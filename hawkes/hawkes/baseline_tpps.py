# %%
from dataclasses import dataclass
from typing import Optional, Tuple
import torch
import torch.nn.functional as F

from .event_utils import MVEventData, BatchedMVEventData
from .utils import inverse_softplus, LinearSpline
from .tpps import NumericalTemporalPointProcess, TemporalPointProcess


@dataclass
class PoissonProcessParams:
    mu: torch.Tensor  # Shape: (D,)


class PoissonProcess(TemporalPointProcess):
    def __init__(
        self,
        params: Optional[PoissonProcessParams] = None,
        D: Optional[int] = None,
        seed: Optional[int] = 42,
        terminating: bool = False,
    ):
        """Initialize the Poisson process.

        Args:
            params: Poisson process parameters or None to initialize randomly
            D: Event dimension (required if params is None)
            seed: random seed for initialization
            terminating: If True, each event type can only occur once
        """
        super().__init__(D, seed, use_analytical_ci=True, terminating=terminating)

        generator = torch.Generator()
        if seed is not None:
            generator = generator.manual_seed(seed)

        self.generator = generator

        if params is None and D is None:
            raise ValueError("Either params or D (number of dimensions) must be provided.")

        if params is None:
            # The only parameters are the baserates mu.
            mu = torch.rand(size=(D,)) * (10 - 1 / 12) + 1 / 12
            self.mu = torch.nn.Parameter(inverse_softplus(mu))
            self.D = D

        else:
            if D is None:
                D = params.mu.shape[0]

            assert params.mu.shape[0] == D, "Dimension mismatch in mu"
            self.mu = torch.nn.Parameter(inverse_softplus(params.mu))
            self.D = D

    def get_baserate(self):
        mu = self.transform_params()

        return {"mu": mu}

    def transform_params(self):
        # Constrain parameters mu to be positive
        return (F.softplus(self.mu),)

    def _raw_intensity(self, t: torch.Tensor, ts: BatchedMVEventData):
        # Computes the raw intensity at time t for each batch (before termination masking).
        # Returns intensity with shape (batch_size, D)

        # lambda_d(t) = mu_d (constant intensity for homogeneous Poisson process)

        (mu,) = self.transform_params()
        return mu.unsqueeze(0).repeat(ts.shape[0], 1)  # Shape: (B,D)

    def _raw_analytical_cumulative_intensity(
        self, T: torch.Tensor, ts: BatchedMVEventData, termination_times: Optional[torch.Tensor] = None
    ):
        """
        Computes the cumulative intensity function from t=0 up to T.
        For terminating processes, computes up to min(T, termination_times) for each dimension.

        Args:
            T: End times. Shape: (batch_size,)
            ts: Historical events. Shape: (batch_size, seq_len)
            termination_times: Optional termination times per dimension. Shape: (batch_size, D)
                If provided, CI is computed up to min(T, termination_times) for each dimension.

        Returns:
            Cumulative intensity. Shape: (batch_size, D)
        """
        (mu,) = self.transform_params()  # Shape: (D,)

        # Effective end time for each dimension
        if termination_times is not None:
            # T: (B,), termination_times: (B, D)
            # Expand T to (B, D) and take min
            T_expanded = T.unsqueeze(1).expand_as(termination_times)  # (B, D)
            effective_T = torch.minimum(T_expanded, termination_times)  # (B, D)
        else:
            # Non-terminating: use T for all dimensions
            effective_T = T.unsqueeze(1).expand(T.shape[0], self.D)  # (B, D)

        # For Poisson: Λ_d(t) = μ_d * t
        # integral: (B, D) = μ (D,) * effective_T (B, D)
        integral = mu.unsqueeze(0) * effective_T  # Shape: (B, D)

        return integral

    def inverse_marginal_cumulative_intensity(
        self,
        u: torch.Tensor,
        ts: BatchedMVEventData,
    ) -> torch.Tensor:
        """
        Invert the marginal cumulative intensity to find t such that Λ_marginal(t) = u.

        For terminating processes, the marginal intensity only includes types that
        haven't occurred yet, and the cumulative intensity accounts for termination:
            Λ_marginal(t) = Σ_d μ_d * min(t, t_term_d)

        For non-terminating: Λ_marginal(t) = (Σ_d μ_d) * t, so t = u / Σ_d μ_d

        Args:
            u: Target marginal cumulative intensity. Shape: (B,)
            ts: Historical events. Shape: (B, N)

        Returns:
            Time t such that Λ_marginal(t) = u. Shape: (B,)
        """
        assert u.shape[0] == ts.time_points.shape[0], "u must have the same batch size as ts"

        device = ts.time_points.device
        (mu,) = self.transform_params()  # (D,)
        lamb = mu.sum()  # Total rate
        B = u.shape[0]

        if not self.terminating:
            # Simple case: Λ(t) = (Σ μ_d) * t => t = u / Σ μ_d
            return u / lamb

        # For terminating processes, we need to handle the piecewise nature.
        # Λ_marginal(t) = Σ_d μ_d * min(t, t_term_d)
        # This is piecewise linear with breakpoints at each t_term_d.

        termination_times = self._get_termination_times(ts)  # (B, D)

        # Check which batch elements have no terminations yet (all inf)
        # For those, we can use the simple formula
        no_terminations = torch.isinf(termination_times).all(dim=1)  # (B,)

        # Simple result for no-termination case
        t_simple = u / lamb  # (B,)

        # If all batch elements have no terminations, return early
        if no_terminations.all():
            return t_simple

        # For batch elements with some terminations, we need the piecewise approach.
        # To avoid inf - inf = nan, replace inf with a large finite value that's
        # larger than any realistic t we'll need to compute.
        # The max CI we might need to invert is u.max(), so max t ≈ u.max() / min(μ)
        # Use a conservative upper bound.
        max_t_needed = (u.max() / mu.min() + 1.0).detach()
        termination_times_finite = torch.where(
            torch.isinf(termination_times), torch.full_like(termination_times, max_t_needed.item()), termination_times
        )

        # Strategy: Sort termination times, then solve piecewise.
        # Before any termination: slope = Σ μ_d
        # After t_term_d: slope decreases by μ_d

        # Sort termination times for each batch
        sorted_terms, sort_indices = torch.sort(termination_times_finite, dim=1)  # (B, D)

        # Get corresponding mu values in sorted order
        mu_sorted = mu[sort_indices]  # (B, D)

        # Cumulative sum of mu in reverse order gives remaining intensity
        mu_cumsum_rev = torch.flip(torch.cumsum(torch.flip(mu_sorted, [1]), dim=1), [1])  # (B, D)
        # mu_cumsum_rev[:, k] = Σ_{j>=k} μ_sorted[:, j] = slope in segment k

        # Prepend 0 to sorted_terms for the segment from 0 to first breakpoint
        zeros = torch.zeros(B, 1, device=device, dtype=sorted_terms.dtype)
        breakpoints = torch.cat([zeros, sorted_terms], dim=1)  # (B, D+1)

        # Compute segment lengths
        segment_lengths = breakpoints[:, 1:] - breakpoints[:, :-1]  # (B, D)

        # Slopes for each segment (slope in segment k is mu_cumsum_rev[:, k])
        slopes = mu_cumsum_rev  # (B, D)

        # CI increment in each segment
        ci_increments = slopes * segment_lengths  # (B, D)

        # Cumulative CI at end of each segment
        ci_cumsum = torch.cumsum(ci_increments, dim=1)  # (B, D)
        # Prepend 0: CI at start of segment k
        ci_at_start = torch.cat([zeros, ci_cumsum[:, :-1]], dim=1)  # (B, D)

        # Find which segment contains u for each batch
        # u falls in segment k if ci_at_start[:, k] <= u < ci_at_start[:, k] + ci_increments[:, k]

        # Use searchsorted to find segment
        segment_idx = torch.searchsorted(ci_cumsum, u.unsqueeze(1)).squeeze(1)  # (B,)
        segment_idx = segment_idx.clamp(max=self.D - 1)  # Clamp to valid range

        # Gather the relevant values for each batch
        batch_indices = torch.arange(B, device=device)

        ci_start = ci_at_start[batch_indices, segment_idx]  # (B,)
        slope = slopes[batch_indices, segment_idx]  # (B,)
        t_start = breakpoints[batch_indices, segment_idx]  # (B,)

        # Solve: ci_start + slope * (t - t_start) = u
        # => t = t_start + (u - ci_start) / slope

        # Handle zero slope (all types terminated) - return inf
        safe_slope = torch.where(slope > 1e-12, slope, torch.ones_like(slope))
        delta_t = (u - ci_start) / safe_slope
        delta_t = torch.where(slope > 1e-12, delta_t, torch.full_like(delta_t, float("inf")))

        t_star = t_start + delta_t

        # Also check if u exceeds the max achievable CI (all types terminated before reaching u)
        # In this case, we need to return inf
        # Max CI is ci_cumsum[:, -1] when all terminations are finite
        # But we replaced inf with max_t_needed, so we need to check against original
        max_ci_per_batch = (
            mu.unsqueeze(0) * torch.minimum(termination_times, torch.full_like(termination_times, float("inf")))
        ).sum(dim=1)  # This would be inf if any termination is inf

        # Actually, simpler: check if t_star exceeds the last finite termination time
        # and slope is 0 at that point
        last_finite_term = termination_times.masked_fill(torch.isinf(termination_times), 0.0).max(dim=1).values
        all_terminated = ~torch.isinf(termination_times).any(dim=1)  # All types have terminated
        max_ci_when_all_terminated = (mu.unsqueeze(0) * termination_times).sum(dim=1)  # Only valid when all_terminated

        # If all types terminated and u > max achievable CI, return inf
        exceeds_max = all_terminated & (u > max_ci_when_all_terminated)
        t_star = torch.where(exceeds_max, torch.full_like(t_star, float("inf")), t_star)

        # Use simple formula for batch elements with no terminations
        t_star = torch.where(no_terminations, t_simple, t_star)

        return t_star

    def positive_likelihood(
        self,
        ts: BatchedMVEventData,
        log: bool = True,
    ):
        # Shape ts: [batch_size, len]
        # ts arrays are right padded. np.inf for time and -1 for type

        # Identify valid (non-padded) events
        valid_events = ts.event_types != -1

        # Shape: D
        (mu,) = self.transform_params()

        intensities = mu[ts.event_types.clamp(min=0)]  # Shape: (B,N,)

        min_intensity = 1e-12
        if log:
            # Clamp intensities to avoid log(0) = -inf or log(negative) = nan
            intensities_clamped = torch.clamp(intensities, min=min_intensity)
            log_intensities = torch.log(intensities_clamped)
            # Screen out invalid contributions. Set them to 0.0 so they dont contribute to the next sum.
            log_intensities[~valid_events] = 0.0
            positive_likelihood = torch.sum(input=log_intensities, dim=-1)  # Shape: (B,)
        else:
            # Have to set the invalid events to 1.0, so they dont contribute to the product.
            # Also clamp to ensure non-negative intensities
            intensities = torch.clamp(intensities, min=min_intensity)
            intensities[~valid_events] = 1.0
            positive_likelihood = torch.prod(intensities, dim=-1)  # Shape: (B,)

        # Mask out event sequences with no elements. They only contribute negativly.
        num_0 = valid_events.sum(dim=1) == 0
        positive_likelihood[num_0] = torch.tensor(0.0) if log else torch.tensor(1.0)

        return positive_likelihood

    def sample(
        self,
        ts: MVEventData,
        num_steps: int = 1,
        rng: Optional[torch.Generator | int] = None,
    ):
        """Closed form-sampling for Poisson Processes. Use the fact, that the CDF is an exponential distribution for sampling.

        For terminating processes, sampling stops when all event types have occurred.

        Args:
            ts (MVEventData): Conditioned past events.
            num_steps (int, optional): Number of steps for generation (for terminating: min(num_steps, remaining types)). Defaults to 1.
            rng (Optional[torch.Generator  |  int], optional): Random state. Defaults to None.
        """
        step = 0
        time_samples = []
        dist_samples = []

        (mu,) = self.transform_params()
        device = ts.time_points.device

        if rng is None:
            rng = torch.Generator()
        elif isinstance(rng, int):
            rng = torch.Generator().manual_seed(rng)

        for step in range(0, num_steps):
            # For terminating processes, check if all types have occurred
            if self.terminating:
                batched = BatchedMVEventData([ts.time_points], [ts.event_types])
                mask = self._get_termination_mask(batched)[0]  # (D,)
                remaining_types = mask.sum().item()
                if remaining_types == 0:
                    # All types have occurred, stop sampling
                    break
                # Use masked mu for sampling
                active_mu = mu * mask.float()
            else:
                active_mu = mu

            # Get the last event time (or 0 if no events yet)
            t_last = ts.time_points[-1].item() if ts.time_points.numel() > 0 else 0.0

            # Intensity is constant
            u = torch.rand(size=(), generator=rng).item()

            # The rate of the superimposed process is the sum of the individual rates
            lamb = active_mu.sum()

            # Check for degenerate case (all intensities zero)
            if lamb <= 1e-12:
                break

            CI_at_last = (active_mu * t_last).sum().item()

            # We now solve for inter-arrival time delta in u = CDF(delta) = 1 - exp(-lamb * delta)
            # => delta = -log(1-u)/lamb

            # As we deal in absolute times here, we need to adjust for the cumulative intensity at t_last
            t_star = (-torch.log1p(-torch.tensor(u)).item() + CI_at_last) / lamb

            # Compute type-distribution at t_star
            probs = active_mu / lamb
            type_idx = torch.multinomial(probs, num_samples=1, generator=rng)

            time_samples.append(t_star)  # Store inter-arrival time
            dist_samples.append(probs)

            T_tensor = torch.tensor([t_star], dtype=ts.time_points.dtype, device=device)

            new_time_points = torch.cat([ts.time_points, T_tensor], dim=0)
            new_event_types = torch.cat([ts.event_types, type_idx], dim=0)

            ts = MVEventData(new_time_points, new_event_types)

        return ts, time_samples, dist_samples


@dataclass
class ConditionalIPPParams:
    mu: torch.Tensor  # D intercepts per disease
    slope: torch.Tensor  # D slopes per disease


class ConditionalInhomogeniousPoissonProcess(TemporalPointProcess):
    def __init__(
        self,
        params: Optional[ConditionalIPPParams] = None,
        D: Optional[int] = None,
        seed: Optional[int] = 42,
        terminating: bool = False,
    ):
        """Initialize the Conditional Poisson process.

        Args:
            params: Poisson process parameters or None to initialize randomly
            D: Event dimension (required if params is None)
            seed: random seed for initialization
            terminating: If True, each event type can only occur once
        """
        super().__init__(D, seed, use_analytical_ci=True, terminating=terminating)

        generator = torch.Generator()
        if seed is not None:
            generator = generator.manual_seed(seed)

        self.generator = generator

        if params is None and D is None:
            raise ValueError("Either params or D (number of dimensions) must be provided.")

        if params is None:
            # The only parameters are the baserates mu.
            # TODO parameter inits problematic.
            mu = torch.log(torch.rand(size=(D,)))  # torch.log(torch.rand(size=(D,)) * (10 - 1 / 12) + 1 / 12)
            slope = torch.zeros(size=(D,))  # torch.log(torch.rand(size=(D,)) * (10 - 1 / 12) + 1 / 12)
            self.mu = torch.nn.Parameter(mu)
            self.slope = torch.nn.Parameter(slope)
            self.D = D

        else:
            if D is None:
                D = params.mu.shape[0]

            assert params.mu.shape[0] == D, "Dimension mismatch in mu"
            self.mu = torch.nn.Parameter(params.mu)
            self.slope = torch.nn.Parameter(params.slope)
            self.D = D

    def get_rate_slope(self):
        mu = self.transform_params()

        return {"mu": mu, "slope": self.slope}

    def transform_params(self):
        # Constrain parameters mu to be positive
        return self.mu, self.slope

    def _raw_intensity(self, t: torch.Tensor, ts: BatchedMVEventData):
        # Computes the raw intensity at time t for each batch (before termination masking).
        # Returns intensity with shape (batch_size, D)

        # lambda_d(t) = exp(mu_d + slope_d * t) (inhomogeneous Poisson with log-linear intensity)

        mu, slope = self.transform_params()

        intensity = torch.exp(mu.unsqueeze(0) + slope.unsqueeze(0) * t.unsqueeze(1))  # Shape: (B,D)
        return intensity

    def _raw_analytical_cumulative_intensity(
        self, T: torch.Tensor, ts: BatchedMVEventData, termination_times: Optional[torch.Tensor] = None
    ):
        """
        Computes the cumulative intensity function from t=0 up to T.
        For terminating processes, computes up to min(T, termination_times) for each dimension.

        Args:
            T: End times. Shape: (batch_size,)
            ts: Historical events. Shape: (batch_size, seq_len)
            termination_times: Optional termination times per dimension. Shape: (batch_size, D)
                If provided, CI is computed up to min(T, termination_times) for each dimension.

        Returns:
            Cumulative intensity. Shape: (batch_size, D)
        """
        mu, slope = self.transform_params()

        # Effective end time for each dimension
        if termination_times is not None:
            # T: (B,), termination_times: (B, D)
            T_expanded = T.unsqueeze(1).expand_as(termination_times)  # (B, D)
            effective_T = torch.minimum(T_expanded, termination_times)  # (B, D)
        else:
            # Non-terminating: use T for all dimensions
            effective_T = T.unsqueeze(1).expand(T.shape[0], self.D)  # (B, D)

        slope_0 = torch.abs(slope) < 1e-8

        # At slope == 0 (or very close) we use the Taylor expansion to avoid division by zero.
        # Taylor expansion: exp(bT) ~ 1 + bT + b^2T^2/2 + ...
        # effective_T shape: (B, D), need to broadcast with (D,) parameters
        taylor_expansion_at_b = torch.exp(mu.unsqueeze(0)) * (
            effective_T + slope.unsqueeze(0) * effective_T**2 / 2 + slope.unsqueeze(0) ** 2 * effective_T**3 / 6
        )  # Shape: (B, D)

        fixed_slope = slope.clone()
        fixed_slope[slope_0] = 1.0  # avoid division by zero

        closed_form_integral = (
            torch.exp(mu.unsqueeze(0))
            / fixed_slope.unsqueeze(0)
            * (torch.exp(fixed_slope.unsqueeze(0) * effective_T) - 1)
        )  # Shape: (B, D)

        # Select between Taylor and closed form based on slope magnitude
        integral = torch.where(
            slope_0.unsqueeze(0),  # (D,) -> (1, D) -> broadcast to (B, D)
            taylor_expansion_at_b,
            closed_form_integral,
        )  # Shape: (B, D)

        return integral

    def positive_likelihood(
        self,
        ts: BatchedMVEventData,
        log: bool = True,
    ):
        # Shape ts: [batch_size, len]
        # ts arrays are right padded. -1 for type and the largest number for time.

        # Identify valid (non-padded) events
        valid_events = ts.event_types != -1

        # Shape: D
        mu, slope = self.transform_params()

        # Clamp event_types to avoid negative indexing for padded entries (-1)
        clamped_types = ts.event_types.clamp(min=0)
        intensities = torch.exp(mu[clamped_types] + slope[clamped_types] * ts.time_points)  # Shape: (B,N,)

        min_intensity = 1e-12
        if log:
            # Clamp intensities to avoid log(0) = -inf or log(negative) = nan
            intensities_clamped = torch.clamp(intensities, min=min_intensity)
            log_intensities = torch.log(intensities_clamped)
            # Screen out invalid contributions. Set them to 0.0 so they dont contribute to the next sum.
            # log_intensities[~valid_events] = 0.0
            log_intensities *= valid_events.float()
            positive_likelihood = torch.sum(input=log_intensities, dim=-1)  # Shape: (B,)
        else:
            # Have to set the invalid events to 1.0, so they dont contribute to the product.
            # Also clamp to ensure non-negative intensities
            intensities = torch.clamp(intensities, min=min_intensity)
            intensities[~valid_events] = 1.0
            positive_likelihood = torch.prod(intensities, dim=-1)  # Shape: (B,)

        # Mask out event sequences with no elements. They only contribute negativly.
        num_0 = valid_events.sum(dim=1) == 0
        positive_likelihood[num_0] = torch.tensor(0.0) if log else torch.tensor(1.0)

        return positive_likelihood


@dataclass
class SplineProcessParams:
    h_knots: torch.Tensor  # Shape: (D, num_knots)
    delta_ts: torch.Tensor  # Time intervals between knots, Shape: (num_knots,) or scalar


class SplinePoissonProcess(TemporalPointProcess):
    def __init__(
        self,
        D: int,
        num_knots: int,
        delta_t: torch.Tensor | float,
        params: Optional[SplineProcessParams] = None,
        seed: Optional[int] = 42,
        terminating: bool = False,
    ):
        super().__init__(D, seed, use_analytical_ci=True, terminating=terminating)

        if params is None:
            self.D = D
            self.num_knots = num_knots

            # Initialize heights randomly in log-space/softplus-space
            # Target range ~[0.1, 10.0]
            initial_h = torch.rand(D, num_knots) + 0.1  # [0-1] = 0-80 years.
            self.h_knots = torch.nn.Parameter(inverse_softplus(initial_h))

            if isinstance(delta_t, float) or delta_t.numel() == 1:
                knot_locs = torch.arange(num_knots, dtype=torch.float32) * delta_t
            else:
                knot_locs = torch.cumsum(delta_t, dim=0)

            # Register as buffer so it moves with .to(device) and is saved/loaded
            self.register_buffer("knot_locs", knot_locs)

            assert self.knot_locs.shape[0] == self.h_knots.shape[1]

        else:
            self.h_knots = torch.nn.Parameter(inverse_softplus(params.h_knots))
            knot_locs = torch.cumsum(params.delta_ts, dim=0)
            self.register_buffer("knot_locs", knot_locs)
            self.D = params.h_knots.shape[0]
            self.num_knots = params.h_knots.shape[1]

        assert self.num_knots == self.h_knots.shape[1], "Number of knots mismatch"

    def transform_params(self):
        "Return the parameters in their acutal/constrained form."
        return (F.softplus(self.h_knots),)

    def get_heights(self):
        """Returns non-negative knot heights: Shape (D, num_knots)

        Note: Returns a tuple containing the tensor for consistency with transform_params().
        Use (h,) = self.get_heights() to unpack.
        """
        return self.transform_params()

    def _raw_intensity(self, t: torch.Tensor, ts: BatchedMVEventData):
        """
        lambda_d(t) = h_k + m_k * delta (raw intensity before termination masking)
        Returns: (batch_size, D)
        """
        # Expand T into 1-dimensional tensor if needed
        if t.dim() == 0:
            t = t.unsqueeze(0)

        (h,) = self.get_heights()  # (D, K)

        intensity = LinearSpline.interpolate(self.knot_locs, h, t)  # (B,D)

        return intensity

    def _raw_analytical_cumulative_intensity(
        self, T: torch.Tensor, ts: BatchedMVEventData, termination_times: Optional[torch.Tensor] = None
    ):
        """
        Calculates the cumulative intensity from 0 to T for each dimension.
        For terminating processes, computes up to min(T, termination_times) for each dimension.

        Args:
            T: End times. Shape: (batch_size,)
            ts: Historical events. Shape: (batch_size, seq_len)
            termination_times: Optional termination times per dimension. Shape: (batch_size, D)
                If provided, CI is computed up to min(T, termination_times) for each dimension.

        Returns:
            Cumulative intensity. Shape: (batch_size, D)
        """
        # Expand T into 1-dimensional tensor if needed
        if T.dim() == 0:
            T = T.unsqueeze(0)

        (h,) = self.get_heights()  # (D,K)

        # Effective end time for each dimension
        if termination_times is not None:
            # T: (B,), termination_times: (B, D)
            T_expanded = T.unsqueeze(1).expand_as(termination_times)  # (B, D)
            effective_T = torch.minimum(T_expanded, termination_times)  # (B, D)
        else:
            # Non-terminating: use T for all dimensions
            effective_T = T.unsqueeze(1).expand(T.shape[0], self.D)  # (B, D)

        # Integrate spline from 0 to effective_T for each (batch, dimension) pair
        # LinearSpline.integrate expects t to have shape that broadcasts with (D, K)
        # effective_T has shape (B, D), which works correctly
        integral = LinearSpline.integrate(x_knots=self.knot_locs, y_knots=h, t=effective_T)  # (B, D)

        return integral

    def inverse_marginal_cumulative_intensity(
        self,
        u: torch.Tensor,
        ts: BatchedMVEventData,
    ) -> torch.Tensor:
        """
        Invert the marginal cumulative intensity to find t such that Λ(t) = u.

        This function returns the ABSOLUTE time t, not a relative delta.
        The computation finds t such that the integral of the marginal intensity
        from 0 to t equals u.

        Args:
            u: Target marginal cumulative intensity value (typically -log(1-uniform) + Λ(t_last)). Shape: (batch_size,)
            ts: Historical events (batched, padded sequences). Shape: (batch_size, seq_len)

        Returns:
            Absolute time values t such that Λ(t) ≈ u. Shape: same as u

        Useful for sampling via inverse CDF method.
        """

        assert u.shape[0] == ts.time_points.shape[0], "u must have the same batch size as ts"

        (h,) = self.get_heights()  # (D, K)
        B = u.shape[0]

        if self.terminating:
            # For terminating processes, the marginal CI is complex:
            # Λ_marginal(t) = Σ_d ∫_0^{min(t, t_term_d)} spline_d(s) ds
            # This is a piecewise function with breakpoints at each termination time.
            # We use numerical root finding via the base class.
            return super().inverse_marginal_cumulative_intensity(u, ts)

        # Non-terminating case: simple sum of splines
        # Build the marginal intensity by summing across dimensions
        # h_marginal[k] = sum_d h[d, k]
        h_marginal = h.sum(dim=0, keepdim=True)  # (1, K)

        # Invert: find t such that Lambda_marginal(t) = Lambda_target
        # u has shape (B,), we need (B, 1) to match h_marginal's D=1
        t_next = LinearSpline.inverse_integral(
            self.knot_locs,
            h_marginal,
            u.unsqueeze(1),  # (B, 1)
        )  # (B, 1)
        t_next = t_next.squeeze(1)  # (B,)

        return t_next

    def positive_likelihood(self, ts: BatchedMVEventData, log: bool = True):
        # Handle empty sequences: no events means positive_likelihood = 1 (or 0 in log)
        B = ts.shape[0]
        N = ts.shape[1]
        if N == 0:
            if log:
                return torch.zeros(B, dtype=ts.time_points.dtype, device=ts.time_points.device)
            else:
                return torch.ones(B, dtype=ts.time_points.dtype, device=ts.time_points.device)

        # For terminating processes, use the base class implementation which properly
        # handles the batch structure for termination masks
        if self.terminating:
            return super().positive_likelihood(ts, log=log)

        # Identify valid events
        valid_events = ts.event_types != -1  # (B, N)

        # We need intensity at every event time t_i for type d_i
        # This requires vectorized indexing over the spline
        # Flatten times to process through _get_knot_info
        flat_times = ts.time_points.flatten()  # (B*N)
        flat_events = ts.event_types.flatten()  # (B*N)

        # ATTENTION! Only works because it is independent from the past values. Does not use ts.
        all_intensities = self.intensity(flat_times, ts)  # (B*N, D)
        all_intensities = all_intensities.view(ts.shape[0], ts.shape[1], self.D)  # (B, N, D)

        # Select the intensity corresponding to the actual event type
        # event_types shape: (B, N)
        # Using gather to pick the d-th intensity
        # Not sure if clamp is necesarry, but they get masked out anyway later.
        event_intensities = torch.gather(
            all_intensities, dim=2, index=ts.event_types.unsqueeze(2).clamp(min=0)
        ).squeeze(2)

        min_intensity = 1e-12
        if log:
            log_intensities = torch.log(torch.clamp(event_intensities, min=min_intensity))
            log_intensities[~valid_events] = 0.0
            return torch.sum(log_intensities, dim=-1)
        else:
            event_intensities[~valid_events] = 1.0
            return torch.prod(event_intensities, dim=-1)

    # def sample(self, ts: MVEventData, num_steps: int = 1, rng: Optional[torch.Generator | int] = None):
    #     """
    #     Sampling via Superposition + Inverse CDF Transform.

    #     Strategy:
    #     1. Build the marginal (superposition) process by summing intensities across all D dimensions
    #     2. Sample inter-arrival times from the marginal process using inverse transform
    #     3. At each sampled time, determine event type via categorical distribution weighted by intensities

    #     Args:
    #         ts (MVEventData): Conditioned past events.
    #         num_steps (int, optional): Number of steps for generation. Defaults to 1.
    #         rng (Optional[torch.Generator], optional): Random state. Defaults to None.

    #     Returns:
    #         ts: Updated MVEventData with new events appended.
    #         time_samples: List of sampled inter-arrival times.
    #         dist_samples: List of type probability distributions at each sampled time.
    #     """
    #     if rng is None:
    #         rng = torch.Generator(device=self.h_knots.device)
    #     elif isinstance(rng, int):
    #         rng = torch.Generator(device=self.h_knots.device).manual_seed(rng)

    #     curr_ts = ts  # (N,) can be None
    #     (h,) = self.get_heights()  # (D, K) - unpack from tuple
    #     device = h.device

    #     time_samples = []
    #     dist_samples = []

    #     # Build the marginal intensity by summing across dimensions
    #     # h_marginal[k] = sum_d h[d, k]
    #     h_marginal = h.sum(dim=0, keepdim=True)  # (1, K)

    #     for step in range(num_steps):
    #         t_start = curr_ts.time_points.max() if curr_ts.time_points.numel() > 0 else torch.tensor(0.0, device=device)

    #         # Compute cumulative intensity of marginal process at t_start
    #         # Lambda_marginal(t) = sum_d Lambda_d(t) = integral of sum_d lambda_d(s) from 0 to t
    #         if t_start > 0:
    #             Lambda_start = LinearSpline.integrate(self.knot_locs, h_marginal, t_start.unsqueeze(0))  # (1, 1)
    #             Lambda_start = Lambda_start.squeeze()  # scalar
    #         else:
    #             Lambda_start = torch.tensor(0.0, device=device)

    #         # Sample u ~ Uniform(0, 1) and compute inverse of target cumulative intensity
    #         # u = 1 - exp(-(Lambda_target - Lambda_start)) => CDF inversion
    #         # 1 -u = exp(-(Lambda_target - Lambda_start)) => -log(1-u) = Lambda_target - Lambda_start
    #         u = torch.rand(size=(), generator=rng, device=device)
    #         Lambda_target = Lambda_start - torch.log(1 - u)

    #         # Invert: find t such that Lambda_marginal(t) = Lambda_target
    #         t_next = LinearSpline.inverse_integral(
    #             self.knot_locs,
    #             h_marginal,
    #             Lambda_target.unsqueeze(0).unsqueeze(0),  # (1, 1)
    #         )  # (1, 1)
    #         t_next = t_next.squeeze()  # scalar

    #         # Compute inter-arrival time
    #         delta_t = t_next - t_start
    #         time_samples.append(delta_t.item())

    #         # Determine event type: sample from categorical weighted by intensities at t_next
    #         intensities_at_t = LinearSpline.interpolate(self.knot_locs, h, t_next.unsqueeze(0))  # (1, D)
    #         intensities_at_t = intensities_at_t.squeeze(0)  # (D,)

    #         # Normalize to get probabilities
    #         probs = intensities_at_t / intensities_at_t.sum()
    #         dist_samples.append(probs.detach().clone())

    #         # Sample event type
    #         event_type = torch.multinomial(probs, num_samples=1, generator=rng)  # (1,)

    #         # Append new event to sequence
    #         new_time_points = torch.cat([curr_ts.time_points, t_next.unsqueeze(0)], dim=0)
    #         new_event_types = torch.cat([curr_ts.event_types, event_type], dim=0)

    #         curr_ts = MVEventData(new_time_points, new_event_types)

    #     return curr_ts, time_samples, dist_samples


class DebugNumericalPoisson(NumericalTemporalPointProcess):
    """
    A simple constant-rate Poisson process for debugging NumericalTemporalPointProcess.

    This is a homogeneous Poisson process with constant intensity μ for each dimension.
    The intensity does NOT depend on history, making it easy to verify correctness.

    Analytical solutions (for comparison):
        - Intensity: λ_d(t) = μ_d (constant)
        - Cumulative intensity: Λ_d(T) = μ_d * T
        - Likelihood: log L = sum_i log(μ_{m_i}) - sum_d μ_d * T
    """

    def __init__(
        self,
        D: int,
        mu: Optional[torch.Tensor] = None,
        seed: Optional[int] = 42,
        ci_integration_method: str = "trapezoidal",
        ci_num_points: int = 50,
        terminating: bool = False,
    ):
        """
        Args:
            D: Number of event types
            mu: Base intensity rates. Shape: (D,). If None, defaults to ones.
            seed: Random seed
            ci_integration_method: Integration method for numerical CI
            ci_num_points: Number of integration points
            terminating: If True, each event type can only occur once
        """
        super().__init__(
            D=D,
            seed=seed,
            ci_integration_method=ci_integration_method,
            ci_num_points=ci_num_points,
            terminating=terminating,
        )

        if mu is None:
            mu = torch.ones(D, dtype=torch.float64)
        self.register_buffer("mu", mu)

    def transform_params(self) -> Tuple[torch.Tensor, ...]:
        """Return the intensity rates."""
        return (self.mu,)

    def _raw_intensity(self, t: torch.Tensor, ts: BatchedMVEventData) -> torch.Tensor:
        """Constant intensity μ for all times (before termination masking)."""
        B = t.shape[0]
        return self.mu.unsqueeze(0).expand(B, -1)  # (B, D)

    def _raw_analytical_cumulative_intensity(
        self, T: torch.Tensor, ts: BatchedMVEventData, termination_times: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Analytical CI for verification: Λ_d(T) = μ_d * T
        For terminating processes: Λ_d(T) = μ_d * min(T, t_term_d)
        """
        # Effective end time for each dimension
        if termination_times is not None:
            T_expanded = T.unsqueeze(1).expand_as(termination_times)  # (B, D)
            effective_T = torch.minimum(T_expanded, termination_times)  # (B, D)
        else:
            effective_T = T.unsqueeze(1).expand(T.shape[0], self.D)  # (B, D)

        return self.mu.unsqueeze(0) * effective_T  # (B, D)
