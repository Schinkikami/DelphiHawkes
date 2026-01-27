# %%
from dataclasses import dataclass
from typing import Optional
import torch
import torch.nn.functional as F

from .event_utils import MVEventData, BatchedMVEventData
from .utils import inverse_softplus, LinearSpline
from .tpps import TemporalPointProcess


@dataclass
class PoissonProcessParams:
    mu: torch.Tensor  # Shape: (D,)


class PoissonProcess(TemporalPointProcess):
    def __init__(
        self,
        params: Optional[PoissonProcessParams] = None,
        D: Optional[int] = None,
        seed: Optional[int] = 42,
    ):
        """Initialize the Poisson process.

        Args:
            params: Poisson process parameters or None to initialize randomly
            D: Event dimension (required if params is None)
            seed: random seed for initialization
            reg_lambda: regularization parameter
        """
        super().__init__(D, seed, True)

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

    def intensity(self, t: torch.Tensor, ts: BatchedMVEventData):
        # Computes the intensity at time t for each batch.
        # Returns intensity with shape (batch_size, D)

        # lambda_d(t) = mu_d (constant intensity for homogeneous Poisson process)

        (mu,) = self.transform_params()
        return mu.unsqueeze(0).repeat(ts.shape[0], 1)  # Shape: (B,D)

    def analytical_cumulative_intensity(self, T: torch.Tensor, ts: BatchedMVEventData):
        """
        Computes the cumulative intensity function from t=0 up to T, not from the last event time, as we need the full integral for the likelihood.
        To obtain the cumulative intensity from t_n up to T, consider subtracting the CIs.--> cumulative_intensity(T) - cumulative_intensity(t_n)

        :param self: Description
        :param T: Description
        :type T: torch.Tensor
        :param ts: Description
        :type ts: BatchedMVEventData
        """

        (mu,) = self.transform_params()
        integral = mu.unsqueeze(1) * T.unsqueeze(0)  # Shape: (D,B)

        return integral.T  # Shape: (B,D)

    def inverse_marginal_cumulative_intensity(
        self,
        u: torch.Tensor,
        ts: BatchedMVEventData,
    ) -> torch.Tensor:
        """
        As the Poisson process has constant intensity, we can invert the marginal cumulative intensity function in closed form.
        """

        assert u.shape[0] == ts.time_points.shape[0], "u must have the same batch size as ts"

        device = ts.time_points.device
        (mu,) = self.transform_params()

        lamb = mu.sum()

        t_star = u / lamb  # Shape: (B,)

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

        Args:
            ts (MVEventData): Conditioned past events.
            num_steps (int, optional): Number of steps for generation. Defaults to 1.
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
            # Get the last event time (or 0 if no events yet)
            t_last = ts.time_points[-1].item() if ts.time_points.numel() > 0 else 0.0

            # Intensity is constant
            u = torch.rand(size=(), generator=rng).item()

            # The rate of the superimposed process is the sum of the individual rates
            lamb = mu.sum()
            CI_at_last = (mu * t_last).sum().item()

            # We now solve for inter-arrival time delta in u = CDF(delta) = 1 - exp(-lamb * delta)
            # => delta = -log(1-u)/lamb

            # As we deal in absolute times here, we need to adjust for the cumulative intensity at t_last
            t_star = (-torch.log1p(-torch.tensor(u)).item() + CI_at_last) / lamb

            # Compute type-distribution at t_star
            probs = mu / lamb
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
    ):
        """Initialize the Conditional Poisson process.

        Args:
            params: Poisson process parameters or None to initialize randomly
            D: Event dimension (required if params is None)
            seed: random seed for initialization
            reg_lambda: regularization parameter
        """
        super().__init__(D, seed, True)

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

    def intensity(self, t: torch.Tensor, ts: BatchedMVEventData):
        # Computes the intensity at time t for each batch.
        # Returns intensity with shape (batch_size, D)

        # lambda_d(t) = exp(mu_d + slope_d * t) (inhomogeneous Poisson with log-linear intensity)

        mu, slope = self.transform_params()

        intensity = torch.exp(mu.unsqueeze(0) + slope.unsqueeze(0) * t.unsqueeze(1))  # Shape: (B,D)
        return intensity

    def analytical_cumulative_intensity(self, T: torch.Tensor, ts: BatchedMVEventData):
        """
        Computes the cumulative intensity function from t=0 up to T, not from the last event time, as we need the full integral for the likelihood.
        To obtain the cumulative intensity from t_n up to T, consider subtracting the CIs.--> cumulative_intensity(T) - cumulative_intensity(t_n)

        :param self: Description
        :param T: Description
        :type T: torch.Tensor
        :param ts: Description
        :type ts: BatchedMVEventData
        """

        mu, slope = self.transform_params()

        slope_0 = torch.abs(slope) < 1e-8

        # At slope == 0 (or very close) we use the Taylor expansion to avoid division by zero.
        # This is better than adding a deadzone to the gradient: lim b-> 0 int_0^T exp(a+bt) dt = int_0^T exp(a) dt = exp(a) *T. However this forumulation does not have gradient for b.
        # Taylor at very small b: exp(bT) ~ 1 for very small b.
        # Therefore exp(bT) = 1 + bT + b^2T^2/2 + ... at these very small b.
        # --> (exp(bT) -1)/b = T + bT^2/2 + b^2T^3/6 + ...
        # --> e^mu * (exp(bT) -1)/b = e^mu * (T + bT^2/2 + b^2T^3/6 + ...)
        taylor_expansion_at_b = torch.exp(mu.unsqueeze(1)) * (
            T.unsqueeze(0)
            + slope.unsqueeze(1) * T.unsqueeze(0) ** 2 / 2
            + slope.unsqueeze(1) ** 2 * T.unsqueeze(0) ** 3 / 6
        )

        fixed_slope = slope.clone()
        fixed_slope[slope_0] = 1.0  # avoid division by zero

        closed_form_integral = torch.zeros_like(taylor_expansion_at_b)
        closed_form_integral = (
            torch.exp(mu.unsqueeze(1))
            / fixed_slope.unsqueeze(1)
            * (torch.exp(fixed_slope.unsqueeze(1) * T.unsqueeze(0)) - 1)
        )

        # Here we replace the normal integral form of int_0^T exp(mu +slope t) dt = 1/b exp(mu + slope T) - 1/b exp(mu) = 1/b * exp(mu) * (exp(slope T) - 1) = 1/b *exp(mu) * expm1(slop T).
        integral = torch.where(
            slope_0.unsqueeze(1),
            taylor_expansion_at_b,  # torch.exp(mu.unsqueeze(1)) * T.unsqueeze(0),
            closed_form_integral,  # torch.exp(mu.unsqueeze(1) + slope.unsqueeze(1) * T.unsqueeze(0)) / slope.unsqueeze(1)
        )  # Shape: (D,B)

        return integral.T  # Shape: (B,D)

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
    ):
        super().__init__(D, seed, True)

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

    def intensity(self, t: torch.Tensor, ts: BatchedMVEventData):
        """
        lambda_d(t) = h_k + m_k * delta
        Returns: (batch_size, D)
        """
        # Expand T into 1-dimensional tensor if needed
        if t.dim() == 0:
            t = t.unsqueeze(0)

        (h,) = self.get_heights()  # (D, K)

        intensity = LinearSpline.interpolate(self.knot_locs, h, t)  # (B,D)

        return intensity

    def analytical_cumulative_intensity(self, T: torch.Tensor, ts: BatchedMVEventData):
        """
        Calculates the integral from 0 to T for each dimension.
        Returns: (batch_size, D)
        """

        # Expand T into 1-dimensional tensor if needed
        if T.dim() == 0:
            T = T.unsqueeze(0)

        (h,) = self.get_heights()  # (D,K)

        integral = LinearSpline.integrate(x_knots=self.knot_locs, y_knots=h, t=T)  # (B, D)

        return integral  # (B,D)

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
        # Compute inter-arrival time

        return t_next

    def positive_likelihood(self, ts: BatchedMVEventData, log: bool = True):
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
