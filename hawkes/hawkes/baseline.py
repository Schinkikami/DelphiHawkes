# %%
from dataclasses import dataclass
from typing import Optional
import torch
import torch.nn.functional as F

from .event_utils import MVEventData, BatchedMVEventData
from .utils import inverse_softplus


@dataclass
class PoissonProcessParams:
    mu: torch.Tensor  # Shape: (D,)


class PoissonProcess(torch.nn.Module):
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

        super().__init__()

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
        return F.softplus(self.mu)

    def intensity(self, t: torch.Tensor, ts: BatchedMVEventData):
        # Computes the intensity at time t for each batch.
        # Returns intensity with shape (batch_size, D)

        # lambda_d(t) = mu_d + \sum_d \sum_{t_i < t} \mathbb{1}(m_i == d) \phi_{d,m_i}(t-t_i)

        mu = self.transform_params()
        return mu.unsqueeze(0).repeat(ts.shape[0], 1)  # Shape: (B,D)

    def cumulative_intensity(self, T: torch.Tensor, ts: BatchedMVEventData):
        """
        Computes the cumulative intensity function from t=0 up to T, not from the last event time, as we need the full integral for the likelihood.
        To obtain the cumulative intensity from t_n up to T, consider subtracting the CIs.--> cumulative_intensity(T) - cumulative_intensity(t_n)

        :param self: Description
        :param T: Description
        :type T: torch.Tensor
        :param ts: Description
        :type ts: BatchedMVEventData
        """

        mu = self.transform_params()
        integral = mu.unsqueeze(1) * T.unsqueeze(0)  # Shape: (D,B)

        return integral.T  # Shape: (B,D)

    def PDF(self, T: torch.Tensor, ts: BatchedMVEventData):
        """Returns the joint probability density function at T for every type E: p(t=T, e=E|H_t_).
        args:
            T: torch.Tensor = Points where to evaluate the PDF. Shape: (batch_size)
            ts:BatchedMVEventData = A batch of padded sequences of size (batch_size, sequence_length)

        returns:
            pdfs: torch.Tensor = The PDF per batch element at the batch specific time point T[i] for each type e. p(t=T, e=E|H_t_).
        """

        intensities = self.intensity(T, ts)  # Shape: (B,D)
        ci = self.cumulative_intensity(T, ts) - self.cumulative_intensity(ts.max_time, ts)  # Shape: (B,D)
        pdfs = intensities * torch.exp(-torch.sum(ci, dim=1)).unsqueeze(1)  # Shape: (B,D)
        return pdfs

    def CDF(self, T: torch.Tensor, ts: BatchedMVEventData):
        """Returns the cumulative density function at T for any event.
        This is the CDF of the marginal over all types of the joint distribution.
        Defining CDF(T,E) does not make sense: p(t < T, e < E) for categorical E's?
        args:
            T: torch.Tensor = Points where to evaluate the CDF. Shape: (batch_size)
            ts:BatchedMVEventData = A batch of padded sequences of size (batch_size, sequence_length)

        returns:
            cdfs: torch.Tensor = The CDF per batch element at the batch specific time point T[i].
        """
        ci = self.cumulative_intensity(T, ts) - self.cumulative_intensity(ts.max_time, ts)  # Shape: (B,D)
        cdfs = 1 - torch.exp(-torch.sum(ci, dim=1))  # Shape: (B,)
        return cdfs

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
        mu = self.transform_params()

        intensities = mu  # Shape: (B,N,)

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

    def negative_likelihood(
        self, ts: BatchedMVEventData, T: torch.Tensor, log: bool = True, num_integration_points: int = 0
    ):
        if num_integration_points == 0:
            integral = self.cumulative_intensity(T=T, ts=ts)
        else:
            raise RuntimeError("We should not be here.")

        if log:
            negative_likelihood = torch.sum(-integral, dim=1)  # Shape: (B,)
        else:
            negative_likelihood = torch.prod(torch.exp(-integral), dim=1)  # Shape: (B,)

        return negative_likelihood

    def likelihood(
        self,
        ts: BatchedMVEventData,
        T: torch.Tensor,
        log: bool = True,
    ):
        """
        Compute the likelihood of observed events for a TPP.

        Args:
            ts: Array of event times.
            T: End time for the observation window (defaults to last event). Has shape (batch_size,)
            log: Whether to return the log-likelihood (bool).

        Returns:
            The (log-)likelihood value (float) for the observed events.

        Mathematical context:
            Likelihood = product of intensities at event times
                        minus the integral of the intensity over [0, T] (the log-survival function)
        """

        # Positive likelihood
        # ----
        positive_likelihood = self.positive_likelihood(ts, log)

        # ----
        # Negative likelihood
        # ----
        negative_likelihood = self.negative_likelihood(ts, T, log, 0)

        # ----
        # Return total likelihood

        if log:
            return positive_likelihood + negative_likelihood
        else:
            return positive_likelihood * negative_likelihood

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

        mu = self.transform_params()
        device = ts.time_points.device

        if rng is None:
            rng = torch.Generator()
        elif isinstance(rng, int):
            rng = torch.Generator().manual_seed(rng)

        for step in range(0, num_steps):
            # Intensity is constant
            u = torch.rand(size=(), generator=rng).item()

            # The rate of the superimposed process is the sum of the individual rates
            lamb = mu.sum()
            # We not solve for T* in u = CDF(T*) = 1 - exp(-lamb* * T* )
            # => T* = -log(1-u)/lamb
            t_star = -torch.log1p(-torch.tensor(u)).item() / lamb

            # Compute type-distribution at t_star
            probs = mu / lamb
            type_idx = torch.multinomial(probs, num_samples=1, generator=rng)

            time_samples.append(t_star)
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


class ConditionalInhomogeniousPoissonProcess(torch.nn.Module):
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

        super().__init__()

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

        # lambda_d(t) = mu_d + \sum_d \sum_{t_i < t} \mathbb{1}(m_i == d) \phi_{d,m_i}(t-t_i)

        mu, slope = self.transform_params()

        intensity = torch.exp(mu.unsqueeze(0) + slope.unsqueeze(0) * t.unsqueeze(1))  # Shape: (B,D)
        return intensity

    def cumulative_intensity(self, T: torch.Tensor, ts: BatchedMVEventData):
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

    def PDF(self, T: torch.Tensor, ts: BatchedMVEventData):
        """Returns the joint probability density function at T for every type E: p(t=T, e=E|H_t_).
        args:
            T: torch.Tensor = Points where to evaluate the PDF. Shape: (batch_size)
            ts:BatchedMVEventData = A batch of padded sequences of size (batch_size, sequence_length)

        returns:
            pdfs: torch.Tensor = The PDF per batch element at the batch specific time point T[i] for each type e. p(t=T, e=E|H_t_).
        """

        intensities = self.intensity(T, ts)  # Shape: (B,D)
        ci = self.cumulative_intensity(T, ts) - self.cumulative_intensity(ts.max_time, ts)  # Shape: (B,D)
        pdfs = intensities * torch.exp(-torch.sum(ci, dim=1)).unsqueeze(1)  # Shape: (B,D)
        return pdfs

    def CDF(self, T: torch.Tensor, ts: BatchedMVEventData):
        """Returns the cumulative density function at T for any event.
        This is the CDF of the marginal over all types of the joint distribution.
        Defining CDF(T,E) does not make sense: p(t < T, e < E) for categorical E's?
        args:
            T: torch.Tensor = Points where to evaluate the CDF. Shape: (batch_size)
            ts:BatchedMVEventData = A batch of padded sequences of size (batch_size, sequence_length)

        returns:
            cdfs: torch.Tensor = The CDF per batch element at the batch specific time point T[i].
        """
        ci = self.cumulative_intensity(T, ts) - self.cumulative_intensity(ts.max_time, ts)  # Shape: (B,D)
        cdfs = 1 - torch.exp(-torch.sum(ci, dim=1))  # Shape: (B,)
        return cdfs

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

        intensities = torch.exp(mu[ts.event_types] + slope[ts.event_types] * ts.time_points)  # Shape: (B,N,)

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

    def negative_likelihood(
        self, ts: BatchedMVEventData, T: torch.Tensor, log: bool = True, num_integration_points: int = 0
    ):
        if num_integration_points == 0:
            integral = self.cumulative_intensity(T=T, ts=ts)
        else:
            raise RuntimeError("We should not be here.")

        if log:
            negative_likelihood = torch.sum(-integral, dim=1)  # Shape: (B,)
        else:
            negative_likelihood = torch.prod(torch.exp(-integral), dim=1)  # Shape: (B,)

        return negative_likelihood

    def likelihood(
        self,
        ts: BatchedMVEventData,
        T: torch.Tensor,
        log: bool = True,
    ):
        """
        Compute the likelihood of observed events for a TPP.

        Args:
            ts: Array of event times.
            T: End time for the observation window (defaults to last event). Has shape (batch_size,)
            log: Whether to return the log-likelihood (bool).

        Returns:
            The (log-)likelihood value (float) for the observed events.

        Mathematical context:
            Likelihood = product of intensities at event times
                        minus the integral of the intensity over [0, T] (the log-survival function)
        """

        # Positive likelihood
        # ----
        positive_likelihood = self.positive_likelihood(ts, log)

        # ----
        # Negative likelihood
        # ----
        negative_likelihood = self.negative_likelihood(ts, T, log, 0)

        # ----
        # Return total likelihood

        if log:
            # TODO debug change
            return positive_likelihood + negative_likelihood
        else:
            return positive_likelihood * negative_likelihood

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
        raise NotImplementedError("Sampling not implemented for Conditional Inhomogenious Poisson Process.")
        step = 0
        time_samples = []
        dist_samples = []

        mu = self.transform_params()
        device = ts.time_points.device

        if rng is None:
            rng = torch.Generator()
        elif isinstance(rng, int):
            rng = torch.Generator().manual_seed(rng)

        for step in range(0, num_steps):
            # Intensity is constant
            u = torch.rand(size=(), generator=rng).item()

            # The rate of the superimposed process is the sum of the individual rates
            lamb = mu.sum()
            # We not solve for T* in u = CDF(T*) = 1 - exp(-lamb* * T* )
            # => T* = -log(1-u)/lamb
            t_star = -torch.log1p(-torch.tensor(u)).item() / lamb

            # Compute type-distribution at t_star
            probs = mu / lamb
            type_idx = torch.multinomial(probs, num_samples=1, generator=rng)

            time_samples.append(t_star)
            dist_samples.append(probs)

            T_tensor = torch.tensor([t_star], dtype=ts.time_points.dtype, device=device)

            new_time_points = torch.cat([ts.time_points, T_tensor], dim=0)
            new_event_types = torch.cat([ts.event_types, type_idx], dim=0)

            ts = MVEventData(new_time_points, new_event_types)

        return ts, time_samples, dist_samples


@dataclass
class SplineProcessParams:
    h_knots: torch.Tensor  # Shape: (D, num_knots)
    delta_ts: float  # Time between knots


class LinearSpline(torch.nn.Module):
    @staticmethod
    def interpolate(x_knots: torch.Tensor, y_knots: torch.Tensor, t: torch.Tensor):
        """Interpolates the splines at the given positions.

        Args:
            x_knots (torch.Tensor): Position of the splines. Shared between all splines. Shape: (K,)
            y_knots (torch.Tensor): Height of the splines for each knot in each spline. Shape: (D,K)
            t (torch.Tensor): Time-points where to interpolate the splines. Shape: (B,)

        Returns:
            interpolation (torch.Tensor): Interpolated splines at t. Shape: (B,D)
        """
        k, delta = LinearSpline._get_knot_info(x_knots, t)  # k: (B,), delta: (B,)

        h_k = y_knots[:, k]  # (D,B)

        delta_times = x_knots[1:] - x_knots[:-1]  # (K-1)
        delta_heights = y_knots[:, 1:] - y_knots[:, :-1]  # (K-1)
        slopes = delta_heights / delta_times  # (D, K-1)
        segment_slopes = F.pad(slopes, (0, 1), value=0.0)  # (D, K)

        slope_k = segment_slopes[:, k]  # (D,B)

        interpolation = h_k + slope_k * delta.unsqueeze(0)  # (D,B)

        return interpolation.T  # (B,D)

    @staticmethod
    def integrate(x_knots: torch.Tensor, y_knots: torch.Tensor, t: torch.Tensor):
        """Calculates the integral of the splines at the given positions.

        Args:
            x_knots (torch.Tensor): Position of the splines. Shared between all splines. Shape: (K,)
            y_knots (torch.Tensor): Height of the splines for each spline and each knot. Shape: (D,K)
            t (torch.Tensor): Time-points where up to where to integrate the splines (starting from 0). Shape: (B,)

        Returns:
            integral: (torch.Tensor): Integrated splines up to t. Shape: (B,D)
        """
        k, delta = LinearSpline._get_knot_info(x_knots=x_knots, t=t)  # (B,), (B,)

        delta_times = x_knots[1:] - x_knots[:-1]  # (K-1)
        delta_heights = y_knots[:, 1:] - y_knots[:, :-1]  # (K-1)
        slopes = delta_heights / delta_times  # (D, K-1)

        segment_areas = (
            y_knots[:, :-1] * delta_times + slopes / 2.0 * delta_times.unsqueeze(0) ** 2
        )  # integral_0^t ( h + slope * s)  ds, Shape: (D,K-1)

        prefix_sums = torch.cumsum(segment_areas, dim=1)  # Shape: (D,K-1)
        prefix_sums = F.pad(prefix_sums, (1, 0), value=0.0)  # (D, K)
        segment_slopes = F.pad(slopes, (0, 1), value=0.0)  # (D, K)

        # Extract info for current interval
        C_k = prefix_sums[:, k]  # (D,B)
        s_k = segment_slopes[:, k]  # (D,B)
        h_k = y_knots[:, k]  # (D,B)
        last_knot_loc = x_knots[k]  # (B,)
        d_t = t - last_knot_loc  # (B,)

        # Integral of previous segments + square_area + triangular area
        integral = C_k + h_k * d_t.unsqueeze(0) + s_k / 2 * delta.unsqueeze(0) ** 2  # (D,B)

        return integral.T  # (B,D)

    @staticmethod
    def inverse_integral(x_knots: torch.Tensor, y_knots: torch.Tensor, u: torch.Tensor):
        """Computes the inverse of the integral if defined (y_knots has to be strictly positive or negative).

        Given u, finds t such that u = int_0^t spline(x_knots, y_knots, s) ds.

        Args:
            x_knots (torch.Tensor): Position of the splines. Shared between all splines. Shape: (K,)
            y_knots (torch.Tensor): Height of the splines for each spline and each knot. Shape: (D,K)
            u (torch.Tensor): Target integral values. Shape: (B, D)

        Returns:
            t (torch.Tensor): The time values producing the desired u's. Shape: (B, D)
        """
        assert (y_knots > 0).all() or (y_knots < 0).all(), (
            "y_knots must be strictly positive or negative for invertibility"
        )

        D, K = y_knots.shape
        B = u.shape[0]

        # Compute segment properties
        delta_times = x_knots[1:] - x_knots[:-1]  # (K-1,)
        delta_heights = y_knots[:, 1:] - y_knots[:, :-1]  # (D, K-1)
        slopes = delta_heights / delta_times  # (D, K-1)

        segment_areas = y_knots[:, :-1] * delta_times + slopes / 2.0 * delta_times**2  # (D, K-1)

        prefix_sums = torch.cumsum(segment_areas, dim=1)  # (D, K-1)
        prefix_sums = F.pad(prefix_sums, (1, 0), value=0.0)  # (D, K) - integral from 0 to x_knots[k]

        segment_slopes = F.pad(slopes, (0, 1), value=0.0)  # (D, K)

        # For each (batch, dim), find which segment the target u falls into
        # u has shape (B, D), prefix_sums has shape (D, K)
        # Find k such that prefix_sums[d, k] <= u[b, d] < prefix_sums[d, k+1]
        k = torch.searchsorted(prefix_sums.unsqueeze(0).expand((B, D, K)), u.unsqueeze(2), right=True) - 1
        k = k.squeeze(2).T  # (B,D)

        # Extract segment info for each (d, b) pair
        # k has shape (D, B)
        C_k = torch.gather(prefix_sums, 1, k)  # (D, B) - cumulative integral up to knot k
        h_k = torch.gather(y_knots, 1, k)  # (D, B) - height at knot k
        s_k = torch.gather(segment_slopes, 1, k)  # (D, B) - slope in segment k
        x_k = x_knots[k]  # (D, B) - position of knot k

        # We need to solve:  h_k * delta + s_k/2 * delta^2 = u - C_k
        # Rearranging: s_k/2 * delta^2 + h_k * delta - (u - C_k) = 0
        #            = s_k/2 * delta^2 + h_k * delta - c = 0

        # Using quadratic formula: delta = (-h_k + sqrt(h_k^2 + 2*s_k*c)) / s_k

        c_val = u.T - C_k  # (D, B)

        # Handle two cases: non-zero slope (quadratic) and zero slope (linear)
        slope_is_zero = torch.abs(s_k) < 1e-10

        # For zero slope: h_k * delta = u - C_k => delta = (u - C_k) / h_k = c_val / h_k
        delta_linear = c_val / h_k  # (D, B)

        # For non-zero slope: solve quadratic
        # s_k/2 * delta^2 + h_k * delta + c_val = 0
        # delta = (-h_k + sqrt(h_k^2 + 2*s_k*c_val)) / s_k
        discriminant = h_k**2 + 2 * s_k * c_val  # (D, B)
        # Clamp to avoid numerical issues.
        discriminant = torch.clamp(discriminant, min=0.0)
        sqrt_disc = torch.sqrt(discriminant)

        # Safe division for quadratic case to avoid NaN in tensors. Can lead to problems, even when masked.
        # Learned that the hard way... Will be masked out later anyway.
        s_k_safe = torch.where(slope_is_zero, torch.ones_like(s_k), s_k)
        delta_quadratic = (-h_k + sqrt_disc) / s_k_safe
        # Select based on whether slope is zero
        delta = torch.where(slope_is_zero, delta_linear, delta_quadratic)  # (D, B)

        # Compute final t values
        t = x_k + delta  # (D, B)

        return t.T  # (B, D)

    @staticmethod
    def _get_knot_info(x_knots, t: torch.Tensor):
        """
        Calculates indices and relative offsets for time points.
        x_knots: Location of the knots (K,)
        t: (batch_size,)
        returns: k (indices, Shape:(B,)), delta (t - t_k, Shape:(B,))
        """

        k = torch.searchsorted(x_knots, t, right=True) - 1
        delta = t - x_knots[k]
        return k, delta


class SplinePoissonProcess(torch.nn.Module):
    def __init__(
        self,
        D: int,
        num_knots: int,
        delta_t: torch.Tensor | float,
        params: Optional[SplineProcessParams] = None,
        seed: Optional[int] = 42,
    ):
        super().__init__()

        if params is None:
            self.D = D
            self.num_knots = num_knots

            # Initialize heights randomly in log-space/softplus-space
            # Target range ~[0.1, 10.0]
            initial_h = torch.rand(D, num_knots) + 0.1  # [0-1] = 0-80 years.
            self.h_knots = torch.nn.Parameter(inverse_softplus(initial_h))

            if isinstance(delta_t, float) or delta_t.numel() == 1:
                self.knot_locs = torch.arange(num_knots) * delta_t
            else:
                self.knot_locs = torch.cumsum(delta_t, dim=0)

            assert self.knot_locs.shape[0] == self.h_knots.shape[1]

        else:
            self.h_knots = torch.nn.Parameter(inverse_softplus(params.h_knots))
            self.knot_locs = torch.cumsum(params.delta_ts, dim=0)
            self.D = params.h_knots.shape[0]
            self.num_knots = params.h_knots.shape[1]

        assert self.num_knots == self.h_knots.shape[1], "Number of knots mismatch"

    def transform_parameters(self):
        "Return the parameters in their acutal/constrained form."
        return F.softplus(self.h_knots)

    def get_heights(self):
        """Returns non-negative knot heights: Shape (D, num_knots)"""
        return self.transform_parameters()

    def intensity(self, t: torch.Tensor, ts: BatchedMVEventData):
        """
        lambda_d(t) = h_k + m_k * delta
        Returns: (batch_size, D)
        """
        # Expand T into 1-dimensional tensor if needed
        if t.dim() == 0:
            t = t.unsqueeze(0)

        h = self.get_heights()  # (D, K)

        intensity = LinearSpline.interpolate(self.knot_locs, h, t)  # (B,D)

        return intensity

    def cumulative_intensity(self, T: torch.Tensor, ts: BatchedMVEventData):
        """
        Calculates the integral from 0 to T for each dimension.
        Returns: (batch_size, D)
        """

        # Expand T into 1-dimensional tensor if needed
        if T.dim() == 0:
            T = T.unsqueeze(0)

        h = self.get_heights()  # (D,K)

        integral = LinearSpline.integrate(self.knot_locs, h, T)  # (B, D)

        return integral  # (B,D)

    def positive_likelihood(self, ts: BatchedMVEventData, log: bool = True):
        # Identify valid events
        valid_events = ts.event_types != -1  # (B, N)

        # We need intensity at every event time t_i for type d_i
        # This requires vectorized indexing over the spline
        # Flatten times to process through _get_knot_info
        flat_times = ts.time_points.flatten()  # (B*N)
        flat_events = ts.event_types.flatten()  # (B*N)

        all_intensities = self.intensity(flat_times, ts)  # (B*N, D)
        all_intensities = all_intensities.T.view(ts.shape[0], ts.shape[1], self.D)  # (B, N, D)

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

    def negative_likelihood(self, ts: BatchedMVEventData, T: torch.Tensor, log: bool = True):
        # Integral from 0 to T over all D dimensions
        integral = self.cumulative_intensity(T, ts)  # (B, D)
        sum_integral = torch.sum(integral, dim=1)  # (B,)

        if log:
            return -sum_integral
        else:
            return torch.exp(-sum_integral)

    def likelihood(self, ts: BatchedMVEventData, T: torch.Tensor, log: bool = True):
        pos = self.positive_likelihood(ts, log)
        neg = self.negative_likelihood(ts, T, log)
        return pos + neg if log else pos * neg

    def sample(self, ts: MVEventData, num_steps: int = 1, rng: Optional[torch.Generator] = None):
        """
        Sampling via Superposition + Inverse CDF Transform.

        Strategy:
        1. Build the marginal (superposition) process by summing intensities across all D dimensions
        2. Sample inter-arrival times from the marginal process using inverse transform
        3. At each sampled time, determine event type via categorical distribution weighted by intensities

        Args:
            ts (MVEventData): Conditioned past events.
            num_steps (int, optional): Number of steps for generation. Defaults to 1.
            rng (Optional[torch.Generator], optional): Random state. Defaults to None.

        Returns:
            ts: Updated MVEventData with new events appended.
            time_samples: List of sampled inter-arrival times.
            dist_samples: List of type probability distributions at each sampled time.
        """
        if rng is None:
            rng = torch.Generator(device=self.h_knots.device)

        curr_ts = ts  # (N,) can be None
        h = self.get_heights()  # (D, K)
        device = h.device

        time_samples = []
        dist_samples = []

        # Build the marginal intensity by summing across dimensions
        # h_marginal[k] = sum_d h[d, k]
        h_marginal = h.sum(dim=0, keepdim=True)  # (1, K)

        for step in range(num_steps):
            t_start = curr_ts.time_points.max() if curr_ts.time_points.numel() > 0 else torch.tensor(0.0, device=device)

            # Compute cumulative intensity of marginal process at t_start
            # Lambda_marginal(t) = sum_d Lambda_d(t) = integral of sum_d lambda_d(s) from 0 to t
            if t_start > 0:
                Lambda_start = LinearSpline.integrate(self.knot_locs, h_marginal, t_start.unsqueeze(0))  # (1, 1)
                Lambda_start = Lambda_start.squeeze()  # scalar
            else:
                Lambda_start = torch.tensor(0.0, device=device)

            # Sample u ~ Uniform(0, 1) and compute inverse of target cumulative intensity
            # u = 1 - exp(-(Lambda_target - Lambda_start)) => CDF inversion
            # 1 -u = exp(-(Lambda_target - Lambda_start)) => -log(1-u) = Lambda_target - Lambda_start
            u = torch.rand(size=(), generator=rng, device=device)
            Lambda_target = Lambda_start - torch.log(1 - u)

            # Invert: find t such that Lambda_marginal(t) = Lambda_target
            t_next = LinearSpline.inverse_integral(
                self.knot_locs,
                h_marginal,
                Lambda_target.unsqueeze(0).unsqueeze(0),  # (1, 1)
            )  # (1, 1)
            t_next = t_next.squeeze()  # scalar

            # Compute inter-arrival time
            delta_t = t_next - t_start
            time_samples.append(delta_t.item())

            # Determine event type: sample from categorical weighted by intensities at t_next
            intensities_at_t = LinearSpline.interpolate(self.knot_locs, h, t_next.unsqueeze(0))  # (1, D)
            intensities_at_t = intensities_at_t.squeeze(0)  # (D,)

            # Normalize to get probabilities
            probs = intensities_at_t / intensities_at_t.sum()
            dist_samples.append(probs.detach().clone())

            # Sample event type
            event_type = torch.multinomial(probs, num_samples=1, generator=rng)  # (1,)

            # Append new event to sequence
            new_time_points = torch.cat([curr_ts.time_points, t_next.unsqueeze(0)], dim=0)
            new_event_types = torch.cat([curr_ts.event_types, event_type], dim=0)

            curr_ts = MVEventData(new_time_points, new_event_types)

        return curr_ts, time_samples, dist_samples
