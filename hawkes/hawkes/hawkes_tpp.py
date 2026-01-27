# My own implementation of the Hawkes process
# %%
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from .tpps import TemporalPointProcess

from .event_utils import BatchedMVEventData
from .utils import inverse_softplus, LinearSpline

# %%
# ============================================================================
# Abstract Modular Architecture for Hawkes Processes
# ============================================================================


class BaselineIntensityModule(nn.Module, ABC):
    """
    Abstract base class for baseline intensity modules in Hawkes processes.

    A baseline module represents the exogenous intensity (spontaneous event rate)
    in the Hawkes process. It is independent of historical events.
    """

    def __init__(self, D: int):
        """
        Args:
            D: Number of event types (dimensions)
        """
        super().__init__()
        self.D = D

    @abstractmethod
    def transform_params(self) -> Tuple[torch.Tensor, ...]:
        """
        Return baseline parameters in their constrained form.

        Returns:
            Baseline intensity parameters. Shape: (D,)
        """
        pass

    @abstractmethod
    def intensity(self, t: torch.Tensor, ts: BatchedMVEventData) -> torch.Tensor:
        """
        Compute baseline intensity at time t.

        Args:
            t: Time points. Shape: (B,)
            ts: Historical events (for context, may not be used by all baselines). Shape: (B,N)

        Returns:
            Baseline intensity for each event type. Shape: (B,D)
        """
        pass

    @abstractmethod
    def cumulative_intensity(self, T: torch.Tensor, ts: BatchedMVEventData) -> torch.Tensor:
        """
        Compute the integral of baseline intensity from 0 to T.

        Args:
            T: End times. Shape: (B,)
            ts: Historical events (for context, may not be used by all baselines). Shape: (B,N)

        Returns:
            Cumulative baseline intensity for each event type. Shape: (B,D)
        """
        pass

    def positive_likelihood_intensities(self, ts: BatchedMVEventData) -> torch.Tensor:
        """
        Compute the positive part of the likelihood for the baseline intensities: the baseline intensities at event times for the correct class.

        This corresponds to the "data fit" term in the likelihood, measuring how well
        the model predicts the observed events.

        Can be overwritten to provide a far more efficient implementation that only computes the intensities of the "correct" classes at each time-point.

        Args:
            ts: Observed event sequences (batched, padded). Shape: (batch_size, seq_len)
            log: If True, return log-likelihood; otherwise return likelihood

        Returns:
            Baseline intensities for each event in batch. Shape: (batch_size, seq_len)

        """

        B = ts.time_points.shape[0]
        L = ts.time_points.shape[1]
        valid_events = ts.event_types != -1  # (B, L) Mask for valid events (not padding)

        # Gather intensities at event times
        intensities = torch.stack([self.intensity(ts.time_points[:, l], ts=ts) for l in range(L)], dim=1)  # (B, L, D)
        event_types = ts.event_types.unsqueeze(-1)  # (B, L, 1)
        intensities = torch.gather(intensities, dim=2, index=event_types.clamp(min=0)).squeeze(-1)  # (B, L)

        intensities[~valid_events] = 0.0  # Set invalid events to zero intensity

        return intensities  # (B, L)

    def negative_likelihood_intensities(self, T: torch.Tensor, ts: BatchedMVEventData) -> torch.Tensor:
        """
        Compute the negative part of the likelihood for the baseline intensities: the cumulative baseline intensities up to time T.

        This corresponds to the "penalty" term in the likelihood, accounting for the total intensity over time.

        Args:
            T: End times for each sequence in batch. Shape: (batch_size,)
            ts: Observed event sequences (batched, padded). Shape: (batch_size, seq_len)

        Returns:
            Cumulative baseline intensities for each sequence in batch. Shape: (batch_size, D)
        """
        return self.cumulative_intensity(T, ts)  # (B, D)


class KernelIntensityModule(nn.Module, ABC):
    """
    Abstract base class for kernel functions in Hawkes processes.

    A kernel module represents the endogenous component (self-exciting/self-inhibiting),
    capturing how past events influence future event intensity.
    """

    def __init__(self, D: int):
        """
        Args:
            D: Number of event types (dimensions)
        """
        super().__init__()
        self.D = D

    @abstractmethod
    def transform_params(self) -> Tuple[torch.Tensor, ...]:
        """
        Return kernel parameters in their constrained form.

        Returns:
            Tuple of kernel parameters (structure depends on specific kernel)
        """
        pass

    @abstractmethod
    def intensity(self, t: torch.Tensor, ts: BatchedMVEventData) -> torch.Tensor:
        """
        Compute kernel contribution to intensity at time t.

        Args:
            t: Time points. Shape: (B,)
            ts: Historical events. Shape: (B,N)

        Returns:
            Kernel intensity contribution for each event type. Shape: (B,D)
        """
        pass

    @abstractmethod
    def cumulative_intensity(self, T: torch.Tensor, ts: BatchedMVEventData) -> torch.Tensor:
        """
        Compute the integral of kernel intensity from 0 to T.

        Args:
            T: End times. Shape: (B,)
            ts: Historical events. Shape: (B,N)

        Returns:
            Cumulative kernel intensity for each event type. Shape: (B,D)
        """
        pass

    def positive_likelihood_intensities(self, ts: BatchedMVEventData) -> torch.Tensor:
        """
        Compute the positive part of the likelihood for the kernel intensities: the kernel intensities at event times for the correct class.

        This corresponds to the "data fit" term in the likelihood, measuring how well
        the model predicts the observed events.

        Can be overwritten to provide a far more efficient implementation that only computes the intensities of the "correct" classes at each time-point.

        Args:
            ts: Observed event sequences (batched, padded). Shape: (batch_size, seq_len)
            log: If True, return log-likelihood; otherwise return likelihood

        Returns:
            Baseline intensities for each event in batch. Shape: (batch_size, seq_len)

        """

        B = ts.time_points.shape[0]
        L = ts.time_points.shape[1]
        valid_events = ts.event_types != -1  # (B, L) Mask for valid events (not padding)

        # Gather intensities at event times
        intensities = torch.stack([self.intensity(ts.time_points[:, l], ts=ts) for l in range(L)], dim=1)  # (B, L, D)
        event_types = ts.event_types.unsqueeze(-1)  # (B, L, 1)
        intensities = torch.gather(intensities, dim=2, index=event_types).squeeze(-1)  # (B, L)

        intensities[~valid_events] = 0.0  # Set invalid events to zero intensity

        return intensities  # (B, L)

    def negative_likelihood_intensities(self, T: torch.Tensor, ts: BatchedMVEventData) -> torch.Tensor:
        """
        Compute the negative part of the likelihood for the kernel intensities: the cumulative kernel intensities up to time T.

        This corresponds to the "penalty" term in the likelihood, accounting for the total intensity over time.

        Args:
            T: End times for each sequence in batch. Shape: (batch_size,)
            ts: Observed event sequences (batched, padded). Shape: (batch_size, seq_len)

        Returns:
            Cumulative kernel intensities for each sequence in batch. Shape: (batch_size, D)
        """
        return self.cumulative_intensity(T, ts)  # (B, D)


class HawkesProcess(TemporalPointProcess, ABC):
    """
    Abstract base class for Hawkes processes with modular baseline and kernel.

    A Hawkes process combines:
    - A baseline module (exogenous intensity)
    - A kernel module (endogenous self-exciting component)

    The total intensity is: λ(t) = baseline(t) + kernel(t)
    """

    def __init__(
        self,
        baseline: BaselineIntensityModule,
        kernel: KernelIntensityModule,
        D: int,
        seed: Optional[int] = 42,
    ):
        """
        Args:
            baseline: BaselineIntensityModule instance
            kernel: KernelIntensityModule instance
            D: Number of event types
            seed: Random seed
        """
        super().__init__(D, seed, use_analytical_ci=True)
        self.baseline = baseline
        self.kernel = kernel

        assert self.baseline.D == D, "Baseline dimension mismatch"
        assert self.kernel.D == D, "Kernel dimension mismatch"

    def transform_params(self) -> Tuple[torch.Tensor, ...]:
        """
        Return all parameters in constrained form.

        Returns:
            (baseline_params, *kernel_params)
        """
        baseline_params = self.baseline.transform_params()
        kernel_params = self.kernel.transform_params()
        return (*baseline_params, *kernel_params)

    def intensity(self, t: torch.Tensor, ts: BatchedMVEventData) -> torch.Tensor:
        """
        Compute total intensity = baseline + kernel.

        Args:
            t: Time points. Shape: (B,)
            ts: Historical events. Shape: (B,N)

        Returns:
            Total intensity. Shape: (B,D)
        """
        baseline_intensity = self.baseline.intensity(t, ts)  # (B,D)
        kernel_intensity = self.kernel.intensity(t, ts)  # (B,D)
        return baseline_intensity + kernel_intensity

    def analytical_cumulative_intensity(self, T: torch.Tensor, ts: BatchedMVEventData) -> torch.Tensor:
        """
        Compute total cumulative intensity = baseline CI + kernel CI.

        Args:
            T: End times. Shape: (B,)
            ts: Historical events. Shape: (B,N)

        Returns:
            Total cumulative intensity. Shape: (B,D)
        """
        baseline_ci = self.baseline.cumulative_intensity(T, ts)  # (B,D)
        kernel_ci = self.kernel.cumulative_intensity(T, ts)  # (B,D)
        return baseline_ci + kernel_ci

    def positive_likelihood(self, ts: BatchedMVEventData, log: bool = True) -> torch.Tensor:
        """
        Compute the likelihood of observing the events (product of intensities at event times).

        This is the standard Hawkes process likelihood computation, shared across all variants.
        """
        # Identify valid (non-padded) events
        valid_events = ts.event_types != -1

        baseline_intensities = self.baseline.positive_likelihood_intensities(ts)  # (B,N)
        interaction_terms = self.kernel.positive_likelihood_intensities(ts)  # (B,N)

        intensities = baseline_intensities + interaction_terms  # (B,N)

        min_intensity = 1e-12
        if log:
            intensities_clamped = torch.clamp(intensities, min=min_intensity)
            log_intensities = torch.log(intensities_clamped)
            log_intensities[~valid_events] = 0.0
            positive_likelihood = torch.sum(log_intensities, dim=-1)  # Shape: (B,)
        else:
            intensities = torch.clamp(intensities, min=min_intensity)
            intensities[~valid_events] = 1.0
            positive_likelihood = torch.prod(intensities, dim=-1)  # Shape: (B,)

        # Mask out sequences with no events
        num_0 = valid_events.sum(dim=1) == 0
        positive_likelihood[num_0] = torch.tensor(0.0) if log else torch.tensor(1.0)

        return positive_likelihood


class InhibitiveHawkesProcess(TemporalPointProcess, ABC):
    """
    Abstract base class for Inhibitive Hawkes processes with modular baseline and kernel.

    A Hawkes process combines:
    - A baseline module (exogenous intensity)
    - A kernel module (endogenous self-exciting component)

    The total intensity is: λ(t) = baseline(t) + kernel(t)
    """

    def __init__(
        self,
        positive_constraint: Callable,
        baseline: BaselineIntensityModule,
        kernel: KernelIntensityModule,
        D: int,
        seed: Optional[int] = 42,
    ):
        """
        Args:
            baseline: BaselineIntensityModule instance
            kernel: KernelIntensityModule instance
            D: Number of event types
            seed: Random seed
        """
        super().__init__(D, seed, use_analytical_ci=False)
        self.baseline = baseline
        self.kernel = kernel
        self.positivity_constraint = positive_constraint

        assert self.baseline.D == D, "Baseline dimension mismatch"
        assert self.kernel.D == D, "Kernel dimension mismatch"

    def transform_params(self) -> Tuple[torch.Tensor, ...]:
        """
        Return all parameters in constrained form.

        Returns:
            (baseline_params, *kernel_params)
        """
        baseline_params = self.baseline.transform_params()
        kernel_params = self.kernel.transform_params()
        return (*baseline_params, *kernel_params)

    def intensity(self, t: torch.Tensor, ts: BatchedMVEventData) -> torch.Tensor:
        """
        Compute total intensity = baseline + kernel.

        Args:
            t: Time points. Shape: (B,)
            ts: Historical events. Shape: (B,N)

        Returns:
            Total intensity. Shape: (B,D)
        """
        baseline_intensity = self.baseline.intensity(t, ts)  # (B,D)
        kernel_intensity = self.kernel.intensity(t, ts)  # (B,D)
        return self.positivity_constraint(baseline_intensity + kernel_intensity)

    def positive_likelihood(self, ts: BatchedMVEventData, log: bool = True) -> torch.Tensor:
        """
        Compute the likelihood of observing the events (product of intensities at event times).

        This is the standard Hawkes process likelihood computation, shared across all variants.
        """
        # Identify valid (non-padded) events
        valid_events = ts.event_types != -1

        baseline_intensities = self.baseline.positive_likelihood_intensities(ts)  # (B,N)
        interaction_terms = self.kernel.positive_likelihood_intensities(ts)  # (B,N)

        intensities = self.positivity_constraint(baseline_intensities + interaction_terms)  # (B,N)

        min_intensity = 1e-12
        if log:
            intensities_clamped = torch.clamp(intensities, min=min_intensity)
            log_intensities = torch.log(intensities_clamped)
            log_intensities[~valid_events] = 0.0
            positive_likelihood = torch.sum(log_intensities, dim=-1)  # Shape: (B,)
        else:
            intensities = torch.clamp(intensities, min=min_intensity)
            intensities[~valid_events] = 1.0
            positive_likelihood = torch.prod(intensities, dim=-1)  # Shape: (B,)

        # Mask out sequences with no events
        num_0 = valid_events.sum(dim=1) == 0
        positive_likelihood[num_0] = torch.tensor(0.0) if log else torch.tensor(1.0)

        return positive_likelihood


# ============================================================================
# Concrete Implementations of Baseline and Kernel Modules
# ============================================================================


@dataclass
class ConstantBaselineParams:
    mu: torch.Tensor  # Shape: (D,)


class ConstantBaselineModule(BaselineIntensityModule):
    """Constant (time-independent) baseline intensity."""

    def __init__(self, D: int, params: Optional[ConstantBaselineParams] = None):
        """
        Args:
            D: Number of event types
            mu: Initial baseline intensity values. If None, initialized randomly.
        """
        super().__init__(D)

        if params is None:
            mu = torch.rand(D) * (10 - 1 / 12) + 1 / 12
        else:
            mu = params.mu
            assert mu.shape[0] == D

        self.mu = nn.Parameter(inverse_softplus(mu))

    def transform_params(self) -> Tuple[torch.Tensor]:
        """Return constrained baseline parameters."""
        return (F.softplus(self.mu),)

    def intensity(self, t: torch.Tensor, ts: BatchedMVEventData) -> torch.Tensor:
        """Baseline intensity is constant regardless of time or history."""
        (mu,) = self.transform_params()  # Shape: (D,)
        batch_size = t.shape[0]
        return mu.unsqueeze(0).expand(batch_size, self.D)  # Shape: (B,D)

    def cumulative_intensity(self, T: torch.Tensor, ts: BatchedMVEventData) -> torch.Tensor:
        """Cumulative intensity: mu * T."""
        (mu,) = self.transform_params()  # Shape: (D,)
        return mu.unsqueeze(0) * T.unsqueeze(1)  # Shape: (B,D)


@dataclass
class LinearBaselineParams:
    mu: torch.Tensor  # Shape: (D,)
    slope: torch.Tensor  # Shape: (D,)


class LinearBaselineModule(BaselineIntensityModule):
    def __init__(
        self,
        params: Optional[LinearBaselineParams] = None,
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
        super().__init__(D)

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

    def positive_likelihood_intensities(
        self,
        ts: BatchedMVEventData,
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
        masked_intensities = intensities * valid_events.float()
        return masked_intensities


@dataclass
class SplineBaselineParams:
    h_knots: torch.Tensor  # Shape: (D, num_knots)
    delta_ts: torch.Tensor  # Shape: (num_knots - 1,)


class SplineBaselineModule(BaselineIntensityModule):
    """Spline-based baseline intensity module."""

    def __init__(
        self,
        D: int,
        num_knots: int,
        delta_t: torch.Tensor | float,
        params: Optional[SplineBaselineParams] = None,
        seed: Optional[int] = 42,
    ):
        super().__init__(D)

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

    def cumulative_intensity(self, T: torch.Tensor, ts: BatchedMVEventData):
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

    def positive_likelihood_intensities(self, ts: BatchedMVEventData):
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

        event_intensities = event_intensities.masked_fill(~valid_events, 0.0)

        return event_intensities


@dataclass
class ExpKernelParams:
    alpha: torch.Tensor  # Shape: (D,D)
    beta: torch.Tensor  # Shape: (D,D)


# UnconstrainedExpKernelParams is an alias for ExpKernelParams
# Both have the same structure, but the semantics differ:
# - ExpKernelParams: alpha values should be positive (for excitatory processes)
# - UnconstrainedExpKernelParams: alpha can be positive or negative (for inhibitive processes)
UnconstrainedExpKernelParams = ExpKernelParams


class ExpKernelModule(KernelIntensityModule):
    """Exponential kernel module for self-exciting Hawkes processes.

    Alpha values are constrained to be positive via softplus.
    """

    def __init__(
        self,
        D: int,
        params: Optional[ExpKernelParams] = None,
        seed: Optional[int] = 42,
    ):
        """
        Args:
            D: Number of event types
            params: Kernel parameters. If None, initialized randomly.
            seed: Random seed
        """
        super().__init__(D)

        generator = torch.Generator()
        if seed is not None:
            generator = generator.manual_seed(seed)
        self.generator = generator

        if params is None:
            alpha, beta = self._init_random_params(D, generator)
        else:
            alpha = params.alpha
            beta = params.beta

        self._init_parameters(alpha, beta)

    def _init_random_params(self, D: int, generator: torch.Generator) -> Tuple[torch.Tensor, torch.Tensor]:
        """Initialize random alpha and beta values."""
        # Random decay rates: effect decays with random time constants
        ratios_every_years = torch.rand(size=(D, D), generator=generator) * (10 - 1 / 12) + 1 / 12
        beta = 1 / ratios_every_years

        # Random amplitude, then rescale for target branching ratio
        alphas = torch.rand(size=(D, D), generator=generator) * 100 + 1
        max_eig = torch.max(torch.abs(torch.linalg.eigvals(alphas)))
        targeted_branching_ratio = 0.5
        alpha = (1 / max_eig) * targeted_branching_ratio * alphas

        return alpha, beta

    def _init_parameters(self, alpha: torch.Tensor, beta: torch.Tensor):
        """Initialize the learnable parameters. Override in subclasses for different constraints."""
        self.log_alpha = nn.Parameter(inverse_softplus(alpha))
        self.log_beta = nn.Parameter(inverse_softplus(beta))

    def transform_params(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return constrained parameters (alpha, beta). Override in subclasses."""
        return F.softplus(self.log_alpha), F.softplus(self.log_beta)

    def intensity(self, t: torch.Tensor, ts: BatchedMVEventData) -> torch.Tensor:
        """
        Compute kernel contribution to intensity.

        λ_kernel_d(t) = Σ_{t_i < t} Σ_j α_{d,j} β_{d,j} exp(-β_{d,j} (t - t_i))
        """
        alpha, beta = self.transform_params()

        # Only include events before time t
        valid_mask = ts.event_types != -1  # Shape: (B,N)
        past_event_mask = ts.time_points < t.unsqueeze(1)  # Shape: (B,N)
        valid_mask = valid_mask & past_event_mask

        # Clamp event_types to avoid negative indexing for padded entries (-1)
        clamped_types = ts.event_types.clamp(min=0)
        relevant_alphas = alpha[:, clamped_types]  # Shape: (D,B,N)
        relevant_betas = beta[:, clamped_types]  # Shape: (D,B,N)

        delta_t = t.unsqueeze(0).unsqueeze(2) - ts.time_points.unsqueeze(0)  # Shape: (1,B,N)
        delta_t = torch.where(valid_mask.unsqueeze(0), delta_t, torch.zeros_like(delta_t))

        contributions = relevant_alphas * relevant_betas * torch.exp(-relevant_betas * delta_t)  # Shape: (D,B,N)
        contributions = contributions.masked_fill(~valid_mask.unsqueeze(0), 0)
        contributions = torch.sum(contributions, dim=2)  # Shape: (D,B)

        return contributions.T  # Shape: (B,D)

    def cumulative_intensity(self, T: torch.Tensor, ts: BatchedMVEventData) -> torch.Tensor:
        """
        Compute cumulative kernel intensity from 0 to T.

        Λ_kernel_d(T) = Σ_{t_i < T} Σ_j α_{d,j} (1 - exp(-β_{d,j} (T - t_i)))
        """

        valid_mask = ts.event_types != -1  # Shape: (B,N)
        past_event_mask = ts.time_points < T.unsqueeze(1)  # Shape: (B,N)
        valid_mask = valid_mask & past_event_mask

        alpha, beta = self.transform_params()

        integral = torch.zeros(self.D, T.shape[0], device=T.device, dtype=T.dtype)

        if len(ts) > 0:
            # Vectorized multivariate implementation
            delta_t = (T.unsqueeze(1) - ts.time_points).unsqueeze(0)  # Shape: (1,B,N)
            delta_t = torch.where(valid_mask.unsqueeze(0), delta_t, torch.zeros_like(delta_t))

            # Clamp event_types to avoid negative indexing for padded entries (-1)
            clamped_types = ts.event_types.clamp(min=0)
            relevant_alpha = alpha[:, clamped_types]  # Shape: (D,B,N)
            relevant_beta = beta[:, clamped_types]  # Shape: (D,B,N)
            contributions = relevant_alpha * (1 - torch.exp(-relevant_beta * delta_t))  # Shape: (D,B,N)
            contributions = contributions.masked_fill(~valid_mask.unsqueeze(0), 0)
            integral = integral + torch.sum(contributions, dim=2)  # Shape: (D,B)

        integral = integral.T
        return integral

    def positive_likelihood_intensities(self, ts: BatchedMVEventData) -> torch.Tensor:
        """
        Efficiently compute the pairwise interaction terms for all events in ts, using a lower triangular matrix approach.

        Args:
            ts (BatchedMVEventData): Batch of past events

        Returns:
            torch.Tensor: Tensor of positive likelihood intensities
        """
        # Uses a lower triangular matrix to compute all intensities at once

        # Shape ts: [batch_size, len]
        # ts arrays are right padded. -1 for type, the time points just have larger numbers

        # Identify valid (non-padded) events
        valid_events = ts.event_types != -1
        valid_event_mask = valid_events.unsqueeze(2) & valid_events.unsqueeze(1)

        # Computes the time-difference matrix T.
        # $T_{b,i,j} = t_b,i - t_b,j$ if $i > j$ else 0
        time_diffs = ts.time_points.unsqueeze(2) - ts.time_points.unsqueeze(1)  # Shape: (B, N, N)
        # Set time_diffs to 0, where we have interactions with padded events (can results in +-inf or NaN).
        time_diffs_safe = torch.where(valid_event_mask, time_diffs, torch.zeros_like(time_diffs))

        # Get receiver (j) types: (B, N) -> (B, N, 1)
        # Clamp to avoid negative indexing for padded entries (-1)
        clamped_types = ts.event_types.clamp(min=0)
        receiver_types = clamped_types.unsqueeze(2)

        # Get trigger (i) types: (B, N) -> (B, 1, N)
        trigger_types = clamped_types.unsqueeze(1)

        # Shape: D, (D,D), (D,D)
        alpha, beta = self.transform_params()

        # Create a matrix of shape (B, N, N) where each entry (b,i,j) corresponds to alpha_{receiver_types[b,i], trigger_types[b,j]}
        alpha_matrix = alpha[receiver_types, trigger_types]  # Shape: (B, N, N)
        beta_matrix = beta[receiver_types, trigger_types]  # Shape: (B, N, N)

        # Compute the kernel values for all time differences
        # We include clamping to avoid underflows.
        # Formula: alpha * beta * exp(-beta * dt)
        exponent = -beta_matrix * time_diffs_safe
        # Clamp exponent to avoid overflow/underflow. Should not happen, just to be sure.
        exponent = torch.clamp(exponent, min=-20, max=0)
        exp_kernel = torch.exp(exponent)
        interaction_terms = alpha_matrix * beta_matrix * exp_kernel  # Shape: (B,N, N)

        interaction_terms = torch.tril(
            interaction_terms, diagonal=-1
        )  # Zero out diagonal and above, to ensure causality. This works also with batched matrices.

        # Mask out invalid events.
        interaction_terms[~valid_event_mask] = 0.0

        return interaction_terms.sum(dim=2)  # Shape: (B,N)


class UnconstrainedExpKernelModule(ExpKernelModule):
    """
    Exponential kernel module for inhibitive Hawkes processes.

    Inherits from ExpKernelModule but with unconstrained alpha (can be negative for inhibition).
    Beta remains positive (decay rate).

    Kernel function: φ_{d,j}(Δt) = α_{d,j} β_{d,j} exp(-β_{d,j} Δt)

    When α < 0, past events inhibit (reduce) future intensity.
    When α > 0, past events excite (increase) future intensity.
    """

    def _init_random_params(self, D: int, generator: torch.Generator) -> Tuple[torch.Tensor, torch.Tensor]:
        """Initialize random alpha (centered around 0) and positive beta values."""
        # Random decay rates: effect decays with random time constants
        ratios_every_years = torch.rand(size=(D, D), generator=generator) * (10 - 1 / 12) + 1 / 12
        beta = 1 / ratios_every_years

        # Random alpha - unconstrained, can be positive or negative
        # Initialize with small values centered around 0
        alpha = torch.randn(size=(D, D), generator=generator) * 0.1

        return alpha, beta

    def _init_parameters(self, alpha: torch.Tensor, beta: torch.Tensor):
        """Initialize parameters with unconstrained alpha."""
        # Alpha is unconstrained (stored directly)
        self.alpha = nn.Parameter(alpha)
        # Beta must be positive (use softplus constraint)
        self.log_beta = nn.Parameter(inverse_softplus(beta))

    def transform_params(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return parameters: unconstrained alpha, positive beta."""
        return self.alpha, F.softplus(self.log_beta)


class ExpKernelHawkesProcess(HawkesProcess):
    """
    Reimplementation of Hawkes process with exponential kernel using modular architecture.

    This version uses the HawkesProcess base class with ConstantBaselineModule and ExpKernelModule.
    """

    def __init__(
        self,
        D: int,
        baseline_params: Optional[ConstantBaselineParams] = None,
        kernel_params: Optional[ExpKernelParams] = None,
        seed: Optional[int] = 42,
    ):
        """
        Initialize exponential kernel Hawkes process.

        Args:
            D: Number of event types
            baseline_mu: Baseline intensity (D,). If None, initialized randomly.
            alpha: Branching ratio matrix (D,D). If None, initialized randomly.
            beta: Decay rate matrix (D,D). If None, initialized randomly.
            seed: Random seed
        """
        baseline = ConstantBaselineModule(D, baseline_params)
        kernel = ExpKernelModule(D, kernel_params, seed)

        super().__init__(baseline, kernel, D, seed)


class LinearBaselineExpKernelHawkesProcess(HawkesProcess):
    """
    Reimplementation of Hawkes process with exponential kernel using modular architecture.

    This version uses the HawkesProcess base class with ConstantBaselineModule and ExpKernelModule.
    """

    def __init__(
        self,
        D: int,
        baseline_params: Optional[LinearBaselineParams] = None,
        kernel_params: Optional[ExpKernelParams] = None,
        seed: Optional[int] = 42,
    ):
        """
        Initialize exponential kernel Hawkes process with a linear function baseline intensity

        Args:
            D: Number of event types
            baseline_params: If None, initialized randomly.
            kernel_params: If None, initialized randomly.
            seed: Random seed
        """
        baseline = LinearBaselineModule(baseline_params, D, seed)
        kernel = ExpKernelModule(D, kernel_params, seed)

        super().__init__(baseline, kernel, D, seed)


class SplineBaselineExpKernelHawkesProcess(HawkesProcess):
    """
    Hawkes process with spline-based baseline intensity and exponential kernel.

    This version uses the HawkesProcess base class with SplineBaselineModule and ExpKernelModule.
    """

    def __init__(
        self,
        D: int,
        num_knots: int,
        delta_t: torch.Tensor | float,
        baseline_params: Optional[SplineBaselineParams] = None,
        kernel_params: Optional[ExpKernelParams] = None,
        seed: Optional[int] = 42,
    ):
        """
        Initialize spline-based baseline exponential kernel Hawkes process.

        Args:
            D: Number of event types
            baseline_knots: Knots for spline baseline (num_knots,). If None, initialized randomly.
            baseline_coefs: Coefficients for spline baseline (D, num_coefs). If None, initialized randomly.
            alpha: Branching ratio matrix (D,D). If None, initialized randomly.
            beta: Decay rate matrix (D,D). If None, initialized randomly.
            seed: Random seed
        """
        baseline = SplineBaselineModule(D, num_knots=num_knots, delta_t=delta_t, params=baseline_params, seed=seed)
        kernel = ExpKernelModule(D, kernel_params, seed)

        super().__init__(baseline, kernel, D, seed)


class NumericalSplineBaselineExpKernelHawkesProcess(HawkesProcess):
    """
    Hawkes process with spline-based baseline intensity and exponential kernel.

    This version uses the HawkesProcess base class with SplineBaselineModule and ExpKernelModule.
    """

    def __init__(
        self,
        D: int,
        num_knots: int,
        delta_t: torch.Tensor | float,
        baseline_params: Optional[SplineBaselineParams] = None,
        kernel_params: Optional[ExpKernelParams] = None,
        seed: Optional[int] = 42,
    ):
        """
        Initialize spline-based baseline exponential kernel Hawkes process.

        Args:
            D: Number of event types
            baseline_knots: Knots for spline baseline (num_knots,). If None, initialized randomly.
            baseline_coefs: Coefficients for spline baseline (D, num_coefs). If None, initialized randomly.
            alpha: Branching ratio matrix (D,D). If None, initialized randomly.
            beta: Decay rate matrix (D,D). If None, initialized randomly.
            seed: Random seed
        """
        baseline = SplineBaselineModule(D, num_knots=num_knots, delta_t=delta_t, params=baseline_params, seed=seed)
        kernel = ExpKernelModule(D, kernel_params, seed)

        super().__init__(baseline, kernel, D, seed)

        self.use_analytical_ci = False  # Use numerical CI from base class
        self.ci_num_points = 10
        self.ci_integration_method = "mc_trapezoidal"


class SoftplusConstExpIHawkesProcess(InhibitiveHawkesProcess):
    """
    Inhibitive Hawkes process with constant baseline and unconstrained exponential kernel.

    Uses softplus to ensure non-negative total intensity: λ(t) = softplus(μ + kernel(t))

    The kernel alpha values are unconstrained, allowing:
    - Positive alpha: excitation (past events increase future intensity)
    - Negative alpha: inhibition (past events decrease future intensity)
    """

    def __init__(
        self,
        D,
        baseline_params: Optional[ConstantBaselineParams],
        kernel_params: Optional[UnconstrainedExpKernelParams],
        seed: Optional[int] = 42,
    ):
        baseline = ConstantBaselineModule(D, baseline_params)
        kernel = UnconstrainedExpKernelModule(D, kernel_params, seed)
        super().__init__(F.softplus, baseline, kernel, D, seed)


class SoftplusSplineExpIHawkesProcess(InhibitiveHawkesProcess):
    """
    Inhibitive Hawkes process with spline baseline and unconstrained exponential kernel.

    Uses softplus to ensure non-negative total intensity: λ(t) = softplus(spline(t) + kernel(t))

    The kernel alpha values are unconstrained, allowing:
    - Positive alpha: excitation (past events increase future intensity)
    - Negative alpha: inhibition (past events decrease future intensity)
    """

    def __init__(
        self,
        D: int,
        num_knots: int,
        delta_t: torch.Tensor | float,
        baseline_params: Optional[SplineBaselineParams] = None,
        kernel_params: Optional[UnconstrainedExpKernelParams] = None,
        seed: Optional[int] = 42,
    ):
        """
        Initialize spline-based baseline inhibitive Hawkes process.

        Args:
            D: Number of event types
            num_knots: Number of knots for the spline baseline
            delta_t: Spacing between knots (float for uniform, tensor for variable)
            baseline_params: Spline baseline parameters. If None, initialized randomly.
            kernel_params: Kernel parameters. If None, initialized randomly.
            seed: Random seed
        """
        baseline = SplineBaselineModule(D, num_knots=num_knots, delta_t=delta_t, params=baseline_params, seed=seed)
        kernel = UnconstrainedExpKernelModule(D, kernel_params, seed)
        super().__init__(F.softplus, baseline, kernel, D, seed)


# %%


# @dataclass
# class ExpKernelParams:
#     """
#     Parameters for a Hawkes Process with Exponential Kernel.
#     Shapes have to be consisttent: (D,), (D,D) and (D,D).
#     For alpha and beta the first dimension is the receiving event type, the second the triggering event type.
#     --> alpha[i,j] scales the impact of an event of type j on the intensity of events of type i.
#     """

#     mu: torch.Tensor
#     alpha: torch.Tensor
#     beta: torch.Tensor


# class ExpKernelMVHawkesProcess(TemporalPointProcess):
#     def __init__(
#         self,
#         params: Optional[ExpKernelParams] = None,
#         D: Optional[int] = None,
#         seed: Optional[int] = 42,
#     ):
#         """Initialize the MultivariateHawkes process.

#         Args:
#             params: Hawkes process parameters or None to initialize randomly
#             D: Event dimension (required if params is None)
#             seed: random seed for initialization
#         """
#         super().__init__(D, seed, True)

#         generator = torch.Generator()
#         if seed is not None:
#             generator = generator.manual_seed(seed)

#         self.generator = generator

#         if params is None and D is None:
#             raise ValueError("Either params or D (number of dimensions) must be provided.")

#         if params is None:
#             # Decay rates: Effect goes down a factor of e every x steps.
#             # Our time scale is years, so maybe we can init it sensibly, between ratioing every month to 10 years.
#             ratios_every_years = torch.rand(size=(D, D)) * (10 - 1 / 12) + 1 / 12  # Random from one month to 10 years.
#             betas = 1 / ratios_every_years

#             # Amplitude is alpha*beta. Therefore branching ratio is max_abs_eig alpha.
#             targeted_branching_ratio = 0.5
#             alphas = torch.rand(size=(D, D)) * 100 + 1
#             max_eig = torch.max(torch.abs(torch.linalg.eigvals(alphas)))
#             alphas = (1 / max_eig) * targeted_branching_ratio * alphas

#             # Finally init mu. Not sure what to do here.
#             mu = torch.rand(size=(D,)) * (10 - 1 / 12) + 1 / 12

#             self.mu = torch.nn.Parameter(inverse_softplus(mu))
#             self.log_alpha = torch.nn.Parameter(inverse_softplus(alphas))
#             self.log_beta = torch.nn.Parameter(inverse_softplus(betas))
#             self.ensure_stability(radius=0.55)  # Limit spectral radius of alpha/beta matrix to be < 1.

#         else:
#             if D is None:
#                 D = params.mu.shape[0]

#             assert params.mu.shape[0] == D, "Dimension mismatch in mu"
#             assert params.alpha.shape == (D, D), "Dimension mismatch in alpha"
#             assert params.beta.shape == (D, D), "Dimension mismatch in beta"
#             self.mu = torch.nn.Parameter(inverse_softplus(params.mu))
#             self.log_alpha = torch.nn.Parameter(inverse_softplus(params.alpha))
#             self.log_beta = torch.nn.Parameter(inverse_softplus(params.beta))
#             stable, max_eig = self.check_stability()
#             if not stable:
#                 print(f"Warning: Provided parameters are unstable. Spectral radius: {max_eig} >= 1.")

#     def check_stability(self):
#         # Check if the Hawkes process is stable (spectral radius of alpha < 1)
#         # For exponential kernels, alpha is the branching ratio matrix.
#         _, alpha, _ = self.transform_params()
#         kernel_matrix = alpha  # Shape: (D,D)
#         eigenvalues = torch.linalg.eigvals(kernel_matrix)
#         spectral_radius = torch.max(torch.abs(eigenvalues)).item()
#         return spectral_radius < 1, spectral_radius

#     def ensure_stability(self, radius: float = 1.0):
#         # Compute the spectral radius
#         _, alpha, _ = self.transform_params()
#         kernel_matrix = alpha  # Shape: (D,D)
#         eigenvalues = torch.linalg.eigvals(kernel_matrix)
#         spectral_radius = torch.max(torch.abs(eigenvalues)).item()

#         if spectral_radius < radius:
#             return

#         self.log_alpha.data = inverse_softplus(alpha / ((1 / radius) * spectral_radius) - 1e-7)

#     def get_baserate_amplitude_decay(self):
#         mu, alpha, beta = self.transform_params()

#         amplitude = alpha * beta
#         decay = beta

#         return mu, amplitude, decay

#     def transform_params(self, mu=True, alpha=True, beta=True):
#         # Constrain parameters alpha and beta and mu to be positive

#         parameters = []

#         if mu:
#             parameters += [F.softplus(self.mu)]
#         if alpha:
#             parameters += [F.softplus(self.log_alpha)]
#         if beta:
#             parameters += [F.softplus(self.log_beta)]

#         return tuple(parameters)

#     def _unb_exp_kernel(self, delta_t: torch.Tensor, event_types: torch.Tensor):
#         # Warning: Unbatched!
#         # Returns the exponential kernel values: alpha * beta * exp(-beta * delta_t)
#         # delta_t: Shape (N,), event_types: Shape (N,)
#         # Returns: Shape (D, N)
#         _, _alpha, _beta = self.transform_params()  # Shape: (D,D) and (D,D)
#         alpha = _alpha[:, event_types].T  # Shape: (N,D)
#         beta = _beta[:, event_types].T  # Shape: (N,D)

#         delta_t = delta_t.unsqueeze(-1)  # Shape: (N,1)

#         return (alpha * beta * torch.exp(-beta * delta_t)).T  # Shape: (D,N)

#     def intensity(self, t: torch.Tensor, ts: BatchedMVEventData):
#         """
#         Computes the intensity at time t for each sequence in the batch.

#         :param t: Time points. Shape: (B,)
#         :type t: torch.Tensor
#         :param ts: Batched event data. Shape: (B,N)
#         :type ts: BatchedMVEventData
#         """

#         # lambda_d(t) = mu_d + \sum_{t_i < t} \sum_d \mathbb{1}(m_i == d) \phi_{d,m_i}(t-t_i)

#         mu, alpha, beta = self.transform_params()

#         # Only include events before time t
#         valid_mask = ts.event_types != -1  # Shape: (B,N)
#         past_event_mask = ts.time_points < t.unsqueeze(1)  # Shape: (B,N)
#         valid_mask = valid_mask & past_event_mask

#         # Clamp event_types to avoid negative indexing for padded entries (-1)
#         clamped_types = ts.event_types.clamp(min=0)
#         relevant_alphas = alpha[:, clamped_types]  # Shape: (D,B,N)
#         relevant_betas = beta[:, clamped_types]  # Shape: (D,B,N)

#         delta_t = t.unsqueeze(0).unsqueeze(2) - ts.time_points.unsqueeze(0)  # Shape: (1,B,N)
#         delta_t = torch.where(valid_mask.unsqueeze(0), delta_t, torch.zeros_like(delta_t))

#         contributions = relevant_alphas * relevant_betas * torch.exp(-relevant_betas * delta_t)  # Shape: (D,B,N)
#         contributions = contributions.masked_fill(~valid_mask.unsqueeze(0), 0)

#         intensity = mu.unsqueeze(1) + torch.sum(contributions, dim=2)  # Shape: (D,B)

#         return intensity.T  # Shape: (B,D)

#     def analytical_cumulative_intensity(self, T: torch.Tensor, ts: BatchedMVEventData):
#         """
#         Computes the cumulative intensity function from t=0 up to T.
#         To obtain the cumulative intensity from t_n up to T, consider subtracting the CIs.

#         :param T: Time points. Shape: (B,)
#         :type T: torch.Tensor
#         :param ts: Batched event data. Shape: (B,N)
#         :type ts: BatchedMVEventData
#         """

#         valid_mask = ts.event_types != -1  # Shape: (B,N)
#         future_event_mask = ts.time_points >= T.unsqueeze(1)  # Shape: (B,N)

#         valid_mask = valid_mask & ~future_event_mask

#         # DONE if T <= ts.time_points.max(), we include future events with a negative effect.
#         # This is definitly a bug. Should be fixed now...

#         mu, alpha, beta = self.transform_params()

#         integral = mu.unsqueeze(1) * T.unsqueeze(0)  # Shape: (D,B)

#         if len(ts) > 0:
#             # Vectorized multivariate implementation
#             delta_t = (T.unsqueeze(1) - ts.time_points).unsqueeze(0)  # Shape: (1,B,N)
#             delta_t = torch.where(valid_mask.unsqueeze(0), delta_t, torch.zeros_like(delta_t))

#             # Clamp event_types to avoid negative indexing for padded entries (-1)
#             clamped_types = ts.event_types.clamp(min=0)
#             relevant_alpha = alpha[:, clamped_types]  # Shape: (D,B,N)
#             relevant_beta = beta[:, clamped_types]  # Shape: (D,B,N)
#             contributions = relevant_alpha * (1 - torch.exp(-relevant_beta * delta_t))  # Shape: (D,B,N)
#             contributions = contributions.masked_fill(~valid_mask.unsqueeze(0), 0)
#             integral = integral + torch.sum(contributions, dim=2)  # Shape: (D,B)
#         integral = integral.T
#         return integral

#     def positive_likelihood(
#         self,
#         ts: BatchedMVEventData,
#         log: bool = True,
#     ):
#         # Uses a lower triangular matrix to compute all intensities at once

#         # Shape ts: [batch_size, len]
#         # ts arrays are right padded. -1 for type, the time points just have larger numbers

#         # Identify valid (non-padded) events
#         valid_events = ts.event_types != -1
#         valid_event_mask = valid_events.unsqueeze(2) & valid_events.unsqueeze(1)

#         # Computes the time-difference matrix T.
#         # $T_{b,i,j} = t_b,i - t_b,j$ if $i > j$ else 0
#         time_diffs = ts.time_points.unsqueeze(2) - ts.time_points.unsqueeze(1)  # Shape: (B, N, N)
#         # Set time_diffs to 0, where we have interactions with padded events (can results in +-inf or NaN).
#         time_diffs_safe = torch.where(valid_event_mask, time_diffs, torch.zeros_like(time_diffs))

#         # Get receiver (j) types: (B, N) -> (B, N, 1)
#         # Clamp to avoid negative indexing for padded entries (-1)
#         clamped_types = ts.event_types.clamp(min=0)
#         receiver_types = clamped_types.unsqueeze(2)

#         # Get trigger (i) types: (B, N) -> (B, 1, N)
#         trigger_types = clamped_types.unsqueeze(1)

#         # Shape: D, (D,D), (D,D)
#         mu, alpha, beta = self.transform_params()

#         # Create a matrix of shape (B, N, N) where each entry (b,i,j) corresponds to alpha_{receiver_types[b,i], trigger_types[b,j]}
#         alpha_matrix = alpha[receiver_types, trigger_types]  # Shape: (B, N, N)
#         beta_matrix = beta[receiver_types, trigger_types]  # Shape: (B, N, N)

#         # Compute the kernel values for all time differences
#         # We include clamping to avoid underflows.
#         # Formula: alpha * beta * exp(-beta * dt)
#         exponent = -beta_matrix * time_diffs_safe
#         # Clamp exponent to avoid overflow/underflow. Should not happen, just to be sure.
#         exponent = torch.clamp(exponent, min=-20, max=0)
#         exp_kernel = torch.exp(exponent)
#         interaction_terms = alpha_matrix * beta_matrix * exp_kernel  # Shape: (B,N, N)

#         interaction_terms = torch.tril(
#             interaction_terms, diagonal=-1
#         )  # Zero out diagonal and above, to ensure causality. This works also with batched matrices.

#         # Mask out invalid events.
#         interaction_terms[~valid_event_mask] = 0.0

#         relevant_mu = mu[clamped_types] * valid_events  # Shape (B,N)

#         # We now want the impact of past events on the current event.
#         # For this we sum the rows. To get the intensities we also add the correct mu values.
#         intensities = relevant_mu + torch.sum(interaction_terms, dim=2)  # Shape: (B,N,)

#         min_intensity = 1e-12
#         if log:
#             # Clamp intensities to avoid log(0) = -inf or log(negative) = nan
#             intensities_clamped = torch.clamp(intensities, min=min_intensity)
#             log_intensities = torch.log(intensities_clamped)
#             # Screen out invalid contributions. Set them to 0.0 so they dont contribute to the next sum.
#             log_intensities[~valid_events] = 0.0
#             positive_likelihood = torch.sum(input=log_intensities, dim=-1)  # Shape: (B,)
#         else:
#             # Have to set the invalid events to 1.0, so they dont contribute to the product.
#             # Also clamp to ensure non-negative intensities
#             intensities = torch.clamp(intensities, min=min_intensity)
#             intensities[~valid_events] = 1.0
#             positive_likelihood = torch.prod(intensities, dim=-1)  # Shape: (B,)

#         # Mask out event sequences with no elements. They only contribute negativly.
#         num_0 = valid_events.sum(dim=1) == 0
#         positive_likelihood[num_0] = torch.tensor(0.0) if log else torch.tensor(1.0)

#         return positive_likelihood

#     # def sample_inverse(
#     #     self,
#     #     ts: MVEventData,
#     #     num_steps: int = 1,
#     #     rng: Optional[torch.Generator | int] = None,
#     #     tol: float = 1e-6,
#     #     max_newton_iters: int = 50,
#     # ):
#     #     """
#     #     Sample next event by inverting the total (superposed) CDF and then sampling the event type.

#     #     Steps:
#     #     1. Draw `u ~ Uniform(0,1)` and set `target = -log(1-u)`.
#     #     2. Invert `C_total(T) = sum_d C_d(T)` to find T* such that C_total(T*) = target using the helpers in `hawkes.utils`.
#     #     3. Evaluate `lambda_d(T*)` and sample the event type from the categorical distribution with probs proportional to lambda_d(T*).

#     #     Args:
#     #         ts: past events as `MVEventData`.
#     #         rng: optional torch.Generator for reproducibility.
#     #         tol: tolerance passed to the root solver.
#     #         max_newton_iters: maximum iterations passed to the root solver.

#     #     Returns:
#     #         (t_star, type_idx)
#     #     """

#     #     # TODO allow drawing more samples.
#     #     # TODO fix CUDA (prob in whole class). BatchedMVEventData .seq_length and .max_time have cuda problems.

#     #     step = 0
#     #     time_samples = []
#     #     dist_samples = []

#     #     if rng is None:
#     #         rng = torch.Generator()
#     #     elif isinstance(rng, int):
#     #         rng = torch.Generator().manual_seed(rng)

#     #     for step in range(0, num_steps):
#     #         # Pack single sequence into BatchedMVEventData
#     #         batched = BatchedMVEventData([ts.time_points], [ts.event_types])

#     #         device = ts.time_points.device

#     #         # current time baseline (last observed event time) or 0
#     #         if len(ts) == 0:
#     #             t0 = 0.0
#     #         else:
#     #             t0 = float(ts.time_points[-1].item())

#     #         # Draw a single uniform for the superposed process
#     #         u = torch.rand(size=(), generator=rng, device=device).item()
#     #         target_total = -torch.log1p(-torch.tensor(u)).item()

#     #         # define scalar-evaluators for the total cumulative and total intensity
#     #         def ci_total(T_scalar: float, ci_low, ts) -> float:
#     #             T_tensor = torch.tensor([T_scalar], dtype=ts.time_points.dtype, device=device)
#     #             ci_vec = (self.cumulative_intensity(T_tensor, batched) - ci_low)[0]  # shape (D,)
#     #             return float(ci_vec.sum().item())

#     #         def lambda_total(T_scalar: float, ts) -> float:
#     #             T_tensor = torch.tensor([T_scalar], dtype=ts.time_points.dtype, device=device)
#     #             lam_vec = self.intensity(T_tensor, batched)[0]  # shape (D,)
#     #             return float(lam_vec.sum().item())

#     #         low = float(t0)
#     #         ci_low = self.cumulative_intensity(torch.tensor([low], dtype=ts.time_points.dtype, device=device), batched)

#     #         # bracket the root
#     #         ci_kwargs = {"ci_low": ci_low, "ts": batched}
#     #         lambda_kwargs = {"ts": batched}
#     #         low, high, ci_low, ci_high = bracket_monotone(ci_total, float(t0), target_total, func_kwargs=ci_kwargs)

#     #         if ci_high < target_total:
#     #             # failed to bracket: return high and sample type at high
#     #             t_star = float(high)
#     #         else:
#     #             # invert using safeguarded Newton
#     #             t_star = invert_monotone_newton(
#     #                 ci_total,
#     #                 lambda_total,
#     #                 target_total,
#     #                 low,
#     #                 high,
#     #                 tol=tol,
#     #                 max_iters=max_newton_iters,
#     #                 mono_kwargs=ci_kwargs,
#     #                 d_kwargs=lambda_kwargs,
#     #             )

#     #         # Compute per-type intensities at t_star and sample type
#     #         T_tensor = torch.tensor([t_star], dtype=ts.time_points.dtype, device=device)
#     #         lam_vec = self.intensity(T_tensor, batched)[0]  # shape (D,)
#     #         lam_sum = float(lam_vec.sum().item())

#     #         D = lam_vec.shape[0]
#     #         if lam_sum <= 1e-12 or not (lam_sum == lam_sum):
#     #             # Degenerate: no intensity; choose uniformly
#     #             type_idx = int(torch.randint(high=D, size=(1,), generator=rng).item())
#     #         else:
#     #             probs = lam_vec / lam_sum
#     #             # torch.multinomial expects 2D or 1D float tensor
#     #             type_idx = torch.multinomial(probs, num_samples=1, generator=rng)

#     #         time_samples.append(t_star)
#     #         dist_samples.append(probs)

#     #         new_time_points = torch.cat([ts.time_points, T_tensor], dim=0)
#     #         new_event_types = torch.cat([ts.event_types, type_idx], dim=0)

#     #         ts = MVEventData(new_time_points, new_event_types)

#     #     return ts, time_samples, dist_samples
