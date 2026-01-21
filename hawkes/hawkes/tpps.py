"""
Abstract Base Classes for Temporal Point Processes (TPPs)

This module defines the foundational abstract base classes for implementing
multivariate temporal point processes, extracting common interfaces from
implementations like Hawkes processes, Poisson processes, and spline-based processes.

Key features:
- Flexible intensity computation (standard or log-space)
- Optional numerical integration for cumulative intensity (trapezoidal + MC hybrid)
- CDF inversion via rootfinding (uses existing Hawkes utilities) or direct implementation
- Time-rescaling for goodness-of-fit testing
- Helper methods for common TPP operations
"""

from abc import ABC, abstractmethod
from typing import Optional, Tuple, List
import torch
import torch.nn as nn

from .event_utils import MVEventData, BatchedMVEventData
from .utils import invert_CI, numerical_integration


class TemporalPointProcess(nn.Module, ABC):
    """
    Abstract base class for multivariate temporal point processes.

    A temporal point process models the occurrence of events over time, where each
    event has a timestamp and an associated type/mark. The key component is the
    intensity function λ_d(t) which characterizes the instantaneous rate of events
    of type d at time t.

    Core Methods (Required):
    - intensity(t, ts) OR log_intensity(t, ts): The intensity or log-intensity function
    - transform_params(): Return parameters in their constrained form

    Likelihood Methods (Must implement one of):
    - cumulative_intensity(T, ts): Analytical integral (preferred)
    - OR provide neither and use numerical_integration (slower but flexible)

    Helper Methods (Optional overrides):
    - positive_likelihood(ts, log): Product of intensities at event times
    - negative_likelihood(ts, T, log): Integral of intensity (survival term)
    - inverse_cumulative_intensity(u, ts): Invert CDF analytically
    - OR use root-finding solver (automatic fallback)

    Default Behavior:
    - If log_intensity is provided but not intensity, intensity is computed via exp(log_intensity)
    - If neither intensity nor cumulative_intensity are analytical, numerical integration is used
    - If CDF inversion is not provided, root-finding solver is used (Brent's method)
    - PDF and CDF use intensity and cumulative_intensity
    """

    def __init__(
        self,
        D: int,
        seed: Optional[int] = 42,
        use_analytical_ci: bool = True,
        ci_integration_method: str = "trapezoidal",
        ci_num_points: int = 100,
    ):
        """
        Initialize the temporal point process.

        Args:
            D: Number of event types (dimensions)
            seed: Random seed for reproducibility
            use_analytical_ci: Whether to prefer analytical cumulative_intensity if available.
                If False and both analytical and numerical are available, uses numerical
                (useful for testing/debugging).
            ci_integration_method: Method for numerical integration ("trapezoidal" or "mc_trapezoidal")
            ci_num_points: Number of points for numerical integration
        """
        super().__init__()
        self.D = D
        self.seed = seed
        self.use_analytical_ci = use_analytical_ci
        self.ci_integration_method = ci_integration_method
        self.ci_num_points = ci_num_points

        if seed is not None:
            self.generator = torch.Generator().manual_seed(seed)
        else:
            self.generator = torch.Generator()

    # =========================================================================
    # Utility functions
    # ========================================================================
    @abstractmethod
    def transform_params(self) -> Tuple[torch.Tensor, ...]:
        """
        Return the model parameters in their constrained/transformed form.

        Many TPP parameters have natural constraints (e.g., rates must be positive).
        This method returns the parameters after applying appropriate transformations
        like softplus, exp, etc. to ensure constraints are satisfied.

        Returns:
            Tuple of transformed parameter tensors. The exact structure depends on
            the specific TPP model (e.g., (mu,) for Poisson, (mu, alpha, beta) for Hawkes)
        """
        raise NotImplementedError("Subclasses must implement parameter transformation")

    # =========================================================================
    # Intensity Functions (At least one must be implemented)
    # =========================================================================

    def intensity(self, t: torch.Tensor, ts: BatchedMVEventData) -> torch.Tensor:
        """
        Compute the intensity function λ_d(t) for each event type d.

        The intensity function represents the instantaneous rate of events of each
        type conditional on the history of events up to time t.

        Default behavior: Computes exp(log_intensity(t, ts)) if log_intensity is overridden.
        Override this method if direct computation is more stable/efficient.

        Args:
            t: Time points where to evaluate intensity. Shape: (batch_size,)
            ts: Historical events (batched, padded sequences). Shape: (batch_size, seq_len)

        Returns:
            Intensity values for each type and batch. Shape: (batch_size, D)

        Mathematical definition:
            λ_d(t) = lim_{dt→0} P(event of type d in [t, t+dt] | H_t) / dt
            where H_t is the history of events up to time t.
        """
        return torch.exp(self.log_intensity(t, ts))

    def log_intensity(self, t: torch.Tensor, ts: BatchedMVEventData) -> torch.Tensor:
        """
        Compute the log intensity function log(λ_d(t)) for numerical stability.

        Computing in log-space can be more numerically stable, especially when
        intensities are very small or very large.

        Default behavior: Computes log(intensity(t, ts)).
        Override this method if log-computation is more stable/efficient than computing
        intensity and then taking log.

        Args:
            t: Time points where to evaluate intensity. Shape: (batch_size,)
            ts: Historical events (batched, padded sequences). Shape: (batch_size, seq_len)

        Returns:
            Log-intensity values for each type and batch. Shape: (batch_size, D)
        """
        return torch.log(self.intensity(t, ts))

    # =========================================================================
    # Cumulative Intensity Functions. If analytical solution is not provided, numerical integration will be used
    # =========================================================================

    def analytical_cumulative_intensity(self, T: torch.Tensor, ts: BatchedMVEventData):
        """
        Computes the cumulative intensity function (see cumuluative_intensity) in a analytical form.
        """
        raise NotImplementedError("You can provide an analytical solution to your CI here.")

    def numerical_cumulative_intensity(self, T: torch.Tensor, ts: BatchedMVEventData):
        """
        Compute the cumulative intensity function Λ_d(T) using numerical integration.

        The cumulative intensity represents the integral of the intensity function
        from time 0 to T, measuring the expected number of events of each type
        up to time T.

        Args:
            T: End times where to evaluate cumulative intensity. Shape: (batch_size,)
            ts: Historical events (batched, padded sequences). Shape: (batch_size, seq_len)
        """

        return numerical_integration(
            self.intensity,
            t_start=torch.zeros_like(T),
            t_end=T,
            ts=ts,
            method=self.ci_integration_method,
            num_points=self.ci_num_points,
            rng=self.generator,
        )

    def cumulative_intensity(self, T: torch.Tensor, ts: BatchedMVEventData, use_analytical_ci: Optional[bool] = True):
        """
        Compute the cumulative intensity function Λ_d(T).

        The cumulative intensity represents the integral of the intensity function
        from time 0 to T, measuring the expected number of events of each type
        up to time T.

        Default behavior: Uses numerical integration if analytical is not overridden
        or if use_analytical_ci is False.

        Args:
            T: End times where to evaluate cumulative intensity. Shape: (batch_size,)
            ts: Historical events (batched, padded sequences). Shape: (batch_size, seq_len)
        """

        if use_analytical_ci is None:
            use_analytical_ci = self.use_analytical_ci

        # Use the analytical solution. Throws an not-implemented error if not provided.
        if use_analytical_ci:
            return self.analytical_cumulative_intensity(T, ts)
        else:
            return self.numerical_cumulative_intensity(T, ts)

    # ========================================================================
    # CDF/CI Inversion (Override for analytical solution)
    # =======================================================================

    def inverse_marginal_cumulative_intensity(
        self,
        u: torch.Tensor,
        ts: BatchedMVEventData,
    ) -> torch.Tensor:
        """
        Invert the marginal cumulative intensity to find t such that Λ(t) - Λ(t_last) = u.
        t is here in absolute time.

        Default implementation: Uses Bracketed-Newton's method from utils.py.
        Override for analytical solutions or more efficient implementations.

        Args:
            u: Target cumulative intensity values. Shape: (batch_size,)
            ts: Historical events (batched, padded sequences). Shape: (batch_size, seq_len)

        Returns:
            Time values delta such that cumulative_intensity(delta) ≈ u. Shape: same as u

        Useful for sampling and other applications requiring CI inversion.
        """

        assert u.shape[0] == ts.time_points.shape[0], "u must have the same batch size as ts"

        device = ts.time_points.device

        last_event_CI = self.cumulative_intensity(ts.max_time, ts)  # (B,D)

        def ci_func(t: torch.Tensor) -> torch.Tensor:
            ci = self.cumulative_intensity(t, ts)  # (B,D)
            return ci.sum(dim=1) - last_event_CI.sum(dim=1)  # (B,)

        def intensity_func(t: torch.Tensor) -> torch.Tensor:
            lam = self.intensity(t, ts)  # (B,D)
            return lam.sum(dim=1)  # (B,)

        t_star = invert_CI(
            cumulative_intensity_func=ci_func,
            intensity_func=intensity_func,
            target=u,
            bracket_init=(ts.max_time, ts.max_time + 0.1),
        )

        return t_star

    # =========================================================================
    # Likelihood Functions (Can override positive/negative likelihoods)
    # =========================================================================

    def positive_likelihood(
        self,
        ts: BatchedMVEventData,
        log: bool = True,
    ) -> torch.Tensor:
        """
        Compute the positive part of the likelihood: product of intensities at event times.

        This corresponds to the "data fit" term in the likelihood, measuring how well
        the model predicts the observed events.

        Can be overwritten to provide a far more efficient implementation that only computes the likelihood of the "correct" classes at each time-point.

        Args:
            ts: Observed event sequences (batched, padded). Shape: (batch_size, seq_len)
            log: If True, return log-likelihood; otherwise return likelihood

        Returns:
            Positive likelihood for each batch. Shape: (batch_size,)

        Mathematical definition:
            L⁺ = ∏ᵢ λ(tᵢ, mᵢ)  or  log L⁺ = ∑ᵢ log λ(tᵢ, mᵢ)
            where tᵢ are event times and mᵢ are event types.
        """
        B = ts.time_points.shape[0]
        L = ts.time_points.shape[1]
        valid_events = ts.event_types != -1  # (B, L) Mask for valid events (not padding)

        # Gather intensities at event times
        intensities = torch.stack([self.intensity(ts.time_points[:, l], ts=ts) for l in range(L)], dim=1)  # (B, L, D)
        event_types = ts.event_types.unsqueeze(-1)  # (B, L, 1)
        intensities = torch.gather(intensities, dim=2, index=event_types).squeeze(-1)  # (B, L)

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
        self,
        ts: BatchedMVEventData,
        T: torch.Tensor,
        log: bool = True,
    ) -> torch.Tensor:
        """
        Compute the negative part of the likelihood: integral of intensity (survival term).

        This corresponds to the "survival" term in the likelihood, penalizing the model
        for predicting events that did not occur.

        Args:
            ts: Observed event sequences (batched, padded). Shape: (batch_size, seq_len)
            T: End time of observation window. Shape: (batch_size,)
            log: If True, return log-likelihood; otherwise return likelihood

        Returns:
            Negative likelihood for each batch. Shape: (batch_size,)

        Mathematical definition:
            L⁻ = exp(-∑_d Λ_d(T))  or  log L⁻ = -∑_d Λ_d(T)
            where Λ_d(T) = ∫₀ᵀ λ_d(t) dt is the cumulative intensity.
        """
        integral = self.cumulative_intensity(T=T, ts=ts)  # (B,D)

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
    ) -> torch.Tensor:
        """
        Compute the full likelihood of observed events.

        The likelihood combines the positive term (product of intensities at events)
        and the negative term (integral of intensity over the observation window).

        Args:
            ts: Observed event sequences (batched, padded). Shape: (batch_size, seq_len)
            T: End time of observation window. Shape: (batch_size,)
            log: If True, return log-likelihood; otherwise return likelihood

        Returns:
            Likelihood for each batch. Shape: (batch_size,)

        Mathematical definition:
            L = (∏ᵢ λ(tᵢ, mᵢ)) × exp(-∑_d ∫₀ᵀ λ_d(t) dt)
            log L = ∑ᵢ log λ(tᵢ, mᵢ) - ∑_d ∫₀ᵀ λ_d(t) dt

        This is the standard likelihood for temporal point processes, which can be
        derived from the probability of observing exactly the given events in [0,T].
        """
        positive = self.positive_likelihood(ts, log=log)
        negative = self.negative_likelihood(ts, T, log=log)

        if log:
            return positive + negative
        else:
            return positive * negative

    def PDF(self, T: torch.Tensor, ts: BatchedMVEventData) -> torch.Tensor:
        """
        Compute the joint probability density function p(t=T, e=E | H_t) for each type E.

        This gives the probability density of the next event occurring at time T with
        type E, given the history of events up to (but not including) T.

        Args:
            T: Time points to evaluate. Shape: (batch_size,)
            ts: Historical events (batched, padded). Shape: (batch_size, seq_len)

        Returns:
            PDF values for each type and batch. Shape: (batch_size, D)

        Mathematical definition:
            p(t=T, e=E | H_t) = λ_E(T) × exp(-∑_d Λ_d(T))
            where Λ_d(T) is the cumulative intensity from the last event to T.

        Note:
            This is the probability density for the joint distribution over (time, type).
            To get marginal over time: sum over all types.
        """
        intensities = self.intensity(T, ts)  # Shape: (B, D)
        ci = self.cumulative_intensity(T, ts) - self.cumulative_intensity(ts.max_time, ts)  # Shape: (B, D)
        pdfs = intensities * torch.exp(-torch.sum(ci, dim=1, keepdim=True))  # Shape: (B, D)
        return pdfs

    def CDF(self, T: torch.Tensor, ts: BatchedMVEventData) -> torch.Tensor:
        """
        Compute the cumulative distribution function P(t < T | H_t) (marginal over types).

        This gives the probability that the next event (of any type) occurs before time T,
        given the history.

        Args:
            T: Time points to evaluate. Shape: (batch_size,)
            ts: Historical events (batched, padded). Shape: (batch_size, seq_len)

        Returns:
            CDF values for each batch. Shape: (batch_size,)

        Mathematical definition:
            P(t < T) = 1 - exp(-∑_d Λ_d(T))
            where Λ_d(T) is the cumulative intensity from the last event to T.

        Note:
            CDF for the categorical event type is not well-defined. This returns the
            marginal CDF over time, integrating out the event type.
        """
        ci = self.cumulative_intensity(T, ts) - self.cumulative_intensity(ts.max_time, ts)  # Shape: (B, D)
        cdfs = 1 - torch.exp(-torch.sum(ci, dim=1))  # Shape: (B,)
        return cdfs

    # =========================================================================
    # Sampling
    # =========================================================================

    def sample(
        self,
        ts: MVEventData,
        num_steps: int = 1,
        rng: Optional[torch.Generator | int] = None,
    ) -> Tuple[MVEventData, List[float], List[torch.Tensor]]:
        """
        Generate new events by sampling from the temporal point process.

        Default implementation: Uses inverse CDF method.
        Override for model-specific sampling strategies (thinning, etc.).

        Args:
            ts: Initial event history (single sequence, not batched)
            num_steps: Number of events to generate
            rng: Random number generator or seed for reproducibility

        Returns:
            Tuple containing:
            - Updated MVEventData with new events appended
            - List of sampled inter-arrival times
            - List of type probability distributions at each sampled time

        Note:
            Sampling strategies vary by model:
            - Poisson: Direct inverse CDF sampling
            - Hawkes: Thinning algorithm or inverse CDF (if tractable)
            - Spline: Inverse CDF via numerical inversion

            Not all models may have efficient sampling implementations.
        """
        if rng is None:
            rng = torch.Generator()
        elif isinstance(rng, int):
            rng = torch.Generator().manual_seed(rng)

        device = ts.time_points.device
        batched = BatchedMVEventData([ts.time_points], [ts.event_types])

        time_samples = []
        dist_samples = []

        for _ in range(num_steps):
            # Sample the target quantile of the (class marginalized) CDF from uniform.
            u = torch.rand(size=(), generator=rng, device=device)

            # Transform u. Remember that our CI implementation is in absolute time (for easy computation of the likelihood integral).
            # We search for delta such that:
            # CDF(delta) = 1 - exp(- (CI(t_last + delta) - CI(t_last)) ) = u
            # CI(t_last + delta) - CI(t_last) = -log(1-u)
            target = -torch.log1p(-torch.tensor(u, device=device)).unsqueeze(0)  # (1,)

            # Invert CDF to get new event time. The offset with t_last is handled in the inversion implementation.
            t_star = self.inverse_marginal_cumulative_intensity(target, batched)  # (1,)

            # Sample event type from intensity-weighted categorical
            lam_vec = self.intensity(t_star, batched)[0]  # Shape: (D,)
            lam_sum = float(lam_vec.sum().item())

            if lam_sum <= 1e-12 or not (lam_sum == lam_sum):
                # Degenerate: choose uniformly
                type_idx = torch.randint(high=self.D, size=(1,), generator=rng)
            else:
                probs = lam_vec / lam_sum
                type_idx = torch.multinomial(probs, num_samples=1, generator=rng)

            time_samples.append(t_star)
            dist_samples.append(lam_vec / lam_sum if lam_sum > 0 else torch.ones(self.D, device=device) / self.D)

            # Update sequence
            new_time_points = torch.cat([ts.time_points, t_star], dim=0)
            new_event_types = torch.cat([ts.event_types, type_idx], dim=0)
            ts = MVEventData(new_time_points, new_event_types)
            batched = BatchedMVEventData([ts.time_points], [ts.event_types])

        return ts, time_samples, dist_samples
