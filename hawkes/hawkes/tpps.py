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
import warnings
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
    - _raw_intensity(t, ts): The raw intensity function (before termination masking)
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
    - intensity(t, ts) automatically applies termination masking to _raw_intensity(t, ts)
    - If cumulative_intensity is not analytical, numerical integration is used
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
        terminating: bool = False,
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
            terminating: If True, each event type can only occur once (first occurrence data).
                After an event type occurs, its intensity becomes zero and it cannot occur again.
        """
        super().__init__()
        self.D = D
        self.seed = seed
        self.use_analytical_ci = use_analytical_ci
        self.ci_integration_method = ci_integration_method
        self.ci_num_points = ci_num_points
        self.terminating = terminating

        if seed is not None:
            self.generator = torch.Generator().manual_seed(seed)
        else:
            self.generator = torch.Generator()

    # =========================================================================
    # Utility functions
    # ========================================================================

    def _get_termination_mask(self, ts: BatchedMVEventData, t: Optional[torch.Tensor] = None, full: bool = False):
        """
        Compute a mask indicating which event types have already occurred (terminated).

        For terminating processes, once an event type occurs, it should no longer
        contribute to the intensity.

        Args:
            ts: Historical events (batched, padded sequences). Shape: (batch_size, seq_len)
            t: Optional time points to check termination at. Shape: (batch_size,)
               If provided, only considers events that occurred strictly before time t.
               If None, considers all events in the history.
            full: If True, returns the full termination mask for all time points in ts.
                    Cannot be true if t is provided.

        Returns:
            Boolean mask where True indicates the event type has NOT occurred yet.
            Shape: (batch_size, D)
            For non-terminating processes, returns all True (no types are masked).
        """

        if full and t is not None:
            raise ValueError("Cannot use full=True and provide t simultaneously.")

        if not self.terminating:
            # Non-terminating: all event types are always active
            B = ts.time_points.shape[0]
            device = ts.time_points.device
            return torch.ones(B, self.D, dtype=torch.bool, device=device)

        # For terminating processes, check which event types have occurred
        B = ts.time_points.shape[0]
        L = ts.time_points.shape[1]
        device = ts.time_points.device
        dtype = ts.time_points.dtype

        # Valid events (not padding)
        valid_events = ts.event_types != -1  # (B, L)

        # If t is provided, only consider events strictly before t
        if t is not None:
            # Events before time t
            before_t = ts.time_points < t.unsqueeze(1)  # (B, L)
            valid_events = valid_events & before_t

        # Vectorized: Create one-hot encoding for each event type
        # For each (batch, event) pair, create a D-dimensional indicator
        # event_types: (B, L), values in [0, D-1] or -1 for padding
        # Clamp to [0, D-1] for scatter, will be masked out by valid_events anyway
        event_types_clamped = ts.event_types.clamp(min=0, max=self.D - 1)  # (B, L)

        # Create one-hot encoding: (B, L, D)
        one_hot = torch.nn.functional.one_hot(event_types_clamped, num_classes=self.D).bool()  # (B, L, D)

        # Mask out invalid events
        one_hot = one_hot & valid_events.unsqueeze(2)  # (B, L, D)

        # Check if any event of each type has occurred: (B, D)
        type_occurred = one_hot.any(dim=1)  # (B, D)

        # Return mask where True = type has NOT occurred
        mask = ~type_occurred  # (B, D)

        return mask

    def _get_termination_times(self, ts: BatchedMVEventData) -> torch.Tensor:
        """
        Get the termination time for each event type (time when it first occurred).

        For terminating processes, this returns the time at which each event type
        first occurred. For types that haven't occurred, returns inf.

        Args:
            ts: Historical events (batched, padded sequences). Shape: (batch_size, seq_len)

        Returns:
            Termination times for each type. Shape: (batch_size, D)
            Contains inf for types that haven't occurred yet.
            For non-terminating processes, returns all inf.
        """
        B = ts.time_points.shape[0]
        L = ts.time_points.shape[1]
        device = ts.time_points.device
        dtype = ts.time_points.dtype

        if not self.terminating:
            # Non-terminating: no types ever terminate
            return torch.full((B, self.D), float("inf"), dtype=dtype, device=device)

        # Handle empty sequences (no events)
        if L == 0:
            return torch.full((B, self.D), float("inf"), dtype=dtype, device=device)

        # Valid events (not padding)
        valid_events = ts.event_types != -1  # (B, L)

        # Clamp event types to valid range for scatter
        event_types_clamped = ts.event_types.clamp(min=0, max=self.D - 1)  # (B, L)

        # Create one-hot encoding: (B, L, D)
        one_hot = torch.nn.functional.one_hot(event_types_clamped, num_classes=self.D).float()  # (B, L, D)

        # Mask out invalid events by setting their one-hot to 0
        one_hot = one_hot * valid_events.unsqueeze(2).float()  # (B, L, D)

        # Expand time_points to match one_hot shape: (B, L, D)
        time_expanded = ts.time_points.unsqueeze(2).expand(B, L, self.D)  # (B, L, D)

        # Set times for non-occurrences to inf
        time_expanded = torch.where(
            one_hot.bool(), time_expanded, torch.tensor(float("inf"), dtype=dtype, device=device)
        )

        # Find minimum time for each type (first occurrence): (B, D)
        termination_times, _ = time_expanded.min(dim=1)  # (B, D)

        return termination_times

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
    # Intensity Functions (Must be implemented by subclasses)
    # =========================================================================

    @abstractmethod
    def _raw_intensity(self, t: torch.Tensor, ts: BatchedMVEventData) -> torch.Tensor:
        """
        Compute the raw (unmasked) intensity function λ_d(t) for each event type d.

        This is the intensity before applying termination masking. Subclasses must
        override this method to define their intensity function.

        Args:
            t: Time points where to evaluate intensity. Shape: (batch_size,)
            ts: Historical events (batched, padded sequences). Shape: (batch_size, seq_len)

        Returns:
            Raw intensity values for each type and batch. Shape: (batch_size, D)
        """
        raise NotImplementedError("Subclasses must implement _raw_intensity")

    def intensity(self, t: torch.Tensor, ts: BatchedMVEventData) -> torch.Tensor:
        """
        Compute the intensity function λ_d(t) for each event type d.

        The intensity function represents the instantaneous rate of events of each
        type conditional on the history of events up to time t.

        For terminating processes, the intensity is zero for event types that have
        already occurred (strictly before time t).

        Subclasses should override _raw_intensity() instead of this method to ensure
        termination is properly handled.

        Args:
            t: Time points where to evaluate intensity. Shape: (batch_size,)
            ts: Historical events (batched, padded sequences). Shape: (batch_size, seq_len)

        Returns:
            Intensity values for each type and batch. Shape: (batch_size, D)
            For terminating processes, intensities are zero for already-occurred types.

        Mathematical definition:
            λ_d(t) = lim_{dt→0} P(event of type d in [t, t+dt] | H_t) / dt
            where H_t is the history of events up to time t.
            For terminating processes: λ_d(t) = 0 if d has occurred at time t' < t.
        """
        intensities = self._raw_intensity(t, ts)

        if self.terminating:
            # Zero out intensities for event types that have already occurred before time t
            mask = self._get_termination_mask(ts, t=t)  # (B, D)
            intensities = intensities * mask.float()

        return intensities

    # =========================================================================
    # Cumulative Intensity Functions. If analytical solution is not provided, numerical integration will be used
    # =========================================================================

    def _raw_analytical_cumulative_intensity(
        self, T: torch.Tensor, ts: BatchedMVEventData, termination_times: Optional[torch.Tensor] = None
    ):
        """
        Computes the cumulative intensity function Λ_d(T) from time 0 to T.

        For terminating processes, subclasses should handle the termination_times parameter
        to compute: Λ_d(T) = ∫_0^{min(t_d, T)} λ_d(t) dt
        where t_d is the time when type d first occurred.

        Args:
            T: End times where to evaluate cumulative intensity. Shape: (batch_size,)
            ts: Historical events (batched, padded sequences). Shape: (batch_size, seq_len)
            termination_times: Optional termination time for each type. Shape: (batch_size, D)
                If provided, CI should be computed up to min(T, termination_times) for each dimension.
                For non-terminating processes, this will be None.

        Returns:
            Cumulative intensity for each type. Shape: (batch_size, D)

        Notes:
            Subclasses should implement efficient termination handling internally.
            For example, Poisson: μ * min(T, t_term)
            This avoids the need for dimension-wise iteration in the base class.
        """
        raise NotImplementedError("You can provide an analytical solution to your CI here.")

    def analytical_cumulative_intensity(self, T: torch.Tensor, ts: BatchedMVEventData):
        """
        Computes the cumulative intensity function with termination handling.

        For terminating processes, the integral for each type d is:
            Λ_d(T) = ∫_0^{min(t_d, T)} λ_d(t) dt
        where t_d is the time when type d first occurred.

        This method passes termination information to _raw_analytical_cumulative_intensity
        which should handle it efficiently within the model-specific computation.
        """
        if not self.terminating:
            return self._raw_analytical_cumulative_intensity(T, ts, termination_times=None)

        # For terminating processes, compute termination times and pass to implementation.
        termination_times = self._get_termination_times(ts)  # (B, D)
        return self._raw_analytical_cumulative_intensity(T, ts, termination_times=termination_times)

    def numerical_cumulative_intensity(
        self, T: torch.Tensor, ts: BatchedMVEventData, T_low: Optional[torch.Tensor] = None
    ):
        """
        Compute the cumulative intensity function Λ_d(T) using numerical integration.

        The cumulative intensity represents the integral of the intensity function
        from time 0 to T, measuring the expected number of events of each type
        up to time T.

        If T_low is provided, computes the integral from T_low to T instead.

        For terminating processes, this automatically handles termination because
        intensity() returns 0 after termination.

        Args:
            T: End times where to evaluate cumulative intensity. Shape: (batch_size,)
            ts: Historical events (batched, padded sequences). Shape: (batch_size, seq_len)
            T_low: Optional start times for integration. Shape: (batch_size,)
        """

        return numerical_integration(
            self.intensity,
            t_start=torch.zeros_like(T) if T_low is None else T_low,
            t_end=T,
            ts=ts,
            method=self.ci_integration_method,
            num_points=self.ci_num_points,
            rng=None,
        )

    def cumulative_intensity(
        self,
        T: torch.Tensor,
        ts: BatchedMVEventData,
        T_low: Optional[torch.Tensor] = None,
    ):
        """
        Compute the cumulative intensity function Λ_d(T).

        The cumulative intensity represents the integral of the intensity function
        from time 0 to T, measuring the expected number of events of each type
        up to time T.

        If T_low is provided, computes the integral from T_low to T instead.

        For terminating processes the logic is handeled in the analytical/numerical implementations.
        For the numerical case, this is automatic since intensity() returns 0 after termination.

        Args:
            T: End times where to evaluate cumulative intensity. Shape: (batch_size,)
            ts: Historical events (batched, padded sequences). Shape: (batch_size, seq_len)
            T_low: Optional start times for integration. Shape: (batch_size,)
        """

        # Check legality of T:
        assert (T >= 0.0).all(), "Cumulative intensity is only defined for T >= 0."
        assert T.shape[0] == ts.time_points.shape[0], "T must have the same batch size as ts"

        if T_low is not None:
            assert (T_low >= 0.0).all(), "Cumulative intensity is only defined for T_low >= 0."
            assert T_low.shape[0] == ts.time_points.shape[0], "T_low must have the same batch size as ts"
            assert (T >= T_low).all(), "Cumulative intensity requires T >= T_low."

        # Use the analytical solution. Throws an not-implemented error if not provided.
        if self.use_analytical_ci:
            ci = self.analytical_cumulative_intensity(T, ts)  # (B,D)
            if T_low is not None:
                ci_low = self.analytical_cumulative_intensity(T_low, ts)  # (B,D)
                ci = ci - ci_low
            return ci  # (B,D)

        else:
            return self.numerical_cumulative_intensity(T, ts, T_low=T_low)  # (B,D)

    # ========================================================================
    # CDF/CI Inversion (Override for analytical solution)
    # ========================================================================

    def inverse_marginal_cumulative_intensity(
        self,
        u: torch.Tensor,
        ts: BatchedMVEventData,
    ) -> torch.Tensor:
        """
        Invert the marginal cumulative intensity to find t such that Λ(t) = u.
        t is here in absolute time.

        Default implementation: Uses Bracketed-Newton's method from utils.py.
        Override for analytical solutions or more efficient implementations.

        Args:
            u: Target cumulative intensity values. Shape: (batch_size,)
            ts: Historical events (batched, padded sequences). Shape: (batch_size, seq_len)

        Returns:
            Time values t such that cumulative_intensity(t) ≈ u. Shape: same as u

        Useful for sampling and other applications requiring CI inversion.
        """
        # TODO is there a way to cache and reuse the inbetween intensities if we have numerical integration??

        assert u.shape[0] == ts.time_points.shape[0], "u must have the same batch size as ts"

        device = ts.time_points.device

        def ci_func(t: torch.Tensor) -> torch.Tensor:
            ci = self.cumulative_intensity(t, ts)  # (B,D)
            return ci.sum(dim=1)  # (B,)

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

    def inverse_CDF(self, u: torch.Tensor, ts: BatchedMVEventData) -> torch.Tensor:
        """
        Invert the CDF to find T such that P( t_next < T) = u.
        Also known as the quantile function. It is only defined for T >= last event time.
        If you want to sample inbetween events, use the inverse_marginal_cumulative_intensity function,
        or cut of the history ts.


        Uses the inverse marginal cumulative intensity.
        Args:
            u: Target CDF values. Shape: (batch_size,)
            ts: Historical events. Shape: (batch_size, seq_len)

        Returns:
            Time values T such that CDF(T) ≈ u. Shape: same as u

        Useful for sampling and other applications requiring CDF inversion.
        """
        device = ts.time_points.device

        # Transform u to target cumulative intensity
        target_ci = -torch.log1p(-u).to(device)  # (B,)

        # CDF is only defined for t >= last event time
        target_ci += self.cumulative_intensity(ts.max_time, ts).sum(dim=1)  # (B,)

        t_star = self.inverse_marginal_cumulative_intensity(target_ci, ts)  # (B,)

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
        device = ts.time_points.device

        if L == 0:
            # No events in the sequence
            return torch.zeros(B, device=device) if log else torch.ones(B, device=device)

        # Gather intensities at event times
        intensities = torch.stack([self.intensity(ts.time_points[:, l], ts=ts) for l in range(L)], dim=1)  # (B, L, D)
        # Clamp event_types to valid range for gather (replace -1 with 0, will be masked out later)
        event_types_clamped = ts.event_types.clamp(min=0).unsqueeze(-1)  # (B, L, 1)
        intensities = torch.gather(intensities, dim=2, index=event_types_clamped).squeeze(-1)  # (B, L)

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
        # Convert T to double to match ts.max_time dtype
        T = T.double()

        # If T is very close to ts.max_time, round up to avoid numerical issues
        close_mask = torch.abs(T - ts.max_time) < 1e-8
        T = torch.where(close_mask, ts.max_time, T)

        assert (T >= ts.max_time).all(), "PDF is only defined for T >= last event time."

        intensities = self.intensity(T, ts)  # Shape: (B, D)
        ci = self.cumulative_intensity(T, ts, T_low=ts.max_time)  # Shape: (B, D)
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
        # Convert T to double to match ts.max_time dtype
        T = T.double()

        # If T is very close to ts.max_time, round up to avoid numerical issues
        close_mask = torch.abs(T - ts.max_time) < 1e-6
        T = torch.where(close_mask, ts.max_time, T)

        assert (T >= ts.max_time).all(), "CDF is only defined for T >= last event time."

        ci = self.cumulative_intensity(T, ts, T_low=ts.max_time)  # Shape: (B, D)
        cdfs = 1 - torch.exp(-torch.sum(ci, dim=1))  # Shape: (B,)
        return cdfs

    def marginal_class_distribution(
        self,
        ts: BatchedMVEventData,
        T_max: Optional[torch.Tensor] = None,
        num_points: int = 200,
    ) -> torch.Tensor:
        """
        Compute the marginal class distribution p(m | H_t) for the next event.

        This marginalizes over time to get the probability of each event type
        for the next event, regardless of when it occurs:

            p(m | H_t) = ∫_{t_N}^{∞} p(m, s | H_t) ds
                       = ∫_{t_N}^{∞} λ_m(s) exp(-∑_d (Λ_d(s) - Λ_d(t_N))) ds

        Since we cannot integrate to infinity, we integrate up to T_max where
        the survival probability is negligible (CDF ≈ 1).

        Args:
            ts: Historical events (batched, padded). Shape: (batch_size, seq_len)
            T_max: Upper integration limit. Shape: (batch_size,) or None.
                   If None, automatically determined based on when CDF ≈ 0.999.
            num_points: Number of quadrature points for numerical integration.

        Returns:
            Marginal class probabilities for each batch. Shape: (batch_size, D)
            Each row sums to approximately 1 (may be slightly less if T_max is too small).

        Mathematical derivation:
            The joint PDF of the next event at time s with type m is:
                p(s, m | H_t) = λ_m(s) × exp(-∑_d (Λ_d(s) - Λ_d(t_N)))

            Integrating over time gives the marginal over event types:
                p(m | H_t) = ∫_{t_N}^{∞} λ_m(s) × S(s) ds

            where S(s) = exp(-∑_d (Λ_d(s) - Λ_d(t_N))) is the survival function.
        """
        # TODO optimize computation for numerical integration. Currently the CI is computed so often, which is extremly inefficient if we use
        # numerical CI. Would be much better to cache the intensities we use for CI_inversion and later for the integration of probs.
        device = ts.time_points.device
        t_last = ts.max_time  # Shape: (B,)

        # Get baseline cumulative intensity at t_last
        ci_base = self.cumulative_intensity(t_last, ts)  # Shape: (B, D)

        # Determine T_max if not provided
        # We want to integrate until CDF ≈ target_cdf (e.g., 0.999)
        if T_max is None:
            # Use CDF inversion: CDF(T) = 1 - exp(-Λ(T)) = target_cdf
            # => Λ(T) = -log(1 - target_cdf)
            target_cdf = 0.999
            with torch.no_grad():
                T_max = self.inverse_CDF(
                    torch.tensor(target_cdf, device=device).expand(ts.time_points.shape[0]), ts
                )  # (B,)

        # At this point T_max is guaranteed to be a tensor
        T_max_tensor: torch.Tensor = T_max

        # Create integration grid: t_last to T_max
        # Shape: (B, num_points)
        t_grid = torch.linspace(0, 1, num_points, device=device).unsqueeze(0)  # (1, num_points)
        t_grid = t_last.unsqueeze(1) + t_grid * (T_max_tensor - t_last).unsqueeze(1)  # (B, num_points)

        # Compute integrand at each grid point
        # integrand[b, k, d] = λ_d(t_grid[b,k]) * exp(-sum_d' (Λ_d'(t_grid[b,k]) - Λ_d'(t_last[b])))
        integrands = []
        for k in range(num_points):
            t_k = t_grid[:, k]  # (B,)

            # Intensity at t_k
            lam_k = self.intensity(t_k, ts)  # (B, D)

            # Cumulative intensity at t_k
            if self.use_analytical_ci:
                ci_k = self.cumulative_intensity(t_k, ts)  # (B, D)
                delta_ci = ci_k - ci_base  # (B, D)
            else:
                delta_ci = self.cumulative_intensity(t_k, ts, T_low=t_last)  # (B, D)

            # Survival function: exp(-sum_d (Λ_d(t_k) - Λ_d(t_last)))
            survival = torch.exp(-delta_ci.sum(dim=1, keepdim=True))  # (B, 1)

            # Integrand: λ_m(t_k) * S(t_k)
            integrand_k = lam_k * survival  # (B, D)
            integrands.append(integrand_k)

        # Stack: (B, num_points, D)
        integrands = torch.stack(integrands, dim=1)

        # Integrate: (B, D)
        marginal_probs = torch.trapezoid(integrands, t_grid.unsqueeze(-1).expand_as(integrands), dim=1)

        # Normalize to ensure valid probability distribution
        # (should sum to ~1 if T_max is large enough)
        total_prob = marginal_probs.sum(dim=1, keepdim=True).clamp(min=1e-12)
        marginal_probs = marginal_probs / total_prob

        return marginal_probs

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

        For terminating processes, sampling stops when all event types have occurred.

        Args:
            ts: Initial event history (single sequence, not batched)
            num_steps: Number of events to generate (for terminating: min(num_steps, remaining types))
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

            For terminating processes, sampling automatically stops when all event
            types have occurred, even if num_steps is larger.
        """
        if rng is None:
            rng = torch.Generator()
        elif isinstance(rng, int):
            rng = torch.Generator().manual_seed(rng)

        device = ts.time_points.device
        batched = BatchedMVEventData([ts.time_points], [ts.event_types])

        time_samples = []
        dist_samples = []

        for step in range(num_steps):
            # For terminating processes, check if all types have occurred
            if self.terminating:
                mask = self._get_termination_mask(batched)  # (1, D)
                remaining_types = mask[0].sum().item()
                if remaining_types == 0:
                    # All types have occurred, stop sampling
                    break

            # Sample the target quantile of the (class marginalized) CDF from uniform.
            u = torch.rand(size=(), generator=rng, device=device)

            # Transform u. Remember that our CI implementation is in absolute time (for easy computation of the likelihood integral).
            # We search for delta such that:
            # CDF(delta) = 1 - exp(- (CI(t_last + delta) - CI(t_last)) ) = u
            # CI(t_last + delta) - CI(t_last) = -log(1-u)
            # Invert CDF to get new event time. The offset with t_last is handled in the inversion implementation.
            t_star = self.inverse_CDF(u.unsqueeze(0), batched)  # (1,)

            # Ensure the sampled time is after the last event (important for terminating processes
            # where inverse_CDF may not properly account for reduced marginal intensity)
            t_last = batched.max_time if hasattr(batched, "max_time") else batched.time_points.max()
            if t_star.item() < t_last.item():
                t_star = t_last + torch.rand(size=(), generator=rng, device=device) * 0.1

            # Sample event type from intensity-weighted categorical
            lam_vec = self.intensity(t_star, batched)[0]  # Shape: (D,)
            lam_sum = float(lam_vec.sum().item())

            if lam_sum <= 1e-12 or not (lam_sum == lam_sum):
                # Degenerate case
                if self.terminating:
                    # For terminating: choose uniformly from remaining types
                    mask = self._get_termination_mask(batched)[0]  # (D,) - squeeze batch dim
                    remaining_indices = torch.where(mask)[0]
                    if len(remaining_indices) > 0:
                        random_idx = torch.randint(high=len(remaining_indices), size=(1,), generator=rng, device=device)
                        type_idx = remaining_indices[random_idx]
                    else:
                        # Should not happen as we check above, but handle gracefully
                        break
                else:
                    # Non-terminating: choose uniformly from all types
                    type_idx = torch.randint(high=self.D, size=(1,), generator=rng, device=device)
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


class NumericalTemporalPointProcess(TemporalPointProcess, ABC):
    """
    Abstract base class for temporal point processes that require numerical integration.

    This is a thin wrapper around TemporalPointProcess that always uses numerical
    integration for cumulative intensity by setting use_analytical_ci=False.

    Useful for models where:
    - Analytical cumulative intensity is intractable
    - Intensity involves non-linear transformations (e.g., softplus)
    - Testing/debugging numerical integration methods
    """

    def __init__(
        self,
        D: int,
        seed: Optional[int] = 42,
        ci_integration_method: str = "mc_trapezoidal",
        ci_num_points: int = 10,
        terminating: bool = False,
    ):
        """
        Initialize a numerical temporal point process.

        Args:
            D: Number of event types (dimensions)
            seed: Random seed for reproducibility
            ci_integration_method: Method for numerical integration ("trapezoidal" or "mc_trapezoidal")
                Default is "mc_trapezoidal" which uses Monte Carlo sampling with trapezoidal rule
            ci_num_points: Number of points for numerical integration
                Default is 10, which is relatively coarse but fast
            terminating: If True, each event type can only occur once (first occurrence data)
        """
        super().__init__(
            D=D,
            seed=seed,
            use_analytical_ci=False,  # Always use numerical integration
            ci_integration_method=ci_integration_method,
            ci_num_points=ci_num_points,
            terminating=terminating,
        )

    def likelihood(
        self,
        ts: BatchedMVEventData,
        T: torch.Tensor,
        log: bool = True,
    ) -> torch.Tensor:
        """
        Compute the positive part of the likelihood: product of intensities at event times.

        This corresponds to the "data fit" term in the likelihood, measuring how well
        the model predicts the observed events.

        Can be overwritten to provide a far more efficient implementation that only computes the likelihood of the "correct" classes at each time-point.

        Args:
            ts: Observed event sequences (batched, padded). Shape: (batch_size, seq_len)
            T: End time of observation window. Shape: (batch_size,)
            log: If True, return log-likelihood; otherwise return likelihood

        Returns:
            Likelihood for each batch. Shape: (batch_size,)

        Mathematical definition:
            L = (∏ᵢ λ(tᵢ, mᵢ)) × exp(-∑_d ∫₀ᵀ λ_d(t) dt)
            log L = ∑ᵢ log λ(tᵢ, mᵢ) - ∑_d ∫₀ᵀ λ_d(t) dt
        """
        B = ts.time_points.shape[0]
        L = ts.time_points.shape[1]
        valid_events = ts.event_types != -1  # (B, L) Mask for valid events (not padding)
        device = ts.time_points.device

        # Convert T to double to match ts.time_points dtype
        T = T.double()

        # Validate T shape
        if T.shape[0] != B:
            raise ValueError(f"T must have shape ({B},) to match batch size, but got shape {T.shape}")

        # Validate that T >= 0 for all sequences
        if (T < 0).any():
            raise ValueError(f"T must be non-negative for all sequences. Got min T = {T.min().item()}")

        # Validate that T >= ts.max_time for all sequences (observation window must extend beyond last event)
        # For empty sequences, ts.max_time is 0, so this is always satisfied if T >= 0
        if (T < ts.max_time).any():
            invalid_mask = T < ts.max_time
            raise ValueError(
                f"T must be >= last event time (ts.max_time) for all sequences. "
                f"Found {invalid_mask.sum().item()} sequences where T < ts.max_time. "
                f"Min difference: {(ts.max_time - T).max().item():.6f}"
            )

        # If T > ts.max_time provide a warning that sampling over ts.max_time is sparser then below it.
        if (T > ts.max_time).any():
            warnings.warn(
                "Some T values are greater than the last event time in ts. "
                "Numerical integration may be less accurate beyond the last event time."
            )

        # Handle empty sequences (L=0) early - only negative likelihood
        if L == 0:
            # No events, only negative likelihood contribution
            return self.negative_likelihood(ts, T, log=log)

        # Gather intensities at event times
        intensities = torch.stack([self.intensity(ts.time_points[:, l], ts=ts) for l in range(L)], dim=1)  # (B, L, D)
        # Clamp event_types to valid range for gather (replace -1 with 0, will be masked out later)
        event_types_clamped = ts.event_types.clamp(min=0).unsqueeze(-1)  # (B, L, 1)
        selected_intensities = torch.gather(intensities, dim=2, index=event_types_clamped).squeeze(-1)  # (B, L)

        min_intensity = 1e-12
        if log:
            # Clamp intensities to avoid log(0) = -inf or log(negative) = nan
            selected_intensities_clamped = torch.clamp(selected_intensities, min=min_intensity)
            selected_log_intensities = torch.log(selected_intensities_clamped)
            # Screen out invalid contributions. Set them to 0.0 so they dont contribute to the next sum.
            selected_log_intensities[~valid_events] = 0.0
            positive_likelihood = torch.sum(input=selected_log_intensities, dim=-1)  # Shape: (B,)
        else:
            # Have to set the invalid events to 1.0, so they dont contribute to the product.
            # Also clamp to ensure non-negative intensities
            selected_intensities_clamped = torch.clamp(selected_intensities, min=min_intensity)
            selected_intensities_clamped[~valid_events] = 1.0
            positive_likelihood = torch.prod(selected_intensities_clamped, dim=-1)  # Shape: (B,)

        # Mask out event sequences with no elements. They only contribute negativly.
        num_0 = valid_events.sum(dim=1) == 0
        positive_likelihood[num_0] = torch.tensor(0.0) if log else torch.tensor(1.0)

        # Compute negative likelihood using numerical integration

        # Each batch needs ci_num_points total points. We already have valid_events.sum(dim=1) points.
        # Use minimum valid events to ensure we have enough sampled points for all batches
        min_valid_events = valid_events.sum(dim=1).min().item()
        num_to_sample: int = max(2, self.ci_num_points - min_valid_events)

        if self.ci_integration_method == "trapezoidal":
            # Sample points uniformly in [0, max_time] for each batch
            t_vals = torch.linspace(0, 1, num_to_sample, device=device)
            t_vals = T.unsqueeze(1) * t_vals.unsqueeze(0)  # (B, num_to_sample)
        else:  # self.ci_integration_method == "mc_trapezoidal"
            # Sample points randomly in [0, max_time], always include 0 and max_time
            uniform_samples = torch.rand((B, num_to_sample - 2), generator=self.generator, device=device)
            uniform_samples = torch.cat(
                [torch.zeros((B, 1), device=device), uniform_samples, torch.ones((B, 1), device=device)], dim=1
            )  # (B, num_to_sample)
            t_vals = T.unsqueeze(1) * uniform_samples  # (B, num_to_sample)

        # Compute intensity value at sampled points
        sample_intensities = torch.stack(
            [self.intensity(t=t_vals[:, k], ts=ts) for k in range(num_to_sample)], dim=1
        )  # Shape: (B, num_to_sample, D)

        # Only include VALID event times and their intensities (exclude padding)
        # Replace invalid times with -1 (will be filtered after sort)
        valid_event_times = ts.time_points.clone()  # (B, L)
        valid_event_times[~valid_events] = -1.0  # Mark invalid with -1

        # Combine sampled and valid event times/intensities
        time_points = torch.cat([t_vals, valid_event_times], dim=1)  # (B, num_to_sample + L)
        combined_intensities = torch.cat([sample_intensities, intensities], dim=1)  # (B, num_to_sample + L, D)

        # Sort by time
        time_points, indices = torch.sort(time_points, dim=1)
        indices_expanded = indices.unsqueeze(2).expand(-1, -1, self.D)
        combined_intensities = torch.gather(combined_intensities, 1, indices_expanded)

        # Create mask for valid points (time >= 0, i.e., not the -1 markers)
        valid_points = time_points >= 0  # (B, num_to_sample + L)

        # For trapezoidal integration, we need consecutive valid points
        # A segment [i, i+1] is valid if both endpoints are valid
        segment_valid = valid_points[:, :-1] & valid_points[:, 1:]  # (B, num_to_sample + L - 1)

        # Compute segment-wise integrals using trapezoidal rule
        dt = time_points[:, 1:] - time_points[:, :-1]  # (B, num_to_sample + L - 1)
        avg_intensity = (
            combined_intensities[:, 1:, :] + combined_intensities[:, :-1, :]
        ) / 2  # (B, num_to_sample + L - 1, D)

        # Segment areas: dt * avg_intensity, but only for valid segments
        segment_areas = dt.unsqueeze(2) * avg_intensity  # (B, num_to_sample + L - 1, D)
        segment_areas[~segment_valid.unsqueeze(2).expand_as(segment_areas)] = 0.0

        # Sum all segment areas to get total integral
        integrals = segment_areas.sum(dim=1)  # (B, D)

        if log:
            negative_likelihood = torch.sum(-integrals, dim=1)  # Shape: (B,)
        else:
            negative_likelihood = torch.prod(torch.exp(-integrals), dim=1)  # Shape: (B,)

        if log:
            return positive_likelihood + negative_likelihood
        else:
            return positive_likelihood * negative_likelihood
