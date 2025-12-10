# My own implementation of the Hawkes process
# %%
from dataclasses import dataclass
from typing import Literal, Optional
import torch
import torch.nn.functional as F

from event_utils import MVEventData, BatchedMVEventData
from utils import inverse_softplus


# %%


@dataclass
class ExpKernelParams:
    """
    Parameters for a Hawkes Process with Exponential Kernel.
    Shapes have to be consisttent: (D,), (D,D) and (D,D).
    For alpha and beta the first dimension is the receiving event type, the second the triggering event type.
    --> alpha[i,j] scales the impact of an event of type j on the intensity of events of type i.
    """

    mu: torch.Tensor
    alpha: torch.Tensor
    beta: torch.Tensor


class AbstractHawkesProcess(torch.nn.Module):
    def __init__(
        self,
        D: Optional[int] = None,
        seed: Optional[int] = 42,
        integration_mode: Literal["trapezoidal", "monte_carlo", "mc_trapezoidal"] = "trapezoidal",
    ) -> None:
        self.D = D
        self.seed = seed
        self.integration_mode = integration_mode

        super().__init__()

    def kernel(self, delta_t: float, type_1: int, type_2: int):
        raise NotImplementedError()

    def intensity(self, t: float | torch.Tensor, ts: MVEventData | list[MVEventData]):
        raise NotImplementedError("Overwrite this function to implement a kernel.")

    def intensity_integral(self, T: float | torch.Tensor, ts: MVEventData | list[MVEventData]):
        raise NotImplementedError("Optional.")

    def get_parameters(self):
        raise NotImplementedError("Implement: Return the true parameters of the kernel.")

    def _enocde_parameters(self):
        raise NotImplementedError()

    def likelihood(self, T: float, ts: MVEventData | list[MVEventData], log: bool = True):
        raise NotImplementedError()

    def sample(self, T, ts: MVEventData):
        raise NotImplementedError()

    def integral_numerical(self, T: float, ts: MVEventData, num_integration_points: int = 1000):
        # Currently not used as we use the exponential kernel.
        # However, good for comparing performance for later kernels.

        if self.INTEGRATION_MODE == "trapezoidal":
            evaluation_points = torch.linspace(0, T, num_integration_points)

            ## Find the indices of events to include at each evaluation point
            ## Vectorized PyTorch implementation
            included_events_idcs = torch.searchsorted(ts.time_points, evaluation_points, right=False)
            ## Old loop-based implementation
            # included_events_idcs = torch.zeros_like(evaluation_points, dtype=torch.long)
            # curr_idx = 0
            # for i,t in enumerate(evaluation_points):
            #    #Only include events before time t
            #    while curr_idx < len(ts) and ts.time_points[curr_idx] < t:
            #        curr_idx += 1
            #    included_events_idcs[i] = curr_idx

            ## Compute intensity values at each evaluation point
            ## Fast implementation.
            intensity_values = torch.stack(
                [self.intensity(t, ts[:ie_idx]) for t, ie_idx in zip(evaluation_points, included_events_idcs)]
            )  # Shape: (num_integration_points, D)
            integral = torch.trapezoid(intensity_values, evaluation_points, dim=0)  # Shape: (D,)

            ## Old loop-based implementation
            # for i, t in enumerate(evaluation_points):
            #    #Only include events before time t
            #    while included_events_idx < len(ts) and ts.time_points[included_events_idx] < t:
            #        included_events_idx += 1
            #    relevant_events = ts[:included_events_idx]
            #    intensity_values[i] = self.intensity( t, relevant_events)
            # integral = _trapz(intensity_values, evaluation_points)

            return integral

        elif self.INTEGRATION_MODE == "monte_carlo":
            # Monte-Carlo integration
            # Should not be used (high variance and unoptimized implementation)
            # Raise warning on first use that it is not recommended.
            if not hasattr(self, "_mc_warning_issued"):
                print(
                    "Warning: Monte Carlo integration mode is not recommended due to high variance and unoptimized implementation."
                )
                self._mc_warning_issued = True

            rng = torch.Generator()

            evaluation_points = T * torch.rand(size=(num_integration_points,), generator=rng)
            included_events_idcs = torch.searchsorted(ts.time_points, evaluation_points, right=False)
            intensity_values = torch.stack(
                [self.intensity(t, ts[:ie_idx]) for t, ie_idx in zip(evaluation_points, included_events_idcs)]
            )  # Shape: (num_integration_points, D)

            integral = (T / num_integration_points) * torch.sum(intensity_values)

            return integral

        elif self.INTEGRATION_MODE == "mc_trapezoidal":
            # Hybrid Monte-Carlo + Trapezoidal integration
            # Trapezoidal integration on random evaluation points. Beginning and end points are always included.

            rng = torch.Generator()

            evaluation_points = T * torch.rand(size=(num_integration_points - 2,), generator=rng)
            evaluation_points = torch.cat([torch.tensor([0.0]), evaluation_points, torch.tensor([T])])
            evaluation_points, _ = torch.sort(evaluation_points)

            included_events_idcs = torch.searchsorted(ts.time_points, evaluation_points, right=False)
            intensity_values = torch.stack(
                [self.intensity(t, ts[:ie_idx]) for t, ie_idx in zip(evaluation_points, included_events_idcs)]
            )  # Shape: (num_integration_points, D)

            integral = torch.trapezoid(intensity_values, evaluation_points, dim=0)  # Shape: (D,)

            return integral

        else:
            raise ValueError(f"Unknown INTEGRATION_MODE {self.INTEGRATION_MODE}")


class ExpKernelMVHawkesProcess(torch.nn.Module):
    def __init__(
        self,
        params: Optional[ExpKernelParams],
        D: Optional[int] = None,
        seed: Optional[int] = 42,
        integration_mode: Literal["trapezoidal", "monte_carlo", "mc_trapezoidal"] = "trapezoidal",
    ):
        """Initialize the MultivariateHawkes process.

        Args:
            params: Hawkes process parameters or None to initialize randomly
            D: Event dimension (required if params is None)
            seed: random seed for initialization
            reg_lambda: regularization parameter
            integration_mode: How to estimate the intensity integral in the likelihood computation
        """

        super().__init__()

        generator = torch.Generator()
        if seed is not None:
            generator = generator.manual_seed(seed)

        self.generator = generator

        if params is None and D is None:
            raise ValueError("Either params or D (number of dimensions) must be provided.")

        if params is None:
            # Decay rates: Effect goes down a factor of e every x steps.
            # Our time scale is years, so maybe we can init it sensibly, between ratioing every month to 10 years.
            ratios_every_years = torch.rand(size=(D, D)) * (10 - 1 / 12) + 1 / 12  # Random from one month to 10 years.
            betas = 1 / ratios_every_years

            # Amplitude is alpha*beta. Therefore branching ratio is max_abs_eig alpha.
            targeted_branching_ratio = 0.5
            alphas = torch.rand(size=(D, D)) * 100 + 1
            max_eig = torch.max(torch.abs(torch.linalg.eigvals(alphas)))
            alphas = (1 / max_eig) * targeted_branching_ratio * alphas

            # Finally init mu. Not sure what to do here.
            mu = torch.rand(size=(D,)) * (10 - 1 / 12) + 1 / 12

            self.mu = torch.nn.Parameter(inverse_softplus(mu))
            self.log_alpha = torch.nn.Parameter(inverse_softplus(alphas))
            self.log_beta = torch.nn.Parameter(inverse_softplus(betas))
            self.ensure_stability(radius=0.55)  # Limit spectral radius of alpha/beta matrix to be < 1.

        else:
            if D is None:
                D = params.mu.shape[0]

            assert params.mu.shape[0] == D, "Dimension mismatch in mu"
            assert params.alpha.shape == (D, D), "Dimension mismatch in alpha"
            assert params.beta.shape == (D, D), "Dimension mismatch in beta"
            self.mu = torch.nn.Parameter(inverse_softplus(params.mu))
            self.log_alpha = torch.nn.Parameter(inverse_softplus(params.alpha))
            self.log_beta = torch.nn.Parameter(inverse_softplus(params.beta))
            stable, max_eig = self.check_stability()
            if not stable:
                print(f"Warning: Provided parameters are unstable. Spectral radius: {max_eig} >= 1.")
        self.INTEGRATION_MODE = integration_mode

    def check_stability(self):
        # Check if the Hawkes process is stable (spectral radius of alpha/beta < 1)
        _, alpha, _ = self.transform_params()
        kernel_matrix = alpha  # Shape: (D,D)
        eigenvalues = torch.linalg.eigvals(kernel_matrix)
        spectral_radius = torch.max(torch.abs(eigenvalues)).item()
        return spectral_radius < 1, spectral_radius

    def ensure_stability(self, radius: float = 1.0):
        # Compute the spectral radius
        _, alpha, _ = self.transform_params()
        kernel_matrix = alpha  # Shape: (D,D)
        eigenvalues = torch.linalg.eigvals(kernel_matrix)
        spectral_radius = torch.max(torch.abs(eigenvalues)).item()

        if spectral_radius < radius:
            return

        self.log_alpha.data = inverse_softplus(alpha / ((1 / radius) * spectral_radius) - 1e-7)

    def get_amplitude_decay(self):
        _, alpha, beta = self.transform_params()

        amplitude = alpha * beta
        decay = beta

        return amplitude, decay

    def transform_params(self, mu=True, alpha=True, beta=True):
        # Constrain parameters alpha and beta and mu to be positive

        parameters = []

        if mu:
            parameters += [F.softplus(self.mu)]
        if alpha:
            parameters += [F.softplus(self.log_alpha)]
        if beta:
            parameters += [F.softplus(self.log_beta)]

        return tuple(parameters)

    def integral_exp_kernel(self, T: torch.Tensor, ts: BatchedMVEventData):
        mu, alpha, beta = self.transform_params()

        valid_mask = ts.event_types != -1  # Shape: (B,N)
        integral = mu.unsqueeze(1) * T.unsqueeze(0)  # Shape: (D,B)

        if len(ts) > 0:
            # Vectorized multivariate implementation
            delta_t = (T.unsqueeze(1) - ts.time_points).unsqueeze(0)  # Shape: (1,B,N)
            delta_t = torch.where(valid_mask.unsqueeze(0), delta_t, torch.zeros_like(delta_t))

            relevant_alpha = alpha[:, ts.event_types]  # Shape: (D,B,N)
            relevant_beta = beta[:, ts.event_types]  # Shape: (D,B,N)
            contributions = relevant_alpha * (1 - torch.exp(-relevant_beta * delta_t))  # Shape: (D,B,N)
            contributions = contributions.masked_fill(~valid_mask.unsqueeze(0), 0)
            integral = integral + torch.sum(contributions, dim=2)  # Shape: (D,B)
        integral = integral.T
        return integral

    def positive_likelihood(
        self,
        ts: BatchedMVEventData,
        log: bool = True,
    ):
        # Uses a lower triangular matrix to compute all intensities at once

        # Shape ts: [batch_size, len]
        # ts arrays are right padded. np.inf for time and -1 for type

        # Identify valid (non-padded) events
        valid_events = ts.event_types != -1
        valid_event_mask = valid_events.unsqueeze(2) & valid_events.unsqueeze(1)

        # Computes the time-difference matrix T.
        # $T_{b,i,j} = t_b,i - t_b,j$ if $i > j$ else 0
        time_diffs = ts.time_points.unsqueeze(2) - ts.time_points.unsqueeze(1)  # Shape: (B, N, N)
        # Set time_diffs to 0, where we have interactions with padded events (can results in +-inf or NaN).
        time_diffs_safe = torch.where(valid_event_mask, time_diffs, torch.zeros_like(time_diffs))

        # Get receiver (j) types: (B, N) -> (B, N, 1)
        receiver_types = ts.event_types.unsqueeze(2)

        # Get trigger (i) types: (B, N) -> (B, 1, N)
        trigger_types = ts.event_types.unsqueeze(1)

        # Shape: D, (D,D), (D,D)
        mu, alpha, beta = self.transform_params()

        # Create a matrix of shape (B, N, N) where each entry (b,i,j) corresponds to alpha_{receiver_types[b,i], trigger_types[b,j]}
        alpha_matrix = alpha[receiver_types, trigger_types].clone()  # Shape: (B, N, N)
        beta_matrix = beta[receiver_types, trigger_types].clone()  # Shape: (B, N, N)

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

        relevant_mu = mu[ts.event_types] * valid_events  # Shape (B,N)

        # We now want the impact of past events on the current event.
        # For this we sum the rows. To get the intensities we also add the correct mu values.
        intensities = relevant_mu + torch.sum(interaction_terms, dim=2)  # Shape: (B,N,)

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
            # For the exponential kernel, we can compute the integral analytically.
            integral = self.integral_exp_kernel(T=T, ts=ts)
        else:
            raise RuntimeError("We should not be here.")
            # Rely on numerical integration.
            integral = self._integral_numerical(T, ts, num_integration_points)

        if log:
            negative_likelihood = torch.sum(-integral, dim=1)  # Shape: (B,)
        else:
            negative_likelihood = torch.prod(torch.exp(-integral), dim=1)  # Shape: (B,)

        return negative_likelihood

    def likelihood(
        self,
        ts: MVEventData,
        T: torch.Tensor,
        log: bool = True,
    ):
        """
        Compute the likelihood of observed event times under the Hawkes process model.

        Args:
            params: Hawkes process parameters (Params NamedTuple).
            ts: Array of event times.
            T: End time for the observation window (defaults to last event).
            num_integration_points: Number of points for numerical integration (trapezoidal rule).

        Returns:
            The likelihood value (float) for the observed events.

        Mathematical context:
            Likelihood = product of intensities at event times
                        minus the integral of the intensity over [0, T].
        """

        # Positive likelihood
        # ----
        # Efficient vectorized implementation.
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

    def _integral_numerical(self, T: float, ts: MVEventData, num_integration_points: int = 1000):
        # Currently not used as we use the exponential kernel.
        # However, good for comparing performance for later kernels.

        if self.INTEGRATION_MODE == "trapezoidal":
            evaluation_points = torch.linspace(0, T, num_integration_points)

            ## Find the indices of events to include at each evaluation point
            ## Vectorized PyTorch implementation
            included_events_idcs = torch.searchsorted(ts.time_points, evaluation_points, right=False)
            ## Old loop-based implementation
            # included_events_idcs = torch.zeros_like(evaluation_points, dtype=torch.long)
            # curr_idx = 0
            # for i,t in enumerate(evaluation_points):
            #    #Only include events before time t
            #    while curr_idx < len(ts) and ts.time_points[curr_idx] < t:
            #        curr_idx += 1
            #    included_events_idcs[i] = curr_idx

            ## Compute intensity values at each evaluation point
            ## Fast implementation.
            intensity_values = torch.stack(
                [self.intensity(t, ts[:ie_idx]) for t, ie_idx in zip(evaluation_points, included_events_idcs)]
            )  # Shape: (num_integration_points, D)
            integral = torch.trapezoid(intensity_values, evaluation_points, dim=0)  # Shape: (D,)

            ## Old loop-based implementation
            # for i, t in enumerate(evaluation_points):
            #    #Only include events before time t
            #    while included_events_idx < len(ts) and ts.time_points[included_events_idx] < t:
            #        included_events_idx += 1
            #    relevant_events = ts[:included_events_idx]
            #    intensity_values[i] = self.intensity( t, relevant_events)
            # integral = _trapz(intensity_values, evaluation_points)

            return integral

        elif self.INTEGRATION_MODE == "monte_carlo":
            # Monte-Carlo integration
            # Should not be used (high variance and unoptimized implementation)
            # Raise warning on first use that it is not recommended.
            if not hasattr(self, "_mc_warning_issued"):
                print(
                    "Warning: Monte Carlo integration mode is not recommended due to high variance and unoptimized implementation."
                )
                self._mc_warning_issued = True

            rng = torch.Generator()

            evaluation_points = T * torch.rand(size=(num_integration_points,), generator=rng)
            included_events_idcs = torch.searchsorted(ts.time_points, evaluation_points, right=False)
            intensity_values = torch.stack(
                [self.intensity(t, ts[:ie_idx]) for t, ie_idx in zip(evaluation_points, included_events_idcs)]
            )  # Shape: (num_integration_points, D)

            integral = (T / num_integration_points) * torch.sum(intensity_values)

            return integral

        elif self.INTEGRATION_MODE == "mc_trapezoidal":
            # Hybrid Monte-Carlo + Trapezoidal integration
            # Trapezoidal integration on random evaluation points. Beginning and end points are always included.

            rng = torch.Generator()

            evaluation_points = T * torch.rand(size=(num_integration_points - 2,), generator=rng)
            evaluation_points = torch.cat([torch.tensor([0.0]), evaluation_points, torch.tensor([T])])
            evaluation_points, _ = torch.sort(evaluation_points)

            included_events_idcs = torch.searchsorted(ts.time_points, evaluation_points, right=False)
            intensity_values = torch.stack(
                [self.intensity(t, ts[:ie_idx]) for t, ie_idx in zip(evaluation_points, included_events_idcs)]
            )  # Shape: (num_integration_points, D)

            integral = torch.trapezoid(intensity_values, evaluation_points, dim=0)  # Shape: (D,)

            return integral

        else:
            raise ValueError(f"Unknown INTEGRATION_MODE {self.INTEGRATION_MODE}")

    def sample(self, Tstart: float, Tend: float, ts: Optional[MVEventData] = None, max_samples: Optional[float] = None):
        """Generate a sample from the Hawkes process using Ogata's thinning algorithm.

        Args:
            Tstart: Start time
            Tend: End time
            events: Initial events (optional)
            seed: Random seed
        """

        # TODO include seeding for reproducibility.

        # Required if the process is potentially unstable (infinite loop)
        num_samples = 0

        if ts is None:
            events = []
            event_types = []
        else:
            # unpack times and types
            events = ts.time_points.tolist()
            event_types = ts.event_types.tolist()

        # Generators do not work with torch distributions...
        # generator = torch.Generator()

        t = Tstart

        while t < Tend:
            lambdas = self.intensity(
                t,
                MVEventData(time_points=torch.tensor(events), event_types=torch.tensor(event_types, dtype=torch.long)),
            )
            lambda_sum = lambdas.sum().item() + 1e-8  # total intensity
            delta_t = torch.distributions.Exponential(lambda_sum).sample().item()
            t_proposed = t + delta_t
            if t_proposed > Tend:
                break
            # Get intensity at proposed time
            lambdas_prop = self.intensity(
                t_proposed,
                MVEventData(time_points=torch.tensor(events), event_types=torch.tensor(event_types, dtype=torch.long)),
            )
            lambda_sum_prop = lambdas_prop.sum().item()
            if torch.rand(1) <= lambda_sum_prop / lambda_sum:
                # Accept
                probs = lambdas_prop / lambda_sum_prop
                event_type = torch.multinomial(probs, 1).item()
                events.append(t_proposed)
                event_types.append(event_type)
                num_samples += 1
                if max_samples is not None and num_samples >= max_samples:
                    break
            t = t_proposed
        return MVEventData(time_points=torch.tensor(events), event_types=torch.tensor(event_types, dtype=torch.long))

    def sample_naive(self, Tstart: float, Tend: float, step_size: float, ts: Optional[MVEventData] = None):
        """Generate a sample from the Hawkes process using a very naive stepping algorithm.

        Args:
            Tstart: Start time
            Tend: End time
            delta_t: Time step
            events: Initial events (optional)
            seed: Random seed
        """

        if ts is None:
            events = []
            event_types = []
        else:
            # unpack times and types
            events = ts.time_points.tolist()
            event_types = ts.event_types.tolist()

        t = Tstart

        while t < Tend:
            lambdas = self.intensity(
                t,
                MVEventData(time_points=torch.tensor(events), event_types=torch.tensor(event_types, dtype=torch.long)),
            )
            # We approximate the intensities as locally constant. Then we can treat them all as if they were
            # exponentially distributed in a very small time window [t,t+delta_t]

            # We sample all exponentials at the same time, through batching.
            samples = torch.distributions.Exponential(lambdas).sample((1,))[0]

            # These samples are only "valid" for a very small time-windows (as the intensities are not constant but decreasing).
            accepted_samples = samples <= step_size
            accepted_event_types = torch.arange(len(lambdas))[accepted_samples]
            accepted_event_times = t + samples[accepted_samples]

            sort_idcs = torch.argsort(accepted_event_times)

            accepted_event_times = accepted_event_times[sort_idcs]
            accepted_event_types = accepted_event_types[sort_idcs]

            for t, e in zip(accepted_event_times, accepted_event_types):
                events.append(t.item())
                event_types.append(e.item())

        return MVEventData(time_points=torch.tensor(events), event_types=torch.tensor(event_types, dtype=torch.long))
