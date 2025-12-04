# My own implementation of the Hawkes process
# %%
from dataclasses import dataclass
from typing import Literal, Optional
import torch
import torch.nn.functional as F
from einops import rearrange

from event_utils import MVEventData, inverse_softplus

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
            self.mu = torch.nn.Parameter(inverse_softplus(torch.ones(size=(D,), dtype=torch.float32)) * 1e-8)
            self.log_alpha = torch.nn.Parameter(inverse_softplus(x=torch.rand(D, D, generator=generator) + 1e-8))
            self.log_beta = torch.nn.Parameter(inverse_softplus(torch.rand(D, D, generator=generator) + 1e-8))
            self.ensure_stability(radius=1.0)  # Limit spectral radius of alpha/beta matrix to be < 1.

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
        _, alpha, beta = self.transform_params()
        kernel_matrix = alpha / beta  # Shape: (D,D)
        eigenvalues = torch.linalg.eigvals(kernel_matrix)
        spectral_radius = torch.max(torch.abs(eigenvalues)).item()
        return spectral_radius < 1, spectral_radius

    def ensure_stability(self, radius: float = 1.0):
        # Compute the spectral radius
        _, max_eig = self.check_stability()
        if max_eig >= 1.0:
            # print(f"Rescaling alpha to ensure stability. Current spectral radius: {max_eig}")
            _, alpha, beta = self.transform_params()
            self.log_alpha.data = inverse_softplus(alpha / ((1 / radius) * max_eig) + 1e-7)

            # self.log_alpha.data = inverse_softplus(alpha / ((1/radius) * max_eig)+1e-7)
            # self.log_beta.data = inverse_softplus(beta * np.sqrt(((1/radius)*max_eig)+1e-7))

            # print(f"New spectral radius: {self.check_stability()[1]}")

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

    def integral_exp_kernel(self, T, ts: BatchedMVEventData):
        mu, alpha, beta = self.transform_params()

        valid_mask = ts.event_types != -1
        integral = (mu * T).unsqueeze(-1)  # Shape: (D,1)

        if len(ts) > 0:
            # Vectorized multivariate implementation
            delta_t = (T - ts.time_points).unsqueeze(0)  # Shape: (1,B,N)
            relevant_alpha = alpha[:, ts.event_types]  # Shape: (D,B,N)
            relevant_beta = beta[:, ts.event_types]  # Shape: (D,B,N)
            contributions = relevant_alpha * (1 - torch.exp(-relevant_beta * delta_t))  # Shape: (D,B,N)
            contributions = contributions.permute(1, 2, 0)  # Shape (B,N,D)
            contributions.masked_fill(~valid_mask.unsqueeze(-1), 0)  # Set non valid numbers to zero. Use broadcasting.
            contributions = contributions.permute(2, 0, 1)  # Shape: (D,B,N)
            integral += torch.sum(contributions, dim=2)  # Shape: (D,B)
        integral = integral.T
        return integral

    def positive_likelihood(
        self,
        ts: MVEventData,
        log: bool = True,
    ):
        # Uses a lower triangular matrix to compute all intensities at once

        # Shape ts: [batch_size, len]
        # ts arrays are right padded. np.inf for time and -1 for type

        # Computes the time-difference matrix T.
        # $T_{b,i,j} = t_b,i - t_b,j$ if $i > j$ else 0

        time_diffs = rearrange(ts.time_points, "b l -> b l 1") - rearrange(
            ts.time_points, "b l -> b 1 l"
        )  # Shape B N N

        # Get receiver (j) types: (B, N) -> (B, N, 1)
        receiver_types = rearrange(ts.event_types, "b l -> b l 1")

        # Get trigger (i) types: (B, N) -> (B, 1, N)
        trigger_types = rearrange(ts.event_types, "b l -> b 1 l")

        mu, alpha, beta = self.transform_params()

        # Create a matrix of shape (B, N, N) where each entry (b, i,j) corresponds to alpha_{receiver_types[b,i], trigger_types[b,j]}
        alpha_matrix = alpha[receiver_types, trigger_types]  # Shape: (B, N, N)
        beta_matrix = beta[receiver_types, trigger_types]  # Shape: (B, N, N)

        # Compute the kernel values for all time differences
        interaction_terms = alpha_matrix * beta_matrix * torch.exp(-beta_matrix * time_diffs)  # Shape: (B,N, N)
        interaction_terms = torch.tril(
            interaction_terms, diagonal=-1
        )  # Zero out diagonal and above, to ensure causality. This works also with batched matrices.

        # Mask out invalid events.
        valid_events = ts.event_types != -1
        valid_event_mask = rearrange(valid_events, "B N -> B N 1") & rearrange(valid_events, "B N -> B 1 N")
        # TODO Make sure multiplication works here. Maybe we need indexing.
        interaction_terms *= valid_event_mask

        relevant_mu = mu[ts.event_types] * valid_events  # Shape (B,N)

        # We now want the impact of past events on the current event.
        # For this we sum the rows. To get the intensities we also add the correct mu values.
        intensities = relevant_mu + torch.sum(interaction_terms, dim=2)  # Shape: (B,N,)

        # TODO make sure it works with empty event sequences for some batch elements.
        if log:
            positive_likelihood = torch.sum(input=torch.log(intensities), dim=-1)  # Shape: (B,)
        else:
            positive_likelihood = torch.prod(intensities, dim=-1)  # Shape: (B,)

        return positive_likelihood

    def likelihood(
        self,
        ts: MVEventData,
        T: float,
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

        num_integration_points = 0

        # TODO throw error when T <= last event time
        num_events = (ts.event_types != -1).sum(dim=1)

        # Positive likelihood
        # ----
        # Efficient vectorized implementation.
        positive_likelihood = self.positive_likelihood(ts, log)

        # Mask out event sequences with no elements. They only contribute negativly.
        num_0 = num_events == 0
        positive_likelihood[num_0] = torch.tensor(0.0) if log else torch.tensor(1.0)

        # ----
        # Negative likelihood
        # ----

        if num_integration_points == 0:
            # For the exponential kernel, we can compute the integral analytically.
            integral = self.integral_exp_kernel(T, ts)
        else:
            # Rely on numerical integration.
            integral = self._integral_numerical(T, ts, num_integration_points)

        if log:
            negative_likelihood = torch.sum(-integral, dim=1)  # Shape: (B,)
        else:
            negative_likelihood = torch.prod(torch.exp(-integral), dim=1)  # Shape: (B,)

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


# %%

from event_utils import MVEventData
from torch import Tensor
from tensordict import TensorDict


class BatchedMVEventData(TensorDict):
    def __init__(self, time_points: list[Tensor], event_types: list[Tensor], sort: bool = False):
        assert isinstance(time_points, list)
        assert isinstance(event_types, list)
        assert len(time_points) == len(event_types)

        batch_size = len(time_points)

        for t, e in zip(time_points, event_types):
            assert t.dim() == 1
            assert e.dim() == 1
            assert t.shape == e.shape

        # Assert time_points have float type
        assert time_points[0].dtype.is_floating_point
        # Assert event_types have integer type
        assert not event_types[1].dtype.is_floating_point

        if sort:
            for i in range(len(time_points)):
                sorted_indices = torch.argsort(time_points[i])
                time_points[i] = time_points[i][sorted_indices]
                event_types[i] = event_types[i][sorted_indices]

        lengths = [len(b) for b in time_points]
        length = max(lengths)
        device = time_points[0].device
        dtype_time = time_points[0].dtype
        dtype_event = event_types[0].dtype

        padded_time_points = torch.full((batch_size, length), float("inf"), dtype=dtype_time, device=device)
        padded_event_types = torch.full((batch_size, length), -1, dtype=dtype_event, device=device)

        for i, (t, e) in enumerate(zip(time_points, event_types)):
            n = len(t)
            padded_time_points[i, :n] = t
            padded_event_types[i, :n] = e

        super().__init__(
            {"time_points": padded_time_points, "event_types": padded_event_types}, batch_size=(batch_size, length)
        )

    @property
    def time_points(self):
        return self["time_points"]

    @property
    def event_types(self):
        return self["event_types"]

    def __iter__(self):
        # Iteration yields a tuple for each event
        for i in range(len(self)):
            yield self.time_points[i], self.event_types[i]

    # def __repr__(self):

    #     def abbrev(tensor, max_len=8, float_fmt="{:.2f}"):
    #         l = tensor.tolist()
    #         if tensor.dtype.is_floating_point:
    #             l = [float_fmt.format(v) for v in l]
    #         else:
    #             l = [str(v) for v in l]
    #         if len(l) > max_len:
    #             return "[" + ", ".join(l[:4]) + ", ..., " + ", ".join(l[-3:]) + "]"
    #         return "[" + ", ".join(l) + "]"

    #     batch_size = self.time_points.shape[0]
    #     length = self.time_points.shape[1]

    #     if batch_size == 0:
    #         return (
    #             f"BatchedMVEventData(batch=0, len={length}, [{self.time_points.dtype}, {self.event_types.dtype}])\n"
    #             f"  time_points: {tps}\n"
    #             f"  event_types: {ets}"
    #         )

    #     if batch_size in [0, 1]:
    #         tps = abbrev(self.time_points)
    #         ets = abbrev(self.event_types)
    #     return (
    #         f"BatchedMVEventData(len={length}, [{self.time_points.dtype}, {self.event_types.dtype}])\n"
    #         f"  time_points: {tps}\n"
    #         f"  event_types: {ets}"
    #     )


# %%

t = [torch.tensor([0.0, 0.2, 0.7]), torch.tensor([0.6, 0.7]), torch.tensor([])]
e = [torch.tensor([0, 1, 0]), torch.tensor([1, 1]), torch.tensor([])]

batch = BatchedMVEventData(t, e)
# %%
hp = ExpKernelMVHawkesProcess(None, D=2)
# %%
hp.positive_likelihood(batch)
# %%
