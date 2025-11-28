# My own implementation of the Hawkes process
#%%
from dataclasses import dataclass
from typing import Literal, NamedTuple, Optional
import numpy as np
from tqdm import tqdm
from jaxtyping import Float as tFloat, Integer as tInteger
from matplotlib import pyplot as plt
from tensordict import TensorDict
import torch
import torch.nn.functional as F

from torch import tensor, Tensor

from event_utils import MVEventData, inverse_softplus

#%%

@dataclass
class Params:
    mu: torch.Tensor
    alpha: torch.Tensor
    beta: torch.Tensor


class UnivariateHawkesProcess(torch.nn.Module):

    def __init__(self, params:Optional[Params], seed=42, reg_lambda=None, integration_mode:Literal["trapezoidal", "monte_carlo", "mc_quadrature"]="trapezoidal"):
        """Initialize the Hawkes process.
        
        Args:
            key: random key for initialization
            reg_lambda: regularization parameter
        """
        
        super().__init__()

        self.reg_lambda = reg_lambda

        generator = torch.Generator()
        if seed is not None:
            generator = generator.manual_seed(seed)

        self.generator = generator

        if params is None:
            self.mu = torch.nn.Parameter(tensor(0.0, dtype=torch.float32))
            self.log_alpha = torch.nn.Parameter(torch.log(torch.rand(generator=generator)+1e-8))
            self.log_beta = torch.nn.Parameter(torch.log(torch.rand(generator=generator)+1e-8))
        else:
            self.mu = torch.nn.Parameter(params.mu)
            self.log_alpha = torch.nn.Parameter(torch.log(params.alpha))
            self.log_beta = torch.nn.Parameter(torch.log(params.beta))

        self.INTEGRATION_MODE = integration_mode


    def transform_params(self):
        # Constrain parameters alpha and beta to be positive
        return torch.exp(self.log_alpha), torch.exp(self.log_beta)

    def exp_kernel(self, x):
        # Returns pdf of exponential distribution with rate beta scaled by alpha
        alpha, beta = self.transform_params()
        return alpha * beta * torch.exp(-beta * x)

    def integral_exp_kernel(self, T, ts):
        alpha, beta = self.transform_params()

        integral = self.mu*T
        if len(ts) > 0:
            integral += torch.sum(torch.stack([(alpha/beta)*(1 - torch.exp(-beta*(T - ti))) for ti in ts]))
        return integral

    def intensity(self, t, ts):
        # Compute the intensity at time t given past events ts
        
        if len(ts) == 0:
            return self.mu
        
        mask = ts < t
        contributions = torch.stack([self.exp_kernel(t - ti) for ti in ts])
        current_intensity = self.mu + torch.sum(contributions * mask)
        return current_intensity

    def likelihood(
        self,
        ts: torch.Tensor,
        T: float,
        log: bool = True,
        num_integration_points: int = 1000,
        vectorized: bool = False,
        analy_kernel: bool = True,
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

        
        #TODO handel empty ts case
        if len(ts) == 0:
            # We received an empty event list
            intensities = torch.stack([self.intensity( T, ts)])
        else:
            indices = torch.arange(len(ts))
            intensities = torch.stack([self.intensity( ts[i], ts[:i]) for i in indices])

        if log:
            positive_likelihood = torch.sum(input=torch.log(intensities))
        else:
            positive_likelihood = torch.prod(intensities)
        
            
        if num_integration_points == 0 and analy_kernel:
            # For the exponential kernel, we can compute the integral analytically
            integral = self.integral_exp_kernel( T, ts)

        else:
            integral = self._integral_numerical( T, ts,  num_integration_points)

        if log:
            negative_likelihood = -integral
            return positive_likelihood + negative_likelihood
        else:
            negative_likelihood = torch.exp(-integral)
            return positive_likelihood * negative_likelihood


    def _integral_numerical(self, T:float, ts:torch.Tensor, num_integration_points:int=1000):




        if self.INTEGRATION_MODE == "trapezoidal":

            evaluation_points = torch.linspace(0, T, num_integration_points)


            # Make sure events are time-sorted
            ts, _ = torch.sort(ts)

            intensity_values = torch.zeros_like(evaluation_points)

            included_events_idx = 0


            for i, t in enumerate(evaluation_points):
                #Only include events before time t
                while included_events_idx < len(ts) and ts[included_events_idx] < t:
                    included_events_idx += 1
                relevant_events = ts[:included_events_idx]
                intensity_values[i] = self.intensity( t, relevant_events)
            

            # Old inefficient version
            #intensity_values = torch.tensor([self.intensity( t, ts) for t in evaluation_points])

            integral = _trapz(intensity_values, evaluation_points)
            return integral
        
        elif self.INTEGRATION_MODE == "monte_carlo":

            # Monte Carlo integration
            rng = torch.Generator()

            sample_points = T * torch.rand(size=(num_integration_points,), generator=rng)

            intensity_values = torch.stack([self.intensity( t, ts) for t in sample_points])

            integral = (T / num_integration_points) * torch.sum(intensity_values)
            return integral


    def sample(self, Tstart:float, Tend:float, ts:Optional[torch.Tensor]=None, seed:Optional[int]=None):
        """Generate a sample from the Hawkes process using Ogata's thinning algorithm.
        
        Args:
            Tstart: Start time
            Tend: End time
            events: Initial events (optional)
            seed: Random seed
        """
        if ts is None:
            events = []
        else:
            events = ts.tolist()

        generator = torch.Generator()
        if seed is not None:
            generator = generator.manual_seed(seed)

        intensities = []
        t = Tstart

        while t < Tend:
            # Find a value lambda_star >= intensity everywhere after t
            # For Hawkes: intensity increases after each event, so set lambda_star = current intensity + epsilon
            lambda_star = self.intensity(t, torch.tensor(events))
            # Small epsilon to avoid zero intensity
            lambda_star = lambda_star.item() + 1e-8

            # Propose next event
            delta_t = torch.distributions.Exponential(lambda_star).sample().item()
            t_proposed = t + delta_t

            if t_proposed > Tend:
                break

            # Accept or reject
            lambda_prop = self.intensity(t_proposed, torch.tensor(events)).item()
            D = torch.rand(size=(), generator=generator).item()
            p_accept = lambda_prop / lambda_star

            intensities.append(SampleStorage(intensity=lambda_prop, uni_s=D, delta_t=delta_t, p_event=p_accept))

            if D <= p_accept:
                events.append(t_proposed)
                t = t_proposed
            else:
                t = t_proposed
                    
        return torch.tensor(events), intensities

class MultiVariateHawkesProcess(torch.nn.Module):
    #TODO handle the empty event list case in intensity and likelihood. Might still be off.
    # Is it correct that the intensity is always mu? Or should it be mu*T?

    #TODO 
    def __init__(self, params:Optional[Params], D:Optional[int]=None, seed:Optional[int]=42, reg_lambda:Optional[float]=None, integration_mode:Literal["trapezoidal", "monte_carlo"]="trapezoidal"):
        """Initialize the MultivariateHawkes process.
        
        Args:
            params: Hawkes process parameters or None to initialize randomly
            D: Event dimension (required if params is None)
            seed: random seed for initialization
            reg_lambda: regularization parameter
            integration_mode: How to estimate the intensity integral in the likelihood computation
        """
        
        super().__init__()

        self.reg_lambda = reg_lambda # Unused for now

        generator = torch.Generator()
        if seed is not None:
            generator = generator.manual_seed(seed)

        self.generator = generator

        if params is None and D is None:
            raise ValueError("Either params or D (number of dimensions) must be provided.")

        if params is None:
            self.mu = torch.nn.Parameter(inverse_softplus(torch.ones(size=(D,), dtype=torch.float32))*1e-8)
            self.log_alpha = torch.nn.Parameter(inverse_softplus(torch.rand(D,D,generator=generator)+1e-8))
            self.log_beta = torch.nn.Parameter(inverse_softplus(torch.rand(D,D,generator=generator)+1e-8))
            self.ensure_stability(radius=0.5) # Limit spectral radius of alpha/beta matrix to be < 1.
            
        else:
            if D is None:
                D = params.mu.shape[0]
            
            assert params.mu.shape[0] == D, "Dimension mismatch in mu"
            assert params.alpha.shape == (D,D), "Dimension mismatch in alpha"
            assert params.beta.shape == (D,D), "Dimension mismatch in beta"
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

    def ensure_stability(self, radius:float= 1.0):
        # Compute the spectral radius
        _, max_eig = self.check_stability()
        if max_eig >= 1.0:
            print(f"Rescaling alpha to ensure stability. Current spectral radius: {max_eig}")
            _, alpha, beta = self.transform_params()
            self.log_alpha.data = inverse_softplus(alpha / np.sqrt(((1/radius) * max_eig)+1e-7))
            self.log_beta.data = inverse_softplus(beta * np.sqrt(((1/radius)*max_eig)+1e-7))

            print(f"New spectral radius: {self.check_stability()[1]}")
        
    def transform_params(self):
        # Constrain parameters alpha and beta and mu to be positive
        return F.softplus(self.mu), F.softplus(self.log_alpha), F.softplus(self.log_beta)

    def exp_kernel(self, x:torch.Tensor, event_type:torch.Tensor):
        # Returns pdf of exponential distribution with rate beta scaled by alpha
        # This is a tensor of shape (D).
        _, _alpha, _beta = self.transform_params() # Shape: (D,D) and (D,D)
        alpha = _alpha[event_type, :] # Shape: (N,D)
        beta = _beta[event_type, :] # Shape: (N,D)

        x = x.unsqueeze(-1)  # Shape: (N,1)

        return alpha * beta * torch.exp(-beta * x)

    def integral_exp_kernel(self, T, ts:MVEventData):
        mu, alpha, beta = self.transform_params()

        integral = mu*T
        if len(ts) > 0:


            #Old Univariate Implementation
            #integral += torch.sum(torch.stack([(alpha/beta)*(1 - torch.exp(-beta*(T - ti))) for ti in ts]))

            #Loopy multivariate implementation
            #contributions = []
            #for ti,ei in ts:
            #    relevant_alpha = alpha[ei, :] # Shape: (D,)
            #    relevant_beta = beta[ei, :]
            #    contributions.append( (relevant_alpha/relevant_beta)*(1 - torch.exp(-relevant_beta*(T - ti))) ) # Shape: (D,)
            #integral += torch.sum(torch.stack(contributions), dim=0)

            #Vectorized multivariate implementation
            # TODO memory problems for many events, might require chunking
            delta_t = (T - ts.time_points).unsqueeze(-1) # Shape: (N,1)
            relevant_alpha = alpha[ts.event_types, :] # Shape: (N,D)
            relevant_beta = beta[ts.event_types, :] # Shape: (N,D)
            contributions = (relevant_alpha/relevant_beta)*(1 - torch.exp(-relevant_beta*delta_t)) # Shape: (N,D)
            integral += torch.sum(contributions, dim=0) # Shape: (D,)

        return integral

    def intensity(self, t:float, ts:MVEventData):
        # Compute the intensity at time t for all event types given past events ts
        
        mu, _, _ = self.transform_params()
        if len(ts) == 0:
            return mu # Shape: (D,)
        
        #mask = ts.time_points < t
        #contributions = torch.stack([self.exp_kernel(t-ti, ei) for ti,ei in zip(ts.time_points, ts.event_type)], dim=0) # Shape: (num_time_points, D)
        #current_intensity = self.mu + torch.sum(contributions * mask, dim=0)

        # Time_points are always sorted
        # Find index of first event after time t
        idx_until_t = torch.searchsorted(ts.time_points, t)
        relevant_events = ts[:idx_until_t] # Returns an MVEventData object
        contributions = self.exp_kernel(t- relevant_events.time_points, relevant_events.event_types) # Shape: (num_events, D)
        intensity = mu + torch.sum(contributions, dim=0)
        return intensity # Shape: (D,)

    def _positive_likelihood_vetorized(
        self,
        ts: MVEventData,
        log: bool = True,
    ):
        # Uses a lower triangular matrix to compute all intensities at once
        
        num_events = len(ts)

        #TODO move triling to the end, to make interaction with exp correct

        # Computes the time-difference matrix T.
        # $T_{i,j} = t_i - t_j$ if $i > j$ else 0
        time_diffs = ts.time_points.unsqueeze(1) - ts.time_points.unsqueeze(0) # Shape: (num_events, num_events)
        time_diffs = torch.tril(time_diffs, diagonal=-1)  # Zero out diagonal and above.


        # Get receiver (j) types: (N) -> (N, 1)
        receiver_types = ts.event_types.unsqueeze(-1)
        
        # Get trigger (i) types: (N) -> (1, N)
        trigger_types = ts.event_types.unsqueeze(0)

        mu, alpha, beta = self.transform_params()

        # Create a matrix of shape (N, N) where each entry (i,j) corresponds to alpha_{receiver_types[i], trigger_types[j]}
        alpha_matrix = alpha[receiver_types, trigger_types] # Shape: (N, N)
        beta_matrix = beta[receiver_types, trigger_types] # Shape: (N, N)

        # Compute the kernel values for all time differences
        interaction_terms = alpha_matrix * beta_matrix * torch.exp(-beta_matrix * time_diffs) # Shape: (N, N)

        # We now want the impact of past events on the current event.
        # For this we sum the rows. To get the intensities we also add the correct mu values.
        intensities = mu[ts.event_types] + torch.sum(interaction_terms, dim=1) # Shape: (N,)

        if log:
            positive_likelihood = torch.sum(input=torch.log(intensities)) # Shape: ()
        else:
            positive_likelihood = torch.prod(intensities) # Shape: ()

        return positive_likelihood
    
    def _negative_likelihood_vectorized(
        self,
        ts: MVEventData,
        T: float,
        log: bool = True,
        num_integration_points: int = 1000,
        analy_kernel: bool = True,
    ):

    def likelihood(
        self,
        ts: MVEventData,
        T: float,
        log: bool = True,
        num_integration_points: int = 1000,
        vectorized: bool = False,
        analy_kernel: bool = True,
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

        num_events = len(ts)
        
        if vectorized and analy_kernel:
            return self._likelihood_vectorized( ts, T, log, num_integration_points, analy_kernel)

        # Positive likelihood
        #----
        
        if num_events == 0:
            #TODO this prob. wrong. Positive likelihood should be zero? But then what about the LL?
            # We received an empty event list
            intensities = torch.stack([self.intensity( T, ts)]) # Shape: (1, D)
        else:
            indices = torch.arange(num_events)
            intensities = torch.stack([self.intensity( ts.time_points[i], ts=ts[:i]) for i in indices]) # Shape: (num_events, D)

        # Only use the occured events.
        occured_intensities = intensities[torch.arange(num_events), ts.event_types] # Shape: (num_events,)
        if log:
            positive_likelihood = torch.sum(input=torch.log(occured_intensities)) # Shape: ()
        else:

            positive_likelihood = torch.prod(occured_intensities) # Shape: ()
        
        #----
        # Negative likelihood
        #----
        #I think everything is correct for the log case...

        if num_integration_points == 0 and analy_kernel:
            # For the exponential kernel, we can compute the integral analytically
            integral = self.integral_exp_kernel( T, ts)

        else:
            integral = self._integral_numerical( T, ts,  num_integration_points)

        if log:
            negative_likelihood = -torch.sum(integral)
            return positive_likelihood + negative_likelihood
        else:
            #TODO Not sure?
            negative_likelihood = torch.prod(torch.exp(-integral))
            return positive_likelihood * negative_likelihood


    def _integral_numerical(self, T:float, ts:MVEventData, num_integration_points:int=1000):

        if self.INTEGRATION_MODE == "trapezoidal":

            evaluation_points = torch.linspace(0, T, num_integration_points)

            # TODO Gurantee that events are time-sorted. Should be done in the MVEventData class

            # Vectorized PyTorch implementation
            included_events_idcs = torch.searchsorted(ts.time_points, evaluation_points, right=False)
            #included_events_idcs = torch.zeros_like(evaluation_points, dtype=torch.long)
            #curr_idx = 0
            #for i,t in enumerate(evaluation_points):
            #    #Only include events before time t
            #    while curr_idx < len(ts) and ts.time_points[curr_idx] < t:
            #        curr_idx += 1
            #    included_events_idcs[i] = curr_idx

            intensity_values = torch.stack([self.intensity(t,ts[:ie_idx]) for t, ie_idx in zip(evaluation_points, included_events_idcs)]) # Shape: (num_integration_points, D)
            integral = torch.trapezoid(intensity_values, evaluation_points, dim=0) # Shape: (D,)            
            

            # Old implementation
            #for i, t in enumerate(evaluation_points):
            #    #Only include events before time t
            #    while included_events_idx < len(ts) and ts.time_points[included_events_idx] < t:
            #        included_events_idx += 1
            #    relevant_events = ts[:included_events_idx]
            #    intensity_values[i] = self.intensity( t, relevant_events)
            #integral = _trapz(intensity_values, evaluation_points)


            return integral
        
        elif self.INTEGRATION_MODE == "monte_carlo":

            # Monte Carlo integration
            # Should not be used (high variance and unoptimized implementation)
            raise ValueError("Monte Carlo integration not implemented for multivariate Hawkes process.")
            rng = torch.Generator()

            sample_points = T * torch.rand(size=(num_integration_points,), generator=rng)

            intensity_values = torch.stack([self.intensity( t, ts) for t in sample_points])

            integral = (T / num_integration_points) * torch.sum(intensity_values)
            return integral
        else:
            raise ValueError(f"Unknown INTEGRATION_MODE {self.INTEGRATION_MODE}")

    def sample(self, Tstart:float, Tend:float, ts:Optional[MVEventData]=None, max_samples:Optional[float]=None):
        """Generate a sample from the Hawkes process using Ogata's thinning algorithm.
        
        Args:
            Tstart: Start time
            Tend: End time
            events: Initial events (optional)
            seed: Random seed
        """

        #TODO include seeding

        num_samples = 0

        if ts is None:
            events = []
            event_types = []
        else:
            # unpack times and types
            events = ts.time_points.tolist()
            event_types = ts.event_types.tolist()

        generator = torch.Generator()

        t = Tstart

        while t < Tend:
            lambdas = self.intensity(t, MVEventData(time_points=torch.tensor(events), event_types=torch.tensor(event_types, dtype=torch.long)))
            lambda_sum = lambdas.sum().item() + 1e-8  # total intensity
            delta_t = torch.distributions.Exponential(lambda_sum).sample().item()
            t_proposed = t + delta_t
            if t_proposed > Tend:
                break
            # Get intensity at proposed time
            lambdas_prop = self.intensity(t_proposed, MVEventData(time_points=torch.tensor(events), event_types=torch.tensor(event_types, dtype=torch.long)))
            lambda_sum_prop = lambdas_prop.sum().item()
            if torch.rand(1) <= lambda_sum_prop / lambda_sum:
                # Accept
                probs = (lambdas_prop / lambda_sum_prop)
                event_type = torch.multinomial(probs, 1).item()
                events.append(t_proposed)
                event_types.append(event_type)
                num_samples += 1
                if max_samples is not None and num_samples >= max_samples:
                    break
            t = t_proposed
        return MVEventData(time_points=torch.tensor(events), event_types=torch.tensor(event_types, dtype=torch.long))
    