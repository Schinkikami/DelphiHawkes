# My own implementation of the univariate Hawkes process
#%%
from typing import NamedTuple
import jax.numpy as jnp
import numpy as np
from jax import random
from functools import partial
from tqdm import tqdm
import jax
import jax.lax as lax

from matplotlib import pyplot as plt

class Params(NamedTuple):
    mu: jax.Array
    log_alpha: jax.Array
    log_beta: jax.Array


def init_params(key, reg_lambda=None):
    """Initialize the Hawkes process.
    
    Args:
        key: JAX random key for initialization
        reg_lambda: regularization parameter
    """
    reg_lambda = reg_lambda

    key1, key2 = random.split(key)
    mu = jnp.array(0.0, dtype=float)
    log_alpha = jnp.log(random.uniform(key1, shape=())+1e-8)
    log_beta = jnp.log(random.uniform(key2, shape=())+1e-8)

    return Params(mu, log_alpha, log_beta)

@jax.jit
def transform_params(params:Params):
    # Constrain parameters alpha and beta to be positive
    return jnp.exp(params.log_alpha), jnp.exp(params.log_beta)
@jax.jit
def exp_kernel(params:Params, x):
    # Returns pdf of exponential distribution with rate beta scaled by alpha
    alpha, beta = transform_params(params)
    return alpha * beta * jnp.exp(-beta * x)

@jax.jit
def integral_exp_kernel(params:Params, T, ts):
    alpha, beta = transform_params(params)
    integral = params.mu*T + jnp.sum(jnp.array([(alpha/beta)*(1 - jnp.exp(-beta*(T - ti))) for ti in ts], dtype=float))
    return integral

@jax.jit
def intensity(params:Params, t, ts):
    # Compute the intensity at time t given past events ts using JAX vectorization
    mask = ts < t
    contributions = jax.vmap(lambda ti: exp_kernel(params, t - ti))(ts)
    current_intensity = params.mu + jnp.sum(contributions * mask)
    return current_intensity

#@partial(jax.jit, static_argnames=['log', 'num_integration_points', 'vectorized', 'analy_kernel'])
def likelihood(
    params: Params,
    ts: jnp.ndarray,
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

    
    if vectorized:
        # Jit and Vmap compatible version. However requires passing in the whole ts array each time. Very suboptimal for early time points.
        if log:
            positive_likelihood = jnp.sum(jax.vmap(lambda ti: jnp.log(intensity(params, ti, ts)))(ts))
        else:
            positive_likelihood = jnp.prod(jax.vmap(lambda ti: intensity(params, ti, ts))(ts))
    else:
        # General non-jit version using a python loop
        indices = jnp.arange(len(ts))
        intensities = jnp.array([intensity(params, ts[i], ts[:i]) for i in indices])
        if log:
            positive_likelihood = jnp.sum(jnp.log(intensities))
        else:
            positive_likelihood = jnp.prod(intensities)
        
        
    if num_integration_points == 0 and analy_kernel:
        # For the exponential kernel, we can compute the integral analytically
        integral = integral_exp_kernel(params, T, ts)

    else:
        integral = _integral_numerical(params, T, ts, intensity, num_integration_points)

    if log:
        negative_likelihood = -integral
        return positive_likelihood + negative_likelihood
    else:
        negative_likelihood = jnp.exp(-integral)
        return positive_likelihood * negative_likelihood

@partial(jax.jit, static_argnames=['intensity_function', 'num_integration_points'])
def _integral_numerical(params:Params, T, ts, intensity_function, num_integration_points=1000):
    # TODO implement Monte Carlo Quadrature, or other methods to get an unbiased estimate.
    evaluation_points = jnp.linspace(0, T, num_integration_points)

    # This is jitable and vmappable. 
    # TODO If jit is disabled a loop version that explicitly indexes ts can be more performant.
    intensity_values = jax.vmap(lambda t: intensity_function(params, t, ts))(evaluation_points)

    integral = _trapz(intensity_values, evaluation_points)  # Use our JAX-compatible trapz
    return integral

def sample(key, params:Params, Tstart:float, Tend:float, events=None):
    """Generate a sample from the Hawkes process using Ogata's thinning algorithm.
    
    Args:
        key: JAX random key for sampling
        Tstart: Start time
        Tend: End time
        events: Initial events (optional)
    """
    if events is None:
        events = []

    intensities = []
    t = Tstart
    last_t = t
    
    while t < Tend:
        jnp_events = jnp.array(events)
        key, subkey1, subkey2 = random.split(key, 3)
        lambda_t = intensity(params,t, jnp_events)
        intensities.append(lambda_t)
        
        # Generate uniform random numbers using JAX
        u = random.uniform(subkey1)
        w = -jnp.log(u) / lambda_t
        t += w
        
        D = random.uniform(subkey2)
        if D <= intensity(params, t, jnp.array(jnp_events)) / lambda_t and t < Tend:
            events.append(t)
        
        last_t = t

    return jnp.array(events), jnp.array(intensities)

    
@jax.jit
def _trapz(y, x):
    """JAX compatible trapezoidal integration."""
    return jnp.sum((x[1:] - x[:-1]) * (y[1:] + y[:-1])) / 2.0



# Generate some demo data
# %%
global_rng = random.PRNGKey(0)
true_params = Params(mu= jnp.array(0.2), log_alpha= jnp.log(0.5), log_beta= jnp.log(0.4))
fake_params = Params(mu= jnp.array(0.8), log_alpha= jnp.log(0.0), log_beta= jnp.log(0.05))

#%%
key, *sk = random.split(random.PRNGKey(42), 100)

start, stop, num_eval = 0.0, 100.0, 100

events, _ = sample(sk[0], true_params, start, stop, None)
print(f"Generated {len(events)} events.")
fake_events, _ = sample(sk[1], fake_params, start, stop, None)
print(f"Generated {len(fake_events)} events.")

eval_points = jnp.linspace(0, stop, num_eval)
intensity_values = jnp.array([intensity(true_params, t, events) for t in eval_points])
print("Computed intensity values.")

fig, axes= plt.subplots(2,1)
axes[0].scatter(events, np.zeros_like(events), c='red', label='Events')
axes[0].scatter(fake_events, -0.1*np.ones_like(fake_events), c="black", label="Fake Events")

axes[0].plot(eval_points, intensity_values, label="Intensity")
axes[0].legend()
print("Plotted intensity and events.")

num_integration_points = 0
timeline_evalp = jnp.linspace(start, stop, num_eval)
timeline_ll = [likelihood(log= True, params=true_params, ts=events[events<=t], T=t, num_integration_points=num_integration_points)/t for t in timeline_evalp]
timeline_ll_fake = [likelihood(log= True, params=true_params, ts=fake_events[fake_events<=t], T=t, num_integration_points=num_integration_points)/t for t in timeline_evalp]


axes[1].plot(jnp.linspace(start, stop, num_eval), timeline_ll, c="red", label="Log-Likelihood")
axes[1].plot(jnp.linspace(start, stop, 100), timeline_ll_fake, c="black", label="False Data Log-Likelihood")

axes[1].set_title("Normalized Log-Likelihood of observed events")
axes[1].legend()
# Set y axes to log
#axes[1].set_yscale('log')



# %%
# Generate multiple trajectories from demo_params

def generate_dataset(params, N, T, key):   
    all_events = []
    for n in range(N):
        key, sk = random.split(key)
        events, _ = sample(sk, params, 0, T, None)
        all_events.append(events)
    return all_events

global_rng, sk = random.split(global_rng)
all_events = generate_dataset(true_params, N=20, T=100.0, key=sk)
# Plot the dataset
for i,events in enumerate(all_events):
    plt.plot(events, jnp.ones_like(events)*i, 'ro', alpha=0.3)
plt.show()

# %%

num_steps = 1000
step_size = jnp.array(0.0001)

key, sk = random.split(random.PRNGKey(12))
test_params = Params(mu= jnp.array(0.1), log_alpha= jnp.log(0.3), log_beta= jnp.log(0.2))

def fit_model_basic(params, events_batch, T):
    likelihoods = []

    def compute_batch_ll(params, events_batch, T):
        # Does not work without padding the sequences to the same length
        # return jnp.sum(jax.vmap(lambda event_timeline: log_likelihood(params, event_timeline, T))(events_batch))
    
        ind_ll = [likelihood(log=True, params=params, ts=event_timeline, T=T) for event_timeline in events_batch]
        return jnp.sum(jnp.array(ind_ll))

    grad_ll_function = jax.value_and_grad(compute_batch_ll, 0)
    
    for step in tqdm(range(num_steps)):

        ll, grads = grad_ll_function(params, events_batch, T)
        likelihoods.append(ll)
        print(ll)
        # Step gradient ascent
        params = Params(
            mu= params.mu + step_size * grads.mu,
            log_alpha= params.log_alpha + step_size * grads.log_alpha,
            log_beta= params.log_beta + step_size * grads.log_beta,
        )

    return params, likelihoods


baseline_ll =jnp.sum(np.array([likelihood(log=True, params=true_params, ts=event_timeline, T=100.0) for event_timeline in all_events]))

# %%
fitted_params, likelihoods = fit_model_basic(test_params, all_events, T=100.0)
# %%
