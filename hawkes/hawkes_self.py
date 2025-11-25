# My own implementation of the univariate Hawkes process
#%%
import jax.numpy as np
from jax import random
from functools import partial
from tqdm import tqdm

class UVHawkes:
    def __init__(self, key, reg_lambda=None):
        """Initialize the Hawkes process.
        
        Args:
            key: JAX random key for initialization
            reg_lambda: regularization parameter
        """
        self.reg_lambda = reg_lambda
        self.mu = np.empty(0, dtype=float)
        self.log_alpha = np.empty(0, dtype=float)
        self.log_beta = np.empty(0, dtype=float)
        self._init_params(key)

    def _init_params(self, key):
        """Initialize parameters using JAX random key."""
        key1, key2 = random.split(key)
        self.mu = np.array(0.0, dtype=float)
        self.log_alpha = random.uniform(key1, shape=())
        self.log_beta = random.uniform(key2, shape=())

    def _transform_params(self):
        # Constrain parameters alpha and beta to be positive
        return self.mu, np.exp(self.log_alpha), np.exp(self.log_beta)

    def _kernel(self, x):
        # Returns pdf of exponential distribution with rate beta scaled by alpha
        _, alpha, beta = self._transform_params()
        return alpha * beta * np.exp(-beta * x)

    def intensity(self, t, ts):
        # Compute the intensity at time t given past events ts
        intensity = self.mu
        for ti in ts:
            if ti < t:
                intensity += self._kernel(t - ti)
        return intensity

    def sample(self, key, Tstart, Tend, events=None):
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
        #pbar = tqdm(total=Tend-Tstart, desc='Simulating', unit='time')
        last_t = t
        
        while t < Tend:
            key, subkey1, subkey2 = random.split(key, 3)
            lambda_t = self.intensity(t, events)
            intensities.append(lambda_t)
            
            # Generate uniform random numbers using JAX
            u = random.uniform(subkey1)
            w = -np.log(u) / lambda_t
            t += w
            
            D = random.uniform(subkey2)
            if D <= self.intensity(t, events) / lambda_t and t < Tend:
                events.append(t)
            
            #pbar.update(t - last_t)
            last_t = t

        #pbar.close()
        return np.array(events), np.array(intensities)

    def likelihood(self, ts, T=None, num_integration_points=1000):
        if T is None:
            T = ts[-1]

        positive_likelihood = np.ones(1, dtype=float)
        for i, ti in enumerate(ts):
            positive_likelihood *= self.intensity(ti, ts[:i])
        
        evaluation_points = np.linspace(0, T, num_integration_points)
        intensity_values = np.array([self.intensity(t, ts) for t in evaluation_points])
        
        num_integral = trapz(intensity_values, evaluation_points)  # Use our JAX-compatible trapz
        negative_likelihood = np.exp(-num_integral)

        return positive_likelihood - negative_likelihood
    

def trapz(y, x):
    """JAX compatible trapezoidal integration."""
    return np.sum((x[1:] - x[:-1]) * (y[1:] + y[:-1])) / 2.0

#%%

#Stateless Hawkes model for Jax



#%%
key = random.PRNGKey(0)
events = np.array([0.3, 1.2, 1.4, 1.6, 2.8, 4.0, 4.2])
key, sk = random.split(key)
model = UVHawkes(key=sk)


#%%
model.likelihood(events, T=np.max(events)+1)
# %%
model.intensity(3.0, [1.0])

# Generate some demo data
# %%
key, sk = random.split(random.PRNGKey(42))
demo_model = UVHawkes(key=sk)
demo_model.mu = 0.2
demo_model.log_alpha = np.log(0.5)
demo_model.log_beta = np.log(0.4)
#%%
start, stop = 0.0, 100.0
events, intensities = demo_model.sample(random.split(sk)[1], start, stop, None)

eval_points = np.linspace(0, 100, 1000)
intensity_values = np.array([demo_model.intensity(t, events) for t in eval_points])
from matplotlib import pyplot as plt
plt.plot(events, np.zeros_like(events), 'ro', label='Events')
plt.plot(eval_points, intensity_values, label="Intensity")

# %%
