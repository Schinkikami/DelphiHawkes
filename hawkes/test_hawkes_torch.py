
#%%
# Generate some demo data
from hawkes_torch import UnivariateHawkesProcess, Params
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
#%%

true_params = Params(mu= torch.tensor(0.1), alpha= torch.tensor(0.9), beta= torch.tensor(1.0))
fake_params = Params(mu= torch.tensor(0.5), alpha= torch.tensor(0.0), beta= torch.tensor(0.00))

true_model = UnivariateHawkesProcess(true_params)
fake_model = UnivariateHawkesProcess(fake_params)
#%%

start, stop, num_eval = 0.0, 100.0, 100

with torch.no_grad():
    events, _ = true_model.sample(  start, stop, None)
    print(f"Generated {len(events)} events.")
    fake_events, _ = fake_model.sample(  start, stop, None)
    print(f"Generated {len(fake_events)} events.")

    eval_points = torch.linspace(0, stop, num_eval)
    intensity_values = torch.tensor([true_model.intensity( t, events) for t in eval_points])
    print("Computed intensity values.")

    fig, axes= plt.subplots(2,1)
    axes[0].scatter(events, np.zeros_like(events), c='red', label='Events')
    axes[0].scatter(fake_events, -0.1*np.ones_like(fake_events), c="black", label="Fake Events")

    axes[0].plot(eval_points, intensity_values, label="Intensity")
    axes[0].legend()
    print("Plotted intensity and events.")

    num_integration_points = 0
    timeline_evalp = torch.linspace(start, stop, num_eval)
    timeline_ll = [true_model.likelihood(ts=events[events<=t], T=float(t), log=True, num_integration_points=num_integration_points)/t for t in timeline_evalp]
    timeline_ll_fake = [true_model.likelihood(ts=fake_events[fake_events<=t], T=float(t), log=True, num_integration_points=num_integration_points)/t for t in timeline_evalp]


    axes[1].plot(torch.linspace(start, stop, num_eval), timeline_ll, c="red", label="Log-Likelihood")
    axes[1].plot(torch.linspace(start, stop, 100), timeline_ll_fake, c="black", label="False Data Log-Likelihood")

    axes[1].set_title("Normalized Log-Likelihood of observed events")
    axes[1].legend()
    # Set y axes to log
    #axes[1].set_yscale('log')


#%%


with torch.no_grad():
    # Evaluate the effect of numerical integration

    true_model.INTEGRATION_MODE = "trapezoidal"
    timeline_ll_trap = [true_model.likelihood(ts=events[events<t], T=float(t), log=True, num_integration_points=100, analy_kernel=False)/t for t in timeline_evalp]

    true_model.INTEGRATION_MODE = "monte_carlo"
    timeline_ll_mc = [true_model.likelihood(ts=events[events<t], T=float(t), log=True, num_integration_points=100, analy_kernel=False)/t for t in timeline_evalp]


    plt.plot(np.array(timeline_ll), label="Analytical")
    plt.plot(np.array(timeline_ll_trap), label="Trapezoidal")
    plt.plot(np.array(timeline_ll_mc), label="Monte Carlo")
    plt.legend()
    plt.show()

# %%
N = 100
T = 100.0

true_params = Params(mu= torch.tensor(0.1), alpha= torch.tensor(0.9), beta= torch.tensor(1.0))
true_model = UnivariateHawkesProcess(true_params)


#%%
def generate_dataset(model, N, T):   
    all_events = []
    for n in range(N):
        events, _ = model.sample(0, T, None)
        all_events.append(events)
    return all_events

all_events = generate_dataset(true_model, N=N, T=T)
#%%
# Plot the dataset
for i,events in enumerate(all_events):
    plt.plot(events, torch.ones_like(events, dtype=torch.int8)*i, 'ro', alpha=0.3)
plt.show()

#%%


def batch_ll(model:UnivariateHawkesProcess, events_batch, T):
    # Does not work without padding the sequences to the same length
    # return torch.sum(jax.vmap(lambda event_timeline: log_likelihood(params, event_timeline, T))(events_batch))

    ind_ll = [model.likelihood(ts=event_timeline, T=T, num_integration_points=0) for event_timeline in events_batch]
    return torch.sum(torch.stack(ind_ll))


def fit_model_basic(model, events_batch, T):
    likelihoods = []




    for step in tqdm(range(num_steps)):

        optim.zero_grad()
        ll= batch_ll(model, events_batch, T)
        
        nll = -ll

        nll.backward()

        likelihoods.append(ll.detach().item())        

    
        print("NLL:", nll.detach().item(), " Params:", model.mu.item(), torch.exp(model.log_alpha).item(), torch.exp(model.log_beta).item(),
              "Grads:", model.mu.grad.item(), model.log_alpha.grad.item(), model.log_beta.grad.item())
    
                
        optim.step()
        #with torch.no_grad():
        #    for param in model.parameters():
        #        param += step_size * param.grad
        #        param.grad.zero_()

    return model, likelihoods


#%%

test_params = Params(mu= torch.tensor(0.4), alpha= torch.tensor(0.5), beta= torch.tensor(0.3))
test_model = UnivariateHawkesProcess(test_params)

baseline_ll = torch.sum(torch.tensor([true_model.likelihood(ts=ts, T=T, num_integration_points=0) for ts in all_events]))
baseline_ll_wrong = torch.sum(torch.tensor([test_model.cpu().likelihood(ts=ts, T=T, num_integration_points=0) for ts in all_events]))

print(f"Baseline log-likelihood of true model on one trajectory: {baseline_ll.item()}")
print(f"Baseline log-likelihood of test model on one trajectory: {baseline_ll_wrong.item()}")

# %%

use_cuda = False



if use_cuda:
    test_model_in = test_model.cuda()
    event_batch = [ts.cuda() for ts in all_events]
else:
    test_model_in = test_model.cpu()
    event_batch = [ts.cpu() for ts in all_events]

num_steps = 1000
step_size = torch.tensor(0.01)

optim = torch.optim.Adam(test_model_in.parameters(), lr=step_size.item(), weight_decay=1e-3)


fitted_params, likelihoods = fit_model_basic(test_model_in, event_batch, T=T)
# %%

fitted_params.sample(0, T, None)
# %%
