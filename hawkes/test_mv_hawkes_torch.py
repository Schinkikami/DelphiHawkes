#%%
# Generate some demo data for Multivariate Hawkes Process
from typing import Optional
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from hawkes_torch import MultiVariateHawkesProcess, Params, MVEventData
#%%

D = 2  # Number of event types (dimensions)

# Choose some "ground truth" multivariate parameters
true_params = Params(
    mu=torch.tensor([0.1, 0.5]),           # Baseline for each type
    alpha=torch.tensor([[0.5, 0.2],         # Impact from type j to type i
                        [0.1, 0.4]]),
    beta=torch.tensor([[0.5, 2.0],
                       [0.2, 0.8]])
)
fake_params = Params(
    mu=torch.tensor([0.5, 0.1]),           # Baseline for each type
    alpha=torch.tensor([[0.2, 0.5],         # Impact from type j to type i
                        [0.4, 0.1]]),
    beta=torch.tensor([[0.2, 0.5],
                       [0.8, 0.2]])
)

true_model = MultiVariateHawkesProcess(true_params)
fake_model = MultiVariateHawkesProcess(fake_params)

true_model.ensure_stability()
fake_model.ensure_stability()
#%%

start, stop, num_eval = 0.0, 100.0, 100

with torch.no_grad():
    events = true_model.sample(start, stop)
    
    print(f"Generated {len(events)} events (multivariate).")
    fake_events = fake_model.sample(start, stop)
    
    print(f"Generated {len(fake_events)} events (fake, multivariate).")

    eval_points = torch.linspace(0, stop, num_eval)
    intensity_values_true = torch.stack([true_model.intensity(t, events) for t in eval_points]) # Shape (num_eval, D)

    # Plotting
    fig, axes = plt.subplots(D+2, 1, figsize=(10, 4*(D+2)))

    colors = ['red', 'blue', 'green', 'orange']
    for d in range(D):
        # Events of each type
        mask = (events.event_types == d)
        mask_fake = (fake_events.event_types == d)
        axes[0].scatter(events.time_points[mask], np.ones_like(events.time_points[mask]) * (d+0.1), c="red", label=f'True Events type {d}', marker="o" if d==0 else "x", alpha=0.6)
        axes[0].scatter(fake_events.time_points[mask_fake], np.ones_like(fake_events.time_points[mask_fake]) * (d-0.1), c="blue", marker="o" if d==0 else "x", label=f'Fake Events type {d}', alpha=0.6)
        # Intensities
        axes[d+1].plot(eval_points, intensity_values_true[:, d], label=f'Intensity type {d}')
        axes[d+1].set_ylabel(f'Intensity type {d}')
        axes[d+1].legend()

    axes[0].set_title("Events (per type)")
    axes[0].legend()
    axes[0].set_yticks([i for i in range(D)])
    # Compute normalized log-likelihood over time
    timeline_evalp = torch.linspace(start+1.0, stop, num_eval)  # Avoid t=0 for normalization
    timeline_ll_true = [true_model.likelihood(
        ts=MVEventData(
            time_points=events.time_points[events.time_points < t],
            event_types=events.event_types[events.time_points < t]
        ),
        T=float(t), log=True, num_integration_points=0)/t for t in timeline_evalp]
    timeline_ll_fake_data = [true_model.likelihood(
        ts=MVEventData(
            time_points=fake_events.time_points[fake_events.time_points < t],
            event_types=fake_events.event_types[fake_events.time_points < t]
        ), 
        T=float(t), log=True, num_integration_points=0)/t for t in timeline_evalp]

    axes[-1].plot(timeline_evalp, timeline_ll_true, c='red', label="True Model- True Data LL")
    axes[-1].plot(timeline_evalp, timeline_ll_fake_data, c='black', label="True Model - Fake Data LL")

    axes[-1].set_title("Normalized Log-Likelihood (multivariate)")
    axes[-1].legend()
    #plt.show()

#%%

def generate_dataset_mv(model, N, T):
    all_events = []
    for _ in tqdm(range(N)):
        events = model.sample(0, T)
        all_events.append(events)
    return all_events

print("Generating dataset...")
N = 10000
T = 50.0
all_events = generate_dataset_mv(true_model, N, T)

N_test = 1000
all_events_test = generate_dataset_mv(true_model, N_test, T)
print("Generated dataset.")
#%% Plot the dataset (multitype raster plot)
# Only a subset...
subset_datashowcase = 50

fig, ax = plt.subplots(1, 1, figsize=(10, 6))
offset = 0.2
ax.hlines(np.arange(subset_datashowcase), xmin=0, xmax=T, color='lightgray')
for i,events in enumerate(all_events[:subset_datashowcase]):
    for d in range(D):
        mask = (events.event_types == d)
        ax.plot(events.time_points[mask], np.ones_like(events.time_points[mask])*i, 
                'o', color=colors[d], alpha=0.4)
plt.title(f"Raster plot of {subset_datashowcase} generated multivariate event sequences")
plt.xlabel("Time")
plt.ylabel("Sequence index")
#plt.show()

#%%
# TODO step on sum of log-likelihoods or on mean??
def batch_ll_mv(model, events_batch, T):
    ind_ll = [model.likelihood(ts=e, T=T, num_integration_points=0) for e in events_batch]
    return torch.sum(torch.stack(ind_ll))

def fit_model_basic_mv(model:MultiVariateHawkesProcess, events_batch:list[MVEventData], T:float, test_events:Optional[list[MVEventData]]=None, eval_freq:float=10, num_steps=100, step_size=0.01, batch_size:Optional[float]=None):
    optim = torch.optim.AdamW(model.parameters(), lr=step_size, weight_decay=0.0)
    likelihoods = []
    test_likelihoods = []
    for step in tqdm(range(num_steps)):
        if batch_size is not None:
            batch_indices = np.random.choice(len(events_batch), size=batch_size, replace=False)
            batch = [events_batch[i] for i in batch_indices]
        else:
            batch = events_batch

        optim.zero_grad()
        ll = batch_ll_mv(model, batch, T)
        nll = -ll
        nll.backward()
        likelihoods.append(ll.item())
        
        optim.step()
        with torch.no_grad():
            model.ensure_stability()
        if test_events is not None and (step % eval_freq == 0 or step == num_steps-1):
            with torch.no_grad():
                test_ll = batch_ll_mv(model, test_events, T)
                print(f" Step {step}, Test LL (normed): {test_ll.item()/len(test_events)}")
                test_likelihoods.append(test_ll.item())
    return model, likelihoods, test_likelihoods

#%% Baseline log-likelihoods

test_params = Params(
    mu=torch.tensor([0.4, 0.2]),
    alpha=torch.tensor([[0.2, 0.1],
                        [0.1, 0.2]]),
    beta=torch.tensor([[0.3, 0.2],
                       [0.2, 0.3]])
)
init_model = MultiVariateHawkesProcess(test_params)
init_model.ensure_stability()
fit_model = MultiVariateHawkesProcess(test_params)
fit_model.ensure_stability()

true_ll_train = torch.mean(torch.stack([true_model.likelihood(ts=ev, T=T, num_integration_points=0) for ev in all_events]))
init_ll_train = torch.mean(torch.stack([init_model.likelihood(ts=ev, T=T, num_integration_points=0) for ev in all_events]))

print(f"Baseline log-likelihood of true model on dataset: {true_ll_train.item()}")
print(f"Baseline log-likelihood of init model on dataset: {init_ll_train.item()}")

true_ll_test = torch.mean(torch.stack([true_model.likelihood(ts=ev, T=T, num_integration_points=0) for ev in all_events_test]))
init_ll_test = torch.mean(torch.stack([init_model.likelihood(ts=ev, T=T, num_integration_points=0) for ev in all_events_test]))

print(f"Baseline log-likelihood of true model on dataset: {true_ll_test.item()}")
print(f"Baseline log-likelihood of init model on dataset: {init_ll_test.item()}")


#%%
# Fit the model to the data.
num_steps = 1000
step_size = 1e-3
batch_size = 100
fit_model, likelihoods, test_likelihoods = fit_model_basic_mv(model=fit_model.cuda(), events_batch=[e.cuda() for e in all_events], T=torch.tensor(T).cuda(), test_events=[e.cuda() for e in all_events_test], num_steps=num_steps, step_size=step_size, batch_size=batch_size)
#%%

model_ll_train = torch.mean(torch.stack([fit_model.likelihood(ts=ev.cuda(), T=T, num_integration_points=0) for ev in all_events])).cpu()
model_ll_test = torch.mean(torch.stack([fit_model.likelihood(ts=ev.cuda(), T=T, num_integration_points=0) for ev in all_events_test])).cpu()

#%%

print(f"Baseline log-likelihood of true model on train dataset: {true_ll_train.item()}")
print(f"Baseline log-likelihood of init model on train dataset: {init_ll_train.item()}")
print(f"Baseline log-likelihood of fitted model on train dataset: {model_ll_train.item()}")

print(f"Baseline log-likelihood of true model on test dataset: {true_ll_test.item()}")
print(f"Baseline log-likelihood of init model on test dataset: {init_ll_test.item()}")
print(f"Baseline log-likelihood of fitted model on test dataset: {model_ll_test.item()}")

#%%

fig, axes = plt.subplots(2,1)


axes[0].plot([l/batch_size for l in likelihoods])
axes[0].hlines(true_ll_train.item(), xmin=0, xmax=num_steps, colors='red', linestyles='dashed', label='True Model LL')
axes[0].hlines(init_ll_train.item(), xmin=0, xmax=num_steps, colors='blue', linestyles='dashed', label='Start Model LL')
axes[0].hlines(model_ll_train.item(), xmin=0, xmax=num_steps, colors='green', linestyles='dashed', label='Final Model LL')

axes[0].set_xlabel("Step")
axes[0].set_ylabel("Total Train Log-Likelihood")
axes[0].legend()
axes[1].plot(np.linspace(0, num_steps, len(test_likelihoods)), [l/len(all_events_test) for l in test_likelihoods], marker="x", markersize=2)
axes[1].hlines(true_ll_test.item(), xmin=0, xmax=num_steps, colors='red', linestyles='dashed', label='True Model LL Test')
axes[1].hlines(init_ll_test.item(), xmin=0, xmax=num_steps, colors='blue', linestyles='dashed', label='Start Model LL Test')
axes[1].hlines(model_ll_test.item(), xmin=0, xmax=num_steps, colors='green', linestyles='dashed', label='Final Model LL Test')

axes[1].set_xlabel("Step")
axes[1].set_ylabel("Total Test Log-Likelihood")
axes[1].legend()

plt.show()

#%% Sample fitted model
sampled = fit_model.cpu().sample(0, T)
print("Sampled events from fitted model:", sampled)

#%%
with torch.no_grad():

    eval_points = torch.linspace(0, stop, num_eval)
    intensity_values_true = torch.stack([true_model.intensity(t, events) for t in eval_points]) # Shape (num_eval, D)
    intensity_values_model = torch.stack([fit_model.cpu().intensity(t, events) for t in eval_points]) # Shape (num_eval, D)
    intensity_values_init = torch.stack([init_model.cpu().intensity(t, events) for t in eval_points]) # Shape (num_eval, D)

    # Plotting
    fig, axes = plt.subplots(D+2, 1, figsize=(10, 4*(D+1)))

    colors = ['red', 'blue', 'green', 'orange']
    for d in range(D):
        # Events of each type
        mask = (events.event_types == d)
        axes[0].scatter(events.time_points[mask], np.zeros_like(events.time_points[mask]) +d *0.1, c=colors[d], label=f'True Events type {d}', alpha=0.6)
        # Intensities
        axes[d+1].plot(eval_points, intensity_values_true[:, d], c=colors[d], label=f'Intensity type {d} (True)')
        axes[d+1].plot(eval_points, intensity_values_model[:, d], c=colors[d], label=f'Intensity type {d} (Model)', linestyle='dashed')
        axes[d+1].plot(eval_points, intensity_values_init[:, d], c=colors[d], label=f'Intensity type {d} (Init values)', linestyle='dotted')

        axes[d+1].set_ylabel(f'Intensity type {d}')
        axes[d+1].legend()

    axes[0].set_title("Events (per type)")
    axes[0].legend()
    axes[0].get_yaxis().set_visible(False)

    # Compute normalized log-likelihood over time
    timeline_evalp = torch.linspace(start+1.0, stop, num_eval)  # Avoid t=0 for normalization
    timeline_ll_true = [true_model.likelihood(
        ts=MVEventData(
            time_points=events.time_points[events.time_points <= t],
            event_types=events.event_types[events.time_points <= t]
        ),
        T=float(t), log=True, num_integration_points=0)/t for t in timeline_evalp]
    timeline_ll_model = [fit_model.cpu().likelihood(
        ts=MVEventData(
            time_points=events.time_points[events.time_points <= t],
            event_types=events.event_types[events.time_points <= t]
        ), 
        T=float(t), log=True, num_integration_points=0)/t for t in timeline_evalp]
    timeline_ll_fake_data = [init_model.cpu().likelihood(
        ts=MVEventData(
            time_points=events.time_points[events.time_points <= t],
            event_types=events.event_types[events.time_points <= t]
        ),
        T=float(t), log=True, num_integration_points=0)/t for t in timeline_evalp]
    axes[-1].plot(timeline_evalp, timeline_ll_true, c='red', label="True Model LL")
    axes[-1].plot(timeline_evalp, timeline_ll_model, c='black', label="Fitted Model LL", linestyle='dashed')
    axes[-1].plot(timeline_evalp, timeline_ll_fake_data, c='black', label="Init Model LL", linestyle='dotted')

    axes[-1].set_title("Normalized Log-Likelihood (multivariate)")
    axes[-1].legend()
    plt.show()


# %%
add_test_dataset = generate_dataset_mv(true_model, 200, 100)
#%%
aa = torch.mean(torch.stack([fit_model.cuda().likelihood(ts=ev.cuda(), T=100, num_integration_points=0) for ev in add_test_dataset])).cpu()
bb = torch.mean(torch.stack([true_model.cuda().likelihood(ts=ev.cuda(), T=100, num_integration_points=0) for ev in add_test_dataset])).cpu()
cc = torch.mean(torch.stack([init_model.cuda().likelihood(ts=ev.cuda(), T=100, num_integration_points=0) for ev in add_test_dataset])).cpu()
print(aa, bb, cc)
# %%
bbb = [true_model.cuda().likelihood(ts=ev.cuda(), T=100, num_integration_points=0) for ev in add_test_dataset]

# %%
