#%%
# Generate some demo data for Multivariate Hawkes Process
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from hawkes_torch import MultiVariateHawkesProcess, Params, MVEventData

#%%

D = 2  # Number of event types (dimensions)

# Choose some "ground truth" multivariate parameters
true_params = Params(
    mu=torch.tensor([0.1, 0.1]),           # Baseline for each type
    alpha=torch.tensor([[0.5, 0.2],         # Impact from type j to type i
                        [0.1, 0.4]]),
    beta=torch.tensor([[1.0, 1.0],
                       [1.0, 1.0]])
)
fake_params = Params(
    mu=torch.tensor([0.3, 0.3]),
    alpha=torch.tensor([[0.0, 0.0],
                        [0.0, 0.0]]),
    beta=torch.tensor([[1.0, 1.0],
                       [1.0, 1.0]])
)

true_model = MultiVariateHawkesProcess(true_params)
fake_model = MultiVariateHawkesProcess(fake_params)

#%%

start, stop, num_eval = 0.0, 100.0, 100

with torch.no_grad():
    events = true_model.sample(start, stop)
    
    print(f"Generated {len(events)} events (multivariate).")
    fake_events = fake_model.sample(start, stop)
    
    print(f"Generated {len(fake_events)} events (fake, multivariate).")

    eval_points = torch.linspace(0, stop, num_eval)
    intensity_values = torch.stack([true_model.intensity(t, events) for t in eval_points]) # Shape (num_eval, D)

    # Plotting
    fig, axes = plt.subplots(D+1, 1, figsize=(10, 4*(D+1)))

    colors = ['red', 'blue', 'green', 'orange']
    for d in range(D):
        # Events of each type
        mask = (events.event_types == d)
        mask_fake = (fake_events.event_types == d)
        axes[0].scatter(events.time_points[mask], np.ones_like(events.time_points[mask]) * (d+0.1), c=colors[d], label=f'True Events type {d}', alpha=0.6)
        axes[0].scatter(fake_events.time_points[mask_fake], np.ones_like(fake_events.time_points[mask_fake]) * (d-0.1), c=colors[d], marker='x', label=f'Fake Events type {d}', alpha=0.6)
        # Intensities
        axes[d+1].plot(eval_points, intensity_values[:, d], c=colors[d], label=f'Intensity type {d}')
        axes[d+1].set_ylabel(f'Intensity type {d}')
        axes[d+1].legend()

    axes[0].set_title("Events (per type)")
    axes[0].legend()
    axes[0].set_yticks([i for i in range(D)])
#%%
with torch.no_grad():
    # Compute normalized log-likelihood over time
    timeline_evalp = torch.linspace(start+1.0, stop, num_eval)  # Avoid t=0 for normalization
    timeline_ll = [true_model.likelihood(
        ts=MVEventData(
            time_points=events.time_points[events.time_points <= t],
            event_types=events.event_types[events.time_points <= t]
        ),
        T=float(t), log=True, num_integration_points=0)/t for t in timeline_evalp]
    timeline_ll_fake = [true_model.likelihood(
        ts=MVEventData(
            time_points=fake_events.time_points[fake_events.time_points <= t],
            event_types=fake_events.event_types[fake_events.time_points <= t]
        ), 
        T=float(t), log=True, num_integration_points=0)/t for t in timeline_evalp]

    fig2, ax2 = plt.subplots(1,1)
    ax2.plot(timeline_evalp, timeline_ll, c='red', label="True Data LL")
    ax2.plot(timeline_evalp, timeline_ll_fake, c='black', label="Fake Data LL")
    ax2.set_title("Normalized Log-Likelihood (multivariate)")
    ax2.legend()
    plt.show()

#%%

def generate_dataset_mv(model, N, T):
    all_events = []
    for _ in range(N):
        events = model.sample(0, T)
        all_events.append(events)
    return all_events

N = 1000
T = 50.0
all_events = generate_dataset_mv(true_model, N, T)

#%% Plot the dataset (multitype raster plot)
fig, ax = plt.subplots(1, 1, figsize=(10, 6))
offset = 0.2
ax.hlines(np.arange(N), xmin=0, xmax=T, color='lightgray')
for i,events in enumerate(all_events):
    for d in range(D):
        mask = (events.event_types == d)
        ax.plot(events.time_points[mask], np.ones_like(events.time_points[mask])*i, 
                'o', color=colors[d], alpha=0.4)
plt.title("Raster plot of generated multivariate events")
plt.show()

#%%

def batch_ll_mv(model, events_batch, T):
    ind_ll = [model.likelihood(ts=e, T=T, num_integration_points=0) for e in events_batch]
    return torch.sum(torch.stack(ind_ll))

def fit_model_basic_mv(model, events_batch, T, num_steps=100, step_size=0.01):
    optim = torch.optim.Adam(model.parameters(), lr=step_size, weight_decay=1e-3)
    likelihoods = []
    for step in tqdm(range(num_steps)):
        optim.zero_grad()
        ll = batch_ll_mv(model, events_batch, T)
        nll = -ll
        nll.backward()
        likelihoods.append(ll.item())
        with torch.no_grad():
            mu_v, alpha_v, beta_v = model.transform_params()
        print(f"Step {step}, NLL (normed): {nll.item()/len(events_batch)}, Params mu: {mu_v.detach().cpu().numpy()}, alpha: {alpha_v.detach().cpu().numpy()}, beta: {beta_v.detach().cpu().numpy()}")
        optim.step()
    return model, likelihoods

#%% Baseline log-likelihoods

test_params = Params(
    mu=torch.tensor([0.4, 0.2]),
    alpha=torch.tensor([[0.2, 0.1],
                        [0.1, 0.2]]),
    beta=torch.tensor([[0.3, 0.2],
                       [0.2, 0.3]])
)
test_model = MultiVariateHawkesProcess(test_params)

baseline_ll = torch.mean(torch.stack([true_model.likelihood(ts=ev, T=T, num_integration_points=0) for ev in all_events]))
baseline_ll_wrong = torch.mean(torch.stack([test_model.likelihood(ts=ev, T=T, num_integration_points=0) for ev in all_events]))

print(f"Baseline log-likelihood of true model on dataset: {baseline_ll.item()}")
print(f"Baseline log-likelihood of test model on dataset: {baseline_ll_wrong.item()}")

#%%

num_steps = 200
step_size = 0.01
fitted_params, likelihoods = fit_model_basic_mv(test_model.cuda(), [e.cuda() for e in all_events], torch.tensor(T).cuda(), num_steps=num_steps, step_size=step_size)
#%%
plt.plot(likelihoods)
plt.hlines(baseline_ll.item(), xmin=0, xmax=num_steps, colors='red', linestyles='dashed', label='True Model LL')
plt.xlabel("Step")
plt.ylabel("Total Log-Likelihood")
plt.title("Parameter Fitting Progress (Multivariate Hawkes)")
plt.show()

#%% Sample fitted model
sampled = fitted_params.sample(0, T)
print("Sampled events from fitted model:", sampled)


#%%
from pathlib import Path

# Load some real data
data_path = Path("../data/ukb_simulated_data/train.bin")
assert data_path.exists(), "Data file does not exist."
np_data = np.memmap(str(data_path), mode='r', dtype=np.uint32).reshape(-1,3)

batch_ids = np_data[:,0].astype(int)
time_points = np_data[:,1]
event_types = np_data[:,2].astype(int)

unique_org_event_ids = np.unique(event_types)
num_event_types = len(unique_org_event_ids)
#Remap event types to contiguous ids
event_type_mapping = {orig_id: new_id for new_id, orig_id in enumerate(unique_org_event_ids)}
event_types = np.array([event_type_mapping[et] for et in event_types], dtype=int)


# Split data by batch id
all_events_real = []
for b_id in np.unique(batch_ids):
    mask = (batch_ids == b_id)
    ev_data = MVEventData(
        time_points=torch.tensor(time_points[mask], dtype=torch.float32),
        event_types=torch.tensor(event_types[mask], dtype=torch.long),
        sort=True
    )
    all_events_real.append(ev_data)


# %%
D = int(event_types.max())
T = max(time_points)+1.0

real_MVHP = MultiVariateHawkesProcess(None, D)
# %%
baseline_ll = torch.mean(torch.stack([real_MVHP.likelihood(ts=ev, T=T, num_integration_points=0) for ev in all_events_real]))

# %%
