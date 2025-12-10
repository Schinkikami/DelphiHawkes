# %%
import torch
from Hawkes import ExpKernelMVHawkesProcess
from hawkes_torch import MultiVariateHawkesProcess, Params
from event_utils import *
import numpy as np
import time

# %%
# Correctness check.
D = 1250
max_len = 80
num_seq = 1000
max_T = 80
batch_size = 500
number = 1
device = "cuda:0"

sequences = generate_data(D, max_len, num_seq, max_T)
mv_events, batched_mv_events = prepare_data(sequences=sequences, batch_size=batch_size)

# TODO parameters dont align
hp = ExpKernelMVHawkesProcess(None, D=D)
hp.ensure_stability()

m, a, b = hp.transform_params()
hp_org = MultiVariateHawkesProcess(Params(m, a, b))
# %%

t = [torch.tensor([0.0, 0.2, 0.7, 0.8]), torch.tensor([0.6, 0.6, 0.9]), torch.tensor([])]
e = [
    torch.tensor([0, 1, 0, 0], dtype=torch.long),
    torch.tensor([1, 1, 1], dtype=torch.long),
    torch.tensor([], dtype=torch.long),
]

# mv_events = [MVEventData(tt, ee) for tt, ee in zip(t, e)]
# batched_mv_events = [BatchedMVEventData(t, e)]
# %%
# Check positive likelihood
with torch.no_grad():
    posll_batch = torch.cat([hp.positive_likelihood(batch) for batch in batched_mv_events])
    posll_org = torch.stack([hp_org._positive_likelihood_vectorized(ts) for ts in mv_events])
    print(posll_batch, posll_org, posll_batch.allclose(posll_org))
    int_batch = torch.cat([hp.integral_exp_kernel(T=torch.tensor([max_T]), ts=batch) for batch in batched_mv_events])
    int_org = torch.stack([hp_org.integral_exp_kernel(T=max_T, ts=ts) for ts in mv_events])
    print(int_batch, int_org, int_batch.to(torch.float32).allclose(int_org))
    ll_batch = torch.cat([hp.likelihood(batch, T=torch.tensor([max_T]), log=True) for batch in batched_mv_events])
    ll_org = torch.stack([hp_org.likelihood(ts, T=max_T, log=True) for ts in mv_events])
    print(ll_batch, ll_org, ll_batch.allclose(ll_org))
# %%
# Benchmark speed

D = 1250
max_len = 80
num_seq = 50000
max_T = 80
batch_size = 500
number = 1
device = "cuda:0"


# %%
batched_hp = ExpKernelMVHawkesProcess(None, D=D)
batched_hp.ensure_stability()
m, a, b = batched_hp.transform_params()
iter_hp = MultiVariateHawkesProcess(Params(m, a, b), D=D)

sequences = generate_data(D, max_len, num_seq, max_T)
mv_events, batched_mv_events = prepare_data(sequences, batch_size)

# %%


def org_likelihood(mv_events, hp, T, device):
    l = []
    for ts in mv_events:
        l.append(hp.likelihood(ts.to(device), T.to(device)))
    return l


def new_likelihood(batched_mv_events, hp, T, device):
    l = []
    for batch in batched_mv_events:
        l.append(hp.likelihood(batch.to(device), T.to(device)))
    return l


T_ = torch.tensor(max_T + 1)

org_speed = []
batch_speed = []

print("Start measurement")

iter_hp = iter_hp.to(device)
batched_hp = batched_hp.to(device)

for i in range(number):
    print(f"Measurement: {i}")

    print("Starting batched version")
    with torch.no_grad():
        start = time.time()
        _ = new_likelihood(batched_mv_events, batched_hp, T_, device=device)
        end = time.time()
        batch_speed.append(end - start)

    print("Starting unbatched version")
    with torch.no_grad():
        start = time.time()
        _ = org_likelihood(mv_events, iter_hp, T_, device=device)
        end = time.time()
        org_speed.append(end - start)

print(np.mean(org_speed), np.min(org_speed), np.max(org_speed))
print(np.mean(batch_speed), np.min(batch_speed), np.max(batch_speed))

# %%
