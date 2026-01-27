# %%
import sys

sys.path.append("..")
# %%

import torch
from hawkes.event_utils import BatchedMVEventData, MVEventData
from hawkes.hawkes_tpp import ExpKernelHawkesProcess, SplineBaselineExpKernelHawkesProcess
from hawkes.baseline_tpps import SplinePoissonProcess

# %%
seq = MVEventData(torch.tensor([0.2, 0.8, 2.0, 2.3], dtype=float), torch.tensor([0, 0, 1, 0], dtype=int))
batch = BatchedMVEventData([seq.time_points], [seq.event_types])
# %%
model = ExpKernelHawkesProcess(D=2)
model = SplineBaselineExpKernelHawkesProcess(D=2, num_knots=5, delta_t=2.4 / 5)
model = SplinePoissonProcess(D=2, num_knots=5, delta_t=2.4 / 5)
# %%
model.likelihood(batch, torch.tensor(4.0).unsqueeze(0))
# %%
model.intensity(torch.tensor(4.0).unsqueeze(0), batch)

# %%
# Test integrate numerically
model.cumulative_intensity(torch.tensor(5.0).unsqueeze(dim=0), batch)

# %%
# Test CDF functionality
model.CDF(torch.tensor(2.300001).unsqueeze(0), batch)
# %%
model.PDF(torch.tensor(2.30001).unsqueeze(0), batch)
# %%
model.sample(seq, num_steps=10)

# %%
model.inverse_CDF(torch.tensor([0.0]), batch)
# %%
model.inverse_CDF(torch.tensor([0.5]), batch)
# %%
model.CDF(torch.tensor([model.inverse_CDF(torch.tensor([0.5]), batch)[0]]), batch)
# %%

# Check correctness of positive likelihood

pos_likelihood1 = model.positive_likelihood(batch, log=True)
pos_likelihood = super(SplinePoissonProcess, model).positive_likelihood(batch, log=True)
# Stack the sequence on itself with increasing lengths
stacked_batch = BatchedMVEventData(
    time_points=[
        seq.time_points[:i]
        for i in range(
            len(seq.time_points),
        )
    ],
    event_types=[
        seq.event_types[:i]
        for i in range(
            len(seq.event_types),
        )
    ],
)

valid_events = stacked_batch.event_types != -1
stacked_intensities = model.intensity(seq.time_points, stacked_batch)
intensity_at_points = stacked_intensities[torch.arange(len(seq.time_points)), seq.event_types]
log_intensity_at_points = intensity_at_points.log()

print(pos_likelihood, log_intensity_at_points.sum())
# %%
# Check correctness of PDF and CDF via numerical integration
integrants = [model.PDF(torch.tensor([t]), batch) for t in torch.linspace(2.3, 2.5, steps=100)]
integrants = torch.stack(integrants).squeeze()
integral = torch.trapz(integrants, torch.linspace(2.3, 2.5, steps=100), dim=0).sum()
print("Integral over PDF from 2.3 to 2.5:", integral)
cdf = model.CDF(torch.tensor([2.5]), batch) - model.CDF(torch.tensor([2.3]), batch)
print("CDF difference from 2.3 to 2.5:", cdf)

# %%
