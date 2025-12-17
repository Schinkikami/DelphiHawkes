# %%
import sys

sys.path.append("..")
# %%

import torch
from hawkes.event_utils import BatchedMVEventData, MVEventData
from hawkes.Hawkes import ExpKernelMVHawkesProcess

# %%
seq = MVEventData(torch.tensor([0.2, 0.8, 2.0, 2.3], dtype=float), torch.tensor([0, 0, 1, 0], dtype=int))
batch = BatchedMVEventData([seq.time_points], [seq.event_types])
# %%
model = ExpKernelMVHawkesProcess(None, D=2)
# %%
model.likelihood(batch, torch.tensor(4.0).unsqueeze(0))
# %%
model.intensity(torch.tensor(4.0).unsqueeze(0), batch)
# %%
# Test integrate numerically
model._unb_integral_numerical(torch.tensor(5.0), seq, num_integration_points=3000)

# %%
# Test integrate numerically
model.cumulative_intensity(torch.tensor(5.0).unsqueeze(dim=0), batch)

# %%
# Test CDF functionality
model.CDF(torch.tensor(2.300001).unsqueeze(0), batch)
# %%
model.PDF(torch.tensor(2.30001).unsqueeze(0), batch)
# %%
model.sample_inverse(seq, num_steps=10)
