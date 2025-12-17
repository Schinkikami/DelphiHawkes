# %%
from pathlib import Path
import torch
import numpy as np
from .event_utils import MVEventData


# %%
# Load some real data
data_path = Path("../data/ukb_simulated_data/expansion.bin")
assert data_path.exists(), "Data file does not exist."
np_data = np.memmap(str(data_path), mode="r", dtype=np.uint32).reshape(-1, 3)

batch_ids = np_data[:, 0].astype(int)
time_points = np_data[:, 1].astype(float)
event_types = np_data[:, 2].astype(int)
# %%

male_token = 2
female_token = 3
death_token = 1269

# Remap tokens, so that we start at 0. (0 and 1 are reserved for Padding and NoEvent).
event_types -= 2
male_token -= 2
female_token -= 2
death_token -= 2


unique_org_event_ids = np.unique(event_types)
num_event_types = death_token + 1  # We start at 0, and death is the last event type.

time_points = time_points / 365.0  # Convert to years
# %%
# Split data by batch id
all_events_real = []

batch_ids_np = np.asarray(batch_ids)
change_idcs = np.flatnonzero(batch_ids_np[1:] != batch_ids_np[:-1]) + 1
change_idcs = np.concatenate([np.array([0]), change_idcs, np.array([len(batch_ids_np)])])

for start, stop in zip(change_idcs[:-1], change_idcs[1:]):
    sequence_time = time_points[start:stop]
    sequence_event = event_types[start:stop]

    ev_data = MVEventData(
        time_points=torch.tensor(sequence_time, dtype=torch.float32),
        event_types=torch.tensor(sequence_event, dtype=torch.long),
        sort=True,
    )
    all_events_real.append(ev_data)

# %%
# Sequences always start with male/female. Filter illegal sequences.
first_tokens = np.array([ts.event_types[0] for ts in all_events_real])

legal_sequence_icds = np.flatnonzero(np.logical_or(first_tokens == male_token, first_tokens == female_token))
all_events_real = [all_events_real[i] for i in legal_sequence_icds]

# %%
num_illegal_sequences = 0

occurences = torch.zeros(num_event_types)
counts = torch.zeros(num_event_types)

for sequence in all_events_real:
    count = np.bincount(sequence.event_types, minlength=num_event_types)
    occured = counts > 0

    if max(count) > 1:
        num_illegal_sequences += 1

    occurences += occured
    counts += count

# --> Results in 0. No Pruning needed it seems.
# %%
