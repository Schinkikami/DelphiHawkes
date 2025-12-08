# %%
from pathlib import Path
from typing import Optional
import torch
import numpy as np
from tqdm import tqdm
from hawkes import ExpKernelMVHawkesProcess
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from event_utils import MVEventData
# %%


def batched_train_loop(
    model: ExpKernelMVHawkesProcess,
    events_batch: DataLoader,
    T: float,
    test_events: Optional[DataLoader] = None,
    num_steps=100,
    step_size=0.01,
    eval_freq: float = 10,
    save_file_name: Optional[str] = None,
):
    optim = torch.optim.AdamW(model.parameters(), lr=step_size, weight_decay=0.0)
    likelihoods = []
    test_likelihoods = []

    last_ll = 0.0
    last_test_ll = 0.0
    global_step = 0
    epoch = 0

    T = torch.tensor(T).cuda()
    progress_bar = tqdm(total=num_steps)

    while True:
        print(f"Starting epoch {epoch}...")
        epoch += 1

        for step, batch in enumerate(events_batch):
            batch = batch.cuda()

            optim.zero_grad()
            ll = model.likelihood(batch, T)
            nll = -ll
            nll.backward()
            last_ll = ll.item()
            likelihoods.append(ll.item())
            optim.step()
            with torch.no_grad():
                model.ensure_stability()

            if test_events is not None and (step % eval_freq == 0 or step == num_steps - 1):
                with torch.no_grad():
                    test_lls = []
                    for test_batch in test_events:
                        test_batch = test_batch.cuda()
                        batch_ll = model.likelihood(test_batch, T)
                        test_lls.append(batch_ll.detach().cpu())
                        last_test_ll = batch_ll.item()
                    test_ll = torch.sum(torch.stack(test_lls))
                    print(f" Step {step}, Test LL (normed): {test_ll.item() / len(test_events)}")
                    test_likelihoods.append(test_ll.item())
                    if save_file_name is not None:
                        torch.save(model.state_dict(), save_file_name)

            global_step += 1
            # Update the progress bar description and metrics
            progress_bar.set_postfix(epoch=epoch, LL_train=f"{last_ll}", LL_test=f"{last_test_ll}", refresh=True)
            progress_bar.update(1)
            if global_step >= num_steps:
                break
        if global_step >= num_steps:
            break

    progress_bar.close()

    return model, likelihoods, test_likelihoods


# %%
# Load some real data
data_path = Path("../data/ukb_simulated_data/train.bin")
assert data_path.exists(), "Data file does not exist."
np_data = np.memmap(str(data_path), mode="r", dtype=np.uint32).reshape(-1, 3)

batch_ids = np_data[:, 0].astype(int)
time_points = np_data[:, 1].astype(float)
event_types = np_data[:, 2].astype(int)

unique_org_event_ids = np.unique(event_types)
num_event_types = len(unique_org_event_ids)
# Remap event types to contiguous ids
event_type_mapping = {orig_id: new_id for new_id, orig_id in enumerate(unique_org_event_ids)}
event_types = np.array([event_type_mapping[et] for et in event_types], dtype=int)

# Transform time to years
time_points = time_points / 365.0

# Split data by batch id
all_events_real = []
for b_id in np.unique(batch_ids):
    mask = batch_ids == b_id
    ev_data = MVEventData(
        time_points=torch.tensor(time_points[mask], dtype=torch.float32),
        event_types=torch.tensor(event_types[mask], dtype=torch.long),
        sort=True,
    )
    all_events_real.append(ev_data)

# Take 20% of the data as test set and 10% as validatation set.


# %%


indices = np.arange(len(all_events_real))

TEST_RATIO = 0.20
VALIDATION_RATIO = 0.10
# -> TRAIN_RATIO = 0.70

train_val_indices, test_indices = train_test_split(indices, test_size=TEST_RATIO, random_state=42, shuffle=True)

VAL_SPLIT_RATIO = VALIDATION_RATIO / (1.0 - TEST_RATIO)

train_indices, validation_indices = train_test_split(
    train_val_indices,
    test_size=VAL_SPLIT_RATIO,
    random_state=42,
    shuffle=False,
)

train_sequences = [all_events_real[i] for i in train_indices]
validation_sequences = [all_events_real[i] for i in validation_indices]
test_sequences = [all_events_real[i] for i in test_indices]
# %%

from torch.utils.data import DataLoader

BATCH_SIZE = 512


class ListDataset(torch.utils.data.Dataset):
    def __init__(self, data_list):
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]


def collate_batch(mv_l: list[MVEventData]):
    time_points = [t.time_points for ts in mv_l]
    event_types = [t.event_types for ts in mv_l]

    return BatchedMVEventData(time_points, event_types)


dataset_train = ListDataset(all_events_real)
dataset_val = ListDataset(validation_sequences)
dataset_test = ListDataset(test_sequences)

dataloader_train = DataLoader(dataset=dataset_train, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch)
dataloader_val = DataLoader(dataset=dataset_val, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_batch)
dataloader_test = DataLoader(dataset=dataset_test, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_batch)


# %%
D = int(event_types.max() + 1)  # Number of event types
T = max(time_points) + 1.0  # Maximum simulation time
N = len(all_events_real)  # Number of time-series

load_path = None
if load_path is not None and load_path.exists():
    print("Loading pre-trained model...")
    loaded_model = ExpKernelMVHawkesProcess(None, D)
    loaded_model.load_state_dict(torch.load(str(load_path)))
    real_MVHP = loaded_model
else:
    real_MVHP = ExpKernelMVHawkesProcess(None, D, seed=43)

cuda_real_MVHP = real_MVHP.cuda()
# %%

init_ll_train = torch.mean(torch.stack([real_MVHP.likelihood(ts=ev, T=T) for ev in train_sequences]))
init_ll_val = torch.mean(torch.stack([real_MVHP.likelihood(ts=ev, T=T) for ev in validation_sequences]))
init_ll_test = torch.mean(torch.stack([real_MVHP.likelihood(ts=ev, T=T) for ev in test_sequences]))

print(f"Baseline log-likelihood of init model on train dataset: {init_ll_train.item()}")
print(f"Baseline log-likelihood of init model on val dataset: {init_ll_val.item()}")
print(f"Baseline log-likelihood of init model on test dataset: {init_ll_test.item()}")

# %%
with torch.no_grad():
    ll = torch.sum(
        torch.stack(
            [
                cuda_real_MVHP.likelihood(
                    all_events_real[i].cuda(), T=torch.tensor(data=T).cuda(), num_integration_points=0
                )
                for i in range(1000)
            ]
        )
    )
    print(f"Log-likelihood: {ll.item()}")

# %%
step_size = 1e-4
num_steps = 100

fit_model, train_lls, val_lls = basic_train_loop(
    cuda_real_MVHP,
    dataloader_train,
    torch.tensor(T).cuda(),
    dataloader_val,
    num_steps=num_steps,
    step_size=step_size,
    save_file_name="models/latest.pth",
)
# %%
# Implement loading and saving of mode
torch.save(fit_model.state_dict(), "models/first_mv_hawkes_model.pth")
# %%
fit_model.cpu().sample(0, 80.0)
# %%
