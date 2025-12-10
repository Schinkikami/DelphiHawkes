# %%
from pathlib import Path
from typing import Optional
import torch
import numpy as np
from tqdm import tqdm
from Hawkes import ExpKernelMVHawkesProcess
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from event_utils import MVEventData, BatchedMVEventData
from ukb_loading import load_ukb_sequences
# %%


def batched_train_loop(
    model: ExpKernelMVHawkesProcess,
    events_batch: DataLoader,
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

    progress_bar = tqdm(total=num_steps)

    while True:
        print(f"Starting epoch {epoch}...")
        epoch += 1

        for step, batch in enumerate(events_batch):
            T = batch.max_time + (1 / 365)
            batch = batch.cuda()
            T = T.to(batch.device)
            optim.zero_grad()
            ll = model.likelihood(batch, T)
            ll = torch.mean(ll)
            nll = -ll
            nll.backward()
            last_ll = ll.item()
            likelihoods.append(ll.item())
            optim.step()
            # with torch.no_grad():
            #    model.ensure_stability()

            if test_events is not None and (step % eval_freq == 0 or step == num_steps - 1):
                with torch.no_grad():
                    val_lls = []
                    for val_batch in test_events:
                        T = val_batch.max_time + (1 / 365)
                        val_batch = val_batch.cuda()
                        T = T.to(val_batch.device)

                        batch_val_lls = model.likelihood(val_batch, T)
                        batch_val_lls = torch.sum(batch_val_lls)
                        val_lls.append(batch_val_lls.item())
                        last_test_ll = batch_val_lls.item()
                    val_ll = torch.sum(torch.tensor(val_lls))
                    print(f" Step {step}, Test LL (normed): {val_ll.item() / len(test_events)}")
                    test_likelihoods.append(val_ll.item())
                    # if save_file_name is not None:
                    #    torch.save(model.state_dict(), save_file_name)

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

# Load UKB data
data_path = Path("../data/ukb_simulated_data/expansion.bin")
sequences, sexes, num_event_types = load_ukb_sequences(data_path)

# %%
LIMIT_DATSET_SIZE = 100000

sequences = sequences[:LIMIT_DATSET_SIZE]
sexes = sexes[:LIMIT_DATSET_SIZE]

# Take 20% of the data as test set and 10% as validatation set.

indices = np.arange(len(sequences))

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

train_sequences = [sequences[i] for i in train_indices]
validation_sequences = [sequences[i] for i in validation_indices]
test_sequences = [sequences[i] for i in test_indices]
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
    time_points = [ts.time_points for ts in mv_l]
    event_types = [ts.event_types for ts in mv_l]

    return BatchedMVEventData(time_points, event_types)


dataset_train = ListDataset(sequences)
dataset_val = ListDataset(validation_sequences)
dataset_test = ListDataset(test_sequences)

dataloader_train = DataLoader(dataset=dataset_train, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch)
dataloader_val = DataLoader(dataset=dataset_val, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_batch)
dataloader_test = DataLoader(dataset=dataset_test, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_batch)


# %%
D = int(num_event_types)  # Number of event types
sequence_length = torch.tensor([ts.time_points[-1] for ts in sequences])
T = sequence_length + (1 / 365)
N = len(sequences)  # Number of time-series

load_path = None
if load_path is not None and load_path.exists():
    print("Loading pre-trained model...")
    loaded_model = ExpKernelMVHawkesProcess(None, D)
    loaded_model.load_state_dict(torch.load(str(load_path)))
    real_MVHP = loaded_model
else:
    real_MVHP = ExpKernelMVHawkesProcess(None, D, seed=43)

real_MVHP = real_MVHP.cuda()
# %%

with torch.no_grad():
    init_ll_train = torch.mean(
        input=torch.cat(
            [real_MVHP.likelihood(ts=ev.cuda(), T=ev.max_time.cuda() + (1 / 365)) for ev in dataloader_train]
        )
    ).item()
    init_ll_val = torch.mean(
        torch.cat([real_MVHP.likelihood(ts=ev.cuda(), T=ev.max_time.cuda() + (1 / 365)) for ev in dataloader_val])
    ).item()
    init_ll_test = torch.mean(
        torch.cat([real_MVHP.likelihood(ts=ev.cuda(), T=ev.max_time.cuda() + (1 / 365)) for ev in dataloader_test])
    ).item()

print(f"Baseline log-likelihood of init model on train dataset: {init_ll_train}")
print(f"Baseline log-likelihood of init model on val dataset: {init_ll_val}")
print(f"Baseline log-likelihood of init model on test dataset: {init_ll_test}")

# %%
step_size = 1e-4
num_steps = 100

fit_model, train_lls, val_lls = batched_train_loop(
    real_MVHP,
    dataloader_train,
    dataloader_val,
    num_steps=num_steps,
    step_size=step_size,
    save_file_name="models/latest.pth",
)
# %%
# Implement loading and saving of mode
torch.save(fit_model.state_dict(), "models/first_batched_mv_hawkes_model.pth")
# %%
fit_model.cpu().sample(0, 80.0)
# %%
