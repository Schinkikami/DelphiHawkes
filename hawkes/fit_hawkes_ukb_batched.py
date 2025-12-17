# %%
from pathlib import Path
from typing import Optional
import torch
from tqdm import tqdm
from .Hawkes import ExpKernelMVHawkesProcess
from torch.utils.data import DataLoader
from .event_utils import MVEventData, BatchedMVEventData
from .ukb_loading import load_ukb_sequences
# %%


def measure_likelihood(model, dataloader, device):
    model = model.to(device)

    lls = []

    for batch in dataloader:
        T = batch.max_time
        batch = batch.to(device)
        T = T.to(device)

        with torch.no_grad():
            ll = model.likelihood(batch, T)
            lls.append(ll.detach().cpu())

    lls = torch.cat(lls, dim=0)
    model = model.cpu()
    return lls


def batched_train_loop(
    model: ExpKernelMVHawkesProcess,
    events_batch: DataLoader,
    test_events: Optional[DataLoader] = None,
    num_steps=100,
    step_size=0.01,
    eval_freq: float = 100,
    save_file_name: Optional[str] = None,
    device: str = "cpu",
):
    model = model.to(device=device)
    optim = torch.optim.AdamW(model.parameters(), lr=step_size, weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optim, step_size=200, gamma=0.1)

    likelihoods = []
    test_likelihoods = []
    last_test_ll = None
    last_ll = 0.0
    global_step = 0
    epoch = 0

    progress_bar = tqdm(total=num_steps)

    while True:
        print(f"Starting epoch {epoch}...")
        epoch += 1

        for step, batch in enumerate(events_batch):
            T = batch.max_time
            batch = batch.to(device)
            T = T.to(device)
            optim.zero_grad()
            ll = model.likelihood(batch, T)
            ll /= T
            ll = torch.mean(ll)
            nll = -ll
            nll.backward()
            last_ll = ll.item()
            likelihoods.append(ll.item())
            optim.step()
            scheduler.step()

            # with torch.no_grad():
            #    model.ensure_stability()

            if test_events is not None and (step % eval_freq == 0 or step == num_steps - 1):
                with torch.no_grad():
                    val_lls = []
                    for val_batch in test_events:
                        T = val_batch.max_time + (1 / 365)
                        val_batch = val_batch.to(device)
                        T = T.to(device)

                        batch_val_lls = model.likelihood(val_batch, T)
                        batch_val_lls /= T
                        batch_val_lls = torch.mean(batch_val_lls)
                        val_lls.append(batch_val_lls.item())
                    val_ll = torch.mean(torch.tensor(val_lls))

                    print(f" Step {step}, Test LL (normed): {val_ll.item()}")
                    test_likelihoods.append(val_ll.item())
                    # if save_file_name is not None:
                    #    torch.save(model.state_dict(), save_file_name)

            global_step += 1
            # Update the progress bar description and metrics
            progress_bar.set_postfix(epoch=epoch, LL_train=f"{last_ll}", refresh=True)
            progress_bar.update(1)
            if global_step >= num_steps:
                break
        if global_step >= num_steps:
            break

    progress_bar.close()

    return model, likelihoods, test_likelihoods


# %%

# Load UKB data
LIMIT_DATSET_SIZE = None
step_size = 1e-0
NUM_STEPS = 1000
DEVICE = torch.device("cuda")

BATCH_SIZE = 512

TEST_RATIO = 0.20
VALIDATION_RATIO = 0.10
# -> TRAIN_RATIO = 0.70


# Make file paths relative to repository root for robust execution
ROOT = Path(__file__).resolve().parent.parent
base_data_path = ROOT / "data" / "ukb_simulated_data"
sequences, sexes, num_event_types = load_ukb_sequences(base_data_path / "train.bin", limit_size=LIMIT_DATSET_SIZE)
validation_sequences, _, _ = load_ukb_sequences(base_data_path / "val.bin", limit_size=LIMIT_DATSET_SIZE)
# test_sequences, _, _ = load_ukb_sequences(base_data_path / "test.bin", limit_size=LIMIT_DATSET_SIZE)

# %%

# Take 20% of the data as test set and 10% as validatation set.

# indices = np.arange(len(sequences))

# train_val_indices, test_indices = train_test_split(indices, test_size=TEST_RATIO, random_state=42, shuffle=True)

# VAL_SPLIT_RATIO = VALIDATION_RATIO / (1.0 - TEST_RATIO)

# train_indices, validation_indices = train_test_split(
#    train_val_indices,
#    test_size=VAL_SPLIT_RATIO,
#    random_state=42,
#    shuffle=False,
# )

# train_sequences = [sequences[i] for i in train_indices]
# validation_sequences = [sequences[i] for i in validation_indices]
# test_sequences = [sequences[i] for i in test_indices]


# %%

from torch.utils.data import DataLoader


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
# dataset_test = ListDataset(test_sequences)

dataloader_train = DataLoader(dataset=dataset_train, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch)
dataloader_val = DataLoader(dataset=dataset_val, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_batch)
# dataloader_test = DataLoader(dataset=dataset_test, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_batch)


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

real_MVHP = real_MVHP
# %%

init_ll_train = torch.mean(measure_likelihood(model=real_MVHP, dataloader=dataloader_train, device=DEVICE))
init_ll_val = torch.mean(measure_likelihood(model=real_MVHP, dataloader=dataloader_val, device=DEVICE))
# init_ll_test = torch.mean(measure_likelihood(model=real_MVHP, dataloader=dataloader_test, device=DEVICE))

print(f"Baseline log-likelihood of init model on train dataset: {init_ll_train}")
print(f"Baseline log-likelihood of init model on val dataset: {init_ll_val}")
# print(f"Baseline log-likelihood of init model on test dataset: {init_ll_test}")

# %%

num_epochs_training = (NUM_STEPS * BATCH_SIZE) / len(sequences)
print(f"Will train {num_epochs_training} epochs!")

# %%
# Run a short training for evaluation purposes
# Run a short training for evaluation purposes
save_path = ROOT / "models" / "hawkes_trained.pth"
fit_model, train_lls, val_lls = batched_train_loop(
    real_MVHP,
    dataloader_train,
    dataloader_val,
    num_steps=NUM_STEPS,
    step_size=step_size,
    save_file_name=str(save_path),
    device=DEVICE,
)
# %%
# Implement loading and saving of mode
torch.save(fit_model.state_dict(), str(save_path))
