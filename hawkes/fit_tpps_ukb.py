# %%
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, asdict
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader

# Optional Weights & Biases support (safe to leave unconfigured)
try:
    import wandb
except ImportError:  # pragma: no cover - optional dependency
    wandb = None

from hawkes.event_utils import MVEventData, BatchedMVEventData
from hawkes.tpps import TemporalPointProcess
from hawkes.baseline_tpps import (
    PoissonProcess,
    ConditionalInhomogeniousPoissonProcess,
    SplinePoissonProcess,
)
from hawkes.hawkes_tpp import ExpKernelHawkesProcess, SplineBaselineExpKernelHawkesProcess
from hawkes.ukb_loading import load_ukb_sequences
# %%


@dataclass
class TrainingConfig:
    """Configuration for training temporal point processes."""

    # Data settings
    limit_dataset_size: int = 10000
    batch_size: int = 512

    # Training settings
    num_steps: int = 300
    learning_rate: float = 1e-0
    weight_decay: float = 0.0
    eval_freq: int = 100

    # Learning rate scheduler settings
    scheduler_type: str = "step"  # "step", "cosine", "none"
    scheduler_step_size: int = 200
    scheduler_gamma: float = 0.1

    # Model settings
    model_type: str = (
        "hawkes_modular"  # "poisson", "inhomogeneous_poisson", "spline_poisson", "hawkes", "hawkes_modular"
    )
    model_seed: int = 43

    # Spline-specific settings (for spline_poisson)
    spline_K: int = 5  # Number of knots
    spline_delta_t: float = 0.3  # Time spacing (default: 1.5/5)

    # Device settings
    device: str = "cuda:0"

    # Weights & Biases settings
    wandb_enable: bool = False  # Set True to enable W&B logging
    wandb_project: str = "your-project-name"  # e.g., "delphi-hawkes" (fill me)
    wandb_entity: Optional[str] = None  # e.g., "your-team" or keep None
    wandb_run_name: Optional[str] = None  # e.g., "experiment-001"
    wandb_tags: Optional[list] = None  # optional list of tags
    wandb_mode: str = "online"  # "online", "offline", "dryrun"

    def __post_init__(self):
        """Compute derived values after initialization."""
        # Only compute if explicitly set to a special sentinel (we use default value instead)
        pass

    def to_dict(self):
        """Convert config to dictionary for logging."""
        return asdict(self)


# %%


def measure_likelihood(model: TemporalPointProcess, dataloader: DataLoader, device):
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
    model: TemporalPointProcess,
    events_batch: DataLoader,
    config: TrainingConfig,
    test_events: Optional[DataLoader] = None,
    save_file_name: Optional[str] = None,
    wandb_run=None,
):
    device = config.device
    model = model.to(device=device)
    optim = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    # Configure learning rate scheduler
    if config.scheduler_type == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer=optim, step_size=config.scheduler_step_size, gamma=config.scheduler_gamma
        )
    elif config.scheduler_type == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optim, T_max=config.num_steps)
    elif config.scheduler_type == "none":
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=optim, lr_lambda=lambda epoch: 1.0)
    else:
        raise ValueError(f"Unknown scheduler type: {config.scheduler_type}")

    torch.autograd.set_detect_anomaly(True)

    likelihoods = []
    test_likelihoods = []
    last_ll = 0.0
    global_step = 0
    epoch = 0

    progress_bar = tqdm(total=config.num_steps)

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

            if wandb_run is not None and wandb is not None:
                wandb.log({"train/log_likelihood": last_ll, "train/step": global_step})

            if test_events is not None and (step % config.eval_freq == 0 or step == config.num_steps - 1):
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
                    if wandb_run is not None and wandb is not None:
                        wandb.log({"val/log_likelihood": val_ll.item(), "val/step": global_step})
                    # if save_file_name is not None:
                    #    torch.save(model.state_dict(), save_file_name)

            global_step += 1
            # Update the progress bar description and metrics
            progress_bar.set_postfix(epoch=epoch, LL_train=f"{last_ll}", refresh=True)
            progress_bar.update(1)
            if global_step >= config.num_steps:
                break
        if global_step >= config.num_steps:
            break

    progress_bar.close()

    return model, likelihoods, test_likelihoods


# %%
# =============================================================================
# Configuration
# =============================================================================

config = TrainingConfig(
    # Data settings
    limit_dataset_size=10000,
    batch_size=512,
    # Training settings
    num_steps=300,
    learning_rate=1e-0,
    weight_decay=0.0,  # L2 regularization (0.0 = no regularization)
    eval_freq=100,
    # Learning rate scheduler settings
    scheduler_type="step",  # Options: "step", "cosine", "none"
    scheduler_step_size=100,  # For step scheduler
    scheduler_gamma=0.1,  # For step scheduler
    # Model settings
    model_type="spline_exp_hawkes",  # Options: "poisson", "inhomogeneous_poisson", "spline_poisson", "hawkes", "spline_exp_hawkes"
    model_seed=43,
    # Spline-specific settings (only used for spline_poisson)
    spline_K=5,
    spline_delta_t=0.3,  # Time spacing for splines
    # Device settings
    device="cuda:0",
    # Weights & Biases settings
    wandb_enable=True,  # Set False to disable logging
    wandb_project="tpp-compare",  # Fill in your project name
    wandb_entity=None,  # Fill in your team/entity name if needed
    wandb_run_name=None,  # Auto-set to model_type if None
    wandb_tags=[],  # Add custom tags like ["experiment-1", "baseline"]
    wandb_mode="online",  # Options: "online", "offline", "dryrun"
)


def init_wandb_run(config: TrainingConfig):
    """Create a Weights & Biases run if configured, otherwise return None."""

    if not config.wandb_enable:
        return None

    if wandb is None:
        print("wandb not installed; set wandb_enable=False or install wandb to enable logging.")
        return None

    run = wandb.init(
        project=config.wandb_project,
        entity=config.wandb_entity,
        name=config.wandb_run_name,
        tags=config.wandb_tags,
        mode=config.wandb_mode,  # type: ignore
        config=config.to_dict(),
    )

    return run


# Make file paths relative to repository root for robust execution
ROOT = Path(__file__).resolve().parent.parent
base_data_path = ROOT / "data" / "ukb_simulated_data"
sequences, sexes, num_event_types = load_ukb_sequences(
    base_data_path / "train.bin", limit_size=config.limit_dataset_size
)
validation_sequences, _, _ = load_ukb_sequences(base_data_path / "val.bin", limit_size=config.limit_dataset_size)
# test_sequences, _, _ = load_ukb_sequences(base_data_path / "test.bin", limit_size=config.limit_dataset_size)

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

dataloader_train = DataLoader(
    dataset=dataset_train, batch_size=config.batch_size, shuffle=True, collate_fn=collate_batch
)
dataloader_val = DataLoader(dataset=dataset_val, batch_size=config.batch_size, shuffle=False, collate_fn=collate_batch)
# dataloader_test = DataLoader(dataset=dataset_test, batch_size=config.batch_size, shuffle=False, collate_fn=collate_batch)


# %%

# Determine model type and instantiate the appropriate model

D = int(num_event_types)  # Number of event types
sequence_length = torch.tensor([ts.time_points[-1] for ts in sequences])
T = sequence_length + (1 / 365)
N = len(sequences)  # Number of time-series


load_path = None

# Model factory
if config.model_type == "poisson":
    print("Creating Poisson Process model...")
    if load_path is not None and load_path.exists():
        print("Loading pre-trained model...")
        model = PoissonProcess(D=D, seed=config.model_seed)
        model.load_state_dict(torch.load(str(load_path)))
    else:
        model = PoissonProcess(D=D, seed=config.model_seed)
    save_filename = "poisson.pth"

elif config.model_type == "inhomogeneous_poisson":
    print("Creating Inhomogeneous Poisson Process model...")
    if load_path is not None and load_path.exists():
        print("Loading pre-trained model...")
        model = ConditionalInhomogeniousPoissonProcess(D=D, seed=config.model_seed)
        model.load_state_dict(torch.load(str(load_path)))
    else:
        model = ConditionalInhomogeniousPoissonProcess(D=D, seed=config.model_seed)
    save_filename = "inhomogeneous_poisson.pth"

elif config.model_type == "spline_poisson":
    print("Creating Spline Poisson Process model...")
    if load_path is not None and load_path.exists():
        print("Loading pre-trained model...")
        model = SplinePoissonProcess(D, config.spline_K, config.spline_delta_t)
        model.load_state_dict(torch.load(str(load_path)))
    else:
        model = SplinePoissonProcess(D, config.spline_K, config.spline_delta_t, seed=config.model_seed)
    save_filename = "splinepp.pth"

elif config.model_type == "hawkes":
    print("Creating Exponential Kernel Hawkes Process model...")
    if load_path is not None and load_path.exists():
        print("Loading pre-trained model...")
        model = ExpKernelHawkesProcess(D=D, seed=config.model_seed)
        model.load_state_dict(torch.load(str(load_path)))
    else:
        model = ExpKernelHawkesProcess(D=D, seed=config.model_seed)
    save_filename = "hawkes.pth"

elif config.model_type == "spline_exp_hawkes":
    print("Creating new modular Exponential Kernel Hawkes Process model...")
    if load_path is not None and load_path.exists():
        print("Loading pre-trained model...")
        model = SplineBaselineExpKernelHawkesProcess(
            D=D, num_knots=config.spline_K, delta_t=config.spline_delta_t, seed=config.model_seed
        )
        model.load_state_dict(torch.load(str(load_path)))
    else:
        model = SplineBaselineExpKernelHawkesProcess(
            D=D, num_knots=config.spline_K, delta_t=config.spline_delta_t, seed=config.model_seed
        )
    save_filename = "spline_hawkes.pth"

else:
    raise ValueError(
        f"Unknown model type: {config.model_type}. Choose from: poisson, inhomogeneous_poisson, spline_poisson, hawkes"
    )
# %%

wandb_run = init_wandb_run(config)

init_ll_train = torch.mean(measure_likelihood(model=model, dataloader=dataloader_train, device=config.device))
init_ll_val = torch.mean(measure_likelihood(model=model, dataloader=dataloader_val, device=config.device))
# init_ll_test = torch.mean(measure_likelihood(model=real_MVHP, dataloader=dataloader_test, device=DEVICE))

print(f"Baseline log-likelihood of init model on train dataset: {init_ll_train}")
print(f"Baseline log-likelihood of init model on val dataset: {init_ll_val}")
# print(f"Baseline log-likelihood of init model on test dataset: {init_ll_test}")

if wandb_run is not None and wandb is not None:
    wandb.log(
        {
            "baseline/train_log_likelihood": float(init_ll_train),
            "baseline/val_log_likelihood": float(init_ll_val),
            "dataset/N": len(sequences),
        },
        step=0,
    )

# %%

num_epochs_training = (config.num_steps * config.batch_size) / len(sequences)
print(f"Will train {num_epochs_training} epochs!")

# %%
# Run a short training for evaluation purposes
save_path = ROOT / "models" / f"new_{save_filename}"
fit_model, train_lls, val_lls = batched_train_loop(
    model,
    dataloader_train,
    config,
    test_events=dataloader_val,
    save_file_name=str(save_path),
    wandb_run=wandb_run,
)
# %%
# Implement loading and saving of mode
torch.save(fit_model.state_dict(), str(save_path))

if wandb_run is not None and wandb is not None:
    # Log final metrics and close out the run.
    final_train_ll = float(train_lls[-1]) if len(train_lls) > 0 else None
    final_val_ll = float(val_lls[-1]) if len(val_lls) > 0 else None
    log_payload = {
        "final/train_log_likelihood": final_train_ll,
        "final/val_log_likelihood": final_val_ll,
        "artifact/save_path": str(save_path),
    }
    wandb.log(log_payload, step=config.num_steps)
    wandb.finish()
