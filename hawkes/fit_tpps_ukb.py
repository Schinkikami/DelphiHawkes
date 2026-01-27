# %%
from pathlib import Path
from typing import Optional, List, Union, Any
from dataclasses import dataclass, asdict
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from simple_parsing import ArgumentParser, field, Serializable

# Optional argcomplete for tab completion (activate with: eval "$(register-python-argcomplete fit_tpps_ukb.py)")
try:
    import argcomplete
except ImportError:
    argcomplete = None

# Optional Weights & Biases support (safe to leave unconfigured)
try:
    import wandb
except ImportError:  # pragma: no cover - optional dependency
    wandb = None

# PyTorch profiler for performance analysis
from torch.profiler import profile, ProfilerActivity

from hawkes.event_utils import MVEventData, BatchedMVEventData
from hawkes.tpps import TemporalPointProcess
from hawkes.baseline_tpps import (
    PoissonProcess,
    ConditionalInhomogeniousPoissonProcess,
    SplinePoissonProcess,
)
from hawkes.hawkes_tpp import (
    ExpKernelHawkesProcess,
    LinearBaselineExpKernelHawkesProcess,
    SplineBaselineExpKernelHawkesProcess,
    SoftplusConstExpIHawkesProcess,
    SoftplusSplineExpIHawkesProcess,
    NumericalSplineBaselineExpKernelHawkesProcess,
)
from hawkes.ukb_loading import load_ukb_sequences
from hawkes.evaluation import evaluate_tpp, format_metrics

# Valid model types for reference
VALID_MODEL_TYPES = [
    "poisson",
    "inhomogeneous_poisson",
    "spline_poisson",
    "hawkes",
    "linear_exp_hawkes",
    "spline_exp_hawkes",
    "softplus_const_exp_ihawkes",
    "softplus_spline_exp_ihawkes",
    "numerical_spline_exp_hawkes",
]

# %%


@dataclass
class TrainingConfig(Serializable):
    """Configuration for training temporal point processes."""

    # ==========================================================================
    # CRITICAL SETTINGS (no defaults - must be explicitly provided)
    # ==========================================================================
    model_type: str = field(help=f"Model type to train. Choices: {', '.join(VALID_MODEL_TYPES)}")

    # ==========================================================================
    # Data settings
    # ==========================================================================
    data_dir: str = field(
        default="data/ukb_simulated_data", help="Directory containing data files (relative to repo root)"
    )
    train_file: str = field(default="train.bin", help="Training data file")
    val_file: str = field(default="val.bin", help="Validation data file")
    test_file: str = field(default="test.bin", help="Test data file")
    limit_dataset_size: int = field(default=10000, help="Max sequences to load (-1 for all)")
    batch_size: int = field(default=512, help="Batch size for training")

    # ==========================================================================
    # Training settings
    # ==========================================================================
    num_steps: int = field(default=500, help="Total training steps")
    learning_rate: float = field(default=1.0, help="Initial learning rate")
    weight_decay: float = field(default=0.0, help="L2 regularization (0.0 = none)")
    eval_freq: int = field(default=100, help="Evaluate every N steps")

    # ==========================================================================
    # Learning rate scheduler settings
    # ==========================================================================
    scheduler_type: str = field(default="step", help="LR scheduler type: step, cosine, none")
    scheduler_step_size: int = field(default=100, help="Step scheduler: decay every N steps")
    scheduler_gamma: float = field(default=0.1, help="Step scheduler: LR multiplier")

    # ==========================================================================
    # Model settings
    # ==========================================================================
    model_seed: int = field(default=42, help="Random seed for reproducibility")
    num_ci_integration_steps: int = field(default=100, help="Integration steps for cumulative intensity")
    spline_K: int = field(default=5, help="Number of spline knots")
    max_spline_T: float = field(default=1.25, help="Max time for spline baseline (delta_t = max_spline_T / spline_K)")

    # ==========================================================================
    # Device settings
    # ==========================================================================
    device: str = field(default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda/cpu)")

    # ==========================================================================
    # Weights & Biases settings
    # ==========================================================================
    wandb_enable: bool = field(default=False, help="Enable W&B logging")
    wandb_project: str = field(default="tpp_compare", help="W&B project name")
    wandb_entity: Optional[str] = field(default=None, help="W&B entity (None = personal)")
    wandb_run_name: Optional[str] = field(default=None, help="W&B run name (None = auto)")
    wandb_tags: Optional[List[str]] = field(default=None, help="W&B tags")
    wandb_mode: str = field(default="online", help="W&B mode: online, offline, dryrun")

    # ==========================================================================
    # Test evaluation settings
    # ==========================================================================
    run_test_eval: bool = field(default=True, help="Run evaluation on test set after training")
    test_limit_sequences: Optional[int] = field(default=None, help="Limit test sequences (None = use all data)")
    compute_marginal_type: bool = field(default=True, help="Compute marginal type p(m|H) - expensive")

    # ==========================================================================
    # Model saving settings
    # ==========================================================================
    save_path: Optional[str] = field(default=None, help="Path to save model (None = auto-generate from model_type)")
    overwrite: bool = field(default=False, help="Overwrite existing model file if it exists")
    auto_increment: bool = field(
        default=True, help="Auto-increment filename if file exists (ignored if overwrite=True)"
    )

    # ==========================================================================
    # Profiling settings
    # ==========================================================================
    profile: bool = field(default=False, help="Enable PyTorch profiler for performance analysis")
    profile_steps: int = field(default=20, help="Number of steps to profile (after 5 warmup steps)")
    profile_output: str = field(default="profile_trace.json", help="Output file for profiler trace")

    # ==========================================================================
    # Compilation settings (torch.compile for PyTorch 2.x)
    # ==========================================================================
    compile: bool = field(default=False, help="Enable torch.compile() for faster training (PyTorch 2.x)")
    compile_mode: str = field(
        default="default", help="Compile mode: default, reduce-overhead, max-autotune, max-autotune-no-cudagraphs"
    )
    compile_backend: str = field(default="inductor", help="Compile backend: inductor, cudagraphs, eager, etc.")

    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.model_type not in VALID_MODEL_TYPES:
            raise ValueError(f"Unknown model_type: '{self.model_type}'. Choose from: {', '.join(VALID_MODEL_TYPES)}")
        # Handle -1 as "load all"
        if self.limit_dataset_size is not None and self.limit_dataset_size < 0:
            self.limit_dataset_size = None  # type: ignore

    @property
    def spline_delta_t(self) -> float:
        """Compute delta_t from max_spline_T and spline_K."""
        return self.max_spline_T / self.spline_K

    def as_dict(self):
        """Convert config to dictionary for logging."""
        return asdict(self)


def parse_args() -> TrainingConfig:
    """Parse command line arguments and return a TrainingConfig."""
    parser = ArgumentParser(description="Train temporal point process models on UKB data.")
    parser.add_arguments(TrainingConfig, dest="config")

    # Enable tab completion if argcomplete is installed
    if argcomplete is not None:
        argcomplete.autocomplete(parser)

    args = parser.parse_args()
    return args.config


# %%


def measure_likelihood(model: Union[TemporalPointProcess, Any], dataloader: DataLoader, device):
    model = model.to(device)

    lls = []

    for batch in tqdm(dataloader):
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
    model: Union[TemporalPointProcess, Any],
    events_batch: DataLoader,
    config: TrainingConfig,
    test_events: Optional[DataLoader] = None,
    save_file_name: Optional[str] = None,
    wandb_run=None,
    profiler=None,
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

            # Step the profiler if enabled
            if profiler is not None:
                profiler.step()

            if global_step >= config.num_steps:
                break
        if global_step >= config.num_steps:
            break

    progress_bar.close()

    return model, likelihoods, test_likelihoods


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
        config=config.as_dict(),
    )

    return run


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


def create_model(config: TrainingConfig, D: int, load_path: Optional[Path] = None):
    """Factory function to create the appropriate TPP model.

    Args:
        config: Training configuration
        D: Number of event types
        load_path: Optional path to load pre-trained weights

    Returns:
        Tuple of (model, save_filename)
    """
    if config.model_type == "poisson":
        print("Creating Poisson Process model...")
        model = PoissonProcess(D=D, seed=config.model_seed)
        save_filename = "poisson.pth"

    elif config.model_type == "inhomogeneous_poisson":
        print("Creating Inhomogeneous Poisson Process model...")
        model = ConditionalInhomogeniousPoissonProcess(D=D, seed=config.model_seed)
        save_filename = "inhomogeneous_poisson.pth"

    elif config.model_type == "spline_poisson":
        print("Creating Spline Poisson Process model...")
        model = SplinePoissonProcess(D, config.spline_K, config.spline_delta_t, seed=config.model_seed)
        save_filename = "splinepp.pth"

    elif config.model_type == "hawkes":
        print("Creating Exponential Kernel Hawkes Process model...")
        model = ExpKernelHawkesProcess(D=D, seed=config.model_seed)
        save_filename = "hawkes.pth"

    elif config.model_type == "linear_exp_hawkes":
        print("Creating Linear Baseline Exponential Kernel Hawkes Process model...")
        model = LinearBaselineExpKernelHawkesProcess(D=D, seed=config.model_seed)
        save_filename = "linear_exp_hawkes.pth"

    elif config.model_type == "spline_exp_hawkes":
        print("Creating Spline Baseline Exponential Kernel Hawkes Process model...")
        model = SplineBaselineExpKernelHawkesProcess(
            D=D, num_knots=config.spline_K, delta_t=config.spline_delta_t, seed=config.model_seed
        )
        save_filename = "spline_hawkes.pth"

    elif config.model_type == "softplus_const_exp_ihawkes":
        print("Creating Softplus Const Exp Inhibitive Hawkes Process model...")
        model = SoftplusConstExpIHawkesProcess(
            D=D,
            seed=config.model_seed,
            baseline_params=None,
            kernel_params=None,
        )
        model.ci_num_points = config.num_ci_integration_steps
        save_filename = "softplus_const_exp_ihawkes.pth"

    elif config.model_type == "softplus_spline_exp_ihawkes":
        print("Creating Softplus Spline Exp Inhibitive Hawkes Process model...")
        model = SoftplusSplineExpIHawkesProcess(
            D=D,
            num_knots=config.spline_K,
            delta_t=config.spline_delta_t,
            seed=config.model_seed,
            baseline_params=None,
            kernel_params=None,
        )
        model.ci_num_points = config.num_ci_integration_steps
        save_filename = "softplus_spline_exp_ihawkes.pth"

    elif config.model_type == "numerical_spline_exp_hawkes":
        print("Creating Numerical Spline Baseline Exponential Kernel Hawkes Process model...")
        model = NumericalSplineBaselineExpKernelHawkesProcess(
            D=D, num_knots=config.spline_K, delta_t=config.spline_delta_t, seed=config.model_seed
        )
        save_filename = "numerical_spline_hawkes.pth"

    else:
        # This should never happen due to validation in TrainingConfig.__post_init__
        raise ValueError(f"Unknown model type: {config.model_type}")

    # Load pre-trained weights if specified
    if load_path is not None and load_path.exists():
        print(f"Loading pre-trained model from {load_path}...")
        model.load_state_dict(torch.load(str(load_path)))

    return model, save_filename


def main():
    """Main training function."""
    # Parse command line arguments
    config = parse_args()

    print("=" * 60)
    print("Training Configuration:")
    print("=" * 60)
    for key, value in config.as_dict().items():
        print(f"  {key}: {value}")
    print("=" * 60)

    # Make file paths relative to repository root for robust execution
    ROOT = Path(__file__).resolve().parent.parent
    base_data_path = ROOT / config.data_dir
    train_path = base_data_path / config.train_file
    val_path = base_data_path / config.val_file
    test_path = base_data_path / config.test_file

    # Load data
    print(f"\nLoading training data from {train_path}...")
    sequences, sexes, num_event_types = load_ukb_sequences(train_path, limit_size=config.limit_dataset_size)
    print(f"Loaded {len(sequences)} training sequences with {num_event_types} event types")

    print(f"Loading validation data from {val_path}...")
    validation_sequences, _, _ = load_ukb_sequences(val_path, limit_size=config.limit_dataset_size)
    print(f"Loaded {len(validation_sequences)} validation sequences")

    # Create datasets and dataloaders
    dataset_train = ListDataset(sequences)
    dataset_val = ListDataset(validation_sequences)

    dataloader_train = DataLoader(
        dataset=dataset_train, batch_size=config.batch_size, shuffle=True, collate_fn=collate_batch
    )
    dataloader_val = DataLoader(
        dataset=dataset_val, batch_size=config.batch_size, shuffle=False, collate_fn=collate_batch
    )

    # Create model
    D = int(num_event_types)
    model, default_save_filename = create_model(config, D, load_path=None)

    # ==========================================================================
    # Compile model with torch.compile() for PyTorch 2.x speedups
    # ==========================================================================
    if config.compile:
        print(
            f"\nCompiling model with torch.compile(mode='{config.compile_mode}', backend='{config.compile_backend}')..."
        )
        print("Note: First forward pass will be slow due to compilation.")
        model = torch.compile(model, mode=config.compile_mode, backend=config.compile_backend)  # type: ignore

    # ==========================================================================
    # Determine save path and check for existing files BEFORE training
    # ==========================================================================
    if config.save_path is not None:
        save_path = Path(config.save_path)
        # Make absolute if relative
        if not save_path.is_absolute():
            save_path = ROOT / save_path
    else:
        save_path = ROOT / "models" / f"new_{default_save_filename}"

    # Ensure parent directory exists
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # Check if file exists and handle accordingly
    if save_path.exists():
        if config.overwrite:
            print(f"\nWarning: {save_path} exists and will be overwritten.")
        elif config.auto_increment:
            # Find next available filename
            base = save_path.stem
            suffix = save_path.suffix
            parent = save_path.parent
            counter = 1
            while save_path.exists():
                save_path = parent / f"{base}_{counter}{suffix}"
                counter += 1
            print(f"\nFile exists, using incremented path: {save_path}")
        else:
            raise FileExistsError(
                f"Model file already exists: {save_path}\n"
                f"Use --overwrite to overwrite, or --auto_increment to auto-increment filename."
            )

    print(f"Model will be saved to: {save_path}")

    # Initialize W&B
    wandb_run = init_wandb_run(config)

    # Measure initial likelihoods
    print("\nMeasuring initial likelihoods...")
    init_ll_train = torch.mean(measure_likelihood(model=model, dataloader=dataloader_train, device=config.device))
    init_ll_val = torch.mean(measure_likelihood(model=model, dataloader=dataloader_val, device=config.device))

    print(f"Initial log-likelihood on train dataset: {init_ll_train:.4f}")
    print(f"Initial log-likelihood on val dataset: {init_ll_val:.4f}")

    if wandb_run is not None and wandb is not None:
        wandb.log(
            {
                "baseline/train_log_likelihood": float(init_ll_train),
                "baseline/val_log_likelihood": float(init_ll_val),
                "dataset/N": len(sequences),
            },
            step=0,
        )

    # Calculate epochs
    num_epochs_training = (config.num_steps * config.batch_size) / len(sequences)
    print(f"\nWill train for approximately {num_epochs_training:.2f} epochs")

    # Train model
    print("\nStarting training...")

    if config.profile:
        # Profile training with PyTorch profiler
        print(f"\nProfiling enabled: will profile {config.profile_steps} steps after 5 warmup steps")
        print(f"Profile output will be saved to: {config.profile_output}")
        print("View the trace in Chrome at chrome://tracing or in TensorBoard")

        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            schedule=torch.profiler.schedule(
                wait=1,  # Skip first step
                warmup=5,  # Warmup steps (not recorded)
                active=config.profile_steps,  # Steps to record
                repeat=1,  # Only do this once
            ),
            on_trace_ready=torch.profiler.tensorboard_trace_handler("./profiler_logs"),
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
        ) as prof:
            # Use a modified training loop that steps the profiler
            fit_model, train_lls, val_lls = batched_train_loop(
                model,
                dataloader_train,
                config,
                test_events=dataloader_val,
                save_file_name=str(save_path),
                wandb_run=wandb_run,
                profiler=prof,
            )

        # Export Chrome trace
        prof.export_chrome_trace(config.profile_output)
        print(f"\nProfile trace saved to: {config.profile_output}")

        # Print summary table
        print("\n" + "=" * 60)
        print("Profiler Summary (sorted by CUDA time):")
        print("=" * 60)
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))

        print("\n" + "=" * 60)
        print("Profiler Summary (sorted by CPU time):")
        print("=" * 60)
        print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=20))
    else:
        fit_model, train_lls, val_lls = batched_train_loop(
            model,
            dataloader_train,
            config,
            test_events=dataloader_val,
            save_file_name=str(save_path),
            wandb_run=wandb_run,
        )

    # Compute best validation log-likelihood
    best_val_ll = max(val_lls) if val_lls else float("-inf")

    # Save final model
    torch.save(fit_model.state_dict(), str(save_path))
    print(f"\nModel saved to {save_path}")

    print("\nTraining complete!")
    if len(train_lls) > 0:
        print(f"Final train log-likelihood: {train_lls[-1]:.4f}")
    if len(val_lls) > 0:
        print(f"Final val log-likelihood: {val_lls[-1]:.4f}")
        print(f"Best val log-likelihood: {best_val_ll:.4f}")

    # ==========================================================================
    # Test Evaluation
    # ==========================================================================
    test_metrics = None
    if config.run_test_eval:
        print("\n" + "=" * 60)
        print("Running Test Evaluation")
        print("=" * 60)

        # Load test data (None means load all)
        test_limit = config.test_limit_sequences
        print(f"Loading test data from {test_path}...")
        test_sequences, _, _ = load_ukb_sequences(test_path, limit_size=test_limit)
        print(f"Loaded {len(test_sequences)} test sequences")

        # Run evaluation
        print("\nEvaluating model on test set...")
        test_metrics = evaluate_tpp(
            model=fit_model,
            sequences=test_sequences,
            num_event_types=D,
            device=config.device,
            compute_marginal_type=config.compute_marginal_type,
        )

        # Print results
        print("\n" + "=" * 60)
        print("Test Evaluation Results:")
        print("=" * 60)
        print(format_metrics(test_metrics, prefix="  "))

    # ==========================================================================
    # Log final metrics to W&B (using summary for ranking/filtering)
    # ==========================================================================
    if wandb_run is not None and wandb is not None:
        final_train_ll = float(train_lls[-1]) if len(train_lls) > 0 else None
        final_val_ll = float(val_lls[-1]) if len(val_lls) > 0 else None

        # Use wandb.summary for metrics you want to rank/filter by in the UI
        # These appear as columns in the runs table and can be sorted/filtered
        wandb.summary["best_val_ll"] = best_val_ll
        wandb.summary["final_val_ll"] = final_val_ll
        wandb.summary["final_train_ll"] = final_train_ll

        # Add test metrics to summary (for ranking/filtering)
        if test_metrics is not None:
            for key, value in test_metrics.items():
                wandb.summary[f"test_{key}"] = value

        # Also log as regular metrics for the chart history
        log_payload = {
            "final/train_log_likelihood": final_train_ll,
            "final/val_log_likelihood": final_val_ll,
            "final/best_val_log_likelihood": best_val_ll,
            "artifact/save_path": str(save_path),
        }

        # Add test metrics to regular log as well
        if test_metrics is not None:
            for key, value in test_metrics.items():
                log_payload[f"test/{key}"] = value

        wandb.log(log_payload, step=config.num_steps)
        wandb.finish()


if __name__ == "__main__":
    main()
