"""
Visualization script for intensity curves.

This script loads trained Poisson process models (constant, linear, and spline intensities)
and visualizes their intensity functions over the lifetime.
"""

from pathlib import Path
import torch
import matplotlib.pyplot as plt
from hawkes.hawkes.hawkes_tpp import (
    ExpKernelHawkesProcess,
    LinearBaselineExpKernelHawkesProcess,
    SplineBaselineExpKernelHawkesProcess,
)
from hawkes.hawkes.event_utils import BatchedMVEventData


def load_model(model_path: Path, model_type: str, D: int = 1268, num_knots: int = 15, delta_t: float = 0.1):
    """Load a trained model from disk.

    Args:
        model_path: Path to the saved model weights
        model_type: Type of model ('poisson', 'inhomogeneous', 'spline')
        D: Number of event dimensions (default: 1268)
        num_knots: Number of knots for spline model (default: 15)
        delta_t: Time spacing for spline knots (default: 0.1 = 1.5/15)

    Returns:
        Loaded model
    """
    if model_type == "poisson":
        model = ExpKernelHawkesProcess(D=D, seed=42)
    elif model_type == "inhomogeneous":
        model = LinearBaselineExpKernelHawkesProcess(D=D, seed=42)
    elif model_type == "spline":
        model = SplineBaselineExpKernelHawkesProcess(D=D, num_knots=num_knots, delta_t=delta_t, seed=42)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    if model_path.exists():
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
        print(f"Loaded {model_type} model from {model_path}")
    else:
        print(f"Warning: Model file {model_path} not found. Using randomly initialized model.")

    model.eval()
    return model


def plot_interaction_matrix(model, time_range, ts: BatchedMVEventData):
    matrix = 1 - torch.exp(-model.cumulative_intensity(time_range.unsqueeze(0), ts).squeeze(0))


model = load_model(Path("models/new_splinepp.pth"), "spline", D=1268, num_knots=15, delta_t=0.1)
time_range = torch.tensor([0.0, 1.0])  # Normalized time range
ts = BatchedMVEventData(mv_events=[])

plot_cumulative_intensity_curves(model, time_range, ts)

exit()


def compute_intensity_curves(models: dict, time_range: torch.Tensor, event_dims: list = None):
    """Compute intensity curves for given models over a time range.

    Args:
        models: Dictionary mapping model names to model instances
        time_range: 1D tensor of time points
        event_dims: List of event dimension indices to plot (None = marginal intensity)

    Returns:
        Dictionary mapping model names to intensity curves
    """
    intensities = {}

    # Create a dummy batched event data (batch size 1) for intensity computation
    # BatchedMVEventData expects lists of 1D tensors
    dummy_batch = BatchedMVEventData(
        time_points=[torch.zeros(1)],
        event_types=[torch.zeros(1, dtype=torch.long)],
    )

    for name, model in models.items():
        with torch.no_grad():
            # Compute intensity for each time point
            intensity_list = []
            for t in time_range:
                intensity = model.intensity(t.unsqueeze(0), dummy_batch)  # Shape: (1, D)
                intensity_list.append(intensity)

            # Stack into (T, D)
            intensity_tensor = torch.cat(intensity_list, dim=0)  # Shape: (T, D)

            if event_dims is None:
                # Compute marginal intensity (sum over all dimensions)
                intensity_tensor = intensity_tensor.sum(dim=1)  # Shape: (T,)
            else:
                # Select specific dimensions
                intensity_tensor = intensity_tensor[:, event_dims]  # Shape: (T, len(event_dims))

            intensities[name] = intensity_tensor

    return intensities


def plot_marginal_intensities(
    intensities: dict, time_range: torch.Tensor, title: str = "Marginal Intensity Curves", save_path: str = None
):
    """Plot marginal intensity curves for multiple models.

    Args:
        intensities: Dictionary mapping model names to intensity curves (1D tensors)
        time_range: 1D tensor of time points
        title: Plot title
        save_path: Path to save the figure (optional)
    """
    plt.figure(figsize=(12, 6))

    colors = {"Constant (Poisson)": "blue", "Linear (Inhomogeneous Poisson)": "green", "Spline (Spline Poisson)": "red"}

    for name, intensity in intensities.items():
        color = colors.get(name, None)
        plt.plot(time_range.numpy(), intensity.numpy(), label=name, linewidth=2, color=color)

    plt.xlabel("Time (normalized: 0-1 = 0-80 years)", fontsize=12)
    plt.ylabel("Marginal Intensity λ(t)", fontsize=12)
    plt.title(title, fontsize=14, fontweight="bold")
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved figure to {save_path}")

    plt.show()


def plot_dimension_intensities(
    intensities: dict,
    time_range: torch.Tensor,
    dim_names: list = None,
    title: str = "Intensity Curves by Dimension",
    save_path: str = None,
):
    """Plot intensity curves for specific dimensions across models.

    Args:
        intensities: Dictionary mapping model names to intensity curves (2D tensors: T x D)
        time_range: 1D tensor of time points
        dim_names: Names of the dimensions being plotted
        title: Plot title
        save_path: Path to save the figure (optional)
    """
    n_models = len(intensities)
    n_dims = list(intensities.values())[0].shape[1]

    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5))
    if n_models == 1:
        axes = [axes]

    for ax, (name, intensity) in zip(axes, intensities.items()):
        for i in range(n_dims):
            dim_label = dim_names[i] if dim_names else f"Dimension {i}"
            ax.plot(time_range.numpy(), intensity[:, i].numpy(), label=dim_label, linewidth=2)

        ax.set_xlabel("Time (normalized: 0-1 = 0-80 years)", fontsize=11)
        ax.set_ylabel("Intensity λ_d(t)", fontsize=11)
        ax.set_title(name, fontsize=12, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved figure to {save_path}")

    plt.show()


def main():
    """Main visualization workflow."""

    # Configuration
    models_dir = Path("models")
    D = 1268  # Number of event types
    num_knots = 15  # Determined from saved model state dict
    delta_t = 0.1  # 1.5 / 15

    # Load models
    print("Loading models...")
    models = {
        "Constant (Poisson)": load_model(models_dir / "new_poisson.pth", "poisson", D=D),
        "Linear (Inhomogeneous Poisson)": load_model(
            models_dir / "new_inhomogeneous_poisson.pth", "inhomogeneous", D=D
        ),
        "Spline (Spline Poisson)": load_model(
            models_dir / "new_splinepp.pth", "spline", D=D, num_knots=num_knots, delta_t=delta_t
        ),
    }

    # Time range (normalized: 0-1 represents 0-80 years in the original data)
    time_range = torch.linspace(0, 1, 200)

    # Compute marginal intensities
    print("\nComputing marginal intensities...")
    marginal_intensities = compute_intensity_curves(models, time_range, event_dims=None)

    # Plot marginal intensities
    print("\nPlotting marginal intensities...")
    plot_marginal_intensities(
        marginal_intensities,
        time_range,
        title="Marginal Intensity Functions: Comparison of Poisson Process Models",
        save_path="poisson_marginal_intensities.png",
    )

    # Select a few interesting dimensions to visualize
    # For example, dimensions 0, 10, 50 (arbitrary selection)
    selected_dims = [0, 10, 50]
    print(f"\nComputing intensities for selected dimensions: {selected_dims}...")
    dim_intensities = compute_intensity_curves(models, time_range, event_dims=selected_dims)

    # Plot dimension-specific intensities
    print("\nPlotting dimension-specific intensities...")
    plot_dimension_intensities(
        dim_intensities,
        time_range,
        dim_names=[f"Event Type {d}" for d in selected_dims],
        title="Intensity Functions for Selected Event Types",
        save_path="poisson_dimension_intensities.png",
    )

    # Print summary statistics
    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)

    for name, intensity in marginal_intensities.items():
        intensity_np = intensity.numpy()
        print(f"\n{name}:")
        print(f"  Mean intensity: {intensity_np.mean():.4f}")
        print(f"  Min intensity:  {intensity_np.min():.4f}")
        print(f"  Max intensity:  {intensity_np.max():.4f}")
        print(f"  Std intensity:  {intensity_np.std():.4f}")

    print("\n" + "=" * 60)
    print("Visualization complete!")


if __name__ == "__main__":
    main()
