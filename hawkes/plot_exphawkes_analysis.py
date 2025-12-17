import sys
from pathlib import Path

# Ensure project root (parent of this script's parent) is on sys.path so
# `import hawkes...` works regardless of the current working directory.
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
import matplotlib.pyplot as plt
from hawkes.event_utils import BatchedMVEventData, MVEventData
from hawkes.Hawkes import ExpKernelMVHawkesProcess


def main():
    # Sequence from test_exphawkes.py
    seq = MVEventData(
        torch.tensor([0.2, 0.8, 2.0, 2.3], dtype=torch.float32), torch.tensor([0, 0, 1, 0], dtype=torch.long)
    )

    # Forecast horizon: we'll plot forecasts up to this absolute time.
    t_end = 6.0
    num_points = 2000

    # Small epsilon to evaluate just after conditioning times
    eps = 1e-4

    # Conditioning times: from 0.0 and after each observed event time
    conditioning_times = [0.0] + seq.time_points.tolist()

    # Prepare a color map
    import matplotlib

    cmap = matplotlib.cm.get_cmap("tab10")

    model = ExpKernelMVHawkesProcess(None, D=2)

    # We'll collect results for plotting: for each conditioning time we compute
    # times_i (absolute times >= cond_time+eps) and the 4 quantities per-dim.
    forecasts = []
    for i, s in enumerate(conditioning_times):
        s = float(s)
        times_i = torch.linspace(s + eps, t_end, num_points)

        # Build batch where each batch element uses the subsequence of observed events up to s
        mask = seq.time_points <= s + 1e-12
        past_times = seq.time_points[mask]
        past_types = seq.event_types[mask]
        # create BatchedMVEventData repeated for each evaluation time
        time_list = [past_times.clone() for _ in range(len(times_i))]
        type_list = [past_types.clone() for _ in range(len(times_i))]
        batch_i = BatchedMVEventData(time_list, type_list)

        T = times_i.clone()
        T.requires_grad_(False)

        pdf_t = model.PDF(T, batch_i)  # shape (B, D)
        cdf_t = model.CDF(T, batch_i)
        intensity_t = model.intensity(T, batch_i)
        ci_t = model.cumulative_intensity(T, batch_i)

        forecasts.append(
            (
                s,
                times_i.detach().numpy(),
                pdf_t.detach().numpy(),
                cdf_t.detach().numpy(),
                intensity_t.detach().numpy(),
                ci_t.detach().numpy(),
            )
        )

    # Print numeric summaries per forecast (PDF integral vs CDF delta; sample first values)
    import numpy as np

    print("Per-forecast summary (pdf trapz vs cdf delta, and first evaluation sample):")
    for s, times_i, pdf_i, cdf_i, intensity_i, ci_i in forecasts:
        print(f"\nForecast conditioned at t={s:.6f}:")
        for d in range(pdf_i.shape[1]):
            trap = np.trapz(pdf_i[:, d], x=times_i)
            cdf_diff = cdf_i[-1, d] - cdf_i[0, d]
            print(f" dim {d}: trapz={trap:.6e}, cdf_delta={cdf_diff:.6e}, abs_err={abs(trap - cdf_diff):.6e}")

        # initial sample values (first evaluation point)
        t0 = times_i[0]
        print(f" sample t0 = {t0:.6f}")
        for d in range(ci_i.shape[1]):
            print(
                f"  dim {d}: ci={ci_i[0, d]:.6e}, intensity={intensity_i[0, d]:.6e}, pdf={pdf_i[0, d]:.6e}, cdf={cdf_i[0, d]:.6e}"
            )

    # Plots
    # Create multi-start forecast plots
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))

    for idx, (s, times_i, pdf_i, cdf_i, intensity_i, ci_i) in enumerate(forecasts):
        color = cmap(idx % 10)
        label = f"forecast at t={s:.3f}"
        for d in range(pdf_i.shape[1]):
            axs[0, 0].plot(
                times_i, pdf_i[:, d], color=color, alpha=0.9, linewidth=1.0, label=(label if d == 0 else None)
            )
            axs[0, 1].plot(
                times_i, cdf_i[:, d], color=color, alpha=0.9, linewidth=1.0, label=(label if d == 0 else None)
            )
            axs[1, 0].plot(
                times_i,
                intensity_i[:, d],
                color=color,
                alpha=0.6,
                linestyle=("-" if d == 0 else "--"),
                label=(label if d == 0 else None),
            )
            axs[1, 1].plot(
                times_i,
                ci_i[:, d],
                color=color,
                alpha=0.6,
                linestyle=("-" if d == 0 else "--"),
                label=(label if d == 0 else None),
            )

    axs[0, 0].set_title("PDF forecasts (per-dim)")
    axs[0, 1].set_title("CDF forecasts (per-dim)")
    axs[1, 0].set_title("Intensity forecasts (per-dim)")
    axs[1, 1].set_title("Cumulative intensity forecasts (per-dim)")

    # Legends and grid
    for ax in axs.flatten():
        ax.grid(True)
        ax.legend()

    plt.tight_layout()
    outpath = "./hawkes/exphawkes_forecasts.png"
    plt.savefig(outpath, dpi=200)
    print(f"Saved multi-start forecast plot to {outpath}")


if __name__ == "__main__":
    main()
