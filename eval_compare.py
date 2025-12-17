#!/usr/bin/env python3
"""
Evaluation script to compare Delphi transformer and Hawkes ExpKernel models
on next-event prediction metrics (class and time).

This script computes top-1/top-5 accuracy for next-event class prediction
and MAE/RMSE for predicted time-to-next-event for each model on their
respective test sets (Transformer uses `data/ukb_simulated_data/val.bin`/`test.bin`,
Hawkes uses `data/ukb_simulated_data/expansion.bin`).

Usage: python eval_compare.py
"""

import argparse
from pathlib import Path
import sys
import numpy as np
import torch

# Ensure local package imports in `hawkes/` resolve correctly when running as a script
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from utils import get_p2i, get_batch
from model import Delphi, DelphiConfig

from hawkes.ukb_loading import load_ukb_sequences
from hawkes.event_utils import BatchedMVEventData, MVEventData
from hawkes.Hawkes import ExpKernelMVHawkesProcess


def evaluate_delphi(ckpt_path: Path, data_dir: Path, device: str = "cpu", max_batches: int = 50):
    # Load checkpoint
    if not ckpt_path.exists():
        print(f"Delphi checkpoint {ckpt_path} not found. Skipping Delphi evaluation.")
        return None

    ckpt = torch.load(str(ckpt_path), map_location=device)
    model_args = ckpt.get("model_args")
    if model_args is None:
        print("No model args found in checkpoint. Can't instantiate Delphi model.")
        return None

    config = DelphiConfig(**model_args)
    model = Delphi(config).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # Load validation/test data memmap
    test_path = data_dir / "test.bin"
    if not test_path.exists():
        print(f"Test data {test_path} not found. Skipping Delphi evaluation.")
        return None

    test_data = np.memmap(str(test_path), dtype=np.uint32, mode="r").reshape(-1, 3)
    test_p2i = get_p2i(test_data)

    SHAWN_DATA_FIX = True
    if SHAWN_DATA_FIX:
        test_data = np.array(test_data, copy=True)

        test_data[:, 2] -= 1

    # iterate a few batches
    total = 0
    top1 = 0
    top5 = 0
    time_abs_errs = []
    time_sq_errs = []

    for batch_idx in range(max_batches):
        # sample a batch of size 16
        ix = np.random.randint(0, len(test_p2i), size=(16,))
        X, A, Y, B = get_batch(
            ix,
            test_data,
            test_p2i,
            select="left",
            block_size=config.block_size,
            device=device,
            padding="regular",
        )

        with torch.no_grad():
            logits, loss, _ = model(X, A)
            # take logits at last position
            last_logits = logits[:, -1, :]
            # convert logits to rates as used in generate(): rate_i = exp(logit_i)
            rates = torch.exp(last_logits)
            rate_sum = rates.sum(dim=1, keepdim=True)
            probs = rates / (rate_sum + 1e-12)

            # predicted time (expected min exponential) in same units as ages
            pred_dt = 1.0 / (rate_sum.squeeze(1) + 1e-12)

        # true next token at this last position is Y[:, -1]
        true_tokens = Y[:, -1].cpu()
        # compute top1/top5
        top1 += (probs.argmax(dim=1).cpu() == true_tokens).sum().item()
        top5 += sum([true_tokens[i].item() in probs[i].topk(5).indices.cpu().tolist() for i in range(len(true_tokens))])
        total += len(true_tokens)

        # true dt
        true_dt = (B - A)[:, -1].cpu().float()
        # mask invalids where target token is padding (0)
        valid_mask = true_tokens != 0
        for i in range(len(true_tokens)):
            if not valid_mask[i]:
                continue
            e = abs(pred_dt[i].cpu().item() - true_dt[i].item())
            time_abs_errs.append(e)
            time_sq_errs.append(e * e)

    metrics = {
        "top1": top1 / total,
        "top5": top5 / total,
        "time_mae": float(np.mean(time_abs_errs)) if time_abs_errs else None,
        "time_rmse": float(np.sqrt(np.mean(time_sq_errs))) if time_sq_errs else None,
    }
    print("Delphi metrics:", metrics)
    return metrics


def evaluate_hawkes(hawkes_state_path: Path | None, expansion_path: Path, device: str = "cpu", max_seqs: int = 2000):
    # Load sequences using ukb_loading
    sequences, sexes, num_event_types = load_ukb_sequences(expansion_path, limit_size=100)

    D = int(num_event_types)
    if hawkes_state_path is not None and hawkes_state_path.exists():
        # need to know D to instantiate
        hawkes = ExpKernelMVHawkesProcess(None, D).to(device)
        state = torch.load(str(hawkes_state_path), map_location=device)
        hawkes.load_state_dict(state)
    else:
        raise RuntimeError("Hawkes- Checkpoint not found!")

    hawkes.eval()

    # TODO Metrics
    total_int_t = []
    total_cumint_t = []
    total_pdf_t = []
    total_cdf_t = []

    for ts_all in sequences:
        for idx in range(2, len(ts_all)):
            ts = ts_all[:idx]

            history = MVEventData(ts.time_points[:-1], ts.event_types[:-1])
            batch = BatchedMVEventData(mv_events=[history])

            if len(history) == 0:
                last_event_time_tensor = torch.tensor(0.0).unsqueeze(0)
            else:
                last_event_time_tensor = torch.tensor(history.time_points[-1]).unsqueeze(0)

            target_time = ts.time_points[-1].item()
            target_type = ts.event_types[-1].item()

            target_time_tensor = torch.tensor(target_time).unsqueeze(0)

            # Evaluate likelihoods and so on here.

            # First the per-type intensities and probabilities.
            type_intensity_at_t = hawkes.intensity(target_time_tensor, batch)
            type_cumulative_intensity_t = hawkes.cumulative_intensity(
                target_time_tensor, batch
            ) - hawkes.cumulative_intensity(last_event_time_tensor, batch)

            # Joint density p(t,e). The likelihoods for all event types at the correct time (but not conditioned!!)
            type_PDF_at_t = hawkes.PDF(target_time_tensor, batch)

            # The likelihood of the correct event type at the correct time
            type_likelihood_at_t = type_PDF_at_t[0, target_type]

            # Distribution over types at time_point t.
            type_distribution_at_t = type_intensity_at_t / torch.sum(type_intensity_at_t)

            # Often TPPs are compared on next event prediction quality. We define the intensities and likelihoods for the next events.
            total_intensity_at_t = torch.sum(type_intensity_at_t)
            total_cumulative_intensity_at_t = torch.sum(type_cumulative_intensity_t)
            total_CDF_at_t = 1 - torch.exp(-total_cumulative_intensity_at_t)
            total_PDF_at_t = total_intensity_at_t * torch.exp(-total_cumulative_intensity_at_t)

            # The expected value of the time_pdf. Often used for L2 and L1 metrics.
            expected_t_given_history = None  # Hard to compute?? We can evaluate the PDF like above. But computing the mean requires sampling I think?
            median_t_given_history = None  # We could probably compute the median of the pdf (and then use abs error as metric), using the inverse of the CDF, as computed in the inverse sampling...

            # Sample multiple continuatations.
            samples = [hawkes.sample_inverse(ts, num_steps=1) for _ in range(100)]
            sample_times = [s[1] for s in samples]
            sample_type_dists = [s[2] for s in samples]

            # Estimate for the expected next event time.
            average_time = torch.mean(sample_times, dim=0)

            # Estimate for the distribution over event types of next token.
            average_sample_dist = torch.mean(sample_type_dists, dim=0)

            # We need to implement metrics here.

    metrics = {
        # TODO
    }
    print("Hawkes metrics:", metrics)
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--delphi_ckpt", type=str, default="models/ckpt.pt", help="Path to Delphi ckpt")
    parser.add_argument(
        "--hawkes_state", type=str, default="models/hawkes_trained.pth", help="Path to Hawkes state dict (optional)"
    )
    parser.add_argument("--data_dir", type=str, default="data/ukb_simulated_data", help="Path to ukb data dir")
    parser.add_argument(
        "--expansion_bin", type=str, default="data/ukb_simulated_data/test.bin", help="Expansion bin for Hawkes"
    )
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    expansion = Path(args.expansion_bin)

    # print("Evaluating Delphi...")
    # evaluate_delphi(Path(args.delphi_ckpt), data_dir, device=args.device)

    print("Evaluating Hawkes...")
    evaluate_hawkes(Path(args.hawkes_state) if args.hawkes_state else None, expansion, device=args.device)


if __name__ == "__main__":
    main()
