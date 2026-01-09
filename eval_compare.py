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
import tqdm

# Ensure local package imports in `hawkes/` resolve correctly when running as a script
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from utils import get_p2i, get_batch
from model import Delphi, DelphiConfig

from hawkes.ukb_loading import load_ukb_sequences
from hawkes.event_utils import BatchedMVEventData
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
    num_sequences = 0
    total_predictions = 0

    joint_likelihood = []
    time_likelihood = []
    type_time_cond_likelihood = []

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
            logits, _, _ = model(X, A, Y, B, validation_loss_mode=True)

            valid = Y >= 13

            # TODO Hawkes so far ignores the emtpy context prediction, a bug I have to fix.. Not complete match.
            # To make them comparable, we also mask the first prediction, so we only have predictions with context.
            has_true = valid.any(dim=1)  # [B]
            first_idx = valid.int().argmax(dim=1)  # [B], undefined if no True
            batch_idx = torch.arange(valid.size(0), device=valid.device)
            valid[batch_idx[has_true], first_idx[has_true]] = False

            delta_t = B - A

            # Eleminate the NoEvent and Padding token, as well as Lifestyle tokens from predictions.
            logits[..., :13] = -torch.inf

            # convert logits to exp dist rates: rate_i = exp(logit_i)
            rates = torch.exp(logits)
            rate_sum = rates.sum(dim=2)
            probs = rates / rate_sum.unsqueeze(-1)

            # P(E=e| T=t, H_t) => P(E=e| H_t), as we constant (t independent) constant intensities.
            type_likelihoods = torch.take_along_dim(probs, Y.unsqueeze(-1), dim=2).squeeze(
                2
            )  # At t, as it is constant over time.
            log_type_likelihoods = torch.log(type_likelihoods)[valid]  # Commpute the LL, only take for valid targets.

            # Compute P(T=t|H_t). Use the fact that the intensities are constant. The the super-process (minimum of all event types)
            # also has constant intensity super_rate = \sum_i rate_i --> Is exponential.
            # Compute log PDF at point of exponential dist.: log(\lambda * exp(-\lambda*t)) == log(lambda) - (lambda*t)
            log_pdf_time = torch.log(rate_sum) - (rate_sum * delta_t)
            log_pdf_time = log_pdf_time[valid]

            # Now compute the joint density p(T=t, E=e|H_t) = p(T=t|H_t) * p(E=e| T=t, H_t) = p(T=t|H_t) * p(E=e| H_t).
            # Last step due to constant intensities.
            # For log joint density: log(p(T=t, E=e|H_t)) = log(p(T=t|H_t) * p(E=e| H_t)) = log(p(T=t|H_t)) + log(p(E=e| H_t))
            log_joint_density = log_type_likelihoods + log_pdf_time

            joint_likelihood.append(log_joint_density)
            time_likelihood.append(log_pdf_time)
            type_time_cond_likelihood.append(log_type_likelihoods)

            total_predictions += torch.sum(valid)

    #     # true next token at this last position is Y[:, -1]
    #     true_tokens = Y[:, -1].cpu()
    #     # compute top1/top5
    #     top1 += (probs.argmax(dim=1).cpu() == true_tokens).sum().item()
    #     top5 += sum([true_tokens[i].item() in probs[i].topk(5).indices.cpu().tolist() for i in range(len(true_tokens))])

    #     # true dt
    #     true_dt = (B - A)[:, -1].cpu().float()
    #     # mask invalids where target token is padding (0)
    #     valid_mask = true_tokens != 0
    #     for i in range(len(true_tokens)):
    #         if not valid_mask[i]:
    #             continue
    #         e = abs(pred_dt[i].cpu().item() - true_dt[i].item())
    #         time_abs_errs.append(e)
    #         time_sq_errs.append(e * e)

    # point_metrics = {
    #     "top1": top1 / total_predictions,
    #     "top5": top5 / total_predictions,
    #     "time_mae": float(x=np.mean(time_abs_errs)) if time_abs_errs else None,
    #     "time_rmse": float(np.sqrt(np.mean(time_sq_errs))) if time_sq_errs else None,
    # }
    likelihood_metrics = {
        "Marginal time-log-likelihood": torch.mean(torch.cat(time_likelihood)),
        "Conditional type-log-likelihood": torch.mean(torch.cat(type_time_cond_likelihood)),
        "Joint log-likelihood": torch.mean(torch.cat(joint_likelihood)),
    }
    print("Delphi metrics:", likelihood_metrics)
    return likelihood_metrics


def evaluate_hawkes(hawkes_state_path: Path | None, expansion_path: Path, device: str = "cpu", max_seqs: int = 2000):
    # Load sequences using ukb_loading
    sequences, sexes, num_event_types = load_ukb_sequences(expansion_path, limit_size=max_seqs)

    D = int(num_event_types)
    if hawkes_state_path is not None and hawkes_state_path.exists():
        # need to know D to instantiate
        hawkes = ExpKernelMVHawkesProcess(None, D).to(device)
        state = torch.load(str(hawkes_state_path), map_location=device)
        hawkes.load_state_dict(state)
    else:
        raise RuntimeError("Hawkes- Checkpoint not found!")

    hawkes.eval()

    DEVICE = torch.device("cuda:0")
    hawkes = hawkes.to(DEVICE)

    joint_likelihood = []
    time_likelihood = []
    type_time_cond_likelihood = []
    num_sequences = 0
    total_predictions = 0

    for ts_all in tqdm.tqdm(sequences):
        if len(ts_all) < 3:
            continue

        num_sequences += 1

        history = [
            ts_all[: idx - 1] for idx in range(2, len(ts_all))
        ]  # TODO need fix this bug, we cant run from empty sequences in batch mode..
        batch = BatchedMVEventData(mv_events=history)
        target_time = ts_all.time_points[2:].to(DEVICE)
        target_type = ts_all.event_types[2:].to(DEVICE)
        total_predictions += len(target_time)
        last_time = batch.max_time.to(DEVICE)
        batch = batch.to(DEVICE)

        valid_targets = target_type >= 11  # Here we dont have padding and no-event tokens at 0 and 1.

        with torch.no_grad():
            # Evaluate likelihoods and so on here.

            # First the per-type intensities and probabilities.
            type_intensity_at_t = hawkes.intensity(target_time, batch)
            type_cumulative_intensity_t = hawkes.cumulative_intensity(target_time, batch) - hawkes.cumulative_intensity(
                last_time, batch
            )

            # Joint density p(t,e). The likelihoods for all event types at the correct time (but not conditioned!!)
            type_PDF_at_t = hawkes.PDF(target_time, batch)

            # The likelihood of the correct event type at the correct time
            joint_likelihood_at_t = type_PDF_at_t[torch.arange(len(history)), target_type]
            joint_likelihood.append(joint_likelihood_at_t[valid_targets] / (365 * 80))

            # Distribution over types at time_point t.
            type_distribution_at_t = type_intensity_at_t / torch.sum(type_intensity_at_t, dim=1).unsqueeze(1)
            type_time_cond_likelihood.append(
                type_distribution_at_t[torch.arange(len(history)), target_type][valid_targets]
            )

            # Often TPPs are compared on next event prediction quality. We define the intensities and likelihoods for the next events.
            total_intensity_at_t = torch.sum(type_intensity_at_t, dim=1)
            total_cumulative_intensity_at_t = torch.sum(type_cumulative_intensity_t, dim=1)
            total_CDF_at_t = 1 - torch.exp(-total_cumulative_intensity_at_t)
            total_PDF_at_t = total_intensity_at_t * torch.exp(-total_cumulative_intensity_at_t)
            time_likelihood.append(total_PDF_at_t[valid_targets] / (365 * 80))

            # The expected value of the time_pdf. Often used for L2 and L1 metrics.
            # expected_t_given_history = None  # Hard to compute?? We can evaluate the PDF like above. But computing the mean requires sampling I think?
            # median_t_given_history = None  # We could probably compute the median of the pdf (and then use abs error as metric), using the inverse of the CDF, as computed in the inverse sampling...

            # Sample multiple continuatations.
            # samples = [hawkes.sample_inverse(ts, num_steps=1) for _ in range(100)]
            # sample_times = [s[1] for s in samples]
            # sample_type_dists = [s[2] for s in samples]

            # Estimate for the expected next event time.
            # average_time = torch.mean(sample_times, dim=0)

            # Estimate for the distribution over event types of next token.
            # average_sample_dist = torch.mean(sample_type_dists, dim=0)

            # We need to implement metrics here.

    metrics = {
        "Marginal time-likelihood": torch.mean(torch.log(torch.cat(time_likelihood))),
        "Conditional type-likelihood": torch.mean(torch.log(torch.cat(type_time_cond_likelihood))),
        "Joint likelihood": torch.mean(torch.log(torch.cat(joint_likelihood))),
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

    print("Evaluating Delphi...")
    evaluate_delphi(Path(args.delphi_ckpt), data_dir, device=args.device)

    print("Evaluating Hawkes...")
    evaluate_hawkes(Path(args.hawkes_state) if args.hawkes_state else None, expansion, device=args.device)


if __name__ == "__main__":
    main()
