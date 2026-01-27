import argparse
from pathlib import Path
import sys
import numpy as np
import torch
import tqdm

from utils import get_batch, get_p2i


# Ensure local package imports in `hawkes/` resolve correctly when running as a script
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from hawkes.hawkes.tpps import TemporalPointProcess
from hawkes.hawkes.ukb_loading import load_ukb_sequences
from hawkes.hawkes.event_utils import BatchedMVEventData

# Delphi imports
from model import Delphi, DelphiConfig


def evaluate_delphi_old(ckpt_path: Path, data_dir: Path, device: str = "cpu", max_batches: int = 50):
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

            # P(E=e| T=t, H_t) => P(E=e| H_t), as we have constant (t independent) intensities.
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


def evaluate_delphi(ckpt_path: Path, data_dir: Path, sequences, device: str = "cpu"):
    """
    Evaluate Delphi model on the same sequences used for TPP evaluation.

    Delphi uses constant intensities (rates) per event type, so:
    - p(m|t, H) = p(m|H) (type distribution doesn't depend on time)
    - Time distribution is exponential with rate = sum of all type rates

    Time is in days for Delphi.

    This implementation uses get_batch() to match the preprocessing used during training
    (NoEvent token insertion, padding, etc.), but iterates through sequences in the same
    order as TPP evaluation.
    """
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

    # Load the raw test data for Delphi format
    test_path = data_dir / "test.bin"
    if not test_path.exists():
        print(f"Test data {test_path} not found. Skipping Delphi evaluation.")
        return None

    # Load data and prepare for get_batch
    test_data = np.memmap(str(test_path), dtype=np.uint32, mode="r").reshape(-1, 3)
    test_p2i = get_p2i(test_data)

    # Apply the Delphi token fix
    test_data = np.array(test_data, copy=True)
    test_data[:, 2] -= 1

    # Collect metrics
    joint_likelihood = []
    time_likelihood = []
    cond_type_likelihood = []
    top1_correct = []
    top5_correct = []
    time_squared_errors = []
    time_abs_errors = []

    total_predictions = 0

    # Build mapping from TPP sequences to test_p2i indices
    # We need to iterate through test_p2i in order and filter the same way as load_ukb_sequences
    male_token_delphi = 1  # After -1 offset
    female_token_delphi = 2

    # Identify indices that correspond to legal sequences (start with sex token)
    legal_indices = []
    for i in range(len(test_p2i)):
        start_idx = test_p2i[i][0]
        first_token = test_data[start_idx, 2]
        if first_token == male_token_delphi or first_token == female_token_delphi:
            # Check if non-empty after sex removal
            seq_len = test_p2i[i][1]
            if seq_len > 1:  # More than just sex token
                legal_indices.append(i)

    # Now legal_indices[i] corresponds to sequences[i]
    # We iterate through min(len(legal_indices), len(sequences))
    num_to_eval = min(len(legal_indices), len(sequences))

    # Process in batches of 16 for efficiency
    batch_size = 16
    for batch_start in tqdm.tqdm(range(0, num_to_eval, batch_size)):
        batch_end = min(batch_start + batch_size, num_to_eval)
        ix = np.array(legal_indices[batch_start:batch_end])

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

            # Valid targets: disease tokens >= 13 for Delphi
            valid = Y >= 13

            # Mask out the first valid disease prediction per sequence to match TPP evaluation
            # TPP requires at least 1 event in history, but Delphi can predict from sex token alone
            # has_true = valid.any(dim=1)
            # first_idx = valid.int().argmax(dim=1)
            # batch_idx_tensor = torch.arange(valid.size(0), device=valid.device)
            # valid[batch_idx_tensor[has_true], first_idx[has_true]] = False

            # Compute delta_t (time between consecutive events)
            delta_t = B - A

            # Eliminate NoEvent, Padding, and Lifestyle tokens from predictions
            logits[..., :13] = -torch.inf

            # Convert logits to rates: rate_i = exp(logit_i)
            rates = torch.exp(logits)
            rate_sum = rates.sum(dim=2)
            probs = rates / rate_sum.unsqueeze(-1)

            # p(m|H) = p(m|t,H) for constant intensities
            type_probs_at_target = torch.take_along_dim(probs, Y.unsqueeze(-1), dim=2).squeeze(2)
            log_type_ll = torch.log(type_probs_at_target.clamp(min=1e-12))

            # p(t|H) - exponential distribution with rate = sum of rates
            log_time_ll = torch.log(rate_sum.clamp(min=1e-12)) - rate_sum * delta_t

            # Joint log-likelihood
            log_joint_ll = log_type_ll + log_time_ll

            # Collect valid predictions
            if valid.sum() > 0:
                joint_likelihood.append(log_joint_ll[valid])
                time_likelihood.append(log_time_ll[valid])
                cond_type_likelihood.append(log_type_ll[valid])

                # Top-1 and Top-5 accuracy
                pred_top1 = probs.argmax(dim=2)
                top1_correct.append((pred_top1 == Y)[valid])

                _, pred_top5 = probs.topk(5, dim=2)
                top5_hits = (pred_top5 == Y.unsqueeze(-1)).any(dim=2)
                top5_correct.append(top5_hits[valid])

                # Time prediction: median of exponential distribution = ln(2) / rate
                pred_delta_t = torch.log(torch.tensor(2.0, device=device)) / rate_sum
                time_errors = (pred_delta_t - delta_t)[valid]
                time_squared_errors.append(time_errors**2)
                time_abs_errors.append(torch.abs(time_errors))

                total_predictions += valid.sum().item()

    if total_predictions == 0:
        print("No valid predictions found for Delphi evaluation.")
        return None

    # Compute final metrics (time is in days for Delphi)
    metrics = {
        "Marginal time-log-likelihood": torch.mean(torch.cat(time_likelihood)),
        "Conditional type-log-likelihood (p(m|t,H))": torch.mean(torch.cat(cond_type_likelihood)),
        "Marginal type-log-likelihood (p(m|H))": torch.mean(torch.cat(cond_type_likelihood)),  # Same for constant rates
        "Joint log-likelihood": torch.mean(torch.cat(joint_likelihood)),
        "Top-1 Accuracy": torch.cat(top1_correct).float().mean(),
        "Top-5 Accuracy": torch.cat(top5_correct).float().mean(),
        "Time RMSE (days)": torch.sqrt(torch.mean(torch.cat(time_squared_errors))),
        "Time MAE (days)": torch.mean(torch.cat(time_abs_errors)),
    }
    return metrics


def evaluate_tpp(model: TemporalPointProcess, sequences, D, device: str = "cpu", max_seqs: int = 2000):
    """
    Evaluate a temporal point process model on sequences.

    Note on time units:
        TPP models use scaled time where 1.0 = 80 years (time_in_days / 365.25 / 80).
        Delphi uses time in days.
        To make metrics comparable, we convert all time-related metrics to days:
        - Log-likelihoods of time densities: subtract log(365.25 * 80) (Jacobian correction)
        - Time errors (RMSE, MAE): multiply by 365.25 * 80
        Type-only distributions (conditional and marginal) don't need conversion.
    """
    # Time unit conversion factor: TPP scaled time -> days
    # TPP uses: t_scaled = t_days / 365.25 / 80, so t_days = t_scaled * 365.25 * 80
    TIME_SCALE_FACTOR = 365.25 * 80  # ~29220 days = 80 years
    LOG_TIME_SCALE = np.log(TIME_SCALE_FACTOR)

    model.eval()

    DEVICE = torch.device(device)
    model = model.to(DEVICE)

    joint_likelihood = []
    time_likelihood = []
    time_cond_type_likelihood = []
    marginal_type_likelihood = []
    top1_correct = []
    top5_correct = []
    time_squared_errors = []
    time_abs_errors = []

    num_sequences = 0
    total_predictions = 0

    for ts_all in tqdm.tqdm(sequences):
        if len(ts_all) < 2:
            continue

        num_sequences += 1

        # For predicting event at index i (i >= 1), we use events 0..i-1 as history
        # So for target_time[j] = ts_all.time_points[1+j], history[j] = ts_all[:1+j]
        # Minimum history has 1 event (we skip empty histories to avoid edge cases)
        # First token is also always sex, and we dont predict that (<11)
        history = [ts_all[:idx] for idx in range(1, len(ts_all))]
        batch = BatchedMVEventData(mv_events=history)
        target_time = ts_all.time_points[1:].to(DEVICE)
        target_type = ts_all.event_types[1:].to(DEVICE)
        total_predictions += len(target_time)
        last_time = batch.max_time.to(DEVICE)
        batch = batch.to(DEVICE)

        valid_targets = (
            target_type >= 11
        )  # Filter out lifestyle tokens (0-10 after ukb_loading shift). Disease tokens start at 11.

        with torch.no_grad():
            # Evaluate likelihoods and so on here.

            # First the per-type intensities and probabilities.
            type_intensity_at_t = model.intensity(target_time, batch)
            type_cumulative_intensity_t = model.cumulative_intensity(target_time, batch) - model.cumulative_intensity(
                last_time, batch
            )

            # Joint density p(t,e). The likelihoods for all event types at the correct time (but not conditioned!!)
            type_PDF_at_t = model.PDF(target_time, batch)

            # The likelihood of the correct event type at the correct time
            joint_likelihood_at_t = type_PDF_at_t[torch.arange(len(history)), target_type]
            joint_likelihood.append(joint_likelihood_at_t[valid_targets])

            # Distribution over types at time_point t, conditioned on time: p(m | t, H_t)
            type_distribution_at_t = type_intensity_at_t / torch.sum(type_intensity_at_t, dim=1).unsqueeze(1)
            time_cond_type_likelihood.append(
                type_distribution_at_t[torch.arange(len(history)), target_type][valid_targets]
            )

            # Marginal type likelihood p(m | H_t) - marginalizing over time
            # Uses numerical integration from base class
            marginal_type_probs = model.marginal_class_distribution(batch)  # (B, D)
            marginal_type_likelihood_at_target = marginal_type_probs[torch.arange(len(history)), target_type]
            marginal_type_likelihood.append(marginal_type_likelihood_at_target[valid_targets])

            # Compute top-1 and top-5 accuracy based on marginal type probabilities p(m | H_t)
            pred_top1 = marginal_type_probs.argmax(dim=1)  # (B,)
            top1_correct.append((pred_top1 == target_type)[valid_targets])

            _, pred_top5 = marginal_type_probs.topk(5, dim=1)  # (B, 5)
            top5_hits = (pred_top5 == target_type.unsqueeze(1)).any(dim=1)  # (B,)
            top5_correct.append(top5_hits[valid_targets])

            # Often TPPs are compared on next event prediction quality. We define the intensities and likelihoods for the next events.
            total_intensity_at_t = torch.sum(type_intensity_at_t, dim=1)
            total_cumulative_intensity_at_t = torch.sum(type_cumulative_intensity_t, dim=1)
            total_CDF_at_t = 1 - torch.exp(-total_cumulative_intensity_at_t)
            total_PDF_at_t = total_intensity_at_t * torch.exp(-total_cumulative_intensity_at_t)
            time_likelihood.append(total_PDF_at_t[valid_targets])

            # Compute median time prediction using inverse CDF at u=0.5
            # median_t = inverse_CDF(0.5) gives us the time where P(t_next < median_t | H_t) = 0.5
            u_median = torch.full((len(history),), 0.5, device=DEVICE, dtype=torch.float64)
            median_time_pred = model.inverse_CDF(u_median, batch)  # (B,)

            # Compute time prediction errors (RMSE and MAE)
            time_errors = (median_time_pred - target_time)[valid_targets]
            time_squared_errors.append(time_errors**2)
            time_abs_errors.append(torch.abs(time_errors))

    # Compute metrics with proper unit conversions
    # Time likelihoods need Jacobian correction: log(p_days) = log(p_scaled) - log(TIME_SCALE_FACTOR)
    # Time errors need scaling: error_days = error_scaled * TIME_SCALE_FACTOR

    marginal_time_ll = torch.mean(torch.log(torch.cat(time_likelihood))) - LOG_TIME_SCALE
    joint_ll = torch.mean(torch.log(torch.cat(joint_likelihood))) - LOG_TIME_SCALE

    # Type likelihoods don't need conversion (dimensionless probabilities)
    cond_type_ll = torch.mean(torch.log(torch.cat(time_cond_type_likelihood)))
    marginal_type_ll = torch.mean(torch.log(torch.cat(marginal_type_likelihood)))

    # Time errors converted to days
    time_rmse = torch.sqrt(torch.mean(torch.cat(time_squared_errors))) * TIME_SCALE_FACTOR
    time_mae = torch.mean(torch.cat(time_abs_errors)) * TIME_SCALE_FACTOR

    metrics = {
        "Marginal time-log-likelihood": marginal_time_ll,
        "Conditional type-log-likelihood (p(m|t,H))": cond_type_ll,
        "Marginal type-log-likelihood (p(m|H))": marginal_type_ll,
        "Joint log-likelihood": joint_ll,
        "Top-1 Accuracy": torch.cat(top1_correct).float().mean(),
        "Top-5 Accuracy": torch.cat(top5_correct).float().mean(),
        "Time RMSE (days)": time_rmse,
        "Time MAE (days)": time_mae,
    }
    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate temporal point process models on UKB data.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate ExpKernelHawkes model
  python eval_compare.py --model exp_hawkes --weights models/new_hawkes.pth

  # Evaluate SplineBaselineExpKernelHawkes model  
  python eval_compare.py --model spline_hawkes --weights models/new_spline_hawkes.pth --num_knots 5 --delta_t 0.3

  # Evaluate Poisson model
  python eval_compare.py --model poisson --weights models/new_poisson.pth

  # Evaluate Delphi model
  python eval_compare.py --model delphi --weights models/ckpt.pt
""",
    )

    # Model selection
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=[
            "exp_hawkes",
            "spline_hawkes",
            "numerical_spline_hawkes",
            "poisson",
            "inhomogeneous_poisson",
            "spline_poisson",
            "delphi",
            "delphi_old",
        ],
        help="Type of model to evaluate",
    )
    parser.add_argument("--weights", type=str, required=True, help="Path to model weights/checkpoint")

    # Data arguments
    parser.add_argument("--data_dir", type=str, default="data/ukb_simulated_data", help="Path to UKB data directory")
    parser.add_argument("--data_file", type=str, default="test.bin", help="Path to data bin file")
    parser.add_argument("--limit_sequences", type=int, default=int(1e8), help="Max number of sequences to load")

    # Device
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to run evaluation on")

    # Model-specific arguments (for spline models)
    parser.add_argument("--num_knots", type=int, default=5, help="Number of knots for spline models")
    parser.add_argument("--delta_t", type=float, default=0.3, help="Knot spacing (delta_t) for spline models")

    args = parser.parse_args()

    # Load data
    data_dir = Path(args.data_dir)
    data_file = Path(args.data_file)
    data_file = data_dir / data_file

    print(f"Loading sequences from {data_file}...")
    sequences, sexes, num_event_types = load_ukb_sequences(data_file, limit_size=args.limit_sequences)
    D = num_event_types
    print(f"Loaded {len(sequences)} sequences with {D} event types")

    # Evaluate based on model type
    if args.model == "delphi":
        print(f"Evaluating Delphi model from {args.weights}...")
        metrics = evaluate_delphi(Path(args.weights), data_dir, sequences, device=args.device)
    elif args.model == "delphi_old":
        print(f"Evaluating Delphi model from {args.weights} using old evaluation...")
        metrics = evaluate_delphi_old(Path(args.weights), data_dir, device=args.device)
    else:
        # Load TPP model based on type
        model = load_tpp_model(args.model, D, args)

        # Load weights
        print(f"Loading weights from {args.weights}...")
        state = torch.load(args.weights, map_location=args.device)
        model.load_state_dict(state)

        print(f"Evaluating {args.model} model...")
        metrics = evaluate_tpp(model, sequences, D, device=args.device)

    # Print results
    if metrics:
        print("\n" + "=" * 50)
        print("Evaluation Results:")
        print("=" * 50)
        for key, value in metrics.items():
            if isinstance(value, torch.Tensor):
                print(f"  {key}: {value.item():.6f}")
            else:
                print(f"  {key}: {value:.6f}" if value is not None else f"  {key}: N/A")


def load_tpp_model(model_type: str, D: int, args) -> TemporalPointProcess:
    """Load a TPP model based on the model type."""
    from hawkes.hawkes.hawkes_tpp import (
        ExpKernelHawkesProcess,
        SplineBaselineExpKernelHawkesProcess,
        NumericalSplineBaselineExpKernelHawkesProcess,
        SoftplusConstExpIHawkesProcess,
        SoftplusSplineExpIHawkesProcess,
    )
    from hawkes.hawkes.baseline_tpps import (
        PoissonProcess,
        ConditionalInhomogeniousPoissonProcess,
        SplinePoissonProcess,
    )

    if model_type == "exp_hawkes":
        return ExpKernelHawkesProcess(D)

    elif model_type == "spline_hawkes":
        return SplineBaselineExpKernelHawkesProcess(D, num_knots=args.num_knots, delta_t=args.delta_t)

    elif model_type == "numerical_spline_hawkes":
        return NumericalSplineBaselineExpKernelHawkesProcess(D, num_knots=args.num_knots, delta_t=args.delta_t)

    elif model_type == "soft_plus_const_exp_ihawkes":
        return SoftplusConstExpIHawkesProcess(D, baseline_params=None, kernel_params=None)

    elif model_type == "soft_plus_spline_exp_ihawkes":
        return SoftplusSplineExpIHawkesProcess(
            D, num_knots=args.num_knots, delta_t=args.delta_t, baseline_params=None, kernel_params=None
        )

    elif model_type == "poisson":
        return PoissonProcess(D=D)

    elif model_type == "inhomogeneous_poisson":
        return ConditionalInhomogeniousPoissonProcess(D=D)

    elif model_type == "spline_poisson":
        return SplinePoissonProcess(D, num_knots=args.num_knots, delta_t=args.delta_t)

    else:
        raise ValueError(f"Unknown model type: {model_type}")


if __name__ == "__main__":
    main()
