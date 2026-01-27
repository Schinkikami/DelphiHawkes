"""
Evaluation utilities for temporal point process models.

This module provides functions for evaluating TPP models on test sequences,
computing various metrics like likelihoods, accuracy, and time prediction errors.
"""

from typing import Dict, List, Union, Any
import numpy as np
import torch
from tqdm import tqdm

from hawkes.tpps import TemporalPointProcess
from hawkes.event_utils import MVEventData, BatchedMVEventData


# Time unit conversion factor: TPP scaled time -> days
# TPP uses: t_scaled = t_days / 365.25 / 80, so t_days = t_scaled * 365.25 * 80
TIME_SCALE_FACTOR = 365.25 * 80  # ~29220 days = 80 years
LOG_TIME_SCALE = np.log(TIME_SCALE_FACTOR)


def evaluate_tpp(
    model: TemporalPointProcess,
    sequences: List[MVEventData],
    num_event_types: int,
    device: str = "cpu",
    compute_marginal_type: bool = False,
    min_disease_token: int = 11,
) -> Dict[str, Any]:
    """
    Evaluate a temporal point process model on sequences.

    This function computes comprehensive metrics for evaluating TPP models:
    - Joint log-likelihood p(t, m | H)
    - Marginal time log-likelihood p(t | H)
    - Conditional type log-likelihood p(m | t, H)
    - Marginal type log-likelihood p(m | H) (optional, expensive)
    - Top-1 and Top-5 accuracy
    - Time prediction RMSE and MAE

    Note on time units:
        TPP models use scaled time where 1.0 = 80 years (time_in_days / 365.25 / 80).
        All time-related metrics are converted to days for comparability with Delphi.

    Args:
        model: The TPP model to evaluate
        sequences: List of MVEventData sequences to evaluate on
        num_event_types: Number of event types (D)
        device: Device to run evaluation on
        compute_marginal_type: Whether to compute marginal type distribution p(m|H).
            This is expensive as it requires numerical integration. If False,
            uses conditional type distribution p(m|t,H) for accuracy metrics.
        min_disease_token: Minimum token index for disease events (default 11).
            Tokens below this are lifestyle/demographic and are filtered out.

    Returns:
        Dictionary with evaluation metrics (all values are floats):
            - marginal_time_ll: Marginal time log-likelihood (in days)
            - cond_type_ll: Conditional type log-likelihood p(m|t,H)
            - marginal_type_ll: Marginal type log-likelihood p(m|H) (if computed)
            - joint_ll: Joint log-likelihood (in days)
            - top1_accuracy: Top-1 type prediction accuracy
            - top5_accuracy: Top-5 type prediction accuracy
            - time_rmse_days: Time prediction RMSE in days
            - time_mae_days: Time prediction MAE in days
            - num_predictions: Total number of predictions made
    """
    model.eval()
    DEVICE = torch.device(device)
    model = model.to(DEVICE)

    # Accumulators for metrics
    joint_likelihood = []
    time_likelihood = []
    time_cond_type_likelihood = []
    marginal_type_likelihood = []
    top1_correct = []
    top5_correct = []
    time_squared_errors = []
    time_abs_errors = []

    total_predictions = 0

    for ts_all in tqdm(sequences, desc="Evaluating"):
        if len(ts_all) < 2:
            continue

        # For predicting event at index i (i >= 1), we use events 0..i-1 as history
        # Minimum history has 1 event (we skip empty histories to avoid edge cases)
        history = [ts_all[:idx] for idx in range(1, len(ts_all))]
        batch = BatchedMVEventData(mv_events=history)
        target_time: torch.Tensor = ts_all.time_points[1:].to(DEVICE)  # type: ignore
        target_type: torch.Tensor = ts_all.event_types[1:].to(DEVICE)  # type: ignore
        last_time = batch.max_time.to(DEVICE)
        batch = batch.to(DEVICE)

        # Filter out lifestyle tokens - only evaluate on disease tokens
        valid_targets = target_type >= min_disease_token

        if valid_targets.sum() == 0:
            continue

        total_predictions += valid_targets.sum().item()

        with torch.no_grad():
            # Per-type intensities and cumulative intensities
            type_intensity_at_t = model.intensity(target_time, batch)
            type_cumulative_intensity_t = model.cumulative_intensity(target_time, batch) - model.cumulative_intensity(
                last_time, batch
            )

            # Joint density p(t, m | H)
            type_PDF_at_t = model.PDF(target_time, batch)
            joint_likelihood_at_t = type_PDF_at_t[torch.arange(len(history)), target_type]
            joint_likelihood.append(joint_likelihood_at_t[valid_targets])

            # Conditional type distribution p(m | t, H)
            type_distribution_at_t = type_intensity_at_t / torch.sum(type_intensity_at_t, dim=1, keepdim=True)
            cond_type_probs = type_distribution_at_t[torch.arange(len(history)), target_type]
            time_cond_type_likelihood.append(cond_type_probs[valid_targets])

            # Marginal type distribution p(m | H) - expensive numerical integration
            if compute_marginal_type:
                marginal_type_probs = model.marginal_class_distribution(batch)  # (B, D)
                marginal_type_likelihood_at_target = marginal_type_probs[torch.arange(len(history)), target_type]
                marginal_type_likelihood.append(marginal_type_likelihood_at_target[valid_targets])

                # Top-1 and Top-5 accuracy based on marginal type probs
                pred_top1 = marginal_type_probs.argmax(dim=1)
                top1_correct.append((pred_top1 == target_type)[valid_targets])

                _, pred_top5 = marginal_type_probs.topk(5, dim=1)
                top5_hits = (pred_top5 == target_type.unsqueeze(1)).any(dim=1)
                top5_correct.append(top5_hits[valid_targets])
            else:
                # Use conditional type distribution for accuracy (cheaper)
                pred_top1 = type_distribution_at_t.argmax(dim=1)
                top1_correct.append((pred_top1 == target_type)[valid_targets])

                _, pred_top5 = type_distribution_at_t.topk(5, dim=1)
                top5_hits = (pred_top5 == target_type.unsqueeze(1)).any(dim=1)
                top5_correct.append(top5_hits[valid_targets])

            # Marginal time distribution p(t | H)
            total_intensity_at_t = torch.sum(type_intensity_at_t, dim=1)
            total_cumulative_intensity_at_t = torch.sum(type_cumulative_intensity_t, dim=1)
            total_PDF_at_t = total_intensity_at_t * torch.exp(-total_cumulative_intensity_at_t)
            time_likelihood.append(total_PDF_at_t[valid_targets])

            # Time prediction using median (inverse CDF at u=0.5)
            u_median = torch.full((len(history),), 0.5, device=DEVICE, dtype=torch.float64)
            try:
                median_time_pred = model.inverse_CDF(u_median, batch)
                time_errors = (median_time_pred - target_time)[valid_targets]
                time_squared_errors.append(time_errors**2)
                time_abs_errors.append(torch.abs(time_errors))
            except Exception:
                # Some models may not support inverse_CDF
                pass

    if total_predictions == 0:
        return {"error": "No valid predictions made"}

    # Compute metrics with proper unit conversions
    # Time likelihoods need Jacobian correction: log(p_days) = log(p_scaled) - log(TIME_SCALE_FACTOR)
    marginal_time_ll = torch.mean(torch.log(torch.cat(time_likelihood).clamp(min=1e-12))) - LOG_TIME_SCALE
    joint_ll = torch.mean(torch.log(torch.cat(joint_likelihood).clamp(min=1e-12))) - LOG_TIME_SCALE

    # Type likelihoods don't need conversion (dimensionless probabilities)
    cond_type_ll = torch.mean(torch.log(torch.cat(time_cond_type_likelihood).clamp(min=1e-12)))

    metrics = {
        "marginal_time_ll": float(marginal_time_ll),
        "cond_type_ll": float(cond_type_ll),
        "joint_ll": float(joint_ll),
        "top1_accuracy": float(torch.cat(top1_correct).float().mean()),
        "top5_accuracy": float(torch.cat(top5_correct).float().mean()),
        "num_predictions": total_predictions,
    }

    # Add marginal type likelihood if computed
    if compute_marginal_type and len(marginal_type_likelihood) > 0:
        metrics["marginal_type_ll"] = float(torch.mean(torch.log(torch.cat(marginal_type_likelihood).clamp(min=1e-12))))

    # Add time errors if computed (converted to days)
    if len(time_squared_errors) > 0:
        metrics["time_rmse_days"] = float(torch.sqrt(torch.mean(torch.cat(time_squared_errors))) * TIME_SCALE_FACTOR)
        metrics["time_mae_days"] = float(torch.mean(torch.cat(time_abs_errors)) * TIME_SCALE_FACTOR)

    return metrics


def format_metrics(metrics: Dict[str, Union[float, torch.Tensor]], prefix: str = "") -> str:
    """Format evaluation metrics as a readable string.

    Args:
        metrics: Dictionary of metrics from evaluate_tpp
        prefix: Optional prefix for each line

    Returns:
        Formatted string with metrics
    """
    lines = []
    metric_names = {
        "marginal_time_ll": "Marginal time log-likelihood",
        "cond_type_ll": "Conditional type log-likelihood p(m|t,H)",
        "marginal_type_ll": "Marginal type log-likelihood p(m|H)",
        "joint_ll": "Joint log-likelihood",
        "top1_accuracy": "Top-1 Accuracy",
        "top5_accuracy": "Top-5 Accuracy",
        "time_rmse_days": "Time RMSE (days)",
        "time_mae_days": "Time MAE (days)",
        "num_predictions": "Total predictions",
    }

    for key, display_name in metric_names.items():
        if key in metrics:
            value = metrics[key]
            if isinstance(value, (int, np.integer)):
                lines.append(f"{prefix}{display_name}: {value}")
            else:
                lines.append(f"{prefix}{display_name}: {value:.6f}")

    return "\n".join(lines)
