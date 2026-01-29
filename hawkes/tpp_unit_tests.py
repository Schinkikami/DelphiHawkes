"""
Unit tests for Temporal Point Process models.

Tests various TPP models for correctness of intensity, cumulative intensity,
PDF, CDF, likelihood, sampling, and marginal distributions.
"""

import sys
import lovely_tensors as lt

lt.monkey_patch()
sys.path.append("..")

import torch
from hawkes.event_utils import BatchedMVEventData, MVEventData
from hawkes.hawkes_tpp import (
    ExpKernelHawkesProcess,
    LinearBaselineExpKernelHawkesProcess,
    SplineBaselineExpKernelHawkesProcess,
)
from hawkes.baseline_tpps import SplinePoissonProcess, PoissonProcess, ConditionalInhomogeniousPoissonProcess

# Test configuration constants
MODEL_TYPE = "spline_exp_hawkes"
SPLINE_K = 100
SPLINE_DELTA_T = 0.1
D = 2  # Number of event types
SEED = 32

# Tolerance settings for numerical tests
TIGHT_TOLERANCE = 1e-4  # For exact matches (e.g., edge cases)
STANDARD_TOLERANCE = 1e-2  # For numerical differentiation/integration
SAMPLING_TOLERANCE = 0.05  # For stochastic sampling tests

# Numerical integration settings
NUM_INTEGRATION_STEPS = 1000
NUM_SAMPLING_ITERATIONS = 1000

# Small epsilon to avoid discontinuities at event times
EPSILON = 1e-4


def create_model(model_type: str, d: int, seed: int):
    """Factory function to create TPP models."""
    if model_type == "poisson":
        print("Creating Poisson Process model...")
        return PoissonProcess(D=d, seed=seed)

    elif model_type == "inhomogeneous_poisson":
        print("Creating Inhomogeneous Poisson Process model...")
        return ConditionalInhomogeniousPoissonProcess(D=d, seed=seed)

    elif model_type == "spline_poisson":
        return SplinePoissonProcess(d, SPLINE_K, SPLINE_DELTA_T, seed=seed)

    elif model_type == "hawkes":
        return ExpKernelHawkesProcess(D=d, seed=seed)

    elif model_type == "linear_exp_hawkes":
        print("Creating linear baseline exponential kernel Hawkes Process model...")
        return LinearBaselineExpKernelHawkesProcess(D=d, seed=seed)

    elif model_type == "spline_exp_hawkes":
        print("Creating spline baseline exponential kernel Hawkes Process model...")
        return SplineBaselineExpKernelHawkesProcess(D=d, num_knots=SPLINE_K, delta_t=SPLINE_DELTA_T, seed=seed)

    else:
        raise ValueError(f"Unknown model type: {model_type}.")


def create_test_data():
    """
    Create test data including empty, single, and batch sequences.

    Returns:
        dict: Dictionary containing various test sequences and batches
    """
    # Empty sequence
    emp_seq = MVEventData(torch.tensor([]), torch.tensor([], dtype=int))

    # First sequence with 4 events
    seq1 = MVEventData(torch.tensor([0.2, 0.8, 2.0, 2.3], dtype=float), torch.tensor([0, 0, 1, 0], dtype=int))

    # Second sequence with 5 events (for batch testing)
    seq2 = MVEventData(torch.tensor([0.1, 0.5, 1.2, 1.8, 2.5], dtype=float), torch.tensor([1, 0, 1, 0, 1], dtype=int))

    # Padded version of first sequence
    padded_seq1 = MVEventData(
        torch.tensor([0.2, 0.8, 2.0, 2.3, 2.5, 2.8], dtype=float), torch.tensor([0, 0, 1, 0, -1, -1], dtype=int)
    )

    return {
        "empty_seq": emp_seq,
        "seq1": seq1,
        "seq2": seq2,
        "padded_seq1": padded_seq1,
        "batch_single": BatchedMVEventData([seq1.time_points], [seq1.event_types]),
        "batch_multi": BatchedMVEventData([seq1.time_points, seq2.time_points], [seq1.event_types, seq2.event_types]),
        "empty_batch": BatchedMVEventData([emp_seq.time_points], [emp_seq.event_types]),
        "padded_batch": BatchedMVEventData([padded_seq1.time_points], [padded_seq1.event_types]),
    }


def test_intensity_correctness(model, data: dict, d: int):
    """Test correctness of intensity computation."""
    batch_single = data["batch_single"]
    batch_multi = data["batch_multi"]
    padded_batch = data["padded_batch"]
    empty_batch = data["empty_batch"]
    seq1 = data["seq1"]
    seq2 = data["seq2"]

    # Test single batch - intensity after sequence
    after_sequence_intensity = model.intensity(torch.tensor([4.0]), batch_single)
    assert after_sequence_intensity.shape == (1, d), "Intensity shape mismatch after sequence."
    assert torch.all(after_sequence_intensity > 0), "Intensity should be positive."

    # Test multi-batch - intensity after sequences
    after_multi_intensity = model.intensity(torch.tensor([4.0, 3.0]), batch_multi)
    assert after_multi_intensity.shape == (2, d), "Intensity shape mismatch for multi-batch."
    assert torch.all(after_multi_intensity > 0), "Intensity should be positive for multi-batch."

    # Verify batch consistency: batched call should equal individual calls combined
    batch1_individual = BatchedMVEventData([seq1.time_points], [seq1.event_types])
    batch2_individual = BatchedMVEventData([seq2.time_points], [seq2.event_types])
    intensity1 = model.intensity(torch.tensor([4.0]), batch1_individual)
    intensity2 = model.intensity(torch.tensor([3.0]), batch2_individual)
    intensities_combined = torch.cat([intensity1, intensity2], dim=0)
    assert torch.allclose(after_multi_intensity.float(), intensities_combined.float(), atol=TIGHT_TOLERANCE), (
        "Batched intensity should equal individual intensities combined."
    )

    # Padding with -1 events should not affect intensity
    after_padded_sequence_intensity = model.intensity(torch.tensor([4.0]), padded_batch)
    assert torch.allclose(
        after_sequence_intensity.float(), after_padded_sequence_intensity.float(), atol=TIGHT_TOLERANCE
    ), "Intensity should be the same with padded events."

    # Test empty sequence
    empty_intensity = model.intensity(torch.tensor([1.0]), empty_batch)
    assert empty_intensity.shape == (1, d), "Intensity shape mismatch for empty sequence."
    assert torch.all(empty_intensity > 0), "Intensity should be positive for empty sequence."

    # Intensity exactly at event should match intensity just before it
    intensity_exactly_at_event = model.intensity(torch.tensor([2.3]), batch_single)
    intensity_exactly_at_event_last_cut = model.intensity(torch.tensor([2.3]), batch_single[:, :-1])
    assert torch.allclose(
        intensity_exactly_at_event.float(), intensity_exactly_at_event_last_cut.float(), atol=TIGHT_TOLERANCE
    ), "Intensity exactly at event should match intensity after last event."

    # Intensity between events should match intensity after last event before that time
    intensity_inbetween = model.intensity(torch.tensor([2.2]), batch_single)
    intensity_last_cut = model.intensity(torch.tensor([2.2]), batch_single[:, :-1])
    assert torch.allclose(intensity_inbetween.float(), intensity_last_cut.float(), atol=TIGHT_TOLERANCE), (
        "Intensity between events should match intensity after last event."
    )

    # Intensity at t=0 should match baseline
    intensity_at_0 = model.intensity(torch.tensor([0.0]), batch_single)
    intensity_at_0_empty = model.intensity(torch.tensor([0.0]), empty_batch)
    assert torch.allclose(intensity_at_0.float(), intensity_at_0_empty.float(), atol=TIGHT_TOLERANCE), (
        "Intensity at t=0 should match baseline intensity."
    )


def check_cumulative_intensity_correctness(model, data: dict, d: int):
    """Test correctness of cumulative intensity computation."""
    batch_single = data["batch_single"]
    batch_multi = data["batch_multi"]
    padded_batch = data["padded_batch"]
    empty_batch = data["empty_batch"]
    seq1 = data["seq1"]
    seq2 = data["seq2"]

    # Test single batch
    after_sequence_cum_intensity = model.cumulative_intensity(torch.tensor([4.0]), batch_single)
    assert after_sequence_cum_intensity.shape == (1, d), "Cumulative Intensity shape mismatch after sequence."
    assert torch.all(after_sequence_cum_intensity > 0), "Cumulative Intensity should be positive."

    # Test multi-batch
    after_multi_cum_intensity = model.cumulative_intensity(torch.tensor([4.0, 3.0]), batch_multi)
    assert after_multi_cum_intensity.shape == (2, d), "Cumulative Intensity shape mismatch for multi-batch."
    assert torch.all(after_multi_cum_intensity > 0), "Cumulative Intensity should be positive for multi-batch."

    # Verify batch consistency: batched call should equal individual calls combined
    batch1_individual = BatchedMVEventData([seq1.time_points], [seq1.event_types])
    batch2_individual = BatchedMVEventData([seq2.time_points], [seq2.event_types])
    cum_intensity1 = model.cumulative_intensity(torch.tensor([4.0]), batch1_individual)
    cum_intensity2 = model.cumulative_intensity(torch.tensor([3.0]), batch2_individual)
    cum_intensities_combined = torch.cat([cum_intensity1, cum_intensity2], dim=0)
    assert torch.allclose(after_multi_cum_intensity.float(), cum_intensities_combined.float(), atol=TIGHT_TOLERANCE), (
        "Batched cumulative intensity should equal individual cumulative intensities combined."
    )

    # Padding with -1 events should not affect cumulative intensity
    after_padded_sequence_cum_intensity = model.cumulative_intensity(torch.tensor([4.0]), padded_batch)
    assert torch.allclose(
        after_sequence_cum_intensity.float(), after_padded_sequence_cum_intensity.float(), atol=TIGHT_TOLERANCE
    ), "Cumulative Intensity should be the same with padded events."

    # Test empty sequence
    empty_cum_intensity = model.cumulative_intensity(torch.tensor([1.0]), empty_batch)
    assert empty_cum_intensity.shape == (1, d), "Cumulative Intensity shape mismatch for empty sequence."
    assert torch.all(empty_cum_intensity > 0), "Cumulative Intensity should be positive for empty sequence."

    # Cumulative intensity exactly at event
    cum_intensity_exactly_at_event = model.cumulative_intensity(torch.tensor([2.3]), batch_single)
    cum_intensity_exactly_at_event_last_cut = model.cumulative_intensity(torch.tensor([2.3]), batch_single[:, :-1])
    assert torch.allclose(
        cum_intensity_exactly_at_event.float(), cum_intensity_exactly_at_event_last_cut.float(), atol=TIGHT_TOLERANCE
    ), "Cumulative Intensity exactly at event should match cumulative Intensity after last event."

    # Cumulative intensity between events
    cum_intensity_inbetween = model.cumulative_intensity(torch.tensor([2.2]), batch_single)
    cum_intensity_last_cut = model.cumulative_intensity(torch.tensor([2.2]), batch_single[:, :-1])
    assert torch.allclose(cum_intensity_inbetween.float(), cum_intensity_last_cut.float(), atol=TIGHT_TOLERANCE), (
        "Cumulative Intensity between events should match cumulative intensity after last event with last event cut."
    )


def check_intensity_cumulative_intensity_relation(model, data: dict):
    """Check relation between intensity and cumulative intensity via numerical differentiation."""
    batch_single = data["batch_single"]
    seq1 = data["seq1"]

    # Start slightly after the last event time to avoid discontinuities at event times
    last_time = seq1.time_points.max().item()
    times = torch.linspace(last_time + EPSILON, 4.0, steps=NUM_INTEGRATION_STEPS)
    delta_t = times[1] - times[0]

    cum_intensities = torch.stack([model.cumulative_intensity(t.unsqueeze(0), batch_single) for t in times]).squeeze()
    intensities_numerical = (cum_intensities[1:] - cum_intensities[:-1]) / delta_t
    intensities_model = torch.stack([model.intensity(t.unsqueeze(0), batch_single) for t in times[0:-1]]).squeeze()

    assert torch.allclose(intensities_numerical.float(), intensities_model.float(), atol=STANDARD_TOLERANCE), (
        "Numerical derivative of cumulative intensity should match intensity."
    )


def check_pdf_correctness(model, data: dict, d: int):
    """Test correctness of PDF computation."""
    batch_single = data["batch_single"]
    batch_multi = data["batch_multi"]
    seq1 = data["seq1"]
    seq2 = data["seq2"]

    # Check PDF shape and non-negativity
    pdf_at_time = model.PDF(torch.tensor([2.5]), batch_single)
    assert pdf_at_time.shape == (1, d), "PDF shape mismatch."
    assert torch.all(pdf_at_time >= 0), "PDF should be non-negative."

    # Test multi-batch PDF (need times after each sequence's last event)
    pdf_multi = model.PDF(torch.tensor([3.0, 3.0]), batch_multi)
    assert pdf_multi.shape == (2, d), "PDF shape mismatch for multi-batch."
    assert torch.all(pdf_multi >= 0), "PDF should be non-negative for multi-batch."

    # Verify batch consistency: batched call should equal individual calls combined
    batch1_individual = BatchedMVEventData([seq1.time_points], [seq1.event_types])
    batch2_individual = BatchedMVEventData([seq2.time_points], [seq2.event_types])
    pdf1 = model.PDF(torch.tensor([3.0]), batch1_individual)
    pdf2 = model.PDF(torch.tensor([3.0]), batch2_individual)
    pdfs_combined = torch.cat([pdf1, pdf2], dim=0)
    assert torch.allclose(pdf_multi.float(), pdfs_combined.float(), atol=TIGHT_TOLERANCE), (
        "Batched PDF should equal individual PDFs combined."
    )

    # Check relation between PDF and intensity and cumulative intensity
    intensity_at_time = model.intensity(torch.tensor([2.5]), batch_single)
    cum_intensity_at_time = model.cumulative_intensity(torch.tensor([2.5]), batch_single) - model.cumulative_intensity(
        torch.tensor([2.3]), batch_single
    )

    pdf_expected = intensity_at_time * torch.exp(-cum_intensity_at_time.sum())
    assert torch.allclose(pdf_at_time.float(), pdf_expected.float(), atol=TIGHT_TOLERANCE), (
        "PDF should equal intensity times exp(-cumulative intensity)."
    )


def check_cdf_correctness(model, data: dict):
    """Test correctness of CDF computation."""
    batch_single = data["batch_single"]
    batch_multi = data["batch_multi"]
    seq1 = data["seq1"]
    seq2 = data["seq2"]

    # Check CDF shape and non-negativity
    cdf_at_time = model.CDF(torch.tensor([2.5]), batch_single)
    assert cdf_at_time.shape == (1,), "CDF shape mismatch."
    assert torch.all(cdf_at_time >= 0), "CDF should be non-negative."

    # Test multi-batch CDF (need times after each sequence's last event)
    cdf_multi = model.CDF(torch.tensor([3.0, 3.0]), batch_multi)
    assert cdf_multi.shape == (2,), "CDF shape mismatch for multi-batch."
    assert torch.all(cdf_multi >= 0), "CDF should be non-negative for multi-batch."

    # Verify batch consistency: batched call should equal individual calls combined
    batch1_individual = BatchedMVEventData([seq1.time_points], [seq1.event_types])
    batch2_individual = BatchedMVEventData([seq2.time_points], [seq2.event_types])
    cdf1 = model.CDF(torch.tensor([3.0]), batch1_individual)
    cdf2 = model.CDF(torch.tensor([3.0]), batch2_individual)
    cdfs_combined = torch.cat([cdf1, cdf2], dim=0)
    assert torch.allclose(cdf_multi.float(), cdfs_combined.float(), atol=TIGHT_TOLERANCE), (
        "Batched CDF should equal individual CDFs combined."
    )

    # Check relation between CDF and cumulative intensity
    cum_intensity_at_time = model.cumulative_intensity(torch.tensor([2.5]), batch_single) - model.cumulative_intensity(
        torch.tensor([2.3]), batch_single
    )
    cdf_expected = 1 - torch.exp(-cum_intensity_at_time.sum())
    assert torch.allclose(cdf_at_time.float(), cdf_expected.float(), atol=TIGHT_TOLERANCE), (
        "CDF should equal 1 - exp(-cumulative intensity)."
    )

    # CDF should be near zero just after last event
    cdf_at_last_point = model.CDF(torch.tensor([2.3 + EPSILON]), batch_single)
    assert torch.allclose(cdf_at_last_point, torch.zeros_like(cdf_at_last_point), atol=STANDARD_TOLERANCE), (
        "CDF just after last event time should be near zero."
    )


def check_pdf_cdf_correctness(model, data: dict):
    """Check correctness of PDF and CDF via numerical integration."""
    batch_single = data["batch_single"]
    seq1 = data["seq1"]

    last_time = seq1.time_points.max().item()
    start_time = last_time + EPSILON
    end_time = 2.5

    times = torch.linspace(start_time, end_time, steps=NUM_INTEGRATION_STEPS)
    integrants = [model.PDF(torch.tensor([t]), batch_single) for t in times]
    integrants = torch.stack(integrants).squeeze()
    integral = torch.trapezoid(integrants, times, dim=0).sum()

    print(f"Integral over PDF from {start_time:.3f} to {end_time}: {integral.item():.6f}")

    cdf = model.CDF(torch.tensor([end_time]), batch_single) - model.CDF(torch.tensor([start_time]), batch_single)
    assert torch.allclose(integral.float(), cdf.sum().float(), atol=STANDARD_TOLERANCE), (
        "Integral of PDF should match CDF difference."
    )


def check_likelihood_correctness(model, data: dict):
    """Test likelihood computation correctness."""
    batch_single = data["batch_single"]
    batch_multi = data["batch_multi"]
    seq1 = data["seq1"]
    seq2 = data["seq2"]

    # Test single batch likelihood
    likelihood = model.likelihood(batch_single, torch.tensor([4.0]), log=False)
    assert likelihood.shape == (1,), "Likelihood shape mismatch."
    assert torch.all(likelihood > 0), "Likelihood should be positive."

    # Test multi-batch likelihood
    likelihood_multi = model.likelihood(batch_multi, torch.tensor([4.0, 3.0]), log=False)
    assert likelihood_multi.shape == (2,), "Likelihood shape mismatch for multi-batch."
    assert torch.all(likelihood_multi > 0), "Likelihood should be positive for multi-batch."

    # Verify batch consistency: batched call should equal individual calls combined
    batch1_individual = BatchedMVEventData([seq1.time_points], [seq1.event_types])
    batch2_individual = BatchedMVEventData([seq2.time_points], [seq2.event_types])
    likelihood1 = model.likelihood(batch1_individual, torch.tensor([4.0]), log=False)
    likelihood2 = model.likelihood(batch2_individual, torch.tensor([3.0]), log=False)
    likelihoods_combined = torch.cat([likelihood1, likelihood2], dim=0)
    assert torch.allclose(likelihood_multi.float(), likelihoods_combined.float(), atol=TIGHT_TOLERANCE), (
        "Batched likelihood should equal individual likelihoods combined."
    )

    # Test log likelihood
    log_likelihood = model.likelihood(batch_single, torch.tensor([4.0]), log=True)
    assert log_likelihood.shape == (1,), "Log Likelihood shape mismatch."
    assert torch.all(log_likelihood <= 0), "Log Likelihood should be non-positive."

    assert torch.allclose(torch.exp(log_likelihood).float(), likelihood.float(), atol=TIGHT_TOLERANCE), (
        "Exponentiated log likelihood should match likelihood."
    )

    # Check relation between likelihood and positive likelihood
    pos_likelihood = model.positive_likelihood(batch_single, log=False)
    cum_intensity_end = model.cumulative_intensity(torch.tensor([4.0]), batch_single).sum()
    likelihood_expected = pos_likelihood * torch.exp(-cum_intensity_end)
    assert torch.allclose(likelihood.float(), likelihood_expected.float(), atol=TIGHT_TOLERANCE), (
        "Likelihood should equal positive likelihood times exp(-cumulative intensity at end)."
    )

    # Check correctness of positive likelihood
    pos_likelihood = model.positive_likelihood(batch_single, log=True)

    # Stack the sequence on itself with increasing lengths
    stacked_batch = BatchedMVEventData(
        time_points=[seq1.time_points[:i] for i in range(len(seq1.time_points))],
        event_types=[seq1.event_types[:i] for i in range(len(seq1.event_types))],
    )

    stacked_intensities = model.intensity(seq1.time_points, stacked_batch)
    intensity_at_points = stacked_intensities[torch.arange(len(seq1.time_points)), seq1.event_types]
    log_intensity_at_points = intensity_at_points.log()

    assert torch.allclose(pos_likelihood.float(), log_intensity_at_points.sum().float(), atol=TIGHT_TOLERANCE), (
        "Positive log-likelihood should equal sum of log intensities at event times."
    )


def check_sampling(model, data: dict):
    """Test sampling functionality."""
    seq1 = data["seq1"]

    num_steps = 10
    samples, tstar, dstar = model.sample(seq1, num_steps=num_steps)

    assert (torch.tensor(tstar) >= seq1.time_points.max()).all(), "Sampled times should be after last event time."
    assert len(samples) == len(seq1) + num_steps, "Number of sampled events should match."


def check_inverse_cdf(model, data: dict):
    """Test inverse CDF computation."""
    batch_single = data["batch_single"]
    batch_multi = data["batch_multi"]
    seq1 = data["seq1"]
    seq2 = data["seq2"]

    # Test at various quantiles
    u_values = torch.tensor([0.01, 0.5, 0.9, 0.99])
    for u in u_values:
        t_inv = model.inverse_CDF(u.unsqueeze(0), batch_single)
        cdf_at_t_inv = model.CDF(t_inv, batch_single)
        assert torch.allclose(cdf_at_t_inv.float(), u.unsqueeze(0).float(), atol=STANDARD_TOLERANCE), (
            f"CDF at inverse CDF({u.item()}) should equal {u.item()}."
        )

    # Test multi-batch inverse CDF
    u_multi = torch.tensor([0.5, 0.7])
    t_inv_multi = model.inverse_CDF(u_multi, batch_multi)
    assert t_inv_multi.shape == (2,), "Inverse CDF shape mismatch for multi-batch."

    # Verify batch consistency: batched call should equal individual calls combined
    batch1_individual = BatchedMVEventData([seq1.time_points], [seq1.event_types])
    batch2_individual = BatchedMVEventData([seq2.time_points], [seq2.event_types])
    t_inv1 = model.inverse_CDF(torch.tensor([0.5]), batch1_individual)
    t_inv2 = model.inverse_CDF(torch.tensor([0.7]), batch2_individual)
    t_invs_combined = torch.cat([t_inv1, t_inv2], dim=0)
    assert torch.allclose(t_inv_multi.float(), t_invs_combined.float(), atol=STANDARD_TOLERANCE), (
        "Batched inverse CDF should equal individual inverse CDFs combined."
    )

    # Test inverse CDF at edge cases
    t_inv_0 = model.inverse_CDF(torch.tensor([0.0]), batch_single)
    cdf_at_t_inv_0 = model.CDF(t_inv_0, batch_single)
    assert torch.allclose(cdf_at_t_inv_0.float(), torch.tensor([0.0]).float(), atol=TIGHT_TOLERANCE), (
        "CDF at inverse CDF(0) should be 0."
    )

    t_inv_1 = model.inverse_CDF(torch.tensor([1.0]), batch_single)
    cdf_at_t_inv_1 = model.CDF(t_inv_1, batch_single)
    assert torch.allclose(cdf_at_t_inv_1.float(), torch.tensor([1.0]).float(), atol=TIGHT_TOLERANCE), (
        "CDF at inverse CDF(1) should be 1."
    )


def check_class_and_time_marginals(model, data: dict):
    """Test class and time marginal distributions via sampling."""
    batch_single = data["batch_single"]
    seq1 = data["seq1"]

    # Find the 0.999 quantile time
    t_999 = model.inverse_CDF(torch.tensor([0.999]), batch_single)[0].item()

    xs = torch.linspace((batch_single.max_time + EPSILON).item(), t_999, steps=100)

    # Compute class marginals up to t_999
    joint_time_type_pdf = []
    for t in xs:
        pdf = model.PDF(t.unsqueeze(0), batch_single)  # Joint PDF over time and type
        joint_time_type_pdf.append(pdf)

    joint_time_type_pdf = torch.stack(joint_time_type_pdf).squeeze()  # Shape: (num_times, D)
    type_marginals = torch.trapezoid(joint_time_type_pdf, xs, dim=0)  # Shape: (D,)
    assert 0.99 <= type_marginals.sum() <= 1.0, "Type marginals should sum to approximately 1.0."
    type_marginals = type_marginals / type_marginals.sum()  # Normalize to sum to 1

    time_marginal = joint_time_type_pdf.sum(dim=1)  # Is a density. Does not sum to 1.

    time_cdf = torch.cumulative_trapezoid(time_marginal, xs, dim=0)
    assert torch.all(type_marginals >= 0), "Type marginals should be non-negative."
    assert torch.all(time_cdf >= 0) and torch.all(time_cdf <= 1), "Time CDF should be between 0 and 1."

    # Sample continuations and compute empirical marginals
    tstar, dstar = [], []
    for i in range(NUM_SAMPLING_ITERATIONS):
        _, tt, dd = model.sample(seq1, num_steps=1, rng=i)
        tstar.append(tt[0])
        dstar.append(dd[0])

    tstar = torch.tensor(tstar)
    dstar = torch.stack(dstar)
    empirical_type_marginals = torch.mean(dstar, dim=0)
    empirical_time_cdf = torch.tensor([(tstar <= t).float().mean() for t in xs])

    assert torch.allclose(type_marginals.float(), empirical_type_marginals.float(), atol=SAMPLING_TOLERANCE), (
        "Empirical type marginals should match computed type marginals."
    )

    assert torch.allclose(time_cdf.float(), empirical_time_cdf[1:].float(), atol=SAMPLING_TOLERANCE), (
        "Empirical time CDF should match computed time CDF."
    )


def run_tests_for_model(model_type: str):
    """Run all unit tests for a specific model."""
    print(f"\n{'=' * 60}")
    print(f"Running TPP Unit Tests for {model_type}")
    print(f"{'=' * 60}\n")

    # Create model and test data
    model = create_model(model_type, D, SEED)
    data = create_test_data()

    with torch.no_grad():
        print("Testing intensity correctness...")
        test_intensity_correctness(model, data, D)
        print("✓ Intensity tests passed")

        print("Testing cumulative intensity correctness...")
        check_cumulative_intensity_correctness(model, data, D)
        print("✓ Cumulative intensity tests passed")

        print("Testing intensity-cumulative intensity relation...")
        check_intensity_cumulative_intensity_relation(model, data)
        print("✓ Intensity-cumulative intensity relation tests passed")

        print("Testing PDF correctness...")
        check_pdf_correctness(model, data, D)
        print("✓ PDF tests passed")

        print("Testing CDF correctness...")
        check_cdf_correctness(model, data)
        print("✓ CDF tests passed")

        print("Testing PDF-CDF correctness via integration...")
        check_pdf_cdf_correctness(model, data)
        print("✓ PDF-CDF integration tests passed")

        print("Testing likelihood correctness...")
        check_likelihood_correctness(model, data)
        print("✓ Likelihood tests passed")

        print("Testing sampling...")
        check_sampling(model, data)
        print("✓ Sampling tests passed")

        print("Testing inverse CDF...")
        check_inverse_cdf(model, data)
        print("✓ Inverse CDF tests passed")

        print("Testing class and time marginals...")
        check_class_and_time_marginals(model, data)
        print("✓ Marginal distribution tests passed")

    print(f"\n{'=' * 60}")
    print(f"All tests passed for {model_type}!")
    print(f"{'=' * 60}\n")


def run_all_tests():
    """Run all unit tests for all model types."""
    model_types = [
        "poisson",
        "inhomogeneous_poisson",
        "spline_poisson",
        "hawkes",
        "linear_exp_hawkes",
        "spline_exp_hawkes",
    ]

    print("\n" + "=" * 60)
    print("RUNNING COMPREHENSIVE TPP UNIT TESTS")
    print("=" * 60)

    failed_models = []

    for model_type in model_types:
        try:
            run_tests_for_model(model_type)
        except Exception as e:
            print(f"\n{'=' * 60}")
            print(f"❌ Tests FAILED for {model_type}")
            print(f"Error: {e}")
            print(f"{'=' * 60}\n")
            failed_models.append((model_type, str(e)))

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"Total models tested: {len(model_types)}")
    print(f"Passed: {len(model_types) - len(failed_models)}")
    print(f"Failed: {len(failed_models)}")

    if failed_models:
        print("\nFailed models:")
        for model_type, error in failed_models:
            print(f"  - {model_type}: {error[:100]}...")
    else:
        print("\n🎉 ALL TESTS PASSED FOR ALL MODELS! 🎉")

    print("=" * 60 + "\n")

    return len(failed_models) == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
