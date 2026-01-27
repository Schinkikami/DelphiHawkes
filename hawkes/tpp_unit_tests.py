# %%
# Test a Temporal Point Process for correctness

import sys
import lovely_tensors as lt

lt.monkey_patch()
sys.path.append("..")
# %%

import torch
from hawkes.event_utils import BatchedMVEventData, MVEventData
from hawkes.hawkes_tpp import (
    ExpKernelHawkesProcess,
    LinearBaselineExpKernelHawkesProcess,
    SplineBaselineExpKernelHawkesProcess,
)
from hawkes.baseline_tpps import SplinePoissonProcess, PoissonProcess, ConditionalInhomogeniousPoissonProcess

# %%
MODEL_TYPE = "spline_exp_hawkes"
spline_k = 100
spline_delta_t = 0.1
D = 2
seed = 32

# Model factory
if MODEL_TYPE == "poisson":
    print("Creating Poisson Process model...")
    model = PoissonProcess(D=D, seed=seed)

elif MODEL_TYPE == "inhomogeneous_poisson":
    print("Creating Inhomogeneous Poisson Process model...")
    model = ConditionalInhomogeniousPoissonProcess(D=D, seed=seed)

elif MODEL_TYPE == "spline_poisson":
    model = SplinePoissonProcess(D, spline_k, spline_delta_t, seed=seed)

elif MODEL_TYPE == "hawkes":
    model = ExpKernelHawkesProcess(D=D, seed=seed)

elif MODEL_TYPE == "linear_exp_hawkes":
    print("Creating new linear baseline exponential kernel Hawkes Process model...")
    model = LinearBaselineExpKernelHawkesProcess(D=D, seed=seed)
    save_filename = "linear_exp_hawkes.pth"

elif MODEL_TYPE == "spline_exp_hawkes":
    print("Creating new spline baseline exponential kernel Hawkes Process model...")
    model = SplineBaselineExpKernelHawkesProcess(D=D, num_knots=spline_k, delta_t=spline_delta_t, seed=seed)

else:
    raise ValueError(f"Unknown model type: {MODEL_TYPE}.")


emp_seq = MVEventData(torch.tensor([]), torch.tensor([], dtype=int))
seq = MVEventData(torch.tensor([0.2, 0.8, 2.0, 2.3], dtype=float), torch.tensor([0, 0, 1, 0], dtype=int))
padded_seq = MVEventData(
    torch.tensor([0.2, 0.8, 2.0, 2.3, 2.5, 2.8], dtype=float), torch.tensor([0, 0, 1, 0, -1, -1], dtype=int)
)
batch = BatchedMVEventData([seq.time_points], [seq.event_types])
empty_batch = BatchedMVEventData([emp_seq.time_points], [emp_seq.event_types])
padded_batch = BatchedMVEventData([padded_seq.time_points], [padded_seq.event_types])

# %% Start testing with intensity.


def test_intensity_correctness():
    # Check correctness of intensity
    after_sequence_intensity = model.intensity(torch.tensor([4.0]), batch)
    assert after_sequence_intensity.shape == (1, D), "Intensity shape mismatch after sequence."
    assert torch.all(after_sequence_intensity > 0), "Intensity should be positive."

    # Padding with -1 events should not have an effect on intensity
    after_padded_sequence_intensity = model.intensity(torch.tensor([4.0]), padded_batch)
    assert torch.allclose(after_sequence_intensity.float(), after_padded_sequence_intensity.float()), (
        "Intensity should be the same with padded events."
    )

    empty_intensity = model.intensity(torch.tensor([1.0]), empty_batch)
    assert empty_intensity.shape == (1, D), "Intensity shape mismatch for empty sequence."
    assert torch.all(empty_intensity > 0), "Intensity should be positive for empty sequence."

    intensity_exactly_at_event = model.intensity(torch.tensor([2.3]), batch)
    intensity_exactly_at_event_last_cut = model.intensity(torch.tensor([2.3]), batch[:, :-1])
    assert torch.allclose(intensity_exactly_at_event.float(), intensity_exactly_at_event_last_cut.float()), (
        "Intensity exactly at event should match intensity after last event."
    )

    intensity_inbetween = model.intensity(torch.tensor([2.2]), batch)
    intensity_last_cut = model.intensity(torch.tensor([2.2]), batch[:, :-1])
    assert torch.allclose(intensity_inbetween.float(), intensity_last_cut.float()), (
        "Intensity between events should match intensity after last event."
    )

    intensity_at_0 = model.intensity(torch.tensor([0.0]), batch)
    intensity_at_0_empty = model.intensity(torch.tensor([0.0]), empty_batch)
    assert torch.allclose(intensity_at_0.float(), intensity_at_0_empty.float()), (
        "Intensity exactly at event should match intensity after last event."
    )


# %%


def check_cumulative_intensity_correctness():
    # Check correctness of cumulative intensity
    after_sequence_cum_intensity = model.cumulative_intensity(torch.tensor([4.0]), batch)
    assert after_sequence_cum_intensity.shape == (1, D), "Cumulative Intensity shape mismatch after sequence."
    assert torch.all(after_sequence_cum_intensity > 0), "Cumulative Intensity should be positive."

    # Padding with -1 events should not have an effect on intensity
    after_padded_sequence_cum_intensity = model.cumulative_intensity(torch.tensor([4.0]), padded_batch)
    assert torch.allclose(after_sequence_cum_intensity.float(), after_padded_sequence_cum_intensity.float()), (
        "Cumulative Intensity should be the same with padded events."
    )

    empty_cum_intensity = model.cumulative_intensity(torch.tensor([1.0]), empty_batch)
    assert empty_cum_intensity.shape == (1, D), "Cumulative Intensity shape mismatch for empty sequence."
    assert torch.all(empty_cum_intensity > 0), "Cumulative Intensity should be positive for empty sequence."

    cum_intensity_exactly_at_event = model.cumulative_intensity(torch.tensor([2.3]), batch)
    cum_intensity_exactly_at_event_last_cut = model.cumulative_intensity(torch.tensor([2.3]), batch[:, :-1])
    assert torch.allclose(cum_intensity_exactly_at_event.float(), cum_intensity_exactly_at_event_last_cut.float()), (
        "Cumulative Intensity exactly at event should match cumulative Intensity after last event."
    )

    cum_intensity_inbetween = model.cumulative_intensity(torch.tensor([2.2]), batch)
    cum_intensity_last_cut = model.cumulative_intensity(torch.tensor([2.2]), batch[:, :-1])
    assert torch.allclose(cum_intensity_inbetween.float(), cum_intensity_last_cut.float()), (
        "Cumulative Intensity between events should match cumulative intensity after last event with last event cut."
    )


def check_intensity_cumulative_intensity_relation():
    # Check relation between intensity and cumulative intensity via numerical differentiation
    # Start slightly after the last event time to avoid discontinuities at event times
    times = torch.linspace(2.3 + 1e-4, 4.0, steps=1000)
    delta_t = times[1] - times[0]
    cum_intensities = torch.stack([model.cumulative_intensity(t.unsqueeze(0), batch) for t in times]).squeeze()
    intensities_numerical = (cum_intensities[1:] - cum_intensities[:-1]) / delta_t
    intensities_model = torch.stack([model.intensity(t.unsqueeze(0), batch) for t in times[0:-1]]).squeeze()
    assert torch.allclose(intensities_numerical.float(), intensities_model.float(), atol=1e-2), (
        "Numerical derivative of cumulative intensity should match intensity."
    )


def check_pdf_correctness():
    # Check correctness of PDF
    pdf_at_time = model.PDF(torch.tensor([2.5]), batch)
    assert pdf_at_time.shape == (1, D), "PDF shape mismatch."
    assert torch.all(pdf_at_time >= 0), "PDF should be non-negative."

    # Check relation between PDF and intensity and cumulative intensity
    intensity_at_time = model.intensity(torch.tensor([2.5]), batch)
    cum_intensity_at_time = model.cumulative_intensity(torch.tensor([2.5]), batch) - model.cumulative_intensity(
        torch.tensor([2.3]), batch
    )

    pdf_expected = intensity_at_time * torch.exp(-cum_intensity_at_time.sum())
    assert torch.allclose(pdf_at_time.float(), pdf_expected.float()), (
        "PDF should equal intensity times exp(-cumulative intensity)."
    )


def check_cdf_correctness():
    # Check correctness of PDF
    cdf_at_time = model.CDF(torch.tensor([2.5]), batch)
    assert cdf_at_time.shape == (1,), "PDF shape mismatch."
    assert torch.all(cdf_at_time >= 0), "PDF should be non-negative."

    # Check relation between CDF and cumulative intensity
    cum_intensity_at_time = model.cumulative_intensity(torch.tensor([2.5]), batch) - model.cumulative_intensity(
        torch.tensor([2.3]), batch
    )
    cdf_expected = 1 - torch.exp(-cum_intensity_at_time.sum())
    assert torch.allclose(cdf_at_time.float(), cdf_expected.float())

    cdf_at_last_point = model.CDF(torch.tensor([2.3001]), batch)
    assert torch.allclose(cdf_at_last_point, torch.zeros_like(cdf_at_last_point), atol=0.01), (
        "CDF at last event time should be zero."
    )


# Check correctness of PDF and CDF via numerical integration
def check_pdf_cdf_correctness():
    integrants = [model.PDF(torch.tensor([t]), batch) for t in torch.linspace(2.301, 2.5, steps=1000)]
    integrants = torch.stack(integrants).squeeze()
    integral = torch.trapz(integrants, torch.linspace(2.301, 2.5, steps=1000), dim=0).sum()
    print("Integral over PDF from 2.301 to 2.5:", integral)
    cdf = model.CDF(torch.tensor([2.5]), batch) - model.CDF(torch.tensor([2.301]), batch)
    assert torch.allclose(integral.float(), cdf.sum().float(), atol=1e-2)


# Check likelihood correctness
def check_likelihood_correctness():
    likelihood = model.likelihood(batch, torch.tensor([4.0]), log=False)
    assert likelihood.shape == (1,), "Likelihood shape mismatch."
    assert torch.all(likelihood > 0), "Likelihood should be positive."

    log_likelihood = model.likelihood(batch, torch.tensor([4.0]), log=True)
    assert log_likelihood.shape == (1,), "Log Likelihood shape mismatch."
    assert torch.all(log_likelihood <= 0), "Log Likelihood should be non-positive."

    assert torch.allclose(torch.exp(log_likelihood).float(), likelihood.float()), (
        "Exponentiated log likelihood should match likelihood."
    )

    # Check relation between likelihood and positive likelihood
    pos_likelihood = model.positive_likelihood(batch, log=False)
    cum_intensity_end = model.cumulative_intensity(torch.tensor([4.0]), batch).sum()
    likelihood_expected = pos_likelihood * torch.exp(-cum_intensity_end)
    assert torch.allclose(likelihood.float(), likelihood_expected.float()), (
        "Likelihood should equal positive likelihood times exp(-cumulative intensity at end)."
    )

    # Check correctness of positive likelihood

    pos_likelihood = model.positive_likelihood(batch, log=True)
    # pos_likelihood_tpp = super(type(model), model).positive_likelihood(batch, log=True)
    # Stack the sequence on itself with increasing lengths
    stacked_batch = BatchedMVEventData(
        time_points=[
            seq.time_points[:i]
            for i in range(
                len(seq.time_points),
            )
        ],
        event_types=[
            seq.event_types[:i]
            for i in range(
                len(seq.event_types),
            )
        ],
    )

    stacked_intensities = model.intensity(seq.time_points, stacked_batch)
    intensity_at_points = stacked_intensities[torch.arange(len(seq.time_points)), seq.event_types]
    log_intensity_at_points = intensity_at_points.log()

    assert torch.allclose(pos_likelihood.float(), log_intensity_at_points.sum().float())


def check_sampling():
    samples, tstar, dstar = model.sample(seq, num_steps=10)

    assert (torch.tensor(tstar) >= seq.time_points.max()).all(), "Sampled time should be after last event time."
    assert len(seq) + 10 == len(samples), "Number of sampled events should match."


def check_inverse_cdf():
    u_values = torch.tensor([0.01, 0.5, 0.9, 0.99])
    for u in u_values:
        t_inv = model.inverse_CDF(u.unsqueeze(0), batch)
        cdf_at_t_inv = model.CDF(t_inv, batch)
        assert torch.allclose(cdf_at_t_inv.float(), u.unsqueeze(0).float(), atol=1e-2), (
            f"CDF at inverse CDF({u.item()}) should equal {u.item()}."
        )

    # Test inverse CDF at edge cases
    t_inv_0 = model.inverse_CDF(torch.tensor([0.0]), batch)
    cdf_at_t_inv_0 = model.CDF(t_inv_0, batch)
    assert torch.allclose(cdf_at_t_inv_0.float(), torch.tensor([0.0]).float(), atol=1e-4), (
        "CDF at inverse CDF(0) should be 0."
    )

    t_inv_1 = model.inverse_CDF(torch.tensor([1.0]), batch)
    cdf_at_t_inv_1 = model.CDF(t_inv_1, batch)
    assert torch.allclose(cdf_at_t_inv_1.float(), torch.tensor([1.0]).float(), atol=1e-4), (
        "CDF at inverse CDF(1) should be 1."
    )


def check_class_and_time_marginals():
    # Find the 0.999 quantile time
    t_999 = model.inverse_CDF(torch.tensor([0.999]), batch)[0].item()

    xs = torch.linspace((batch.max_time + 1e-4).item(), t_999, steps=100)
    # Compute class marginals up to t_999
    joint_time_type_pdf = []
    for t in xs:
        pdf = model.PDF(t.unsqueeze(0), batch)  # Joint PDF over time and type
        joint_time_type_pdf.append(pdf)

    joint_time_type_pdf = torch.stack(joint_time_type_pdf).squeeze()  # Shape: (num_times, D)
    type_marginals = torch.trapz(joint_time_type_pdf, xs, dim=0)  # Shape: (D,)
    assert 0.99 <= type_marginals.sum() <= 1.0
    type_marginals = type_marginals / type_marginals.sum()  # Normalize to sum to 1

    time_marginal = joint_time_type_pdf.sum(dim=1)  # Is a density. Does not sum to 1.

    time_cdf = torch.cumulative_trapezoid(time_marginal, xs, dim=0)
    assert torch.all(type_marginals >= 0), "Type marginals should be non-negative."
    assert torch.all(time_cdf >= 0) and torch.all(time_cdf <= 1), "Time CDF should be between 0 and 1."

    # Sample 10.000 continuations and compute empirical marginals
    num_samples = 1000
    tstar, dstar = [], []
    for i in range(num_samples):
        _, tt, dd = model.sample(seq, num_steps=1, rng=i)
        tstar.append(tt[0])
        dstar.append(dd[0])

    tstar = torch.tensor(tstar)
    dstar = torch.stack(dstar)
    empirical_type_marginals = torch.mean(dstar, dim=0)
    empirical_time_cdf = torch.tensor([(tstar <= t).float().mean() for t in xs])

    assert torch.allclose(type_marginals.float(), empirical_type_marginals.float(), atol=0.05), (
        "Empirical type marginals should match computed type marginals."
    )
    assert torch.allclose(time_cdf.float(), empirical_time_cdf[1:].float(), atol=0.05), (
        "Empirical time CDF should match computed time CDF."
    )


with torch.no_grad():
    test_intensity_correctness()
    check_cumulative_intensity_correctness()
    check_intensity_cumulative_intensity_relation()
    check_pdf_correctness()
    check_cdf_correctness()
    check_pdf_cdf_correctness()
    check_likelihood_correctness()
    check_sampling()
    check_inverse_cdf()
    check_class_and_time_marginals()
