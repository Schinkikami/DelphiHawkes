"""Debug script for NumericalTemporalPointProcess likelihood implementation."""

import torch
from hawkes.tpps import DebugNumericalPoisson
from hawkes.event_utils import BatchedMVEventData
import lovely_tensors as lt

lt.monkey_patch()
torch.set_printoptions(precision=6)

# Create a simple test case
D = 3
mu = torch.tensor([0.5, 0.3, 0.8], dtype=torch.float64)

# Create model
model = DebugNumericalPoisson(D=D, mu=mu, ci_num_points=100, ci_integration_method="trapezoidal")

# Create test data: 3 sequences (including one empty)
time_points = [
    torch.tensor([0.5, 1.5], dtype=torch.float64),
    torch.tensor([0.3, 0.8, 1.8], dtype=torch.float64),
    torch.tensor([], dtype=torch.float64),  # Empty sequence
]  # List of (L,) tensors
event_types = [
    torch.tensor([0, 1], dtype=torch.int64),
    torch.tensor([1, 0, 2], dtype=torch.int64),
    torch.tensor([], dtype=torch.int64),  # Empty sequence
]  # List of (L,) tensors
ts = BatchedMVEventData(time_points, event_types)

T = torch.tensor([1.5, 1.8, 2.0], dtype=torch.float64)  # End time

print("=" * 60)
print("DEBUG: NumericalTemporalPointProcess Likelihood")
print("=" * 60)
print(f"mu = {mu}")
print(f"Event times: {time_points}")
print(f"Event types: {event_types}")
print(f"T = {T}")
print()

# Analytical solution
print("ANALYTICAL SOLUTION:")
# Positive: log(mu[event_type]) for each event in each sequence
pos_analytical = (
    torch.log(mu[0]) + torch.log(mu[1]),  # Seq 1: events 0, 1
    torch.log(mu[1]) + torch.log(mu[0]) + torch.log(mu[2]),  # Seq 2: events 1, 0, 2
    torch.tensor(0.0, dtype=torch.float64),  # Seq 3: no events
)
print(
    f"  Positive log-likelihood:  [{pos_analytical[0].item():.6f}, {pos_analytical[1].item():.6f}, {pos_analytical[2].item():.6f}]"
)

# Negative: -sum_d mu_d * T for each sequence
neg_analytical = (
    -(mu.sum() * T[0]),
    -(mu.sum() * T[1]),
    -(mu.sum() * T[2]),
)
print(
    f"  Negative log-likelihood: "
    f"-({mu.sum().item():.1f}) * {T[0].item():.1f} = {neg_analytical[0].item():.6f}, "
    f"-({mu.sum().item():.1f}) * {T[1].item():.1f} = {neg_analytical[1].item():.6f}, "
    f"-({mu.sum().item():.1f}) * {T[2].item():.1f} = {neg_analytical[2].item():.6f}"
)

total_analytical = (
    pos_analytical[0] + neg_analytical[0],
    pos_analytical[1] + neg_analytical[1],
    pos_analytical[2] + neg_analytical[2],
)
print(
    f"  Total log-likelihood: [{total_analytical[0].item():.6f}, {total_analytical[1].item():.6f}, {total_analytical[2].item():.6f}]"
)
print()

# Test base class implementation (separate positive/negative)
print("BASE CLASS IMPLEMENTATION (separate pos/neg):")
with torch.no_grad():
    pos_base = model.positive_likelihood(ts, log=True)
    print(f"  Positive log-likelihood: {pos_base.tolist()}")

    # Use analytical CI for negative (for comparison)
    model.use_analytical_ci = True
    neg_base_analytical = model.negative_likelihood(ts, T, log=True)
    print(f"  Negative log-likelihood (analytical CI): {neg_base_analytical.tolist()}")

    # Use numerical CI for negative
    model.use_analytical_ci = False
    neg_base_numerical = model.negative_likelihood(ts, T, log=True)
    print(f"  Negative log-likelihood (numerical CI): {neg_base_numerical.tolist()}")

    total_base = pos_base + neg_base_numerical
    print(f"  Total log-likelihood (numerical): {total_base.tolist()}")
print()

# Test optimized implementation
print("OPTIMIZED IMPLEMENTATION (combined likelihood):")
with torch.no_grad():
    try:
        total_optimized = model.likelihood(ts, T, log=True)
        print(f"  Total log-likelihood: {total_optimized.tolist()}")
    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback

        traceback.print_exc()
print()

# Compare
print("COMPARISON:")
print(f"  Analytical (seq 1):  {total_analytical[0].item():.6f}")
print(f"  Analytical (seq 2):  {total_analytical[1].item():.6f}")
print(f"  Analytical (seq 3):  {total_analytical[2].item():.6f}")
if isinstance(total_base, torch.Tensor):
    print(f"  Base class (seq 1):  {total_base[0].item():.6f}")
    print(f"  Base class (seq 2):  {total_base[1].item():.6f}")
    print(f"  Base class (seq 3):  {total_base[2].item():.6f}")
    print(f"  Optimized (seq 1):   {total_optimized[0].item():.6f}")
    print(f"  Optimized (seq 2):   {total_optimized[1].item():.6f}")
    print(f"  Optimized (seq 3):   {total_optimized[2].item():.6f}")
    print(f"  Error base vs analytical (seq 1):   {abs(total_base[0].item() - total_analytical[0].item()):.6e}")
    print(f"  Error base vs analytical (seq 2):   {abs(total_base[1].item() - total_analytical[1].item()):.6e}")
    print(f"  Error base vs analytical (seq 3):   {abs(total_base[2].item() - total_analytical[2].item()):.6e}")
    print(f"  Error opt vs analytical (seq 1):    {abs(total_optimized[0].item() - total_analytical[0].item()):.6e}")
    print(f"  Error opt vs analytical (seq 2):    {abs(total_optimized[1].item() - total_analytical[1].item()):.6e}")
    print(f"  Error opt vs analytical (seq 3):    {abs(total_optimized[2].item() - total_analytical[2].item()):.6e}")
else:
    print(f"  Base class:  {total_base:.6f}")
    print(f"  Error (base vs analytical): {abs(total_base - total_analytical):.6e}")
