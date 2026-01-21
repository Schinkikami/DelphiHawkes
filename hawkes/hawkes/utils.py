from collections.abc import Callable
from typing import Optional, Tuple
import torch
from torch import Tensor

from .event_utils import BatchedMVEventData


# ==============================================================================
# Function Utilities
# ==============================================================================


def inverse_softplus(x: Tensor):
    # Computes the inverse of the softplus function,
    # using the numerically stable log(expm1(x)) implementation
    # (sadly torch does not provide logexpm1)
    return torch.log(torch.expm1(x))


# ==============================================================================
# Numerical Integration Utilities
# ==============================================================================


def trapezoidal_integration(
    intensity_func: Callable,
    t_start: torch.Tensor,
    t_end: torch.Tensor,
    ts: BatchedMVEventData,
    num_points: int = 100,
) -> torch.Tensor:
    """
    Compute integral using trapezoidal rule.

    Args:
        func: Function to integrate, should accept (t: Tensor, ts: BatchedMVEventData) -> Tensor of shape (B, D)
        t_start: Start times. Shape: (B,)
        t_end: End times. Shape: (B,)
        ts: Batched event data
        num_points: Number of integration points

    Returns:
        Integral values. Shape: (B, D)
    """

    device = t_start.device

    t_vals = torch.linspace(0, 1, num_points, device=device)
    t_vals = t_start.unsqueeze(1) + (t_end - t_start).unsqueeze(1) * t_vals.unsqueeze(0)  # (B, num_points)

    intensities = [
        intensity_func(t=t_vals[:, k], ts=ts) for k in range(num_points)
    ]  # List of length num_points, each (B,D)
    intensities = torch.stack(intensities, dim=2)  # Shape: (B, D, num_points)

    # Trapezoidal rule
    integrals = torch.trapezoid(intensities, t_vals.unsqueeze(1), dim=2)  # Shape: (B, D)

    return integrals


def monte_carlo_trapezoidal_integration(
    intensity_func: Callable,
    t_start: torch.Tensor,
    t_end: torch.Tensor,
    ts: BatchedMVEventData,
    num_points: int = 100,
    rng: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """
    Hybrid Monte Carlo + Trapezoidal integration.

    Places evaluation points randomly in [t_start, t_end] while including boundaries,
    then uses trapezoidal rule on the sorted points.

    Args:
        func: Function to integrate
        t_start: Start times. Shape: (B,)
        t_end: End times. Shape: (B,)
        ts: Batched event data
        num_points: Number of integration points
        rng: Random number generator

    Returns:
        Integral values. Shape: (B, D)
    """
    if rng is None:
        rng = torch.Generator()

    B = ts.shape[0]

    device = t_start.device

    random_points = (t_end - t_start).unsqueeze(1) * torch.rand(
        (B, num_points - 2), generator=rng, device=device
    ) + t_start.unsqueeze(1)  # (B, num_points-2)

    t_vals = torch.cat(
        [torch.tensor([t_start], device=device), random_points, torch.tensor([t_end], device=device)], dim=1
    )  # (B, num_points)

    # Sort
    t_vals, _ = torch.sort(t_vals, dim=1)

    intensities = [
        intensity_func(t=t_vals[:, k], ts=ts) for k in range(num_points)
    ]  # List of length num_points, each (B,D)
    intensities = torch.stack(intensities, dim=2)  # Shape: (B, D, num_points)

    # Trapezoidal rule
    integrals = torch.trapezoid(intensities, t_vals.unsqueeze(1), dim=2)  # Shape: (B, D)

    return integrals


def numerical_integration(
    func: Callable,
    t_start: torch.Tensor,
    t_end: torch.Tensor,
    ts: BatchedMVEventData,
    method: str = "trapezoidal",
    num_points: int = 100,
    rng: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """
    Numerically integrate a function using the specified method.

    Args:
        func: Function to integrate, should accept (t: Tensor, ts: BatchedMVEventData) -> Tensor of shape (B, D)
        t_start: Start times. Shape: (B,)
        t_end: End times. Shape: (B,)
        ts: Batched event data
        method: One of "trapezoidal", "mc_trapezoidal"
        num_points: Number of integration points
        rng: Random number generator (required for MC methods)

    Returns:
        Integral values. Shape: (B, D)
    """
    if method == "trapezoidal":
        return trapezoidal_integration(func, t_start, t_end, ts, num_points)
    elif method == "mc_trapezoidal":
        return monte_carlo_trapezoidal_integration(func, t_start, t_end, ts, num_points, rng)
    else:
        raise ValueError(f"Unknown integration method: {method}")


# ==============================================================================
# Monotone Function Inversion Utilities
# ==============================================================================


def bracket_monotone(
    monot_func,
    low: torch.Tensor,
    target: torch.Tensor,
    initial_high: torch.Tensor | None = None,
    expand_factor: float = 2.0,
    max_expands: int = 60,
    func_kwargs: dict = dict(),
):
    """
    Find a bracket [low, high] such that ci_func(high) >= target assuming ci_func is monotonically increasing in its scalar argument.

    Args:
        ci_func: callable T -> scalar cumulative value (float)
        low: starting lower bound (float)
        target: target cumulative value (float)
        initial_high: optional initial high to try; if None, uses low + 1.0
        expand_factor: multiplicative expansion factor for high
        max_expands: maximum number of expansions

    Returns:
        (low, high, ci_low, ci_high)
    """
    if initial_high is None:
        high = low + 1.0
    else:
        high = initial_high

    ci_low = float(monot_func(low, **func_kwargs))
    ci_high = float(monot_func(high, **func_kwargs))

    expands = 0
    while ci_high < target and expands < max_expands:
        # expand geometrically
        span = high - low
        if span <= 0:
            high = low + (2**expands)
        else:
            high = low + max(span * expand_factor, 1.0 * (2**expands))

        ci_high = float(monot_func(high, **func_kwargs))
        expands += 1

    return low, high, ci_low, ci_high


def invert_monotone_newton(
    monot_func,
    d_func,
    target: torch.Tensor,
    low: torch.Tensor,
    high: torch.Tensor,
    tol: float = 1e-6,
    max_iters: int = 50,
    mono_kwargs: dict = dict(),
    d_kwargs: dict = dict(),
) -> Tensor:
    """
    Invert a monotone cumulative function ci_func(T) to find T such that ci_func(T) = target.

    Uses Newton's method with derivative provided by d_func.
    Falls back to bisection when Newton proposes values outside the [low, high] bracket or derivative is too small.

    Args:
        monot_func: callable T -> monot_func(T) (float)
        d_func: callable T -> d_func(T) (float, derivative)
        target: target cumulative value
        low, high: bracket with ci(low) <= target <= ci(high)
        tol: tolerance on |ci(T)-target|
        max_iters: maximum iterations

    Returns:
        T (float)
    """
    # Start at midpoint
    T = 0.5 * (low + high)

    for _ in range(max_iters):
        ci_val = float(monot_func(T, **mono_kwargs))
        resid = ci_val - target
        if abs(resid) <= tol:
            return T

        lam = float(d_func(T, **d_kwargs))
        # avoid tiny derivative
        if lam <= 1e-12 or not (lam == lam):
            # bisection
            T_new = 0.5 * (low + high)
        else:
            T_new = T - resid / lam

        # If Newton step leaves bracket or is nan/inf, fall back to bisection
        if not (T_new > low and T_new < high and (T_new == T_new)):
            T_new = 0.5 * (low + high)

        ci_new = float(monot_func(T_new, **mono_kwargs))
        # update bracket
        if ci_new < target:
            low = T_new
        else:
            high = T_new

        T = T_new

    return T


def invert_CI(
    cumulative_intensity_func: Callable[[torch.Tensor], torch.Tensor],
    intensity_func: Callable[[torch.Tensor], torch.Tensor],
    target: torch.Tensor,
    bracket_init: Tuple[torch.Tensor, torch.Tensor],
    tol: float = 1e-6,
    max_newton_iters: int = 50,
    expand_factor: float = 2.0,
    max_expansions: int = 100,
) -> torch.Tensor:
    """
    Invert cumulative intensity using Newton's method with bracketing.

    Args:
        cumulative_intensity_func: Callable that returns cumulative intensity at time t (thats absolute time)
        intensity_func: Callable that returns intensity (derivative of CDF) at time t (thats absolute time)
        target: Target cumulative intensity value to invert
        bracket_init: Initial bracket interval
        tol: Tolerance for convergence
        max_newton_iters: Maximum Newton iterations
        expand_factor: Factor to expand bracket by
        max_expansions: Maximum bracket expansion iterations

    Returns:
        Time t such that cumulative_intensity_func(t) ≈ target
    """
    # First, bracket the root
    low, high, ci_low, ci_high = bracket_monotone(
        cumulative_intensity_func,
        bracket_init[0],
        target,
        initial_high=bracket_init[1],
        expand_factor=expand_factor,
        max_expands=max_expansions,
    )

    # if ci_high < target:
    #    # Failed to bracket: return high value
    #    return float(high)

    # Use Newton's method with bisection fallback
    t_star = invert_monotone_newton(
        cumulative_intensity_func,
        intensity_func,
        target,
        low,
        high,
        tol=tol,
        max_iters=max_newton_iters,
    )

    return t_star
