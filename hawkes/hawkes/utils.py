from collections.abc import Callable
from typing import Optional, Tuple
import torch
from torch import Tensor
import torch.nn.functional as F
from .event_utils import BatchedMVEventData


# ==============================================================================
# Function Utilities
# ==============================================================================


def inverse_softplus(x: Tensor):
    # Computes the inverse of the softplus function,
    # using the numerically stable log(expm1(x)) implementation
    # (sadly torch does not provide logexpm1)
    return torch.log(torch.expm1(x))


class LinearSpline(torch.nn.Module):
    @staticmethod
    def interpolate(x_knots: torch.Tensor, y_knots: torch.Tensor, t: torch.Tensor):
        """Interpolates the splines at the given positions.

        Args:
            x_knots (torch.Tensor): Position of the splines. Shared between all splines. Shape: (K,)
            y_knots (torch.Tensor): Height of the splines for each knot in each spline. Shape: (D,K)
            t (torch.Tensor): Time-points where to interpolate the splines. Shape: (B,)

        Returns:
            interpolation (torch.Tensor): Interpolated splines at t. Shape: (B,D)
        """
        k, delta = LinearSpline._get_knot_info(x_knots, t)  # k: (B,), delta: (B,)

        h_k = y_knots[:, k]  # (D,B)

        delta_times = x_knots[1:] - x_knots[:-1]  # (K-1)
        delta_heights = y_knots[:, 1:] - y_knots[:, :-1]  # (K-1)
        slopes = delta_heights / delta_times  # (D, K-1)
        segment_slopes = F.pad(slopes, (0, 1), value=0.0)  # (D, K)

        slope_k = segment_slopes[:, k]  # (D,B)

        interpolation = h_k + slope_k * delta.unsqueeze(0)  # (D,B)

        return interpolation.T  # (B,D)

    @staticmethod
    def integrate(x_knots: torch.Tensor, y_knots: torch.Tensor, t: torch.Tensor):
        """Calculates the integral of the splines at the given positions.

        Args:
            x_knots (torch.Tensor): Position of the splines. Shared between all splines. Shape: (K,)
            y_knots (torch.Tensor): Height of the splines for each spline and each knot. Shape: (D,K)
            t (torch.Tensor): Time-points where up to where to integrate the splines (starting from 0). Shape: (B,)

        Returns:
            integral: (torch.Tensor): Integrated splines up to t. Shape: (B,D)
        """
        k, delta = LinearSpline._get_knot_info(x_knots=x_knots, t=t)  # (B,), (B,)

        delta_times = x_knots[1:] - x_knots[:-1]  # (K-1)
        delta_heights = y_knots[:, 1:] - y_knots[:, :-1]  # (K-1)
        slopes = delta_heights / delta_times  # (D, K-1)

        segment_areas = (
            y_knots[:, :-1] * delta_times + slopes / 2.0 * delta_times.unsqueeze(0) ** 2
        )  # integral_0^t ( h + slope * s)  ds, Shape: (D,K-1)

        prefix_sums = torch.cumsum(segment_areas, dim=1)  # Shape: (D,K-1)
        prefix_sums = F.pad(prefix_sums, (1, 0), value=0.0)  # (D, K)
        segment_slopes = F.pad(slopes, (0, 1), value=0.0)  # (D, K)

        # Extract info for current interval
        C_k = prefix_sums[:, k]  # (D,B)
        s_k = segment_slopes[:, k]  # (D,B)
        h_k = y_knots[:, k]  # (D,B)
        last_knot_loc = x_knots[k]  # (B,)
        d_t = t - last_knot_loc  # (B,)

        # Integral of previous segments + square_area + triangular area
        # Done like this to avoid zero multiplications which can lead to NaNs if d_t is Infinite
        integral = C_k
        integral += torch.where(h_k != 0.0, h_k * d_t.unsqueeze(0), torch.zeros_like(h_k))
        integral += torch.where(s_k != 0.0, s_k / 2 * delta.unsqueeze(0) ** 2, torch.zeros_like(s_k))
        # (D,B)
        # integral = C_k + h_k * d_t.unsqueeze(0) + s_k / 2 * delta.unsqueeze(0) ** 2  # (D,B)

        return integral.T  # (B,D)

    @staticmethod
    def inverse_integral(x_knots: torch.Tensor, y_knots: torch.Tensor, u: torch.Tensor):
        """Computes the inverse of the integral if defined (y_knots has to be strictly positive or negative).

        Given u, finds t such that u = int_0^t spline(x_knots, y_knots, s) ds.

        Args:
            x_knots (torch.Tensor): Position of the splines. Shared between all splines. Shape: (K,)
            y_knots (torch.Tensor): Height of the splines for each spline and each knot. Shape: (D,K)
            u (torch.Tensor): Target integral values. Shape: (B, D)

        Returns:
            t (torch.Tensor): The time values producing the desired u's. Shape: (B, D)
        """
        assert (y_knots > 0).all() or (y_knots < 0).all(), (
            "y_knots must be strictly positive or negative for invertibility"
        )

        D, K = y_knots.shape
        B = u.shape[0]

        # Compute segment properties
        delta_times = x_knots[1:] - x_knots[:-1]  # (K-1,)
        delta_heights = y_knots[:, 1:] - y_knots[:, :-1]  # (D, K-1)
        slopes = delta_heights / delta_times  # (D, K-1)

        segment_areas = y_knots[:, :-1] * delta_times + slopes / 2.0 * delta_times**2  # (D, K-1)

        prefix_sums = torch.cumsum(segment_areas, dim=1)  # (D, K-1)
        prefix_sums = F.pad(prefix_sums, (1, 0), value=0.0)  # (D, K) - integral from 0 to x_knots[k]

        segment_slopes = F.pad(slopes, (0, 1), value=0.0)  # (D, K)

        # For each (batch, dim), find which segment the target u falls into
        # u has shape (B, D), prefix_sums has shape (D, K)
        # Find k such that prefix_sums[d, k] <= u[b, d] < prefix_sums[d, k+1]
        k = torch.searchsorted(prefix_sums.unsqueeze(0).expand((B, D, K)), u.unsqueeze(2), right=True) - 1
        k = k.squeeze(2).T  # (B,D)

        # Extract segment info for each (d, b) pair
        # k has shape (D, B)
        C_k = torch.gather(prefix_sums, 1, k)  # (D, B) - cumulative integral up to knot k
        h_k = torch.gather(y_knots, 1, k)  # (D, B) - height at knot k
        s_k = torch.gather(segment_slopes, 1, k)  # (D, B) - slope in segment k
        x_k = x_knots[k]  # (D, B) - position of knot k

        # We need to solve:  h_k * delta + s_k/2 * delta^2 = u - C_k
        # Rearranging: s_k/2 * delta^2 + h_k * delta - (u - C_k) = 0
        #            = s_k/2 * delta^2 + h_k * delta - c = 0

        # Using quadratic formula: delta = (-h_k + sqrt(h_k^2 + 2*s_k*c)) / s_k

        c_val = u.T - C_k  # (D, B)

        # Handle two cases: non-zero slope (quadratic) and zero slope (linear)
        slope_is_zero = torch.abs(s_k) < 1e-10

        # For zero slope: h_k * delta = u - C_k => delta = (u - C_k) / h_k = c_val / h_k
        delta_linear = c_val / h_k  # (D, B)

        # For non-zero slope: solve quadratic
        # s_k/2 * delta^2 + h_k * delta + c_val = 0
        # delta = (-h_k + sqrt(h_k^2 + 2*s_k*c_val)) / s_k
        discriminant = h_k**2 + 2 * s_k * c_val  # (D, B)
        # Clamp to avoid numerical issues.
        discriminant = torch.clamp(discriminant, min=0.0)
        sqrt_disc = torch.sqrt(discriminant)

        # Safe division for quadratic case to avoid NaN in tensors. Can lead to problems, even when masked.
        # Learned that the hard way... Will be masked out later anyway.
        s_k_safe = torch.where(slope_is_zero, torch.ones_like(s_k), s_k)
        delta_quadratic = (-h_k + sqrt_disc) / s_k_safe
        # Select based on whether slope is zero
        delta = torch.where(slope_is_zero, delta_linear, delta_quadratic)  # (D, B)

        # Compute final t values
        t = x_k + delta  # (D, B)

        return t.T  # (B, D)

    @staticmethod
    def _get_knot_info(x_knots, t: torch.Tensor):
        """
        Calculates indices and relative offsets for time points.
        x_knots: Location of the knots (K,)
        t: (batch_size,)
        returns: k (indices, Shape:(B,)), delta (t - t_k, Shape:(B,))
        """

        k = torch.searchsorted(x_knots, t, right=True) - 1
        delta = t - x_knots[k]
        return k, delta


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
    device = ts.time_points.device

    if rng is None:
        rng = torch.Generator(device=device)

    B = ts.shape[0]

    random_points = (t_end - t_start).unsqueeze(1) * torch.rand(
        (B, num_points - 2), generator=rng, device=device
    ) + t_start.unsqueeze(1)  # (B, num_points-2)

    t_vals = torch.cat([t_start.unsqueeze(1), random_points, t_end.unsqueeze(1)], dim=1)  # (B, num_points)

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
    Find a bracket [low, high] such that monot_func(high) >= target assuming monot_func is monotonically increasing.

    Supports batched operations where low, target, and monot_func output are tensors of shape (T,).

    Args:
        monot_func: callable that takes tensor of shape (T,) and returns tensor of shape (T,)
        low: starting lower bounds. Shape: (T,)
        target: target cumulative values. Shape: (T,)
        initial_high: optional initial high to try; if None, uses low + 1.0. Shape: (T,) or None
        expand_factor: multiplicative expansion factor for high
        max_expands: maximum number of expansions

    Returns:
        (low, high, ci_low, ci_high) - all tensors of shape (T,)
    """
    if initial_high is None:
        high = low + 1.0
    else:
        high = initial_high.clone()

    ci_low = monot_func(low, **func_kwargs)
    ci_high = monot_func(high, **func_kwargs)

    # Track which elements still need expansion
    needs_expand = ci_high < target

    expands = 0
    while needs_expand.any() and expands < max_expands:
        # Exponentially expand the span for elements that need it
        span = high - low
        new_high = low + span * expand_factor

        # Only update elements that need expansion
        high = torch.where(needs_expand, new_high, high)

        ci_high = monot_func(high, **func_kwargs)
        needs_expand = ci_high < target
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
