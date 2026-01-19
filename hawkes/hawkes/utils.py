import torch
from torch import Tensor


def inverse_softplus(x: Tensor):
    # Computes the inverse of the softplus function,
    # using the numerically stable log(expm1(x)) implementation
    # (sadly torch does not provide logexpm1)
    return torch.log(torch.expm1(x))


def bracket_monotone(
    monot_func,
    low: float,
    target: float,
    initial_high: float | None = None,
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
    target: float,
    low: float,
    high: float,
    tol: float = 1e-6,
    max_iters: int = 50,
    mono_kwargs: dict = dict(),
    d_kwargs: dict = dict(),
):
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
