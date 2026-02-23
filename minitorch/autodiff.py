"""
Automatic differentiation utilities for MiniTorch.
"""

from typing import Callable, List, Tuple, Any


def central_difference(f: Callable[..., float], *vals: float, arg: int = 0, epsilon: float = 1e-6) -> float:
    """
    Compute numerical derivative of f with respect to argument `arg`.

    Uses central difference: (f(x+h) - f(x-h)) / (2h)

    Args:
        f: Function to differentiate
        *vals: Input values to f
        arg: Which argument to differentiate with respect to (0-indexed)
        epsilon: Step size for numerical differentiation

    Returns:
        Approximate derivative

    Example:
        >>> def mul(x, y): return x * y
        >>> central_difference(mul, 3, 4, arg=0)  # df/dx at (3,4)
        4.0
        >>> central_difference(mul, 3, 4, arg=1)  # df/dy at (3,4)
        3.0
    """
    vals_list = list(vals)

    # Create vals with arg incremented by epsilon
    vals_plus = vals_list.copy()
    vals_plus[arg] = vals_plus[arg] + epsilon  # Q1: Add what?

    # Create vals with arg decremented by epsilon
    vals_minus = vals_list.copy()
    vals_minus[arg] = vals_minus[arg] - epsilon # Q2: Subtract what?

    # Compute central difference
    f_plus = f(*vals_plus)
    f_minus = f(*vals_minus)

    return (f_plus - f_minus) / (2 * epsilon)  # Q3: Divide by what?
