
from .scalar import Scalar

def assert_close(a, b, tol=1e-6):
    """
    Check if two numbers are approximately equal.
    """
    assert abs(a - b) < tol, f"{a} != {b}"