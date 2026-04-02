from __future__ import annotations
from typing import Iterable, Optional, Sequence, Tuple, Union
import numpy as np
import numpy.typing as npt
from numpy import array, float64

MAX_DIMS = 32

class IndexingError(RuntimeError):
    """Exception raised for indexing errors."""
    pass

# Type aliases for clarity
Storage = npt.NDArray[np.float64]
OutIndex = npt.NDArray[np.int32]
Index = npt.NDArray[np.int32]
Shape = npt.NDArray[np.int32]
Strides = npt.NDArray[np.int32]
UserIndex = Sequence[int]
UserShape = Sequence[int]
UserStrides = Sequence[int]


def index_to_position(index: Index, strides: Strides) -> int:
    """
    Convert a multidimensional tensor index into a single-dimensional
    position in storage based on strides.

    Args:
        index: index tuple of ints (as numpy array)
        strides: tensor strides (as numpy array)

    Returns:
        Position in storage
    """
    # TODO: Implement for Task 2.1
    # Hint: The position is the dot product of index and strides
    # Example: index=[1, 2], strides=[4, 1] -> position = 1*4 + 2*1 = 6
    position = ___  # Q1: What operation combines index and strides?
    return ___      # Q2: Return type should be int


def to_index(ordinal: int, shape: Shape, out_index: OutIndex) -> None:
    """
    Convert an ordinal (flat position 0...size-1) to a multi-dimensional
    index in the given shape.

    This is the inverse mapping: given position in enumeration order,
    produce the corresponding index.

    Args:
        ordinal: ordinal position to convert
        shape: tensor shape
        out_index: output array to fill with index values

    Returns:
        None (modifies out_index in place)
    """
    # TODO: Implement for Task 2.1
    # Hint: Work from the last dimension to the first
    # Use modulo and integer division
    # Example: ordinal=5, shape=(2,3) -> out_index=[1, 2]
    #   5 % 3 = 2 (last index)
    #   5 // 3 = 1 (remaining ordinal for next dimension)

    cur_ord = ordinal
    for i in range(len(shape) - 1, -1, -1):
        out_index[i] = ___  # Q3: What operation gives the index for dimension i?
        cur_ord = ___       # Q4: How do you get the remaining ordinal?
  