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
    position = 0  # Q1: What operation combines index and strides?
    for i, s in zip(index, strides):
        position += i * s
    return int(position)     # Q2: Return type should be int


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
        out_index[i] = cur_ord % shape[i]
        cur_ord = cur_ord // shape[i]

    for i in range(len(shape) - 1, -1, -1):
        out_index[i] = cur_ord % shape[i]
        cur_ord = cur_ord // shape[i]      # Q4: How do you get the remaining ordinal?
  


class TensorData:
    _storage: Storage
    _strides: Strides
    _shape: Shape
    strides: UserStrides
    shape: UserShape
    dims: int

    def __init__(
        self,
        storage: Union[Sequence[float], Storage],
        shape: UserShape,
        strides: Optional[UserStrides] = None,
    ):
        if isinstance(storage, np.ndarray):
            self._storage = storage
        else:
            self._storage = array(storage, dtype=float64)

        if strides is None:
            strides = strides_from_shape(shape)

        assert isinstance(strides, tuple), "Strides must be tuple"
        assert isinstance(shape, tuple), "Shape must be tuple"
        if len(strides) != len(shape):
            raise IndexingError(f"Len of strides {strides} must match {shape}.")

        self._strides = array(strides)
        self._shape = array(shape)
        self.strides = strides
        self.dims = len(strides)
        self.size = int(prod(shape))
        self.shape = shape
        assert len(self._storage) == self.size

    # In TensorData class:
    def permute(self, *order: int) -> TensorData:
        """Permute tensor dimensions."""
        assert list(sorted(order)) == list(range(len(self.shape)))

        new_shape = tuple(self.shape[o] for o in order)
        new_strides = tuple(self.strides[o] for o in order)

        return TensorData(self._storage, new_shape, new_strides)
   
def shape_broadcast(shape1: UserShape, shape2: UserShape) -> UserShape:
    """Broadcast two shapes to create a new union shape."""
    result = []
    len1, len2 = len(shape1), len(shape2)
    max_len = max(len1, len2)

    for i in range(max_len):
        d1 = shape1[len1 - 1 - i] if i < len1 else 1
        d2 = shape2[len2 - 1 - i] if i < len2 else 1

        if d1 == d2:
            result.append(d1)
        elif d1 == 1:
            result.append(d2)
        elif d2 == 1:
            result.append(d1)
        else:
            raise IndexingError(f"Cannot broadcast shapes {shape1} and {shape2}")

    return tuple(reversed(result))


def broadcast_index(
    big_index: Index,
    big_shape: Shape,
    shape: Shape,
    out_index: OutIndex
) -> None:
    """Convert index from broadcasted shape to original shape."""
    offset = len(big_shape) - len(shape)

    for i in range(len(shape)):
        if shape[i] == 1:
            out_index[i] = 0
        else:
            out_index[i] = big_index[i + offset]
