from __future__ import annotations
from typing import Callable, Optional, Type
import numpy as np

from .tensor_data import (
    MAX_DIMS,
    broadcast_index,
    index_to_position,
    shape_broadcast,
    to_index,
    Index, Shape, Storage, Strides,
)

def tensor_map(
    fn: Callable[[float], float]
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides], None]:
    """
    Low-level implementation of tensor map between tensors with
    possibly different strides.

    Args:
        fn: function from float to float to apply

    Returns:
        Tensor map function
    """
    def _map(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        in_storage: Storage,
        in_shape: Shape,
        in_strides: Strides,
    ) -> None:
        # TODO: Implement for Task 2.3
        # For each position in output:
        #   1. Convert position to output index
        #   2. Map output index to input index (handle broadcasting)
        #   3. Get input value, apply fn, store in output

        out_index = np.zeros(MAX_DIMS, dtype=np.int32)
        in_index = np.zeros(MAX_DIMS, dtype=np.int32)

        for i in range(int(np.prod(out_shape))):
            to_index(i, out_shape, out_index)
            broadcast_index(out_index, out_shape, in_shape, in_index)

            out_pos = index_to_position(out_index, out_strides)
            in_pos = index_to_position(in_index[:len(in_shape)], in_strides)

            out[out_pos] = fn(in_storage[in_pos])


    return _map


def tensor_zip(
    fn: Callable[[float, float], float]
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides, Storage, Shape, Strides], None]:
    """
    Low-level implementation of tensor zip between tensors with
    possibly different strides.

    Args:
        fn: function from two floats to float

    Returns:
        Tensor zip function
    """
    def _zip(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        b_storage: Storage,
        b_shape: Shape,
        b_strides: Strides,
    ) -> None:
        # TODO: Implement for Task 2.3
        # Similar to map, but get values from two inputs

        out_index = np.zeros(MAX_DIMS, dtype=np.int32)
        a_index = np.zeros(MAX_DIMS, dtype=np.int32)
        b_index = np.zeros(MAX_DIMS, dtype=np.int32)

        for i in range(int(np.prod(out_shape))):
            to_index(i, out_shape, out_index)
            broadcast_index(out_index, out_shape, a_shape, a_index)
            broadcast_index(out_index, out_shape, b_shape, b_index)

            out_pos = index_to_position(out_index, out_strides)
            a_pos = index_to_position(a_index[:len(a_shape)], a_strides)
            b_pos = index_to_position(b_index[:len(b_shape)], b_strides)

            out[out_pos] = fn(a_storage[a_pos], b_storage[b_pos])


    return _zip


def tensor_reduce(
    fn: Callable[[float, float], float]
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides, int], None]:
    """
    Low-level implementation of tensor reduce.

    The output shape is the same as input except reduce_dim is size 1.

    Args:
        fn: reduction function (e.g., add, mul)

    Returns:
        Tensor reduce function
    """
    def _reduce(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        reduce_dim: int,
    ) -> None:
        # TODO: Implement for Task 2.3
        # For each output position:
        #   Iterate over the reduce dimension
        #   Accumulate using fn

        out_index = np.zeros(MAX_DIMS, dtype=np.int32)
        a_index = np.zeros(MAX_DIMS, dtype=np.int32)

        for i in range(int(np.prod(out_shape))):
            to_index(i, out_shape, out_index)
            out_pos = index_to_position(out_index, out_strides)

            # Copy output index to a_index
            for j in range(len(out_shape)):
                a_index[j] = out_index[j]

            # Iterate over reduce dimension
            for i in range(int(np.prod(out_shape))):
            to_index(i, out_shape, out_index)
            out_pos = index_to_position(out_index, out_strides)

            for j in range(len(out_shape)):
                a_index[j] = out_index[j]

            for j in range(a_shape[reduce_dim]):
                a_index[reduce_dim] = j
                a_pos = index_to_position(a_index[:len(a_shape)], a_strides)
                out[out_pos] = fn(out[out_pos], a_storage[a_pos])

    return _reduce
