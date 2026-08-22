from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F


def effective_scale_count(
    spatial_shape: tuple[int, ...], *, block_size: int, requested_scales: int
) -> int:
    if len(spatial_shape) not in {2, 3}:
        raise ValueError("MINER supports 2D or 3D spatial signals")
    if block_size <= 0 or requested_scales <= 0:
        raise ValueError("block_size and requested_scales must be positive")
    if len(spatial_shape) == 3:
        return int(requested_scales)
    maximum = int(requested_scales)
    while maximum > 1:
        factor = int(block_size) * (2 ** (maximum - 1))
        if all(int(size) >= factor and int(size) % factor == 0 for size in spatial_shape):
            break
        maximum -= 1
    if any(int(size) < int(block_size) for size in spatial_shape):
        raise ValueError("Every spatial dimension must be at least one MINER block")
    return maximum


def pad_to_scale_compatible(
    values: np.ndarray, *, block_size: int, scales: int
) -> tuple[np.ndarray, tuple[tuple[int, int], ...]]:
    array = np.asarray(values, dtype=np.float32)
    multiple = int(block_size) * (2 ** (int(scales) - 1))
    padding = tuple((0, (-int(size)) % multiple) for size in array.shape)
    if not any(after for _, after in padding):
        return np.ascontiguousarray(array), padding
    for size, (_, after) in zip(array.shape, padding):
        if after >= int(size):
            raise ValueError(
                "Reflect padding would be at least as large as a signal dimension; "
                "reduce scales or block_size"
            )
    return np.pad(array, padding, mode="reflect").astype(np.float32, copy=False), padding


def crop_padding(values: np.ndarray, original_shape: tuple[int, ...]) -> np.ndarray:
    slices = tuple(slice(0, int(size)) for size in original_shape)
    return np.asarray(values[slices], dtype=np.float32)


def resize_signal(values: torch.Tensor, shape: tuple[int, ...]) -> torch.Tensor:
    if tuple(values.shape) == tuple(shape):
        return values
    ndim = values.ndim
    if ndim not in {2, 3}:
        raise ValueError("resize_signal supports 2D or 3D tensors")
    source = values[None, None]
    shrinking = all(int(dst) <= int(src) for dst, src in zip(shape, values.shape))
    mode = "area" if shrinking else ("bilinear" if ndim == 2 else "trilinear")
    kwargs = {} if mode == "area" else {"align_corners": False}
    return F.interpolate(source, size=shape, mode=mode, **kwargs)[0, 0]


def build_pyramid(values: np.ndarray, scales: int) -> list[torch.Tensor]:
    finest = torch.from_numpy(np.array(values, dtype=np.float32, copy=True))
    result: list[torch.Tensor] = []
    for scale_index in range(int(scales)):
        divisor = 2 ** (int(scales) - scale_index - 1)
        shape = tuple(int(size) // divisor for size in finest.shape)
        result.append(resize_signal(finest, shape))
    return result


def blockify(values: torch.Tensor, block_size: int) -> tuple[torch.Tensor, tuple[int, ...]]:
    if values.ndim not in {2, 3}:
        raise ValueError("MINER blockify supports 2D or 3D tensors")
    if any(int(size) % int(block_size) != 0 for size in values.shape):
        raise ValueError("Signal shape must be divisible by block_size")
    grid_shape = tuple(int(size) // int(block_size) for size in values.shape)
    unfolded = values
    for dimension in range(values.ndim):
        unfolded = unfolded.unfold(dimension, int(block_size), int(block_size))
    return unfolded.contiguous().reshape(math.prod(grid_shape), -1), grid_shape


def unblockify(
    blocks: torch.Tensor, grid_shape: tuple[int, ...], block_size: int
) -> torch.Tensor:
    ndim = len(grid_shape)
    expected = math.prod(grid_shape)
    if blocks.ndim != 2 or int(blocks.shape[0]) != expected:
        raise ValueError("Block tensor does not match grid_shape")
    shaped = blocks.reshape(*grid_shape, *([int(block_size)] * ndim))
    order: list[int] = []
    for dimension in range(ndim):
        order.extend((dimension, ndim + dimension))
    return shaped.permute(order).contiguous().reshape(
        *(int(size) * int(block_size) for size in grid_shape)
    )


def local_coordinate_grid(block_size: int, dimensions: int) -> torch.Tensor:
    axes = [torch.linspace(-1.0, 1.0, int(block_size)) for _ in range(int(dimensions))]
    mesh = torch.meshgrid(*axes, indexing="ij")
    return torch.stack(mesh, dim=-1).reshape(-1, int(dimensions))
