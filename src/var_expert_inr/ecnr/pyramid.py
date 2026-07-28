from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from numpy.lib.format import open_memmap


@dataclass(frozen=True)
class PyramidScale:
    level: int
    values: np.ndarray
    time_indices: np.ndarray


def retained_time_positions(length: int) -> np.ndarray:
    length = int(length)
    if length <= 1:
        return np.array([0], dtype=np.int64)
    indices = list(range(0, length, 2))
    if indices[-1] != length - 1:
        indices.append(length - 1)
    return np.asarray(indices, dtype=np.int64)


def gaussian_kernel_1d(kernel_size: int = 5, sigma: float = 1.0) -> torch.Tensor:
    if int(kernel_size) != 5:
        raise ValueError("Formal ECNR reproduction fixes Gaussian kernel_size=5")
    radius = int(kernel_size) // 2
    axis = torch.arange(-radius, radius + 1, dtype=torch.float32)
    kernel = torch.exp(-(axis * axis) / (2.0 * float(sigma) ** 2))
    return kernel / kernel.sum()


def _blur_frame(frame: np.ndarray, *, sigma: float) -> np.ndarray:
    tensor = torch.from_numpy(np.array(frame, dtype=np.float32, copy=True)).view(1, 1, *frame.shape)
    kernel = gaussian_kernel_1d(5, sigma)
    # Apply three separable filters. Reflect is valid when a dimension is > 2;
    # replicate is the closest defined behavior for degenerate test dimensions.
    for axis in range(3):
        shape = [1, 1, 1]
        shape[axis] = 5
        weight = kernel.view(1, 1, *shape)
        pads = [0, 0, 0, 0, 0, 0]
        pad_pair = 2 - axis
        pads[2 * pad_pair] = 2
        pads[2 * pad_pair + 1] = 2
        dimension_size = int(tensor.shape[2 + axis])
        mode = "reflect" if dimension_size > 2 else "replicate"
        tensor = F.conv3d(F.pad(tensor, pads, mode=mode), weight)
    return tensor[0, 0, ::2, ::2, ::2].numpy().astype(np.float32, copy=False)


def downsample_once(
    values: np.ndarray,
    time_indices: np.ndarray,
    *,
    sigma: float = 1.0,
    output_path: str | Path | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    source = np.asarray(values)
    if source.ndim != 4:
        raise ValueError("values must have shape [T,Z,Y,X]")
    temporal_positions = retained_time_positions(source.shape[0])
    new_times = np.asarray(time_indices, dtype=np.int64)[temporal_positions]
    output_shape = (
        len(temporal_positions),
        (source.shape[1] + 1) // 2,
        (source.shape[2] + 1) // 2,
        (source.shape[3] + 1) // 2,
    )
    if output_path is None:
        output = np.empty(output_shape, dtype=np.float32)
    else:
        target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        output = open_memmap(target, mode="w+", dtype=np.float32, shape=output_shape)
    for output_index, source_index in enumerate(temporal_positions.tolist()):
        output[output_index] = _blur_frame(source[source_index], sigma=float(sigma))
    if hasattr(output, "flush"):
        output.flush()
    return output, new_times


def build_three_scale_pyramid(
    values: np.ndarray,
    *,
    sigma: float = 1.0,
    cache_dir: str | Path | None = None,
) -> list[PyramidScale]:
    source = np.asarray(values)
    time_indices = np.arange(source.shape[0], dtype=np.int64)
    scales = [PyramidScale(level=0, values=source, time_indices=time_indices)]
    current, times = source, time_indices
    for level in (1, 2):
        path = None if cache_dir is None else Path(cache_dir) / f"pyramid_scale_{level}.npy"
        current, times = downsample_once(current, times, sigma=sigma, output_path=path)
        scales.append(PyramidScale(level=level, values=current, time_indices=times))
    return scales


def upsample_to_scale(
    coarse_values: np.ndarray,
    coarse_time_indices: np.ndarray,
    *,
    fine_shape_tzyx: tuple[int, int, int, int],
    fine_time_indices: np.ndarray,
    output_path: str | Path | None = None,
) -> np.ndarray:
    coarse = np.asarray(coarse_values)
    target_t, target_z, target_y, target_x = (int(value) for value in fine_shape_tzyx)
    if output_path is None:
        output = np.empty((target_t, target_z, target_y, target_x), dtype=np.float32)
    else:
        target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        output = open_memmap(
            target,
            mode="w+",
            dtype=np.float32,
            shape=(target_t, target_z, target_y, target_x),
        )
    coarse_times = np.asarray(coarse_time_indices, dtype=np.int64)
    fine_times = np.asarray(fine_time_indices, dtype=np.int64)
    spatial_cache: dict[int, np.ndarray] = {}

    def spatial_frame(index: int) -> np.ndarray:
        index = int(index)
        if index not in spatial_cache:
            tensor = torch.from_numpy(
                np.array(coarse[index], dtype=np.float32, copy=True)
            ).view(1, 1, *coarse.shape[1:])
            spatial_cache[index] = F.interpolate(
                tensor,
                size=(target_z, target_y, target_x),
                mode="trilinear",
                align_corners=False,
            )[0, 0].numpy()
        # Only adjacent temporal frames are needed; bound memory to two frames.
        for cached in list(spatial_cache):
            if cached not in {index, index - 1, index + 1}:
                del spatial_cache[cached]
        return spatial_cache[index]

    for output_index, time_value in enumerate(fine_times.tolist()):
        right = int(np.searchsorted(coarse_times, time_value, side="left"))
        if right == 0:
            output[output_index] = spatial_frame(0)
        elif right >= len(coarse_times):
            output[output_index] = spatial_frame(len(coarse_times) - 1)
        elif coarse_times[right] == time_value:
            output[output_index] = spatial_frame(right)
        else:
            left = right - 1
            denominator = int(coarse_times[right] - coarse_times[left])
            alpha = float(time_value - coarse_times[left]) / float(denominator)
            output[output_index] = (
                (1.0 - alpha) * spatial_frame(left) + alpha * spatial_frame(right)
            )
    if hasattr(output, "flush"):
        output.flush()
    return output
