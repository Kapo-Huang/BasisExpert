from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np

from ...data.base import ensure_pre_normalized_range


class ScalarVolumeReader:
    def __init__(self, path: str | Path, shape: Mapping[str, int]) -> None:
        self.path = str(path)
        self.shape_tzyx = tuple(int(shape[axis]) for axis in ("T", "Z", "Y", "X"))
        raw = np.load(self.path, mmap_mode="r", allow_pickle=False)
        expected_size = int(np.prod(self.shape_tzyx, dtype=np.int64))
        normalized = raw
        if normalized.ndim == 5 and normalized.shape[-1] == 1:
            normalized = normalized[..., 0]
        elif normalized.ndim == 2 and normalized.shape[1] == 1:
            normalized = normalized[:, 0]
        if normalized.ndim == 1 and normalized.size == expected_size:
            normalized = normalized.reshape(self.shape_tzyx)
        if tuple(normalized.shape) != self.shape_tzyx:
            raise ValueError(
                f"MINER target shape mismatch: expected {self.shape_tzyx}, "
                f"got {tuple(normalized.shape)}"
            )
        if not np.issubdtype(normalized.dtype, np.floating):
            raise TypeError("MINER targets must be floating point")
        ensure_pre_normalized_range(normalized, label=f"target_path '{self.path}'")
        self.array = normalized
        _, z_size, y_size, x_size = self.shape_tzyx
        if z_size == 1 and y_size > 1 and x_size > 1:
            self.spatial_dimensions = 2
            self.spatial_shape = (y_size, x_size)
        elif z_size > 1 and y_size > 1 and x_size > 1:
            self.spatial_dimensions = 3
            self.spatial_shape = (z_size, y_size, x_size)
        else:
            raise ValueError(
                "MINER requires either Z=1 with 2D Y/X data or fully 3D Z/Y/X data"
            )

    @property
    def timesteps(self) -> int:
        return int(self.shape_tzyx[0])

    def timestep(self, index: int) -> np.ndarray:
        if not 0 <= int(index) < self.timesteps:
            raise IndexError(f"Timestep {index} is outside [0,{self.timesteps})")
        frame = np.asarray(self.array[int(index)], dtype=np.float32)
        return np.ascontiguousarray(frame[0] if self.spatial_dimensions == 2 else frame)

    def restore_storage_shape(self, frame: np.ndarray) -> np.ndarray:
        result = np.asarray(frame, dtype=np.float32)
        return result[None, ...] if self.spatial_dimensions == 2 else result

    def raw_bytes(self, time_indices: list[int] | tuple[int, ...]) -> int:
        per_frame = int(np.prod(self.shape_tzyx[1:], dtype=np.int64))
        return per_frame * len(time_indices) * int(self.array.dtype.itemsize)
