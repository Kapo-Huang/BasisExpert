from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Mapping

import numpy as np
import torch
import torch.nn.functional as F

from ..data.base import ensure_pre_normalized_range


def _spatial_shape_from_volume_shape(volume_shape: Mapping[str, int]) -> tuple[int, int, int]:
    return (
        int(volume_shape["Z"]),
        int(volume_shape["Y"]),
        int(volume_shape["X"]),
    )


def _expected_full_shape(volume_shape: Mapping[str, int]) -> tuple[int, int, int, int]:
    spatial = _spatial_shape_from_volume_shape(volume_shape)
    return (int(volume_shape["T"]), spatial[0], spatial[1], spatial[2])


def _normalize_loaded_array(array: np.ndarray) -> np.ndarray:
    if array.ndim == 5 and int(array.shape[-1]) == 1:
        return array[..., 0]
    return array


def make_coord_grid(
    shape: tuple[int, int, int],
    device: torch.device | str,
    *,
    flatten: bool = True,
    align_corners: bool = True,
) -> torch.Tensor:
    coord_seqs = []
    for axis_size in shape:
        if axis_size <= 1:
            seq = torch.zeros((axis_size,), device=device, dtype=torch.float32)
        elif align_corners:
            seq = torch.linspace(-1.0, 1.0, steps=int(axis_size), device=device, dtype=torch.float32)
        else:
            step = 2.0 / float(int(axis_size) + 1)
            seq = (-1.0 + step) + step * torch.arange(int(axis_size), device=device, dtype=torch.float32)
        coord_seqs.append(seq)
    grid = torch.meshgrid(*coord_seqs, indexing="ij")
    coords = torch.stack(grid, dim=-1)
    if flatten:
        coords = coords.view(-1, coords.shape[-1])
    return coords.flip(-1)


@dataclass(frozen=True)
class TimestepMetrics:
    mse: float
    mae: float
    psnr: float

    def to_dict(self) -> dict[str, float]:
        return {
            "mse": float(self.mse),
            "mae": float(self.mae),
            "psnr": float(self.psnr),
        }


class IonizationTargetReader:
    def __init__(self, target_path: str | Path, volume_shape: Mapping[str, int]) -> None:
        self.target_path = str(target_path)
        self.volume_shape = {
            "X": int(volume_shape["X"]),
            "Y": int(volume_shape["Y"]),
            "Z": int(volume_shape["Z"]),
            "T": int(volume_shape["T"]),
        }
        loaded = np.load(self.target_path, mmap_mode="r")
        self._array = _normalize_loaded_array(loaded)
        expected_shape = _expected_full_shape(self.volume_shape)
        if self._array.ndim != 4:
            raise ValueError(
                f"APMGSRN ionization target must have shape (T, Z, Y, X), got {self._array.shape}"
            )
        if tuple(int(value) for value in self._array.shape) != expected_shape:
            raise ValueError(
                f"DATA.volume_shape does not match target array shape: expected {expected_shape}, got {tuple(self._array.shape)}"
            )
        ensure_pre_normalized_range(self._array, label=f"target_path '{self.target_path}'")

    @property
    def time_count(self) -> int:
        return int(self.volume_shape["T"])

    @property
    def spatial_shape(self) -> tuple[int, int, int]:
        return _spatial_shape_from_volume_shape(self.volume_shape)

    @property
    def dtype(self) -> np.dtype:
        return np.dtype(self._array.dtype)

    @property
    def itemsize(self) -> int:
        return int(self.dtype.itemsize)

    def timestep_array(self, time_index: int) -> np.ndarray:
        time_index = int(time_index)
        if time_index < 0 or time_index >= self.time_count:
            raise IndexError(f"time_index must be in [0, {self.time_count - 1}], got {time_index}")
        return np.array(self._array[time_index], dtype=np.float32, copy=True)

    def raw_bytes_for_indices(self, time_indices: list[int]) -> int:
        values_per_timestep = int(np.prod(self.spatial_shape, dtype=np.int64))
        return int(values_per_timestep * len(time_indices) * self.itemsize)


class IonizationTimestepDataset:
    def __init__(
        self,
        reader: IonizationTargetReader,
        *,
        time_index: int,
        align_corners: bool = True,
        device: str | torch.device = "cpu",
    ) -> None:
        self.reader = reader
        self.time_index = int(time_index)
        self.align_corners = bool(align_corners)
        self.device = torch.device(device)
        self.slice_np = reader.timestep_array(self.time_index)
        self.data = torch.from_numpy(self.slice_np).unsqueeze(0).unsqueeze(0).to(self.device)
        self.data_min = float(self.slice_np.min())
        self.data_max = float(self.slice_np.max())

    @property
    def spatial_shape(self) -> tuple[int, int, int]:
        return self.reader.spatial_shape

    @property
    def n_voxels(self) -> int:
        return int(np.prod(self.spatial_shape, dtype=np.int64))

    def sample_points(self, coords: torch.Tensor) -> torch.Tensor:
        coords = coords.to(self.device, dtype=torch.float32)
        if coords.ndim != 2 or int(coords.shape[1]) != 3:
            raise ValueError(f"coords must have shape [N, 3], got {tuple(coords.shape)}")
        grid = coords.view(1, 1, 1, coords.shape[0], 3)
        sampled = F.grid_sample(
            self.data,
            grid,
            mode="bilinear",
            align_corners=self.align_corners,
        )
        return sampled.view(1, coords.shape[0]).transpose(0, 1)

    def sample_random_points(self, n_points: int) -> tuple[torch.Tensor, torch.Tensor]:
        coords = torch.rand((int(n_points), 3), device=self.device, dtype=torch.float32) * 2.0 - 1.0
        return coords, self.sample_points(coords)

    @cached_property
    def full_coords_cpu(self) -> torch.Tensor:
        return make_coord_grid(self.spatial_shape, device=torch.device("cpu"), flatten=True, align_corners=self.align_corners)

    def reshape_flat_predictions(self, flat_values: np.ndarray) -> np.ndarray:
        array = np.asarray(flat_values, dtype=np.float32)
        if array.ndim == 2:
            if int(array.shape[1]) != 1:
                raise ValueError(f"Expected scalar flat predictions with shape [N, 1], got {array.shape}")
            array = array[:, 0]
        if int(array.size) != self.n_voxels:
            raise ValueError(f"Expected {self.n_voxels} flat values, got {array.size}")
        return array.reshape(self.spatial_shape)

    def reconstruct(
        self,
        model: torch.nn.Module,
        *,
        batch_size: int,
        model_device: torch.device,
    ) -> np.ndarray:
        batch_size = max(1, int(batch_size))
        outputs = np.empty((self.n_voxels, 1), dtype=np.float32)
        model.eval()
        with torch.no_grad():
            for start in range(0, self.n_voxels, batch_size):
                stop = min(start + batch_size, self.n_voxels)
                batch_coords = self.full_coords_cpu[start:stop].to(model_device, non_blocking=True)
                outputs[start:stop] = model(batch_coords).detach().cpu().numpy().astype(np.float32, copy=False)
        return self.reshape_flat_predictions(outputs)

    def target_flat(self) -> np.ndarray:
        return self.slice_np.reshape(-1, 1).astype(np.float32, copy=False)

    def target_array(self) -> np.ndarray:
        return np.asarray(self.slice_np, dtype=np.float32)
