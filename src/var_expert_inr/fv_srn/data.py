from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


def _normalize_array(array: np.ndarray) -> np.ndarray:
    if array.ndim == 5 and array.shape[-1] == 1:
        return array[..., 0]
    if array.ndim == 2 and array.shape[1] == 1:
        return array[:, 0]
    return array


class TemporalVolume:
    def __init__(self, path: str | Path, shape: dict[str, int]) -> None:
        self.path = str(path)
        self.shape = {axis: int(shape[axis]) for axis in ("T", "Z", "Y", "X")}
        raw = np.load(self.path, mmap_mode="r")
        normalized = _normalize_array(raw)
        expected = tuple(self.shape[axis] for axis in ("T", "Z", "Y", "X"))
        expected_size = int(np.prod(expected, dtype=np.int64))
        if normalized.ndim == 1 and normalized.size == expected_size:
            normalized = normalized.reshape(expected)
        if tuple(normalized.shape) != expected:
            raise ValueError(
                f"Target shape mismatch for {self.path}: expected {expected} "
                f"({expected_size} values), got {tuple(normalized.shape)} ({normalized.size} values)"
            )
        if not np.issubdtype(normalized.dtype, np.floating):
            raise TypeError(f"fV-SRN target must be floating point, got {normalized.dtype}")
        self.array = normalized
        self.raw_bytes = expected_size * int(normalized.dtype.itemsize)
        # NPY files in this project are pre-normalized. Sample each frame first,
        # then rely on training/evaluation reads to avoid an eager multi-GB scan.
        for t in range(self.shape["T"]):
            frame = self.array[t]
            if float(np.min(frame)) < -1.00001 or float(np.max(frame)) > 1.00001:
                raise ValueError(f"Target values must be pre-normalized to [-1,1], violation at t={t}")

    @property
    def spatial_shape(self) -> tuple[int, int, int]:
        return (self.shape["Z"], self.shape["Y"], self.shape["X"])

    def frame(self, timestep: int) -> np.ndarray:
        return self.array[int(timestep)]

    def sample(self, timestep: int, coords: np.ndarray) -> np.ndarray:
        """Trilinear sampling at XYZ coordinates in [0,1]."""
        xyz = np.asarray(coords, dtype=np.float32)
        if xyz.ndim != 2 or xyz.shape[1] != 3:
            raise ValueError(f"coords must have shape [N,3], got {xyz.shape}")
        xyz = np.clip(xyz, 0.0, 1.0)
        zyx_size = np.array([self.shape["X"], self.shape["Y"], self.shape["Z"]], dtype=np.int64)
        scaled = xyz * (zyx_size.astype(np.float32) - 1.0)
        low = np.floor(scaled).astype(np.int64)
        high = np.minimum(low + 1, zyx_size - 1)
        frac = scaled - low
        x0, y0, z0 = low.T
        x1, y1, z1 = high.T
        fx, fy, fz = frac.T
        frame = self.frame(timestep)
        c000 = np.asarray(frame[z0, y0, x0], dtype=np.float32)
        c100 = np.asarray(frame[z0, y0, x1], dtype=np.float32)
        c010 = np.asarray(frame[z0, y1, x0], dtype=np.float32)
        c110 = np.asarray(frame[z0, y1, x1], dtype=np.float32)
        c001 = np.asarray(frame[z1, y0, x0], dtype=np.float32)
        c101 = np.asarray(frame[z1, y0, x1], dtype=np.float32)
        c011 = np.asarray(frame[z1, y1, x0], dtype=np.float32)
        c111 = np.asarray(frame[z1, y1, x1], dtype=np.float32)
        c00 = c000 * (1 - fx) + c100 * fx
        c10 = c010 * (1 - fx) + c110 * fx
        c01 = c001 * (1 - fx) + c101 * fx
        c11 = c011 * (1 - fx) + c111 * fx
        c0 = c00 * (1 - fy) + c10 * fy
        c1 = c01 * (1 - fy) + c11 * fy
        return (c0 * (1 - fz) + c1 * fz).reshape(-1, 1).astype(np.float32)

    def full_coords(self, start: int, stop: int) -> np.ndarray:
        z_size, y_size, x_size = self.spatial_shape
        rows = np.arange(int(start), int(stop), dtype=np.int64)
        x = rows % x_size
        rows //= x_size
        y = rows % y_size
        z = rows // y_size
        return np.stack(
            [
                x / max(x_size - 1, 1),
                y / max(y_size - 1, 1),
                z / max(z_size - 1, 1),
            ],
            axis=1,
        ).astype(np.float32)


@dataclass
class SamplePool:
    coords: list[np.ndarray]
    targets: list[np.ndarray]
    train_indices: list[np.ndarray]
    val_indices: list[np.ndarray]

    def state_dict(self) -> dict:
        return {
            "coords": self.coords,
            "targets": self.targets,
            "train_indices": self.train_indices,
            "val_indices": self.val_indices,
        }

    @classmethod
    def from_state_dict(cls, state: dict) -> "SamplePool":
        return cls(**state)


def _rejection_sample(
    volume: TemporalVolume,
    timestep: int,
    count: int,
    rng: np.random.Generator,
    floor: float,
    error_grid: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    accepted_coords: list[np.ndarray] = []
    accepted_values: list[np.ndarray] = []
    remaining = int(count)
    while remaining > 0:
        candidate_count = max(remaining * 2, 1024)
        coords = rng.random((candidate_count, 3), dtype=np.float32)
        values = volume.sample(timestep, coords)
        if error_grid is None:
            score = np.clip((values[:, 0] + 1.0) * 0.5, 0.0, 1.0)
        else:
            grid_size = error_grid.shape[0]
            indices = np.minimum((coords * grid_size).astype(np.int64), grid_size - 1)
            score = error_grid[indices[:, 2], indices[:, 1], indices[:, 0]]
        probability = float(floor) + (1.0 - float(floor)) * score
        mask = rng.random(candidate_count) <= probability
        if np.any(mask):
            take = min(remaining, int(mask.sum()))
            accepted_coords.append(coords[mask][:take])
            accepted_values.append(values[mask][:take])
            remaining -= take
    return np.concatenate(accepted_coords), np.concatenate(accepted_values)


def build_sample_pool(
    volume: TemporalVolume,
    *,
    count_per_timestep: int,
    validation_fraction: float,
    floor: float,
    rng: np.random.Generator,
    error_grids: list[np.ndarray] | None = None,
) -> SamplePool:
    coords_all, targets_all, train_all, val_all = [], [], [], []
    for t in range(volume.shape["T"]):
        coords, targets = _rejection_sample(
            volume, t, count_per_timestep, rng, floor,
            None if error_grids is None else error_grids[t],
        )
        order = rng.permutation(count_per_timestep)
        val_count = int(round(count_per_timestep * validation_fraction))
        coords_all.append(coords)
        targets_all.append(targets)
        val_all.append(order[:val_count])
        train_all.append(order[val_count:])
    return SamplePool(coords_all, targets_all, train_all, val_all)
