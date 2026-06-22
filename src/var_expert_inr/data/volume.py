from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import torch

from ..config.schema import VolumeShape
from .base import (
    DatasetMeta,
    FieldBatch,
    FieldDataset,
    ensure_pre_normalized_range,
    infer_volume_shape,
    normalize_index_coordinates,
    peek_array,
    target_dim_from_array,
)


def _flatten_volume(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 5:
        return arr.reshape(-1, arr.shape[-1])
    if arr.ndim == 4:
        return arr.reshape(-1, 1)
    if arr.ndim == 2:
        return arr
    if arr.ndim == 1:
        return arr.reshape(-1, 1)
    raise ValueError(f"Unsupported target shape: {arr.shape}")


class VolumeFieldDataset(FieldDataset):
    def __init__(
        self,
        *,
        target_path: str | None = None,
        targets: dict[str, str] | None = None,
        target_dir: str | None = None,
        volume_shape: VolumeShape | None = None,
    ) -> None:
        if target_path is None and not targets and target_dir is None:
            raise ValueError("VolumeFieldDataset requires target_path, targets, or target_dir")
        self.target_path = target_path
        self.targets_map = dict(targets or {})
        self.target_dir = target_dir

        if target_dir is not None and not self.targets_map and target_path is None:
            target_root = Path(target_dir)
            self.targets_map = {
                path.stem.replace("target_", "", 1): str(path)
                for path in sorted(target_root.glob("*.npy"))
            }
            if not self.targets_map:
                raise ValueError(f"No target files found in {target_dir}")

        if target_path is not None:
            arr = peek_array(target_path)
            self._target_names = ("target",)
            self._targets_np = {"target": arr}
        else:
            self._target_names = tuple(sorted(self.targets_map.keys()))
            self._targets_np = {name: peek_array(path) for name, path in self.targets_map.items()}

        first = self._targets_np[self._target_names[0]]
        self.volume_shape = infer_volume_shape(first, volume_shape)
        for name, arr in self._targets_np.items():
            shape = infer_volume_shape(arr, self.volume_shape)
            if shape != self.volume_shape:
                raise ValueError(f"Volume shape mismatch for {name}: {shape} vs {self.volume_shape}")

        self.meta = DatasetMeta(
            kind="volume",
            n_samples=int(self.volume_shape.N),
            input_dim=4,
            target_names=self._target_names,
            target_dims={name: target_dim_from_array(arr) for name, arr in self._targets_np.items()},
            volume_shape=self.volume_shape,
        )

        self._targets_flat = {name: _flatten_volume(arr) for name, arr in self._targets_np.items()}
        for name, flat in self._targets_flat.items():
            ensure_pre_normalized_range(flat, label=f"target '{name}'")

    def _indices_to_coords(self, rows: np.ndarray) -> np.ndarray:
        x = rows % self.volume_shape.X
        rows = rows // self.volume_shape.X
        y = rows % self.volume_shape.Y
        rows = rows // self.volume_shape.Y
        z = rows % self.volume_shape.Z
        rows = rows // self.volume_shape.Z
        t = rows
        return np.stack(
            [
                normalize_index_coordinates(x, self.volume_shape.X),
                normalize_index_coordinates(y, self.volume_shape.Y),
                normalize_index_coordinates(z, self.volume_shape.Z),
                normalize_index_coordinates(t, self.volume_shape.T),
            ],
            axis=1,
        ).astype(np.float32)

    def fetch_batch(
        self,
        indices: Iterable[int],
        *,
        include_targets: bool = True,
        assignments: np.ndarray | None = None,
    ) -> FieldBatch:
        rows = np.asarray(list(indices), dtype=np.int64)
        coords = self._indices_to_coords(rows)
        xb = torch.from_numpy(coords)

        expert_ids = None
        if assignments is not None:
            if assignments.shape == (self.volume_shape.T,):
                per_timestep = self.volume_shape.X * self.volume_shape.Y * self.volume_shape.Z
                time_ids = (rows // per_timestep).astype(np.int64)
                expert_ids = torch.from_numpy(np.asarray(assignments[time_ids], dtype=np.int64))
            else:
                per_timestep = self.volume_shape.X * self.volume_shape.Y * self.volume_shape.Z
                voxels = (rows % per_timestep).astype(np.int64)
                expert_ids = torch.from_numpy(np.asarray(assignments[voxels], dtype=np.int64))

        if not include_targets:
            return FieldBatch(indices=torch.from_numpy(rows), coords=xb, expert_ids=expert_ids)

        if len(self._target_names) == 1:
            name = self._target_names[0]
            arr = np.asarray(self._targets_flat[name][rows], dtype=np.float32)
            yb: torch.Tensor | dict[str, torch.Tensor] = torch.from_numpy(arr)
        else:
            yb = {}
            for name in self._target_names:
                arr = np.asarray(self._targets_flat[name][rows], dtype=np.float32)
                yb[name] = torch.from_numpy(arr)
        return FieldBatch(indices=torch.from_numpy(rows), coords=xb, targets=yb, expert_ids=expert_ids)

    def load_targets_flat(self) -> dict[str, np.ndarray]:
        return {name: np.asarray(flat, dtype=np.float32) for name, flat in self._targets_flat.items()}

    def reshape_flat_predictions(self, name: str, flat_values: np.ndarray) -> np.ndarray:
        dims = self.meta.target_dims[name]
        if dims == 1:
            return flat_values.reshape(
                self.volume_shape.T,
                self.volume_shape.Z,
                self.volume_shape.Y,
                self.volume_shape.X,
            )
        return flat_values.reshape(
            self.volume_shape.T,
            self.volume_shape.Z,
            self.volume_shape.Y,
            self.volume_shape.X,
            dims,
        )
