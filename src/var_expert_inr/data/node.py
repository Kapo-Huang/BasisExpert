from __future__ import annotations

from typing import Iterable

import numpy as np
import torch

from .base import (
    DatasetMeta,
    FieldBatch,
    FieldDataset,
    ensure_pre_normalized_range,
    peek_array,
    target_dim_from_array,
)


class NodeFieldDataset(FieldDataset):
    def __init__(
        self,
        *,
        coords_path: str,
        target_path: str | None = None,
        targets: dict[str, str] | None = None,
    ) -> None:
        if target_path is None and not targets:
            raise ValueError("NodeFieldDataset requires target_path or targets")
        self.coords_path = coords_path
        self.target_path = target_path
        self.targets_map = dict(targets or {})

        coords_np = peek_array(coords_path)
        if coords_np.ndim != 2:
            raise ValueError(f"coords_path must contain a 2D array, got {coords_np.shape}")
        ensure_pre_normalized_range(coords_np, label=f"coords_path '{coords_path}'")
        self._coords_np = coords_np

        if target_path is not None:
            arr = peek_array(target_path)
            self._target_names = ("target",)
            self._targets_np = {"target": arr}
        else:
            self._target_names = tuple(sorted(self.targets_map.keys()))
            self._targets_np = {name: peek_array(path) for name, path in self.targets_map.items()}

        for name, arr in self._targets_np.items():
            if int(arr.shape[0]) != int(coords_np.shape[0]):
                raise ValueError(
                    f"Target sample count mismatch for {name}: {arr.shape[0]} vs {coords_np.shape[0]}"
                )
            ensure_pre_normalized_range(arr, label=f"target '{name}'")

        self.meta = DatasetMeta(
            kind="node",
            n_samples=int(coords_np.shape[0]),
            input_dim=int(coords_np.shape[1]),
            target_names=self._target_names,
            target_dims={name: target_dim_from_array(arr) for name, arr in self._targets_np.items()},
            volume_shape=None,
        )

    def _target_tensor(self, name: str, rows: np.ndarray) -> torch.Tensor:
        arr = np.asarray(self._targets_np[name][rows], dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        return torch.from_numpy(arr)

    def fetch_batch(
        self,
        indices: Iterable[int],
        *,
        include_targets: bool = True,
        assignments: np.ndarray | None = None,
    ) -> FieldBatch:
        rows = np.asarray(list(indices), dtype=np.int64)
        coords = np.asarray(self._coords_np[rows], dtype=np.float32)
        xb = torch.from_numpy(coords)

        expert_ids = None
        if assignments is not None:
            expert_ids = torch.from_numpy(np.asarray(assignments[rows], dtype=np.int64))

        if not include_targets:
            return FieldBatch(indices=torch.from_numpy(rows), coords=xb, expert_ids=expert_ids)

        if len(self._target_names) == 1:
            target_name = self._target_names[0]
            yb: torch.Tensor | dict[str, torch.Tensor] = self._target_tensor(target_name, rows)
        else:
            yb = {name: self._target_tensor(name, rows) for name in self._target_names}
        return FieldBatch(indices=torch.from_numpy(rows), coords=xb, targets=yb, expert_ids=expert_ids)

    def load_targets_flat(self) -> dict[str, np.ndarray]:
        flat: dict[str, np.ndarray] = {}
        for name in self._target_names:
            arr = np.asarray(self._targets_np[name], dtype=np.float32)
            if arr.ndim == 1:
                arr = arr.reshape(-1, 1)
            flat[name] = arr
        return flat

    def reshape_flat_predictions(self, name: str, flat_values: np.ndarray) -> np.ndarray:
        return np.asarray(flat_values)

    def pretrain_assignment_kind(self) -> str:
        return "sample"

    def sample_cluster_features(self) -> np.ndarray:
        blocks = []
        for name in self._target_names:
            arr = np.asarray(self._targets_np[name], dtype=np.float32)
            if arr.ndim == 1:
                arr = arr.reshape(-1, 1)
            blocks.append(arr)
        return np.concatenate(blocks, axis=1)

    def raw_coords(self) -> np.ndarray:
        return np.asarray(self._coords_np, dtype=np.float32)
