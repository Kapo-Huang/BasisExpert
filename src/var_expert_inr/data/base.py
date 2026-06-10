from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch
from torch.utils.data import Dataset

from ..config.schema import VolumeShape

PRE_NORMALIZED_RANGE_MIN = -1.0
PRE_NORMALIZED_RANGE_MAX = 1.0
PRE_NORMALIZED_RANGE_ATOL = 1e-6
DEFAULT_VALIDATION_CHUNK_VALUES = 1_000_000


def peek_array(path: str) -> np.ndarray:
    return np.load(path, mmap_mode="r")


def target_dim_from_array(arr: np.ndarray) -> int:
    if arr.ndim in {2, 5}:
        return int(arr.shape[-1])
    if arr.ndim in {1, 4}:
        return 1
    raise ValueError(f"Unsupported target shape: {arr.shape}")


def infer_volume_shape(arr: np.ndarray, supplied: VolumeShape | None) -> VolumeShape:
    if supplied is not None:
        return supplied
    if arr.ndim not in {4, 5}:
        raise ValueError("volume_shape is required for flat targets")
    return VolumeShape(
        X=int(arr.shape[3]),
        Y=int(arr.shape[2]),
        Z=int(arr.shape[1]),
        T=int(arr.shape[0]),
    )


def _iter_validation_chunks(arr: np.ndarray, *, max_values: int = DEFAULT_VALIDATION_CHUNK_VALUES):
    if arr.ndim == 0:
        yield np.asarray(arr, dtype=np.float32).reshape(1)
        return
    trailing = int(np.prod(arr.shape[1:], dtype=np.int64)) if arr.ndim > 1 else 1
    rows_per_chunk = max(1, int(max_values) // max(trailing, 1))
    for start in range(0, int(arr.shape[0]), rows_per_chunk):
        stop = min(start + rows_per_chunk, int(arr.shape[0]))
        yield np.asarray(arr[start:stop], dtype=np.float32)


def ensure_pre_normalized_range(
    arr: np.ndarray,
    *,
    label: str,
    min_value: float = PRE_NORMALIZED_RANGE_MIN,
    max_value: float = PRE_NORMALIZED_RANGE_MAX,
    atol: float = PRE_NORMALIZED_RANGE_ATOL,
) -> None:
    observed_min = float("inf")
    observed_max = float("-inf")
    for chunk in _iter_validation_chunks(arr):
        if not np.isfinite(chunk).all():
            raise ValueError(f"{label} contains NaN or Inf values.")
        observed_min = min(observed_min, float(chunk.min()))
        observed_max = max(observed_max, float(chunk.max()))
    if observed_min < float(min_value) - float(atol) or observed_max > float(max_value) + float(atol):
        raise ValueError(
            f"{label} must already be scaled into [-1, 1], "
            f"but observed range was [{observed_min:.6g}, {observed_max:.6g}]."
        )
def normalize_index_coordinates(values: np.ndarray, size: int) -> np.ndarray:
    coords = np.asarray(values, dtype=np.float32)
    if int(size) <= 1:
        return np.zeros_like(coords, dtype=np.float32)
    return (2.0 * coords / float(int(size) - 1)) - 1.0


@dataclass(frozen=True)
class DatasetMeta:
    kind: str
    n_samples: int
    input_dim: int
    target_names: tuple[str, ...]
    target_dims: dict[str, int]
    volume_shape: VolumeShape | None = None

    @property
    def is_multitarget(self) -> bool:
        return len(self.target_names) > 1


@dataclass
class FieldBatch:
    indices: torch.Tensor
    coords: torch.Tensor
    targets: torch.Tensor | dict[str, torch.Tensor] | None = None
    expert_ids: torch.Tensor | None = None


class FieldDataset(Dataset[int], ABC):
    meta: DatasetMeta

    def __len__(self) -> int:
        return int(self.meta.n_samples)

    def __getitem__(self, idx: int) -> int:
        return int(idx)

    def target_names(self) -> tuple[str, ...]:
        return self.meta.target_names

    def view_specs(self) -> dict[str, int]:
        return dict(self.meta.target_dims)

    @abstractmethod
    def fetch_batch(
        self,
        indices: Iterable[int],
        *,
        include_targets: bool = True,
        assignments: np.ndarray | None = None,
    ) -> FieldBatch:
        raise NotImplementedError

    @abstractmethod
    def load_targets_flat(self) -> dict[str, np.ndarray]:
        raise NotImplementedError

    @abstractmethod
    def reshape_flat_predictions(self, name: str, flat_values: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def pretrain_assignment_kind(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def sample_cluster_features(self) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def raw_coords(self) -> np.ndarray:
        raise NotImplementedError
