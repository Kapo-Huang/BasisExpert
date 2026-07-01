from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from ..config.schema import VolumeShape
from ..data.base import ensure_pre_normalized_range, normalize_index_coordinates, peek_array


@dataclass(frozen=True)
class BlockShape:
    sx: int
    sy: int
    sz: int

    def __post_init__(self) -> None:
        if int(self.sx) <= 0 or int(self.sy) <= 0 or int(self.sz) <= 0:
            raise ValueError(f"BlockShape values must be positive, got {(self.sx, self.sy, self.sz)}")

    @property
    def voxel_count(self) -> int:
        return int(self.sx) * int(self.sy) * int(self.sz)

    def to_dict(self) -> dict[str, int]:
        return {"sx": int(self.sx), "sy": int(self.sy), "sz": int(self.sz)}


@dataclass(frozen=True)
class BlockGridShape:
    gx: int
    gy: int
    gz: int

    def __post_init__(self) -> None:
        if int(self.gx) <= 0 or int(self.gy) <= 0 or int(self.gz) <= 0:
            raise ValueError(f"BlockGridShape values must be positive, got {(self.gx, self.gy, self.gz)}")

    @property
    def n_blocks(self) -> int:
        return int(self.gx) * int(self.gy) * int(self.gz)

    def to_dict(self) -> dict[str, int]:
        return {"gx": int(self.gx), "gy": int(self.gy), "gz": int(self.gz)}


def block_grid_shape_for_volume(volume_shape: VolumeShape, block_shape: BlockShape) -> BlockGridShape:
    if int(volume_shape.X) % int(block_shape.sx) != 0:
        raise ValueError(
            f"Block shape sx={block_shape.sx} must divide volume X={int(volume_shape.X)} exactly"
        )
    if int(volume_shape.Y) % int(block_shape.sy) != 0:
        raise ValueError(
            f"Block shape sy={block_shape.sy} must divide volume Y={int(volume_shape.Y)} exactly"
        )
    if int(volume_shape.Z) % int(block_shape.sz) != 0:
        raise ValueError(
            f"Block shape sz={block_shape.sz} must divide volume Z={int(volume_shape.Z)} exactly"
        )
    return BlockGridShape(
        gx=int(volume_shape.X) // int(block_shape.sx),
        gy=int(volume_shape.Y) // int(block_shape.sy),
        gz=int(volume_shape.Z) // int(block_shape.sz),
    )


def block_shape_from_payload(payload: dict[str, int]) -> BlockShape:
    return BlockShape(
        sx=int(payload["sx"]),
        sy=int(payload["sy"]),
        sz=int(payload["sz"]),
    )


def block_grid_shape_from_payload(payload: dict[str, int]) -> BlockGridShape:
    return BlockGridShape(
        gx=int(payload["gx"]),
        gy=int(payload["gy"]),
        gz=int(payload["gz"]),
    )


class DCTargetVolume:
    def __init__(
        self,
        *,
        target_path: str | Path,
        target_name: str,
        volume_shape: VolumeShape,
    ) -> None:
        self.target_path = str(target_path)
        self.target_name = str(target_name)
        self.volume_shape = volume_shape

        raw = peek_array(self.target_path)
        normalized = self._normalize_loaded_array(raw)
        expected_shape = (
            int(self.volume_shape.T),
            int(self.volume_shape.Z),
            int(self.volume_shape.Y),
            int(self.volume_shape.X),
        )
        if normalized.ndim == 1 and int(normalized.size) == int(self.volume_shape.N):
            normalized = normalized.reshape(expected_shape)
        if normalized.shape != expected_shape:
            raise ValueError(
                f"Target shape mismatch for {self.target_path}: expected {expected_shape}, got {normalized.shape}"
            )
        ensure_pre_normalized_range(normalized, label=f"target_path '{self.target_path}'")
        self.array = normalized

    @staticmethod
    def _normalize_loaded_array(raw: np.ndarray) -> np.ndarray:
        if raw.ndim == 5:
            if int(raw.shape[-1]) != 1:
                raise ValueError(f"DC-INR only supports scalar targets, got shape={raw.shape}")
            return raw[..., 0]
        if raw.ndim == 4:
            return raw
        if raw.ndim == 2:
            if int(raw.shape[1]) != 1:
                raise ValueError(f"DC-INR only supports scalar targets, got shape={raw.shape}")
            return raw[:, 0]
        return raw

    @property
    def dtype(self) -> np.dtype:
        return np.dtype(self.array.dtype)

    @property
    def itemsize(self) -> int:
        return int(self.dtype.itemsize)

    @property
    def raw_bytes(self) -> int:
        return int(self.volume_shape.N) * int(self.itemsize)

    @property
    def shape_tzyx(self) -> tuple[int, int, int, int]:
        return (
            int(self.volume_shape.T),
            int(self.volume_shape.Z),
            int(self.volume_shape.Y),
            int(self.volume_shape.X),
        )

    def array_tzyx(self) -> np.ndarray:
        if self.array.ndim == 4:
            return self.array
        return np.asarray(self.array, dtype=np.float32).reshape(self.shape_tzyx)

    def block_view(self, block_shape: BlockShape) -> np.ndarray:
        array = self.array_tzyx()
        grid_shape = block_grid_shape_for_volume(self.volume_shape, block_shape)
        view = array.reshape(
            int(self.volume_shape.T),
            int(grid_shape.gz),
            int(block_shape.sz),
            int(grid_shape.gy),
            int(block_shape.sy),
            int(grid_shape.gx),
            int(block_shape.sx),
        )
        transposed = view.transpose(1, 3, 5, 0, 2, 4, 6)
        return transposed.reshape(
            int(grid_shape.n_blocks),
            int(self.volume_shape.T),
            int(block_shape.sz),
            int(block_shape.sy),
            int(block_shape.sx),
        )


def block_id_to_grid_indices(block_id: int, grid_shape: BlockGridShape) -> tuple[int, int, int]:
    block_id = int(block_id)
    bx = block_id % int(grid_shape.gx)
    rem = block_id // int(grid_shape.gx)
    by = rem % int(grid_shape.gy)
    bz = rem // int(grid_shape.gy)
    return bx, by, bz


def local_spatial_coords(block_shape: BlockShape) -> np.ndarray:
    voxel_count = int(block_shape.voxel_count)
    indices = np.arange(voxel_count, dtype=np.int64)
    x = indices % int(block_shape.sx)
    rem = indices // int(block_shape.sx)
    y = rem % int(block_shape.sy)
    z = rem // int(block_shape.sy)
    return np.stack(
        [
            normalize_index_coordinates(x, int(block_shape.sx)),
            normalize_index_coordinates(y, int(block_shape.sy)),
            normalize_index_coordinates(z, int(block_shape.sz)),
        ],
        axis=1,
    ).astype(np.float32, copy=False)


def normalized_time_values(time_count: int) -> np.ndarray:
    values = np.arange(int(time_count), dtype=np.int64)
    return normalize_index_coordinates(values, int(time_count)).astype(np.float32, copy=False)


def full_block_query_coords(
    *,
    block_shape: BlockShape,
    time_count: int,
) -> np.ndarray:
    spatial = local_spatial_coords(block_shape)
    time_values = normalized_time_values(time_count)
    chunks: list[np.ndarray] = []
    for time_value in time_values.tolist():
        time_column = np.full((int(spatial.shape[0]), 1), float(time_value), dtype=np.float32)
        chunks.append(np.concatenate([spatial, time_column], axis=1))
    return np.concatenate(chunks, axis=0)


def sample_block_training_batch(
    *,
    block_values: np.ndarray,
    block_shape: BlockShape,
    points_per_timestep: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    array = np.asarray(block_values, dtype=np.float32)
    if array.ndim != 4:
        raise ValueError(f"block_values must have shape [T, sz, sy, sx], got {array.shape}")
    time_count = int(array.shape[0])
    flat_values = array.reshape(time_count, -1)
    spatial = local_spatial_coords(block_shape)
    time_values = normalized_time_values(time_count)
    coords_parts: list[np.ndarray] = []
    target_parts: list[np.ndarray] = []
    voxel_count = int(block_shape.voxel_count)
    sample_count = int(points_per_timestep)
    if sample_count <= 0:
        raise ValueError("points_per_timestep must be positive")
    for time_index in range(time_count):
        replace = voxel_count < sample_count
        picked = rng.choice(voxel_count, size=sample_count, replace=replace).astype(np.int64)
        coords_t = spatial[picked]
        time_column = np.full((sample_count, 1), float(time_values[time_index]), dtype=np.float32)
        coords_parts.append(np.concatenate([coords_t, time_column], axis=1))
        target_parts.append(flat_values[time_index, picked].reshape(sample_count, 1))
    return (
        np.concatenate(coords_parts, axis=0).astype(np.float32, copy=False),
        np.concatenate(target_parts, axis=0).astype(np.float32, copy=False),
    )


def sample_balanced_block_training_batch(
    *,
    block_values: np.ndarray,
    block_shape: BlockShape,
    batch_size: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample one exact-size batch, distributing rows across all timesteps."""
    array = np.asarray(block_values, dtype=np.float32)
    if array.ndim != 4:
        raise ValueError(f"block_values must have shape [T, sz, sy, sx], got {array.shape}")
    time_count = int(array.shape[0])
    requested = int(batch_size)
    if requested <= 0:
        raise ValueError("batch_size must be positive")
    flat_values = array.reshape(time_count, -1)
    spatial = local_spatial_coords(block_shape)
    time_values = normalized_time_values(time_count)
    voxel_count = int(block_shape.voxel_count)
    base, remainder = divmod(requested, time_count)
    coords_parts: list[np.ndarray] = []
    target_parts: list[np.ndarray] = []
    for time_index in range(time_count):
        count = base + (1 if time_index < remainder else 0)
        if count <= 0:
            continue
        picked = rng.choice(voxel_count, size=count, replace=voxel_count < count).astype(np.int64)
        time_column = np.full((count, 1), float(time_values[time_index]), dtype=np.float32)
        coords_parts.append(np.concatenate([spatial[picked], time_column], axis=1))
        target_parts.append(flat_values[time_index, picked].reshape(count, 1))
    return (
        np.concatenate(coords_parts, axis=0).astype(np.float32, copy=False),
        np.concatenate(target_parts, axis=0).astype(np.float32, copy=False),
    )


def candidate_voxel_counts(candidate_shapes: Iterable[BlockShape]) -> set[int]:
    return {int(shape.voxel_count) for shape in candidate_shapes}
