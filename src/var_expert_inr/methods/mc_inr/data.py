from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from typing import Iterable

import numpy as np
import torch

from ...data.base import DatasetMeta, FieldDataset, normalize_index_coordinates


@dataclass(frozen=True)
class TargetLayoutEntry:
    name: str
    start: int
    end: int
    dim: int


@dataclass
class MCBatch:
    indices: np.ndarray
    coords: torch.Tensor
    targets_concat: torch.Tensor
    targets_dict: dict[str, torch.Tensor]
    cluster_ids: torch.Tensor


def target_layout_from_dataset(dataset: FieldDataset) -> tuple[TargetLayoutEntry, ...]:
    offset = 0
    entries: list[TargetLayoutEntry] = []
    for name in dataset.target_names():
        dim = int(dataset.meta.target_dims[name])
        entries.append(TargetLayoutEntry(name=name, start=offset, end=offset + dim, dim=dim))
        offset += dim
    return tuple(entries)


def layout_to_payload(layout: tuple[TargetLayoutEntry, ...]) -> list[dict[str, int | str]]:
    return [
        {"name": entry.name, "start": int(entry.start), "end": int(entry.end), "dim": int(entry.dim)}
        for entry in layout
    ]


def layout_from_payload(payload: Iterable[dict[str, int | str]]) -> tuple[TargetLayoutEntry, ...]:
    return tuple(
        TargetLayoutEntry(
            name=str(entry["name"]),
            start=int(entry["start"]),
            end=int(entry["end"]),
            dim=int(entry["dim"]),
        )
        for entry in payload
    )


def concat_targets(
    targets: torch.Tensor | dict[str, torch.Tensor],
    layout: tuple[TargetLayoutEntry, ...],
) -> torch.Tensor:
    if isinstance(targets, torch.Tensor):
        if len(layout) != 1:
            raise ValueError("Tensor targets can only be used with a single target layout")
        return targets
    return torch.cat([targets[entry.name] for entry in layout], dim=-1)


def split_tensor_by_layout(
    tensor: torch.Tensor,
    layout: tuple[TargetLayoutEntry, ...],
) -> dict[str, torch.Tensor]:
    return {entry.name: tensor[:, entry.start:entry.end] for entry in layout}


def cluster_ids_for_rows(rows: np.ndarray, assignments: np.ndarray, meta: DatasetMeta) -> np.ndarray:
    row_ids = np.asarray(rows, dtype=np.int64)
    cluster_ids = np.asarray(assignments, dtype=np.int64)
    if meta.kind == "node":
        return cluster_ids[row_ids]
    if meta.volume_shape is None:
        raise ValueError("Volume metadata is required for volume cluster routing")
    voxel_count = int(meta.volume_shape.X) * int(meta.volume_shape.Y) * int(meta.volume_shape.Z)
    if cluster_ids.shape != (voxel_count,):
        raise ValueError(
            f"Unsupported MC-INR assignment shape {cluster_ids.shape} for volume dataset; "
            f"expected {(voxel_count,)}"
        )
    return cluster_ids[row_ids % voxel_count]


def _sample_ids_by_cluster(assignments: np.ndarray, sampling_ratio: float, rng: np.random.Generator) -> np.ndarray:
    ratio = float(sampling_ratio)
    if ratio >= 1.0:
        return np.arange(int(assignments.shape[0]), dtype=np.int64)
    sampled: list[np.ndarray] = []
    for cluster_id in np.unique(assignments):
        cluster_rows = np.flatnonzero(assignments == cluster_id)
        if cluster_rows.size == 0:
            continue
        take = min(cluster_rows.size, max(1, int(ceil(cluster_rows.size * ratio))))
        sampled.append(rng.choice(cluster_rows, size=take, replace=False).astype(np.int64))
    if not sampled:
        return np.empty((0,), dtype=np.int64)
    return np.concatenate(sampled, axis=0)


def sample_node_rows(assignments: np.ndarray, sampling_ratio: float, rng: np.random.Generator) -> np.ndarray:
    return _sample_ids_by_cluster(np.asarray(assignments, dtype=np.int64), sampling_ratio, rng)


def sample_node_rows_from_cluster(
    assignments: np.ndarray,
    cluster_id: int,
    row_count: int,
    rng: np.random.Generator,
) -> np.ndarray:
    assignments_np = np.asarray(assignments, dtype=np.int64)
    members = np.flatnonzero(assignments_np == int(cluster_id))
    if members.size == 0:
        return np.empty((0,), dtype=np.int64)
    n = min(max(0, int(row_count)), int(members.size))
    if n <= 0:
        return np.empty((0,), dtype=np.int64)
    return rng.choice(members, size=n, replace=False).astype(np.int64)


def volume_voxel_count(meta: DatasetMeta) -> int:
    if meta.volume_shape is None:
        raise ValueError("volume_voxel_count requires volume metadata")
    return int(meta.volume_shape.X) * int(meta.volume_shape.Y) * int(meta.volume_shape.Z)


def volume_rows_from_voxels_and_times(
    voxel_ids: np.ndarray,
    time_ids: np.ndarray,
    meta: DatasetMeta,
) -> np.ndarray:
    if meta.volume_shape is None:
        raise ValueError("volume_rows_from_voxels_and_times requires volume metadata")

    voxels = np.asarray(voxel_ids, dtype=np.int64)
    times = np.asarray(time_ids, dtype=np.int64)
    if voxels.shape != times.shape:
        raise ValueError("voxel_ids and time_ids must have the same shape")

    voxel_count = volume_voxel_count(meta)
    time_count = int(meta.volume_shape.T)
    if voxels.size > 0:
        if int(voxels.min()) < 0 or int(voxels.max()) >= voxel_count:
            raise ValueError("voxel_ids out of valid range")
        if int(times.min()) < 0 or int(times.max()) >= time_count:
            raise ValueError("time_ids out of valid range")

    return (times * voxel_count + voxels).astype(np.int64, copy=False)


def sample_volume_rows_from_cluster(
    assignments: np.ndarray,
    cluster_id: int,
    meta: DatasetMeta,
    row_count: int,
    rng: np.random.Generator,
) -> np.ndarray:
    assignments_np = np.asarray(assignments, dtype=np.int64)
    members = np.flatnonzero(assignments_np == int(cluster_id))
    if members.size == 0:
        return np.empty((0,), dtype=np.int64)

    time_count = int(meta.volume_shape.T) if meta.volume_shape is not None else 0
    local_total_rows = int(members.size) * int(time_count)
    n = min(max(0, int(row_count)), local_total_rows)
    if n <= 0:
        return np.empty((0,), dtype=np.int64)

    offsets = rng.choice(local_total_rows, size=n, replace=False)
    local_voxel_indices = offsets % int(members.size)
    time_ids = offsets // int(members.size)
    voxel_ids = members[local_voxel_indices]
    return volume_rows_from_voxels_and_times(voxel_ids, time_ids, meta)


def sample_volume_rows_global(
    assignments: np.ndarray,
    meta: DatasetMeta,
    row_count: int,
    rng: np.random.Generator,
) -> np.ndarray:
    del assignments
    voxel_count = volume_voxel_count(meta)
    time_count = int(meta.volume_shape.T) if meta.volume_shape is not None else 0
    total_rows = voxel_count * time_count
    n = min(max(0, int(row_count)), total_rows)
    if n <= 0:
        return np.empty((0,), dtype=np.int64)
    return rng.choice(total_rows, size=n, replace=False).astype(np.int64)


def volume_spatial_coords_for_voxels(voxel_ids: np.ndarray, meta: DatasetMeta) -> np.ndarray:
    if meta.volume_shape is None:
        raise ValueError("volume_spatial_coords_for_voxels requires volume metadata")
    voxels = np.asarray(voxel_ids, dtype=np.int64)
    X = int(meta.volume_shape.X)
    Y = int(meta.volume_shape.Y)
    Z = int(meta.volume_shape.Z)
    x = voxels % X
    rem = voxels // X
    y = rem % Y
    z = rem // Y
    return np.stack(
        [
            normalize_index_coordinates(x, X),
            normalize_index_coordinates(y, Y),
            normalize_index_coordinates(z, Z),
        ],
        axis=1,
    ).astype(np.float32, copy=False)


def prediction_shape(meta: DatasetMeta, target_dim: int) -> tuple[int, ...]:
    dims = int(target_dim)
    if meta.volume_shape is None:
        return (int(meta.n_samples), dims)
    if dims == 1:
        return (
            int(meta.volume_shape.T),
            int(meta.volume_shape.Z),
            int(meta.volume_shape.Y),
            int(meta.volume_shape.X),
        )
    return (
        int(meta.volume_shape.T),
        int(meta.volume_shape.Z),
        int(meta.volume_shape.Y),
        int(meta.volume_shape.X),
        dims,
    )


def fetch_mc_batch(
    dataset: FieldDataset,
    rows: np.ndarray,
    layout: tuple[TargetLayoutEntry, ...],
    assignments: np.ndarray,
) -> MCBatch:
    row_ids = np.asarray(rows, dtype=np.int64)
    batch = dataset.fetch_batch(row_ids.tolist(), include_targets=True)
    targets_concat = concat_targets(batch.targets, layout)
    if isinstance(batch.targets, torch.Tensor):
        targets_dict = {layout[0].name: batch.targets}
    else:
        targets_dict = batch.targets
    cluster_ids = torch.from_numpy(cluster_ids_for_rows(row_ids, assignments, dataset.meta))
    return MCBatch(
        indices=row_ids,
        coords=batch.coords,
        targets_concat=targets_concat,
        targets_dict=targets_dict,
        cluster_ids=cluster_ids,
    )


def compute_volume_centroids(
    meta: DatasetMeta,
    assignments: np.ndarray,
    num_clusters: int,
    *,
    chunk_size: int = 1_000_000,
) -> np.ndarray:
    if meta.volume_shape is None:
        raise ValueError("Volume metadata is required to compute centroids")
    cluster_ids = np.asarray(assignments, dtype=np.int64)
    X = int(meta.volume_shape.X)
    Y = int(meta.volume_shape.Y)
    Z = int(meta.volume_shape.Z)
    voxel_count = X * Y * Z
    if cluster_ids.shape != (voxel_count,):
        raise ValueError(
            f"Volume centroids require voxel-level assignments with shape {(voxel_count,)}, "
            f"got {cluster_ids.shape}"
        )

    sum_x = np.zeros((num_clusters,), dtype=np.float64)
    sum_y = np.zeros((num_clusters,), dtype=np.float64)
    sum_z = np.zeros((num_clusters,), dtype=np.float64)
    counts = np.zeros((num_clusters,), dtype=np.float64)

    for start in range(0, voxel_count, int(chunk_size)):
        stop = min(start + int(chunk_size), voxel_count)
        idx = np.arange(start, stop, dtype=np.int64)
        cids = cluster_ids[start:stop]

        x = idx % X
        rem = idx // X
        y = rem % Y
        z = rem // Y

        x_n = normalize_index_coordinates(x, X).astype(np.float64, copy=False)
        y_n = normalize_index_coordinates(y, Y).astype(np.float64, copy=False)
        z_n = normalize_index_coordinates(z, Z).astype(np.float64, copy=False)

        counts += np.bincount(cids, minlength=num_clusters).astype(np.float64)
        sum_x += np.bincount(cids, weights=x_n, minlength=num_clusters).astype(np.float64)
        sum_y += np.bincount(cids, weights=y_n, minlength=num_clusters).astype(np.float64)
        sum_z += np.bincount(cids, weights=z_n, minlength=num_clusters).astype(np.float64)

    counts = np.maximum(counts, 1.0)
    centroids = np.stack([sum_x / counts, sum_y / counts, sum_z / counts], axis=1)
    return centroids.astype(np.float32)
