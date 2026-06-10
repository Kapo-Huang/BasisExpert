from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from sklearn.cluster import MiniBatchKMeans


@dataclass
class PretrainAssignmentConfig:
    method: str = "random_uniform"
    seed: int = 42
    cache_path: str = ""
    cluster_num_time_samples: int = 16
    spatial_blocks: Optional[Tuple[int, int, int]] = None
    time_block_size: int = 0


def _load_cached(cache_path: str) -> Optional[np.ndarray]:
    if not cache_path:
        return None
    cache = Path(cache_path)
    if not cache.exists():
        return None
    return np.asarray(np.load(cache), dtype=np.int64)


def _save_cached(cache_path: str, assignments: np.ndarray):
    if not cache_path:
        return
    cache = Path(cache_path)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache, assignments)


def _balanced_random_assignments(n_samples: int, num_experts: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    reps = int(ceil(n_samples / float(num_experts)))
    base = np.tile(np.arange(num_experts, dtype=np.int64), reps)[:n_samples]
    rng.shuffle(base)
    return base


def _kmeans_assignments(features: np.ndarray, num_experts: int, seed: int) -> np.ndarray:
    chunk_size = min(50_000, int(features.shape[0]))
    kmeans = MiniBatchKMeans(
        n_clusters=int(num_experts),
        random_state=int(seed),
        batch_size=max(256, chunk_size),
        n_init=3,
    )
    for start in range(0, int(features.shape[0]), chunk_size):
        end = min(start + chunk_size, int(features.shape[0]))
        kmeans.partial_fit(features[start:end])
    assignments = np.empty((features.shape[0],), dtype=np.int64)
    for start in range(0, int(features.shape[0]), chunk_size):
        end = min(start + chunk_size, int(features.shape[0]))
        assignments[start:end] = kmeans.predict(features[start:end]).astype(np.int64)
    return assignments


def _choose_grid_dims(num_experts: int) -> Tuple[int, int, int]:
    nx = int(round(num_experts ** (1.0 / 3.0)))
    nx = max(1, nx)
    while num_experts % nx != 0 and nx > 1:
        nx -= 1
    rem = max(1, num_experts // nx)
    ny = int(round(rem ** 0.5))
    ny = max(1, ny)
    while rem % ny != 0 and ny > 1:
        ny -= 1
    nz = max(1, rem // ny)
    return int(nx), int(ny), int(nz)


def _block_indices(values: np.ndarray, num_blocks: int) -> np.ndarray:
    if num_blocks <= 1:
        return np.zeros(values.shape[0], dtype=np.int64)
    vmin = float(values.min())
    vmax = float(values.max())
    if vmax <= vmin:
        return np.zeros(values.shape[0], dtype=np.int64)
    edges = np.linspace(vmin, vmax, num_blocks + 1)
    block_ids = np.searchsorted(edges[1:-1], values, side="right")
    return np.asarray(block_ids, dtype=np.int64)


def _spatial_block_assignments_node(coords: np.ndarray, num_experts: int, spatial_blocks) -> np.ndarray:
    if spatial_blocks is None:
        nx, ny, nz = _choose_grid_dims(num_experts)
    else:
        nx, ny, nz = map(int, spatial_blocks)
    bx = _block_indices(coords[:, 0], max(1, nx))
    by = _block_indices(coords[:, 1], max(1, ny))
    bz = _block_indices(coords[:, 2], max(1, nz))
    return np.asarray(((bz * max(1, ny) + by) * max(1, nx) + bx) % num_experts, dtype=np.int64)


def _time_block_assignments_node(coords: np.ndarray, num_experts: int, time_block_size: int) -> np.ndarray:
    order = np.argsort(coords[:, 3], kind="stable")
    if time_block_size <= 0:
        time_block_size = int(ceil(coords.shape[0] / float(num_experts)))
    assignments = np.empty((coords.shape[0],), dtype=np.int64)
    for rank, sample_index in enumerate(order):
        assignments[sample_index] = (rank // int(time_block_size)) % num_experts
    return assignments


def _sample_time_indices(T: int, num_time_samples: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    replace = T < num_time_samples
    return rng.choice(T, size=num_time_samples, replace=replace).astype(np.int64)


def _build_voxel_cluster_features(dataset, num_time_samples: int, seed: int) -> np.ndarray:
    volume_shape = dataset.meta.volume_shape
    V = int(volume_shape.X) * int(volume_shape.Y) * int(volume_shape.Z)
    time_ids = _sample_time_indices(int(volume_shape.T), num_time_samples, seed)
    targets_flat = dataset.load_targets_flat()
    idx = np.arange(V, dtype=np.int64)[:, None] + V * time_ids[None, :]
    blocks = []
    for name in dataset.target_names():
        flat = np.asarray(targets_flat[name][idx.reshape(-1)], dtype=np.float32)
        flat = flat.reshape(V, len(time_ids), -1)
        blocks.append(flat)
    merged = np.concatenate(blocks, axis=2)
    return merged.reshape(V, -1)


def _spatial_block_assignments_volume(dataset, num_experts: int, spatial_blocks) -> np.ndarray:
    volume_shape = dataset.meta.volume_shape
    X, Y, Z = int(volume_shape.X), int(volume_shape.Y), int(volume_shape.Z)
    V = X * Y * Z
    if spatial_blocks is None:
        nx, ny, nz = _choose_grid_dims(num_experts)
    else:
        nx, ny, nz = map(int, spatial_blocks)
    nx = max(1, nx)
    ny = max(1, ny)
    nz = max(1, nz)
    block_x = max(1, int(ceil(X / float(nx))))
    block_y = max(1, int(ceil(Y / float(ny))))
    block_z = max(1, int(ceil(Z / float(nz))))
    assignments = np.empty((V,), dtype=np.int64)
    v = 0
    for z in range(Z):
        bz = min(nz - 1, z // block_z)
        for y in range(Y):
            by = min(ny - 1, y // block_y)
            for x in range(X):
                bx = min(nx - 1, x // block_x)
                assignments[v] = ((bz * ny + by) * nx + bx) % num_experts
                v += 1
    return assignments


def _time_block_assignments_volume(dataset, num_experts: int, time_block_size: int) -> np.ndarray:
    T = int(dataset.meta.volume_shape.T)
    if time_block_size <= 0:
        time_block_size = int(ceil(T / float(num_experts)))
    assignments = np.empty((T,), dtype=np.int64)
    for t in range(T):
        assignments[t] = (t // int(time_block_size)) % num_experts
    return assignments


def compute_pretrain_assignments(dataset, num_experts: int, cfg: PretrainAssignmentConfig) -> np.ndarray:
    method = str(cfg.method).strip().lower()
    cached = _load_cached(cfg.cache_path)
    if cached is not None:
        return cached

    if dataset.pretrain_assignment_kind() == "sample":
        coords = dataset.raw_coords()
        features = dataset.sample_cluster_features()
        n_samples = int(dataset.meta.n_samples)
        if method in {"sample_clustering", "clustering", "kmeans"}:
            assignments = _kmeans_assignments(features, num_experts, int(cfg.seed))
        elif method in {"random", "uniform", "random_uniform"}:
            assignments = _balanced_random_assignments(n_samples, num_experts, int(cfg.seed))
        elif method in {"spatial_block", "spatial_blocks", "space_block"}:
            assignments = _spatial_block_assignments_node(coords, num_experts, cfg.spatial_blocks)
        elif method in {"time_block", "time_blocks", "temporal_block"}:
            assignments = _time_block_assignments_node(coords, num_experts, int(cfg.time_block_size))
        else:
            raise ValueError(f"Unsupported node pretrain assignment method: {cfg.method}")
    else:
        volume_shape = dataset.meta.volume_shape
        V = int(volume_shape.X) * int(volume_shape.Y) * int(volume_shape.Z)
        if method in {"voxel_clustering", "clustering", "kmeans"}:
            assignments = _kmeans_assignments(
                _build_voxel_cluster_features(dataset, int(cfg.cluster_num_time_samples), int(cfg.seed)),
                num_experts,
                int(cfg.seed),
            )
        elif method in {"random", "uniform", "random_uniform"}:
            assignments = _balanced_random_assignments(V, num_experts, int(cfg.seed))
        elif method in {"spatial_block", "spatial_blocks", "space_block"}:
            assignments = _spatial_block_assignments_volume(dataset, num_experts, cfg.spatial_blocks)
        elif method in {"time_block", "time_blocks", "temporal_block"}:
            assignments = _time_block_assignments_volume(dataset, num_experts, int(cfg.time_block_size))
        else:
            raise ValueError(f"Unsupported volume pretrain assignment method: {cfg.method}")

    _save_cached(cfg.cache_path, assignments)
    return assignments
