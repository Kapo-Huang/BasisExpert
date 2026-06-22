from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
import time

import numpy as np
from sklearn.cluster import MiniBatchKMeans

logger = logging.getLogger(__name__)


@dataclass
class PretrainAssignmentConfig:
    seed: int = 42
    cache_path: str = ""


def _load_cached(cache_path: str, *, expected_size: int, num_experts: int) -> np.ndarray | None:
    if not cache_path:
        return None
    cache = Path(cache_path)
    if not cache.exists():
        return None
    logger.info("Loading pretrain assignments cache: %s", cache)
    cached = np.asarray(np.load(cache), dtype=np.int64)
    if cached.shape != (int(expected_size),):
        raise ValueError(
            f"Cached pretrain assignments at {cache} have shape {cached.shape}, expected {(int(expected_size),)}"
        )
    if cached.size > 0 and (int(cached.min()) < 0 or int(cached.max()) >= int(num_experts)):
        raise ValueError(
            f"Cached pretrain assignments at {cache} contain ids outside [0, {int(num_experts) - 1}]"
        )
    return cached


def _save_cached(cache_path: str, assignments: np.ndarray):
    if not cache_path:
        return
    cache = Path(cache_path)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache, assignments)
    logger.info("Saved pretrain assignments cache: %s shape=%s", cache, tuple(assignments.shape))


def _sample_time_indices(T: int, num_time_samples: int, seed: int) -> np.ndarray:
    if T <= 0:
        raise ValueError(f"volume_shape.T must be positive, got {T}")
    if num_time_samples <= 0:
        raise ValueError(f"num_time_samples must be positive, got {num_time_samples}")
    rng = np.random.default_rng(seed)
    replace = T < num_time_samples
    return rng.choice(T, size=num_time_samples, replace=replace).astype(np.int64)


def _build_voxel_cluster_features(dataset, num_time_samples: int, seed: int) -> np.ndarray:
    volume_shape = dataset.meta.volume_shape
    if volume_shape is None:
        raise ValueError("voxel_clustering pretrain requires a volume dataset with volume_shape")
    V = int(volume_shape.X) * int(volume_shape.Y) * int(volume_shape.Z)
    started_at = time.perf_counter()
    time_ids = _sample_time_indices(int(volume_shape.T), int(num_time_samples), int(seed))
    logger.info(
        "Building voxel clustering features: V=%d T=%d sampled_time_count=%d sampled_times=%s targets=%s",
        V,
        int(volume_shape.T),
        int(len(time_ids)),
        time_ids.tolist(),
        list(dataset.target_names()),
    )

    targets_flat = dataset.load_targets_flat()
    idx = np.arange(V, dtype=np.int64)[:, None] + V * time_ids[None, :]
    feature_blocks = []
    for name in dataset.target_names():
        flat = np.asarray(targets_flat[name][idx.reshape(-1)], dtype=np.float32)
        if flat.ndim == 1:
            flat = flat.reshape(-1, 1)
        flat = flat.reshape(V, len(time_ids), -1)
        feature_blocks.append(flat)
    merged = np.concatenate(feature_blocks, axis=2)
    features = merged.reshape(V, -1)
    logger.info(
        "Voxel clustering features ready: shape=%s dtype=%s time=%.2fs",
        tuple(features.shape),
        features.dtype,
        time.perf_counter() - started_at,
    )
    return features


def _kmeans_assignments(features: np.ndarray, num_experts: int, seed: int) -> np.ndarray:
    if int(num_experts) <= 0:
        raise ValueError(f"num_experts must be positive, got {num_experts}")
    if int(features.shape[0]) < int(num_experts):
        raise ValueError(
            f"voxel_clustering pretrain requires voxel_count >= num_experts, got {features.shape[0]} < {num_experts}"
        )
    chunk_size = min(50_000, int(features.shape[0]))
    kmeans = MiniBatchKMeans(
        n_clusters=int(num_experts),
        random_state=int(seed),
        batch_size=max(256, chunk_size),
        n_init=3,
    )
    logger.info(
        "MiniBatchKMeans start: samples=%d feature_dim=%d n_clusters=%d chunk_size=%d",
        int(features.shape[0]),
        int(features.shape[1]),
        int(num_experts),
        int(chunk_size),
    )
    fit_started_at = time.perf_counter()
    for start in range(0, int(features.shape[0]), chunk_size):
        end = min(start + chunk_size, int(features.shape[0]))
        kmeans.partial_fit(features[start:end])
    logger.info("MiniBatchKMeans partial_fit done: %.2fs", time.perf_counter() - fit_started_at)

    assignments = np.empty((features.shape[0],), dtype=np.int64)
    predict_started_at = time.perf_counter()
    for start in range(0, int(features.shape[0]), chunk_size):
        end = min(start + chunk_size, int(features.shape[0]))
        assignments[start:end] = kmeans.predict(features[start:end]).astype(np.int64)
    logger.info("MiniBatchKMeans predict done: %.2fs", time.perf_counter() - predict_started_at)
    return assignments


def compute_pretrain_assignments(dataset, num_experts: int, cfg: PretrainAssignmentConfig) -> np.ndarray:
    volume_shape = dataset.meta.volume_shape
    if volume_shape is None:
        raise ValueError("Configured pretraining requires a volume dataset because only voxel_clustering is supported")
    voxel_count = int(volume_shape.X) * int(volume_shape.Y) * int(volume_shape.Z)
    started_at = time.perf_counter()
    cached = _load_cached(str(cfg.cache_path or ""), expected_size=voxel_count, num_experts=int(num_experts))
    if cached is not None:
        logger.info(
            "Using cached pretrain assignments: method=voxel_clustering shape=%s time=%.2fs",
            tuple(cached.shape),
            time.perf_counter() - started_at,
        )
        return cached

    logger.info(
        "Building pretrain assignments: method=voxel_clustering num_experts=%d num_time_samples=%d volume_shape=%s cache=%s",
        int(num_experts),
        int(num_experts),
        volume_shape,
        str(cfg.cache_path or "<none>"),
    )
    assignments = _kmeans_assignments(
        _build_voxel_cluster_features(dataset, int(num_experts), int(cfg.seed)),
        int(num_experts),
        int(cfg.seed),
    )
    _save_cached(str(cfg.cache_path or ""), assignments)
    logger.info(
        "Pretrain assignments ready: method=voxel_clustering shape=%s time=%.2fs",
        tuple(assignments.shape),
        time.perf_counter() - started_at,
    )
    return assignments
