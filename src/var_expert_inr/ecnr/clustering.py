from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.cluster import KMeans


@dataclass(frozen=True)
class BalancedClustering:
    labels: np.ndarray
    centroids: np.ndarray
    slots: np.ndarray
    cluster_sizes: np.ndarray


def squared_euclidean_distances(points: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    x = np.asarray(points, dtype=np.float32)
    c = np.asarray(centroids, dtype=np.float32)
    x2 = np.sum(x * x, axis=1, keepdims=True)
    c2 = np.sum(c * c, axis=1, keepdims=True).T
    result = x2 + c2 - 2.0 * (x @ c.T)
    return np.maximum(result, 0.0).astype(np.float32, copy=False)


def balanced_kmeans(
    normalized_blocks: np.ndarray,
    *,
    target_blocks_per_mlp: int,
    seed: int,
    n_init: int = 1,
    max_iter: int = 300,
    tol: float = 1.0e-4,
) -> BalancedClustering:
    """Cluster normalized block values, then perform one deterministic balance pass."""
    points = np.asarray(normalized_blocks, dtype=np.float32)
    if points.ndim != 2 or points.shape[0] == 0:
        raise ValueError("normalized_blocks must have shape [N, block_voxels] with N > 0")
    n_points = int(points.shape[0])
    target = int(target_blocks_per_mlp)
    if target <= 0:
        raise ValueError("target_blocks_per_mlp must be positive")
    cluster_count = int(np.ceil(n_points / target))
    if cluster_count == 1:
        labels = np.zeros(n_points, dtype=np.int64)
        centroids = points.mean(axis=0, keepdims=True, dtype=np.float32)
    else:
        estimator = KMeans(
            n_clusters=cluster_count,
            init="k-means++",
            n_init=int(n_init),
            max_iter=int(max_iter),
            tol=float(tol),
            algorithm="lloyd",
            random_state=int(seed),
        )
        estimator.fit(points)
        centroids = np.asarray(estimator.cluster_centers_, dtype=np.float32)
        distances = squared_euclidean_distances(points, centroids)
        delta = distances.max(axis=1) - distances.min(axis=1)
        # Primary key: descending delta. Secondary key: original traversal index.
        order = np.lexsort((np.arange(n_points, dtype=np.int64), -delta.astype(np.float64)))
        n_min = n_points // cluster_count
        n_max = int(np.ceil(n_points / cluster_count))
        counts = np.zeros(cluster_count, dtype=np.int64)
        labels = np.full(n_points, -1, dtype=np.int64)
        cluster_ids = np.arange(cluster_count, dtype=np.int64)
        for point_index in order.tolist():
            under_minimum = cluster_ids[counts < n_min]
            allowed = under_minimum if under_minimum.size else cluster_ids[counts < n_max]
            if allowed.size == 0:
                raise RuntimeError("Capacity-constrained K-means reassignment exhausted all clusters")
            local_distances = distances[point_index, allowed]
            # np.argmin returns the first entry; allowed is ascending cluster id.
            selected = int(allowed[int(np.argmin(local_distances))])
            labels[point_index] = selected
            counts[selected] += 1
        centroids = np.empty((cluster_count, points.shape[1]), dtype=np.float32)
        for cluster_index in range(cluster_count):
            centroids[cluster_index] = points[labels == cluster_index].mean(axis=0, dtype=np.float32)

    cluster_sizes = np.bincount(labels, minlength=cluster_count).astype(np.int64)
    slots = np.empty(n_points, dtype=np.int64)
    next_slot = np.zeros(cluster_count, dtype=np.int64)
    # Input order already is time-major then spatial-block-major.
    for block_index, cluster_index in enumerate(labels.tolist()):
        slots[block_index] = next_slot[cluster_index]
        next_slot[cluster_index] += 1
    return BalancedClustering(
        labels=labels,
        centroids=np.asarray(centroids, dtype=np.float32),
        slots=slots,
        cluster_sizes=cluster_sizes,
    )
