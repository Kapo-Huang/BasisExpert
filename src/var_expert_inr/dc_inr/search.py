from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.cluster import DBSCAN

from .data import BlockGridShape, BlockShape, DCTargetVolume, block_grid_shape_for_volume
from .model import adjusted_even_width, dc_inr_parameter_count


@dataclass(frozen=True)
class CandidateSummary:
    block_shape: BlockShape
    grid_shape: BlockGridShape
    representative_block_ids: np.ndarray
    block_to_representative: np.ndarray
    entropies: np.ndarray
    widths: np.ndarray
    selected_M: int
    mean_capacity: float
    payload_bytes: int
    distance_matrix_bytes: int


def validate_candidate_block_shapes(candidate_shapes: tuple[BlockShape, ...]) -> None:
    if not candidate_shapes:
        raise ValueError("partition.candidate_block_shapes must be non-empty")
    voxel_counts = {int(shape.voxel_count) for shape in candidate_shapes}
    if len(voxel_counts) != 1:
        raise ValueError("All partition.candidate_block_shapes must contain the same voxel count")


def compute_spatiotemporal_distance_matrix(
    block_features: np.ndarray,
    *,
    max_bytes: int,
) -> np.ndarray:
    features = np.asarray(block_features, dtype=np.float32)
    if features.ndim != 3:
        raise ValueError(f"block_features must have shape [N, T, V], got {features.shape}")
    num_blocks = int(features.shape[0])
    distance_bytes = num_blocks * num_blocks * np.dtype(np.float32).itemsize
    if distance_bytes > int(max_bytes):
        raise ValueError(
            f"Distance matrix requires {distance_bytes} bytes, which exceeds "
            f"partition.distance_matrix_max_bytes={int(max_bytes)}. Use coarser blocks."
        )
    # DC-INR's clustering threshold must be independent of both the block voxel
    # count and the number of timesteps.  Summed Euclidean distances scale as
    # O(T * sqrt(V)); with production blocks this made an eps such as 0.01
    # effectively zero and turned every block into its own representative.
    # Average per-timestep RMSE keeps the distance in the target value scale.
    distance = np.zeros((num_blocks, num_blocks), dtype=np.float32)
    voxel_count = int(features.shape[2])
    time_count = int(features.shape[1])
    for time_index in range(int(features.shape[1])):
        current = features[:, time_index, :]
        norms = np.sum(current * current, axis=1, dtype=np.float64)
        gram = current @ current.T
        squared = np.maximum(norms[:, None] + norms[None, :] - (2.0 * gram), 0.0)
        distance += np.sqrt(squared / float(voxel_count)).astype(np.float32) / float(time_count)
    np.fill_diagonal(distance, 0.0)
    return distance


def cluster_representatives(
    distance_matrix: np.ndarray,
    *,
    eps: float,
    min_samples: int,
) -> tuple[np.ndarray, np.ndarray]:
    labels = DBSCAN(metric="precomputed", eps=float(eps), min_samples=int(min_samples)).fit_predict(distance_matrix)
    labels = np.asarray(labels, dtype=np.int32)
    if np.any(labels < 0):
        next_label = int(labels.max()) + 1
        for index in np.flatnonzero(labels < 0).tolist():
            labels[index] = next_label
            next_label += 1
    representative_block_ids: list[int] = []
    block_to_representative = np.empty((labels.shape[0],), dtype=np.int32)
    for cluster_index, label in enumerate(sorted(np.unique(labels).tolist())):
        members = np.flatnonzero(labels == int(label))
        if members.size == 1:
            representative = int(members[0])
        else:
            submatrix = distance_matrix[np.ix_(members, members)].astype(np.float64, copy=False)
            representative = int(members[int(np.argmin(submatrix.sum(axis=1)))])
        representative_block_ids.append(representative)
        block_to_representative[members] = int(cluster_index)
    return (
        np.asarray(representative_block_ids, dtype=np.int32),
        block_to_representative.astype(np.int32, copy=False),
    )


def compute_spatiotemporal_entropies(
    blocks: np.ndarray,
    representative_block_ids: np.ndarray,
    *,
    entropy_bins: int,
) -> np.ndarray:
    block_array = np.asarray(blocks, dtype=np.float32)
    reps = np.asarray(representative_block_ids, dtype=np.int64)
    entropies = np.zeros((reps.shape[0],), dtype=np.float64)
    for rep_offset, block_id in enumerate(reps.tolist()):
        block = block_array[int(block_id)]
        entropy_value = 0.0
        for time_index in range(int(block.shape[0])):
            hist, _ = np.histogram(
                np.asarray(block[time_index], dtype=np.float32).reshape(-1),
                bins=int(entropy_bins),
                range=(-1.0, 1.0),
            )
            total = int(hist.sum())
            if total <= 0:
                continue
            probs = hist.astype(np.float64) / float(total)
            probs = probs[probs > 0.0]
            entropy_value += float(-(probs * np.log(probs)).sum())
        entropies[rep_offset] = entropy_value
    return entropies.astype(np.float32)


def allocate_widths_for_entropies(
    entropies: np.ndarray,
    *,
    max_initial_neurons: int,
    min_initial_neurons: int,
) -> np.ndarray:
    entropy_array = np.asarray(entropies, dtype=np.float32)
    max_entropy = float(np.max(entropy_array)) if entropy_array.size > 0 else 0.0
    widths = np.empty((entropy_array.shape[0],), dtype=np.int32)
    if max_entropy <= 0.0:
        widths.fill(adjusted_even_width(int(min_initial_neurons), minimum=int(min_initial_neurons)))
        return widths
    for index, entropy_value in enumerate(entropy_array.tolist()):
        scaled = int(np.floor((float(entropy_value) / max_entropy) * float(max_initial_neurons)))
        widths[index] = int(
            adjusted_even_width(
                max(int(min_initial_neurons), scaled),
                minimum=int(min_initial_neurons),
            )
        )
    return widths


def estimate_payload_bytes(
    *,
    volume: DCTargetVolume,
    block_shape: BlockShape,
    grid_shape: BlockGridShape,
    target_name: str,
    block_to_representative: np.ndarray,
    representative_block_ids: np.ndarray,
    widths: np.ndarray,
) -> int:
    assignment_count = int(np.asarray(block_to_representative).size)
    representative_count = int(np.asarray(representative_block_ids).size)
    model_param_bytes = sum(dc_inr_parameter_count(int(width)) * 2 for width in np.asarray(widths, dtype=np.int32).tolist())
    metadata_bytes = 0
    metadata_bytes += 4 * 4
    metadata_bytes += 3 * 4
    metadata_bytes += 3 * 4
    metadata_bytes += 4 + len(str(target_name).encode("utf-8"))
    metadata_bytes += 2 * 4
    metadata_bytes += 4 * assignment_count
    metadata_bytes += 4 * representative_count
    metadata_bytes += 4 * representative_count
    metadata_bytes += 4 * 4
    metadata_bytes += 4
    metadata_bytes += 4
    return int(model_param_bytes + metadata_bytes)


def select_best_candidate(
    *,
    volume: DCTargetVolume,
    candidate_shapes: tuple[BlockShape, ...],
    dbscan_eps: float,
    dbscan_min_samples: int,
    entropy_bins: int,
    distance_matrix_max_bytes: int,
    target_cr: float,
    max_initial_neurons: int,
    min_initial_neurons: int,
) -> tuple[CandidateSummary, list[dict[str, object]]]:
    validate_candidate_block_shapes(candidate_shapes)
    if float(target_cr) <= 0.0:
        raise ValueError("compression.target_cr must be positive")
    if int(max_initial_neurons) < int(min_initial_neurons):
        raise ValueError("compression.max_initial_neurons must be at least compression.min_initial_neurons")

    best_summary: CandidateSummary | None = None
    summaries: list[dict[str, object]] = []
    size_limit_bytes = float(volume.raw_bytes) / float(target_cr)

    for shape in candidate_shapes:
        grid_shape = block_grid_shape_for_volume(volume.volume_shape, shape)
        blocks = volume.block_view(shape)
        features = blocks.reshape(int(grid_shape.n_blocks), int(volume.volume_shape.T), int(shape.voxel_count))
        distance_matrix = compute_spatiotemporal_distance_matrix(
            features,
            max_bytes=int(distance_matrix_max_bytes),
        )
        representative_block_ids, block_to_representative = cluster_representatives(
            distance_matrix,
            eps=float(dbscan_eps),
            min_samples=int(dbscan_min_samples),
        )
        entropies = compute_spatiotemporal_entropies(
            blocks,
            representative_block_ids,
            entropy_bins=int(entropy_bins),
        )

        feasible_M: int | None = None
        feasible_widths: np.ndarray | None = None
        feasible_payload = 0
        low = int(min_initial_neurons)
        high = int(max_initial_neurons)
        while low <= high:
            mid = (low + high) // 2
            widths = allocate_widths_for_entropies(
                entropies,
                max_initial_neurons=int(mid),
                min_initial_neurons=int(min_initial_neurons),
            )
            payload_bytes = estimate_payload_bytes(
                volume=volume,
                block_shape=shape,
                grid_shape=grid_shape,
                target_name=volume.target_name,
                block_to_representative=block_to_representative,
                representative_block_ids=representative_block_ids,
                widths=widths,
            )
            if float(payload_bytes) <= size_limit_bytes:
                feasible_M = int(mid)
                feasible_widths = widths
                feasible_payload = int(payload_bytes)
                low = mid + 1
            else:
                high = mid - 1

        if feasible_M is None or feasible_widths is None:
            summaries.append(
                {
                    "block_shape": shape.to_dict(),
                    "grid_shape": grid_shape.to_dict(),
                    "status": "infeasible",
                    "reason": "target_cr_unreachable",
                }
            )
            continue

        mean_capacity = float(np.mean(feasible_widths.astype(np.float64))) if feasible_widths.size > 0 else 0.0
        summary = CandidateSummary(
            block_shape=shape,
            grid_shape=grid_shape,
            representative_block_ids=representative_block_ids,
            block_to_representative=block_to_representative,
            entropies=entropies,
            widths=feasible_widths,
            selected_M=int(feasible_M),
            mean_capacity=mean_capacity,
            payload_bytes=int(feasible_payload),
            distance_matrix_bytes=int(distance_matrix.nbytes),
        )
        summaries.append(
            {
                "block_shape": shape.to_dict(),
                "grid_shape": grid_shape.to_dict(),
                "status": "feasible",
                "representative_count": int(representative_block_ids.size),
                "selected_M": int(feasible_M),
                "mean_capacity": float(mean_capacity),
                "payload_bytes": int(feasible_payload),
                "distance_matrix_bytes": int(distance_matrix.nbytes),
            }
        )
        if best_summary is None:
            best_summary = summary
            continue
        if float(summary.mean_capacity) > float(best_summary.mean_capacity):
            best_summary = summary
            continue
        if float(summary.mean_capacity) == float(best_summary.mean_capacity):
            if int(summary.payload_bytes) < int(best_summary.payload_bytes):
                best_summary = summary

    if best_summary is None:
        raise ValueError("No candidate block shape can satisfy compression.target_cr")
    return best_summary, summaries
