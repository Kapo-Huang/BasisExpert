from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import numpy as np


DEPENDENCY_SCHEMA_VERSION = 1
DEPENDENCY_ALGORITHM_VERSION = "dependency_histogram_v1"
DEFAULT_SAMPLE_RATIO = 0.2
DEFAULT_SAMPLE_SEED = 42
DEFAULT_MAX_BINS = 64
DEFAULT_VARIANCE_EPS = 1.0e-12
_UINT64_MASK = np.uint64(0xFFFFFFFFFFFFFFFF)


@dataclass(frozen=True)
class DependencyTarget:
    name: str
    transform: str = "scalar"

    def __post_init__(self) -> None:
        if self.transform not in {"scalar", "magnitude"}:
            raise ValueError(f"Unsupported dependency transform: {self.transform!r}")


@dataclass(frozen=True)
class DependencyStatistics:
    pearson: np.ndarray
    mutual_info: np.ndarray
    pearson_valid_pairs: np.ndarray
    mi_valid_pairs: np.ndarray
    bin_edges: tuple[np.ndarray, ...]
    sample_count: int


def _splitmix64(values: np.ndarray) -> np.ndarray:
    """Stable, vectorized SplitMix64 finalizer used only for index selection."""
    z = np.asarray(values, dtype=np.uint64).copy()
    z = (z + np.uint64(0x9E3779B97F4A7C15)) & _UINT64_MASK
    z = ((z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)) & _UINT64_MASK
    z = ((z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)) & _UINT64_MASK
    return z ^ (z >> np.uint64(31))


def iter_sample_index_chunks(
    frame_size: int,
    *,
    timestep: int,
    sample_ratio: float = DEFAULT_SAMPLE_RATIO,
    seed: int = DEFAULT_SAMPLE_SEED,
    chunk_size: int = 4_000_000,
) -> Iterator[np.ndarray]:
    frame_size = int(frame_size)
    ratio = float(sample_ratio)
    if frame_size <= 0:
        raise ValueError(f"frame_size must be positive, got {frame_size}")
    if not 0.0 < ratio <= 1.0:
        raise ValueError(f"sample_ratio must be in (0, 1], got {ratio}")
    threshold = int(ratio * (1 << 64))
    salt = np.uint64((int(seed) + int(timestep)) & ((1 << 64) - 1))
    for start in range(0, frame_size, int(chunk_size)):
        stop = min(frame_size, start + int(chunk_size))
        rows = np.arange(start, stop, dtype=np.uint64)
        hashed = _splitmix64(rows ^ salt)
        if threshold >= (1 << 64):
            selected = rows
        else:
            selected = rows[hashed < np.uint64(threshold)]
        if selected.size:
            yield selected.astype(np.int64, copy=False)


def stable_sample_indices(
    frame_size: int,
    *,
    timestep: int,
    sample_ratio: float = DEFAULT_SAMPLE_RATIO,
    seed: int = DEFAULT_SAMPLE_SEED,
) -> np.ndarray:
    chunks = list(
        iter_sample_index_chunks(
            frame_size,
            timestep=timestep,
            sample_ratio=sample_ratio,
            seed=seed,
        )
    )
    if not chunks:
        raise ValueError(
            f"Sampling selected no points for timestep={timestep}, frame_size={frame_size}"
        )
    indices = np.concatenate(chunks)
    if indices.size < 2:
        raise ValueError(
            f"Dependency metrics require at least two samples, got {indices.size}"
        )
    return indices


def sample_index_digest(indices: np.ndarray) -> str:
    values = np.asarray(indices, dtype="<u8")
    return hashlib.sha256(values.tobytes(order="C")).hexdigest()


def scalarize_frame(
    frame: np.ndarray,
    *,
    frame_size: int,
    transform: str,
) -> np.ndarray:
    values = np.asarray(frame)
    if values.size % int(frame_size) != 0:
        raise ValueError(
            f"Frame with shape {values.shape} cannot be aligned to {frame_size} points"
        )
    components = int(values.size // int(frame_size))
    flat = values.reshape(int(frame_size), components)
    if transform == "scalar":
        if components != 1:
            raise ValueError(
                f"Scalar dependency target has {components} components: {values.shape}"
            )
        result = flat[:, 0]
    elif transform == "magnitude":
        result = np.linalg.norm(flat.astype(np.float64, copy=False), axis=1)
    else:
        raise ValueError(f"Unsupported dependency transform: {transform!r}")
    return np.asarray(result)


def sampled_channel(
    frame: np.ndarray,
    indices: np.ndarray,
    *,
    frame_size: int,
    transform: str,
) -> np.ndarray:
    return np.asarray(
        scalarize_frame(frame, frame_size=frame_size, transform=transform)[indices],
        dtype=np.float64,
    )


def sampled_matrix(
    frames: Mapping[str, np.ndarray],
    targets: Sequence[DependencyTarget],
    indices: np.ndarray,
    *,
    frame_size: int,
) -> np.ndarray:
    columns = []
    for target in targets:
        if target.name not in frames:
            raise KeyError(f"Missing dependency target frame: {target.name}")
        column = sampled_channel(
            frames[target.name],
            indices,
            frame_size=frame_size,
            transform=target.transform,
        )
        if not np.all(np.isfinite(column)):
            raise ValueError(f"Non-finite values in dependency target {target.name!r}")
        columns.append(column)
    return np.column_stack(columns)


def _quantile_edges(values: np.ndarray, *, max_bins: int) -> np.ndarray:
    count = int(values.size)
    requested = min(int(max_bins), max(2, int(math.floor(count ** (1.0 / 3.0)))))
    quantiles = np.linspace(0.0, 1.0, requested + 1, dtype=np.float64)
    raw = np.quantile(values, quantiles)
    finite = np.unique(np.asarray(raw[1:-1], dtype=np.float64))
    if finite.size:
        lo, hi = float(np.min(values)), float(np.max(values))
        finite = finite[(finite > lo) & (finite < hi)]
    return np.concatenate(([-np.inf], finite, [np.inf])).astype(np.float64)


def derive_bin_edges(matrix: np.ndarray, *, max_bins: int = DEFAULT_MAX_BINS) -> tuple[np.ndarray, ...]:
    values = np.asarray(matrix, dtype=np.float64)
    return tuple(_quantile_edges(values[:, channel], max_bins=max_bins) for channel in range(values.shape[1]))


def _pearson_matrix(
    matrix: np.ndarray,
    *,
    variance_eps: float,
    valid_pairs_from_gt: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(matrix, dtype=np.float64)
    count, channels = values.shape
    means = np.zeros(channels, dtype=np.float64)
    m2 = np.zeros((channels, channels), dtype=np.float64)
    seen = 0
    for start in range(0, count, 1_000_000):
        block = values[start : start + 1_000_000]
        block_count = int(block.shape[0])
        block_mean = np.mean(block, axis=0, dtype=np.float64)
        centered = block - block_mean
        block_m2 = centered.T @ centered
        if seen == 0:
            means = block_mean
            m2 = block_m2
            seen = block_count
            continue
        delta = block_mean - means
        total = seen + block_count
        m2 += block_m2 + np.outer(delta, delta) * (seen * block_count / total)
        means += delta * (block_count / total)
        seen = total
    variances = np.diag(m2) / max(seen, 1)
    nonconstant = variances > float(variance_eps)
    denominator = np.sqrt(np.outer(np.diag(m2), np.diag(m2)))
    pearson = np.zeros_like(m2)
    np.divide(m2, denominator, out=pearson, where=denominator > 0.0)
    pearson = np.clip(pearson, -1.0, 1.0)
    np.fill_diagonal(pearson, np.where(nonconstant, 1.0, 0.0))
    if valid_pairs_from_gt is None:
        valid = np.outer(nonconstant, nonconstant)
        np.fill_diagonal(valid, False)
    else:
        valid = np.asarray(valid_pairs_from_gt, dtype=bool).copy()
    return pearson, valid


def _digitize_matrix(matrix: np.ndarray, edges: Sequence[np.ndarray]) -> tuple[np.ndarray, tuple[int, ...]]:
    values = np.asarray(matrix, dtype=np.float64)
    labels = np.empty(values.shape, dtype=np.int16)
    bins: list[int] = []
    for channel, channel_edges in enumerate(edges):
        interior = np.asarray(channel_edges, dtype=np.float64)[1:-1]
        labels[:, channel] = np.searchsorted(interior, values[:, channel], side="right")
        bins.append(int(interior.size + 1))
    return labels, tuple(bins)


def _mutual_info_matrix(
    matrix: np.ndarray,
    edges: Sequence[np.ndarray],
    *,
    valid_pairs_from_gt: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    labels, bins = _digitize_matrix(matrix, edges)
    channels = labels.shape[1]
    output = np.zeros((channels, channels), dtype=np.float64)
    if valid_pairs_from_gt is None:
        valid = np.zeros((channels, channels), dtype=bool)
    else:
        valid = np.asarray(valid_pairs_from_gt, dtype=bool).copy()
    for left in range(channels):
        for right in range(left + 1, channels):
            if valid_pairs_from_gt is None and (bins[left] < 2 or bins[right] < 2):
                continue
            if valid_pairs_from_gt is not None and not valid[left, right]:
                continue
            joint = np.bincount(
                labels[:, left].astype(np.int64) * bins[right] + labels[:, right],
                minlength=bins[left] * bins[right],
            ).reshape(bins[left], bins[right]).astype(np.float64)
            total = float(np.sum(joint))
            px = np.sum(joint, axis=1)
            py = np.sum(joint, axis=0)
            expected = np.outer(px, py)
            occupied = joint > 0.0
            mi = float(
                np.sum(
                    (joint[occupied] / total)
                    * np.log((joint[occupied] * total) / expected[occupied])
                )
            )
            output[left, right] = output[right, left] = max(0.0, mi)
            if valid_pairs_from_gt is None:
                valid[left, right] = valid[right, left] = True
    np.fill_diagonal(valid, False)
    return output, valid


def compute_dependency_statistics(
    matrix: np.ndarray,
    *,
    bin_edges: Sequence[np.ndarray] | None = None,
    pearson_valid_pairs: np.ndarray | None = None,
    mi_valid_pairs: np.ndarray | None = None,
    max_bins: int = DEFAULT_MAX_BINS,
    variance_eps: float = DEFAULT_VARIANCE_EPS,
) -> DependencyStatistics:
    values = np.asarray(matrix, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] < 2:
        raise ValueError(f"Dependency matrix must have shape [N>=2, V>=2], got {values.shape}")
    if not np.all(np.isfinite(values)):
        raise ValueError("Dependency samples contain NaN or infinity")
    edges = tuple(bin_edges) if bin_edges is not None else derive_bin_edges(values, max_bins=max_bins)
    pearson, pearson_valid = _pearson_matrix(
        values,
        variance_eps=variance_eps,
        valid_pairs_from_gt=pearson_valid_pairs,
    )
    mutual_info, mi_valid = _mutual_info_matrix(
        values,
        edges,
        valid_pairs_from_gt=mi_valid_pairs,
    )
    return DependencyStatistics(
        pearson=pearson,
        mutual_info=mutual_info,
        pearson_valid_pairs=pearson_valid,
        mi_valid_pairs=mi_valid,
        bin_edges=edges,
        sample_count=int(values.shape[0]),
    )


def dependency_errors(
    gt: DependencyStatistics,
    reconstruction: DependencyStatistics,
) -> dict[str, Any]:
    pearson_abs = np.abs(reconstruction.pearson - gt.pearson)
    mi_abs = np.abs(reconstruction.mutual_info - gt.mutual_info)
    upper = np.triu(np.ones(gt.pearson.shape, dtype=bool), k=1)
    pearson_mask = upper & gt.pearson_valid_pairs
    mi_mask = upper & gt.mi_valid_pairs
    pearson_values = pearson_abs[pearson_mask]
    mi_values = mi_abs[mi_mask]
    return {
        "pearson_error": float(np.mean(pearson_values)) if pearson_values.size else float("nan"),
        "mi_error": float(np.mean(mi_values)) if mi_values.size else float("nan"),
        "pearson_valid_pair_count": int(pearson_values.size),
        "mi_valid_pair_count": int(mi_values.size),
        "pearson_error_matrix": pearson_abs,
        "mi_error_matrix": mi_abs,
    }


def pack_bin_edges(
    edges_by_timestep: Sequence[Sequence[np.ndarray]],
    *,
    max_bins: int,
) -> tuple[np.ndarray, np.ndarray]:
    timesteps = len(edges_by_timestep)
    channels = len(edges_by_timestep[0]) if timesteps else 0
    packed = np.full((timesteps, channels, int(max_bins) - 1), np.nan, dtype=np.float64)
    counts = np.zeros((timesteps, channels), dtype=np.int16)
    for time_index, channel_edges in enumerate(edges_by_timestep):
        for channel, edges in enumerate(channel_edges):
            interior = np.asarray(edges, dtype=np.float64)[1:-1]
            packed[time_index, channel, : interior.size] = interior
            counts[time_index, channel] = int(interior.size)
    return packed, counts


def unpack_bin_edges(packed: np.ndarray, counts: np.ndarray, timestep_position: int) -> tuple[np.ndarray, ...]:
    result = []
    for channel in range(int(packed.shape[1])):
        count = int(counts[timestep_position, channel])
        interior = np.asarray(packed[timestep_position, channel, :count], dtype=np.float64)
        result.append(np.concatenate(([-np.inf], interior, [np.inf])))
    return tuple(result)


def load_dependency_cache(path: str | Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    cache_path = Path(path)
    metadata_path = cache_path.with_suffix(".json")
    if not cache_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(f"Dependency GT cache is incomplete: {cache_path}")
    arrays = dict(np.load(cache_path, allow_pickle=False))
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if int(metadata.get("schema_version", -1)) != DEPENDENCY_SCHEMA_VERSION:
        raise ValueError(f"Unsupported dependency cache schema: {metadata.get('schema_version')}")
    if metadata.get("algorithm_version") != DEPENDENCY_ALGORITHM_VERSION:
        raise ValueError(f"Unsupported dependency cache algorithm: {metadata.get('algorithm_version')!r}")
    return arrays, metadata


def cache_statistics(arrays: Mapping[str, np.ndarray], timestep_position: int) -> DependencyStatistics:
    return DependencyStatistics(
        pearson=np.asarray(arrays["pearson_gt"][timestep_position]),
        mutual_info=np.asarray(arrays["mi_gt"][timestep_position]),
        pearson_valid_pairs=np.asarray(arrays["pearson_valid_pairs"][timestep_position], dtype=bool),
        mi_valid_pairs=np.asarray(arrays["mi_valid_pairs"][timestep_position], dtype=bool),
        bin_edges=unpack_bin_edges(arrays["bin_interiors"], arrays["bin_interior_counts"], timestep_position),
        sample_count=int(arrays["sample_counts"][timestep_position]),
    )


def aggregate_dependency_rows(
    rows: Iterable[Mapping[str, Any]],
) -> dict[str, float | None]:
    records = list(rows)
    output: dict[str, float | None] = {}
    for metric in ("pearson_error", "mi_error"):
        values = [float(row[metric]) for row in records if row.get(metric) is not None and math.isfinite(float(row[metric]))]
        output[metric] = float(np.mean(values)) if values else None
    return output
