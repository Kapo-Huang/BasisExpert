from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from ..config.io import load_evaluation_experiment_config
from .dependency import (
    DEFAULT_MAX_BINS,
    DEFAULT_SAMPLE_RATIO,
    DEFAULT_SAMPLE_SEED,
    DEFAULT_VARIANCE_EPS,
    DEPENDENCY_ALGORITHM_VERSION,
    DEPENDENCY_SCHEMA_VERSION,
    DependencyTarget,
    compute_dependency_statistics,
    pack_bin_edges,
    sample_index_digest,
    sampled_channel,
    stable_sample_indices,
)
from .data_paths import normalize_experiment_data_paths
from .ground_truth import target_paths_from_config


def _path_fingerprint(path: str | Path) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    stat = resolved.stat()
    return {
        "path": str(resolved),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def load_dependency_configuration(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("datasets"), dict):
        raise ValueError(f"Dependency configuration must define a datasets mapping: {config_path}")
    payload["_path"] = str(config_path)
    return payload


def dependency_targets(dataset_spec: dict[str, Any]) -> tuple[DependencyTarget, ...]:
    raw = dataset_spec.get("targets")
    if not isinstance(raw, list) or len(raw) < 2:
        raise ValueError("Each dependency dataset needs at least two targets")
    targets = []
    for item in raw:
        if isinstance(item, str):
            targets.append(DependencyTarget(item))
        elif isinstance(item, dict) and item.get("name"):
            targets.append(DependencyTarget(str(item["name"]), str(item.get("transform", "scalar"))))
        else:
            raise ValueError(f"Invalid dependency target specification: {item!r}")
    names = [target.name for target in targets]
    if len(set(names)) != len(names):
        raise ValueError(f"Dependency targets contain duplicates: {names}")
    return tuple(targets)


def _repo_root(configuration: dict[str, Any]) -> Path:
    return Path(configuration["_path"]).resolve().parents[2]


def dependency_cache_path(
    configuration: dict[str, Any],
    dataset_name: str,
) -> Path:
    root_value = configuration.get("cache_root", "EvalResult/artifacts/dependency_gt")
    root = Path(root_value)
    if not root.is_absolute():
        root = _repo_root(configuration) / root
    return root.resolve() / dataset_name.lower() / "cache.npz"


def _node_indexers(coords: np.ndarray) -> list[slice]:
    if coords.ndim != 2 or coords.shape[0] == 0:
        raise ValueError(f"Invalid node coordinate array shape: {coords.shape}")
    times = coords[:, -1]
    boundaries: list[int] = []
    previous = times[0]
    for start in range(1, len(times), 1_000_000):
        stop = min(len(times), start + 1_000_000)
        block = np.asarray(times[start:stop])
        if block[0] != previous:
            boundaries.append(start)
        boundaries.extend(int(item) for item in (np.flatnonzero(block[1:] != block[:-1]) + start + 1))
        previous = block[-1]
    starts, stops = [0, *boundaries], [*boundaries, len(times)]
    indexers = [slice(int(start), int(stop)) for start, stop in zip(starts, stops)]
    labels = [float(times[indexer.start]) for indexer in indexers]
    if len(set(labels)) != len(labels):
        raise ValueError("Node timesteps must be stored in contiguous coordinate blocks")
    return indexers


def _sample_matrix_from_paths(
    paths: dict[str, Path],
    targets: tuple[DependencyTarget, ...],
    indices: np.ndarray,
    *,
    timestep: int,
    indexer: slice,
    volume: bool,
) -> np.ndarray:
    """Gather only selected rows, avoiding large flat-file frame views on Windows."""
    columns: list[np.ndarray] = []
    sample_rows = np.arange(indices.size, dtype=np.int64)
    for target in targets:
        array = np.load(paths[target.name], mmap_mode="r", allow_pickle=False)
        try:
            if volume and array.ndim >= 4:
                frame = np.asarray(array[int(timestep)])
                values = sampled_channel(
                    frame,
                    indices,
                    frame_size=int(indexer.stop - indexer.start),
                    transform=target.transform,
                )
            else:
                selected = np.asarray(array[int(indexer.start) + indices])
                values = sampled_channel(
                    selected,
                    sample_rows,
                    frame_size=int(indices.size),
                    transform=target.transform,
                )
        finally:
            memory_map = getattr(array, "_mmap", None)
            if memory_map is not None:
                memory_map.close()
        if not np.all(np.isfinite(values)):
            raise ValueError(
                f"Non-finite GT values in dataset target={target.name!r}, timestep={timestep}"
            )
        columns.append(values)
    return np.column_stack(columns)


def _validate_existing_cache(
    cache_path: Path,
    metadata_path: Path,
    *,
    dataset_name: str,
    experiment_config_path: Path,
    targets: tuple[DependencyTarget, ...],
    gt_paths: dict[str, Path],
    timestep_count: int,
    sample_ratio: float,
    seed: int,
    max_bins: int,
    variance_eps: float,
) -> None:
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        arrays = np.load(cache_path, allow_pickle=False)
        cached_names = tuple(str(value) for value in arrays["target_names"])
        cached_transforms = tuple(str(value) for value in arrays["target_transforms"])
        cached_timestep_count = int(arrays["timesteps"].size)
        arrays.close()
    except Exception as exc:
        raise ValueError(
            f"Dependency GT cache is unreadable: {cache_path}; rerun with --overwrite"
        ) from exc
    expected_names = tuple(target.name for target in targets)
    expected_transforms = tuple(target.transform for target in targets)
    expected = {
        "schema_version": DEPENDENCY_SCHEMA_VERSION,
        "algorithm_version": DEPENDENCY_ALGORITHM_VERSION,
        "dataset": str(dataset_name),
        "dataset_config": _path_fingerprint(experiment_config_path),
        "target_names": list(expected_names),
        "target_transforms": list(expected_transforms),
        "ground_truth": {name: _path_fingerprint(path) for name, path in gt_paths.items()},
        "timestep_count": int(timestep_count),
        "sample_ratio": float(sample_ratio),
        "sample_seed": int(seed),
        "max_bins": int(max_bins),
        "variance_eps": float(variance_eps),
    }
    mismatches = [
        name for name, value in expected.items() if metadata.get(name) != value
    ]
    if cached_names != expected_names:
        mismatches.append("npz.target_names")
    if cached_transforms != expected_transforms:
        mismatches.append("npz.target_transforms")
    if cached_timestep_count != int(timestep_count):
        mismatches.append("npz.timesteps")
    if mismatches:
        raise ValueError(
            f"Dependency GT cache is stale for {dataset_name}: mismatched "
            f"{sorted(set(mismatches))}; rerun with --overwrite"
        )


def build_dataset_dependency_cache(
    configuration: dict[str, Any],
    dataset_name: str,
    *,
    overwrite: bool = False,
) -> Path:
    datasets = configuration["datasets"]
    key = next((name for name in datasets if str(name).lower() == str(dataset_name).lower()), None)
    if key is None:
        raise KeyError(f"Unknown dependency dataset {dataset_name!r}; available: {list(datasets)}")
    dataset_spec = dict(datasets[key])
    repo_root = _repo_root(configuration)
    print(f"[{key}] loading dataset configuration", flush=True)
    experiment_config_path = Path(dataset_spec["config"])
    if not experiment_config_path.is_absolute():
        experiment_config_path = repo_root / experiment_config_path
    experiment = normalize_experiment_data_paths(
        load_evaluation_experiment_config(experiment_config_path),
        repo_root=repo_root,
    )
    print(f"[{key}] loaded dataset configuration", flush=True)
    targets = dependency_targets(dataset_spec)
    target_names = tuple(target.name for target in targets)
    all_paths = target_paths_from_config(experiment.data, repo_root=repo_root)
    print(f"[{key}] resolved GT paths", flush=True)
    missing = [name for name in target_names if name not in all_paths]
    if missing:
        raise FileNotFoundError(f"GT target paths are missing for {key}: {missing}")
    gt_paths = {name: all_paths[name].expanduser().resolve() for name in target_names}
    missing_files = [name for name, path in gt_paths.items() if not path.is_file()]
    if missing_files:
        raise FileNotFoundError(f"GT files are missing for {key}: {missing_files}")
    array_shapes: dict[str, tuple[int, ...]] = {}
    for name, path in gt_paths.items():
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        array_shapes[name] = tuple(int(value) for value in array.shape)
        memory_map = getattr(array, "_mmap", None)
        if memory_map is not None:
            memory_map.close()
    print(f"[{key}] validated {len(array_shapes)} GT headers", flush=True)
    if experiment.data.volume_shape is not None:
        volume = True
        shape = experiment.data.volume_shape
        frame_size = int(shape.X) * int(shape.Y) * int(shape.Z)
        indexers = [slice(t * frame_size, (t + 1) * frame_size) for t in range(int(shape.T))]
        expected_samples = int(shape.T) * frame_size
        for name, array_shape in array_shapes.items():
            actual_samples = (
                int(np.prod(array_shape[:4], dtype=np.int64))
                if len(array_shape) >= 4
                else int(array_shape[0])
            )
            if actual_samples != expected_samples:
                raise ValueError(
                    f"GT shape mismatch for {key}/{name}: expected {expected_samples} "
                    f"samples, got {array_shape}"
                )
    else:
        volume = False
        coords_path = Path(str(experiment.data.coords_path))
        coords = np.load(coords_path, mmap_mode="r", allow_pickle=False)
        indexers = _node_indexers(coords)
        for name, array_shape in array_shapes.items():
            if not array_shape or int(array_shape[0]) != int(coords.shape[0]):
                raise ValueError(
                    f"GT shape mismatch for {key}/{name}: expected first dimension "
                    f"{coords.shape[0]}, got {array_shape}"
                )
    print(f"[{key}] prepared {len(indexers)} timestep indexers", flush=True)

    cache_path = dependency_cache_path(configuration, str(key))
    metadata_path = cache_path.with_suffix(".json")
    settings = dict(configuration.get("settings") or {})
    sample_ratio = float(settings.get("sample_ratio", DEFAULT_SAMPLE_RATIO))
    seed = int(settings.get("seed", DEFAULT_SAMPLE_SEED))
    max_bins = int(settings.get("max_bins", DEFAULT_MAX_BINS))
    variance_eps = float(settings.get("variance_eps", DEFAULT_VARIANCE_EPS))
    if cache_path.exists() != metadata_path.exists() and not overwrite:
        raise ValueError(
            f"Dependency GT cache is incomplete: {cache_path}; rerun with --overwrite"
        )
    if cache_path.is_file() and metadata_path.is_file() and not overwrite:
        _validate_existing_cache(
            cache_path,
            metadata_path,
            dataset_name=str(key),
            experiment_config_path=experiment_config_path,
            targets=targets,
            gt_paths=gt_paths,
            timestep_count=len(indexers),
            sample_ratio=sample_ratio,
            seed=seed,
            max_bins=max_bins,
            variance_eps=variance_eps,
        )
        return cache_path
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    time_count = len(indexers)
    channels = len(targets)
    pearson_gt = np.zeros((time_count, channels, channels), dtype=np.float64)
    mi_gt = np.zeros_like(pearson_gt)
    pearson_valid = np.zeros_like(pearson_gt, dtype=bool)
    mi_valid = np.zeros_like(pearson_gt, dtype=bool)
    frame_counts = np.zeros(time_count, dtype=np.int64)
    sample_counts = np.zeros(time_count, dtype=np.int64)
    index_digests = np.empty(time_count, dtype="<U64")
    edges_by_timestep: list[tuple[np.ndarray, ...]] = []
    print(f"[{key}] allocated cache outputs", flush=True)

    for timestep, indexer in enumerate(indexers):
        current_frame_size = int(indexer.stop - indexer.start)
        if current_frame_size >= 1_000_000:
            print(
                f"[{key}] timestep {timestep + 1}/{time_count}: selecting indices",
                flush=True,
            )
        indices = stable_sample_indices(
            current_frame_size,
            timestep=timestep,
            sample_ratio=sample_ratio,
            seed=seed,
        )
        matrix = _sample_matrix_from_paths(
            gt_paths,
            targets,
            indices,
            timestep=timestep,
            indexer=indexer,
            volume=volume,
        )
        if current_frame_size >= 1_000_000:
            print(
                f"[{key}] timestep {timestep + 1}/{time_count}: computing statistics",
                flush=True,
            )
        stats = compute_dependency_statistics(
            matrix,
            max_bins=max_bins,
            variance_eps=variance_eps,
        )
        pearson_gt[timestep] = stats.pearson
        mi_gt[timestep] = stats.mutual_info
        pearson_valid[timestep] = stats.pearson_valid_pairs
        mi_valid[timestep] = stats.mi_valid_pairs
        frame_counts[timestep] = current_frame_size
        sample_counts[timestep] = int(indices.size)
        index_digests[timestep] = sample_index_digest(indices)
        edges_by_timestep.append(stats.bin_edges)
        print(
            f"[{key}] timestep {timestep + 1}/{time_count}: "
            f"frame={current_frame_size} sample={indices.size}",
            flush=True,
        )

    bin_interiors, bin_interior_counts = pack_bin_edges(edges_by_timestep, max_bins=max_bins)
    temporary = cache_path.with_name(cache_path.stem + ".tmp.npz")
    np.savez_compressed(
        temporary,
        target_names=np.asarray(target_names, dtype=str),
        target_transforms=np.asarray([target.transform for target in targets], dtype=str),
        timesteps=np.arange(time_count, dtype=np.int64),
        pearson_gt=pearson_gt,
        mi_gt=mi_gt,
        pearson_valid_pairs=pearson_valid,
        mi_valid_pairs=mi_valid,
        bin_interiors=bin_interiors,
        bin_interior_counts=bin_interior_counts,
        frame_counts=frame_counts,
        sample_counts=sample_counts,
        sample_index_digests=index_digests,
    )
    os.replace(temporary, cache_path)
    metadata = {
        "schema_version": DEPENDENCY_SCHEMA_VERSION,
        "algorithm_version": DEPENDENCY_ALGORITHM_VERSION,
        "dataset": str(key),
        "dataset_config": _path_fingerprint(experiment_config_path),
        "target_names": list(target_names),
        "target_transforms": [target.transform for target in targets],
        "ground_truth": {name: _path_fingerprint(path) for name, path in gt_paths.items()},
        "timestep_count": time_count,
        "sample_ratio": sample_ratio,
        "sample_seed": seed,
        "sampling": "splitmix64_threshold",
        "max_bins": max_bins,
        "binning": "per_timestep_gt_quantiles",
        "mi_log_base": "e",
        "mi_unit": "nats",
        "variance_eps": variance_eps,
    }
    temporary_metadata = metadata_path.with_name(metadata_path.stem + ".tmp.json")
    temporary_metadata.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary_metadata, metadata_path)
    return cache_path


def build_dependency_caches(
    config_path: str | Path,
    *,
    dataset: str = "all",
    overwrite: bool = False,
) -> list[Path]:
    configuration = load_dependency_configuration(config_path)
    selected = (
        list(configuration["datasets"])
        if str(dataset).lower() == "all"
        else [str(dataset)]
    )
    return [
        build_dataset_dependency_cache(configuration, name, overwrite=overwrite)
        for name in selected
    ]
