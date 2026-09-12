from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import yaml

from ..config.io import load_evaluation_experiment_config
from ..training.engine import _predict_batch
from .dependency import (
    DEPENDENCY_ALGORITHM_VERSION,
    DependencyTarget,
    aggregate_dependency_rows,
    cache_statistics,
    compute_dependency_statistics,
    dependency_errors,
    load_dependency_cache,
    sample_index_digest,
    sampled_channel,
    stable_sample_indices,
)
from .dependency_cache import (
    dependency_cache_path,
    dependency_targets,
    load_dependency_configuration,
)
from .artifacts import ArtifactStore, LAYOUT_SCHEMA_VERSION
from .data_paths import normalize_experiment_data_paths, normalize_raw_data_paths
from .ground_truth import target_paths_from_config
from .reporting import (
    cache_key,
    environment_manifest,
    find_cached_evaluation,
    path_fingerprint,
    write_json,
    write_metrics_csv,
)
from .selection import parse_timestep_selection


DEPENDENCY_METRICS = frozenset({"pearson_error", "mi_error"})


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _mapping(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a YAML mapping: {path}")
    return payload


def _section(raw: Mapping[str, Any], lower: str, upper: str) -> dict[str, Any]:
    value = raw.get(lower)
    if not isinstance(value, dict):
        value = raw.get(upper)
    return dict(value) if isinstance(value, dict) else {}


def _run_config(run_dir: Path) -> Path:
    candidates = (
        run_dir / "configs" / "config.yaml",
        run_dir / "config.yaml",
    )
    for path in candidates:
        if path.is_file():
            return path
    searched = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"Saved run config does not exist; searched: {searched}")


def _dataset_name(raw: Mapping[str, Any]) -> str:
    name = str(_section(raw, "data", "DATA").get("dataset_name", "")).strip()
    if not name:
        raise ValueError("Run config does not define dataset_name")
    return name


def _selected_target(raw: Mapping[str, Any]) -> str | None:
    data = _section(raw, "data", "DATA")
    value = data.get("target")
    if value not in (None, ""):
        return str(value)
    configured = data.get("targets")
    if isinstance(configured, dict) and len(configured) == 1:
        return str(next(iter(configured)))
    return None


@dataclass(frozen=True)
class DependencyRunGroup:
    dataset: str
    anchor: Path
    targets: tuple[DependencyTarget, ...]
    run_by_target: Mapping[str, Path]
    joint_run: Path | None

    @property
    def run_dirs(self) -> tuple[Path, ...]:
        if self.joint_run is not None:
            return (self.joint_run,)
        return tuple(self.run_by_target[target.name] for target in self.targets)


def resolve_dependency_group(
    run_dir: str | Path,
    configuration: dict[str, Any],
) -> DependencyRunGroup:
    resolved = Path(run_dir).expanduser().resolve()
    raw = _mapping(_run_config(resolved))
    dataset = _dataset_name(raw)
    dataset_key = next(
        (name for name in configuration["datasets"] if str(name).lower() == dataset.lower()),
        None,
    )
    if dataset_key is None:
        raise KeyError(f"Dataset {dataset!r} has no dependency specification")
    targets = dependency_targets(dict(configuration["datasets"][dataset_key]))
    selected = _selected_target(raw)
    if selected is None:
        return DependencyRunGroup(
            dataset=str(dataset_key),
            anchor=resolved,
            targets=targets,
            run_by_target={target.name: resolved for target in targets},
            joint_run=resolved,
        )

    siblings: dict[str, Path] = {}
    for candidate in sorted(resolved.parent.iterdir()):
        if not candidate.is_dir():
            continue
        try:
            candidate_config = _run_config(candidate)
        except FileNotFoundError:
            continue
        candidate_raw = _mapping(candidate_config)
        if _dataset_name(candidate_raw).lower() != dataset.lower():
            continue
        candidate_target = _selected_target(candidate_raw)
        if candidate_target is None:
            continue
        if candidate_target in siblings:
            raise ValueError(
                f"Duplicate sibling dependency target {candidate_target!r}: "
                f"{siblings[candidate_target]} and {candidate}"
            )
        siblings[candidate_target] = candidate.resolve()
    required = [target.name for target in targets]
    missing = [name for name in required if name not in siblings]
    if missing:
        raise ValueError(
            f"Incomplete dependency group under {resolved.parent}: missing {missing}; "
            f"available={sorted(siblings)}"
        )
    return DependencyRunGroup(
        dataset=str(dataset_key),
        anchor=resolved.parent.resolve(),
        targets=targets,
        run_by_target={name: siblings[name] for name in required},
        joint_run=None,
    )


def discover_dependency_groups(
    run_root: str | Path,
    configuration: dict[str, Any],
) -> tuple[DependencyRunGroup, ...]:
    root = Path(run_root).expanduser().resolve()
    candidates = sorted(
        {
            path.parent.parent if path.parent.name == "configs" else path.parent
            for path in root.rglob("config.yaml")
        }
    )
    groups: dict[Path, DependencyRunGroup] = {}
    failures: dict[Path, Exception] = {}
    for run_dir in candidates:
        try:
            group = resolve_dependency_group(run_dir, configuration)
        except (KeyError, ValueError, FileNotFoundError) as exc:
            failures[run_dir] = exc
            continue
        groups.setdefault(group.anchor, group)
    if not groups and failures:
        example = next(iter(failures.values()))
        raise ValueError(f"No complete dependency groups found under {root}: {example}")
    return tuple(groups[path] for path in sorted(groups))


def canonical_dependency_run_dirs(
    run_dirs: Sequence[str | Path],
    configuration: dict[str, Any],
) -> tuple[Path, ...]:
    """Collapse sibling runs to one canonical entry per dependency group.

    Runs that cannot be resolved are preserved so the normal evaluation path
    still reports their original configuration or completeness error.
    """
    canonical: list[Path] = []
    seen_anchors: set[Path] = set()
    for run_dir in run_dirs:
        resolved = Path(run_dir).expanduser().resolve()
        try:
            group = resolve_dependency_group(resolved, configuration)
        except (KeyError, ValueError, FileNotFoundError):
            canonical.append(resolved)
            continue
        if group.anchor in seen_anchors:
            continue
        seen_anchors.add(group.anchor)
        if group.joint_run is not None:
            canonical.append(group.joint_run)
        else:
            canonical.append(group.run_by_target[group.targets[0].name])
    return tuple(canonical)


def _output_dir(
    group: DependencyRunGroup,
    result_root: str | Path,
    evaluation_id: str,
) -> Path:
    repo_root = _repo_root()
    root = Path(result_root)
    if not root.is_absolute():
        root = repo_root / root
    try:
        relative = group.anchor.relative_to(repo_root / "Result")
    except ValueError:
        try:
            relative = Path("_external") / group.anchor.relative_to(repo_root)
        except ValueError:
            relative = Path("_external") / group.dataset / group.anchor.name
    return (root / "evaluations" / evaluation_id / relative).resolve()


class _StandardSampleDecoder:
    def __init__(self, run_dir: Path, targets: tuple[str, ...], source: str, device: torch.device) -> None:
        from .service import (
            EvaluationRequest,
            _InferenceOnlyDataset,
            _frame_indexers,
            _load_prediction_arrays,
            _load_standard_model,
            _resolve_standard_source,
        )

        self.run_dir = run_dir
        self.config = normalize_experiment_data_paths(
            load_evaluation_experiment_config(_run_config(run_dir)),
            repo_root=_repo_root(),
        )
        available = tuple(
            [self.config.data.target]
            if self.config.data.target
            else list(self.config.data.targets.keys())
        )
        self.dataset = _InferenceOnlyDataset(self.config, available)
        self.indexers = _frame_indexers(self.dataset)
        request = EvaluationRequest(run_dir=run_dir, source=source, metrics=("pearson_error",))
        self.source_kind, self.source_path = _resolve_standard_source(request, self.config)
        self.device = device
        self.targets = targets
        if self.source_kind == "prediction":
            self.arrays = _load_prediction_arrays(
                self.source_path,
                exp_id=self.config.exp_id,
                targets=targets,
            )
            self.model = None
        else:
            self.arrays = None
            self.model = _load_standard_model(self.config, self.dataset, device, self.source_path)

    @property
    def fingerprint(self) -> dict[str, Any]:
        if self.arrays is not None and len(self.targets) > 1:
            return path_fingerprint(self.source_path.parent)
        return path_fingerprint(self.source_path)

    def sample(self, timestep: int, local_indices: np.ndarray) -> dict[str, np.ndarray]:
        from .service import _prediction_frame

        indexer = self.indexers[int(timestep)]
        if self.arrays is not None:
            result = {}
            frame_size = int(indexer.stop - indexer.start) if isinstance(indexer, slice) else len(indexer)
            for target in self.targets:
                frame = _prediction_frame(
                    self.arrays[target], self.dataset, target, int(timestep), indexer
                )
                result[target] = np.asarray(frame).reshape(frame_size, -1)[local_indices]
            return result
        if isinstance(indexer, slice):
            rows = local_indices + int(indexer.start)
        else:
            rows = np.asarray(indexer, dtype=np.int64)[local_indices]
        parts = {name: [] for name in self.targets}
        batch_size = int(self.config.evaluation.batch_size)
        self.model.eval()
        with torch.inference_mode():
            for start in range(0, int(rows.size), batch_size):
                batch_rows = rows[start : start + batch_size]
                batch = self.dataset.fetch_batch(batch_rows, include_targets=False)
                predictions = _predict_batch(
                    self.model,
                    batch.coords.to(self.device, non_blocking=True),
                    self.dataset.target_names(),
                    hard_topk=True,
                    target_dims=self.dataset.meta.target_dims,
                )
                for name in self.targets:
                    parts[name].append(predictions[name].detach().cpu().numpy())
        return {name: np.concatenate(values, axis=0) for name, values in parts.items()}


class _StandaloneSampleDecoder:
    def __init__(self, run_dir: Path, target: str, source: str, device: torch.device) -> None:
        from .standalone import (
            _data_section,
            _find_source,
            identify_subsystem,
        )

        self.run_dir = run_dir
        self.config_path = _run_config(run_dir)
        self.raw = normalize_raw_data_paths(
            _mapping(self.config_path),
            repo_root=_repo_root(),
            config_path=self.config_path,
        )
        self.subsystem = identify_subsystem(self.raw)
        if self.subsystem is None:
            raise ValueError(f"Run is not a standalone subsystem: {run_dir}")
        self.target = target
        self.device = device
        data = _data_section(self.raw)
        volume_cfg = data.get("volume_shape")
        self.shape_tzyx = None
        self.coords = None
        self.coords_path = None
        if volume_cfg:
            self.shape_tzyx = (
                int(volume_cfg["T"]), int(volume_cfg["Z"]),
                int(volume_cfg["Y"]), int(volume_cfg["X"]),
            )
            per_frame = int(np.prod(self.shape_tzyx[1:], dtype=np.int64))
            self.indexers = [slice(t * per_frame, (t + 1) * per_frame) for t in range(self.shape_tzyx[0])]
        else:
            value = data.get("coords_path") or data.get("source_path")
            self.coords_path = Path(str(value))
            self.coords = np.load(self.coords_path, mmap_mode="r", allow_pickle=False)
            from .dependency_cache import _node_indexers
            self.indexers = _node_indexers(self.coords)
        self.source_kind, self.source_path = _find_source(
            run_dir, self.subsystem, source, None, None
        )

    @property
    def fingerprint(self) -> dict[str, Any]:
        return path_fingerprint(self.source_path)

    def sample(self, timestep: int, local_indices: np.ndarray) -> np.ndarray:
        from .standalone import (
            _array_frame,
            _decode_apmgsrn_frames,
            _decode_miner_frames,
            _decode_neural_expert_frames,
            _invoke_predict,
            _portable_standalone_config,
            _prediction_paths,
        )

        decoded_positions = None
        if self.source_kind == "prediction":
            prediction_paths = _prediction_paths({}, self.source_path, (self.target,))
            array = np.load(prediction_paths[self.target], mmap_mode="r", allow_pickle=False)
            frame = _array_frame(
                array, timestep=timestep, indexer=self.indexers[timestep], shape_tzyx=self.shape_tzyx
            )
        elif self.subsystem in {"apmgsrn", "miner", "neural_expert"}:
            if self.subsystem == "apmgsrn":
                decoded = _decode_apmgsrn_frames(
                    self.source_path, self.raw, timesteps=(timestep,), targets=(self.target,),
                    shape_tzyx=self.shape_tzyx, device=self.device,
                )
            elif self.subsystem == "miner":
                decoded = _decode_miner_frames(
                    self.source_path, timesteps=(timestep,), targets=(self.target,),
                    shape_tzyx=self.shape_tzyx, device=self.device,
                )
            else:
                decoded = _decode_neural_expert_frames(
                    self.source_path, self.raw, timesteps=(timestep,), targets=(self.target,),
                    indexers=self.indexers, shape_tzyx=self.shape_tzyx, coords=self.coords,
                    repo_root=_repo_root(), config_path=self.config_path, device=self.device,
                )
            frame = decoded[(self.target, timestep)]
        else:
            with _portable_standalone_config(self.raw, device=str(self.device)) as portable_config:
                result = _invoke_predict(
                    self.subsystem, portable_config, self.source_kind, self.source_path,
                    self.target, (timestep,),
                )
            prediction_paths = _prediction_paths(result, self.source_path, (self.target,))
            if result.get("decoded_timesteps") is not None:
                decoded_positions = {
                    int(value): position for position, value in enumerate(result["decoded_timesteps"])
                }
            array = np.load(prediction_paths[self.target], mmap_mode="r", allow_pickle=False)
            frame = _array_frame(
                array, timestep=timestep, indexer=self.indexers[timestep],
                shape_tzyx=self.shape_tzyx, decoded_positions=decoded_positions,
            )
        frame_size = int(self.indexers[timestep].stop - self.indexers[timestep].start)
        return np.asarray(frame).reshape(frame_size, -1)[local_indices]


def _is_standalone(run_dir: Path) -> bool:
    from .standalone import identify_subsystem
    return identify_subsystem(_mapping(_run_config(run_dir))) is not None


def _make_decoders(
    group: DependencyRunGroup,
    *,
    source: str,
    device: torch.device,
) -> tuple[list[Any], dict[str, Any]]:
    fingerprints: dict[str, Any] = {}
    if group.joint_run is not None:
        if _is_standalone(group.joint_run):
            raise ValueError("Standalone dependency checkpoints must use one sibling run per target")
        decoder = _StandardSampleDecoder(
            group.joint_run, tuple(target.name for target in group.targets), source, device
        )
        fingerprints[str(group.joint_run)] = decoder.fingerprint
        return [decoder], fingerprints
    decoders = []
    for target in group.targets:
        run_dir = group.run_by_target[target.name]
        decoder = (
            _StandaloneSampleDecoder(run_dir, target.name, source, device)
            if _is_standalone(run_dir)
            else _StandardSampleDecoder(run_dir, (target.name,), source, device)
        )
        decoders.append(decoder)
        fingerprints[str(run_dir)] = decoder.fingerprint
    return decoders, fingerprints


def evaluate_dependency_group(
    group: DependencyRunGroup,
    *,
    configuration: dict[str, Any],
    metrics: Sequence[str] = ("pearson_error", "mi_error"),
    timesteps: str = "all",
    source: str = "checkpoint",
    overwrite: bool = False,
    device: str | None = None,
    result_root: str | Path = "EvalResult",
    evaluation_id: str = "dependency",
) -> dict[str, Any]:
    requested = tuple(str(metric).lower() for metric in metrics)
    unknown = set(requested).difference(DEPENDENCY_METRICS)
    if unknown or not requested:
        raise ValueError(f"Dependency evaluation accepts only {sorted(DEPENDENCY_METRICS)}; got {requested}")
    cache_path = dependency_cache_path(configuration, group.dataset)
    cache_arrays, cache_metadata = load_dependency_cache(cache_path)
    dataset_spec = dict(configuration["datasets"][group.dataset])
    configuration_root = Path(configuration["_path"]).resolve().parents[2]
    canonical_config_path = Path(dataset_spec["config"])
    if not canonical_config_path.is_absolute():
        canonical_config_path = configuration_root / canonical_config_path
    canonical_config = load_evaluation_experiment_config(canonical_config_path)
    if cache_metadata.get("dataset_config") != path_fingerprint(canonical_config_path):
        raise ValueError(
            f"Dependency GT cache dataset config is stale for {group.dataset}; "
            f"rebuild {cache_path}"
        )
    current_gt_paths = target_paths_from_config(canonical_config.data, repo_root=configuration_root)
    current_gt_fingerprints = {
        target.name: path_fingerprint(current_gt_paths[target.name]) for target in group.targets
    }
    if cache_metadata.get("ground_truth") != current_gt_fingerprints:
        raise ValueError(
            f"Dependency GT cache is stale for {group.dataset}; rebuild {cache_path}"
        )
    configured_settings = dict(configuration.get("settings") or {})
    for config_key, metadata_key in (
        ("sample_ratio", "sample_ratio"),
        ("seed", "sample_seed"),
        ("max_bins", "max_bins"),
        ("variance_eps", "variance_eps"),
    ):
        if config_key in configured_settings and configured_settings[config_key] != cache_metadata.get(metadata_key):
            raise ValueError(
                f"Dependency GT cache setting {metadata_key} is stale: "
                f"{cache_metadata.get(metadata_key)!r} != {configured_settings[config_key]!r}"
            )
    cached_names = tuple(str(value) for value in cache_arrays["target_names"])
    cached_transforms = tuple(str(value) for value in cache_arrays["target_transforms"])
    expected_names = tuple(target.name for target in group.targets)
    expected_transforms = tuple(target.transform for target in group.targets)
    if cached_names != expected_names or cached_transforms != expected_transforms:
        raise ValueError(
            f"Dependency cache target layout mismatch: cache={(cached_names, cached_transforms)} "
            f"group={(expected_names, expected_transforms)}"
        )
    total_timesteps = int(cache_arrays["timesteps"].size)
    selected_timesteps = parse_timestep_selection(timesteps, total_timesteps)
    torch_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    decoders, source_fingerprints = _make_decoders(group, source=source, device=torch_device)
    artifacts = ArtifactStore(repo_root=_repo_root(), result_root=result_root)
    artifacts.initialize()
    key_payload = {
        "schema_version": LAYOUT_SCHEMA_VERSION,
        "algorithm_version": DEPENDENCY_ALGORITHM_VERSION,
        "dataset": group.dataset,
        "targets": list(expected_names),
        "metrics": list(requested),
        "timesteps": list(selected_timesteps),
        "sources": source_fingerprints,
        "gt_cache": artifacts.content_fingerprint(cache_path),
        "gt_cache_metadata": artifacts.content_fingerprint(cache_path.with_suffix(".json")),
    }
    evaluation_key = cache_key(key_payload)
    output_dir = _output_dir(group, result_root, evaluation_id)
    if not overwrite:
        cached = find_cached_evaluation(output_dir, evaluation_key)
        if cached is not None:
            return cached
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    pearson_recon = []
    mi_recon = []
    pearson_error_matrices = []
    mi_error_matrices = []
    for timestep in selected_timesteps:
        position = int(timestep)
        frame_size = int(cache_arrays["frame_counts"][position])
        indices = stable_sample_indices(
            frame_size,
            timestep=timestep,
            sample_ratio=float(cache_metadata["sample_ratio"]),
            seed=int(cache_metadata["sample_seed"]),
        )
        expected_count = int(cache_arrays["sample_counts"][position])
        expected_digest = str(cache_arrays["sample_index_digests"][position])
        actual_digest = sample_index_digest(indices)
        if int(indices.size) != expected_count or actual_digest != expected_digest:
            raise ValueError(
                f"GT/reconstruction sampling mismatch at timestep {timestep}: "
                f"count={indices.size}/{expected_count}, digest={actual_digest}/{expected_digest}"
            )
        columns: dict[str, np.ndarray] = {}
        if group.joint_run is not None:
            raw_samples = decoders[0].sample(timestep, indices)
            for target in group.targets:
                values = np.asarray(raw_samples[target.name])
                columns[target.name] = sampled_channel(
                    values, np.arange(values.shape[0]), frame_size=values.shape[0],
                    transform=target.transform,
                )
        else:
            for target, decoder in zip(group.targets, decoders):
                raw_samples = decoder.sample(timestep, indices)
                if isinstance(raw_samples, Mapping):
                    raw_samples = raw_samples[target.name]
                values = np.asarray(raw_samples)
                columns[target.name] = sampled_channel(
                    values, np.arange(values.shape[0]), frame_size=values.shape[0],
                    transform=target.transform,
                )
        matrix = np.column_stack([columns[target.name] for target in group.targets])
        gt_stats = cache_statistics(cache_arrays, position)
        reconstruction_stats = compute_dependency_statistics(
            matrix,
            bin_edges=gt_stats.bin_edges,
            pearson_valid_pairs=gt_stats.pearson_valid_pairs,
            mi_valid_pairs=gt_stats.mi_valid_pairs,
            variance_eps=float(cache_metadata["variance_eps"]),
        )
        errors = dependency_errors(gt_stats, reconstruction_stats)
        row = {
            "row_type": "dependency_per_timestep",
            "timestep": int(timestep),
            "sample_count": int(indices.size),
            "pearson_valid_pair_count": errors["pearson_valid_pair_count"],
            "mi_valid_pair_count": errors["mi_valid_pair_count"],
        }
        if "pearson_error" in requested:
            row["pearson_error"] = errors["pearson_error"]
        if "mi_error" in requested:
            row["mi_error"] = errors["mi_error"]
        rows.append(row)
        pearson_recon.append(reconstruction_stats.pearson)
        mi_recon.append(reconstruction_stats.mutual_info)
        pearson_error_matrices.append(errors["pearson_error_matrix"])
        mi_error_matrices.append(errors["mi_error_matrix"])
        write_json(
            output_dir / "progress.json",
            {"status": "running", "completed_rows": len(rows), "per_timestep": rows},
        )
        write_metrics_csv(output_dir / "metrics.csv", rows)

    aggregate = aggregate_dependency_rows(rows)
    payload = {
        "schema_version": LAYOUT_SCHEMA_VERSION,
        "status": "complete",
        "aggregate": {key: value for key, value in aggregate.items() if key in requested},
        "dependency": {
            "dataset": group.dataset,
            "target_names": list(expected_names),
            "target_transforms": list(expected_transforms),
            "sample_ratio": float(cache_metadata["sample_ratio"]),
            "mi_unit": "nats",
        },
        "per_timestep": rows,
    }
    manifest = {
        "schema_version": LAYOUT_SCHEMA_VERSION,
        "evaluation_id": evaluation_id,
        "result_root": ".",
        "artifact_root": "artifacts",
        "cache_key": evaluation_key,
        "algorithm_version": DEPENDENCY_ALGORITHM_VERSION,
        "dataset_name": group.dataset,
        "group_anchor": str(group.anchor),
        "run_dirs": [str(path) for path in group.run_dirs],
        "joint": group.joint_run is not None,
        "metrics": list(requested),
        "timesteps": list(selected_timesteps),
        "targets": list(expected_names),
        "sources": source_fingerprints,
        "ground_truth_cache": {
            "path": artifacts.reference(cache_path),
            "fingerprint": artifacts.content_fingerprint(cache_path),
            "metadata": cache_metadata,
        },
        "device": str(torch_device),
        "environment": environment_manifest(),
    }
    np.savez_compressed(
        output_dir / "dependency_metrics.npz",
        target_names=np.asarray(expected_names, dtype=str),
        timesteps=np.asarray(selected_timesteps, dtype=np.int64),
        pearson_gt=np.asarray(cache_arrays["pearson_gt"])[list(selected_timesteps)],
        mi_gt=np.asarray(cache_arrays["mi_gt"])[list(selected_timesteps)],
        pearson_reconstruction=np.asarray(pearson_recon),
        mi_reconstruction=np.asarray(mi_recon),
        pearson_error=np.asarray(pearson_error_matrices),
        mi_error=np.asarray(mi_error_matrices),
        pearson_valid_pairs=np.asarray(cache_arrays["pearson_valid_pairs"])[list(selected_timesteps)],
        mi_valid_pairs=np.asarray(cache_arrays["mi_valid_pairs"])[list(selected_timesteps)],
    )
    manifest_path = write_json(output_dir / "manifest.json", manifest)
    metrics_path = write_json(output_dir / "metrics.json", payload)
    csv_path = write_metrics_csv(output_dir / "metrics.csv", rows)
    write_json(output_dir / "progress.json", {"status": "complete", "completed_rows": len(rows)})
    log_path = output_dir / "logs" / "evaluate.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(
        f"dataset={group.dataset}\ngroup={group.anchor}\nstatus=complete\n",
        encoding="utf-8",
    )
    return {
        "output_dir": output_dir,
        "manifest_path": manifest_path,
        "metrics_path": metrics_path,
        "csv_path": csv_path,
        "log_path": log_path,
        "metrics": payload,
    }


def evaluate_dependency_run(
    run_dir: str | Path,
    *,
    metrics: Sequence[str] = ("pearson_error", "mi_error"),
    timesteps: str = "all",
    source: str = "checkpoint",
    overwrite: bool = False,
    device: str | None = None,
    result_root: str | Path = "EvalResult",
    evaluation_id: str = "dependency",
    dependency_config: str | Path | None = None,
) -> dict[str, Any]:
    config_path = dependency_config or (
        _repo_root() / "configs/evaluation/evaluation_result_dependency.yaml"
    )
    configuration = load_dependency_configuration(config_path)
    group = resolve_dependency_group(run_dir, configuration)
    return evaluate_dependency_group(
        group,
        configuration=configuration,
        metrics=metrics,
        timesteps=timesteps,
        source=source,
        overwrite=overwrite,
        device=device,
        result_root=result_root,
        evaluation_id=evaluation_id,
    )
