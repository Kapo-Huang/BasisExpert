from __future__ import annotations

import logging
import time
from contextlib import nullcontext
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from ..config.io import load_evaluation_experiment_config
from ..data.base import DatasetMeta, FieldBatch, FieldDataset, normalize_index_coordinates
from ..models import build_model
from ..training.engine import _predict_batch
from ..utils.checkpoint import (
    read_checkpoint_payload,
    validate_checkpoint_target_layout,
)
from .data_paths import normalize_experiment_data_paths, normalize_raw_data_paths
from .ground_truth import target_paths_from_config, validate_ground_truth_paths
from .artifacts import ArtifactStore, LAYOUT_SCHEMA_VERSION
from .metrics import (
    ErrorAccumulator,
    QualityAccumulator,
    error_frame_statistics,
    error_percentage,
    mae,
    mse,
    pointwise_absolute_error,
    psnr,
    summarize_selected_quality,
    validate_error_bounds,
)
from .performance import DecodeMeasurement, combine_memory_samples, synchronize_cuda
from .rendering import (
    VolumeRenderSession,
    compare_rendered_images,
    load_render_profile,
    preflight_rendering,
    profile_fingerprint,
    render_error_image_frame,
    render_error_node_frame,
    render_image_frame,
    render_node_frame,
    renderer_name,
)
from .reporting import (
    cache_key,
    environment_manifest,
    evaluation_output_dir,
    find_cached_evaluation,
    write_json,
    write_metrics_csv,
)
from .selection import (
    metrics_require_ground_truth,
    metrics_require_rendering,
    parse_metric_selection,
    parse_name_selection,
    parse_timestep_selection,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EvaluationRequest:
    run_dir: Path
    metrics: tuple[str, ...] = ("psnr",)
    timesteps: str = "all"
    targets: tuple[str, ...] | None = None
    source: str = "auto"
    checkpoint: Path | None = None
    prediction: Path | None = None
    render: bool = False
    render_profile: Path | str | None = None
    overwrite: bool = False
    device: str | None = None
    result_root: Path | str = "EvalResult"
    evaluation_id: str = "default"
    error_vmin: float = 0.0
    error_vmax: float = 5.0
    training_probe_samples: int = 72_000_000
    training_total_samples: int = 14_400_000_000
    inference_fraction: float = 0.1

    def __post_init__(self) -> None:
        error_vmin, error_vmax = validate_error_bounds(
            self.error_vmin,
            self.error_vmax,
        )
        object.__setattr__(self, "error_vmin", error_vmin)
        object.__setattr__(self, "error_vmax", error_vmax)

        if int(self.training_probe_samples) <= 0:
            raise ValueError("training_probe_samples must be positive")
        if int(self.training_total_samples) <= 0:
            raise ValueError("training_total_samples must be positive")
        if not 0.0 < float(self.inference_fraction) <= 1.0:
            raise ValueError("inference_fraction must be in (0, 1]")

class _InferenceOnlyDataset(FieldDataset):
    """Coordinate dataset used when performance/render tasks have no GT."""

    def __init__(self, config, targets: tuple[str, ...]) -> None:
        self._target_names = tuple(targets)
        self.volume_shape = config.data.volume_shape
        if self.volume_shape is not None:
            n_samples = int(self.volume_shape.N)
            self._coordinate_axes = tuple(config.data.coordinate_axes or ("x", "y", "z", "t"))
            input_dim = len(self._coordinate_axes)
            self._coords_np = None
        else:
            self._coordinate_axes = None
            coords_path = Path(config.data.coords_path).expanduser().resolve()
            if not coords_path.is_file():
                raise FileNotFoundError(f"Coordinate file is required for GT-free node evaluation: {coords_path}")
            self._coords_np = np.load(coords_path, mmap_mode="r", allow_pickle=False)
            n_samples = int(self._coords_np.shape[0])
            input_dim = int(self._coords_np.shape[1])
        configured_out = int(config.model.params.get("out_features", 1))
        configured_paths = dict(config.data.targets or {})
        if config.data.target_path:
            configured_paths[str(config.data.target or targets[0])] = config.data.target_path
        target_dims: dict[str, int] = {}
        for name in targets:
            path_value = configured_paths.get(name)
            if path_value and Path(path_value).is_file():
                array = np.load(path_value, mmap_mode="r", allow_pickle=False)
                if int(array.size) % n_samples != 0:
                    raise ValueError(
                        f"Target {name!r} contains {array.size} values for {n_samples} samples"
                    )
                target_dims[name] = int(array.size) // n_samples
            else:
                target_dims[name] = configured_out if len(targets) == 1 else 1
        self.meta = DatasetMeta(
            kind=config.data.kind, n_samples=n_samples, input_dim=input_dim,
            target_names=self._target_names, target_dims=target_dims,
            volume_shape=self.volume_shape,
        )

    def fetch_batch(self, indices, *, include_targets=True, assignments=None) -> FieldBatch:
        rows = np.asarray(list(indices), dtype=np.int64)
        if self.volume_shape is not None:
            remaining = rows.copy()
            x = remaining % int(self.volume_shape.X)
            remaining //= int(self.volume_shape.X)
            y = remaining % int(self.volume_shape.Y)
            remaining //= int(self.volume_shape.Y)
            z = remaining % int(self.volume_shape.Z)
            remaining //= int(self.volume_shape.Z)
            t = remaining
            normalized = {
                "x": normalize_index_coordinates(x, self.volume_shape.X),
                "y": normalize_index_coordinates(y, self.volume_shape.Y),
                "z": normalize_index_coordinates(z, self.volume_shape.Z),
                "t": normalize_index_coordinates(t, self.volume_shape.T),
            }
            coords = np.stack(
                [normalized[axis] for axis in self._coordinate_axes], axis=1
            ).astype(np.float32)
        else:
            coords = np.asarray(self._coords_np[rows], dtype=np.float32)
        return FieldBatch(indices=torch.from_numpy(rows), coords=torch.from_numpy(coords), targets=None)

    def load_targets_flat(self):
        raise RuntimeError("Ground Truth is unavailable for this evaluation")

    def align_target_order(self, target_names: tuple[str, ...]) -> None:
        self.align_target_layout(
            target_names,
            tuple(self.meta.target_dims[name] for name in target_names),
        )

    def align_target_layout(
        self,
        target_names: tuple[str, ...],
        target_dims: tuple[int, ...],
    ) -> None:
        requested = tuple(str(name) for name in target_names)
        current = tuple(self._target_names)
        if set(requested) != set(current):
            raise ValueError(
                f"Checkpoint targets do not match run config: checkpoint={list(requested)} config={list(current)}"
            )
        if len(target_dims) != len(requested):
            raise ValueError(
                "Checkpoint target dimension count does not match target names: "
                f"names={len(requested)} dims={len(target_dims)}"
            )
        resolved_dims = {
            name: int(dimension)
            for name, dimension in zip(requested, target_dims)
        }
        if any(dimension <= 0 for dimension in resolved_dims.values()):
            raise ValueError(
                f"Checkpoint target dimensions must be positive: {resolved_dims}"
            )
        self._target_names = requested
        self.meta = replace(
            self.meta,
            target_names=requested,
            target_dims=resolved_dims,
        )

    def reshape_flat_predictions(self, name, flat_values):
        values = np.asarray(flat_values)
        if self.volume_shape is None:
            return values
        dims = int(self.meta.target_dims[name])
        shape = (self.volume_shape.T, self.volume_shape.Z, self.volume_shape.Y, self.volume_shape.X)
        return values.reshape(shape) if dims == 1 else values.reshape((*shape, dims))


def resolve_run_config(run_dir: str | Path) -> Path:
    root = Path(run_dir).expanduser().resolve()
    candidates = (
        root / "configs" / "config.yaml",
        root / "config.yaml",
    )
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(f"No run config found under {root}")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _select_targets(available: tuple[str, ...], selected: tuple[str, ...] | None) -> tuple[str, ...]:
    if selected is None:
        return available
    lookup = {name.lower(): name for name in available}
    result: list[str] = []
    missing: list[str] = []
    for requested in selected:
        actual = lookup.get(requested.lower())
        if actual is None:
            missing.append(requested)
        elif actual not in result:
            result.append(actual)
    if missing:
        raise KeyError(f"Unknown targets: {missing}. Available: {list(available)}")
    return tuple(result)


def _node_time_indexers(coords: np.ndarray) -> list[slice | np.ndarray]:
    if coords.ndim != 2 or coords.shape[1] < 1:
        raise ValueError(f"Node coordinates must have shape (N, D), got {coords.shape}")
    times = coords[:, -1]
    size = int(times.shape[0])
    if size == 0:
        return []
    boundaries: list[int] = []
    block_size = 1_000_000
    previous = times[0]
    for start in range(1, size, block_size):
        stop = min(size, start + block_size)
        block = np.asarray(times[start:stop])
        if block.size == 0:
            continue
        if block[0] != previous:
            boundaries.append(start)
        local = np.flatnonzero(block[1:] != block[:-1]) + start + 1
        boundaries.extend(int(item) for item in local)
        previous = block[-1]
    starts = [0, *boundaries]
    stops = [*boundaries, size]
    contiguous = [slice(int(start), int(stop)) for start, stop in zip(starts, stops)]
    labels = [times[indexer.start] for indexer in contiguous]
    if len(set(float(item) for item in labels)) != len(labels):
        raise ValueError("Node timesteps must be stored in contiguous coordinate blocks")
    return contiguous


def _frame_indexers(dataset) -> list[slice | np.ndarray]:
    shape = dataset.meta.volume_shape
    if shape is not None:
        per_time = int(shape.X) * int(shape.Y) * int(shape.Z)
        return [slice(t * per_time, (t + 1) * per_time) for t in range(int(shape.T))]
    return _node_time_indexers(dataset._coords_np)


def _indices(indexer: slice | np.ndarray) -> np.ndarray:
    if isinstance(indexer, slice):
        return np.arange(int(indexer.start), int(indexer.stop), dtype=np.int64)
    return np.asarray(indexer, dtype=np.int64)


def _resolve_standard_source(request: EvaluationRequest, config) -> tuple[str, Path]:
    if request.checkpoint is not None:
        return "checkpoint", request.checkpoint.resolve()
    if request.prediction is not None:
        return "prediction", request.prediction.resolve()
    requested = str(request.source).lower()
    checkpoint_dir = request.run_dir / "checkpoints"
    prediction_dir = request.run_dir / "predictions"
    checkpoints = sorted(checkpoint_dir.glob("*.pth")) if checkpoint_dir.exists() else []
    canonical = checkpoint_dir / f"{config.exp_id}.pth"
    predictions = sorted(prediction_dir.glob("*.npy")) if prediction_dir.exists() else []
    if requested in {"auto", "checkpoint"} and canonical.is_file():
        return "checkpoint", canonical
    if requested in {"auto", "checkpoint"} and checkpoints:
        finals = [path for path in checkpoints if "_epoch" not in path.stem]
        return "checkpoint", (finals[-1] if finals else checkpoints[-1])
    if requested == "checkpoint":
        raise FileNotFoundError(f"No checkpoint found under {checkpoint_dir}")
    if requested in {"auto", "prediction"} and predictions:
        return "prediction", predictions[0]
    raise FileNotFoundError(f"No usable evaluation source found in {request.run_dir}")


def _load_prediction_arrays(
    prediction_dir_or_file: Path,
    *,
    exp_id: str,
    targets: tuple[str, ...],
) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    if prediction_dir_or_file.is_file() and len(targets) == 1:
        return {targets[0]: np.load(prediction_dir_or_file, mmap_mode="r", allow_pickle=False)}
    root = prediction_dir_or_file if prediction_dir_or_file.is_dir() else prediction_dir_or_file.parent
    for target in targets:
        candidates = [root / f"{exp_id}_{target}.npy", root / f"{exp_id}.npy"]
        path = next((item for item in candidates if item.is_file()), None)
        if path is None:
            raise FileNotFoundError(f"Prediction file is missing for target {target!r} under {root}")
        result[target] = np.load(path, mmap_mode="r", allow_pickle=False)
    return result


def _load_standard_model(config, dataset, device: torch.device, source: Path):
    payload = read_checkpoint_payload(source)
    checkpoint_format = payload.get("format")
    if checkpoint_format not in {None, "inference_checkpoint_v1"}:
        raise ValueError(f"Unsupported inference checkpoint: {checkpoint_format!r}")
    if "model_state" not in payload:
        raise ValueError(f"Checkpoint does not contain model_state: {source}")
    if payload.get("target_names_order"):
        target_names = tuple(payload["target_names_order"])
        target_dims = payload.get("target_dims_order")
        if target_dims is None:
            dataset.align_target_order(target_names)
        else:
            dataset.align_target_layout(target_names, tuple(target_dims))
    validate_checkpoint_target_layout(
        payload,
        dataset.target_names(),
        dataset.meta.target_dims,
    )
    model = build_model(config.model, dataset.meta).to(device)
    model.load_state_dict(payload["model_state"])
    return model


def _predict_selected_frames(
    model,
    dataset,
    *,
    frame_indexers: list[slice | np.ndarray],
    timesteps: tuple[int, ...],
    targets: tuple[str, ...],
    batch_size: int,
    device: torch.device,
) -> dict[tuple[str, int], np.ndarray]:
    collected: dict[tuple[str, int], np.ndarray] = {}
    model.eval()
    with torch.inference_mode():
        for timestep in timesteps:
            rows = _indices(frame_indexers[timestep])
            parts = {name: [] for name in targets}
            for start in range(0, int(rows.size), int(batch_size)):
                batch_rows = rows[start : start + int(batch_size)]
                batch = dataset.fetch_batch(batch_rows, include_targets=False)
                coords = batch.coords.to(device, non_blocking=True)
                predictions = _predict_batch(
                    model,
                    coords,
                    dataset.target_names(),
                    hard_topk=True,
                    target_dims=dataset.meta.target_dims,
                )
                for name in targets:
                    parts[name].append(predictions[name].detach().cpu().numpy())
            for name in targets:
                collected[(name, timestep)] = np.concatenate(parts[name], axis=0)
    return collected


def _prediction_frame(array: np.ndarray, dataset, target: str, timestep: int, indexer) -> np.ndarray:
    shaped = np.asarray(array)
    shape = dataset.meta.volume_shape
    if shape is not None and shaped.ndim >= 4 and int(shaped.shape[0]) == int(shape.T):
        return np.asarray(shaped[timestep])
    flat = shaped.reshape(-1, shaped.shape[-1]) if shaped.ndim == 2 else shaped.reshape(-1, 1)
    return np.asarray(flat[indexer])


def _ground_truth_frame(raw: np.ndarray, dataset, timestep: int, indexer) -> np.ndarray:
    shape = dataset.meta.volume_shape
    if shape is not None and raw.ndim >= 4:
        return np.asarray(raw[timestep])
    return np.asarray(raw[indexer])


def _reshape_render_frame(dataset, values: np.ndarray) -> np.ndarray:
    shape = dataset.meta.volume_shape
    if shape is None:
        return np.asarray(values)
    dims = values.shape[-1] if values.ndim == 2 else 1
    spatial = (int(shape.Z), int(shape.Y), int(shape.X))
    return values.reshape((*spatial, dims))[..., 0] if dims == 1 else values.reshape((*spatial, dims))


def run_standard_evaluation(request: EvaluationRequest) -> dict[str, Any]:
    request = EvaluationRequest(**{**request.__dict__, "run_dir": request.run_dir.resolve()})
    repo_root = _repo_root()
    artifacts = ArtifactStore(repo_root=repo_root, result_root=request.result_root)
    artifacts.initialize()
    metrics = () if request.metrics == () else parse_metric_selection(request.metrics)
    needs_gt = metrics_require_ground_truth(metrics)
    needs_render = bool(request.render or metrics_require_rendering(metrics))
    config_path = resolve_run_config(request.run_dir)
    config = normalize_experiment_data_paths(
        load_evaluation_experiment_config(config_path),
        repo_root=repo_root,
    )
    available_targets = tuple(
        [config.data.target] if config.data.target else
        list(config.data.targets.keys()) if config.data.targets else ["target"]
    )
    targets = _select_targets(available_targets, request.targets)
    gt_paths = target_paths_from_config(config.data, repo_root=repo_root)
    ground_truth_available = all(name in gt_paths and gt_paths[name].is_file() for name in targets)
    volume_shape = None
    node_count = None
    if config.data.volume_shape is not None:
        volume_shape = (
            int(config.data.volume_shape.T), int(config.data.volume_shape.Z),
            int(config.data.volume_shape.Y), int(config.data.volume_shape.X),
        )
        total_timesteps = int(config.data.volume_shape.T)
    else:
        coords_path = Path(str(config.data.coords_path))
        coords = np.load(coords_path, mmap_mode="r", allow_pickle=False)
        node_count = int(coords.shape[0])
        total_timesteps = len(_node_time_indexers(coords))
    timesteps = parse_timestep_selection(request.timesteps, total_timesteps)
    if needs_gt:
        validate_ground_truth_paths(
            gt_paths, targets, volume_shape=volume_shape, node_count=node_count
        )
    elif needs_render and ground_truth_available:
        try:
            validate_ground_truth_paths(
                gt_paths, targets, volume_shape=volume_shape, node_count=node_count
            )
        except (OSError, ValueError):
            ground_truth_available = False

    # Evaluation uses coordinate-only metadata and memory-mapped GT frames. This
    # avoids validating or materializing an entire multi-gigabyte target array.
    dataset = _InferenceOnlyDataset(config, available_targets)
    gt_arrays = {
        name: np.load(gt_paths[name], mmap_mode="r", allow_pickle=False)
        for name in targets if ground_truth_available and (needs_gt or needs_render)
    }
    frame_indexers = _frame_indexers(dataset)
    profile = None
    selected_renderer = None
    if needs_render:
        profile = load_render_profile(
            config.data.dataset_name, request.render_profile or config.evaluation.render_profile,
            repo_root=repo_root,
        )
        if str(profile.get("kind", config.data.kind)).lower() != config.data.kind:
            raise ValueError("Render profile kind does not match dataset kind")
        selected_renderer = renderer_name(profile, dataset_kind=config.data.kind)
        frame_coordinates = None
        if config.data.kind == "node":
            frame_coordinates = {
                timestep: np.asarray(dataset._coords_np[frame_indexers[timestep]])
                for timestep in timesteps
            }
        spatial_shape = None
        if config.data.volume_shape is not None:
            spatial_shape = (
                int(config.data.volume_shape.Z),
                int(config.data.volume_shape.Y),
                int(config.data.volume_shape.X),
            )
        preflight_rendering(
            profile,
            dataset_kind=config.data.kind,
            targets=targets,
            timesteps=timesteps,
            frame_sizes={
                timestep: int(frame_indexers[timestep].stop - frame_indexers[timestep].start)
                for timestep in timesteps
            },
            prediction_only=not ground_truth_available,
            metrics=metrics,
            frame_coordinates=frame_coordinates,
            spatial_shape=spatial_shape,
        )
    kind, source_path = _resolve_standard_source(request, config)
    if kind == "checkpoint" and not source_path.is_file():
        raise FileNotFoundError(f"Evaluation checkpoint does not exist: {source_path}")
    if kind == "prediction" and not source_path.exists():
        raise FileNotFoundError(f"Evaluation prediction source does not exist: {source_path}")
    ground_truth_fingerprints = {
        name: artifacts.content_fingerprint(gt_paths[name]) for name in targets
    } if ground_truth_available and (needs_gt or needs_render) else {}
    source_fingerprint = artifacts.content_fingerprint(source_path)
    key_payload = {
        "schema_version": LAYOUT_SCHEMA_VERSION,
        "config": artifacts.content_fingerprint(config_path),
        "source_kind": kind,
        "source": source_fingerprint,
        "metrics": list(metrics),
        "timesteps": list(timesteps),
        "targets": list(targets),
        "render": bool(needs_render),
        "render_profile": profile_fingerprint(profile) if profile is not None else None,
        "ground_truth": ground_truth_fingerprints or None,
        "error_analysis": {
            "version": 1,
            "gt_range": [-1.0, 1.0],
            "percentage_formula": "abs_error / 2 * 100",
            "error_vmin": request.error_vmin,
            "error_vmax": request.error_vmax,
        } if "error" in metrics else None,
    }
    evaluation_cache_key = cache_key(key_payload)
    output_dir = evaluation_output_dir(
        request.run_dir,
        repo_root=repo_root,
        result_root=request.result_root,
        evaluation_id=request.evaluation_id,
    )
    render_fingerprint = profile_fingerprint(profile) if profile is not None else None
    cache_allowed = not {"decode_time", "memory"}.intersection(metrics)
    if cache_allowed and not request.overwrite:
        cached = find_cached_evaluation(output_dir, evaluation_cache_key)
        if cached is not None:
            return cached
    # Evaluation runs independently from the device used during training.
    # A caller may select a device explicitly; otherwise use the current CUDA
    # device instead of inheriting a potentially stale archived training device.
    device = torch.device(request.device or "cuda")
    if device.type == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")
    output_dir.mkdir(parents=True, exist_ok=True)

    load_seconds = reconstruction_seconds = 0.0
    memory_samples: list[dict[str, Any]] = []
    load_measurement = DecodeMeasurement(device=device)
    with load_measurement:
        started = time.perf_counter()
        if kind == "prediction":
            source_arrays = _load_prediction_arrays(source_path, exp_id=config.exp_id, targets=targets)
            model = None
        else:
            source_arrays = None
            model = _load_standard_model(config, dataset, device, source_path)
        synchronize_cuda(device)
        load_seconds = float(time.perf_counter() - started)
    memory_samples.append(load_measurement.as_dict())

    rows: list[dict[str, Any]] = []
    selected_values = 0
    target_accumulators = {name: QualityAccumulator() for name in targets}
    error_accumulators = {name: ErrorAccumulator() for name in targets}
    volume_context = VolumeRenderSession(profile) if needs_render and selected_renderer == "volume" else nullcontext()
    with volume_context as volume_renderer:
        for timestep in timesteps:
            frame_measurement = DecodeMeasurement(device=device)
            with frame_measurement:
                started = time.perf_counter()
                if source_arrays is None:
                    decoded_frame = _predict_selected_frames(
                        model, dataset, frame_indexers=frame_indexers, timesteps=(timestep,),
                        targets=targets, batch_size=int(config.evaluation.batch_size), device=device,
                    )
                else:
                    decoded_frame = {
                        (name, timestep): _prediction_frame(
                            source_arrays[name], dataset, name, timestep, frame_indexers[timestep]
                        )
                        for name in targets
                    }
                synchronize_cuda(device)
                reconstruction_seconds += float(time.perf_counter() - started)
            memory_samples.append(frame_measurement.as_dict())
            selected_values += sum(int(np.asarray(value).size) for value in decoded_frame.values())
            if not needs_gt and not needs_render:
                del decoded_frame
                continue
            for target in targets:
                pred = np.asarray(decoded_frame[(target, timestep)])
                gt = _ground_truth_frame(
                    gt_arrays[target], dataset, timestep, frame_indexers[timestep]
                ) if target in gt_arrays else None
                pred_for_metrics = pred.reshape(gt.shape) if gt is not None and pred.size == gt.size else pred
                row: dict[str, Any] = {
                    "row_type": "per_timestep", "target": target, "timestep": timestep,
                    "status": "ok", "prediction_source": str(source_path),
                }
                if "psnr" in metrics:
                    target_accumulators[target].update(gt, pred_for_metrics)
                    row.update({"mse": mse(gt, pred_for_metrics), "mae": mae(gt, pred_for_metrics), "psnr": psnr(gt, pred_for_metrics)})
                abs_error = None
                if "error" in metrics:
                    abs_error = pointwise_absolute_error(gt, pred_for_metrics)
                    error_accumulators[target].update(abs_error)
                    row.update(error_frame_statistics(abs_error))
                if needs_render:
                    pred_render = _reshape_render_frame(dataset, pred)
                    gt_render = None if gt is None else _reshape_render_frame(dataset, gt)
                    frame_coords = (
                        np.asarray(dataset._coords_np[frame_indexers[timestep]])
                        if selected_renderer == "mesh" else None
                    )
                    gt_record = None
                    if gt_render is not None:
                        gt_spec = artifacts.ground_truth_spec(
                            dataset=config.data.dataset_name,
                            target=target,
                            timestep=timestep,
                            ground_truth_path=gt_paths[target],
                            profile=profile,
                        )

                        def produce_gt(path: Path) -> dict[str, Any]:
                            if selected_renderer == "volume":
                                return volume_renderer.render(gt_render, path, target=target)
                            if selected_renderer == "image2d":
                                return render_image_frame(
                                    gt_render, path, profile=profile, gt_values=gt_render, target=target,
                                )
                            return render_node_frame(
                                gt_render, path, profile=profile, time_index=timestep,
                                gt_values=gt_render, coordinates=frame_coords, target=target,
                            )

                        gt_record = artifacts.materialize(gt_spec, produce_gt)

                    pred_spec = artifacts.prediction_spec(
                        dataset=config.data.dataset_name,
                        model=config.model.name,
                        target=target,
                        timestep=timestep,
                        source_path=source_path,
                        profile=profile,
                        ground_truth_fingerprint=ground_truth_fingerprints.get(target),
                    )

                    def produce_pred(path: Path) -> dict[str, Any]:
                        if selected_renderer == "volume":
                            return volume_renderer.render(pred_render, path, target=target)
                        if selected_renderer == "image2d":
                            return render_image_frame(
                                pred_render, path, profile=profile, gt_values=gt_render, target=target,
                            )
                        return render_node_frame(
                            pred_render, path, profile=profile, time_index=timestep,
                            gt_values=gt_render, coordinates=frame_coords, target=target,
                        )

                    pred_record = artifacts.materialize(
                        pred_spec,
                        produce_pred,
                        overwrite=request.overwrite,
                    )
                    row["pred_render_path"] = artifacts.relative(pred_record.path)
                    row["render_info"] = artifacts.describe(pred_record)
                    if gt_record is not None:
                        row["gt_render_path"] = artifacts.relative(gt_record.path)
                        row["gt_render_info"] = artifacts.describe(gt_record)
                    if abs_error is not None:
                        error_spec = artifacts.error_spec(
                            dataset=config.data.dataset_name,
                            model=config.model.name,
                            target=target,
                            timestep=timestep,
                            source_path=source_path,
                            ground_truth_fingerprint=ground_truth_fingerprints[target],
                            profile=profile,
                            error_vmin=request.error_vmin,
                            error_vmax=request.error_vmax,
                        )

                        def produce_error(path: Path) -> dict[str, Any]:
                            error_render = _reshape_render_frame(
                                dataset,
                                error_percentage(abs_error),
                            )
                            if selected_renderer == "volume":
                                return volume_renderer.render_error(
                                    error_render,
                                    path,
                                    target=target,
                                    error_vmin=request.error_vmin,
                                    error_vmax=request.error_vmax,
                                )
                            if selected_renderer == "image2d":
                                return render_error_image_frame(
                                    error_render,
                                    path,
                                    profile=profile,
                                    error_vmin=request.error_vmin,
                                    error_vmax=request.error_vmax,
                                )
                            return render_error_node_frame(
                                error_render,
                                path,
                                profile=profile,
                                time_index=timestep,
                                coordinates=frame_coords,
                                error_vmin=request.error_vmin,
                                error_vmax=request.error_vmax,
                            )

                        error_record = artifacts.materialize(
                            error_spec,
                            produce_error,
                            overwrite=request.overwrite,
                        )
                        row["error_render_path"] = artifacts.relative(error_record.path)
                        row["error_render_info"] = artifacts.describe(error_record)
                    if metrics_require_rendering(metrics):
                        if gt_record is None:
                            raise RuntimeError("Rendered image metrics require a ground-truth artifact")
                        row.update(
                            compare_rendered_images(
                                gt_record.path, pred_record.path, metrics, device=str(device)
                            )
                        )
                rows.append(row)
                partial_targets, partial_aggregate = summarize_selected_quality(
                    rows,
                    target_accumulators,
                    targets,
                    metrics,
                    error_accumulators,
                )
                write_json(
                    output_dir / "metrics.json",
                    {
                        "schema_version": LAYOUT_SCHEMA_VERSION,
                        "status": "running",
                        "targets": partial_targets,
                        "aggregate": partial_aggregate,
                        "performance": {},
                        "per_timestep": rows,
                    },
                )
                write_metrics_csv(output_dir / "metrics.csv", rows)
                write_json(
                    output_dir / "progress.json",
                    {"status": "running", "completed_rows": len(rows), "per_timestep": rows},
                )
            del decoded_frame

    per_target, aggregate = summarize_selected_quality(
        rows,
        target_accumulators,
        targets,
        metrics,
        error_accumulators,
    )
    performance: dict[str, Any] = {}
    if "decode_time" in metrics:
        total_decode = float(load_seconds + reconstruction_seconds)
        performance.update({
            "load_seconds": load_seconds,
            "reconstruction_seconds": reconstruction_seconds,
            "total_decode_seconds": total_decode,
            "selected_values": int(selected_values),
            "total_values": int(dataset.meta.n_samples) * sum(int(dataset.meta.target_dims[name]) for name in targets),
            "values_per_second": float(selected_values / reconstruction_seconds) if reconstruction_seconds > 0 else None,
            "decode_selection_mode": "selected",
        })
    if "memory" in metrics:
        performance.update(combine_memory_samples(memory_samples))

    manifest = {
        "schema_version": LAYOUT_SCHEMA_VERSION,
        "evaluation_id": request.evaluation_id,
        "result_root": ".",
        "artifact_root": "artifacts",
        "run_dir": str(request.run_dir), "config_path": str(config_path),
        "model": config.model.name, "dataset_kind": config.data.kind,
        "dataset_name": config.data.dataset_name,
        "ground_truth_available": bool(ground_truth_available),
        "ground_truth_required": bool(needs_gt),
        "ground_truth": ground_truth_fingerprints,
        "metrics": list(metrics), "timesteps": list(timesteps), "targets": list(targets),
        "source_kind": kind, "source_path": str(source_path), "device": str(device),
        "source_fingerprint": source_fingerprint,
        "render_requested": bool(needs_render),
        "render_profile": None if profile is None else {
            "path": profile.get("_path"),
            "fingerprint": profile_fingerprint(profile),
            "renderer": selected_renderer,
        },
        "error_analysis": {
            "enabled": True,
            "version": 1,
            "space": "normalized",
            "vector_reduction": "pointwise_l2",
            "gt_range": [-1.0, 1.0],
            "percentage_formula": "abs_error / 2 * 100",
            "percentage_unit": "%",
            "error_vmin": request.error_vmin,
            "error_vmax": request.error_vmax,
            "pointwise_fields_saved": False,
            "pooled_percentiles": False,
        } if "error" in metrics else {"enabled": False},
        "environment": environment_manifest(),
        "cache_key": evaluation_cache_key,
    }
    payload = {
        "schema_version": LAYOUT_SCHEMA_VERSION, "status": "complete", "targets": per_target, "aggregate": aggregate,
        "performance": performance, "per_timestep": rows,
    }
    manifest_path = write_json(output_dir / "manifest.json", manifest)
    metrics_path = write_json(output_dir / "metrics.json", payload)
    csv_rows = rows or [{"row_type": "performance", **performance}]
    csv_path = write_metrics_csv(output_dir / "metrics.csv", csv_rows)
    write_json(output_dir / "progress.json", {"status": "complete", "completed_rows": len(rows)})
    log_path = output_dir / "logs" / "evaluate.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(
        "\n".join(
            [
                f"run_dir={request.run_dir}",
                f"source={kind}:{source_path}",
                f"metrics={','.join(metrics)}",
                f"timesteps={','.join(str(item) for item in timesteps)}",
                f"targets={','.join(targets)}",
                f"ground_truth_available={ground_truth_available}",
                "status=complete",
            ]
        ) + "\n",
        encoding="utf-8",
    )
    return {
        "output_dir": output_dir, "manifest_path": manifest_path,
        "metrics_path": metrics_path, "csv_path": csv_path, "log_path": log_path,
        "metrics": payload,
    }


def evaluate_run(
    run_dir: str | Path,
    *,
    metrics: str | tuple[str, ...] | None = None,
    timesteps: str | None = None,
    targets: str | tuple[str, ...] | None = None,
    source: str | None = None,
    checkpoint: str | Path | None = None,
    prediction: str | Path | None = None,
    render: bool = False,
    render_profile: str | Path | None = None,
    overwrite: bool = False,
    device: str | None = None,
    result_root: str | Path = "EvalResult",
    evaluation_id: str = "default",
    error_vmin: float | None = None,
    error_vmax: float | None = None,
    training_probe_samples: int = 72_000_000,
    training_total_samples: int = 14_400_000_000,
    inference_fraction: float = 0.1,
) -> dict[str, Any]:
    resolved_run = Path(run_dir).expanduser().resolve()
    config_path = resolve_run_config(resolved_run)
    raw = normalize_raw_data_paths(
        yaml.safe_load(config_path.read_text(encoding="utf-8")) or {},
        repo_root=_repo_root(),
        config_path=config_path,
    )
    configured_evaluation = dict(raw.get("evaluation") or raw.get("EVALUATION") or {})
    if metrics is not None:
        selected_metrics = metrics
    elif render:
        selected_metrics = ()
    else:
        selected_metrics = configured_evaluation.get("metrics", "psnr")
    selected_timesteps = timesteps if timesteps is not None else configured_evaluation.get("timesteps", "all")
    selected_targets = targets if targets is not None else configured_evaluation.get("targets", "all")
    selected_source = source if source is not None else configured_evaluation.get("source", "auto")
    selected_profile = render_profile if render_profile is not None else configured_evaluation.get("render_profile")
    selected_error_vmin = (
        error_vmin
        if error_vmin is not None
        else configured_evaluation.get("error_vmin", 0.0)
    )
    selected_error_vmax = (
        error_vmax
        if error_vmax is not None
        else configured_evaluation.get("error_vmax", 5.0)
    )
    parsed_metrics = () if selected_metrics == () else parse_metric_selection(selected_metrics)
    dependency_metrics = tuple(
        metric for metric in parsed_metrics if metric in {"pearson_error", "mi_error"}
    )
    if dependency_metrics:
        ordinary_metrics = tuple(metric for metric in parsed_metrics if metric not in dependency_metrics)
        if ordinary_metrics:
            raise ValueError(
                "pearson_error/mi_error are group-level metrics and cannot be mixed with "
                f"per-target metrics in one evaluate call: {ordinary_metrics}"
            )
        if render:
            raise ValueError("Dependency-only evaluation does not render images")
        if checkpoint is not None or prediction is not None:
            raise ValueError(
                "Explicit checkpoint/prediction paths are ambiguous for dependency groups; "
                "select sources through the run group"
            )
        from .dependency_service import evaluate_dependency_run

        return evaluate_dependency_run(
            resolved_run,
            metrics=dependency_metrics,
            timesteps=str(selected_timesteps or "all"),
            source=str(selected_source or "auto"),
            overwrite=bool(overwrite),
            device=device,
            result_root=result_root,
            evaluation_id=evaluation_id,
        )
    request = EvaluationRequest(
        run_dir=resolved_run,
        metrics=parsed_metrics,
        timesteps=str(selected_timesteps or "all"),
        targets=parse_name_selection(selected_targets),
        source=str(selected_source or "auto"),
        checkpoint=None if checkpoint is None else Path(checkpoint),
        prediction=None if prediction is None else Path(prediction),
        render=bool(render), render_profile=selected_profile, overwrite=bool(overwrite), device=device,
        result_root=result_root, evaluation_id=evaluation_id,
        error_vmin=float(selected_error_vmin), error_vmax=float(selected_error_vmax),
        training_probe_samples=int(training_probe_samples),
        training_total_samples=int(training_total_samples),
        inference_fraction=float(inference_fraction),
    )
    from .adapters import select_run_adapter

    adapter = select_run_adapter(raw)
    runtime_metrics = tuple(
        metric for metric in parsed_metrics
        if metric in {"training_time", "inference_time"}
    )
    ordinary_metrics = tuple(
        metric for metric in parsed_metrics
        if metric not in {"training_time", "inference_time"}
    )
    ordinary_result = None
    if ordinary_metrics or request.render:
        ordinary_result = adapter.evaluate(
            replace(request, metrics=ordinary_metrics), raw, config_path
        )
    if runtime_metrics:
        from .runtime import run_runtime_evaluation

        return run_runtime_evaluation(
            replace(request, metrics=runtime_metrics),
            raw,
            config_path,
            adapter_name=adapter.name,
            existing_result=ordinary_result,
        )
    if ordinary_result is None:
        raise ValueError("Evaluation selected no metrics or rendering work")
    return ordinary_result
