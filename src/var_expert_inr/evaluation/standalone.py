from __future__ import annotations

import time
import tempfile
from contextlib import contextmanager, nullcontext
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from .ground_truth import portable_data_path, validate_ground_truth_paths
from .metrics import QualityAccumulator, mae, mse, psnr, summarize_selected_quality
from .performance import DecodeMeasurement
from .rendering import (
    VolumeRenderSession,
    compare_rendered_images,
    load_render_profile,
    preflight_rendering,
    profile_fingerprint,
    render_node_frame,
)
from .reporting import (
    cache_key,
    environment_manifest,
    evaluation_token,
    find_cached_evaluation,
    path_fingerprint,
    write_json,
    write_metrics_csv,
)
from .selection import metrics_require_ground_truth, metrics_require_rendering, parse_timestep_selection


def identify_subsystem(raw: dict[str, Any]) -> str | None:
    if "MODEL" in raw:
        name = str((raw.get("MODEL") or {}).get("model_name", "")).lower()
        if "apmgsrn" in name:
            return "apmgsrn"
        if "neural" in name or "inr_moe" in name:
            return "neural_expert"
    name = str((raw.get("model") or {}).get("name", "")).lower().replace("-", "_")
    return name if name in {"mc_inr", "fv_srn", "rmdsrn", "ecnr", "miner"} else None


def _resolve_path(value: str | Path, *, repo_root: Path, config_path: Path) -> Path:
    text = str(value).replace("${REPO_ROOT}", str(repo_root))
    path = Path(text)
    return (config_path.parent / path).resolve() if not path.is_absolute() else path.resolve()


def _data_section(raw: dict[str, Any]) -> dict[str, Any]:
    return dict(raw.get("data") or raw.get("DATA") or {})


def _target_paths(raw: dict[str, Any], *, repo_root: Path, config_path: Path) -> dict[str, Path]:
    data = _data_section(raw)
    dataset_name = data.get("dataset_name")
    if data.get("target_path"):
        resolved = _resolve_path(data["target_path"], repo_root=repo_root, config_path=config_path)
        return {str(data.get("target") or "target"): portable_data_path(resolved, dataset_name=dataset_name, repo_root=repo_root)}
    result = {
        str(name): portable_data_path(
            _resolve_path(path, repo_root=repo_root, config_path=config_path),
            dataset_name=dataset_name,
            repo_root=repo_root,
        )
        for name, path in (data.get("targets") or {}).items()
    }
    selected = data.get("target")
    return {str(selected): result[str(selected)]} if selected and str(selected) in result else result


def _find_source(run_dir: Path, subsystem: str, requested: str, checkpoint, artifact, prediction):
    if prediction:
        return "prediction", Path(prediction).resolve()
    if artifact:
        return "artifact", Path(artifact).resolve()
    if checkpoint:
        return "checkpoint", Path(checkpoint).resolve()
    predictions = sorted((run_dir / "predictions").glob("*.npy")) if (run_dir / "predictions").exists() else []
    artifacts = sorted((run_dir / "artifacts").glob("*")) if (run_dir / "artifacts").exists() else []
    checkpoints = sorted((run_dir / "checkpoints").glob("*.pth")) if (run_dir / "checkpoints").exists() else []
    if subsystem in {"apmgsrn", "miner"}:
        timestep_checkpoints = sorted((run_dir / "timesteps").glob("t*/checkpoint.pth"))
        if requested in {"auto", "checkpoint"} and timestep_checkpoints:
            return "checkpoint", run_dir / "timesteps"
    if requested in {"auto", "artifact"} and artifacts:
        return "artifact", artifacts[-1]
    if requested == "artifact":
        raise FileNotFoundError(f"No artifact found under {run_dir / 'artifacts'}")
    if requested in {"auto", "checkpoint"} and checkpoints:
        finals = [path for path in checkpoints if "epoch" not in path.stem and "iter" not in path.stem]
        return "checkpoint", (finals[-1] if finals else checkpoints[-1])
    if requested == "checkpoint":
        raise FileNotFoundError(f"No checkpoint found under {run_dir / 'checkpoints'}")
    if predictions:
        return "prediction", predictions[0]
    if subsystem == "neural_expert":
        validate = run_dir / "validate_artifacts"
        candidates = sorted(validate.glob("*.pth")) if validate.exists() else []
        if candidates:
            return "checkpoint", candidates[-1]
    raise FileNotFoundError(f"No evaluation source found for {subsystem} under {run_dir}")


def _load_torch_payload(path: Path, device: torch.device) -> dict[str, Any]:
    try:
        payload = torch.load(path, map_location=device, weights_only=False)
    except TypeError:  # PyTorch < 2.6
        payload = torch.load(path, map_location=device)
    if not isinstance(payload, dict):
        raise ValueError(f"Checkpoint payload must be a mapping: {path}")
    return payload


def _decode_miner_frames(
    source_root: Path,
    *,
    timesteps: tuple[int, ...],
    targets: tuple[str, ...],
    shape_tzyx: tuple[int, int, int, int],
    device: torch.device,
) -> dict[tuple[str, int], np.ndarray]:
    from ..miner.runner import decode_checkpoint

    if len(targets) != 1:
        raise ValueError("MINER checkpoints contain one scalar target per run")
    decoded: dict[tuple[str, int], np.ndarray] = {}
    for timestep in timesteps:
        if source_root.is_file():
            checkpoint_path = source_root
        else:
            checkpoint_path = source_root / f"t{int(timestep):04d}" / "checkpoint.pth"
        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                f"Missing MINER checkpoint for timestep {timestep}: {checkpoint_path}"
            )
        frame = decode_checkpoint(checkpoint_path, device=device)
        if shape_tzyx[1] == 1 and frame.ndim == 2:
            frame = frame[None, ...]
        expected = tuple(int(value) for value in shape_tzyx[1:])
        if tuple(frame.shape) != expected:
            raise ValueError(
                f"MINER decoded shape mismatch: expected {expected}, got {tuple(frame.shape)}"
            )
        decoded[(targets[0], int(timestep))] = frame
    return decoded


def _normalized_axis(values: np.ndarray, size: int) -> np.ndarray:
    if int(size) <= 1:
        return np.zeros_like(values, dtype=np.float32)
    return (values.astype(np.float32) * (2.0 / float(int(size) - 1)) - 1.0).astype(np.float32)


def _decode_apmgsrn_frames(
    source_root: Path,
    raw: dict[str, Any],
    *,
    timesteps: tuple[int, ...],
    targets: tuple[str, ...],
    shape_tzyx: tuple[int, int, int, int],
    device: torch.device,
) -> dict[tuple[str, int], np.ndarray]:
    from ..apmgsrn.model import APMGSRN

    if len(targets) != 1:
        raise ValueError("APMGSRN checkpoints contain one target per run")
    training = dict(raw.get("TRAINING") or raw.get("training") or {})
    batch_size = max(1, int(training.get("prediction_points_per_batch", training.get("pred_batch_size", 16_000))))
    _, z_size, y_size, x_size = shape_tzyx
    spatial_size = int(z_size * y_size * x_size)
    decoded: dict[tuple[str, int], np.ndarray] = {}
    for timestep in timesteps:
        checkpoint_path = source_root if source_root.is_file() else source_root / f"t{int(timestep):03d}" / "checkpoint.pth"
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Missing APMGSRN checkpoint for timestep {timestep}: {checkpoint_path}")
        payload = _load_torch_payload(checkpoint_path, device)
        if int(payload.get("time_index", timestep)) != int(timestep):
            raise ValueError(f"APMGSRN checkpoint time_index mismatch: {checkpoint_path}")
        checkpoint_target = str(payload.get("target_name", targets[0]))
        if checkpoint_target != targets[0]:
            raise ValueError(
                f"APMGSRN checkpoint target mismatch: expected {targets[0]!r}, got {checkpoint_target!r}"
            )
        model_cfg = dict(payload.get("model_config") or raw.get("MODEL") or {})
        state = payload["model_state"]
        uses_tcnn_state = any(str(key).startswith("decoder.params") for key in state)
        model = APMGSRN(
            model_cfg,
            data_min=float(payload["data_min"]),
            data_max=float(payload["data_max"]),
            use_tcnn=uses_tcnn_state,
        ).to(device)
        model.load_state_dict(state, strict=True)
        model.eval()
        flat_output = np.empty((spatial_size, int(model.n_outputs)), dtype=np.float32)
        with torch.inference_mode():
            for start in range(0, spatial_size, batch_size):
                stop = min(spatial_size, start + batch_size)
                rows = np.arange(start, stop, dtype=np.int64)
                x = rows % x_size
                remaining = rows // x_size
                y = remaining % y_size
                z = remaining // y_size
                coords = np.stack(
                    [_normalized_axis(x, x_size), _normalized_axis(y, y_size), _normalized_axis(z, z_size)],
                    axis=1,
                )
                batch = torch.from_numpy(coords).to(device, non_blocking=True)
                flat_output[start:stop] = model(batch).detach().cpu().numpy().astype(np.float32, copy=False)
        frame = flat_output.reshape(z_size, y_size, x_size, int(model.n_outputs))
        decoded[(targets[0], int(timestep))] = frame[..., 0] if int(model.n_outputs) == 1 else frame
        del model
    return decoded


def _neural_expert_prediction(output: dict[str, torch.Tensor]) -> torch.Tensor:
    selected = output.get("selected_nonmanifold_pnts_pred")
    if selected is not None:
        return selected
    values = output.get("nonmanifold_pnts_pred")
    if values is None:
        raise ValueError("NeuralExpert model output is missing prediction tensors")
    return values.permute(0, 2, 1)


def _decode_neural_expert_frames(
    source_path: Path,
    raw: dict[str, Any],
    *,
    timesteps: tuple[int, ...],
    targets: tuple[str, ...],
    indexers: list[slice],
    shape_tzyx: tuple[int, int, int, int] | None,
    coords: np.ndarray | None,
    repo_root: Path,
    config_path: Path,
    device: torch.device,
) -> dict[tuple[str, int], np.ndarray]:
    if len(targets) != 1:
        raise ValueError("NeuralExpert checkpoints contain one target per run")
    data = dict(raw.get("DATA") or {})
    dataset_name = str(data.get("dataset_name", "")).strip().lower()
    model_name = str((raw.get("MODEL") or {}).get("model_name", ""))
    if dataset_name in {"ionization", "combustion_40nh3_1"}:
        if model_name == "inr_moe_ionization":
            from ..neural_expert.ionization.inr_moe import INR_MoE as ModelClass
        elif model_name == "inr_ionization":
            from ..neural_expert.ionization.inr import INR as ModelClass
        else:
            raise ValueError(f"Unsupported NeuralExpert volume model: {model_name!r}")
    else:
        if model_name == "inr_moe_mesh":
            from ..neural_expert.mesh.inr_moe import INR_MoE as ModelClass
        elif model_name == "inr_mesh":
            from ..neural_expert.mesh.inr import INR as ModelClass
        else:
            raise ValueError(f"Unsupported NeuralExpert mesh model: {model_name!r}")
    model = ModelClass(raw)
    payload = _load_torch_payload(source_path, device)
    state = payload.get("model_state", payload)
    model.load_state_dict(state, strict=True)
    model = model.to(device).eval()
    x_mean = np.asarray(payload.get("x_mean", 0.0), dtype=np.float32).reshape(1, -1)
    x_std = np.maximum(np.asarray(payload.get("x_std", 1.0), dtype=np.float32).reshape(1, -1), 1.0e-12)
    y_mean = np.asarray(payload.get("y_mean", 0.0), dtype=np.float32).reshape(1, -1)
    y_std = np.maximum(np.asarray(payload.get("y_std", 1.0), dtype=np.float32).reshape(1, -1), 1.0e-12)
    normalize_inputs = bool(data.get("normalize_inputs", dataset_name == "ionization"))
    normalize_targets = bool(data.get("normalize_targets", False))
    training = dict(raw.get("TRAINING") or {})
    batch_size = max(1, int(training.get("prediction_points_per_batch", training.get("n_points", 16_000))))
    decoded: dict[tuple[str, int], np.ndarray] = {}
    with torch.inference_mode():
        for timestep in timesteps:
            indexer = indexers[timestep]
            count = int(indexer.stop - indexer.start)
            parts: list[np.ndarray] = []
            for local_start in range(0, count, batch_size):
                local_stop = min(count, local_start + batch_size)
                if shape_tzyx is not None:
                    _, z_size, y_size, x_size = shape_tzyx
                    rows = np.arange(local_start, local_stop, dtype=np.int64)
                    x = rows % x_size
                    remaining = rows // x_size
                    y = remaining % y_size
                    z = remaining // y_size
                    raw_coords = np.stack(
                        [x.astype(np.float32), y.astype(np.float32), z.astype(np.float32), np.full(rows.shape, timestep, dtype=np.float32)],
                        axis=1,
                    )
                else:
                    assert coords is not None
                    raw_coords = np.asarray(coords[indexer.start + local_start:indexer.start + local_stop], dtype=np.float32)
                model_coords = (raw_coords - x_mean) / x_std if normalize_inputs else raw_coords
                output = model(torch.from_numpy(model_coords).to(device).unsqueeze(0))
                values = _neural_expert_prediction(output).squeeze(0).detach().cpu().numpy().astype(np.float32, copy=False)
                if normalize_targets:
                    values = values * y_std + y_mean
                parts.append(values)
            flat = np.concatenate(parts, axis=0)
            if shape_tzyx is not None:
                _, z_size, y_size, x_size = shape_tzyx
                flat = flat.reshape(z_size, y_size, x_size, flat.shape[-1])
                if flat.shape[-1] == 1:
                    flat = flat[..., 0]
            decoded[(targets[0], int(timestep))] = flat
    return decoded


def _invoke_predict(subsystem: str, config_path: Path, source_kind: str, source_path: Path, target: str | None):
    if subsystem == "mc_inr":
        from ..mc_inr.runner import run_predict
        return run_predict(config_path, checkpoint_path=source_path if source_kind == "checkpoint" else None)
    if subsystem == "fv_srn":
        from ..fv_srn.runner import run_predict
        return run_predict(config_path, target=target, artifact=source_path if source_kind == "artifact" else None, checkpoint=source_path if source_kind == "checkpoint" else None)
    if subsystem == "rmdsrn":
        from ..rmdsrn.runner import run_predict
        return run_predict(config_path, target=target, artifact=source_path if source_kind == "artifact" else None, checkpoint=source_path if source_kind == "checkpoint" else None)
    if subsystem == "ecnr":
        from ..ecnr.runner import run_predict
        return run_predict(config_path, target=target, artifact=source_path if source_kind == "artifact" else None, checkpoint=source_path if source_kind == "checkpoint" else None)
    raise RuntimeError(
        f"{subsystem} has no standalone decode API; evaluate it from an exported prediction array"
    )


def _prediction_paths(result: dict[str, Any], source_path: Path, targets: tuple[str, ...]) -> dict[str, Path]:
    if result.get("prediction_paths"):
        return {str(name): Path(path) for name, path in result["prediction_paths"].items() if str(name) in targets}
    path = result.get("mean_prediction_path") or result.get("prediction_path") or source_path
    if len(targets) != 1:
        raise ValueError("A single prediction file cannot represent multiple selected targets")
    return {targets[0]: Path(path)}


@contextmanager
def _portable_standalone_config(
    raw: dict[str, Any],
    *,
    gt_paths: dict[str, Path],
    coords_path: Path | None,
):
    payload = deepcopy(raw)
    key = "DATA" if "DATA" in payload else "data"
    data = dict(payload.get(key) or {})
    if data.get("target_path"):
        selected = str(data.get("target") or next(iter(gt_paths), "target"))
        if selected in gt_paths:
            data["target_path"] = str(gt_paths[selected])
    if data.get("targets"):
        data["targets"] = {
            name: str(gt_paths.get(str(name), Path(path)))
            for name, path in data["targets"].items()
        }
    if coords_path is not None:
        if data.get("coords_path") is not None:
            data["coords_path"] = str(coords_path)
        if data.get("source_path") is not None:
            data["source_path"] = str(coords_path)
    payload[key] = data
    with tempfile.TemporaryDirectory(prefix="var_expert_eval_") as temp_dir:
        path = Path(temp_dir) / "config.yaml"
        path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
        yield path


def run_standalone_evaluation(request, raw: dict[str, Any], subsystem: str, config_path: Path) -> dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[3]
    data = _data_section(raw)
    gt_paths_all = _target_paths(raw, repo_root=repo_root, config_path=config_path)
    available = tuple(gt_paths_all) or (str(data.get("target") or "target"),)
    targets = request.targets or available
    unknown = set(targets).difference(available)
    if unknown:
        raise KeyError(f"Unknown targets: {sorted(unknown)}. Available: {list(available)}")
    volume_cfg = data.get("volume_shape")
    is_volume = str(data.get("kind", "volume" if volume_cfg else "node")).lower() == "volume"
    if is_volume:
        if not volume_cfg:
            raise ValueError("Standalone volume config must provide data.volume_shape")
        shape_tzyx = (int(volume_cfg["T"]), int(volume_cfg["Z"]), int(volume_cfg["Y"]), int(volume_cfg["X"]))
        total_timesteps = shape_tzyx[0]
        coords = None
        coords_path = None
        indexers = [slice(t * int(np.prod(shape_tzyx[1:])), (t + 1) * int(np.prod(shape_tzyx[1:]))) for t in range(total_timesteps)]
    else:
        coords_value = data.get("coords_path") or data.get("source_path")
        coords_path = portable_data_path(
            _resolve_path(coords_value, repo_root=repo_root, config_path=config_path),
            dataset_name=data.get("dataset_name"),
            repo_root=repo_root,
        )
        coords = np.load(coords_path, mmap_mode="r", allow_pickle=False)
        times = coords[:, -1]
        boundaries: list[int] = []
        previous = times[0]
        for start in range(1, len(times), 1_000_000):
            block = np.asarray(times[start:min(len(times), start + 1_000_000)])
            if block[0] != previous:
                boundaries.append(start)
            boundaries.extend(int(item) for item in (np.flatnonzero(block[1:] != block[:-1]) + start + 1))
            previous = block[-1]
        starts, stops = [0, *boundaries], [*boundaries, len(times)]
        indexers = [slice(int(a), int(b)) for a, b in zip(starts, stops)]
        total_timesteps = len(indexers)
        shape_tzyx = None
    timesteps = parse_timestep_selection(request.timesteps, total_timesteps)
    needs_gt = metrics_require_ground_truth(request.metrics)
    ground_truth_available = all(name in gt_paths_all and gt_paths_all[name].is_file() for name in targets)
    if needs_gt:
        validate_ground_truth_paths(
            gt_paths_all, tuple(targets), volume_shape=shape_tzyx,
            node_count=None if is_volume else int(coords.shape[0]),
        )
    needs_render = bool(request.render or metrics_require_rendering(request.metrics))
    if not needs_gt and needs_render and ground_truth_available:
        try:
            validate_ground_truth_paths(
                gt_paths_all, tuple(targets), volume_shape=shape_tzyx,
                node_count=None if is_volume else int(coords.shape[0]),
            )
        except (OSError, ValueError):
            ground_truth_available = False
    profile = load_render_profile(data.get("dataset_name"), request.render_profile, repo_root=repo_root) if needs_render else None
    if profile is not None:
        preflight_rendering(
            profile,
            dataset_kind="volume" if is_volume else "node",
            targets=tuple(targets),
            timesteps=timesteps,
            frame_sizes={timestep: int(indexers[timestep].stop - indexers[timestep].start) for timestep in timesteps},
            prediction_only=not ground_truth_available,
            metrics=tuple(request.metrics),
        )
    source_kind, source_path = _find_source(
        request.run_dir, subsystem, request.source, request.checkpoint,
        request.artifact, request.prediction,
    )
    directory_checkpoint_subsystems = {"apmgsrn", "miner"}
    if source_kind == "checkpoint" and subsystem not in directory_checkpoint_subsystems and not source_path.is_file():
        raise FileNotFoundError(f"Evaluation checkpoint does not exist: {source_path}")
    if source_kind == "checkpoint" and subsystem in directory_checkpoint_subsystems and not source_path.exists():
        raise FileNotFoundError(f"Evaluation {subsystem} checkpoint source does not exist: {source_path}")
    if source_kind in {"artifact", "prediction"} and not source_path.exists():
        raise FileNotFoundError(f"Evaluation {source_kind} does not exist: {source_path}")
    if source_kind == "artifact" and subsystem in {"apmgsrn", "miner", "neural_expert"}:
        raise ValueError(f"{subsystem} does not define a compact artifact format; use checkpoint or prediction")
    key_payload = {
        "schema_version": 1,
        "config": path_fingerprint(config_path),
        "subsystem": subsystem,
        "source_kind": source_kind,
        "source": path_fingerprint(source_path),
        "metrics": list(request.metrics),
        "timesteps": list(timesteps),
        "targets": list(targets),
        "render": bool(needs_render),
        "render_profile": profile_fingerprint(profile) if profile is not None else None,
        "ground_truth": {
            name: path_fingerprint(gt_paths_all[name]) for name in targets
        } if ground_truth_available and (needs_gt or needs_render) else None,
    }
    evaluation_cache_key = cache_key(key_payload)
    cache_allowed = not {"decode_time", "memory"}.intersection(request.metrics)
    if cache_allowed and not request.overwrite:
        cached = find_cached_evaluation(request.run_dir, evaluation_cache_key)
        if cached is not None:
            return cached
    device_text = request.device or str((raw.get("training") or raw.get("TRAINING") or {}).get("device", "cuda"))
    device = torch.device(device_text if not device_text.startswith("cuda") or torch.cuda.is_available() else "cpu")
    measurement = DecodeMeasurement(device=device)
    load_seconds = reconstruction_seconds = 0.0
    decoded_frames: dict[tuple[str, int], np.ndarray] | None = None
    if source_kind == "prediction":
        prediction_result: dict[str, Any] = {}
        prediction_paths = _prediction_paths(prediction_result, source_path, tuple(targets))
    elif subsystem in {"apmgsrn", "miner", "neural_expert"}:
        with measurement:
            started = time.perf_counter()
            if subsystem == "apmgsrn":
                assert shape_tzyx is not None
                decoded_frames = _decode_apmgsrn_frames(
                    source_path, raw, timesteps=timesteps, targets=tuple(targets),
                    shape_tzyx=shape_tzyx, device=device,
                )
            elif subsystem == "miner":
                assert shape_tzyx is not None
                decoded_frames = _decode_miner_frames(
                    source_path,
                    timesteps=timesteps,
                    targets=tuple(targets),
                    shape_tzyx=shape_tzyx,
                    device=device,
                )
            else:
                decoded_frames = _decode_neural_expert_frames(
                    source_path, raw, timesteps=timesteps, targets=tuple(targets),
                    indexers=indexers, shape_tzyx=shape_tzyx, coords=coords,
                    repo_root=repo_root, config_path=config_path, device=device,
                )
            reconstruction_seconds = float(time.perf_counter() - started)
        prediction_paths = {}
    else:
        with measurement:
            started = time.perf_counter()
            # Standalone runners combine loading and reconstruction; record the
            # observable total under reconstruction and retain zero load split.
            with _portable_standalone_config(
                raw, gt_paths=gt_paths_all, coords_path=coords_path
            ) as decode_config_path:
                prediction_result = _invoke_predict(
                    subsystem, decode_config_path, source_kind, source_path, data.get("target")
                )
            reconstruction_seconds = float(time.perf_counter() - started)
        prediction_paths = _prediction_paths(prediction_result, source_path, tuple(targets))
    if source_kind == "prediction" and ({"decode_time", "memory"}.intersection(request.metrics)):
        with measurement:
            started = time.perf_counter()
            arrays = {name: np.load(path, mmap_mode="r", allow_pickle=False) for name, path in prediction_paths.items()}
            for array in arrays.values():
                for timestep in timesteps:
                    _ = np.asarray(array[timestep] if is_volume and array.ndim >= 4 else array[indexers[timestep]])
            reconstruction_seconds = float(time.perf_counter() - started)
    elif decoded_frames is None:
        arrays = {name: np.load(path, mmap_mode="r", allow_pickle=False) for name, path in prediction_paths.items()}
    else:
        arrays = {}
    gt_arrays = {
        name: np.load(gt_paths_all[name], mmap_mode="r", allow_pickle=False)
        for name in targets if ground_truth_available and (needs_gt or needs_render)
    }
    output_dir = request.run_dir / "evaluations" / evaluation_token()
    output_dir.mkdir(parents=True, exist_ok=False)
    rows, accumulators = [], {name: QualityAccumulator() for name in targets}
    context = VolumeRenderSession(profile) if needs_render and is_volume else nullcontext()
    with context as volume_renderer:
        for name in targets if (needs_gt or needs_render) else ():
            for timestep in timesteps:
                if decoded_frames is not None:
                    pred = np.asarray(decoded_frames[(name, timestep)])
                else:
                    pred = np.asarray(arrays[name][timestep] if is_volume and arrays[name].ndim >= 4 else arrays[name][indexers[timestep]])
                gt = None
                if ground_truth_available:
                    gt = np.asarray(gt_arrays[name][timestep] if is_volume and gt_arrays[name].ndim >= 4 else gt_arrays[name][indexers[timestep]])
                    if pred.size == gt.size:
                        pred = pred.reshape(gt.shape)
                row: dict[str, Any] = {"row_type": "per_timestep", "target": name, "timestep": timestep, "status": "ok"}
                if "psnr" in request.metrics:
                    accumulators[name].update(gt, pred)
                    row.update({"mse": mse(gt, pred), "mae": mae(gt, pred), "psnr": psnr(gt, pred)})
                if needs_render:
                    render_dir = output_dir / "renders" / name
                    pred_path, gt_path = render_dir / f"pred_t{timestep:04d}.png", render_dir / f"gt_t{timestep:04d}.png"
                    if is_volume:
                        pred_info = volume_renderer.render(pred, pred_path, target=name)
                        if gt is not None:
                            gt_info = volume_renderer.render(gt, gt_path, target=name)
                    else:
                        pred_info = render_node_frame(pred, pred_path, profile=profile, time_index=timestep, gt_values=gt, target=name)
                        if gt is not None:
                            gt_info = render_node_frame(gt, gt_path, profile=profile, time_index=timestep, gt_values=gt, target=name)
                    row["pred_render_path"] = str(pred_path.resolve())
                    row["render_info"] = pred_info
                    if gt is not None:
                        row["gt_render_path"] = str(gt_path.resolve())
                        row["gt_render_info"] = gt_info
                    if metrics_require_rendering(request.metrics):
                        row.update(compare_rendered_images(gt_path, pred_path, request.metrics, device=str(device)))
                rows.append(row)
                partial_targets, partial_aggregate = summarize_selected_quality(
                    rows, accumulators, tuple(targets), tuple(request.metrics)
                )
                write_json(
                    output_dir / "metrics.json",
                    {
                        "schema_version": 1,
                        "status": "running",
                        "targets": partial_targets,
                        "aggregate": partial_aggregate,
                        "performance": {},
                        "per_timestep": rows,
                    },
                )
                write_metrics_csv(output_dir / "metrics.csv", rows)
                write_json(output_dir / "progress.json", {"status": "running", "completed_rows": len(rows), "per_timestep": rows})
    per_target, aggregate = summarize_selected_quality(
        rows, accumulators, tuple(targets), tuple(request.metrics)
    )
    performance: dict[str, Any] = {}
    if "decode_time" in request.metrics:
        selected_values = sum(
            int(np.asarray(decoded_frames[(name, timestep)]).size)
            if decoded_frames is not None else
            int(np.asarray(arrays[name][timestep] if is_volume and arrays[name].ndim >= 4 else arrays[name][indexers[timestep]]).size)
            for name in targets for timestep in timesteps
        )
        total_decode = load_seconds + reconstruction_seconds
        performance.update({
            "load_seconds": load_seconds,
            "reconstruction_seconds": reconstruction_seconds,
            "total_decode_seconds": total_decode,
            "values_per_second": float(selected_values / reconstruction_seconds) if reconstruction_seconds > 0 else None,
            "decode_selection_mode": "selected" if subsystem in {"apmgsrn", "miner", "neural_expert"} or source_kind == "prediction" else "full_required",
        })
    if "memory" in request.metrics:
        performance.update(measurement.as_dict())
    manifest = {
        "schema_version": 1, "run_dir": str(request.run_dir), "config_path": str(config_path),
        "subsystem": subsystem, "dataset_kind": "volume" if is_volume else "node",
        "dataset_name": data.get("dataset_name"), "ground_truth_available": ground_truth_available,
        "ground_truth_required": needs_gt, "metrics": list(request.metrics), "timesteps": list(timesteps),
        "ground_truth": {
            name: path_fingerprint(gt_paths_all[name])
            for name in targets if name in gt_paths_all and gt_paths_all[name].is_file()
        },
        "targets": list(targets), "source_kind": source_kind, "source_path": str(source_path), "device": str(device),
        "render_requested": bool(needs_render),
        "render_profile": None if profile is None else {
            "path": profile.get("_path"),
            "fingerprint": profile_fingerprint(profile),
        },
        "environment": environment_manifest(),
        "cache_key": evaluation_cache_key,
    }
    payload = {"schema_version": 1, "status": "complete", "targets": per_target, "aggregate": aggregate, "performance": performance, "per_timestep": rows}
    manifest_path = write_json(output_dir / "manifest.json", manifest)
    metrics_path = write_json(output_dir / "metrics.json", payload)
    csv_path = write_metrics_csv(output_dir / "metrics.csv", rows or [{"row_type": "performance", **performance}])
    write_json(output_dir / "progress.json", {"status": "complete", "completed_rows": len(rows)})
    log_path = output_dir / "logs" / "evaluate.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(
        f"run_dir={request.run_dir}\nsubsystem={subsystem}\nsource={source_kind}:{source_path}\nstatus=complete\n",
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
