from __future__ import annotations

import copy
import json
import math
import os
import tempfile
import time
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
import yaml

from .artifacts import LAYOUT_SCHEMA_VERSION
from .reporting import (
    environment_manifest,
    evaluation_output_dir,
    write_json,
    write_metrics_csv,
)
from .selection import parse_timestep_selection
from .performance import synchronize_cuda


RUNTIME_SCHEMA_VERSION = 1
DEFAULT_TRAINING_PROBE_SAMPLES = 72_000_000
DEFAULT_TRAINING_TOTAL_SAMPLES = 14_400_000_000
DEFAULT_INFERENCE_FRACTION = 0.1


def uniform_fraction_timesteps(total: int, fraction: float) -> tuple[int, ...]:
    total = int(total)
    fraction = float(fraction)
    if total <= 0:
        raise ValueError("total timesteps must be positive")
    if not 0.0 < fraction <= 1.0:
        raise ValueError("inference fraction must be in (0, 1]")
    count = max(1, min(total, int(math.ceil(total * fraction))))
    if count == total:
        return tuple(range(total))
    if count == 1:
        return (int((total - 1) // 2),)
    return tuple(
        dict.fromkeys(
            int(round(index * (total - 1) / (count - 1)))
            for index in range(count)
        )
    )


def estimate_training_time(
    *,
    measured_samples: int,
    measured_seconds: float,
    total_samples: int,
) -> dict[str, Any]:
    measured_samples = int(measured_samples)
    measured_seconds = float(measured_seconds)
    total_samples = int(total_samples)
    if measured_samples <= 0 or measured_seconds <= 0.0 or total_samples <= 0:
        raise ValueError("training time estimation requires positive samples and seconds")
    rate = measured_samples / measured_seconds
    estimated = total_samples / rate
    return {
        "schema_version": RUNTIME_SCHEMA_VERSION,
        "probe_samples_actual": measured_samples,
        "probe_seconds": measured_seconds,
        "samples_per_second": rate,
        "seconds_per_sample": 1.0 / rate,
        "total_samples_assumed": total_samples,
        "estimated_training_seconds": estimated,
        "estimated_training_hours": estimated / 3600.0,
        "timing_scope": "training_loop_only",
    }


def estimate_inference_time(
    *,
    load_seconds: float,
    reconstruction_seconds: float,
    selected_values: int,
    total_values: int,
) -> dict[str, Any]:
    load_seconds = float(load_seconds)
    reconstruction_seconds = float(reconstruction_seconds)
    selected_values = int(selected_values)
    total_values = int(total_values)
    if load_seconds < 0.0 or reconstruction_seconds <= 0.0:
        raise ValueError("inference time estimation requires a positive reconstruction time")
    if selected_values <= 0 or total_values < selected_values:
        raise ValueError("invalid selected/total inference value counts")
    rate = selected_values / reconstruction_seconds
    projected_reconstruction = total_values / rate
    estimated_total = load_seconds + projected_reconstruction
    return {
        "schema_version": RUNTIME_SCHEMA_VERSION,
        "load_seconds": load_seconds,
        "reconstruction_seconds": reconstruction_seconds,
        "selected_values": selected_values,
        "total_values": total_values,
        "values_per_second": rate,
        "estimated_full_reconstruction_seconds": projected_reconstruction,
        "estimated_total_inference_seconds": estimated_total,
        "estimated_total_inference_hours": estimated_total / 3600.0,
        "timing_scope": "checkpoint_load_once_plus_reconstruction",
    }


def _section(raw: dict[str, Any], lower: str, upper: str) -> tuple[str, dict[str, Any]]:
    if isinstance(raw.get(lower), dict):
        return lower, dict(raw[lower])
    if isinstance(raw.get(upper), dict):
        return upper, dict(raw[upper])
    return lower, {}


def _total_timesteps(raw: dict[str, Any], config_path: Path) -> int:
    _, data = _section(raw, "data", "DATA")
    shape = data.get("volume_shape")
    if isinstance(shape, dict) and shape.get("T") is not None:
        return int(shape["T"])
    coords_value = data.get("coords_path") or data.get("source_path")
    if not coords_value:
        raise ValueError("Node runtime evaluation requires coords_path/source_path")
    from .ground_truth import portable_data_path
    from .standalone import _resolve_path
    from .service import _node_time_indexers, _repo_root

    coords_path = portable_data_path(
        _resolve_path(coords_value, repo_root=_repo_root(), config_path=config_path),
        dataset_name=data.get("dataset_name"),
        repo_root=_repo_root(),
    )
    coords = np.load(coords_path, mmap_mode="r", allow_pickle=False)
    return len(_node_time_indexers(coords))


def _portable_training_payload(
    raw: dict[str, Any],
    *,
    config_path: Path,
    scratch_root: Path,
    device: str,
) -> dict[str, Any]:
    from .ground_truth import portable_data_path
    from .standalone import _resolve_path
    from .service import _repo_root

    payload = copy.deepcopy(raw)
    data_key, data = _section(payload, "data", "DATA")
    dataset_name = data.get("dataset_name")
    for key in ("target_path", "coords_path", "source_path", "target_stats_path", "coordinate_stats_path"):
        if data.get(key):
            resolved = _resolve_path(
                data[key], repo_root=_repo_root(), config_path=config_path
            )
            data[key] = str(
                portable_data_path(
                    resolved, dataset_name=dataset_name, repo_root=_repo_root()
                )
            )
    if isinstance(data.get("targets"), dict):
        data["targets"] = {
            str(name): str(
                portable_data_path(
                    _resolve_path(path, repo_root=_repo_root(), config_path=config_path),
                    dataset_name=dataset_name,
                    repo_root=_repo_root(),
                )
            )
            for name, path in data["targets"].items()
        }
    payload[data_key] = data
    payload["experiment_root"] = str(scratch_root)
    payload["exp_id"] = "runtime-probe"

    training_key, training = _section(payload, "training", "TRAINING")
    training["device"] = device
    if "save_every" in training:
        training["save_every"] = 0
    if "log_every" in training:
        training["log_every"] = max(1, int(training["log_every"]))
    payload[training_key] = training
    if "exploration_probe" in payload:
        payload["exploration_probe"] = {"enabled": False}

    evaluation_key, evaluation = _section(payload, "evaluation", "EVALUATION")
    for key in ("save_predictions", "run_after_training"):
        if key in evaluation:
            evaluation[key] = False
    payload[evaluation_key] = evaluation
    return payload


def _representative_timesteps(value: Any, total: int) -> list[int]:
    if isinstance(value, list):
        available = [int(item) for item in value]
    elif isinstance(value, tuple):
        available = [int(item) for item in value]
    elif isinstance(value, int):
        available = [int(value)]
    elif isinstance(value, str) and value.strip().lower() != "all":
        available = list(parse_timestep_selection(value, total))
    else:
        available = list(range(total))
    if len(available) <= 3:
        return available
    positions = uniform_fraction_timesteps(len(available), 3.0 / len(available))
    return [available[position] for position in positions]


def _configure_training_probe(
    payload: dict[str, Any],
    *,
    adapter_name: str,
    probe_samples: int,
) -> dict[str, Any]:
    configured = copy.deepcopy(payload)
    _, model = _section(configured, "model", "MODEL")
    training_key, training = _section(configured, "training", "TRAINING")
    data_key, data = _section(configured, "data", "DATA")
    shape = data.get("volume_shape") or {}
    total_t = int(shape.get("T", 1))
    batch_size = max(
        1,
        int(
            training.get(
                "batch_size",
                training.get("n_points", training.get("points_per_iteration", 16_000)),
            )
        ),
    )

    if adapter_name == "unified":
        training["epochs"] = 1
        training["sampler"] = "budgeted_random"
        training["batches_per_epoch_budget"] = int(math.ceil(probe_samples / batch_size))
        training["early_stop_patience"] = 0
        if "val_split" in training:
            training["val_split"] = 0.0
        training["log_psnr_every"] = 0
        multiview_dwa = training.get("multiview_dwa_loss")
        if isinstance(multiview_dwa, dict):
            for legacy_key in ("eta_max", "eta_min", "window_size"):
                multiview_dwa.pop(legacy_key, None)
        if isinstance(training.get("pretrain"), dict):
            training["pretrain"]["enabled"] = False
            training["pretrain"]["epochs"] = 0
    elif adapter_name == "apmgsrn":
        selected = _representative_timesteps(training.get("time_indices", "all"), total_t)
        points = max(1, int(training.get("points_per_iteration", 100_000)))
        training["time_indices"] = selected
        training["iterations"] = max(1, int(math.ceil(probe_samples / (points * len(selected)))))
        training["early_stopping"] = False
    elif adapter_name == "fv_srn":
        epochs = 3
        validation = float(training.get("validation_fraction", 0.0))
        train_fraction = max(1.0 - validation, 1.0e-6)
        per_timestep = int(math.ceil(probe_samples / (epochs * total_t * train_fraction)))
        training["epochs"] = epochs
        training["samples_per_timestep"] = max(1, per_timestep)
        training["rebuild_every"] = 0
    elif adapter_name == "rmdsrn":
        training["steps"] = max(1, int(math.ceil(probe_samples / batch_size)))
        for key in ("lr_schedule_steps", "lambda_schedule_steps"):
            if key in training:
                training[key] = max(int(training[key]), int(training["steps"]))
    elif adapter_name == "ecnr":
        stage_count = 2 * max(1, int(model.get("scales", 1))) + 1
        per_stage = max(1, int(math.ceil(probe_samples / stage_count)))
        training["epochs_per_scale"] = 1
        training["quantization_finetune_epochs"] = 1
        training["sampling_mode"] = "budgeted_random"
        training["scalar_predictions_per_epoch_budget"] = per_stage
        training["pruning_epochs"] = []
        training["pruning_sparsities"] = []
        cnn = dict(configured.get("cnn") or {})
        cnn["epochs"] = 1
        cnn["sampling_mode"] = "budgeted_tiles"
        cnn["core_voxel_budget"] = per_stage
        configured["cnn"] = cnn
    elif adapter_name == "miner":
        selected = _representative_timesteps(training.get("time_indices", "all"), total_t)
        scales = max(1, int((configured.get("model") or {}).get("scales", 1)))
        spatial = max(
            1,
            int(shape.get("X", 1)) * int(shape.get("Y", 1)) * int(shape.get("Z", 1)),
        )
        training["time_indices"] = selected
        training["epochs_per_scale"] = max(
            1, int(math.ceil(probe_samples / (len(selected) * scales * spatial)))
        )
        training["global_mse_threshold"] = 0.0
        training["scale_convergence_delta"] = 0.0
    elif adapter_name == "mc_inr":
        meta_per_iteration = max(
            1,
            int(training.get("meta_batch_clusters", 1))
            * int(training.get("meta_inner_steps", 1))
            * int(training.get("meta_inner_batch_size", batch_size)),
        )
        meta_budget = max(meta_per_iteration, int(round(probe_samples * 0.1)))
        training["meta_iterations"] = max(1, int(math.ceil(meta_budget / meta_per_iteration)))
        remaining = max(batch_size, probe_samples - meta_budget)
        training["finetune_epochs"] = 1
        training["batches_per_epoch_budget"] = max(1, int(math.ceil(remaining / batch_size)))
        training["recluster_after_finetune"] = False
        training["max_recluster_rounds"] = 0
        training["save_intermediate_checkpoints"] = False
    elif adapter_name == "neural_expert":
        points = max(1, int(training.get("n_points", training.get("batch_size", 16_000))))
        training["num_epochs"] = max(1, int(math.ceil(probe_samples / points)))
        training["n_samples"] = int(training["num_epochs"])
    else:
        raise ValueError(f"Unsupported runtime training adapter: {adapter_name}")

    if "save_every" in training:
        training["save_every"] = 0
    if "log_every" in training:
        training["log_every"] = max(1, int(training["log_every"]))
    configured[training_key] = training
    configured[data_key] = data
    return configured


@contextmanager
def _runtime_environment() -> Iterator[None]:
    key = "VAR_EXPERT_RUNTIME_BENCHMARK"
    previous = os.environ.get(key)
    os.environ[key] = "1"
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = previous


@contextmanager
def _runtime_output_directory(path: Path) -> Iterator[None]:
    key = "VAR_EXPERT_EVALUATION_OUTPUT_DIR"
    previous = os.environ.get(key)
    path.mkdir(parents=True, exist_ok=True)
    os.environ[key] = str(path)
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = previous


def _invoke_training(config_path: Path, adapter_name: str, device: str) -> dict[str, Any]:
    if adapter_name == "unified":
        from ..cli import run_train
        return run_train(config_path)
    if adapter_name == "mc_inr":
        from ..methods.mc_inr.runner import run_train
        return run_train(config_path)
    if adapter_name == "apmgsrn":
        from ..methods.apmgsrn.runner import run_train
        return run_train(config_path)
    if adapter_name == "fv_srn":
        from ..methods.fv_srn.runner import run_train
        return run_train(config_path)
    if adapter_name == "rmdsrn":
        from ..methods.rmdsrn.runner import run_train
        return run_train(config_path)
    if adapter_name == "ecnr":
        from ..methods.ecnr.runner import run_train
        return run_train(config_path)
    if adapter_name == "miner":
        from ..methods.miner.runner import run_train
        return run_train(config_path)
    if adapter_name == "neural_expert":
        from ..methods.neural_expert.cli import run_train
        gpu = 0
        if str(device).startswith("cuda:"):
            gpu = int(str(device).split(":", 1)[1])
        return run_train(config_path, gpu=gpu)
    raise ValueError(f"Unsupported runtime training adapter: {adapter_name}")


def _json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _extract_training_measurement(
    result: dict[str, Any],
    *,
    adapter_name: str,
    requested_samples: int,
    wall_seconds: float,
) -> tuple[int, float, list[dict[str, Any]], str]:
    direct = result.get("runtime_training")
    if isinstance(direct, dict):
        return (
            int(direct["samples"]),
            float(direct["seconds"]),
            list(direct.get("strata") or []),
            "native_loop",
        )
    if adapter_name == "rmdsrn" and result.get("elapsed_seconds") is not None:
        steps = int(result.get("steps", 0))
        batch = int(result.get("batch_size", 0))
        samples = steps * batch or requested_samples
        return samples, float(result["elapsed_seconds"]), [], "native_loop"
    manifest_value = result.get("manifest_path")
    if adapter_name in {"apmgsrn", "miner"} and manifest_value:
        manifest = _json(manifest_value)
        strata = []
        for name, entry in (manifest.get("timesteps") or {}).items():
            if adapter_name == "apmgsrn":
                seconds = float(entry.get("training_loop_seconds", entry.get("elapsed_seconds", 0.0)))
                samples = int(entry.get("training_samples", 0))
            else:
                seconds = float(entry.get("training_loop_seconds", entry.get("elapsed_seconds", 0.0)))
                samples = int(entry.get("logical_samples", 0))
            if seconds > 0.0 and samples > 0:
                strata.append({"name": name, "samples": samples, "seconds": seconds})
        if strata:
            return (
                sum(int(item["samples"]) for item in strata),
                sum(float(item["seconds"]) for item in strata),
                strata,
                "native_loop",
            )
    if adapter_name == "ecnr" and result.get("training_cost_path"):
        cost = _json(result["training_cost_path"])
        strata = []
        for scale in cost.get("scales") or []:
            samples = int(scale.get("actual_scalar_predictions", 0)) + int(
                scale.get("quantization_finetune_actual_predictions", 0)
            )
            seconds = float(scale.get("primary_training_seconds", 0.0)) + float(
                scale.get("quantization_and_finetune_seconds", 0.0)
            )
            if samples > 0 and seconds > 0.0:
                strata.append({
                    "name": f"scale_{scale.get('level')}",
                    "samples": samples,
                    "seconds": seconds,
                })
        cnn = cost.get("cnn") or {}
        if int(cnn.get("core_voxel_visits", 0)) > 0 and float(cnn.get("seconds", 0.0)) > 0.0:
            strata.append({
                "name": "boundary_cnn",
                "samples": int(cnn["core_voxel_visits"]),
                "seconds": float(cnn["seconds"]),
            })
        if strata:
            return (
                sum(int(item["samples"]) for item in strata),
                sum(float(item["seconds"]) for item in strata),
                strata,
                "native_loop",
            )
    return int(requested_samples), float(wall_seconds), [], "process_wall_fallback"


def benchmark_training(
    request,
    raw: dict[str, Any],
    config_path: Path,
    *,
    adapter_name: str,
) -> dict[str, Any]:
    device_text = request.device or "cuda"
    if device_text.startswith("cuda") and not torch.cuda.is_available():
        device_text = "cpu"
    with tempfile.TemporaryDirectory(prefix="var-expert-runtime-train-") as temp_name:
        scratch = Path(temp_name)
        payload = _portable_training_payload(
            raw,
            config_path=config_path,
            scratch_root=scratch / "runs",
            device=device_text,
        )
        payload = _configure_training_probe(
            payload,
            adapter_name=adapter_name,
            probe_samples=int(request.training_probe_samples),
        )
        probe_config = scratch / "config.yaml"
        probe_config.write_text(
            yaml.safe_dump(payload, sort_keys=False),
            encoding="utf-8",
        )
        synchronize_cuda(device_text)
        started = time.perf_counter()
        with _runtime_environment():
            result = _invoke_training(probe_config, adapter_name, device_text)
        synchronize_cuda(device_text)
        wall_seconds = float(time.perf_counter() - started)
        samples, seconds, strata, timing_source = _extract_training_measurement(
            result,
            adapter_name=adapter_name,
            requested_samples=int(request.training_probe_samples),
            wall_seconds=wall_seconds,
        )
    payload = estimate_training_time(
        measured_samples=samples,
        measured_seconds=seconds,
        total_samples=int(request.training_total_samples),
    )
    payload.update({
        "probe_samples_requested": int(request.training_probe_samples),
        "probe_samples_overshoot": int(samples - int(request.training_probe_samples)),
        "adapter": adapter_name,
        "device": device_text,
        "timing_source": timing_source,
        "strata": strata,
    })
    return payload


def _checkpoint_load_seconds(path: Path) -> float:
    started = time.perf_counter()
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    elapsed = float(time.perf_counter() - started)
    del payload
    return elapsed


def benchmark_inference(
    request,
    raw: dict[str, Any],
    config_path: Path,
) -> dict[str, Any]:
    if request.prediction is not None or str(request.source).lower() == "prediction":
        raise ValueError("inference_time requires a checkpoint source")
    total_timesteps = _total_timesteps(raw, config_path)
    selected = uniform_fraction_timesteps(total_timesteps, request.inference_fraction)
    from .adapters import select_run_adapter

    with tempfile.TemporaryDirectory(prefix="var-expert-runtime-infer-") as temp_name:
        probe_request = replace(
            request,
            metrics=("decode_time",),
            timesteps=",".join(str(item) for item in selected),
            targets=None,
            source="checkpoint",
            prediction=None,
            render=False,
            overwrite=True,
            result_root=Path(temp_name) / "EvalResult",
            evaluation_id="runtime-probe",
        )
        with _runtime_output_directory(Path(temp_name) / "predictions"):
            probe = select_run_adapter(raw).evaluate(probe_request, raw, config_path)
        measured = dict(probe["metrics"].get("performance") or {})
        probe_manifest = _json(probe["manifest_path"])
        source_path = Path(str(probe_manifest.get("source_path", "")))

    load_seconds = float(measured.get("load_seconds", 0.0))
    reconstruction_seconds = float(measured["reconstruction_seconds"])
    load_timing_source = str(
        measured.get("load_timing_source") or "adapter_split"
    )
    separate_load_required = load_timing_source != "outer_container_ignored"
    if load_seconds <= 0.0 and source_path.is_file() and separate_load_required:
        separately_measured_load = _checkpoint_load_seconds(source_path)
        load_seconds = separately_measured_load
        reconstruction_seconds = max(
            reconstruction_seconds - separately_measured_load,
            float(np.finfo(np.float64).eps),
        )
        load_timing_source = "separate_checkpoint_load"

    decode_selection_mode = str(measured.get("decode_selection_mode") or "selected")
    selected_values = int(measured["selected_values"])
    total_values = int(measured["total_values"])
    if decode_selection_mode == "full_required":
        selected_values = total_values

    result = estimate_inference_time(
        load_seconds=load_seconds,
        reconstruction_seconds=reconstruction_seconds,
        selected_values=selected_values,
        total_values=total_values,
    )
    measured_timing_scope = measured.get("timing_scope")
    if measured_timing_scope:
        result["timing_scope"] = str(measured_timing_scope)
    result.update({
        "fraction_requested": float(request.inference_fraction),
        "fraction_measured": float(selected_values / max(total_values, 1)),
        "total_timesteps": total_timesteps,
        "selected_timesteps": list(selected),
        "selected_timestep_count": len(selected),
        "selection_mode": "uniform_timesteps",
        "decode_selection_mode": decode_selection_mode,
        "load_timing_source": load_timing_source,
        "device": str(request.device or ("cuda" if torch.cuda.is_available() else "cpu")),
    })
    return result


def _runtime_csv_rows(performance: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    training = performance.get("training_time")
    if isinstance(training, dict):
        rows.append({
            "row_type": "runtime_training",
            **{key: value for key, value in training.items() if key != "strata"},
        })
        for item in training.get("strata") or []:
            rows.append({"row_type": "runtime_training_stratum", **item})
    inference = performance.get("inference_time")
    if isinstance(inference, dict):
        rows.append({
            "row_type": "runtime_inference",
            **{
                key: value
                for key, value in inference.items()
                if key != "selected_timesteps"
            },
            "selected_timesteps": ",".join(
                str(item) for item in inference.get("selected_timesteps") or []
            ),
        })
    return rows


def _completed_runtime_metrics(payload: dict[str, Any]) -> set[str]:
    performance = payload.get("performance") or {}
    return {
        metric
        for metric in ("training_time", "inference_time")
        if isinstance(performance.get(metric), dict)
    }


def run_runtime_evaluation(
    request,
    raw: dict[str, Any],
    config_path: Path,
    *,
    adapter_name: str,
    existing_result: dict[str, Any] | None = None,
) -> dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[3]
    output_dir = evaluation_output_dir(
        request.run_dir,
        repo_root=repo_root,
        result_root=request.result_root,
        evaluation_id=request.evaluation_id,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.json"
    manifest_path = output_dir / "manifest.json"

    if existing_result is not None:
        payload = dict(existing_result["metrics"])
    elif metrics_path.is_file():
        payload = _json(metrics_path)
    else:
        payload = {
            "schema_version": LAYOUT_SCHEMA_VERSION,
            "status": "complete",
            "targets": {},
            "aggregate": {},
            "performance": {},
            "per_timestep": [],
        }
    performance = dict(payload.get("performance") or {})
    completed = _completed_runtime_metrics(payload)
    pending = [
        metric
        for metric in request.metrics
        if request.overwrite or metric not in completed
    ]
    if "training_time" in pending:
        performance["training_time"] = benchmark_training(
            request, raw, config_path, adapter_name=adapter_name
        )
    if "inference_time" in pending:
        performance["inference_time"] = benchmark_inference(
            request, raw, config_path
        )
    payload.update({
        "schema_version": LAYOUT_SCHEMA_VERSION,
        "status": "complete",
        "performance": performance,
    })

    previous_manifest = _json(manifest_path) if manifest_path.is_file() else {}
    existing_metrics = list(previous_manifest.get("metrics") or [])
    all_metrics = list(dict.fromkeys([*existing_metrics, *request.metrics]))
    manifest = {
        **previous_manifest,
        "schema_version": LAYOUT_SCHEMA_VERSION,
        "evaluation_id": request.evaluation_id,
        "result_root": ".",
        "artifact_root": "artifacts",
        "run_dir": str(request.run_dir),
        "config_path": str(config_path),
        "metrics": all_metrics,
        "runtime_benchmark": {
            "schema_version": RUNTIME_SCHEMA_VERSION,
            "training_probe_samples": int(request.training_probe_samples),
            "training_total_samples": int(request.training_total_samples),
            "inference_fraction": float(request.inference_fraction),
            "cache_policy": "experiment",
            "adapter": adapter_name,
        },
        "environment": environment_manifest(),
    }
    written_manifest = write_json(manifest_path, manifest)
    written_metrics = write_json(metrics_path, payload)
    csv_rows = list(payload.get("per_timestep") or []) + _runtime_csv_rows(performance)
    csv_path = write_metrics_csv(output_dir / "metrics.csv", csv_rows)
    write_json(
        output_dir / "progress.json",
        {"status": "complete", "completed_rows": len(csv_rows)},
    )
    log_path = output_dir / "logs" / "evaluate.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(
        "\n".join([
            f"run_dir={request.run_dir}",
            f"metrics={','.join(request.metrics)}",
            f"adapter={adapter_name}",
            f"cache_policy=experiment",
            f"reused={','.join(sorted(set(request.metrics) - set(pending)))}",
            "status=complete",
        ]) + "\n",
        encoding="utf-8",
    )
    return {
        "output_dir": output_dir,
        "manifest_path": written_manifest,
        "metrics_path": written_metrics,
        "csv_path": csv_path,
        "log_path": log_path,
        "metrics": payload,
        "cache_hit": not pending,
    }
