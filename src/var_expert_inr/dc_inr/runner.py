from __future__ import annotations

import json
import logging
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from numpy.lib.format import open_memmap

from ..evaluation.metrics import PSNRAccumulator, mae, mse, psnr, save_metrics
from ..utils.io import sha256_payload
from ..utils.logging_utils import close_file_handlers, setup_logging
from ..utils.runtime import apply_runtime_thread_limits, set_random_seed
from .checkpoint import load_dc_checkpoint, save_dc_checkpoint, validate_dc_checkpoint
from .config import DCExperimentConfig, config_payload, load_config, save_config
from .data import (
    DCTargetVolume,
    block_grid_shape_from_payload,
    block_id_to_grid_indices,
    block_shape_from_payload,
    full_block_query_coords,
    sample_block_training_batch,
    sample_balanced_block_training_batch,
)
from .model import DCINRTiny, dc_inr_parameter_count
from .search import CandidateSummary, select_best_candidate

logger = logging.getLogger(__name__)
TIMESTAMP_RUN_PATTERN = re.compile(r"^\d{8}_\d{6}_\d{6}$")


def _resolve_device(requested: str) -> torch.device:
    requested_norm = str(requested).strip().lower()
    if requested_norm.startswith("cuda") and not torch.cuda.is_available():
        logger.warning("CUDA requested but unavailable. Falling back to CPU.")
        return torch.device("cpu")
    return torch.device(requested)


def _experiment_dir(config: DCExperimentConfig) -> Path:
    return Path(config.experiment_root) / config.exp_id


def _is_timestamped_run_dir(path: Path) -> bool:
    return path.is_dir() and bool(TIMESTAMP_RUN_PATTERN.fullmatch(path.name))


def _build_run_dirs(run_dir: Path) -> dict[str, Path | str]:
    checkpoint_dir = run_dir / "checkpoints"
    config_dir = run_dir / "configs"
    prediction_dir = run_dir / "predictions"
    metrics_dir = run_dir / "metrics"
    logs_dir = run_dir / "logs"
    return {
        "experiment_dir": run_dir.parent,
        "run_dir": run_dir,
        "run_token": run_dir.name,
        "checkpoint_dir": checkpoint_dir,
        "config_dir": config_dir,
        "prediction_dir": prediction_dir,
        "metrics_dir": metrics_dir,
        "logs_dir": logs_dir,
    }


def _ensure_run_dirs(run_dir: Path) -> dict[str, Path | str]:
    dirs = _build_run_dirs(run_dir)
    for path in (
        dirs["run_dir"],
        dirs["checkpoint_dir"],
        dirs["config_dir"],
        dirs["prediction_dir"],
        dirs["metrics_dir"],
        dirs["logs_dir"],
    ):
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def _create_train_run_dirs(config: DCExperimentConfig):
    experiment_dir = _experiment_dir(config)
    experiment_dir.mkdir(parents=True, exist_ok=True)
    while True:
        run_token = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        run_dir = experiment_dir / run_token
        if not run_dir.exists():
            return _ensure_run_dirs(run_dir)
        time.sleep(0.001)


def _resolve_checkpoint_run_dir(checkpoint_path: str | Path) -> Path | None:
    resolved = Path(checkpoint_path).resolve()
    if resolved.parent.name != "checkpoints":
        return None
    run_dir = resolved.parent.parent
    if not _is_timestamped_run_dir(run_dir):
        raise FileNotFoundError(
            f"Checkpoint path must live under runs/<exp_id>/<timestamp>/checkpoints: {resolved}"
        )
    return run_dir


def _resolve_latest_run_dir(config: DCExperimentConfig) -> Path:
    experiment_dir = _experiment_dir(config)
    candidates: list[Path] = []
    if experiment_dir.exists():
        candidates = sorted(path for path in experiment_dir.iterdir() if _is_timestamped_run_dir(path))
    if not candidates:
        raise FileNotFoundError(
            f"No timestamped run directory found for exp_id '{config.exp_id}' under '{experiment_dir}'."
        )
    return candidates[-1]


def _resolve_existing_run_dirs(config: DCExperimentConfig, checkpoint_path: str | Path | None = None):
    run_dir = None
    if checkpoint_path is not None:
        run_dir = _resolve_checkpoint_run_dir(checkpoint_path)
    if run_dir is None:
        run_dir = _resolve_latest_run_dir(config)
    return _ensure_run_dirs(run_dir)


def _json_dump(path: str | Path, payload: dict[str, Any]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
    return target


def _sum_model_stats(widths: np.ndarray) -> dict[str, int | float]:
    param_count = int(sum(dc_inr_parameter_count(int(width)) for width in np.asarray(widths, dtype=np.int32).tolist()))
    fp16_bytes = int(param_count * 2)
    return {
        "param_count": param_count,
        "trainable_param_count": param_count,
        "fp16_size_bytes": fp16_bytes,
        "fp16_size_mb": float(fp16_bytes) / (1024.0 * 1024.0),
    }


def _train_representative_models(
    *,
    volume: DCTargetVolume,
    selection: CandidateSummary,
    config: DCExperimentConfig,
    device: torch.device,
) -> tuple[list[dict[str, torch.Tensor]], list[dict[str, float]]]:
    blocks = volume.block_view(selection.block_shape)
    widths = np.asarray(selection.widths, dtype=np.int32)
    model_states: list[dict[str, torch.Tensor]] = []
    summaries: list[dict[str, float]] = []
    for model_index, block_id in enumerate(np.asarray(selection.representative_block_ids, dtype=np.int32).tolist()):
        width = int(widths[model_index])
        model = DCINRTiny(width).to(device)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=float(config.training.lr),
            betas=(float(config.training.beta_1), float(config.training.beta_2)),
        )
        rng = np.random.default_rng(int(config.training.seed) + int(model_index))
        block_values = np.asarray(blocks[int(block_id)], dtype=np.float32)
        final_loss = float("nan")
        started_at = time.perf_counter()
        if int(config.training.total_steps) > 0:
            base_steps, extra_steps = divmod(
                int(config.training.total_steps),
                int(selection.representative_block_ids.size),
            )
            model_steps = base_steps + (1 if model_index < extra_steps else 0)
        else:
            model_steps = int(config.training.epochs)
        if int(config.training.total_steps) > 0:
            milestones = sorted(
                {
                    max(
                        1,
                        int(
                            round(
                                model_steps
                                * float(milestone)
                                / float(config.training.total_steps)
                            )
                        ),
                    )
                    for milestone in config.training.lr_milestones
                }
            )
        else:
            milestones = list(config.training.lr_milestones)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=milestones,
            gamma=float(config.training.lr_gamma),
        )
        for epoch in range(1, model_steps + 1):
            if int(config.training.total_steps) > 0:
                coords_np, targets_np = sample_balanced_block_training_batch(
                    block_values=block_values,
                    block_shape=selection.block_shape,
                    batch_size=int(config.training.batch_size),
                    rng=rng,
                )
            else:
                coords_np, targets_np = sample_block_training_batch(
                    block_values=block_values,
                    block_shape=selection.block_shape,
                    points_per_timestep=int(config.training.points_per_timestep),
                    rng=rng,
                )
            coords = torch.from_numpy(coords_np).to(device, non_blocking=True)
            targets = torch.from_numpy(targets_np).to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            predictions = model(coords)
            loss = F.mse_loss(predictions, targets)
            loss.backward()
            optimizer.step()
            scheduler.step()
            final_loss = float(loss.detach().item())
            if int(config.training.log_every) > 0 and (
                epoch == 1
                or epoch == model_steps
                or (epoch % int(config.training.log_every) == 0)
            ):
                logger.info(
                    "DC-INR rep %d/%d block_id=%d width=%d epoch %d/%d loss=%.6e lr=%.3e",
                    model_index + 1,
                    int(widths.size),
                    int(block_id),
                    width,
                    epoch,
                    model_steps,
                    final_loss,
                    float(optimizer.param_groups[0]["lr"]),
                )
        cpu_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
        model_states.append(cpu_state)
        summaries.append(
            {
                "representative_index": float(model_index),
                "block_id": float(block_id),
                "width": float(width),
                "steps": int(model_steps),
                "samples": int(
                    model_steps
                    * (
                        int(config.training.batch_size)
                        if int(config.training.total_steps) > 0
                        else int(config.training.points_per_timestep) * int(volume.volume_shape.T)
                    )
                ),
                "final_loss": float(final_loss),
                "elapsed_seconds": float(time.perf_counter() - started_at),
            }
        )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return model_states, summaries


def _predict_volume_from_payload(
    *,
    payload: dict[str, Any],
    device: torch.device,
    prediction_batch_size: int,
    prediction_path: str | Path,
) -> Path:
    block_shape = block_shape_from_payload(payload["block_shape"])
    grid_shape = block_grid_shape_from_payload(payload["block_grid_shape"])
    time_count = int(payload["volume_shape"]["T"])
    volume_shape_tzyx = (
        int(payload["volume_shape"]["T"]),
        int(payload["volume_shape"]["Z"]),
        int(payload["volume_shape"]["Y"]),
        int(payload["volume_shape"]["X"]),
    )
    assignments = np.asarray(payload["block_to_representative"], dtype=np.int32)
    widths = np.asarray(payload["model_widths"], dtype=np.int32)
    model_states = list(payload["model_states"])
    coords_cpu = torch.from_numpy(
        full_block_query_coords(block_shape=block_shape, time_count=time_count)
    )
    prediction = open_memmap(
        prediction_path,
        mode="w+",
        dtype=np.float32,
        shape=volume_shape_tzyx,
    )
    voxel_count = int(block_shape.voxel_count)
    for representative_index, width in enumerate(widths.tolist()):
        model = DCINRTiny(int(width)).to(device)
        model.load_state_dict(model_states[representative_index], strict=True)
        model.eval()
        outputs = np.empty((int(coords_cpu.shape[0]), 1), dtype=np.float32)
        with torch.no_grad():
            for start in range(0, int(coords_cpu.shape[0]), int(prediction_batch_size)):
                stop = min(start + int(prediction_batch_size), int(coords_cpu.shape[0]))
                batch_coords = coords_cpu[start:stop].to(device, non_blocking=True)
                outputs[start:stop] = model(batch_coords).detach().cpu().numpy().astype(np.float32, copy=False)
        block_prediction = outputs.reshape(time_count, voxel_count).reshape(
            time_count,
            int(block_shape.sz),
            int(block_shape.sy),
            int(block_shape.sx),
        )
        for block_id in np.flatnonzero(assignments == int(representative_index)).tolist():
            bx, by, bz = block_id_to_grid_indices(int(block_id), grid_shape)
            x0 = bx * int(block_shape.sx)
            y0 = by * int(block_shape.sy)
            z0 = bz * int(block_shape.sz)
            prediction[
                :,
                z0 : z0 + int(block_shape.sz),
                y0 : y0 + int(block_shape.sy),
                x0 : x0 + int(block_shape.sx),
            ] = block_prediction
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    prediction.flush()
    return Path(prediction_path)


def _evaluate_prediction(
    *,
    volume: DCTargetVolume,
    prediction_path: str | Path,
    payload_bytes: int,
    checkpoint_bytes: int,
) -> dict[str, Any]:
    prediction = np.load(str(prediction_path), mmap_mode="r")
    target = volume.array_tzyx()
    if prediction.shape != target.shape:
        raise ValueError(f"Prediction shape mismatch: expected {target.shape}, got {prediction.shape}")
    accumulator = PSNRAccumulator()
    total_abs_error = 0.0
    total_count = 0
    per_time: list[dict[str, Any]] = []
    for time_index in range(int(target.shape[0])):
        gt = np.asarray(target[time_index], dtype=np.float32)
        pred = np.asarray(prediction[time_index], dtype=np.float32)
        accumulator.update(gt, pred)
        diff = pred.astype(np.float64) - gt.astype(np.float64)
        total_abs_error += float(np.abs(diff).sum())
        total_count += int(diff.size)
        per_time.append(
            {
                "t": int(time_index),
                "mse": mse(gt, pred),
                "mae": mae(gt, pred),
                "psnr": psnr(gt, pred),
            }
        )
    total_mse = float(accumulator.total_squared_error) / max(total_count, 1)
    payload_cr = float(volume.raw_bytes) / float(max(int(payload_bytes), 1))
    checkpoint_cr = float(volume.raw_bytes) / float(max(int(checkpoint_bytes), 1))
    return {
        "targets": {
            volume.target_name: {
                "mse": total_mse,
                "mae": float(total_abs_error) / max(total_count, 1),
                "psnr": accumulator.compute(),
                "per_time": per_time,
            }
        },
        "aggregate": {
            "mse": total_mse,
            "mae": float(total_abs_error) / max(total_count, 1),
            "psnr": accumulator.compute(),
            "cr": payload_cr,
            "payload_cr": payload_cr,
            "checkpoint_cr": checkpoint_cr,
            "payload_bytes": int(payload_bytes),
            "checkpoint_bytes": int(checkpoint_bytes),
            "raw_target_bytes": int(volume.raw_bytes),
        },
    }


def _build_checkpoint_payload(
    *,
    config: DCExperimentConfig,
    config_hash: str,
    selection: CandidateSummary,
    model_states: list[dict[str, torch.Tensor]],
    search_summaries: list[dict[str, object]],
    model_stats: dict[str, int | float],
) -> dict[str, Any]:
    return {
        "model_name": "dc_inr",
        "config_hash": str(config_hash),
        "target_name": str(config.data.target),
        "target_path": str(config.data.target_path),
        "volume_shape": config.data.volume_shape.to_dict() if config.data.volume_shape is not None else None,
        "block_shape": selection.block_shape.to_dict(),
        "block_grid_shape": selection.grid_shape.to_dict(),
        "block_to_representative": np.asarray(selection.block_to_representative, dtype=np.int32),
        "representative_block_ids": np.asarray(selection.representative_block_ids, dtype=np.int32),
        "model_widths": np.asarray(selection.widths, dtype=np.int32),
        "model_states": list(model_states),
        "selected_M": int(selection.selected_M),
        "mean_capacity": float(selection.mean_capacity),
        "payload_bytes": int(selection.payload_bytes),
        "value_range": [-1.0, 1.0],
        "partition_search": list(search_summaries),
        "model_stats": dict(model_stats),
    }


def _prepare_volume(config: DCExperimentConfig) -> DCTargetVolume:
    if config.data.target_path is None or config.data.volume_shape is None or config.data.target is None:
        raise ValueError("DC-INR config did not resolve a single target volume")
    return DCTargetVolume(
        target_path=config.data.target_path,
        target_name=config.data.target,
        volume_shape=config.data.volume_shape,
    )


def run_train(config_path: str | Path, *, target: str | None = None) -> dict[str, Any]:
    apply_runtime_thread_limits()
    try:
        config = load_config(config_path, target_override=target)
        dirs = _create_train_run_dirs(config)
        setup_logging(log_dir=dirs["logs_dir"], log_file=f"run_{dirs['run_token']}.log")
        config_hash = sha256_payload(config_payload(config))
        set_random_seed(int(config.training.seed))
        save_config(config, Path(dirs["config_dir"]) / "config.yaml")
        volume = _prepare_volume(config)
        device = _resolve_device(config.training.device)
        effective_target_cr = (
            float(config.compression.target_cr)
            if config.compression.target_cr is not None
            else float(volume.raw_bytes)
            / (float(config.compression.target_size_mib) * 1024.0 * 1024.0)
        )
        logger.info(
            "DC-INR target=%s volume_shape=%s candidates=%d target_cr=%.4f target_size_mib=%s",
            config.data.target,
            config.data.volume_shape.to_dict() if config.data.volume_shape is not None else {},
            len(config.partition.candidate_block_shapes),
            effective_target_cr,
            config.compression.target_size_mib,
        )
        selection, search_summaries = select_best_candidate(
            volume=volume,
            candidate_shapes=config.partition.candidate_block_shapes,
            dbscan_eps=float(config.partition.dbscan_eps),
            dbscan_min_samples=int(config.partition.dbscan_min_samples),
            entropy_bins=int(config.partition.entropy_bins),
            distance_matrix_max_bytes=int(config.partition.distance_matrix_max_bytes),
            target_cr=effective_target_cr,
            max_initial_neurons=int(config.compression.max_initial_neurons),
            min_initial_neurons=int(config.compression.min_initial_neurons),
        )
        _json_dump(Path(dirs["metrics_dir"]) / "partition_search.json", {"candidates": search_summaries})
        logger.info(
            "DC-INR selected block_shape=%s representatives=%d M=%d mean_capacity=%.2f payload_bytes=%d",
            selection.block_shape.to_dict(),
            int(selection.representative_block_ids.size),
            int(selection.selected_M),
            float(selection.mean_capacity),
            int(selection.payload_bytes),
        )

        if config.compression.target_size_mib is not None:
            selected_stats = _sum_model_stats(np.asarray(selection.widths, dtype=np.int32))
            relative_error = abs(
                float(selected_stats["fp16_size_mb"]) - float(config.compression.target_size_mib)
            ) / float(config.compression.target_size_mib)
            if relative_error > 0.05:
                raise ValueError(
                    "DC-INR selected FP16 parameter size is outside the 5% target tolerance: "
                    f"actual={selected_stats['fp16_size_mb']:.6f}MiB "
                    f"target={config.compression.target_size_mib:.6f}MiB"
                )

        model_states, training_summary = _train_representative_models(
            volume=volume,
            selection=selection,
            config=config,
            device=device,
        )
        model_stats = _sum_model_stats(np.asarray(selection.widths, dtype=np.int32))
        if config.log.model_stats:
            logger.info(
                "DC-INR ensemble size: params=%d trainable=%d size(fp16)=%.2f MB",
                int(model_stats["param_count"]),
                int(model_stats["trainable_param_count"]),
                float(model_stats["fp16_size_mb"]),
            )
        checkpoint_payload = _build_checkpoint_payload(
            config=config,
            config_hash=config_hash,
            selection=selection,
            model_states=model_states,
            search_summaries=search_summaries,
            model_stats=model_stats,
        )
        checkpoint_path, saved_payload = save_dc_checkpoint(
            Path(dirs["checkpoint_dir"]) / f"{config.exp_id}.pth",
            checkpoint_payload,
        )
        _json_dump(
            Path(dirs["metrics_dir"]) / "training_summary.json",
            {
                "selected_block_shape": selection.block_shape.to_dict(),
                "selected_block_grid_shape": selection.grid_shape.to_dict(),
                "selected_M": int(selection.selected_M),
                "representative_count": int(selection.representative_block_ids.size),
                "training": training_summary,
            },
        )
        if not bool(config.evaluation.save_predictions):
            logger.info("Skipping automatic prediction/evaluation after training.")
            return {
                "checkpoint_path": str(checkpoint_path),
            }
        prediction_path = _predict_volume_from_payload(
            payload=saved_payload,
            device=device,
            prediction_batch_size=int(config.training.prediction_batch_size),
            prediction_path=Path(dirs["prediction_dir"]) / f"{config.exp_id}.npy",
        )
        metrics_payload = _evaluate_prediction(
            volume=volume,
            prediction_path=prediction_path,
            payload_bytes=int(saved_payload["payload_bytes"]),
            checkpoint_bytes=int(saved_payload["checkpoint_bytes"]),
        )
        metrics_path = save_metrics(Path(dirs["metrics_dir"]) / f"{config.exp_id}.json", metrics_payload)
        return {
            "checkpoint_path": str(checkpoint_path),
            "prediction_path": str(prediction_path),
            "metrics_path": str(metrics_path),
            "metrics": metrics_payload,
        }
    finally:
        close_file_handlers()


def _load_checkpoint_for_inference(
    *,
    config: DCExperimentConfig,
    checkpoint_path: str | Path,
    config_hash: str,
) -> dict[str, Any]:
    payload = load_dc_checkpoint(checkpoint_path)
    validate_dc_checkpoint(
        payload,
        expected_target_name=config.data.target,
        expected_volume_shape=config.data.volume_shape.to_dict() if config.data.volume_shape is not None else None,
        expected_config_hash=config_hash,
    )
    return payload


def run_predict(
    config_path: str | Path,
    *,
    checkpoint_path: str | Path | None = None,
    target: str | None = None,
) -> dict[str, Any]:
    apply_runtime_thread_limits()
    try:
        config = load_config(config_path, target_override=target)
        dirs = _resolve_existing_run_dirs(config, checkpoint_path=checkpoint_path)
        setup_logging(log_dir=dirs["logs_dir"], log_file=f"run_{dirs['run_token']}.log")
        resolved_checkpoint = (
            Path(checkpoint_path).resolve()
            if checkpoint_path is not None
            else Path(dirs["checkpoint_dir"]) / f"{config.exp_id}.pth"
        )
        config_hash = sha256_payload(config_payload(config))
        payload = _load_checkpoint_for_inference(
            config=config,
            checkpoint_path=resolved_checkpoint,
            config_hash=config_hash,
        )
        prediction_path = _predict_volume_from_payload(
            payload=payload,
            device=_resolve_device(config.training.device),
            prediction_batch_size=int(config.evaluation.batch_size or config.training.prediction_batch_size),
            prediction_path=Path(dirs["prediction_dir"]) / f"{config.exp_id}.npy",
        )
        return {
            "checkpoint_path": str(resolved_checkpoint),
            "prediction_path": str(prediction_path),
        }
    finally:
        close_file_handlers()


def run_evaluate(
    config_path: str | Path,
    *,
    checkpoint_path: str | Path | None = None,
    target: str | None = None,
) -> dict[str, Any]:
    apply_runtime_thread_limits()
    try:
        config = load_config(config_path, target_override=target)
        dirs = _resolve_existing_run_dirs(config, checkpoint_path=checkpoint_path)
        setup_logging(log_dir=dirs["logs_dir"], log_file=f"run_{dirs['run_token']}.log")
        resolved_checkpoint = (
            Path(checkpoint_path).resolve()
            if checkpoint_path is not None
            else Path(dirs["checkpoint_dir"]) / f"{config.exp_id}.pth"
        )
        config_hash = sha256_payload(config_payload(config))
        payload = _load_checkpoint_for_inference(
            config=config,
            checkpoint_path=resolved_checkpoint,
            config_hash=config_hash,
        )
        prediction_path = _predict_volume_from_payload(
            payload=payload,
            device=_resolve_device(config.training.device),
            prediction_batch_size=int(config.evaluation.batch_size or config.training.prediction_batch_size),
            prediction_path=Path(dirs["prediction_dir"]) / f"{config.exp_id}.npy",
        )
        volume = _prepare_volume(config)
        metrics_payload = _evaluate_prediction(
            volume=volume,
            prediction_path=prediction_path,
            payload_bytes=int(payload["payload_bytes"]),
            checkpoint_bytes=int(payload["checkpoint_bytes"]),
        )
        metrics_path = save_metrics(Path(dirs["metrics_dir"]) / f"{config.exp_id}.json", metrics_payload)
        return {
            "checkpoint_path": str(resolved_checkpoint),
            "prediction_path": str(prediction_path),
            "metrics_path": str(metrics_path),
            "metrics": metrics_payload,
        }
    finally:
        close_file_handlers()
