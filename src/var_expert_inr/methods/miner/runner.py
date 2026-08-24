from __future__ import annotations

import copy
import json
import logging
import math
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from numpy.lib.format import open_memmap

from ...evaluation.metrics import mae, mse, psnr, save_metrics
from ...utils.exploration_probe import (
    HEADER as EXPLORATION_PROBE_HEADER,
    fixed_sample_indices,
    normalize_probe,
    psnr_from_arrays,
)
from ...utils.io import sha256_payload
from ...utils.logging_utils import close_file_handlers, setup_logging
from ...utils.runtime import apply_runtime_thread_limits, set_random_seed
from ...utils.temporal_checkpoint import TemporalCheckpointReader, TemporalCheckpointWriter
from .blocks import (
    blockify,
    build_pyramid,
    crop_padding,
    effective_scale_count,
    local_coordinate_grid,
    pad_to_scale_compatible,
    resize_signal,
    unblockify,
)
from .config import load_config, save_config
from .data import ScalarVolumeReader
from .model import (
    BlockSiren,
    cpu_state_dict,
    merge_state_channels,
    propagate_state_to_finer_grid,
    select_state_channels,
)


logger = logging.getLogger(__name__)
CHECKPOINT_FORMAT = "miner_timestep_inference_v1"
BUNDLE_FORMAT = "miner_temporal_inference_v1"


def _device(requested: str) -> torch.device:
    normalized = str(requested).strip().lower()
    if normalized.startswith("cuda") and not torch.cuda.is_available():
        logger.warning("CUDA requested but unavailable; falling back to CPU")
        return torch.device("cpu")
    return torch.device(requested)


def _config_hash(cfg: dict[str, Any]) -> str:
    payload = copy.deepcopy(cfg)
    payload.pop("CONFIG_PATH", None)
    return sha256_payload(payload)


def _write_json(path: str | Path, payload: dict[str, Any]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
    temporary.replace(target)
    return target


def _scale_exploration_metrics(
    cfg: dict[str, Any],
    *,
    target: torch.Tensor,
    reconstruction: torch.Tensor,
    time_index: int,
    scale_index: int,
    effective_scales: int,
    elapsed_seconds: float,
) -> dict[str, Any] | None:
    probe = normalize_probe(cfg.get("exploration_probe"))
    if not probe.enabled:
        return None
    target_values = target.detach().cpu().numpy().reshape(-1)
    prediction_values = reconstruction.detach().cpu().numpy().reshape(-1)
    indices = fixed_sample_indices(
        int(target_values.size),
        probe,
        salt=int(time_index) * 1000 + int(scale_index),
    )
    progress = math.ceil(
        int(probe.total_epoch_equivalents)
        * (int(scale_index) + 1)
        / max(int(effective_scales), 1)
    )
    return {
        "time_index": int(time_index),
        "scale_index": int(scale_index),
        "effective_scales": int(effective_scales),
        "progress": int(progress),
        "psnr": psnr_from_arrays(target_values[indices], prediction_values[indices]),
        "sample_count": int(indices.size),
        "shape": [int(value) for value in target.shape],
        "elapsed_seconds": float(elapsed_seconds),
        "full_resolution": False,
    }


def _write_exploration_trajectory(
    metrics_dir: Path,
    entries: list[dict[str, Any]],
    cfg: dict[str, Any],
) -> Path | None:
    probe = normalize_probe(cfg.get("exploration_probe"))
    if not probe.enabled:
        return None
    rows: list[dict[str, Any]] = []
    for entry in entries:
        metrics_path = Path(entry["metrics_path"])
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        rows.extend(list(payload.get("scale_metrics") or []))
    rows.sort(key=lambda item: (int(item["time_index"]), int(item["scale_index"])))
    target = metrics_dir / "exploration_psnr.tsv"
    temporary = target.with_suffix(target.suffix + ".tmp")
    lines = [EXPLORATION_PROBE_HEADER]
    for row in rows:
        details = {
            "time_index": int(row["time_index"]),
            "scale_index": int(row["scale_index"]),
            "effective_scales": int(row["effective_scales"]),
            "shape": list(row["shape"]),
            "full_resolution": bool(row.get("full_resolution", False)),
        }
        detail_text = json.dumps(
            details, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
        lines.append(
            f"{int(row['progress'])}\t{int(probe.total_epoch_equivalents)}\t"
            f"timestep:t{int(row['time_index']):04d}\t{float(row['psnr']):.8g}\t"
            f"{int(row['sample_count'])}\t{float(row['elapsed_seconds']):.6f}\t"
            f"{detail_text}\n"
        )
    temporary.write_text("".join(lines), encoding="utf-8")
    temporary.replace(target)
    return target


def _run_layout(run_dir: Path, *, create: bool = True) -> dict[str, Path]:
    result = {
        "run": run_dir,
        "configs": run_dir / "configs",
        "checkpoints": run_dir / "checkpoints",
        "timesteps": run_dir / "timesteps",
        "predictions": run_dir / "predictions",
        "metrics": run_dir / "metrics",
        "logs": run_dir / "logs",
    }
    if create:
        for path in result.values():
            path.mkdir(parents=True, exist_ok=True)
    return result


def _new_run(cfg: dict[str, Any]) -> dict[str, Path]:
    experiment = Path(cfg["experiment_root"]) / cfg["exp_id"]
    experiment.mkdir(parents=True, exist_ok=True)
    while True:
        run_dir = experiment / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        if not run_dir.exists():
            return _run_layout(run_dir)
        time.sleep(0.001)


def _latest_run(cfg: dict[str, Any]) -> dict[str, Path]:
    experiment = Path(cfg["experiment_root"]) / cfg["exp_id"]
    candidates = sorted(path for path in experiment.glob("*") if path.is_dir())
    if not candidates:
        raise FileNotFoundError(f"No MINER run found under {experiment}")
    return _run_layout(candidates[-1])


def _timestep_dir(layout: dict[str, Path], time_index: int) -> Path:
    result = layout["timesteps"] / f"t{int(time_index):04d}"
    result.mkdir(parents=True, exist_ok=True)
    return result


def _build_model(
    *, channels: int, dimensions: int, hidden_features: int, model_cfg: dict[str, Any], scale_index: int
) -> BlockSiren:
    return BlockSiren(
        channels=int(channels),
        in_features=int(dimensions),
        hidden_features=int(hidden_features),
        hidden_layers=int(model_cfg["hidden_layers"]),
        omega_0=float(model_cfg["omega_0"]),
        initialization_scale=2.0 ** (-int(scale_index)),
    )


def _predict_blocks(
    model: BlockSiren,
    coordinates: torch.Tensor,
    *,
    count: int,
    batch_blocks: int,
    device: torch.device,
) -> torch.Tensor:
    result = torch.empty((int(count), int(coordinates.shape[0])), dtype=torch.float32)
    model.eval()
    with torch.inference_mode():
        for start in range(0, int(count), int(batch_blocks)):
            stop = min(int(count), start + int(batch_blocks))
            indices = torch.arange(start, stop, dtype=torch.long, device=device)
            prediction = model(coordinates.to(device), indices)[..., 0]
            result[start:stop] = prediction.detach().cpu()
    return result


def _decode_scale_payloads(
    scales: list[dict[str, Any]], *, device: torch.device, batch_blocks: int
) -> torch.Tensor:
    reconstruction: torch.Tensor | None = None
    for scale in scales:
        shape = tuple(int(value) for value in scale["shape"])
        if reconstruction is None:
            reconstruction = torch.zeros(shape, dtype=torch.float32)
        else:
            reconstruction = resize_signal(reconstruction, shape)
        if bool(scale.get("empty", False)):
            continue
        model = BlockSiren(**scale["model_config"]).to(device)
        model.load_state_dict(scale["model_state"], strict=True)
        master = torch.as_tensor(scale["master_indices"], dtype=torch.long)
        coordinates = local_coordinate_grid(
            int(scale["block_size"]), int(scale["dimensions"])
        )
        predictions = _predict_blocks(
            model,
            coordinates,
            count=int(master.numel()),
            batch_blocks=int(batch_blocks),
            device=device,
        )
        all_blocks = torch.zeros(
            (math.prod(tuple(int(value) for value in scale["grid_shape"])), predictions.shape[1]),
            dtype=torch.float32,
        )
        all_blocks[master] = predictions
        reconstruction = reconstruction + unblockify(
            all_blocks,
            tuple(int(value) for value in scale["grid_shape"]),
            int(scale["block_size"]),
        )
        del model
    if reconstruction is None:
        raise ValueError("MINER checkpoint contains no completed scales")
    return reconstruction


def decode_checkpoint(
    checkpoint_path: str | Path,
    *,
    time_index: int | None = None,
    device: torch.device | str = "cpu",
    batch_blocks: int | None = None,
) -> np.ndarray:
    resolved_device = torch.device(device)
    with TemporalCheckpointReader(
        checkpoint_path, expected_format=BUNDLE_FORMAT
    ) as checkpoint_reader:
        if time_index is None:
            if len(checkpoint_reader.time_indices) != 1:
                raise ValueError("time_index is required for a multi-timestep MINER checkpoint")
            time_index = checkpoint_reader.time_indices[0]
        payload = checkpoint_reader.load_timestep(time_index, map_location="cpu")
    if payload.get("format") != CHECKPOINT_FORMAT:
        raise ValueError(f"Unsupported MINER timestep payload: {payload.get('format')!r}")
    if str(payload.get("status", "")) != "complete":
        raise ValueError("MINER inference requires a completed final timestep checkpoint")
    configured_batch = int(payload.get("max_active_blocks_per_step", 2048))
    reconstructed = _decode_scale_payloads(
        list(payload["scales"]),
        device=resolved_device,
        batch_blocks=int(batch_blocks or configured_batch),
    )
    original_shape = tuple(int(value) for value in payload["original_shape"])
    return crop_padding(reconstructed.numpy(), original_shape)


def _train_active_model(
    model: BlockSiren,
    targets: torch.Tensor,
    *,
    cfg: dict[str, Any],
    scale_index: int,
    final_scale: bool,
    device: torch.device,
) -> tuple[dict[str, Any], torch.Tensor]:
    training = cfg["training"]
    coordinates = local_coordinate_grid(
        int(cfg["model"]["block_size"]), int(model.in_features)
    ).to(device)
    model.to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(training["lr"]),
        betas=(float(training["beta_1"]), float(training["beta_2"])),
    )
    active = torch.arange(model.channels, dtype=torch.long)
    previous_loss: float | None = None
    logical_samples = 0
    optimizer_steps = 0
    epochs_executed = 0
    started = time.perf_counter()
    batch_blocks = int(training["max_active_blocks_per_step"])
    base_lr = float(training["lr"]) / (4.0 if int(scale_index) == 0 else 1.0)
    for epoch in range(int(training["epochs_per_scale"])):
        if active.numel() == 0:
            break
        model.train()
        lr = (
            base_lr
            * float(active.numel())
            / float(max(model.channels, 1))
            * (float(training["lr_decay"]) ** epoch)
        )
        for group in optimizer.param_groups:
            group["lr"] = lr
        loss_sum = 0.0
        chunk_count = 0
        for start in range(0, int(active.numel()), batch_blocks):
            selected = active[start : start + batch_blocks].to(device)
            target_batch = targets[selected.cpu()].to(device, non_blocking=True)[..., None]
            optimizer.zero_grad(set_to_none=True)
            prediction = model(coordinates, selected)
            loss = F.mse_loss(prediction, target_batch)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach())
            chunk_count += 1
            logical_samples += int(selected.numel()) * int(coordinates.shape[0])
            optimizer_steps += 1
        epochs_executed = epoch + 1
        current_loss = loss_sum / max(chunk_count, 1)
        predictions = _predict_blocks(
            model,
            coordinates.detach().cpu(),
            count=model.channels,
            batch_blocks=batch_blocks,
            device=device,
        )
        errors = torch.mean((predictions - targets) ** 2, dim=1)
        active = torch.nonzero(
            errors > float(training["block_mse_threshold"]), as_tuple=False
        ).flatten()
        global_mse = float(errors.mean())
        if (
            bool(cfg["log"]["epoch_summary"])
            and int(training["log_every"]) > 0
            and (epochs_executed == 1 or epochs_executed % int(training["log_every"]) == 0)
        ):
            logger.info(
                "MINER scale=%d epoch=%d/%d loss=%.6e global_mse=%.6e active=%d/%d lr=%.3e",
                scale_index,
                epochs_executed,
                int(training["epochs_per_scale"]),
                current_loss,
                global_mse,
                int(active.numel()),
                model.channels,
                lr,
            )
        threshold = float(training["global_mse_threshold"])
        if threshold > 0.0 and global_mse <= threshold:
            break
        if (
            not final_scale
            and previous_loss is not None
            and abs(current_loss - previous_loss) < float(training["scale_convergence_delta"])
        ):
            break
        previous_loss = current_loss
    predictions = _predict_blocks(
        model,
        coordinates.detach().cpu(),
        count=model.channels,
        batch_blocks=batch_blocks,
        device=device,
    )
    return {
        "epochs_executed": epochs_executed,
        "logical_samples": logical_samples,
        "optimizer_steps": optimizer_steps,
        "remaining_active_blocks": int(active.numel()),
        "elapsed_seconds": float(time.perf_counter() - started),
    }, predictions


def _train_timestep(
    cfg: dict[str, Any],
    reader: ScalarVolumeReader,
    *,
    time_index: int,
    timestep_dir: Path,
    config_hash: str,
    device: torch.device,
) -> dict[str, Any]:
    set_random_seed(int(cfg["training"]["seed"]) + int(time_index))
    frame = reader.timestep(time_index)
    requested_scales = int(cfg["model"]["scales"])
    effective_scales = effective_scale_count(
        tuple(frame.shape),
        block_size=int(cfg["model"]["block_size"]),
        requested_scales=requested_scales,
    )
    padded, padding = pad_to_scale_compatible(
        frame,
        block_size=int(cfg["model"]["block_size"]),
        scales=effective_scales,
    )
    pyramid = build_pyramid(padded, effective_scales)
    completed_scales: list[dict[str, Any]] = []
    carry_state: dict[str, torch.Tensor] | None = None
    reconstruction = None
    probe = normalize_probe(cfg.get("exploration_probe"))
    total_parameters = 0
    total_logical_samples = 0
    total_optimizer_steps = 0
    started = time.perf_counter()
    for scale_index in range(effective_scales):
        target = pyramid[scale_index]
        previous = torch.zeros_like(target) if reconstruction is None else resize_signal(reconstruction, tuple(target.shape))
        residual = target - previous
        residual_blocks, grid_shape = blockify(residual, int(cfg["model"]["block_size"]))
        energies = torch.mean(residual_blocks**2, dim=1)
        if scale_index == 0:
            master = torch.arange(residual_blocks.shape[0], dtype=torch.long)
        else:
            master = torch.nonzero(
                energies > float(cfg["training"]["block_mse_threshold"]),
                as_tuple=False,
            ).flatten()
        hidden_features = int(cfg["model"]["hidden_features"])
        if scale_index == 0:
            hidden_features *= int(cfg["model"]["coarse_feature_multiplier"])
        scale_payload: dict[str, Any] = {
            "scale_index": int(scale_index),
            "shape": list(target.shape),
            "grid_shape": list(grid_shape),
            "block_size": int(cfg["model"]["block_size"]),
            "dimensions": int(reader.spatial_dimensions),
            "master_indices": master.cpu(),
            "empty": bool(master.numel() == 0),
        }
        next_carry: dict[str, torch.Tensor] | None = None
        if master.numel() == 0:
            reconstruction = previous
            if (
                scale_index + 1 < effective_scales
                and scale_index + 1 >= int(cfg["model"]["carry_start_scale"])
                and carry_state is not None
            ):
                next_carry = propagate_state_to_finer_grid(carry_state, grid_shape)
            scale_payload.update({
                "model_config": None,
                "model_state": None,
                "parameter_count": 0,
                "training": {
                    "epochs_executed": 0,
                    "logical_samples": 0,
                    "optimizer_steps": 0,
                    "remaining_active_blocks": 0,
                    "elapsed_seconds": 0.0,
                },
            })
        else:
            full_channels = int(residual_blocks.shape[0])
            if scale_index >= int(cfg["model"]["carry_start_scale"]) and carry_state is not None:
                full_state = {name: value.detach().cpu() for name, value in carry_state.items()}
                if int(next(iter(full_state.values())).shape[0]) != full_channels:
                    raise ValueError("Propagated MINER state does not match the finer block grid")
            else:
                torch.manual_seed(int(cfg["training"]["seed"]) + int(time_index) * 1000 + scale_index)
                full_model = _build_model(
                    channels=full_channels,
                    dimensions=reader.spatial_dimensions,
                    hidden_features=hidden_features,
                    model_cfg=cfg["model"],
                    scale_index=scale_index,
                )
                full_state = cpu_state_dict(full_model)
                del full_model
            model = _build_model(
                channels=int(master.numel()),
                dimensions=reader.spatial_dimensions,
                hidden_features=hidden_features,
                model_cfg=cfg["model"],
                scale_index=scale_index,
            )
            model.load_state_dict(select_state_channels(full_state, master), strict=True)
            training_info, predicted_active = _train_active_model(
                model,
                residual_blocks[master],
                cfg=cfg,
                scale_index=scale_index,
                final_scale=scale_index == effective_scales - 1,
                device=device,
            )
            selected_state = cpu_state_dict(model)
            merged_state = merge_state_channels(full_state, selected_state, master)
            if scale_index + 1 < effective_scales and scale_index + 1 >= int(cfg["model"]["carry_start_scale"]):
                next_carry = propagate_state_to_finer_grid(merged_state, grid_shape)
            all_residual_blocks = torch.zeros_like(residual_blocks)
            all_residual_blocks[master] = predicted_active
            reconstruction = previous + unblockify(
                all_residual_blocks, grid_shape, int(cfg["model"]["block_size"])
            )
            parameter_count = sum(int(value.numel()) for value in selected_state.values())
            total_parameters += parameter_count
            total_logical_samples += int(training_info["logical_samples"])
            total_optimizer_steps += int(training_info["optimizer_steps"])
            scale_payload.update({
                "model_config": model.config_dict(),
                "model_state": selected_state,
                "parameter_count": parameter_count,
                "training": training_info,
            })
            del model
        scale_payload["exploration_metrics"] = _scale_exploration_metrics(
            cfg,
            target=target,
            reconstruction=reconstruction,
            time_index=time_index,
            scale_index=scale_index,
            effective_scales=effective_scales,
            elapsed_seconds=sum(
                float(item.get("training", {}).get("elapsed_seconds", 0.0))
                for item in completed_scales
            )
            + float(scale_payload["training"].get("elapsed_seconds", 0.0)),
        )
        completed_scales.append(scale_payload)
        carry_state = next_carry
        logger.info(
            "MINER timestep=%d scale=%d/%d blocks=%d params=%d",
            time_index,
            scale_index + 1,
            effective_scales,
            int(master.numel()),
            int(scale_payload["parameter_count"]),
        )
    if reconstruction is None:
        raise ValueError("MINER produced no reconstruction")
    prediction = crop_padding(reconstruction.numpy(), tuple(frame.shape))
    final_mse = mse(frame, prediction)
    final_mae = mae(frame, prediction)
    final_psnr = psnr(frame, prediction)
    if probe.enabled and completed_scales[-1].get("exploration_metrics") is not None:
        completed_scales[-1]["exploration_metrics"].update(
            {
                "progress": int(probe.total_epoch_equivalents),
                "psnr": float(final_psnr),
                "sample_count": int(frame.size),
                "shape": [int(value) for value in frame.shape],
                "full_resolution": True,
            }
        )
    scale_metrics = [
        dict(item["exploration_metrics"])
        for item in completed_scales
        if item.get("exploration_metrics") is not None
    ]
    inference_scales = [
        {
            key: scale[key]
            for key in (
                "shape",
                "grid_shape",
                "block_size",
                "dimensions",
                "master_indices",
                "empty",
                "model_config",
                "model_state",
            )
        }
        for scale in completed_scales
    ]
    final_payload = {
        "format": CHECKPOINT_FORMAT,
        "status": "complete",
        "config_hash": config_hash,
        "time_index": int(time_index),
        "target_name": str(cfg["data"]["target"]),
        "original_shape": list(frame.shape),
        "max_active_blocks_per_step": int(cfg["training"]["max_active_blocks_per_step"]),
        "scales": inference_scales,
    }
    metrics_payload = {
        "time_index": int(time_index),
        "target": str(cfg["data"]["target"]),
        "mse": final_mse,
        "mae": final_mae,
        "psnr": final_psnr,
        "parameter_count": int(total_parameters),
        "fp16_size_bytes": int(total_parameters * 2),
        "logical_samples": int(total_logical_samples),
        "optimizer_steps": int(total_optimizer_steps),
        "elapsed_seconds": float(time.perf_counter() - started),
        "effective_scales": int(effective_scales),
        "scale_metrics": scale_metrics,
    }
    metrics_path = _write_json(timestep_dir / "metrics.json", metrics_payload)
    return {
        "status": "completed",
        "config_hash": config_hash,
        "time_index": int(time_index),
        "metrics_path": str(metrics_path),
        "checkpoint_payload": final_payload,
        **metrics_payload,
    }


def run_train(
    config_path: str | Path,
) -> dict[str, Any]:
    apply_runtime_thread_limits()
    cfg = load_config(config_path)
    layout = _new_run(cfg)
    setup_logging(log_dir=layout["logs"], log_file="train.log")
    try:
        config_hash = _config_hash(cfg)
        saved_config = layout["configs"] / "config.yaml"
        manifest_path = layout["run"] / "manifest.json"
        save_config(cfg, saved_config)
        reader = ScalarVolumeReader(cfg["data"]["target_path"], cfg["data"]["volume_shape"])
        if cfg["log"]["effective_config"]:
            logger.info("MINER effective config: %s", json.dumps({k: v for k, v in cfg.items() if k != "CONFIG_PATH"}, default=str))
        manifest = {
            "schema_version": 1,
            "status": "running",
            "experiment": cfg["experiment"],
            "exp_id": cfg["exp_id"],
            "config_hash": config_hash,
            "dataset_name": cfg["data"]["dataset_name"],
            "target": cfg["data"]["target"],
            "volume_shape": cfg["data"]["volume_shape"],
            "time_indices": cfg["training"]["time_indices"],
            "timesteps": {},
        }
        _write_json(manifest_path, manifest)
        resolved_device = _device(cfg["training"]["device"])
        completed: list[int] = []
        skipped: list[int] = []
        checkpoint_path = layout["checkpoints"] / f"{cfg['exp_id']}.pth"
        with TemporalCheckpointWriter(
            checkpoint_path,
            metadata={
                "format": BUNDLE_FORMAT,
                "model_name": "miner",
                "config_hash": config_hash,
                "target": cfg["data"]["target"],
                "volume_shape": cfg["data"]["volume_shape"],
            },
        ) as checkpoint_writer:
            for time_index in cfg["training"]["time_indices"]:
                token = f"t{int(time_index):04d}"
                timestep_dir = _timestep_dir(layout, int(time_index))
                result = _train_timestep(
                    cfg,
                    reader,
                    time_index=int(time_index),
                    timestep_dir=timestep_dir,
                    config_hash=config_hash,
                    device=resolved_device,
                )
                checkpoint_writer.write_timestep(
                    int(time_index), result.pop("checkpoint_payload")
                )
                manifest["timesteps"][token] = result
                _write_json(manifest_path, manifest)
                completed.append(int(time_index))
        entries = [manifest["timesteps"][f"t{int(index):04d}"] for index in cfg["training"]["time_indices"]]
        exploration_metrics_path = _write_exploration_trajectory(
            layout["metrics"], entries, cfg
        )
        checkpoint_bytes = int(checkpoint_path.stat().st_size)
        raw_target_bytes = int(reader.raw_bytes(cfg["training"]["time_indices"]))
        cr = float(raw_target_bytes / checkpoint_bytes) if checkpoint_bytes else float("inf")
        model_stats = {
            "target": cfg["data"]["target"],
            "timestep_count": len(entries),
            "parameter_count": sum(int(item["parameter_count"]) for item in entries),
            "fp16_size_bytes": sum(int(item["fp16_size_bytes"]) for item in entries),
            "checkpoint_bytes": checkpoint_bytes,
            "raw_target_bytes": raw_target_bytes,
            "cr": cr,
            "logical_samples": sum(int(item["logical_samples"]) for item in entries),
            "optimizer_steps": sum(int(item["optimizer_steps"]) for item in entries),
        }
        model_stats_path = _write_json(layout["metrics"] / "model_stats.json", model_stats)
        manifest.update({"status": "complete", "model_stats": model_stats})
        _write_json(manifest_path, manifest)
        prediction_path = None
        if cfg["evaluation"]["save_predictions"] or cfg["evaluation"]["run_after_training"]:
            prediction_path = run_predict(config_path, checkpoint=checkpoint_path)["prediction_path"]
        return {
            "run_dir": layout["run"],
            "checkpoint_path": checkpoint_path,
            "checkpoint_bytes": checkpoint_bytes,
            "raw_target_bytes": raw_target_bytes,
            "cr": cr,
            "manifest_path": manifest_path,
            "model_stats_path": model_stats_path,
            "exploration_metrics_path": exploration_metrics_path,
            "prediction_path": prediction_path,
            "completed_timesteps": completed,
            "skipped_timesteps": skipped,
        }
    finally:
        close_file_handlers()


def run_predict(
    config_path: str | Path,
    *,
    checkpoint: str | Path | None = None,
) -> dict[str, Any]:
    cfg = load_config(config_path)
    layout = _latest_run(cfg) if checkpoint is None else None
    source = (
        layout["checkpoints"] / f"{cfg['exp_id']}.pth"
        if layout is not None
        else Path(checkpoint).resolve()
    )
    if layout is not None:
        output_run = layout["run"]
    else:
        output_run = source.parent.parent if source.parent.name == "checkpoints" else source.parent
    prediction_dir = output_run / "predictions"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    reader = ScalarVolumeReader(cfg["data"]["target_path"], cfg["data"]["volume_shape"])
    output_path = prediction_dir / f"{cfg['exp_id']}.npy"
    output = open_memmap(output_path, mode="w+", dtype=np.float32, shape=reader.shape_tzyx)
    resolved_device = _device(cfg["training"]["device"])
    decoded_indices: list[int] = []
    with TemporalCheckpointReader(source, expected_format=BUNDLE_FORMAT) as checkpoint_reader:
        available = set(checkpoint_reader.time_indices)
    selected = [
        int(index) for index in cfg["training"]["time_indices"] if int(index) in available
    ]
    if not selected:
        raise ValueError("MINER checkpoint contains none of the configured time_indices")
    for time_index in selected:
        decoded = decode_checkpoint(
            source,
            time_index=int(time_index),
            device=resolved_device,
            batch_blocks=int(cfg["training"]["max_active_blocks_per_step"]),
        )
        output[int(time_index)] = reader.restore_storage_shape(decoded)
        decoded_indices.append(int(time_index))
    output.flush()
    return {"prediction_path": output_path, "decoded_timesteps": decoded_indices}


def run_evaluate(
    config_path: str | Path,
    *,
    checkpoint: str | Path | None = None,
) -> dict[str, Any]:
    cfg = load_config(config_path)
    prediction_result = run_predict(config_path, checkpoint=checkpoint)
    prediction = np.load(prediction_result["prediction_path"], mmap_mode="r")
    reader = ScalarVolumeReader(cfg["data"]["target_path"], cfg["data"]["volume_shape"])
    selected = prediction_result["decoded_timesteps"]
    gt = np.asarray(reader.array[selected], dtype=np.float32)
    estimated = np.asarray(prediction[selected], dtype=np.float32)
    metrics = {
        "targets": {
            cfg["data"]["target"]: {
                "mse": mse(gt, estimated),
                "mae": mae(gt, estimated),
                "psnr": psnr(gt, estimated),
            }
        },
        "aggregate": {
            "mse": mse(gt, estimated),
            "mae": mae(gt, estimated),
            "psnr": psnr(gt, estimated),
        },
    }
    metrics_path = save_metrics(Path(prediction_result["prediction_path"]).parent.parent / "metrics" / f"{cfg['exp_id']}.json", metrics)
    return {**prediction_result, "metrics": metrics, "metrics_path": metrics_path}
