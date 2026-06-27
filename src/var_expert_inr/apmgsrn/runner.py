from __future__ import annotations

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from ..evaluation.metrics import PSNRAccumulator, mae, mse, psnr, save_metrics
from ..utils.io import sha256_payload
from ..utils.logging_utils import close_file_handlers, setup_logging
from ..utils.model_stats import collect_model_statistics
from ..utils.runtime import apply_runtime_thread_limits, set_random_seed
from .config import config_payload, experiment_dir_from_config, load_config, save_config
from .dataset import IonizationTargetReader, IonizationTimestepDataset
from .model import APMGSRN

logger = logging.getLogger(__name__)


def _resolve_device(requested: str) -> torch.device:
    requested_norm = str(requested).strip().lower()
    if requested_norm.startswith("cuda") and not torch.cuda.is_available():
        logger.warning("CUDA requested but unavailable. Falling back to CPU.")
        return torch.device("cpu")
    return torch.device(requested)


def _resolve_data_device(requested: str, *, train_device: torch.device) -> torch.device:
    requested_norm = str(requested).strip().lower()
    if requested_norm == "same":
        return train_device
    if requested_norm.startswith("cuda") and not torch.cuda.is_available():
        logger.warning("CUDA data_device requested but unavailable. Falling back to CPU.")
        return torch.device("cpu")
    return torch.device(requested)


def _timestep_token(time_index: int) -> str:
    return f"t{int(time_index):03d}"


def _json_dump(path: str | Path, payload: dict[str, Any]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
    return target


def _checkpoint_payload(
    *,
    model: APMGSRN,
    cfg: dict[str, Any],
    config_hash: str,
    time_index: int,
    target_name: str,
    data_min: float,
    data_max: float,
    stats: dict[str, int | float],
) -> dict[str, Any]:
    return {
        "model_state": model.state_dict(),
        "config_hash": str(config_hash),
        "time_index": int(time_index),
        "target_name": str(target_name),
        "model_config": dict(cfg["MODEL"]),
        "data_min": float(data_min),
        "data_max": float(data_max),
        "stats": dict(stats),
    }


def _save_checkpoint(path: str | Path, payload: dict[str, Any]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, target)
    return target


def _build_run_dirs(run_dir: Path) -> dict[str, Path | str]:
    dirs = {
        "experiment_dir": run_dir.parent,
        "run_dir": run_dir,
        "run_token": run_dir.name,
        "config_dir": run_dir / "configs",
        "timesteps_dir": run_dir / "timesteps",
        "logs_dir": run_dir / "logs",
        "predictions_dir": run_dir / "predictions",
        "metrics_dir": run_dir / "metrics",
    }
    return dirs


def _ensure_run_layout(run_dir: Path) -> dict[str, Path | str]:
    dirs = _build_run_dirs(run_dir)
    for path in dirs.values():
        if isinstance(path, Path):
            path.mkdir(parents=True, exist_ok=True)
    return dirs


def _create_train_run_dirs(cfg: dict[str, Any]) -> dict[str, Path | str]:
    experiment_dir = experiment_dir_from_config(cfg).resolve()
    experiment_dir.mkdir(parents=True, exist_ok=True)
    while True:
        run_token = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        run_dir = experiment_dir / run_token
        if not run_dir.exists():
            return _ensure_run_layout(run_dir)
        time.sleep(0.001)


def _ensure_timestep_dir(run_dir: Path, *, time_index: int) -> Path:
    timestep_dir = run_dir / "timesteps" / _timestep_token(time_index)
    timestep_dir.mkdir(parents=True, exist_ok=True)
    return timestep_dir


def _manifest_path(run_dir: Path) -> Path:
    return run_dir / "manifest.json"


def _initial_manifest(cfg: dict[str, Any], *, config_hash: str) -> dict[str, Any]:
    return {
        "experiment": str(cfg["experiment"]),
        "exp_id": str(cfg["exp_id"]),
        "source_config_path": str(cfg["CONFIG_PATH"]),
        "config_hash": str(config_hash),
        "status": "running",
        "dataset_name": str(cfg["DATA"]["dataset_name"]),
        "target": str(cfg["DATA"]["target"]),
        "target_path": str(cfg["DATA"]["target_path"]),
        "volume_shape": dict(cfg["DATA"]["volume_shape"]),
        "time_indices": list(cfg["TRAINING"]["time_indices"]),
        "timesteps": {},
        "aggregate": {},
    }


def _can_skip_timestep(entry: dict[str, Any] | None, *, config_hash: str) -> bool:
    if not entry:
        return False
    if str(entry.get("config_hash", "")) != str(config_hash):
        raise ValueError(
            f"Timestep {entry.get('time_index')} was trained with a different config hash. "
            "Choose a new exp_id or clean the run directory."
        )
    if str(entry.get("status", "")) != "completed":
        return False
    required_paths = (
        entry.get("checkpoint_path"),
        entry.get("prediction_path"),
        entry.get("metrics_path"),
    )
    return all(path and Path(path).exists() for path in required_paths)


def _build_model(cfg: dict[str, Any], *, data_min: float, data_max: float, device: torch.device) -> APMGSRN:
    model_cfg = dict(cfg["MODEL"])
    allow_tcnn = bool(model_cfg.get("use_tcnn_if_available", True) and device.type == "cuda")
    if allow_tcnn:
        try:
            return APMGSRN(model_cfg, data_min=data_min, data_max=data_max, use_tcnn=True)
        except RuntimeError:
            logger.warning("tinycudann unavailable for APMGSRN. Falling back to pure PyTorch decoder.")
    return APMGSRN(model_cfg, data_min=data_min, data_max=data_max, use_tcnn=False)


def _format_loss(grid_loss: float | None) -> str:
    if grid_loss is None:
        return "grid_loss=<none>"
    return f"grid_loss={grid_loss:.6e}"


def _train_single_timestep(
    *,
    cfg: dict[str, Any],
    reader: IonizationTargetReader,
    time_index: int,
    run_dir: Path,
    config_hash: str,
    train_device: torch.device,
    data_device: torch.device,
) -> dict[str, Any]:
    timestep_dir = _ensure_timestep_dir(run_dir, time_index=time_index)
    dataset = IonizationTimestepDataset(
        reader,
        time_index=time_index,
        align_corners=bool(cfg["DATA"]["align_corners"]),
        device=data_device,
    )
    set_random_seed(int(cfg["TRAINING"]["seed"]) + int(time_index))
    model = _build_model(cfg, data_min=dataset.data_min, data_max=dataset.data_max, device=train_device).to(train_device)
    stats = collect_model_statistics(model)

    optimizer_model = torch.optim.Adam(
        model.get_model_parameters(),
        lr=float(cfg["TRAINING"]["lr"]),
        betas=(float(cfg["TRAINING"]["beta_1"]), float(cfg["TRAINING"]["beta_2"])),
        eps=1.0e-14,
    )
    optimizer_grid = torch.optim.Adam(
        model.get_transform_parameters(),
        lr=float(cfg["TRAINING"]["lr"]) * 0.05,
        betas=(float(cfg["TRAINING"]["beta_1"]), float(cfg["TRAINING"]["beta_2"])),
        eps=1.0e-14,
    )
    scheduler_model = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_model,
        mode="min",
        patience=500,
        threshold=1.0e-4,
        threshold_mode="rel",
        cooldown=250,
        factor=0.1,
    )
    scheduler_grid = torch.optim.lr_scheduler.LinearLR(
        optimizer_grid,
        start_factor=1.0,
        end_factor=0.5,
    )

    iterations = int(cfg["TRAINING"]["iterations"])
    points_per_iteration = int(cfg["TRAINING"]["points_per_iteration"])
    log_every = int(cfg["TRAINING"]["log_every"])
    save_every = int(cfg["TRAINING"]["save_every"])
    prediction_batch_size = int(cfg["TRAINING"]["prediction_points_per_batch"])

    early_stop_reconstruction = False
    early_stop_grid = False
    reconstruction_losses = torch.zeros((iterations,), dtype=torch.float32, device=train_device)
    grid_losses = torch.zeros((iterations,), dtype=torch.float32, device=train_device)
    grid_convergence_streak = 0
    started_at = time.time()

    logger.info(
        "APMGSRN timestep %s start: iterations=%d points_per_iteration=%d device=%s data_device=%s",
        _timestep_token(time_index),
        iterations,
        points_per_iteration,
        train_device,
        data_device,
    )

    for iteration in range(iterations):
        if early_stop_reconstruction and early_stop_grid:
            logger.info("APMGSRN timestep %s early stopped at iteration %d", _timestep_token(time_index), iteration)
            break

        optimizer_model.zero_grad(set_to_none=True)
        coords, targets = dataset.sample_random_points(points_per_iteration)
        coords = coords.to(train_device, non_blocking=True)
        targets = targets.to(train_device, non_blocking=True)

        transformed_coords = model.transform(coords)
        model_output = model.forward_pre_transformed(transformed_coords)
        reconstruction_loss = F.mse_loss(model_output, targets, reduction="none").sum(dim=1, keepdim=True)
        reconstruction_loss_mean = reconstruction_loss.mean()
        reconstruction_loss_mean.backward()

        reconstruction_losses[iteration] = reconstruction_loss_mean.detach()
        early_stop_reconstruction = optimizer_model.param_groups[0]["lr"] < float(cfg["TRAINING"]["lr"]) * 1.0e-2

        grid_loss_mean_value: float | None = None
        if iteration > 500 and iteration < int(iterations * 0.8) and not early_stop_grid:
            optimizer_grid.zero_grad(set_to_none=True)
            density = model.feature_density_pre_transformed(transformed_coords)
            density = density / density.sum().detach().clamp_min(1.0e-16)
            target_density = torch.exp(
                torch.log(density + 1.0e-16) * (reconstruction_loss_mean / (reconstruction_loss + 1.0e-16))
            )
            target_density = target_density / target_density.sum().clamp_min(1.0e-16)
            density_loss = F.kl_div(
                torch.log(density + 1.0e-16),
                torch.log(target_density.detach() + 1.0e-16),
                reduction="none",
                log_target=True,
            )
            density_loss_mean = density_loss.mean()
            density_loss_mean.backward()
            optimizer_grid.step()
            scheduler_grid.step()

            grid_losses[iteration] = density_loss_mean.detach()
            grid_loss_mean_value = float(density_loss_mean.detach().item())
            if iteration >= 2500:
                previous_avg = float(grid_losses[iteration - 2000 : iteration - 1000].mean().item())
                current_avg = float(grid_losses[iteration - 1000 : iteration].mean().item())
                threshold = previous_avg * 1.0e-4
                threshold_met = previous_avg - current_avg < threshold
                if threshold_met:
                    grid_convergence_streak += 1
                else:
                    grid_convergence_streak = 0
                early_stop_grid = threshold_met and grid_convergence_streak > 1

        optimizer_model.step()
        if early_stop_grid and iteration >= 1000:
            scheduler_model.step(float(reconstruction_losses[iteration - 1000 : iteration].mean().item()))

        if log_every > 0 and ((iteration + 1) % log_every == 0 or iteration == 0):
            logger.info(
                "APMGSRN timestep %s iter %d/%d recon_loss=%.6e %s lr_model=%.3e lr_grid=%.3e elapsed=%.1fs",
                _timestep_token(time_index),
                iteration + 1,
                iterations,
                float(reconstruction_loss_mean.detach().item()),
                _format_loss(grid_loss_mean_value),
                float(optimizer_model.param_groups[0]["lr"]),
                float(optimizer_grid.param_groups[0]["lr"]),
                time.time() - started_at,
            )

        if save_every > 0 and (iteration + 1) % save_every == 0:
            interval_checkpoint_path = timestep_dir / f"checkpoint_iter{iteration + 1:06d}.pth"
            _save_checkpoint(
                interval_checkpoint_path,
                _checkpoint_payload(
                    model=model,
                    cfg=cfg,
                    config_hash=config_hash,
                    time_index=time_index,
                    target_name=cfg["DATA"]["target"],
                    data_min=dataset.data_min,
                    data_max=dataset.data_max,
                    stats=stats,
                ),
            )

    checkpoint_path = _save_checkpoint(
        timestep_dir / "checkpoint.pth",
        _checkpoint_payload(
            model=model,
            cfg=cfg,
            config_hash=config_hash,
            time_index=time_index,
            target_name=cfg["DATA"]["target"],
            data_min=dataset.data_min,
            data_max=dataset.data_max,
            stats=stats,
        ),
    )
    prediction = dataset.reconstruct(model, batch_size=prediction_batch_size, model_device=train_device)
    prediction_path = timestep_dir / "prediction.npy"
    np.save(prediction_path, prediction)

    timestep_metrics = {
        "time_index": int(time_index),
        "target": str(cfg["DATA"]["target"]),
        "mse": mse(dataset.target_array(), prediction),
        "mae": mae(dataset.target_array(), prediction),
        "psnr": psnr(dataset.target_array(), prediction),
        "checkpoint_bytes": int(checkpoint_path.stat().st_size),
        "elapsed_seconds": float(time.time() - started_at),
    }
    metrics_path = save_metrics(timestep_dir / "metrics.json", timestep_metrics)
    logger.info(
        "APMGSRN timestep %s complete: psnr=%.4f checkpoint=%s prediction=%s",
        _timestep_token(time_index),
        float(timestep_metrics["psnr"]),
        checkpoint_path,
        prediction_path,
    )

    return {
        "time_index": int(time_index),
        "status": "completed",
        "config_hash": str(config_hash),
        "checkpoint_path": str(checkpoint_path),
        "prediction_path": str(prediction_path),
        "metrics_path": str(metrics_path),
        "checkpoint_bytes": int(checkpoint_path.stat().st_size),
        "model_stats": dict(stats),
        "psnr": float(timestep_metrics["psnr"]),
        "mse": float(timestep_metrics["mse"]),
        "mae": float(timestep_metrics["mae"]),
    }


def _build_aggregate_artifacts(
    *,
    cfg: dict[str, Any],
    reader: IonizationTargetReader,
    manifest: dict[str, Any],
    run_dir: Path,
) -> tuple[Path, Path, dict[str, Any]]:
    time_indices = [int(value) for value in cfg["TRAINING"]["time_indices"]]
    spatial_shape = reader.spatial_shape
    prediction_path = run_dir / "predictions" / f"{cfg['exp_id']}.npy"
    prediction_memmap = np.lib.format.open_memmap(
        prediction_path,
        mode="w+",
        dtype=np.float32,
        shape=(len(time_indices), spatial_shape[0], spatial_shape[1], spatial_shape[2]),
    )

    accumulator = PSNRAccumulator()
    total_abs_error = 0.0
    total_count = 0
    total_checkpoint_bytes = 0
    per_time: list[dict[str, Any]] = []

    for position, time_index in enumerate(time_indices):
        token = _timestep_token(time_index)
        entry = manifest["timesteps"].get(token)
        if not _can_skip_timestep(entry, config_hash=str(manifest["config_hash"])):
            raise FileNotFoundError(f"Missing completed timestep artifacts for {token}")
        prediction_slice = np.asarray(np.load(entry["prediction_path"], mmap_mode="r"), dtype=np.float32)
        if tuple(int(value) for value in prediction_slice.shape) != spatial_shape:
            raise ValueError(
                f"Timestep prediction shape mismatch for {token}: expected {spatial_shape}, got {prediction_slice.shape}"
            )
        prediction_memmap[position] = prediction_slice
        target_slice = reader.timestep_array(time_index)
        accumulator.update(target_slice, prediction_slice)
        diff = prediction_slice.astype(np.float64) - target_slice.astype(np.float64)
        total_abs_error += float(np.abs(diff).sum())
        total_count += int(diff.size)
        total_checkpoint_bytes += int(entry["checkpoint_bytes"])
        per_time.append(
            {
                "t": int(time_index),
                "mse": mse(target_slice, prediction_slice),
                "mae": mae(target_slice, prediction_slice),
                "psnr": psnr(target_slice, prediction_slice),
            }
        )

    prediction_memmap.flush()
    total_squared_error = float(accumulator.total_squared_error)
    aggregate_payload = {
        "targets": {
            str(cfg["DATA"]["target"]): {
                "mse": total_squared_error / max(total_count, 1),
                "mae": total_abs_error / max(total_count, 1),
                "psnr": accumulator.compute(),
                "per_time": per_time,
            }
        },
        "aggregate": {
            "mse": total_squared_error / max(total_count, 1),
            "mae": total_abs_error / max(total_count, 1),
            "psnr": accumulator.compute(),
            "cr": float(reader.raw_bytes_for_indices(time_indices) / max(total_checkpoint_bytes, 1)),
            "checkpoint_bytes": int(total_checkpoint_bytes),
            "raw_target_bytes": int(reader.raw_bytes_for_indices(time_indices)),
        },
        "time_indices": time_indices,
    }
    metrics_path = save_metrics(run_dir / "metrics" / "aggregate.json", aggregate_payload)
    return prediction_path, metrics_path, aggregate_payload


def run_train(config_path: str | Path, *, target: str | None = None, identifier: str | None = None) -> dict[str, Any]:
    apply_runtime_thread_limits()
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
    try:
        cfg = load_config(config_path, target_override=target, identifier=identifier)
        config_hash = sha256_payload(config_payload(cfg))
        dirs = _create_train_run_dirs(cfg)
        run_dir = Path(dirs["run_dir"])
        setup_logging(log_dir=Path(dirs["logs_dir"]), log_file=f"run_{dirs['run_token']}.log")
        logger.info("APMGSRN config source: %s", cfg["CONFIG_PATH"])

        reader = IonizationTargetReader(cfg["DATA"]["target_path"], cfg["DATA"]["volume_shape"])
        train_device = _resolve_device(cfg["TRAINING"]["device"])
        data_device = _resolve_data_device(cfg["TRAINING"]["data_device"], train_device=train_device)
        logger.info(
            "APMGSRN target=%s volume_shape=%s selected_timesteps=%d",
            cfg["DATA"]["target"],
            cfg["DATA"]["volume_shape"],
            len(cfg["TRAINING"]["time_indices"]),
        )

        manifest_file = _manifest_path(run_dir)
        manifest = _initial_manifest(cfg, config_hash=config_hash)
        _json_dump(manifest_file, manifest)
        save_config(cfg, Path(dirs["config_dir"]) / "config.yaml")

        completed_timesteps: list[int] = []
        skipped_timesteps: list[int] = []
        for time_index in cfg["TRAINING"]["time_indices"]:
            token = _timestep_token(time_index)
            result = _train_single_timestep(
                cfg=cfg,
                reader=reader,
                time_index=int(time_index),
                run_dir=run_dir,
                config_hash=config_hash,
                train_device=train_device,
                data_device=data_device,
            )
            manifest["timesteps"][token] = result
            manifest["status"] = "running"
            _json_dump(manifest_file, manifest)
            completed_timesteps.append(int(time_index))

        prediction_path, metrics_path, aggregate_payload = _build_aggregate_artifacts(
            cfg=cfg,
            reader=reader,
            manifest=manifest,
            run_dir=run_dir,
        )
        manifest["aggregate"] = {
            "prediction_path": str(prediction_path),
            "metrics_path": str(metrics_path),
            "checkpoint_bytes": int(aggregate_payload["aggregate"]["checkpoint_bytes"]),
            "raw_target_bytes": int(aggregate_payload["aggregate"]["raw_target_bytes"]),
            "cr": float(aggregate_payload["aggregate"]["cr"]),
            "psnr": float(aggregate_payload["aggregate"]["psnr"]),
            "mse": float(aggregate_payload["aggregate"]["mse"]),
            "mae": float(aggregate_payload["aggregate"]["mae"]),
        }
        manifest["status"] = "completed"
        _json_dump(manifest_file, manifest)
        logger.info(
            "APMGSRN aggregate complete: prediction=%s metrics=%s skipped=%d completed=%d",
            prediction_path,
            metrics_path,
            len(skipped_timesteps),
            len(completed_timesteps),
        )
        return {
            "run_dir": str(run_dir),
            "manifest_path": str(manifest_file),
            "prediction_path": str(prediction_path),
            "metrics_path": str(metrics_path),
            "completed_timesteps": completed_timesteps,
            "skipped_timesteps": skipped_timesteps,
        }
    finally:
        close_file_handlers()
