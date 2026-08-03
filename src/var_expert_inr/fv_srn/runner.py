from __future__ import annotations

import copy
import json
import logging
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
from ..utils.model_stats import collect_model_statistics
from ..utils.runtime import apply_runtime_thread_limits, set_random_seed
from .checkpoint import load_payload, restore_rng, save_checkpoint, validate
from .config import load_config, save_config
from .data import SamplePool, TemporalVolume, build_sample_pool
from .model import TemporalFVSRN
from .quantization import export_compact, load_compact

logger = logging.getLogger(__name__)


def _device(requested: str) -> torch.device:
    if str(requested).lower().startswith("cuda") and not torch.cuda.is_available():
        logger.warning("CUDA requested but unavailable; falling back to CPU")
        return torch.device("cpu")
    return torch.device(requested)


def _hash(cfg: dict) -> str:
    payload = copy.deepcopy(cfg)
    payload.pop("CONFIG_PATH", None)
    return sha256_payload(payload)


def _dirs(run_dir: Path, *, create: bool = True) -> dict[str, Path]:
    result = {
        "run": run_dir,
        "configs": run_dir / "configs",
        "checkpoints": run_dir / "checkpoints",
        "artifacts": run_dir / "artifacts",
        "predictions": run_dir / "predictions",
        "metrics": run_dir / "metrics",
        "logs": run_dir / "logs",
    }
    if create:
        for value in result.values():
            value.mkdir(parents=True, exist_ok=True)
    return result


def _new_run(cfg: dict) -> dict[str, Path]:
    root = Path(cfg["experiment_root"]) / cfg["exp_id"]
    root.mkdir(parents=True, exist_ok=True)
    while True:
        run = root / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        if not run.exists():
            return _dirs(run)
        time.sleep(0.001)


def _latest_run(cfg: dict) -> dict[str, Path]:
    root = Path(cfg["experiment_root"]) / cfg["exp_id"]
    candidates = sorted(path for path in root.glob("*") if path.is_dir())
    if not candidates:
        raise FileNotFoundError(f"No fV-SRN run found under {root}")
    return _dirs(candidates[-1])


def _run_for_path(cfg: dict, path: str | Path | None) -> dict[str, Path]:
    if path is None:
        return _latest_run(cfg)
    resolved = Path(path).resolve()
    if resolved.parent.name in {"artifacts", "checkpoints"}:
        return _dirs(resolved.parent.parent)
    return _latest_run(cfg)


def _weighted_loss(prediction: torch.Tensor, target: torch.Tensor, training: dict) -> torch.Tensor:
    total = prediction.new_zeros(())
    if training["l1_weight"]:
        total = total + float(training["l1_weight"]) * F.l1_loss(prediction, target)
    if training["l2_weight"]:
        total = total + float(training["l2_weight"]) * F.mse_loss(prediction, target)
    return total


def _error_grids(
    model: TemporalFVSRN,
    volume: TemporalVolume,
    training: dict,
    *,
    device: torch.device,
    rng: np.random.Generator,
) -> list[np.ndarray]:
    size = int(training["rebuild_grid_size"])
    repeats = int(training["rebuild_samples_per_cell"])
    cells = np.stack(np.meshgrid(np.arange(size), np.arange(size), np.arange(size), indexing="ij"), axis=-1)
    cells = cells.reshape(-1, 3)[:, [2, 1, 0]]
    grids: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for t in range(volume.shape["T"]):
            errors = np.zeros(size**3, dtype=np.float32)
            for _ in range(repeats):
                coords = (cells + rng.random(cells.shape)) / size
                target = volume.sample(t, coords)
                outputs = []
                for start in range(0, len(coords), int(training["prediction_batch_size"])):
                    batch = torch.from_numpy(coords[start : start + int(training["prediction_batch_size"])].astype(np.float32)).to(device)
                    outputs.append(model(batch, t).cpu().numpy())
                errors += np.abs(np.concatenate(outputs)[:, 0] - target[:, 0])
            errors /= repeats
            maximum = float(errors.max())
            if maximum > 0:
                errors /= maximum
            grids.append(errors.reshape(size, size, size))
    return grids


def _validate_epoch(
    model: TemporalFVSRN,
    pool: SamplePool,
    *,
    device: torch.device,
    training: dict,
) -> float:
    total, count = 0.0, 0
    model.eval()
    with torch.no_grad():
        for t, indices in enumerate(pool.val_indices):
            for start in range(0, len(indices), int(training["batch_size"])):
                picked = indices[start : start + int(training["batch_size"])]
                coords = torch.from_numpy(pool.coords[t][picked]).to(device)
                targets = torch.from_numpy(pool.targets[t][picked]).to(device)
                total += float(_weighted_loss(model(coords, t), targets, training)) * len(picked)
                count += len(picked)
    return total / max(count, 1)


def _load_inference_model(
    cfg: dict,
    *,
    device: torch.device,
    artifact: str | Path | None,
    checkpoint: str | Path | None,
    dirs: dict[str, Path],
):
    if artifact is None and checkpoint is None:
        if cfg["evaluation"]["default_model"] == "compact":
            artifact = dirs["artifacts"] / f"{cfg['exp_id']}.pt"
        else:
            checkpoint = dirs["checkpoints"] / f"{cfg['exp_id']}.pth"
    if artifact is not None:
        model, payload = load_compact(artifact, device=device)
        if payload["target_name"] != cfg["data"]["target"] or payload["volume_shape"] != cfg["data"]["volume_shape"]:
            raise ValueError("Compact fV-SRN artifact does not match target or volume shape")
        return model, Path(artifact), payload
    payload = load_payload(checkpoint)
    validate(payload, cfg)
    model = TemporalFVSRN(cfg["model"]).to(device)
    model.load_state_dict(payload["model_state"])
    return model, Path(checkpoint), payload


def _predict(
    model,
    volume: TemporalVolume,
    *,
    device: torch.device,
    batch_size: int,
    output_path: Path,
) -> Path:
    prediction = open_memmap(
        output_path,
        mode="w+",
        dtype=np.float32,
        shape=(volume.shape["T"], volume.shape["Z"], volume.shape["Y"], volume.shape["X"]),
    )
    spatial_count = int(np.prod(volume.spatial_shape, dtype=np.int64))
    model.eval()
    with torch.no_grad():
        for t in range(volume.shape["T"]):
            flat = prediction[t].reshape(-1)
            for start in range(0, spatial_count, batch_size):
                stop = min(start + batch_size, spatial_count)
                coords = torch.from_numpy(volume.full_coords(start, stop)).to(device)
                flat[start:stop] = model(coords, t).detach().cpu().numpy()[:, 0]
            prediction.flush()
    return output_path


def _evaluate(volume: TemporalVolume, prediction_path: Path, model_path: Path) -> dict[str, Any]:
    prediction = np.load(prediction_path, mmap_mode="r")
    accumulator = PSNRAccumulator()
    squared_error = absolute_error = 0.0
    total_count = 0
    per_time = []
    for t in range(volume.shape["T"]):
        gt = np.asarray(volume.frame(t), dtype=np.float32)
        pred = np.asarray(prediction[t], dtype=np.float32)
        accumulator.update(gt, pred)
        diff = pred.astype(np.float64) - gt.astype(np.float64)
        squared_error += float(np.sum(diff * diff))
        absolute_error += float(np.sum(np.abs(diff)))
        total_count += int(diff.size)
        per_time.append({"t": t, "mse": mse(gt, pred), "mae": mae(gt, pred), "psnr": psnr(gt, pred)})
    model_bytes = int(model_path.stat().st_size)
    return {
        "target": volume.path,
        "per_time": per_time,
        "aggregate": {
            "mse": squared_error / total_count,
            "mae": absolute_error / total_count,
            "psnr": accumulator.compute(),
            "raw_target_bytes": int(volume.raw_bytes),
            "model_bytes": model_bytes,
            "cr": float(volume.raw_bytes / max(model_bytes, 1)),
        },
    }


def run_train(config_path: str | Path, *, target: str | None = None, resume: str | Path | None = None) -> dict:
    apply_runtime_thread_limits()
    cfg = load_config(config_path, target_override=target)
    config_hash = _hash(cfg)
    dirs = _new_run(cfg)
    setup_logging(log_dir=dirs["logs"], log_file="run.log")
    try:
        save_config(cfg, dirs["configs"] / "config.yaml")
        device = _device(cfg["training"]["device"])
        set_random_seed(int(cfg["training"]["seed"]))
        volume = TemporalVolume(cfg["data"]["target_path"], cfg["data"]["volume_shape"])
        model = TemporalFVSRN(cfg["model"]).to(device)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=float(cfg["training"]["lr"]),
            betas=(float(cfg["training"]["beta_1"]), float(cfg["training"]["beta_2"])),
        )
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=int(cfg["training"]["lr_step"]), gamma=float(cfg["training"]["lr_gamma"])
        )
        start_epoch = 0
        rng = np.random.default_rng(int(cfg["training"]["seed"]))
        if resume:
            payload = load_payload(resume)
            validate(payload, cfg, config_hash=config_hash)
            model.load_state_dict(payload["model_state"])
            optimizer.load_state_dict(payload["optimizer_state"])
            scheduler.load_state_dict(payload["scheduler_state"])
            pool = SamplePool.from_state_dict(payload["sample_pool"])
            start_epoch = int(payload["epoch"])
            restore_rng(payload)
            rng.bit_generator.state = payload["sampler_rng_state"]
        else:
            pool = build_sample_pool(
                volume,
                count_per_timestep=int(cfg["training"]["samples_per_timestep"]),
                validation_fraction=float(cfg["training"]["validation_fraction"]),
                floor=float(cfg["training"]["importance_floor"]),
                rng=rng,
            )
        logger.info("fV-SRN model statistics: %s", collect_model_statistics(model))
        from ..utils.exploration_probe import (
            ExplorationProbeRecorder,
            normalize_probe,
            probe_due,
            probe_progress,
            probe_temporal_volume_model,
        )

        probe_cfg = normalize_probe(cfg.get("exploration_probe"))
        probe_recorder = ExplorationProbeRecorder(dirs["metrics"], probe_cfg) if probe_cfg.enabled else None
        for epoch in range(start_epoch, int(cfg["training"]["epochs"])):
            if cfg["training"]["rebuild_every"] and (epoch + 1) % int(cfg["training"]["rebuild_every"]) == 0:
                errors = _error_grids(model, volume, cfg["training"], device=device, rng=rng)
                pool = build_sample_pool(
                    volume,
                    count_per_timestep=int(cfg["training"]["samples_per_timestep"]),
                    validation_fraction=float(cfg["training"]["validation_fraction"]),
                    floor=float(cfg["training"]["importance_floor"]),
                    rng=rng,
                    error_grids=errors,
                )
            model.train()
            train_total = train_count = 0
            for t in rng.permutation(volume.shape["T"]).tolist():
                indices = rng.permutation(pool.train_indices[t])
                for start in range(0, len(indices), int(cfg["training"]["batch_size"])):
                    picked = indices[start : start + int(cfg["training"]["batch_size"])]
                    coords = torch.from_numpy(pool.coords[t][picked]).to(device)
                    targets = torch.from_numpy(pool.targets[t][picked]).to(device)
                    optimizer.zero_grad(set_to_none=True)
                    loss = _weighted_loss(model(coords, t), targets, cfg["training"])
                    loss.backward()
                    optimizer.step()
                    train_total += float(loss.detach()) * len(picked)
                    train_count += len(picked)
            scheduler.step()
            val_loss = _validate_epoch(model, pool, device=device, training=cfg["training"])
            if (epoch + 1) % int(cfg["training"]["log_every"]) == 0:
                logger.info(
                    "epoch=%d/%d train_loss=%.7g val_loss=%.7g lr=%.7g",
                    epoch + 1, cfg["training"]["epochs"], train_total / max(train_count, 1),
                    val_loss, scheduler.get_last_lr()[0],
                )
            if cfg["training"]["save_every"] and (epoch + 1) % int(cfg["training"]["save_every"]) == 0:
                save_checkpoint(
                    dirs["checkpoints"] / f"epoch_{epoch + 1:04d}.pth",
                    model=model, optimizer=optimizer, scheduler=scheduler, epoch=epoch + 1,
                    cfg=cfg, config_hash=config_hash, sample_pool=pool,
                    sampler_rng_state=rng.bit_generator.state,
                )
            if probe_recorder is not None and probe_due(epoch + 1, int(cfg["training"]["epochs"]), probe_cfg):
                probe_started = time.perf_counter()
                probe_psnr, probe_count = probe_temporal_volume_model(
                    model=model,
                    volume=volume,
                    device=device,
                    batch_size=int(cfg["training"]["prediction_batch_size"]),
                    probe=probe_cfg,
                )
                probe_recorder.record(
                    progress=probe_progress(epoch + 1, int(cfg["training"]["epochs"]), probe_cfg),
                    scope=str(cfg["data"]["target"]),
                    aggregate_psnr=probe_psnr,
                    sample_count=probe_count,
                    elapsed_seconds=time.perf_counter() - probe_started,
                )
                model.train()
        checkpoint = save_checkpoint(
            dirs["checkpoints"] / f"{cfg['exp_id']}.pth",
            model=model, optimizer=optimizer, scheduler=scheduler, epoch=int(cfg["training"]["epochs"]),
            cfg=cfg, config_hash=config_hash, sample_pool=pool,
            sampler_rng_state=rng.bit_generator.state,
        )
        artifact, artifact_payload = export_compact(
            model=model, cfg=cfg, target_name=cfg["data"]["target"],
            volume_shape=cfg["data"]["volume_shape"],
            path=dirs["artifacts"] / f"{cfg['exp_id']}.pt", config_hash=config_hash,
        )
        summary = {
            "checkpoint_path": str(checkpoint),
            "artifact_path": str(artifact),
            "artifact_bytes": int(artifact_payload["artifact_bytes"]),
            "raw_target_bytes": int(volume.raw_bytes),
            "compact_cr": float(volume.raw_bytes / artifact_payload["artifact_bytes"]),
        }
        (dirs["metrics"] / "training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        if cfg["evaluation"]["run_after_training"]:
            evaluated = run_evaluate(config_path, target=target, artifact=artifact)
            summary.update(evaluated)
        return summary
    finally:
        close_file_handlers()


def run_predict(
    config_path: str | Path,
    *,
    target: str | None = None,
    artifact: str | Path | None = None,
    checkpoint: str | Path | None = None,
) -> dict:
    cfg = load_config(config_path, target_override=target)
    explicit = artifact or checkpoint
    dirs = _run_for_path(cfg, explicit)
    device = _device(cfg["training"]["device"])
    volume = TemporalVolume(cfg["data"]["target_path"], cfg["data"]["volume_shape"])
    model, model_path, _ = _load_inference_model(
        cfg, device=device, artifact=artifact, checkpoint=checkpoint, dirs=dirs
    )
    prediction_path = _predict(
        model, volume, device=device, batch_size=int(cfg["evaluation"]["batch_size"]),
        output_path=dirs["predictions"] / f"{cfg['exp_id']}.npy",
    )
    return {"prediction_path": str(prediction_path), "model_path": str(model_path)}


def run_evaluate(
    config_path: str | Path,
    *,
    target: str | None = None,
    artifact: str | Path | None = None,
    checkpoint: str | Path | None = None,
) -> dict:
    prediction_result = run_predict(
        config_path, target=target, artifact=artifact, checkpoint=checkpoint
    )
    cfg = load_config(config_path, target_override=target)
    dirs = _run_for_path(cfg, artifact or checkpoint or prediction_result["model_path"])
    volume = TemporalVolume(cfg["data"]["target_path"], cfg["data"]["volume_shape"])
    metrics = _evaluate(
        volume, Path(prediction_result["prediction_path"]), Path(prediction_result["model_path"])
    )
    metrics_path = save_metrics(dirs["metrics"] / f"{cfg['exp_id']}.json", metrics)
    return {**prediction_result, "metrics_path": str(metrics_path), "metrics": metrics}
