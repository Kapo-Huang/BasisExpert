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

from ...evaluation.metrics import PSNRAccumulator, mae, mse, psnr, save_metrics
from ...evaluation.selection import parse_timestep_selection
from ...utils.io import sha256_payload
from ...utils.checkpoint import read_checkpoint_payload
from ...utils.logging_utils import close_file_handlers, setup_logging
from ...utils.model_stats import collect_model_statistics
from ...utils.runtime import apply_runtime_thread_limits, set_random_seed
from .config import load_config, save_config
from .data import SamplePool, TemporalVolume, build_sample_pool
from .model import TemporalFVSRN
from .quantization import load_inference_checkpoint, save_inference_checkpoint

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
    if resolved.parent.name == "checkpoints":
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
    checkpoint: str | Path | None,
    dirs: dict[str, Path],
):
    checkpoint = checkpoint or dirs["checkpoints"] / f"{cfg['exp_id']}.pth"
    payload = read_checkpoint_payload(checkpoint)
    checkpoint_format = payload.get("format")
    if checkpoint_format == "fv_srn_inference_v1":
        model, payload = load_inference_checkpoint(checkpoint, device=device)
        if payload["target_name"] != cfg["data"]["target"] or payload["volume_shape"] != cfg["data"]["volume_shape"]:
            raise ValueError("fV-SRN checkpoint does not match target or volume shape")
    elif checkpoint_format == "inference_checkpoint_v1":
        target_names = [str(value) for value in payload.get("target_names_order", [])]
        if target_names and target_names != [str(cfg["data"]["target"])]:
            raise ValueError(
                "fV-SRN checkpoint target does not match config: "
                f"checkpoint={target_names} config={[str(cfg['data']['target'])]}"
            )
        model = TemporalFVSRN(cfg["model"]).to(device)
        model.load_state_dict(payload["model_state"], strict=True)
    else:
        raise ValueError(f"Unsupported fV-SRN inference checkpoint: {checkpoint_format!r}")
    return model, Path(checkpoint), payload


def _predict(
    model,
    volume: TemporalVolume,
    *,
    device: torch.device,
    batch_size: int,
    output_path: Path,
    time_indices: tuple[int, ...] | None = None,
) -> Path:
    selected = time_indices or tuple(range(volume.shape["T"]))
    prediction = open_memmap(
        output_path,
        mode="w+",
        dtype=np.float32,
        shape=(len(selected), volume.shape["Z"], volume.shape["Y"], volume.shape["X"]),
    )
    spatial_count = int(np.prod(volume.spatial_shape, dtype=np.int64))
    model.eval()
    with torch.no_grad():
        for output_index, t in enumerate(selected):
            flat = prediction[output_index].reshape(-1)
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
    checkpoint_bytes = int(model_path.stat().st_size)
    return {
        "target": volume.path,
        "per_time": per_time,
        "aggregate": {
            "mse": squared_error / total_count,
            "mae": absolute_error / total_count,
            "psnr": accumulator.compute(),
            "raw_target_bytes": int(volume.raw_bytes),
            "checkpoint_bytes": checkpoint_bytes,
            "cr": float(volume.raw_bytes / max(checkpoint_bytes, 1)),
        },
    }


def run_train(config_path: str | Path, *, target: str | None = None) -> dict:
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
            eps=float(cfg["training"]["eps"]),
            weight_decay=float(cfg["training"]["weight_decay"]),
        )
        if cfg["training"]["lr_scheduler"] == "constant":
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _: 1.0)
        else:
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=int(cfg["training"]["lr_step"]),
                gamma=float(cfg["training"]["lr_gamma"]),
            )
        rng = np.random.default_rng(int(cfg["training"]["seed"]))
        pool = build_sample_pool(
            volume,
            count_per_timestep=int(cfg["training"]["samples_per_timestep"]),
            validation_fraction=float(cfg["training"]["validation_fraction"]),
            floor=float(cfg["training"]["importance_floor"]),
            rng=rng,
        )
        logger.info("fV-SRN model statistics: %s", collect_model_statistics(model))
        from ...utils.exploration_probe import (
            ExplorationProbeRecorder,
            normalize_probe,
            probe_due,
            probe_progress,
            probe_temporal_volume_model,
        )

        probe_cfg = normalize_probe(cfg.get("exploration_probe"))
        probe_recorder = ExplorationProbeRecorder(dirs["metrics"], probe_cfg) if probe_cfg.enabled else None
        for epoch in range(int(cfg["training"]["epochs"])):
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
                save_inference_checkpoint(
                    model=model, cfg=cfg, target_name=cfg["data"]["target"],
                    volume_shape=cfg["data"]["volume_shape"],
                    path=dirs["checkpoints"] / f"epoch_{epoch + 1:04d}.pth",
                    config_hash=config_hash,
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
        checkpoint, checkpoint_payload = save_inference_checkpoint(
            model=model, cfg=cfg, target_name=cfg["data"]["target"],
            volume_shape=cfg["data"]["volume_shape"],
            path=dirs["checkpoints"] / f"{cfg['exp_id']}.pth", config_hash=config_hash,
        )
        summary = {
            "checkpoint_path": str(checkpoint),
            "checkpoint_bytes": int(checkpoint_payload["checkpoint_bytes"]),
            "raw_target_bytes": int(volume.raw_bytes),
            "cr": float(volume.raw_bytes / checkpoint_payload["checkpoint_bytes"]),
        }
        (dirs["metrics"] / "training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        if cfg["evaluation"]["run_after_training"]:
            evaluated = run_evaluate(config_path, target=target, checkpoint=checkpoint)
            summary.update(evaluated)
        return summary
    finally:
        close_file_handlers()


def run_predict(
    config_path: str | Path,
    *,
    target: str | None = None,
    checkpoint: str | Path | None = None,
    time_indices: str | tuple[int, ...] | list[int] | None = None,
) -> dict:
    cfg = load_config(config_path, target_override=target)
    dirs = _run_for_path(cfg, checkpoint)
    device = _device(cfg["training"]["device"])
    volume = TemporalVolume(cfg["data"]["target_path"], cfg["data"]["volume_shape"])
    model, model_path, _ = _load_inference_model(
        cfg, device=device, checkpoint=checkpoint, dirs=dirs
    )
    selected = parse_timestep_selection(time_indices, volume.shape["T"])
    prediction_path = _predict(
        model, volume, device=device, batch_size=int(cfg["evaluation"]["batch_size"]),
        output_path=dirs["predictions"] / f"{cfg['exp_id']}.npy",
        time_indices=selected,
    )
    return {
        "prediction_path": str(prediction_path),
        "model_path": str(model_path),
        "decoded_timesteps": list(selected),
    }


def run_evaluate(
    config_path: str | Path,
    *,
    target: str | None = None,
    checkpoint: str | Path | None = None,
) -> dict:
    prediction_result = run_predict(
        config_path, target=target, checkpoint=checkpoint
    )
    cfg = load_config(config_path, target_override=target)
    dirs = _run_for_path(cfg, checkpoint or prediction_result["model_path"])
    volume = TemporalVolume(cfg["data"]["target_path"], cfg["data"]["volume_shape"])
    metrics = _evaluate(
        volume, Path(prediction_result["prediction_path"]), Path(prediction_result["model_path"])
    )
    metrics_path = save_metrics(dirs["metrics"] / f"{cfg['exp_id']}.json", metrics)
    return {**prediction_result, "metrics_path": str(metrics_path), "metrics": metrics}
