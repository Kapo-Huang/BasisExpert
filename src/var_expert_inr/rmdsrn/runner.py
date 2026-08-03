from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.lib.format import open_memmap

from ..evaluation.metrics import PSNRAccumulator, save_metrics
from ..utils.io import sha256_payload
from ..utils.logging_utils import close_file_handlers, setup_logging
from ..utils.model_stats import collect_model_statistics
from ..utils.runtime import apply_runtime_thread_limits, set_random_seed
from .checkpoint import (
    load_artifact,
    load_checkpoint,
    restore_training_random_state,
    save_artifact,
    save_checkpoint,
    validate_payload,
)
from .config import config_payload, load_config, save_config
from .data import TemporalFrameSampler, TemporalVolume, sample_voxel_batch
from .losses import exponential_variance_weight, rmdsrn_loss
from .model import RMDSRN

logger = logging.getLogger(__name__)


def _device(requested: str) -> torch.device:
    if str(requested).lower().startswith("cuda") and not torch.cuda.is_available():
        logger.warning("CUDA requested but unavailable; falling back to CPU")
        return torch.device("cpu")
    return torch.device(requested)


def _config_hash(cfg: dict[str, Any]) -> str:
    return sha256_payload(config_payload(cfg))


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
        for path in result.values():
            path.mkdir(parents=True, exist_ok=True)
    return result


def _new_run(cfg: dict[str, Any]) -> dict[str, Path]:
    root = Path(cfg["experiment_root"]) / cfg["exp_id"]
    root.mkdir(parents=True, exist_ok=True)
    while True:
        run = root / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        if not run.exists():
            return _dirs(run)
        time.sleep(0.001)


def _latest_run(cfg: dict[str, Any]) -> dict[str, Path]:
    root = Path(cfg["experiment_root"]) / cfg["exp_id"]
    candidates = sorted(path for path in root.glob("*") if path.is_dir())
    if not candidates:
        raise FileNotFoundError(f"No RMDSRN run found under {root}")
    return _dirs(candidates[-1])


def _run_for_path(cfg: dict[str, Any], path: str | Path | None) -> dict[str, Path]:
    if path is None:
        return _latest_run(cfg)
    resolved = Path(path).resolve()
    if resolved.parent.name in {"artifacts", "checkpoints"}:
        return _dirs(resolved.parent.parent)
    return _latest_run(cfg)


def _json_dump(path: str | Path, payload: dict[str, Any]) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
    return output


def _load_inference_model(
    cfg: dict[str, Any],
    *,
    device: torch.device,
    artifact: str | Path | None,
    checkpoint: str | Path | None,
    dirs: dict[str, Path],
) -> tuple[RMDSRN, Path, dict]:
    if artifact is None and checkpoint is None:
        if cfg["evaluation"]["default_model"] == "artifact":
            artifact = dirs["artifacts"] / f"{cfg['exp_id']}.pt"
        else:
            checkpoint = dirs["checkpoints"] / f"{cfg['exp_id']}.pth"
    if artifact is not None:
        model_path = Path(artifact)
        payload = load_artifact(model_path)
    else:
        model_path = Path(checkpoint)
        payload = load_checkpoint(model_path)
    validate_payload(payload, cfg, config_hash=_config_hash(cfg))
    model = RMDSRN(payload["model_config"]).to(device)
    model.load_state_dict(payload["model_state"], strict=True)
    model.eval()
    return model, model_path, payload


def _predict(
    model: RMDSRN,
    volume: TemporalVolume,
    *,
    device: torch.device,
    batch_size: int,
    mean_path: Path,
    variance_path: Path,
) -> tuple[Path, Path]:
    shape = (
        volume.shape["T"],
        volume.shape["Z"],
        volume.shape["Y"],
        volume.shape["X"],
    )
    mean_output = open_memmap(mean_path, mode="w+", dtype=np.float32, shape=shape)
    variance_output = open_memmap(variance_path, mode="w+", dtype=np.float32, shape=shape)
    spatial_count = int(np.prod(volume.spatial_shape, dtype=np.int64))
    model.eval()
    with torch.no_grad():
        for timestep in range(volume.shape["T"]):
            mean_flat = mean_output[timestep].reshape(-1)
            variance_flat = variance_output[timestep].reshape(-1)
            for start in range(0, spatial_count, int(batch_size)):
                stop = min(start + int(batch_size), spatial_count)
                coords = torch.from_numpy(volume.full_coords(start, stop)).to(
                    device, non_blocking=True
                )
                mean, variance = model(coords, timestep)
                mean_flat[start:stop] = mean.detach().cpu().numpy()[:, 0]
                variance_flat[start:stop] = variance.detach().cpu().numpy()[:, 0]
            mean_output.flush()
            variance_output.flush()
            logger.info("RMDSRN predicted timestep %d/%d", timestep + 1, volume.shape["T"])
    return mean_path, variance_path


@dataclass
class _PearsonAccumulator:
    count: int = 0
    sum_x: float = 0.0
    sum_y: float = 0.0
    sum_x2: float = 0.0
    sum_y2: float = 0.0
    sum_xy: float = 0.0

    def update(self, x: np.ndarray, y: np.ndarray) -> None:
        x64 = np.asarray(x, dtype=np.float64)
        y64 = np.asarray(y, dtype=np.float64)
        self.count += int(x64.size)
        self.sum_x += float(x64.sum())
        self.sum_y += float(y64.sum())
        self.sum_x2 += float(np.sum(x64 * x64))
        self.sum_y2 += float(np.sum(y64 * y64))
        self.sum_xy += float(np.sum(x64 * y64))

    def compute(self) -> float:
        if self.count <= 1:
            return 0.0
        count = float(self.count)
        covariance = count * self.sum_xy - self.sum_x * self.sum_y
        scale_x = count * self.sum_x2 - self.sum_x * self.sum_x
        scale_y = count * self.sum_y2 - self.sum_y * self.sum_y
        denominator = math.sqrt(max(scale_x, 0.0) * max(scale_y, 0.0))
        if denominator <= 0.0:
            return 0.0
        return float(covariance / denominator)


def _sample_indices(total_count: int, sample_size: int, rng: np.random.Generator) -> np.ndarray:
    count = min(int(total_count), int(sample_size))
    if count >= int(total_count):
        return np.arange(int(total_count), dtype=np.int64)
    if count > int(total_count) // 4:
        return rng.choice(int(total_count), size=count, replace=False, shuffle=False)
    selected = np.empty((0,), dtype=np.int64)
    while selected.size < count:
        needed = count - int(selected.size)
        candidates = rng.integers(
            0,
            int(total_count),
            size=max(needed + needed // 20, 1024),
            dtype=np.int64,
        )
        selected = np.unique(np.concatenate([selected, candidates]))
        if selected.size > count:
            selected = rng.choice(selected, size=count, replace=False, shuffle=False)
    return selected


def _topk_hit_rates(
    errors: np.ndarray,
    variances: np.ndarray,
    fractions: list[float],
) -> dict[str, float]:
    error_values = np.asarray(errors, dtype=np.float64).reshape(-1)
    variance_values = np.asarray(variances, dtype=np.float64).reshape(-1)
    result: dict[str, float] = {}
    for fraction in fractions:
        count = max(1, int(round(error_values.size * float(fraction))))
        error_indices = np.argpartition(error_values, -count)[-count:]
        variance_indices = np.argpartition(variance_values, -count)[-count:]
        error_mask = np.zeros(error_values.size, dtype=np.bool_)
        error_mask[error_indices] = True
        hits = int(np.count_nonzero(error_mask[variance_indices]))
        result[f"{float(fraction):g}"] = float(hits) / float(count)
    return result


def _evaluate(
    volume: TemporalVolume,
    *,
    mean_path: Path,
    variance_path: Path,
    model_path: Path,
    batch_size: int,
    sample_size: int,
    topk_fractions: list[float],
    seed: int,
) -> dict[str, Any]:
    mean_prediction = np.load(mean_path, mmap_mode="r")
    variance_prediction = np.load(variance_path, mmap_mode="r")
    expected_shape = (
        volume.shape["T"],
        volume.shape["Z"],
        volume.shape["Y"],
        volume.shape["X"],
    )
    if tuple(mean_prediction.shape) != expected_shape or tuple(variance_prediction.shape) != expected_shape:
        raise ValueError("RMDSRN prediction shapes do not match the configured volume")

    global_psnr = PSNRAccumulator()
    global_pearson = _PearsonAccumulator()
    absolute_error_sum = 0.0
    per_time: list[dict[str, Any]] = []
    spatial_count = int(np.prod(volume.spatial_shape, dtype=np.int64))
    for timestep in range(volume.shape["T"]):
        target = np.asarray(volume.frame(timestep)).reshape(-1)
        predicted = np.asarray(mean_prediction[timestep]).reshape(-1)
        predicted_variance = np.asarray(variance_prediction[timestep]).reshape(-1)
        frame_psnr = PSNRAccumulator()
        frame_pearson = _PearsonAccumulator()
        frame_abs = 0.0
        for start in range(0, spatial_count, int(batch_size)):
            stop = min(start + int(batch_size), spatial_count)
            target_chunk = target[start:stop]
            predicted_chunk = predicted[start:stop]
            difference = predicted_chunk.astype(np.float64) - target_chunk.astype(np.float64)
            squared_error = difference * difference
            frame_psnr.update(target_chunk, predicted_chunk)
            global_psnr.update(target_chunk, predicted_chunk)
            frame_abs += float(np.abs(difference).sum())
            absolute_error_sum += float(np.abs(difference).sum())
            frame_pearson.update(predicted_variance[start:stop], squared_error)
            global_pearson.update(predicted_variance[start:stop], squared_error)
        frame_mse = frame_psnr.total_squared_error / max(frame_psnr.total_count, 1)
        per_time.append(
            {
                "t": int(timestep),
                "mse": float(frame_mse),
                "mae": float(frame_abs) / max(frame_psnr.total_count, 1),
                "psnr": frame_psnr.compute(),
                "variance_error_pearson": frame_pearson.compute(),
            }
        )

    total_count = int(np.prod(expected_shape, dtype=np.int64))
    sampled_indices = _sample_indices(
        total_count,
        int(sample_size),
        np.random.default_rng(int(seed)),
    )
    target_flat = np.asarray(volume.array).reshape(-1)
    mean_flat = np.asarray(mean_prediction).reshape(-1)
    variance_flat = np.asarray(variance_prediction).reshape(-1)
    sampled_error = (
        mean_flat[sampled_indices].astype(np.float64)
        - target_flat[sampled_indices].astype(np.float64)
    ) ** 2
    topk = _topk_hit_rates(
        sampled_error,
        variance_flat[sampled_indices],
        topk_fractions,
    )
    model_bytes = int(model_path.stat().st_size)
    total_mse = global_psnr.total_squared_error / max(global_psnr.total_count, 1)
    return {
        "target": volume.path,
        "mean_prediction_path": str(mean_path),
        "variance_prediction_path": str(variance_path),
        "per_time": per_time,
        "aggregate": {
            "mse": float(total_mse),
            "mae": float(absolute_error_sum) / max(global_psnr.total_count, 1),
            "psnr": global_psnr.compute(),
            "variance_error_pearson": global_pearson.compute(),
            "topk_hit_rate": topk,
            "topk_sample_count": int(sampled_indices.size),
            "raw_target_bytes": int(volume.raw_bytes),
            "model_bytes": model_bytes,
            "cr": float(volume.raw_bytes) / max(model_bytes, 1),
        },
    }


def run_train(
    config_path: str | Path,
    *,
    target: str | None = None,
    resume: str | Path | None = None,
) -> dict[str, Any]:
    apply_runtime_thread_limits()
    cfg = load_config(config_path, target_override=target)
    config_hash = _config_hash(cfg)
    dirs = _new_run(cfg)
    setup_logging(log_dir=dirs["logs"], log_file="run.log")
    try:
        save_config(cfg, dirs["configs"] / "config.yaml")
        set_random_seed(int(cfg["training"]["seed"]))
        device = _device(cfg["training"]["device"])
        volume = TemporalVolume(cfg["data"]["target_path"], cfg["data"]["volume_shape"])
        model = RMDSRN(cfg["model"]).to(device)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=float(cfg["training"]["lr"]),
            betas=(
                float(cfg["training"]["beta_1"]),
                float(cfg["training"]["beta_2"]),
            ),
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=int(cfg["training"]["steps"]),
            eta_min=float(cfg["training"]["min_lr"]),
        )
        start_step = 0
        if resume is not None:
            payload = load_checkpoint(resume)
            validate_payload(payload, cfg, config_hash=config_hash)
            model.load_state_dict(payload["model_state"], strict=True)
            optimizer.load_state_dict(payload["optimizer_state"])
            scheduler.load_state_dict(payload["scheduler_state"])
            start_step = int(payload["step"])
            frame_sampler = restore_training_random_state(payload)
            if frame_sampler.time_count != volume.shape["T"]:
                raise ValueError("Checkpoint temporal sampler does not match the volume")
        else:
            frame_sampler = TemporalFrameSampler.create(
                volume.shape["T"],
                np.random.default_rng(int(cfg["training"]["seed"])),
            )
        if start_step >= int(cfg["training"]["steps"]):
            raise ValueError(
                f"Checkpoint step {start_step} has already reached configured training.steps"
            )
        if cfg["log"]["model_stats"]:
            logger.info("RMDSRN model statistics: %s", collect_model_statistics(model))

        final_losses = {"total": float("nan"), "member": float("nan"), "variance": float("nan")}
        from ..utils.exploration_probe import (
            ExplorationProbeRecorder,
            normalize_probe,
            probe_due,
            probe_progress,
            probe_temporal_volume_model,
        )

        probe_cfg = normalize_probe(cfg.get("exploration_probe"))
        probe_recorder = ExplorationProbeRecorder(dirs["metrics"], probe_cfg) if probe_cfg.enabled else None
        started_at = time.perf_counter()
        for step in range(start_step + 1, int(cfg["training"]["steps"]) + 1):
            timestep = frame_sampler.next()
            coords_np, targets_np = sample_voxel_batch(
                volume,
                timestep=timestep,
                count=int(cfg["training"]["batch_size"]),
                rng=frame_sampler.rng,
            )
            coords = torch.from_numpy(coords_np).to(device, non_blocking=True)
            targets = torch.from_numpy(targets_np).to(device, non_blocking=True)
            weight = exponential_variance_weight(
                step,
                int(cfg["training"]["steps"]),
                minimum=float(cfg["training"]["lambda_min"]),
                maximum=float(cfg["training"]["lambda_max"]),
                growth_rate=float(cfg["training"]["lambda_growth_rate"]),
            )
            model.train()
            optimizer.zero_grad(set_to_none=True)
            output = rmdsrn_loss(
                model.forward_members(coords, timestep),
                targets,
                variance_weight=weight,
                epsilon=float(cfg["training"]["epsilon"]),
            )
            output.total.backward()
            optimizer.step()
            scheduler.step()
            final_losses = {
                "total": float(output.total.detach().item()),
                "member": float(output.member.detach().item()),
                "variance": float(output.variance.detach().item()),
            }
            log_every = int(cfg["training"]["log_every"])
            if log_every > 0 and (step == 1 or step == int(cfg["training"]["steps"]) or step % log_every == 0):
                logger.info(
                    "step=%d/%d t=%d total=%.7g member=%.7g variance_kl=%.7g lambda=%.7g lr=%.7g",
                    step,
                    cfg["training"]["steps"],
                    timestep,
                    final_losses["total"],
                    final_losses["member"],
                    final_losses["variance"],
                    weight,
                    optimizer.param_groups[0]["lr"],
                )
            save_every = int(cfg["training"]["save_every"])
            if save_every > 0 and step % save_every == 0:
                save_checkpoint(
                    dirs["checkpoints"] / f"step_{step:06d}.pth",
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    step=step,
                    cfg=cfg,
                    config_hash=config_hash,
                    frame_sampler=frame_sampler,
                )
            if probe_recorder is not None and probe_due(step, int(cfg["training"]["steps"]), probe_cfg):
                probe_started = time.perf_counter()
                probe_psnr, probe_count = probe_temporal_volume_model(
                    model=model,
                    volume=volume,
                    device=device,
                    batch_size=int(cfg["evaluation"]["batch_size"]),
                    probe=probe_cfg,
                )
                probe_recorder.record(
                    progress=probe_progress(step, int(cfg["training"]["steps"]), probe_cfg),
                    scope=str(cfg["data"]["target"]),
                    aggregate_psnr=probe_psnr,
                    sample_count=probe_count,
                    elapsed_seconds=time.perf_counter() - probe_started,
                )
                model.train()

        checkpoint_path = save_checkpoint(
            dirs["checkpoints"] / f"{cfg['exp_id']}.pth",
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            step=int(cfg["training"]["steps"]),
            cfg=cfg,
            config_hash=config_hash,
            frame_sampler=frame_sampler,
        )
        artifact_path, artifact_payload = save_artifact(
            dirs["artifacts"] / f"{cfg['exp_id']}.pt",
            model=model,
            cfg=cfg,
            config_hash=config_hash,
        )
        summary: dict[str, Any] = {
            "checkpoint_path": str(checkpoint_path),
            "artifact_path": str(artifact_path),
            "artifact_bytes": int(artifact_payload["artifact_bytes"]),
            "raw_target_bytes": int(volume.raw_bytes),
            "cr": float(volume.raw_bytes) / max(int(artifact_payload["artifact_bytes"]), 1),
            "steps": int(cfg["training"]["steps"]),
            "final_losses": final_losses,
            "elapsed_seconds": float(time.perf_counter() - started_at),
        }
        _json_dump(dirs["metrics"] / "training_summary.json", summary)
        if cfg["evaluation"]["run_after_training"]:
            evaluated = run_evaluate(
                config_path,
                target=target,
                artifact=artifact_path,
            )
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
) -> dict[str, Any]:
    apply_runtime_thread_limits()
    cfg = load_config(config_path, target_override=target)
    explicit = artifact or checkpoint
    dirs = _run_for_path(cfg, explicit)
    device = _device(cfg["training"]["device"])
    volume = TemporalVolume(cfg["data"]["target_path"], cfg["data"]["volume_shape"])
    model, model_path, _ = _load_inference_model(
        cfg,
        device=device,
        artifact=artifact,
        checkpoint=checkpoint,
        dirs=dirs,
    )
    mean_path, variance_path = _predict(
        model,
        volume,
        device=device,
        batch_size=int(cfg["evaluation"]["batch_size"]),
        mean_path=dirs["predictions"] / f"{cfg['exp_id']}_mean.npy",
        variance_path=dirs["predictions"] / f"{cfg['exp_id']}_variance.npy",
    )
    return {
        "mean_prediction_path": str(mean_path),
        "variance_prediction_path": str(variance_path),
        "model_path": str(model_path),
    }


def run_evaluate(
    config_path: str | Path,
    *,
    target: str | None = None,
    artifact: str | Path | None = None,
    checkpoint: str | Path | None = None,
) -> dict[str, Any]:
    prediction = run_predict(
        config_path,
        target=target,
        artifact=artifact,
        checkpoint=checkpoint,
    )
    cfg = load_config(config_path, target_override=target)
    dirs = _run_for_path(cfg, artifact or checkpoint or prediction["model_path"])
    volume = TemporalVolume(cfg["data"]["target_path"], cfg["data"]["volume_shape"])
    metrics = _evaluate(
        volume,
        mean_path=Path(prediction["mean_prediction_path"]),
        variance_path=Path(prediction["variance_prediction_path"]),
        model_path=Path(prediction["model_path"]),
        batch_size=int(cfg["evaluation"]["batch_size"]),
        sample_size=int(cfg["evaluation"]["uncertainty_sample_size"]),
        topk_fractions=list(cfg["evaluation"]["topk_fractions"]),
        seed=int(cfg["evaluation"]["seed"]),
    )
    metrics_path = save_metrics(dirs["metrics"] / f"{cfg['exp_id']}.json", metrics)
    return {**prediction, "metrics_path": str(metrics_path), "metrics": metrics}
