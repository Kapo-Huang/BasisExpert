from __future__ import annotations

from dataclasses import asdict, is_dataclass
import json
import logging
import math
from pathlib import Path
import time
from typing import Any

import numpy as np

from ..config.schema import ExplorationProbeConfig


LOGGER = logging.getLogger(__name__)
HEADER = "progress\ttotal\tscope\taggregate_psnr\tsample_count\telapsed_seconds\tdetails\n"


def normalize_probe(value: Any) -> ExplorationProbeConfig:
    if isinstance(value, ExplorationProbeConfig):
        return value
    if is_dataclass(value):
        value = asdict(value)
    return ExplorationProbeConfig(**dict(value or {}))


def probe_interval(total_steps: int, probe: ExplorationProbeConfig) -> int:
    return max(
        1,
        int(round(int(total_steps) * int(probe.every_epoch_equivalents) / int(probe.total_epoch_equivalents))),
    )


def probe_progress(current_step: int, total_steps: int, probe: ExplorationProbeConfig) -> int:
    return min(
        int(probe.total_epoch_equivalents),
        max(1, int(round(int(current_step) * int(probe.total_epoch_equivalents) / max(int(total_steps), 1)))),
    )


def probe_due(current_step: int, total_steps: int, probe: ExplorationProbeConfig) -> bool:
    if not probe.enabled:
        return False
    interval = probe_interval(total_steps, probe)
    return int(current_step) == int(total_steps) or int(current_step) % interval == 0


def fixed_sample_indices(total_size: int, probe: ExplorationProbeConfig, *, salt: int = 0) -> np.ndarray:
    total = int(total_size)
    if total <= 0:
        return np.empty((0,), dtype=np.int64)
    requested = max(1, int(round(total * float(probe.sample_ratio))))
    count = min(total, requested, int(probe.max_samples))
    if count == total:
        return np.arange(total, dtype=np.int64)
    rng = np.random.default_rng(int(probe.seed) + int(salt))
    return np.sort(rng.choice(total, size=count, replace=False).astype(np.int64))


class ExplorationProbeRecorder:
    def __init__(self, metrics_dir: str | Path, probe: ExplorationProbeConfig):
        self.probe = normalize_probe(probe)
        self.path = Path(metrics_dir) / "exploration_psnr.tsv"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.write_text(HEADER, encoding="utf-8")

    def record(
        self,
        *,
        progress: int,
        scope: str,
        aggregate_psnr: float,
        sample_count: int,
        elapsed_seconds: float,
        details: dict[str, Any] | None = None,
    ) -> None:
        detail_text = json.dumps(details or {}, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        with self.path.open("a", encoding="utf-8", newline="") as handle:
            handle.write(
                f"{int(progress)}\t{int(self.probe.total_epoch_equivalents)}\t{scope}\t"
                f"{float(aggregate_psnr):.8g}\t{int(sample_count)}\t{float(elapsed_seconds):.6f}\t{detail_text}\n"
            )
        LOGGER.info(
            "Exploration PSNR progress=%d/%d scope=%s aggregate=%.4f samples=%d probe_time=%.2fs",
            int(progress),
            int(self.probe.total_epoch_equivalents),
            scope,
            float(aggregate_psnr),
            int(sample_count),
            float(elapsed_seconds),
        )


def psnr_from_arrays(target: np.ndarray, prediction: np.ndarray) -> float:
    target64 = np.asarray(target, dtype=np.float64).reshape(-1)
    prediction64 = np.asarray(prediction, dtype=np.float64).reshape(-1)
    if target64.size == 0 or prediction64.size != target64.size:
        return float("nan")
    mse = float(np.mean((prediction64 - target64) ** 2))
    data_range = float(np.max(target64) - np.min(target64))
    if not math.isfinite(data_range) or data_range <= 0.0:
        data_range = max(float(np.max(np.abs(target64))), 1.0e-12)
    return float("inf") if mse <= 0.0 else 10.0 * math.log10((data_range * data_range) / mse)


def timed_probe(function):
    started = time.perf_counter()
    value = function()
    return value, time.perf_counter() - started


def probe_temporal_volume_model(
    *,
    model,
    volume,
    device,
    batch_size: int,
    probe: ExplorationProbeConfig,
) -> tuple[float, int]:
    import torch

    spatial_count = int(np.prod(volume.spatial_shape, dtype=np.int64))
    indices = fixed_sample_indices(spatial_count * int(volume.shape["T"]), probe)
    targets: list[np.ndarray] = []
    predictions: list[np.ndarray] = []
    z_size, y_size, x_size = volume.spatial_shape
    model.eval()
    with torch.no_grad():
        for timestep in np.unique(indices // spatial_count).tolist():
            spatial = indices[indices // spatial_count == int(timestep)] % spatial_count
            target_flat = np.asarray(volume.frame(int(timestep))).reshape(-1)
            for start in range(0, int(spatial.size), int(batch_size)):
                rows = spatial[start : start + int(batch_size)]
                x = rows % x_size
                yz = rows // x_size
                y = yz % y_size
                z = yz // y_size
                coords = np.stack(
                    [
                        x / max(x_size - 1, 1),
                        y / max(y_size - 1, 1),
                        z / max(z_size - 1, 1),
                    ],
                    axis=1,
                ).astype(np.float32)
                output = model(torch.from_numpy(coords).to(device), int(timestep))
                if isinstance(output, tuple):
                    output = output[0]
                predictions.append(output.detach().cpu().numpy().reshape(-1))
                targets.append(np.asarray(target_flat[rows], dtype=np.float32).reshape(-1))
    return psnr_from_arrays(np.concatenate(targets), np.concatenate(predictions)), int(indices.size)


def probe_temporal_volume_ensemble_model(
    *,
    model,
    volume,
    device,
    batch_size: int,
    probe: ExplorationProbeConfig,
    variance_weight: float,
    epsilon: float,
    topk_fractions: list[float],
) -> tuple[float, int, dict[str, Any]]:
    """Probe an ensemble temporal model's reconstruction and uncertainty quality."""
    import torch

    spatial_count = int(np.prod(volume.spatial_shape, dtype=np.int64))
    indices = fixed_sample_indices(spatial_count * int(volume.shape["T"]), probe)
    targets: list[np.ndarray] = []
    members: list[np.ndarray] = []
    z_size, y_size, x_size = volume.spatial_shape
    model.eval()
    with torch.no_grad():
        for timestep in np.unique(indices // spatial_count).tolist():
            spatial = indices[indices // spatial_count == int(timestep)] % spatial_count
            target_flat = np.asarray(volume.frame(int(timestep))).reshape(-1)
            for start in range(0, int(spatial.size), int(batch_size)):
                rows = spatial[start : start + int(batch_size)]
                x = rows % x_size
                yz = rows // x_size
                y = yz % y_size
                z = yz // y_size
                coords = np.stack(
                    [
                        x / max(x_size - 1, 1),
                        y / max(y_size - 1, 1),
                        z / max(z_size - 1, 1),
                    ],
                    axis=1,
                ).astype(np.float32)
                output = model.forward_members(
                    torch.from_numpy(coords).to(device), int(timestep)
                )
                members.append(output.detach().cpu().numpy()[:, :, 0])
                targets.append(np.asarray(target_flat[rows], dtype=np.float32).reshape(-1))

    target = np.concatenate(targets).astype(np.float64, copy=False)
    member_values = np.concatenate(members, axis=0).astype(np.float64, copy=False)
    mean = member_values.mean(axis=1)
    variance = member_values.var(axis=1, ddof=1)
    squared_error = np.square(mean - target)
    member_mse = float(np.mean(np.square(member_values - target[:, None])))

    error_mass = squared_error + float(epsilon)
    variance_mass = variance + float(epsilon)
    error_density = error_mass / error_mass.sum()
    variance_density = variance_mass / variance_mass.sum()
    variance_kl = float(
        np.sum(error_density * (np.log(error_density) - np.log(variance_density)))
    )

    centered_error = squared_error - squared_error.mean()
    centered_variance = variance - variance.mean()
    denominator = float(
        np.sqrt(np.sum(centered_error**2) * np.sum(centered_variance**2))
    )
    pearson = (
        0.0
        if denominator <= 0.0
        or np.allclose(squared_error, squared_error[0], rtol=1.0e-6, atol=1.0e-12)
        or np.allclose(variance, variance[0], rtol=1.0e-6, atol=1.0e-12)
        else float(np.sum(centered_error * centered_variance) / denominator)
    )

    topk: dict[str, float] = {}
    for fraction in topk_fractions:
        count = max(1, int(round(target.size * float(fraction))))
        error_indices = np.argpartition(squared_error, -count)[-count:]
        variance_indices = np.argpartition(variance, -count)[-count:]
        error_mask = np.zeros(target.size, dtype=np.bool_)
        error_mask[error_indices] = True
        topk[f"{float(fraction):g}"] = float(
            np.count_nonzero(error_mask[variance_indices])
        ) / float(count)

    weighted_variance = float(variance_weight) * variance_kl
    details = {
        "member_mse": member_mse,
        "variance_kl": variance_kl,
        "variance_weight": float(variance_weight),
        "weighted_variance_loss": weighted_variance,
        "weighted_variance_to_member_ratio": weighted_variance
        / max(member_mse, float(epsilon)),
        "variance_error_pearson": pearson,
        "topk_hit_rate": topk,
    }
    return psnr_from_arrays(target, mean), int(indices.size), details
