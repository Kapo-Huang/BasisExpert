from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

EPS = 1e-12


class PSNRAccumulator:
    def __init__(self) -> None:
        self.total_squared_error = 0.0
        self.total_count = 0
        self.gt_min = float("inf")
        self.gt_max = float("-inf")

    def update(self, gt: np.ndarray, pred: np.ndarray) -> None:
        gt_f = np.asarray(gt, dtype=np.float64)
        pred_f = np.asarray(pred, dtype=np.float64)
        if gt_f.shape != pred_f.shape:
            raise ValueError(f"PSNRAccumulator shape mismatch: {gt_f.shape} vs {pred_f.shape}")
        self.total_squared_error += float(np.sum((gt_f - pred_f) ** 2))
        self.total_count += int(gt_f.size)
        if gt_f.size > 0:
            self.gt_min = min(self.gt_min, float(np.min(gt_f)))
            self.gt_max = max(self.gt_max, float(np.max(gt_f)))

    def compute(self) -> float:
        if self.total_count <= 0:
            return float("nan")
        data_range = float(self.gt_max - self.gt_min)
        if data_range <= 0:
            data_range = max(abs(float(self.gt_max)), abs(float(self.gt_min))) + EPS
        mse_val = float(self.total_squared_error) / float(self.total_count)
        if mse_val <= 0:
            return float("inf")
        return 10.0 * math.log10((data_range ** 2) / (mse_val + EPS))


class QualityAccumulator(PSNRAccumulator):
    """Streaming numeric quality accumulator used by selected evaluations."""

    def __init__(self) -> None:
        super().__init__()
        self.total_absolute_error = 0.0

    def update(self, gt: np.ndarray, pred: np.ndarray) -> None:
        gt_f = np.asarray(gt, dtype=np.float64)
        pred_f = np.asarray(pred, dtype=np.float64)
        if gt_f.shape != pred_f.shape:
            raise ValueError(f"QualityAccumulator shape mismatch: {gt_f.shape} vs {pred_f.shape}")
        self.total_absolute_error += float(np.sum(np.abs(gt_f - pred_f)))
        super().update(gt_f, pred_f)

    def as_dict(self) -> dict[str, float | int]:
        if self.total_count <= 0:
            return {"count": 0, "mse": float("nan"), "mae": float("nan"), "psnr": float("nan")}
        return {
            "count": int(self.total_count),
            "mse": float(self.total_squared_error / self.total_count),
            "mae": float(self.total_absolute_error / self.total_count),
            "psnr": float(self.compute()),
        }


def summarize_selected_quality(
    rows: list[dict[str, Any]],
    accumulators: dict[str, QualityAccumulator],
    targets: tuple[str, ...],
    metrics: tuple[str, ...],
) -> tuple[dict[str, dict[str, Any]], dict[str, float]]:
    per_target: dict[str, dict[str, Any]] = {}
    for target in targets:
        summary: dict[str, Any] = {}
        if "psnr" in metrics:
            summary.update(accumulators[target].as_dict())
        target_rows = [row for row in rows if row.get("target") == target]
        for metric in ("ssim", "lpips"):
            values = [float(row[metric]) for row in target_rows if row.get(metric) is not None]
            if metric in metrics and values:
                summary[metric] = float(np.mean(values))
        if summary:
            per_target[target] = summary
    aggregate: dict[str, float] = {}
    for metric in ("mse", "mae", "psnr", "ssim", "lpips"):
        values = [float(summary[metric]) for summary in per_target.values() if metric in summary]
        if values:
            aggregate[metric] = float(np.mean(values))
    return per_target, aggregate


def mse(gt: np.ndarray, pred: np.ndarray) -> float:
    return float(np.mean((np.asarray(gt) - np.asarray(pred)) ** 2))


def mae(gt: np.ndarray, pred: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(gt) - np.asarray(pred))))


def psnr(gt: np.ndarray, pred: np.ndarray) -> float:
    gt_f = np.asarray(gt, dtype=np.float64)
    pred_f = np.asarray(pred, dtype=np.float64)
    mse_val = float(np.mean((gt_f - pred_f) ** 2))
    data_range = float(np.max(gt_f) - np.min(gt_f))
    if data_range <= 0:
        data_range = float(np.max(np.abs(gt_f))) + EPS
    if mse_val <= 0:
        return float("inf")
    return 10.0 * math.log10((data_range ** 2) / (mse_val + EPS))


def evaluate_predictions(dataset, predictions: dict[str, np.ndarray], checkpoint_path: str | Path | None = None) -> dict[str, Any]:
    gt = dataset.load_targets_flat()
    results: dict[str, Any] = {"targets": {}, "aggregate": {}}
    psnrs = []
    mses = []
    maes = []
    for name in dataset.target_names():
        pred = np.asarray(predictions[name], dtype=np.float32)
        gt_arr = np.asarray(gt[name], dtype=np.float32)
        if pred.shape != gt_arr.shape:
            raise ValueError(f"Prediction shape mismatch for {name}: {pred.shape} vs {gt_arr.shape}")
        target_result = {
            "mse": mse(gt_arr, pred),
            "mae": mae(gt_arr, pred),
            "psnr": psnr(gt_arr, pred),
        }
        if dataset.meta.volume_shape is not None:
            reshaped_gt = dataset.reshape_flat_predictions(name, gt_arr)
            reshaped_pred = dataset.reshape_flat_predictions(name, pred)
            per_time = []
            for t in range(int(dataset.meta.volume_shape.T)):
                per_time.append(
                    {
                        "t": t,
                        "mse": mse(reshaped_gt[t], reshaped_pred[t]),
                        "mae": mae(reshaped_gt[t], reshaped_pred[t]),
                        "psnr": psnr(reshaped_gt[t], reshaped_pred[t]),
                    }
                )
            target_result["per_time"] = per_time
        results["targets"][name] = target_result
        psnrs.append(target_result["psnr"])
        mses.append(target_result["mse"])
        maes.append(target_result["mae"])
    results["aggregate"] = {
        "mse": float(np.mean(mses)),
        "mae": float(np.mean(maes)),
        "psnr": float(np.mean(psnrs)),
    }
    if checkpoint_path is not None:
        ckpt_size = Path(checkpoint_path).stat().st_size
        raw_bytes = sum(np.asarray(array).nbytes for array in gt.values())
        results["aggregate"]["cr"] = float(raw_bytes / ckpt_size) if ckpt_size > 0 else float("nan")
        results["aggregate"]["checkpoint_bytes"] = int(ckpt_size)
        results["aggregate"]["raw_target_bytes"] = int(raw_bytes)
    return results


def save_metrics(path: str | Path, payload: dict[str, Any]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
    return target
