from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

EPS = 1e-12


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
