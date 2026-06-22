from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import torch
import yaml

from .config.io import load_experiment_config, save_experiment_config
from .data import build_dataset
from .evaluation.metrics import evaluate_predictions, save_metrics
from .models import build_model, materialize_model_config
from .training.engine import predict_dataset, save_predictions, train_model
from .utils.checkpoint import (
    read_checkpoint_payload,
    validate_checkpoint_target_order,
)
from .utils.io import sha256_payload
from .utils.logging_utils import close_file_handlers, setup_logging
from .utils.model_stats import (
    build_model_catalog_row,
    collect_model_statistics,
    format_fp16_size_megabytes,
    format_param_count,
    upsert_model_catalog,
)
from .utils.runtime import apply_runtime_thread_limits, set_random_seed

logger = logging.getLogger(__name__)


def _resolve_device(requested: str) -> torch.device:
    requested_norm = str(requested).strip().lower()
    if requested_norm.startswith("cuda") and not torch.cuda.is_available():
        logger.warning("CUDA requested but unavailable. Falling back to CPU.")
        return torch.device("cpu")
    return torch.device(requested)


def _prepare_run_dirs(config):
    run_dir = config.run_dir
    checkpoint_dir = run_dir / "checkpoints"
    prediction_dir = run_dir / "predictions"
    metrics_dir = run_dir / "metrics"
    logs_dir = run_dir / "logs"
    for path in (run_dir, checkpoint_dir, prediction_dir, metrics_dir, logs_dir):
        path.mkdir(parents=True, exist_ok=True)
    return {
        "run_dir": run_dir,
        "checkpoint_dir": checkpoint_dir,
        "prediction_dir": prediction_dir,
        "metrics_dir": metrics_dir,
        "logs_dir": logs_dir,
    }


def _build_effective_config_payload(config, dataset_meta) -> dict:
    payload = config.to_dict()
    payload["model"] = materialize_model_config(config.model, dataset_meta)
    return payload


def _log_effective_config(config, payload: dict) -> None:
    logger.info("Using config source: %s", config.source_config_path or "<memory>")
    logger.info("Effective config:\n%s", yaml.safe_dump(payload, sort_keys=False).rstrip())


def _prepare_runtime(config_path: str | Path):
    config_started_at = time.perf_counter()
    config = load_experiment_config(config_path)
    config_load_seconds = time.perf_counter() - config_started_at

    dirs_started_at = time.perf_counter()
    dirs = _prepare_run_dirs(config)
    run_dir_prepare_seconds = time.perf_counter() - dirs_started_at

    setup_logging(log_dir=dirs["logs_dir"])

    dataset_started_at = time.perf_counter()
    dataset = build_dataset(config.data, model_name=config.model.name)
    dataset_init_seconds = time.perf_counter() - dataset_started_at

    effective_payload = _build_effective_config_payload(config, dataset.meta)
    save_experiment_config(effective_payload, dirs["run_dir"] / "config.yaml")
    if config.log.effective_config:
        _log_effective_config(config, effective_payload)
    if config.log.startup_timing:
        logger.info("Config load: %.2fs", config_load_seconds)
        logger.info("Run dir prepare: %.2fs", run_dir_prepare_seconds)
        logger.info("Dataset init: %.2fs", dataset_init_seconds)
    device = _resolve_device(config.training.device)
    return config, dirs, dataset, device, effective_payload


def _predict_from_runtime(config, dirs, dataset, device: torch.device, checkpoint_path: str | Path | None = None) -> dict:
    model = build_model(config.model, dataset.meta).to(device)
    if checkpoint_path is None:
        checkpoint_path = dirs["checkpoint_dir"] / f"{config.exp_id}.pth"
    payload = read_checkpoint_payload(checkpoint_path)
    validate_checkpoint_target_order(payload, dataset.target_names())
    model.load_state_dict(payload["model_state"])
    predictions = predict_dataset(
        model,
        dataset,
        batch_size=config.evaluation.batch_size or config.training.pred_batch_size,
        device=device,
        hard_topk=True,
    )
    prediction_paths = save_predictions(dataset, predictions, dirs["prediction_dir"], config.exp_id)
    return {"checkpoint_path": checkpoint_path, "predictions": predictions, "prediction_paths": prediction_paths}


def run_train(config_path: str | Path) -> dict:
    apply_runtime_thread_limits()
    train_started_at = time.perf_counter()
    try:
        config, dirs, dataset, device, effective_payload = _prepare_runtime(config_path)
        set_random_seed(int(config.training.seed))
        model_started_at = time.perf_counter()
        model = build_model(config.model, dataset.meta)
        model_build_seconds = time.perf_counter() - model_started_at
        stats = collect_model_statistics(model)
        if config.log.model_stats:
            logger.info(
                "Model size: params=%s trainable=%s size(fp16, weights+bias)=%s",
                format_param_count(int(stats["param_count"])),
                format_param_count(int(stats["trainable_param_count"])),
                format_fp16_size_megabytes(int(stats["fp16_size_bytes"])),
            )
        if config.log.startup_timing:
            logger.info("Model build: %.2fs", model_build_seconds)
        catalog_row = build_model_catalog_row(
            model_name=config.model.name,
            model_params=config.model.params,
            stats=stats,
        )
        upsert_model_catalog(Path(config.experiment_root) / "model_size_catalog.csv", catalog_row)
        config_hash = sha256_payload(effective_payload)
        result = train_model(
            model=model,
            dataset=dataset,
            cfg=config.training,
            log_cfg=config.log,
            device=device,
            checkpoint_dir=dirs["checkpoint_dir"],
            config_hash=config_hash,
            prediction_dir=dirs["prediction_dir"],
            exp_id=config.exp_id,
        )
        if config.log.startup_timing:
            logger.info("Train total: %.2fs", time.perf_counter() - train_started_at)
        metrics = evaluate_predictions(dataset, result["predictions"], checkpoint_path=result["checkpoint_path"])
        save_metrics(dirs["metrics_dir"] / f"{config.exp_id}.json", metrics)
        return result
    finally:
        close_file_handlers()


def run_predict(config_path: str | Path, checkpoint_path: str | Path | None = None) -> dict:
    apply_runtime_thread_limits()
    try:
        config, dirs, dataset, device, _ = _prepare_runtime(config_path)
        result = _predict_from_runtime(config, dirs, dataset, device, checkpoint_path=checkpoint_path)
        return {"predictions": result["predictions"], "prediction_paths": result["prediction_paths"]}
    finally:
        close_file_handlers()


def run_evaluate(config_path: str | Path, checkpoint_path: str | Path | None = None) -> dict:
    apply_runtime_thread_limits()
    try:
        config, dirs, dataset, device, _ = _prepare_runtime(config_path)
        predict_result = _predict_from_runtime(config, dirs, dataset, device, checkpoint_path=checkpoint_path)
        metrics = evaluate_predictions(
            dataset,
            predict_result["predictions"],
            checkpoint_path=predict_result["checkpoint_path"],
        )
        metrics_path = save_metrics(dirs["metrics_dir"] / f"{config.exp_id}.json", metrics)
        return {"metrics": metrics, "metrics_path": metrics_path}
    finally:
        close_file_handlers()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="VarExpert-INR unified CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train a model")
    train_parser.add_argument("--config", required=True, help="Path to the experiment config")

    predict_parser = subparsers.add_parser("predict", help="Generate predictions from a checkpoint")
    predict_parser.add_argument("--config", required=True, help="Path to the experiment config")
    predict_parser.add_argument("--checkpoint", default=None, help="Optional explicit checkpoint path")

    eval_parser = subparsers.add_parser("evaluate", help="Evaluate a checkpoint")
    eval_parser.add_argument("--config", required=True, help="Path to the experiment config")
    eval_parser.add_argument("--checkpoint", default=None, help="Optional explicit checkpoint path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "train":
        result = run_train(args.config)
        logger.info("Training completed. Checkpoint: %s", result["checkpoint_path"])
        return
    if args.command == "predict":
        result = run_predict(args.config, checkpoint_path=args.checkpoint)
        logger.info("Predictions saved: %s", result["prediction_paths"])
        return
    if args.command == "evaluate":
        result = run_evaluate(args.config, checkpoint_path=args.checkpoint)
        logger.info("Metrics saved: %s", result["metrics_path"])
        return
    raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
