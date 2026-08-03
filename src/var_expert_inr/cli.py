from __future__ import annotations

import argparse
import logging
import re
import time
from datetime import datetime
from pathlib import Path

import torch
import yaml

from .config.io import load_experiment_config, save_experiment_config
from .data import build_dataset
from .evaluation.metrics import evaluate_predictions, save_metrics
from .models import build_model, effective_model_config, materialize_model_config
from .models.common import ModelAdapter
from .models.sota.compact_ngp import load_compact_ngp_artifact
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
TIMESTAMP_RUN_PATTERN = re.compile(r"^\d{8}_\d{6}_\d{6}$")


def _configured_method(config_path: str | Path) -> str:
    with Path(config_path).open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    model = payload.get("model") or payload.get("MODEL") or {}
    return str(model.get("name", "")).strip().lower().replace("-", "_")


def _resolve_device(requested: str) -> torch.device:
    requested_norm = str(requested).strip().lower()
    if requested_norm.startswith("cuda") and not torch.cuda.is_available():
        logger.warning("CUDA requested but unavailable. Falling back to CPU.")
        return torch.device("cpu")
    return torch.device(requested)


def _experiment_dir(config) -> Path:
    return Path(config.experiment_root) / config.exp_id


def _is_timestamped_run_dir(path: Path) -> bool:
    return path.is_dir() and bool(TIMESTAMP_RUN_PATTERN.fullmatch(path.name))


def _build_run_dirs(run_dir: Path) -> dict[str, Path | str]:
    checkpoint_dir = run_dir / "checkpoints"
    config_dir = run_dir / "configs"
    prediction_dir = run_dir / "predictions"
    metrics_dir = run_dir / "metrics"
    logs_dir = run_dir / "logs"
    artifact_dir = run_dir / "artifacts"
    return {
        "experiment_dir": run_dir.parent,
        "run_dir": run_dir,
        "run_token": run_dir.name,
        "checkpoint_dir": checkpoint_dir,
        "config_dir": config_dir,
        "prediction_dir": prediction_dir,
        "metrics_dir": metrics_dir,
        "logs_dir": logs_dir,
        "artifact_dir": artifact_dir,
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
        dirs["artifact_dir"],
    ):
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def _create_train_run_dirs(config):
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


def _resolve_artifact_run_dir(artifact_path: str | Path) -> Path | None:
    resolved = Path(artifact_path).resolve()
    if resolved.parent.name != "artifacts":
        return None
    run_dir = resolved.parent.parent
    if not _is_timestamped_run_dir(run_dir):
        raise FileNotFoundError(
            f"Artifact path must live under runs/<exp_id>/<timestamp>/artifacts: {resolved}"
        )
    return run_dir


def _resolve_latest_run_dir(config) -> Path:
    experiment_dir = _experiment_dir(config)
    candidates = []
    if experiment_dir.exists():
        candidates = sorted(path for path in experiment_dir.iterdir() if _is_timestamped_run_dir(path))
    if not candidates:
        raise FileNotFoundError(
            f"No timestamped run directory found for exp_id '{config.exp_id}' under '{experiment_dir}'."
        )
    return candidates[-1]


def _resolve_existing_run_dirs(
    config,
    checkpoint_path: str | Path | None = None,
    artifact_path: str | Path | None = None,
):
    if checkpoint_path is not None and artifact_path is not None:
        raise ValueError("checkpoint_path and artifact_path are mutually exclusive")
    run_dir = None
    if checkpoint_path is not None:
        run_dir = _resolve_checkpoint_run_dir(checkpoint_path)
    if artifact_path is not None:
        run_dir = _resolve_artifact_run_dir(artifact_path)
    if run_dir is None:
        run_dir = _resolve_latest_run_dir(config)
    return _ensure_run_dirs(run_dir)


def _build_effective_config_payload(config, dataset_meta) -> dict:
    payload = config.to_dict()
    payload["model"] = effective_model_config(config.model, dataset_meta)
    if dataset_meta.volume_shape is not None:
        payload["data"]["volume_shape"] = dataset_meta.volume_shape.to_dict()
    return payload


def _log_effective_config(config, payload: dict) -> None:
    logger.info("Using config source: %s", config.source_config_path or "<memory>")
    logger.info("Effective config:\n%s", yaml.safe_dump(payload, sort_keys=False).rstrip())


def _prepare_runtime(
    config_path: str | Path,
    *,
    create_run: bool,
    checkpoint_path: str | Path | None = None,
    artifact_path: str | Path | None = None,
):
    config_started_at = time.perf_counter()
    config = load_experiment_config(config_path)
    config_load_seconds = time.perf_counter() - config_started_at

    dirs_started_at = time.perf_counter()
    if create_run:
        dirs = _create_train_run_dirs(config)
    else:
        dirs = _resolve_existing_run_dirs(
            config,
            checkpoint_path=checkpoint_path,
            artifact_path=artifact_path,
        )
    run_dir_prepare_seconds = time.perf_counter() - dirs_started_at

    setup_logging(log_dir=dirs["logs_dir"], log_file=f"run_{dirs['run_token']}.log")

    dataset_started_at = time.perf_counter()
    dataset = build_dataset(config.data, model_name=config.model.name)
    dataset_init_seconds = time.perf_counter() - dataset_started_at

    effective_payload = _build_effective_config_payload(config, dataset.meta)
    if create_run:
        save_experiment_config(effective_payload, dirs["config_dir"] / "config.yaml")
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


def _predict_from_artifact(
    config,
    dirs,
    dataset,
    device: torch.device,
    artifact_path: str | Path,
) -> dict:
    model, payload = load_compact_ngp_artifact(artifact_path, device=device)
    expected_model_config = effective_model_config(config.model, dataset.meta)
    if payload.get("model_config") != expected_model_config:
        raise ValueError(
            "CompactNGP artifact model config mismatch. "
            f"artifact={payload.get('model_config')} current={expected_model_config}"
        )
    expected_targets = list(dataset.target_names())
    if list(payload.get("target_names_order", [])) != expected_targets:
        raise ValueError(
            "CompactNGP artifact target order mismatch. "
            f"artifact={payload.get('target_names_order')} current={expected_targets}"
        )
    expected_shape = (
        dataset.meta.volume_shape.to_dict()
        if dataset.meta.volume_shape is not None
        else None
    )
    if payload.get("volume_shape") != expected_shape:
        raise ValueError(
            "CompactNGP artifact volume shape mismatch. "
            f"artifact={payload.get('volume_shape')} current={expected_shape}"
        )
    predictions = predict_dataset(
        ModelAdapter(model),
        dataset,
        batch_size=config.evaluation.batch_size or config.training.pred_batch_size,
        device=device,
        hard_topk=True,
    )
    prediction_paths = save_predictions(
        dataset, predictions, dirs["prediction_dir"], config.exp_id
    )
    return {
        "artifact_path": Path(artifact_path),
        "predictions": predictions,
        "prediction_paths": prediction_paths,
    }


def run_train(config_path: str | Path, *, resume_path: str | Path | None = None) -> dict:
    if _configured_method(config_path) == "ecnr":
        from .ecnr.runner import run_train as run_ecnr_train

        return run_ecnr_train(config_path, resume=resume_path)
    if resume_path is not None:
        raise ValueError("--resume is currently supported by ECNR only in the unified CLI")
    apply_runtime_thread_limits()
    train_started_at = time.perf_counter()
    try:
        config, dirs, dataset, device, effective_payload = _prepare_runtime(config_path, create_run=True)
        set_random_seed(int(config.training.seed))
        model_started_at = time.perf_counter()
        model = build_model(config.model, dataset.meta)
        model_build_seconds = time.perf_counter() - model_started_at
        stats = collect_model_statistics(model)
        if config.log.model_stats:
            logger.info(
                "Model size: params=%s trainable=%s size(fp16, all parameters)=%s",
                format_param_count(int(stats["param_count"])),
                format_param_count(int(stats["trainable_param_count"])),
                format_fp16_size_megabytes(int(stats["fp16_size_bytes"])),
            )
        if config.log.startup_timing:
            logger.info("Model build: %.2fs", model_build_seconds)
        catalog_model_payload = {
            key: value
            for key, value in effective_payload["model"].items()
            if key != "name"
        }
        catalog_row = build_model_catalog_row(
            model_name=config.model.name,
            model_params=catalog_model_payload,
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
            predict_after_training=bool(config.evaluation.save_predictions),
            artifact_dir=dirs["artifact_dir"],
            model_config=effective_payload["model"],
            exploration_probe=config.exploration_probe,
        )
        if "artifact_path" in result:
            result.update(
                {
                    "training_param_count": int(stats["param_count"]),
                    "training_fp16_size_bytes": int(stats["fp16_size_bytes"]),
                }
            )
        if config.log.startup_timing:
            logger.info("Train total: %.2fs", time.perf_counter() - train_started_at)
        if "predictions" in result:
            metrics = evaluate_predictions(dataset, result["predictions"], checkpoint_path=result["checkpoint_path"])
            save_metrics(dirs["metrics_dir"] / f"{config.exp_id}.json", metrics)
        return result
    finally:
        close_file_handlers()


def run_predict(
    config_path: str | Path,
    checkpoint_path: str | Path | None = None,
    artifact_path: str | Path | None = None,
) -> dict:
    if _configured_method(config_path) == "ecnr":
        from .ecnr.runner import run_predict as run_ecnr_predict

        return run_ecnr_predict(
            config_path,
            checkpoint=checkpoint_path,
            artifact=artifact_path,
        )
    apply_runtime_thread_limits()
    try:
        config, dirs, dataset, device, _ = _prepare_runtime(
            config_path,
            create_run=False,
            checkpoint_path=checkpoint_path,
            artifact_path=artifact_path,
        )
        if artifact_path is not None:
            result = _predict_from_artifact(config, dirs, dataset, device, artifact_path)
        else:
            result = _predict_from_runtime(
                config, dirs, dataset, device, checkpoint_path=checkpoint_path
            )
        return {"predictions": result["predictions"], "prediction_paths": result["prediction_paths"]}
    finally:
        close_file_handlers()


def run_evaluate(
    config_path: str | Path,
    checkpoint_path: str | Path | None = None,
    artifact_path: str | Path | None = None,
) -> dict:
    if _configured_method(config_path) == "ecnr":
        from .ecnr.runner import run_evaluate as run_ecnr_evaluate

        return run_ecnr_evaluate(
            config_path,
            checkpoint=checkpoint_path,
            artifact=artifact_path,
        )
    apply_runtime_thread_limits()
    try:
        config, dirs, dataset, device, _ = _prepare_runtime(
            config_path,
            create_run=False,
            checkpoint_path=checkpoint_path,
            artifact_path=artifact_path,
        )
        if artifact_path is not None:
            predict_result = _predict_from_artifact(
                config, dirs, dataset, device, artifact_path
            )
            source_path = predict_result["artifact_path"]
        else:
            predict_result = _predict_from_runtime(
                config, dirs, dataset, device, checkpoint_path=checkpoint_path
            )
            source_path = predict_result["checkpoint_path"]
        metrics = evaluate_predictions(
            dataset,
            predict_result["predictions"],
            checkpoint_path=source_path,
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
    train_parser.add_argument("--resume", default=None, help="Optional ECNR scale checkpoint to resume")

    predict_parser = subparsers.add_parser("predict", help="Generate predictions from a checkpoint")
    predict_parser.add_argument("--config", required=True, help="Path to the experiment config")
    predict_source = predict_parser.add_mutually_exclusive_group()
    predict_source.add_argument("--checkpoint", default=None, help="Optional explicit checkpoint path")
    predict_source.add_argument("--artifact", default=None, help="Optional inference artifact (.ecnr for ECNR)")

    eval_parser = subparsers.add_parser("evaluate", help="Evaluate a run or checkpoint")
    eval_identity = eval_parser.add_mutually_exclusive_group(required=True)
    eval_identity.add_argument("--config", help="Path to the experiment config")
    eval_identity.add_argument("--run", help="Path to an existing run directory")
    eval_source = eval_parser.add_mutually_exclusive_group()
    eval_source.add_argument("--checkpoint", default=None, help="Optional explicit checkpoint path")
    eval_source.add_argument("--artifact", default=None, help="Optional inference artifact (.ecnr for ECNR)")
    eval_source.add_argument("--prediction", default=None, help="Optional prediction file or directory")
    eval_parser.add_argument("--source", choices=("auto", "checkpoint", "artifact", "prediction"), default=None)
    eval_parser.add_argument("--metrics", default=None, help="Comma-separated: psnr,ssim,lpips,decode_time,memory")
    eval_parser.add_argument("--timesteps", default=None, help="all, N, start:end[:step], or comma combinations")
    eval_parser.add_argument("--targets", default=None, help="all or comma-separated target names")
    eval_parser.add_argument("--render", action="store_true", help="Render selected prediction frames")
    eval_parser.add_argument("--eval-config", default=None, help="Optional render-profile YAML")
    eval_parser.add_argument("--overwrite", action="store_true", help="Bypass compatible cached evaluations")
    eval_parser.add_argument("--device", default=None, help="Optional evaluation device override")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "train":
        result = run_train(args.config, resume_path=args.resume)
        logger.info("Training completed. Checkpoint: %s", result["checkpoint_path"])
        return
    if args.command == "predict":
        result = run_predict(
            args.config,
            checkpoint_path=args.checkpoint,
            artifact_path=args.artifact,
        )
        logger.info(
            "Predictions saved: %s",
            result.get("prediction_paths", result.get("prediction_path")),
        )
        return
    if args.command == "evaluate":
        from .evaluation.service import evaluate_run

        if args.run:
            run_dir = Path(args.run)
        else:
            loaded = load_experiment_config(args.config)
            source_path = args.checkpoint or args.artifact
            if source_path:
                candidate = Path(source_path).resolve()
                run_dir = candidate.parent.parent if candidate.parent.name in {"checkpoints", "artifacts"} else candidate.parent
            else:
                run_dir = _resolve_latest_run_dir(loaded)
        result = evaluate_run(
            run_dir,
            metrics=args.metrics,
            timesteps=args.timesteps,
            targets=args.targets,
            source=args.source,
            checkpoint=args.checkpoint,
            artifact=args.artifact,
            prediction=args.prediction,
            render=args.render,
            render_profile=args.eval_config,
            overwrite=args.overwrite,
            device=args.device,
        )
        logger.info("Evaluation report saved: %s", result["metrics_path"])
        return
    raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
