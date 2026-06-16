from __future__ import annotations

from dataclasses import fields
from pathlib import Path
from typing import Any

from .schema import (
    DataConfig,
    EvaluationConfig,
    ExperimentConfig,
    GradientBalancerConfig,
    LogConfig,
    ModelConfig,
    MultiAttrEMALossConfig,
    PSNRLogConfig,
    PretrainConfig,
    SchedulerConfig,
    TimingLogConfig,
    TrainingConfig,
    VolumeShape,
)
from ..utils.io import dump_yaml, load_yaml, resolve_mapping_paths, resolve_path

TOP_LEVEL_CONFIG_KEYS = {
    "experiment",
    "exp_id",
    "experiment_root",
    "data",
    "model",
    "training",
    "evaluation",
    "log",
}
TARGET_PLACEHOLDER = "{target}"


def _field_names(config_cls) -> set[str]:
    return {field.name for field in fields(config_cls)}


def _ensure_mapping(value: Any, *, label: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise TypeError(f"{label} must be a mapping, got {type(value).__name__}")
    return dict(value)


def _reject_unknown_keys(payload: dict[str, Any], allowed: set[str], *, label: str) -> None:
    unknown = sorted(set(payload).difference(allowed))
    if unknown:
        raise ValueError(f"Unknown {label} keys: {', '.join(unknown)}")


def _parse_volume_shape(payload: dict[str, Any] | None) -> VolumeShape | None:
    if payload is None:
        return None
    return VolumeShape(
        X=int(payload["X"]),
        Y=int(payload["Y"]),
        Z=int(payload["Z"]),
        T=int(payload["T"]),
    )


def _resolve_data_paths(data: dict[str, Any], *, base_dir: Path) -> dict[str, Any]:
    payload = dict(data)
    payload["coords_path"] = resolve_path(payload.get("coords_path"), base_dir=base_dir)
    payload["target_path"] = resolve_path(payload.get("target_path"), base_dir=base_dir)
    payload["target_dir"] = resolve_path(payload.get("target_dir"), base_dir=base_dir)
    payload["targets"] = resolve_mapping_paths(payload.get("targets"), base_dir=base_dir)
    return payload


def _resolve_target_placeholder(value: Any, *, target: str | None, field_name: str) -> str | None:
    if value is None:
        return None
    text = str(value)
    if TARGET_PLACEHOLDER not in text:
        return text
    if target is None:
        raise ValueError(f"{field_name} uses '{TARGET_PLACEHOLDER}' but data.target is not set")
    return text.replace(TARGET_PLACEHOLDER, target)


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    config_path = Path(path).resolve()
    payload = load_yaml(config_path)
    _reject_unknown_keys(payload, TOP_LEVEL_CONFIG_KEYS, label="config")

    data_section = _ensure_mapping(payload.get("data"), label="data")
    model_payload = _ensure_mapping(payload.get("model"), label="model")
    training_payload = _ensure_mapping(payload.get("training"), label="training")
    evaluation_payload = _ensure_mapping(payload.get("evaluation"), label="evaluation")
    log_payload = _ensure_mapping(payload.get("log"), label="log")

    _reject_unknown_keys(data_section, _field_names(DataConfig), label="data")
    _reject_unknown_keys(training_payload, _field_names(TrainingConfig), label="training")
    _reject_unknown_keys(evaluation_payload, _field_names(EvaluationConfig), label="evaluation")
    _reject_unknown_keys(log_payload, _field_names(LogConfig), label="log")
    data_payload = _resolve_data_paths(data_section, base_dir=config_path.parent)

    if "name" not in model_payload:
        raise ValueError("model.name is required")

    gradient_balancer_payload = _ensure_mapping(training_payload.get("gradient_balancer"), label="training.gradient_balancer")
    multiview_ema_payload = _ensure_mapping(
        training_payload.get("multiview_ema_loss"),
        label="training.multiview_ema_loss",
    )
    scheduler_payload = _ensure_mapping(training_payload.get("scheduler"), label="training.scheduler")
    pretrain_payload = _ensure_mapping(training_payload.get("pretrain"), label="training.pretrain")
    psnr_log_payload = _ensure_mapping(log_payload.get("psnr"), label="log.psnr")
    timing_log_payload = _ensure_mapping(log_payload.get("timing"), label="log.timing")

    _reject_unknown_keys(psnr_log_payload, _field_names(PSNRLogConfig), label="log.psnr")
    _reject_unknown_keys(timing_log_payload, _field_names(TimingLogConfig), label="log.timing")

    data_cfg = DataConfig(
        kind=data_payload["kind"],
        dataset_name=data_payload.get("dataset_name"),
        split=str(data_payload.get("split", "train")),
        coords_path=data_payload.get("coords_path"),
        target_path=data_payload.get("target_path"),
        targets=data_payload.get("targets"),
        target=data_payload.get("target"),
        target_dir=data_payload.get("target_dir"),
        volume_shape=_parse_volume_shape(data_payload.get("volume_shape")),
    )
    model_cfg = ModelConfig(name=str(model_payload.pop("name")), params=model_payload)
    training_cfg = TrainingConfig(
        epochs=int(training_payload.get("epochs", 100)),
        batch_size=int(training_payload.get("batch_size", 8192)),
        pred_batch_size=int(training_payload.get("pred_batch_size", training_payload.get("batch_size", 8192))),
        num_workers=int(training_payload.get("num_workers", 0)),
        lr=float(training_payload.get("lr", 5e-5)),
        weight_decay=float(training_payload.get("weight_decay", 0.0)),
        loss_type=str(training_payload.get("loss_type", "mse")),
        val_split=float(training_payload.get("val_split", 0.1)),
        log_every=int(training_payload.get("log_every", 10)),
        log_psnr_every=int(training_payload.get("log_psnr_every", 0)),
        psnr_sample_ratio=float(training_payload.get("psnr_sample_ratio", 1.0)),
        save_every=int(training_payload.get("save_every", 0)),
        early_stop_patience=int(training_payload.get("early_stop_patience", 0)),
        seed=int(training_payload.get("seed", 42)),
        device=str(training_payload.get("device", "cuda")),
        sampler=str(training_payload.get("sampler", "uniform_random")),
        batches_per_epoch_budget=int(training_payload.get("batches_per_epoch_budget", 0)),
        freeze_router_at=float(training_payload.get("freeze_router_at", 0.0)),
        hard_topk_warmup_epochs=int(training_payload.get("hard_topk_warmup_epochs", 0)),
        gradient_balancer=GradientBalancerConfig(**gradient_balancer_payload),
        multiview_ema_loss=MultiAttrEMALossConfig(**multiview_ema_payload),
        scheduler=SchedulerConfig(**scheduler_payload),
        pretrain=PretrainConfig(**pretrain_payload),
    )
    evaluation_cfg = EvaluationConfig(**evaluation_payload)
    log_cfg = LogConfig(
        effective_config=bool(log_payload.get("effective_config", True)),
        model_stats=bool(log_payload.get("model_stats", True)),
        epoch_summary=bool(log_payload.get("epoch_summary", True)),
        startup_timing=bool(log_payload.get("startup_timing", True)),
        psnr=PSNRLogConfig(**psnr_log_payload),
        timing=TimingLogConfig(**timing_log_payload),
    )
    experiment = _resolve_target_placeholder(payload.get("experiment"), target=data_cfg.target, field_name="experiment")
    exp_id = _resolve_target_placeholder(payload.get("exp_id"), target=data_cfg.target, field_name="exp_id")

    return ExperimentConfig(
        experiment=experiment,
        exp_id=str(exp_id or model_cfg.name.lower()),
        experiment_root=str(payload.get("experiment_root", "runs")),
        data=data_cfg,
        model=model_cfg,
        training=training_cfg,
        evaluation=evaluation_cfg,
        log=log_cfg,
        source_config_path=str(config_path),
    )


def save_experiment_config(config: ExperimentConfig, path: str | Path) -> Path:
    payload = config.to_dict() if hasattr(config, "to_dict") else dict(config)
    return dump_yaml(path, payload)
