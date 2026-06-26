from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

from ..config.schema import (
    DataConfig,
    EvaluationConfig,
    LogConfig,
    PSNRLogConfig,
    SchedulerConfig,
    TimingLogConfig,
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


def _field_names(config_cls) -> set[str]:
    return {entry.name for entry in fields(config_cls)}


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


def _normalize_loss_type(loss_type: str) -> str:
    normalized = str(loss_type).strip().lower()
    if normalized not in {"mse", "l1"}:
        raise ValueError(f"Unsupported loss_type: {loss_type!r}")
    return normalized


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
    payload["targets"] = resolve_mapping_paths(payload.get("targets"), base_dir=base_dir)
    return payload


@dataclass(frozen=True)
class MCModelConfig:
    name: str
    hidden_features: int = 64
    gfe_layers: int = 5
    lfe_layers: int = 6

    def __post_init__(self) -> None:
        normalized_name = str(self.name).strip().lower()
        if normalized_name != "mc_inr":
            raise ValueError(f"Unsupported MC-INR model.name: {self.name!r}")
        object.__setattr__(self, "name", normalized_name)


@dataclass(frozen=True)
class MCTrainingConfig:
    epochs: int = 200
    batch_size: int = 65536
    pred_batch_size: int = 65536
    num_workers: int = 0
    lr: float = 5.0e-5
    weight_decay: float = 0.0
    loss_type: str = "mse"
    log_every: int = 10
    save_every: int = 0
    seed: int = 42
    device: str = "cuda"
    resume_path: str | None = None
    initial_k: int = 20
    cluster_init_method: str = "auto"
    assignments_cache_path: str = ""
    meta_sampling_ratio: float | None = None
    meta_iterations: int = 2000
    meta_inner_steps: int = 5
    meta_inner_lr: float = 1.0e-4
    meta_batch_clusters: int = 4
    meta_support_ratio: float = 0.3
    meta_outer_lr: float = 1.0e-3
    convergence_patience: int = 30
    convergence_delta: float = 0.0
    finetune_epochs: int = 200
    finetune_lr: float | None = None
    finetune_sampling_ratio: float = 1.0
    recluster_after_finetune: bool = False
    split_threshold: float = 5.0e-4
    min_split_points: int = 32
    max_recluster_rounds: int = 3
    cluster_aware_batches: bool = True
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)

    def __post_init__(self) -> None:
        object.__setattr__(self, "loss_type", _normalize_loss_type(self.loss_type))
        object.__setattr__(self, "cluster_init_method", str(self.cluster_init_method).strip().lower())
        if int(self.epochs) <= 0:
            raise ValueError("training.epochs must be positive")
        if int(self.batch_size) <= 0:
            raise ValueError("training.batch_size must be positive")
        if int(self.pred_batch_size) <= 0:
            raise ValueError("training.pred_batch_size must be positive")
        if int(self.initial_k) <= 0:
            raise ValueError("training.initial_k must be positive")
        if int(self.meta_iterations) <= 0:
            raise ValueError("training.meta_iterations must be positive")
        if int(self.meta_inner_steps) <= 0:
            raise ValueError("training.meta_inner_steps must be positive")
        if int(self.meta_batch_clusters) <= 0:
            raise ValueError("training.meta_batch_clusters must be positive")
        if float(self.meta_inner_lr) <= 0.0:
            raise ValueError("training.meta_inner_lr must be positive")
        if float(self.meta_outer_lr) <= 0.0:
            raise ValueError("training.meta_outer_lr must be positive")
        if not (0.0 < float(self.meta_support_ratio) <= 1.0):
            raise ValueError("training.meta_support_ratio must be in (0, 1]")
        if self.meta_sampling_ratio is not None and not (0.0 < float(self.meta_sampling_ratio) <= 1.0):
            raise ValueError("training.meta_sampling_ratio must be in (0, 1] when provided")
        if int(self.finetune_epochs) <= 0:
            raise ValueError("training.finetune_epochs must be positive")
        if float(self.finetune_sampling_ratio) <= 0.0:
            raise ValueError("training.finetune_sampling_ratio must be positive")
        if float(self.split_threshold) < 0.0:
            raise ValueError("training.split_threshold must be non-negative")
        if int(self.min_split_points) < 2:
            raise ValueError("training.min_split_points must be at least 2")
        if int(self.max_recluster_rounds) < 0:
            raise ValueError("training.max_recluster_rounds must be non-negative")


@dataclass(frozen=True)
class MCExperimentConfig:
    experiment: str | None
    exp_id: str
    experiment_root: str
    data: DataConfig
    model: MCModelConfig
    training: MCTrainingConfig
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    log: LogConfig = field(default_factory=LogConfig)
    source_config_path: str | None = None

    @property
    def run_dir(self) -> Path:
        return Path(self.experiment_root) / self.exp_id

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["data"]["volume_shape"] = (
            self.data.volume_shape.to_dict() if self.data.volume_shape is not None else None
        )
        return payload


def load_config(path: str | Path) -> MCExperimentConfig:
    config_path = Path(path).resolve()
    payload = load_yaml(config_path)
    _reject_unknown_keys(payload, TOP_LEVEL_CONFIG_KEYS, label="config")

    data_section = _ensure_mapping(payload.get("data"), label="data")
    model_section = _ensure_mapping(payload.get("model"), label="model")
    training_section = _ensure_mapping(payload.get("training"), label="training")
    evaluation_section = _ensure_mapping(payload.get("evaluation"), label="evaluation")
    log_section = _ensure_mapping(payload.get("log"), label="log")

    _reject_unknown_keys(data_section, _field_names(DataConfig), label="data")
    _reject_unknown_keys(
        model_section,
        {"name", "hidden_features", "gfe_layers", "lfe_layers"},
        label="model",
    )
    _reject_unknown_keys(training_section, _field_names(MCTrainingConfig), label="training")
    _reject_unknown_keys(evaluation_section, _field_names(EvaluationConfig), label="evaluation")
    _reject_unknown_keys(log_section, _field_names(LogConfig), label="log")

    scheduler_section = _ensure_mapping(training_section.get("scheduler"), label="training.scheduler")
    _reject_unknown_keys(scheduler_section, _field_names(SchedulerConfig), label="training.scheduler")
    psnr_section = _ensure_mapping(log_section.get("psnr"), label="log.psnr")
    timing_section = _ensure_mapping(log_section.get("timing"), label="log.timing")
    _reject_unknown_keys(psnr_section, _field_names(PSNRLogConfig), label="log.psnr")
    _reject_unknown_keys(timing_section, _field_names(TimingLogConfig), label="log.timing")

    data_payload = _resolve_data_paths(data_section, base_dir=config_path.parent)
    data_cfg = DataConfig(
        kind=data_payload["kind"],
        dataset_name=data_payload.get("dataset_name"),
        split=str(data_payload.get("split", "train")),
        coords_path=data_payload.get("coords_path"),
        target_path=data_payload.get("target_path"),
        targets=data_payload.get("targets"),
        target=data_payload.get("target"),
        volume_shape=_parse_volume_shape(data_payload.get("volume_shape")),
    )
    if data_cfg.target is not None:
        raise ValueError("mc_inr does not support data.target; use target_path or targets")

    model_cfg = MCModelConfig(
        name=str(model_section.get("name", "mc_inr")),
        hidden_features=int(model_section.get("hidden_features", 64)),
        gfe_layers=int(model_section.get("gfe_layers", 5)),
        lfe_layers=int(model_section.get("lfe_layers", 6)),
    )
    legacy_meta_sampling_ratio = training_section.get("meta_sampling_ratio")
    meta_support_ratio = training_section.get("meta_support_ratio", legacy_meta_sampling_ratio)
    meta_iterations = training_section.get("meta_iterations")
    if meta_iterations is None:
        meta_iterations = training_section.get("epochs", 2000)
    meta_outer_lr = training_section.get("meta_outer_lr")
    if meta_outer_lr is None:
        meta_outer_lr = training_section.get("lr", 5.0e-5)

    training_cfg = MCTrainingConfig(
        epochs=int(training_section.get("epochs", 200)),
        batch_size=int(training_section.get("batch_size", 65536)),
        pred_batch_size=int(training_section.get("pred_batch_size", training_section.get("batch_size", 65536))),
        num_workers=int(training_section.get("num_workers", 0)),
        lr=float(training_section.get("lr", 5.0e-5)),
        weight_decay=float(training_section.get("weight_decay", 0.0)),
        loss_type=str(training_section.get("loss_type", "mse")),
        log_every=int(training_section.get("log_every", 10)),
        save_every=int(training_section.get("save_every", 0)),
        seed=int(training_section.get("seed", 42)),
        device=str(training_section.get("device", "cuda")),
        resume_path=resolve_path(training_section.get("resume_path"), base_dir=config_path.parent),
        initial_k=int(training_section.get("initial_k", 20)),
        cluster_init_method=str(training_section.get("cluster_init_method", "auto")),
        assignments_cache_path=str(
            resolve_path(training_section.get("assignments_cache_path"), base_dir=config_path.parent) or ""
        ),
        meta_sampling_ratio=(
            float(legacy_meta_sampling_ratio) if legacy_meta_sampling_ratio is not None else None
        ),
        meta_iterations=int(meta_iterations),
        meta_inner_steps=int(training_section.get("meta_inner_steps", 5)),
        meta_inner_lr=float(training_section.get("meta_inner_lr", 1.0e-4)),
        meta_batch_clusters=int(training_section.get("meta_batch_clusters", 4)),
        meta_support_ratio=float(meta_support_ratio if meta_support_ratio is not None else 0.3),
        meta_outer_lr=float(meta_outer_lr),
        convergence_patience=int(training_section.get("convergence_patience", 30)),
        convergence_delta=float(training_section.get("convergence_delta", 0.0)),
        finetune_epochs=int(training_section.get("finetune_epochs", 200)),
        finetune_lr=(
            float(training_section["finetune_lr"])
            if training_section.get("finetune_lr") is not None
            else None
        ),
        finetune_sampling_ratio=float(training_section.get("finetune_sampling_ratio", 1.0)),
        recluster_after_finetune=bool(training_section.get("recluster_after_finetune", False)),
        split_threshold=float(training_section.get("split_threshold", 5.0e-4)),
        min_split_points=int(training_section.get("min_split_points", 32)),
        max_recluster_rounds=int(training_section.get("max_recluster_rounds", 3)),
        cluster_aware_batches=bool(training_section.get("cluster_aware_batches", True)),
        scheduler=SchedulerConfig(**scheduler_section),
    )
    evaluation_cfg = EvaluationConfig(**evaluation_section)
    log_cfg = LogConfig(
        effective_config=bool(log_section.get("effective_config", True)),
        model_stats=bool(log_section.get("model_stats", True)),
        epoch_summary=bool(log_section.get("epoch_summary", True)),
        startup_timing=bool(log_section.get("startup_timing", True)),
        psnr=PSNRLogConfig(**psnr_section),
        timing=TimingLogConfig(**timing_section),
    )
    experiment = payload.get("experiment")
    exp_id = payload.get("exp_id") or model_cfg.name
    return MCExperimentConfig(
        experiment=str(experiment) if experiment is not None else None,
        exp_id=str(exp_id),
        experiment_root=str(payload.get("experiment_root", "runs")),
        data=data_cfg,
        model=model_cfg,
        training=training_cfg,
        evaluation=evaluation_cfg,
        log=log_cfg,
        source_config_path=str(config_path),
    )


def save_config(config: MCExperimentConfig, path: str | Path) -> Path:
    return dump_yaml(path, config.to_dict())
