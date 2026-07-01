from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

from ..config.schema import EvaluationConfig, LogConfig, PSNRLogConfig, TimingLogConfig, VolumeShape
from ..utils.io import dump_yaml, load_yaml, resolve_mapping_paths, resolve_path
from .data import BlockShape

TARGET_PLACEHOLDER = "{target}"
TOP_LEVEL_CONFIG_KEYS = {
    "experiment",
    "exp_id",
    "experiment_root",
    "data",
    "model",
    "partition",
    "compression",
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


def _replace_target_placeholder(value: Any, target: str | None) -> Any:
    if isinstance(value, dict):
        return {key: _replace_target_placeholder(item, target) for key, item in value.items()}
    if isinstance(value, list):
        return [_replace_target_placeholder(item, target) for item in value]
    if isinstance(value, str) and TARGET_PLACEHOLDER in value:
        if target is None:
            raise ValueError(f"Found '{TARGET_PLACEHOLDER}' placeholder without a selected target")
        return value.replace(TARGET_PLACEHOLDER, target)
    return value


def _parse_volume_shape(payload: dict[str, Any] | None) -> VolumeShape | None:
    if payload is None:
        return None
    return VolumeShape(
        X=int(payload["X"]),
        Y=int(payload["Y"]),
        Z=int(payload["Z"]),
        T=int(payload["T"]),
    )


def _resolve_paths(value: Any, *, base_dir: Path, parent_key: str | None = None) -> Any:
    if isinstance(value, dict):
        if parent_key == "targets":
            return {str(name): resolve_path(str(path), base_dir=base_dir) for name, path in value.items()}
        return {key: _resolve_paths(item, base_dir=base_dir, parent_key=key) for key, item in value.items()}
    if isinstance(value, list):
        return [_resolve_paths(item, base_dir=base_dir, parent_key=parent_key) for item in value]
    if isinstance(value, str) and parent_key in {"target_path"}:
        return resolve_path(value, base_dir=base_dir)
    return value


@dataclass(frozen=True)
class DCDataConfig:
    kind: str
    dataset_name: str | None = None
    split: str = "train"
    target: str | None = None
    target_path: str | None = None
    targets: dict[str, str] | None = None
    volume_shape: VolumeShape | None = None

    def __post_init__(self) -> None:
        normalized_kind = str(self.kind).strip().lower()
        if normalized_kind != "volume":
            raise ValueError(f"DC-INR only supports data.kind='volume', got {self.kind!r}")
        object.__setattr__(self, "kind", normalized_kind)
        if self.volume_shape is None:
            raise ValueError("DC-INR requires data.volume_shape")
        if self.target_path is None and not self.targets:
            raise ValueError("DC-INR requires data.target_path or data.targets")
        if self.target_path is not None and self.targets:
            raise ValueError("DC-INR must use either data.target_path or data.targets, not both")
        if self.target_path is not None and self.target is None:
            object.__setattr__(self, "target", "target")
        if self.targets:
            if self.target is None:
                raise ValueError("DC-INR data.targets requires data.target")
            if str(self.target) not in self.targets:
                available = ", ".join(sorted(self.targets.keys()))
                raise ValueError(f"Unknown data.target {self.target!r}. Available targets: {available}")
        if self.target is None:
            raise ValueError("DC-INR target name must be resolved")


@dataclass(frozen=True)
class DCModelConfig:
    name: str = "dc_inr"

    def __post_init__(self) -> None:
        normalized = str(self.name).strip().lower()
        if normalized != "dc_inr":
            raise ValueError(f"Unsupported DC-INR model.name: {self.name!r}")
        object.__setattr__(self, "name", normalized)


@dataclass(frozen=True)
class DCPartitionConfig:
    candidate_block_shapes: tuple[BlockShape, ...]
    dbscan_eps: float = 1.0e-2
    dbscan_min_samples: int = 1
    entropy_bins: int = 256
    distance_matrix_max_bytes: int = 536_870_912

    def __post_init__(self) -> None:
        if not self.candidate_block_shapes:
            raise ValueError("partition.candidate_block_shapes must be non-empty")
        voxel_counts = {int(shape.voxel_count) for shape in self.candidate_block_shapes}
        if len(voxel_counts) != 1:
            raise ValueError("All partition.candidate_block_shapes must contain the same voxel count")
        if float(self.dbscan_eps) <= 0.0:
            raise ValueError("partition.dbscan_eps must be positive")
        if int(self.dbscan_min_samples) <= 0:
            raise ValueError("partition.dbscan_min_samples must be positive")
        if int(self.entropy_bins) <= 1:
            raise ValueError("partition.entropy_bins must be greater than 1")
        if int(self.distance_matrix_max_bytes) <= 0:
            raise ValueError("partition.distance_matrix_max_bytes must be positive")


@dataclass(frozen=True)
class DCCompressionConfig:
    max_initial_neurons: int
    target_cr: float | None = None
    target_size_mib: float | None = None
    min_initial_neurons: int = 4

    def __post_init__(self) -> None:
        if (self.target_cr is None) == (self.target_size_mib is None):
            raise ValueError("compression requires exactly one of target_cr or target_size_mib")
        if self.target_cr is not None and float(self.target_cr) <= 0.0:
            raise ValueError("compression.target_cr must be positive")
        if self.target_size_mib is not None and float(self.target_size_mib) <= 0.0:
            raise ValueError("compression.target_size_mib must be positive")
        if int(self.max_initial_neurons) < 4:
            raise ValueError("compression.max_initial_neurons must be at least 4")
        if int(self.min_initial_neurons) < 4:
            raise ValueError("compression.min_initial_neurons must be at least 4")
        if int(self.max_initial_neurons) < int(self.min_initial_neurons):
            raise ValueError("compression.max_initial_neurons must be at least compression.min_initial_neurons")


@dataclass(frozen=True)
class DCTrainingConfig:
    epochs: int = 300
    total_steps: int = 0
    batch_size: int = 16_000
    lr: float = 1.0e-4
    beta_1: float = 0.9
    beta_2: float = 0.999
    points_per_timestep: int = 32_000
    prediction_batch_size: int = 65_536
    lr_milestones: tuple[int, ...] = (150, 225)
    lr_gamma: float = 0.5
    log_every: int = 10
    seed: int = 42
    device: str = "cuda"

    def __post_init__(self) -> None:
        if int(self.epochs) <= 0:
            raise ValueError("training.epochs must be positive")
        if int(self.total_steps) < 0:
            raise ValueError("training.total_steps must be non-negative")
        if int(self.batch_size) <= 0:
            raise ValueError("training.batch_size must be positive")
        if float(self.lr) <= 0.0:
            raise ValueError("training.lr must be positive")
        if not (0.0 < float(self.beta_1) < 1.0):
            raise ValueError("training.beta_1 must be in (0, 1)")
        if not (0.0 < float(self.beta_2) < 1.0):
            raise ValueError("training.beta_2 must be in (0, 1)")
        if int(self.points_per_timestep) <= 0:
            raise ValueError("training.points_per_timestep must be positive")
        if int(self.prediction_batch_size) <= 0:
            raise ValueError("training.prediction_batch_size must be positive")
        if float(self.lr_gamma) <= 0.0 or float(self.lr_gamma) >= 1.0:
            raise ValueError("training.lr_gamma must be in (0, 1)")
        if int(self.log_every) < 0:
            raise ValueError("training.log_every must be non-negative")
        if any(int(milestone) <= 0 for milestone in self.lr_milestones):
            raise ValueError("training.lr_milestones must contain positive integers")


@dataclass(frozen=True)
class DCExperimentConfig:
    experiment: str | None
    exp_id: str
    experiment_root: str
    data: DCDataConfig
    model: DCModelConfig
    partition: DCPartitionConfig
    compression: DCCompressionConfig
    training: DCTrainingConfig
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
        payload["partition"]["candidate_block_shapes"] = [shape.to_dict() for shape in self.partition.candidate_block_shapes]
        return payload


def _normalize_target_selection(cfg: dict[str, Any], *, target_override: str | None) -> dict[str, Any]:
    data_cfg = _ensure_mapping(cfg.get("data"), label="data")
    targets = data_cfg.get("targets")
    selected_target = str(target_override or data_cfg.get("target") or "").strip() or None
    if targets:
        if not isinstance(targets, dict) or not targets:
            raise ValueError("data.targets must be a non-empty mapping when provided")
        if selected_target is None:
            raise ValueError("data.targets requires data.target or --target")
        if selected_target not in targets:
            available = ", ".join(sorted(map(str, targets.keys())))
            raise ValueError(f"Unknown target '{selected_target}'. Available targets: {available}")
        data_cfg["target"] = selected_target
        data_cfg["target_path"] = targets[selected_target]
    elif data_cfg.get("target_path"):
        data_cfg["target"] = str(selected_target or data_cfg.get("target") or "target")
    else:
        raise ValueError("data must provide target_path or targets")
    cfg["data"] = data_cfg
    return cfg


def _parse_candidate_shapes(payload: Any) -> tuple[BlockShape, ...]:
    if not isinstance(payload, list) or not payload:
        raise ValueError("partition.candidate_block_shapes must be a non-empty list")
    shapes: list[BlockShape] = []
    for index, item in enumerate(payload):
        if not isinstance(item, dict):
            raise TypeError(f"partition.candidate_block_shapes[{index}] must be a mapping")
        shapes.append(
            BlockShape(
                sx=int(item["sx"]),
                sy=int(item["sy"]),
                sz=int(item["sz"]),
            )
        )
    return tuple(shapes)


def load_config(path: str | Path, *, target_override: str | None = None) -> DCExperimentConfig:
    config_path = Path(path).resolve()
    payload = load_yaml(config_path)
    _reject_unknown_keys(payload, TOP_LEVEL_CONFIG_KEYS, label="config")
    payload = _normalize_target_selection(payload, target_override=target_override)
    selected_target = payload["data"].get("target")
    payload = _replace_target_placeholder(payload, selected_target)
    payload = _resolve_paths(payload, base_dir=config_path.parent)

    data_section = _ensure_mapping(payload.get("data"), label="data")
    model_section = _ensure_mapping(payload.get("model"), label="model")
    partition_section = _ensure_mapping(payload.get("partition"), label="partition")
    compression_section = _ensure_mapping(payload.get("compression"), label="compression")
    training_section = _ensure_mapping(payload.get("training"), label="training")
    evaluation_section = _ensure_mapping(payload.get("evaluation"), label="evaluation")
    log_section = _ensure_mapping(payload.get("log"), label="log")

    _reject_unknown_keys(
        data_section,
        {"kind", "dataset_name", "split", "target", "target_path", "targets", "volume_shape"},
        label="data",
    )
    _reject_unknown_keys(model_section, _field_names(DCModelConfig), label="model")
    _reject_unknown_keys(
        partition_section,
        {"candidate_block_shapes", "dbscan_eps", "dbscan_min_samples", "entropy_bins", "distance_matrix_max_bytes"},
        label="partition",
    )
    _reject_unknown_keys(compression_section, _field_names(DCCompressionConfig), label="compression")
    _reject_unknown_keys(training_section, _field_names(DCTrainingConfig), label="training")
    _reject_unknown_keys(evaluation_section, _field_names(EvaluationConfig), label="evaluation")
    _reject_unknown_keys(log_section, _field_names(LogConfig), label="log")

    psnr_section = _ensure_mapping(log_section.get("psnr"), label="log.psnr")
    timing_section = _ensure_mapping(log_section.get("timing"), label="log.timing")
    _reject_unknown_keys(psnr_section, _field_names(PSNRLogConfig), label="log.psnr")
    _reject_unknown_keys(timing_section, _field_names(TimingLogConfig), label="log.timing")

    data_cfg = DCDataConfig(
        kind=str(data_section.get("kind", "volume")),
        dataset_name=data_section.get("dataset_name"),
        split=str(data_section.get("split", "train")),
        target=str(data_section.get("target")) if data_section.get("target") is not None else None,
        target_path=resolve_path(data_section.get("target_path"), base_dir=config_path.parent),
        targets=(
            None
            if data_section.get("target_path") is not None
            else resolve_mapping_paths(data_section.get("targets"), base_dir=config_path.parent)
        ),
        volume_shape=_parse_volume_shape(data_section.get("volume_shape")),
    )
    model_cfg = DCModelConfig(name=str(model_section.get("name", "dc_inr")))
    partition_cfg = DCPartitionConfig(
        candidate_block_shapes=_parse_candidate_shapes(partition_section.get("candidate_block_shapes")),
        dbscan_eps=float(partition_section.get("dbscan_eps", 1.0e-2)),
        dbscan_min_samples=int(partition_section.get("dbscan_min_samples", 1)),
        entropy_bins=int(partition_section.get("entropy_bins", 256)),
        distance_matrix_max_bytes=int(partition_section.get("distance_matrix_max_bytes", 536_870_912)),
    )
    if "max_initial_neurons" not in compression_section:
        raise ValueError("compression.max_initial_neurons is required")
    compression_cfg = DCCompressionConfig(
        max_initial_neurons=int(compression_section["max_initial_neurons"]),
        target_cr=(
            float(compression_section["target_cr"])
            if compression_section.get("target_cr") is not None
            else None
        ),
        target_size_mib=(
            float(compression_section["target_size_mib"])
            if compression_section.get("target_size_mib") is not None
            else None
        ),
        min_initial_neurons=int(compression_section.get("min_initial_neurons", 4)),
    )
    training_cfg = DCTrainingConfig(
        epochs=int(training_section.get("epochs", 300)),
        total_steps=int(training_section.get("total_steps", 0)),
        batch_size=int(training_section.get("batch_size", 16_000)),
        lr=float(training_section.get("lr", 1.0e-4)),
        beta_1=float(training_section.get("beta_1", 0.9)),
        beta_2=float(training_section.get("beta_2", 0.999)),
        points_per_timestep=int(training_section.get("points_per_timestep", 32_000)),
        prediction_batch_size=int(training_section.get("prediction_batch_size", 65_536)),
        lr_milestones=tuple(int(value) for value in training_section.get("lr_milestones", [150, 225])),
        lr_gamma=float(training_section.get("lr_gamma", 0.5)),
        log_every=int(training_section.get("log_every", 10)),
        seed=int(training_section.get("seed", 42)),
        device=str(training_section.get("device", "cuda")),
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
    exp_id = payload.get("exp_id") or f"dc-inr-{data_cfg.target}"
    return DCExperimentConfig(
        experiment=str(experiment) if experiment is not None else None,
        exp_id=str(exp_id),
        experiment_root=str(payload.get("experiment_root", "runs")),
        data=data_cfg,
        model=model_cfg,
        partition=partition_cfg,
        compression=compression_cfg,
        training=training_cfg,
        evaluation=evaluation_cfg,
        log=log_cfg,
        source_config_path=str(config_path),
    )


def config_payload(config: DCExperimentConfig) -> dict[str, Any]:
    return config.to_dict()


def save_config(config: DCExperimentConfig, path: str | Path) -> Path:
    return dump_yaml(path, config.to_dict())
