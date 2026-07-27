from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


def _normalize_loss_type(loss_type: str) -> str:
    normalized = str(loss_type).strip().lower()
    if normalized not in {"mse", "l1"}:
        raise ValueError(f"Unsupported loss_type: {loss_type!r}")
    return normalized


def _normalize_data_kind(kind: str) -> str:
    normalized = str(kind).strip().lower()
    if normalized not in {"node", "volume"}:
        raise ValueError(f"Unsupported data.kind: {kind!r}")
    return normalized


def _normalize_sampler(name: str) -> str:
    normalized = str(name).strip().lower()
    if normalized not in {"uniform_random", "budgeted_random", "time_stratified"}:
        raise ValueError(f"Unsupported sampler: {name!r}")
    return normalized


def _normalize_scheduler_interval(interval: str) -> str:
    normalized = str(interval).strip().lower()
    if normalized not in {"epoch", "optimizer_step"}:
        raise ValueError(
            f"Unsupported scheduler.interval: {interval!r}"
        )
    return normalized


@dataclass(frozen=True)
class VolumeShape:
    X: int
    Y: int
    Z: int
    T: int

    @property
    def N(self) -> int:
        return int(self.X) * int(self.Y) * int(self.Z) * int(self.T)

    def to_dict(self) -> dict[str, int]:
        return {"X": int(self.X), "Y": int(self.Y), "Z": int(self.Z), "T": int(self.T)}


@dataclass(frozen=True)
class GradientBalancerConfig:
    enabled: bool = False
    method: str = "pcgrad"
    cagrad_c: float = 0.4
    solver_max_iter: int = 50
    solver_lr: float = 0.25
    gradnorm_alpha: float = 0.5
    gradnorm_lr: float = 1e-3
    gradnorm_every_n_steps: int = 100


@dataclass(frozen=True)
class MultiAttrEMALossConfig:
    enabled: bool = False
    beta: float = 0.95
    eps: float = 1e-8
    w_min: float = 0.2
    w_max: float = 5.0
    warmup_steps: int = 0
    alpha: float = 1.0


@dataclass(frozen=True)
class MultiAttrDWALossConfig:
    enabled: bool = False
    temperature: float = 2.0
    eps: float = 1e-12


@dataclass(frozen=True)
class SchedulerConfig:
    enabled: bool = False
    step_size: int = 0
    gamma: float = 1.0
    interval: str = "epoch"
    milestones: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "interval",
            _normalize_scheduler_interval(self.interval),
        )
        milestones = tuple(int(value) for value in self.milestones)
        if any(value <= 0 for value in milestones):
            raise ValueError("scheduler.milestones must contain positive integers")
        if tuple(sorted(set(milestones))) != milestones:
            raise ValueError(
                "scheduler.milestones must be strictly increasing"
            )
        object.__setattr__(self, "milestones", milestones)
        if self.enabled and not milestones and int(self.step_size) <= 0:
            raise ValueError(
                "enabled scheduler requires positive step_size or milestones"
            )
        if float(self.gamma) <= 0.0:
            raise ValueError("scheduler.gamma must be positive")


@dataclass(frozen=True)
class PretrainConfig:
    enabled: bool = False
    epochs: int = 0
    lr: float = 5e-5
    cluster_seed: int = 42
    assignments_cache_path: str = ""


@dataclass(frozen=True)
class EvaluationConfig:
    batch_size: int = 16384
    save_predictions: bool = True


@dataclass(frozen=True)
class PSNRLogConfig:
    enabled: bool = True
    per_target: bool = True


@dataclass(frozen=True)
class TimingLogConfig:
    enabled: bool = True
    epoch_breakdown: bool = True
    step_window: bool = False
    step_window_every_steps: int = 100
    cuda_sync: bool = False


@dataclass(frozen=True)
class LogConfig:
    effective_config: bool = True
    model_stats: bool = True
    epoch_summary: bool = True
    startup_timing: bool = True
    psnr: PSNRLogConfig = field(default_factory=PSNRLogConfig)
    timing: TimingLogConfig = field(default_factory=TimingLogConfig)


@dataclass(frozen=True)
class DataConfig:
    kind: str
    dataset_name: str | None = None
    split: str = "train"
    coords_path: str | None = None
    target_path: str | None = None
    targets: dict[str, str] | None = None
    target: str | None = None
    volume_shape: VolumeShape | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", _normalize_data_kind(self.kind))
        if self.target is not None:
            object.__setattr__(self, "target", str(self.target))
            if self.targets is None:
                raise ValueError("data.target requires data.targets")
            if self.target_path is not None:
                raise ValueError("data.target cannot be combined with data.target_path")
        if self.kind == "node":
            if self.coords_path is None:
                raise ValueError("node datasets require data.coords_path")
            if self.target_path is None and not self.targets:
                raise ValueError("node datasets require target_path or targets")
            if self.volume_shape is not None:
                raise ValueError("node datasets must not define volume_shape")
        else:
            if self.target_path is None and not self.targets:
                raise ValueError("volume datasets require target_path or targets")
            if self.target_path is not None and self.targets:
                raise ValueError("volume datasets must use either target_path or targets, not both")
        if self.targets is not None and not self.targets:
            raise ValueError("data.targets must be non-empty when provided")


@dataclass(frozen=True)
class ModelConfig:
    name: str
    params: dict[str, Any] = field(default_factory=dict)

    def as_builder_kwargs(self) -> dict[str, Any]:
        payload = dict(self.params)
        payload["name"] = self.name
        return payload


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 100
    batch_size: int = 8192
    pred_batch_size: int = 8192
    num_workers: int = 0
    lr: float = 5e-5
    beta_1: float = 0.9
    beta_2: float = 0.999
    epsilon: float = 1e-8
    weight_decay: float = 0.0
    loss_type: str = "mse"
    val_split: float = 0.1
    log_every: int = 10
    log_psnr_every: int = 0
    psnr_sample_ratio: float = 1.0
    save_every: int = 0
    early_stop_patience: int = 0
    seed: int = 42
    device: str = "cuda"
    sampler: str = "uniform_random"
    batches_per_epoch_budget: int = 0
    gradient_accumulation_steps: int = 1
    freeze_router_at: float = 0.0
    hard_topk_warmup_epochs: int = 0
    gradient_balancer: GradientBalancerConfig = field(default_factory=GradientBalancerConfig)
    multiview_ema_loss: MultiAttrEMALossConfig = field(default_factory=MultiAttrEMALossConfig)
    multiview_dwa_loss: MultiAttrDWALossConfig = field(default_factory=MultiAttrDWALossConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    pretrain: PretrainConfig = field(default_factory=PretrainConfig)

    def __post_init__(self) -> None:
        object.__setattr__(self, "loss_type", _normalize_loss_type(self.loss_type))
        object.__setattr__(self, "sampler", _normalize_sampler(self.sampler))
        if not (0.0 <= float(self.beta_1) < 1.0):
            raise ValueError(f"training.beta_1 must be in [0, 1), got {self.beta_1}")
        if not (0.0 <= float(self.beta_2) < 1.0):
            raise ValueError(f"training.beta_2 must be in [0, 1), got {self.beta_2}")
        if float(self.epsilon) <= 0.0:
            raise ValueError(f"training.epsilon must be positive, got {self.epsilon}")
        if not (0.0 <= float(self.val_split) < 1.0):
            raise ValueError(f"training.val_split must be in [0, 1), got {self.val_split}")
        if int(self.gradient_accumulation_steps) <= 0:
            raise ValueError(
                "training.gradient_accumulation_steps must be positive"
            )


@dataclass(frozen=True)
class ExperimentConfig:
    experiment: str | None
    exp_id: str
    experiment_root: str
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig
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
