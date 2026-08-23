from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

from ...utils.io import dump_yaml, load_yaml, resolve_path

TOP_LEVEL = {
    "experiment", "exp_id", "experiment_root", "data", "model",
    "training", "evaluation", "log", "exploration_probe",
}

DEFAULT_MODEL = {
    "name": "fv_srn",
    "grid_resolution": 32,
    "grid_channels": 16,
    "grid_init_std": 0.01,
    "keyframe_indices": list(range(0, 100, 9)),
    "fourier_features": 14,
    "fourier_mode": "nerf",
    "hidden_features": 32,
    "hidden_layers": 3,
    "activation": "snake_alt",
    "activation_frequency": 1.0,
    "time_encoding": "none",
}
DEFAULT_TRAINING = {
    "epochs": 200,
    "samples_per_timestep": 8192,
    "validation_fraction": 0.2,
    "batch_size": 8192,
    "prediction_batch_size": 65536,
    "lr": 5.0e-5,
    "beta_1": 0.9,
    "beta_2": 0.999,
    "eps": 1.0e-8,
    "weight_decay": 0.0,
    "lr_scheduler": "step",
    "lr_step": 40,
    "lr_gamma": 0.92,
    "l1_weight": 1.0,
    "l2_weight": 0.0,
    "importance_floor": 0.01,
    "rebuild_every": 51,
    "rebuild_grid_size": 32,
    "rebuild_samples_per_cell": 2,
    "save_every": 20,
    "log_every": 1,
    "seed": 42,
    "device": "cuda",
}
DEFAULT_EVALUATION = {
    "batch_size": 65536,
    "save_predictions": True,
    "run_after_training": True,
    "default_model": "compact",
}
DEFAULT_LOG = {
    "effective_config": True,
    "model_stats": True,
    "epoch_summary": True,
    "startup_timing": True,
}


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise TypeError(f"{label} must be a mapping")
    return dict(value)


def _reject(payload: dict[str, Any], allowed: set[str], label: str) -> None:
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise ValueError(f"Unknown {label} keys: {', '.join(unknown)}")


def _replace_target(value: Any, target: str) -> Any:
    if isinstance(value, dict):
        return {key: _replace_target(item, target) for key, item in value.items()}
    if isinstance(value, list):
        return [_replace_target(item, target) for item in value]
    if isinstance(value, str):
        return value.replace("{target}", target)
    return value


def _positive(value: Any, label: str, *, integer: bool = False):
    result = int(value) if integer else float(value)
    if result <= 0:
        raise ValueError(f"{label} must be positive")
    return result


def load_config(path: str | Path, *, target_override: str | None = None) -> dict[str, Any]:
    config_path = Path(path).resolve()
    raw = load_yaml(config_path)
    _reject(raw, TOP_LEVEL, "config")
    cfg = deepcopy(raw)
    data = _mapping(cfg.get("data"), "data")
    targets = data.get("targets")
    selected = str(target_override or data.get("target") or "").strip()
    if targets:
        if not isinstance(targets, dict) or not targets:
            raise ValueError("data.targets must be a non-empty mapping")
        if selected not in targets:
            raise ValueError(f"Unknown target {selected!r}. Available targets: {', '.join(sorted(targets))}")
        data["target"] = selected
        data["target_path"] = resolve_path(str(targets[selected]), base_dir=config_path.parent)
        data["targets"] = {
            str(name): resolve_path(str(value), base_dir=config_path.parent)
            for name, value in targets.items()
        }
    elif data.get("target_path"):
        selected = selected or str(data.get("target") or "target")
        data["target"] = selected
        data["target_path"] = resolve_path(str(data["target_path"]), base_dir=config_path.parent)
    else:
        raise ValueError("data requires target_path or targets")
    cfg["data"] = data
    cfg = _replace_target(cfg, selected)

    _reject(
        data,
        {"kind", "dataset_name", "split", "target", "target_path", "targets", "volume_shape"},
        "data",
    )
    if str(data.get("kind", "volume")).lower() != "volume":
        raise ValueError("fV-SRN only supports data.kind='volume'")
    shape = _mapping(data.get("volume_shape"), "data.volume_shape")
    _reject(shape, {"T", "Z", "Y", "X"}, "data.volume_shape")
    if set(shape) != {"T", "Z", "Y", "X"}:
        raise ValueError("data.volume_shape requires T, Z, Y, and X")
    data["volume_shape"] = {axis: _positive(shape[axis], f"volume_shape.{axis}", integer=True) for axis in ("T", "Z", "Y", "X")}

    model = {**DEFAULT_MODEL, **_mapping(cfg.get("model"), "model")}
    _reject(model, set(DEFAULT_MODEL), "model")
    if str(model["name"]).lower() != "fv_srn":
        raise ValueError("model.name must be 'fv_srn'")
    if str(model["fourier_mode"]).lower() != "nerf":
        raise ValueError("Only model.fourier_mode='nerf' is supported")
    if str(model["activation"]).lower() != "snake_alt":
        raise ValueError("Only model.activation='snake_alt' is supported")
    if str(model["time_encoding"]).lower() != "none":
        raise ValueError("This temporal fV-SRN requires model.time_encoding='none'")
    for key in ("grid_resolution", "grid_channels", "fourier_features", "hidden_features", "hidden_layers"):
        model[key] = _positive(model[key], f"model.{key}", integer=True)
    model["grid_init_std"] = _positive(model["grid_init_std"], "model.grid_init_std")
    model["activation_frequency"] = _positive(model["activation_frequency"], "model.activation_frequency")
    keyframes = [int(value) for value in model["keyframe_indices"]]
    if keyframes != sorted(set(keyframes)) or len(keyframes) < 2:
        raise ValueError("model.keyframe_indices must be strictly increasing and contain at least two entries")
    if keyframes[0] != 0 or keyframes[-1] != data["volume_shape"]["T"] - 1:
        raise ValueError("model.keyframe_indices must include temporal endpoints 0 and T-1")
    model["keyframe_indices"] = keyframes
    cfg["model"] = model

    training = {**DEFAULT_TRAINING, **_mapping(cfg.get("training"), "training")}
    _reject(training, set(DEFAULT_TRAINING), "training")
    for key in (
        "epochs", "samples_per_timestep", "batch_size", "prediction_batch_size",
        "lr_step", "rebuild_grid_size", "rebuild_samples_per_cell", "log_every",
    ):
        training[key] = _positive(training[key], f"training.{key}", integer=True)
    training["save_every"] = int(training["save_every"])
    training["rebuild_every"] = int(training["rebuild_every"])
    if training["save_every"] < 0 or training["rebuild_every"] < 0:
        raise ValueError("training.save_every and rebuild_every must be non-negative")
    for key in ("lr", "beta_1", "beta_2", "eps", "lr_gamma"):
        training[key] = _positive(training[key], f"training.{key}")
    training["weight_decay"] = float(training["weight_decay"])
    if training["weight_decay"] < 0.0:
        raise ValueError("training.weight_decay must be non-negative")
    scheduler_name = str(training["lr_scheduler"]).strip().lower()
    if scheduler_name not in {"constant", "step"}:
        raise ValueError("training.lr_scheduler must be 'constant' or 'step'")
    training["lr_scheduler"] = scheduler_name
    validation_fraction = float(training["validation_fraction"])
    if not 0.0 <= validation_fraction < 1.0:
        raise ValueError("training.validation_fraction must be in [0,1)")
    training["validation_fraction"] = validation_fraction
    training["l1_weight"] = float(training["l1_weight"])
    training["l2_weight"] = float(training["l2_weight"])
    if training["l1_weight"] < 0 or training["l2_weight"] < 0 or not (training["l1_weight"] or training["l2_weight"]):
        raise ValueError("At least one non-negative L1/L2 loss weight must be active")
    floor = float(training["importance_floor"])
    if not 0.0 < floor <= 1.0:
        raise ValueError("training.importance_floor must be in (0,1]")
    training["importance_floor"] = floor
    training["seed"] = int(training["seed"])
    cfg["training"] = training

    evaluation = {**DEFAULT_EVALUATION, **_mapping(cfg.get("evaluation"), "evaluation")}
    _reject(evaluation, set(DEFAULT_EVALUATION), "evaluation")
    evaluation["batch_size"] = _positive(evaluation["batch_size"], "evaluation.batch_size", integer=True)
    if evaluation["default_model"] not in {"compact", "checkpoint"}:
        raise ValueError("evaluation.default_model must be compact or checkpoint")
    cfg["evaluation"] = evaluation
    log = {**DEFAULT_LOG, **_mapping(cfg.get("log"), "log")}
    _reject(log, set(DEFAULT_LOG), "log")
    cfg["log"] = log
    cfg["experiment"] = str(cfg.get("experiment") or f"fv_srn_{selected}")
    cfg["exp_id"] = str(cfg.get("exp_id") or f"fv-srn-{selected}")
    cfg["experiment_root"] = str(
        resolve_path(str(cfg.get("experiment_root") or "runs/fv_srn"), base_dir=config_path.parent)
    )
    cfg["CONFIG_PATH"] = str(config_path)
    return cfg


def save_config(cfg: dict[str, Any], path: str | Path) -> Path:
    payload = deepcopy(cfg)
    payload.pop("CONFIG_PATH", None)
    return dump_yaml(path, payload)
