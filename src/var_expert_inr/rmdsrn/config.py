from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

from ..utils.io import dump_yaml, load_yaml, resolve_path

TOP_LEVEL_KEYS = {
    "experiment",
    "exp_id",
    "experiment_root",
    "data",
    "model",
    "training",
    "evaluation",
    "log",
    "exploration_probe",
}
DATA_KEYS = {
    "kind",
    "dataset_name",
    "split",
    "target",
    "target_path",
    "targets",
    "volume_shape",
}
MODEL_DEFAULTS = {
    "name": "rmdsrn",
    "base_encoder": "temporal_fv_srn",
    "grid_resolution": 32,
    "grid_channels": 16,
    "grid_init_std": 0.01,
    "keyframe_indices": [0, 9, 18, 27, 36, 45, 54, 63, 72, 81, 90, 99],
    "fourier_features": 14,
    "fourier_mode": "nerf",
    "decoder_count": 5,
    "decoder_hidden_features": 64,
    "decoder_hidden_layers": 2,
    "activation": "snake_alt",
    "activation_frequency": 1.0,
}
TRAINING_DEFAULTS = {
    "steps": 50_000,
    "batch_size": 131_072,
    "lr": 5.0e-3,
    "beta_1": 0.9,
    "beta_2": 0.999,
    "min_lr": 1.0e-7,
    "lambda_min": 0.0,
    "lambda_max": 10.0,
    "lambda_growth_rate": 500.0,
    "epsilon": 1.0e-12,
    "save_every": 5_000,
    "log_every": 100,
    "seed": 42,
    "device": "cuda",
}
EVALUATION_DEFAULTS = {
    "batch_size": 65_536,
    "save_mean": True,
    "save_variance": True,
    "run_after_training": True,
    "default_model": "artifact",
    "uncertainty_sample_size": 1_000_000,
    "topk_fractions": [0.01, 0.05],
    "seed": 42,
}
LOG_DEFAULTS = {
    "effective_config": True,
    "model_stats": True,
    "startup_timing": True,
}


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise TypeError(f"{label} must be a mapping")
    return dict(value)


def _reject(payload: dict[str, Any], allowed: set[str], label: str) -> None:
    unknown = sorted(set(payload).difference(allowed))
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
    resolved = int(value) if integer else float(value)
    if resolved <= 0:
        raise ValueError(f"{label} must be positive")
    return resolved


def _volume_shape(value: Any) -> dict[str, int]:
    payload = _mapping(value, "data.volume_shape")
    _reject(payload, {"T", "Z", "Y", "X"}, "data.volume_shape")
    if set(payload) != {"T", "Z", "Y", "X"}:
        raise ValueError("data.volume_shape requires T, Z, Y, and X")
    return {
        axis: _positive(payload[axis], f"data.volume_shape.{axis}", integer=True)
        for axis in ("T", "Z", "Y", "X")
    }


def _normalize_data(
    payload: dict[str, Any],
    *,
    config_path: Path,
    target_override: str | None,
) -> tuple[dict[str, Any], str]:
    _reject(payload, DATA_KEYS, "data")
    if str(payload.get("kind", "volume")).strip().lower() != "volume":
        raise ValueError("RMDSRN only supports data.kind='volume'")
    dataset_name = str(payload.get("dataset_name", "ionization")).strip().lower()
    if dataset_name not in {"ionization", "combustion_40nh3_1"}:
        raise ValueError(
            "RMDSRN only supports structured volume datasets "
            "'ionization' and 'combustion_40NH3_1'"
        )
    targets = payload.get("targets")
    selected = str(target_override or payload.get("target") or "").strip()
    if targets:
        if not isinstance(targets, dict) or not targets:
            raise ValueError("data.targets must be a non-empty mapping")
        if selected not in targets:
            raise ValueError(
                f"Unknown target {selected!r}. Available targets: {', '.join(sorted(map(str, targets)))}"
            )
        target_path = resolve_path(str(targets[selected]), base_dir=config_path.parent)
        resolved_targets = {
            str(name): resolve_path(str(path), base_dir=config_path.parent)
            for name, path in targets.items()
        }
    elif payload.get("target_path"):
        selected = selected or "target"
        target_path = resolve_path(str(payload["target_path"]), base_dir=config_path.parent)
        resolved_targets = None
    else:
        raise ValueError("data requires target_path or targets")
    return {
        "kind": "volume",
        "dataset_name": dataset_name,
        "split": str(payload.get("split", "train")),
        "target": selected,
        "target_path": str(target_path),
        "targets": resolved_targets,
        "volume_shape": _volume_shape(payload.get("volume_shape")),
    }, selected


def _normalize_model(payload: dict[str, Any], *, time_count: int) -> dict[str, Any]:
    model = {**MODEL_DEFAULTS, **payload}
    _reject(model, set(MODEL_DEFAULTS), "model")
    if str(model["name"]).strip().lower() != "rmdsrn":
        raise ValueError("model.name must be 'rmdsrn'")
    if str(model["base_encoder"]).strip().lower() != "temporal_fv_srn":
        raise ValueError("model.base_encoder must be 'temporal_fv_srn'")
    if str(model["fourier_mode"]).strip().lower() != "nerf":
        raise ValueError("model.fourier_mode must be 'nerf'")
    if str(model["activation"]).strip().lower() != "snake_alt":
        raise ValueError("model.activation must be 'snake_alt'")
    for key in (
        "grid_resolution",
        "grid_channels",
        "fourier_features",
        "decoder_hidden_features",
        "decoder_hidden_layers",
    ):
        model[key] = _positive(model[key], f"model.{key}", integer=True)
    model["decoder_count"] = int(model["decoder_count"])
    if model["decoder_count"] < 2:
        raise ValueError("model.decoder_count must be at least 2")
    model["grid_init_std"] = _positive(model["grid_init_std"], "model.grid_init_std")
    model["activation_frequency"] = _positive(
        model["activation_frequency"], "model.activation_frequency"
    )
    keyframes = [int(value) for value in model["keyframe_indices"]]
    if len(keyframes) < 2 or keyframes != sorted(set(keyframes)):
        raise ValueError("model.keyframe_indices must be strictly increasing")
    if keyframes[0] != 0 or keyframes[-1] != int(time_count) - 1:
        raise ValueError("model.keyframe_indices must include temporal endpoints 0 and T-1")
    model["keyframe_indices"] = keyframes
    model["name"] = "rmdsrn"
    model["base_encoder"] = "temporal_fv_srn"
    model["fourier_mode"] = "nerf"
    model["activation"] = "snake_alt"
    return model


def _normalize_training(payload: dict[str, Any]) -> dict[str, Any]:
    training = {**TRAINING_DEFAULTS, **payload}
    _reject(training, set(TRAINING_DEFAULTS), "training")
    for key in ("steps", "batch_size"):
        training[key] = _positive(training[key], f"training.{key}", integer=True)
    for key in ("save_every", "log_every"):
        training[key] = int(training[key])
        if training[key] < 0:
            raise ValueError(f"training.{key} must be non-negative")
    for key in ("lr", "min_lr", "epsilon"):
        training[key] = _positive(training[key], f"training.{key}")
    for key in ("beta_1", "beta_2"):
        training[key] = float(training[key])
        if not 0.0 < training[key] < 1.0:
            raise ValueError(f"training.{key} must be in (0, 1)")
    training["lambda_min"] = float(training["lambda_min"])
    training["lambda_max"] = float(training["lambda_max"])
    if training["lambda_min"] < 0.0:
        raise ValueError("training.lambda_min must be non-negative")
    if training["lambda_max"] < training["lambda_min"]:
        raise ValueError("training.lambda_max must be >= training.lambda_min")
    training["lambda_growth_rate"] = float(training["lambda_growth_rate"])
    if training["lambda_growth_rate"] < 1.0:
        raise ValueError("training.lambda_growth_rate must be at least 1")
    training["seed"] = int(training["seed"])
    training["device"] = str(training["device"])
    return training


def _normalize_evaluation(payload: dict[str, Any]) -> dict[str, Any]:
    evaluation = {**EVALUATION_DEFAULTS, **payload}
    _reject(evaluation, set(EVALUATION_DEFAULTS), "evaluation")
    evaluation["batch_size"] = _positive(
        evaluation["batch_size"], "evaluation.batch_size", integer=True
    )
    evaluation["uncertainty_sample_size"] = _positive(
        evaluation["uncertainty_sample_size"],
        "evaluation.uncertainty_sample_size",
        integer=True,
    )
    evaluation["save_mean"] = bool(evaluation["save_mean"])
    evaluation["save_variance"] = bool(evaluation["save_variance"])
    if not evaluation["save_mean"] or not evaluation["save_variance"]:
        raise ValueError("RMDSRN evaluation requires save_mean=true and save_variance=true")
    evaluation["run_after_training"] = bool(evaluation["run_after_training"])
    evaluation["default_model"] = str(evaluation["default_model"]).strip().lower()
    if evaluation["default_model"] not in {"artifact", "checkpoint"}:
        raise ValueError("evaluation.default_model must be 'artifact' or 'checkpoint'")
    fractions = [float(value) for value in evaluation["topk_fractions"]]
    if not fractions or any(value <= 0.0 or value >= 1.0 for value in fractions):
        raise ValueError("evaluation.topk_fractions must contain values in (0, 1)")
    evaluation["topk_fractions"] = fractions
    evaluation["seed"] = int(evaluation["seed"])
    return evaluation


def load_config(path: str | Path, *, target_override: str | None = None) -> dict[str, Any]:
    config_path = Path(path).resolve()
    raw = deepcopy(load_yaml(config_path))
    _reject(raw, TOP_LEVEL_KEYS, "config")
    data, selected = _normalize_data(
        _mapping(raw.get("data"), "data"),
        config_path=config_path,
        target_override=target_override,
    )
    raw = _replace_target(raw, selected)
    model = _normalize_model(
        _mapping(raw.get("model"), "model"),
        time_count=int(data["volume_shape"]["T"]),
    )
    training = _normalize_training(_mapping(raw.get("training"), "training"))
    evaluation = _normalize_evaluation(_mapping(raw.get("evaluation"), "evaluation"))
    log = {**LOG_DEFAULTS, **_mapping(raw.get("log"), "log")}
    _reject(log, set(LOG_DEFAULTS), "log")
    return {
        "experiment": str(raw.get("experiment") or f"rmdsrn_ionization_{selected}"),
        "exp_id": str(raw.get("exp_id") or f"rmdsrn-ionization-{selected}"),
        "experiment_root": str(
            resolve_path(str(raw.get("experiment_root") or "runs/rmdsrn"), base_dir=config_path.parent)
        ),
        "data": data,
        "model": model,
        "training": training,
        "evaluation": evaluation,
        "log": {key: bool(value) for key, value in log.items()},
        "exploration_probe": deepcopy(raw.get("exploration_probe") or {}),
        "CONFIG_PATH": str(config_path),
    }


def config_payload(cfg: dict[str, Any]) -> dict[str, Any]:
    return {key: deepcopy(value) for key, value in cfg.items() if key != "CONFIG_PATH"}


def save_config(cfg: dict[str, Any], path: str | Path) -> Path:
    return dump_yaml(path, config_payload(cfg))
