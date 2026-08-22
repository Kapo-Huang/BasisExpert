from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

from ...utils.io import dump_yaml, load_yaml, resolve_path


TARGET_PLACEHOLDER = "{target}"
MODEL_KEYS = {
    "name",
    "scales",
    "block_size",
    "hidden_features",
    "hidden_layers",
    "omega_0",
    "coordinate_type",
    "propagation",
    "carry_start_scale",
    "coarse_feature_multiplier",
}
TRAINING_KEYS = {
    "epochs_per_scale",
    "lr",
    "beta_1",
    "beta_2",
    "block_mse_threshold",
    "scale_convergence_delta",
    "global_mse_threshold",
    "lr_decay",
    "max_active_blocks_per_step",
    "time_indices",
    "seed",
    "device",
    "log_every",
}
EVALUATION_KEYS = {"save_predictions", "run_after_training", "default_model"}
LOG_KEYS = {"effective_config", "startup_timing", "epoch_summary"}


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise TypeError(f"{label} must be a mapping")
    return dict(value)


def _reject_unknown(payload: dict[str, Any], allowed: set[str], label: str) -> None:
    unknown = sorted(set(payload).difference(allowed))
    if unknown:
        raise ValueError(f"Unknown {label} keys: {', '.join(unknown)}")


def _replace_target(value: Any, target: str) -> Any:
    if isinstance(value, dict):
        return {key: _replace_target(item, target) for key, item in value.items()}
    if isinstance(value, list):
        return [_replace_target(item, target) for item in value]
    if isinstance(value, str):
        return value.replace(TARGET_PLACEHOLDER, target)
    return value


def parse_time_indices(value: Any, total: int) -> list[int]:
    if value is None or (isinstance(value, str) and value.strip().lower() == "all"):
        return list(range(int(total)))
    if isinstance(value, int):
        result = [int(value)]
    elif isinstance(value, str):
        result = []
        for token in value.split(","):
            text = token.strip()
            if not text:
                continue
            if ":" in text:
                parts = [int(item) if item else None for item in text.split(":")]
                if len(parts) not in {2, 3}:
                    raise ValueError(f"Invalid timestep selection: {text!r}")
                start = 0 if parts[0] is None else parts[0]
                stop = int(total) if parts[1] is None else parts[1]
                step = 1 if len(parts) == 2 or parts[2] is None else parts[2]
                result.extend(range(start, stop, step))
            else:
                result.append(int(text))
    elif isinstance(value, (list, tuple)):
        result = [int(item) for item in value]
    else:
        raise TypeError("training.time_indices must be all, an integer, range text, or a list")
    if not result:
        raise ValueError("training.time_indices selected no timesteps")
    if len(set(result)) != len(result):
        raise ValueError("training.time_indices must not contain duplicates")
    if any(item < 0 or item >= int(total) for item in result):
        raise ValueError(f"training.time_indices must stay within [0,{int(total)})")
    return result


def _normalize_data(cfg: dict[str, Any], config_path: Path) -> tuple[dict[str, Any], int]:
    data = _mapping(cfg.get("data"), "data")
    allowed = {
        "kind", "dataset_name", "split", "target", "target_path", "targets",
        "volume_shape", "coordinate_axes", "spatial_dimensions",
    }
    _reject_unknown(data, allowed, "data")
    if str(data.get("kind", "volume")).lower() != "volume":
        raise ValueError("MINER only supports data.kind=volume")
    targets = _mapping(data.get("targets"), "data.targets")
    selected = str(data.get("target") or "").strip()
    if targets:
        if not selected:
            raise ValueError("data.targets requires data.target")
        if selected not in targets:
            raise ValueError(f"Unknown MINER target {selected!r}")
        resolved_targets = {
            str(name): resolve_path(str(path), base_dir=config_path.parent)
            for name, path in targets.items()
        }
        data["targets"] = resolved_targets
        data["target_path"] = resolved_targets[selected]
    elif data.get("target_path"):
        selected = selected or "target"
        data["target_path"] = resolve_path(str(data["target_path"]), base_dir=config_path.parent)
        data["targets"] = {}
    else:
        raise ValueError("MINER data requires target_path or targets")
    shape = _mapping(data.get("volume_shape"), "data.volume_shape")
    if set(shape) != {"T", "Z", "Y", "X"}:
        raise ValueError("data.volume_shape requires exactly T, Z, Y, and X")
    shape = {axis: int(shape[axis]) for axis in ("T", "Z", "Y", "X")}
    if any(value <= 0 for value in shape.values()):
        raise ValueError("All data.volume_shape values must be positive")
    if shape["Z"] == 1 and shape["Y"] > 1 and shape["X"] > 1:
        dimensions = 2
    elif shape["Z"] > 1 and shape["Y"] > 1 and shape["X"] > 1:
        dimensions = 3
    else:
        raise ValueError("MINER volume must be 2D (Z=1) or fully 3D")
    data.update({
        "kind": "volume",
        "dataset_name": str(data.get("dataset_name") or "volume").strip().lower(),
        "split": str(data.get("split") or "train"),
        "target": selected,
        "volume_shape": shape,
        "spatial_dimensions": dimensions,
    })
    return data, dimensions


def load_config(config_path: str | Path) -> dict[str, Any]:
    path = Path(config_path).resolve()
    cfg = deepcopy(load_yaml(path))
    if not isinstance(cfg, dict):
        raise TypeError("MINER config must contain a mapping")
    data, dimensions = _normalize_data(cfg, path)
    cfg["data"] = data
    cfg = _replace_target(cfg, data["target"])

    model_defaults = {
        "name": "miner",
        "scales": 4,
        "block_size": 32 if dimensions == 2 else 16,
        "hidden_features": 18 if dimensions == 2 else 20,
        "hidden_layers": 2,
        "omega_0": 150.0 if dimensions == 2 else 30.0,
        "coordinate_type": "local",
        "propagation": "coarse_to_fine",
        "carry_start_scale": 2,
        "coarse_feature_multiplier": 4,
    }
    raw_model = _mapping(cfg.get("model"), "model")
    _reject_unknown(raw_model, MODEL_KEYS, "model")
    model = {**model_defaults, **raw_model}
    if str(model["name"]).lower().replace("-", "_") != "miner":
        raise ValueError("model.name must be miner")
    for key in ("scales", "block_size", "hidden_features", "hidden_layers", "carry_start_scale", "coarse_feature_multiplier"):
        model[key] = int(model[key])
    model["omega_0"] = float(model["omega_0"])
    if model["scales"] <= 0 or model["block_size"] <= 0 or model["hidden_features"] <= 0:
        raise ValueError("MINER scale, block, and width values must be positive")
    if model["hidden_layers"] < 0 or not 0 <= model["carry_start_scale"] <= model["scales"]:
        raise ValueError("Invalid hidden_layers or carry_start_scale")
    if model["coordinate_type"] != "local" or model["propagation"] != "coarse_to_fine":
        raise ValueError("MINER reproduction fixes local coordinates and coarse_to_fine propagation")
    cfg["model"] = model

    training_defaults = {
        "epochs_per_scale": 500 if dimensions == 2 else 2000,
        "lr": 5.0e-4 if dimensions == 2 else 1.0e-3,
        "beta_1": 0.9,
        "beta_2": 0.999,
        "block_mse_threshold": 1.0e-4 if dimensions == 2 else 2.0e-4,
        "scale_convergence_delta": 5.0e-7 if dimensions == 2 else 2.0e-6,
        "global_mse_threshold": 1.0e-4 if dimensions == 2 else 0.0,
        "lr_decay": 0.999,
        "max_active_blocks_per_step": 16384 if dimensions == 2 else 2048,
        "time_indices": "all",
        "seed": 42,
        "device": "cuda",
        "log_every": 25,
    }
    raw_training = _mapping(cfg.get("training"), "training")
    _reject_unknown(raw_training, TRAINING_KEYS, "training")
    training = {**training_defaults, **raw_training}
    for key in ("epochs_per_scale", "max_active_blocks_per_step", "seed", "log_every"):
        training[key] = int(training[key])
    for key in (
        "lr", "beta_1", "beta_2", "block_mse_threshold",
        "scale_convergence_delta", "global_mse_threshold", "lr_decay",
    ):
        training[key] = float(training[key])
    if training["epochs_per_scale"] < 0 or training["max_active_blocks_per_step"] <= 0:
        raise ValueError("epochs_per_scale must be non-negative and max_active_blocks_per_step positive")
    if training["lr"] <= 0 or training["block_mse_threshold"] < 0:
        raise ValueError("training.lr must be positive and threshold non-negative")
    if not 0 <= training["beta_1"] < 1 or not 0 <= training["beta_2"] < 1:
        raise ValueError("Adam beta values must be in [0,1)")
    training["time_indices"] = parse_time_indices(training["time_indices"], data["volume_shape"]["T"])
    training["device"] = str(training["device"])
    cfg["training"] = training

    raw_evaluation = _mapping(cfg.get("evaluation"), "evaluation")
    _reject_unknown(raw_evaluation, EVALUATION_KEYS, "evaluation")
    evaluation = {
        "save_predictions": False,
        "run_after_training": False,
        "default_model": "checkpoint",
        **raw_evaluation,
    }
    if evaluation["default_model"] != "checkpoint":
        raise ValueError("MINER only defines checkpoint inference")
    evaluation["save_predictions"] = bool(evaluation["save_predictions"])
    evaluation["run_after_training"] = bool(evaluation["run_after_training"])
    cfg["evaluation"] = evaluation

    raw_log = _mapping(cfg.get("log"), "log")
    _reject_unknown(raw_log, LOG_KEYS, "log")
    cfg["log"] = {
        "effective_config": True,
        "startup_timing": True,
        "epoch_summary": True,
        **raw_log,
    }
    cfg["experiment"] = str(cfg.get("experiment") or f"miner_{data['dataset_name']}_{data['target']}")
    cfg["exp_id"] = str(cfg.get("exp_id") or f"miner-{data['dataset_name']}-{data['target']}")
    cfg["experiment_root"] = str(
        resolve_path(str(cfg.get("experiment_root") or "runs/miner"), base_dir=path.parent)
    )
    cfg["CONFIG_PATH"] = str(path)
    return cfg


def save_config(cfg: dict[str, Any], path: str | Path) -> Path:
    payload = deepcopy(cfg)
    payload.pop("CONFIG_PATH", None)
    return dump_yaml(path, payload)
