from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

from ..utils.io import dump_yaml, load_yaml, resolve_path

TARGET_PLACEHOLDER = "{target}"
PATH_KEYS = {
    "experiment_root",
    "target_path",
}
TOP_LEVEL_KEYS = {
    "experiment",
    "exp_id",
    "experiment_root",
    "MODEL",
    "DATA",
    "TRAINING",
    "EVALUATION",
    "exploration_probe",
}
MODEL_KEYS = {
    "model_name",
    "n_dims",
    "n_outputs",
    "feature_grid_shape",
    "n_features",
    "n_grids",
    "nodes_per_layer",
    "n_layers",
    "use_bias",
    "use_tcnn_if_available",
    "grid_initialization",
    "requires_padded_feats",
}
DATA_KEYS = {
    "dataset_name",
    "target",
    "target_path",
    "targets",
    "volume_shape",
    "align_corners",
}
TRAINING_KEYS = {
    "iterations",
    "points_per_iteration",
    "prediction_points_per_batch",
    "lr",
    "beta_1",
    "beta_2",
    "device",
    "data_device",
    "save_every",
    "log_every",
    "time_indices",
    "seed",
    "early_stopping",
}
EVALUATION_KEYS = {
    "run_after_training",
}


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


def _resolve_paths(value: Any, *, base_dir: Path, parent_key: str | None = None) -> Any:
    if isinstance(value, dict):
        if parent_key == "targets":
            return {str(name): resolve_path(str(path), base_dir=base_dir) for name, path in value.items()}
        return {key: _resolve_paths(item, base_dir=base_dir, parent_key=key) for key, item in value.items()}
    if isinstance(value, list):
        return [_resolve_paths(item, base_dir=base_dir, parent_key=parent_key) for item in value]
    if isinstance(value, str) and parent_key in PATH_KEYS:
        return resolve_path(value, base_dir=base_dir)
    return value


def _normalize_target_selection(cfg: dict[str, Any], *, target_override: str | None) -> dict[str, Any]:
    data_cfg = _ensure_mapping(cfg.get("DATA"), label="DATA")
    targets = data_cfg.get("targets")
    selected_target = str(target_override or data_cfg.get("target") or "").strip()

    if targets:
        if not isinstance(targets, dict) or not targets:
            raise ValueError("DATA.targets must be a non-empty mapping when provided")
        if not selected_target:
            raise ValueError("DATA.targets requires DATA.target or --target")
        if selected_target not in targets:
            available = ", ".join(sorted(map(str, targets.keys())))
            raise ValueError(f"Unknown target '{selected_target}'. Available targets: {available}")
        data_cfg["target"] = selected_target
        data_cfg["attr_name"] = selected_target
        data_cfg["target_path"] = targets[selected_target]
    elif data_cfg.get("target_path"):
        selected_target = selected_target or "target"
        data_cfg["target"] = selected_target
        data_cfg["attr_name"] = selected_target
    else:
        raise ValueError("DATA must provide target_path or targets")

    cfg["DATA"] = data_cfg
    return cfg


def _normalize_volume_shape(payload: Any) -> dict[str, int]:
    if not isinstance(payload, dict):
        raise TypeError("DATA.volume_shape must be a mapping")
    try:
        volume_shape = {
            "X": int(payload["X"]),
            "Y": int(payload["Y"]),
            "Z": int(payload["Z"]),
            "T": int(payload["T"]),
        }
    except KeyError as exc:
        raise ValueError(f"DATA.volume_shape is missing key: {exc.args[0]}") from exc
    for key, value in volume_shape.items():
        if value <= 0:
            raise ValueError(f"DATA.volume_shape.{key} must be positive, got {value}")
    return volume_shape


def _normalize_feature_grid_shape(payload: Any) -> list[int]:
    if not isinstance(payload, list) or len(payload) != 3:
        raise ValueError("MODEL.feature_grid_shape must be a list of three integers")
    shape = [int(value) for value in payload]
    if any(value <= 0 for value in shape):
        raise ValueError("MODEL.feature_grid_shape values must be positive")
    return shape


def _expand_time_indices(raw_value: Any, *, time_count: int) -> list[int]:
    if raw_value is None or (isinstance(raw_value, str) and str(raw_value).strip().lower() == "all"):
        return list(range(int(time_count)))

    if isinstance(raw_value, int):
        indices = [int(raw_value)]
    elif isinstance(raw_value, list):
        indices = [int(value) for value in raw_value]
    else:
        raise TypeError("TRAINING.time_indices must be 'all', an integer, or a list of integers")

    expanded: list[int] = []
    seen: set[int] = set()
    for index in indices:
        if index in seen:
            raise ValueError(f"Duplicate time index in TRAINING.time_indices: {index}")
        if index < 0 or index >= int(time_count):
            raise ValueError(f"Time index {index} is outside [0, {int(time_count) - 1}]")
        expanded.append(index)
        seen.add(index)
    return expanded


def _normalize_data_section(data_cfg: dict[str, Any]) -> dict[str, Any]:
    _reject_unknown_keys(data_cfg, DATA_KEYS | {"attr_name"}, label="DATA")
    dataset_name = str(data_cfg.get("dataset_name", "")).strip().lower()
    if dataset_name != "ionization":
        raise ValueError(f"APMGSRN only supports DATA.dataset_name='ionization', got {data_cfg.get('dataset_name')!r}")
    volume_shape = _normalize_volume_shape(data_cfg.get("volume_shape"))
    return {
        "dataset_name": dataset_name,
        "target": str(data_cfg["target"]),
        "attr_name": str(data_cfg.get("attr_name", data_cfg["target"])),
        "target_path": str(data_cfg["target_path"]),
        "targets": dict(data_cfg.get("targets") or {}),
        "volume_shape": volume_shape,
        "align_corners": bool(data_cfg.get("align_corners", True)),
    }


def _normalize_model_section(model_cfg: dict[str, Any]) -> dict[str, Any]:
    _reject_unknown_keys(model_cfg, MODEL_KEYS, label="MODEL")
    model_name = str(model_cfg.get("model_name", "apmgsrn")).strip().lower()
    if model_name != "apmgsrn":
        raise ValueError(f"Unsupported APMGSRN MODEL.model_name: {model_cfg.get('model_name')!r}")
    n_dims = int(model_cfg.get("n_dims", 3))
    if n_dims != 3:
        raise ValueError(f"APMGSRN only supports 3D spatial grids, got MODEL.n_dims={n_dims}")
    n_outputs = int(model_cfg.get("n_outputs", 1))
    if n_outputs != 1:
        raise ValueError(f"APMGSRN only supports scalar outputs per timestep, got MODEL.n_outputs={n_outputs}")
    requires_padded_feats_raw = model_cfg.get("requires_padded_feats")
    requires_padded_feats = None if requires_padded_feats_raw is None else bool(requires_padded_feats_raw)
    return {
        "model_name": model_name,
        "n_dims": n_dims,
        "n_outputs": n_outputs,
        "feature_grid_shape": _normalize_feature_grid_shape(model_cfg.get("feature_grid_shape")),
        "n_features": int(model_cfg.get("n_features", 2)),
        "n_grids": int(model_cfg.get("n_grids", 64)),
        "nodes_per_layer": int(model_cfg.get("nodes_per_layer", 64)),
        "n_layers": int(model_cfg.get("n_layers", 2)),
        "use_bias": bool(model_cfg.get("use_bias", False)),
        "use_tcnn_if_available": bool(model_cfg.get("use_tcnn_if_available", True)),
        "grid_initialization": str(model_cfg.get("grid_initialization", "default")),
        "requires_padded_feats": requires_padded_feats,
    }


def _normalize_training_section(training_cfg: dict[str, Any], *, time_count: int) -> dict[str, Any]:
    _reject_unknown_keys(training_cfg, TRAINING_KEYS, label="TRAINING")
    points_per_iteration = int(training_cfg.get("points_per_iteration", 100_000))
    prediction_points_per_batch = int(training_cfg.get("prediction_points_per_batch", points_per_iteration))
    normalized = {
        "iterations": int(training_cfg.get("iterations", 10_000)),
        "points_per_iteration": points_per_iteration,
        "prediction_points_per_batch": prediction_points_per_batch,
        "lr": float(training_cfg.get("lr", 1.0e-2)),
        "beta_1": float(training_cfg.get("beta_1", 0.9)),
        "beta_2": float(training_cfg.get("beta_2", 0.99)),
        "device": str(training_cfg.get("device", "cuda:0")),
        "data_device": str(training_cfg.get("data_device", "same")),
        "save_every": int(training_cfg.get("save_every", 0)),
        "log_every": int(training_cfg.get("log_every", 0)),
        "time_indices": _expand_time_indices(training_cfg.get("time_indices", "all"), time_count=time_count),
        "seed": int(training_cfg.get("seed", 42)),
        "early_stopping": bool(training_cfg.get("early_stopping", True)),
    }
    if normalized["iterations"] <= 0:
        raise ValueError("TRAINING.iterations must be positive")
    if normalized["points_per_iteration"] <= 0:
        raise ValueError("TRAINING.points_per_iteration must be positive")
    if normalized["prediction_points_per_batch"] <= 0:
        raise ValueError("TRAINING.prediction_points_per_batch must be positive")
    if normalized["lr"] <= 0.0:
        raise ValueError("TRAINING.lr must be positive")
    if not (0.0 < normalized["beta_1"] < 1.0):
        raise ValueError("TRAINING.beta_1 must be in (0, 1)")
    if not (0.0 < normalized["beta_2"] < 1.0):
        raise ValueError("TRAINING.beta_2 must be in (0, 1)")
    if normalized["save_every"] < 0:
        raise ValueError("TRAINING.save_every must be non-negative")
    if normalized["log_every"] < 0:
        raise ValueError("TRAINING.log_every must be non-negative")
    return normalized


def _normalize_evaluation_section(evaluation_cfg: dict[str, Any]) -> dict[str, Any]:
    _reject_unknown_keys(evaluation_cfg, EVALUATION_KEYS, label="EVALUATION")
    return {
        "run_after_training": bool(evaluation_cfg.get("run_after_training", True)),
    }


def config_payload(cfg: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in cfg.items() if key != "CONFIG_PATH"}


def load_config(config_path: str | Path, *, target_override: str | None = None, identifier: str | None = None) -> dict[str, Any]:
    path = Path(config_path).resolve()
    raw_cfg = deepcopy(load_yaml(path))
    _reject_unknown_keys(raw_cfg, TOP_LEVEL_KEYS, label="config")
    cfg = _normalize_target_selection(raw_cfg, target_override=target_override)
    selected_target = cfg["DATA"].get("target")
    cfg = _replace_target_placeholder(cfg, selected_target)
    cfg = _resolve_paths(cfg, base_dir=path.parent)

    normalized_data = _normalize_data_section(_ensure_mapping(cfg.get("DATA"), label="DATA"))
    normalized_model = _normalize_model_section(_ensure_mapping(cfg.get("MODEL"), label="MODEL"))
    normalized_training = _normalize_training_section(
        _ensure_mapping(cfg.get("TRAINING"), label="TRAINING"),
        time_count=int(normalized_data["volume_shape"]["T"]),
    )
    normalized_evaluation = _normalize_evaluation_section(
        _ensure_mapping(cfg.get("EVALUATION"), label="EVALUATION"),
    )

    exp_id_default = f"apmgsrn-ionization-{normalized_data['target']}"
    return {
        "experiment": str(cfg.get("experiment") or exp_id_default),
        "exp_id": str(identifier or cfg.get("exp_id") or exp_id_default),
        "experiment_root": str(cfg.get("experiment_root") or "runs"),
        "MODEL": normalized_model,
        "DATA": normalized_data,
        "TRAINING": normalized_training,
        "EVALUATION": normalized_evaluation,
        "exploration_probe": deepcopy(cfg.get("exploration_probe") or {}),
        "CONFIG_PATH": str(path),
    }


def experiment_dir_from_config(cfg: dict[str, Any]) -> Path:
    return Path(cfg["experiment_root"]) / str(cfg["exp_id"])


def run_dir_from_config(cfg: dict[str, Any]) -> Path:
    return experiment_dir_from_config(cfg)


def save_config(cfg: dict[str, Any], path: str | Path) -> Path:
    return dump_yaml(path, config_payload(cfg))
