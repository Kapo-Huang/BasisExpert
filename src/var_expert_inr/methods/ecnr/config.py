from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

from ...utils.io import dump_yaml, load_yaml, resolve_path


TOP_LEVEL = {
    "experiment",
    "exp_id",
    "experiment_root",
    "data",
    "model",
    "clustering",
    "training",
    "quantization",
    "cnn",
    "evaluation",
    "log",
}

DEFAULT_MODEL = {
    "name": "ecnr",
    "scales": 3,
    "block_shape_xyz": [25, 31, 31],
    "residual_threshold": 1.0e-4,
    "gaussian_kernel_size": 5,
    "gaussian_sigma": 1.0,
    "gaussian_padding": "reflect",
    "latent_dim": 8,
    "hidden_features": 24,
    "hidden_layers": 3,
    "omega_0": 30.0,
    "target_blocks_per_mlp": [8, 16, 32],
}

DEFAULT_CLUSTERING = {
    "distance": "squared_euclidean",
    "input": "normalized_block_values",
    "initialization": "kmeans_pp",
    "seed": 42,
    "balancing_passes": 1,
    "centroid_dtype": "float32",
    "tie_break": "lowest_cluster_index",
    "n_init": 1,
    "max_iter": 300,
    "tol": 1.0e-4,
    "algorithm": "lloyd",
}

DEFAULT_TRAINING = {
    "epochs_per_scale": 500,
    "batch_size": 3200,
    "passes_per_epoch": 1,
    "lr": 1.0e-3,
    "beta_1": 0.9,
    "beta_2": 0.999,
    "weight_decay": 2.0e-5,
    "pruning_epochs": [150, 225, 300, 375],
    "pruning_sparsities": [0.30, 0.40, 0.45, 0.50],
    "pruning_loss_weight": 0.1,
    "pruning_lr_gamma": 0.75,
    "quantization_finetune_epochs": 75,
    "quantization_finetune_passes_per_epoch": 1,
    "quantization_finetune_lr": 1.0e-5,
    "save_every": 0,
    "save_intermediate_checkpoints": True,
    "log_every": 1,
    "progress_log_seconds": 60,
    "seed": 42,
    "device": "cuda",
}

DEFAULT_QUANTIZATION = {
    "mlp_weight_bits": 8,
    "mlp_bias_bits": 8,
    "latent_bits": 0,
    "cnn_bits": 9,
    "entropy": "huffman",
}

DEFAULT_CNN = {
    "implementation_choice": "fixed_for_underspecified_method_detail",
    "dimensionality": "3d",
    "layers": 5,
    "hidden_channels": 32,
    "kernel_size": 3,
    "stride": 1,
    "padding": 1,
    "dilation": 1,
    "bias": True,
    "hidden_activation": "relu",
    "output_activation": "none",
    "halo": 5,
    "tile_core_shape_zyx": [32, 64, 64],
    "epochs": 100,
    "lr": 1.0e-5,
}

DEFAULT_EVALUATION = {
    "batch_size": 3200,
    "save_predictions": False,
    "run_after_training": False,
    "default_model": "checkpoint",
}

DEFAULT_LOG = {
    "effective_config": True,
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


def _fixed(payload: dict[str, Any], expected: dict[str, Any], keys: tuple[str, ...], label: str) -> None:
    for key in keys:
        if payload[key] != expected[key]:
            raise ValueError(f"{label}.{key} is fixed to {expected[key]!r}, got {payload[key]!r}")


def _replace_target(value: Any, target: str) -> Any:
    if isinstance(value, dict):
        return {key: _replace_target(item, target) for key, item in value.items()}
    if isinstance(value, list):
        return [_replace_target(item, target) for item in value]
    if isinstance(value, str):
        return value.replace("{target}", target)
    return value


def load_config(path: str | Path, *, target_override: str | None = None) -> dict[str, Any]:
    config_path = Path(path).resolve()
    raw = load_yaml(config_path)
    _reject(raw, TOP_LEVEL, "config")
    cfg = deepcopy(raw)

    data = _mapping(cfg.get("data"), "data")
    _reject(
        data,
        {"kind", "dataset_name", "split", "target", "target_path", "targets", "volume_shape"},
        "data",
    )
    if str(data.get("kind", "volume")).lower() != "volume":
        raise ValueError("ECNR only supports data.kind='volume'")
    targets = data.get("targets")
    selected = str(target_override or data.get("target") or "").strip()
    if targets:
        if not isinstance(targets, dict) or not targets:
            raise ValueError("data.targets must be a non-empty mapping")
        if selected not in targets:
            raise ValueError(f"Unknown target {selected!r}; available: {', '.join(sorted(targets))}")
        data["targets"] = {
            str(name): resolve_path(str(value), base_dir=config_path.parent)
            for name, value in targets.items()
        }
        data["target_path"] = data["targets"][selected]
    elif data.get("target_path"):
        selected = selected or "target"
        data["target_path"] = resolve_path(str(data["target_path"]), base_dir=config_path.parent)
    else:
        raise ValueError("data requires target_path or targets")
    data["target"] = selected
    shape = _mapping(data.get("volume_shape"), "data.volume_shape")
    if set(shape) != {"T", "Z", "Y", "X"}:
        raise ValueError("data.volume_shape requires T, Z, Y, and X")
    data["volume_shape"] = {axis: int(shape[axis]) for axis in ("T", "Z", "Y", "X")}
    if any(value <= 0 for value in data["volume_shape"].values()):
        raise ValueError("All volume dimensions must be positive")
    cfg["data"] = data
    cfg = _replace_target(cfg, selected)

    model = {**DEFAULT_MODEL, **_mapping(cfg.get("model"), "model")}
    _reject(model, set(DEFAULT_MODEL), "model")
    _fixed(
        model,
        DEFAULT_MODEL,
        (
            "name", "scales", "residual_threshold", "gaussian_kernel_size",
            "gaussian_padding", "latent_dim", "hidden_features", "hidden_layers",
            "omega_0", "target_blocks_per_mlp",
        ),
        "model",
    )
    block_shape = [int(value) for value in model["block_shape_xyz"]]
    if len(block_shape) != 3 or any(value <= 0 for value in block_shape):
        raise ValueError("model.block_shape_xyz must contain three positive integers")
    model["block_shape_xyz"] = block_shape
    model["gaussian_sigma"] = float(model["gaussian_sigma"])
    if model["gaussian_sigma"] <= 0:
        raise ValueError("model.gaussian_sigma must be positive")
    cfg["model"] = model

    clustering = {**DEFAULT_CLUSTERING, **_mapping(cfg.get("clustering"), "clustering")}
    _reject(clustering, set(DEFAULT_CLUSTERING), "clustering")
    _fixed(clustering, DEFAULT_CLUSTERING, tuple(DEFAULT_CLUSTERING), "clustering")
    cfg["clustering"] = clustering

    training = {**DEFAULT_TRAINING, **_mapping(cfg.get("training"), "training")}
    _reject(training, set(DEFAULT_TRAINING), "training")
    for key in (
        "epochs_per_scale", "batch_size", "passes_per_epoch",
        "quantization_finetune_epochs", "quantization_finetune_passes_per_epoch",
        "save_every", "log_every", "progress_log_seconds", "seed",
    ):
        training[key] = int(training[key])
        if training[key] < 0:
            raise ValueError(f"training.{key} must be non-negative")
    training["save_intermediate_checkpoints"] = bool(
        training["save_intermediate_checkpoints"]
    )
    for key in ("batch_size", "passes_per_epoch", "quantization_finetune_passes_per_epoch"):
        if training[key] == 0:
            raise ValueError(f"training.{key} must be positive")
    for key in (
        "lr", "beta_1", "beta_2", "weight_decay", "pruning_loss_weight",
        "pruning_lr_gamma", "quantization_finetune_lr",
    ):
        training[key] = float(training[key])
    pruning_epochs = [int(value) for value in training["pruning_epochs"]]
    pruning_sparsities = [float(value) for value in training["pruning_sparsities"]]
    if len(pruning_epochs) != len(pruning_sparsities):
        raise ValueError("pruning_epochs and pruning_sparsities must have equal length")
    if pruning_epochs != sorted(set(pruning_epochs)):
        raise ValueError("pruning_epochs must be strictly increasing")
    if any(not 0.0 <= value <= 1.0 for value in pruning_sparsities):
        raise ValueError("pruning_sparsities must be in [0,1]")
    if pruning_sparsities != sorted(pruning_sparsities):
        raise ValueError("pruning_sparsities must be cumulative and non-decreasing")
    training["pruning_epochs"] = pruning_epochs
    training["pruning_sparsities"] = pruning_sparsities
    cfg["training"] = training

    quantization = {**DEFAULT_QUANTIZATION, **_mapping(cfg.get("quantization"), "quantization")}
    _reject(quantization, set(DEFAULT_QUANTIZATION), "quantization")
    _fixed(quantization, DEFAULT_QUANTIZATION, tuple(DEFAULT_QUANTIZATION), "quantization")
    cfg["quantization"] = quantization

    cnn = {**DEFAULT_CNN, **_mapping(cfg.get("cnn"), "cnn")}
    _reject(cnn, set(DEFAULT_CNN), "cnn")
    structural_cnn = tuple(key for key in DEFAULT_CNN if key not in {"tile_core_shape_zyx", "epochs", "lr"})
    _fixed(cnn, DEFAULT_CNN, structural_cnn, "cnn")
    cnn["tile_core_shape_zyx"] = [int(value) for value in cnn["tile_core_shape_zyx"]]
    if len(cnn["tile_core_shape_zyx"]) != 3 or any(value <= 0 for value in cnn["tile_core_shape_zyx"]):
        raise ValueError("cnn.tile_core_shape_zyx must contain three positive integers")
    cnn["epochs"] = int(cnn["epochs"])
    cnn["lr"] = float(cnn["lr"])
    if cnn["epochs"] < 0 or cnn["lr"] <= 0:
        raise ValueError("cnn.epochs must be non-negative and cnn.lr positive")
    cfg["cnn"] = cnn

    evaluation = {**DEFAULT_EVALUATION, **_mapping(cfg.get("evaluation"), "evaluation")}
    _reject(evaluation, set(DEFAULT_EVALUATION), "evaluation")
    evaluation["batch_size"] = int(evaluation["batch_size"])
    if evaluation["default_model"] != "checkpoint":
        raise ValueError("evaluation.default_model must be checkpoint")
    cfg["evaluation"] = evaluation
    log = {**DEFAULT_LOG, **_mapping(cfg.get("log"), "log")}
    _reject(log, set(DEFAULT_LOG), "log")
    cfg["log"] = log
    cfg["experiment"] = str(cfg.get("experiment") or f"ecnr_{selected}")
    cfg["exp_id"] = str(cfg.get("exp_id") or f"ecnr-{selected}")
    cfg["experiment_root"] = str(
        resolve_path(str(cfg.get("experiment_root") or "runs"), base_dir=config_path.parent)
    )
    cfg["CONFIG_PATH"] = str(config_path)
    return cfg


def save_config(cfg: dict[str, Any], path: str | Path) -> Path:
    payload = deepcopy(cfg)
    payload.pop("CONFIG_PATH", None)
    return dump_yaml(path, payload)
