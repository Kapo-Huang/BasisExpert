from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

from ..utils.io import load_yaml, resolve_path

TARGET_PLACEHOLDER = "{target}"
PATH_KEYS = {
    "experiment_root",
    "manager_pt_path",
    "source_path",
    "target_path",
    "target_stats_path",
    "cache_path",
}


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
    data_cfg = dict(cfg.get("DATA", {}))
    targets = data_cfg.get("targets")
    selected_target = str(target_override or data_cfg.get("target") or "").strip()

    if targets:
        if not selected_target:
            raise ValueError("DATA.targets requires DATA.target or --target")
        if selected_target not in targets:
            available = ", ".join(sorted(map(str, targets.keys())))
            raise ValueError(f"Unknown target '{selected_target}'. Available targets: {available}")
        data_cfg["target"] = selected_target
        data_cfg["attr_name"] = selected_target
        data_cfg["target_path"] = targets[selected_target]
    elif data_cfg.get("target_path"):
        if selected_target:
            data_cfg["target"] = selected_target
            data_cfg["attr_name"] = selected_target
    else:
        raise ValueError("DATA must provide target_path or targets")

    data_cfg["dataset_name"] = str(data_cfg.get("dataset_name", "")).strip().lower()
    cfg["DATA"] = data_cfg
    return cfg


def load_config(config_path: str | Path, *, target_override: str | None = None, identifier: str | None = None) -> dict[str, Any]:
    path = Path(config_path).resolve()
    cfg = deepcopy(load_yaml(path))
    cfg = _normalize_target_selection(cfg, target_override=target_override)
    selected_target = cfg["DATA"].get("target")
    cfg = _replace_target_placeholder(cfg, selected_target)
    cfg = _resolve_paths(cfg, base_dir=path.parent)

    if identifier:
        cfg["exp_id"] = str(identifier)
    else:
        cfg["exp_id"] = str(cfg.get("exp_id") or f"neural-expert-{cfg['DATA']['dataset_name']}")
    cfg["experiment"] = str(cfg.get("experiment") or cfg["exp_id"])
    cfg["experiment_root"] = str(cfg.get("experiment_root") or "runs/neural_expert")
    cfg["CONFIG_PATH"] = str(path)
    return cfg


def run_dir_from_config(cfg: dict[str, Any]) -> Path:
    return Path(cfg["experiment_root"]) / str(cfg["exp_id"])
