from __future__ import annotations

from ..config.schema import DataConfig
from .node import NodeFieldDataset
from .volume import VolumeFieldDataset

SINGLE_TARGET_SELECTOR_MODELS = {"siren", "coordnet", "moe_inr"}

def build_dataset(data_cfg: DataConfig, *, model_name: str | None = None):
    selected_targets = data_cfg.targets
    target_path = data_cfg.target_path
    if data_cfg.target is not None:
        if model_name is None:
            raise ValueError("data.target requires build_dataset(..., model_name=...)")
        normalized_model_name = str(model_name).strip().lower()
        if normalized_model_name not in SINGLE_TARGET_SELECTOR_MODELS:
            allowed = ", ".join(sorted(SINGLE_TARGET_SELECTOR_MODELS))
            raise ValueError(
                f"data.target is only supported for single-target models: {allowed}; "
                f"got model_name={model_name!r}"
            )
        if not data_cfg.targets or data_cfg.target not in data_cfg.targets:
            available = ", ".join(sorted(data_cfg.targets or {}))
            raise ValueError(
                f"data.target={data_cfg.target!r} was not found in data.targets. "
                f"Available targets: {available}"
            )
        selected_targets = {data_cfg.target: data_cfg.targets[data_cfg.target]}
        target_path = None
    if data_cfg.kind == "node":
        return NodeFieldDataset(
            coords_path=str(data_cfg.coords_path),
            target_path=target_path,
            targets=selected_targets,
        )
    return VolumeFieldDataset(
        target_path=target_path,
        targets=selected_targets,
        volume_shape=data_cfg.volume_shape,
    )
