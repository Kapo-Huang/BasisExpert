from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ...utils.checkpoint import read_checkpoint_payload, validate_checkpoint_target_order
from .data import TargetLayoutEntry, layout_from_payload, layout_to_payload

logger = logging.getLogger(__name__)
VALID_MC_STAGES = {"meta", "meta_init", "finetune", "split"}


def save_mc_checkpoint(
    *,
    path: str | Path,
    model: torch.nn.Module | None = None,
    model_state: dict[str, Any] | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any = None,
    epoch: int,
    stage: str,
    config_hash: str,
    target_names: tuple[str, ...],
    target_layout: tuple[TargetLayoutEntry, ...],
    assignments: np.ndarray,
    centroids: np.ndarray | torch.Tensor | None = None,
    best_loss: float,
    epochs_no_improve: int,
    extra_payload: dict[str, Any] | None = None,
) -> Path:
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    if model_state is None and model is not None:
        model_state = model.state_dict()
    if centroids is None:
        if model is None:
            raise ValueError("save_mc_checkpoint requires centroids when model is None")
        centroids = model.centroids.detach().cpu().numpy().astype(np.float32)
    centroids_np = np.asarray(centroids, dtype=np.float32)
    payload = {
        "model_state": model_state,
        "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "epoch": int(epoch),
        "mc_stage": str(stage),
        "config_hash": str(config_hash),
        "target_names_order": [str(name) for name in target_names],
        "data_target_order": [str(name) for name in target_names],
        "target_layout": layout_to_payload(target_layout),
        "cluster_assignments": np.asarray(assignments, dtype=np.int32),
        "centroids": centroids_np,
        "best_loss": float(best_loss),
        "epochs_no_improve": int(epochs_no_improve),
    }
    if extra_payload:
        payload.update(dict(extra_payload))
    torch.save(payload, checkpoint_path)
    return checkpoint_path


def load_mc_checkpoint(path: str | Path) -> dict[str, Any]:
    return read_checkpoint_payload(path)


def validate_mc_checkpoint(
    payload: dict[str, Any],
    current_target_names: tuple[str, ...],
    *,
    expected_assignment_shape: tuple[int, ...] | None = None,
    expected_output_dim: int | None = None,
    expected_spatial_dims: int | None = None,
    expected_config_hash: str | None = None,
) -> None:
    validate_checkpoint_target_order(payload, current_target_names)
    stage = str(payload.get("mc_stage", "") or "")
    if stage and stage not in VALID_MC_STAGES:
        raise ValueError(f"Unsupported MC-INR checkpoint stage: {stage!r}")

    layout = payload.get("target_layout")
    if not layout:
        raise ValueError("MC-INR checkpoint is missing target_layout")
    restored_layout = layout_from_payload(layout)
    layout_names = tuple(entry.name for entry in restored_layout)
    expected_names = tuple(str(name) for name in current_target_names)
    if layout_names != expected_names:
        raise ValueError(
            "MC-INR checkpoint target layout mismatch. "
            f"checkpoint={list(layout_names)} current={list(expected_names)}"
        )
    if expected_output_dim is not None:
        output_dim = sum(int(entry.dim) for entry in restored_layout)
        if output_dim != int(expected_output_dim):
            raise ValueError(
                f"MC-INR checkpoint output dim mismatch: checkpoint={output_dim} expected={int(expected_output_dim)}"
            )

    assignments = payload.get("cluster_assignments")
    if assignments is None:
        raise ValueError("MC-INR checkpoint is missing cluster_assignments")
    assignments_np = np.asarray(assignments)
    if expected_assignment_shape is not None and tuple(assignments_np.shape) != tuple(expected_assignment_shape):
        raise ValueError(
            "MC-INR checkpoint assignment shape mismatch. "
            f"checkpoint={tuple(assignments_np.shape)} expected={tuple(expected_assignment_shape)}"
        )

    centroids = payload.get("centroids")
    if centroids is None:
        raise ValueError("MC-INR checkpoint is missing centroids")
    centroids_np = np.asarray(centroids, dtype=np.float32)
    if centroids_np.ndim != 2 or centroids_np.shape[0] <= 0:
        raise ValueError(f"Invalid checkpoint centroids shape: {tuple(centroids_np.shape)}")
    if expected_spatial_dims is not None and int(centroids_np.shape[1]) != int(expected_spatial_dims):
        raise ValueError(
            "MC-INR checkpoint centroid spatial dim mismatch. "
            f"checkpoint={int(centroids_np.shape[1])} expected={int(expected_spatial_dims)}"
        )

    if stage != "meta_init" and payload.get("model_state") is None:
        raise ValueError(f"MC-INR checkpoint stage {stage!r} requires model_state")
    if stage == "meta_init" and payload.get("template_model_state") is None:
        raise ValueError("MC-INR meta_init checkpoint is missing template_model_state")

    if expected_config_hash is not None:
        actual_hash = str(payload.get("config_hash", "") or "")
        if actual_hash and actual_hash != str(expected_config_hash):
            logger.warning(
                "MC-INR checkpoint config hash mismatch: checkpoint=%s current=%s",
                actual_hash,
                expected_config_hash,
            )


def restore_target_layout(payload: dict[str, Any]) -> tuple[TargetLayoutEntry, ...]:
    layout_payload = payload.get("target_layout")
    if not layout_payload:
        raise ValueError("MC-INR checkpoint is missing target_layout")
    return layout_from_payload(layout_payload)
