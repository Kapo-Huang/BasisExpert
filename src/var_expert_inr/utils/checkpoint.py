from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def read_checkpoint_payload(path: str | Path):
    checkpoint_path = Path(path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)
    try:
        return torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(checkpoint_path, map_location="cpu")


def save_checkpoint(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    scheduler: Any,
    dataset,
    epoch: int,
    config_hash: str,
    path: str | Path,
) -> Path:
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "epoch": int(epoch),
        "target_names_order": list(dataset.target_names()),
        "config_hash": str(config_hash),
    }
    torch.save(payload, checkpoint_path)
    return checkpoint_path


def load_checkpoint(
    *,
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any = None,
    strict: bool = True,
):
    checkpoint_path = Path(path)
    payload = read_checkpoint_payload(checkpoint_path)
    model.load_state_dict(payload["model_state"], strict=strict)
    if optimizer is not None and payload.get("optimizer_state") is not None:
        optimizer.load_state_dict(payload["optimizer_state"])
    if scheduler is not None and payload.get("scheduler_state") is not None:
        scheduler.load_state_dict(payload["scheduler_state"])
    return payload


def validate_checkpoint_target_order(payload, current_target_names) -> None:
    expected = [str(name) for name in current_target_names]
    actual = [str(name) for name in payload.get("target_names_order", [])]
    if not actual or not expected:
        return
    if actual != expected:
        raise ValueError(
            "Checkpoint target order mismatch. "
            f"checkpoint={actual} current={expected}"
        )
