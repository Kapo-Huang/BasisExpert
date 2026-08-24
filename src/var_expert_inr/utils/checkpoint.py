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
    dataset,
    config_hash: str,
    path: str | Path,
) -> Path:
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": "inference_checkpoint_v1",
        "model_state": model.state_dict(),
        "target_names_order": list(dataset.target_names()),
        "target_dims_order": [
            int(dataset.meta.target_dims[name])
            for name in dataset.target_names()
        ],
        "config_hash": str(config_hash),
    }
    torch.save(payload, checkpoint_path)
    return checkpoint_path


def load_checkpoint(
    *,
    path: str | Path,
    model: torch.nn.Module,
    strict: bool = True,
):
    checkpoint_path = Path(path)
    payload = read_checkpoint_payload(checkpoint_path)
    if payload.get("format") != "inference_checkpoint_v1":
        raise ValueError(f"Unsupported inference checkpoint: {payload.get('format')!r}")
    model.load_state_dict(payload["model_state"], strict=strict)
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


def validate_checkpoint_target_layout(payload, current_target_names, current_target_dims) -> None:
    validate_checkpoint_target_order(payload, current_target_names)
    actual = payload.get("target_dims_order")
    if actual is None:
        return
    expected = [
        int(current_target_dims[name])
        for name in current_target_names
    ]
    actual = [int(value) for value in actual]
    if actual != expected:
        raise ValueError(
            "Checkpoint target dimensions mismatch. "
            f"checkpoint={actual} current={expected}"
        )
