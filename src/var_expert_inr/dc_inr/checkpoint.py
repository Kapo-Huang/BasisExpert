from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ..utils.checkpoint import read_checkpoint_payload

logger = logging.getLogger(__name__)


def save_dc_checkpoint(path: str | Path, payload: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    current_payload = dict(payload)
    for _ in range(4):
        torch.save(current_payload, checkpoint_path)
        file_bytes = int(checkpoint_path.stat().st_size)
        if int(current_payload.get("checkpoint_bytes", -1)) == file_bytes:
            current_payload["checkpoint_bytes"] = file_bytes
            return checkpoint_path, current_payload
        current_payload["checkpoint_bytes"] = file_bytes
    torch.save(current_payload, checkpoint_path)
    current_payload["checkpoint_bytes"] = int(checkpoint_path.stat().st_size)
    return checkpoint_path, current_payload


def load_dc_checkpoint(path: str | Path) -> dict[str, Any]:
    return read_checkpoint_payload(path)


def validate_dc_checkpoint(
    payload: dict[str, Any],
    *,
    expected_target_name: str | None = None,
    expected_volume_shape: dict[str, int] | None = None,
    expected_config_hash: str | None = None,
) -> None:
    model_name = str(payload.get("model_name", "")).strip().lower()
    if model_name != "dc_inr":
        raise ValueError(f"Unsupported DC-INR checkpoint model_name: {payload.get('model_name')!r}")

    target_name = str(payload.get("target_name", "") or "")
    if expected_target_name is not None and target_name != str(expected_target_name):
        raise ValueError(
            f"DC-INR checkpoint target mismatch: checkpoint={target_name!r} expected={expected_target_name!r}"
        )

    volume_shape = payload.get("volume_shape")
    if not isinstance(volume_shape, dict):
        raise ValueError("DC-INR checkpoint is missing volume_shape metadata")
    if expected_volume_shape is not None:
        current_shape = {str(key): int(value) for key, value in volume_shape.items()}
        wanted_shape = {str(key): int(value) for key, value in expected_volume_shape.items()}
        if current_shape != wanted_shape:
            raise ValueError(
                f"DC-INR checkpoint volume shape mismatch: checkpoint={current_shape} expected={wanted_shape}"
            )

    block_shape = payload.get("block_shape")
    block_grid_shape = payload.get("block_grid_shape")
    if not isinstance(block_shape, dict) or not isinstance(block_grid_shape, dict):
        raise ValueError("DC-INR checkpoint is missing block partition metadata")

    assignments = np.asarray(payload.get("block_to_representative"))
    representatives = np.asarray(payload.get("representative_block_ids"))
    widths = np.asarray(payload.get("model_widths"))
    model_states = payload.get("model_states")
    if assignments.ndim != 1:
        raise ValueError("DC-INR checkpoint block_to_representative must be a 1D array")
    if representatives.ndim != 1 or widths.ndim != 1:
        raise ValueError("DC-INR checkpoint representative metadata must be 1D")
    if representatives.size != widths.size:
        raise ValueError("DC-INR checkpoint representative_block_ids and model_widths size mismatch")
    if not isinstance(model_states, list) or len(model_states) != int(widths.size):
        raise ValueError("DC-INR checkpoint model_states size mismatch")
    if assignments.size > 0:
        if int(assignments.min()) < 0 or int(assignments.max()) >= int(widths.size):
            raise ValueError("DC-INR checkpoint block_to_representative contains invalid indices")

    if expected_config_hash is not None:
        actual = str(payload.get("config_hash", "") or "")
        if actual and actual != str(expected_config_hash):
            logger.warning(
                "DC-INR checkpoint config hash mismatch: checkpoint=%s current=%s",
                actual,
                expected_config_hash,
            )
