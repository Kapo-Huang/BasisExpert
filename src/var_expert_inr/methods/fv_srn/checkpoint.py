from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch

from .data import SamplePool


def save_checkpoint(
    path: str | Path,
    *,
    model,
    optimizer,
    scheduler,
    epoch: int,
    cfg: dict,
    config_hash: str,
    sample_pool: SamplePool,
    sampler_rng_state: dict,
) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": "fv_srn_checkpoint_v1",
        "model_name": "fv_srn",
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "epoch": int(epoch),
        "target_name": cfg["data"]["target"],
        "volume_shape": dict(cfg["data"]["volume_shape"]),
        "model_config": dict(cfg["model"]),
        "config_hash": str(config_hash),
        "sample_pool": sample_pool.state_dict(),
        "sampler_rng_state": sampler_rng_state,
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "numpy_rng_state": np.random.get_state(),
        "python_rng_state": random.getstate(),
    }
    torch.save(payload, output)
    return output


def load_payload(path: str | Path) -> dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def validate(payload: dict, cfg: dict, *, config_hash: str | None = None) -> None:
    if payload.get("format") != "fv_srn_checkpoint_v1":
        raise ValueError("Unsupported fV-SRN checkpoint")
    if payload.get("target_name") != cfg["data"]["target"]:
        raise ValueError("fV-SRN checkpoint target mismatch")
    if payload.get("volume_shape") != cfg["data"]["volume_shape"]:
        raise ValueError("fV-SRN checkpoint volume shape mismatch")
    if config_hash is not None and payload.get("config_hash") != config_hash:
        raise ValueError("fV-SRN checkpoint config hash mismatch")


def restore_rng(payload: dict) -> None:
    torch.set_rng_state(payload["torch_rng_state"])
    if torch.cuda.is_available() and payload.get("cuda_rng_state") is not None:
        torch.cuda.set_rng_state_all(payload["cuda_rng_state"])
    np.random.set_state(payload["numpy_rng_state"])
    random.setstate(payload["python_rng_state"])
