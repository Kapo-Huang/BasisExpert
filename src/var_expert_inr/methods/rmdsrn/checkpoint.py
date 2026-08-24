from __future__ import annotations

from pathlib import Path

import torch

CHECKPOINT_FORMAT = "rmdsrn_inference_v1"


def _torch_load(path: str | Path) -> dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _model_state_cpu(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        key: value.detach().float().cpu()
        for key, value in model.state_dict().items()
    }


def save_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    cfg: dict,
    config_hash: str,
) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": CHECKPOINT_FORMAT,
        "model_name": "rmdsrn",
        "model_state": _model_state_cpu(model),
        "target_name": str(cfg["data"]["target"]),
        "volume_shape": dict(cfg["data"]["volume_shape"]),
        "model_config": dict(cfg["model"]),
        "config_hash": str(config_hash),
    }
    torch.save(payload, output)
    return output


def load_checkpoint(path: str | Path) -> dict:
    payload = _torch_load(path)
    if payload.get("format") != CHECKPOINT_FORMAT:
        raise ValueError(f"Unsupported RMDSRN inference checkpoint: {payload.get('format')!r}")
    return payload


def validate_payload(payload: dict, cfg: dict, *, config_hash: str | None = None) -> None:
    if payload.get("model_name") != "rmdsrn":
        raise ValueError("RMDSRN model name mismatch")
    if payload.get("target_name") != cfg["data"]["target"]:
        raise ValueError("RMDSRN target mismatch")
    if payload.get("volume_shape") != cfg["data"]["volume_shape"]:
        raise ValueError("RMDSRN volume shape mismatch")
    if payload.get("model_config") != cfg["model"]:
        raise ValueError("RMDSRN model configuration mismatch")
    if config_hash is not None and payload.get("config_hash") != config_hash:
        raise ValueError("RMDSRN config hash mismatch")
