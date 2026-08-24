from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
import yaml


def estimate_model_size_fp16(model) -> tuple[int, int]:
    parameter_count = sum(param.numel() for param in model.parameters())
    size_bytes = parameter_count * 2
    return parameter_count, size_bytes


def format_size_bytes(size_bytes: int) -> str:
    size_mib = size_bytes / (1024**2)
    size_mb = size_bytes / 1.0e6
    return f"{size_bytes} bytes ({size_mib:.2f} MiB, {size_mb:.2f} MB)"


def to_device(data, device: torch.device):
    out = {}
    for key, value in data.items():
        if torch.is_tensor(value):
            out[key] = value.to(device, non_blocking=True)
        else:
            out[key] = value
    return out


def load_state_dict_payload(path: str | Path, device: torch.device | str = "cpu"):
    try:
        payload = torch.load(str(path), map_location=device, weights_only=False)
    except TypeError:
        payload = torch.load(str(path), map_location=device)
    if not isinstance(payload, dict) or payload.get("format") != "neural_expert_inference_v1":
        checkpoint_format = payload.get("format") if isinstance(payload, dict) else None
        raise ValueError(f"Unsupported NeuralExpert checkpoint: {checkpoint_format!r}")
    return payload["model_state"]


def dump_config(cfg: dict, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
