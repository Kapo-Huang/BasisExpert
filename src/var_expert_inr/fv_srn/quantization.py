from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import SnakeAlt


def quantize_grids(grids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    source = grids.detach().float().cpu()
    minimum = source.amin(dim=(2, 3, 4), keepdim=True)
    maximum = source.amax(dim=(2, 3, 4), keepdim=True)
    scale = (maximum - minimum) / 255.0
    safe_scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    quantized = torch.round((source - minimum) / safe_scale).clamp(0, 255).to(torch.uint8)
    scale = torch.where(scale > 0, scale, torch.zeros_like(scale))
    return quantized, minimum, scale


def dequantize_grid(
    quantized: torch.Tensor,
    minimum: torch.Tensor,
    scale: torch.Tensor,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    return quantized.to(device=device, dtype=dtype) * scale.to(device=device, dtype=dtype) + minimum.to(
        device=device, dtype=dtype
    )


class CompactFVSRN(nn.Module):
    def __init__(self, payload: dict, *, device: torch.device) -> None:
        super().__init__()
        cfg = payload["model_config"]
        self.keyframe_indices = tuple(int(v) for v in cfg["keyframe_indices"])
        self.compute_dtype = torch.float16 if device.type == "cuda" else torch.float32
        self.register_buffer("fourier_matrix", payload["fourier_matrix"].to(device=device, dtype=self.compute_dtype))
        self.register_buffer("quantized_grids", payload["quantized_grids"].to(device=device))
        self.register_buffer("grid_minimum", payload["grid_minimum"].to(device=device))
        self.register_buffer("grid_scale", payload["grid_scale"].to(device=device))
        input_channels = 3 + 2 * int(cfg["fourier_features"]) + int(cfg["grid_channels"])
        hidden = int(cfg["hidden_features"])
        modules: list[nn.Module] = []
        current = input_channels
        for _ in range(int(cfg["hidden_layers"])):
            modules.extend([nn.Linear(current, hidden), SnakeAlt(float(cfg["activation_frequency"]))])
            current = hidden
        modules.append(nn.Linear(current, 1))
        self.mlp = nn.Sequential(*modules).to(device=device, dtype=self.compute_dtype)
        state = {key: value.to(dtype=self.compute_dtype) for key, value in payload["mlp_state"].items()}
        self.mlp.load_state_dict(state)

    def keyframe_pair(self, timestep: float) -> tuple[int, int, float]:
        if timestep <= self.keyframe_indices[0]:
            return 0, 0, 0.0
        if timestep >= self.keyframe_indices[-1]:
            last = len(self.keyframe_indices) - 1
            return last, last, 0.0
        right = next(i for i, frame in enumerate(self.keyframe_indices) if frame >= timestep)
        if self.keyframe_indices[right] == timestep:
            return right, right, 0.0
        left = right - 1
        alpha = (timestep - self.keyframe_indices[left]) / (
            self.keyframe_indices[right] - self.keyframe_indices[left]
        )
        return left, right, float(alpha)

    def _features(self, index: int, coords: torch.Tensor) -> torch.Tensor:
        grid = dequantize_grid(
            self.quantized_grids[index],
            self.grid_minimum[index],
            self.grid_scale[index],
            device=coords.device,
            dtype=self.compute_dtype,
        )
        query = coords.view(1, 1, 1, -1, 3).mul(2).sub(1)
        sampled = F.grid_sample(
            grid.unsqueeze(0), query, mode="bilinear", padding_mode="border", align_corners=False
        )
        return sampled[0, :, 0, 0, :].transpose(0, 1)

    def forward(self, coords: torch.Tensor, timestep: float | int) -> torch.Tensor:
        coords = coords.to(dtype=self.compute_dtype)
        left, right, alpha = self.keyframe_pair(float(timestep))
        features = self._features(left, coords)
        if left != right:
            features = features * (1 - alpha) + self._features(right, coords) * alpha
        phases = coords @ self.fourier_matrix.transpose(0, 1)
        encoded = torch.cat([coords, torch.cos(phases), torch.sin(phases), features], dim=1)
        return self.mlp(encoded).float()


def export_compact(
    *,
    model,
    cfg: dict,
    target_name: str,
    volume_shape: dict,
    path: str | Path,
    config_hash: str,
) -> tuple[Path, dict]:
    quantized, minimum, scale = quantize_grids(model.feature_grids)
    payload = {
        "format": "fv_srn_compact_v1",
        "model_name": "fv_srn",
        "model_config": dict(cfg["model"]),
        "target_name": str(target_name),
        "volume_shape": dict(volume_shape),
        "config_hash": str(config_hash),
        "fourier_matrix": model.fourier_matrix.detach().half().cpu(),
        "mlp_state": {key: value.detach().half().cpu() for key, value in model.mlp.state_dict().items()},
        "quantized_grids": quantized,
        "grid_minimum": minimum,
        "grid_scale": scale,
    }
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)
    payload["artifact_bytes"] = int(output.stat().st_size)
    return output, payload


def load_compact(path: str | Path, *, device: torch.device) -> tuple[CompactFVSRN, dict]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if payload.get("format") != "fv_srn_compact_v1":
        raise ValueError(f"Unsupported compact fV-SRN artifact: {payload.get('format')!r}")
    return CompactFVSRN(payload, device=device).to(device), payload
