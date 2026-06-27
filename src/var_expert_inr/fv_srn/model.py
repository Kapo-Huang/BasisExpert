from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SnakeAlt(nn.Module):
    def __init__(self, frequency: float = 1.0) -> None:
        super().__init__()
        self.frequency = float(frequency)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return (inputs + 1.0 - torch.cos(2.0 * self.frequency * inputs)) / (2.0 * self.frequency)


def nerf_fourier_matrix(count: int) -> torch.Tensor:
    blocks = [float(2**index) * torch.eye(3) for index in range(int(math.ceil(count / 3)))]
    return torch.cat(blocks, dim=0)[:count] * (2.0 * math.pi)


class TemporalFVSRN(nn.Module):
    def __init__(self, model_cfg: dict) -> None:
        super().__init__()
        self.grid_resolution = int(model_cfg["grid_resolution"])
        self.grid_channels = int(model_cfg["grid_channels"])
        self.keyframe_indices = tuple(int(value) for value in model_cfg["keyframe_indices"])
        fourier_count = int(model_cfg["fourier_features"])
        self.register_buffer("fourier_matrix", nerf_fourier_matrix(fourier_count))
        self.feature_grids = nn.Parameter(
            torch.randn(
                len(self.keyframe_indices),
                self.grid_channels,
                self.grid_resolution,
                self.grid_resolution,
                self.grid_resolution,
            ) * float(model_cfg["grid_init_std"])
        )
        input_channels = 3 + 2 * fourier_count + self.grid_channels
        hidden = int(model_cfg["hidden_features"])
        hidden_layers = int(model_cfg["hidden_layers"])
        modules: list[nn.Module] = []
        current = input_channels
        for _ in range(hidden_layers):
            modules.extend(
                [
                    nn.Linear(current, hidden),
                    SnakeAlt(float(model_cfg["activation_frequency"])),
                ]
            )
            current = hidden
        modules.append(nn.Linear(current, 1))
        self.mlp = nn.Sequential(*modules)

    def keyframe_pair(self, timestep: float) -> tuple[int, int, float]:
        value = float(timestep)
        if value <= self.keyframe_indices[0]:
            return 0, 0, 0.0
        if value >= self.keyframe_indices[-1]:
            last = len(self.keyframe_indices) - 1
            return last, last, 0.0
        right = next(i for i, frame in enumerate(self.keyframe_indices) if frame >= value)
        if self.keyframe_indices[right] == value:
            return right, right, 0.0
        left = right - 1
        alpha = (value - self.keyframe_indices[left]) / (
            self.keyframe_indices[right] - self.keyframe_indices[left]
        )
        return left, right, float(alpha)

    @staticmethod
    def _sample_grid(grid: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        query = coords.mul(2.0).sub(1.0).view(1, 1, 1, -1, 3)
        sampled = F.grid_sample(
            grid.unsqueeze(0),
            query,
            mode="bilinear",
            padding_mode="border",
            align_corners=False,
        )
        return sampled[0, :, 0, 0, :].transpose(0, 1)

    def grid_features(self, coords: torch.Tensor, timestep: float) -> torch.Tensor:
        left, right, alpha = self.keyframe_pair(timestep)
        low = self._sample_grid(self.feature_grids[left], coords)
        if left == right:
            return low
        high = self._sample_grid(self.feature_grids[right], coords)
        return low * (1.0 - alpha) + high * alpha

    def encode(self, coords: torch.Tensor, timestep: float) -> torch.Tensor:
        phases = coords @ self.fourier_matrix.transpose(0, 1)
        return torch.cat(
            [coords, torch.cos(phases), torch.sin(phases), self.grid_features(coords, timestep)],
            dim=1,
        )

    def forward(self, coords: torch.Tensor, timestep: float | int) -> torch.Tensor:
        return self.mlp(self.encode(coords, float(timestep)))
