from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..fv_srn.model import SnakeAlt, nerf_fourier_matrix


class TemporalFeatureGridEncoder(nn.Module):
    """Temporal fV-SRN encoder with linearly interpolated keyframe grids."""

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
            )
            * float(model_cfg["grid_init_std"])
        )
        self.output_features = 3 + 2 * fourier_count + self.grid_channels

    def keyframe_pair(self, timestep: float) -> tuple[int, int, float]:
        value = float(timestep)
        if value <= self.keyframe_indices[0]:
            return 0, 0, 0.0
        if value >= self.keyframe_indices[-1]:
            last = len(self.keyframe_indices) - 1
            return last, last, 0.0
        right = next(index for index, frame in enumerate(self.keyframe_indices) if frame >= value)
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

    def forward(self, coords: torch.Tensor, timestep: float | int) -> torch.Tensor:
        phases = coords @ self.fourier_matrix.transpose(0, 1)
        return torch.cat(
            [
                coords,
                torch.cos(phases),
                torch.sin(phases),
                self.grid_features(coords, float(timestep)),
            ],
            dim=1,
        )


class RMDSRNDecoder(nn.Sequential):
    def __init__(self, input_features: int, model_cfg: dict) -> None:
        hidden_features = int(model_cfg["decoder_hidden_features"])
        hidden_layers = int(model_cfg["decoder_hidden_layers"])
        frequency = float(model_cfg["activation_frequency"])
        layers: list[nn.Module] = []
        current_features = int(input_features)
        for _ in range(hidden_layers):
            layers.extend(
                [
                    nn.Linear(current_features, hidden_features),
                    SnakeAlt(frequency),
                ]
            )
            current_features = hidden_features
        layers.append(nn.Linear(current_features, 1))
        super().__init__(*layers)


class RMDSRN(nn.Module):
    """Shared temporal feature-grid encoder with independent MLP decoders."""

    def __init__(self, model_cfg: dict) -> None:
        super().__init__()
        self.encoder = TemporalFeatureGridEncoder(model_cfg)
        self.decoder_count = int(model_cfg["decoder_count"])
        self.decoders = nn.ModuleList(
            [
                RMDSRNDecoder(self.encoder.output_features, model_cfg)
                for _ in range(self.decoder_count)
            ]
        )

    def encode(self, coords: torch.Tensor, timestep: float | int) -> torch.Tensor:
        return self.encoder(coords, timestep)

    def forward_members(self, coords: torch.Tensor, timestep: float | int) -> torch.Tensor:
        encoded = self.encode(coords, timestep)
        return torch.stack([decoder(encoded) for decoder in self.decoders], dim=1)

    @staticmethod
    def ensemble_statistics(member_predictions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if member_predictions.ndim != 3:
            raise ValueError(
                "member_predictions must have shape [batch, members, channels], "
                f"got {tuple(member_predictions.shape)}"
            )
        if int(member_predictions.shape[1]) < 2:
            raise ValueError("At least two member predictions are required for unbiased variance")
        mean = member_predictions.mean(dim=1)
        variance = member_predictions.var(dim=1, unbiased=True)
        return mean, variance

    def forward(self, coords: torch.Tensor, timestep: float | int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.ensemble_statistics(self.forward_members(coords, timestep))
