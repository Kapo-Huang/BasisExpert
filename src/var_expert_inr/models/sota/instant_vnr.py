from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from .hash_grid import MultiresolutionHashEncoding


INSTANT_VNR_IN_FEATURES = 4
INSTANT_VNR_OUT_FEATURES = 1
INSTANT_VNR_LEVELS = 8
INSTANT_VNR_FEATURES_PER_LEVEL = 8
INSTANT_VNR_BASE_RESOLUTION = 16
INSTANT_VNR_PER_LEVEL_SCALE = 2.0
INSTANT_VNR_LOG2_HASHMAP_SIZE = 19
INSTANT_VNR_HIDDEN_FEATURES = 64
INSTANT_VNR_HIDDEN_LAYERS = 4


class InstantVNR(nn.Module):
    """Pure-PyTorch 4D extension of InstantVNR's HashGrid representation."""

    def __init__(
        self,
        *,
        in_features: int = INSTANT_VNR_IN_FEATURES,
        out_features: int = INSTANT_VNR_OUT_FEATURES,
        n_levels: int = INSTANT_VNR_LEVELS,
        n_features_per_level: int = INSTANT_VNR_FEATURES_PER_LEVEL,
        base_resolution: int = INSTANT_VNR_BASE_RESOLUTION,
        per_level_scale: float = INSTANT_VNR_PER_LEVEL_SCALE,
        log2_hashmap_size: int = INSTANT_VNR_LOG2_HASHMAP_SIZE,
        hidden_features: int = INSTANT_VNR_HIDDEN_FEATURES,
        hidden_layers: int = INSTANT_VNR_HIDDEN_LAYERS,
    ) -> None:
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.hidden_features = int(hidden_features)
        self.hidden_layers = int(hidden_layers)

        if self.in_features != INSTANT_VNR_IN_FEATURES:
            raise ValueError(
                f"InstantVNR requires in_features={INSTANT_VNR_IN_FEATURES}, "
                f"got {self.in_features}"
            )
        if self.out_features != INSTANT_VNR_OUT_FEATURES:
            raise ValueError(
                f"InstantVNR requires out_features={INSTANT_VNR_OUT_FEATURES}, "
                f"got {self.out_features}"
            )
        if self.hidden_features <= 0:
            raise ValueError("hidden_features must be positive")
        if self.hidden_layers <= 0:
            raise ValueError("hidden_layers must be positive")

        self.encoding = MultiresolutionHashEncoding(
            dimensions=self.in_features,
            n_levels=int(n_levels),
            n_features_per_level=int(n_features_per_level),
            base_resolution=int(base_resolution),
            per_level_scale=float(per_level_scale),
            log2_hashmap_size=int(log2_hashmap_size),
        )

        layers: list[nn.Module] = []
        input_width = self.encoding.output_dim
        for _ in range(self.hidden_layers):
            layer = nn.Linear(input_width, self.hidden_features, bias=False)
            nn.init.xavier_uniform_(layer.weight)
            layers.extend((layer, nn.ReLU()))
            input_width = self.hidden_features
        output = nn.Linear(input_width, self.out_features, bias=False)
        nn.init.xavier_uniform_(output.weight)
        layers.append(output)
        self.decoder = nn.Sequential(*layers)

    def forward(self, coords: torch.Tensor, **_: Any) -> torch.Tensor:
        if coords.ndim != 2 or coords.shape[1] != self.in_features:
            raise ValueError(
                f"InstantVNR expects [B, {self.in_features}] coordinates, "
                f"got {tuple(coords.shape)}"
            )
        return self.decoder(self.encoding(coords)).float()


def build_instant_vnr_from_config(cfg: dict[str, Any]) -> InstantVNR:
    return InstantVNR(**cfg)
