from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from .hash_grid import (
    COHERENT_PRIMES,
    UINT32_MASK,
    MultiresolutionHashEncoding,
    coherent_prime_hash as _coherent_prime_hash,
)

INSTANT_NGP_IN_FEATURES = 4
INSTANT_NGP_OUT_FEATURES = 1
INSTANT_NGP_LEVELS = 16
INSTANT_NGP_FEATURES_PER_LEVEL = 2
INSTANT_NGP_BASE_RESOLUTION = 16
INSTANT_NGP_FINEST_RESOLUTION = 600
INSTANT_NGP_LOG2_HASHMAP_SIZE = 19
INSTANT_NGP_HIDDEN_FEATURES = 64
INSTANT_NGP_HIDDEN_LAYERS = 2
INSTANT_NGP_DECODER_L2_WEIGHT = 1.0e-6


def coherent_prime_hash(vertices: torch.Tensor) -> torch.Tensor:
    if vertices.ndim < 1 or vertices.shape[-1] != 4:
        raise ValueError(
            "CoherentPrime hash expects vertices with four trailing coordinates"
        )
    return _coherent_prime_hash(vertices)


class MultiresolutionHashEncoding4D(MultiresolutionHashEncoding):
    def __init__(
        self,
        *,
        n_levels: int,
        n_features_per_level: int,
        base_resolution: int,
        finest_resolution: int,
        log2_hashmap_size: int,
    ) -> None:
        super().__init__(
            dimensions=4,
            n_levels=n_levels,
            n_features_per_level=n_features_per_level,
            base_resolution=base_resolution,
            finest_resolution=finest_resolution,
            log2_hashmap_size=log2_hashmap_size,
        )

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        if coords.ndim != 2 or coords.shape[1] != 4:
            raise ValueError(
                f"InstantNGP encoding expects [B, 4] coordinates, got {tuple(coords.shape)}"
            )
        return super().forward(coords)


class InstantNGP(nn.Module):
    def __init__(
        self,
        *,
        in_features: int = INSTANT_NGP_IN_FEATURES,
        out_features: int = INSTANT_NGP_OUT_FEATURES,
        n_levels: int = INSTANT_NGP_LEVELS,
        n_features_per_level: int = INSTANT_NGP_FEATURES_PER_LEVEL,
        base_resolution: int = INSTANT_NGP_BASE_RESOLUTION,
        finest_resolution: int = INSTANT_NGP_FINEST_RESOLUTION,
        log2_hashmap_size: int = INSTANT_NGP_LOG2_HASHMAP_SIZE,
        hidden_features: int = INSTANT_NGP_HIDDEN_FEATURES,
        hidden_layers: int = INSTANT_NGP_HIDDEN_LAYERS,
    ) -> None:
        super().__init__()
        actual = {
            "in_features": int(in_features),
            "out_features": int(out_features),
            "n_levels": int(n_levels),
            "n_features_per_level": int(n_features_per_level),
            "base_resolution": int(base_resolution),
            "finest_resolution": int(finest_resolution),
            "log2_hashmap_size": int(log2_hashmap_size),
            "hidden_features": int(hidden_features),
            "hidden_layers": int(hidden_layers),
        }
        expected = {
            "in_features": INSTANT_NGP_IN_FEATURES,
            "out_features": INSTANT_NGP_OUT_FEATURES,
            "n_levels": INSTANT_NGP_LEVELS,
            "n_features_per_level": INSTANT_NGP_FEATURES_PER_LEVEL,
            "base_resolution": INSTANT_NGP_BASE_RESOLUTION,
            "finest_resolution": INSTANT_NGP_FINEST_RESOLUTION,
            "log2_hashmap_size": INSTANT_NGP_LOG2_HASHMAP_SIZE,
            "hidden_features": INSTANT_NGP_HIDDEN_FEATURES,
            "hidden_layers": INSTANT_NGP_HIDDEN_LAYERS,
        }
        mismatches = [
            f"{key}={actual[key]} (expected {value})"
            for key, value in expected.items()
            if actual[key] != value
        ]
        if mismatches:
            raise ValueError(
                "InstantNGP uses a fixed architecture: " + ", ".join(mismatches)
            )

        self.in_features = actual["in_features"]
        self.out_features = actual["out_features"]
        self.encoding = MultiresolutionHashEncoding4D(
            n_levels=actual["n_levels"],
            n_features_per_level=actual["n_features_per_level"],
            base_resolution=actual["base_resolution"],
            finest_resolution=actual["finest_resolution"],
            log2_hashmap_size=actual["log2_hashmap_size"],
        )
        self.decoder = nn.Sequential(
            nn.Linear(
                self.encoding.output_dim,
                actual["hidden_features"],
                bias=False,
            ),
            nn.ReLU(),
            nn.Linear(
                actual["hidden_features"],
                actual["hidden_features"],
                bias=False,
            ),
            nn.ReLU(),
            nn.Linear(
                actual["hidden_features"],
                self.out_features,
                bias=False,
            ),
        )
        for module in self.decoder:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)

    def forward(self, coords: torch.Tensor, **_: Any) -> torch.Tensor:
        if coords.ndim != 2 or coords.shape[1] != self.in_features:
            raise ValueError(
                f"InstantNGP expects [B, 4] coordinates, got {tuple(coords.shape)}"
            )
        encoded = self.encoding(coords)
        return self.decoder(encoded).float()

    def decoder_l2_regularization(self) -> torch.Tensor:
        regularization = None
        for parameter in self.decoder.parameters():
            value = parameter.float().square().sum()
            regularization = (
                value
                if regularization is None
                else regularization + value
            )
        if regularization is None:
            raise RuntimeError("InstantNGP decoder has no trainable parameters")
        return regularization


def build_instant_ngp_from_config(cfg: dict[str, Any]) -> InstantNGP:
    return InstantNGP(**cfg)
