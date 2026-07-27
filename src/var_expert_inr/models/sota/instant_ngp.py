from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn


UINT32_MASK = 0xFFFFFFFF
COHERENT_PRIMES = (1, 2654435761, 805459861, 3674653429)

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


def _corner_offsets(dimensions: int) -> torch.Tensor:
    corner_ids = torch.arange(1 << int(dimensions), dtype=torch.long)
    shifts = torch.arange(int(dimensions), dtype=torch.long)
    return ((corner_ids[:, None] >> shifts[None, :]) & 1).contiguous()


def coherent_prime_hash(vertices: torch.Tensor) -> torch.Tensor:
    if vertices.ndim < 1 or vertices.shape[-1] != 4:
        raise ValueError(
            "CoherentPrime hash expects vertices with four trailing coordinates"
        )
    result = torch.zeros(
        vertices.shape[:-1],
        dtype=torch.long,
        device=vertices.device,
    )
    for dim, prime in enumerate(COHERENT_PRIMES):
        term = torch.bitwise_and(vertices[..., dim] * int(prime), UINT32_MASK)
        result = torch.bitwise_xor(result, term)
    return torch.bitwise_and(result, UINT32_MASK)


class MultiresolutionHashEncoding4D(nn.Module):
    def __init__(
        self,
        *,
        n_levels: int,
        n_features_per_level: int,
        base_resolution: int,
        finest_resolution: int,
        log2_hashmap_size: int,
    ) -> None:
        super().__init__()
        self.n_levels = int(n_levels)
        self.n_features_per_level = int(n_features_per_level)
        self.base_resolution = int(base_resolution)
        self.finest_resolution = int(finest_resolution)
        self.log2_hashmap_size = int(log2_hashmap_size)
        self.max_entries_per_level = 1 << self.log2_hashmap_size

        if self.n_levels < 2:
            raise ValueError("n_levels must be at least two")
        if self.n_features_per_level <= 0:
            raise ValueError("n_features_per_level must be positive")
        if self.base_resolution < 2:
            raise ValueError("base_resolution must be at least two")
        if self.finest_resolution < self.base_resolution:
            raise ValueError(
                "finest_resolution must be greater than or equal to base_resolution"
            )
        if self.log2_hashmap_size <= 0:
            raise ValueError("log2_hashmap_size must be positive")

        self.per_level_scale = math.exp(
            (
                math.log(float(self.finest_resolution))
                - math.log(float(self.base_resolution))
            )
            / float(self.n_levels - 1)
        )
        scales = [
            float(self.base_resolution) * self.per_level_scale**level - 1.0
            for level in range(self.n_levels)
        ]
        resolutions = [int(math.ceil(scale)) + 1 for scale in scales]
        dense_entries = [int(resolution) ** 4 for resolution in resolutions]
        entries = [
            ((min(count, self.max_entries_per_level) + 7) // 8) * 8
            for count in dense_entries
        ]
        dense_levels = [
            count <= self.max_entries_per_level for count in dense_entries
        ]
        # Keep host-side copies for indexing decisions.  Reading CUDA scalar
        # buffers with .item() here would otherwise synchronize the device
        # twice per level on every forward pass.
        self._level_resolutions = tuple(resolutions)
        self._level_entries = tuple(entries)
        self._dense_levels = tuple(dense_levels)

        self.register_buffer(
            "level_scales",
            torch.tensor(scales, dtype=torch.float32),
        )
        self.register_buffer(
            "level_resolutions",
            torch.tensor(resolutions, dtype=torch.long),
        )
        self.register_buffer(
            "level_entries",
            torch.tensor(entries, dtype=torch.long),
        )
        self.register_buffer(
            "dense_levels",
            torch.tensor(dense_levels, dtype=torch.bool),
        )
        self.register_buffer("corner_offsets", _corner_offsets(4))

        self.feature_tables = nn.ParameterList()
        for entries_in_level in entries:
            table = nn.Parameter(
                torch.empty(entries_in_level, self.n_features_per_level)
            )
            nn.init.uniform_(table, -1.0e-4, 1.0e-4)
            self.feature_tables.append(table)

    @property
    def output_dim(self) -> int:
        return self.n_levels * self.n_features_per_level

    def grid_geometry(
        self,
        coords: torch.Tensor,
        level: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        scale = self.level_scales[int(level)].to(
            device=coords.device,
            dtype=coords.dtype,
        )
        grid_position = 0.5 * scale * (coords + 1.0) + 0.5
        lower_float = torch.floor(grid_position)
        lower = lower_float.to(torch.long)
        fractions = grid_position - lower_float
        vertices = lower[:, None, :] + self.corner_offsets[None, :, :]
        return vertices, fractions, grid_position

    def interpolation_weights(self, fractions: torch.Tensor) -> torch.Tensor:
        offsets = self.corner_offsets.to(torch.bool)[None, :, :]
        per_dimension = torch.where(
            offsets,
            fractions[:, None, :],
            1.0 - fractions[:, None, :],
        )
        return per_dimension.prod(dim=-1)

    def vertex_indices(self, vertices: torch.Tensor, level: int) -> torch.Tensor:
        level_index = int(level)
        resolution = self._level_resolutions[level_index]
        entries = self._level_entries[level_index]
        if self._dense_levels[level_index]:
            x, y, z, t = vertices.unbind(dim=-1)
            indices = (
                x
                + resolution * y
                + resolution * resolution * z
                + resolution * resolution * resolution * t
            )
        else:
            indices = coherent_prime_hash(vertices)
        return torch.remainder(indices, entries)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        if coords.ndim != 2 or coords.shape[1] != 4:
            raise ValueError(
                f"InstantNGP encoding expects [B, 4] coordinates, got {tuple(coords.shape)}"
            )
        coords = coords.to(dtype=torch.float32).contiguous()
        encoded_levels = []
        for level, table in enumerate(self.feature_tables):
            vertices, fractions, _ = self.grid_geometry(coords, level)
            indices = self.vertex_indices(vertices, level)
            corner_features = table[indices]
            weights = self.interpolation_weights(fractions)
            encoded_levels.append(
                torch.sum(weights[..., None] * corner_features, dim=1)
            )
        return torch.cat(encoded_levels, dim=-1)


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
