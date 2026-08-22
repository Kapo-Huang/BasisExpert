from __future__ import annotations

import math

import torch
import torch.nn as nn


UINT32_MASK = 0xFFFFFFFF
COHERENT_PRIMES = (1, 2654435761, 805459861, 3674653429)


def corner_offsets(dimensions: int) -> torch.Tensor:
    dimensions = int(dimensions)
    if dimensions <= 0 or dimensions > len(COHERENT_PRIMES):
        raise ValueError(
            f"Hash-grid dimensions must be in [1, {len(COHERENT_PRIMES)}]"
        )
    corner_ids = torch.arange(1 << dimensions, dtype=torch.long)
    shifts = torch.arange(dimensions, dtype=torch.long)
    return ((corner_ids[:, None] >> shifts[None, :]) & 1).contiguous()


def coherent_prime_hash(vertices: torch.Tensor) -> torch.Tensor:
    if vertices.ndim < 1:
        raise ValueError("CoherentPrime hash expects at least one dimension")
    dimensions = int(vertices.shape[-1])
    if dimensions <= 0 or dimensions > len(COHERENT_PRIMES):
        raise ValueError(
            f"CoherentPrime hash supports 1-{len(COHERENT_PRIMES)} coordinates, "
            f"got {dimensions}"
        )
    result = torch.zeros(
        vertices.shape[:-1],
        dtype=torch.long,
        device=vertices.device,
    )
    for dim, prime in enumerate(COHERENT_PRIMES[:dimensions]):
        term = torch.bitwise_and(vertices[..., dim] * int(prime), UINT32_MASK)
        result = torch.bitwise_xor(result, term)
    return torch.bitwise_and(result, UINT32_MASK)


class MultiresolutionHashEncoding(nn.Module):
    def __init__(
        self,
        *,
        dimensions: int,
        n_levels: int,
        n_features_per_level: int,
        base_resolution: int,
        log2_hashmap_size: int,
        finest_resolution: int | None = None,
        per_level_scale: float | None = None,
    ) -> None:
        super().__init__()
        self.dimensions = int(dimensions)
        self.n_levels = int(n_levels)
        self.n_features_per_level = int(n_features_per_level)
        self.base_resolution = int(base_resolution)
        self.log2_hashmap_size = int(log2_hashmap_size)
        self.max_entries_per_level = 1 << self.log2_hashmap_size

        if self.dimensions <= 0 or self.dimensions > len(COHERENT_PRIMES):
            raise ValueError(
                f"dimensions must be in [1, {len(COHERENT_PRIMES)}]"
            )
        if self.n_levels < 2:
            raise ValueError("n_levels must be at least two")
        if self.n_features_per_level <= 0:
            raise ValueError("n_features_per_level must be positive")
        if self.base_resolution < 2:
            raise ValueError("base_resolution must be at least two")
        if self.log2_hashmap_size <= 0:
            raise ValueError("log2_hashmap_size must be positive")
        if (finest_resolution is None) == (per_level_scale is None):
            raise ValueError(
                "Exactly one of finest_resolution or per_level_scale must be provided"
            )

        if finest_resolution is not None:
            self.finest_resolution = int(finest_resolution)
            if self.finest_resolution < self.base_resolution:
                raise ValueError(
                    "finest_resolution must be greater than or equal to base_resolution"
                )
            self.per_level_scale = math.exp(
                (
                    math.log(float(self.finest_resolution))
                    - math.log(float(self.base_resolution))
                )
                / float(self.n_levels - 1)
            )
        else:
            self.per_level_scale = float(per_level_scale)
            if self.per_level_scale <= 1.0:
                raise ValueError("per_level_scale must be greater than one")
            self.finest_resolution = int(
                math.ceil(
                    float(self.base_resolution)
                    * self.per_level_scale ** (self.n_levels - 1)
                )
            )

        scales = [
            float(self.base_resolution) * self.per_level_scale**level - 1.0
            for level in range(self.n_levels)
        ]
        resolutions = [int(math.ceil(scale)) + 1 for scale in scales]
        dense_entries = [int(resolution) ** self.dimensions for resolution in resolutions]
        entries = [
            ((min(count, self.max_entries_per_level) + 7) // 8) * 8
            for count in dense_entries
        ]
        dense_levels = [
            count <= self.max_entries_per_level for count in dense_entries
        ]
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
        self.register_buffer("corner_offsets", corner_offsets(self.dimensions))

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
            indices = torch.zeros(
                vertices.shape[:-1], dtype=torch.long, device=vertices.device
            )
            stride = 1
            for coordinate in vertices.unbind(dim=-1):
                indices = indices + stride * coordinate
                stride *= resolution
        else:
            indices = coherent_prime_hash(vertices)
        return torch.remainder(indices, entries)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        if coords.ndim != 2 or coords.shape[1] != self.dimensions:
            raise ValueError(
                "Hash-grid encoding expects "
                f"[B, {self.dimensions}] coordinates, got {tuple(coords.shape)}"
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
