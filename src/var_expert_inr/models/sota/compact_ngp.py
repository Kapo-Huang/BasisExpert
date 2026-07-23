from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


UINT32_MASK = 0xFFFFFFFF
COHERENT_PRIMES = (1, 2654435761, 805459861, 3674653429)
REVERSED_PRIMES = (2165219737, 1434869437, 2097192037, 3674653429)
COMPACT_NGP_ARTIFACT_FORMAT = "compact_ngp_inference_v1"


def _corner_offsets(dimensions: int) -> torch.Tensor:
    values = torch.arange(1 << int(dimensions), dtype=torch.long)
    shifts = torch.arange(int(dimensions), dtype=torch.long)
    return ((values[:, None] >> shifts[None, :]) & 1).contiguous()


def hash_vertices(vertices: torch.Tensor, factors: tuple[int, ...]) -> torch.Tensor:
    if vertices.shape[-1] != len(factors):
        raise ValueError(
            f"Hash input dimension {vertices.shape[-1]} does not match {len(factors)} factors"
        )
    result = torch.zeros(vertices.shape[:-1], dtype=torch.long, device=vertices.device)
    for dim, factor in enumerate(factors):
        term = torch.bitwise_and(vertices[..., dim] * int(factor), UINT32_MASK)
        result = torch.bitwise_xor(result, term)
    return torch.bitwise_and(result, UINT32_MASK)


def pack_probe_offsets(offsets: torch.Tensor) -> torch.Tensor:
    flat = offsets.detach().to(device="cpu", dtype=torch.uint8).reshape(-1)
    if flat.numel() % 4 != 0:
        raise ValueError("Probe offset count must be divisible by four")
    if flat.numel() and int(flat.max().item()) > 3:
        raise ValueError("2-bit probe offsets must be in [0, 3]")
    values = flat.to(torch.int64).reshape(-1, 4)
    packed = (
        values[:, 0]
        | (values[:, 1] << 2)
        | (values[:, 2] << 4)
        | (values[:, 3] << 6)
    )
    return packed.to(torch.uint8)


def unpack_probe_offsets(packed: torch.Tensor, count: int) -> torch.Tensor:
    packed_i64 = packed.detach().to(device="cpu", dtype=torch.int64).reshape(-1)
    shifts = torch.tensor((0, 2, 4, 6), dtype=torch.int64)
    values = torch.bitwise_and(packed_i64[:, None] >> shifts[None, :], 3)
    return values.reshape(-1)[: int(count)].to(torch.uint8)


def _lookup_packed_offsets(packed: torch.Tensor, rows: torch.Tensor) -> torch.Tensor:
    byte_indices = torch.bitwise_right_shift(rows, 2)
    shifts = torch.bitwise_left_shift(torch.bitwise_and(rows, 3), 1)
    selected = packed[byte_indices].to(torch.long)
    return torch.bitwise_and(torch.bitwise_right_shift(selected, shifts), 3)


class _CompactNGPEncodingBase(nn.Module):
    def __init__(
        self,
        *,
        in_features: int = 4,
        num_levels: int = 16,
        features_per_level: int = 2,
        feature_table_size: int = 1024,
        index_table_size: int = 65536,
        num_probes: int = 4,
        base_resolution: int = 16,
        max_resolution: int = 2048,
    ) -> None:
        super().__init__()
        self.in_features = int(in_features)
        self.num_levels = int(num_levels)
        self.features_per_level = int(features_per_level)
        self.feature_table_size = int(feature_table_size)
        self.index_table_size = int(index_table_size)
        self.num_probes = int(num_probes)
        self.base_resolution = int(base_resolution)
        self.max_resolution = int(max_resolution)
        if self.in_features != 4:
            raise ValueError("CompactNGP requires four input coordinates")
        if self.num_levels <= 1:
            raise ValueError("CompactNGP requires at least two resolution levels")
        if self.features_per_level <= 0:
            raise ValueError("features_per_level must be positive")
        if self.num_probes != 4:
            raise ValueError("CompactNGP currently requires exactly four probes")
        if self.feature_table_size <= 0 or self.feature_table_size % self.num_probes != 0:
            raise ValueError("feature_table_size must be positive and divisible by num_probes")
        if self.index_table_size <= 0 or self.index_table_size % 4 != 0:
            raise ValueError("index_table_size must be positive and divisible by four")
        if self.base_resolution <= 1 or self.max_resolution < self.base_resolution:
            raise ValueError("Invalid CompactNGP resolution range")

        self.per_level_scale = math.exp(
            math.log(float(self.max_resolution) / float(self.base_resolution))
            / float(self.num_levels - 1)
        )
        scales = [
            float(self.base_resolution) * self.per_level_scale**level - 1.0
            for level in range(self.num_levels)
        ]
        resolutions = [int(math.ceil(scale)) + 1 for scale in scales]
        self.register_buffer("level_scales", torch.tensor(scales, dtype=torch.float32))
        self.register_buffer("level_resolutions", torch.tensor(resolutions, dtype=torch.long))
        self.register_buffer("corner_offsets", _corner_offsets(self.in_features))

    @property
    def encoded_features(self) -> int:
        return self.num_levels * self.features_per_level

    def grid_geometry(
        self,
        coords: torch.Tensor,
        level: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        scale = self.level_scales[int(level)].to(dtype=coords.dtype)
        grid_pos = 0.5 * scale * coords + 0.5 * scale + 0.5
        lower_float = torch.floor(grid_pos)
        lower = lower_float.to(torch.long)
        fractions = grid_pos - lower_float
        vertices = lower[:, None, :] + self.corner_offsets[None, :, :]
        return vertices, fractions, grid_pos

    def interpolation_weights(self, fractions: torch.Tensor) -> torch.Tensor:
        offsets = self.corner_offsets.to(torch.bool)[None, :, :]
        per_dimension = torch.where(
            offsets,
            fractions[:, None, :],
            1.0 - fractions[:, None, :],
        )
        return per_dimension.prod(dim=-1)


class CompactNGP(_CompactNGPEncodingBase):
    def __init__(
        self,
        *,
        out_features: int = 1,
        hidden_features: int = 64,
        hidden_layers: int = 2,
        **encoding_kwargs: Any,
    ) -> None:
        super().__init__(**encoding_kwargs)
        self.out_features = int(out_features)
        self.hidden_features = int(hidden_features)
        self.hidden_layers = int(hidden_layers)
        if self.out_features != 1:
            raise ValueError("CompactNGP requires a scalar output")
        if self.hidden_features <= 0 or self.hidden_layers != 2:
            raise ValueError("CompactNGP requires two positive-width hidden layers")

        self.feature_tables = nn.ParameterList()
        self.confidence_tables = nn.ParameterList()
        for _ in range(self.num_levels):
            features = nn.Parameter(
                torch.empty(self.feature_table_size, self.features_per_level)
            )
            nn.init.uniform_(features, -1.0e-4, 1.0e-4)
            self.feature_tables.append(features)
            self.confidence_tables.append(
                nn.Parameter(torch.zeros(self.index_table_size, self.num_probes))
            )

        layers: list[nn.Module] = []
        width = self.encoded_features
        for _ in range(self.hidden_layers):
            layers.extend((nn.Linear(width, self.hidden_features), nn.ReLU()))
            width = self.hidden_features
        layers.append(nn.Linear(width, self.out_features))
        self.decoder = nn.Sequential(*layers)
        self._baked_indices: list[torch.Tensor] | None = None
        self._baked_valid = False

    def invalidate_baked_indices(self) -> None:
        self._baked_indices = None
        self._baked_valid = False

    @torch.no_grad()
    def bake_indices(self) -> list[torch.Tensor]:
        if not self._baked_valid:
            self._baked_indices = [
                table.argmax(dim=-1).to(torch.uint8) for table in self.confidence_tables
            ]
            self._baked_valid = True
        return self._baked_indices

    def train(self, mode: bool = True):
        result = super().train(mode)
        if mode:
            self.invalidate_baked_indices()
        else:
            self.bake_indices()
        return result

    def _load_from_state_dict(self, *args, **kwargs) -> None:
        self.invalidate_baked_indices()
        super()._load_from_state_dict(*args, **kwargs)
        self.invalidate_baked_indices()

    def _training_vertex_features(
        self,
        level: int,
        feature_base: torch.Tensor,
        confidence_rows: torch.Tensor,
    ) -> torch.Tensor:
        logits = self.confidence_tables[level][confidence_rows]
        offsets = torch.arange(self.num_probes, device=feature_base.device)
        candidate_indices = feature_base[:, None] + offsets[None, :]
        candidates = self.feature_tables[level][candidate_indices]
        probabilities = torch.softmax(logits, dim=-1)
        soft_feature = torch.sum(probabilities[..., None] * candidates, dim=1)
        hard_offset = logits.argmax(dim=-1)
        hard_feature = candidates[
            torch.arange(candidates.shape[0], device=candidates.device),
            hard_offset,
        ]
        return soft_feature + (hard_feature - soft_feature).detach()

    def _inference_vertex_features(
        self,
        level: int,
        feature_base: torch.Tensor,
        confidence_rows: torch.Tensor,
    ) -> torch.Tensor:
        baked = self.bake_indices()[level]
        offsets = baked[confidence_rows].to(torch.long)
        return self.feature_tables[level][feature_base + offsets]

    def encode(self, coords: torch.Tensor) -> torch.Tensor:
        if coords.ndim != 2 or coords.shape[1] != self.in_features:
            raise ValueError(f"CompactNGP expects [B, 4] coordinates, got {tuple(coords.shape)}")
        encoded = []
        for level in range(self.num_levels):
            vertices, fractions, _ = self.grid_geometry(coords, level)
            flat_vertices = vertices.reshape(-1, self.in_features)
            feature_hash = hash_vertices(flat_vertices, COHERENT_PRIMES)
            confidence_hash = hash_vertices(flat_vertices, REVERSED_PRIMES)
            feature_base = torch.remainder(
                feature_hash * self.num_probes,
                self.feature_table_size,
            )
            confidence_rows = torch.remainder(confidence_hash, self.index_table_size)
            if self.training:
                flat_features = self._training_vertex_features(
                    level, feature_base, confidence_rows
                )
            else:
                flat_features = self._inference_vertex_features(
                    level, feature_base, confidence_rows
                )
            corner_features = flat_features.reshape(
                coords.shape[0], 1 << self.in_features, self.features_per_level
            )
            weights = self.interpolation_weights(fractions)
            encoded.append(torch.sum(weights[..., None] * corner_features, dim=1))
        return torch.cat(encoded, dim=-1)

    def forward(self, coords: torch.Tensor, **_: Any) -> torch.Tensor:
        return self.decoder(self.encode(coords))

    def inference_payload(
        self,
        *,
        model_config: dict[str, Any],
        target_names: tuple[str, ...],
        volume_shape: dict[str, int] | None,
        config_hash: str,
    ) -> tuple[dict[str, Any], int]:
        baked = self.bake_indices()
        feature_tables = torch.stack(
            [table.detach().cpu().to(torch.float16) for table in self.feature_tables]
        )
        packed_indices = torch.stack([pack_probe_offsets(table) for table in baked])
        decoder_state = {
            name: tensor.detach().cpu().to(torch.float16)
            for name, tensor in self.decoder.state_dict().items()
        }
        tensors = [feature_tables, packed_indices, *decoder_state.values()]
        payload_bytes = sum(tensor.numel() * tensor.element_size() for tensor in tensors)
        payload = {
            "format": COMPACT_NGP_ARTIFACT_FORMAT,
            "model_name": "compact_ngp",
            "model_config": dict(model_config),
            "target_names_order": list(target_names),
            "volume_shape": volume_shape,
            "config_hash": str(config_hash),
            "feature_tables": feature_tables,
            "packed_indices": packed_indices,
            "decoder_state": decoder_state,
            "theoretical_payload_bytes": int(payload_bytes),
        }
        return payload, int(payload_bytes)


class CompactNGPInference(_CompactNGPEncodingBase):
    def __init__(
        self,
        *,
        feature_tables: torch.Tensor,
        packed_indices: torch.Tensor,
        decoder_state: dict[str, torch.Tensor],
        out_features: int = 1,
        hidden_features: int = 64,
        hidden_layers: int = 2,
        **encoding_kwargs: Any,
    ) -> None:
        super().__init__(**encoding_kwargs)
        self.out_features = int(out_features)
        self.hidden_features = int(hidden_features)
        self.hidden_layers = int(hidden_layers)
        expected_features = (
            self.num_levels,
            self.feature_table_size,
            self.features_per_level,
        )
        expected_indices = (self.num_levels, self.index_table_size // 4)
        if tuple(feature_tables.shape) != expected_features:
            raise ValueError(
                f"CompactNGP artifact feature shape {tuple(feature_tables.shape)} "
                f"does not match {expected_features}"
            )
        if tuple(packed_indices.shape) != expected_indices:
            raise ValueError(
                f"CompactNGP artifact index shape {tuple(packed_indices.shape)} "
                f"does not match {expected_indices}"
            )
        self.register_buffer("feature_tables", feature_tables.detach().to(torch.float32))
        self.register_buffer("packed_indices", packed_indices.detach().to(torch.uint8))
        self.decoder = nn.Sequential(
            nn.Linear(self.encoded_features, self.hidden_features),
            nn.ReLU(),
            nn.Linear(self.hidden_features, self.hidden_features),
            nn.ReLU(),
            nn.Linear(self.hidden_features, self.out_features),
        )
        self.decoder.load_state_dict(
            {name: tensor.to(torch.float32) for name, tensor in decoder_state.items()}
        )
        self.requires_grad_(False)
        self.eval()

    def encode(self, coords: torch.Tensor) -> torch.Tensor:
        if coords.ndim != 2 or coords.shape[1] != self.in_features:
            raise ValueError(f"CompactNGP expects [B, 4] coordinates, got {tuple(coords.shape)}")
        encoded = []
        for level in range(self.num_levels):
            vertices, fractions, _ = self.grid_geometry(coords, level)
            flat_vertices = vertices.reshape(-1, self.in_features)
            feature_hash = hash_vertices(flat_vertices, COHERENT_PRIMES)
            confidence_hash = hash_vertices(flat_vertices, REVERSED_PRIMES)
            feature_base = torch.remainder(
                feature_hash * self.num_probes,
                self.feature_table_size,
            )
            rows = torch.remainder(confidence_hash, self.index_table_size)
            offsets = _lookup_packed_offsets(self.packed_indices[level], rows)
            flat_features = self.feature_tables[level][feature_base + offsets]
            corner_features = flat_features.reshape(
                coords.shape[0], 1 << self.in_features, self.features_per_level
            )
            weights = self.interpolation_weights(fractions)
            encoded.append(torch.sum(weights[..., None] * corner_features, dim=1))
        return torch.cat(encoded, dim=-1)

    def forward(self, coords: torch.Tensor, **_: Any) -> torch.Tensor:
        return self.decoder(self.encode(coords))


def save_compact_ngp_artifact(
    path: str | Path,
    *,
    model: nn.Module,
    model_config: dict[str, Any],
    dataset,
    config_hash: str,
) -> dict[str, Any]:
    backbone = getattr(model, "backbone", model)
    if not isinstance(backbone, CompactNGP):
        raise TypeError("save_compact_ngp_artifact requires a CompactNGP model")
    volume_shape = (
        dataset.meta.volume_shape.to_dict()
        if dataset.meta.volume_shape is not None
        else None
    )
    payload, payload_bytes = backbone.inference_payload(
        model_config=model_config,
        target_names=dataset.target_names(),
        volume_shape=volume_shape,
        config_hash=config_hash,
    )
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)
    return {
        "artifact_path": output,
        "compact_payload_bytes": int(payload_bytes),
        "artifact_file_bytes": int(output.stat().st_size),
    }


def load_compact_ngp_artifact(
    path: str | Path,
    *,
    device: torch.device | str = "cpu",
) -> tuple[CompactNGPInference, dict[str, Any]]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if payload.get("format") != COMPACT_NGP_ARTIFACT_FORMAT:
        raise ValueError("Unsupported CompactNGP artifact format")
    config = dict(payload["model_config"])
    config.pop("name", None)
    model = CompactNGPInference(
        feature_tables=payload["feature_tables"],
        packed_indices=payload["packed_indices"],
        decoder_state=payload["decoder_state"],
        **config,
    ).to(device)
    return model, payload


def build_compact_ngp_from_config(cfg: dict[str, Any]) -> CompactNGP:
    return CompactNGP(**cfg)
