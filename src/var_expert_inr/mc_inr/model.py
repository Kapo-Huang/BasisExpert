from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn

from .data import TargetLayoutEntry, split_tensor_by_layout


class SineLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int, *, is_first: bool = False, omega_0: float = 30.0):
        super().__init__()
        self.in_features = int(in_features)
        self.is_first = bool(is_first)
        self.omega_0 = float(omega_0)
        self.linear = nn.Linear(in_features, out_features, bias=True)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        with torch.no_grad():
            if self.is_first:
                bound = 1.0 / float(self.in_features)
            else:
                bound = math.sqrt(6.0 / float(self.in_features)) / float(self.omega_0)
            self.linear.weight.uniform_(-bound, bound)
            if self.linear.bias is not None:
                self.linear.bias.zero_()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return torch.sin(float(self.omega_0) * self.linear(inputs))


class ResBlock(nn.Module):
    def __init__(self, in_features: int, out_features: int, *, is_first_block: bool = False):
        super().__init__()
        self.layers = nn.Sequential(
            SineLayer(in_features, out_features, is_first=bool(is_first_block)),
            SineLayer(out_features, out_features, is_first=False),
        )
        self.proj = SineLayer(in_features, out_features, is_first=False) if in_features != out_features else None

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        residual = features if self.proj is None else self.proj(features)
        return 0.5 * (self.layers(features) + residual)


class PositionalEncoding(nn.Module):
    def __init__(self, in_features: int, *, num_frequencies: int = 6, include_input: bool = True):
        super().__init__()
        self.num_frequencies = int(num_frequencies)
        self.include_input = bool(include_input)
        self.out_dim = int(in_features) * (int(self.include_input) + 2 * self.num_frequencies)
        freq_bands = (2.0 ** torch.arange(self.num_frequencies, dtype=torch.float32)) * math.pi
        self.register_buffer("freq_bands", freq_bands)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        angles = coords.unsqueeze(-1) * self.freq_bands
        encoded = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1).reshape(coords.shape[0], -1)
        if self.include_input:
            return torch.cat([coords, encoded], dim=-1)
        return encoded


class ClusterCoordNet(nn.Module):
    def __init__(
        self,
        *,
        in_features: int,
        out_features: int,
        hidden_features: int,
        gfe_layers: int,
        lfe_layers: int,
    ):
        super().__init__()
        self.positional_encoding = PositionalEncoding(in_features, num_frequencies=6, include_input=True)
        pe_dim = self.positional_encoding.out_dim

        gfe_blocks = [ResBlock(pe_dim, hidden_features, is_first_block=True)]
        for _ in range(max(0, int(gfe_layers) - 1)):
            gfe_blocks.append(ResBlock(hidden_features, hidden_features))
        self.global_feature_extractor = nn.Sequential(*gfe_blocks)

        heads = []
        for _ in range(int(out_features)):
            head_layers: list[nn.Module] = []
            for _ in range(max(0, int(lfe_layers) - 1)):
                head_layers.append(ResBlock(hidden_features, hidden_features))
            head_layers.append(nn.Linear(hidden_features, 1))
            heads.append(nn.Sequential(*head_layers))
        self.heads = nn.ModuleList(heads)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        encoded = self.positional_encoding(coords)
        global_features = self.global_feature_extractor(encoded)
        return torch.cat([head(global_features) for head in self.heads], dim=-1)


class MCINR(nn.Module):
    def __init__(
        self,
        *,
        centroids: np.ndarray | torch.Tensor,
        target_layout: tuple[TargetLayoutEntry, ...],
        in_features: int,
        hidden_features: int = 64,
        gfe_layers: int = 5,
        lfe_layers: int = 6,
    ):
        super().__init__()
        centroid_tensor = torch.as_tensor(centroids, dtype=torch.float32)
        if centroid_tensor.ndim != 2 or centroid_tensor.shape[0] <= 0 or centroid_tensor.shape[1] <= 0:
            raise ValueError(f"Invalid centroid shape: {tuple(centroid_tensor.shape)}")
        if not target_layout:
            raise ValueError("target_layout must be non-empty")

        self.register_buffer("centroids", centroid_tensor.clone())
        self.register_buffer(
            "routing_active",
            torch.ones((int(centroid_tensor.shape[0]),), dtype=torch.bool),
        )
        self.target_layout = tuple(target_layout)
        self.in_features = int(in_features)
        self.hidden_features = int(hidden_features)
        self.gfe_layers = int(gfe_layers)
        self.lfe_layers = int(lfe_layers)
        self.output_dim = sum(entry.dim for entry in self.target_layout)
        self.cluster_networks = nn.ModuleList(
            [
                ClusterCoordNet(
                    in_features=self.in_features,
                    out_features=self.output_dim,
                    hidden_features=self.hidden_features,
                    gfe_layers=self.gfe_layers,
                    lfe_layers=self.lfe_layers,
                )
                for _ in range(int(self.centroids.shape[0]))
            ]
        )

    @property
    def cluster_count(self) -> int:
        return int(self.centroids.shape[0])

    @property
    def spatial_dims(self) -> int:
        return int(self.centroids.shape[1])

    def route(self, coords: torch.Tensor) -> torch.Tensor:
        spatial = coords[:, : self.spatial_dims]
        centroids = self.centroids.to(device=coords.device, dtype=coords.dtype)
        distances = torch.cdist(spatial, centroids)
        active = self.routing_active.to(device=coords.device)
        if not torch.any(active):
            raise RuntimeError("MCINR.route requires at least one active cluster")
        distances[:, ~active] = float("inf")
        return torch.argmin(distances, dim=1)

    def split_cluster(self, parent_id: int, child_centroids: np.ndarray | torch.Tensor) -> tuple[int, int]:
        parent_index = int(parent_id)
        if parent_index < 0 or parent_index >= len(self.cluster_networks):
            raise IndexError(f"Parent cluster index out of range: {parent_id}")
        if not bool(self.routing_active[parent_index].item()):
            raise ValueError(f"Cluster {parent_id} is already inactive and cannot be split again")

        child_centroid_tensor = torch.as_tensor(
            child_centroids,
            dtype=self.centroids.dtype,
            device=self.centroids.device,
        )
        if child_centroid_tensor.shape != (2, self.spatial_dims):
            raise ValueError(
                f"child_centroids must have shape {(2, self.spatial_dims)}, got {tuple(child_centroid_tensor.shape)}"
            )

        parent_network = self.cluster_networks[parent_index]
        device = next(parent_network.parameters()).device
        child_network_1 = ClusterCoordNet(
            in_features=self.in_features,
            out_features=self.output_dim,
            hidden_features=self.hidden_features,
            gfe_layers=self.gfe_layers,
            lfe_layers=self.lfe_layers,
        ).to(device)
        child_network_2 = ClusterCoordNet(
            in_features=self.in_features,
            out_features=self.output_dim,
            hidden_features=self.hidden_features,
            gfe_layers=self.gfe_layers,
            lfe_layers=self.lfe_layers,
        ).to(device)
        child_network_1.load_state_dict(parent_network.state_dict())
        child_network_2.load_state_dict(parent_network.state_dict())

        next_cluster_id = len(self.cluster_networks)
        self.cluster_networks.append(child_network_1)
        self.cluster_networks.append(child_network_2)

        updated_active = self.routing_active.clone()
        updated_active[parent_index] = False
        updated_active = torch.cat(
            [updated_active, torch.ones((2,), dtype=torch.bool, device=updated_active.device)],
            dim=0,
        )
        self.routing_active = updated_active
        self.centroids = torch.cat([self.centroids, child_centroid_tensor], dim=0)
        return int(next_cluster_id), int(next_cluster_id + 1)

    def forward(
        self,
        coords: torch.Tensor,
        cluster_idx: torch.Tensor | None = None,
        *,
        return_concat: bool = False,
    ) -> dict[str, torch.Tensor] | torch.Tensor:
        if cluster_idx is None:
            cluster_idx = self.route(coords)
        else:
            cluster_idx = cluster_idx.to(device=coords.device, dtype=torch.long)

        outputs = torch.zeros(
            (coords.shape[0], self.output_dim),
            device=coords.device,
            dtype=coords.dtype,
        )
        for cluster_id in torch.unique(cluster_idx).tolist():
            if cluster_id < 0 or cluster_id >= len(self.cluster_networks):
                raise IndexError(f"Cluster index out of range: {cluster_id}")
            mask = cluster_idx == int(cluster_id)
            if torch.any(mask):
                outputs[mask] = self.cluster_networks[int(cluster_id)](coords[mask])

        if return_concat:
            return outputs
        return split_tensor_by_layout(outputs, self.target_layout)
