from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


def _weights_init(module: nn.Module) -> None:
    if isinstance(module, nn.Linear):
        nn.init.xavier_normal_(module.weight)
        if module.bias is not None:
            torch.nn.init.normal_(module.bias, 0.0, 0.001)


class ReLULayer(nn.Module):
    def __init__(self, in_features: int, out_features: int, *, bias: bool = True) -> None:
        super().__init__()
        self.linear = nn.Linear(int(in_features), int(out_features), bias=bool(bias), dtype=torch.float32)
        self.init_weights()

    def init_weights(self) -> None:
        with torch.no_grad():
            nn.init.xavier_normal_(self.linear.weight)
            if self.linear.bias is not None:
                nn.init.zeros_(self.linear.bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return F.relu(self.linear(inputs))


class APMGEncoder(nn.Module):
    def __init__(
        self,
        *,
        n_grids: int,
        n_features: int,
        feature_grid_shape: list[int],
        grid_initialization: str,
    ) -> None:
        super().__init__()
        self.n_grids = int(n_grids)
        self.n_features = int(n_features)
        self.feature_grid_shape = [int(value) for value in feature_grid_shape]
        self.transformation_matrices = nn.Parameter(
            torch.zeros((self.n_grids, 4, 4), dtype=torch.float32),
            requires_grad=True,
        )
        self.feature_grids = nn.Parameter(
            torch.empty((self.n_grids, self.n_features, *self.feature_grid_shape), dtype=torch.float32).uniform_(-1.0e-4, 1.0e-4),
            requires_grad=True,
        )

        initialization = str(grid_initialization).strip().lower()
        if "small" in initialization:
            self.init_grids_small()
        elif "large" in initialization:
            self.init_grids_large()
        else:
            self.randomize_grids()

    def _randomized_transform(self, *, base_scale: float) -> torch.Tensor:
        matrices = torch.eye(4, dtype=torch.float32).unsqueeze(0).repeat(self.n_grids, 1, 1) * float(base_scale)
        matrices[:, 0:3, :] += torch.randn_like(matrices[:, 0:3, :]) * 0.05
        matrices[:, 3, 0:3] = 0.0
        matrices[:, 3, 3] = 1.0
        return matrices

    def randomize_grids(self) -> None:
        with torch.no_grad():
            self.transformation_matrices.copy_(self._randomized_transform(base_scale=1.0))

    def init_grids_large(self) -> None:
        with torch.no_grad():
            self.transformation_matrices.copy_(self._randomized_transform(base_scale=0.8))

    def init_grids_small(self) -> None:
        with torch.no_grad():
            matrices = self._randomized_transform(base_scale=8.0)
            translations = (torch.rand((self.n_grids, 3), dtype=torch.float32) * 2.0 - 1.0)
            translations *= matrices.diagonal(0, 1, 2)[:, 0:3]
            matrices[:, 0:3, -1] += translations
            self.transformation_matrices.copy_(matrices)

    def transform(self, coords: torch.Tensor) -> torch.Tensor:
        batch = int(coords.shape[0])
        ones = torch.ones((batch, 1), device=coords.device, dtype=torch.float32)
        homogeneous = torch.cat([coords.to(dtype=torch.float32), ones], dim=1)
        transformed = torch.matmul(self.transformation_matrices, homogeneous.transpose(0, 1)).transpose(1, 2)
        return transformed[..., 0:3]

    def feature_density_pre_transformed(self, transformed_points: torch.Tensor) -> torch.Tensor:
        coeffs = torch.linalg.det(self.transformation_matrices[:, 0:-1, 0:-1]).unsqueeze(0) / (2.0 * torch.pi) ** 1.5
        exps = torch.exp(-0.5 * torch.sum(transformed_points.transpose(0, 1) ** 20, dim=-1))
        return torch.sum(coeffs * exps, dim=-1, keepdim=True)

    def forward_pre_transformed(self, transformed_points: torch.Tensor) -> torch.Tensor:
        grids = int(transformed_points.shape[0])
        batch = int(transformed_points.shape[1])
        grid = transformed_points.reshape(grids, 1, 1, batch, 3)
        sampled = F.grid_sample(
            self.feature_grids,
            grid.detach() if self.training else grid,
            mode="bilinear",
            align_corners=True,
            padding_mode="zeros",
        )
        sampled = sampled.permute(0, 4, 1, 2, 3).reshape(grids, batch, self.n_features)
        return sampled.permute(1, 0, 2).reshape(batch, grids * self.n_features)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        return self.forward_pre_transformed(self.transform(coords))


class APMGSRN(nn.Module):
    def __init__(
        self,
        cfg: dict[str, Any],
        *,
        data_min: float,
        data_max: float,
        use_tcnn: bool,
    ) -> None:
        super().__init__()
        self.n_dims = int(cfg["n_dims"])
        if self.n_dims != 3:
            raise ValueError(f"APMGSRN only supports 3D spatial inputs, got n_dims={self.n_dims}")
        self.n_outputs = int(cfg["n_outputs"])
        self.n_grids = int(cfg["n_grids"])
        self.n_features = int(cfg["n_features"])
        self.feature_grid_shape = [int(value) for value in cfg["feature_grid_shape"]]
        self.nodes_per_layer = int(cfg["nodes_per_layer"])
        self.n_layers = int(cfg["n_layers"])
        self.use_bias = bool(cfg["use_bias"])
        self.padding_size = 0
        requires_padded_feats = cfg.get("requires_padded_feats")
        if requires_padded_feats is None:
            requires_padded_feats = bool(use_tcnn and ((self.n_grids * self.n_features) % 16 != 0))
        self.requires_padded_feats = bool(requires_padded_feats)
        if self.requires_padded_feats:
            padded_input = 16 * int(math.ceil(max(1, (self.n_grids * self.n_features) / 16.0)))
            self.padding_size = int(padded_input - (self.n_grids * self.n_features))

        self.encoder = APMGEncoder(
            n_grids=self.n_grids,
            n_features=self.n_features,
            feature_grid_shape=self.feature_grid_shape,
            grid_initialization=str(cfg.get("grid_initialization", "default")),
        )

        self.using_tcnn = False
        if bool(use_tcnn):
            self.decoder = self._init_decoder_tcnn()
            self.using_tcnn = True
        else:
            self.decoder = self._init_decoder_pytorch()

        self.reset_parameters()
        self.register_buffer("volume_min", torch.tensor([float(data_min)], dtype=torch.float32), persistent=False)
        self.register_buffer("volume_max", torch.tensor([float(data_max)], dtype=torch.float32), persistent=False)

    def _decoder_input_size(self) -> int:
        base_size = self.n_grids * self.n_features
        if self.requires_padded_feats:
            return int(base_size + self.padding_size)
        return int(base_size)

    def _init_decoder_tcnn(self) -> nn.Module:
        try:
            import tinycudann as tcnn
        except ImportError as exc:
            raise RuntimeError("tinycudann is not available") from exc
        return tcnn.Network(
            n_input_dims=self._decoder_input_size(),
            n_output_dims=self.n_outputs,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": self.nodes_per_layer,
                "n_hidden_layers": self.n_layers,
            },
        )

    def _init_decoder_pytorch(self) -> nn.Sequential:
        layers: list[nn.Module] = [
            ReLULayer(self._decoder_input_size(), self.nodes_per_layer, bias=self.use_bias),
        ]
        for layer_index in range(self.n_layers):
            if layer_index == self.n_layers - 1:
                layers.append(nn.Linear(self.nodes_per_layer, self.n_outputs, bias=self.use_bias, dtype=torch.float32))
            else:
                layers.append(ReLULayer(self.nodes_per_layer, self.nodes_per_layer, bias=self.use_bias))
        return nn.Sequential(*layers)

    def get_transform_parameters(self) -> list[dict[str, object]]:
        return [{"params": self.encoder.transformation_matrices}]

    def get_model_parameters(self) -> list[dict[str, object]]:
        return [
            {"params": [self.encoder.feature_grids]},
            {"params": self.decoder.parameters()},
        ]

    def reset_parameters(self) -> None:
        with torch.no_grad():
            self.encoder.feature_grids.uniform_(-1.0e-4, 1.0e-4)
        if not self.using_tcnn:
            self.decoder.apply(_weights_init)

    def transform(self, coords: torch.Tensor) -> torch.Tensor:
        return self.encoder.transform(coords)

    def feature_density_pre_transformed(self, transformed_points: torch.Tensor) -> torch.Tensor:
        return self.encoder.feature_density_pre_transformed(transformed_points)

    def forward_pre_transformed(self, transformed_points: torch.Tensor) -> torch.Tensor:
        features = self.encoder.forward_pre_transformed(transformed_points)
        if self.requires_padded_feats:
            features = F.pad(features, (0, self.padding_size), value=1.0)
        outputs = self.decoder(features).float()
        return outputs * (self.volume_max - self.volume_min) + self.volume_min

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        return self.forward_pre_transformed(self.transform(coords))
