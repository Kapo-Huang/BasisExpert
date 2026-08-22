from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn


MVNET_IN_FEATURES = 4
MVNET_HIDDEN_FEATURES = 206
MVNET_RESIDUAL_BLOCKS = 10
MVNET_OMEGA_0 = 30.0
MVNET_BIAS = True


class SineLayer(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        bias: bool = True,
        is_first: bool = False,
        omega_0: float = MVNET_OMEGA_0,
    ) -> None:
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.is_first = bool(is_first)
        self.omega_0 = float(omega_0)
        self.linear = nn.Linear(
            self.in_features,
            self.out_features,
            bias=bool(bias),
        )
        self.reset_weight()

    def reset_weight(self) -> None:
        if self.is_first:
            bound = 1.0 / float(self.in_features)
        else:
            bound = (
                math.sqrt(6.0 / float(self.in_features))
                / self.omega_0
            )
        with torch.no_grad():
            self.linear.weight.uniform_(-bound, bound)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.omega_0 * self.linear(inputs))


class ResidualSineBlock(nn.Module):
    def __init__(
        self,
        *,
        features: int = MVNET_HIDDEN_FEATURES,
        omega_0: float = MVNET_OMEGA_0,
        bias: bool = MVNET_BIAS,
    ) -> None:
        super().__init__()
        features = int(features)
        self.layer1 = SineLayer(
            features,
            features,
            bias=bias,
            is_first=False,
            omega_0=omega_0,
        )
        self.layer2 = SineLayer(
            features,
            features,
            bias=bias,
            is_first=False,
            omega_0=omega_0,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        residual = inputs
        transformed = self.layer1(inputs)
        transformed = self.layer2(transformed)
        return 0.5 * (transformed + residual)


class MVNet4D(nn.Module):
    def __init__(
        self,
        num_variables: int,
        *,
        in_features: int = MVNET_IN_FEATURES,
        hidden_features: int = MVNET_HIDDEN_FEATURES,
        num_residual_blocks: int = MVNET_RESIDUAL_BLOCKS,
        omega_0: float = MVNET_OMEGA_0,
        bias: bool = MVNET_BIAS,
    ) -> None:
        super().__init__()
        actual = {
            "in_features": int(in_features),
            "hidden_features": int(hidden_features),
            "num_residual_blocks": int(num_residual_blocks),
            "omega_0": float(omega_0),
            "bias": bool(bias),
        }
        expected = {
            "in_features": MVNET_IN_FEATURES,
            "hidden_features": MVNET_HIDDEN_FEATURES,
            "num_residual_blocks": MVNET_RESIDUAL_BLOCKS,
            "omega_0": MVNET_OMEGA_0,
            "bias": MVNET_BIAS,
        }
        mismatches = [
            f"{key}={actual[key]} (expected {value})"
            for key, value in expected.items()
            if actual[key] != value
        ]
        if mismatches:
            raise ValueError(
                "MVNet4D uses a fixed architecture: "
                + ", ".join(mismatches)
            )
        self.num_variables = int(num_variables)
        if self.num_variables < 2:
            raise ValueError("MVNet4D requires at least two variables")

        self.in_features = actual["in_features"]
        self.hidden_features = actual["hidden_features"]
        self.input_layer = SineLayer(
            self.in_features,
            self.hidden_features,
            bias=actual["bias"],
            is_first=True,
            omega_0=actual["omega_0"],
        )
        self.residual_blocks = nn.ModuleList(
            [
                ResidualSineBlock(
                    features=self.hidden_features,
                    omega_0=actual["omega_0"],
                    bias=actual["bias"],
                )
                for _ in range(actual["num_residual_blocks"])
            ]
        )
        self.output_layer = nn.Linear(
            self.hidden_features,
            self.num_variables,
            bias=actual["bias"],
        )
        with torch.no_grad():
            bound = 1.0 / float(self.hidden_features)
            self.output_layer.weight.uniform_(-bound, bound)

    @property
    def expected_parameter_count(self) -> int:
        hidden = self.hidden_features
        return (
            hidden * (self.in_features + 1)
            + 2 * len(self.residual_blocks) * hidden * (hidden + 1)
            + self.num_variables * (hidden + 1)
        )

    def forward(self, coords: torch.Tensor, **_: Any) -> torch.Tensor:
        if coords.ndim != 2 or coords.shape[1] != self.in_features:
            raise ValueError(
                f"MVNet4D expects [B, 4] coordinates, got {tuple(coords.shape)}"
            )
        hidden = self.input_layer(coords.float())
        for block in self.residual_blocks:
            hidden = block(hidden)
        return self.output_layer(hidden)


def build_mvnet_from_config(cfg: dict[str, Any]) -> MVNet4D:
    payload = dict(cfg)
    num_variables = int(payload.pop("out_features"))
    return MVNet4D(num_variables=num_variables, **payload)
