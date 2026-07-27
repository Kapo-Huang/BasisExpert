from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import torch
import torch.nn as nn


DEFAULT_FREQUENCY_COORDINATES = (1.0, 2.0, 3.0)
DEFAULT_TENSOR_RING_RANKS = (22, 88, 3, 3, 5)


class SineLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        omega: float,
        first: bool = False,
    ) -> None:
        super().__init__()
        self.omega = float(omega)
        self.linear = nn.Linear(int(in_features), int(out_features))
        with torch.no_grad():
            if first:
                bound = 1.0 / float(in_features)
            else:
                bound = math.sqrt(5.0 / float(in_features)) / self.omega
            self.linear.weight.uniform_(-bound, bound)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.omega * self.linear(inputs))


class FactorMLP(nn.Module):
    def __init__(
        self,
        left_rank: int,
        right_rank: int,
        *,
        hidden_width: int = 128,
        depth: int = 4,
        omega: float = 19.0,
    ) -> None:
        super().__init__()
        self.left_rank = int(left_rank)
        self.right_rank = int(right_rank)
        self.hidden_width = int(hidden_width)
        self.depth = int(depth)
        if self.left_rank <= 0 or self.right_rank <= 0:
            raise ValueError("Factor ranks must be positive")
        if self.hidden_width <= 0:
            raise ValueError("factor_hidden_width must be positive")
        if self.depth != 4:
            raise ValueError("FA-TR-INR requires factor_mlp_depth=4")

        self.net = nn.Sequential(
            SineLinear(1, self.hidden_width, omega=omega, first=True),
            SineLinear(
                self.hidden_width,
                self.hidden_width,
                omega=omega,
            ),
            SineLinear(
                self.hidden_width,
                self.hidden_width,
                omega=omega,
            ),
            nn.Linear(
                self.hidden_width,
                self.left_rank * self.right_rank,
            ),
        )

    def forward(self, coordinate: torch.Tensor) -> torch.Tensor:
        if coordinate.ndim != 2 or coordinate.shape[1] != 1:
            raise ValueError(
                f"FactorMLP expects [B, 1] coordinates, got {tuple(coordinate.shape)}"
            )
        output = self.net(coordinate)
        return output.reshape(
            coordinate.shape[0],
            self.left_rank,
            self.right_rank,
        )


class FrequencyAwareTRINR(nn.Module):
    def __init__(
        self,
        *,
        in_features: int = 4,
        out_features: int = 1,
        frequency_coordinates: Sequence[float] = DEFAULT_FREQUENCY_COORDINATES,
        omega: float = 19.0,
        factor_mlp_depth: int = 4,
        factor_hidden_width: int = 128,
        integration_mlp_depth: int = 2,
        tensor_ring_ranks: Sequence[int] = DEFAULT_TENSOR_RING_RANKS,
    ) -> None:
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.omega = float(omega)
        self.factor_mlp_depth = int(factor_mlp_depth)
        self.factor_hidden_width = int(factor_hidden_width)
        self.integration_mlp_depth = int(integration_mlp_depth)
        self.tensor_ring_ranks = tuple(int(rank) for rank in tensor_ring_ranks)
        frequencies = tuple(float(value) for value in frequency_coordinates)

        if self.in_features != 4:
            raise ValueError("FA-TR-INR requires four input coordinates")
        if self.out_features != 1:
            raise ValueError("FA-TR-INR requires a scalar output")
        if frequencies != DEFAULT_FREQUENCY_COORDINATES:
            raise ValueError(
                "FA-TR-INR requires frequency_coordinates=[1.0, 2.0, 3.0]"
            )
        if self.omega != 19.0:
            raise ValueError("FA-TR-INR requires omega=19.0")
        if self.factor_mlp_depth != 4:
            raise ValueError("FA-TR-INR requires factor_mlp_depth=4")
        if self.factor_hidden_width != 128:
            raise ValueError("FA-TR-INR requires factor_hidden_width=128")
        if self.integration_mlp_depth != 2:
            raise ValueError("FA-TR-INR requires integration_mlp_depth=2")
        if self.tensor_ring_ranks != DEFAULT_TENSOR_RING_RANKS:
            raise ValueError(
                "FA-TR-INR requires tensor_ring_ranks=[22, 88, 3, 3, 5]"
            )

        rank_x, rank_y, rank_f, rank_z, rank_t = self.tensor_ring_ranks
        factor_kwargs = {
            "hidden_width": self.factor_hidden_width,
            "depth": self.factor_mlp_depth,
            "omega": self.omega,
        }
        self.factor_x = FactorMLP(rank_x, rank_y, **factor_kwargs)
        self.factor_y = FactorMLP(rank_y, rank_f, **factor_kwargs)
        self.factor_f = FactorMLP(rank_f, rank_z, **factor_kwargs)
        self.factor_z = FactorMLP(rank_z, rank_t, **factor_kwargs)
        self.factor_t = FactorMLP(rank_t, rank_x, **factor_kwargs)
        self.integration = nn.Sequential(
            nn.Linear(len(frequencies), len(frequencies)),
            nn.GELU(),
            nn.Linear(len(frequencies), self.out_features),
        )
        self.register_buffer(
            "frequency_coordinates",
            torch.tensor(frequencies, dtype=torch.float32).reshape(-1, 1),
        )

    def frequency_components(self, coords: torch.Tensor) -> torch.Tensor:
        if coords.ndim != 2 or coords.shape[1] != self.in_features:
            raise ValueError(
                f"FA-TR-INR expects [B, 4] coordinates, got {tuple(coords.shape)}"
            )
        x = coords[:, 0:1]
        y = coords[:, 1:2]
        z = coords[:, 2:3]
        t = coords[:, 3:4]

        gx = self.factor_x(x)
        gy = self.factor_y(y)
        gf = self.factor_f(
            self.frequency_coordinates.to(dtype=coords.dtype)
        )
        gz = self.factor_z(z)
        gt = self.factor_t(t)
        return torch.einsum(
            "bij,bjk,fkl,blm,bmi->bf",
            gx,
            gy,
            gf,
            gz,
            gt,
        )

    def forward(self, coords: torch.Tensor, **_: Any) -> torch.Tensor:
        return self.integration(self.frequency_components(coords))


def build_fa_tr_inr_from_config(cfg: dict[str, Any]) -> FrequencyAwareTRINR:
    return FrequencyAwareTRINR(**cfg)
