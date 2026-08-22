from __future__ import annotations

import math
from collections.abc import Mapping

import torch
from torch import nn


class IndexedLinear(nn.Module):
    """A bank of independent linear layers selected by block index."""

    def __init__(self, channels: int, in_features: int, out_features: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(channels, in_features, out_features))
        self.bias = nn.Parameter(torch.empty(channels, 1, out_features))

    def forward(self, values: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        return torch.bmm(values, self.weight[indices]) + self.bias[indices]


class BlockSiren(nn.Module):
    """MINER's vectorized collection of one small SIREN per spatial block."""

    def __init__(
        self,
        *,
        channels: int,
        in_features: int,
        hidden_features: int,
        hidden_layers: int,
        out_features: int = 1,
        omega_0: float = 30.0,
        initialization_scale: float = 1.0,
    ) -> None:
        super().__init__()
        if channels <= 0 or in_features <= 0 or hidden_features <= 0:
            raise ValueError("BlockSiren dimensions must be positive")
        if hidden_layers < 0 or out_features != 1:
            raise ValueError("MINER requires hidden_layers >= 0 and scalar output")
        self.channels = int(channels)
        self.in_features = int(in_features)
        self.hidden_features = int(hidden_features)
        self.hidden_layers = int(hidden_layers)
        self.out_features = int(out_features)
        self.omega_0 = float(omega_0)
        self.initialization_scale = float(initialization_scale)

        layers: list[IndexedLinear] = [
            IndexedLinear(self.channels, self.in_features, self.hidden_features)
        ]
        layers.extend(
            IndexedLinear(self.channels, self.hidden_features, self.hidden_features)
            for _ in range(self.hidden_layers)
        )
        layers.append(IndexedLinear(self.channels, self.hidden_features, self.out_features))
        self.layers = nn.ModuleList(layers)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        with torch.no_grad():
            first_bound = self.initialization_scale / float(self.in_features)
            self.layers[0].weight.uniform_(-first_bound, first_bound)
            self.layers[0].bias.uniform_(-first_bound, first_bound)
            hidden_bound = (
                math.sqrt(self.initialization_scale * 6.0 / float(self.hidden_features))
                / self.omega_0
            )
            for layer in self.layers[1:-1]:
                layer.weight.uniform_(-hidden_bound, hidden_bound)
                layer.bias.uniform_(-hidden_bound, hidden_bound)
            output_bound = (
                math.sqrt(self.initialization_scale / float(self.hidden_features))
                / self.omega_0
            )
            self.layers[-1].weight.uniform_(-output_bound, output_bound)
            self.layers[-1].bias.uniform_(-output_bound, output_bound)

    def forward(
        self,
        coordinates: torch.Tensor,
        indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if indices is None:
            indices = torch.arange(self.channels, device=coordinates.device)
        indices = indices.to(dtype=torch.long, device=coordinates.device)
        if coordinates.ndim == 2:
            values = coordinates.unsqueeze(0).expand(indices.numel(), -1, -1)
        elif coordinates.ndim == 3:
            if coordinates.shape[0] == 1:
                values = coordinates.expand(indices.numel(), -1, -1)
            elif coordinates.shape[0] == indices.numel():
                values = coordinates
            else:
                raise ValueError("Coordinate channel count does not match selected blocks")
        else:
            raise ValueError("Coordinates must have shape [P,D] or [B,P,D]")
        values = torch.sin(self.omega_0 * self.layers[0](values, indices))
        for layer in self.layers[1:-1]:
            values = torch.sin(self.omega_0 * layer(values, indices))
        return self.layers[-1](values, indices)

    def config_dict(self) -> dict[str, int | float]:
        return {
            "channels": self.channels,
            "in_features": self.in_features,
            "hidden_features": self.hidden_features,
            "hidden_layers": self.hidden_layers,
            "out_features": self.out_features,
            "omega_0": self.omega_0,
            "initialization_scale": self.initialization_scale,
        }


def cpu_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def select_state_channels(
    state: Mapping[str, torch.Tensor], indices: torch.Tensor
) -> dict[str, torch.Tensor]:
    selected = indices.detach().cpu().to(torch.long)
    return {name: value.detach().cpu()[selected].clone() for name, value in state.items()}


def merge_state_channels(
    full_state: Mapping[str, torch.Tensor],
    selected_state: Mapping[str, torch.Tensor],
    indices: torch.Tensor,
) -> dict[str, torch.Tensor]:
    result = {name: value.detach().cpu().clone() for name, value in full_state.items()}
    selected = indices.detach().cpu().to(torch.long)
    for name, value in selected_state.items():
        result[name][selected] = value.detach().cpu()
    return result


def propagate_state_to_finer_grid(
    state: Mapping[str, torch.Tensor], grid_shape: tuple[int, ...]
) -> dict[str, torch.Tensor]:
    """Copy each parent block to its 2**D children as in the reference code."""

    if not grid_shape or any(int(value) <= 0 for value in grid_shape):
        raise ValueError("grid_shape must contain positive dimensions")
    parent_count = math.prod(grid_shape)
    first = next(iter(state.values()))
    if int(first.shape[0]) != parent_count:
        raise ValueError("State channel count does not match grid_shape")
    fine_shape = tuple(int(value) * 2 for value in grid_shape)
    fine_rows = torch.arange(math.prod(fine_shape), dtype=torch.long)
    remainder = fine_rows.clone()
    fine_coords: list[torch.Tensor] = []
    for size in reversed(fine_shape):
        fine_coords.append(remainder % size)
        remainder //= size
    fine_coords.reverse()
    parent_coords = [coord // 2 for coord in fine_coords]
    parent_indices = torch.zeros_like(fine_rows)
    for coordinate, size in zip(parent_coords, grid_shape):
        parent_indices = parent_indices * int(size) + coordinate
    divisor = math.sqrt(float(2 ** len(grid_shape)))
    return {
        name: value.detach().cpu()[parent_indices].clone() / divisor
        for name, value in state.items()
    }
