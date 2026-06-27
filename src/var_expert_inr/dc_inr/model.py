from __future__ import annotations

import math

import torch
import torch.nn as nn


def adjusted_even_width(width: int, *, minimum: int = 4) -> int:
    resolved = max(int(minimum), int(width))
    if resolved % 2 != 0:
        resolved += 1
    return resolved


def dc_inr_parameter_count(width: int) -> int:
    resolved = adjusted_even_width(width)
    return (19 * resolved * resolved) + ((37 * resolved) // 2) + 1


def _siren_init_linear(linear: nn.Linear, *, is_first: bool) -> None:
    in_features = int(linear.in_features)
    if is_first:
        bound = 1.0 / float(in_features)
    else:
        bound = math.sqrt(6.0 / float(in_features))
    with torch.no_grad():
        linear.weight.uniform_(-bound, bound)
        if linear.bias is not None:
            linear.bias.zero_()


class LearnableFourierPE(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        resolved_width = adjusted_even_width(width)
        self.width = int(resolved_width)
        self.proj = nn.Linear(4, self.width // 2, bias=True)
        _siren_init_linear(self.proj, is_first=True)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        projected = self.proj(coords)
        angles = 2.0 * math.pi * projected
        return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)


class SineLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, *, is_first: bool = False) -> None:
        super().__init__()
        self.linear = nn.Linear(int(in_features), int(out_features), bias=True)
        _siren_init_linear(self.linear, is_first=bool(is_first))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.linear(inputs))


class ResidualBottleneck(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        resolved_width = adjusted_even_width(width)
        hidden = int(resolved_width)
        expanded = 4 * hidden
        self.fc1 = nn.Linear(expanded, hidden, bias=True)
        self.fc2 = nn.Linear(hidden, hidden, bias=True)
        self.fc3 = nn.Linear(hidden, expanded, bias=True)
        _siren_init_linear(self.fc1, is_first=False)
        _siren_init_linear(self.fc2, is_first=False)
        _siren_init_linear(self.fc3, is_first=False)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        residual = features
        x = torch.sin(self.fc1(features))
        x = torch.sin(self.fc2(x))
        x = self.fc3(x)
        return residual + x


class DCINRTiny(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.width = adjusted_even_width(width)
        self.positional_encoding = LearnableFourierPE(self.width)
        self.fc1 = SineLinear(self.width, 2 * self.width, is_first=False)
        self.fc2 = SineLinear(2 * self.width, 4 * self.width, is_first=False)
        self.residual = ResidualBottleneck(self.width)
        self.decoder = nn.Linear(4 * self.width, 1, bias=True)
        _siren_init_linear(self.decoder, is_first=False)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        encoded = self.positional_encoding(coords)
        hidden = self.fc1(encoded)
        hidden = self.fc2(hidden)
        hidden = self.residual(hidden)
        return self.decoder(hidden)
