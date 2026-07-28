from __future__ import annotations

import math

import torch
from torch import nn


class PackedLinear(nn.Module):
    def __init__(self, mlp_count: int, in_features: int, out_features: int, *, initialization: str) -> None:
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.weight = nn.Parameter(torch.empty(int(mlp_count), self.in_features, self.out_features))
        self.bias = nn.Parameter(torch.empty(int(mlp_count), 1, self.out_features))
        self.reset_parameters(initialization)

    def reset_parameters(self, initialization: str) -> None:
        if initialization == "first":
            bound = 1.0 / float(self.in_features)
        elif initialization in {"sine", "output"}:
            bound = math.sqrt(6.0 / float(self.in_features)) / 30.0
        else:
            raise ValueError(f"Unknown ECNR initialization: {initialization}")
        nn.init.uniform_(self.weight, -bound, bound)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return torch.bmm(inputs, self.weight) + self.bias


class PackedSiren(nn.Module):
    """A parallel group of fixed ECNR SIREN MLPs."""

    def __init__(
        self,
        *,
        mlp_count: int,
        max_slots: int,
        slot_valid: torch.Tensor | None = None,
        latent_dim: int = 8,
        hidden_features: int = 24,
        omega_0: float = 30.0,
    ) -> None:
        super().__init__()
        if int(mlp_count) <= 0 or int(max_slots) <= 0:
            raise ValueError("mlp_count and max_slots must be positive")
        if int(latent_dim) != 8 or int(hidden_features) != 24 or float(omega_0) != 30.0:
            raise ValueError("ECNR architecture is fixed to latent=8, hidden=24, omega_0=30")
        self.mlp_count = int(mlp_count)
        self.max_slots = int(max_slots)
        self.latent_dim = int(latent_dim)
        self.omega_0 = float(omega_0)
        self.layers = nn.ModuleList(
            [
                PackedLinear(self.mlp_count, 3 + self.latent_dim, hidden_features, initialization="first"),
                PackedLinear(self.mlp_count, hidden_features, hidden_features, initialization="sine"),
                PackedLinear(self.mlp_count, hidden_features, hidden_features, initialization="sine"),
                PackedLinear(self.mlp_count, hidden_features, 1, initialization="output"),
            ]
        )
        self.latent = nn.Parameter(torch.randn(self.mlp_count, self.max_slots, self.latent_dim))
        if slot_valid is None:
            slot_valid = torch.ones(self.mlp_count, self.max_slots, dtype=torch.bool)
        self.register_buffer("slot_valid", torch.as_tensor(slot_valid, dtype=torch.bool))

    def forward(self, coordinates: torch.Tensor, slots: torch.Tensor) -> torch.Tensor:
        if coordinates.ndim != 2 or coordinates.shape[1] != 3:
            raise ValueError("coordinates must have shape [B,3]")
        slots = slots.to(device=self.latent.device, dtype=torch.long)
        if slots.ndim != 1 or slots.shape[0] != coordinates.shape[0]:
            raise ValueError("slots must have shape [B]")
        if slots.numel() and (int(slots.min()) < 0 or int(slots.max()) >= self.max_slots):
            raise ValueError("slot index outside latent table")
        batch = int(coordinates.shape[0])
        coords = coordinates.to(self.latent.device).unsqueeze(0).expand(self.mlp_count, batch, 3)
        latent = self.latent[:, slots, :]
        hidden = torch.cat([coords, latent], dim=-1)
        for layer in self.layers[:3]:
            hidden = torch.sin(self.omega_0 * layer(hidden))
        return self.layers[3](hidden).squeeze(-1)

    def expanded_slot_mask(self, slots: torch.Tensor) -> torch.Tensor:
        return self.slot_valid[:, slots.to(self.slot_valid.device, dtype=torch.long)]

    def apply_pruning_masks(self, masks: dict[str, torch.Tensor]) -> None:
        with torch.no_grad():
            for name, mask in masks.items():
                parameter = dict(self.named_parameters())[name]
                parameter.mul_(mask.to(parameter.device, dtype=parameter.dtype))

    def mask_pruned_gradients(self, masks: dict[str, torch.Tensor]) -> None:
        for name, mask in masks.items():
            parameter = dict(self.named_parameters())[name]
            if parameter.grad is not None:
                parameter.grad.mul_(mask.to(parameter.grad.device, dtype=parameter.grad.dtype))


def local_coordinate_grid(block_shape_xyz: tuple[int, int, int]) -> torch.Tensor:
    bx, by, bz = (int(value) for value in block_shape_xyz)

    def axis(size: int) -> torch.Tensor:
        if size == 1:
            return torch.zeros(1, dtype=torch.float32)
        return torch.linspace(-1.0, 1.0, size, dtype=torch.float32)

    z, y, x = torch.meshgrid(axis(bz), axis(by), axis(bx), indexing="ij")
    return torch.stack([x, y, z], dim=-1).reshape(-1, 3)
