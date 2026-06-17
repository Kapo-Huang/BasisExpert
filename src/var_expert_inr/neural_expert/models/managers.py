from __future__ import annotations

import torch
import torch.nn as nn


class DummyManager(nn.Module):
    def __init__(self, n_experts):
        super().__init__()
        self.n_experts = int(n_experts)

    def forward(self, points):
        return torch.zeros(points.shape[0], self.n_experts, device=points.device), (None, None)


class ManagerConditioner(nn.Module):
    def __init__(self, manager_conditioning, last_layer_dim=128, expert_decoder=None):
        super().__init__()
        self.manager_conditioning = str(manager_conditioning)

    def forward(self, x, manager_input, **kwargs):
        if self.manager_conditioning == "max":
            point_rep = torch.max(x, dim=1)[0].unsqueeze(1).expand(-1, manager_input.shape[1], -1)
        elif self.manager_conditioning == "mean":
            point_rep = torch.mean(x, dim=1).unsqueeze(1).expand(-1, manager_input.shape[1], -1)
        elif self.manager_conditioning == "cat":
            point_rep = x
        else:
            point_rep = None

        if point_rep is not None:
            manager_input = torch.cat([manager_input, point_rep], dim=-1)
        return manager_input
