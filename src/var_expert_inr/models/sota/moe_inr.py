import logging
import math
from typing import Optional

import torch
import torch.nn as nn

from .siren import SineLayer

logger = logging.getLogger(__name__)


def _count_parameters(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


class PositionalEncoding(nn.Module):
    def __init__(self, in_features: int, mapping_size: int):
        super().__init__()
        self.in_features = in_features
        self.lin = nn.Linear(in_features, mapping_size, bias=True)

    @property
    def out_dim(self) -> int:
        return 2 * self.lin.out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = self.lin(x)
        return torch.cat([torch.sin(u), torch.cos(u)], dim=-1)


class SirenMLP(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 3,
        first_omega_0: float = 30.0,
        hidden_omega_0: float = 30.0,
    ):
        super().__init__()
        if num_layers < 2:
            raise ValueError("num_layers must be >= 2")

        layers = [SineLayer(in_dim, hidden_dim, omega_0=first_omega_0, is_first=True)]
        for _ in range(num_layers - 2):
            layers.append(SineLayer(hidden_dim, hidden_dim, omega_0=hidden_omega_0))
        self.mlp = nn.Sequential(*layers)
        self.final = nn.Linear(hidden_dim, out_dim)
        with torch.no_grad():
            bound = math.sqrt(6.0 / hidden_dim) / hidden_omega_0
            self.final.weight.uniform_(-bound, bound)
            if self.final.bias is not None:
                self.final.bias.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.final(self.mlp(x))


class BottleneckResBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, bottleneck_dim: Optional[int] = None):
        super().__init__()
        if bottleneck_dim is None:
            bottleneck_dim = max(1, min(in_dim, out_dim) // 4)
        self.fc1 = nn.Linear(in_dim, bottleneck_dim)
        self.fc2 = nn.Linear(bottleneck_dim, bottleneck_dim)
        self.fc3 = nn.Linear(bottleneck_dim, out_dim)
        self.act1 = nn.ReLU()
        self.act2 = nn.ReLU()
        self.act_out = nn.ReLU()
        self.shortcut = nn.Linear(in_dim, out_dim) if in_dim != out_dim else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act1(self.fc1(x))
        h = self.act2(self.fc2(h))
        h = self.fc3(h)
        shortcut = self.shortcut(x) if self.shortcut is not None else x
        return self.act_out(h + shortcut)


class SharedSirenEncoder(nn.Module):
    def __init__(
        self,
        in_features: int = 4,
        feature_dim: int = 256,
        base_dim: Optional[int] = None,
        first_omega_0: float = 30.0,
        hidden_omega_0: float = 30.0,
    ):
        super().__init__()
        if base_dim is None:
            base_dim = max(16, feature_dim // 8)
        pe_mapping_dim = base_dim
        sine1_dim = 4 * base_dim
        sine2_dim = 8 * base_dim
        res_dim = 8 * base_dim

        self.pos_enc = PositionalEncoding(in_features=in_features, mapping_size=pe_mapping_dim)
        self.sine1 = SineLayer(self.pos_enc.out_dim, sine1_dim, omega_0=first_omega_0, is_first=True)
        self.sine2 = SineLayer(sine1_dim, sine2_dim, omega_0=hidden_omega_0)
        self.res_block = BottleneckResBlock(sine2_dim, res_dim)
        self.out_dim = res_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.res_block(self.sine2(self.sine1(self.pos_enc(x))))


class PolicyNetwork(nn.Module):
    def __init__(
        self,
        in_features: int = 4,
        hidden_dim: int = 128,
        num_layers: int = 3,
        num_experts: int = 7,
        gate_in_dim: Optional[int] = None,
        first_omega_0: float = 30.0,
        hidden_omega_0: float = 30.0,
    ):
        super().__init__()
        if num_layers < 2:
            raise ValueError("num_layers must be >= 2")
        layers = [SineLayer(in_features, hidden_dim, omega_0=first_omega_0, is_first=True)]
        for _ in range(num_layers - 2):
            layers.append(SineLayer(hidden_dim, hidden_dim, omega_0=hidden_omega_0))
        self.feature = nn.Sequential(*layers)
        if gate_in_dim is None:
            gate_in_dim = hidden_dim
        self.gate = nn.Linear(gate_in_dim, num_experts)
        with torch.no_grad():
            bound = math.sqrt(6.0 / gate_in_dim) / hidden_omega_0
            self.gate.weight.uniform_(-bound, bound)
            if self.gate.bias is not None:
                self.gate.bias.zero_()

    def forward(self, x: torch.Tensor, encoder_feat: Optional[torch.Tensor] = None):
        feat = self.feature(x)
        gate_input = torch.cat([encoder_feat, feat], dim=-1) if encoder_feat is not None else feat
        logits = self.gate(gate_input)
        return torch.softmax(logits, dim=-1), logits, feat


class ExpertDecoder(nn.Module):
    def __init__(self, in_dim: int, out_features: int = 1):
        super().__init__()
        self.mlp = nn.Linear(in_dim, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class MoEINR(nn.Module):
    def __init__(
        self,
        in_features: int = 4,
        out_features: int = 1,
        num_experts: int = 7,
        encoder_feature_dim: int = 256,
        base_dim: Optional[int] = None,
        encoder_first_omega_0: float = 30.0,
        encoder_hidden_omega_0: float = 30.0,
        policy_hidden_dim: int = 128,
        policy_num_layers: int = 3,
        policy_first_omega_0: float = 30.0,
        policy_hidden_omega_0: float = 30.0,
    ):
        super().__init__()
        self.encoder = SharedSirenEncoder(
            in_features=in_features,
            feature_dim=encoder_feature_dim,
            base_dim=base_dim,
            first_omega_0=encoder_first_omega_0,
            hidden_omega_0=encoder_hidden_omega_0,
        )
        self.policy = PolicyNetwork(
            in_features=in_features,
            hidden_dim=policy_hidden_dim,
            num_layers=policy_num_layers,
            num_experts=num_experts,
            gate_in_dim=encoder_feature_dim + policy_hidden_dim,
            first_omega_0=policy_first_omega_0,
            hidden_omega_0=policy_hidden_omega_0,
        )
        self.num_experts = num_experts
        self.experts = nn.ModuleList(
            [ExpertDecoder(in_dim=encoder_feature_dim, out_features=out_features) for _ in range(num_experts)]
        )
        logger.info(
            "MoEINR init params: policy_network=%s experts=%s shared_encoder=%s",
            f"{_count_parameters(self.policy):,}",
            f"{_count_parameters(self.experts):,}",
            f"{_count_parameters(self.encoder):,}",
        )

    def forward(self, x: torch.Tensor, *, hard_routing: bool = False, return_all: bool = False):
        enc_feat = self.encoder(x)
        probs, logits, _ = self.policy(x, enc_feat)
        preds_all = torch.stack([expert(enc_feat) for expert in self.experts], dim=1)
        if hard_routing:
            indices = torch.argmax(probs, dim=-1)
            y = preds_all[torch.arange(x.shape[0], device=x.device), indices]
        else:
            y = torch.sum(preds_all * probs.unsqueeze(-1), dim=1)
        if return_all:
            return y, preds_all, probs, logits
        return y

    def pretrain_forward(self, x: torch.Tensor) -> torch.Tensor:
        enc_feat = self.encoder(x)
        _, logits, _ = self.policy(x, enc_feat)
        return logits

    def pretrain_parameters(self):
        return list(self.encoder.parameters()) + list(self.policy.parameters())


def build_moe_inr_from_config(cfg) -> MoEINR:
    base_dim = cfg.get("base_dim")
    if base_dim is not None:
        base_dim = int(base_dim)
        encoder_feature_dim = 8 * base_dim
        policy_hidden_dim = base_dim
    else:
        encoder_feature_dim = int(cfg.get("encoder_feature_dim", 256))
        policy_hidden_dim = int(cfg.get("policy_hidden_dim", 128))
    return MoEINR(
        in_features=int(cfg.get("in_features", 4)),
        out_features=int(cfg.get("out_features", 1)),
        num_experts=int(cfg.get("num_experts", 7)),
        encoder_feature_dim=encoder_feature_dim,
        base_dim=base_dim,
        encoder_first_omega_0=float(cfg.get("encoder_first_omega_0", 30.0)),
        encoder_hidden_omega_0=float(cfg.get("encoder_hidden_omega_0", 30.0)),
        policy_hidden_dim=policy_hidden_dim,
        policy_num_layers=int(cfg.get("policy_num_layers", 3)),
        policy_first_omega_0=float(cfg.get("policy_first_omega_0", 30.0)),
        policy_hidden_omega_0=float(cfg.get("policy_hidden_omega_0", 30.0)),
    )
