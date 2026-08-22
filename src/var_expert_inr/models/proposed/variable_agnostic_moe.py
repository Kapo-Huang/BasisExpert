from typing import Dict, Optional

import torch
import torch.nn as nn

from .components import ExpertEncoder, PositionalEncoding, SirenMLP, SpatialGating


class VariableAgnosticMoE(nn.Module):
    supports_expert_routing_aux = True

    def __init__(
        self,
        in_features: int,
        view_specs: Dict[str, int],
        num_experts: int = 7,
        expert_feature_dim: int = 128,
        top_k: int = 3,
        expert_num_frequencies: int = 6,
        expert_hidden_dim: int = 128,
        expert_num_layers: int = 3,
        gate_hidden_dim: int = 128,
        gate_num_layers: int = 3,
        decoder_feature_dim: int = 128,
        decoder_hidden_dim: int = 128,
        decoder_num_layers: int = 3,
        head_hidden_dim: Optional[int] = None,
        head_num_layers: int = 2,
        expert_first_omega_0: float = 30.0,
        expert_hidden_omega_0: float = 30.0,
        gate_first_omega_0: float = 30.0,
        gate_hidden_omega_0: float = 30.0,
        decoder_first_omega_0: float = 30.0,
        decoder_hidden_omega_0: float = 30.0,
        head_first_omega_0: float = 30.0,
        head_hidden_omega_0: float = 30.0,
    ):
        super().__init__()
        if top_k < 1:
            raise ValueError("top_k must be >= 1")
        if not view_specs:
            raise ValueError("view_specs must be a non-empty dict")
        if head_num_layers < 2:
            raise ValueError("head_num_layers must be >= 2")

        self.view_names = list(view_specs.keys())
        self.view_dims = dict(view_specs)
        self.num_views = len(self.view_names)
        self.num_experts = num_experts
        self.top_k = min(top_k, num_experts)

        self.pos_enc = PositionalEncoding(
            in_features=in_features,
            num_frequencies=expert_num_frequencies,
        )
        pe_dim = self.pos_enc.out_dim
        self.gating = SpatialGating(
            in_features=pe_dim,
            num_experts=num_experts,
            hidden_dim=gate_hidden_dim,
            num_layers=gate_num_layers,
            first_omega_0=gate_first_omega_0,
            hidden_omega_0=gate_hidden_omega_0,
        )

        self.experts = nn.ModuleList(
            [
                ExpertEncoder(
                    in_features=pe_dim,
                    feature_dim=expert_feature_dim,
                    use_positional_encoding=False,
                    hidden_dim=expert_hidden_dim,
                    num_layers=expert_num_layers,
                    first_omega_0=expert_first_omega_0,
                    hidden_omega_0=expert_hidden_omega_0,
                )
                for _ in range(num_experts)
            ]
        )

        self.decoder = SirenMLP(
            in_dim=expert_feature_dim,
            out_dim=decoder_feature_dim,
            hidden_dim=decoder_hidden_dim,
            num_layers=decoder_num_layers,
            first_omega_0=decoder_first_omega_0,
            hidden_omega_0=decoder_hidden_omega_0,
        )
        head_hidden_dim = decoder_feature_dim if head_hidden_dim is None else int(head_hidden_dim)
        self.heads = nn.ModuleDict(
            {
                name: SirenMLP(
                    in_dim=decoder_feature_dim,
                    out_dim=out_dim,
                    hidden_dim=head_hidden_dim,
                    num_layers=head_num_layers,
                    first_omega_0=head_first_omega_0,
                    hidden_omega_0=head_hidden_omega_0,
                )
                for name, out_dim in self.view_dims.items()
            }
        )

    def _topk_mask(self, probs: torch.Tensor) -> torch.Tensor:
        _, indices = torch.topk(probs, k=self.top_k, dim=-1)
        mask = torch.zeros_like(probs)
        return mask.scatter_(1, indices, 1.0)

    def forward(
        self,
        coords: torch.Tensor,
        request: Optional[str] = None,
        *,
        hard_topk: bool = True,
        return_aux: bool = False,
    ):
        if request is not None and request not in self.view_dims:
            raise KeyError(f"Unknown view '{request}'")

        x_pe = self.pos_enc(coords)
        expert_feats = torch.stack([expert(x_pe) for expert in self.experts], dim=1)
        probs, _ = self.gating(x_pe)
        mask = self._topk_mask(probs)
        masked_probs = probs * mask
        masked_probs = masked_probs / (masked_probs.sum(dim=-1, keepdim=True) + 1e-9)
        weights = masked_probs if hard_topk else probs

        h_shared = torch.sum(expert_feats * weights.unsqueeze(-1), dim=1)
        shared_feat = self.decoder(h_shared)
        selected_names = self.view_names if request is None else [request]
        preds = {name: self.heads[name](shared_feat) for name in selected_names}

        output = preds if request is None else preds[request]
        if return_aux:
            view_count = len(selected_names)
            return output, {
                "probs": probs.unsqueeze(1).expand(-1, view_count, -1),
                "masks": mask.unsqueeze(1).expand(-1, view_count, -1),
                "H_views": h_shared.unsqueeze(1).expand(-1, view_count, -1),
                "H_shared": shared_feat.unsqueeze(1).expand(-1, view_count, -1),
                "expert_feats": expert_feats,
            }
        return output

    def pretrain_forward(self, coords: torch.Tensor) -> torch.Tensor:
        x_pe = self.pos_enc(coords)
        _, logits = self.gating(x_pe)
        return logits

    def pretrain_parameters(self):
        return list(self.gating.parameters())


def build_variable_agnostic_moe_from_config(
    cfg: Dict,
    view_specs: Dict[str, int],
) -> VariableAgnosticMoE:
    base_dim = cfg.get("base_dim")
    head_hidden_raw = cfg.get("head_hidden_dim")
    decoder_feature_raw = cfg.get("decoder_feature_dim")
    if base_dim is not None:
        base_dim = int(base_dim)
        expert_feature_dim = 8 * base_dim
        expert_hidden_dim = 8 * base_dim
        gate_hidden_dim = int(cfg.get("gate_hidden_dim", 8 * base_dim))
        decoder_hidden_dim = 8 * base_dim
    else:
        expert_feature_dim = int(cfg.get("expert_feature_dim", 128))
        expert_hidden_dim = int(cfg.get("expert_hidden_dim", 128))
        gate_hidden_dim = int(cfg.get("gate_hidden_dim", 128))
        decoder_hidden_dim = int(cfg.get("decoder_hidden_dim", 128))
    decoder_feature_dim = (
        int(decoder_feature_raw) if decoder_feature_raw is not None else expert_feature_dim
    )
    head_hidden_dim = int(head_hidden_raw) if head_hidden_raw is not None else decoder_feature_dim
    return VariableAgnosticMoE(
        in_features=int(cfg.get("in_features", 4)),
        view_specs=view_specs,
        num_experts=int(cfg.get("num_experts", 7)),
        expert_feature_dim=expert_feature_dim,
        top_k=int(cfg.get("top_k", 3)),
        expert_num_frequencies=int(cfg.get("expert_num_frequencies", 6)),
        expert_hidden_dim=expert_hidden_dim,
        expert_num_layers=int(cfg.get("expert_num_layers", 3)),
        gate_hidden_dim=gate_hidden_dim,
        gate_num_layers=int(cfg.get("gate_num_layers", 3)),
        decoder_feature_dim=decoder_feature_dim,
        decoder_hidden_dim=decoder_hidden_dim,
        decoder_num_layers=int(cfg.get("decoder_num_layers", 3)),
        head_hidden_dim=head_hidden_dim,
        head_num_layers=int(cfg.get("head_num_layers", 2)),
        expert_first_omega_0=float(cfg.get("expert_first_omega_0", 30.0)),
        expert_hidden_omega_0=float(cfg.get("expert_hidden_omega_0", 30.0)),
        gate_first_omega_0=float(cfg.get("gate_first_omega_0", 30.0)),
        gate_hidden_omega_0=float(cfg.get("gate_hidden_omega_0", 30.0)),
        decoder_first_omega_0=float(cfg.get("decoder_first_omega_0", 30.0)),
        decoder_hidden_omega_0=float(cfg.get("decoder_hidden_omega_0", 30.0)),
        head_first_omega_0=float(cfg.get("head_first_omega_0", 30.0)),
        head_hidden_omega_0=float(cfg.get("head_hidden_omega_0", 30.0)),
    )
