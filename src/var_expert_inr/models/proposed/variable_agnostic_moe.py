from typing import Dict

import torch

from .var_expert import VarExpert, _var_expert_kwargs_from_config


class VarExpertNoEmbedding(VarExpert):
    """VarExpert ablation with variable-agnostic, coordinate-dependent routing."""

    def _routing_view_embedding(self, view_ids: torch.Tensor) -> torch.Tensor:
        view_embed = self.view_embedding(view_ids)
        return torch.zeros_like(view_embed)


def build_var_expert_no_embedding_from_config(
    cfg: Dict,
    view_specs: Dict[str, int],
) -> VarExpertNoEmbedding:
    return VarExpertNoEmbedding(**_var_expert_kwargs_from_config(cfg, view_specs))


# Backward-compatible direct-Python names for the former unregistered model.
VariableAgnosticMoE = VarExpertNoEmbedding
build_variable_agnostic_moe_from_config = build_var_expert_no_embedding_from_config
