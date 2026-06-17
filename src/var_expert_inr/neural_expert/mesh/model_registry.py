from __future__ import annotations

from .inr import INR
from .inr_moe import INR_MoE
from .losses import MeshReconstructionLoss


def build_model(cfg, loss_cfg):
    model_name = cfg["MODEL"]["model_name"]
    model_dict = {
        "inr_mesh": INR,
        "inr_moe_mesh": INR_MoE,
    }
    if model_name not in model_dict:
        raise NotImplementedError("NeuralExpert mesh only supports inr_mesh and inr_moe_mesh")
    model = model_dict[model_name](cfg_all=cfg)
    loss = MeshReconstructionLoss(cfg=loss_cfg, model_name=model_name, model=model, n_experts=cfg["MODEL"]["n_experts"])
    return model, loss
