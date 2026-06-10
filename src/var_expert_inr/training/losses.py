from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F


def pointwise_loss(pred: torch.Tensor, target: torch.Tensor, loss_type: str) -> torch.Tensor:
    if loss_type == "mse":
        return F.mse_loss(pred, target)
    if loss_type == "l1":
        return F.l1_loss(pred, target)
    raise ValueError(f"Unsupported loss_type: {loss_type}")


def reconstruction_loss_with_breakdown(
    preds: Dict[str, torch.Tensor],
    targets: Dict[str, torch.Tensor],
    *,
    weights: Optional[Dict[str, float]] = None,
    loss_type: str = "mse",
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, object], Dict[str, torch.Tensor]]:
    weighted_attr_sum_loss = None
    weighted_dim_numer = None
    weighted_dim_denom = 0.0
    details = {
        "selected_mode": "attr_sum",
        "selected_loss": 0.0,
        "weighted_attr_sum_loss": 0.0,
        "weighted_dim_normalized_loss": 0.0,
        "per_view": {},
        "weight_mode": "static",
    }
    per_view_loss_tensors: Dict[str, torch.Tensor] = {}

    for name, pred in preds.items():
        target = targets[name]
        weight = 1.0 if weights is None else float(weights.get(name, 1.0))
        loss_per_dim = pointwise_loss(pred, target, loss_type=loss_type)
        out_dim = int(pred.shape[-1]) if pred.ndim > 1 else 1
        loss_sum_dims = loss_per_dim * float(out_dim)

        if weighted_attr_sum_loss is None:
            weighted_attr_sum_loss = weight * loss_per_dim
            weighted_dim_numer = weight * loss_sum_dims
        else:
            weighted_attr_sum_loss = weighted_attr_sum_loss + weight * loss_per_dim
            weighted_dim_numer = weighted_dim_numer + weight * loss_sum_dims
        weighted_dim_denom += weight * float(out_dim)

        per_view_loss_tensors[name] = loss_per_dim
        details["per_view"][name] = {
            "dim": float(out_dim),
            "weight": float(weight),
            "loss_sum_dims": float(loss_sum_dims.detach().item()),
            "loss_per_dim": float(loss_per_dim.detach().item()),
        }

    if weighted_attr_sum_loss is None or weighted_dim_numer is None:
        raise ValueError("reconstruction_loss requires at least one prediction-target pair")

    weighted_dim_normalized_loss = weighted_dim_numer / (weighted_dim_denom + 1e-12)
    details["selected_loss"] = float(weighted_attr_sum_loss.detach().item())
    details["weighted_attr_sum_loss"] = float(weighted_attr_sum_loss.detach().item())
    details["weighted_dim_normalized_loss"] = float(weighted_dim_normalized_loss.detach().item())
    return weighted_attr_sum_loss, weighted_dim_normalized_loss, details, per_view_loss_tensors
