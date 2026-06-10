from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from ..data.base import DatasetMeta


def require_single_target(meta: DatasetMeta, model_name: str) -> int:
    if meta.is_multitarget:
        raise ValueError(f"{model_name} only supports single-target datasets")
    return int(meta.target_dims[meta.target_names[0]])


def view_specs_from_meta(meta: DatasetMeta) -> dict[str, int]:
    return dict(meta.target_dims)


class ModelAdapter(nn.Module):
    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone

    def forward(
        self,
        coords: torch.Tensor,
        *,
        return_aux: bool = False,
        hard_topk: bool = True,
        request: str | None = None,
    ):
        try:
            return self.backbone(
                coords,
                request=request,
                return_aux=return_aux,
                hard_topk=hard_topk,
            )
        except TypeError:
            try:
                return self.backbone(coords, request=request, return_aux=return_aux)
            except TypeError:
                try:
                    return self.backbone(coords, return_aux=return_aux, hard_topk=hard_topk)
                except TypeError:
                    try:
                        return self.backbone(coords, hard_topk=hard_topk)
                    except TypeError:
                        return self.backbone(coords)

    def pretrain_forward(self, coords: torch.Tensor) -> torch.Tensor:
        if not hasattr(self.backbone, "pretrain_forward"):
            raise AttributeError(f"{type(self.backbone).__name__} does not support pretraining")
        return self.backbone.pretrain_forward(coords)

    def pretrain_parameters(self):
        if not hasattr(self.backbone, "pretrain_parameters"):
            raise AttributeError(f"{type(self.backbone).__name__} does not support pretraining")
        return self.backbone.pretrain_parameters()

    def __getattr__(self, name: str) -> Any:
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.backbone, name)
