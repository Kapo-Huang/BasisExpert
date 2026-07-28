from __future__ import annotations

import math

import numpy as np
import torch

from .model import PackedSiren


WEIGHT_CANDIDATES = ("layers.0.weight", "layers.1.weight", "layers.2.weight")
BIAS_CANDIDATES = ("layers.1.bias", "layers.2.bias")


def initial_pruning_masks(model: PackedSiren) -> dict[str, torch.Tensor]:
    parameters = dict(model.named_parameters())
    return {
        name: torch.ones_like(parameters[name], dtype=torch.bool)
        for name in (*WEIGHT_CANDIDATES, *BIAS_CANDIDATES)
    }


def _normalize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return values
    return (values - values.min()) / (values.max() - values.min() + 1.0e-6)


def _prune_family(
    model: PackedSiren,
    masks: dict[str, torch.Tensor],
    names: tuple[str, ...],
    mlp_losses: np.ndarray,
    target_sparsity: float,
    loss_weight: float,
) -> int:
    parameters = dict(model.named_parameters())
    total = sum(int(parameters[name].numel()) for name in names)
    already = sum(int((~masks[name]).sum().item()) for name in names)
    target_zero = int(math.floor(float(target_sparsity) * total))
    new_count = max(0, target_zero - already)
    if new_count == 0:
        return 0

    active_magnitudes: list[np.ndarray] = []
    active_losses: list[np.ndarray] = []
    keys: list[tuple[int, int, int, str, int]] = []
    for layer_order, name in enumerate(names):
        parameter = parameters[name].detach().cpu().numpy()
        mask = masks[name].detach().cpu().numpy().astype(bool)
        flat_parameter = parameter.reshape(model.mlp_count, -1)
        flat_mask = mask.reshape(model.mlp_count, -1)
        for mlp_index in range(model.mlp_count):
            active_indices = np.flatnonzero(flat_mask[mlp_index])
            if active_indices.size:
                active_magnitudes.append(np.abs(flat_parameter[mlp_index, active_indices]))
                active_losses.append(np.full(active_indices.size, mlp_losses[mlp_index], dtype=np.float64))
                keys.extend(
                    (layer_order, mlp_index, int(flat_index), name, int(flat_index))
                    for flat_index in active_indices.tolist()
                )
    magnitudes = np.concatenate(active_magnitudes) if active_magnitudes else np.empty(0)
    losses = np.concatenate(active_losses) if active_losses else np.empty(0)
    importance = _normalize(magnitudes) + float(loss_weight) * _normalize(losses)
    if importance.size < new_count:
        raise RuntimeError("Requested pruning exceeds active candidate parameters")
    order = sorted(
        range(importance.size),
        key=lambda index: (float(importance[index]), keys[index][0], keys[index][1], keys[index][2]),
    )
    for selected in order[:new_count]:
        _, mlp_index, _, name, flat_index = keys[selected]
        mask_view = masks[name].view(model.mlp_count, -1)
        mask_view[mlp_index, flat_index] = False
    return new_count


def apply_cumulative_pruning(
    model: PackedSiren,
    masks: dict[str, torch.Tensor],
    *,
    mlp_losses: np.ndarray,
    target_sparsity: float,
    loss_weight: float = 0.1,
) -> dict[str, int]:
    losses = np.asarray(mlp_losses, dtype=np.float64)
    if losses.shape != (model.mlp_count,):
        raise ValueError(f"mlp_losses must have shape {(model.mlp_count,)}")
    result = {
        "weight_newly_pruned": _prune_family(
            model, masks, WEIGHT_CANDIDATES, losses, target_sparsity, loss_weight
        ),
        "bias_newly_pruned": _prune_family(
            model, masks, BIAS_CANDIDATES, losses, target_sparsity, loss_weight
        ),
    }
    model.apply_pruning_masks(masks)
    return result


def family_sparsity(masks: dict[str, torch.Tensor], names: tuple[str, ...]) -> float:
    total = sum(int(masks[name].numel()) for name in names)
    zero = sum(int((~masks[name]).sum().item()) for name in names)
    return float(zero / max(total, 1))
