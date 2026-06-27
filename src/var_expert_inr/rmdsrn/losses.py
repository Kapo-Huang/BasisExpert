from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .model import RMDSRN


@dataclass(frozen=True)
class RMDSRNLoss:
    total: torch.Tensor
    member: torch.Tensor
    variance: torch.Tensor
    mean: torch.Tensor
    predicted_variance: torch.Tensor
    error_density: torch.Tensor
    variance_density: torch.Tensor


def variance_regularization_loss(
    mean: torch.Tensor,
    predicted_variance: torch.Tensor,
    target: torch.Tensor,
    *,
    epsilon: float = 1.0e-12,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if mean.shape != target.shape or predicted_variance.shape != target.shape:
        raise ValueError(
            "mean, predicted_variance, and target must have matching shapes; "
            f"got {tuple(mean.shape)}, {tuple(predicted_variance.shape)}, {tuple(target.shape)}"
        )
    error = (mean.detach() - target).square().mean(dim=-1)
    variance = predicted_variance.mean(dim=-1)
    error_mass = error + float(epsilon)
    variance_mass = variance + float(epsilon)
    error_density = error_mass / error_mass.sum()
    variance_density = variance_mass / variance_mass.sum()
    kl = torch.sum(error_density * (torch.log(error_density) - torch.log(variance_density)))
    return kl, error_density, variance_density


def rmdsrn_loss(
    member_predictions: torch.Tensor,
    target: torch.Tensor,
    *,
    variance_weight: float,
    epsilon: float = 1.0e-12,
) -> RMDSRNLoss:
    if target.ndim != 2:
        raise ValueError(f"target must have shape [batch, channels], got {tuple(target.shape)}")
    if member_predictions.shape[0] != target.shape[0] or member_predictions.shape[2] != target.shape[1]:
        raise ValueError(
            "member prediction and target shape mismatch: "
            f"{tuple(member_predictions.shape)} vs {tuple(target.shape)}"
        )
    expanded_target = target.unsqueeze(1).expand_as(member_predictions)
    member_loss = F.mse_loss(member_predictions, expanded_target)
    mean, predicted_variance = RMDSRN.ensemble_statistics(member_predictions)
    variance_loss, error_density, variance_density = variance_regularization_loss(
        mean,
        predicted_variance,
        target,
        epsilon=epsilon,
    )
    total = member_loss + float(variance_weight) * variance_loss
    return RMDSRNLoss(
        total=total,
        member=member_loss,
        variance=variance_loss,
        mean=mean,
        predicted_variance=predicted_variance,
        error_density=error_density,
        variance_density=variance_density,
    )


def exponential_variance_weight(
    step: int,
    total_steps: int,
    *,
    minimum: float,
    maximum: float,
    growth_rate: float,
) -> float:
    if int(total_steps) <= 0:
        raise ValueError("total_steps must be positive")
    if int(step) < 1 or int(step) > int(total_steps):
        raise ValueError(f"step must be in [1, {int(total_steps)}], got {step}")
    if float(maximum) < float(minimum):
        raise ValueError("maximum must be greater than or equal to minimum")
    progress = float(step) / float(total_steps)
    rate = float(growth_rate)
    if math_is_close_to_one(rate):
        fraction = progress
    else:
        fraction = (rate**progress - 1.0) / (rate - 1.0)
    return float(minimum) + (float(maximum) - float(minimum)) * fraction


def math_is_close_to_one(value: float) -> bool:
    return abs(float(value) - 1.0) <= 1.0e-12
