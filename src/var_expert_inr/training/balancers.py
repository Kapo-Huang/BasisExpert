from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn


class MultiAttrEMALoss(nn.Module):
    def __init__(
        self,
        attr_names,
        beta: float = 0.95,
        eps: float = 1e-8,
        w_min: float = 0.2,
        w_max: float = 5.0,
        warmup_steps: int = 0,
        alpha: float = 1.0,
    ):
        super().__init__()
        attr_names = list(attr_names)
        if not attr_names:
            raise ValueError("attr_names must be non-empty")
        if len(set(attr_names)) != len(attr_names):
            raise ValueError("attr_names must not contain duplicates")
        self.attr_names = attr_names
        self.beta = float(beta)
        self.eps = float(eps)
        self.w_min = float(w_min)
        self.w_max = float(w_max)
        self.warmup_steps = int(warmup_steps)
        self.alpha = float(alpha)
        n_attrs = len(self.attr_names)
        self.register_buffer("ema", torch.zeros(n_attrs))
        self.register_buffer("baseline_ema", torch.ones(n_attrs))
        self.register_buffer("ema_initialized", torch.tensor(False, dtype=torch.bool))
        self.register_buffer("baseline_initialized", torch.tensor(False, dtype=torch.bool))
        self.register_buffer("step", torch.zeros((), dtype=torch.long))

    @torch.no_grad()
    def _update_ema(self, losses_vec: torch.Tensor) -> None:
        losses_vec = losses_vec.detach()
        if not bool(self.ema_initialized.item()):
            self.ema.copy_(losses_vec)
            self.ema_initialized.fill_(True)
        else:
            self.ema.mul_(self.beta).add_(losses_vec * (1.0 - self.beta))

    @torch.no_grad()
    def _maybe_init_baseline(self, step_after_update: int) -> None:
        effective_warmup = max(1, self.warmup_steps)
        if (
            bool(self.ema_initialized.item())
            and not bool(self.baseline_initialized.item())
            and step_after_update >= effective_warmup
        ):
            self.baseline_ema.copy_(self.ema.clamp_min(self.eps))
            self.baseline_initialized.fill_(True)

    @torch.no_grad()
    def _relative_remaining(self) -> torch.Tensor:
        if not bool(self.baseline_initialized.item()):
            return torch.ones_like(self.ema)
        return (self.ema / self.baseline_ema.clamp_min(self.eps)).clamp_min(self.eps)

    @torch.no_grad()
    def _weights_from_relative_progress(self) -> torch.Tensor:
        relative_remaining = self._relative_remaining()
        # Larger relative_remaining means this attribute has improved more
        # slowly relative to its own warm-up loss, so it receives a larger
        # weight. This is not inverse raw-loss weighting.
        scores = relative_remaining.pow(self.alpha)
        w = scores / scores.mean().clamp_min(self.eps)
        w = torch.clamp(w, self.w_min, self.w_max)
        return w / w.mean().clamp_min(self.eps)

    def _to_ordered_tensor(self, per_attr_losses: Dict[str, torch.Tensor]) -> torch.Tensor:
        return torch.stack([per_attr_losses[name] for name in self.attr_names], dim=0)

    def forward(
        self,
        per_attr_losses: Dict[str, torch.Tensor],
        *,
        return_details: bool = False,
        return_tensors: bool = False,
        update_ema: bool = True,
    ):
        losses_t = self._to_ordered_tensor(per_attr_losses)
        step_now = int(self.step.detach().item())
        if update_ema:
            self._update_ema(losses_t.detach())
            self._maybe_init_baseline(step_now + 1)
        if bool(self.baseline_initialized.item()):
            with torch.no_grad():
                w = self._weights_from_relative_progress()
        else:
            w = torch.ones_like(losses_t)
        if update_ema:
            with torch.no_grad():
                self.step.add_(1)
        w = w.to(device=losses_t.device, dtype=losses_t.dtype)
        total = torch.sum(w * losses_t)
        if not return_details and not return_tensors:
            return total
        details = None
        if return_details:
            with torch.no_grad():
                relative_remaining = self._relative_remaining()
                relative_progress = 1.0 - relative_remaining
            details = {
                "per_attr_loss": {n: float(losses_t[i].detach().item()) for i, n in enumerate(self.attr_names)},
                "ema": {n: float(self.ema[i].detach().item()) for i, n in enumerate(self.attr_names)},
                "baseline_ema": {n: float(self.baseline_ema[i].detach().item()) for i, n in enumerate(self.attr_names)},
                "relative_remaining": {
                    n: float(relative_remaining[i].detach().item()) for i, n in enumerate(self.attr_names)
                },
                "relative_progress": {
                    n: float(relative_progress[i].detach().item()) for i, n in enumerate(self.attr_names)
                },
                "weights": {n: float(w[i].detach().item()) for i, n in enumerate(self.attr_names)},
                "total": float(total.detach().item()),
                "step": int(self.step.detach().item()),
                "warmup_steps": int(self.warmup_steps),
                "alpha": float(self.alpha),
                "baseline_initialized": bool(self.baseline_initialized.item()),
            }
        if return_details and return_tensors:
            return total, losses_t, w, details
        if return_details:
            return total, details
        return total, losses_t, w


class MultiAttrDWALoss(nn.Module):
    """Original Dynamic Weight Average loss balancer.

    Buffer semantics:
    current_weights is the weight vector used by the current epoch.
    epoch_loss_sum accumulates detached per-attribute batch losses in the current epoch.
    epoch_batch_count counts training batches accumulated in the current epoch.
    last_epoch_loss stores the most recent completed epoch mean loss.
    second_last_epoch_loss stores the completed epoch mean loss before last_epoch_loss.
    completed_epochs counts epochs submitted through end_epoch().
    """

    def __init__(
        self,
        attr_names,
        temperature: float = 2.0,
        eps: float = 1e-12,
    ):
        super().__init__()
        attr_names = list(attr_names)
        if not attr_names:
            raise ValueError("attr_names must be non-empty")
        if len(set(attr_names)) != len(attr_names):
            raise ValueError("attr_names must not contain duplicates")
        if float(temperature) <= 0.0:
            raise ValueError("temperature must be > 0")
        if float(eps) <= 0.0:
            raise ValueError("eps must be > 0")
        self.attr_names = attr_names
        self.temperature = float(temperature)
        self.eps = float(eps)
        n_attrs = len(self.attr_names)
        self.register_buffer("current_weights", torch.ones(n_attrs))
        self.register_buffer("epoch_loss_sum", torch.zeros(n_attrs))
        self.register_buffer("last_epoch_loss", torch.zeros(n_attrs))
        self.register_buffer("second_last_epoch_loss", torch.zeros(n_attrs))
        self.register_buffer("epoch_batch_count", torch.zeros((), dtype=torch.long))
        self.register_buffer("completed_epochs", torch.zeros((), dtype=torch.long))

    def _to_ordered_tensor(self, per_attr_losses: Dict[str, torch.Tensor]) -> torch.Tensor:
        for name in self.attr_names:
            if name not in per_attr_losses:
                raise KeyError(f"Missing loss for attribute: {name!r}")
            loss_tensor = per_attr_losses[name]
            if not torch.is_tensor(loss_tensor):
                raise TypeError(f"Loss for attribute {name!r} must be a torch.Tensor")
            if loss_tensor.dim() != 0:
                raise ValueError(f"Loss for attribute {name!r} must be a scalar tensor")
            if not bool(torch.isfinite(loss_tensor.detach()).all().item()):
                raise ValueError(f"Loss for attribute {name!r} must be finite")
        return torch.stack([per_attr_losses[name] for name in self.attr_names], dim=0)

    def forward(
        self,
        per_attr_losses: Dict[str, torch.Tensor],
        *,
        return_details: bool = False,
        return_tensors: bool = False,
        update_stats: bool = True,
    ):
        losses_t = self._to_ordered_tensor(per_attr_losses)
        weights = self.current_weights.to(
            device=losses_t.device,
            dtype=losses_t.dtype,
        )
        total = torch.sum(weights * losses_t)
        if update_stats:
            with torch.no_grad():
                self.epoch_loss_sum.add_(
                    losses_t.detach().to(
                        device=self.epoch_loss_sum.device,
                        dtype=self.epoch_loss_sum.dtype,
                    )
                )
                self.epoch_batch_count.add_(1)
        if not return_details and not return_tensors:
            return total
        details = None
        if return_details:
            details = {
                "method": "dwa",
                "per_attr_loss": {n: float(losses_t[i].detach().item()) for i, n in enumerate(self.attr_names)},
                "weights": {n: float(weights[i].detach().item()) for i, n in enumerate(self.attr_names)},
                "total": float(total.detach().item()),
                "temperature": float(self.temperature),
                "completed_epochs": int(self.completed_epochs.detach().item()),
                "current_epoch_batch_count": int(self.epoch_batch_count.detach().item()),
            }
        if return_details and return_tensors:
            return total, losses_t, weights, details
        if return_details:
            return total, details
        return total, losses_t, weights

    @torch.no_grad()
    def end_epoch(
        self,
        *,
        return_details: bool = False,
    ):
        epoch_batches = int(self.epoch_batch_count.detach().item())
        if epoch_batches <= 0:
            raise ValueError("Cannot end a DWA epoch with no accumulated training batches")
        batch_count_t = self.epoch_batch_count.to(
            device=self.epoch_loss_sum.device,
            dtype=self.epoch_loss_sum.dtype,
        )
        epoch_loss = self.epoch_loss_sum / batch_count_t
        self.second_last_epoch_loss.copy_(self.last_epoch_loss)
        self.last_epoch_loss.copy_(epoch_loss)
        self.completed_epochs.add_(1)

        loss_ratio = None
        if int(self.completed_epochs.detach().item()) >= 2:
            loss_ratio = self.last_epoch_loss / self.second_last_epoch_loss.clamp_min(self.eps)
            next_weights = float(len(self.attr_names)) * torch.softmax(
                loss_ratio / float(self.temperature),
                dim=0,
            )
            self.current_weights.copy_(
                next_weights.to(
                    device=self.current_weights.device,
                    dtype=self.current_weights.dtype,
                )
            )
        else:
            self.current_weights.fill_(1.0)

        details = None
        if return_details:
            details = {
                "method": "dwa",
                "epoch_loss": {n: float(epoch_loss[i].detach().item()) for i, n in enumerate(self.attr_names)},
                "next_weights": {
                    n: float(self.current_weights[i].detach().item()) for i, n in enumerate(self.attr_names)
                },
                "temperature": float(self.temperature),
                "completed_epochs": int(self.completed_epochs.detach().item()),
                "epoch_batch_count": epoch_batches,
            }
            if loss_ratio is not None:
                details["loss_ratio"] = {
                    n: float(loss_ratio[i].detach().item()) for i, n in enumerate(self.attr_names)
                }

        self.epoch_loss_sum.zero_()
        self.epoch_batch_count.zero_()
        return details if return_details else None


def _get_trainable_params(model: torch.nn.Module) -> List[torch.nn.Parameter]:
    return [param for param in model.parameters() if param.requires_grad]


def _flatten_grads(
    grads: Sequence[Optional[torch.Tensor]],
    params: Sequence[torch.nn.Parameter],
) -> torch.Tensor:
    flat_parts = []
    for grad, param in zip(grads, params):
        grad_tensor = grad if grad is not None else torch.zeros_like(param)
        flat_parts.append(grad_tensor.reshape(-1))
    return torch.cat(flat_parts, dim=0) if flat_parts else torch.zeros(0)


class GradNormBalancer:
    def __init__(self, task_names, cfg, device):
        self.task_names = list(task_names)
        self.task_weights = torch.nn.Parameter(torch.ones(len(self.task_names), device=device))
        self.initial_losses = None
        self.alpha = float(cfg.gradnorm_alpha)
        self.optimizer = torch.optim.Adam([self.task_weights], lr=float(cfg.gradnorm_lr))

    def get_weight_dict(self) -> Dict[str, float]:
        weights = self.task_weights.detach().cpu().tolist()
        return {task_name: float(weight) for task_name, weight in zip(self.task_names, weights)}

    def build_weighted_loss(self, task_losses: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        ordered_losses = torch.stack([task_losses[task_name] for task_name in self.task_names], dim=0)
        return torch.sum(self.task_weights * ordered_losses), ordered_losses

    def update(self, task_losses: Dict[str, torch.Tensor], model: torch.nn.Module) -> Dict[str, object]:
        _, ordered_losses = self.build_weighted_loss(task_losses)
        if self.initial_losses is None:
            self.initial_losses = ordered_losses.detach().clamp_min(1e-12)
        shared_params = [param for name, param in model.named_parameters() if param.requires_grad and not name.startswith("backbone.heads.")]
        if not shared_params:
            raise ValueError("No shared trainable parameters found for GradNorm.")
        grad_norm_list = []
        for index, loss_tensor in enumerate(ordered_losses):
            weighted_task_loss_i = self.task_weights[index] * loss_tensor
            grads = torch.autograd.grad(
                weighted_task_loss_i,
                shared_params,
                allow_unused=True,
                create_graph=True,
                retain_graph=True,
            )
            grad_norm_list.append(torch.norm(_flatten_grads(grads, shared_params), p=2))
        grad_norms = torch.stack(grad_norm_list, dim=0)
        loss_ratios = ordered_losses.detach() / self.initial_losses
        relative_rates = loss_ratios / loss_ratios.mean().clamp_min(1e-12)
        grad_norm_avg = grad_norms.detach().mean()
        target_grad_norms = grad_norm_avg * (relative_rates.detach() ** self.alpha)
        grad_loss = torch.sum(torch.abs(grad_norms - target_grad_norms))
        self.optimizer.zero_grad()
        torch.autograd.backward(grad_loss, inputs=[self.task_weights], retain_graph=True)
        self.optimizer.step()
        with torch.no_grad():
            self.task_weights.data.clamp_(min=1e-6)
            self.task_weights.data = self.task_weights.data * (
                len(self.task_names) / self.task_weights.data.sum().clamp_min(1e-12)
            )
        return {
            "grad_loss": float(grad_loss.detach().item()),
            "weights": self.get_weight_dict(),
            "grad_norms": {
                task_name: float(value)
                for task_name, value in zip(self.task_names, grad_norms.detach().cpu().tolist())
            },
            "relative_rates": {
                task_name: float(value)
                for task_name, value in zip(self.task_names, relative_rates.detach().cpu().tolist())
            },
        }


def _set_flat_grad(params: Sequence[torch.nn.Parameter], flat_grad: torch.Tensor) -> None:
    offset = 0
    for param in params:
        numel = param.numel()
        param.grad = flat_grad[offset : offset + numel].view_as(param).clone()
        offset += numel


def _compute_task_grad_matrix(
    task_losses: Dict[str, torch.Tensor],
    params: Sequence[torch.nn.Parameter],
) -> Tuple[List[str], torch.Tensor]:
    task_names = list(task_losses.keys())
    rows = []
    for task_name in task_names:
        grads = torch.autograd.grad(
            task_losses[task_name],
            params,
            allow_unused=True,
            create_graph=False,
            retain_graph=True,
        )
        rows.append(_flatten_grads(grads, params))
    return task_names, torch.stack(rows, dim=0)


def _project_to_simplex(v: torch.Tensor) -> torch.Tensor:
    n = int(v.numel())
    if n <= 1:
        return torch.ones_like(v)
    u, _ = torch.sort(v, descending=True)
    cssv = torch.cumsum(u, dim=0) - 1.0
    ind = torch.arange(1, n + 1, device=v.device, dtype=v.dtype)
    cond = u - cssv / ind > 0
    rho = int(torch.nonzero(cond, as_tuple=False)[-1].item())
    theta = cssv[rho] / ind[rho]
    w = torch.clamp(v - theta, min=0.0)
    return w / torch.sum(w).clamp_min(1e-12)


def _merge_grads_pcgrad(G: torch.Tensor) -> torch.Tensor:
    T = int(G.shape[0])
    projected = G.clone()
    for i in range(T):
        gi = G[i].clone()
        order = torch.randperm(T, device=G.device)
        for j in order.tolist():
            if i == j:
                continue
            gj = G[j]
            dot_ij = torch.dot(gi, gj)
            if dot_ij < 0:
                gi = gi - (dot_ij / (torch.dot(gj, gj) + 1e-12)) * gj
        projected[i] = gi
    return projected.mean(dim=0)


def _solve_mgda_weights(G: torch.Tensor, max_iter: int, lr: float) -> torch.Tensor:
    T = int(G.shape[0])
    if T == 1:
        return torch.ones(1, device=G.device, dtype=G.dtype)
    alpha = torch.full((T,), 1.0 / float(T), device=G.device, dtype=G.dtype)
    A = G @ G.transpose(0, 1)
    eigvals = torch.linalg.eigvalsh(A)
    L = 2.0 * torch.clamp(eigvals.max(), min=1e-12)
    step = min(float(lr), float(1.0 / L))
    for _ in range(int(max_iter)):
        grad = 2.0 * (A @ alpha)
        alpha = _project_to_simplex(alpha - step * grad)
    return alpha


def _merge_grads_mgda(G: torch.Tensor, max_iter: int, lr: float) -> Tuple[torch.Tensor, torch.Tensor]:
    alpha = _solve_mgda_weights(G, max_iter=max_iter, lr=lr)
    return alpha @ G, alpha


def _merge_grads_cagrad(
    G: torch.Tensor,
    c: float,
    max_iter: int,
    lr: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    g0 = torch.mean(G, dim=0)
    A = G @ G.transpose(0, 1)
    b = G @ g0
    T = int(G.shape[0])
    if T == 1:
        return G[0], torch.ones(1, device=G.device, dtype=G.dtype)
    w = torch.full((T,), 1.0 / float(T), device=G.device, dtype=G.dtype)
    eigvals = torch.linalg.eigvalsh(A)
    L = torch.clamp(eigvals.max(), min=1e-12)
    step = min(float(lr), float(1.0 / L))
    for _ in range(int(max_iter)):
        quad = torch.clamp(w @ A @ w, min=1e-12)
        grad = b + float(c) * (A @ w) / torch.sqrt(quad)
        w = _project_to_simplex(w - step * grad)
    gw = w @ G
    return g0 + float(c) * gw / torch.norm(gw).clamp_min(1e-12), w


def apply_multitask_gradient(model: torch.nn.Module, task_losses: Dict[str, torch.Tensor], cfg) -> Dict[str, object]:
    params = _get_trainable_params(model)
    if not params:
        raise ValueError("No trainable parameters found")
    task_names, G = _compute_task_grad_matrix(task_losses, params)
    method = str(cfg.gradient_balancer.method).strip().lower()
    if method == "pcgrad":
        merged_grad = _merge_grads_pcgrad(G)
        task_weights = None
    elif method == "mgda":
        merged_grad, alpha = _merge_grads_mgda(
            G,
            max_iter=int(cfg.gradient_balancer.solver_max_iter),
            lr=float(cfg.gradient_balancer.solver_lr),
        )
        task_weights = alpha.detach().cpu().tolist()
    elif method == "cagrad":
        merged_grad, weights = _merge_grads_cagrad(
            G,
            c=float(cfg.gradient_balancer.cagrad_c),
            max_iter=int(cfg.gradient_balancer.solver_max_iter),
            lr=float(cfg.gradient_balancer.solver_lr),
        )
        task_weights = weights.detach().cpu().tolist()
    else:
        raise ValueError(f"Unsupported gradient balancer method: {method}")
    _set_flat_grad(params, merged_grad)
    return {"method": method, "task_names": task_names, "task_weights": task_weights}
