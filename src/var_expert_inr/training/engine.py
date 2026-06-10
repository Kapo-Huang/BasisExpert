from __future__ import annotations

import copy
import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, random_split

from ..pretrain.assignments import PretrainAssignmentConfig, compute_pretrain_assignments
from ..utils.checkpoint import save_checkpoint
from .balancers import GradNormBalancer, MultiAttrEMALoss, apply_multitask_gradient
from .losses import pointwise_loss, reconstruction_loss_with_breakdown
from .samplers import build_train_sampler

logger = logging.getLogger(__name__)


def _collate_factory(dataset, *, include_targets: bool = True, assignments=None):
    def _collate(indices):
        return dataset.fetch_batch(indices, include_targets=include_targets, assignments=assignments)

    return _collate


def _is_multitarget(targets) -> bool:
    return isinstance(targets, dict)


def _split_dataset(dataset, val_split: float, seed: int):
    if val_split <= 0.0 or len(dataset) <= 1:
        return dataset, None
    train_size = max(1, int(round(len(dataset) * (1.0 - float(val_split)))))
    val_size = len(dataset) - train_size
    if val_size <= 0:
        return dataset, None
    generator = torch.Generator().manual_seed(int(seed))
    return random_split(dataset, [train_size, val_size], generator=generator)


def _build_loader(dataset, cfg, *, include_targets: bool = True, assignments=None, shuffle=False, sampler=None):
    return DataLoader(
        dataset,
        batch_size=int(cfg.batch_size if include_targets else cfg.pretrain.batch_size),
        shuffle=shuffle,
        sampler=sampler,
        num_workers=int(cfg.num_workers),
        pin_memory=True,
        collate_fn=_collate_factory(dataset.dataset if isinstance(dataset, Subset) else dataset, include_targets=include_targets, assignments=assignments),
    )


def predict_dataset(model, dataset, *, batch_size: int, device: torch.device, hard_topk: bool = True):
    loader = DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=_collate_factory(dataset),
    )
    model.eval()
    if dataset.meta.is_multitarget:
        collected = {name: [] for name in dataset.target_names()}
    else:
        collected = {dataset.target_names()[0]: []}
    with torch.no_grad():
        for batch in loader:
            coords = batch.coords.to(device)
            preds = model(coords, hard_topk=hard_topk)
            if isinstance(preds, dict):
                for name, tensor in preds.items():
                    tensor = tensor.detach().cpu()
                    collected[name].append(tensor.numpy())
            else:
                collected[dataset.target_names()[0]].append(preds.detach().cpu().numpy())
    return {name: np.concatenate(parts, axis=0) for name, parts in collected.items()}


def save_predictions(dataset, predictions: dict[str, np.ndarray], output_dir: str | Path, exp_id: str) -> dict[str, Path]:
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    saved = {}
    for name, flat in predictions.items():
        array = dataset.reshape_flat_predictions(name, flat)
        save_path = output_root / (f"{exp_id}.npy" if name == "target" else f"{exp_id}_{name}.npy")
        np.save(save_path, array)
        saved[name] = save_path
    return saved


def _run_pretrain(model, dataset, cfg, device):
    if not cfg.pretrain.enabled or cfg.pretrain.epochs <= 0:
        return
    if not hasattr(model, "pretrain_forward"):
        raise ValueError("Configured pretraining but selected model does not expose pretrain_forward")
    assignments_cfg = PretrainAssignmentConfig(
        method=cfg.pretrain.assignments_method,
        seed=int(cfg.pretrain.cluster_seed),
        cache_path=str(cfg.pretrain.assignments_cache_path or ""),
        cluster_num_time_samples=int(cfg.pretrain.cluster_num_time_samples),
        spatial_blocks=cfg.pretrain.spatial_blocks,
        time_block_size=int(cfg.pretrain.time_block_size),
    )
    assignments = compute_pretrain_assignments(dataset, int(model.num_experts), assignments_cfg)
    pretrain_loader = DataLoader(
        dataset,
        batch_size=int(cfg.pretrain.batch_size),
        shuffle=True,
        num_workers=int(cfg.num_workers),
        pin_memory=True,
        collate_fn=_collate_factory(dataset, include_targets=False, assignments=assignments),
    )
    optimizer = torch.optim.Adam(list(model.pretrain_parameters()), lr=float(cfg.pretrain.lr))
    for epoch in range(1, cfg.pretrain.epochs + 1):
        model.train()
        epoch_loss = 0.0
        total = 0
        correct = 0
        for batch in pretrain_loader:
            coords = batch.coords.to(device)
            expert_ids = batch.expert_ids.to(device)
            logits = model.pretrain_forward(coords)
            loss = torch.nn.functional.cross_entropy(logits, expert_ids)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item()) * coords.shape[0]
            total += int(coords.shape[0])
            correct += int((torch.argmax(logits, dim=-1) == expert_ids).sum().item())
        logger.info(
            "Pretrain epoch %s/%s loss=%.6e acc=%.4f",
            epoch,
            cfg.pretrain.epochs,
            epoch_loss / max(total, 1),
            correct / max(total, 1),
        )


def train_model(
    *,
    model,
    dataset,
    cfg,
    device: torch.device,
    checkpoint_dir: str | Path,
    config_hash: str,
    prediction_dir: str | Path,
    exp_id: str,
):
    train_dataset, val_dataset = _split_dataset(dataset, cfg.val_split, cfg.seed)
    sampler = build_train_sampler(train_dataset, cfg)
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(cfg.batch_size),
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=int(cfg.num_workers),
        pin_memory=True,
        collate_fn=_collate_factory(dataset),
    )
    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=int(cfg.batch_size),
            shuffle=False,
            num_workers=int(cfg.num_workers),
            pin_memory=True,
            collate_fn=_collate_factory(dataset),
        )

    model = model.to(device)
    _run_pretrain(model, dataset, cfg, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg.lr), weight_decay=float(cfg.weight_decay))
    scheduler = None
    if cfg.scheduler.enabled and cfg.scheduler.step_size > 0 and cfg.scheduler.gamma != 1.0:
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=int(cfg.scheduler.step_size),
            gamma=float(cfg.scheduler.gamma),
        )

    ema_balancer = None
    gradnorm_balancer = None
    if dataset.meta.is_multitarget and cfg.multiview_ema_loss.enabled:
        ema_balancer = MultiAttrEMALoss(
            dataset.target_names(),
            beta=float(cfg.multiview_ema_loss.beta),
            eps=float(cfg.multiview_ema_loss.eps),
            w_min=float(cfg.multiview_ema_loss.w_min),
            w_max=float(cfg.multiview_ema_loss.w_max),
            warmup_steps=int(cfg.multiview_ema_loss.warmup_steps),
        ).to(device)
    if dataset.meta.is_multitarget and cfg.gradient_balancer.enabled and cfg.gradient_balancer.method == "gradnorm":
        gradnorm_balancer = GradNormBalancer(dataset.target_names(), cfg.gradient_balancer, device)

    best_state = None
    best_val = float("inf")
    no_improve = 0
    router_frozen = False

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        epoch_loss = 0.0
        steps = 0
        hard_topk = int(epoch) > int(cfg.hard_topk_warmup_epochs)

        if not router_frozen and cfg.freeze_router_at > 0 and (epoch / max(cfg.epochs, 1)) >= cfg.freeze_router_at:
            frozen = 0
            for name, param in model.named_parameters():
                if any(token in name.lower() for token in ("gating", "policy", "router")):
                    param.requires_grad = False
                    frozen += param.numel()
            router_frozen = frozen > 0
            if router_frozen:
                logger.info("Froze router-like parameters at epoch %s", epoch)

        for batch in train_loader:
            coords = batch.coords.to(device)
            optimizer.zero_grad()
            if _is_multitarget(batch.targets):
                targets = {name: tensor.to(device) for name, tensor in batch.targets.items()}
                preds = model(coords, hard_topk=hard_topk)
                total_loss, _, _, task_losses = reconstruction_loss_with_breakdown(
                    preds,
                    targets,
                    loss_type=cfg.loss_type,
                )
                loss_to_log = total_loss
                if ema_balancer is not None:
                    total_loss = ema_balancer(task_losses)
                if gradnorm_balancer is not None:
                    total_loss, _ = gradnorm_balancer.build_weighted_loss(task_losses)
                    total_loss.backward(retain_graph=True)
                    gradnorm_balancer.update(task_losses, model)
                elif cfg.gradient_balancer.enabled:
                    apply_multitask_gradient(model, task_losses, cfg)
                else:
                    total_loss.backward()
            else:
                targets = batch.targets.to(device)
                preds = model(coords, hard_topk=hard_topk)
                total_loss = pointwise_loss(preds, targets, cfg.loss_type)
                loss_to_log = total_loss
                total_loss.backward()
            optimizer.step()
            epoch_loss += float(loss_to_log.detach().item())
            steps += 1

        if scheduler is not None:
            scheduler.step()

        val_loss = None
        if val_loader is not None:
            model.eval()
            loss_sum = 0.0
            count = 0
            with torch.no_grad():
                for batch in val_loader:
                    coords = batch.coords.to(device)
                    if _is_multitarget(batch.targets):
                        targets = {name: tensor.to(device) for name, tensor in batch.targets.items()}
                        preds = model(coords, hard_topk=hard_topk)
                        loss, _, _, _ = reconstruction_loss_with_breakdown(
                            preds,
                            targets,
                            loss_type=cfg.loss_type,
                        )
                    else:
                        targets = batch.targets.to(device)
                        preds = model(coords, hard_topk=hard_topk)
                        loss = pointwise_loss(preds, targets, cfg.loss_type)
                    loss_sum += float(loss.item()) * coords.shape[0]
                    count += int(coords.shape[0])
            val_loss = loss_sum / max(count, 1)
            if val_loss < best_val:
                best_val = val_loss
                best_state = copy.deepcopy(model.state_dict())
                no_improve = 0
            else:
                no_improve += 1

        if epoch % cfg.log_every == 0 or epoch == 1:
            if val_loss is None:
                logger.info("Epoch %s/%s train=%.6e", epoch, cfg.epochs, epoch_loss / max(steps, 1))
            else:
                logger.info(
                    "Epoch %s/%s train=%.6e val=%.6e",
                    epoch,
                    cfg.epochs,
                    epoch_loss / max(steps, 1),
                    val_loss,
                )

        if cfg.save_every > 0 and epoch % cfg.save_every == 0:
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                dataset=dataset,
                epoch=epoch,
                config_hash=config_hash,
                path=Path(checkpoint_dir) / f"{exp_id}_epoch{epoch}.pth",
            )

        if cfg.early_stop_patience > 0 and no_improve >= cfg.early_stop_patience:
            logger.info("Early stopping at epoch %s", epoch)
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    final_ckpt = save_checkpoint(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        dataset=dataset,
        epoch=cfg.epochs,
        config_hash=config_hash,
        path=Path(checkpoint_dir) / f"{exp_id}.pth",
    )
    predictions = predict_dataset(
        model,
        dataset,
        batch_size=cfg.pred_batch_size,
        device=device,
        hard_topk=True,
    )
    saved_predictions = save_predictions(dataset, predictions, prediction_dir, exp_id)
    return {
        "checkpoint_path": final_ckpt,
        "predictions": predictions,
        "prediction_paths": saved_predictions,
    }
