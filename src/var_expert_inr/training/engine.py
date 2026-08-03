from __future__ import annotations

import copy
import logging
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, random_split

from ..config.schema import LogConfig
from ..evaluation.metrics import PSNRAccumulator
from ..models.sota.compact_ngp import CompactNGP, save_compact_ngp_artifact
from ..models.sota.instant_ngp import (
    INSTANT_NGP_DECODER_L2_WEIGHT,
    InstantNGP,
)
from ..models.sota.mvnet import MVNet4D
from ..pretrain.assignments import PretrainAssignmentConfig, compute_pretrain_assignments
from ..utils.checkpoint import save_checkpoint
from ..utils.timing import TimingBreakdown, log_epoch_timing, log_step_timing_window, timing_elapsed, timing_start
from .balancers import GradNormBalancer, MultiAttrDWALoss, MultiAttrEMALoss, apply_multitask_gradient
from .losses import pointwise_loss, reconstruction_loss_with_breakdown
from .samplers import build_pretrain_sampler, build_train_sampler

logger = logging.getLogger(__name__)


def validate_mvnet_training_config(cfg) -> None:
    expected_values = {
        "epochs": (int(cfg.epochs), 300),
        "batch_size": (int(cfg.batch_size), 2048),
        "gradient_accumulation_steps": (
            int(cfg.gradient_accumulation_steps),
            1,
        ),
        "batches_per_epoch_budget": (
            int(cfg.batches_per_epoch_budget),
            1500,
        ),
        "early_stop_patience": (
            int(cfg.early_stop_patience),
            0,
        ),
        "lr": (float(cfg.lr), 1.0e-4),
        "beta_1": (float(cfg.beta_1), 0.9),
        "beta_2": (float(cfg.beta_2), 0.999),
        "epsilon": (float(cfg.epsilon), 1.0e-8),
        "weight_decay": (float(cfg.weight_decay), 0.0),
        "val_split": (float(cfg.val_split), 0.0),
        "loss_type": (str(cfg.loss_type), "mse"),
        "sampler": (str(cfg.sampler), "budgeted_random"),
    }
    mismatches = [
        f"{name}={actual} (expected {expected})"
        for name, (actual, expected) in expected_values.items()
        if actual != expected
    ]
    scheduler = cfg.scheduler
    if not scheduler.enabled:
        mismatches.append("scheduler.enabled=False (expected True)")
    if scheduler.interval != "epoch":
        mismatches.append(
            f"scheduler.interval={scheduler.interval} (expected epoch)"
        )
    if int(scheduler.step_size) != 15:
        mismatches.append(
            f"scheduler.step_size={scheduler.step_size} (expected 15)"
        )
    if float(scheduler.gamma) != 0.8:
        mismatches.append(
            f"scheduler.gamma={scheduler.gamma} (expected 0.8)"
        )
    if scheduler.milestones:
        mismatches.append(
            "scheduler.milestones must be empty for MVNet"
        )
    if cfg.pretrain.enabled:
        mismatches.append("pretrain.enabled=True (expected False)")
    if cfg.gradient_balancer.enabled:
        mismatches.append(
            "gradient_balancer.enabled=True (expected False)"
        )
    if cfg.multiview_ema_loss.enabled:
        mismatches.append(
            "multiview_ema_loss.enabled=True (expected False)"
        )
    if cfg.multiview_dwa_loss.enabled:
        mismatches.append(
            "multiview_dwa_loss.enabled=True (expected False)"
        )
    if mismatches:
        raise ValueError(
            "MVNet training uses a fixed reproduction configuration: "
            + ", ".join(mismatches)
        )


def build_training_scheduler(optimizer, scheduler_cfg):
    if not scheduler_cfg.enabled:
        return None
    if scheduler_cfg.milestones:
        return torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=list(scheduler_cfg.milestones),
            gamma=float(scheduler_cfg.gamma),
        )
    return torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=int(scheduler_cfg.step_size),
        gamma=float(scheduler_cfg.gamma),
    )


def training_budget(
    *,
    epochs: int,
    data_steps_per_epoch: int,
    batch_size: int,
    gradient_accumulation_steps: int,
) -> dict[str, int]:
    data_steps = int(epochs) * int(data_steps_per_epoch)
    accumulation = int(gradient_accumulation_steps)
    return {
        "data_steps": data_steps,
        "samples": data_steps * int(batch_size),
        "optimizer_steps": data_steps // accumulation,
        "remainder": data_steps % accumulation,
    }


def _resolve_base_dataset(dataset):
    return dataset.dataset if isinstance(dataset, Subset) else dataset


def _collate_factory(dataset, *, include_targets: bool = True, assignments=None):
    def _collate(indices):
        return dataset.fetch_batch(indices, include_targets=include_targets, assignments=assignments)

    return _collate


def _is_multitarget(targets) -> bool:
    return isinstance(targets, dict)


def split_multitarget_prediction(
    predictions: torch.Tensor,
    target_names: tuple[str, ...],
) -> dict[str, torch.Tensor]:
    if predictions.ndim != 2:
        raise ValueError(
            "Multi-target tensor predictions must have shape [B, V], "
            f"got {tuple(predictions.shape)}"
        )
    if predictions.shape[1] != len(target_names):
        raise ValueError(
            "Prediction column count does not match target order: "
            f"columns={predictions.shape[1]} targets={len(target_names)}"
        )
    return {
        name: predictions[:, index : index + 1]
        for index, name in enumerate(target_names)
    }


def stack_scalar_targets(
    targets: dict[str, torch.Tensor],
    target_names: tuple[str, ...],
) -> torch.Tensor:
    columns = []
    for name in target_names:
        target = targets[name]
        if target.ndim != 2 or target.shape[1] != 1:
            raise ValueError(
                f"MVNet target {name!r} must have shape [B, 1], "
                f"got {tuple(target.shape)}"
            )
        columns.append(target)
    return torch.cat(columns, dim=-1)


def mvnet_joint_mse(
    predictions: torch.Tensor,
    targets: dict[str, torch.Tensor],
    target_names: tuple[str, ...],
) -> torch.Tensor:
    target_matrix = stack_scalar_targets(targets, target_names)
    if predictions.shape != target_matrix.shape:
        raise ValueError(
            "MVNet prediction and target shapes must match: "
            f"{tuple(predictions.shape)} vs {tuple(target_matrix.shape)}"
        )
    return F.mse_loss(predictions, target_matrix, reduction="mean")


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
        batch_size=int(cfg.batch_size),
        shuffle=shuffle,
        sampler=sampler,
        num_workers=int(cfg.num_workers),
        pin_memory=True,
        collate_fn=_collate_factory(
            _resolve_base_dataset(dataset),
            include_targets=include_targets,
            assignments=assignments,
        ),
    )


def _build_pretrain_loader(dataset, cfg, *, assignments):
    sampler = build_pretrain_sampler(dataset, cfg)
    return DataLoader(
        dataset,
        batch_size=int(cfg.batch_size),
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=int(cfg.num_workers),
        pin_memory=True,
        collate_fn=_collate_factory(
            _resolve_base_dataset(dataset),
            include_targets=False,
            assignments=assignments,
        ),
    )


def _build_prediction_loader(dataset, *, batch_size: int, num_workers: int):
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=int(num_workers),
        pin_memory=True,
        collate_fn=_collate_factory(_resolve_base_dataset(dataset)),
    )


def _predict_batch(model, coords: torch.Tensor, target_names: tuple[str, ...], *, hard_topk: bool):
    try:
        preds = model(coords, hard_topk=hard_topk)
    except TypeError:
        preds = model(coords)
    if isinstance(preds, dict):
        return preds
    if len(target_names) > 1:
        return split_multitarget_prediction(preds, target_names)
    return {target_names[0]: preds}


def _predict_batch_with_aux(model, coords: torch.Tensor, target_names: tuple[str, ...], *, hard_topk: bool):
    try:
        output = model(coords, return_aux=True, hard_topk=hard_topk)
    except TypeError:
        try:
            output = model(coords, return_aux=True)
        except TypeError:
            try:
                output = model(coords, hard_topk=hard_topk)
            except TypeError:
                output = model(coords)
    if isinstance(output, tuple):
        preds = output[0]
        aux = output[1] if len(output) > 1 and output[1] is not None else {}
    else:
        preds = output
        aux = {}
    if not isinstance(preds, dict) and len(target_names) > 1:
        preds = split_multitarget_prediction(preds, target_names)
    elif not isinstance(preds, dict):
        preds = {target_names[0]: preds}
    return preds, aux


def _batch_target_dict(batch_targets, target_names: tuple[str, ...]):
    if isinstance(batch_targets, dict):
        return batch_targets
    return {target_names[0]: batch_targets}


def select_psnr_indices(
    total_size: int,
    sample_ratio: float,
    seed: int,
    max_samples: int | None = None,
) -> np.ndarray | None:
    total_size = int(total_size)
    ratio = float(sample_ratio)
    if total_size <= 0 or ratio <= 0.0 or ratio >= 1.0:
        return None
    sample_size = max(1, int(round(total_size * ratio)))
    if max_samples is not None:
        sample_size = min(sample_size, max(1, int(max_samples)))
    if sample_size >= total_size:
        return None
    rng = np.random.default_rng(int(seed))
    return np.sort(rng.choice(total_size, size=sample_size, replace=False).astype(np.int64))


def _build_psnr_dataset(dataset, *, sample_ratio: float, seed: int, max_samples: int | None = None):
    selected = select_psnr_indices(len(dataset), sample_ratio, seed, max_samples=max_samples)
    if selected is None:
        return dataset
    return Subset(dataset, selected.tolist())


def _compute_streaming_psnr(
    *,
    model,
    dataset,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    hard_topk: bool,
):
    loader = _build_prediction_loader(dataset, batch_size=batch_size, num_workers=num_workers)
    base_dataset = _resolve_base_dataset(dataset)
    target_names = base_dataset.target_names()
    accumulators = {name: PSNRAccumulator() for name in target_names}
    model.eval()
    with torch.no_grad():
        for batch in loader:
            coords = batch.coords.to(device, non_blocking=True)
            preds = _predict_batch(model, coords, target_names, hard_topk=hard_topk)
            targets = _batch_target_dict(batch.targets, target_names)
            for name in target_names:
                pred_np = preds[name].detach().cpu().numpy()
                target_np = targets[name].detach().cpu().numpy()
                accumulators[name].update(target_np, pred_np)
    per_target = {name: accumulators[name].compute() for name in target_names}
    aggregate = float(np.mean(list(per_target.values()))) if per_target else float("nan")
    return aggregate, per_target


def predict_dataset(model, dataset, *, batch_size: int, device: torch.device, hard_topk: bool = True):
    loader = _build_prediction_loader(dataset, batch_size=batch_size, num_workers=0)
    model.eval()
    if dataset.meta.is_multitarget:
        collected = {name: [] for name in dataset.target_names()}
    else:
        collected = {dataset.target_names()[0]: []}
    with torch.no_grad():
        for batch in loader:
            coords = batch.coords.to(device, non_blocking=True)
            preds = _predict_batch(model, coords, dataset.target_names(), hard_topk=hard_topk)
            for name, tensor in preds.items():
                collected[name].append(tensor.detach().cpu().numpy())
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


def _log_step_window_if_needed(
    *,
    prefix: str,
    epoch: int,
    total_epochs: int,
    steps_seen: int,
    window_start_step: int,
    window_started_at: float,
    window_data: float,
    window_transfer: float,
    window_training: float,
) -> tuple[float, int]:
    if steps_seen < window_start_step:
        return window_started_at, window_start_step
    elapsed_seconds = time.perf_counter() - window_started_at
    tracked = float(window_data) + float(window_transfer) + float(window_training)
    other_seconds = max(float(elapsed_seconds) - tracked, 0.0)
    log_step_timing_window(
        prefix=prefix,
        epoch=epoch,
        total_epochs=total_epochs,
        step_start=window_start_step,
        step_end=steps_seen,
        elapsed_seconds=elapsed_seconds,
        data_seconds=window_data,
        transfer_seconds=window_transfer,
        training_seconds=window_training,
        other_seconds=other_seconds,
    )
    return time.perf_counter(), steps_seen + 1


def _run_pretrain(model, dataset, cfg, device, log_cfg: LogConfig):
    if not cfg.pretrain.enabled or cfg.pretrain.epochs <= 0:
        return
    if dataset.meta.volume_shape is None:
        raise ValueError("Configured pretraining requires a volume dataset because only voxel_clustering is supported")
    if not hasattr(model, "pretrain_forward"):
        raise ValueError("Configured pretraining but selected model does not expose pretrain_forward")
    num_experts = getattr(model, "num_experts", None)
    if num_experts is None:
        raise ValueError("Configured pretraining requires model.num_experts")
    logger.info(
        "Pretrain start: epochs=%d batch_size=%d num_workers=%d num_experts=%d batches_per_epoch_budget=%d cache=%s",
        int(cfg.pretrain.epochs),
        int(cfg.batch_size),
        int(cfg.num_workers),
        int(num_experts),
        int(cfg.batches_per_epoch_budget),
        str(cfg.pretrain.assignments_cache_path or "<none>"),
    )
    assignments_cfg = PretrainAssignmentConfig(
        seed=int(cfg.pretrain.cluster_seed),
        cache_path=str(cfg.pretrain.assignments_cache_path or ""),
    )
    assignments_started_at = time.perf_counter()
    assignments = compute_pretrain_assignments(dataset, int(num_experts), assignments_cfg)
    logger.info(
        "Pretrain assignments prepared: shape=%s time=%.2fs",
        tuple(assignments.shape),
        time.perf_counter() - assignments_started_at,
    )
    loader_started_at = time.perf_counter()
    pretrain_loader = _build_pretrain_loader(dataset, cfg, assignments=assignments)
    logger.info(
        "Pretrain DataLoader ready: batches_per_epoch=%d dataset_samples=%d budget_batches=%d batch_size=%d time=%.2fs",
        len(pretrain_loader),
        len(dataset),
        int(cfg.batches_per_epoch_budget),
        int(cfg.batch_size),
        time.perf_counter() - loader_started_at,
    )
    optimizer = torch.optim.Adam(list(model.pretrain_parameters()), lr=float(cfg.pretrain.lr))
    timing_enabled = bool(log_cfg.timing.enabled)
    epoch_timing_enabled = timing_enabled and bool(log_cfg.timing.epoch_breakdown)
    step_window_enabled = timing_enabled and bool(log_cfg.timing.step_window)
    sync_timing = bool(log_cfg.timing.cuda_sync)
    step_window_every = max(1, int(log_cfg.timing.step_window_every_steps))

    for epoch in range(1, cfg.pretrain.epochs + 1):
        model.train()
        epoch_loss = 0.0
        total = 0
        correct = 0
        timing = TimingBreakdown()
        epoch_started_at = time.perf_counter()
        window_start_step = 1
        window_started_at = time.perf_counter()
        window_data = 0.0
        window_transfer = 0.0
        window_training = 0.0
        steps_seen = 0
        logger.info(
            "Pretrain epoch %s/%s start: batches=%d batch_size=%d",
            epoch,
            cfg.pretrain.epochs,
            len(pretrain_loader),
            int(cfg.batch_size),
        )

        iterator = iter(pretrain_loader)
        batch_fetch_started_at = time.perf_counter()
        while True:
            try:
                batch = next(iterator)
            except StopIteration:
                break

            data_seconds = 0.0
            if timing_enabled:
                data_seconds = time.perf_counter() - batch_fetch_started_at
                timing.data += data_seconds
                window_data += data_seconds

            step_started_at = time.perf_counter()
            transfer_seconds = 0.0
            if timing_enabled:
                stage_started_at = timing_start(device, sync_timing)
            coords = batch.coords.to(device, non_blocking=True)
            expert_ids = batch.expert_ids.to(device, non_blocking=True)
            if timing_enabled:
                transfer_seconds = timing_elapsed(stage_started_at, device, sync_timing)
                timing.transfer += transfer_seconds
                window_transfer += transfer_seconds
                stage_started_at = timing_start(device, sync_timing)

            logits = model.pretrain_forward(coords)
            loss = torch.nn.functional.cross_entropy(logits, expert_ids)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            training_seconds = 0.0
            if timing_enabled:
                training_seconds = timing_elapsed(stage_started_at, device, sync_timing)
                timing.training += training_seconds
                window_training += training_seconds
                step_elapsed = time.perf_counter() - step_started_at
                timing.others += max(step_elapsed - transfer_seconds - training_seconds, 0.0)

            epoch_loss += float(loss.item()) * coords.shape[0]
            total += int(coords.shape[0])
            correct += int((torch.argmax(logits, dim=-1) == expert_ids).sum().item())
            steps_seen += 1

            if step_window_enabled and steps_seen % step_window_every == 0:
                log_started_at = time.perf_counter()
                window_started_at, window_start_step = _log_step_window_if_needed(
                    prefix="Pretrain",
                    epoch=epoch,
                    total_epochs=cfg.pretrain.epochs,
                    steps_seen=steps_seen,
                    window_start_step=window_start_step,
                    window_started_at=window_started_at,
                    window_data=window_data,
                    window_transfer=window_transfer,
                    window_training=window_training,
                )
                timing.others += time.perf_counter() - log_started_at
                window_data = 0.0
                window_transfer = 0.0
                window_training = 0.0

            batch_fetch_started_at = time.perf_counter()

        if step_window_enabled and steps_seen >= window_start_step:
            log_started_at = time.perf_counter()
            _log_step_window_if_needed(
                prefix="Pretrain",
                epoch=epoch,
                total_epochs=cfg.pretrain.epochs,
                steps_seen=steps_seen,
                window_start_step=window_start_step,
                window_started_at=window_started_at,
                window_data=window_data,
                window_transfer=window_transfer,
                window_training=window_training,
            )
            timing.others += time.perf_counter() - log_started_at

        if log_cfg.epoch_summary:
            log_started_at = time.perf_counter() if timing_enabled else 0.0
            logger.info(
                "Pretrain epoch %s/%s loss=%.6e acc=%.4f",
                epoch,
                cfg.pretrain.epochs,
                epoch_loss / max(total, 1),
                correct / max(total, 1),
            )
            if timing_enabled:
                timing.others += time.perf_counter() - log_started_at

        if epoch_timing_enabled:
            epoch_total = time.perf_counter() - epoch_started_at
            timing.others += max(epoch_total - timing.tracked_total(), 0.0)
            log_epoch_timing(
                prefix="Pretrain",
                epoch=epoch,
                total_epochs=cfg.pretrain.epochs,
                total_seconds=epoch_total,
                breakdown=timing,
            )


def train_model(
    *,
    model,
    dataset,
    cfg,
    log_cfg: LogConfig,
    device: torch.device,
    checkpoint_dir: str | Path,
    config_hash: str,
    prediction_dir: str | Path,
    exp_id: str,
    predict_after_training: bool = True,
    artifact_dir: str | Path | None = None,
    model_config: dict | None = None,
    exploration_probe=None,
):
    backbone = getattr(model, "backbone", model)
    is_var_expert = type(backbone).__name__ == "VarExpert"
    is_instant_ngp = isinstance(backbone, InstantNGP)
    is_mvnet = isinstance(backbone, MVNet4D)
    if is_mvnet:
        validate_mvnet_training_config(cfg)

    timing_enabled = bool(log_cfg.timing.enabled)
    epoch_timing_enabled = timing_enabled and bool(log_cfg.timing.epoch_breakdown)
    step_window_enabled = timing_enabled and bool(log_cfg.timing.step_window)
    sync_timing = bool(log_cfg.timing.cuda_sync)
    step_window_every = max(1, int(log_cfg.timing.step_window_every_steps))

    dataloader_started_at = time.perf_counter()
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
    if log_cfg.startup_timing:
        logger.info("DataLoader build: %.2fs", time.perf_counter() - dataloader_started_at)

    accumulation_steps = int(cfg.gradient_accumulation_steps)
    budget = training_budget(
        epochs=int(cfg.epochs),
        data_steps_per_epoch=len(train_loader),
        batch_size=int(cfg.batch_size),
        gradient_accumulation_steps=accumulation_steps,
    )
    if budget["remainder"] != 0:
        raise ValueError(
            "Training data-step budget must be divisible by "
            "gradient_accumulation_steps: "
            f"data_steps={budget['data_steps']} "
            f"accumulation={accumulation_steps}"
        )
    if (
        int(cfg.save_every) > 0
        and (int(cfg.save_every) * len(train_loader)) % accumulation_steps != 0
    ):
        raise ValueError(
            "save_every must land on a completed gradient accumulation window"
        )
    if accumulation_steps > 1 and (
        cfg.gradient_balancer.enabled
        or cfg.multiview_ema_loss.enabled
        or cfg.multiview_dwa_loss.enabled
    ):
        raise ValueError(
            "gradient accumulation greater than one is not supported with "
            "multi-task loss or gradient balancers"
        )

    model = model.to(device)
    _run_pretrain(model, dataset, cfg, device, log_cfg)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(cfg.lr),
        betas=(float(cfg.beta_1), float(cfg.beta_2)),
        eps=float(cfg.epsilon),
        weight_decay=float(cfg.weight_decay),
        amsgrad=False,
    )
    scheduler = build_training_scheduler(optimizer, cfg.scheduler)

    if cfg.multiview_ema_loss.enabled and cfg.multiview_dwa_loss.enabled:
        raise ValueError("training.multiview_ema_loss and training.multiview_dwa_loss cannot both be enabled")

    ema_balancer = None
    dwa_balancer = None
    gradnorm_balancer = None
    if dataset.meta.is_multitarget and cfg.multiview_ema_loss.enabled:
        ema_balancer = MultiAttrEMALoss(
            dataset.target_names(),
            beta=float(cfg.multiview_ema_loss.beta),
            eps=float(cfg.multiview_ema_loss.eps),
            w_min=float(cfg.multiview_ema_loss.w_min),
            w_max=float(cfg.multiview_ema_loss.w_max),
            warmup_steps=int(cfg.multiview_ema_loss.warmup_steps),
            alpha=float(cfg.multiview_ema_loss.alpha),
        ).to(device)
    if dataset.meta.is_multitarget and cfg.multiview_dwa_loss.enabled:
        dwa_balancer = MultiAttrDWALoss(
            dataset.target_names(),
            temperature=float(cfg.multiview_dwa_loss.temperature),
            eps=float(cfg.multiview_dwa_loss.eps),
            total_epochs=int(cfg.epochs),
            window_size=int(cfg.multiview_dwa_loss.window_size),
            warmup_epochs=int(cfg.multiview_dwa_loss.warmup_epochs),
            eta_max=float(cfg.multiview_dwa_loss.eta_max),
            eta_min=float(cfg.multiview_dwa_loss.eta_min),
        ).to(device)
    if dataset.meta.is_multitarget and cfg.gradient_balancer.enabled and cfg.gradient_balancer.method == "gradnorm":
        gradnorm_balancer = GradNormBalancer(dataset.target_names(), cfg.gradient_balancer, device)

    best_state = None
    best_val = float("inf")
    no_improve = 0
    router_frozen = False
    from ..utils.exploration_probe import ExplorationProbeRecorder, normalize_probe, probe_progress

    probe_cfg = normalize_probe(exploration_probe)
    psnr_dataset = _build_psnr_dataset(
        dataset,
        sample_ratio=probe_cfg.sample_ratio if probe_cfg.enabled else cfg.psnr_sample_ratio,
        seed=probe_cfg.seed if probe_cfg.enabled else cfg.seed,
        max_samples=probe_cfg.max_samples if probe_cfg.enabled else None,
    )
    probe_recorder = (
        ExplorationProbeRecorder(Path(checkpoint_dir).parent / "metrics", probe_cfg)
        if probe_cfg.enabled
        else None
    )
    started_at = time.time()
    global_data_step = 0
    global_optimizer_step = 0
    accumulation_count = 0
    stopped_early = False
    completed_epoch = 0
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(1, cfg.epochs + 1):
        completed_epoch = epoch
        model.train()
        epoch_loss = 0.0
        epoch_regularization = 0.0
        epoch_optimizer_steps = 0
        steps = 0
        hard_topk = int(epoch) > int(cfg.hard_topk_warmup_epochs)
        epoch_timing = TimingBreakdown()
        epoch_started_at = time.perf_counter()
        window_start_step = 1
        window_started_at = time.perf_counter()
        window_data = 0.0
        window_transfer = 0.0
        window_training = 0.0
        expert_select_counts = None
        last_ema_state = None
        dwa_epoch_state = None
        ema_target_loss_sums = (
            {name: 0.0 for name in dataset.target_names()}
            if ema_balancer is not None
            else None
        )
        ema_target_loss_steps = 0

        if not router_frozen and cfg.freeze_router_at > 0 and (epoch / max(cfg.epochs, 1)) >= cfg.freeze_router_at:
            other_started_at = time.perf_counter() if timing_enabled else 0.0
            frozen = 0
            for name, param in model.named_parameters():
                if any(token in name.lower() for token in ("gating", "policy", "router")):
                    param.requires_grad = False
                    frozen += param.numel()
            router_frozen = frozen > 0
            if router_frozen:
                logger.info("Froze router-like parameters at epoch %s", epoch)
            if timing_enabled:
                epoch_timing.others += time.perf_counter() - other_started_at

        iterator = iter(train_loader)
        batch_fetch_started_at = time.perf_counter()
        while True:
            try:
                batch = next(iterator)
            except StopIteration:
                break

            data_seconds = 0.0
            if timing_enabled:
                data_seconds = time.perf_counter() - batch_fetch_started_at
                epoch_timing.data += data_seconds
                window_data += data_seconds

            step_started_at = time.perf_counter()
            transfer_seconds = 0.0
            if timing_enabled:
                stage_started_at = timing_start(device, sync_timing)
            coords = batch.coords.to(device, non_blocking=True)
            if _is_multitarget(batch.targets):
                targets = {name: tensor.to(device, non_blocking=True) for name, tensor in batch.targets.items()}
            else:
                targets = batch.targets.to(device, non_blocking=True)
            if timing_enabled:
                transfer_seconds = timing_elapsed(stage_started_at, device, sync_timing)
                epoch_timing.transfer += transfer_seconds
                window_transfer += transfer_seconds
                stage_started_at = timing_start(device, sync_timing)

            if _is_multitarget(batch.targets):
                if is_mvnet:
                    raw_predictions = model(coords)
                    total_loss = mvnet_joint_mse(
                        raw_predictions,
                        targets,
                        dataset.target_names(),
                    )
                    preds = split_multitarget_prediction(
                        raw_predictions,
                        dataset.target_names(),
                    )
                    task_losses = {
                        name: F.mse_loss(
                            preds[name],
                            targets[name],
                            reduction="mean",
                        )
                        for name in dataset.target_names()
                    }
                    aux = {}
                elif is_var_expert:
                    preds, aux = _predict_batch_with_aux(
                        model,
                        coords,
                        dataset.target_names(),
                        hard_topk=hard_topk,
                    )
                else:
                    preds = _predict_batch(model, coords, dataset.target_names(), hard_topk=hard_topk)
                    aux = {}
                if not is_mvnet:
                    total_loss, _, _, task_losses = reconstruction_loss_with_breakdown(
                        preds,
                        targets,
                        loss_type=cfg.loss_type,
                    )
                if ema_target_loss_sums is not None:
                    for name in dataset.target_names():
                        ema_target_loss_sums[name] += float(task_losses[name].detach().item())
                    ema_target_loss_steps += 1
                loss_to_log = total_loss
                if ema_balancer is not None:
                    total_loss, _, _, ema_details = ema_balancer(
                        task_losses,
                        return_details=True,
                        return_tensors=True,
                    )
                    last_ema_state = {
                        "step": int(ema_details["step"]),
                        "warmup_steps": int(ema_details["warmup_steps"]),
                        "effective_weights": dict(ema_details["weights"]),
                    }
                if dwa_balancer is not None:
                    total_loss = dwa_balancer(task_losses)
                if gradnorm_balancer is not None:
                    total_loss, _ = gradnorm_balancer.build_weighted_loss(task_losses)
                    total_loss.backward(retain_graph=True)
                    gradnorm_balancer.update(task_losses, model)
                elif cfg.gradient_balancer.enabled:
                    apply_multitask_gradient(model, task_losses, cfg)
                else:
                    (total_loss / float(accumulation_steps)).backward()
            else:
                if is_var_expert:
                    preds, aux = _predict_batch_with_aux(
                        model,
                        coords,
                        dataset.target_names(),
                        hard_topk=hard_topk,
                    )
                else:
                    preds = _predict_batch(model, coords, dataset.target_names(), hard_topk=hard_topk)
                    aux = {}
                target_name = dataset.target_names()[0]
                total_loss = pointwise_loss(preds[target_name], targets, cfg.loss_type)
                loss_to_log = total_loss
                (total_loss / float(accumulation_steps)).backward()

            global_data_step += 1
            accumulation_count += 1
            if accumulation_count == accumulation_steps:
                regularization = None
                if is_instant_ngp:
                    regularization = (
                        INSTANT_NGP_DECODER_L2_WEIGHT
                        * backbone.decoder_l2_regularization()
                    )
                    regularization.backward()
                    regularization_value = float(
                        regularization.detach().item()
                    )
                    epoch_regularization += regularization_value
                optimizer.step()
                global_optimizer_step += 1
                epoch_optimizer_steps += 1
                if (
                    scheduler is not None
                    and cfg.scheduler.interval == "optimizer_step"
                ):
                    scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                accumulation_count = 0

            if is_var_expert and "probs" in aux:
                probs = aux["probs"].detach()
                reduce_dims = tuple(range(probs.dim() - 1))
                counts = probs.float().sum(dim=reduce_dims).cpu()
                if expert_select_counts is None:
                    expert_select_counts = torch.zeros_like(counts)
                expert_select_counts += counts

            training_seconds = 0.0
            if timing_enabled:
                training_seconds = timing_elapsed(stage_started_at, device, sync_timing)
                epoch_timing.training += training_seconds
                window_training += training_seconds
                step_elapsed = time.perf_counter() - step_started_at
                epoch_timing.others += max(step_elapsed - data_seconds - transfer_seconds - training_seconds, 0.0)

            epoch_loss += float(loss_to_log.detach().item())
            steps += 1

            if step_window_enabled and steps % step_window_every == 0:
                log_started_at = time.perf_counter()
                window_started_at, window_start_step = _log_step_window_if_needed(
                    prefix="Train",
                    epoch=epoch,
                    total_epochs=cfg.epochs,
                    steps_seen=steps,
                    window_start_step=window_start_step,
                    window_started_at=window_started_at,
                    window_data=window_data,
                    window_transfer=window_transfer,
                    window_training=window_training,
                )
                epoch_timing.others += time.perf_counter() - log_started_at
                window_data = 0.0
                window_transfer = 0.0
                window_training = 0.0

            batch_fetch_started_at = time.perf_counter()

        if step_window_enabled and steps >= window_start_step:
            log_started_at = time.perf_counter()
            _log_step_window_if_needed(
                prefix="Train",
                epoch=epoch,
                total_epochs=cfg.epochs,
                steps_seen=steps,
                window_start_step=window_start_step,
                window_started_at=window_started_at,
                window_data=window_data,
                window_transfer=window_transfer,
                window_training=window_training,
            )
            epoch_timing.others += time.perf_counter() - log_started_at

        if dwa_balancer is not None:
            other_started_at = time.perf_counter() if timing_enabled else 0.0
            dwa_epoch_state = dwa_balancer.end_epoch(return_details=True)
            if timing_enabled:
                epoch_timing.others += time.perf_counter() - other_started_at

        other_started_at = time.perf_counter() if timing_enabled else 0.0
        if (
            scheduler is not None
            and cfg.scheduler.interval == "epoch"
        ):
            scheduler.step()
        if timing_enabled:
            epoch_timing.others += time.perf_counter() - other_started_at

        val_loss = None
        if val_loader is not None:
            val_started_at = timing_start(device, sync_timing) if timing_enabled else 0.0
            model.eval()
            loss_sum = 0.0
            count = 0
            with torch.no_grad():
                for batch in val_loader:
                    coords = batch.coords.to(device, non_blocking=True)
                    if _is_multitarget(batch.targets):
                        targets = {name: tensor.to(device, non_blocking=True) for name, tensor in batch.targets.items()}
                        if is_mvnet:
                            raw_predictions = model(coords)
                            loss = mvnet_joint_mse(
                                raw_predictions,
                                targets,
                                dataset.target_names(),
                            )
                        else:
                            preds = _predict_batch(model, coords, dataset.target_names(), hard_topk=hard_topk)
                            loss, _, _, _ = reconstruction_loss_with_breakdown(
                                preds,
                                targets,
                                loss_type=cfg.loss_type,
                            )
                    else:
                        targets = batch.targets.to(device, non_blocking=True)
                        preds = _predict_batch(model, coords, dataset.target_names(), hard_topk=hard_topk)
                        loss = pointwise_loss(preds[dataset.target_names()[0]], targets, cfg.loss_type)
                    loss_sum += float(loss.item()) * coords.shape[0]
                    count += int(coords.shape[0])
            val_loss = loss_sum / max(count, 1)
            if timing_enabled:
                epoch_timing.val += timing_elapsed(val_started_at, device, sync_timing)
            if val_loss < best_val:
                best_val = val_loss
                best_state = copy.deepcopy(model.state_dict())
                no_improve = 0
            else:
                no_improve += 1

        if log_cfg.epoch_summary and cfg.log_every > 0 and (epoch % cfg.log_every == 0 or epoch == 1):
            log_started_at = time.perf_counter() if timing_enabled else 0.0
            elapsed = time.time() - started_at
            if val_loss is None:
                logger.info(
                    "Epoch %s/%s train=%.6e reg=%.6e "
                    "data_step=%d optimizer_step=%d updates=%d lr=%.6e "
                    "time=%.1fs",
                    epoch,
                    cfg.epochs,
                    epoch_loss / max(steps, 1),
                    epoch_regularization / max(epoch_optimizer_steps, 1),
                    global_data_step,
                    global_optimizer_step,
                    epoch_optimizer_steps,
                    float(optimizer.param_groups[0]["lr"]),
                    elapsed,
                )
            else:
                logger.info(
                    "Epoch %s/%s train=%.6e reg=%.6e val=%.6e "
                    "data_step=%d optimizer_step=%d updates=%d lr=%.6e "
                    "time=%.1fs",
                    epoch,
                    cfg.epochs,
                    epoch_loss / max(steps, 1),
                    epoch_regularization / max(epoch_optimizer_steps, 1),
                    val_loss,
                    global_data_step,
                    global_optimizer_step,
                    epoch_optimizer_steps,
                    float(optimizer.param_groups[0]["lr"]),
                    elapsed,
                )
            if last_ema_state is not None:
                logger.info(
                    "EMA balance state: step=%d warmup_steps=%d effective_weights=%s",
                    int(last_ema_state.get("step", 0)),
                    int(last_ema_state.get("warmup_steps", 0)),
                    last_ema_state.get("effective_weights", {}),
                )
                if ema_target_loss_sums is not None and ema_target_loss_steps > 0:
                    ema_target_loss_text = " ".join(
                        f"{name}={ema_target_loss_sums[name] / float(ema_target_loss_steps):.6e}"
                        for name in dataset.target_names()
                    )
                    logger.info("EMA per-target loss (epoch avg): %s", ema_target_loss_text)
            if dwa_epoch_state is not None:
                logger.info(
                    "DWA balance state: completed_epochs=%d window_size=%d "
                    "warmup_complete=%s eta=%s next_epoch_weights=%s",
                    int(dwa_epoch_state.get("completed_epochs", 0)),
                    int(dwa_epoch_state.get("window_size", 0)),
                    bool(dwa_epoch_state.get("warmup_complete", False)),
                    dwa_epoch_state.get("eta"),
                    dwa_epoch_state.get("next_weights", {}),
                )
                dwa_target_loss_text = " ".join(
                    f"{name}={dwa_epoch_state['epoch_loss'][name]:.6e}"
                    for name in dataset.target_names()
                )
                logger.info("DWA per-target loss (epoch avg): %s", dwa_target_loss_text)
                if "current_window_loss" in dwa_epoch_state:
                    logger.info(
                        "DWA moving averages: previous=%s current=%s proposed_weights=%s",
                        dwa_epoch_state.get("previous_window_loss", {}),
                        dwa_epoch_state.get("current_window_loss", {}),
                        dwa_epoch_state.get("proposed_weights", {}),
                    )
            if expert_select_counts is not None:
                sum_count = float(expert_select_counts.sum().item())
                if sum_count > 0.0:
                    counts_text = " ".join(
                        f"E{i}={count:.2f} ({count / sum_count:.2%})"
                        for i, count in enumerate(expert_select_counts.tolist())
                    )
                    logger.info("Expert utilization rate: %s", counts_text)
            if timing_enabled:
                epoch_timing.others += time.perf_counter() - log_started_at

        if log_cfg.psnr.enabled and cfg.log_psnr_every > 0 and epoch % cfg.log_psnr_every == 0:
            probe_wall_started = time.perf_counter()
            psnr_started_at = timing_start(device, sync_timing) if timing_enabled else 0.0
            aggregate_psnr, per_target_psnr = _compute_streaming_psnr(
                model=model,
                dataset=psnr_dataset,
                batch_size=cfg.pred_batch_size,
                num_workers=cfg.num_workers,
                device=device,
                hard_topk=hard_topk,
            )
            elapsed = time.time() - started_at
            if log_cfg.psnr.per_target:
                per_target_text = " ".join(
                    f"{name}={value:.2f}" for name, value in sorted(per_target_psnr.items())
                )
                logger.info(
                    "PSNR epoch %s/%s: aggregate=%.2f %s time=%.1fs",
                    epoch,
                    cfg.epochs,
                    aggregate_psnr,
                    per_target_text,
                    elapsed,
                )
            else:
                logger.info(
                    "PSNR epoch %s/%s: aggregate=%.2f time=%.1fs",
                    epoch,
                    cfg.epochs,
                    aggregate_psnr,
                    elapsed,
                )
            if timing_enabled:
                epoch_timing.psnr += timing_elapsed(psnr_started_at, device, sync_timing)
            if probe_recorder is not None:
                probe_recorder.record(
                    progress=probe_progress(epoch, cfg.epochs, probe_cfg),
                    scope="aggregate",
                    aggregate_psnr=aggregate_psnr,
                    sample_count=len(psnr_dataset),
                    elapsed_seconds=time.perf_counter() - probe_wall_started,
                    details=per_target_psnr,
                )

        if cfg.save_every > 0 and epoch % cfg.save_every == 0:
            if accumulation_count != 0:
                raise RuntimeError(
                    "Checkpoint boundary has incomplete accumulated gradients"
                )
            save_started_at = time.perf_counter() if timing_enabled else 0.0
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                dataset=dataset,
                epoch=epoch,
                config_hash=config_hash,
                global_data_step=global_data_step,
                global_optimizer_step=global_optimizer_step,
                gradient_accumulation_count=accumulation_count,
                path=Path(checkpoint_dir) / f"{exp_id}_epoch{epoch}.pth",
            )
            if timing_enabled:
                epoch_timing.others += time.perf_counter() - save_started_at

        if cfg.early_stop_patience > 0 and no_improve >= cfg.early_stop_patience:
            if accumulation_count != 0:
                raise RuntimeError(
                    "Early stopping cannot discard accumulated gradients"
                )
            logger.info("Early stopping at epoch %s", epoch)
            stopped_early = True
            if epoch_timing_enabled:
                epoch_total = time.perf_counter() - epoch_started_at
                epoch_timing.others += max(epoch_total - epoch_timing.tracked_total(), 0.0)
                log_epoch_timing(
                    prefix="Train",
                    epoch=epoch,
                    total_epochs=cfg.epochs,
                    total_seconds=epoch_total,
                    breakdown=epoch_timing,
                )
            break

        if epoch_timing_enabled:
            epoch_total = time.perf_counter() - epoch_started_at
            epoch_timing.others += max(epoch_total - epoch_timing.tracked_total(), 0.0)
            log_epoch_timing(
                prefix="Train",
                epoch=epoch,
                total_epochs=cfg.epochs,
                total_seconds=epoch_total,
                breakdown=epoch_timing,
            )

    if accumulation_count != 0:
        raise RuntimeError(
            "Training ended with an incomplete gradient accumulation window"
        )
    if not stopped_early and global_data_step != budget["data_steps"]:
        raise RuntimeError(
            f"Expected {budget['data_steps']} data steps, got {global_data_step}"
        )
    if not stopped_early and global_optimizer_step != budget["optimizer_steps"]:
        raise RuntimeError(
            "Unexpected optimizer update count: "
            f"expected {budget['optimizer_steps']}, "
            f"got {global_optimizer_step}"
        )

    if best_state is not None:
        model.load_state_dict(best_state)

    final_ckpt = save_checkpoint(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        dataset=dataset,
        epoch=completed_epoch,
        config_hash=config_hash,
        global_data_step=global_data_step,
        global_optimizer_step=global_optimizer_step,
        gradient_accumulation_count=accumulation_count,
        path=Path(checkpoint_dir) / f"{exp_id}.pth",
    )
    result = {
        "checkpoint_path": final_ckpt,
        "global_data_step": global_data_step,
        "global_optimizer_step": global_optimizer_step,
        "gradient_accumulation_count": accumulation_count,
    }
    backbone = getattr(model, "backbone", model)
    if isinstance(backbone, CompactNGP):
        if artifact_dir is None or model_config is None:
            raise ValueError("CompactNGP training requires artifact_dir and model_config")
        artifact_result = save_compact_ngp_artifact(
            Path(artifact_dir) / f"{exp_id}.pt",
            model=model,
            model_config=model_config,
            dataset=dataset,
            config_hash=config_hash,
        )
        result.update(artifact_result)
        logger.info(
            "CompactNGP artifact: path=%s payload=%s bytes file=%s bytes",
            artifact_result["artifact_path"],
            artifact_result["compact_payload_bytes"],
            artifact_result["artifact_file_bytes"],
        )
    if not bool(predict_after_training):
        logger.info("Skipping automatic prediction/evaluation after training.")
        return result
    predictions = predict_dataset(
        model,
        dataset,
        batch_size=cfg.pred_batch_size,
        device=device,
        hard_topk=True,
    )
    saved_predictions = save_predictions(dataset, predictions, prediction_dir, exp_id)
    result.update({"predictions": predictions, "prediction_paths": saved_predictions})
    return result
