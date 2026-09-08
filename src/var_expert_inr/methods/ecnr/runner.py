from __future__ import annotations

import copy
import json
import logging
import os
import signal
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.lib.format import open_memmap

from ...evaluation.metrics import PSNRAccumulator, mae, mse, psnr, save_metrics
from ...evaluation.selection import parse_timestep_selection
from ...utils.io import sha256_payload
from ...utils.logging_utils import close_file_handlers, setup_logging
from ...utils.runtime import apply_runtime_thread_limits, set_random_seed
from .checkpoint_codec import FORMAT as INFERENCE_FORMAT
from .checkpoint_codec import load_inference_checkpoint, save_inference_checkpoint
from .blocks import (
    ScaleBlocks,
    attach_clustering,
    build_training_targets,
    prepare_scale_blocks,
    reconstruct_from_normalized_blocks,
    slot_valid_matrix,
)
from .cache import CacheWorkspace
from .clustering import balanced_kmeans
from .cnn import BoundaryCNN, forward_tiled, train_boundary_cnn
from .config import load_config, save_config
from .model import PackedSiren, local_coordinate_grid
from .pruning import (
    BIAS_CANDIDATES,
    WEIGHT_CANDIDATES,
    apply_cumulative_pruning,
    family_sparsity,
    initial_pruning_masks,
)
from .pyramid import PyramidScale, build_three_scale_pyramid, upsample_to_scale
from .quantization import (
    ModelQuantization,
    quantize_array,
    quantize_model,
    unquantized_parameters,
)


logger = logging.getLogger(__name__)


def _raise_sigterm(signum, frame) -> None:
    del signum, frame
    raise KeyboardInterrupt("ECNR received SIGTERM")


def _device(requested: str) -> torch.device:
    if str(requested).lower().startswith("cuda") and not torch.cuda.is_available():
        logger.warning("CUDA requested but unavailable; falling back to CPU")
        return torch.device("cpu")
    return torch.device(requested)


def _config_hash(cfg: dict[str, Any]) -> str:
    payload = copy.deepcopy(cfg)
    payload.pop("CONFIG_PATH", None)
    return sha256_payload(payload)


def _dirs(run_dir: Path, *, create: bool = True) -> dict[str, Path]:
    result = {
        "run": run_dir,
        "configs": run_dir / "configs",
        "checkpoints": run_dir / "checkpoints",
        "predictions": run_dir / "predictions",
        "metrics": run_dir / "metrics",
        "logs": run_dir / "logs",
        "cache": run_dir / "cache",
    }
    if create:
        for key, path in result.items():
            if key == "predictions":
                continue
            path.mkdir(parents=True, exist_ok=True)
    return result


def _new_run(cfg: dict[str, Any]) -> dict[str, Path]:
    root = Path(cfg["experiment_root"]) / cfg["exp_id"]
    root.mkdir(parents=True, exist_ok=True)
    while True:
        run = root / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        if not run.exists():
            return _dirs(run)
        time.sleep(0.001)


def _latest_run(cfg: dict[str, Any]) -> dict[str, Path]:
    root = Path(cfg["experiment_root"]) / cfg["exp_id"]
    candidates = sorted(path for path in root.glob("*") if path.is_dir())
    if not candidates:
        raise FileNotFoundError(f"No ECNR run found under {root}")
    return _dirs(candidates[-1])


def _run_for_path(cfg: dict[str, Any], explicit: str | Path | None) -> dict[str, Path]:
    if explicit is not None:
        resolved = Path(explicit).resolve()
        if resolved.parent.name == "checkpoints":
            return _dirs(resolved.parent.parent)
    return _latest_run(cfg)


def _load_volume(path: str | Path, shape: dict[str, int]) -> np.ndarray:
    raw = np.load(path, mmap_mode="r")
    expected = tuple(int(shape[axis]) for axis in ("T", "Z", "Y", "X"))
    expected_size = int(np.prod(expected, dtype=np.int64))
    array = raw
    if array.ndim == 5 and array.shape[-1] == 1:
        array = array[..., 0]
    elif array.ndim == 2 and array.shape[1] == 1:
        array = array[:, 0]
    if array.ndim == 1 and array.size == expected_size:
        array = array.reshape(expected)
    if tuple(array.shape) != expected:
        raise ValueError(f"ECNR target shape mismatch: expected {expected}, got {tuple(array.shape)}")
    if not np.issubdtype(array.dtype, np.floating):
        raise TypeError("ECNR target must be floating point")
    for time_index in range(expected[0]):
        frame = np.asarray(array[time_index])
        if not np.isfinite(frame).all():
            raise ValueError(f"ECNR target contains NaN/Inf at t={time_index}")
        if float(frame.min()) < -1.000001 or float(frame.max()) > 1.000001:
            raise ValueError(f"ECNR target must be pre-normalized to [-1,1], violation at t={time_index}")
    return array


def _axis_for_blocks(blocks: ScaleBlocks, max_slots: int) -> tuple[torch.Tensor, torch.Tensor]:
    coordinates = local_coordinate_grid(blocks.block_shape_xyz).repeat(max_slots, 1)
    slots = torch.arange(max_slots, dtype=torch.long).repeat_interleave(blocks.block_voxels)
    return coordinates, slots


def _full_pass_batches(
    axis_length: int,
    *,
    batch_size: int,
    rng: np.random.Generator,
):
    permutation = rng.permutation(axis_length)
    for start in range(0, axis_length, int(batch_size)):
        yield permutation[start : min(start + int(batch_size), axis_length)].astype(
            np.int64,
            copy=False,
        )


def _budgeted_batches(
    axis_length: int,
    *,
    logical_samples: int,
    batch_size: int,
    rng: np.random.Generator,
):
    """Yield an exact sample count using shuffled no-replacement cycles."""
    remaining = int(logical_samples)
    if axis_length <= 0 or remaining < 0:
        raise ValueError("axis_length must be positive and logical_samples non-negative")
    parts: list[np.ndarray] = []
    buffered = 0
    while remaining:
        permutation = rng.permutation(axis_length)
        cycle_count = min(remaining, axis_length)
        offset = 0
        while offset < cycle_count:
            take = min(int(batch_size) - buffered, cycle_count - offset)
            parts.append(permutation[offset : offset + take])
            buffered += take
            offset += take
            remaining -= take
            if buffered == int(batch_size):
                yield np.concatenate(parts).astype(np.int64, copy=False)
                parts = []
                buffered = 0
    if buffered:
        yield np.concatenate(parts).astype(np.int64, copy=False)


def _pyramid_scalar_budgets(
    pyramid: list[PyramidScale],
    scalar_predictions_per_epoch_budget: int,
) -> dict[int, int]:
    total_budget = int(scalar_predictions_per_epoch_budget)
    if total_budget <= 0:
        return {int(scale.level): 0 for scale in pyramid}
    populations = {
        int(scale.level): int(np.prod(scale.values.shape, dtype=np.int64))
        for scale in pyramid
    }
    total_population = sum(populations.values())
    budgets = {
        level: total_budget * population // total_population
        for level, population in populations.items()
    }
    budgets[0] += total_budget - sum(budgets.values())
    return budgets


def _masked_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    return torch.sum(mask.to(predictions.dtype) * (predictions - targets) ** 2) / float(predictions.numel())


def _mlp_losses(
    model: PackedSiren,
    targets: np.ndarray,
    coordinates: torch.Tensor,
    slots: torch.Tensor,
    *,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    sums = torch.zeros(model.mlp_count, dtype=torch.float64)
    counts = model.slot_valid.sum(dim=1).cpu().to(torch.float64) * targets.shape[2]
    flat_targets = targets.reshape(model.mlp_count, -1)
    model.eval()
    with torch.no_grad():
        for start in range(0, coordinates.shape[0], int(batch_size)):
            stop = min(start + int(batch_size), coordinates.shape[0])
            batch_slots = slots[start:stop]
            prediction = model(coordinates[start:stop].to(device), batch_slots.to(device))
            target = torch.from_numpy(np.asarray(flat_targets[:, start:stop], dtype=np.float32)).to(device)
            mask = model.expanded_slot_mask(batch_slots).to(device)
            sums += torch.sum(mask * (prediction - target) ** 2, dim=1).detach().cpu().to(torch.float64)
    return (sums / torch.clamp(counts, min=1.0)).numpy()


def _train_scale(
    model: PackedSiren,
    targets: np.ndarray,
    blocks: ScaleBlocks,
    cfg: dict[str, Any],
    *,
    level: int,
    scalar_predictions_per_epoch_budget: int,
    device: torch.device,
    cost: dict[str, Any],
) -> tuple[dict[str, torch.Tensor], ModelQuantization]:
    training = cfg["training"]
    model.to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(training["lr"]),
        betas=(float(training["beta_1"]), float(training["beta_2"])),
        weight_decay=float(training["weight_decay"]),
    )
    masks = initial_pruning_masks(model)
    max_slots = int(targets.shape[1])
    coordinates, slots = _axis_for_blocks(blocks, max_slots)
    flat_targets = targets.reshape(model.mlp_count, -1)
    rng = np.random.default_rng(int(training["seed"]) + int(level) * 1000)
    pruning_schedule = dict(zip(training["pruning_epochs"], training["pruning_sparsities"]))
    scale_started = time.perf_counter()
    primary_started = scale_started
    logical_samples = actual_predictions = optimizer_steps = 0
    batch_size = int(training["batch_size"])
    passes_per_epoch = int(training["passes_per_epoch"])
    axis_length = int(coordinates.shape[0])
    batches_per_pass = (axis_length + batch_size - 1) // batch_size
    sampling_mode = str(training["sampling_mode"])
    if sampling_mode == "budgeted_random":
        logical_samples_per_epoch = int(scalar_predictions_per_epoch_budget) // int(model.mlp_count)
        if logical_samples_per_epoch <= 0:
            raise ValueError(
                f"ECNR scale={level} scalar budget {scalar_predictions_per_epoch_budget} "
                f"is smaller than mlp_count {model.mlp_count}"
            )
        batches_per_epoch = (logical_samples_per_epoch + batch_size - 1) // batch_size
    else:
        logical_samples_per_epoch = axis_length * passes_per_epoch
        batches_per_epoch = passes_per_epoch * batches_per_pass
    epochs_per_scale = int(training["epochs_per_scale"])
    planned_optimizer_steps = epochs_per_scale * batches_per_epoch
    planned_logical_samples = epochs_per_scale * logical_samples_per_epoch
    planned_actual_predictions = planned_logical_samples * int(model.mlp_count)
    progress_log_seconds = int(training["progress_log_seconds"])
    last_progress_log = time.perf_counter()
    logger.info(
        "ECNR scale=%d start sampling_mode=%s effective_blocks=%d mlps=%d max_slots=%d "
        "axis_length=%d logical_samples_per_epoch=%d scalar_budget_per_epoch=%d "
        "batches_per_epoch=%d epochs=%d planned_scalar_predictions=%d "
        "planned_optimizer_steps=%d",
        level,
        sampling_mode,
        blocks.effective_count,
        model.mlp_count,
        max_slots,
        axis_length,
        logical_samples_per_epoch,
        int(scalar_predictions_per_epoch_budget),
        batches_per_epoch,
        epochs_per_scale,
        planned_actual_predictions,
        planned_optimizer_steps,
    )

    for epoch in range(1, epochs_per_scale + 1):
        model.train()
        epoch_loss = 0.0
        epoch_batches = 0
        pruning_loss_sums = (
            torch.zeros(model.mlp_count, dtype=torch.float64)
            if sampling_mode == "budgeted_random" and epoch in pruning_schedule
            else None
        )
        pruning_loss_counts = (
            torch.zeros(model.mlp_count, dtype=torch.float64)
            if pruning_loss_sums is not None
            else None
        )
        if sampling_mode == "budgeted_random":
            epoch_iterator = _budgeted_batches(
                axis_length,
                logical_samples=logical_samples_per_epoch,
                batch_size=batch_size,
                rng=rng,
            )
        else:
            epoch_iterator = (
                indices
                for _ in range(passes_per_epoch)
                for indices in _full_pass_batches(
                    axis_length,
                    batch_size=batch_size,
                    rng=rng,
                )
            )
        for batch_index, indices in enumerate(epoch_iterator, start=1):
            coord_batch = coordinates[indices].to(device)
            slot_batch = slots[indices].to(device)
            target_batch = torch.from_numpy(
                np.asarray(flat_targets[:, indices], dtype=np.float32)
            ).to(device)
            mask_batch = model.expanded_slot_mask(slot_batch).to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(coord_batch, slot_batch)
            if pruning_loss_sums is not None:
                squared = mask_batch.to(prediction.dtype) * (prediction - target_batch) ** 2
                pruning_loss_sums += torch.sum(squared, dim=1).detach().cpu().to(torch.float64)
                pruning_loss_counts += torch.sum(mask_batch, dim=1).detach().cpu().to(torch.float64)
            loss = _masked_loss(prediction, target_batch, mask_batch)
            loss.backward()
            model.mask_pruned_gradients(masks)
            optimizer.step()
            model.apply_pruning_masks(masks)
            epoch_loss += float(loss.detach())
            epoch_batches += 1
            logical_samples += int(indices.size)
            actual_predictions += int(indices.size) * model.mlp_count
            optimizer_steps += 1
            now = time.perf_counter()
            if progress_log_seconds and now - last_progress_log >= progress_log_seconds:
                elapsed = now - scale_started
                eta = elapsed * (planned_optimizer_steps - optimizer_steps) / max(
                    optimizer_steps, 1
                )
                logger.info(
                    "ECNR scale=%d progress epoch=%d/%d batch=%d/%d "
                    "optimizer_steps=%d/%d scalar_predictions=%d/%d "
                    "elapsed_seconds=%.1f eta_seconds=%.1f",
                    level,
                    epoch,
                    epochs_per_scale,
                    batch_index,
                    batches_per_epoch,
                    optimizer_steps,
                    planned_optimizer_steps,
                    actual_predictions,
                    planned_actual_predictions,
                    elapsed,
                    eta,
                )
                last_progress_log = now
        if epoch in pruning_schedule:
            if pruning_loss_sums is None:
                losses = _mlp_losses(
                    model,
                    targets,
                    coordinates,
                    slots,
                    batch_size=int(training["batch_size"]),
                    device=device,
                )
            else:
                losses = (
                    pruning_loss_sums / torch.clamp(pruning_loss_counts, min=1.0)
                ).numpy()
            pruned = apply_cumulative_pruning(
                model,
                masks,
                mlp_losses=losses,
                target_sparsity=float(pruning_schedule[epoch]),
                loss_weight=float(training["pruning_loss_weight"]),
            )
            for group in optimizer.param_groups:
                group["lr"] *= float(training["pruning_lr_gamma"])
            logger.info(
                "ECNR scale=%d epoch=%d pruning=%s weight_sparsity=%.4f bias_sparsity=%.4f",
                level,
                epoch,
                pruned,
                family_sparsity(masks, WEIGHT_CANDIDATES),
                family_sparsity(masks, BIAS_CANDIDATES),
            )
        if int(training["log_every"]) and epoch % int(training["log_every"]) == 0:
            logger.info(
                "ECNR scale=%d epoch=%d/%d loss=%.7g lr=%.7g",
                level,
                epoch,
                epochs_per_scale,
                epoch_loss / max(epoch_batches, 1),
                optimizer.param_groups[0]["lr"],
            )

    primary_seconds = float(time.perf_counter() - primary_started)
    quantization_started = time.perf_counter()
    quantization = quantize_model(
        model,
        masks,
        bits=int(cfg["quantization"]["mlp_weight_bits"]),
        seed=int(training["seed"]) + int(level) * 10_000,
    )
    finetune_epochs = int(training["quantization_finetune_epochs"])
    finetune_passes = int(training["quantization_finetune_passes_per_epoch"])
    if sampling_mode == "budgeted_random":
        finetune_logical_samples_per_epoch = logical_samples_per_epoch
        finetune_batches_per_epoch = batches_per_epoch
    else:
        finetune_logical_samples_per_epoch = axis_length * finetune_passes
        finetune_batches_per_epoch = finetune_passes * batches_per_pass
    finetune_logical_samples = 0
    finetune_actual_predictions = 0
    finetune_optimizer_steps = 0
    if finetune_epochs:
        logger.info(
            "ECNR scale=%d QAT start sampling_mode=%s epochs=%d "
            "logical_samples_per_epoch=%d batches_per_epoch=%d "
            "planned_scalar_predictions=%d planned_optimizer_steps=%d",
            level,
            sampling_mode,
            finetune_epochs,
            finetune_logical_samples_per_epoch,
            finetune_batches_per_epoch,
            finetune_epochs * finetune_logical_samples_per_epoch * int(model.mlp_count),
            finetune_epochs * finetune_batches_per_epoch,
        )
        finetune_optimizer = torch.optim.Adam(
            [*quantization.codebook_parameters(), *unquantized_parameters(model)],
            lr=float(training["quantization_finetune_lr"]),
            betas=(float(training["beta_1"]), float(training["beta_2"])),
            weight_decay=0.0,
        )
        for finetune_epoch in range(1, finetune_epochs + 1):
            finetune_epoch_loss = 0.0
            finetune_epoch_batches = 0
            if sampling_mode == "budgeted_random":
                finetune_iterator = _budgeted_batches(
                    axis_length,
                    logical_samples=logical_samples_per_epoch,
                    batch_size=batch_size,
                    rng=rng,
                )
            else:
                finetune_iterator = (
                    indices
                    for _ in range(finetune_passes)
                    for indices in _full_pass_batches(
                        axis_length,
                        batch_size=batch_size,
                        rng=rng,
                    )
                )
            for batch_index, indices in enumerate(finetune_iterator, start=1):
                quantization.materialize(model)
                finetune_optimizer.zero_grad(set_to_none=True)
                coord_batch = coordinates[indices].to(device)
                slot_batch = slots[indices].to(device)
                target_batch = torch.from_numpy(
                    np.asarray(flat_targets[:, indices], dtype=np.float32)
                ).to(device)
                mask_batch = model.expanded_slot_mask(slot_batch).to(device)
                loss = _masked_loss(model(coord_batch, slot_batch), target_batch, mask_batch)
                loss.backward()
                quantization.collect_codebook_gradients(model)
                finetune_optimizer.step()
                quantization.materialize(model)
                finetune_epoch_loss += float(loss.detach())
                finetune_epoch_batches += 1
                finetune_logical_samples += int(indices.size)
                finetune_actual_predictions += int(indices.size) * model.mlp_count
                finetune_optimizer_steps += 1
                now = time.perf_counter()
                if progress_log_seconds and now - last_progress_log >= progress_log_seconds:
                    qat_planned_steps = finetune_epochs * finetune_batches_per_epoch
                    qat_elapsed = now - quantization_started
                    qat_eta = qat_elapsed * (
                        qat_planned_steps - finetune_optimizer_steps
                    ) / max(finetune_optimizer_steps, 1)
                    logger.info(
                        "ECNR scale=%d QAT progress epoch=%d/%d batch=%d/%d "
                        "optimizer_steps=%d/%d scalar_predictions=%d/%d "
                        "elapsed_seconds=%.1f eta_seconds=%.1f",
                        level,
                        finetune_epoch,
                        finetune_epochs,
                        batch_index,
                        finetune_batches_per_epoch,
                        finetune_optimizer_steps,
                        qat_planned_steps,
                        finetune_actual_predictions,
                        finetune_epochs
                        * finetune_logical_samples_per_epoch
                        * int(model.mlp_count),
                        qat_elapsed,
                        qat_eta,
                    )
                    last_progress_log = now
            if int(training["log_every"]) and finetune_epoch % int(training["log_every"]) == 0:
                logger.info(
                    "ECNR scale=%d QAT epoch=%d/%d loss=%.7g",
                    level,
                    finetune_epoch,
                    finetune_epochs,
                    finetune_epoch_loss / max(finetune_epoch_batches, 1),
                )
        cost["quantization_finetune_logical_samples"] += finetune_logical_samples
        cost["quantization_finetune_actual_predictions"] += finetune_actual_predictions
        cost["quantization_finetune_optimizer_steps"] += finetune_optimizer_steps
    quantization.materialize(model)
    quantization_seconds = float(time.perf_counter() - quantization_started)
    cost["quantization_and_finetune_seconds"] += quantization_seconds
    cost["scales"].append(
        {
            "level": int(level),
            "effective_blocks": int(blocks.effective_count),
            "mlp_count": int(model.mlp_count),
            "max_slots": max_slots,
            "axis_length": axis_length,
            "batches_per_pass": batches_per_pass,
            "batches_per_epoch": batches_per_epoch,
            "passes_per_epoch": passes_per_epoch,
            "sampling_mode": sampling_mode,
            "scalar_predictions_per_epoch_budget": int(scalar_predictions_per_epoch_budget),
            "epochs": epochs_per_scale,
            "planned_logical_samples": planned_logical_samples,
            "planned_optimizer_steps": planned_optimizer_steps,
            "planned_actual_scalar_predictions": planned_actual_predictions,
            "logical_samples": int(logical_samples),
            "actual_scalar_predictions": int(actual_predictions),
            "optimizer_steps": int(optimizer_steps),
            "quantization_finetune_passes_per_epoch": finetune_passes,
            "quantization_finetune_epochs": finetune_epochs,
            "quantization_finetune_planned_logical_samples": (
                finetune_epochs * finetune_logical_samples_per_epoch
            ),
            "quantization_finetune_planned_optimizer_steps": (
                finetune_epochs * finetune_batches_per_epoch
            ),
            "quantization_finetune_planned_actual_predictions": (
                finetune_epochs
                * finetune_logical_samples_per_epoch
                * int(model.mlp_count)
            ),
            "quantization_finetune_logical_samples": finetune_logical_samples,
            "quantization_finetune_actual_predictions": finetune_actual_predictions,
            "quantization_finetune_optimizer_steps": finetune_optimizer_steps,
            "seconds": float(time.perf_counter() - scale_started),
            "primary_training_seconds": primary_seconds,
            "quantization_and_finetune_seconds": quantization_seconds,
            "weight_sparsity": family_sparsity(masks, WEIGHT_CANDIDATES),
            "bias_sparsity": family_sparsity(masks, BIAS_CANDIDATES),
        }
    )
    return masks, quantization


def _decode_scale_model(
    model: PackedSiren,
    blocks: ScaleBlocks,
    *,
    batch_size: int,
    device: torch.device,
    output_path: str | Path | None = None,
    workspace: CacheWorkspace | None = None,
) -> np.ndarray:
    max_slots = int(model.max_slots)
    coordinates, slots = _axis_for_blocks(blocks, max_slots)
    slot_shape = (model.mlp_count, max_slots, blocks.block_voxels)
    if output_path is None:
        decoded_slots = np.empty(slot_shape, dtype=np.float32)
    else:
        normalized_path = Path(output_path)
        normalized_path.parent.mkdir(parents=True, exist_ok=True)
        slots_path = normalized_path.with_name(f"{normalized_path.stem}_slots.npy")
        decoded_slots = open_memmap(slots_path, mode="w+", dtype=np.float32, shape=slot_shape)
        if workspace is not None:
            workspace.register(slots_path, decoded_slots)
    flat = decoded_slots.reshape(model.mlp_count, -1)
    model.eval()
    with torch.no_grad():
        for start in range(0, coordinates.shape[0], int(batch_size)):
            stop = min(start + int(batch_size), coordinates.shape[0])
            flat[:, start:stop] = (
                model(coordinates[start:stop].to(device), slots[start:stop].to(device))
                .detach()
                .cpu()
                .numpy()
            )
    decoded_shape = (blocks.effective_count, blocks.block_voxels)
    if output_path is None:
        decoded = np.empty(decoded_shape, dtype=np.float32)
    else:
        decoded = open_memmap(output_path, mode="w+", dtype=np.float32, shape=decoded_shape)
        if workspace is not None:
            workspace.register(output_path, decoded)
    for block_index in range(blocks.effective_count):
        decoded[block_index] = decoded_slots[
            blocks.block_to_mlp[block_index],
            blocks.block_to_slot[block_index],
        ]
    if hasattr(decoded_slots, "flush"):
        decoded_slots.flush()
    if hasattr(decoded, "flush"):
        decoded.flush()
    if output_path is not None and workspace is not None:
        workspace.release(slots_path, arrays=(decoded_slots,), label="decoded-slots")
    return decoded


def _serialize_blocks(blocks: ScaleBlocks) -> dict[str, Any]:
    return {
        "original_shape_tzyx": list(blocks.original_shape_tzyx),
        "padded_shape_zyx": list(blocks.padded_shape_zyx),
        "padding_zyx": [list(pair) for pair in blocks.padding_zyx],
        "block_shape_xyz": list(blocks.block_shape_xyz),
        "spatial_grid_zyx": list(blocks.spatial_grid_zyx),
        "effective_mask": blocks.effective_mask,
        "effective_positions": blocks.effective_positions,
        "block_min": blocks.block_min,
        "block_max": blocks.block_max,
        "block_to_mlp": blocks.block_to_mlp,
        "block_to_slot": blocks.block_to_slot,
        "cluster_sizes": blocks.cluster_sizes,
    }


def _deserialize_blocks(payload: dict[str, Any]) -> ScaleBlocks:
    return ScaleBlocks(
        original_shape_tzyx=tuple(payload["original_shape_tzyx"]),
        padded_shape_zyx=tuple(payload["padded_shape_zyx"]),
        padding_zyx=tuple(tuple(pair) for pair in payload["padding_zyx"]),
        block_shape_xyz=tuple(payload["block_shape_xyz"]),
        spatial_grid_zyx=tuple(payload["spatial_grid_zyx"]),
        effective_mask=np.asarray(payload["effective_mask"], dtype=bool),
        effective_positions=np.asarray(payload["effective_positions"], dtype=np.int64),
        block_min=np.asarray(payload["block_min"], dtype=np.float32),
        block_max=np.asarray(payload["block_max"], dtype=np.float32),
        normalized_blocks=np.empty((0, 0), dtype=np.float32),
        block_to_mlp=np.asarray(payload["block_to_mlp"], dtype=np.int64),
        block_to_slot=np.asarray(payload["block_to_slot"], dtype=np.int64),
        cluster_sizes=np.asarray(payload["cluster_sizes"], dtype=np.int64),
    )


def _select_scale_blocks(
    blocks: ScaleBlocks,
    scale_time_indices: np.ndarray,
    selected_time_indices: np.ndarray,
) -> ScaleBlocks:
    """Return the block metadata needed to reconstruct selected scale frames."""

    scale_times = np.asarray(scale_time_indices, dtype=np.int64)
    selected_times = np.asarray(selected_time_indices, dtype=np.int64)
    positions_by_time = {int(value): index for index, value in enumerate(scale_times.tolist())}
    missing = [int(value) for value in selected_times if int(value) not in positions_by_time]
    if missing:
        raise IndexError(f"ECNR scale does not contain requested time indices: {missing}")
    selected_positions = [positions_by_time[int(value)] for value in selected_times]
    output_position = {
        int(source_position): output_index
        for output_index, source_position in enumerate(selected_positions)
    }
    block_rows = np.flatnonzero(
        np.isin(blocks.effective_positions[:, 0], np.asarray(selected_positions, dtype=np.int64))
    )
    effective_positions = np.asarray(blocks.effective_positions[block_rows], dtype=np.int64).copy()
    for row in range(effective_positions.shape[0]):
        effective_positions[row, 0] = output_position[int(effective_positions[row, 0])]

    def selected(values: np.ndarray | None) -> np.ndarray | None:
        return None if values is None else np.asarray(values)[block_rows]

    return ScaleBlocks(
        original_shape_tzyx=(len(selected_positions), *blocks.original_shape_tzyx[1:]),
        padded_shape_zyx=blocks.padded_shape_zyx,
        padding_zyx=blocks.padding_zyx,
        block_shape_xyz=blocks.block_shape_xyz,
        spatial_grid_zyx=blocks.spatial_grid_zyx,
        effective_mask=np.asarray(blocks.effective_mask[selected_positions], dtype=bool),
        effective_positions=effective_positions,
        block_min=np.asarray(blocks.block_min[block_rows], dtype=np.float32),
        block_max=np.asarray(blocks.block_max[block_rows], dtype=np.float32),
        normalized_blocks=np.empty((0, 0), dtype=np.float32),
        block_to_mlp=selected(blocks.block_to_mlp),
        block_to_slot=selected(blocks.block_to_slot),
        cluster_sizes=blocks.cluster_sizes,
    )


def _serialize_scale(
    *,
    level: int,
    time_indices: np.ndarray,
    blocks: ScaleBlocks,
    model: PackedSiren | None,
    quantization: ModelQuantization | None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "level": int(level),
        "time_indices": np.asarray(time_indices, dtype=np.int64),
        "blocks": _serialize_blocks(blocks),
        "empty": model is None,
    }
    if model is None:
        return result
    named = dict(model.named_parameters())
    quantized_state = quantization.state_dict()
    for item in quantized_state["parameters"].values():
        mask = np.asarray(item["mask"], dtype=bool)
        labels = np.asarray(item["labels"], dtype=np.int64)
        # Masked entries are ignored during decode; zero is the canonical
        # placeholder that lets all 8-bit labels use an actual uint8 stream.
        item["labels"] = np.where(mask, labels, 0).astype(np.uint8)
    result["model"] = {
        "mlp_count": int(model.mlp_count),
        "max_slots": int(model.max_slots),
        "slot_valid": model.slot_valid.detach().cpu().numpy(),
        "latent": model.latent.detach().cpu().numpy().astype(np.float32),
        "unquantized": {
            name: named[name].detach().cpu().numpy().astype(np.float32)
            for name in ("layers.3.weight", "layers.0.bias", "layers.3.bias")
        },
        "quantization": quantized_state,
    }
    return result


def _model_from_scale(payload: dict[str, Any], device: torch.device) -> PackedSiren | None:
    if payload["empty"]:
        return None
    state = payload["model"]
    model = PackedSiren(
        mlp_count=int(state["mlp_count"]),
        max_slots=int(state["max_slots"]),
        slot_valid=torch.from_numpy(np.asarray(state["slot_valid"], dtype=bool)),
    ).to(device)
    named = dict(model.named_parameters())
    with torch.no_grad():
        model.latent.copy_(torch.from_numpy(np.asarray(state["latent"], dtype=np.float32)).to(device))
        for name, values in state["unquantized"].items():
            named[name].copy_(torch.from_numpy(np.asarray(values, dtype=np.float32)).to(device))
        for name, item in state["quantization"]["parameters"].items():
            labels = np.asarray(item["labels"], dtype=np.int64)
            mask = np.asarray(item["mask"], dtype=bool)
            codebook = torch.from_numpy(np.asarray(item["codebook"], dtype=np.float32)).to(device)
            restored = torch.zeros(labels.shape, dtype=torch.float32, device=device)
            label_tensor = torch.from_numpy(labels).to(device)
            mask_tensor = torch.from_numpy(mask).to(device)
            restored[mask_tensor] = codebook[label_tensor[mask_tensor]]
            named[name].copy_(restored)
    return model


def _decode_scale_payload(
    payload: dict[str, Any],
    *,
    device: torch.device,
    batch_size: int,
    output_path: str | Path | None = None,
    time_indices: np.ndarray | None = None,
    workspace: CacheWorkspace | None = None,
) -> np.ndarray:
    blocks = _deserialize_blocks(payload["blocks"])
    if time_indices is not None:
        blocks = _select_scale_blocks(
            blocks,
            np.asarray(payload["time_indices"], dtype=np.int64),
            np.asarray(time_indices, dtype=np.int64),
        )
    if payload["empty"]:
        if output_path is None:
            return np.zeros(blocks.original_shape_tzyx, dtype=np.float32)
        output = open_memmap(output_path, mode="w+", dtype=np.float32, shape=blocks.original_shape_tzyx)
        output[:] = 0.0
        output.flush()
        return output
    model = _model_from_scale(payload, device)
    decoded_path = None
    if output_path is not None:
        path = Path(output_path)
        decoded_path = path.with_name(f"{path.stem}_normalized.npy")
    decoded = _decode_scale_model(
        model,
        blocks,
        batch_size=batch_size,
        device=device,
        output_path=decoded_path,
        workspace=workspace,
    )
    reconstruction = reconstruct_from_normalized_blocks(
        blocks,
        decoded,
        output_path=output_path,
    )
    if decoded_path is not None and workspace is not None:
        workspace.release(
            decoded_path,
            arrays=(decoded,),
            label=f"predict-scale-{int(payload['level'])}-decoded-blocks",
        )
    return reconstruction


def _framewise_binary(
    left: np.ndarray,
    right: np.ndarray,
    output_path: str | Path,
    *,
    operation: str,
) -> np.ndarray:
    if tuple(left.shape) != tuple(right.shape):
        raise ValueError(f"Framewise {operation} shape mismatch: {left.shape} != {right.shape}")
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    output = open_memmap(path, mode="w+", dtype=np.float32, shape=tuple(left.shape))
    for time_index in range(left.shape[0]):
        left_frame = np.asarray(left[time_index], dtype=np.float32)
        right_frame = np.asarray(right[time_index], dtype=np.float32)
        if operation == "add":
            output[time_index] = left_frame + right_frame
        elif operation == "subtract":
            output[time_index] = left_frame - right_frame
        else:
            raise ValueError(f"Unknown framewise operation: {operation}")
    output.flush()
    return output


def _framewise_add_in_place(destination: np.ndarray, residual: np.ndarray) -> np.ndarray:
    if tuple(destination.shape) != tuple(residual.shape):
        raise ValueError(f"Framewise add shape mismatch: {destination.shape} != {residual.shape}")
    if not destination.flags.writeable:
        raise ValueError("Framewise add destination must be writable")
    for time_index in range(destination.shape[0]):
        destination[time_index] = (
            np.asarray(destination[time_index], dtype=np.float32)
            + np.asarray(residual[time_index], dtype=np.float32)
        )
    if hasattr(destination, "flush"):
        destination.flush()
    return destination


def _clip_in_place(
    values: np.ndarray,
    *,
    lower: float = -1.0,
    upper: float = 1.0,
) -> np.ndarray:
    if not values.flags.writeable:
        raise ValueError("Clip destination must be writable")
    for time_index in range(values.shape[0]):
        np.clip(
            np.asarray(values[time_index], dtype=np.float32),
            float(lower),
            float(upper),
            out=values[time_index],
        )
    if hasattr(values, "flush"):
        values.flush()
    return values


def _quantize_cnn(model: BoundaryCNN, *, bits: int, seed: int) -> dict[str, Any]:
    state: dict[str, Any] = {}
    with torch.no_grad():
        for index, layer in enumerate(model.layers):
            for offset, name in enumerate(("weight", "bias")):
                values = getattr(layer, name).detach().cpu().numpy().astype(np.float32)
                centers, labels = quantize_array(values, bits=bits, seed=seed + index * 2 + offset)
                restored = centers[labels]
                getattr(layer, name).copy_(torch.from_numpy(restored).to(getattr(layer, name).device))
                state[f"layers.{index}.{name}"] = {
                    "labels": labels.astype(np.uint16),
                    "codebook": centers.astype(np.float32),
                }
    return {"bits": int(bits), "parameters": state}


def _cnn_from_payload(payload: dict[str, Any], device: torch.device) -> BoundaryCNN:
    model = BoundaryCNN(hidden_channels=32).to(device)
    named = dict(model.named_parameters())
    with torch.no_grad():
        for name, item in payload["parameters"].items():
            labels = np.asarray(item["labels"], dtype=np.int64)
            codebook = np.asarray(item["codebook"], dtype=np.float32)
            named[name].copy_(torch.from_numpy(codebook[labels]).to(device))
    return model


def decode_checkpoint_payload(
    payload: dict[str, Any],
    *,
    device: torch.device,
    batch_size: int,
    work_dir: str | Path | None = None,
    output_path: str | Path | None = None,
    time_indices: tuple[int, ...] | list[int] | np.ndarray | None = None,
    workspace: CacheWorkspace | None = None,
) -> np.ndarray:
    if payload.get("format") != INFERENCE_FORMAT:
        raise ValueError("Invalid ECNR inference checkpoint payload")
    work = None if work_dir is None else Path(work_dir)
    if work is not None:
        work.mkdir(parents=True, exist_ok=True)
    scales = sorted(payload["scales"], key=lambda item: int(item["level"]))
    finest_times = np.asarray(scales[0]["time_indices"], dtype=np.int64)
    if time_indices is None:
        requested_times = finest_times
    else:
        requested_times = np.asarray(time_indices, dtype=np.int64)
        available = set(int(value) for value in finest_times.tolist())
        missing = [int(value) for value in requested_times if int(value) not in available]
        if missing:
            raise IndexError(f"ECNR checkpoint does not contain time indices: {missing}")

    required_by_level: dict[int, np.ndarray] = {
        int(scales[0]["level"]): requested_times
    }
    for fine_payload, coarse_payload in zip(scales, scales[1:]):
        fine_required = required_by_level[int(fine_payload["level"])]
        coarse_times = np.asarray(coarse_payload["time_indices"], dtype=np.int64)
        support: set[int] = set()
        for time_value in fine_required.tolist():
            right = int(np.searchsorted(coarse_times, int(time_value), side="left"))
            if right == 0:
                support.add(int(coarse_times[0]))
            elif right >= len(coarse_times):
                support.add(int(coarse_times[-1]))
            elif int(coarse_times[right]) == int(time_value):
                support.add(int(coarse_times[right]))
            else:
                support.add(int(coarse_times[right - 1]))
                support.add(int(coarse_times[right]))
        required_by_level[int(coarse_payload["level"])] = np.asarray(
            sorted(support), dtype=np.int64
        )

    composite = None
    composite_paths: tuple[Path, ...] = ()
    previous_times = None
    for scale_payload in reversed(scales):
        level = int(scale_payload["level"])
        current_times = required_by_level[level]
        residual_path = None if work is None else work / f"decode_residual_scale_{level}.npy"
        residual = _decode_scale_payload(
            scale_payload,
            device=device,
            batch_size=batch_size,
            output_path=residual_path,
            time_indices=current_times,
            workspace=workspace,
        )
        residual_paths: tuple[Path, ...] = ()
        if residual_path is not None:
            cropped_residual_path = residual_path.with_name(
                f"{residual_path.stem}_cropped.npy"
            )
            residual_paths = (
                (residual_path, cropped_residual_path)
                if cropped_residual_path.exists()
                else (residual_path,)
            )
            if workspace is not None:
                for path in residual_paths:
                    workspace.register(
                        path,
                        residual if path == residual_paths[-1] else None,
                    )
        if composite is None:
            composite = residual
            composite_paths = residual_paths
        else:
            upsampled_path = (
                None if work is None else work / f"decode_upsampled_scale_{level}.npy"
            )
            upsampled = upsample_to_scale(
                composite,
                previous_times,
                fine_shape_tzyx=tuple(residual.shape),
                fine_time_indices=current_times,
                output_path=upsampled_path,
            )
            if upsampled_path is not None and workspace is not None:
                workspace.register(upsampled_path, upsampled)
                workspace.release(
                    *composite_paths,
                    arrays=(composite,),
                    label=f"predict-scale-{level}-coarse-reconstruction",
                )
            composite = _framewise_add_in_place(upsampled, residual)
            composite_paths = () if upsampled_path is None else (upsampled_path,)
            if residual_paths and workspace is not None:
                workspace.release(
                    *residual_paths,
                    arrays=(residual,),
                    label=f"predict-scale-{level}-decoded-residual",
                )
        previous_times = current_times
    composite = _clip_in_place(composite)
    cnn = _cnn_from_payload(payload["cnn"], device)
    if output_path is None:
        output = np.empty_like(composite)
    else:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output = open_memmap(output_file, mode="w+", dtype=np.float32, shape=tuple(composite.shape))
    core_shape = tuple(int(value) for value in payload["cnn_config"]["tile_core_shape_zyx"])
    halo = int(payload["cnn_config"]["halo"])
    cnn.eval()
    with torch.no_grad():
        for time_index in range(composite.shape[0]):
            output[time_index] = forward_tiled(
                cnn,
                composite[time_index],
                core_shape_zyx=core_shape,
                halo=halo,
                device=device,
            ).numpy()
    if hasattr(output, "flush"):
        output.flush()
    if composite_paths and workspace is not None:
        workspace.release(
            *composite_paths,
            arrays=(composite,),
            label="predict-cnn-input-reconstruction",
        )
    return output


def run_train(
    config_path: str | Path,
    *,
    target: str | None = None,
) -> dict[str, Any]:
    apply_runtime_thread_limits()
    cfg = load_config(config_path, target_override=target)
    config_hash = _config_hash(cfg)
    dirs = _new_run(cfg)
    setup_logging(log_dir=dirs["logs"], log_file="run.log")
    workspace = CacheWorkspace(dirs["cache"])
    cache_cleaned = False
    previous_sigterm = None
    try:
        previous_sigterm = signal.signal(signal.SIGTERM, _raise_sigterm)
    except ValueError:
        pass
    started = time.perf_counter()
    try:
        save_config(cfg, dirs["configs"] / "config.yaml")
        set_random_seed(int(cfg["training"]["seed"]))
        device = _device(cfg["training"]["device"])
        logger.info(
            "ECNR run start target=%s volume_shape=%s device=%s",
            cfg["data"]["target"],
            cfg["data"]["volume_shape"],
            device,
        )
        if torch.cuda.is_available() and device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        logger.info("ECNR loading volume path=%s", cfg["data"]["target_path"])
        volume = _load_volume(cfg["data"]["target_path"], cfg["data"]["volume_shape"])
        logger.info("ECNR volume loaded shape=%s dtype=%s", volume.shape, volume.dtype)
        pyramid_started = time.perf_counter()
        logger.info("ECNR pyramid construction start")
        pyramid = build_three_scale_pyramid(
            volume,
            sigma=float(cfg["model"]["gaussian_sigma"]),
            cache_dir=dirs["cache"],
        )
        for level in (1, 2):
            workspace.register(dirs["cache"] / f"pyramid_scale_{level}.npy", pyramid[level].values)
        pyramid_seconds = float(time.perf_counter() - pyramid_started)
        logger.info("ECNR pyramid construction complete seconds=%.1f", pyramid_seconds)
        scale_payloads: list[dict[str, Any]] = []
        previous_reconstruction: np.ndarray | None = None
        previous_reconstruction_paths: tuple[Path, ...] = ()
        previous_times: np.ndarray | None = None
        cost: dict[str, Any] = {}
        sampling_mode = str(cfg["training"]["sampling_mode"])
        global_scalar_budget = int(cfg["training"]["scalar_predictions_per_epoch_budget"])
        scale_scalar_budgets = _pyramid_scalar_budgets(
            pyramid,
            global_scalar_budget if sampling_mode == "budgeted_random" else 0,
        )
        logger.info(
            "ECNR training budget sampling_mode=%s scalar_predictions_per_epoch=%d "
            "scale_budgets=%s cnn_core_voxel_budget=%d",
            sampling_mode,
            global_scalar_budget,
            scale_scalar_budgets,
            int(cfg["cnn"]["core_voxel_budget"]),
        )
        cost.setdefault("primary_sampling_mode", sampling_mode)
        cost.setdefault("primary_passes_per_epoch", int(cfg["training"]["passes_per_epoch"]))
        cost.setdefault("quantization_finetune_sampling_mode", sampling_mode)
        cost.setdefault(
            "quantization_finetune_passes_per_epoch",
            int(cfg["training"]["quantization_finetune_passes_per_epoch"]),
        )
        cost.setdefault("scales", [])
        cost.setdefault("quantization_finetune_logical_samples", 0)
        cost.setdefault("quantization_finetune_actual_predictions", 0)
        cost.setdefault("quantization_finetune_optimizer_steps", 0)
        cost.setdefault("quantization_and_finetune_seconds", 0.0)
        cost.setdefault("cnn", {})
        cost["pyramid_seconds"] = float(cost.get("pyramid_seconds", 0.0)) + pyramid_seconds
        block_shape = tuple(int(value) for value in cfg["model"]["block_shape_xyz"])

        for level in (2, 1, 0):
            scale: PyramidScale = pyramid[level]
            scale_values = scale.values
            scale_shape = tuple(int(value) for value in scale_values.shape)
            scale_times = np.asarray(scale.time_indices, dtype=np.int64).copy()
            logger.info("ECNR scale=%d preparation start shape=%s", level, scale_shape)
            block_preparation_started = time.perf_counter()
            has_previous = previous_reconstruction is not None
            upsampled: np.ndarray | None = None
            upsampled_path: Path | None = None
            residual_target_path: Path | None = None
            if not has_previous:
                target_values = scale_values
            else:
                upsampled_path = dirs["cache"] / f"upsampled_to_scale_{level}.npy"
                upsampled = upsample_to_scale(
                    previous_reconstruction,
                    previous_times,
                    fine_shape_tzyx=scale_shape,
                    fine_time_indices=scale_times,
                    output_path=upsampled_path,
                )
                workspace.register(upsampled_path, upsampled)
                workspace.release(
                    *previous_reconstruction_paths,
                    arrays=(previous_reconstruction,),
                    label=f"scale-{level}-coarse-reconstruction",
                )
                previous_reconstruction = None
                previous_reconstruction_paths = ()
                residual_target_path = dirs["cache"] / f"residual_target_scale_{level}.npy"
                target_values = _framewise_binary(
                    scale_values,
                    upsampled,
                    residual_target_path,
                    operation="subtract",
                )
                workspace.register(residual_target_path, target_values)
            normalized_path = dirs["cache"] / f"normalized_blocks_scale_{level}.npy"
            blocks = prepare_scale_blocks(
                target_values,
                block_shape_xyz=block_shape,
                residual_threshold=float(cfg["model"]["residual_threshold"]),
                keep_all=level == 2,
                normalized_blocks_path=normalized_path,
            )
            workspace.register(normalized_path, blocks.normalized_blocks)
            logger.info(
                "ECNR scale=%d block preparation complete effective_blocks=%d total_blocks=%d "
                "block_voxels=%d seconds=%.1f",
                level,
                blocks.effective_count,
                blocks.effective_mask.size,
                blocks.block_voxels,
                time.perf_counter() - block_preparation_started,
            )
            if residual_target_path is not None:
                workspace.release(
                    residual_target_path,
                    arrays=(target_values,),
                    label=f"scale-{level}-residual-target",
                )
            if level in (1, 2):
                workspace.release(
                    dirs["cache"] / f"pyramid_scale_{level}.npy",
                    arrays=(scale_values,),
                    label=f"scale-{level}-pyramid",
                )
            if blocks.effective_count == 0:
                logger.info("ECNR scale=%d contains no effective residual blocks", level)
                scale_payload = _serialize_scale(
                    level=level,
                    time_indices=scale_times,
                    blocks=blocks,
                    model=None,
                    quantization=None,
                )
                workspace.release(
                    normalized_path,
                    arrays=(blocks.normalized_blocks,),
                    label=f"scale-{level}-normalized-blocks",
                )
                blocks.normalized_blocks = np.empty((0, 0), dtype=np.float32)
                residual_path = dirs["cache"] / f"decoded_residual_scale_{level}.npy"
                residual_reconstruction = open_memmap(
                    residual_path,
                    mode="w+",
                    dtype=np.float32,
                    shape=scale_shape,
                )
                residual_reconstruction[:] = 0.0
                residual_reconstruction.flush()
                residual_paths = (residual_path,)
                workspace.register(residual_path, residual_reconstruction)
            else:
                target_per_mlp = int(cfg["model"]["target_blocks_per_mlp"][2 - level])
                expected_mlp_count = (
                    blocks.effective_count + target_per_mlp - 1
                ) // target_per_mlp
                clustering_started = time.perf_counter()
                logger.info(
                    "ECNR scale=%d clustering start effective_blocks=%d target_blocks_per_mlp=%d "
                    "expected_mlps=%d",
                    level,
                    blocks.effective_count,
                    target_per_mlp,
                    expected_mlp_count,
                )
                clustering = balanced_kmeans(
                    blocks.normalized_blocks,
                    target_blocks_per_mlp=target_per_mlp,
                    seed=int(cfg["clustering"]["seed"]),
                    n_init=int(cfg["clustering"]["n_init"]),
                    max_iter=int(cfg["clustering"]["max_iter"]),
                    tol=float(cfg["clustering"]["tol"]),
                )
                logger.info(
                    "ECNR scale=%d clustering complete mlps=%d max_slots=%d seconds=%.1f",
                    level,
                    clustering.cluster_sizes.size,
                    int(clustering.cluster_sizes.max()),
                    time.perf_counter() - clustering_started,
                )
                attach_clustering(blocks, clustering)
                targets = build_training_targets(
                    blocks,
                    output_path=dirs["cache"] / f"training_targets_scale_{level}.npy",
                )
                targets_path = dirs["cache"] / f"training_targets_scale_{level}.npy"
                workspace.register(targets_path, targets)
                workspace.release(
                    normalized_path,
                    arrays=(blocks.normalized_blocks,),
                    label=f"scale-{level}-normalized-blocks",
                )
                blocks.normalized_blocks = np.empty((0, 0), dtype=np.float32)
                valid = slot_valid_matrix(blocks.cluster_sizes)
                model = PackedSiren(
                    mlp_count=valid.shape[0],
                    max_slots=valid.shape[1],
                    slot_valid=torch.from_numpy(valid),
                )
                _, quantization = _train_scale(
                    model,
                    targets,
                    blocks,
                    cfg,
                    level=level,
                    scalar_predictions_per_epoch_budget=int(scale_scalar_budgets[level]),
                    device=device,
                    cost=cost,
                )
                workspace.release(
                    targets_path,
                    arrays=(targets,),
                    label=f"scale-{level}-training-targets",
                )
                decoded_path = dirs["cache"] / f"decoded_normalized_scale_{level}.npy"
                decoded = _decode_scale_model(
                    model,
                    blocks,
                    batch_size=int(cfg["evaluation"]["batch_size"]),
                    device=device,
                    output_path=decoded_path,
                    workspace=workspace,
                )
                residual_path = dirs["cache"] / f"decoded_residual_scale_{level}.npy"
                residual_reconstruction = reconstruct_from_normalized_blocks(
                    blocks,
                    decoded,
                    output_path=residual_path,
                )
                cropped_residual_path = residual_path.with_name(f"{residual_path.stem}_cropped.npy")
                residual_paths = (
                    (residual_path, cropped_residual_path)
                    if cropped_residual_path.exists()
                    else (residual_path,)
                )
                for path in residual_paths:
                    workspace.register(
                        path,
                        residual_reconstruction if path == residual_paths[-1] else None,
                    )
                workspace.release(
                    decoded_path,
                    arrays=(decoded,),
                    label=f"scale-{level}-decoded-blocks",
                )
                scale_payload = _serialize_scale(
                    level=level,
                    time_indices=scale_times,
                    blocks=blocks,
                    model=model,
                    quantization=quantization,
                )
            if not has_previous:
                previous_reconstruction = residual_reconstruction
                previous_reconstruction_paths = residual_paths
            else:
                previous_reconstruction = _framewise_add_in_place(
                    upsampled,
                    residual_reconstruction,
                )
                previous_reconstruction_paths = (upsampled_path,)
                workspace.release(
                    *residual_paths,
                    arrays=(residual_reconstruction,),
                    label=f"scale-{level}-decoded-residual",
                )
            previous_times = scale_times
            scale_payloads.append(scale_payload)
        mlp_reconstruction = _clip_in_place(previous_reconstruction)
        cnn_model = BoundaryCNN(hidden_channels=int(cfg["cnn"]["hidden_channels"])).to(device)
        cnn_started = time.perf_counter()
        cnn_cost = train_boundary_cnn(
            cnn_model,
            mlp_reconstruction,
            volume,
            epochs=int(cfg["cnn"]["epochs"]),
            lr=float(cfg["cnn"]["lr"]),
            core_shape_zyx=tuple(int(value) for value in cfg["cnn"]["tile_core_shape_zyx"]),
            halo=int(cfg["cnn"]["halo"]),
            device=device,
            seed=int(cfg["training"]["seed"]),
            sampling_mode=str(cfg["cnn"]["sampling_mode"]),
            core_voxel_budget=int(cfg["cnn"]["core_voxel_budget"]),
            log_every=int(cfg["training"]["log_every"]),
            progress_log_seconds=int(cfg["training"]["progress_log_seconds"]),
        )
        cnn_cost["seconds"] = float(time.perf_counter() - cnn_started)
        cost["cnn"] = cnn_cost
        cnn_quantization_started = time.perf_counter()
        logger.info("ECNR CNN quantization start bits=%d", int(cfg["quantization"]["cnn_bits"]))
        cnn_quantization = _quantize_cnn(
            cnn_model,
            bits=int(cfg["quantization"]["cnn_bits"]),
            seed=int(cfg["training"]["seed"]) + 50_000,
        )
        cost["cnn"]["quantization_seconds"] = float(
            time.perf_counter() - cnn_quantization_started
        )
        logger.info(
            "ECNR CNN quantization complete seconds=%.1f",
            cost["cnn"]["quantization_seconds"],
        )
        workspace.release(
            *previous_reconstruction_paths,
            arrays=(mlp_reconstruction,),
            label="cnn-input-reconstruction",
        )
        workspace.cleanup()
        cache_cleaned = True
        cost["cache"] = workspace.metrics()
        cost["cache_peak_bytes"] = int(cost["cache"]["peak_bytes"])
        cost["cache_released_bytes"] = int(cost["cache"]["released_bytes"])
        cost["cache_final_bytes"] = int(cost["cache"]["final_bytes"])
        cost["cache_cleanup_seconds"] = float(cost["cache"]["cleanup_seconds"])
        if torch.cuda.is_available() and device.type == "cuda":
            cost["peak_cuda_memory_bytes"] = int(torch.cuda.max_memory_allocated(device))
        else:
            cost["peak_cuda_memory_bytes"] = 0
        cost["primary_logical_samples_executed"] = int(
            sum(item["logical_samples"] for item in cost["scales"])
        )
        cost["primary_planned_logical_samples"] = int(
            sum(item["planned_logical_samples"] for item in cost["scales"])
        )
        cost["primary_actual_scalar_predictions"] = int(
            sum(item["actual_scalar_predictions"] for item in cost["scales"])
        )
        cost["primary_planned_scalar_predictions"] = int(
            sum(item["planned_actual_scalar_predictions"] for item in cost["scales"])
        )
        cost["quantization_finetune_planned_scalar_predictions"] = int(
            sum(
                item["quantization_finetune_planned_actual_predictions"]
                for item in cost["scales"]
            )
        )
        cost["total_mlp_actual_scalar_predictions"] = int(
            cost["primary_actual_scalar_predictions"]
            + cost["quantization_finetune_actual_predictions"]
        )
        if sampling_mode == "budgeted_random":
            cost["planned_mlp_sample_sites"] = int(
                (
                    int(cfg["training"]["epochs_per_scale"])
                    + int(cfg["training"]["quantization_finetune_epochs"])
                )
                * int(cfg["training"]["scalar_predictions_per_epoch_budget"])
            )
        else:
            cost["planned_mlp_sample_sites"] = int(
                cost["primary_planned_scalar_predictions"]
                + cost["quantization_finetune_planned_scalar_predictions"]
            )
        cost["main_reference_sample_budget"] = int(
            cost["planned_mlp_sample_sites"]
            + (
                int(cfg["cnn"]["core_voxel_budget"])
                if str(cfg["cnn"]["sampling_mode"]) == "budgeted_tiles"
                else 0
            )
        )
        cost["total_budgeted_sample_sites"] = int(
            cost["total_mlp_actual_scalar_predictions"]
            + int(cost["cnn"]["core_voxel_visits"])
        )
        cost["planned_sample_sites"] = int(cost["main_reference_sample_budget"])
        cost["executed_sample_sites"] = int(cost["total_budgeted_sample_sites"])
        cost["main_reference_budget_utilization"] = float(
            cost["total_budgeted_sample_sites"]
            / max(cost["main_reference_sample_budget"], 1)
        )
        if (
            sampling_mode == "budgeted_random"
            and str(cfg["cnn"]["sampling_mode"]) == "budgeted_tiles"
            and cost["total_budgeted_sample_sites"] > cost["main_reference_sample_budget"]
        ):
            raise RuntimeError("ECNR executed sample sites exceeded the configured main budget")
        logger.info(
            "ECNR training budget complete planned_sample_sites=%d executed_sample_sites=%d "
            "utilization=%.6f primary_predictions=%d qat_predictions=%d "
            "cnn_core_voxels=%d",
            cost["planned_sample_sites"],
            cost["executed_sample_sites"],
            cost["main_reference_budget_utilization"],
            cost["primary_actual_scalar_predictions"],
            cost["quantization_finetune_actual_predictions"],
            cost["cnn"]["core_voxel_visits"],
        )
        cost["primary_optimizer_steps"] = int(
            sum(item["optimizer_steps"] for item in cost["scales"])
        )
        cost["primary_planned_optimizer_steps"] = int(
            sum(item["planned_optimizer_steps"] for item in cost["scales"])
        )
        cost["total_seconds"] = float(time.perf_counter() - started)
        cost.pop("elapsed_active_seconds", None)
        cost_path = dirs["metrics"] / "training_cost.json"
        cost_path.write_text(json.dumps(cost, indent=2), encoding="utf-8")

        checkpoint_payload = {
            "format": INFERENCE_FORMAT,
            "model_name": "ecnr",
            "target_name": cfg["data"]["target"],
            "volume_shape": dict(cfg["data"]["volume_shape"]),
            "block_shape_xyz": list(block_shape),
            "config_hash": config_hash,
            "scales": scale_payloads,
            "cnn": cnn_quantization,
            "cnn_config": dict(cfg["cnn"]),
        }
        checkpoint = save_inference_checkpoint(
            dirs["checkpoints"] / f"{cfg['exp_id']}.pth",
            checkpoint_payload,
        )
        raw_bytes = int(np.prod(volume.shape, dtype=np.int64) * np.dtype(volume.dtype).itemsize)
        checkpoint_bytes = int(checkpoint.stat().st_size)
        summary = {
            "checkpoint_path": str(checkpoint),
            "training_cost_path": str(cost_path),
            "raw_target_bytes": raw_bytes,
            "checkpoint_bytes": checkpoint_bytes,
            "cr": float(raw_bytes / max(checkpoint_bytes, 1)),
        }
        (dirs["metrics"] / "training_summary.json").write_text(
            json.dumps(summary, indent=2),
            encoding="utf-8",
        )
        if cfg["evaluation"]["run_after_training"]:
            summary.update(run_evaluate(config_path, target=target, checkpoint=checkpoint))
        return summary
    finally:
        active_error = sys.exc_info()[0] is not None
        try:
            if not cache_cleaned:
                try:
                    workspace.cleanup()
                except Exception:
                    if active_error:
                        logger.exception("ECNR cache cleanup also failed while handling the original error")
                    else:
                        raise
        finally:
            if previous_sigterm is not None:
                signal.signal(signal.SIGTERM, previous_sigterm)
            close_file_handlers()


def _load_inference_payload(
    cfg: dict[str, Any],
    *,
    checkpoint: str | Path | None,
    dirs: dict[str, Path],
) -> tuple[dict[str, Any], Path]:
    source = Path(checkpoint or dirs["checkpoints"] / f"{cfg['exp_id']}.pth")
    payload = load_inference_checkpoint(source)
    if payload["target_name"] != cfg["data"]["target"] or payload["volume_shape"] != cfg["data"]["volume_shape"]:
        raise ValueError("ECNR inference source target/shape mismatch")
    return payload, source


def run_predict(
    config_path: str | Path,
    *,
    target: str | None = None,
    checkpoint: str | Path | None = None,
    time_indices: str | tuple[int, ...] | list[int] | None = None,
) -> dict[str, Any]:
    cfg = load_config(config_path, target_override=target)
    dirs = _run_for_path(cfg, checkpoint)
    runtime_output = os.environ.get("VAR_EXPERT_EVALUATION_OUTPUT_DIR")
    prediction_dir = Path(runtime_output) if runtime_output else dirs["predictions"]
    prediction_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = prediction_dir / "cache" if runtime_output else dirs["cache"]
    workspace = CacheWorkspace(cache_dir)
    try:
        device = _device(cfg["training"]["device"])
        payload, source = _load_inference_payload(
            cfg,
            checkpoint=checkpoint,
            dirs=dirs,
        )
        selected = parse_timestep_selection(
            time_indices,
            int(cfg["data"]["volume_shape"]["T"]),
        )
        output_path = prediction_dir / f"{cfg['exp_id']}.npy"
        decode_checkpoint_payload(
            payload,
            device=device,
            batch_size=int(cfg["evaluation"]["batch_size"]),
            work_dir=cache_dir / "decode",
            output_path=output_path,
            time_indices=selected,
            workspace=workspace,
        )
        return {
            "prediction_path": str(output_path),
            "model_path": str(source),
            "decoded_timesteps": list(selected),
        }
    finally:
        active_error = sys.exc_info()[0] is not None
        try:
            workspace.cleanup()
        except Exception:
            if active_error:
                logger.exception("ECNR prediction cache cleanup also failed")
            else:
                raise


def _evaluate(volume: np.ndarray, prediction: np.ndarray, model_path: Path) -> dict[str, Any]:
    accumulator = PSNRAccumulator()
    squared_error = absolute_error = 0.0
    count = 0
    per_time = []
    for time_index in range(volume.shape[0]):
        gt = np.asarray(volume[time_index], dtype=np.float32)
        pred = np.asarray(prediction[time_index], dtype=np.float32)
        accumulator.update(gt, pred)
        difference = pred.astype(np.float64) - gt.astype(np.float64)
        squared_error += float(np.sum(difference * difference))
        absolute_error += float(np.sum(np.abs(difference)))
        count += int(difference.size)
        per_time.append(
            {"t": time_index, "mse": mse(gt, pred), "mae": mae(gt, pred), "psnr": psnr(gt, pred)}
        )
    return {
        "per_time": per_time,
        "aggregate": {
            "mse": squared_error / count,
            "mae": absolute_error / count,
            "psnr": accumulator.compute(),
            "raw_target_bytes": int(volume.size * volume.dtype.itemsize),
            "checkpoint_bytes": int(model_path.stat().st_size),
            "cr": float(volume.size * volume.dtype.itemsize / max(model_path.stat().st_size, 1)),
        },
    }


def run_evaluate(
    config_path: str | Path,
    *,
    target: str | None = None,
    checkpoint: str | Path | None = None,
) -> dict[str, Any]:
    prediction_result = run_predict(
        config_path,
        target=target,
        checkpoint=checkpoint,
    )
    cfg = load_config(config_path, target_override=target)
    dirs = _run_for_path(cfg, checkpoint or prediction_result["model_path"])
    volume = _load_volume(cfg["data"]["target_path"], cfg["data"]["volume_shape"])
    prediction_path = Path(prediction_result["prediction_path"])
    prediction = np.load(prediction_path, mmap_mode="r")
    metrics = _evaluate(volume, prediction, Path(prediction_result["model_path"]))
    del prediction
    metrics_path = save_metrics(dirs["metrics"] / f"{cfg['exp_id']}.json", metrics)
    prediction_path.unlink(missing_ok=True)
    if prediction_path.parent.name == "predictions" and prediction_path.parent.is_dir() and not any(prediction_path.parent.iterdir()):
        prediction_path.parent.rmdir()
    return {
        **prediction_result,
        "prediction_path": None,
        "prediction_retained": False,
        "metrics": metrics,
        "metrics_path": str(metrics_path),
    }
