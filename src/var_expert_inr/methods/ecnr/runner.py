from __future__ import annotations

import copy
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.lib.format import open_memmap

from ...evaluation.metrics import PSNRAccumulator, mae, mse, psnr, save_metrics
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
        for path in result.values():
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
    epochs_per_scale = int(training["epochs_per_scale"])
    planned_optimizer_steps = epochs_per_scale * passes_per_epoch * batches_per_pass
    planned_logical_samples = epochs_per_scale * passes_per_epoch * axis_length
    progress_log_seconds = int(training["progress_log_seconds"])
    last_progress_log = time.perf_counter()
    logger.info(
        "ECNR scale=%d start effective_blocks=%d mlps=%d max_slots=%d axis_length=%d "
        "batches_per_pass=%d passes_per_epoch=%d epochs=%d planned_optimizer_steps=%d",
        level,
        blocks.effective_count,
        model.mlp_count,
        max_slots,
        axis_length,
        batches_per_pass,
        passes_per_epoch,
        epochs_per_scale,
        planned_optimizer_steps,
    )

    for epoch in range(1, epochs_per_scale + 1):
        model.train()
        epoch_loss = 0.0
        epoch_batches = 0
        for pass_index in range(1, passes_per_epoch + 1):
            for batch_index, indices in enumerate(
                _full_pass_batches(
                    axis_length,
                    batch_size=batch_size,
                    rng=rng,
                ),
                start=1,
            ):
                coord_batch = coordinates[indices].to(device)
                slot_batch = slots[indices].to(device)
                target_batch = torch.from_numpy(np.asarray(flat_targets[:, indices], dtype=np.float32)).to(device)
                mask_batch = model.expanded_slot_mask(slot_batch).to(device)
                optimizer.zero_grad(set_to_none=True)
                prediction = model(coord_batch, slot_batch)
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
                    logger.info(
                        "ECNR scale=%d progress epoch=%d/%d pass=%d/%d batch=%d/%d "
                        "optimizer_steps=%d elapsed_seconds=%.1f",
                        level,
                        epoch,
                        epochs_per_scale,
                        pass_index,
                        passes_per_epoch,
                        batch_index,
                        batches_per_pass,
                        optimizer_steps,
                        now - scale_started,
                    )
                    last_progress_log = now
        if epoch in pruning_schedule:
            losses = _mlp_losses(
                model,
                targets,
                coordinates,
                slots,
                batch_size=int(training["batch_size"]),
                device=device,
            )
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
    finetune_logical_samples = 0
    finetune_actual_predictions = 0
    finetune_optimizer_steps = 0
    if finetune_epochs:
        logger.info(
            "ECNR scale=%d QAT start epochs=%d passes_per_epoch=%d batches_per_pass=%d "
            "planned_optimizer_steps=%d",
            level,
            finetune_epochs,
            finetune_passes,
            batches_per_pass,
            finetune_epochs * finetune_passes * batches_per_pass,
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
            for pass_index in range(1, finetune_passes + 1):
                for batch_index, indices in enumerate(
                    _full_pass_batches(
                        axis_length,
                        batch_size=batch_size,
                        rng=rng,
                    ),
                    start=1,
                ):
                    quantization.materialize(model)
                    finetune_optimizer.zero_grad(set_to_none=True)
                    coord_batch = coordinates[indices].to(device)
                    slot_batch = slots[indices].to(device)
                    target_batch = torch.from_numpy(np.asarray(flat_targets[:, indices], dtype=np.float32)).to(device)
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
                        logger.info(
                            "ECNR scale=%d QAT progress epoch=%d/%d pass=%d/%d batch=%d/%d "
                            "optimizer_steps=%d elapsed_seconds=%.1f",
                            level,
                            finetune_epoch,
                            finetune_epochs,
                            pass_index,
                            finetune_passes,
                            batch_index,
                            batches_per_pass,
                            finetune_optimizer_steps,
                            now - quantization_started,
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
            "passes_per_epoch": passes_per_epoch,
            "epochs": epochs_per_scale,
            "planned_logical_samples": planned_logical_samples,
            "planned_optimizer_steps": planned_optimizer_steps,
            "logical_samples": int(logical_samples),
            "actual_scalar_predictions": int(actual_predictions),
            "optimizer_steps": int(optimizer_steps),
            "quantization_finetune_passes_per_epoch": finetune_passes,
            "quantization_finetune_epochs": finetune_epochs,
            "quantization_finetune_planned_logical_samples": (
                finetune_epochs * finetune_passes * axis_length
            ),
            "quantization_finetune_planned_optimizer_steps": (
                finetune_epochs * finetune_passes * batches_per_pass
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
    for block_index in range(blocks.effective_count):
        decoded[block_index] = decoded_slots[
            blocks.block_to_mlp[block_index],
            blocks.block_to_slot[block_index],
        ]
    if hasattr(decoded_slots, "flush"):
        decoded_slots.flush()
    if hasattr(decoded, "flush"):
        decoded.flush()
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
) -> np.ndarray:
    blocks = _deserialize_blocks(payload["blocks"])
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
    )
    return reconstruct_from_normalized_blocks(blocks, decoded, output_path=output_path)


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


def _clip_to_memmap(
    values: np.ndarray,
    output_path: str | Path,
    *,
    lower: float = -1.0,
    upper: float = 1.0,
) -> np.ndarray:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    output = open_memmap(path, mode="w+", dtype=np.float32, shape=tuple(values.shape))
    for time_index in range(values.shape[0]):
        output[time_index] = np.clip(
            np.asarray(values[time_index], dtype=np.float32),
            float(lower),
            float(upper),
        )
    output.flush()
    return output


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
) -> np.ndarray:
    if payload.get("format") != INFERENCE_FORMAT:
        raise ValueError("Invalid ECNR inference checkpoint payload")
    work = None if work_dir is None else Path(work_dir)
    if work is not None:
        work.mkdir(parents=True, exist_ok=True)
    composite = None
    previous_times = None
    for scale_payload in sorted(payload["scales"], key=lambda item: int(item["level"]), reverse=True):
        level = int(scale_payload["level"])
        residual_path = None if work is None else work / f"decode_residual_scale_{level}.npy"
        residual = _decode_scale_payload(
            scale_payload,
            device=device,
            batch_size=batch_size,
            output_path=residual_path,
        )
        current_times = np.asarray(scale_payload["time_indices"], dtype=np.int64)
        if composite is None:
            composite = residual
        else:
            upsampled = upsample_to_scale(
                composite,
                previous_times,
                fine_shape_tzyx=tuple(residual.shape),
                fine_time_indices=current_times,
                output_path=None if work is None else work / f"decode_upsampled_scale_{level}.npy",
            )
            if work is None:
                composite = upsampled + residual
            else:
                composite = _framewise_binary(
                    upsampled,
                    residual,
                    work / f"decode_composite_scale_{level}.npy",
                    operation="add",
                )
        previous_times = current_times
    if work is None:
        composite = np.clip(composite, -1.0, 1.0)
    else:
        composite = _clip_to_memmap(composite, work / "decode_mlp_clipped.npy")
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
        pyramid_seconds = float(time.perf_counter() - pyramid_started)
        logger.info("ECNR pyramid construction complete seconds=%.1f", pyramid_seconds)
        scale_payloads: list[dict[str, Any]] = []
        previous_reconstruction: np.ndarray | None = None
        previous_times: np.ndarray | None = None
        cost: dict[str, Any] = {}
        cost.setdefault("primary_sampling_mode", "full_pass")
        cost.setdefault("primary_passes_per_epoch", int(cfg["training"]["passes_per_epoch"]))
        cost.setdefault("quantization_finetune_sampling_mode", "full_pass")
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
            logger.info("ECNR scale=%d preparation start shape=%s", level, scale.values.shape)
            block_preparation_started = time.perf_counter()
            if previous_reconstruction is None:
                target_values = scale.values
            else:
                upsampled = upsample_to_scale(
                    previous_reconstruction,
                    previous_times,
                    fine_shape_tzyx=tuple(scale.values.shape),
                    fine_time_indices=scale.time_indices,
                    output_path=dirs["cache"] / f"upsampled_to_scale_{level}.npy",
                )
                target_values = _framewise_binary(
                    scale.values,
                    upsampled,
                    dirs["cache"] / f"residual_target_scale_{level}.npy",
                    operation="subtract",
                )
            blocks = prepare_scale_blocks(
                target_values,
                block_shape_xyz=block_shape,
                residual_threshold=float(cfg["model"]["residual_threshold"]),
                keep_all=level == 2,
                normalized_blocks_path=dirs["cache"] / f"normalized_blocks_scale_{level}.npy",
            )
            logger.info(
                "ECNR scale=%d block preparation complete effective_blocks=%d total_blocks=%d "
                "block_voxels=%d seconds=%.1f",
                level,
                blocks.effective_count,
                blocks.effective_mask.size,
                blocks.block_voxels,
                time.perf_counter() - block_preparation_started,
            )
            if blocks.effective_count == 0:
                logger.info("ECNR scale=%d contains no effective residual blocks", level)
                scale_payload = _serialize_scale(
                    level=level,
                    time_indices=scale.time_indices,
                    blocks=blocks,
                    model=None,
                    quantization=None,
                )
                residual_reconstruction = open_memmap(
                    dirs["cache"] / f"decoded_residual_scale_{level}.npy",
                    mode="w+",
                    dtype=np.float32,
                    shape=tuple(scale.values.shape),
                )
                residual_reconstruction[:] = 0.0
                residual_reconstruction.flush()
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
                    device=device,
                    cost=cost,
                )
                decoded = _decode_scale_model(
                    model,
                    blocks,
                    batch_size=int(cfg["evaluation"]["batch_size"]),
                    device=device,
                    output_path=dirs["cache"] / f"decoded_normalized_scale_{level}.npy",
                )
                residual_reconstruction = reconstruct_from_normalized_blocks(
                    blocks,
                    decoded,
                    output_path=dirs["cache"] / f"decoded_residual_scale_{level}.npy",
                )
                scale_payload = _serialize_scale(
                    level=level,
                    time_indices=scale.time_indices,
                    blocks=blocks,
                    model=model,
                    quantization=quantization,
                )
            if previous_reconstruction is None:
                previous_reconstruction = residual_reconstruction
            else:
                previous_reconstruction = _framewise_binary(
                    upsampled,
                    residual_reconstruction,
                    dirs["cache"] / f"composite_scale_{level}.npy",
                    operation="add",
                )
            previous_times = scale.time_indices
            scale_payloads.append(scale_payload)
        mlp_reconstruction = _clip_to_memmap(
            previous_reconstruction,
            dirs["cache"] / "mlp_reconstruction_clipped.npy",
        )
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
        )
        cnn_cost["seconds"] = float(time.perf_counter() - cnn_started)
        cost["cnn"] = cnn_cost
        cnn_quantization_started = time.perf_counter()
        cnn_quantization = _quantize_cnn(
            cnn_model,
            bits=int(cfg["quantization"]["cnn_bits"]),
            seed=int(cfg["training"]["seed"]) + 50_000,
        )
        cost["cnn"]["quantization_seconds"] = float(
            time.perf_counter() - cnn_quantization_started
        )
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
) -> dict[str, Any]:
    cfg = load_config(config_path, target_override=target)
    dirs = _run_for_path(cfg, checkpoint)
    device = _device(cfg["training"]["device"])
    payload, source = _load_inference_payload(
        cfg,
        checkpoint=checkpoint,
        dirs=dirs,
    )
    output_path = dirs["predictions"] / f"{cfg['exp_id']}.npy"
    decode_checkpoint_payload(
        payload,
        device=device,
        batch_size=int(cfg["evaluation"]["batch_size"]),
        work_dir=dirs["cache"] / "decode",
        output_path=output_path,
    )
    return {"prediction_path": str(output_path), "model_path": str(source)}


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
    prediction_retained = bool(cfg["evaluation"]["save_predictions"])
    if not prediction_retained:
        prediction_path.unlink(missing_ok=True)
    return {
        **prediction_result,
        "prediction_retained": prediction_retained,
        "metrics": metrics,
        "metrics_path": str(metrics_path),
    }
