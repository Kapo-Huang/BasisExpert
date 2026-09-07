from __future__ import annotations

import logging
import time
from collections.abc import Iterator

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


logger = logging.getLogger(__name__)


class BoundaryCNN(nn.Module):
    def __init__(self, hidden_channels: int = 32) -> None:
        super().__init__()
        channels = [1, int(hidden_channels), int(hidden_channels), int(hidden_channels), int(hidden_channels), 1]
        self.layers = nn.ModuleList(
            [
                nn.Conv3d(channels[index], channels[index + 1], kernel_size=3, stride=1, padding=1, bias=True)
                for index in range(5)
            ]
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = inputs
        for layer in self.layers[:-1]:
            hidden = F.relu(layer(hidden))
        return self.layers[-1](hidden)


def iter_core_slices(
    spatial_shape_zyx: tuple[int, int, int],
    core_shape_zyx: tuple[int, int, int],
) -> Iterator[tuple[slice, slice, slice]]:
    for z0 in range(0, spatial_shape_zyx[0], core_shape_zyx[0]):
        for y0 in range(0, spatial_shape_zyx[1], core_shape_zyx[1]):
            for x0 in range(0, spatial_shape_zyx[2], core_shape_zyx[2]):
                yield (
                    slice(z0, min(z0 + core_shape_zyx[0], spatial_shape_zyx[0])),
                    slice(y0, min(y0 + core_shape_zyx[1], spatial_shape_zyx[1])),
                    slice(x0, min(x0 + core_shape_zyx[2], spatial_shape_zyx[2])),
                )


def extract_core_with_halo(
    frame: np.ndarray | torch.Tensor,
    core_slices: tuple[slice, slice, slice],
    *,
    halo: int = 5,
) -> torch.Tensor:
    source = torch.as_tensor(frame, dtype=torch.float32)
    if source.ndim != 3:
        raise ValueError("frame must have shape [Z,Y,X]")
    slices = []
    for axis, core in enumerate(core_slices):
        start = int(core.start)
        stop = int(core.stop)
        available_start = max(0, start - int(halo))
        available_stop = min(int(source.shape[axis]), stop + int(halo))
        slices.append(slice(available_start, available_stop))
    # Do not explicitly append an input-level zero halo at physical boundaries.
    # Each Conv3d layer's own padding must create those zeros; otherwise biases
    # generate non-zero out-of-domain hidden activations that leak into the core.
    return source[slices[0], slices[1], slices[2]]


def forward_tiled(
    model: BoundaryCNN,
    frame: np.ndarray | torch.Tensor,
    *,
    core_shape_zyx: tuple[int, int, int],
    halo: int = 5,
    device: torch.device | None = None,
) -> torch.Tensor:
    source = torch.as_tensor(frame, dtype=torch.float32)
    output = torch.empty_like(source, device="cpu")
    model_device = device or next(model.parameters()).device
    for core in iter_core_slices(tuple(source.shape), core_shape_zyx):
        tile = extract_core_with_halo(source, core, halo=halo).to(model_device)
        prediction = model(tile[None, None])[0, 0]
        core_shape = tuple(int(item.stop - item.start) for item in core)
        offsets = tuple(min(int(halo), int(item.start)) for item in core)
        cropped = prediction[
            offsets[0] : offsets[0] + core_shape[0],
            offsets[1] : offsets[1] + core_shape[1],
            offsets[2] : offsets[2] + core_shape[2],
        ]
        output[core] = cropped.detach().cpu()
    return output


def train_boundary_cnn(
    model: BoundaryCNN,
    inputs_tzyx: np.ndarray,
    targets_tzyx: np.ndarray,
    *,
    epochs: int,
    lr: float,
    core_shape_zyx: tuple[int, int, int],
    halo: int,
    device: torch.device,
    seed: int,
    sampling_mode: str = "full_volume",
    core_voxel_budget: int = 0,
    log_every: int = 1,
    progress_log_seconds: int = 60,
) -> dict[str, int | float | str]:
    optimizer = torch.optim.Adam(model.parameters(), lr=float(lr))
    rng = np.random.default_rng(int(seed))
    voxel_count = int(np.prod(inputs_tzyx.shape, dtype=np.int64))
    mode = str(sampling_mode).strip().lower()
    if mode not in {"full_volume", "budgeted_tiles"}:
        raise ValueError("sampling_mode must be 'full_volume' or 'budgeted_tiles'")
    if mode == "budgeted_tiles" and int(core_voxel_budget) <= 0:
        raise ValueError("core_voxel_budget must be positive for budgeted_tiles")

    cores = list(iter_core_slices(tuple(inputs_tzyx.shape[1:]), core_shape_zyx))
    core_sizes = [
        int(np.prod(tuple(item.stop - item.start for item in core), dtype=np.int64))
        for core in cores
    ]
    if mode == "full_volume":
        def full_schedules():
            for _ in range(int(epochs)):
                yield (
                    (time_index, core_index)
                    for time_index in rng.permutation(inputs_tzyx.shape[0]).tolist()
                    for core_index in range(len(cores))
                )

        schedules = full_schedules()
        planned_core_voxels = voxel_count * int(epochs)
        budget = planned_core_voxels
        planned_tiles = int(epochs) * int(inputs_tzyx.shape[0]) * len(cores)
    else:
        minimum_core = min(core_sizes)
        pair_count = int(inputs_tzyx.shape[0]) * len(cores)

        def pair_stream():
            while True:
                time_order = rng.permutation(inputs_tzyx.shape[0]).tolist()
                spatial_orders = {
                    int(time_index): rng.permutation(len(cores)).tolist()
                    for time_index in time_order
                }
                for spatial_rank in range(len(cores)):
                    for time_index in time_order:
                        yield int(time_index), int(spatial_orders[int(time_index)][spatial_rank])

        stream = pair_stream()
        epoch_base, epoch_remainder = divmod(int(core_voxel_budget), max(int(epochs), 1))
        schedules: list[list[tuple[int, int]]] = []
        planned_core_voxels = 0
        carry = 0
        for epoch_index in range(int(epochs)):
            available = epoch_base + (1 if epoch_index < epoch_remainder else 0) + carry
            epoch_schedule: list[tuple[int, int]] = []
            rejected = 0
            while available >= minimum_core:
                time_index, core_index = next(stream)
                size = core_sizes[core_index]
                if size <= available:
                    epoch_schedule.append((time_index, core_index))
                    available -= size
                    planned_core_voxels += size
                    rejected = 0
                else:
                    rejected += 1
                    if rejected >= pair_count:
                        break
            carry = available
            schedules.append(epoch_schedule)
        budget = int(core_voxel_budget)

        planned_tiles = sum(len(schedule) for schedule in schedules)
    logger.info(
        "ECNR CNN start sampling_mode=%s epochs=%d candidate_tiles=%d planned_tiles=%d "
        "core_voxel_budget=%d planned_core_voxels=%d",
        mode,
        int(epochs),
        int(inputs_tzyx.shape[0]) * len(cores),
        planned_tiles,
        budget,
        planned_core_voxels,
    )
    steps = 0
    sampled_tiles = 0
    core_voxel_visits = 0
    halo_voxel_visits = 0
    model.train()
    started = time.perf_counter()
    last_progress = started
    for epoch, schedule in enumerate(schedules, start=1):
        epoch_started = time.perf_counter()
        epoch_loss = 0.0
        epoch_steps = 0
        epoch_core_voxels = 0
        epoch_halo_voxels = 0
        current_frame: int | None = None
        if mode == "full_volume":
            optimizer.zero_grad(set_to_none=True)
        for time_index, core_index in schedule:
            if mode == "full_volume" and current_frame != time_index:
                if current_frame is not None:
                    optimizer.step()
                    steps += 1
                    optimizer.zero_grad(set_to_none=True)
                current_frame = time_index
            elif mode == "budgeted_tiles":
                optimizer.zero_grad(set_to_none=True)
            frame = torch.as_tensor(inputs_tzyx[time_index], dtype=torch.float32)
            frame_voxels = int(frame.numel())
            core = cores[core_index]
            tile = extract_core_with_halo(frame, core, halo=halo).to(device)
            prediction = model(tile[None, None])[0, 0]
            core_shape = tuple(int(item.stop - item.start) for item in core)
            offsets = tuple(min(int(halo), int(item.start)) for item in core)
            prediction_core = prediction[
                offsets[0] : offsets[0] + core_shape[0],
                offsets[1] : offsets[1] + core_shape[1],
                offsets[2] : offsets[2] + core_shape[2],
            ]
            target_core = torch.from_numpy(
                np.array(targets_tzyx[time_index][core], dtype=np.float32, copy=True)
            ).to(device)
            if mode == "full_volume":
                loss = torch.sum((prediction_core - target_core) ** 2) / float(frame_voxels)
            else:
                loss = torch.mean((prediction_core - target_core) ** 2)
            loss.backward()
            if mode == "budgeted_tiles":
                optimizer.step()
                steps += 1
            epoch_loss += float(loss.detach())
            epoch_steps += 1
            sampled_tiles += 1
            core_voxel_visits += int(target_core.numel())
            halo_voxel_visits += int(tile.numel())
            epoch_core_voxels += int(target_core.numel())
            epoch_halo_voxels += int(tile.numel())
            now = time.perf_counter()
            if progress_log_seconds and now - last_progress >= int(progress_log_seconds):
                elapsed = now - started
                eta = elapsed * (planned_tiles - sampled_tiles) / max(sampled_tiles, 1)
                logger.info(
                    "ECNR CNN progress epoch=%d/%d tile=%d/%d core_voxels=%d/%d "
                    "halo_voxels=%d optimizer_steps=%d elapsed_seconds=%.1f eta_seconds=%.1f",
                    epoch,
                    int(epochs),
                    sampled_tiles,
                    planned_tiles,
                    core_voxel_visits,
                    budget,
                    halo_voxel_visits,
                    steps,
                    elapsed,
                    eta,
                )
                last_progress = now
        if mode == "full_volume" and current_frame is not None:
            optimizer.step()
            steps += 1
        if log_every and epoch % int(log_every) == 0:
            logger.info(
                "ECNR CNN epoch=%d/%d loss=%.7g sampled_tiles=%d core_voxels=%d "
                "halo_voxels=%d optimizer_steps=%d epoch_seconds=%.1f "
                "total_elapsed_seconds=%.1f",
                epoch,
                int(epochs),
                epoch_loss / max(epoch_steps, 1),
                epoch_steps,
                epoch_core_voxels,
                epoch_halo_voxels,
                steps,
                time.perf_counter() - epoch_started,
                time.perf_counter() - started,
            )
    elapsed = time.perf_counter() - started
    logger.info(
        "ECNR CNN complete sampling_mode=%s tiles=%d core_voxels=%d halo_voxels=%d "
        "optimizer_steps=%d seconds=%.1f",
        mode,
        sampled_tiles,
        core_voxel_visits,
        halo_voxel_visits,
        steps,
        elapsed,
    )
    return {
        "sampling_mode": mode,
        "epochs": int(epochs),
        "optimizer_steps": int(steps),
        "sampled_tiles": int(sampled_tiles),
        "planned_tiles": int(planned_tiles),
        "core_voxel_budget": int(budget),
        "planned_core_voxel_visits": int(planned_core_voxels),
        "core_voxel_visits": int(core_voxel_visits),
        "halo_voxel_visits": int(halo_voxel_visits),
        "voxel_visits": int(core_voxel_visits),
        "budget_utilization": float(core_voxel_visits / max(budget, 1)),
        "seconds": float(elapsed),
    }
