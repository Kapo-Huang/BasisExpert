from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


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
) -> dict[str, int | float]:
    optimizer = torch.optim.Adam(model.parameters(), lr=float(lr))
    rng = np.random.default_rng(int(seed))
    voxel_count = int(np.prod(inputs_tzyx.shape, dtype=np.int64))
    steps = 0
    model.train()
    for _ in range(int(epochs)):
        for time_index in rng.permutation(inputs_tzyx.shape[0]).tolist():
            optimizer.zero_grad(set_to_none=True)
            frame = torch.as_tensor(inputs_tzyx[time_index], dtype=torch.float32)
            target = torch.as_tensor(targets_tzyx[time_index], dtype=torch.float32)
            frame_voxels = int(frame.numel())
            for core in iter_core_slices(tuple(frame.shape), core_shape_zyx):
                tile = extract_core_with_halo(frame, core, halo=halo).to(device)
                prediction = model(tile[None, None])[0, 0]
                core_shape = tuple(int(item.stop - item.start) for item in core)
                offsets = tuple(min(int(halo), int(item.start)) for item in core)
                prediction_core = prediction[
                    offsets[0] : offsets[0] + core_shape[0],
                    offsets[1] : offsets[1] + core_shape[1],
                    offsets[2] : offsets[2] + core_shape[2],
                ]
                target_core = target[core].to(device)
                # Core-only loss. Summing disjoint cores gives the exact full-frame MSE.
                loss = torch.sum((prediction_core - target_core) ** 2) / float(frame_voxels)
                loss.backward()
            optimizer.step()
            steps += 1
    return {
        "epochs": int(epochs),
        "optimizer_steps": int(steps),
        "voxel_visits": int(voxel_count * int(epochs)),
    }
