from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.lib.format import open_memmap

from .clustering import BalancedClustering


@dataclass
class ScaleBlocks:
    original_shape_tzyx: tuple[int, int, int, int]
    padded_shape_zyx: tuple[int, int, int]
    padding_zyx: tuple[tuple[int, int], tuple[int, int], tuple[int, int]]
    block_shape_xyz: tuple[int, int, int]
    spatial_grid_zyx: tuple[int, int, int]
    effective_mask: np.ndarray
    effective_positions: np.ndarray
    block_min: np.ndarray
    block_max: np.ndarray
    normalized_blocks: np.ndarray
    block_to_mlp: np.ndarray | None = None
    block_to_slot: np.ndarray | None = None
    cluster_sizes: np.ndarray | None = None

    @property
    def block_voxels(self) -> int:
        bx, by, bz = self.block_shape_xyz
        return int(bx * by * bz)

    @property
    def effective_count(self) -> int:
        return int(self.effective_positions.shape[0])


def symmetric_padding(size: int, block: int) -> tuple[int, int]:
    amount = (-int(size)) % int(block)
    before = amount // 2
    return before, amount - before


def _iter_frame_blocks(
    frame: np.ndarray,
    *,
    block_shape_xyz: tuple[int, int, int],
    padding_zyx: tuple[tuple[int, int], tuple[int, int], tuple[int, int]],
):
    bx, by, bz = block_shape_xyz
    padded = np.pad(np.asarray(frame, dtype=np.float32), padding_zyx, mode="constant")
    gz, gy, gx = padded.shape[0] // bz, padded.shape[1] // by, padded.shape[2] // bx
    for z_index in range(gz):
        for y_index in range(gy):
            for x_index in range(gx):
                block = padded[
                    z_index * bz : (z_index + 1) * bz,
                    y_index * by : (y_index + 1) * by,
                    x_index * bx : (x_index + 1) * bx,
                ]
                yield block


def prepare_scale_blocks(
    target: np.ndarray,
    *,
    block_shape_xyz: tuple[int, int, int],
    residual_threshold: float,
    keep_all: bool,
    normalized_blocks_path: str | Path | None = None,
) -> ScaleBlocks:
    values = np.asarray(target)
    if values.ndim != 4:
        raise ValueError("target must have shape [T,Z,Y,X]")
    bx, by, bz = (int(value) for value in block_shape_xyz)
    pad_z = symmetric_padding(values.shape[1], bz)
    pad_y = symmetric_padding(values.shape[2], by)
    pad_x = symmetric_padding(values.shape[3], bx)
    padding = (pad_z, pad_y, pad_x)
    padded_shape = (
        values.shape[1] + sum(pad_z),
        values.shape[2] + sum(pad_y),
        values.shape[3] + sum(pad_x),
    )
    grid = (padded_shape[0] // bz, padded_shape[1] // by, padded_shape[2] // bx)
    spatial_count = int(np.prod(grid, dtype=np.int64))
    effective_mask = np.zeros((values.shape[0], spatial_count), dtype=bool)
    mins: list[float] = []
    maxs: list[float] = []
    positions: list[tuple[int, int]] = []

    for time_index in range(values.shape[0]):
        for spatial_index, block in enumerate(
            _iter_frame_blocks(values[time_index], block_shape_xyz=(bx, by, bz), padding_zyx=padding)
        ):
            effective = bool(keep_all or float(np.mean(block * block, dtype=np.float64)) > float(residual_threshold))
            if not effective:
                continue
            effective_mask[time_index, spatial_index] = True
            positions.append((time_index, spatial_index))
            mins.append(float(block.min()))
            maxs.append(float(block.max()))

    count = len(positions)
    block_voxels = bx * by * bz
    if normalized_blocks_path is None:
        normalized = np.empty((count, block_voxels), dtype=np.float32)
    else:
        path = Path(normalized_blocks_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        normalized = open_memmap(path, mode="w+", dtype=np.float32, shape=(count, block_voxels))
    write_index = 0
    for time_index in range(values.shape[0]):
        for spatial_index, block in enumerate(
            _iter_frame_blocks(values[time_index], block_shape_xyz=(bx, by, bz), padding_zyx=padding)
        ):
            if not effective_mask[time_index, spatial_index]:
                continue
            minimum = mins[write_index]
            maximum = maxs[write_index]
            if maximum == minimum:
                normalized[write_index] = 0.0
            else:
                normalized[write_index] = (
                    2.0 * (block.reshape(-1).astype(np.float32) - minimum) / (maximum - minimum) - 1.0
                )
            write_index += 1
    if hasattr(normalized, "flush"):
        normalized.flush()
    return ScaleBlocks(
        original_shape_tzyx=tuple(int(value) for value in values.shape),
        padded_shape_zyx=padded_shape,
        padding_zyx=padding,
        block_shape_xyz=(bx, by, bz),
        spatial_grid_zyx=grid,
        effective_mask=effective_mask,
        effective_positions=np.asarray(positions, dtype=np.int64).reshape(-1, 2),
        block_min=np.asarray(mins, dtype=np.float32),
        block_max=np.asarray(maxs, dtype=np.float32),
        normalized_blocks=normalized,
    )


def attach_clustering(blocks: ScaleBlocks, clustering: BalancedClustering) -> ScaleBlocks:
    if clustering.labels.shape != (blocks.effective_count,):
        raise ValueError("Clustering label count does not match effective blocks")
    blocks.block_to_mlp = clustering.labels.astype(np.int64, copy=False)
    blocks.block_to_slot = clustering.slots.astype(np.int64, copy=False)
    blocks.cluster_sizes = clustering.cluster_sizes.astype(np.int64, copy=False)
    return blocks


def slot_valid_matrix(cluster_sizes: np.ndarray) -> np.ndarray:
    sizes = np.asarray(cluster_sizes, dtype=np.int64)
    max_slots = int(sizes.max())
    return np.arange(max_slots, dtype=np.int64)[None, :] < sizes[:, None]


def build_training_targets(
    blocks: ScaleBlocks,
    *,
    output_path: str | Path | None = None,
) -> np.ndarray:
    if blocks.block_to_mlp is None or blocks.block_to_slot is None or blocks.cluster_sizes is None:
        raise ValueError("Clustering must be attached before building training targets")
    mlp_count = int(blocks.cluster_sizes.shape[0])
    max_slots = int(blocks.cluster_sizes.max())
    shape = (mlp_count, max_slots, blocks.block_voxels)
    if output_path is None:
        targets = np.zeros(shape, dtype=np.float32)
    else:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        targets = open_memmap(path, mode="w+", dtype=np.float32, shape=shape)
        targets[:] = 0.0
    for block_index in range(blocks.effective_count):
        targets[blocks.block_to_mlp[block_index], blocks.block_to_slot[block_index]] = (
            blocks.normalized_blocks[block_index]
        )
    if hasattr(targets, "flush"):
        targets.flush()
    return targets


def reconstruct_from_normalized_blocks(
    blocks: ScaleBlocks,
    decoded: np.ndarray,
    *,
    output_path: str | Path | None = None,
) -> np.ndarray:
    padded_shape = (blocks.original_shape_tzyx[0], *blocks.padded_shape_zyx)
    if output_path is None:
        values = np.zeros(padded_shape, dtype=np.float32)
    else:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        values = open_memmap(path, mode="w+", dtype=np.float32, shape=padded_shape)
        values[:] = 0.0
    bx, by, bz = blocks.block_shape_xyz
    gz, gy, gx = blocks.spatial_grid_zyx
    for block_index, (time_index, spatial_index) in enumerate(blocks.effective_positions.tolist()):
        z_index = spatial_index // (gy * gx)
        remainder = spatial_index % (gy * gx)
        y_index = remainder // gx
        x_index = remainder % gx
        normalized = np.clip(decoded[block_index], -1.0, 1.0)
        restored = (
            (normalized + 1.0) * 0.5 * (blocks.block_max[block_index] - blocks.block_min[block_index])
            + blocks.block_min[block_index]
        ).reshape(bz, by, bx)
        values[
            time_index,
            z_index * bz : (z_index + 1) * bz,
            y_index * by : (y_index + 1) * by,
            x_index * bx : (x_index + 1) * bx,
        ] = restored
    (z_before, z_after), (y_before, y_after), (x_before, x_after) = blocks.padding_zyx
    z_stop = values.shape[1] - z_after if z_after else values.shape[1]
    y_stop = values.shape[2] - y_after if y_after else values.shape[2]
    x_stop = values.shape[3] - x_after if x_after else values.shape[3]
    cropped = values[:, z_before:z_stop, y_before:y_stop, x_before:x_stop]
    if output_path is None or all(sum(pair) == 0 for pair in blocks.padding_zyx):
        if hasattr(values, "flush"):
            values.flush()
        return cropped
    # A .npy memmap cannot expose a cropped view as a standalone artifact.
    cropped_path = Path(output_path).with_name(f"{Path(output_path).stem}_cropped.npy")
    output = open_memmap(cropped_path, mode="w+", dtype=np.float32, shape=blocks.original_shape_tzyx)
    for time_index in range(blocks.original_shape_tzyx[0]):
        output[time_index] = cropped[time_index]
    output.flush()
    return output
