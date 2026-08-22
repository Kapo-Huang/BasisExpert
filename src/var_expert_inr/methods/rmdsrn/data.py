from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..fv_srn.data import TemporalVolume


def sample_voxel_batch(
    volume: TemporalVolume,
    *,
    timestep: int,
    count: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    z_size, y_size, x_size = volume.spatial_shape
    spatial_count = int(z_size * y_size * x_size)
    indices = rng.integers(0, spatial_count, size=int(count), dtype=np.int64)
    x = indices % x_size
    rows = indices // x_size
    y = rows % y_size
    z = rows // y_size
    coords = np.stack(
        [
            x / max(x_size - 1, 1),
            y / max(y_size - 1, 1),
            z / max(z_size - 1, 1),
        ],
        axis=1,
    ).astype(np.float32)
    targets = np.asarray(volume.frame(int(timestep)).reshape(-1)[indices], dtype=np.float32)
    return coords, targets.reshape(-1, 1)


@dataclass
class TemporalFrameSampler:
    time_count: int
    rng: np.random.Generator
    order: np.ndarray
    cursor: int = 0

    @classmethod
    def create(cls, time_count: int, rng: np.random.Generator) -> "TemporalFrameSampler":
        if int(time_count) <= 0:
            raise ValueError("time_count must be positive")
        return cls(
            time_count=int(time_count),
            rng=rng,
            order=rng.permutation(int(time_count)).astype(np.int64),
            cursor=0,
        )

    def next(self) -> int:
        if self.cursor >= self.time_count:
            self.order = self.rng.permutation(self.time_count).astype(np.int64)
            self.cursor = 0
        timestep = int(self.order[self.cursor])
        self.cursor += 1
        return timestep

    def state_dict(self) -> dict:
        return {
            "time_count": int(self.time_count),
            "order": self.order.copy(),
            "cursor": int(self.cursor),
            "rng_state": self.rng.bit_generator.state,
        }

    @classmethod
    def from_state_dict(cls, state: dict) -> "TemporalFrameSampler":
        rng = np.random.default_rng()
        rng.bit_generator.state = state["rng_state"]
        time_count = int(state["time_count"])
        order = np.asarray(state["order"], dtype=np.int64)
        if order.shape != (time_count,):
            raise ValueError("Invalid temporal sampler order in checkpoint")
        cursor = int(state["cursor"])
        if cursor < 0 or cursor > time_count:
            raise ValueError("Invalid temporal sampler cursor in checkpoint")
        return cls(time_count=time_count, rng=rng, order=order, cursor=cursor)
