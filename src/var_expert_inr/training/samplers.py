from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from torch.utils.data import RandomSampler, Sampler, Subset


class TimeStratifiedSampler(Sampler[int]):
    def __init__(
        self,
        dataset,
        *,
        total_samples_budget: int,
        generator=None,
    ):
        self.dataset = dataset
        self.generator = generator
        self.total_samples_budget = int(total_samples_budget)
        base = dataset.dataset if isinstance(dataset, Subset) else dataset
        if base.meta.volume_shape is None:
            raise ValueError("time_stratified sampler requires a volume dataset")
        self.volume_shape = base.meta.volume_shape
        self._per_t_indices = None
        if isinstance(dataset, Subset):
            indices = np.asarray(dataset.indices, dtype=np.int64)
            V = int(self.volume_shape.X) * int(self.volume_shape.Y) * int(self.volume_shape.Z)
            t = indices // V
            self._per_t_indices = [indices[t == ti] for ti in range(int(self.volume_shape.T))]
        self.samples_per_timestep = self.total_samples_budget // int(self.volume_shape.T)
        self.total_samples = self.samples_per_timestep * int(self.volume_shape.T)

    def __iter__(self):
        V = int(self.volume_shape.X) * int(self.volume_shape.Y) * int(self.volume_shape.Z)
        T = int(self.volume_shape.T)
        active_tensor = torch.arange(T, dtype=torch.long)
        remaining = int(self.total_samples)
        chunk_size = 1_000_000
        while remaining > 0:
            cur = min(chunk_size, remaining)
            t_choice = torch.randint(0, T, (cur,), generator=self.generator)
            t_vals = active_tensor[t_choice]
            if self._per_t_indices is None:
                offsets = torch.randint(0, V, (cur,), generator=self.generator)
                indices = t_vals * V + offsets
                for idx in indices.tolist():
                    yield int(idx)
            else:
                t_vals_np = t_vals.cpu().numpy()
                out = np.empty(cur, dtype=np.int64)
                for t in np.unique(t_vals_np):
                    pos = np.where(t_vals_np == t)[0]
                    pool = self._per_t_indices[int(t)]
                    picks = torch.randint(0, len(pool), (len(pos),), generator=self.generator).cpu().numpy()
                    out[pos] = pool[picks]
                for idx in out.tolist():
                    yield int(idx)
            remaining -= cur

    def __len__(self) -> int:
        return int(self.total_samples)


def build_train_sampler(dataset, cfg):
    if cfg.sampler == "uniform_random":
        return None
    if cfg.sampler == "budgeted_random":
        if cfg.batches_per_epoch_budget <= 0:
            raise ValueError("budgeted_random sampler requires batches_per_epoch_budget > 0")
        return RandomSampler(
            dataset,
            replacement=True,
            num_samples=int(cfg.batches_per_epoch_budget) * int(cfg.batch_size),
        )
    if cfg.sampler == "time_stratified":
        if cfg.batches_per_epoch_budget <= 0:
            raise ValueError("time_stratified sampler requires batches_per_epoch_budget > 0")
        generator = torch.Generator().manual_seed(int(cfg.seed))
        return TimeStratifiedSampler(
            dataset,
            total_samples_budget=int(cfg.batches_per_epoch_budget) * int(cfg.batch_size),
            generator=generator,
        )
    raise ValueError(f"Unsupported sampler: {cfg.sampler}")
