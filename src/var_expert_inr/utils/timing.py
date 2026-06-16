from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import torch

logger = logging.getLogger(__name__)


@dataclass
class TimingBreakdown:
    data: float = 0.0
    transfer: float = 0.0
    training: float = 0.0
    val: float = 0.0
    psnr: float = 0.0
    others: float = 0.0

    def tracked_total(self) -> float:
        return (
            float(self.data)
            + float(self.transfer)
            + float(self.training)
            + float(self.val)
            + float(self.psnr)
            + float(self.others)
        )


def maybe_sync_timing(device: torch.device, sync_cuda: bool) -> None:
    if not sync_cuda:
        return
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)


def timing_start(device: torch.device, sync_cuda: bool) -> float:
    maybe_sync_timing(device, sync_cuda)
    return time.perf_counter()


def timing_elapsed(start_time: float, device: torch.device, sync_cuda: bool) -> float:
    maybe_sync_timing(device, sync_cuda)
    return time.perf_counter() - start_time


def log_step_timing_window(
    *,
    prefix: str,
    epoch: int,
    total_epochs: int,
    step_start: int,
    step_end: int,
    elapsed_seconds: float,
    data_seconds: float,
    transfer_seconds: float,
    training_seconds: float,
    other_seconds: float,
) -> None:
    step_count = max(int(step_end) - int(step_start) + 1, 1)
    avg_step = float(elapsed_seconds) / float(step_count)
    steps_per_sec = float(step_count) / max(float(elapsed_seconds), 1e-12)
    logger.info(
        "%s timing window epoch %s/%s steps %s-%s: total=%.2fs data=%.2fs transfer=%.2fs training=%.2fs others=%.2fs avg_step=%.4fs steps_per_sec=%.2f",
        prefix,
        epoch,
        total_epochs,
        step_start,
        step_end,
        elapsed_seconds,
        data_seconds,
        transfer_seconds,
        training_seconds,
        other_seconds,
        avg_step,
        steps_per_sec,
    )


def log_epoch_timing(
    *,
    prefix: str,
    epoch: int,
    total_epochs: int,
    total_seconds: float,
    breakdown: TimingBreakdown,
) -> None:
    logger.info(
        "%s epoch %s/%s timing(total): total=%.2fs data=%.2fs transfer=%.2fs training=%.2fs val=%.2fs psnr=%.2fs others=%.2fs",
        prefix,
        epoch,
        total_epochs,
        total_seconds,
        breakdown.data,
        breakdown.transfer,
        breakdown.training,
        breakdown.val,
        breakdown.psnr,
        breakdown.others,
    )
