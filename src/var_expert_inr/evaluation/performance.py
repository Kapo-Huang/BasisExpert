from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass
from typing import Any

import torch


def _rss_bytes() -> int:
    try:
        import psutil

        return int(psutil.Process(os.getpid()).memory_info().rss)
    except ImportError:
        return 0


def synchronize_cuda(device: torch.device | str | None) -> None:
    if device is None:
        return
    resolved = torch.device(device)
    if resolved.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(resolved)


@dataclass
class DecodeMeasurement:
    device: torch.device | str | None = None
    sample_interval_seconds: float = 0.01

    def __post_init__(self) -> None:
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.started_at = 0.0
        self.baseline_rss_bytes = 0
        self.peak_rss_bytes = 0

    def __enter__(self) -> "DecodeMeasurement":
        synchronize_cuda(self.device)
        resolved = torch.device(self.device) if self.device is not None else None
        if resolved is not None and resolved.type == "cuda" and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(resolved)
        self.baseline_rss_bytes = _rss_bytes()
        self.peak_rss_bytes = self.baseline_rss_bytes
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self._thread.start()
        self.started_at = time.perf_counter()
        return self

    def _sample(self) -> None:
        while not self._stop.wait(self.sample_interval_seconds):
            self.peak_rss_bytes = max(self.peak_rss_bytes, _rss_bytes())

    def __exit__(self, *_: object) -> None:
        synchronize_cuda(self.device)
        self.elapsed_seconds = float(time.perf_counter() - self.started_at)
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, self.sample_interval_seconds * 4))
        self.peak_rss_bytes = max(self.peak_rss_bytes, _rss_bytes())

    def as_dict(self) -> dict[str, Any]:
        resolved = torch.device(self.device) if self.device is not None else None
        cuda_allocated = cuda_reserved = None
        if resolved is not None and resolved.type == "cuda" and torch.cuda.is_available():
            cuda_allocated = int(torch.cuda.max_memory_allocated(resolved))
            cuda_reserved = int(torch.cuda.max_memory_reserved(resolved))
        return {
            "cpu_rss_baseline_bytes": int(self.baseline_rss_bytes),
            "cpu_rss_peak_bytes": int(self.peak_rss_bytes),
            "cpu_rss_peak_delta_bytes": int(max(0, self.peak_rss_bytes - self.baseline_rss_bytes)),
            "cuda_peak_allocated_bytes": cuda_allocated,
            "cuda_peak_reserved_bytes": cuda_reserved,
        }


def combine_memory_samples(samples: list[dict[str, Any]]) -> dict[str, Any]:
    if not samples:
        return {
            "cpu_rss_baseline_bytes": 0,
            "cpu_rss_peak_bytes": 0,
            "cpu_rss_peak_delta_bytes": 0,
            "cuda_peak_allocated_bytes": None,
            "cuda_peak_reserved_bytes": None,
        }
    baseline = int(samples[0]["cpu_rss_baseline_bytes"])
    peak = max(int(item["cpu_rss_peak_bytes"]) for item in samples)
    allocated = [item["cuda_peak_allocated_bytes"] for item in samples if item["cuda_peak_allocated_bytes"] is not None]
    reserved = [item["cuda_peak_reserved_bytes"] for item in samples if item["cuda_peak_reserved_bytes"] is not None]
    return {
        "cpu_rss_baseline_bytes": baseline,
        "cpu_rss_peak_bytes": peak,
        "cpu_rss_peak_delta_bytes": max(0, peak - baseline),
        "cuda_peak_allocated_bytes": max(allocated) if allocated else None,
        "cuda_peak_reserved_bytes": max(reserved) if reserved else None,
    }
