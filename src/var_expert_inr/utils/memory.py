from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

import psutil
import torch


class TrainingMemoryTracker:
    """Measure main-process RSS and PyTorch CUDA allocation during data steps."""

    def __init__(
        self,
        device: torch.device | str,
        *,
        sample_interval_seconds: float = 0.01,
    ) -> None:
        if float(sample_interval_seconds) <= 0.0:
            raise ValueError("sample_interval_seconds must be positive")
        self.device = torch.device(device)
        self.sample_interval_seconds = float(sample_interval_seconds)
        self._process = psutil.Process()
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._active = False
        self._confirmed = False
        self._closed = False
        self._data_steps = 0
        self._optimizer_steps = 0
        self._cpu_baseline_bytes: int | None = None
        self._cpu_peak_bytes: int | None = None
        self._cuda_peak_allocated_bytes: int | None = None
        self._thread = threading.Thread(target=self._sample_rss, daemon=True)
        self._thread.start()

    @property
    def _cuda_enabled(self) -> bool:
        return self.device.type == "cuda" and torch.cuda.is_available()

    def _rss_bytes(self) -> int:
        return int(self._process.memory_info().rss)

    def _update_cpu_peak(self, rss_bytes: int) -> None:
        with self._lock:
            if self._active:
                if self._cpu_baseline_bytes is None:
                    self._cpu_baseline_bytes = int(rss_bytes)
                if self._cpu_peak_bytes is None:
                    self._cpu_peak_bytes = int(rss_bytes)
                else:
                    self._cpu_peak_bytes = max(self._cpu_peak_bytes, int(rss_bytes))

    def _sample_rss(self) -> None:
        while not self._stop.wait(self.sample_interval_seconds):
            with self._lock:
                active = self._active
            if active:
                self._update_cpu_peak(self._rss_bytes())

    def start_data_step(self) -> None:
        if self._closed:
            raise RuntimeError("TrainingMemoryTracker is already closed")
        with self._lock:
            if self._active:
                raise RuntimeError("A training data step is already active")
            self._active = True
            self._confirmed = False
        self._update_cpu_peak(self._rss_bytes())
        if self._cuda_enabled:
            torch.cuda.synchronize(self.device)
            torch.cuda.reset_peak_memory_stats(self.device)

    def confirm_data_step(self) -> None:
        with self._lock:
            if not self._active:
                raise RuntimeError("No training data step is active")
            if not self._confirmed:
                self._confirmed = True
                self._data_steps += 1

    def cancel_data_step(self) -> None:
        with self._lock:
            self._active = False
            self._confirmed = False

    def record_optimizer_step(self) -> None:
        with self._lock:
            if not self._active or not self._confirmed:
                raise RuntimeError("Optimizer steps must be recorded inside a data step")
            self._optimizer_steps += 1

    def finish_data_step(self) -> None:
        with self._lock:
            if not self._active:
                return
        cuda_peak: int | None = None
        if self._cuda_enabled:
            torch.cuda.synchronize(self.device)
            cuda_peak = int(torch.cuda.max_memory_allocated(self.device))
        self._update_cpu_peak(self._rss_bytes())
        with self._lock:
            if cuda_peak is not None:
                if self._cuda_peak_allocated_bytes is None:
                    self._cuda_peak_allocated_bytes = cuda_peak
                else:
                    self._cuda_peak_allocated_bytes = max(
                        self._cuda_peak_allocated_bytes,
                        cuda_peak,
                    )
            self._active = False
            self._confirmed = False

    def close(
        self,
        *,
        status: str,
        error_type: str | None = None,
    ) -> dict[str, Any]:
        if status not in {"completed", "failed"}:
            raise ValueError("status must be 'completed' or 'failed'")
        if not self._closed:
            finish_error: Exception | None = None
            try:
                self.finish_data_step()
            except Exception as error:
                finish_error = error
            finally:
                with self._lock:
                    self._active = False
                    self._confirmed = False
                self._stop.set()
                self._thread.join(timeout=max(1.0, self.sample_interval_seconds * 4.0))
                self._closed = True
            if finish_error is not None and status == "completed":
                raise finish_error
        measured = self._data_steps > 0
        baseline = self._cpu_baseline_bytes if measured else None
        peak = self._cpu_peak_bytes if measured else None
        return {
            "schema_version": 1,
            "status": status,
            "scope": "optimization_steps",
            "device": str(self.device),
            "sample_interval_seconds": self.sample_interval_seconds,
            "measured_data_steps": int(self._data_steps),
            "measured_optimizer_steps": int(self._optimizer_steps),
            "cpu_rss_baseline_bytes": baseline,
            "cpu_rss_peak_bytes": peak,
            "cpu_rss_peak_delta_bytes": (
                None
                if baseline is None or peak is None
                else int(max(0, peak - baseline))
            ),
            "cuda_peak_allocated_bytes": (
                self._cuda_peak_allocated_bytes if measured and self._cuda_enabled else None
            ),
            "error_type": error_type,
        }


def write_training_memory(path: str | Path, payload: dict[str, Any]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(f"{target.suffix}.tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(target)
    return target
