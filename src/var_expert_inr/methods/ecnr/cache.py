from __future__ import annotations

import gc
import logging
import shutil
import time
import weakref
from pathlib import Path

import numpy as np


logger = logging.getLogger(__name__)


def _close_memmap(array: np.ndarray | None) -> None:
    current = array
    seen: set[int] = set()
    while isinstance(current, np.ndarray) and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, np.memmap):
            try:
                current.flush()
            except (ValueError, OSError):
                pass
            mapping = getattr(current, "_mmap", None)
            if mapping is not None and not mapping.closed:
                mapping.close()
            return
        current = getattr(current, "base", None)


class CacheWorkspace:
    """Owns transient ECNR files and enforces run-local cleanup."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.peak_bytes = 0
        self.released_bytes = 0
        self.cleanup_seconds = 0.0
        self._arrays: dict[Path, list[weakref.ReferenceType[np.ndarray]]] = {}

    def _checked(self, path: str | Path) -> Path:
        resolved = Path(path).resolve()
        if not resolved.is_relative_to(self.root):
            raise ValueError(f"Refusing to manage path outside ECNR cache: {resolved}")
        return resolved

    def current_bytes(self) -> int:
        if not self.root.exists():
            return 0
        return sum(path.stat().st_size for path in self.root.rglob("*") if path.is_file())

    def register(self, path: str | Path, array: np.ndarray | None = None) -> Path:
        resolved = self._checked(path)
        if array is not None:
            self._arrays.setdefault(resolved, []).append(weakref.ref(array))
        self.peak_bytes = max(self.peak_bytes, self.current_bytes())
        return resolved

    def release(
        self,
        *paths: str | Path,
        arrays: tuple[np.ndarray | None, ...] = (),
        label: str,
    ) -> int:
        for array in arrays:
            _close_memmap(array)
        released = 0
        files = 0
        for path in paths:
            resolved = self._checked(path)
            for reference in self._arrays.pop(resolved, []):
                _close_memmap(reference())
            if resolved.is_file():
                released += resolved.stat().st_size
                resolved.unlink()
                files += 1
        self.released_bytes += released
        current = self.current_bytes()
        self.peak_bytes = max(self.peak_bytes, current + released)
        logger.info(
            "ECNR cache release stage=%s files=%d released_gib=%.3f current_gib=%.3f peak_gib=%.3f",
            label,
            files,
            released / float(1 << 30),
            current / float(1 << 30),
            self.peak_bytes / float(1 << 30),
        )
        return released

    def cleanup(self) -> None:
        started = time.perf_counter()
        for references in self._arrays.values():
            for reference in references:
                _close_memmap(reference())
        self._arrays.clear()
        gc.collect()
        before = self.current_bytes()
        if self.root.exists():
            shutil.rmtree(self.root)
        self.released_bytes += before
        self.cleanup_seconds += time.perf_counter() - started
        logger.info(
            "ECNR cache cleanup released_gib=%.3f peak_gib=%.3f seconds=%.3f final_bytes=0",
            before / float(1 << 30),
            self.peak_bytes / float(1 << 30),
            self.cleanup_seconds,
        )

    def metrics(self) -> dict[str, int | float]:
        return {
            "peak_bytes": int(self.peak_bytes),
            "released_bytes": int(self.released_bytes),
            "final_bytes": int(self.current_bytes()),
            "cleanup_seconds": float(self.cleanup_seconds),
        }
