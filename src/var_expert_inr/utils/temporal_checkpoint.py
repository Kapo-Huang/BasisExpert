from __future__ import annotations

import io
import json
import os
import zipfile
from pathlib import Path
from typing import Any

import torch


METADATA_MEMBER = "metadata.json"


def _member_name(time_index: int) -> str:
    return f"timesteps/t{int(time_index):04d}.pt"


class TemporalCheckpointWriter:
    """Stream timestep payloads into one atomically published checkpoint."""

    def __init__(self, path: str | Path, *, metadata: dict[str, Any]) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.temporary_path = self.path.with_name(f"{self.path.name}.tmp")
        self.metadata = dict(metadata)
        self.time_indices: list[int] = []
        self._archive = zipfile.ZipFile(
            self.temporary_path,
            mode="w",
            compression=zipfile.ZIP_STORED,
            allowZip64=True,
        )
        self._finalized = False

    def write_timestep(self, time_index: int, payload: dict[str, Any]) -> None:
        index = int(time_index)
        if index in self.time_indices:
            raise ValueError(f"Duplicate timestep in checkpoint bundle: {index}")
        buffer = io.BytesIO()
        torch.save(payload, buffer)
        self._archive.writestr(_member_name(index), buffer.getvalue())
        self.time_indices.append(index)

    def finalize(self) -> Path:
        if self._finalized:
            return self.path
        metadata = {**self.metadata, "time_indices": list(self.time_indices)}
        self._archive.writestr(
            METADATA_MEMBER,
            json.dumps(metadata, ensure_ascii=True, separators=(",", ":")),
        )
        self._archive.close()
        os.replace(self.temporary_path, self.path)
        self._finalized = True
        return self.path

    def abort(self) -> None:
        if self._archive.fp:
            self._archive.close()
        if not self._finalized:
            self.temporary_path.unlink(missing_ok=True)

    def __enter__(self) -> "TemporalCheckpointWriter":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        if exc_type is not None:
            self.abort()
            return
        try:
            self.finalize()
        except Exception:
            self.abort()
            raise


class TemporalCheckpointReader:
    def __init__(self, path: str | Path, *, expected_format: str) -> None:
        self.path = Path(path)
        if not self.path.is_file():
            raise FileNotFoundError(self.path)
        self._archive = zipfile.ZipFile(self.path, mode="r")
        try:
            self.metadata = json.loads(self._archive.read(METADATA_MEMBER).decode("utf-8"))
        except Exception:
            self._archive.close()
            raise
        if self.metadata.get("format") != expected_format:
            self._archive.close()
            raise ValueError(
                f"Unsupported temporal inference checkpoint: {self.metadata.get('format')!r}"
            )

    @property
    def time_indices(self) -> tuple[int, ...]:
        return tuple(int(value) for value in self.metadata.get("time_indices", []))

    def load_timestep(self, time_index: int, *, map_location: Any = "cpu") -> dict[str, Any]:
        member = _member_name(int(time_index))
        try:
            serialized = self._archive.read(member)
        except KeyError as exc:
            raise KeyError(f"Checkpoint does not contain timestep {int(time_index)}") from exc
        buffer = io.BytesIO(serialized)
        try:
            payload = torch.load(buffer, map_location=map_location, weights_only=False)
        except TypeError:
            payload = torch.load(buffer, map_location=map_location)
        if not isinstance(payload, dict):
            raise ValueError(f"Invalid timestep payload for {int(time_index)}")
        return payload

    def close(self) -> None:
        self._archive.close()

    def __enter__(self) -> "TemporalCheckpointReader":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()
