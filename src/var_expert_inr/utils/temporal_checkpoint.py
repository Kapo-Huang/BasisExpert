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

    def __init__(
        self,
        path: str | Path,
        *,
        metadata: dict[str, Any],
        compact: bool = False,
    ) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.temporary_path = self.path.with_name(f"{self.path.name}.tmp")
        self.metadata = dict(metadata)
        self.time_indices: list[int] = []
        self._compact = bool(compact)
        self._chunk_size = 32 if self._compact else 1
        self._pending: list[tuple[int, dict[str, Any]]] = []
        self._chunks: list[dict[str, Any]] = []
        archive_kwargs: dict[str, Any] = {
            "mode": "w",
            "compression": zipfile.ZIP_DEFLATED if self._compact else zipfile.ZIP_STORED,
            "allowZip64": True,
        }
        if self._compact:
            archive_kwargs["compresslevel"] = 9
        self._archive = zipfile.ZipFile(
            self.temporary_path,
            **archive_kwargs,
        )
        self._finalized = False

    def _serialize(self, payload: Any) -> bytes:
        buffer = io.BytesIO()
        if self._compact:
            torch.save(payload, buffer, _use_new_zipfile_serialization=False)
        else:
            torch.save(payload, buffer)
        return buffer.getvalue()

    def _flush_chunk(self) -> None:
        if not self._pending:
            return
        chunk_index = len(self._chunks)
        member = f"chunks/c{chunk_index:04d}.pt"
        indices = [int(index) for index, _ in self._pending]
        payloads = [payload for _, payload in self._pending]
        self._archive.writestr(
            member,
            self._serialize({"time_indices": indices, "payloads": payloads}),
        )
        self._chunks.append({"member": member, "time_indices": indices})
        self._pending.clear()

    def write_timestep(self, time_index: int, payload: dict[str, Any]) -> None:
        index = int(time_index)
        if index in self.time_indices:
            raise ValueError(f"Duplicate timestep in checkpoint bundle: {index}")
        self.time_indices.append(index)
        if not self._compact:
            self._archive.writestr(_member_name(index), self._serialize(payload))
            return
        self._pending.append((index, payload))
        if len(self._pending) >= self._chunk_size:
            self._flush_chunk()

    def finalize(self) -> Path:
        if self._finalized:
            return self.path
        self._flush_chunk()
        metadata = {**self.metadata, "time_indices": list(self.time_indices)}
        if self._compact:
            metadata["storage"] = {
                "kind": "chunked",
                "chunk_size": int(self._chunk_size),
                "chunks": list(self._chunks),
            }
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
        self._chunk_locations: dict[int, tuple[str, int]] = {}
        storage = self.metadata.get("storage")
        if isinstance(storage, dict) and storage.get("kind") == "chunked":
            for chunk in storage.get("chunks", []):
                member = str(chunk["member"])
                for position, index in enumerate(chunk.get("time_indices", [])):
                    resolved = int(index)
                    if resolved in self._chunk_locations:
                        self._archive.close()
                        raise ValueError(f"Duplicate chunked timestep: {resolved}")
                    self._chunk_locations[resolved] = (member, int(position))
        self._cached_chunk_member: str | None = None
        self._cached_chunk_payloads: list[Any] | None = None

    @property
    def time_indices(self) -> tuple[int, ...]:
        return tuple(int(value) for value in self.metadata.get("time_indices", []))

    def load_timestep(self, time_index: int, *, map_location: Any = "cpu") -> dict[str, Any]:
        index = int(time_index)
        chunk_location = self._chunk_locations.get(index)
        member = chunk_location[0] if chunk_location is not None else _member_name(index)
        if chunk_location is not None and self._cached_chunk_member == member:
            payloads = self._cached_chunk_payloads
        else:
            try:
                serialized = self._archive.read(member)
            except KeyError as exc:
                raise KeyError(f"Checkpoint does not contain timestep {index}") from exc
            buffer = io.BytesIO(serialized)
            try:
                loaded = torch.load(buffer, map_location=map_location, weights_only=False)
            except TypeError:
                loaded = torch.load(buffer, map_location=map_location)
            if chunk_location is None:
                payload = loaded
                payloads = None
            else:
                if not isinstance(loaded, dict) or not isinstance(loaded.get("payloads"), list):
                    raise ValueError(f"Invalid chunked checkpoint member: {member}")
                payloads = loaded["payloads"]
                self._cached_chunk_member = member
                self._cached_chunk_payloads = payloads
        if chunk_location is not None:
            if payloads is None or chunk_location[1] >= len(payloads):
                raise ValueError(f"Invalid chunk position for timestep {index}")
            payload = payloads[chunk_location[1]]
        if not isinstance(payload, dict):
            raise ValueError(f"Invalid timestep payload for {index}")
        return payload

    def close(self) -> None:
        self._cached_chunk_member = None
        self._cached_chunk_payloads = None
        self._archive.close()

    def __enter__(self) -> "TemporalCheckpointReader":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()
