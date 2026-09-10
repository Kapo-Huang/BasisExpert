from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


LAYOUT_SCHEMA_VERSION = 2
RENDER_SCHEMA_VERSION = 2
_LOCK_TIMEOUT_SECONDS = 3600.0
_STALE_LOCK_SECONDS = 21600.0


def _json_bytes(payload: Any) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")


def _sha256_payload(payload: Any) -> str:
    return hashlib.sha256(_json_bytes(payload)).hexdigest()


def _safe_component(value: str) -> str:
    text = str(value).strip() or "unknown"
    for character in '<>:"/\\|?*':
        text = text.replace(character, "_")
    return text.rstrip(". ") or "unknown"


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


class _FileLock:
    def __init__(self, path: Path, *, timeout: float = _LOCK_TIMEOUT_SECONDS) -> None:
        self.path = path
        self.timeout = timeout
        self.acquired = False

    def __enter__(self) -> "_FileLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        started = time.monotonic()
        while True:
            try:
                descriptor = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                    handle.write(json.dumps({"pid": os.getpid(), "created": time.time()}))
                self.acquired = True
                return self
            except FileExistsError:
                try:
                    age = time.time() - self.path.stat().st_mtime
                    if age > _STALE_LOCK_SECONDS:
                        self.path.unlink(missing_ok=True)
                        continue
                except FileNotFoundError:
                    continue
                if time.monotonic() - started >= self.timeout:
                    raise TimeoutError(f"Timed out waiting for artifact lock: {self.path}")
                time.sleep(0.1)

    def __exit__(self, exc_type, exc, traceback) -> None:
        if self.acquired:
            self.path.unlink(missing_ok=True)


@dataclass(frozen=True)
class ArtifactSpec:
    path: Path
    key: str
    kind: str
    metadata: dict[str, Any]


@dataclass(frozen=True)
class ArtifactRecord:
    path: Path
    key: str
    kind: str
    cache_hit: bool
    render_info: dict[str, Any]


class ArtifactStore:
    """Content-addressed evaluation render store rooted in one EvalResult tree."""

    def __init__(
        self,
        *,
        repo_root: str | Path,
        result_root: str | Path = "EvalResult",
    ) -> None:
        self.repo_root = Path(repo_root).expanduser().resolve()
        root = Path(result_root).expanduser()
        self.root = (root if root.is_absolute() else self.repo_root / root).resolve()
        self.artifacts_root = self.root / "artifacts"

    def initialize(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        layout = self.root / "layout.json"
        if not layout.is_file():
            with _FileLock(layout.with_suffix(".lock")):
                if not layout.is_file():
                    _atomic_json(
                        layout,
                        {
                            "schema_version": LAYOUT_SCHEMA_VERSION,
                            "artifact_paths": "result_root_relative",
                            "evaluations": "evaluations/<evaluation_id>/<Result-relative-path>",
                            "artifacts": "artifacts/<kind>/...",
                        },
                    )

    def relative(self, path: str | Path) -> str:
        return Path(path).expanduser().resolve().relative_to(self.root).as_posix()

    def reference(self, path: str | Path) -> str:
        resolved = Path(path).expanduser().resolve()
        try:
            return resolved.relative_to(self.root).as_posix()
        except ValueError:
            return str(resolved)

    def resolve(self, reference: str | Path) -> Path:
        path = Path(reference).expanduser()
        return path.resolve() if path.is_absolute() else (self.root / path).resolve()

    @staticmethod
    def _file_sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            while chunk := handle.read(8 * 1024 * 1024):
                digest.update(chunk)
        return digest.hexdigest()

    def content_fingerprint(self, path: str | Path) -> dict[str, Any]:
        """Return a path-independent content fingerprint with a stat-based digest cache."""
        self.initialize()
        resolved = Path(path).expanduser().resolve()
        if not resolved.exists():
            return {"missing": True}
        cache_name = hashlib.sha256(str(resolved).encode("utf-8")).hexdigest() + ".json"
        cache_path = self.artifacts_root / ".fingerprints" / cache_name
        if resolved.is_file():
            stat = resolved.stat()
            signature: dict[str, Any] = {
                "kind": "file",
                "size": int(stat.st_size),
                "mtime_ns": int(stat.st_mtime_ns),
            }
        else:
            entries = []
            for item in sorted(candidate for candidate in resolved.rglob("*") if candidate.is_file()):
                stat = item.stat()
                entries.append(
                    {
                        "path": item.relative_to(resolved).as_posix(),
                        "size": int(stat.st_size),
                        "mtime_ns": int(stat.st_mtime_ns),
                    }
                )
            signature = {"kind": "directory", "entries": entries}
        try:
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            if cached.get("signature") == signature:
                return dict(cached["fingerprint"])
        except (FileNotFoundError, OSError, ValueError, TypeError, KeyError):
            pass

        lock = cache_path.with_suffix(".lock")
        with _FileLock(lock):
            try:
                cached = json.loads(cache_path.read_text(encoding="utf-8"))
                if cached.get("signature") == signature:
                    return dict(cached["fingerprint"])
            except (FileNotFoundError, OSError, ValueError, TypeError, KeyError):
                pass
            if resolved.is_file():
                fingerprint = {"kind": "file", "size": signature["size"], "sha256": self._file_sha256(resolved)}
            else:
                content_entries = [
                    {
                        "path": entry["path"],
                        "size": entry["size"],
                        "sha256": self._file_sha256(resolved / entry["path"]),
                    }
                    for entry in signature["entries"]
                ]
                fingerprint = {
                    "kind": "directory",
                    "entries": content_entries,
                    "sha256": _sha256_payload(content_entries),
                }
            _atomic_json(cache_path, {"signature": signature, "fingerprint": fingerprint})
            return fingerprint

    def _render_spec(
        self,
        profile: dict[str, Any],
        *,
        target: str,
        timestep: int,
    ) -> dict[str, Any]:
        normalized: dict[str, Any] = {}
        asset_keys = {
            "mesh_path",
            "mesh_path_template",
            "vertices_path",
            "vertices_path_template",
            "cells_path",
            "cells_path_template",
        }
        for key, value in profile.items():
            if key == "_path" or key == "mesh_source_hint":
                continue
            if key in asset_keys and value:
                formatted = str(value).format(time_index=timestep, timestep=timestep, t=timestep)
                normalized[key] = self.content_fingerprint(formatted)
            else:
                normalized[key] = value
        preset = None
        if str(profile.get("renderer", "")).lower() == "volume":
            try:
                from volume_vis import load_preset

                preset_name = str((profile.get("target_presets") or {}).get(target, target))
                loaded = load_preset(preset_name, namespace=str(profile.get("preset_namespace", "ionization")))
                preset = {
                    "transfer_function": loaded.transfer_function,
                    "viewport": loaded.viewport,
                }
            except ImportError:
                preset = {"unavailable": True}
        return {
            "render_schema_version": RENDER_SCHEMA_VERSION,
            "profile": normalized,
            "preset": preset,
            "target": str(target),
        }

    def ground_truth_spec(
        self,
        *,
        dataset: str,
        target: str,
        timestep: int,
        ground_truth_path: str | Path,
        profile: dict[str, Any],
        ground_truth_fingerprint: dict[str, Any] | None = None,
    ) -> ArtifactSpec:
        gt = (
            self.content_fingerprint(ground_truth_path)
            if ground_truth_fingerprint is None else dict(ground_truth_fingerprint)
        )
        render = self._render_spec(profile, target=target, timestep=timestep)
        key = _sha256_payload({"ground_truth": gt, "render": render})
        path = (
            self.artifacts_root
            / "ground_truth"
            / _safe_component(dataset)
            / _safe_component(target)
            / key
            / f"gt_t{int(timestep):04d}.png"
        )
        return ArtifactSpec(path, key, "ground_truth", {"dataset": dataset, "target": target, "ground_truth": gt, "render": render})

    def prediction_spec(
        self,
        *,
        dataset: str,
        model: str,
        target: str,
        timestep: int,
        source_path: str | Path,
        profile: dict[str, Any],
        ground_truth_fingerprint: dict[str, Any] | None,
        source_fingerprint: dict[str, Any] | None = None,
    ) -> ArtifactSpec:
        source = (
            self.content_fingerprint(source_path)
            if source_fingerprint is None else dict(source_fingerprint)
        )
        source_key = _sha256_payload(source)
        render = self._render_spec(profile, target=target, timestep=timestep)
        render_key = _sha256_payload({"render": render, "ground_truth": ground_truth_fingerprint})
        path = (
            self.artifacts_root
            / "prediction"
            / _safe_component(dataset)
            / _safe_component(model)
            / source_key
            / _safe_component(target)
            / render_key
            / f"pred_t{int(timestep):04d}.png"
        )
        return ArtifactSpec(path, render_key, "prediction", {"dataset": dataset, "model": model, "target": target, "source": source, "render": render, "ground_truth": ground_truth_fingerprint})

    def error_spec(
        self,
        *,
        dataset: str,
        model: str,
        target: str,
        timestep: int,
        source_path: str | Path,
        ground_truth_fingerprint: dict[str, Any],
        profile: dict[str, Any],
        error_vmin: float,
        error_vmax: float,
        source_fingerprint: dict[str, Any] | None = None,
    ) -> ArtifactSpec:
        source = (
            self.content_fingerprint(source_path)
            if source_fingerprint is None else dict(source_fingerprint)
        )
        comparison_key = _sha256_payload({"source": source, "ground_truth": ground_truth_fingerprint})
        render = self._render_spec(profile, target=target, timestep=timestep)
        render_key = _sha256_payload({"render": render, "error_vmin": error_vmin, "error_vmax": error_vmax})
        path = (
            self.artifacts_root
            / "error"
            / _safe_component(dataset)
            / _safe_component(model)
            / comparison_key
            / _safe_component(target)
            / render_key
            / f"error_t{int(timestep):04d}.png"
        )
        return ArtifactSpec(path, render_key, "error", {"dataset": dataset, "model": model, "target": target, "source": source, "ground_truth": ground_truth_fingerprint, "render": render, "error_vmin": error_vmin, "error_vmax": error_vmax})

    def materialize(
        self,
        spec: ArtifactSpec,
        producer: Callable[[Path], dict[str, Any] | None],
        *,
        overwrite: bool = False,
    ) -> ArtifactRecord:
        self.initialize()
        if spec.path.is_file() and not overwrite:
            return ArtifactRecord(spec.path, spec.key, spec.kind, True, {})
        lock_path = spec.path.with_suffix(spec.path.suffix + ".lock")
        with _FileLock(lock_path):
            if spec.path.is_file() and not overwrite:
                return ArtifactRecord(spec.path, spec.key, spec.kind, True, {})
            spec.path.parent.mkdir(parents=True, exist_ok=True)
            temporary = spec.path.with_name(
                f".{spec.path.stem}.{os.getpid()}.{uuid.uuid4().hex}.tmp{spec.path.suffix}"
            )
            try:
                info = dict(producer(temporary) or {})
                if not temporary.is_file():
                    raise RuntimeError(f"Artifact producer did not create its output: {temporary}")
                os.replace(temporary, spec.path)
            finally:
                temporary.unlink(missing_ok=True)
            info["path"] = self.relative(spec.path)
            self._record_frame(spec, info)
            return ArtifactRecord(spec.path, spec.key, spec.kind, False, info)

    def import_existing(
        self,
        spec: ArtifactSpec,
        source: str | Path,
        *,
        allow_semantic_variant: bool = False,
    ) -> ArtifactRecord:
        """Import a verified legacy frame without rendering or duplicating file data."""
        self.initialize()
        resolved_source = Path(source).expanduser().resolve()
        if not resolved_source.is_file():
            raise FileNotFoundError(f"Legacy artifact does not exist: {resolved_source}")
        lock_path = spec.path.with_suffix(spec.path.suffix + ".lock")
        cache_hit = spec.path.is_file()
        with _FileLock(lock_path):
            semantic_variant = False
            source_digest = None
            destination_digest = None
            if spec.path.is_file():
                if not os.path.samefile(resolved_source, spec.path):
                    source_digest = self._file_sha256(resolved_source)
                    destination_digest = self._file_sha256(spec.path)
                    if source_digest != destination_digest:
                        if not allow_semantic_variant:
                            raise ValueError(
                                f"Legacy artifact conflicts with existing cache entry: {resolved_source} -> {spec.path}"
                            )
                        semantic_variant = True
            else:
                spec.path.parent.mkdir(parents=True, exist_ok=True)
                try:
                    os.link(resolved_source, spec.path)
                except OSError:
                    shutil.copy2(resolved_source, spec.path)
                cache_hit = False
            info = {
                "imported": True,
                "source_size": resolved_source.stat().st_size,
                "semantic_variant": semantic_variant,
            }
            if semantic_variant:
                info.update(
                    {
                        "legacy_source_sha256": source_digest,
                        "canonical_sha256": destination_digest,
                    }
                )
            self._record_frame(spec, info)
        return ArtifactRecord(spec.path, spec.key, spec.kind, cache_hit, info)

    def _record_frame(self, spec: ArtifactSpec, render_info: dict[str, Any]) -> None:
        manifest_path = spec.path.parent / "manifest.json"
        with _FileLock(manifest_path.with_suffix(".lock")):
            try:
                payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (FileNotFoundError, OSError, ValueError, TypeError):
                payload = {
                    "schema_version": LAYOUT_SCHEMA_VERSION,
                    "artifact_kind": spec.kind,
                    "artifact_key": spec.key,
                    "inputs": spec.metadata,
                    "frames": {},
                }
            payload.setdefault("frames", {})[spec.path.name] = {
                "path": self.relative(spec.path),
                "size": spec.path.stat().st_size,
                "render_info": render_info,
                "completed_at": time.time(),
            }
            _atomic_json(manifest_path, payload)

    def describe(self, record: ArtifactRecord) -> dict[str, Any]:
        return {
            "artifact_key": record.key,
            "artifact_kind": record.kind,
            "cache_hit": record.cache_hit,
            "path": self.relative(record.path),
            **record.render_info,
        }


def resolve_artifact_reference(result_root: str | Path, reference: str | Path) -> Path:
    root = Path(result_root).expanduser().resolve()
    path = Path(reference).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()
