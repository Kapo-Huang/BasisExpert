from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml


def ensure_parent(path: str | Path) -> Path:
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise TypeError(f"Expected YAML mapping in {path}, got {type(data)!r}")
    return data


def dump_yaml(path: str | Path, payload: dict[str, Any]) -> Path:
    target = ensure_parent(path)
    with target.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)
    return target


def stable_json_dumps(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_payload(payload: Any) -> str:
    return sha256_text(stable_json_dumps(payload))


def resolve_path(path_value: str | None, *, base_dir: str | Path | None = None) -> str | None:
    if path_value is None:
        return None
    path = Path(path_value)
    if path.is_absolute():
        return str(path)
    if base_dir is None:
        return str(path)
    return str((Path(base_dir) / path).resolve())


def resolve_mapping_paths(
    mapping: dict[str, str] | None,
    *,
    base_dir: str | Path | None = None,
) -> dict[str, str] | None:
    if mapping is None:
        return None
    return {str(name): resolve_path(path, base_dir=base_dir) for name, path in mapping.items()}
