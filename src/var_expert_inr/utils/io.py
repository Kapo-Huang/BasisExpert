from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT_PLACEHOLDER = "${REPO_ROOT}"


def find_repo_root(start: str | Path | None = None) -> Path:
    """Find the project root used by generated configs.

    Generated YAML files can refer to paths with ``${REPO_ROOT}``.  Prefer a
    root discovered from the config location, but fall back to this module's
    source tree so callers outside the repo cwd still resolve consistently.
    """

    search_starts: list[Path] = []
    if start is not None:
        search_starts.append(Path(start).resolve())
    search_starts.append(Path(__file__).resolve())

    for candidate_start in search_starts:
        current = candidate_start if candidate_start.is_dir() else candidate_start.parent
        for parent in (current, *current.parents):
            if (parent / "src" / "var_expert_inr").exists() and (parent / "configs").exists():
                return parent
            if (parent / "pyproject.toml").exists() and (parent / "src").exists():
                return parent
    return Path(__file__).resolve().parents[3]


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
    text = str(path_value)
    if (
        text == REPO_ROOT_PLACEHOLDER
        or text.startswith(f"{REPO_ROOT_PLACEHOLDER}/")
        or text.startswith(f"{REPO_ROOT_PLACEHOLDER}\\")
    ):
        suffix = text[len(REPO_ROOT_PLACEHOLDER) :].lstrip("/\\")
        return str((find_repo_root(base_dir) / suffix).resolve())
    path = Path(text)
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
