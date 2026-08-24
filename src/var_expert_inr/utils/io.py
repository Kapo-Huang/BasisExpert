from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT_PLACEHOLDER = "${REPO_ROOT}"
DATASETS_ROOT_PLACEHOLDER = "${DATASETS_ROOT}"
RUNS_ROOT_PLACEHOLDER = "${RUNS_ROOT}"
DATASET_ROOT_PLACEHOLDERS = {
    "${REDSEA_ROOT}": ("RedSea", Path("data/Mesh/RedSea")),
    "${KATRINA_ROOT}": ("Katrina", Path("data/Mesh/Katrina")),
    "${IONIZATION_ROOT}": ("Ionization", Path("data/Volume/Ionization")),
    "${COMBUSTION_ROOT}": ("Combustion", Path("data/Volume/Combustion")),
}


def _dataset_root(
    placeholder: str,
    *,
    base_dir: str | Path | None,
) -> Path:
    dataset_name, original_relative = DATASET_ROOT_PLACEHOLDERS[placeholder]
    override = os.environ.get(f"{dataset_name.upper()}_ROOT")
    if override:
        return Path(override).expanduser()

    server_env = os.environ.get("SERVER_ENV", "original").strip().lower()
    if server_env == "autodl":
        autodl_root = Path(os.environ.get("AUTODL_DATA_ROOT", "/root/autodl-tmp"))
        return autodl_root / dataset_name
    if server_env != "original":
        raise ValueError(
            f"Unsupported SERVER_ENV={server_env!r}; expected 'original' or 'autodl'"
        )

    repo_root = find_repo_root(base_dir)
    if original_relative is not None:
        return repo_root / original_relative
    configured_root = os.environ.get("DATASETS_ROOT")
    datasets_root = Path(configured_root) if configured_root else repo_root.parent.parent / "Datasets"
    return datasets_root / dataset_name


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


def _runs_root(*, base_dir: str | Path | None) -> Path:
    configured_root = os.environ.get("RUNS_ROOT")
    if configured_root:
        return Path(configured_root).expanduser()

    server_env = os.environ.get("SERVER_ENV", "original").strip().lower()
    if server_env == "autodl":
        return Path("/root/autodl-tmp/runs")
    if server_env == "original":
        return find_repo_root(base_dir) / "runs"
    raise ValueError(
        f"Unsupported SERVER_ENV={server_env!r}; expected 'original' or 'autodl'"
    )


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
        text == RUNS_ROOT_PLACEHOLDER
        or text.startswith(f"{RUNS_ROOT_PLACEHOLDER}/")
        or text.startswith(f"{RUNS_ROOT_PLACEHOLDER}\\")
    ):
        suffix = text[len(RUNS_ROOT_PLACEHOLDER) :].lstrip("/\\")
        return str((_runs_root(base_dir=base_dir) / suffix).resolve())
    if (
        text == REPO_ROOT_PLACEHOLDER
        or text.startswith(f"{REPO_ROOT_PLACEHOLDER}/")
        or text.startswith(f"{REPO_ROOT_PLACEHOLDER}\\")
    ):
        suffix = text[len(REPO_ROOT_PLACEHOLDER) :].lstrip("/\\")
        return str((find_repo_root(base_dir) / suffix).resolve())
    if (
        text == DATASETS_ROOT_PLACEHOLDER
        or text.startswith(f"{DATASETS_ROOT_PLACEHOLDER}/")
        or text.startswith(f"{DATASETS_ROOT_PLACEHOLDER}\\")
    ):
        suffix = text[len(DATASETS_ROOT_PLACEHOLDER) :].lstrip("/\\")
        configured_root = os.environ.get("DATASETS_ROOT")
        datasets_root = (
            Path(configured_root)
            if configured_root
            else find_repo_root(base_dir).parent.parent / "Datasets"
        )
        return str((datasets_root / suffix).resolve())
    for placeholder in DATASET_ROOT_PLACEHOLDERS:
        if (
            text == placeholder
            or text.startswith(f"{placeholder}/")
            or text.startswith(f"{placeholder}\\")
        ):
            suffix = text[len(placeholder) :].lstrip("/\\")
            return str((_dataset_root(placeholder, base_dir=base_dir) / suffix).resolve())
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
