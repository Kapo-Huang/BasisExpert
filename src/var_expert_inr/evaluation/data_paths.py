from __future__ import annotations

import copy
import os
from dataclasses import replace
from pathlib import Path
from typing import Any

from ..utils.io import (
    DATASET_ROOT_DEFINITIONS,
    canonical_dataset_name,
    dataset_root_placeholder,
    resolve_path,
)


# Paths stored by both the current schema and historical standalone-method
# configs. Keeping this inventory here prevents evaluation entrypoints from
# gradually developing different portability behaviour.
SCALAR_DATA_PATH_FIELDS = (
    "target_path",
    "coords_path",
    "source_path",
    "target_stats_path",
    "coordinate_stats_path",
)


def _resolve_config_path(
    value: str | Path,
    *,
    repo_root: Path | None,
    config_path: Path | None,
) -> Path:
    text = str(value)
    if repo_root is not None:
        text = text.replace("${REPO_ROOT}", str(repo_root))
    base_dir = config_path.parent if config_path is not None else repo_root
    resolved = resolve_path(text, base_dir=base_dir)
    if resolved is None:
        raise ValueError(f"Data path cannot be null: {value!r}")
    return Path(resolved).expanduser()


def _environment_dataset_root(
    placeholder: str,
    *,
    repo_root: Path | None,
) -> Path:
    # Delegate defaults and override precedence to the same placeholders used
    # by training. Evaluation therefore has no separate environment layout.
    resolved = resolve_path(placeholder, base_dir=repo_root)
    if resolved is None:  # pragma: no cover - placeholders never resolve null
        raise ValueError(f"Unable to resolve dataset root {placeholder}")
    return Path(resolved).expanduser()


def resolve_evaluation_data_path(
    value: str | Path,
    *,
    dataset_name: str | None,
    repo_root: Path | None,
    config_path: Path | None = None,
) -> Path:
    """Resolve one archived data path for the active execution environment.

    Existing source paths always win. For AutoDL (or an explicit dataset-root
    override), the configured dataset root is authoritative even when the
    candidate file is missing, so errors identify the current environment's
    expected path instead of an obsolete archived server path.
    """

    original = _resolve_config_path(
        value,
        repo_root=repo_root,
        config_path=config_path,
    )
    if original.is_file():
        return original

    canonical_name = canonical_dataset_name(dataset_name)
    definition = DATASET_ROOT_DEFINITIONS.get(canonical_name)
    placeholder = dataset_root_placeholder(canonical_name)
    if definition is None or placeholder is None:
        return original

    server_env = os.environ.get("SERVER_ENV", "original").strip().lower()
    root_environment_variable = placeholder[2:-1]
    has_root_override = bool(os.environ.get(root_environment_variable))
    if server_env == "autodl" or has_root_override:
        return (
            _environment_dataset_root(placeholder, repo_root=repo_root)
            / original.name
        )
    if server_env != "original":
        raise ValueError(
            f"Unsupported SERVER_ENV={server_env!r}; expected 'original' or 'autodl'"
        )

    if repo_root is None:
        return original
    _, local_relative_path = definition
    candidates = [
        repo_root.parent.parent / "Datasets" / canonical_name / original.name,
        repo_root / local_relative_path / original.name,
    ]
    return next((candidate for candidate in candidates if candidate.is_file()), original)


def normalize_raw_data_paths(
    raw: dict[str, Any],
    *,
    repo_root: Path | None,
    config_path: Path | None = None,
) -> dict[str, Any]:
    """Return a copy whose complete data section uses portable paths."""

    payload = copy.deepcopy(raw)
    data_key = "data" if isinstance(payload.get("data"), dict) else "DATA"
    if not isinstance(payload.get(data_key), dict):
        return payload
    data = dict(payload[data_key])
    dataset_name = data.get("dataset_name")
    for field in SCALAR_DATA_PATH_FIELDS:
        if data.get(field) is not None:
            data[field] = str(
                resolve_evaluation_data_path(
                    data[field],
                    dataset_name=dataset_name,
                    repo_root=repo_root,
                    config_path=config_path,
                )
            )
    if isinstance(data.get("targets"), dict):
        data["targets"] = {
            str(name): str(
                resolve_evaluation_data_path(
                    path,
                    dataset_name=dataset_name,
                    repo_root=repo_root,
                    config_path=config_path,
                )
            )
            for name, path in data["targets"].items()
        }
    payload[data_key] = data
    return payload


def normalize_experiment_data_paths(config: Any, *, repo_root: Path | None) -> Any:
    """Return an ExperimentConfig with every schema data path normalized."""

    data = config.data

    def normalized(value: str | None) -> str | None:
        if value is None:
            return None
        return str(
            resolve_evaluation_data_path(
                value,
                dataset_name=data.dataset_name,
                repo_root=repo_root,
            )
        )

    targets = None
    if data.targets is not None:
        targets = {str(name): normalized(path) for name, path in data.targets.items()}
    return replace(
        config,
        data=replace(
            data,
            target_path=normalized(data.target_path),
            targets=targets,
            coords_path=normalized(data.coords_path),
            coordinate_stats_path=normalized(data.coordinate_stats_path),
        ),
    )

