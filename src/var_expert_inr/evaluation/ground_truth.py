from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


DATASET_PATH_ALIASES = {"bathymetry": "redsea"}
DATASET_LOCAL_FOLDERS = {
    "redsea": "RedSea",
    "katrina": "Katrina",
    "ionization": "Ionization",
    "combustion_40nh3_1": "Combustion",
}


def portable_data_path(
    path: str | Path,
    *,
    dataset_name: str | None,
    repo_root: Path | None,
) -> Path:
    original = Path(path).expanduser()
    if original.is_file() or repo_root is None:
        return original
    dataset_name = str(dataset_name or "").strip()
    canonical_name = DATASET_PATH_ALIASES.get(dataset_name.lower(), dataset_name)
    display_name = DATASET_LOCAL_FOLDERS.get(canonical_name.lower(), canonical_name.capitalize())
    candidates = [
        repo_root.parent.parent / "Datasets" / canonical_name / original.name,
        repo_root / "data" / "Volume" / display_name / original.name,
        repo_root / "data" / "Mesh" / display_name / original.name,
    ]
    existing = [candidate for candidate in candidates if candidate.is_file()]
    return existing[0] if existing else original


def _portable_data_path(path: str | Path, data_config: Any, repo_root: Path | None) -> Path:
    return portable_data_path(
        path,
        dataset_name=getattr(data_config, "dataset_name", None),
        repo_root=repo_root,
    )


def target_paths_from_config(data_config: Any, *, repo_root: Path | None = None) -> dict[str, Path]:
    target_path = getattr(data_config, "target_path", None)
    targets = getattr(data_config, "targets", None)
    selected = getattr(data_config, "target", None)
    if target_path:
        return {str(selected or "target"): _portable_data_path(target_path, data_config, repo_root)}
    result = {
        str(name): _portable_data_path(path, data_config, repo_root)
        for name, path in (targets or {}).items()
    }
    if selected is not None:
        return {str(selected): result[str(selected)]} if str(selected) in result else {}
    return result


def validate_ground_truth_paths(
    paths: dict[str, Path],
    targets: tuple[str, ...],
    *,
    volume_shape: tuple[int, int, int, int] | None = None,
    node_count: int | None = None,
) -> dict[str, Path]:
    missing_names = [name for name in targets if name not in paths]
    if missing_names:
        raise FileNotFoundError(f"Ground Truth target paths are missing for: {', '.join(missing_names)}")
    validated: dict[str, Path] = {}
    for name in targets:
        path = paths[name].expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Ground Truth file does not exist for target {name!r}: {path}")
        try:
            array = np.load(path, mmap_mode="r", allow_pickle=False)
        except Exception as exc:
            raise ValueError(f"Ground Truth file is unreadable for target {name!r}: {path}: {exc}") from exc
        if volume_shape is not None:
            expected_samples = int(np.prod(volume_shape, dtype=np.int64))
            actual_samples = int(np.prod(array.shape[:4], dtype=np.int64)) if array.ndim >= 4 else int(array.shape[0])
            if actual_samples != expected_samples:
                raise ValueError(
                    f"Ground Truth shape mismatch for {name!r}: expected {volume_shape} "
                    f"({expected_samples} samples), got {array.shape}"
                )
        elif node_count is not None and (array.ndim == 0 or int(array.shape[0]) != int(node_count)):
            raise ValueError(
                f"Ground Truth shape mismatch for {name!r}: expected first dimension {node_count}, got {array.shape}"
            )
        validated[name] = path
    return validated
