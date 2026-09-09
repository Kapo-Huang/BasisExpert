from __future__ import annotations

import csv
import hashlib
import json
import math
import platform
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


def evaluation_output_dir(
    run_dir: Path,
    *,
    repo_root: Path,
    result_root: str | Path = "EvalResult",
    evaluation_id: str = "default",
) -> Path:
    """Map a Result run into the schema-v2 evaluation namespace."""
    resolved_run = run_dir.expanduser().resolve()
    run_root = (repo_root / "Result").resolve()
    evaluation_root = Path(result_root).expanduser()
    if not evaluation_root.is_absolute():
        evaluation_root = repo_root / evaluation_root
    evaluation_root = evaluation_root.resolve() / "evaluations" / _safe_path_component(evaluation_id)
    try:
        return evaluation_root / resolved_run.relative_to(run_root)
    except ValueError:
        # AutoDL stores Result outside the repository. Preserve everything after
        # the final Result component so distinct experiments cannot collide.
        result_indices = [
            index
            for index, part in enumerate(resolved_run.parts)
            if part.casefold() == "result"
        ]
        if result_indices:
            suffix = resolved_run.parts[result_indices[-1] + 1:]
            if suffix:
                return evaluation_root / Path(*suffix)

        # Keep arbitrary external runs unique even when their leaf names match.
        try:
            return evaluation_root / "_external" / resolved_run.relative_to(repo_root.resolve())
        except ValueError:
            digest = hashlib.sha256(str(resolved_run).encode("utf-8")).hexdigest()[:12]
            leaf = _safe_path_component(resolved_run.name)
            return evaluation_root / "_external" / f"{leaf}-{digest}"


def path_fingerprint(path: str | Path) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    if resolved.is_file():
        stat = resolved.stat()
        return {"path": str(resolved), "size": int(stat.st_size), "mtime_ns": int(stat.st_mtime_ns)}
    if resolved.is_dir():
        entries = []
        for item in sorted(candidate for candidate in resolved.rglob("*") if candidate.is_file()):
            stat = item.stat()
            entries.append({
                "path": str(item.relative_to(resolved)),
                "size": int(stat.st_size),
                "mtime_ns": int(stat.st_mtime_ns),
            })
        return {"path": str(resolved), "entries": entries}
    return {"path": str(resolved), "missing": True}


def cache_key(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _safe_path_component(value: str) -> str:
    text = str(value).strip() or "unknown"
    for character in '<>:"/\\|?*':
        text = text.replace(character, "_")
    return text.rstrip(". ") or "unknown"


def find_cached_evaluation(output_dir: Path, key: str) -> dict[str, Any] | None:
    if not output_dir.is_dir():
        return None
    manifest_path = output_dir / "manifest.json"
    metrics_path = output_dir / "metrics.json"
    csv_path = output_dir / "metrics.csv"
    log_path = output_dir / "logs" / "evaluate.log"
    if not (manifest_path.is_file() and metrics_path.is_file() and csv_path.is_file()):
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("cache_key") != key:
            return None
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return None
    return {
        "output_dir": output_dir,
        "manifest_path": manifest_path,
        "metrics_path": metrics_path,
        "csv_path": csv_path,
        "log_path": log_path,
        "metrics": metrics,
        "cache_hit": True,
    }


def environment_manifest() -> dict[str, Any]:
    cuda_available = bool(torch.cuda.is_available())
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "cuda_available": cuda_available,
        "cuda_version": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if cuda_available else None,
    }


def _json_safe(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return "Infinity" if value > 0 else "-Infinity" if value < 0 else "NaN"
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path: str | Path, payload: dict[str, Any]) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(_json_safe(payload), ensure_ascii=False, indent=2), encoding="utf-8")
    return output


def write_metrics_csv(path: str | Path, rows: list[dict[str, Any]]) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames or ["row_type"])
        writer.writeheader()
        writer.writerows(_json_safe(rows))
    return output
