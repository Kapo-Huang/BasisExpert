"""Schema-v2 batch evaluation entry point."""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import logging
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from var_expert_inr.evaluation.rendering import (
    load_render_profile,
    profile_fingerprint,
    renderer_name,
)
from var_expert_inr.evaluation.metrics import validate_error_bounds
from var_expert_inr.evaluation.dependency_cache import load_dependency_configuration
from var_expert_inr.evaluation.dependency_service import canonical_dependency_run_dirs
from var_expert_inr.evaluation.reporting import (
    evaluation_output_dir,
    write_json,
    write_metrics_csv,
)
from var_expert_inr.evaluation.artifacts import ArtifactStore, LAYOUT_SCHEMA_VERSION, resolve_artifact_reference
from var_expert_inr.evaluation.service import evaluate_run, resolve_run_config
from var_expert_inr.evaluation.selection import parse_timestep_selection


LOGGER = logging.getLogger("evaluation_exploration")
COMPLETED_STATUSES = {"success", "skipped"}
SUMMARY_FIELDS = (
    "index",
    "status",
    "model",
    "dataset",
    "target",
    "run_dir",
    "output_dir",
    "metrics_path",
    "log_path",
    "worker_log",
    "return_code",
    "elapsed_seconds",
    "reuse_reason",
    "error_type",
    "error",
)


def _load_mapping(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a YAML mapping: {path}")
    return payload


def _resolve_repo_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def _server_environment(value: str | None = None) -> str:
    selected = str(value or os.environ.get("SERVER_ENV", "original")).strip().lower()
    if selected not in {"original", "autodl"}:
        raise ValueError(
            f"Unsupported server environment {selected!r}; expected original or autodl"
        )
    return selected


def _resolve_run_source_path(value: str | Path, *, server_env: str) -> Path:
    """Resolve Result inputs using the repository's selected server profile."""
    raw = str(value).strip().replace("\\", "/")
    relative_parts = tuple(part for part in raw.split("/") if part not in {"", "."})
    if (
        server_env == "autodl"
        and relative_parts
        and relative_parts[0].lower() == "result"
        and not Path(value).expanduser().is_absolute()
    ):
        autodl_root = Path(os.environ.get("AUTODL_DATA_ROOT", "/root/autodl-tmp"))
        return autodl_root / "Result" / Path(*relative_parts[1:])
    return _resolve_repo_path(value)


def _read_run_list(path: Path) -> list[Path]:
    runs: list[Path] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        runs.append(_resolve_repo_path(line))
    return runs


def _read_manifest_runs(
    path: Path,
    *,
    datasets: tuple[str, ...] = (),
    missing_fields: tuple[str, ...] = (),
) -> list[Path]:
    """Select evaluable Result entries from the archive manifest."""
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        fields = set(reader.fieldnames or ())
        required = {"dataset", "status", "result_path", *missing_fields}
        absent = sorted(required - fields)
        if absent:
            raise ValueError(
                f"Result manifest is missing required columns {absent}: {path}"
            )
        selected_datasets = {name.strip().lower() for name in datasets if name.strip()}
        runs: set[Path] = set()
        for row in reader:
            if str(row.get("status", "")).strip().lower() == "missing":
                continue
            dataset = str(row.get("dataset", "")).strip().lower()
            if selected_datasets and dataset not in selected_datasets:
                continue
            if missing_fields and all(
                str(row.get(field, "")).strip() for field in missing_fields
            ):
                continue
            run_dir = _resolve_repo_path(str(row.get("result_path", "")).strip())
            try:
                resolve_run_config(run_dir)
            except FileNotFoundError as exc:
                raise FileNotFoundError(
                    f"Selected Result entry has no evaluation config: {run_dir}"
                ) from exc
            checkpoint_dir = run_dir / "checkpoints"
            if not any(checkpoint_dir.glob("*.pth")):
                raise FileNotFoundError(
                    f"Selected Result entry has no checkpoint: {run_dir}"
                )
            runs.add(run_dir.resolve())
    return sorted(runs)


def _discover_checkpoint_runs(root: Path) -> list[Path]:
    """Find archived runs that can be evaluated from their local checkpoint."""
    if not root.is_dir():
        raise FileNotFoundError(f"Evaluation run root does not exist: {root}")
    runs: set[Path] = set()
    for config_path in sorted(root.rglob("config.yaml")):
        run_dir = (
            config_path.parent.parent
            if config_path.parent.name == "configs"
            else config_path.parent
        )
        checkpoint_dir = run_dir / "checkpoints"
        if any(checkpoint_dir.glob("*.pth")):
            runs.add(run_dir.resolve())
    return sorted(runs)


def _section(payload: dict[str, Any], lower: str, upper: str) -> dict[str, Any]:
    value = payload.get(lower)
    if not isinstance(value, dict):
        value = payload.get(upper)
    return value if isinstance(value, dict) else {}


def _run_identity(run_dir: Path) -> tuple[str, str, Path, dict[str, Any]]:
    config_path = resolve_run_config(run_dir)
    raw = _load_mapping(config_path)
    data = _section(raw, "data", "DATA")
    model = _section(raw, "model", "MODEL")
    dataset = str(data.get("dataset_name", "")).strip()
    model_name = str(model.get("name") or model.get("model_name") or "unknown").strip()
    if not dataset:
        raise ValueError(
            f"Run config does not define data.dataset_name/DATA.dataset_name: {config_path}"
        )
    return model_name, dataset, config_path, raw


def _normalized_target_map(value: Any) -> dict[str, str]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError("evaluation.targets_by_dataset must be a mapping when provided")
    return {
        str(dataset).strip().lower(): str(target).strip()
        for dataset, target in value.items()
    }


def _normalized_timestep_map(value: Any) -> dict[str, str]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError("evaluation.timesteps_by_dataset must be a mapping when provided")
    return {
        str(dataset).strip().lower(): str(selection).strip()
        for dataset, selection in value.items()
    }


def _configure_logging(log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(log_path, encoding="utf-8")],
        force=True,
    )


def _write_runtime_summaries(output_dir: Path, records: list[dict[str, Any]]) -> None:
    entries: list[dict[str, Any]] = []
    for record in records:
        if record.get("status") not in COMPLETED_STATUSES or not record.get("metrics_path"):
            continue
        try:
            payload = json.loads(Path(record["metrics_path"]).read_text(encoding="utf-8"))
        except (OSError, TypeError, ValueError):
            continue
        performance = payload.get("performance") or {}
        training = performance.get("training_time") or {}
        inference = performance.get("inference_time") or {}
        if not training and not inference:
            continue
        entries.append({
            "model": record.get("model", ""),
            "dataset": record.get("dataset", ""),
            "target": record.get("target", ""),
            "run_dir": record.get("run_dir", ""),
            "estimated_training_seconds": training.get("estimated_training_seconds"),
            "estimated_training_hours": training.get("estimated_training_hours"),
            "estimated_total_inference_seconds": inference.get("estimated_total_inference_seconds"),
            "estimated_total_inference_hours": inference.get("estimated_total_inference_hours"),
            "training_samples_per_second": training.get("samples_per_second"),
            "inference_values_per_second": inference.get("values_per_second"),
        })
    if not entries:
        return

    grouped: dict[tuple[str, str], dict[str, Any]] = {}
    for entry in entries:
        key = (str(entry["model"]), str(entry["dataset"]))
        group = grouped.setdefault(
            key,
            {
                "model": key[0],
                "dataset": key[1],
                "experiment_count": 0,
                "estimated_training_seconds": 0.0,
                "estimated_total_inference_seconds": 0.0,
            },
        )
        group["experiment_count"] += 1
        for field in ("estimated_training_seconds", "estimated_total_inference_seconds"):
            value = entry.get(field)
            if value is not None:
                group[field] += float(value)
    groups = list(grouped.values())
    for group in groups:
        group["estimated_training_hours"] = group["estimated_training_seconds"] / 3600.0
        group["estimated_total_inference_hours"] = group["estimated_total_inference_seconds"] / 3600.0

    write_json(output_dir / "runtime_summary.json", {
        "schema_version": LAYOUT_SCHEMA_VERSION,
        "grouping": ["model", "dataset"],
        "experiment_count": len(entries),
        "entries": entries,
        "groups": groups,
    })
    for name, rows in (("runtime_entries.tsv", entries), ("runtime_groups.tsv", groups)):
        fields = list(dict.fromkeys(key for row in rows for key in row))
        with (output_dir / name).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)


def _write_summary(output_dir: Path, records: list[dict[str, Any]], config_path: Path) -> None:
    succeeded = sum(row["status"] == "success" for row in records)
    skipped = sum(row["status"] == "skipped" for row in records)
    completed = succeeded + skipped
    summary = {
        "schema_version": LAYOUT_SCHEMA_VERSION,
        "config_path": str(config_path),
        "status": "complete" if completed == len(records) else "failed",
        "total": len(records),
        "succeeded": succeeded,
        "skipped": skipped,
        "failed": len(records) - completed,
        "timed_out": sum(row["status"] == "timeout" for row in records),
        "records": records,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )
    with (output_dir / "summary.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=SUMMARY_FIELDS,
            delimiter="\t",
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(records)
    _write_runtime_summaries(output_dir, records)


def _explicit_timesteps(
    value: str,
    *,
    stored_timesteps: tuple[int, ...] = (),
) -> tuple[int, ...] | None:
    tokens = [token.strip() for token in value.split(",") if token.strip()]
    if not tokens or any(token.lower() == "all" for token in tokens):
        return None
    if len(tokens) == 1 and tokens[0].lower().startswith("uniform:"):
        if not stored_timesteps:
            return None
        try:
            return parse_timestep_selection(value, stored_timesteps[-1] + 1)
        except (IndexError, ValueError):
            return None
    if any(":" in token for token in tokens):
        return None
    try:
        return tuple(int(token) for token in tokens)
    except ValueError:
        return None


def _is_uniform_timestep_request(value: str) -> bool:
    tokens = [token.strip() for token in value.split(",") if token.strip()]
    return len(tokens) == 1 and tokens[0].lower().startswith("uniform:")


def _matching_stored_timesteps(
    value: str,
    *,
    stored_timesteps: tuple[int, ...],
) -> tuple[tuple[int, ...], str] | None:
    """Return the completed coverage and its reuse reason, if reusable.

    Explicit selections require exact equality.  A uniform request can reuse a
    denser completed uniform evaluation because the archived metrics cover at
    least as many uniformly selected frames as the current request.
    """
    requested_timesteps = _explicit_timesteps(
        value,
        stored_timesteps=stored_timesteps,
    )
    if requested_timesteps is None:
        return None
    if _is_uniform_timestep_request(value):
        if len(stored_timesteps) < len(requested_timesteps):
            return None
        reason = (
            "exact-timestep-match"
            if stored_timesteps == requested_timesteps
            else "uniform-coverage"
        )
        return stored_timesteps, reason
    if stored_timesteps != requested_timesteps:
        return None
    return requested_timesteps, "exact-timestep-match"


def _requested_target_names(raw: dict[str, Any], target: str) -> tuple[str, ...]:
    if target != "all":
        return (target,)
    data = _section(raw, "data", "DATA")
    if data.get("target"):
        return (str(data["target"]),)
    configured = data.get("targets")
    return tuple(str(name) for name in configured) if isinstance(configured, dict) else ()


def _source_fingerprint_matches(
    stored: Any,
    source_path: Path,
    artifact_store: ArtifactStore,
) -> bool:
    if stored is None:
        return True
    current = artifact_store.content_fingerprint(source_path)
    if stored == current:
        return True
    if not isinstance(stored, dict) or "path" not in stored:
        return False
    try:
        legacy_path = Path(str(stored["path"])).expanduser().resolve()
        stat = source_path.stat()
        return (
            legacy_path == source_path
            and int(stored["size"]) == int(stat.st_size)
            and int(stored["mtime_ns"]) == int(stat.st_mtime_ns)
        )
    except (KeyError, OSError, TypeError, ValueError):
        return False


def _existing_evaluation_state_at(
    run_dir: Path,
    *,
    raw: dict[str, Any],
    target: str,
    timesteps: str,
    requested_metrics: tuple[str, ...],
    render: bool,
    current_profile_fingerprint: str | None,
    result_root: Path,
    output_dir: Path,
    error_vmin: float,
    error_vmax: float,
) -> dict[str, Any] | None:
    manifest_path = output_dir / "manifest.json"
    metrics_path = output_dir / "metrics.json"
    csv_path = output_dir / "metrics.csv"
    if not (manifest_path.is_file() and metrics_path.is_file() and csv_path.is_file()):
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return None
    if payload.get("status") != "complete":
        return None
    performance = payload.get("performance") or {}
    runtime_completed = {
        metric
        for metric in ("training_time", "inference_time")
        if isinstance(performance.get(metric), dict)
    }
    if (
        requested_metrics
        and set(requested_metrics).issubset(runtime_completed)
        and not render
    ):
        return {
            "output_dir": output_dir,
            "manifest": manifest,
            "metrics": payload,
            "completed_metrics": runtime_completed,
            "render_complete": False,
            "reuse_reason": "experiment-runtime-cache",
        }
    if manifest.get("source_kind") != "checkpoint":
        return None

    source_path = Path(str(manifest.get("source_path", ""))).expanduser().resolve()
    checkpoint_dir = (run_dir / "checkpoints").resolve()
    try:
        source_path.relative_to(checkpoint_dir)
    except ValueError:
        return None
    if not source_path.is_file():
        return None
    stored_source_fingerprint = manifest.get("source_fingerprint")
    artifact_store = ArtifactStore(repo_root=REPO_ROOT, result_root=result_root)
    if not _source_fingerprint_matches(
        stored_source_fingerprint,
        source_path,
        artifact_store,
    ):
        return None

    stored_timesteps = tuple(int(item) for item in (manifest.get("timesteps") or ()))
    timestep_match = _matching_stored_timesteps(
        timesteps,
        stored_timesteps=stored_timesteps,
    )
    if timestep_match is None:
        return None
    covered_timesteps, reuse_reason = timestep_match
    requested_targets = _requested_target_names(raw, target)
    stored_targets = tuple(str(name) for name in (manifest.get("targets") or ()))
    if requested_targets and set(stored_targets) != set(requested_targets):
        return None

    rows = list(payload.get("per_timestep") or ())
    expected = {(name, step) for name in stored_targets for step in covered_timesteps}
    rows_by_key = {
        (str(row.get("target")), int(row.get("timestep"))): row
        for row in rows
        if row.get("target") is not None and row.get("timestep") is not None
    }
    completed_metrics: set[str] = set()
    for metric in requested_metrics:
        if metric in {"training_time", "inference_time"}:
            if isinstance(performance.get(metric), dict):
                completed_metrics.add(metric)
        elif metric in {"decode_time", "memory"}:
            if metric == "decode_time" and payload.get("performance", {}).get("total_decode_seconds") is not None:
                completed_metrics.add(metric)
            elif metric == "memory" and payload.get("performance", {}).get("peak_memory_bytes") is not None:
                completed_metrics.add(metric)
        elif metric == "error" and expected and all(
            all(
                rows_by_key.get(key, {}).get(field) is not None
                for field in (
                    "mean_absolute_error",
                    "max_absolute_error",
                    "p95_absolute_error",
                    "p99_absolute_error",
                    "mean_error_percentage",
                    "max_error_percentage",
                    "p95_error_percentage",
                    "p99_error_percentage",
                )
            )
            for key in expected
        ):
            completed_metrics.add(metric)
        elif expected and all(rows_by_key.get(key, {}).get(metric) is not None for key in expected):
            completed_metrics.add(metric)

    render_complete = False
    if render and expected and bool(manifest.get("render_requested")):
        stored_profile = (manifest.get("render_profile") or {}).get("fingerprint")
        render_complete = stored_profile == current_profile_fingerprint and all(
            resolve_artifact_reference(result_root, str(rows_by_key.get(key, {}).get("pred_render_path", ""))).is_file()
            and resolve_artifact_reference(result_root, str(rows_by_key.get(key, {}).get("gt_render_path", ""))).is_file()
            for key in expected
        )
        if render_complete and "error" in requested_metrics:
            stored_error = manifest.get("error_analysis") or {}
            try:
                render_complete = (
                    bool(stored_error.get("enabled"))
                    and float(stored_error.get("error_vmin")) == error_vmin
                    and float(stored_error.get("error_vmax")) == error_vmax
                    and stored_error.get("gt_range") == [-1.0, 1.0]
                    and all(
                        resolve_artifact_reference(result_root, str(rows_by_key.get(key, {}).get("error_render_path", ""))).is_file()
                        for key in expected
                    )
                )
            except (TypeError, ValueError):
                render_complete = False
    return {
        "output_dir": output_dir,
        "manifest": manifest,
        "metrics": payload,
        "completed_metrics": completed_metrics,
        "render_complete": render_complete,
        "reuse_reason": reuse_reason,
    }


def _existing_evaluation_state(
    run_dir: Path,
    *,
    raw: dict[str, Any],
    target: str,
    timesteps: str,
    requested_metrics: tuple[str, ...],
    render: bool,
    current_profile_fingerprint: str | None,
    result_root: Path,
    evaluation_id: str,
    error_vmin: float,
    error_vmax: float,
) -> dict[str, Any] | None:
    current_output = evaluation_output_dir(
        run_dir,
        repo_root=REPO_ROOT,
        result_root=result_root,
        evaluation_id=evaluation_id,
    )
    candidates = [current_output]
    evaluation_root = result_root / "evaluations"
    if evaluation_root.is_dir():
        for namespace in sorted(path for path in evaluation_root.iterdir() if path.is_dir()):
            candidate = evaluation_output_dir(
                run_dir,
                repo_root=REPO_ROOT,
                result_root=result_root,
                evaluation_id=namespace.name,
            )
            if candidate != current_output and candidate.is_dir():
                candidates.append(candidate)

    current_state: dict[str, Any] | None = None
    for candidate in candidates:
        state = _existing_evaluation_state_at(
            run_dir,
            raw=raw,
            target=target,
            timesteps=timesteps,
            requested_metrics=requested_metrics,
            render=render,
            current_profile_fingerprint=current_profile_fingerprint,
            result_root=result_root,
            output_dir=candidate,
            error_vmin=error_vmin,
            error_vmax=error_vmax,
        )
        if state is None:
            continue
        is_current = candidate == current_output
        if is_current:
            current_state = state
        metrics_complete = set(requested_metrics).issubset(state["completed_metrics"])
        render_complete = not render or bool(state["render_complete"])
        if metrics_complete and render_complete:
            if not is_current:
                state["reuse_reason"] = (
                    "cross-namespace-" + str(state["reuse_reason"])
                )
            return state
    # Partial results are merged only inside the requested namespace. This
    # prevents a worker from overwriting a legacy namespace while still letting
    # complete legacy metrics satisfy a new request.
    return current_state


def _merge_incremental_result(
    state: dict[str, Any],
    *,
    requested_metrics: tuple[str, ...],
) -> None:
    output_dir = Path(state["output_dir"])
    previous_payload = state["metrics"]
    previous_manifest = state["manifest"]
    current_payload = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    current_manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))

    rows_by_key: dict[tuple[str, int], dict[str, Any]] = {}
    order: list[tuple[str, int]] = []
    for payload in (previous_payload, current_payload):
        for row in payload.get("per_timestep") or ():
            key = (str(row.get("target")), int(row.get("timestep")))
            if key not in rows_by_key:
                rows_by_key[key] = {}
                order.append(key)
            rows_by_key[key].update(row)
    merged_rows = [rows_by_key[key] for key in order]

    merged_targets: dict[str, dict[str, Any]] = {}
    for payload in (previous_payload, current_payload):
        for name, values in (payload.get("targets") or {}).items():
            merged_targets.setdefault(str(name), {}).update(values)
    merged_aggregate = dict(previous_payload.get("aggregate") or {})
    merged_aggregate.update(current_payload.get("aggregate") or {})
    merged_performance = dict(previous_payload.get("performance") or {})
    merged_performance.update(current_payload.get("performance") or {})
    merged_payload = {
        "schema_version": LAYOUT_SCHEMA_VERSION,
        "status": "complete",
        "targets": merged_targets,
        "aggregate": merged_aggregate,
        "performance": merged_performance,
        "per_timestep": merged_rows,
    }
    write_json(output_dir / "metrics.json", merged_payload)
    write_metrics_csv(output_dir / "metrics.csv", merged_rows or [{"row_type": "performance", **merged_performance}])

    merged_manifest = dict(previous_manifest)
    merged_manifest.update(current_manifest)
    merged_manifest["metrics"] = list(requested_metrics)
    merged_manifest["render_requested"] = bool(
        previous_manifest.get("render_requested") or current_manifest.get("render_requested")
    )
    merged_manifest["cache_key"] = None
    merged_manifest["incremental"] = True
    write_json(output_dir / "manifest.json", merged_manifest)


def _restore_existing_result(state: dict[str, Any]) -> None:
    output_dir = Path(state["output_dir"])
    payload = state["metrics"]
    write_json(output_dir / "manifest.json", state["manifest"])
    write_json(output_dir / "metrics.json", payload)
    rows = list(payload.get("per_timestep") or ())
    write_metrics_csv(output_dir / "metrics.csv", rows or [{"row_type": "performance", **(payload.get("performance") or {})}])
    write_json(output_dir / "progress.json", {"status": "complete", "completed_rows": len(rows)})


def _require_module(module: str, feature: str) -> None:
    try:
        importlib.import_module(module)
    except ImportError as exc:
        raise RuntimeError(
            f"{feature} requires Python module {module!r}. Install the documented evaluation dependencies first."
        ) from exc


def _preflight_dependencies(
    run_dirs: list[Path],
    *,
    metrics: tuple[str, ...],
    render: bool,
    render_profile: Path | None,
) -> None:
    needs_render = render or bool({"ssim", "lpips"}.intersection(metrics))
    if not needs_render:
        return
    _require_module("PIL", "Rendering")
    _require_module("matplotlib", "Rendering")
    if "ssim" in metrics:
        _require_module("skimage", "SSIM")
    if "lpips" in metrics:
        _require_module("lpips", "LPIPS")
        _require_module("torchvision", "LPIPS")
        import torch

        alexnet_path = (
            Path(torch.hub.get_dir())
            / "checkpoints"
            / "alexnet-owt-7be5be79.pth"
        )
        if not alexnet_path.is_file():
            raise RuntimeError(
                "LPIPS AlexNet weights are missing. Download "
                "https://download.pytorch.org/models/alexnet-owt-7be5be79.pth "
                f"to {alexnet_path} before running the render configuration."
            )
    required_renderers: set[str] = set()
    for run_dir in run_dirs:
        _, dataset, _, raw = _run_identity(run_dir)
        data = _section(raw, "data", "DATA")
        dataset_kind = str(
            data.get("kind", "volume" if data.get("volume_shape") else "node")
        ).lower()
        profile = load_render_profile(
            dataset,
            render_profile,
            repo_root=REPO_ROOT,
        )
        required_renderers.add(renderer_name(profile, dataset_kind=dataset_kind))
    if "volume" in required_renderers:
        _require_module("volume_vis", "Volume rendering")
    if "mesh" in required_renderers:
        _require_module("pyvista", "Mesh rendering")
        _require_module("vtk", "Mesh rendering")


def _worker_main(request_path: Path, result_path: Path) -> int:
    request = json.loads(request_path.read_text(encoding="utf-8"))
    response: dict[str, Any]
    try:
        selected_env = _server_environment(request.get("server_env"))
        os.environ["SERVER_ENV"] = selected_env
        result = evaluate_run(
            request["run_dir"],
            metrics=tuple(request["metrics"]),
            timesteps=request["timesteps"],
            targets=request["target"],
            source=request["source"],
            render=bool(request["render"]),
            render_profile=request.get("render_profile"),
            overwrite=bool(request["overwrite"]),
            device=request.get("device"),
            result_root=request.get("result_root", "EvalResult"),
            evaluation_id=request.get("evaluation_id", "default"),
            error_vmin=request.get("error_vmin"),
            error_vmax=request.get("error_vmax"),
            training_probe_samples=int(request.get("training_probe_samples", 72_000_000)),
            training_total_samples=int(request.get("training_total_samples", 14_400_000_000)),
            inference_fraction=float(request.get("inference_fraction", 0.1)),
        )
        response = {
            "status": "success",
            "output_dir": str(Path(result["output_dir"]).resolve()),
            "metrics_path": str(Path(result["metrics_path"]).resolve()),
            "log_path": str(Path(result["log_path"]).resolve()),
            "error_type": "",
            "error": "",
        }
        return_code = 0
    except BaseException as exc:
        traceback.print_exc()
        response = {
            "status": "failed",
            "output_dir": "",
            "metrics_path": "",
            "log_path": "",
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        return_code = 1
    result_path.write_text(
        json.dumps(response, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )
    return return_code


def run_batch(
    config_path: str | Path,
    *,
    server_env: str | None = None,
) -> tuple[Path, list[dict[str, Any]]]:
    selected_env = _server_environment(server_env)
    os.environ["SERVER_ENV"] = selected_env
    resolved_config = _resolve_repo_path(config_path)
    batch = _load_mapping(resolved_config)
    evaluation = batch.get("evaluation")
    if not isinstance(evaluation, dict):
        raise ValueError(f"Missing evaluation mapping in {resolved_config}")

    list_path_value = batch.get("list_path")
    manifest_path_value = batch.get("manifest_path")
    run_root_value = batch.get("run_root")
    run_roots_value = batch.get("run_roots")
    configured_sources = sum(
        value is not None
        for value in (
            list_path_value,
            manifest_path_value,
            run_root_value,
            run_roots_value,
        )
    )
    if configured_sources > 1:
        raise ValueError(
            "Specify only one of list_path, manifest_path, run_root, or run_roots"
        )
    if configured_sources == 0:
        raise ValueError("Missing list_path, manifest_path, run_root, or run_roots")
    list_path = None if list_path_value is None else _resolve_repo_path(list_path_value)
    manifest_path = (
        None
        if manifest_path_value is None
        else _resolve_repo_path(manifest_path_value)
    )
    run_root = (
        None
        if run_root_value is None
        else _resolve_run_source_path(run_root_value, server_env=selected_env)
    )
    if run_roots_value is None:
        run_roots: tuple[Path, ...] = ()
    else:
        if not isinstance(run_roots_value, (list, tuple)) or not run_roots_value:
            raise ValueError("run_roots must be a non-empty list of directories")
        run_roots = tuple(
            _resolve_run_source_path(value, server_env=selected_env)
            for value in run_roots_value
        )
    result_root = _resolve_repo_path(batch.get("result_root", "EvalResult"))
    evaluation_id = str(batch.get("evaluation_id") or "").strip()
    if not evaluation_id:
        raise ValueError("evaluation_id is required")
    summary_root = result_root / "batches" / evaluation_id
    continue_on_error = bool(batch.get("continue_on_error", True))
    timeout_seconds = int(batch.get("item_timeout_seconds", 3600))
    if timeout_seconds <= 0:
        raise ValueError("item_timeout_seconds must be positive")
    metrics_value = evaluation.get("metrics", ["psnr"])
    if isinstance(metrics_value, str):
        metrics = tuple(
            token.strip().lower()
            for token in metrics_value.split(",")
            if token.strip()
        )
    elif isinstance(metrics_value, (list, tuple)):
        metrics = tuple(str(metric).strip().lower() for metric in metrics_value)
    else:
        raise ValueError("evaluation.metrics must be a string or a list of metric names")
    timesteps = str(evaluation.get("timesteps", "all"))
    render = bool(evaluation.get("render", False))
    incremental = bool(evaluation.get("incremental", False))
    source = str(evaluation.get("source", "checkpoint"))
    overwrite = bool(evaluation.get("overwrite", False))
    device_value = evaluation.get("device")
    device = None if device_value in (None, "", "auto") else str(device_value)
    error_vmin, error_vmax = validate_error_bounds(
        evaluation.get("error_vmin", 0.0),
        evaluation.get("error_vmax", 5.0),
    )
    training_probe_samples = int(evaluation.get("training_probe_samples", 72_000_000))
    training_total_samples = int(evaluation.get("training_total_samples", 14_400_000_000))
    inference_fraction = float(evaluation.get("inference_fraction", 0.1))
    if training_probe_samples <= 0 or training_total_samples <= 0:
        raise ValueError("evaluation training sample budgets must be positive")
    if not 0.0 < inference_fraction <= 1.0:
        raise ValueError("evaluation.inference_fraction must be in (0, 1]")
    profile_value = evaluation.get("render_profile", "auto")
    render_profile = (
        None
        if profile_value in (None, "", "auto")
        else _resolve_repo_path(profile_value)
    )
    targets_by_dataset = _normalized_target_map(
        evaluation.get("targets_by_dataset")
    )
    timesteps_by_dataset = _normalized_timestep_map(
        evaluation.get("timesteps_by_dataset")
    )
    if list_path is not None:
        run_dirs = _read_run_list(list_path)
        run_source: Any = list_path
    elif manifest_path is not None:
        selection = batch.get("manifest_selection") or {}
        if not isinstance(selection, dict):
            raise ValueError("manifest_selection must be a mapping when provided")
        datasets_value = selection.get("datasets") or ()
        missing_fields_value = selection.get("missing_fields") or ()
        if isinstance(datasets_value, str):
            datasets = (datasets_value,)
        elif isinstance(datasets_value, (list, tuple)):
            datasets = tuple(str(value) for value in datasets_value)
        else:
            raise ValueError("manifest_selection.datasets must be a string or list")
        if isinstance(missing_fields_value, str):
            missing_fields = (missing_fields_value,)
        elif isinstance(missing_fields_value, (list, tuple)):
            missing_fields = tuple(str(value) for value in missing_fields_value)
        else:
            raise ValueError(
                "manifest_selection.missing_fields must be a string or list"
            )
        run_dirs = _read_manifest_runs(
            manifest_path,
            datasets=datasets,
            missing_fields=missing_fields,
        )
        run_source = manifest_path
    elif run_root is not None:
        run_dirs = _discover_checkpoint_runs(run_root)
        run_source = run_root
    else:
        run_dirs = sorted(
            {
                run_dir
                for root in run_roots
                for run_dir in _discover_checkpoint_runs(root)
            }
        )
        run_source = run_roots
    discovered_run_count = len(run_dirs)
    dependency_metrics = {"pearson_error", "mi_error"}
    if metrics and set(metrics).issubset(dependency_metrics):
        dependency_configuration = load_dependency_configuration(resolved_config)
        run_dirs = list(
            canonical_dependency_run_dirs(run_dirs, dependency_configuration)
        )
    _preflight_dependencies(
        run_dirs,
        metrics=metrics,
        render=render,
        render_profile=render_profile,
    )

    output_dir = summary_root
    output_dir.mkdir(parents=True, exist_ok=True)
    worker_dir = output_dir / "workers"
    worker_dir.mkdir(parents=True, exist_ok=True)
    _configure_logging(output_dir / "batch.log")

    records: list[dict[str, Any]] = []
    if len(run_dirs) != discovered_run_count:
        LOGGER.info(
            "Collapsed %d discovered dependency runs to %d canonical groups",
            discovered_run_count,
            len(run_dirs),
        )
    LOGGER.info("Loaded %d evaluation runs from %s", len(run_dirs), run_source)
    for index, run_dir in enumerate(run_dirs, start=1):
        worker_log = worker_dir / f"{index:03d}.log"
        request_path = worker_dir / f"{index:03d}.request.json"
        result_path = worker_dir / f"{index:03d}.result.json"
        record: dict[str, Any] = {
            "index": index,
            "status": "failed",
            "model": "",
            "dataset": "",
            "target": "",
            "run_dir": str(run_dir),
            "output_dir": "",
            "metrics_path": "",
            "log_path": "",
            "worker_log": str(worker_log.resolve()),
            "return_code": "",
            "elapsed_seconds": 0.0,
            "reuse_reason": "",
            "error_type": "",
            "error": "",
        }
        started = time.perf_counter()
        existing_state: dict[str, Any] | None = None
        try:
            model_name, dataset, _, raw = _run_identity(run_dir)
            target = targets_by_dataset.get(dataset.lower(), "all")
            run_timesteps = timesteps_by_dataset.get(dataset.lower(), timesteps)
            record.update(model=model_name, dataset=dataset, target=target)
            current_profile_fingerprint = None
            if render:
                current_profile = load_render_profile(
                    dataset,
                    render_profile,
                    repo_root=REPO_ROOT,
                )
                current_profile_fingerprint = profile_fingerprint(current_profile)
            if incremental and not overwrite and source == "checkpoint":
                existing_state = _existing_evaluation_state(
                    run_dir,
                    raw=raw,
                    target=target,
                    timesteps=run_timesteps,
                    requested_metrics=metrics,
                    render=render,
                    current_profile_fingerprint=current_profile_fingerprint,
                    result_root=result_root,
                    evaluation_id=evaluation_id,
                    error_vmin=error_vmin,
                    error_vmax=error_vmax,
                )
            completed_metrics = (
                set(existing_state["completed_metrics"])
                if existing_state is not None
                else set()
            )
            pending_metrics = tuple(metric for metric in metrics if metric not in completed_metrics)
            pending_render = bool(
                render and (
                    existing_state is None
                    or not bool(existing_state["render_complete"])
                )
            )
            if pending_render and "error" in metrics and "error" not in pending_metrics:
                pending_metrics = (*pending_metrics, "error")
            if existing_state is not None and not pending_metrics and not pending_render:
                existing_output = Path(existing_state["output_dir"])
                record.update(
                    status="skipped",
                    output_dir=str(existing_output.resolve()),
                    metrics_path=str((existing_output / "metrics.json").resolve()),
                    log_path=str((existing_output / "logs" / "evaluate.log").resolve()),
                    return_code=0,
                    reuse_reason=str(existing_state["reuse_reason"]),
                )
                LOGGER.info(
                    "[%d/%d] Skipped completed checkpoint (%s): %s",
                    index,
                    len(run_dirs),
                    existing_state["reuse_reason"],
                    run_dir,
                )
                record["elapsed_seconds"] = round(time.perf_counter() - started, 6)
                records.append(record)
                _write_summary(output_dir, records, resolved_config)
                continue
            worker_render = bool(
                pending_render
                or {"ssim", "lpips"}.intersection(pending_metrics)
            )
            LOGGER.info(
                "[%d/%d] Evaluating model=%s dataset=%s target=%s metrics=%s render=%s run=%s",
                index,
                len(run_dirs),
                model_name,
                dataset,
                target,
                ",".join(pending_metrics) or "none",
                worker_render,
                run_dir,
            )
            request_path.write_text(
                json.dumps(
                    {
                        "run_dir": str(run_dir),
                        "server_env": selected_env,
                        "metrics": list(pending_metrics),
                        "timesteps": run_timesteps,
                        "target": target,
                        "source": source,
                        "render": worker_render,
                        "render_profile": None if render_profile is None else str(render_profile),
                        "overwrite": overwrite,
                        "device": device,
                        "result_root": str(result_root),
                        "evaluation_id": evaluation_id,
                        "error_vmin": error_vmin,
                        "error_vmax": error_vmax,
                        "training_probe_samples": training_probe_samples,
                        "training_total_samples": training_total_samples,
                        "inference_fraction": inference_fraction,
                    },
                    indent=2,
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--worker-request",
                str(request_path),
                "--worker-result",
                str(result_path),
            ]
            try:
                with worker_log.open("w", encoding="utf-8") as handle:
                    completed = subprocess.run(
                        command,
                        cwd=REPO_ROOT,
                        stdout=handle,
                        stderr=subprocess.STDOUT,
                        timeout=timeout_seconds,
                        check=False,
                    )
                record["return_code"] = completed.returncode
            except subprocess.TimeoutExpired:
                record.update(
                    status="timeout",
                    error_type="TimeoutExpired",
                    error=f"Evaluation exceeded {timeout_seconds} seconds",
                )
            else:
                if result_path.is_file():
                    response = json.loads(result_path.read_text(encoding="utf-8"))
                    record.update(response)
                else:
                    record.update(
                        error_type="ChildProcessError",
                        error=f"Worker exited with code {completed.returncode} without a result file",
                    )
            if record["status"] == "success":
                if existing_state is not None:
                    _merge_incremental_result(
                        existing_state,
                        requested_metrics=metrics,
                    )
                LOGGER.info(
                    "[%d/%d] Completed: %s",
                    index,
                    len(run_dirs),
                    record["metrics_path"],
                )
            else:
                if existing_state is not None:
                    _restore_existing_result(existing_state)
                LOGGER.error(
                    "[%d/%d] Failed run=%s type=%s error=%s worker_log=%s",
                    index,
                    len(run_dirs),
                    run_dir,
                    record["error_type"],
                    record["error"],
                    worker_log,
                )
        except Exception as exc:
            if existing_state is not None:
                _restore_existing_result(existing_state)
            record.update(status="failed", error_type=type(exc).__name__, error=str(exc))
            LOGGER.exception("[%d/%d] Failed run=%s: %s", index, len(run_dirs), run_dir, exc)
        record["elapsed_seconds"] = round(time.perf_counter() - started, 6)
        records.append(record)
        _write_summary(output_dir, records, resolved_config)
        if record["status"] not in COMPLETED_STATUSES and not continue_on_error:
            break

    _write_summary(output_dir, records, resolved_config)
    LOGGER.info(
        "Batch finished: total=%d succeeded=%d skipped=%d failed=%d summary=%s",
        len(records),
        sum(row["status"] == "success" for row in records),
        sum(row["status"] == "skipped" for row in records),
        sum(row["status"] not in COMPLETED_STATUSES for row in records),
        output_dir / "summary.json",
    )
    return output_dir, records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the Result reconstruction exploration list and summarize all outcomes."
    )
    parser.add_argument(
        "--config",
        default="configs/evaluation/evaluation_exploration.yaml",
        help="Repository-relative or absolute batch evaluation YAML.",
    )
    parser.add_argument(
        "--env",
        choices=("original", "autodl"),
        default=None,
        help=(
            "Server environment (defaults to SERVER_ENV or original); autodl maps "
            "relative Result paths to /root/autodl-tmp/Result."
        ),
    )
    parser.add_argument("--worker-request", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--worker-result", default=None, help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.worker_request or args.worker_result:
        if not args.worker_request or not args.worker_result:
            raise SystemExit("--worker-request and --worker-result must be provided together")
        raise SystemExit(
            _worker_main(Path(args.worker_request), Path(args.worker_result))
        )
    output_dir, records = run_batch(args.config, server_env=args.env)
    print(f"Evaluation batch summary: {output_dir / 'summary.json'}")
    if any(record["status"] not in COMPLETED_STATUSES for record in records):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
