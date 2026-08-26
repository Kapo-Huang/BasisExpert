from __future__ import annotations

import argparse
import csv
import importlib
import json
import logging
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
from var_expert_inr.evaluation.reporting import (
    evaluation_output_dir,
    path_fingerprint,
    write_json,
    write_metrics_csv,
)
from var_expert_inr.evaluation.service import evaluate_run, resolve_run_config


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


def _read_run_list(path: Path) -> list[Path]:
    runs: list[Path] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        runs.append(_resolve_repo_path(line))
    return runs


def _discover_checkpoint_runs(root: Path) -> list[Path]:
    """Find archived runs that can be evaluated from their local checkpoint."""
    if not root.is_dir():
        raise FileNotFoundError(f"Evaluation run root does not exist: {root}")
    runs: list[Path] = []
    for config_path in sorted(root.rglob("configs/config.yaml")):
        run_dir = config_path.parent.parent
        checkpoint_dir = run_dir / "checkpoints"
        if any(checkpoint_dir.glob("*.pth")):
            runs.append(run_dir.resolve())
    return runs


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


def _configure_logging(log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(log_path, encoding="utf-8")],
        force=True,
    )


def _write_summary(output_dir: Path, records: list[dict[str, Any]], config_path: Path) -> None:
    succeeded = sum(row["status"] == "success" for row in records)
    skipped = sum(row["status"] == "skipped" for row in records)
    completed = succeeded + skipped
    summary = {
        "schema_version": 1,
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


def _explicit_timesteps(value: str) -> tuple[int, ...] | None:
    tokens = [token.strip() for token in value.split(",") if token.strip()]
    if not tokens or any(":" in token or token.lower() == "all" for token in tokens):
        return None
    try:
        return tuple(int(token) for token in tokens)
    except ValueError:
        return None


def _requested_target_names(raw: dict[str, Any], target: str) -> tuple[str, ...]:
    if target != "all":
        return (target,)
    data = _section(raw, "data", "DATA")
    if data.get("target"):
        return (str(data["target"]),)
    configured = data.get("targets")
    return tuple(str(name) for name in configured) if isinstance(configured, dict) else ()


def _existing_evaluation_state(
    run_dir: Path,
    *,
    raw: dict[str, Any],
    target: str,
    timesteps: str,
    requested_metrics: tuple[str, ...],
    render: bool,
    current_profile_fingerprint: str | None,
) -> dict[str, Any] | None:
    output_dir = evaluation_output_dir(run_dir, repo_root=REPO_ROOT)
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
    if payload.get("status") != "complete" or manifest.get("source_kind") != "checkpoint":
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
    if stored_source_fingerprint is not None and stored_source_fingerprint != path_fingerprint(source_path):
        return None

    selected_timesteps = _explicit_timesteps(timesteps)
    if selected_timesteps is None or tuple(manifest.get("timesteps") or ()) != selected_timesteps:
        return None
    requested_targets = _requested_target_names(raw, target)
    stored_targets = tuple(str(name) for name in (manifest.get("targets") or ()))
    if requested_targets and set(stored_targets) != set(requested_targets):
        return None

    rows = list(payload.get("per_timestep") or ())
    expected = {(name, step) for name in stored_targets for step in selected_timesteps}
    rows_by_key = {
        (str(row.get("target")), int(row.get("timestep"))): row
        for row in rows
        if row.get("target") is not None and row.get("timestep") is not None
    }
    completed_metrics: set[str] = set()
    for metric in requested_metrics:
        if metric in {"decode_time", "memory"}:
            if metric == "decode_time" and payload.get("performance", {}).get("total_decode_seconds") is not None:
                completed_metrics.add(metric)
            elif metric == "memory" and payload.get("performance", {}).get("peak_memory_bytes") is not None:
                completed_metrics.add(metric)
        elif expected and all(rows_by_key.get(key, {}).get(metric) is not None for key in expected):
            completed_metrics.add(metric)

    render_complete = False
    if render and expected and bool(manifest.get("render_requested")):
        stored_profile = (manifest.get("render_profile") or {}).get("fingerprint")
        render_complete = stored_profile == current_profile_fingerprint and all(
            Path(str(rows_by_key.get(key, {}).get("pred_render_path", ""))).is_file()
            and Path(str(rows_by_key.get(key, {}).get("gt_render_path", ""))).is_file()
            for key in expected
        )
    return {
        "output_dir": output_dir,
        "manifest": manifest,
        "metrics": payload,
        "completed_metrics": completed_metrics,
        "render_complete": render_complete,
    }


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
        "schema_version": 1,
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


def run_batch(config_path: str | Path) -> tuple[Path, list[dict[str, Any]]]:
    resolved_config = _resolve_repo_path(config_path)
    batch = _load_mapping(resolved_config)
    evaluation = batch.get("evaluation")
    if not isinstance(evaluation, dict):
        raise ValueError(f"Missing evaluation mapping in {resolved_config}")

    list_path_value = batch.get("list_path")
    run_root_value = batch.get("run_root")
    if list_path_value is not None and run_root_value is not None:
        raise ValueError("Specify either list_path or run_root, not both")
    if list_path_value is None and run_root_value is None:
        raise ValueError("Missing list_path or run_root")
    list_path = None if list_path_value is None else _resolve_repo_path(list_path_value)
    run_root = None if run_root_value is None else _resolve_repo_path(run_root_value)
    summary_root = _resolve_repo_path(
        batch.get("summary_root", "batch_logs/evaluation_exploration")
    )
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
    profile_value = evaluation.get("render_profile", "auto")
    render_profile = (
        None
        if profile_value in (None, "", "auto")
        else _resolve_repo_path(profile_value)
    )
    targets_by_dataset = _normalized_target_map(
        evaluation.get("targets_by_dataset")
    )
    run_dirs = _read_run_list(list_path) if list_path is not None else _discover_checkpoint_runs(run_root)
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
    run_source = list_path if list_path is not None else run_root
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
            "error_type": "",
            "error": "",
        }
        started = time.perf_counter()
        existing_state: dict[str, Any] | None = None
        try:
            model_name, dataset, _, raw = _run_identity(run_dir)
            target = targets_by_dataset.get(dataset.lower(), "all")
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
                    timesteps=timesteps,
                    requested_metrics=metrics,
                    render=render,
                    current_profile_fingerprint=current_profile_fingerprint,
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
            if existing_state is not None and not pending_metrics and not pending_render:
                existing_output = Path(existing_state["output_dir"])
                record.update(
                    status="skipped",
                    output_dir=str(existing_output.resolve()),
                    metrics_path=str((existing_output / "metrics.json").resolve()),
                    log_path=str((existing_output / "logs" / "evaluate.log").resolve()),
                    return_code=0,
                )
                LOGGER.info(
                    "[%d/%d] Skipped completed checkpoint: %s",
                    index,
                    len(run_dirs),
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
                        "metrics": list(pending_metrics),
                        "timesteps": timesteps,
                        "target": target,
                        "source": source,
                        "render": worker_render,
                        "render_profile": None if render_profile is None else str(render_profile),
                        "overwrite": overwrite,
                        "device": device,
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
    output_dir, records = run_batch(args.config)
    print(f"Evaluation batch summary: {output_dir / 'summary.json'}")
    if any(record["status"] not in COMPLETED_STATUSES for record in records):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
