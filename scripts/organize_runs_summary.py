#!/usr/bin/env python3
"""Build a cleaned, classified runs_summary tree.

The command is dry-run by default. Pass --apply to copy complete run directories
and selected log metadata into a staging directory, verify the copies, and rename
the staging directory to the final output path. Source directories are read-only.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import re
import shutil
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Iterable, Mapping, Sequence


PSNR_THRESHOLD_DB = 20.0
DEPRECATED_MODELS = {"CompactNGP", "DC-INR", "FA-TR-INR"}
METADATA_SUFFIXES = {".log", ".yaml", ".yml", ".json", ".tsv", ".csv", ".txt"}
TIMESTAMP_RE = re.compile(r"^\d{8}_\d{6}(?:_\d+)?$")
TERMINAL_PAIR_RE = re.compile(
    r"(?i)\b(?:epoch|progress)\s*[=:]?\s*(\d+)\s*/\s*(\d+)"
)
PSNR_RE = re.compile(
    r"(?im)^.*?PSNR[^\r\n]*?aggregate\s*[=:]\s*"
    r"(nan|[-+]?inf(?:inity)?|[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)"
)
NATIVE_COMPLETION_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("training-loop-finished", re.compile(r"(?i)Training loop finished:\s*steps=\d+")),
    ("training-completed", re.compile(r"(?i)Training completed(?:\.|,)")),
    ("saved-final-state", re.compile(r"(?i)Saved final state dict to\s+\S+")),
)

MODEL_ALIASES = {
    "apmgsrn": "APMGSRN",
    "compactngp": "CompactNGP",
    "coordnet": "CoordNet",
    "dcinr": "DC-INR",
    "ecnr": "ECNR",
    "fatrinr": "FA-TR-INR",
    "fvsrn": "fV-SRN",
    "instantngp": "InstantNGP",
    "instantvnr": "InstantVNR",
    "mcinr": "MC-INR",
    "moeinr": "MoE-INR",
    "mvnet": "MVNet",
    "neuralexpert": "NeuralExpert",
    "rmdsrn": "RMDSRN",
    "sharedencinr": "VarExpert",
    "siren": "SIREN",
    "varexpert": "VarExpert",
}

MANIFEST_FIELDS = (
    "record_id",
    "source_kind",
    "source_path",
    "log_path",
    "model",
    "experiment",
    "run_id",
    "experiment_group",
    "completion_reason",
    "final_psnr_db",
    "psnr_state",
    "labels",
    "destination_paths",
)
EXCLUDED_FIELDS = (
    "source_kind",
    "source_path",
    "log_path",
    "reason",
    "log_bytes",
)


@dataclass
class Record:
    record_id: str
    source_kind: str
    source_path: Path
    log_path: Path
    model: str
    experiment: str
    run_id: str
    experiment_group: str
    complete: bool
    completion_reason: str
    final_psnr: float | None
    psnr_state: str
    root: Path | None = None
    config_path: Path | None = None
    status_row: dict[str, str] | None = None
    summary_row: dict[str, str] | None = None
    exploration_version: str = ""
    batch_id: str = ""
    labels: list[str] = field(default_factory=list)
    destinations: list[Path] = field(default_factory=list)


@dataclass(frozen=True)
class Excluded:
    source_kind: str
    source_path: Path
    log_path: Path | None
    reason: str
    log_bytes: int


@dataclass
class Inventory:
    records: list[Record]
    excluded: list[Excluded]
    raw_bytes: int
    metadata_bytes: int

    @property
    def required_bytes(self) -> int:
        return self.raw_bytes + self.metadata_bytes


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _normalise_model_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.lower())


def canonical_model(value: str) -> str:
    key = _normalise_model_key(value)
    return MODEL_ALIASES.get(key, value.strip() or "Unknown")


def _safe_component(value: str, *, limit: int = 80) -> str:
    cleaned = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", value.strip())
    cleaned = re.sub(r"\s+", "_", cleaned).strip(" .") or "unknown"
    if len(cleaned) <= limit:
        return cleaned
    digest = hashlib.sha1(cleaned.encode("utf-8")).hexdigest()[:10]
    return f"{cleaned[: limit - 11]}-{digest}"


def _path_text(path: Path) -> str:
    return path.as_posix()


def _extract_simple_config(config_path: Path | None) -> dict[str, str]:
    """Extract the few scalar fields needed without requiring PyYAML."""
    if config_path is None or not config_path.is_file():
        return {}
    result: dict[str, str] = {}
    section = ""
    for line in _read_text(config_path).splitlines():
        top = re.match(r"^([A-Za-z_][\w-]*):(?:\s*(.*))?$", line)
        if top:
            key, value = top.group(1), (top.group(2) or "").strip()
            section = key.lower() if not value else ""
            if value:
                result[key.lower()] = value.strip("'\"")
            continue
        nested = re.match(r"^\s{2}([A-Za-z_][\w-]*):\s*(.*?)\s*$", line)
        if nested and section:
            result[f"{section}.{nested.group(1).lower()}"] = nested.group(2).strip("'\"")
    return result


def _find_run_config(root: Path) -> Path | None:
    candidates = (root / "configs" / "config.yaml", root / "config.yaml")
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    yaml_files = sorted(root.glob("*.yaml")) + sorted(root.glob("*.yml"))
    return yaml_files[0] if yaml_files else None


def infer_model(config: Mapping[str, str], experiment: str, source_path: Path) -> str:
    if "neural_expert" in {part.lower() for part in source_path.parts}:
        return "NeuralExpert"
    configured = config.get("model.name")
    if configured:
        return canonical_model(configured)
    haystack = f"{experiment} {_path_text(source_path)}".lower()
    candidates = sorted(MODEL_ALIASES.items(), key=lambda item: len(item[0]), reverse=True)
    normalised = _normalise_model_key(haystack)
    for key, canonical in candidates:
        if key in normalised:
            return canonical
    return "Unknown"


def infer_group(*values: str) -> str:
    combined = "/".join(values)
    return "Size" if re.search(r"(?i)(?:^|[/_-])size\d+", combined) else "Main"


def completion_from_log(text: str) -> tuple[bool, str]:
    if not text.strip():
        return False, "empty-log"
    pairs = list(TERMINAL_PAIR_RE.finditer(text))
    if pairs:
        current, total = (int(value) for value in pairs[-1].groups())
        if total > 0 and current == total:
            return True, f"terminal-count:{current}/{total}"
        return False, f"incomplete-count:{current}/{total}"
    for name, pattern in NATIVE_COMPLETION_PATTERNS:
        if pattern.search(text):
            return True, f"native:{name}"
    return False, "missing-terminal-marker"


def final_psnr_from_log(text: str) -> tuple[float | None, str]:
    matches = list(PSNR_RE.finditer(text))
    if not matches:
        return None, "missing"
    token = matches[-1].group(1)
    try:
        value = float(token)
    except ValueError:
        return None, "nonfinite"
    if not math.isfinite(value):
        return value, "nonfinite"
    return value, "finite"


def _float_from_summary(row: Mapping[str, str] | None) -> tuple[float | None, str] | None:
    if not row:
        return None
    nonfinite = (row.get("has_nonfinite") or row.get("has_nan_or_inf") or "").lower()
    if nonfinite == "true":
        return None, "nonfinite"
    token = (row.get("final_psnr") or "").strip()
    if not token:
        return None
    try:
        value = float(token)
    except ValueError:
        return None, "nonfinite"
    return (value, "finite") if math.isfinite(value) else (value, "nonfinite")


def classify_record(record: Record) -> list[str]:
    if not record.complete:
        return []
    # Deprecated is intentionally exclusive from success/fail/exploration.
    if record.model in DEPRECATED_MODELS:
        return ["deprecate"]
    if record.source_kind == "batch-exploration":
        failed_quality = record.psnr_state == "nonfinite" or (
            record.final_psnr is not None and record.final_psnr < PSNR_THRESHOLD_DB
        )
        return ["fail", "exploration"] if failed_quality else ["exploration"]
    return [
        "fail"
        if record.psnr_state == "nonfinite"
        or (record.final_psnr is not None and record.final_psnr < PSNR_THRESHOLD_DB)
        else "success"
    ]


def _run_root_from_log(log_path: Path) -> Path:
    if log_path.name == "out.log":
        return log_path.parent
    if log_path.parent.name == "logs":
        return log_path.parent.parent
    return log_path.parent


def _experiment_and_run_id(root: Path, runs_root: Path, config: Mapping[str, str]) -> tuple[str, str]:
    rel = root.relative_to(runs_root)
    experiment = config.get("exp_id", "")
    if not experiment:
        if TIMESTAMP_RE.match(root.name) and len(rel.parts) >= 2:
            experiment = rel.parts[-2]
        else:
            experiment = root.name
    run_id = root.name if TIMESTAMP_RE.match(root.name) else "direct"
    return experiment, run_id


def scan_runs(runs_root: Path) -> tuple[list[Record], list[Excluded]]:
    records: list[Record] = []
    excluded: list[Excluded] = []
    logs = sorted(runs_root.rglob("run_*.log")) + sorted(runs_root.rglob("out.log"))
    seen_roots: set[Path] = set()
    for log_path in logs:
        root = _run_root_from_log(log_path)
        if root in seen_roots:
            excluded.append(
                Excluded("runs", root, log_path, "duplicate-primary-log", log_path.stat().st_size)
            )
            continue
        seen_roots.add(root)
        text = _read_text(log_path)
        complete, reason = completion_from_log(text)
        config_path = _find_run_config(root)
        config = _extract_simple_config(config_path)
        experiment, run_id = _experiment_and_run_id(root, runs_root, config)
        rel = root.relative_to(runs_root)
        if rel.parts and rel.parts[0].lower() == "backup":
            continue
        model = infer_model(config, experiment, rel)
        psnr, psnr_state = final_psnr_from_log(text)
        record = Record(
            record_id=f"runs:{rel.as_posix()}",
            source_kind="runs",
            source_path=root,
            log_path=log_path,
            model=model,
            experiment=experiment,
            run_id=run_id,
            experiment_group=infer_group(experiment, rel.as_posix(), config.get("config_path", "")),
            complete=complete,
            completion_reason=reason,
            final_psnr=psnr,
            psnr_state=psnr_state,
            root=root,
            config_path=config_path,
        )
        if complete:
            # Runs-side exploration is retained in raw; batch_logs is authoritative
            # for exploration quality classifications. Deprecated remains an
            # exclusive model-level classification regardless of source tree.
            is_exploration = bool(
                re.search(r"(?i)(?:^|/)exploration(?:_v\d+)?(?:/|$)", rel.as_posix())
            )
            if is_exploration and record.model in DEPRECATED_MODELS:
                record.labels = ["deprecate"]
            elif not is_exploration:
                record.labels = classify_record(record)
            records.append(record)
        else:
            excluded.append(
                Excluded("runs", root, log_path, reason, log_path.stat().st_size)
            )

    for auxiliary in (runs_root / "visualizations",):
        if auxiliary.exists():
            excluded.append(Excluded("runs", auxiliary, None, "auxiliary-no-training-log", 0))
    for child in sorted(runs_root.iterdir()):
        if child.is_dir() and child.name.lower().startswith("exploration"):
            if not any(child.rglob("run_*.log")):
                excluded.append(Excluded("runs", child, None, "empty-experiment-directory", 0))
    return records, excluded


def _read_tsv(path: Path) -> list[dict[str, str]]:
    if not path.is_file() or path.stat().st_size == 0:
        return []
    with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def _summary_by_config(batch_root: Path) -> dict[str, dict[str, str]]:
    path = batch_root / "exploration_summary.tsv"
    return {row.get("config", "").replace("\\", "/"): row for row in _read_tsv(path)}


def _batch_config_parts(config_key: str) -> tuple[str, str, str]:
    parts = PurePosixPath(config_key.replace("\\", "/")).parts
    model_index = 1 if len(parts) > 1 else 0
    raw_model = parts[model_index] if parts else "Unknown"
    model = canonical_model(raw_model)
    tail = list(parts[model_index + 1 :])
    if tail:
        tail[-1] = PurePosixPath(tail[-1]).stem
    experiment = "__".join(tail) or PurePosixPath(config_key).stem
    return model, experiment, infer_group(config_key)


def scan_batch_exploration(repo_root: Path, batch_logs_root: Path) -> tuple[list[Record], list[Excluded]]:
    records: list[Record] = []
    excluded: list[Excluded] = []
    referenced_logs: set[Path] = set()
    version_dirs = sorted(
        path for path in batch_logs_root.iterdir()
        if path.is_dir() and re.fullmatch(r"exploration(?:_v\d+)?", path.name, re.I)
    )
    for version_dir in version_dirs:
        for batch_root in sorted(path for path in version_dir.iterdir() if path.is_dir()):
            status_path = batch_root / "status.tsv"
            statuses = _read_tsv(status_path)
            final_by_config: dict[str, dict[str, str]] = {}
            for row in statuses:
                config_key = (row.get("config") or "").replace("\\", "/")
                if config_key:
                    final_by_config[config_key] = row
            summaries = _summary_by_config(batch_root)
            for config_key, status_row in sorted(final_by_config.items()):
                log_name = PurePosixPath((status_row.get("log") or "").replace("\\", "/")).name
                log_path = batch_root / "logs" / log_name
                if log_path.is_file():
                    referenced_logs.add(log_path.resolve())
                    text = _read_text(log_path)
                    log_bytes = log_path.stat().st_size
                else:
                    text = ""
                    log_bytes = 0
                final_status = (status_row.get("status") or "").lower()
                complete = final_status == "ok" and bool(text.strip())
                if not text.strip():
                    reason = "empty-or-missing-log"
                elif final_status != "ok":
                    reason = f"batch-status:{final_status or 'missing'}"
                else:
                    terminal, terminal_reason = completion_from_log(text)
                    reason = terminal_reason if terminal else "batch-status:ok"
                summary_row = summaries.get(config_key)
                summary_psnr = _float_from_summary(summary_row)
                if summary_psnr is None:
                    psnr, psnr_state = final_psnr_from_log(text)
                else:
                    psnr, psnr_state = summary_psnr
                model, experiment, group = _batch_config_parts(config_key)
                local_config = repo_root / Path(*PurePosixPath(config_key).parts)
                record = Record(
                    record_id=f"batch:{version_dir.name}/{batch_root.name}:{config_key}",
                    source_kind="batch-exploration",
                    source_path=batch_root,
                    log_path=log_path,
                    model=model,
                    experiment=experiment,
                    run_id=batch_root.name,
                    experiment_group=group,
                    complete=complete,
                    completion_reason=reason,
                    final_psnr=psnr,
                    psnr_state=psnr_state,
                    config_path=local_config if local_config.is_file() else None,
                    status_row=status_row,
                    summary_row=summary_row,
                    exploration_version=version_dir.name,
                    batch_id=batch_root.name,
                )
                if complete:
                    record.labels = classify_record(record)
                    records.append(record)
                else:
                    excluded.append(
                        Excluded("batch-exploration", batch_root, log_path, reason, log_bytes)
                    )

            logs_dir = batch_root / "logs"
            if logs_dir.is_dir():
                for log_path in sorted(logs_dir.glob("*.log")):
                    if log_path.resolve() not in referenced_logs:
                        reason = "unreferenced-empty-log" if log_path.stat().st_size == 0 else "no-final-status"
                        excluded.append(
                            Excluded(
                                "batch-exploration",
                                batch_root,
                                log_path,
                                reason,
                                log_path.stat().st_size,
                            )
                        )
    return records, excluded


def _iter_files(root: Path) -> Iterable[Path]:
    for path in root.rglob("*"):
        if path.is_file():
            yield path


def _tree_bytes(root: Path) -> int:
    return sum(path.stat().st_size for path in _iter_files(root))


def _metadata_sources(record: Record) -> list[Path]:
    if record.source_kind == "runs" and record.root is not None:
        return [path for path in _iter_files(record.root) if path.suffix.lower() in METADATA_SUFFIXES]
    sources = [record.log_path] if record.log_path.is_file() else []
    if record.config_path is not None and record.config_path.is_file():
        sources.append(record.config_path)
    return sources


def destination_for(record: Record, label: str, output_root: Path, runs_root: Path) -> Path:
    model = _safe_component(record.model)
    experiment = _safe_component(record.experiment)
    run_id = _safe_component(record.run_id)
    if label in {"success", "fail"}:
        return output_root / label / record.experiment_group / model / experiment / run_id
    if label == "deprecate":
        return output_root / label / model / record.experiment_group / experiment / run_id
    if label == "exploration":
        return (
            output_root
            / label
            / _safe_component(record.exploration_version)
            / _safe_component(record.batch_id)
            / model
            / experiment
            / run_id
        )
    if label == "raw" and record.root is not None:
        return output_root / "raw" / record.root.relative_to(runs_root)
    raise ValueError(f"Unsupported destination label: {label}")


def build_inventory(repo_root: Path, runs_root: Path, batch_logs_root: Path, output_root: Path) -> Inventory:
    run_records, run_excluded = scan_runs(runs_root)
    batch_records, batch_excluded = scan_batch_exploration(repo_root, batch_logs_root)
    records = run_records + batch_records
    raw_bytes = 0
    metadata_bytes = 0
    for record in records:
        record.destinations.clear()
        if record.source_kind == "runs" and record.root is not None:
            raw_destination = destination_for(record, "raw", output_root, runs_root)
            record.destinations.append(raw_destination)
            raw_bytes += _tree_bytes(record.root)
        metadata_size = sum(path.stat().st_size for path in _metadata_sources(record))
        for label in record.labels:
            record.destinations.append(destination_for(record, label, output_root, runs_root))
            metadata_bytes += metadata_size
    catalog = runs_root / "model_size_catalog.csv"
    if catalog.is_file():
        raw_bytes += catalog.stat().st_size
    return Inventory(records, run_excluded + batch_excluded, raw_bytes, metadata_bytes)


def _copy_metadata(record: Record, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=False)
    if record.source_kind == "runs" and record.root is not None:
        for source in _metadata_sources(record):
            relative = source.relative_to(record.root)
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
        return
    if record.log_path.is_file():
        target = destination / "logs" / record.log_path.name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(record.log_path, target)
    if record.config_path is not None and record.config_path.is_file():
        target = destination / "configs" / record.config_path.name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(record.config_path, target)
    if record.status_row:
        _write_tsv(destination / "status.tsv", [record.status_row], tuple(record.status_row))
    if record.summary_row:
        _write_tsv(destination / "exploration_summary.tsv", [record.summary_row], tuple(record.summary_row))


def _write_tsv(path: Path, rows: Sequence[Mapping[str, object]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def _manifest_row(record: Record, output_root: Path, repo_root: Path) -> dict[str, object]:
    psnr = ""
    if record.final_psnr is not None:
        psnr = str(record.final_psnr)
    return {
        "record_id": record.record_id,
        "source_kind": record.source_kind,
        "source_path": _display_path(record.source_path, repo_root),
        "log_path": _display_path(record.log_path, repo_root),
        "model": record.model,
        "experiment": record.experiment,
        "run_id": record.run_id,
        "experiment_group": record.experiment_group,
        "completion_reason": record.completion_reason,
        "final_psnr_db": psnr,
        "psnr_state": record.psnr_state,
        "labels": ";".join(record.labels),
        "destination_paths": ";".join(_display_path(path, output_root.parent) for path in record.destinations),
    }


def _display_path(path: Path, base: Path) -> str:
    try:
        return path.resolve().relative_to(base.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _write_manifests(stage: Path, inventory: Inventory, repo_root: Path, final_output: Path) -> None:
    manifest_rows = [_manifest_row(record, final_output, repo_root) for record in inventory.records]
    _write_tsv(stage / "classification_manifest.tsv", manifest_rows, MANIFEST_FIELDS)
    excluded_rows = [
        {
            "source_kind": item.source_kind,
            "source_path": _display_path(item.source_path, repo_root),
            "log_path": _display_path(item.log_path, repo_root) if item.log_path else "",
            "reason": item.reason,
            "log_bytes": item.log_bytes,
        }
        for item in inventory.excluded
    ]
    _write_tsv(stage / "excluded_manifest.tsv", excluded_rows, EXCLUDED_FIELDS)


def _write_filtered_exploration_tables(stage: Path, records: Sequence[Record]) -> None:
    grouped: dict[tuple[str, str], list[Record]] = defaultdict(list)
    for record in records:
        if "exploration" in record.labels:
            grouped[(record.exploration_version, record.batch_id)].append(record)
    for (version, batch_id), selected in grouped.items():
        batch_dest = stage / "exploration" / _safe_component(version) / _safe_component(batch_id)
        status_rows = [record.status_row for record in selected if record.status_row]
        if status_rows:
            _write_tsv(batch_dest / "status.tsv", status_rows, tuple(status_rows[0]))
        summary_rows = [record.summary_row for record in selected if record.summary_row]
        if summary_rows:
            _write_tsv(
                batch_dest / "exploration_summary.tsv",
                summary_rows,
                tuple(summary_rows[0]),
            )


def _write_readme(stage: Path, inventory: Inventory) -> None:
    label_counts = Counter(label for record in inventory.records for label in record.labels)
    complete_runs = sum(record.source_kind == "runs" for record in inventory.records)
    complete_exploration = sum(record.source_kind == "batch-exploration" for record in inventory.records)
    content = f"""# Runs Summary

Generated from `runs` and `batch_logs` without modifying either source.

- Complete run directories in `raw`: {complete_runs}
- Complete batch exploration logs: {complete_exploration}
- Success metadata records: {label_counts['success']}
- Fail metadata records: {label_counts['fail']}
- Deprecated metadata records: {label_counts['deprecate']}
- Exploration metadata records: {label_counts['exploration']}
- Excluded records: {len(inventory.excluded)}
- PSNR failure threshold: strictly below {PSNR_THRESHOLD_DB:g} dB; NaN/Inf also fails

Deprecated models (`CompactNGP`, `DC-INR`, `FA-TR-INR`) are exclusive: they are
stored only under `deprecate` among the derived categories. Complete runs still
remain under `raw`. The source `runs/backup` tree is ignored completely.

Completed exploration records are stored under `exploration`, never `success`.
An exploration record below the PSNR threshold (or with NaN/Inf) is additionally
stored under `fail`.

See `classification_manifest.tsv` and `excluded_manifest.tsv` for the audit trail.
"""
    (stage / "README.md").write_text(content, encoding="utf-8")


def _sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_tree(source: Path, target: Path) -> tuple[int, int]:
    source_files = {path.relative_to(source): path for path in _iter_files(source)}
    target_files = {path.relative_to(target): path for path in _iter_files(target)}
    if set(source_files) != set(target_files):
        missing = sorted(str(path) for path in set(source_files) - set(target_files))[:5]
        extra = sorted(str(path) for path in set(target_files) - set(source_files))[:5]
        raise RuntimeError(f"Raw tree mismatch for {source}: missing={missing}, extra={extra}")
    total_bytes = 0
    for relative, source_file in source_files.items():
        target_file = target_files[relative]
        source_size = source_file.stat().st_size
        if source_size != target_file.stat().st_size:
            raise RuntimeError(f"Size mismatch: {source_file} -> {target_file}")
        if _sha256(source_file) != _sha256(target_file):
            raise RuntimeError(f"SHA-256 mismatch: {source_file} -> {target_file}")
        total_bytes += source_size
    return len(source_files), total_bytes


def _selected_source_files(records: Sequence[Record], catalog: Path) -> set[Path]:
    """Return only source files that contribute to the generated summary."""
    selected: set[Path] = set()
    for record in records:
        if record.source_kind == "runs" and record.root is not None:
            selected.update(_iter_files(record.root))
        else:
            selected.update(_metadata_sources(record))
    if catalog.is_file():
        selected.add(catalog)
    return selected


def _snapshot_selected_sources(
    records: Sequence[Record], catalog: Path
) -> dict[Path, tuple[int, int]]:
    snapshot: dict[Path, tuple[int, int]] = {}
    for path in _selected_source_files(records, catalog):
        stat = path.stat()
        snapshot[path] = (stat.st_size, stat.st_mtime_ns)
    return snapshot


def _assert_selected_sources_unchanged(
    snapshot: Mapping[Path, tuple[int, int]], records: Sequence[Record], catalog: Path
) -> None:
    current_paths = _selected_source_files(records, catalog)
    previous_paths = set(snapshot)
    if current_paths != previous_paths:
        added = sorted(str(path) for path in current_paths - previous_paths)[:5]
        removed = sorted(str(path) for path in previous_paths - current_paths)[:5]
        raise RuntimeError(f"Source file set changed: added={added}, removed={removed}")
    for path, before in snapshot.items():
        stat = path.stat()
        after = (stat.st_size, stat.st_mtime_ns)
        if after != before:
            raise RuntimeError(f"Source changed during organization: {path}")


def _inventory_signature(inventory: Inventory) -> tuple[tuple[object, ...], ...]:
    """Stable representation of all completed records that affect output."""
    return tuple(
        sorted(
            (
                record.record_id,
                record.source_kind,
                str(record.source_path),
                str(record.log_path),
                record.model,
                record.experiment,
                record.run_id,
                record.experiment_group,
                record.completion_reason,
                record.final_psnr,
                record.psnr_state,
                tuple(record.labels),
            )
            for record in inventory.records
        )
    )


def _copy_and_verify(
    repo_root: Path,
    runs_root: Path,
    batch_logs_root: Path,
    output_root: Path,
    inventory: Inventory,
    *,
    verify_hashes: bool,
) -> None:
    stage = output_root.with_name(f"{output_root.name}.staging")
    if output_root.exists():
        raise FileExistsError(f"Output already exists: {output_root}")
    if stage.exists():
        raise FileExistsError(f"Staging directory already exists: {stage}")
    free = shutil.disk_usage(output_root.parent).free
    required_with_margin = int(inventory.required_bytes * 1.05)
    if free < required_with_margin:
        raise OSError(
            f"Insufficient free space: need {required_with_margin} bytes including margin, have {free}"
        )
    catalog = runs_root / "model_size_catalog.csv"
    snapshot = _snapshot_selected_sources(inventory.records, catalog)
    stage.mkdir(parents=False)
    complete_runs = [record for record in inventory.records if record.source_kind == "runs"]
    try:
        for index, record in enumerate(complete_runs, start=1):
            assert record.root is not None
            target = destination_for(record, "raw", stage, runs_root)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(record.root, target, copy_function=shutil.copy2)
            print(f"raw {index}/{len(complete_runs)}: {record.root}", flush=True)
        if catalog.is_file():
            raw_root = stage / "raw"
            raw_root.mkdir(parents=True, exist_ok=True)
            shutil.copy2(catalog, raw_root / catalog.name)

        metadata_records = [record for record in inventory.records if record.labels]
        for index, record in enumerate(metadata_records, start=1):
            for label in record.labels:
                target = destination_for(record, label, stage, runs_root)
                _copy_metadata(record, target)
            if index % 25 == 0 or index == len(metadata_records):
                print(f"metadata {index}/{len(metadata_records)}", flush=True)

        _write_filtered_exploration_tables(stage, inventory.records)
        _write_manifests(stage, inventory, repo_root, output_root)
        _write_readme(stage, inventory)

        if verify_hashes:
            verified_files = 0
            verified_bytes = 0
            for index, record in enumerate(complete_runs, start=1):
                assert record.root is not None
                target = destination_for(record, "raw", stage, runs_root)
                count, byte_count = _verify_tree(record.root, target)
                verified_files += count
                verified_bytes += byte_count
                if index % 10 == 0 or index == len(complete_runs):
                    print(f"verify {index}/{len(complete_runs)}", flush=True)
            print(
                f"verified raw files={verified_files} bytes={verified_bytes}",
                flush=True,
            )
        _assert_selected_sources_unchanged(snapshot, inventory.records, catalog)
        final_inventory = build_inventory(repo_root, runs_root, batch_logs_root, output_root)
        if _inventory_signature(final_inventory) != _inventory_signature(inventory):
            raise RuntimeError(
                "Completed-run inventory changed during organization; rerun to include the new state"
            )
        stage.rename(output_root)
    except Exception:
        print(f"Organization failed; partial staging retained at {stage}", file=sys.stderr, flush=True)
        raise


def _print_summary(inventory: Inventory) -> None:
    counts = Counter(label for record in inventory.records for label in record.labels)
    complete_runs = sum(record.source_kind == "runs" for record in inventory.records)
    complete_batch = sum(record.source_kind == "batch-exploration" for record in inventory.records)
    print(f"complete_runs={complete_runs}")
    print(f"complete_batch_exploration={complete_batch}")
    print(f"success={counts['success']}")
    print(f"fail={counts['fail']}")
    print(f"deprecate={counts['deprecate']}")
    print(f"exploration={counts['exploration']}")
    print(f"excluded={len(inventory.excluded)}")
    print(f"raw_bytes={inventory.raw_bytes}")
    print(f"metadata_bytes_estimate={inventory.metadata_bytes}")
    print(f"required_bytes_estimate={inventory.required_bytes}")


def organize(
    repo_root: Path,
    *,
    runs: Path | None = None,
    batch_logs: Path | None = None,
    output: Path | None = None,
    apply: bool = False,
    verify_hashes: bool = True,
) -> Inventory:
    repo_root = repo_root.resolve()
    runs_root = (runs or repo_root / "runs").resolve()
    batch_logs_root = (batch_logs or repo_root / "batch_logs").resolve()
    output_root = (output or repo_root / "runs_summary").resolve()
    if not runs_root.is_dir():
        raise FileNotFoundError(f"Runs directory not found: {runs_root}")
    if not batch_logs_root.is_dir():
        raise FileNotFoundError(f"Batch logs directory not found: {batch_logs_root}")
    inventory = build_inventory(repo_root, runs_root, batch_logs_root, output_root)
    _print_summary(inventory)
    if apply:
        _copy_and_verify(
            repo_root,
            runs_root,
            batch_logs_root,
            output_root,
            inventory,
            verify_hashes=verify_hashes,
        )
        print(f"created={output_root}", flush=True)
    return inventory


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--runs", type=Path)
    parser.add_argument("--batch-logs", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--apply", action="store_true", help="Create and verify runs_summary")
    parser.add_argument(
        "--no-hash-verify",
        action="store_true",
        help="Verify paths and sizes but skip SHA-256 (not recommended)",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    organize(
        args.repo_root,
        runs=args.runs,
        batch_logs=args.batch_logs,
        output=args.output,
        apply=args.apply,
        verify_hashes=not args.no_hash_verify,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
