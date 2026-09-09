"""Inspect, migrate, verify, and prune schema-v2 EvalResult trees."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from var_expert_inr.config.io import load_evaluation_experiment_config
from var_expert_inr.evaluation.artifacts import ArtifactStore, LAYOUT_SCHEMA_VERSION
from var_expert_inr.evaluation.data_paths import normalize_experiment_data_paths
from var_expert_inr.evaluation.ground_truth import target_paths_from_config
from var_expert_inr.evaluation.rendering import load_render_profile, profile_fingerprint
from var_expert_inr.evaluation.standalone import _target_paths


LEGACY_NAMESPACES = {
    "EvalResultMeric": "legacy_metrics",
    "EvalResultFig": "legacy_figures",
    "EvalResultErr": "legacy_errors",
    "EvalResultDependency": "legacy_dependency",
}
RENDER_FIELDS = {
    "gt_render_path": "ground_truth",
    "pred_render_path": "prediction",
    "error_render_path": "error",
}


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.migration.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _legacy_roots(result_root: Path) -> list[tuple[Path, str]]:
    found: list[tuple[Path, str]] = []
    for name, namespace in LEGACY_NAMESPACES.items():
        for candidate in (result_root / name, REPO_ROOT / name):
            if candidate.is_dir() and all(candidate.resolve() != path.resolve() for path, _ in found):
                found.append((candidate.resolve(), namespace))
    direct_markers = {"Main", "RD Curve", "Ablation", "Scaling", "Sensitivity", "_external"}
    if any((result_root / name).is_dir() for name in direct_markers):
        found.append((result_root.resolve(), "legacy_metrics"))
    return found


def _manifest_dirs(root: Path) -> list[Path]:
    excluded = {"GroundTruth", "artifacts", "evaluations", "batches", "migration"}
    return sorted(
        path.parent
        for path in root.rglob("manifest.json")
        if not any(part in excluded for part in path.relative_to(root).parts)
    )


def _current_target_paths(manifest: dict[str, Any], config_path: Path) -> dict[str, Path]:
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    try:
        config = normalize_experiment_data_paths(
            load_evaluation_experiment_config(config_path),
            repo_root=REPO_ROOT,
        )
        return target_paths_from_config(config.data, repo_root=REPO_ROOT)
    except (KeyError, TypeError, ValueError):
        return _target_paths(raw, repo_root=REPO_ROOT, config_path=config_path)


def _actual_legacy_frame(output_dir: Path, row: dict[str, Any], field: str) -> Path | None:
    value = row.get(field)
    if not value:
        return None
    filename = Path(str(value)).name
    target = str(row.get("target") or "target")
    expected = output_dir / "renders" / target / filename
    if expected.is_file():
        return expected
    original = Path(str(value)).expanduser()
    if original.is_file():
        return original.resolve()
    matches = list((output_dir / "renders").rglob(filename)) if (output_dir / "renders").is_dir() else []
    return matches[0].resolve() if len(matches) == 1 else None


def _copy_non_render_files(source: Path, destination: Path) -> None:
    for path in source.rglob("*"):
        relative = path.relative_to(source)
        if relative.parts and relative.parts[0] == "renders":
            continue
        target = destination / relative
        if path.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        elif path.name not in {"manifest.json", "metrics.json"}:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)


def _import_row_artifacts(
    *,
    store: ArtifactStore,
    output_dir: Path,
    row: dict[str, Any],
    manifest: dict[str, Any],
    profile: dict[str, Any],
    gt_paths: dict[str, Path],
    mappings: list[dict[str, str]],
) -> dict[str, Any]:
    migrated = dict(row)
    target = str(row.get("target") or "target")
    timestep = int(row["timestep"])
    dataset = str(manifest.get("dataset_name") or "unknown")
    model = str(manifest.get("model") or manifest.get("subsystem") or "unknown")
    source_path = Path(str(manifest.get("source_path") or "")).expanduser()
    gt_fingerprint = store.content_fingerprint(gt_paths[target]) if target in gt_paths else None
    records: dict[str, Any] = {}

    gt_source = _actual_legacy_frame(output_dir, row, "gt_render_path")
    if gt_source is not None:
        if target not in gt_paths:
            raise KeyError(f"No ground-truth source for target {target!r}")
        spec = store.ground_truth_spec(
            dataset=dataset,
            target=target,
            timestep=timestep,
            ground_truth_path=gt_paths[target],
            profile=profile,
        )
        records["gt_render_path"] = store.import_existing(
            spec, gt_source, allow_semantic_variant=True
        )

    pred_source = _actual_legacy_frame(output_dir, row, "pred_render_path")
    if pred_source is not None:
        if not source_path.exists():
            raise FileNotFoundError(f"Prediction source is unavailable: {source_path}")
        spec = store.prediction_spec(
            dataset=dataset,
            model=model,
            target=target,
            timestep=timestep,
            source_path=source_path,
            profile=profile,
            ground_truth_fingerprint=gt_fingerprint,
        )
        records["pred_render_path"] = store.import_existing(
            spec, pred_source, allow_semantic_variant=True
        )

    error_source = _actual_legacy_frame(output_dir, row, "error_render_path")
    if error_source is not None:
        if not source_path.exists() or gt_fingerprint is None:
            raise FileNotFoundError("Error artifact inputs are unavailable")
        error = manifest.get("error_analysis") or {}
        spec = store.error_spec(
            dataset=dataset,
            model=model,
            target=target,
            timestep=timestep,
            source_path=source_path,
            ground_truth_fingerprint=gt_fingerprint,
            profile=profile,
            error_vmin=float(error.get("error_vmin", 0.0)),
            error_vmax=float(error.get("error_vmax", 5.0)),
        )
        records["error_render_path"] = store.import_existing(
            spec, error_source, allow_semantic_variant=True
        )

    for field, record in records.items():
        old = str(row.get(field) or "")
        new = store.relative(record.path)
        migrated[field] = new
        info_field = "render_info" if field == "pred_render_path" else field.replace("_path", "_info")
        migrated[info_field] = store.describe(record)
        mappings.append(
            {
                "source": old,
                "artifact": new,
                "kind": RENDER_FIELDS[field],
                "semantic_variant": str(bool(record.render_info.get("semantic_variant"))).lower(),
            }
        )
    return migrated


def _migrate_manifest(
    *,
    store: ArtifactStore,
    source_dir: Path,
    legacy_root: Path,
    namespace: str,
    mappings: list[dict[str, str]],
) -> tuple[Path, int]:
    manifest = _read_json(source_dir / "manifest.json")
    metrics_path = source_dir / "metrics.json"
    metrics = _read_json(metrics_path) if metrics_path.is_file() else {"status": "complete", "per_timestep": []}
    relative = source_dir.relative_to(legacy_root)
    if relative.parts and relative.parts[0] == "batch_logs":
        raise ValueError("Batch logs are not evaluation manifests")
    destination = store.root / "evaluations" / namespace / relative
    rows = list(metrics.get("per_timestep") or [])
    render_rows = [row for row in rows if isinstance(row, dict) and any(row.get(field) for field in RENDER_FIELDS)]
    if render_rows:
        config_path = Path(str(manifest.get("config_path") or "")).expanduser()
        if not config_path.is_file():
            raise FileNotFoundError(f"Evaluation config is unavailable: {config_path}")
        profile = load_render_profile(str(manifest.get("dataset_name") or ""), None, repo_root=REPO_ROOT)
        stored_profile = str((manifest.get("render_profile") or {}).get("fingerprint") or "")
        if stored_profile and stored_profile != profile_fingerprint(profile):
            raise ValueError(f"Render profile changed for {source_dir}")
        gt_paths = _current_target_paths(manifest, config_path)
        migrated_rows = [
            _import_row_artifacts(
                store=store,
                output_dir=source_dir,
                row=row,
                manifest=manifest,
                profile=profile,
                gt_paths=gt_paths,
                mappings=mappings,
            )
            if isinstance(row, dict) else row
            for row in rows
        ]
    else:
        migrated_rows = rows
    migrated_metrics = dict(metrics)
    migrated_metrics["schema_version"] = LAYOUT_SCHEMA_VERSION
    migrated_metrics["per_timestep"] = migrated_rows
    migrated_manifest = dict(manifest)
    migrated_manifest.update(
        {
            "schema_version": LAYOUT_SCHEMA_VERSION,
            "evaluation_id": namespace,
            "result_root": ".",
            "artifact_root": "artifacts",
        }
    )
    migrated_manifest.pop("ground_truth_render_cache", None)
    _copy_non_render_files(source_dir, destination)
    _write_json(destination / "manifest.json", migrated_manifest)
    _write_json(destination / "metrics.json", migrated_metrics)
    return destination, len(render_rows)


def verify(result_root: Path) -> dict[str, Any]:
    errors: list[str] = []
    manifests = 0
    references = 0
    for manifest_path in sorted((result_root / "evaluations").rglob("manifest.json")):
        manifests += 1
        try:
            manifest = _read_json(manifest_path)
            if int(manifest.get("schema_version", 0)) != LAYOUT_SCHEMA_VERSION:
                errors.append(f"schema:{manifest_path}")
            metrics_path = manifest_path.with_name("metrics.json")
            metrics = _read_json(metrics_path)
            for row in metrics.get("per_timestep") or []:
                if not isinstance(row, dict):
                    continue
                for field in RENDER_FIELDS:
                    if not row.get(field):
                        continue
                    references += 1
                    value = str(row[field])
                    if Path(value).is_absolute() or not value.startswith("artifacts/"):
                        errors.append(f"reference-format:{metrics_path}:{field}:{value}")
                    elif not (result_root / value).is_file():
                        errors.append(f"reference-missing:{metrics_path}:{field}:{value}")
            if (manifest_path.parent / "renders").exists():
                errors.append(f"local-renders:{manifest_path.parent}")
        except Exception as exc:
            errors.append(f"unreadable:{manifest_path}:{type(exc).__name__}:{exc}")
    return {"schema_version": LAYOUT_SCHEMA_VERSION, "manifests": manifests, "references": references, "errors": errors}


def status(result_root: Path) -> dict[str, Any]:
    rows: dict[tuple[str, str], dict[str, int]] = defaultdict(lambda: {"files": 0, "bytes": 0})
    artifacts = result_root / "artifacts"
    for path in artifacts.rglob("*.png") if artifacts.is_dir() else ():
        relative = path.relative_to(artifacts)
        kind = relative.parts[0] if relative.parts else "unknown"
        dataset = relative.parts[1] if len(relative.parts) > 1 else "unknown"
        rows[(kind, dataset)]["files"] += 1
        rows[(kind, dataset)]["bytes"] += path.stat().st_size
    coverage = [
        {"kind": kind, "dataset": dataset, **values}
        for (kind, dataset), values in sorted(rows.items())
    ]
    verification = verify(result_root)
    return {"schema_version": LAYOUT_SCHEMA_VERSION, "result_root": str(result_root), "coverage": coverage, "verification": verification}


def _copy_batch_logs(legacy_root: Path, namespace: str, result_root: Path) -> None:
    source = legacy_root / "batch_logs"
    if source.is_dir():
        destination = result_root / "batches" / namespace / "legacy"
        shutil.copytree(source, destination, dirs_exist_ok=True)


def _normalize_layout_metadata(result_root: Path) -> int:
    """Keep schema-v2 root metadata portable after moving EvalResult."""
    updated = 0
    for manifest_path in sorted((result_root / "evaluations").rglob("manifest.json")):
        manifest = _read_json(manifest_path)
        if int(manifest.get("schema_version", 0)) != LAYOUT_SCHEMA_VERSION:
            continue
        if manifest.get("result_root") == "." and manifest.get("artifact_root") == "artifacts":
            continue
        manifest["result_root"] = "."
        manifest["artifact_root"] = "artifacts"
        _write_json(manifest_path, manifest)
        updated += 1
    return updated


def _tree_stats(path: Path) -> tuple[int, int]:
    files = list(candidate for candidate in path.rglob("*") if candidate.is_file())
    return len(files), sum(candidate.stat().st_size for candidate in files)


def _all_png_digests(root: Path, store: ArtifactStore) -> set[str]:
    return {store._file_sha256(path) for path in root.rglob("*.png")}


def _preserve_unverified_ground_truth(
    old_root: Path,
    store: ArtifactStore,
    mappings: list[dict[str, str]],
) -> int:
    canonical_root = store.root / "artifacts" / "ground_truth"
    canonical_digests = _all_png_digests(canonical_root, store) if canonical_root.is_dir() else set()
    preserved = 0
    for source in old_root.rglob("*.png"):
        digest = store._file_sha256(source)
        if digest in canonical_digests:
            continue
        relative = source.relative_to(old_root)
        destination = store.root / "artifacts" / "legacy_unverified" / "ground_truth" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.is_file():
            try:
                destination.hardlink_to(source)
            except OSError:
                shutil.copy2(source, destination)
        mappings.append(
            {
                "source": str(source),
                "artifact": store.relative(destination),
                "kind": "legacy_unverified_ground_truth",
                "semantic_variant": "true",
            }
        )
        preserved += 1
    return preserved


def migrate(result_root: Path, *, apply: bool, prune: bool) -> dict[str, Any]:
    store = ArtifactStore(repo_root=REPO_ROOT, result_root=result_root)
    roots = _legacy_roots(result_root)
    source_manifest_count = sum(len(_manifest_dirs(path)) for path, _ in roots)
    report: dict[str, Any] = {
        "schema_version": LAYOUT_SCHEMA_VERSION,
        "started_at": time.time(),
        "mode": "apply" if apply else "dry-run",
        "legacy_roots": [{"path": str(path), "namespace": namespace} for path, namespace in roots],
        "source_manifests": source_manifest_count,
        "migrated_manifests": 0,
        "render_rows": 0,
        "errors": [],
        "deleted": [],
    }
    if not apply:
        return report
    store.initialize()
    report["normalized_manifests"] = _normalize_layout_metadata(result_root)
    if not roots:
        migration_root = result_root / "migration"
        report_path = migration_root / "report.json"
        verification = verify(result_root)
        if report_path.is_file():
            previous = _read_json(report_path)
            previous["normalized_manifests"] = report["normalized_manifests"]
            previous["verification"] = verification
            previous["layout_normalized_at"] = time.time()
            _write_json(report_path, previous)
            return previous
        report["verification"] = verification
        report["completed_at"] = time.time()
        _write_json(report_path, report)
        return report
    mappings: list[dict[str, str]] = []
    for legacy_root, namespace in roots:
        for source_dir in _manifest_dirs(legacy_root):
            try:
                _migrate_manifest(
                    store=store,
                    source_dir=source_dir,
                    legacy_root=legacy_root,
                    namespace=namespace,
                    mappings=mappings,
                )
                report["migrated_manifests"] += 1
                metrics = _read_json(source_dir / "metrics.json") if (source_dir / "metrics.json").is_file() else {}
                report["render_rows"] += sum(
                    1 for row in metrics.get("per_timestep") or []
                    if isinstance(row, dict) and any(row.get(field) for field in RENDER_FIELDS)
                )
            except Exception as exc:
                report["errors"].append(f"{source_dir}:{type(exc).__name__}:{exc}")
        _copy_batch_logs(legacy_root, namespace, result_root)

    migration_root = result_root / "migration"
    migration_root.mkdir(parents=True, exist_ok=True)
    with (migration_root / "path_mapping.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("kind", "source", "artifact", "semantic_variant"),
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerows(mappings)
    verification = verify(result_root)
    report["verification"] = verification
    can_prune = (
        prune
        and not report["errors"]
        and not verification["errors"]
        and report["migrated_manifests"] == source_manifest_count
    )
    old_ground_truth = result_root / "GroundTruth"
    if can_prune and old_ground_truth.is_dir():
        report["preserved_unverified_ground_truth"] = _preserve_unverified_ground_truth(
            old_ground_truth, store, mappings
        )
        with (migration_root / "path_mapping.tsv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=("kind", "source", "artifact", "semantic_variant"),
                delimiter="\t",
            )
            writer.writeheader()
            writer.writerows(mappings)
    if can_prune:
        for path, _ in roots:
            if path.resolve() == result_root.resolve():
                continue
            files, size = _tree_stats(path)
            shutil.rmtree(path)
            report["deleted"].append({"path": str(path), "files": files, "bytes": size})
        if old_ground_truth.is_dir():
            files, size = _tree_stats(old_ground_truth)
            shutil.rmtree(old_ground_truth)
            report["deleted"].append({"path": str(old_ground_truth), "files": files, "bytes": size})
    report["pruned"] = can_prune
    report["semantic_variants"] = sum(
        row.get("semantic_variant") == "true" for row in mappings
    )
    report["completed_at"] = time.time()
    report_path = migration_root / "report.json"
    _write_json(report_path, report)
    return report


def prune_unreferenced(result_root: Path, *, apply: bool) -> dict[str, Any]:
    referenced = set()
    for metrics_path in (result_root / "evaluations").rglob("metrics.json"):
        for row in _read_json(metrics_path).get("per_timestep") or []:
            if isinstance(row, dict):
                referenced.update(str(row[field]) for field in RENDER_FIELDS if row.get(field))
    candidates = [
        path for path in (result_root / "artifacts").rglob("*.png")
        if path.relative_to(result_root).as_posix() not in referenced
    ]
    removed_bytes = sum(path.stat().st_size for path in candidates)
    if apply:
        for path in candidates:
            path.unlink()
    return {"candidates": len(candidates), "bytes": removed_bytes, "applied": apply}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", default="EvalResult")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("status")
    subparsers.add_parser("verify")
    migration = subparsers.add_parser("migrate")
    migration.add_argument("--apply", action="store_true")
    migration.add_argument("--prune-legacy", action="store_true")
    pruning = subparsers.add_parser("prune")
    pruning.add_argument("--apply", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result_root = Path(args.result_root).expanduser()
    if not result_root.is_absolute():
        result_root = REPO_ROOT / result_root
    result_root = result_root.resolve()
    if args.command == "status":
        payload = status(result_root)
    elif args.command == "verify":
        payload = verify(result_root)
    elif args.command == "migrate":
        payload = migrate(result_root, apply=bool(args.apply), prune=bool(args.prune_legacy))
    else:
        payload = prune_unreferenced(result_root, apply=bool(args.apply))
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 1 if payload.get("errors") or (payload.get("verification") or {}).get("errors") else 0


if __name__ == "__main__":
    raise SystemExit(main())
