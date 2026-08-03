from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import yaml


SUMMARY_HEADER = (
    "config",
    "family",
    "profile",
    "target",
    "trajectory",
    "final_psnr",
    "has_nan_or_inf",
    "training_status",
    "scope_count",
    "metrics_path",
)


def _latest_statuses(path: Path) -> dict[str, str]:
    latest: dict[str, str] = {}
    if not path.exists():
        return latest
    with path.open("r", encoding="utf-8", newline="") as handle:
        for index, row in enumerate(csv.reader(handle, delimiter="\t")):
            if index == 0 or len(row) < 2:
                continue
            latest[row[0]] = row[1] if row[1] in {"running", "ok", "failed"} else "invalid"
    return latest


def _latest_metrics(run_root: Path, exp_id: str) -> Path | None:
    experiment_dir = run_root / exp_id
    candidates = list(experiment_dir.glob("**/metrics/exploration_psnr.tsv"))
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime_ns)


def _summarize_metrics(path: Path | None) -> tuple[str, str, str, int]:
    if path is None:
        return "", "", "", 0
    by_progress: dict[int, list[float]] = defaultdict(list)
    scopes: set[str] = set()
    has_nonfinite = False
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            try:
                value = float(row["aggregate_psnr"])
                progress = int(row["progress"])
            except (KeyError, TypeError, ValueError):
                has_nonfinite = True
                continue
            scopes.add(row.get("scope", ""))
            if math.isfinite(value):
                by_progress[progress].append(value)
            else:
                has_nonfinite = True
    means = {
        progress: sum(values) / len(values)
        for progress, values in by_progress.items()
        if values
    }
    trajectory = ",".join(f"{progress}:{means[progress]:.8g}" for progress in sorted(means))
    final_psnr = f"{means[max(means)]:.8g}" if means else ""
    return trajectory, final_psnr, str(has_nonfinite).lower(), len(scopes)


def build_summary(config_root: Path, status_path: Path, output_path: Path, repo_root: Path) -> None:
    statuses = _latest_statuses(status_path)
    run_root = repo_root / "runs" / "exploration"
    rows = []
    for config_path in sorted(config_root.rglob("*.yaml")):
        relative = config_path.relative_to(repo_root).as_posix()
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        parts = config_path.relative_to(config_root).parts
        family = parts[0]
        profile = parts[2]
        data = payload.get("data") or payload.get("DATA") or {}
        target = str(data.get("target") or data.get("attr_name") or "aggregate")
        metrics_path = _latest_metrics(run_root, str(payload["exp_id"]))
        trajectory, final_psnr, has_nonfinite, scope_count = _summarize_metrics(metrics_path)
        rows.append(
            {
                "config": relative,
                "family": family,
                "profile": profile,
                "target": target,
                "trajectory": trajectory,
                "final_psnr": final_psnr,
                "has_nan_or_inf": has_nonfinite,
                "training_status": statuses.get(relative, "missing"),
                "scope_count": scope_count,
                "metrics_path": "" if metrics_path is None else metrics_path.relative_to(repo_root).as_posix(),
            }
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_HEADER, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize the isolated Size163 exploration batch.")
    parser.add_argument("--config-root", type=Path, required=True)
    parser.add_argument("--status", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    args = parser.parse_args()
    build_summary(args.config_root.resolve(), args.status.resolve(), args.output.resolve(), args.repo_root.resolve())
    print(f"Wrote exploration summary to {args.output}")


if __name__ == "__main__":
    main()
