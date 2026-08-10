from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import yaml


HEADER = (
    "config",
    "family",
    "size",
    "target",
    "training_status",
    "probe_complete",
    "trajectory",
    "initial_psnr",
    "peak_psnr",
    "peak_progress",
    "final_psnr",
    "gain_from_initial_db",
    "drop_from_peak_db",
    "has_nonfinite",
    "needs_attention",
    "attention_reason",
    "metrics_path",
)


def latest_statuses(path: Path) -> dict[str, str]:
    latest: dict[str, str] = {}
    if not path.exists():
        return latest
    with path.open("r", encoding="utf-8", newline="") as handle:
        for index, row in enumerate(csv.reader(handle, delimiter="\t")):
            if index == 0 or len(row) < 2:
                continue
            latest[row[0]] = row[1]
    return latest


def latest_metrics(run_root: Path, exp_id: str) -> Path | None:
    candidates = list((run_root / exp_id).glob("**/metrics/exploration_psnr.tsv"))
    return max(candidates, key=lambda path: path.stat().st_mtime_ns) if candidates else None


def read_trajectory(path: Path | None) -> tuple[dict[int, float], bool]:
    if path is None:
        return {}, False
    values: dict[int, list[float]] = {}
    has_nonfinite = False
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            try:
                progress = int(row["progress"])
                value = float(row["aggregate_psnr"])
            except (KeyError, TypeError, ValueError):
                has_nonfinite = True
                continue
            if not math.isfinite(value):
                has_nonfinite = True
                continue
            values.setdefault(progress, []).append(value)
    return (
        {progress: sum(samples) / len(samples) for progress, samples in values.items() if samples},
        has_nonfinite,
    )


def build_summary(
    *,
    config_root: Path,
    status_path: Path,
    output_path: Path,
    attention_path: Path,
    repo_root: Path,
    run_root: Path,
    collapse_threshold_db: float,
    minimum_gain_db: float,
) -> int:
    statuses = latest_statuses(status_path)
    rows: list[dict[str, object]] = []
    attention: list[str] = []
    for config_path in sorted(config_root.rglob("*.yaml")):
        relative_repo = config_path.relative_to(repo_root).as_posix()
        relative_config = config_path.relative_to(config_root)
        family, size = relative_config.parts[:2]
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        data = payload.get("data") or payload.get("DATA") or {}
        target = str(data.get("target") or "aggregate")
        metrics_path = latest_metrics(run_root, str(payload["exp_id"]))
        trajectory, has_nonfinite = read_trajectory(metrics_path)
        ordered = sorted(trajectory)
        initial = trajectory[ordered[0]] if ordered else None
        final = trajectory[ordered[-1]] if ordered else None
        peak_progress = max(ordered, key=lambda progress: trajectory[progress]) if ordered else None
        peak = trajectory[peak_progress] if peak_progress is not None else None
        gain = final - initial if initial is not None and final is not None else None
        drop = peak - final if peak is not None and final is not None else None
        status = statuses.get(relative_repo, "missing")
        probe_complete = 50 in trajectory
        reasons: list[str] = []
        if status != "ok":
            reasons.append(f"status={status}")
        if has_nonfinite:
            reasons.append("nonfinite_psnr")
        if not probe_complete:
            reasons.append("missing_progress_50")
        if drop is not None and drop > collapse_threshold_db:
            reasons.append(f"collapse={drop:.4f}dB")
        if gain is not None and gain < minimum_gain_db:
            reasons.append(f"gain={gain:.4f}dB")
        if not trajectory:
            reasons.append("missing_probe_metrics")
        if reasons:
            attention.append(f"{relative_repo}\t{','.join(reasons)}")
        rows.append(
            {
                "config": relative_repo,
                "family": family,
                "size": size,
                "target": target,
                "training_status": status,
                "probe_complete": str(probe_complete).lower(),
                "trajectory": ",".join(f"{progress}:{trajectory[progress]:.8g}" for progress in ordered),
                "initial_psnr": "" if initial is None else f"{initial:.8g}",
                "peak_psnr": "" if peak is None else f"{peak:.8g}",
                "peak_progress": "" if peak_progress is None else peak_progress,
                "final_psnr": "" if final is None else f"{final:.8g}",
                "gain_from_initial_db": "" if gain is None else f"{gain:.8g}",
                "drop_from_peak_db": "" if drop is None else f"{drop:.8g}",
                "has_nonfinite": str(has_nonfinite).lower(),
                "needs_attention": str(bool(reasons)).lower(),
                "attention_reason": ",".join(reasons),
                "metrics_path": "" if metrics_path is None else metrics_path.relative_to(repo_root).as_posix(),
            }
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=HEADER, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    attention_path.write_text("\n".join(attention) + ("\n" if attention else ""), encoding="utf-8")
    return len(attention)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize all exploration-v3 Size smoke runs.")
    parser.add_argument("--config-root", type=Path, required=True)
    parser.add_argument("--status", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--attention-output", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--collapse-threshold-db", type=float, default=1.0)
    parser.add_argument("--minimum-gain-db", type=float, default=0.1)
    parser.add_argument("--fail-on-attention", action="store_true")
    args = parser.parse_args()
    count = build_summary(
        config_root=args.config_root.resolve(),
        status_path=args.status.resolve(),
        output_path=args.output.resolve(),
        attention_path=args.attention_output.resolve(),
        repo_root=args.repo_root.resolve(),
        run_root=args.run_root.resolve(),
        collapse_threshold_db=args.collapse_threshold_db,
        minimum_gain_db=args.minimum_gain_db,
    )
    print(f"Wrote exploration-v3 summary to {args.output}; attention={count}")
    if args.fail_on_attention and count:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
