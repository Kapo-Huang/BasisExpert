from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from statistics import median

import yaml


HEADER = (
    "config",
    "size",
    "profile",
    "learning_rate",
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
    "best_checkpoint",
    "metrics_path",
)

PROFILE_HEADER = (
    "size",
    "profile",
    "learning_rate",
    "runs",
    "clean_runs",
    "catastrophic_runs",
    "candidate_eligible",
    "selection_rank",
    "median_final_psnr",
    "median_gain_db",
    "median_drop_db",
)


def latest_statuses(path: Path) -> dict[str, str]:
    latest: dict[str, str] = {}
    if not path.exists():
        return latest
    with path.open("r", encoding="utf-8", newline="") as handle:
        for index, row in enumerate(csv.reader(handle, delimiter="\t")):
            if index and len(row) >= 2:
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
        {progress: sum(samples) / len(samples) for progress, samples in values.items()},
        has_nonfinite,
    )


def _numbers(rows: list[dict[str, object]], key: str) -> list[float]:
    result: list[float] = []
    for row in rows:
        try:
            value = float(row[key])
        except (KeyError, TypeError, ValueError):
            continue
        if math.isfinite(value):
            result.append(value)
    return result


def _write_profile_summary(rows: list[dict[str, object]], path: Path) -> None:
    groups: dict[tuple[str, str], list[dict[str, object]]] = {}
    for row in rows:
        groups.setdefault((str(row["size"]), str(row["profile"])), []).append(row)

    output: list[dict[str, object]] = []
    for (size, profile), group in sorted(groups.items()):
        clean_runs = sum(row["needs_attention"] == "false" for row in group)
        catastrophic_runs = sum(
            row["drop_from_peak_db"] != ""
            and float(row["drop_from_peak_db"]) > 3.0
            for row in group
        )
        summary: dict[str, object] = {
            "size": size,
            "profile": profile,
            "learning_rate": group[0]["learning_rate"],
            "runs": len(group),
            "clean_runs": clean_runs,
            "catastrophic_runs": catastrophic_runs,
            "candidate_eligible": str(
                len(group) == 3 and clean_runs == len(group) and catastrophic_runs == 0
            ).lower(),
            "selection_rank": "",
        }
        for output_key, source_key in (
            ("median_final_psnr", "final_psnr"),
            ("median_gain_db", "gain_from_initial_db"),
            ("median_drop_db", "drop_from_peak_db"),
        ):
            samples = _numbers(group, source_key)
            summary[output_key] = "" if not samples else f"{median(samples):.8g}"
        output.append(summary)

    for size in sorted({str(row["size"]) for row in output}):
        candidates = [
            row
            for row in output
            if row["size"] == size and row["candidate_eligible"] == "true"
        ]
        candidates.sort(
            key=lambda row: (-float(row["median_final_psnr"]), str(row["profile"]))
        )
        for rank, row in enumerate(candidates, start=1):
            row["selection_rank"] = rank

    output.sort(key=lambda row: (str(row["size"]), str(row["profile"])))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=PROFILE_HEADER, delimiter="\t", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(output)


def build_summary(
    *,
    config_root: Path,
    status_path: Path,
    output_path: Path,
    profile_output_path: Path,
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
        relative = config_path.relative_to(config_root)
        family, size, profile = relative.parts[:3]
        if family != "CoordNet":
            raise ValueError(f"Unexpected family in exploration_CoordNet: {family}")
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        target = str(payload["data"]["target"])
        learning_rate = float(payload["training"]["lr"])
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

        reasons: list[str] = []
        if status != "ok":
            reasons.append(f"status={status}")
        if has_nonfinite:
            reasons.append("nonfinite_psnr")
        if 50 not in trajectory:
            reasons.append("missing_progress_50")
        if drop is not None and drop > collapse_threshold_db:
            reasons.append(f"collapse={drop:.4f}dB")
        if gain is not None and gain < minimum_gain_db:
            reasons.append(f"gain={gain:.4f}dB")
        if not trajectory:
            reasons.append("missing_probe_metrics")
        if reasons:
            attention.append(f"{relative_repo}\t{','.join(reasons)}")

        best_checkpoint = ""
        if metrics_path is not None:
            candidate = (
                metrics_path.parent.parent
                / "checkpoints"
                / f"{payload['exp_id']}_best_probe.pth"
            )
            if candidate.exists():
                best_checkpoint = candidate.relative_to(repo_root).as_posix()

        rows.append(
            {
                "config": relative_repo,
                "size": size,
                "profile": profile,
                "learning_rate": f"{learning_rate:.8g}",
                "target": target,
                "training_status": status,
                "probe_complete": str(50 in trajectory).lower(),
                "trajectory": ",".join(
                    f"{progress}:{trajectory[progress]:.8g}" for progress in ordered
                ),
                "initial_psnr": "" if initial is None else f"{initial:.8g}",
                "peak_psnr": "" if peak is None else f"{peak:.8g}",
                "peak_progress": "" if peak_progress is None else peak_progress,
                "final_psnr": "" if final is None else f"{final:.8g}",
                "gain_from_initial_db": "" if gain is None else f"{gain:.8g}",
                "drop_from_peak_db": "" if drop is None else f"{drop:.8g}",
                "has_nonfinite": str(has_nonfinite).lower(),
                "needs_attention": str(bool(reasons)).lower(),
                "attention_reason": ",".join(reasons),
                "best_checkpoint": best_checkpoint,
                "metrics_path": ""
                if metrics_path is None
                else metrics_path.relative_to(repo_root).as_posix(),
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=HEADER, delimiter="\t", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)
    attention_path.write_text(
        "\n".join(attention) + ("\n" if attention else ""), encoding="utf-8"
    )
    _write_profile_summary(rows, profile_output_path)
    return len(attention)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize exploration_CoordNet low-learning-rate Size runs."
    )
    parser.add_argument("--config-root", type=Path, required=True)
    parser.add_argument("--status", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile-output", type=Path, required=True)
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
        profile_output_path=args.profile_output.resolve(),
        attention_path=args.attention_output.resolve(),
        repo_root=args.repo_root.resolve(),
        run_root=args.run_root.resolve(),
        collapse_threshold_db=args.collapse_threshold_db,
        minimum_gain_db=args.minimum_gain_db,
    )
    print(f"Wrote exploration_CoordNet summary to {args.output}; attention={count}")
    if args.fail_on_attention and count:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
