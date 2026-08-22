from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import math
from pathlib import Path
from statistics import median

import yaml


EXPECTED_PROGRESS = tuple(range(5, 51, 5))
EXPECTED_TARGETS = ("GT", "H2", "H_plus")
FV_REFERENCE_FINAL_PSNR = {
    "GT": 24.956097,
    "H2": 33.397329,
    "H_plus": 24.312176,
}
BASELINES = {
    "fV-SRN": ("formal_size163", "lr1e2_step100"),
    "InstantVNR": ("official_default", "official_control"),
}

HEADER = (
    "config",
    "family",
    "structure",
    "profile",
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
    "historical_reference_psnr",
    "historical_delta_db",
    "has_nonfinite",
    "needs_attention",
    "attention_reason",
    "metrics_path",
)

PROFILE_HEADER = (
    "family",
    "structure",
    "profile",
    "runs",
    "clean_runs",
    "candidate_eligible",
    "selection_rank",
    "median_final_psnr",
    "median_gain_db",
    "median_drop_db",
    "median_delta_vs_control_db",
)


@dataclass(frozen=True)
class SummaryResult:
    attention_count: int
    missing_eligible_families: tuple[str, ...]


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


def _target(payload: dict) -> str:
    data = payload.get("data") or {}
    if data.get("target"):
        return str(data["target"])
    targets = data.get("targets") or {}
    if isinstance(targets, dict) and len(targets) == 1:
        return str(next(iter(targets)))
    return "aggregate"


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


def _write_profile_summary(
    rows: list[dict[str, object]], path: Path
) -> tuple[str, ...]:
    controls: dict[tuple[str, str], float] = {}
    for row in rows:
        family = str(row["family"])
        baseline = BASELINES.get(family)
        if (
            baseline is not None
            and (str(row["structure"]), str(row["profile"])) == baseline
            and row["final_psnr"] != ""
        ):
            controls[(family, str(row["target"]))] = float(row["final_psnr"])

    groups: dict[tuple[str, str, str], list[dict[str, object]]] = {}
    for row in rows:
        key = (str(row["family"]), str(row["structure"]), str(row["profile"]))
        groups.setdefault(key, []).append(row)

    output: list[dict[str, object]] = []
    for (family, structure, profile), group in sorted(groups.items()):
        deltas = [
            float(row["final_psnr"]) - controls[(family, str(row["target"]))]
            for row in group
            if row["final_psnr"] != ""
            and (family, str(row["target"])) in controls
        ]
        clean_runs = sum(row["needs_attention"] == "false" for row in group)
        targets = {str(row["target"]) for row in group}
        eligible = (
            len(group) == len(EXPECTED_TARGETS)
            and targets == set(EXPECTED_TARGETS)
            and clean_runs == len(EXPECTED_TARGETS)
        )
        values: dict[str, object] = {
            "family": family,
            "structure": structure,
            "profile": profile,
            "runs": len(group),
            "clean_runs": clean_runs,
            "candidate_eligible": str(eligible).lower(),
            "selection_rank": "",
        }
        for output_key, source_key in (
            ("median_final_psnr", "final_psnr"),
            ("median_gain_db", "gain_from_initial_db"),
            ("median_drop_db", "drop_from_peak_db"),
        ):
            samples = _numbers(group, source_key)
            values[output_key] = "" if not samples else f"{median(samples):.8g}"
        values["median_delta_vs_control_db"] = (
            "" if not deltas else f"{median(deltas):.8g}"
        )
        output.append(values)

    families = sorted({str(row["family"]) for row in output})
    for family in families:
        candidates = [
            row
            for row in output
            if row["family"] == family and row["candidate_eligible"] == "true"
        ]
        candidates.sort(
            key=lambda row: (
                -float(row["median_delta_vs_control_db"] or "-inf"),
                float(row["median_drop_db"] or "inf"),
                -float(row["median_final_psnr"] or "-inf"),
                str(row["structure"]),
                str(row["profile"]),
            )
        )
        for rank, row in enumerate(candidates, start=1):
            row["selection_rank"] = rank

    output.sort(
        key=lambda row: (
            str(row["family"]),
            0 if row["selection_rank"] != "" else 1,
            int(row["selection_rank"]) if row["selection_rank"] != "" else 0,
            str(row["structure"]),
            str(row["profile"]),
        )
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=PROFILE_HEADER, delimiter="\t", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(output)

    return tuple(
        family
        for family in BASELINES
        if not any(
            row["family"] == family and row["candidate_eligible"] == "true"
            for row in output
        )
    )


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
    fv_reference_tolerance_db: float,
) -> SummaryResult:
    statuses = latest_statuses(status_path)
    rows: list[dict[str, object]] = []
    attention: list[str] = []

    for config_path in sorted(config_root.rglob("*.yaml")):
        relative_repo = config_path.relative_to(repo_root).as_posix()
        relative = config_path.relative_to(config_root)
        if len(relative.parts) != 4:
            raise ValueError(f"Expected family/structure/profile/config layout: {config_path}")
        family, structure, profile = relative.parts[:3]
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        target = _target(payload)
        metrics_path = latest_metrics(run_root, str(payload["exp_id"]))
        trajectory, has_nonfinite = read_trajectory(metrics_path)
        ordered = sorted(trajectory)
        initial = trajectory[ordered[0]] if ordered else None
        final = trajectory[ordered[-1]] if ordered else None
        peak_progress = max(ordered, key=lambda item: trajectory[item]) if ordered else None
        peak = trajectory[peak_progress] if peak_progress is not None else None
        gain = final - initial if initial is not None and final is not None else None
        drop = peak - final if peak is not None and final is not None else None
        reference = FV_REFERENCE_FINAL_PSNR.get(target) if family == "fV-SRN" else None
        historical_delta = final - reference if final is not None and reference is not None else None
        status = statuses.get(relative_repo, "missing")
        missing_progress = sorted(set(EXPECTED_PROGRESS) - set(trajectory))

        reasons: list[str] = []
        if status != "ok":
            reasons.append(f"status={status}")
        if has_nonfinite:
            reasons.append("nonfinite_psnr")
        if missing_progress:
            reasons.append("missing_progress=" + ",".join(str(value) for value in missing_progress))
        if drop is not None and drop > collapse_threshold_db:
            reasons.append(f"collapse={drop:.4f}dB")
        if gain is not None and gain < minimum_gain_db:
            reasons.append(f"gain={gain:.4f}dB")
        if historical_delta is not None and historical_delta < -fv_reference_tolerance_db:
            reasons.append(f"historical_regression={historical_delta:.4f}dB")
        if not trajectory:
            reasons.append("missing_probe_metrics")
        if reasons:
            attention.append(f"{relative_repo}\t{','.join(reasons)}")

        rows.append(
            {
                "config": relative_repo,
                "family": family,
                "structure": structure,
                "profile": profile,
                "target": target,
                "training_status": status,
                "probe_complete": str(not missing_progress).lower(),
                "trajectory": ",".join(
                    f"{progress}:{trajectory[progress]:.8g}" for progress in ordered
                ),
                "initial_psnr": "" if initial is None else f"{initial:.8g}",
                "peak_psnr": "" if peak is None else f"{peak:.8g}",
                "peak_progress": "" if peak_progress is None else peak_progress,
                "final_psnr": "" if final is None else f"{final:.8g}",
                "gain_from_initial_db": "" if gain is None else f"{gain:.8g}",
                "drop_from_peak_db": "" if drop is None else f"{drop:.8g}",
                "historical_reference_psnr": "" if reference is None else f"{reference:.8g}",
                "historical_delta_db": "" if historical_delta is None else f"{historical_delta:.8g}",
                "has_nonfinite": str(has_nonfinite).lower(),
                "needs_attention": str(bool(reasons)).lower(),
                "attention_reason": ",".join(reasons),
                "metrics_path": (
                    ""
                    if metrics_path is None
                    else metrics_path.relative_to(repo_root).as_posix()
                ),
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=HEADER, delimiter="\t", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)

    missing_families = _write_profile_summary(rows, profile_output_path)
    attention.extend(
        f"FAMILY:{family}\tno_eligible_profile" for family in missing_families
    )
    attention_path.parent.mkdir(parents=True, exist_ok=True)
    attention_path.write_text(
        "\n".join(attention) + ("\n" if attention else ""), encoding="utf-8"
    )
    return SummaryResult(
        attention_count=sum(row["needs_attention"] == "true" for row in rows),
        missing_eligible_families=missing_families,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize exploration-v5 runs.")
    parser.add_argument("--config-root", type=Path, required=True)
    parser.add_argument("--status", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile-output", type=Path, required=True)
    parser.add_argument("--attention-output", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--collapse-threshold-db", type=float, default=1.0)
    parser.add_argument("--minimum-gain-db", type=float, default=0.1)
    parser.add_argument("--fv-reference-tolerance-db", type=float, default=1.0)
    parser.add_argument("--fail-if-no-eligible-profile", action="store_true")
    args = parser.parse_args()
    result = build_summary(
        config_root=args.config_root.resolve(),
        status_path=args.status.resolve(),
        output_path=args.output.resolve(),
        profile_output_path=args.profile_output.resolve(),
        attention_path=args.attention_output.resolve(),
        repo_root=args.repo_root.resolve(),
        run_root=args.run_root.resolve(),
        collapse_threshold_db=args.collapse_threshold_db,
        minimum_gain_db=args.minimum_gain_db,
        fv_reference_tolerance_db=args.fv_reference_tolerance_db,
    )
    missing = ",".join(result.missing_eligible_families) or "none"
    print(
        f"Wrote exploration-v5 summary to {args.output}; "
        f"attention={result.attention_count}; no_eligible={missing}"
    )
    if args.fail_if_no_eligible_profile and result.missing_eligible_families:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
