from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import median

import yaml


EXPECTED_TARGETS = ("GT", "H2", "H_plus")
CONTROL_PROFILE = "official_control"

HEADER = (
    "config",
    "family",
    "structure",
    "profile",
    "target",
    "training_status",
    "metrics_complete",
    "psnr",
    "delta_vs_control_db",
    "mse",
    "mae",
    "artifact_bytes",
    "compression_ratio",
    "training_seconds",
    "peak_cuda_memory_bytes",
    "has_nonfinite",
    "needs_attention",
    "attention_reason",
    "artifact_path",
    "metrics_path",
    "training_cost_path",
)

PROFILE_HEADER = (
    "family",
    "structure",
    "profile",
    "runs",
    "clean_runs",
    "candidate_eligible",
    "selection_rank",
    "median_psnr",
    "median_delta_vs_control_db",
    "median_artifact_bytes",
    "median_compression_ratio",
    "median_training_seconds",
    "max_peak_cuda_memory_bytes",
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
    candidates = list((run_root / exp_id).glob(f"*/metrics/{exp_id}.json"))
    return max(candidates, key=lambda path: path.stat().st_mtime_ns) if candidates else None


def read_json(path: Path | None) -> dict:
    if path is None or not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def finite_number(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def contains_nonfinite(value: object) -> bool:
    if isinstance(value, dict):
        return any(contains_nonfinite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(contains_nonfinite(item) for item in value)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return not math.isfinite(float(value))
    return False


def portable_path(value: object, repo_root: Path) -> str:
    if not value:
        return ""
    path = Path(str(value))
    try:
        return path.resolve().relative_to(repo_root).as_posix()
    except (OSError, ValueError):
        return str(value)


def numbers(rows: list[dict[str, object]], key: str) -> list[float]:
    result: list[float] = []
    for row in rows:
        value = finite_number(row.get(key))
        if value is not None:
            result.append(value)
    return result


def build_summary(
    *,
    config_root: Path,
    status_path: Path,
    output_path: Path,
    profile_output_path: Path,
    attention_path: Path,
    repo_root: Path,
    run_root: Path,
    regression_tolerance_db: float,
) -> tuple[int, bool]:
    statuses = latest_statuses(status_path)
    rows: list[dict[str, object]] = []

    for config_path in sorted(config_root.rglob("*.yaml")):
        relative_repo = config_path.relative_to(repo_root).as_posix()
        relative = config_path.relative_to(config_root)
        if len(relative.parts) != 4:
            raise ValueError(f"Expected family/structure/profile/config layout: {config_path}")
        family, structure, profile = relative.parts[:3]
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        target = str(payload["data"]["target"])
        metrics_path = latest_metrics(run_root, str(payload["exp_id"]))
        metrics = read_json(metrics_path)
        aggregate = metrics.get("aggregate") if isinstance(metrics.get("aggregate"), dict) else {}
        run_metrics = metrics_path.parent if metrics_path is not None else None
        training_summary_path = run_metrics / "training_summary.json" if run_metrics else None
        training_cost_path = run_metrics / "training_cost.json" if run_metrics else None
        training_summary = read_json(training_summary_path)
        training_cost = read_json(training_cost_path)

        psnr = finite_number(aggregate.get("psnr"))
        mse = finite_number(aggregate.get("mse"))
        mae = finite_number(aggregate.get("mae"))
        artifact_bytes = finite_number(aggregate.get("model_bytes"))
        compression_ratio = finite_number(aggregate.get("cr"))
        training_seconds = finite_number(training_cost.get("total_seconds"))
        peak_memory = finite_number(training_cost.get("peak_cuda_memory_bytes"))
        has_nonfinite = contains_nonfinite(metrics) or contains_nonfinite(training_cost)
        status = statuses.get(relative_repo, "missing")

        reasons: list[str] = []
        if status != "ok":
            reasons.append(f"status={status}")
        if metrics_path is None or not aggregate:
            reasons.append("missing_metrics")
        if psnr is None:
            reasons.append("missing_psnr")
        if artifact_bytes is None or artifact_bytes <= 0:
            reasons.append("missing_artifact_size")
        if compression_ratio is None or compression_ratio <= 0:
            reasons.append("missing_compression_ratio")
        if not training_cost:
            reasons.append("missing_training_cost")
        if has_nonfinite:
            reasons.append("nonfinite_metrics")

        rows.append(
            {
                "config": relative_repo,
                "family": family,
                "structure": structure,
                "profile": profile,
                "target": target,
                "training_status": status,
                "metrics_complete": str(not reasons).lower(),
                "psnr": "" if psnr is None else f"{psnr:.8g}",
                "delta_vs_control_db": "",
                "mse": "" if mse is None else f"{mse:.8g}",
                "mae": "" if mae is None else f"{mae:.8g}",
                "artifact_bytes": "" if artifact_bytes is None else int(artifact_bytes),
                "compression_ratio": "" if compression_ratio is None else f"{compression_ratio:.8g}",
                "training_seconds": "" if training_seconds is None else f"{training_seconds:.8g}",
                "peak_cuda_memory_bytes": "" if peak_memory is None else int(peak_memory),
                "has_nonfinite": str(has_nonfinite).lower(),
                "needs_attention": "",
                "attention_reason": reasons,
                "artifact_path": portable_path(training_summary.get("artifact_path"), repo_root),
                "metrics_path": "" if metrics_path is None else metrics_path.relative_to(repo_root).as_posix(),
                "training_cost_path": "" if training_cost_path is None else training_cost_path.relative_to(repo_root).as_posix(),
            }
        )

    controls = {
        str(row["target"]): float(row["psnr"])
        for row in rows
        if row["profile"] == CONTROL_PROFILE and row["psnr"] != ""
    }
    for row in rows:
        reasons = list(row["attention_reason"])
        target = str(row["target"])
        if row["psnr"] != "":
            if target not in controls:
                reasons.append("missing_control")
            else:
                delta = float(row["psnr"]) - controls[target]
                row["delta_vs_control_db"] = f"{delta:.8g}"
                if row["profile"] != CONTROL_PROFILE and delta < -regression_tolerance_db:
                    reasons.append(f"psnr_regression={delta:.4f}dB")
        row["needs_attention"] = str(bool(reasons)).lower()
        row["attention_reason"] = ",".join(reasons)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=HEADER, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    groups: dict[tuple[str, str, str], list[dict[str, object]]] = {}
    for row in rows:
        key = (str(row["family"]), str(row["structure"]), str(row["profile"]))
        groups.setdefault(key, []).append(row)
    profiles: list[dict[str, object]] = []
    for (family, structure, profile), group in sorted(groups.items()):
        targets = {str(row["target"]) for row in group}
        clean_runs = sum(row["needs_attention"] == "false" for row in group)
        deltas = numbers(group, "delta_vs_control_db")
        eligible = (
            len(group) == len(EXPECTED_TARGETS)
            and targets == set(EXPECTED_TARGETS)
            and clean_runs == len(EXPECTED_TARGETS)
            and len(deltas) == len(EXPECTED_TARGETS)
        )
        psnrs = numbers(group, "psnr")
        artifact_sizes = numbers(group, "artifact_bytes")
        ratios = numbers(group, "compression_ratio")
        seconds = numbers(group, "training_seconds")
        memories = numbers(group, "peak_cuda_memory_bytes")
        profiles.append(
            {
                "family": family,
                "structure": structure,
                "profile": profile,
                "runs": len(group),
                "clean_runs": clean_runs,
                "candidate_eligible": str(eligible).lower(),
                "selection_rank": "",
                "median_psnr": "" if not psnrs else f"{median(psnrs):.8g}",
                "median_delta_vs_control_db": "" if not deltas else f"{median(deltas):.8g}",
                "median_artifact_bytes": "" if not artifact_sizes else f"{median(artifact_sizes):.8g}",
                "median_compression_ratio": "" if not ratios else f"{median(ratios):.8g}",
                "median_training_seconds": "" if not seconds else f"{median(seconds):.8g}",
                "max_peak_cuda_memory_bytes": "" if not memories else int(max(memories)),
            }
        )

    candidates = [row for row in profiles if row["candidate_eligible"] == "true"]
    candidates.sort(
        key=lambda row: (
            -float(row["median_delta_vs_control_db"]),
            -float(row["median_compression_ratio"]),
            float(row["median_training_seconds"]),
            str(row["profile"]),
        )
    )
    for rank, row in enumerate(candidates, start=1):
        row["selection_rank"] = rank
    profiles.sort(
        key=lambda row: (
            0 if row["selection_rank"] != "" else 1,
            int(row["selection_rank"] or 0),
            str(row["profile"]),
        )
    )
    with profile_output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=PROFILE_HEADER, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(profiles)

    attention = [
        f"{row['config']}\t{row['attention_reason']}"
        for row in rows
        if row["needs_attention"] == "true"
    ]
    no_eligible = not candidates
    if no_eligible:
        attention.append("FAMILY:ECNR\tno_eligible_profile")
    attention_path.write_text(
        "\n".join(attention) + ("\n" if attention else ""),
        encoding="utf-8",
    )
    return sum(row["needs_attention"] == "true" for row in rows), no_eligible


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize exploration-v6 ECNR runs.")
    parser.add_argument("--config-root", type=Path, required=True)
    parser.add_argument("--status", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile-output", type=Path, required=True)
    parser.add_argument("--attention-output", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--regression-tolerance-db", type=float, default=1.0)
    parser.add_argument("--fail-if-no-eligible-profile", action="store_true")
    args = parser.parse_args()
    attention, no_eligible = build_summary(
        config_root=args.config_root.resolve(),
        status_path=args.status.resolve(),
        output_path=args.output.resolve(),
        profile_output_path=args.profile_output.resolve(),
        attention_path=args.attention_output.resolve(),
        repo_root=args.repo_root.resolve(),
        run_root=args.run_root.resolve(),
        regression_tolerance_db=args.regression_tolerance_db,
    )
    print(
        f"Wrote exploration-v6 summary to {args.output.resolve()}; "
        f"attention={attention}; no_eligible={str(no_eligible).lower()}"
    )
    if args.fail_if_no_eligible_profile and no_eligible:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
