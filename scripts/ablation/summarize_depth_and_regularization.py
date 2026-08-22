from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import median

import yaml

from var_expert_inr.models.baselines.coordnet import CoordNet
from var_expert_inr.methods.rmdsrn.model import RMDSRN


HEADER = (
    "config", "family", "size", "profile", "target", "training_status",
    "probe_complete", "trajectory", "initial_psnr", "peak_psnr", "peak_progress",
    "final_psnr", "gain_from_initial_db", "drop_from_peak_db", "has_nonfinite",
    "needs_attention", "attention_reason", "param_count", "fp16_bytes",
    "member_mse", "variance_kl", "variance_weight", "weighted_variance_loss",
    "weighted_variance_to_member_ratio", "variance_error_pearson",
    "topk_hit_rate_0.01", "topk_hit_rate_0.05", "best_checkpoint", "metrics_path",
)

PROFILE_HEADER = (
    "family", "profile", "runs", "clean_runs", "catastrophic_runs",
    "candidate_eligible", "selection_rank", "res_depth",
    "median_final_psnr", "median_gain_db", "median_drop_db",
    "median_delta_vs_res10_db", "median_variance_error_pearson",
    "median_topk_hit_rate_0.01", "median_topk_hit_rate_0.05",
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


def _contains_nonfinite(value: object) -> bool:
    if isinstance(value, dict):
        return any(_contains_nonfinite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_contains_nonfinite(item) for item in value)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return not math.isfinite(float(value))
    return False


def read_probe(path: Path | None) -> tuple[dict[int, float], dict[int, dict], bool]:
    if path is None:
        return {}, {}, False
    values: dict[int, list[float]] = {}
    details: dict[int, dict] = {}
    has_nonfinite = False
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            try:
                progress = int(row["progress"])
                value = float(row["aggregate_psnr"])
                payload = json.loads(row.get("details") or "{}")
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                has_nonfinite = True
                continue
            if not math.isfinite(value):
                has_nonfinite = True
                continue
            if _contains_nonfinite(payload):
                has_nonfinite = True
            values.setdefault(progress, []).append(value)
            if isinstance(payload, dict):
                details[progress] = payload
    return (
        {progress: sum(samples) / len(samples) for progress, samples in values.items()},
        details,
        has_nonfinite,
    )


def model_param_count(family: str, model: dict) -> int:
    if family == "CoordNet":
        built = CoordNet(
            in_features=int(model.get("in_features", 4)),
            out_features=int(model.get("out_features", 1)),
            init_features=int(model["init_features"]),
            num_res=int(model["num_res"]),
        )
    elif family == "RMDSRN":
        built = RMDSRN(model)
    else:
        raise ValueError(f"Unsupported exploration-v4 family: {family}")
    return sum(int(parameter.numel()) for parameter in built.parameters())


def _finite_detail(details: dict, key: str) -> float | None:
    try:
        value = float(details[key])
    except (KeyError, TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def build_summary(
    *, config_root: Path, status_path: Path, output_path: Path,
    profile_output_path: Path, attention_path: Path, repo_root: Path,
    run_root: Path, collapse_threshold_db: float, minimum_gain_db: float,
) -> int:
    statuses = latest_statuses(status_path)
    rows: list[dict[str, object]] = []
    attention: list[str] = []
    param_cache: dict[str, int] = {}
    for config_path in sorted(config_root.rglob("*.yaml")):
        relative_repo = config_path.relative_to(repo_root).as_posix()
        relative = config_path.relative_to(config_root)
        family, size, profile = relative.parts[:3]
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        data = payload.get("data") or {}
        target = str(data.get("target") or "aggregate")
        metrics_path = latest_metrics(run_root, str(payload["exp_id"]))
        trajectory, probe_details, has_nonfinite = read_probe(metrics_path)
        ordered = sorted(trajectory)
        initial = trajectory[ordered[0]] if ordered else None
        final = trajectory[ordered[-1]] if ordered else None
        peak_progress = max(ordered, key=lambda item: trajectory[item]) if ordered else None
        peak = trajectory[peak_progress] if peak_progress is not None else None
        gain = final - initial if initial is not None and final is not None else None
        drop = peak - final if peak is not None and final is not None else None
        status = statuses.get(relative_repo, "missing")
        reasons: list[str] = []
        if status != "ok":
            reasons.append(f"status={status}")
        if has_nonfinite:
            reasons.append("nonfinite_probe")
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

        signature = json.dumps(payload["model"], sort_keys=True)
        if signature not in param_cache:
            param_cache[signature] = model_param_count(family, payload["model"])
        params = param_cache[signature]
        final_details = probe_details.get(ordered[-1], {}) if ordered else {}
        topk = final_details.get("topk_hit_rate") or {}
        best_checkpoint = ""
        if metrics_path is not None:
            candidate = metrics_path.parent.parent / "checkpoints" / f"{payload['exp_id']}_best_probe.pth"
            if candidate.exists():
                best_checkpoint = candidate.relative_to(repo_root).as_posix()

        row = {
            "config": relative_repo, "family": family, "size": size,
            "profile": profile, "target": target, "training_status": status,
            "probe_complete": str(50 in trajectory).lower(),
            "trajectory": ",".join(f"{p}:{trajectory[p]:.8g}" for p in ordered),
            "initial_psnr": "" if initial is None else f"{initial:.8g}",
            "peak_psnr": "" if peak is None else f"{peak:.8g}",
            "peak_progress": "" if peak_progress is None else peak_progress,
            "final_psnr": "" if final is None else f"{final:.8g}",
            "gain_from_initial_db": "" if gain is None else f"{gain:.8g}",
            "drop_from_peak_db": "" if drop is None else f"{drop:.8g}",
            "has_nonfinite": str(has_nonfinite).lower(),
            "needs_attention": str(bool(reasons)).lower(),
            "attention_reason": ",".join(reasons), "param_count": params,
            "fp16_bytes": params * 2, "best_checkpoint": best_checkpoint,
            "metrics_path": "" if metrics_path is None else metrics_path.relative_to(repo_root).as_posix(),
        }
        for key in (
            "member_mse", "variance_kl", "variance_weight", "weighted_variance_loss",
            "weighted_variance_to_member_ratio", "variance_error_pearson",
        ):
            value = _finite_detail(final_details, key)
            row[key] = "" if value is None else f"{value:.8g}"
        for fraction in ("0.01", "0.05"):
            try:
                value = float(topk[fraction])
            except (KeyError, TypeError, ValueError):
                value = None
            row[f"topk_hit_rate_{fraction}"] = "" if value is None else f"{value:.8g}"
        rows.append(row)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=HEADER, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    attention_path.write_text("\n".join(attention) + ("\n" if attention else ""), encoding="utf-8")
    _write_profile_summary(rows, profile_output_path)
    return len(attention)


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
    baseline = {
        (str(row["size"]), str(row["target"])): float(row["final_psnr"])
        for row in rows
        if row["family"] == "CoordNet"
        and row["profile"] == "res10_base_lr"
        and row["final_psnr"] != ""
    }
    output: list[dict[str, object]] = []
    groups: dict[tuple[str, str], list[dict[str, object]]] = {}
    for row in rows:
        groups.setdefault((str(row["family"]), str(row["profile"])), []).append(row)
    for (family, profile), group in sorted(groups.items()):
        deltas = [
            float(row["final_psnr"]) - baseline[(str(row["size"]), str(row["target"]))]
            for row in group
            if row["final_psnr"] != ""
            and (str(row["size"]), str(row["target"])) in baseline
        ]
        values = {
            "family": family, "profile": profile, "runs": len(group),
            "clean_runs": sum(row["needs_attention"] == "false" for row in group),
            "catastrophic_runs": sum(
                row["drop_from_peak_db"] != "" and float(row["drop_from_peak_db"]) > 3.0
                for row in group
            ),
            "candidate_eligible": "false",
            "selection_rank": "",
            "res_depth": _profile_res_depth(profile) if family == "CoordNet" else "",
        }
        for output_key, source_key in (
            ("median_final_psnr", "final_psnr"), ("median_gain_db", "gain_from_initial_db"),
            ("median_drop_db", "drop_from_peak_db"),
            ("median_variance_error_pearson", "variance_error_pearson"),
            ("median_topk_hit_rate_0.01", "topk_hit_rate_0.01"),
            ("median_topk_hit_rate_0.05", "topk_hit_rate_0.05"),
        ):
            samples = _numbers(group, source_key)
            values[output_key] = "" if not samples else f"{median(samples):.8g}"
        values["median_delta_vs_res10_db"] = "" if not deltas else f"{median(deltas):.8g}"
        if family == "CoordNet" and deltas and values["catastrophic_runs"] == 0:
            values["candidate_eligible"] = "true"
        output.append(values)

    coord_candidates = [
        row for row in output if row["candidate_eligible"] == "true"
    ]
    coord_candidates.sort(
        key=lambda row: (
            -int(row["clean_runs"]),
            -float(row["median_delta_vs_res10_db"]),
            int(row["res_depth"]),
            str(row["profile"]),
        )
    )
    for rank, row in enumerate(coord_candidates, start=1):
        row["selection_rank"] = rank
    output.sort(
        key=lambda row: (
            0 if row["family"] == "CoordNet" else 1,
            0 if row["selection_rank"] != "" else 1,
            int(row["selection_rank"]) if row["selection_rank"] != "" else 0,
            str(row["profile"]),
        )
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=PROFILE_HEADER, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(output)


def _profile_res_depth(profile: str) -> int:
    prefix = str(profile).split("_", 1)[0]
    if prefix.startswith("res") and prefix[3:].isdigit():
        return int(prefix[3:])
    return 1_000_000


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize exploration-v4 runs.")
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
        config_root=args.config_root.resolve(), status_path=args.status.resolve(),
        output_path=args.output.resolve(), profile_output_path=args.profile_output.resolve(),
        attention_path=args.attention_output.resolve(), repo_root=args.repo_root.resolve(),
        run_root=args.run_root.resolve(), collapse_threshold_db=args.collapse_threshold_db,
        minimum_gain_db=args.minimum_gain_db,
    )
    print(f"Wrote exploration-v4 summary to {args.output}; attention={count}")
    if args.fail_on_attention and count:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
