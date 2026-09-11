#!/usr/bin/env python3
"""Import dataset-total runtime results into ``EXPERIMENT_SUMMARY.md``.

Independent per-variable runs receive a representative per-variable estimate
in detailed tables, while aggregated tables use dataset-total values. Missing
runtime groups are filled from documented proxies instead of being shown as
zero or silently omitted.
"""

from __future__ import annotations

import argparse
import csv
import statistics
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable


RUNTIME_START = "<!-- AGGREGATED_RUNTIME_START -->"
RUNTIME_END = "<!-- AGGREGATED_RUNTIME_END -->"
RUNTIME_MARKER_PREFIX = "> Compression/inference runtime values"
RUNTIME_MARKER = (
    "> Compression/inference runtime values are imported from the dataset-total "
    "runtime evaluation. Representative and proxy estimates are explicitly "
    "labelled; compression is reported in hours and inference in seconds."
)
RESULT_CATEGORIES = {"Main", "RD Curve", "Ablation", "Sensitivity", "Scaling"}
MAIN_METHOD_ORDER = (
    "Ours", "CoordNet", "SIREN", "Neural Experts", "MoE-INR", "fV-SRN",
    "APMGSRN", "InstantVNR", "MINER", "ECNR", "STSR-INR", "MVNet",
)
MAIN_DATASET_ORDER = ("Ionization", "Combustion", "Katrina", "RedSea")
RD_METHOD_ORDER = ("Ours", "CoordNet", "MoE-INR", "fV-SRN", "STSR-INR")
RD_DATASET_ORDER = ("Ionization", "Combustion")
RD_SIZE_ORDER = ("0.41", "0.82", "1.63", "3.26")
SENSITIVITY_FAMILY_ORDER = ("ExpertNum", "TopK", "Seed")

# Runtime probes for these Result groups failed. Use the closest available
# implementation family on the same dataset and keep the estimate auditable.
PROXY_METHODS = {"MINER": "ECNR", "Neural Experts": "MoE-INR"}


@dataclass(frozen=True)
class RuntimeGroup:
    category: str
    method: str
    dataset: str
    variant: str
    aggregation_mode: str
    representative_target: str
    variable_count: int
    group_member_count: int
    compression_hours: float
    inference_seconds: float
    source_kind: str
    source_label: str

    @property
    def key(self) -> tuple[str, str, str, str]:
        return self.category, self.method, self.dataset, self.variant


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo", type=Path, default=Path(__file__).resolve().parents[2]
    )
    parser.add_argument("--result", type=Path, default=None)
    parser.add_argument("--runtime-groups", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def group_variant(row: dict[str, str]) -> str:
    category = str(row.get("category", ""))
    item = str(row.get("item", ""))
    if category == "Main":
        return ""
    if category == "RD Curve":
        return item.split("/", 1)[0]
    return item


def group_key(row: dict[str, str]) -> tuple[str, str, str, str]:
    return (
        str(row.get("category", "")),
        str(row.get("method", "")),
        str(row.get("dataset", "")),
        group_variant(row),
    )


def read_runtime_groups(path: Path) -> dict[tuple[str, str, str, str], RuntimeGroup]:
    groups: dict[tuple[str, str, str, str], RuntimeGroup] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            aggregation_mode = str(row["aggregation_mode"])
            group = RuntimeGroup(
                category=str(row["category"]),
                method=str(row["method"]),
                dataset=str(row["dataset"]),
                variant=str(row.get("variant", "")),
                aggregation_mode=aggregation_mode,
                representative_target=str(row.get("representative_target", "")),
                variable_count=max(1, int(row.get("variable_count") or 1)),
                group_member_count=max(1, int(row.get("group_member_count") or 1)),
                compression_hours=float(row["estimated_training_hours"]),
                inference_seconds=float(row["estimated_total_inference_seconds"]),
                source_kind=(
                    "representative" if aggregation_mode == "representative_scaled"
                    else "measured"
                ),
                source_label=str(path),
            )
            groups[group.key] = group
    return groups


def expected_group_keys(rows: list[dict[str, str]]) -> list[tuple[str, str, str, str]]:
    seen: set[tuple[str, str, str, str]] = set()
    result: list[tuple[str, str, str, str]] = []
    for row in rows:
        key = group_key(row)
        if key[0] not in RESULT_CATEGORIES or key in seen:
            continue
        seen.add(key)
        result.append(key)
    return result


def sibling_candidates(
    groups: dict[tuple[str, str, str, str], RuntimeGroup],
    key: tuple[str, str, str, str],
) -> list[RuntimeGroup]:
    category, method, dataset, variant = key
    candidates = [
        group for group in groups.values()
        if group.category == category
        and group.method == method
        and group.dataset == dataset
        and group.source_kind != "proxy"
    ]
    if category == "Sensitivity" and "/" in variant:
        family = variant.split("/", 1)[0]
        family_candidates = [
            group for group in candidates
            if group.variant.startswith(f"{family}/")
        ]
        if family_candidates:
            return family_candidates
    return candidates


def estimate_missing_groups(
    groups: dict[tuple[str, str, str, str], RuntimeGroup],
    expected: Iterable[tuple[str, str, str, str]],
) -> dict[tuple[str, str, str, str], RuntimeGroup]:
    result = dict(groups)
    pending = [key for key in expected if key not in result]

    for key in list(pending):
        category, method, dataset, variant = key
        proxy_method = PROXY_METHODS.get(method)
        proxy = result.get((category, proxy_method or "", dataset, variant))
        if proxy is None:
            continue
        result[key] = replace(
            proxy,
            method=method,
            source_kind="proxy",
            source_label=f"estimated from {proxy_method}/{dataset}",
        )
        pending.remove(key)

    # Future missing ablation, sensitivity, scaling, or RD groups use the
    # median of their closest measured siblings. The supplied batch does not
    # currently need this fallback.
    for key in list(pending):
        candidates = sibling_candidates(result, key)
        if not candidates:
            _, method, dataset, _ = key
            main = result.get(("Main", method, dataset, ""))
            candidates = [main] if main is not None else []
        if not candidates:
            continue
        category, method, dataset, variant = key
        template = candidates[0]
        result[key] = RuntimeGroup(
            category=category,
            method=method,
            dataset=dataset,
            variant=variant,
            aggregation_mode="estimated",
            representative_target="",
            variable_count=template.variable_count,
            group_member_count=template.group_member_count,
            compression_hours=statistics.median(
                group.compression_hours for group in candidates
            ),
            inference_seconds=statistics.median(
                group.inference_seconds for group in candidates
            ),
            source_kind="sibling_estimate",
            source_label="estimated from sibling median",
        )
        pending.remove(key)

    if pending:
        labels = ["/".join(part for part in key if part) for key in pending]
        raise ValueError(f"Unable to estimate runtime groups: {labels}")
    return result


def detail_runtime(group: RuntimeGroup) -> tuple[float, float, str]:
    if group.source_kind == "proxy":
        divisor = max(1, group.variable_count)
        return (
            group.compression_hours / divisor,
            group.inference_seconds / divisor,
            group.source_label,
        )
    if group.aggregation_mode == "representative_scaled":
        divisor = max(1, group.variable_count)
        return (
            group.compression_hours / divisor,
            group.inference_seconds / divisor,
            f"representative: {group.representative_target}",
        )
    if group.source_kind == "sibling_estimate":
        return group.compression_hours, group.inference_seconds, group.source_label
    return group.compression_hours, group.inference_seconds, "measured"


def is_separator(cells: list[str]) -> bool:
    return bool(cells) and all(cell and set(cell) <= {"-", ":"} for cell in cells)


def update_detail_tables(
    path: Path,
    groups: dict[tuple[str, str, str, str], RuntimeGroup],
) -> int:
    lines = path.read_text(encoding="utf-8").splitlines()
    category = ""
    updates = 0
    output: list[str] = []
    for line in lines:
        if line.startswith("## "):
            category = line[3:].strip()
        if category in RESULT_CATEGORIES and line.startswith("| ") and line.endswith(" |"):
            cells = [cell.strip() for cell in line[1:-1].split("|")]
            if "PSNR(dB)" in cells:
                cells = [
                    *cells[:7], "Compression(h)", "Inference(s)", "Runtime basis"
                ]
            elif is_separator(cells):
                cells = ["---"] * 10
            elif len(cells) >= 7:
                row = {
                    "category": category,
                    "method": cells[0],
                    "dataset": cells[1],
                    "item": cells[2],
                }
                group = groups[group_key(row)]
                compression, inference, basis = detail_runtime(group)
                cells = [
                    *cells[:7], f"{compression:.3f}", f"{inference:.2f}", basis
                ]
                updates += 1
            line = "| " + " | ".join(cells) + " |"
        output.append(line)

    output = [line for line in output if not line.startswith(RUNTIME_MARKER_PREFIX)]
    insertion = next(
        (index + 1 for index, line in enumerate(output) if line.startswith("# ")), 1
    )
    while insertion < len(output) and not output[insertion].strip():
        del output[insertion]
    output[insertion:insertion] = ["", RUNTIME_MARKER, ""]
    path.write_text("\n".join(output) + "\n", encoding="utf-8")
    return updates


def markdown_table(headers: Iterable[str], rows: Iterable[Iterable[str]]) -> list[str]:
    header = list(headers)
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join("---" for _ in header) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return lines


def display_value(group: RuntimeGroup | None, metric: str) -> str:
    if group is None:
        return "—"
    value = (
        group.compression_hours if metric == "compression"
        else group.inference_seconds
    )
    suffix = "‡" if group.source_kind in {"proxy", "sibling_estimate"} else (
        "†" if group.aggregation_mode == "representative_scaled" else ""
    )
    return f"{value:.2f}{suffix}"


def comparison_tables(
    groups: dict[tuple[str, str, str, str], RuntimeGroup],
    *,
    metric: str,
    label: str,
) -> list[str]:
    lines = [f"### {label}", "", "#### Main", ""]
    lines.extend(markdown_table(
        ["Model", *MAIN_DATASET_ORDER],
        [
            [
                method,
                *[
                    display_value(groups.get(("Main", method, dataset, "")), metric)
                    for dataset in MAIN_DATASET_ORDER
                ],
            ]
            for method in MAIN_METHOD_ORDER
        ],
    ))
    lines.append("")
    for dataset in RD_DATASET_ORDER:
        lines.extend([f"#### RD Curve - {dataset}", ""])
        lines.extend(markdown_table(
            ["Model", *RD_SIZE_ORDER],
            [
                [
                    method,
                    *[
                        display_value(
                            groups.get(("RD Curve", method, dataset, size)), metric
                        )
                        for size in RD_SIZE_ORDER
                    ],
                ]
                for method in RD_METHOD_ORDER
            ],
        ))
        lines.append("")
    return lines


def variant_rows(
    expected: list[tuple[str, str, str, str]],
    groups: dict[tuple[str, str, str, str], RuntimeGroup],
    category: str,
) -> list[list[str]]:
    return [
        [
            key[3],
            display_value(groups[key], "compression"),
            display_value(groups[key], "inference"),
            groups[key].source_label if groups[key].source_kind != "measured" else "measured",
        ]
        for key in expected if key[0] == category
    ]


def sensitivity_tables(
    expected: list[tuple[str, str, str, str]],
    groups: dict[tuple[str, str, str, str], RuntimeGroup],
) -> list[str]:
    lines = ["### Sensitivity", ""]
    for family in SENSITIVITY_FAMILY_ORDER:
        keys = [
            key for key in expected
            if key[0] == "Sensitivity" and key[3].startswith(f"{family}/")
        ]
        if not keys:
            continue
        labels = [key[3].split("/", 1)[1] for key in keys]
        lines.extend([f"#### {family}", ""])
        lines.extend(markdown_table(
            ["Metric", *labels],
            [
                ["Compression (h)", *[display_value(groups[key], "compression") for key in keys]],
                ["Inference (s)", *[display_value(groups[key], "inference") for key in keys]],
            ],
        ))
        lines.append("")
    return lines


def aggregated_runtime_block(
    expected: list[tuple[str, str, str, str]],
    groups: dict[tuple[str, str, str, str], RuntimeGroup],
) -> list[str]:
    lines = [
        RUNTIME_START,
        "## Aggregated Compression and Inference Time",
        "",
        "Compression time uses the common 14.4-billion-sample budget. Inference "
        "time is checkpoint load once plus projected full-dataset reconstruction. "
        "`†` denotes representative-target scaling; `‡` denotes a documented proxy "
        "or sibling estimate. Unconfigured experiments remain `—`.",
        "",
    ]
    lines.extend(comparison_tables(
        groups, metric="compression", label="Compression Time (hours)"
    ))
    lines.extend(comparison_tables(
        groups, metric="inference", label="Inference Time (seconds)"
    ))
    lines.extend(["### Ablation Study", ""])
    lines.extend(markdown_table(
        ["Variant", "Compression (h)", "Inference (s)", "Runtime basis"],
        variant_rows(expected, groups, "Ablation"),
    ))
    lines.append("")
    lines.extend(sensitivity_tables(expected, groups))
    lines.extend(["### Variable Scaling", ""])
    lines.extend(markdown_table(
        ["Variant", "Compression (h)", "Inference (s)", "Runtime basis"],
        variant_rows(expected, groups, "Scaling"),
    ))
    lines.extend(["", RUNTIME_END])
    return lines


def update_aggregated_block(
    path: Path,
    expected: list[tuple[str, str, str, str]],
    groups: dict[tuple[str, str, str, str], RuntimeGroup],
) -> None:
    lines = path.read_text(encoding="utf-8").splitlines()
    block = aggregated_runtime_block(expected, groups)
    if RUNTIME_START in lines:
        start = lines.index(RUNTIME_START)
        end = lines.index(RUNTIME_END, start) + 1
        lines[start:end] = block
    else:
        try:
            insertion = lines.index("<!-- AGGREGATED_PSNR_END -->") + 1
        except ValueError:
            insertion = next(
                (index for index, line in enumerate(lines) if line == "## 逐实验明细"),
                len(lines),
            )
        lines[insertion:insertion] = ["", *block, ""]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_audit(
    path: Path,
    runtime_groups_path: Path,
    expected: list[tuple[str, str, str, str]],
    groups: dict[tuple[str, str, str, str], RuntimeGroup],
) -> None:
    counts = Counter(groups[key].source_kind for key in expected)
    rows = [
        [
            "/".join(part for part in key if part),
            f"{groups[key].compression_hours:.6f}",
            f"{groups[key].inference_seconds:.6f}",
            groups[key].aggregation_mode,
            groups[key].source_kind,
            groups[key].source_label,
        ]
        for key in expected
    ]
    lines = [
        "# Evaluation Runtime Import",
        "",
        f"- Runtime groups source: `{runtime_groups_path.as_posix()}`",
        f"- Expected Result groups: {len(expected)}",
        "- Sources: " + ", ".join(f"{key}={value}" for key, value in sorted(counts.items())),
        "- MINER uses ECNR as its proxy; Neural Experts uses MoE-INR. Proxy values "
        "are estimates, not measurements of the missing method.",
        "",
        *markdown_table(
            [
                "Group", "Compression (h)", "Inference (s)",
                "Aggregation", "Source kind", "Source",
            ],
            rows,
        ),
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    repo = args.repo.resolve()
    result_root = (args.result or repo / "Result").resolve()
    runtime_groups_path = (
        args.runtime_groups
        or repo / "EvalResult" / "batches" / "result_runtime_dataset_total_v2" / "runtime_groups.tsv"
    ).resolve()
    rows = read_manifest(result_root / "MANIFEST.tsv")
    expected = expected_group_keys(rows)
    measured = read_runtime_groups(runtime_groups_path)
    groups = estimate_missing_groups(measured, expected)
    counts = Counter(groups[key].source_kind for key in expected)
    print(
        f"runtime_groups={len(measured)} expected={len(expected)} "
        + " ".join(f"{key}={value}" for key, value in sorted(counts.items()))
    )
    if args.dry_run:
        return 0

    summary_path = result_root / "EXPERIMENT_SUMMARY.md"
    updates = update_detail_tables(summary_path, groups)
    update_aggregated_block(summary_path, expected, groups)
    write_audit(
        result_root / "EVALUATION_RUNTIME.md",
        runtime_groups_path,
        expected,
        groups,
    )
    print(f"summary_updates={updates} summary={summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
