#!/usr/bin/env python3
"""Import completed training-memory summaries into ``EXPERIMENT_SUMMARY.md``."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable


START = "<!-- AGGREGATED_TRAINING_MEMORY_START -->"
END = "<!-- AGGREGATED_TRAINING_MEMORY_END -->"
GIB = float(1024**3)
MAIN_METHODS = ("Ours", "CoordNet", "MoE-INR", "fV-SRN", "MINER", "STSR-INR")
MAIN_DATASETS = ("Ionization", "Combustion", "Katrina", "RedSea")
RD_DATASETS = ("Ionization", "Combustion")
RD_SIZES = ("0.41", "0.82", "1.63", "3.26")


def parse_args() -> argparse.Namespace:
    repo = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary",
        type=Path,
        default=repo / "EvalResult/batches/training_memory_selected_v1/training_memory_summary.json",
    )
    parser.add_argument(
        "--result",
        type=Path,
        default=repo / "Result/EXPERIMENT_SUMMARY.md",
    )
    return parser.parse_args()


def markdown_table(headers: Iterable[str], rows: Iterable[Iterable[str]]) -> list[str]:
    header = list(headers)
    return [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join("---" for _ in header) + " |",
        *("| " + " | ".join(row) + " |" for row in rows),
    ]


def memory_value(entry: dict[str, Any] | None, field: str) -> str:
    if entry is None or entry.get(field) is None:
        return "-"
    suffix = "*" if entry.get("aggregation_mode") == "representative" else ""
    return f"{float(entry[field]) / GIB:.2f}{suffix}"


def sort_expert(entry: dict[str, Any]) -> int:
    label = str(entry["variant"]).rsplit("/", 1)[-1]
    return int(label[1:])


def grouped(entries: Iterable[dict[str, Any]]) -> dict[tuple[str, str, str, str], dict[str, Any]]:
    return {
        (str(row["category"]), str(row["method"]), str(row["dataset"]), str(row["variant"])): row
        for row in entries
    }


def comparison_tables(
    values: dict[tuple[str, str, str, str], dict[str, Any]],
    *,
    field: str,
    label: str,
) -> list[str]:
    lines = [f"### {label}", "", "#### Main", ""]
    lines.extend(markdown_table(
        ["Model", *MAIN_DATASETS],
        [
            [method, *[
                memory_value(values.get(("Main", method, dataset, "")), field)
                for dataset in MAIN_DATASETS
            ]]
            for method in MAIN_METHODS
        ],
    ))
    lines.append("")
    for dataset in RD_DATASETS:
        lines.extend([f"#### RD Curve - {dataset}", ""])
        lines.extend(markdown_table(
            ["Model", *RD_SIZES],
            [
                [method, *[
                    memory_value(values.get(("RD Curve", method, dataset, size)), field)
                    for size in RD_SIZES
                ]]
                for method in MAIN_METHODS
            ],
        ))
        lines.append("")
    return lines


def variant_tables(entries: list[dict[str, Any]]) -> list[str]:
    ablation = sorted(
        (row for row in entries if row["category"] == "Ablation"),
        key=lambda row: str(row["variant"]).casefold(),
    )
    experts = sorted(
        (
            row for row in entries
            if row["category"] == "Sensitivity"
            and str(row["variant"]).startswith("ExpertNum/")
        ),
        key=sort_expert,
    )
    lines = ["### Ablation Study", ""]
    lines.extend(markdown_table(
        ["Variant", "CPU RSS peak (GiB)", "CUDA allocated (GiB)", "CUDA reserved (GiB)"],
        [[
            str(row["variant"]),
            memory_value(row, "cpu_rss_peak_bytes"),
            memory_value(row, "cuda_peak_allocated_bytes"),
            memory_value(row, "cuda_peak_reserved_bytes"),
        ] for row in ablation],
    ))
    lines.extend(["", "### Sensitivity", "", "#### ExpertNum", ""])
    labels = [str(row["variant"]).split("/", 1)[1] for row in experts]
    lines.extend(markdown_table(
        ["Metric", *labels],
        [
            ["CPU RSS peak (GiB)", *[memory_value(row, "cpu_rss_peak_bytes") for row in experts]],
            ["CUDA allocated (GiB)", *[memory_value(row, "cuda_peak_allocated_bytes") for row in experts]],
            ["CUDA reserved (GiB)", *[memory_value(row, "cuda_peak_reserved_bytes") for row in experts]],
        ],
    ))
    return lines


def detail_table(entries: list[dict[str, Any]]) -> list[str]:
    ordered = sorted(
        entries,
        key=lambda row: tuple(str(row[key]).casefold() for key in ("category", "method", "dataset", "variant")),
    )
    rows = [[
        str(row["category"]), str(row["method"]), str(row["dataset"]),
        str(row["variant"] or "-"), str(row["aggregation_mode"]),
        str(row["representative_target"]),
        memory_value(row, "cpu_rss_peak_bytes"),
        memory_value(row, "cpu_rss_peak_delta_bytes"),
        memory_value(row, "cuda_peak_allocated_bytes"),
        memory_value(row, "cuda_peak_reserved_bytes"),
    ] for row in ordered]
    return [
        "### Per-probe Training Memory",
        "",
        *markdown_table(
            [
                "Category", "Method", "Dataset", "Variant", "Mode", "Representative",
                "CPU RSS peak (GiB)", "CPU RSS delta (GiB)",
                "CUDA allocated (GiB)", "CUDA reserved (GiB)",
            ],
            rows,
        ),
    ]


def build_block(entries: list[dict[str, Any]]) -> list[str]:
    values = grouped(entries)
    lines = [
        START,
        "## Aggregated Training Peak Memory",
        "",
        "Values are peak memory from the complete bounded training probe (model and "
        "optimizer construction, data loading, and training). CUDA reserved is the "
        "PyTorch caching-allocator peak; CPU values are process RSS peaks. All values "
        "are GiB. `*` denotes a representative per-variable probe; it is not scaled "
        "by the dataset variable count.",
        "",
    ]
    lines.extend(comparison_tables(values, field="cuda_peak_reserved_bytes", label="CUDA Peak Reserved (GiB)"))
    lines.extend(comparison_tables(values, field="cpu_rss_peak_bytes", label="CPU RSS Peak (GiB)"))
    lines.extend(variant_tables(entries))
    lines.extend(["", *detail_table(entries), "", END])
    return lines


def update_report(path: Path, entries: list[dict[str, Any]]) -> None:
    lines = path.read_text(encoding="utf-8").splitlines()
    block = build_block(entries)
    if START in lines:
        end = lines.index(END, lines.index(START)) + 1
        lines[lines.index(START):end] = block
    else:
        insertion = lines.index("<!-- AGGREGATED_RUNTIME_END -->") + 1
        lines[insertion:insertion] = ["", *block, ""]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    payload = json.loads(args.summary.read_text(encoding="utf-8"))
    entries = list(payload.get("entries") or [])
    if not entries:
        raise ValueError(f"No training-memory entries in {args.summary}")
    update_report(args.result, entries)
    print(f"Imported {len(entries)} training-memory probes into {args.result}")


if __name__ == "__main__":
    main()
