"""Plot per-attribute PSNR against logged training time for VarExpert main runs.

Each dataset produces one figure.  The bottom x-axis uses the cumulative
training time recorded on each ``PSNR epoch`` log line, while the top x-axis
shows the corresponding epoch.  Aggregate PSNR is intentionally excluded so
that every plotted line represents one dataset attribute.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LOGS = {
    "ionization": REPO_ROOT
    / "runs"
    / "var-expert-ionization"
    / "20260716_133736_725222"
    / "logs"
    / "run_20260716_133736_725222.log",
    "combustion": REPO_ROOT
    / "runs"
    / "var-expert-combustion-40NH3-1"
    / "20260807_070953_263505"
    / "logs"
    / "run_20260807_070953_263505.log",
}
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT / "Metric_Fig_Result" / "Fig" / "Main" / "VarExpert-Training-Curves"
)

PSNR_LINE_RE = re.compile(
    r"PSNR epoch\s+(?P<epoch>\d+)/(?P<total_epochs>\d+):\s+"
    r"(?P<metrics>.*?)\s+time=(?P<seconds>[0-9.]+)s(?:\s|$)"
)
METRIC_RE = re.compile(
    r"(?P<name>[A-Za-z][A-Za-z0-9_]*)="
    r"(?P<value>[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)"
)


@dataclass(frozen=True)
class PsnrRecord:
    epoch: int
    total_epochs: int
    seconds: float
    values: dict[str, float]


def parse_psnr_log(log_path: Path) -> list[PsnrRecord]:
    """Extract all epoch-level PSNR records from a VarExpert training log."""
    records: list[PsnrRecord] = []
    with log_path.open("r", encoding="utf-8", errors="replace") as stream:
        for line in stream:
            match = PSNR_LINE_RE.search(line)
            if match is None:
                continue
            values = {
                metric.group("name"): float(metric.group("value"))
                for metric in METRIC_RE.finditer(match.group("metrics"))
            }
            values.pop("aggregate", None)
            records.append(
                PsnrRecord(
                    epoch=int(match.group("epoch")),
                    total_epochs=int(match.group("total_epochs")),
                    seconds=float(match.group("seconds")),
                    values=values,
                )
            )

    if not records:
        raise ValueError(f"No 'PSNR epoch' records found in {log_path}")

    records.sort(key=lambda record: record.epoch)
    reference_names = set(records[0].values)
    for record in records:
        if set(record.values) != reference_names:
            raise ValueError(
                f"Inconsistent per-attribute metrics at epoch {record.epoch} in {log_path}"
            )
    if any(current.seconds <= previous.seconds for previous, current in zip(records, records[1:])):
        raise ValueError(f"Logged training time is not strictly increasing in {log_path}")
    return records


def plot_dataset(
    dataset_name: str,
    records: Iterable[PsnrRecord],
    output_path: Path,
    dpi: int,
) -> None:
    """Create one time-based figure with one line per dataset attribute."""
    records = list(records)
    attributes = list(records[0].values)
    hours = [record.seconds / 3600.0 for record in records]
    epochs = [record.epoch for record in records]

    fig, ax = plt.subplots(figsize=(15.5, 8.0), constrained_layout=True)
    colors = plt.get_cmap("tab20").colors
    markers = ("o", "s", "^", "D", "v", "P", "X", "<", ">", "h", "*", "p", "8")

    for index, attribute in enumerate(attributes):
        ax.plot(
            hours,
            [record.values[attribute] for record in records],
            color=colors[index % len(colors)],
            marker=markers[index % len(markers)],
            linewidth=1.9,
            markersize=5.5,
            label=attribute,
        )

    ax.set_title(f"VarExpert-INR Main: {dataset_name} per-attribute PSNR")
    ax.set_xlabel("Logged cumulative training time (hours)")
    ax.set_ylabel("PSNR (dB)")
    ax.set_xticks(hours)
    ax.set_xticklabels([f"{hour:.2f}" for hour in hours])
    ax.grid(True, which="major", color="#b0b0b0", alpha=0.35, linewidth=0.8)

    epoch_axis = ax.twiny()
    epoch_axis.set_xlim(ax.get_xlim())
    epoch_axis.set_xticks(hours)
    epoch_axis.set_xticklabels([str(epoch) for epoch in epochs])
    epoch_axis.set_xlabel("Epoch")

    ax.legend(
        title="Attribute",
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        frameon=False,
        fontsize=9,
        title_fontsize=10,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Plot VarExpert main-run per-attribute PSNR against logged cumulative "
            "training time, with epoch labels on a secondary x-axis."
        )
    )
    parser.add_argument(
        "--ionization-log",
        type=Path,
        default=DEFAULT_LOGS["ionization"],
        help="Ionization VarExpert main log.",
    )
    parser.add_argument(
        "--combustion-log",
        type=Path,
        default=DEFAULT_LOGS["combustion"],
        help="Combustion VarExpert main log.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for the two PNG figures.",
    )
    parser.add_argument("--dpi", type=int, default=220, help="Output resolution.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    jobs = (
        ("Ionization", args.ionization_log, "ionization_psnr_training_time.png"),
        (
            "Combustion 40NH3 1",
            args.combustion_log,
            "combustion_40NH3_1_psnr_training_time.png",
        ),
    )
    for display_name, log_path, filename in jobs:
        records = parse_psnr_log(log_path)
        output_path = args.output_dir / filename
        plot_dataset(display_name, records, output_path, args.dpi)
        print(f"{display_name}: {len(records)} PSNR checkpoints -> {output_path}")


if __name__ == "__main__":
    main()
