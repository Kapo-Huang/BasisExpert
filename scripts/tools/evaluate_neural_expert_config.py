from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

# This file is invoked directly by the batch runner. In that mode Python adds the
# scripts directory, rather than the repository or its src tree, to sys.path.
_SRC_ROOT = Path(__file__).resolve().parents[2] / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from var_expert_inr.evaluation.service import evaluate_run
from var_expert_inr.methods.neural_expert.config import load_config, run_dir_from_config


RESULT_MARKER = "NEURAL_EXPERT_PSNR"


def evaluate_config(config_path: str | Path, *, device: str | None = None) -> dict[str, object]:
    path = Path(config_path).resolve()
    cfg = load_config(path)
    if bool(cfg["TRAINING"].get("segmentation_mode", False)):
        raise ValueError(f"Manager-pretrain config cannot be evaluated for reconstruction PSNR: {path}")

    target = str(cfg["DATA"]["target"])
    run_dir = run_dir_from_config(cfg).resolve()
    result = evaluate_run(
        run_dir,
        metrics="psnr",
        timesteps="all",
        targets=target,
        source="auto",
        device=device,
    )
    metrics = result.get("metrics") or {}
    if metrics.get("status") != "complete":
        raise RuntimeError(f"NeuralExpert evaluation did not complete for {path}")
    aggregate = metrics.get("aggregate") or {}
    if "psnr" not in aggregate:
        raise RuntimeError(f"NeuralExpert evaluation did not report aggregate PSNR for {path}")
    psnr = float(aggregate["psnr"])
    if not math.isfinite(psnr):
        raise RuntimeError(f"NeuralExpert evaluation reported non-finite PSNR for {path}: {psnr}")

    return {
        "config": path,
        "target": target,
        "psnr": psnr,
        "run_dir": run_dir,
        "metrics_path": Path(result["metrics_path"]).resolve(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate one completed NeuralExpert main config")
    parser.add_argument("--config", required=True, help="Path to a NeuralExpert main YAML")
    parser.add_argument("--device", default=None, help="Optional evaluation device override")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    record = evaluate_config(args.config, device=args.device)
    print(
        "\t".join(
            [
                RESULT_MARKER,
                str(record["target"]),
                format(float(record["psnr"]), ".12g"),
                str(record["run_dir"]),
                str(record["metrics_path"]),
            ]
        )
    )


if __name__ == "__main__":
    main()
