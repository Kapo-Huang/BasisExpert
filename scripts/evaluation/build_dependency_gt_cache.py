from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from var_expert_inr.evaluation.dependency_cache import build_dependency_caches


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build reusable GT Pearson/MI dependency caches")
    parser.add_argument(
        "--config",
        default=str(REPO_ROOT / "configs/evaluation/evaluation_result_dependency.yaml"),
    )
    parser.add_argument("--dataset", default="all", help="all or one configured dataset name")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = build_dependency_caches(
        args.config,
        dataset=args.dataset,
        overwrite=args.overwrite,
    )
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
