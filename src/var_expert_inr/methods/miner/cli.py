from __future__ import annotations

import argparse
from pathlib import Path

from ...evaluation.cli import add_run_evaluation_arguments, execute_run_evaluation
from .runner import run_predict, run_train


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Native MINER baseline")
    subparsers = parser.add_subparsers(dest="command", required=True)
    train = subparsers.add_parser("train")
    train.add_argument("--config", required=True)
    predict = subparsers.add_parser("predict")
    predict.add_argument("--config", required=True)
    predict.add_argument("--checkpoint", default=None)
    evaluate = subparsers.add_parser("evaluate")
    add_run_evaluation_arguments(evaluate)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "train":
        run_train(Path(args.config))
        return
    if args.command == "predict":
        run_predict(Path(args.config), checkpoint=args.checkpoint)
        return
    if args.command == "evaluate":
        execute_run_evaluation(args)
        return
    raise ValueError(f"Unsupported MINER command: {args.command}")


if __name__ == "__main__":
    main()
