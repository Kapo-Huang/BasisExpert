from __future__ import annotations

import argparse
from pathlib import Path

from .runner import run_evaluate, run_predict, run_train


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Standalone DC-INR entrypoint")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train a DC-INR model")
    train_parser.add_argument("--config", required=True, help="Path to DC-INR config YAML")
    train_parser.add_argument("--target", default=None, help="Optional target override when data.targets is used")

    predict_parser = subparsers.add_parser("predict", help="Generate predictions from a DC-INR checkpoint")
    predict_parser.add_argument("--config", required=True, help="Path to DC-INR config YAML")
    predict_parser.add_argument("--checkpoint", default=None, help="Optional explicit checkpoint path")
    predict_parser.add_argument("--target", default=None, help="Optional target override when data.targets is used")

    evaluate_parser = subparsers.add_parser("evaluate", help="Evaluate a DC-INR checkpoint")
    evaluate_parser.add_argument("--config", required=True, help="Path to DC-INR config YAML")
    evaluate_parser.add_argument("--checkpoint", default=None, help="Optional explicit checkpoint path")
    evaluate_parser.add_argument("--target", default=None, help="Optional target override when data.targets is used")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    if args.command == "train":
        run_train(config_path, target=args.target)
        return
    if args.command == "predict":
        run_predict(config_path, checkpoint_path=args.checkpoint, target=args.target)
        return
    if args.command == "evaluate":
        run_evaluate(config_path, checkpoint_path=args.checkpoint, target=args.target)
        return
    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
