from __future__ import annotations

import argparse
from pathlib import Path

from ..evaluation.cli import add_run_evaluation_arguments, execute_run_evaluation
from .runner import run_evaluate, run_predict, run_train


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Standalone MC-INR entrypoint")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train an MC-INR model")
    train_parser.add_argument("--config", required=True, help="Path to MC-INR config YAML")
    train_parser.add_argument("--resume", default=None, help="Optional checkpoint path to resume from")

    predict_parser = subparsers.add_parser("predict", help="Generate predictions from an MC-INR checkpoint")
    predict_parser.add_argument("--config", required=True, help="Path to MC-INR config YAML")
    predict_parser.add_argument("--checkpoint", default=None, help="Optional explicit checkpoint path")

    evaluate_parser = subparsers.add_parser("evaluate", help="Evaluate an MC-INR checkpoint")
    evaluate_parser.add_argument("--config", required=False, help="Legacy config-based evaluation")
    evaluate_parser.add_argument("--checkpoint", default=None, help="Optional explicit checkpoint path")
    add_run_evaluation_arguments(evaluate_parser, run_required=False, include_source_paths=False)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "evaluate" and args.run:
        execute_run_evaluation(args)
        return
    if not getattr(args, "config", None):
        raise ValueError("--config is required unless evaluate uses --run")
    config_path = Path(args.config)
    if args.command == "train":
        run_train(config_path, resume_path=args.resume)
        return
    if args.command == "predict":
        run_predict(config_path, checkpoint_path=args.checkpoint)
        return
    if args.command == "evaluate":
        run_evaluate(config_path, checkpoint_path=args.checkpoint)
        return
    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
