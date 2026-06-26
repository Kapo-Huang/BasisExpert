from __future__ import annotations

import argparse
from pathlib import Path

from .runner import run_train


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Standalone APMGSRN entrypoint")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train APMGSRN models across ionization timesteps")
    train_parser.add_argument("--config", required=True, help="Path to APMGSRN config YAML")
    train_parser.add_argument("--target", default=None, help="Optional target override when DATA.targets is used")
    train_parser.add_argument("--identifier", default=None, help="Optional exp_id override")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "train":
        run_train(Path(args.config), target=args.target, identifier=args.identifier)
        return
    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
