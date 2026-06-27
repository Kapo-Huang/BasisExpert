from __future__ import annotations

import argparse

from .runner import run_evaluate, run_predict, run_train


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Standalone temporal RMDSRN entrypoint")
    commands = parser.add_subparsers(dest="command", required=True)

    train = commands.add_parser("train", help="Train an RMDSRN model")
    train.add_argument("--config", required=True)
    train.add_argument("--target", default=None)
    train.add_argument("--resume", default=None)

    for name in ("predict", "evaluate"):
        command = commands.add_parser(name, help=f"{name.title()} an RMDSRN model")
        command.add_argument("--config", required=True)
        command.add_argument("--target", default=None)
        source = command.add_mutually_exclusive_group()
        source.add_argument("--artifact", default=None)
        source.add_argument("--checkpoint", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "train":
        run_train(args.config, target=args.target, resume=args.resume)
    elif args.command == "predict":
        run_predict(
            args.config,
            target=args.target,
            artifact=args.artifact,
            checkpoint=args.checkpoint,
        )
    else:
        run_evaluate(
            args.config,
            target=args.target,
            artifact=args.artifact,
            checkpoint=args.checkpoint,
        )


if __name__ == "__main__":
    main()
