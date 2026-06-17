from __future__ import annotations

import argparse
from pathlib import Path

from .config import load_config


def run_train(config_path: str | Path, *, target: str | None = None, identifier: str | None = None, gpu: int = 0) -> dict:
    cfg = load_config(config_path, target_override=target, identifier=identifier)
    dataset_name = cfg["DATA"]["dataset_name"]
    if dataset_name == "ionization":
        from .ionization.runner import run_train as run_ionization_train

        return run_ionization_train(cfg, gpu=gpu)
    if dataset_name in {"katrina", "linkage_p", "linkage_c"}:
        from .mesh.runner import run_train as run_mesh_train

        return run_mesh_train(cfg, gpu=gpu)
    raise ValueError(f"Unsupported NeuralExpert dataset_name: {dataset_name!r}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Standalone NeuralExpert entrypoint")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train a NeuralExpert model")
    train_parser.add_argument("--config", required=True, help="Path to NeuralExpert config YAML")
    train_parser.add_argument("--target", default=None, help="Optional target override when DATA.targets is used")
    train_parser.add_argument("--identifier", default=None, help="Optional run id override")
    train_parser.add_argument("--gpu", default=0, type=int, help="GPU index to use when CUDA is available")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "train":
        run_train(args.config, target=args.target, identifier=args.identifier, gpu=int(args.gpu))
        return
    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
