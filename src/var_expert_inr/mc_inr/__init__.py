from __future__ import annotations

from .runner import run_evaluate, run_predict, run_train


def main(argv: list[str] | None = None) -> int:
    from .cli import main as cli_main

    return cli_main(argv)


__all__ = ["main", "run_train", "run_predict", "run_evaluate"]
