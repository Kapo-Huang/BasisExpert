from __future__ import annotations

import logging
import os
from datetime import datetime
from pathlib import Path


def _resolve_level(level: str | None) -> int:
    level_name = (level or os.getenv("VAR_EXPERT_INR_LOG_LEVEL", "INFO")).strip().upper()
    mapping = getattr(logging, "_nameToLevel", {})
    return int(mapping.get(level_name, logging.INFO))


def setup_logging(
    *,
    level: str | None = None,
    log_dir: str | Path | None = None,
    log_file: str | None = None,
) -> None:
    resolved_level = _resolve_level(level)
    root = logging.getLogger()
    if not root.handlers:
        logging.basicConfig(
            level=resolved_level,
            format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        )
    else:
        root.setLevel(resolved_level)

    if log_dir is None:
        return

    log_dir_path = Path(log_dir)
    log_dir_path.mkdir(parents=True, exist_ok=True)
    if log_file is None:
        log_file = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    log_path = log_dir_path / log_file

    for handler in list(root.handlers):
        if isinstance(handler, logging.FileHandler):
            if Path(handler.baseFilename) == log_path:
                return
            root.removeHandler(handler)
            handler.close()

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(resolved_level)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    )
    root.addHandler(file_handler)


def close_file_handlers() -> None:
    root = logging.getLogger()
    for handler in list(root.handlers):
        if isinstance(handler, logging.FileHandler):
            root.removeHandler(handler)
            handler.close()
