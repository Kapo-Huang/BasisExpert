from __future__ import annotations

from importlib import import_module
from typing import Any


_EXPORT_MODULES = {
    "load_checkpoint": ".checkpoint",
    "save_checkpoint": ".checkpoint",
    "dump_yaml": ".io",
    "load_yaml": ".io",
    "close_file_handlers": ".logging_utils",
    "setup_logging": ".logging_utils",
    "build_model_catalog_row": ".model_stats",
    "collect_model_statistics": ".model_stats",
    "format_fp16_size_megabytes": ".model_stats",
    "format_param_count": ".model_stats",
    "upsert_model_catalog": ".model_stats",
    "configure_thread_env": ".runtime",
    "set_random_seed": ".runtime",
}


def __getattr__(name: str) -> Any:
    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(name)
    value = getattr(import_module(module_name, __name__), name)
    globals()[name] = value
    return value


__all__ = list(_EXPORT_MODULES)
