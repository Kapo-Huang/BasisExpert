from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Mapping

import torch

from .io import ensure_parent, sha256_payload, stable_json_dumps

MODEL_CATALOG_FIXED_COLUMNS = (
    "model_name",
    "param_count",
    "trainable_param_count",
    "fp16_size_bytes",
    "fp16_size_mb",
    "model_config_hash",
)
_FP16_BYTES_PER_PARAM = 2
_BYTES_PER_MIB = 1024.0 * 1024.0


def _weight_bias_numel(model: torch.nn.Module) -> int:
    total = 0
    for name, param in model.named_parameters():
        if name.endswith("weight") or name.endswith("bias"):
            total += int(param.numel())
    return total


def collect_model_statistics(model: torch.nn.Module) -> dict[str, int | float]:
    param_count = sum(int(param.numel()) for param in model.parameters())
    trainable_param_count = sum(int(param.numel()) for param in model.parameters() if param.requires_grad)
    fp16_size_bytes = _weight_bias_numel(model) * _FP16_BYTES_PER_PARAM
    return {
        "param_count": param_count,
        "trainable_param_count": trainable_param_count,
        "fp16_size_bytes": fp16_size_bytes,
        "fp16_size_mb": fp16_size_bytes / _BYTES_PER_MIB,
    }


def format_param_count(count: int) -> str:
    return f"{int(count):,}"


def format_fp16_size_megabytes(num_bytes: int) -> str:
    return f"{float(num_bytes) / _BYTES_PER_MIB:.2f} MB"


def build_model_config_hash(model_name: str, model_params: Mapping[str, Any]) -> str:
    return sha256_payload(
        {
            "model_name": str(model_name),
            "model_params": _normalize_json_value(dict(model_params)),
        }
    )


def build_model_catalog_row(
    *,
    model_name: str,
    model_params: Mapping[str, Any],
    stats: Mapping[str, int | float],
) -> dict[str, str]:
    row = {
        "model_name": str(model_name),
        "param_count": str(int(stats["param_count"])),
        "trainable_param_count": str(int(stats["trainable_param_count"])),
        "fp16_size_bytes": str(int(stats["fp16_size_bytes"])),
        "fp16_size_mb": _serialize_csv_value(float(stats["fp16_size_mb"])),
        "model_config_hash": build_model_config_hash(model_name, model_params),
    }
    for key, value in model_params.items():
        row[str(key)] = _serialize_csv_value(value)
    return row


def upsert_model_catalog(path: str | Path, row: Mapping[str, Any]) -> bool:
    catalog_path = ensure_parent(path)
    existing_rows = _load_catalog_rows(catalog_path)
    existing_by_key: dict[tuple[str, str], dict[str, str]] = {}
    inserted = True
    for existing in existing_rows:
        key = _catalog_row_key(existing)
        if key not in existing_by_key:
            existing_by_key[key] = dict(existing)

    key = _catalog_row_key(row)
    if key in existing_by_key:
        inserted = False
        existing_by_key[key] = {**existing_by_key[key], **_stringify_row(row)}
    else:
        existing_by_key[key] = _stringify_row(row)

    merged_rows = sorted(
        existing_by_key.values(),
        key=lambda current: (current.get("model_name", ""), current.get("model_config_hash", "")),
    )
    fieldnames = _catalog_fieldnames(merged_rows)
    with catalog_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for current in merged_rows:
            writer.writerow({field: current.get(field, "") for field in fieldnames})
    return inserted


def _normalize_json_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _normalize_json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalize_json_value(item) for item in value]
    if hasattr(value, "item") and callable(getattr(value, "item")):
        try:
            return value.item()
        except ValueError:
            return value
    return value


def _serialize_csv_value(value: Any) -> str:
    normalized = _normalize_json_value(value)
    if normalized is None:
        return ""
    if isinstance(normalized, bool):
        return "true" if normalized else "false"
    if isinstance(normalized, int):
        return str(normalized)
    if isinstance(normalized, float):
        return format(normalized, ".15g")
    if isinstance(normalized, str):
        return normalized
    return stable_json_dumps(normalized)


def _catalog_row_key(row: Mapping[str, Any]) -> tuple[str, str]:
    return str(row.get("model_name", "")), str(row.get("model_config_hash", ""))


def _catalog_fieldnames(rows: list[Mapping[str, str]]) -> list[str]:
    dynamic = sorted(
        {
            key
            for row in rows
            for key in row.keys()
            if key not in MODEL_CATALOG_FIXED_COLUMNS
        }
    )
    return [*MODEL_CATALOG_FIXED_COLUMNS, *dynamic]


def _stringify_row(row: Mapping[str, Any]) -> dict[str, str]:
    return {str(key): _serialize_csv_value(value) for key, value in row.items()}


def _load_catalog_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            return []
        return [{str(key): value or "" for key, value in row.items()} for row in reader]
