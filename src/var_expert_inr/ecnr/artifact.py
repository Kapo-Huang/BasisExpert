from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .huffman import HuffmanStream, decode as huffman_decode, encode as huffman_encode


FORMAT = "ecnr_artifact_v1"
_HUFFMAN_MARKER = "__ecnr_huffman__"


def _pack(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        if value.dtype.kind in {"b", "i", "u"}:
            stream = huffman_encode(value)
            return {_HUFFMAN_MARKER: asdict(stream)}
        if value.dtype.kind == "f":
            return np.asarray(value, dtype="<f4")
        return value
    if isinstance(value, dict):
        return {str(key): _pack(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_pack(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _unpack(value: Any) -> Any:
    if isinstance(value, dict) and set(value) == {_HUFFMAN_MARKER}:
        payload = value[_HUFFMAN_MARKER]
        stream = HuffmanStream(
            shape=tuple(payload["shape"]),
            dtype=str(payload["dtype"]),
            symbols=tuple(payload["symbols"]),
            code_lengths=tuple(payload["code_lengths"]),
            bit_length=int(payload["bit_length"]),
            payload=bytes(payload["payload"]),
        )
        return huffman_decode(stream)
    if isinstance(value, dict):
        return {key: _unpack(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_unpack(item) for item in value]
    return value


def save_artifact(path: str | Path, payload: dict[str, Any]) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"format": FORMAT, "payload": _pack(payload)}, output)
    return output


def load_artifact(path: str | Path) -> dict[str, Any]:
    try:
        wrapper = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        wrapper = torch.load(path, map_location="cpu")
    if wrapper.get("format") != FORMAT:
        raise ValueError("Unsupported ECNR artifact format")
    payload = _unpack(wrapper["payload"])
    if payload.get("format") != FORMAT:
        raise ValueError("Invalid ECNR artifact payload")
    return payload
