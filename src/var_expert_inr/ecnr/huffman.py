from __future__ import annotations

import heapq
from collections import Counter
from dataclasses import dataclass
from itertools import count

import numpy as np


@dataclass(frozen=True)
class HuffmanStream:
    shape: tuple[int, ...]
    dtype: str
    symbols: tuple[int, ...]
    code_lengths: tuple[int, ...]
    bit_length: int
    payload: bytes


def _code_lengths(values: np.ndarray) -> dict[int, int]:
    frequencies = Counter(int(value) for value in values.tolist())
    if not frequencies:
        return {}
    if len(frequencies) == 1:
        return {next(iter(frequencies)): 1}
    serial = count()
    heap = [(frequency, symbol, next(serial), symbol) for symbol, frequency in sorted(frequencies.items())]
    heapq.heapify(heap)
    while len(heap) > 1:
        first = heapq.heappop(heap)
        second = heapq.heappop(heap)
        minimum_symbol = min(first[1], second[1])
        heapq.heappush(
            heap,
            (first[0] + second[0], minimum_symbol, next(serial), (first[3], second[3])),
        )
    lengths: dict[int, int] = {}

    def visit(node, depth: int) -> None:
        if isinstance(node, int):
            lengths[node] = max(depth, 1)
            return
        visit(node[0], depth + 1)
        visit(node[1], depth + 1)

    visit(heap[0][3], 0)
    return lengths


def _canonical_codes(lengths: dict[int, int]) -> dict[int, tuple[int, int]]:
    ordered = sorted(lengths.items(), key=lambda item: (item[1], item[0]))
    codes: dict[int, tuple[int, int]] = {}
    code = 0
    previous_length = 0
    for symbol, length in ordered:
        code <<= length - previous_length
        codes[symbol] = (code, length)
        code += 1
        previous_length = length
    return codes


def encode(array: np.ndarray) -> HuffmanStream:
    values = np.asarray(array)
    if values.dtype.kind not in {"b", "i", "u"}:
        raise TypeError("Huffman encoding only supports integer and boolean arrays")
    flat = values.reshape(-1)
    lengths = _code_lengths(flat)
    codes = _canonical_codes(lengths)
    output = bytearray()
    accumulator = 0
    bits_in_accumulator = 0
    bit_length = 0
    for raw in flat.tolist():
        code, length = codes[int(raw)]
        accumulator = (accumulator << length) | code
        bits_in_accumulator += length
        bit_length += length
        while bits_in_accumulator >= 8:
            shift = bits_in_accumulator - 8
            output.append((accumulator >> shift) & 0xFF)
            accumulator &= (1 << shift) - 1 if shift else 0
            bits_in_accumulator = shift
    if bits_in_accumulator:
        output.append((accumulator << (8 - bits_in_accumulator)) & 0xFF)
    symbols = tuple(sorted(lengths))
    return HuffmanStream(
        shape=tuple(int(value) for value in values.shape),
        dtype=values.dtype.str,
        symbols=symbols,
        code_lengths=tuple(int(lengths[symbol]) for symbol in symbols),
        bit_length=int(bit_length),
        payload=bytes(output),
    )


def decode(stream: HuffmanStream) -> np.ndarray:
    lengths = dict(zip(stream.symbols, stream.code_lengths))
    codes = _canonical_codes(lengths)
    reverse = {(length, code): symbol for symbol, (code, length) in codes.items()}
    expected = int(np.prod(stream.shape, dtype=np.int64))
    result: list[int] = []
    code = 0
    length = 0
    consumed = 0
    for byte in stream.payload:
        for shift in range(7, -1, -1):
            if consumed >= stream.bit_length:
                break
            code = (code << 1) | ((byte >> shift) & 1)
            length += 1
            consumed += 1
            key = (length, code)
            if key in reverse:
                result.append(reverse[key])
                code = 0
                length = 0
    if len(result) != expected or length != 0:
        raise ValueError("Invalid or truncated Huffman stream")
    return np.asarray(result, dtype=np.dtype(stream.dtype)).reshape(stream.shape)
