from __future__ import annotations

from collections.abc import Iterable


ALL_TOKEN = "all"


def parse_name_selection(value: str | Iterable[str] | None) -> tuple[str, ...] | None:
    """Return selected names, or ``None`` for all names."""
    if value is None:
        return None
    raw = [value] if isinstance(value, str) else list(value)
    tokens: list[str] = []
    for item in raw:
        tokens.extend(part.strip() for part in str(item).split(",") if part.strip())
    if not tokens or any(token.lower() == ALL_TOKEN for token in tokens):
        if len(tokens) > 1:
            raise ValueError("'all' cannot be combined with explicit target names")
        return None
    return tuple(dict.fromkeys(tokens))


def parse_timestep_selection(value: str | Iterable[int] | None, total: int) -> tuple[int, ...]:
    """Parse single values and inclusive ``start:end[:step]`` ranges."""
    total = int(total)
    if total <= 0:
        raise ValueError(f"total timesteps must be positive, got {total}")
    if value is None or (isinstance(value, str) and value.strip().lower() == ALL_TOKEN):
        return tuple(range(total))
    if not isinstance(value, str):
        selected = [int(item) for item in value]
    else:
        selected: list[int] = []
        tokens = [part.strip() for part in value.split(",") if part.strip()]
        if not tokens:
            raise ValueError("timestep selection must not be empty")
        if any(token.lower() == ALL_TOKEN for token in tokens):
            raise ValueError("'all' cannot be combined with explicit timesteps")
        for token in tokens:
            fields = token.split(":")
            if len(fields) == 1:
                selected.append(int(fields[0]))
                continue
            if len(fields) not in {2, 3} or not fields[0] or not fields[1]:
                raise ValueError(f"invalid timestep range: {token!r}")
            start, end = int(fields[0]), int(fields[1])
            step = int(fields[2]) if len(fields) == 3 and fields[2] else 1
            if step <= 0:
                raise ValueError(f"timestep range step must be positive: {token!r}")
            if end < start:
                raise ValueError(f"timestep range end precedes start: {token!r}")
            selected.extend(range(start, end + 1, step))
    result = tuple(dict.fromkeys(selected))
    if not result:
        raise ValueError("timestep selection must not be empty")
    invalid = [item for item in result if item < 0 or item >= total]
    if invalid:
        raise IndexError(f"timesteps out of range [0, {total - 1}]: {invalid}")
    return result


def parse_metric_selection(value: str | Iterable[str] | None) -> tuple[str, ...]:
    allowed = ("psnr", "ssim", "lpips", "decode_time", "memory")
    if value is None:
        return ("psnr",)
    raw = [value] if isinstance(value, str) else list(value)
    tokens: list[str] = []
    for item in raw:
        tokens.extend(part.strip().lower() for part in str(item).split(",") if part.strip())
    if not tokens:
        raise ValueError("at least one metric must be selected")
    unknown = sorted(set(tokens).difference(allowed))
    if unknown:
        raise ValueError(f"unsupported metrics: {', '.join(unknown)}; allowed: {', '.join(allowed)}")
    return tuple(dict.fromkeys(tokens))


def metrics_require_ground_truth(metrics: Iterable[str]) -> bool:
    return bool({"psnr", "ssim", "lpips"}.intersection(metrics))


def metrics_require_rendering(metrics: Iterable[str]) -> bool:
    return bool({"ssim", "lpips"}.intersection(metrics))
