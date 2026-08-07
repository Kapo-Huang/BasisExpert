"""Inspect and export dynamically wet Katrina ADCIRC node samples."""

from __future__ import annotations

import argparse
import gc
import json
import logging
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np


LOGGER = logging.getLogger("katrina_wet")
DRY_SENTINEL = -99999.0
COORDINATE_FILE = "source_XYZT.npy"
TARGET_COMPONENTS = {
    "fort63": 1,
    "fort64": 1,
    "fort73": 1,
    "speed": 1,
    "v": 3,
}
OUTPUT_DTYPES = {
    COORDINATE_FILE: np.dtype(np.float32),
    **{f"target_{name}.npy": np.dtype(np.float32) for name in TARGET_COMPONENTS},
    "wet_node_indices.npy": np.dtype(np.int32),
    "frame_offsets.npy": np.dtype(np.int64),
}


@dataclass(frozen=True)
class ScanResult:
    frame_counts: tuple[int, ...]
    frame_offsets: tuple[int, ...]
    coordinate_min: tuple[float, ...]
    coordinate_max: tuple[float, ...]
    target_min: Mapping[str, tuple[float, ...]]
    target_max: Mapping[str, tuple[float, ...]]

    @property
    def sample_count(self) -> int:
        return int(self.frame_offsets[-1])


class KatrinaReader:
    """Memory-mapped reader for the aligned Katrina training arrays."""

    def __init__(self, input_dir: str | Path) -> None:
        self.input_dir = Path(input_dir).expanduser().resolve()
        if not self.input_dir.is_dir():
            raise FileNotFoundError(f"Input directory does not exist: {self.input_dir}")

        self.paths = {
            "coordinates": self.input_dir / COORDINATE_FILE,
            **{
                name: self.input_dir / f"target_{name}.npy"
                for name in TARGET_COMPONENTS
            },
        }
        missing = [str(path) for path in self.paths.values() if not path.is_file()]
        if missing:
            raise FileNotFoundError("Missing Katrina input arrays: " + ", ".join(missing))

        self.coordinates = np.load(
            self.paths["coordinates"], mmap_mode="r", allow_pickle=False
        )
        self.targets = {
            name: np.load(self.paths[name], mmap_mode="r", allow_pickle=False)
            for name in TARGET_COMPONENTS
        }
        self._validate_shapes()
        self.total_rows = int(self.coordinates.shape[0])
        self.node_count, self.frame_count = self._infer_layout()
        self._validate_frame_times()

    def _validate_shapes(self) -> None:
        if self.coordinates.ndim != 2 or self.coordinates.shape[1] != 4:
            raise ValueError(
                f"{COORDINATE_FILE} must have shape (N, 4), got {self.coordinates.shape}"
            )
        if not np.issubdtype(self.coordinates.dtype, np.number):
            raise TypeError(f"{COORDINATE_FILE} must be numeric, got {self.coordinates.dtype}")
        row_count = int(self.coordinates.shape[0])
        if row_count <= 0:
            raise ValueError("Katrina input arrays must not be empty")
        for name, components in TARGET_COMPONENTS.items():
            array = self.targets[name]
            expected = (row_count, components)
            if array.shape != expected:
                raise ValueError(
                    f"target_{name}.npy must have shape {expected}, got {array.shape}"
                )
            if not np.issubdtype(array.dtype, np.number):
                raise TypeError(f"target_{name}.npy must be numeric, got {array.dtype}")

    def _infer_layout(self) -> tuple[int, int]:
        time_values = self.coordinates[:, 3]
        first_time = time_values[0]
        if not np.isfinite(first_time):
            raise ValueError("The first Katrina time coordinate is not finite")
        node_count = int(np.searchsorted(time_values, first_time, side="right"))
        if node_count <= 0 or self.coordinates.shape[0] % node_count != 0:
            raise ValueError(
                "Could not infer a fixed node count from contiguous time-coordinate blocks"
            )
        return node_count, int(self.coordinates.shape[0] // node_count)

    def _validate_frame_times(self) -> None:
        previous = -np.inf
        for frame in range(self.frame_count):
            block = self.coordinates[self.frame_slice(frame), 3]
            time_value = float(block[0])
            if not np.isfinite(block).all() or not np.all(block == time_value):
                raise ValueError(f"Frame {frame} does not contain one finite constant time")
            if time_value <= previous:
                raise ValueError("Katrina frame times must be strictly increasing")
            previous = time_value

    def frame_slice(self, frame: int) -> slice:
        start = int(frame) * self.node_count
        return slice(start, start + self.node_count)

    def source_summary(self) -> dict[str, Any]:
        files: dict[str, Any] = {}
        for key, path in self.paths.items():
            array = self.coordinates if key == "coordinates" else self.targets[key]
            stat = path.stat()
            files[path.name] = {
                "path": str(path),
                "shape": list(array.shape),
                "dtype": str(array.dtype),
                "size_bytes": int(stat.st_size),
                "mtime_ns": int(stat.st_mtime_ns),
            }
        return {
            "input_dir": str(self.input_dir),
            "files": files,
            "sample_count": self.total_rows,
            "frame_count": self.frame_count,
            "nodes_per_frame": self.node_count,
        }


def _update_range(
    low: np.ndarray, high: np.ndarray, values: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    if values.size == 0:
        return low, high
    return np.minimum(low, values.min(axis=0)), np.maximum(high, values.max(axis=0))


def scan_wet_samples(reader: KatrinaReader, *, dry_sentinel: float) -> ScanResult:
    coordinate_min = np.full(4, np.inf, dtype=np.float64)
    coordinate_max = np.full(4, -np.inf, dtype=np.float64)
    target_min = {
        name: np.full(components, np.inf, dtype=np.float64)
        for name, components in TARGET_COMPONENTS.items()
    }
    target_max = {
        name: np.full(components, -np.inf, dtype=np.float64)
        for name, components in TARGET_COMPONENTS.items()
    }
    frame_counts: list[int] = []
    frame_offsets = [0]
    reference_spatial = np.asarray(
        reader.coordinates[reader.frame_slice(0), :3]
    )

    for frame in range(reader.frame_count):
        frame_slice = reader.frame_slice(frame)
        coordinates = np.asarray(reader.coordinates[frame_slice])
        if frame and not np.array_equal(coordinates[:, :3], reference_spatial):
            raise ValueError(f"Frame {frame} spatial coordinates do not match frame 0")

        fort63 = np.asarray(reader.targets["fort63"][frame_slice, 0])
        wet = np.logical_and(np.isfinite(fort63), fort63 != dry_sentinel)
        wet_count = int(wet.sum())
        if wet_count == 0:
            raise ValueError(f"Frame {frame} contains no wet nodes")
        frame_counts.append(wet_count)
        frame_offsets.append(frame_offsets[-1] + wet_count)

        wet_coordinates = coordinates[wet]
        if not np.isfinite(wet_coordinates).all():
            raise ValueError(f"Frame {frame} contains non-finite wet coordinates")
        coordinate_min, coordinate_max = _update_range(
            coordinate_min, coordinate_max, wet_coordinates
        )

        for name, array in reader.targets.items():
            selected = np.asarray(array[frame_slice][wet])
            if not np.isfinite(selected).all():
                raise ValueError(f"Frame {frame} target {name!r} has non-finite wet values")
            target_min[name], target_max[name] = _update_range(
                target_min[name], target_max[name], selected
            )

        if frame == 0 or (frame + 1) % 20 == 0 or frame + 1 == reader.frame_count:
            LOGGER.info(
                "Scanned frame %d/%d: wet=%d, cumulative=%d",
                frame + 1,
                reader.frame_count,
                wet_count,
                frame_offsets[-1],
            )

    return ScanResult(
        frame_counts=tuple(frame_counts),
        frame_offsets=tuple(frame_offsets),
        coordinate_min=tuple(float(value) for value in coordinate_min),
        coordinate_max=tuple(float(value) for value in coordinate_max),
        target_min={
            name: tuple(float(value) for value in values)
            for name, values in target_min.items()
        },
        target_max={
            name: tuple(float(value) for value in values)
            for name, values in target_max.items()
        },
    )


def normalize_minmax(
    values: np.ndarray,
    low: np.ndarray | tuple[float, ...] | float,
    high: np.ndarray | tuple[float, ...] | float,
) -> np.ndarray:
    """Normalize columns to [-1, 1], mapping constant ranges to zero."""

    values32 = np.asarray(values, dtype=np.float32)
    low32 = np.asarray(low, dtype=np.float32)
    high32 = np.asarray(high, dtype=np.float32)
    width = high32 - low32
    constant = width == 0
    safe_width = np.where(constant, np.float32(1.0), width)
    normalized = (values32 - low32) * (np.float32(2.0) / safe_width) - np.float32(1.0)
    return np.where(constant, np.float32(0.0), normalized).astype(np.float32, copy=False)


def _normalization_manifest(
    low: tuple[float, ...] | float,
    high: tuple[float, ...] | float,
    *,
    method: str,
) -> dict[str, Any]:
    low_array = np.atleast_1d(np.asarray(low, dtype=np.float64))
    high_array = np.atleast_1d(np.asarray(high, dtype=np.float64))
    constant = low_array == high_array
    return {
        "method": method,
        "source_min": low_array.tolist(),
        "source_max": high_array.tolist(),
        "constant_components": constant.tolist(),
        "inverse": (
            "raw = (normalized + 1) * (source_max - source_min) / 2 + source_min; "
            "constant components decode to source_min"
        ),
    }


def _open_output_arrays(staging: Path, sample_count: int, frame_count: int) -> dict[str, np.memmap]:
    arrays = {
        COORDINATE_FILE: np.lib.format.open_memmap(
            staging / COORDINATE_FILE,
            mode="w+",
            dtype=np.float32,
            shape=(sample_count, 4),
        ),
        "wet_node_indices.npy": np.lib.format.open_memmap(
            staging / "wet_node_indices.npy",
            mode="w+",
            dtype=np.int32,
            shape=(sample_count,),
        ),
        "frame_offsets.npy": np.lib.format.open_memmap(
            staging / "frame_offsets.npy",
            mode="w+",
            dtype=np.int64,
            shape=(frame_count + 1,),
        ),
    }
    for name, components in TARGET_COMPONENTS.items():
        filename = f"target_{name}.npy"
        arrays[filename] = np.lib.format.open_memmap(
            staging / filename,
            mode="w+",
            dtype=np.float32,
            shape=(sample_count, components),
        )
    return arrays


def _close_memmap(array: np.ndarray, *, flush: bool = False) -> None:
    if flush and hasattr(array, "flush"):
        array.flush()
    mmap = getattr(array, "_mmap", None)
    if mmap is not None:
        mmap.close()


def _close_output_arrays(arrays: Mapping[str, np.ndarray], *, flush: bool = False) -> None:
    for array in arrays.values():
        _close_memmap(array, flush=flush)


def _publish_staging(staging: Path, output: Path, overwrite: bool) -> None:
    if output.exists() and not overwrite:
        raise FileExistsError(f"Output directory already exists: {output}; use --overwrite")
    backup = output.parent / f".{output.name}.backup-{uuid.uuid4().hex}"
    moved_old = False
    try:
        if output.exists():
            output.replace(backup)
            moved_old = True
        staging.replace(output)
    except Exception:
        if moved_old and not output.exists() and backup.exists():
            backup.replace(output)
        raise
    if backup.exists():
        shutil.rmtree(backup)


def _output_specs(scan: ScanResult, frame_count: int) -> dict[str, tuple[tuple[int, ...], np.dtype]]:
    specs = {
        COORDINATE_FILE: ((scan.sample_count, 4), np.dtype(np.float32)),
        "wet_node_indices.npy": ((scan.sample_count,), np.dtype(np.int32)),
        "frame_offsets.npy": ((frame_count + 1,), np.dtype(np.int64)),
    }
    specs.update(
        {
            f"target_{name}.npy": ((scan.sample_count, components), np.dtype(np.float32))
            for name, components in TARGET_COMPONENTS.items()
        }
    )
    return specs


def validate_export(staging: Path, reader: KatrinaReader, scan: ScanResult) -> None:
    for filename, (shape, dtype) in _output_specs(scan, reader.frame_count).items():
        array = np.load(staging / filename, mmap_mode="r", allow_pickle=False)
        try:
            if array.shape != shape or array.dtype != dtype:
                raise RuntimeError(
                    f"Output validation failed for {filename}: {array.shape}/{array.dtype}, "
                    f"expected {shape}/{dtype}"
                )
            if np.issubdtype(dtype, np.floating):
                for start in range(0, array.shape[0], 1_000_000):
                    block = np.array(array[start : start + 1_000_000], copy=True)
                    if not np.isfinite(block).all():
                        raise RuntimeError(f"Output {filename} contains NaN or Inf")
                    if block.size and (
                        float(block.min()) < -1.00001 or float(block.max()) > 1.00001
                    ):
                        raise RuntimeError(f"Output {filename} is outside [-1, 1]")
                    del block
        finally:
            _close_memmap(array)

    offsets = np.load(staging / "frame_offsets.npy", mmap_mode="r", allow_pickle=False)
    indices = np.load(staging / "wet_node_indices.npy", mmap_mode="r", allow_pickle=False)
    try:
        if not np.array_equal(offsets, np.asarray(scan.frame_offsets, dtype=np.int64)):
            raise RuntimeError("frame_offsets.npy does not match the scanned wet counts")
        if int(offsets[-1]) != scan.sample_count:
            raise RuntimeError("The final frame offset does not match the exported sample count")
        for frame in range(reader.frame_count):
            start, stop = int(offsets[frame]), int(offsets[frame + 1])
            block = np.array(indices[start:stop], copy=True)
            if block.size != scan.frame_counts[frame]:
                raise RuntimeError(f"Frame {frame} wet-node index count is incorrect")
            if block[0] < 0 or block[-1] >= reader.node_count or np.any(np.diff(block) <= 0):
                raise RuntimeError(f"Frame {frame} wet-node indices are invalid or unordered")
            del block
    finally:
        _close_memmap(indices)
        _close_memmap(offsets)


def _manifest(reader: KatrinaReader, scan: ScanResult, dry_sentinel: float) -> dict[str, Any]:
    coordinate_normalization = _normalization_manifest(
        scan.coordinate_min,
        scan.coordinate_max,
        method="per_column_minmax_to_minus_one_one",
    )
    coordinate_normalization["columns"] = ["X", "Y", "Z", "T"]

    targets: dict[str, Any] = {}
    for name, components in TARGET_COMPONENTS.items():
        if name == "v":
            low = min(scan.target_min[name])
            high = max(scan.target_max[name])
            normalization = _normalization_manifest(
                low, high, method="joint_component_minmax_to_minus_one_one"
            )
        else:
            normalization = _normalization_manifest(
                scan.target_min[name][0],
                scan.target_max[name][0],
                method="minmax_to_minus_one_one",
            )
        targets[name] = {
            "file": f"target_{name}.npy",
            "shape": [scan.sample_count, components],
            "dtype": "float32",
            "normalization": normalization,
        }

    return {
        "format_version": 1,
        "dataset_name": "Katrina_Wet",
        "source": reader.source_summary(),
        "wet_selection": {
            "mask_source": "target_fort63.npy",
            "definition": "isfinite(fort63) and fort63 != dry_sentinel",
            "dry_sentinel": float(dry_sentinel),
            "policy": "dynamic_per_frame",
            "sample_count": scan.sample_count,
            "frame_wet_counts": list(scan.frame_counts),
            "minimum_frame_wet_count": min(scan.frame_counts),
            "maximum_frame_wet_count": max(scan.frame_counts),
        },
        "layout": {
            "row_order": "time ascending, then original node index ascending",
            "frame_offsets_file": "frame_offsets.npy",
            "wet_node_indices_file": "wet_node_indices.npy",
            "frame_offsets_shape": [reader.frame_count + 1],
            "wet_node_indices_shape": [scan.sample_count],
        },
        "coordinates": {
            "file": COORDINATE_FILE,
            "shape": [scan.sample_count, 4],
            "dtype": "float32",
            "normalization": coordinate_normalization,
        },
        "targets": targets,
    }


def inspect_dataset(input_dir: str | Path, *, dry_sentinel: float = DRY_SENTINEL) -> dict[str, Any]:
    reader = KatrinaReader(input_dir)
    scan = scan_wet_samples(reader, dry_sentinel=dry_sentinel)
    payload = _manifest(reader, scan, dry_sentinel)
    print(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False))
    return payload


def export_dataset(
    input_dir: str | Path,
    output: str | Path,
    *,
    dry_sentinel: float = DRY_SENTINEL,
    overwrite: bool = False,
) -> dict[str, Any]:
    reader = KatrinaReader(input_dir)
    scan = scan_wet_samples(reader, dry_sentinel=dry_sentinel)
    output_path = Path(output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Output directory already exists: {output_path}; use --overwrite")
    staging = output_path.parent / f".{output_path.name}.staging-{uuid.uuid4().hex}"
    staging.mkdir(parents=False, exist_ok=False)
    arrays: dict[str, np.memmap] = {}

    try:
        arrays = _open_output_arrays(staging, scan.sample_count, reader.frame_count)
        arrays["frame_offsets.npy"][:] = np.asarray(scan.frame_offsets, dtype=np.int64)
        coordinate_low = np.asarray(scan.coordinate_min, dtype=np.float64)
        coordinate_high = np.asarray(scan.coordinate_max, dtype=np.float64)

        for frame in range(reader.frame_count):
            frame_slice = reader.frame_slice(frame)
            fort63 = np.asarray(reader.targets["fort63"][frame_slice, 0])
            wet = np.logical_and(np.isfinite(fort63), fort63 != dry_sentinel)
            node_indices = np.flatnonzero(wet).astype(np.int32, copy=False)
            start, stop = scan.frame_offsets[frame : frame + 2]
            if node_indices.size != stop - start:
                raise RuntimeError(f"Wet mask changed between scan and export at frame {frame}")

            arrays["wet_node_indices.npy"][start:stop] = node_indices
            arrays[COORDINATE_FILE][start:stop] = normalize_minmax(
                reader.coordinates[frame_slice][wet], coordinate_low, coordinate_high
            )
            for name in TARGET_COMPONENTS:
                selected = reader.targets[name][frame_slice][wet]
                if name == "v":
                    low = min(scan.target_min[name])
                    high = max(scan.target_max[name])
                else:
                    low = scan.target_min[name][0]
                    high = scan.target_max[name][0]
                arrays[f"target_{name}.npy"][start:stop] = normalize_minmax(
                    selected, low, high
                )

            if frame == 0 or (frame + 1) % 20 == 0 or frame + 1 == reader.frame_count:
                LOGGER.info(
                    "Exported frame %d/%d: rows=%d/%d",
                    frame + 1,
                    reader.frame_count,
                    stop,
                    scan.sample_count,
                )

        _close_output_arrays(arrays, flush=True)
        arrays.clear()
        gc.collect()

        manifest = _manifest(reader, scan, dry_sentinel)
        (staging / "manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False),
            encoding="utf-8",
        )
        validate_export(staging, reader, scan)
        _publish_staging(staging, output_path, overwrite)
    except Exception:
        _close_output_arrays(arrays)
        arrays.clear()
        gc.collect()
        if staging.exists():
            shutil.rmtree(staging)
        raise

    result = {
        "output_dir": str(output_path),
        "manifest_path": str(output_path / "manifest.json"),
        "sample_count": scan.sample_count,
        "frame_count": reader.frame_count,
        "nodes_per_frame": reader.node_count,
        "minimum_frame_wet_count": min(scan.frame_counts),
        "maximum_frame_wet_count": max(scan.frame_counts),
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--log-level", choices=("DEBUG", "INFO", "WARNING", "ERROR"), default="INFO"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser("inspect", help="Inspect dynamic Katrina wet points")
    inspect_parser.add_argument("--input-dir", required=True)
    inspect_parser.add_argument("--dry-sentinel", type=float, default=DRY_SENTINEL)

    export_parser = subparsers.add_parser("export", help="Export normalized wet-point arrays")
    export_parser.add_argument("--input-dir", required=True)
    export_parser.add_argument("--output", required=True)
    export_parser.add_argument("--dry-sentinel", type=float, default=DRY_SENTINEL)
    export_parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    if args.command == "inspect":
        inspect_dataset(args.input_dir, dry_sentinel=args.dry_sentinel)
    else:
        export_dataset(
            args.input_dir,
            args.output,
            dry_sentinel=args.dry_sentinel,
            overwrite=args.overwrite,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
