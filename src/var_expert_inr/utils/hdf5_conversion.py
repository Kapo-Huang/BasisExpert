from __future__ import annotations

import argparse
import os
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

try:
    import h5py
except ImportError:  # pragma: no cover - exercised only when dependency is missing.
    h5py = None


@dataclass
class DatasetConversionSpec:
    dataset_name: str
    dataset_kind: str
    default_input_dir: Path
    default_output_filename: str
    targets: dict[str, str]
    coords_file: str | None = None
    extra_root_attrs: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.default_input_dir = Path(self.default_input_dir)
        if self.dataset_kind not in {"node", "volume"}:
            raise ValueError(f"Unsupported dataset_kind: {self.dataset_kind}")
        if self.dataset_kind == "node" and self.coords_file is None:
            raise ValueError("Node dataset conversion requires coords_file.")
        if self.dataset_kind == "volume" and self.coords_file is not None:
            raise ValueError("Volume dataset conversion cannot declare coords_file.")
        if not self.targets:
            raise ValueError("At least one target file is required.")


@dataclass(frozen=True)
class SourceArraySpec:
    source_path: Path
    array: np.ndarray
    dataset_name: str
    group_name: str | None = None

    @property
    def hdf5_path(self) -> str:
        if self.group_name is None:
            return f"/{self.dataset_name}"
        return f"/{self.group_name}/{self.dataset_name}"


def build_arg_parser(spec: DatasetConversionSpec) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=f"Convert the {spec.dataset_name} dataset from .npy files to a single HDF5 file."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=spec.default_input_dir,
        help=f"Dataset directory containing the source .npy files (default: {spec.default_input_dir})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=spec.default_input_dir / spec.default_output_filename,
        help="Destination .h5 file path.",
    )
    parser.add_argument(
        "--chunk-rows",
        type=int,
        default=250_000,
        help="Number of rows copied per write chunk.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing an existing output .h5 file.",
    )
    return parser


def main_for_dataset(spec: DatasetConversionSpec, argv: list[str] | None = None) -> int:
    args = build_arg_parser(spec).parse_args(argv)
    try:
        dataset_paths = convert_dataset(
            spec,
            input_dir=args.input_dir,
            output_path=args.output,
            chunk_rows=args.chunk_rows,
            overwrite=args.overwrite,
        )
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(f"Finished conversion: {Path(args.output).resolve()}")
    print("HDF5 dataset keys:")
    for dataset_path in dataset_paths:
        print(f"  {dataset_path}")
    return 0


def convert_dataset(
    spec: DatasetConversionSpec,
    *,
    input_dir: str | Path,
    output_path: str | Path,
    chunk_rows: int,
    overwrite: bool,
) -> list[str]:
    h5py_module = _require_h5py()
    input_dir = Path(input_dir)
    output_path = Path(output_path)
    if chunk_rows <= 0:
        raise ValueError("chunk_rows must be a positive integer.")
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Refusing to overwrite existing HDF5 file: {output_path}. Pass --overwrite to replace it."
        )

    source_specs = _collect_source_specs(spec, input_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_output = _reserve_temp_output(output_path.parent, output_path.stem, output_path.suffix or ".h5")

    try:
        with h5py_module.File(temp_output, "w") as h5_file:
            _write_root_attributes(h5_file, spec)
            if any(item.group_name == "targets" for item in source_specs):
                h5_file.require_group("targets")
            dataset_paths: list[str] = []
            for source in source_specs:
                dataset_paths.append(_write_source_array(h5_file, source, chunk_rows))
        temp_output.replace(output_path)
    except Exception:
        temp_output.unlink(missing_ok=True)
        raise
    return dataset_paths


def _require_h5py():
    if h5py is None:
        raise RuntimeError("h5py is required to run HDF5 conversion scripts.")
    return h5py


def _collect_source_specs(spec: DatasetConversionSpec, input_dir: Path) -> list[SourceArraySpec]:
    if spec.dataset_kind == "node":
        return _collect_node_source_specs(spec, input_dir)
    return _collect_volume_source_specs(spec, input_dir)


def _collect_node_source_specs(spec: DatasetConversionSpec, input_dir: Path) -> list[SourceArraySpec]:
    coords_path = input_dir / str(spec.coords_file)
    coords = _load_source_array(coords_path)
    if coords.ndim != 2:
        raise ValueError(f"Node coords must be a 2D array, but {coords_path} has shape {coords.shape}.")
    if int(coords.shape[0]) == 0:
        raise ValueError(f"Node coords array is empty: {coords_path}")

    source_specs = [SourceArraySpec(source_path=coords_path, array=coords, dataset_name="coords")]
    expected_rows = int(coords.shape[0])
    for target_name, filename in spec.targets.items():
        target_path = input_dir / filename
        target = _load_source_array(target_path)
        if target.ndim < 1:
            raise ValueError(f"Target array must have at least one dimension: {target_path}")
        if int(target.shape[0]) != expected_rows:
            raise ValueError(
                f"Target sample count mismatch for {target_path}: {target.shape[0]} vs coords rows {expected_rows}."
            )
        source_specs.append(
            SourceArraySpec(
                source_path=target_path,
                array=target,
                dataset_name=target_name,
                group_name="targets",
            )
        )
    return source_specs


def _collect_volume_source_specs(spec: DatasetConversionSpec, input_dir: Path) -> list[SourceArraySpec]:
    source_specs: list[SourceArraySpec] = []
    expected_shape: tuple[int, ...] | None = None
    for target_name, filename in spec.targets.items():
        target_path = input_dir / filename
        target = _load_source_array(target_path)
        if target.ndim < 1:
            raise ValueError(f"Target array must have at least one dimension: {target_path}")
        shape = tuple(int(dim) for dim in target.shape)
        if expected_shape is None:
            expected_shape = shape
        elif shape != expected_shape:
            raise ValueError(f"Volume target shape mismatch for {target_path}: {shape} vs {expected_shape}.")
        source_specs.append(
            SourceArraySpec(
                source_path=target_path,
                array=target,
                dataset_name=target_name,
                group_name="targets",
            )
        )
    return source_specs


def _load_source_array(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Required source file is missing: {path}")
    try:
        return np.load(path, mmap_mode="r")
    except Exception as exc:  # pragma: no cover - depends on corrupt user input.
        raise ValueError(f"Failed to load {path} as a .npy array.") from exc


def _write_root_attributes(h5_file, spec: DatasetConversionSpec) -> None:
    h5_file.attrs["dataset_name"] = spec.dataset_name
    h5_file.attrs["dataset_kind"] = spec.dataset_kind
    h5_file.attrs["source_format"] = "npy"
    h5_file.attrs["stored_dtype"] = "float32"
    h5_file.attrs["compression"] = "lzf"
    for key, value in spec.extra_root_attrs.items():
        h5_file.attrs[key] = value


def _write_source_array(h5_file, source: SourceArraySpec, chunk_rows: int) -> str:
    print(
        "Converting "
        f"{source.source_path.name}: shape={source.array.shape}, "
        f"original_dtype={source.array.dtype}, stored_dtype=float32, hdf5_path={source.hdf5_path}"
    )
    target_shape = tuple(int(dim) for dim in source.array.shape)
    dataset_kwargs = {
        "shape": target_shape,
        "dtype": np.float32,
        "compression": "lzf",
        "chunks": _chunk_shape(target_shape, chunk_rows),
    }
    if source.group_name is None:
        dataset = h5_file.create_dataset(source.dataset_name, **dataset_kwargs)
    else:
        group = h5_file.require_group(source.group_name)
        dataset = group.create_dataset(source.dataset_name, **dataset_kwargs)

    dataset.attrs["source_file"] = source.source_path.name
    dataset.attrs["original_dtype"] = str(source.array.dtype)
    dataset.attrs["original_shape"] = np.asarray(target_shape, dtype=np.int64)

    total_rows = int(source.array.shape[0])
    for start in range(0, total_rows, chunk_rows):
        stop = min(start + chunk_rows, total_rows)
        dataset[start:stop, ...] = np.asarray(source.array[start:stop, ...], dtype=np.float32)
    return source.hdf5_path


def _chunk_shape(shape: tuple[int, ...], chunk_rows: int) -> tuple[int, ...]:
    if not shape:
        raise ValueError("Scalar arrays are not supported for HDF5 conversion.")
    if int(shape[0]) <= 0:
        raise ValueError(f"Array has no rows to write: shape={shape}")
    return (min(int(chunk_rows), int(shape[0])),) + tuple(int(dim) for dim in shape[1:])


def _reserve_temp_output(parent: Path, stem: str, suffix: str) -> Path:
    fd, temp_name = tempfile.mkstemp(prefix=f"{stem}.", suffix=f"{suffix}.tmp", dir=parent)
    os.close(fd)
    temp_path = Path(temp_name)
    temp_path.unlink(missing_ok=True)
    return temp_path
