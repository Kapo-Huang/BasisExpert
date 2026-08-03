"""Standalone RealPDEBench combustion inspection, rendering, and export tool."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import logging
import re
import shutil
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np


LOGGER = logging.getLogger("combustion")
FLOAT32_LE = np.dtype("<f4")
FLOAT64_LE = np.dtype("<f8")
BASE_COLUMNS = ("sim_id", "shape_t", "shape_h", "shape_w")
NUMERICAL_CHANNELS = (
    "Absolute_Pressure",
    "Chemistry_Heat_Release_Rate",
    "Mole_Fraction_of_CH4",
    "Mole_Fraction_of_CO",
    "Mole_Fraction_of_CO2",
    "Mole_Fraction_of_H2O",
    "Mole_Fraction_of_NH2",
    "Mole_Fraction_of_NH3",
    "Mole_Fraction_of_OH",
    "Pressure",
    "Temperature",
    "Velocity[i]",
    "Velocity[j]",
    "Velocity[k]",
    "Velocity_Magnitude",
)
VELOCITY_INDICES = (11, 12, 13)
EXPECTED_EXPORT_SHAPE = (2001, 128, 128, 15)


@dataclass(frozen=True)
class TrajectoryMeta:
    sim_id: str
    row_index: int
    shape: tuple[int, int, int]
    numerical_channels: int | None

    @property
    def observed_nbytes(self) -> int:
        return int(np.prod(self.shape, dtype=np.int64)) * FLOAT32_LE.itemsize

    @property
    def numerical_nbytes(self) -> int | None:
        if self.numerical_channels is None:
            return None
        return self.observed_nbytes * int(self.numerical_channels)

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["shape"] = list(self.shape)
        result["observed_nbytes"] = self.observed_nbytes
        result["numerical_nbytes"] = self.numerical_nbytes
        return result


class CombustionArrowReader:
    """Read complete real or numerical combustion trajectories from Arrow."""

    def __init__(self, dataset_dir: str | Path, *, dataset: Any | None = None) -> None:
        self.dataset_dir = Path(dataset_dir).expanduser().resolve()
        if dataset is None:
            if not self.dataset_dir.is_dir():
                raise FileNotFoundError(f"Dataset directory does not exist: {self.dataset_dir}")
            try:
                from datasets import load_from_disk
            except ImportError as exc:
                raise RuntimeError(
                    "The standalone combustion script requires Hugging Face datasets. "
                    "Run it with the compression conda environment."
                ) from exc
            dataset = load_from_disk(str(self.dataset_dir))

        columns = set(getattr(dataset, "column_names", ()))
        missing = sorted(set(BASE_COLUMNS).difference(columns))
        if missing:
            raise ValueError(f"Arrow dataset is missing required columns: {', '.join(missing)}")
        if "observed" not in columns and "numerical" not in columns:
            raise ValueError("Arrow dataset contains neither observed nor numerical payloads")

        self._dataset = dataset
        self._columns = columns
        meta_columns = list(BASE_COLUMNS)
        if "numerical_channels" in columns:
            meta_columns.append("numerical_channels")
        self._metadata_dataset = dataset.select_columns(meta_columns)
        self._payload_datasets = {
            name: dataset.select_columns([name])
            for name in ("observed", "numerical")
            if name in columns
        }
        coordinate_columns = [name for name in ("x", "y", "t") if name in columns]
        self._coordinate_dataset = (
            dataset.select_columns(coordinate_columns) if coordinate_columns else None
        )
        self._metadata: dict[str, TrajectoryMeta] = {}
        self._build_metadata()
        self.dataset_fingerprint = self._fingerprint()

    def _build_metadata(self) -> None:
        for row_index in range(len(self._metadata_dataset)):
            row = self._metadata_dataset[row_index]
            sim_id = str(row["sim_id"])
            if not sim_id or sim_id in self._metadata:
                raise ValueError(f"Invalid or duplicate sim_id at Arrow row {row_index}: {sim_id!r}")
            shape = tuple(int(row[name]) for name in ("shape_t", "shape_h", "shape_w"))
            if any(size <= 0 for size in shape):
                raise ValueError(f"Invalid trajectory shape for {sim_id!r}: {shape}")
            channels = row.get("numerical_channels")
            channels = None if channels is None else int(channels)
            self._metadata[sim_id] = TrajectoryMeta(sim_id, row_index, shape, channels)

    def _fingerprint(self) -> str:
        digest = hashlib.sha256()
        digest.update(str(self.dataset_dir).encode("utf-8"))
        backend = getattr(self._dataset, "_fingerprint", None)
        if backend:
            digest.update(str(backend).encode("utf-8"))
        if self.dataset_dir.is_dir():
            for path in sorted(self.dataset_dir.glob("*.arrow")):
                stat = path.stat()
                digest.update(path.name.encode("utf-8"))
                digest.update(str(stat.st_size).encode("ascii"))
                digest.update(str(stat.st_mtime_ns).encode("ascii"))
            for name in ("dataset_info.json", "state.json"):
                path = self.dataset_dir / name
                if path.is_file():
                    digest.update(path.read_bytes())
        return digest.hexdigest()

    @property
    def simulation_ids(self) -> tuple[str, ...]:
        return tuple(self._metadata)

    @property
    def has_observed(self) -> bool:
        return "observed" in self._payload_datasets

    @property
    def has_numerical(self) -> bool:
        return "numerical" in self._payload_datasets

    def trajectory_meta(self, sim_id: str) -> TrajectoryMeta:
        try:
            return self._metadata[str(sim_id)]
        except KeyError as exc:
            raise KeyError(f"Unknown combustion sim_id: {sim_id!r}") from exc

    def _payload(self, sim_id: str, field: str, shape: tuple[int, ...]) -> np.ndarray:
        meta = self.trajectory_meta(sim_id)
        if field not in self._payload_datasets:
            raise ValueError(f"Dataset has no {field!r} payload")
        binary = self._payload_datasets[field][meta.row_index][field]
        if binary is None:
            raise ValueError(f"Trajectory {sim_id!r} has no {field} payload")
        expected = int(np.prod(shape, dtype=np.int64)) * FLOAT32_LE.itemsize
        if len(binary) != expected:
            raise ValueError(
                f"Trajectory {sim_id!r} {field} byte-size mismatch: "
                f"expected {expected}, got {len(binary)}"
            )
        return np.frombuffer(binary, dtype=FLOAT32_LE).reshape(shape)

    def load_observed(self, sim_id: str) -> np.ndarray:
        meta = self.trajectory_meta(sim_id)
        return self._payload(sim_id, "observed", meta.shape)

    def load_numerical(self, sim_id: str) -> np.ndarray:
        meta = self.trajectory_meta(sim_id)
        if meta.numerical_channels is None:
            raise ValueError(f"Trajectory {sim_id!r} has no numerical channel count")
        return self._payload(
            sim_id,
            "numerical",
            (*meta.shape, int(meta.numerical_channels)),
        )

    def load_coordinate(self, sim_id: str, axis: str) -> np.ndarray | None:
        if axis not in {"x", "y", "t"}:
            raise ValueError(f"Unknown coordinate axis: {axis!r}")
        if self._coordinate_dataset is None or axis not in self._columns:
            return None
        meta = self.trajectory_meta(sim_id)
        binary = self._coordinate_dataset[meta.row_index][axis]
        if binary is None:
            return None
        expected_size = meta.shape[0] if axis == "t" else meta.shape[2 if axis == "x" else 1]
        expected_bytes = expected_size * FLOAT64_LE.itemsize
        if len(binary) != expected_bytes:
            raise ValueError(
                f"Trajectory {sim_id!r} coordinate {axis} byte-size mismatch: "
                f"expected {expected_bytes}, got {len(binary)}"
            )
        return np.frombuffer(binary, dtype=FLOAT64_LE)

    def metadata_summary(self) -> dict[str, Any]:
        return {
            "dataset_dir": str(self.dataset_dir),
            "dataset_fingerprint": self.dataset_fingerprint,
            "trajectory_count": len(self._metadata),
            "has_observed": self.has_observed,
            "has_numerical": self.has_numerical,
            "trajectories": [meta.to_dict() for meta in self._metadata.values()],
        }


def parse_frame_selection(selection: str | Sequence[int] | None, frame_count: int) -> tuple[int, ...]:
    if selection is None or (isinstance(selection, str) and selection.strip().lower() == "all"):
        return tuple(range(frame_count))
    if not isinstance(selection, str):
        result = tuple(int(value) for value in selection)
    else:
        result_list: list[int] = []
        for token in (part.strip() for part in selection.split(",")):
            if not token:
                continue
            if ":" not in token:
                result_list.append(int(token))
                continue
            parts = token.split(":")
            if len(parts) not in {2, 3}:
                raise ValueError(f"Invalid frame range: {token!r}")
            start = int(parts[0]) if parts[0] else 0
            stop = int(parts[1]) if parts[1] else frame_count - 1
            step = int(parts[2]) if len(parts) == 3 and parts[2] else 1
            if step <= 0:
                raise ValueError("Frame range step must be positive")
            result_list.extend(range(start, stop + 1, step))
        result = tuple(dict.fromkeys(result_list))
    if not result:
        raise ValueError("Frame selection is empty")
    invalid = [value for value in result if value < 0 or value >= frame_count]
    if invalid:
        raise IndexError(f"Frame indices outside [0, {frame_count - 1}]: {invalid}")
    return result


def _scan_array(values: np.ndarray, *, chunk_rows: int) -> dict[str, Any]:
    flat = values.reshape(-1, values.shape[-1]) if values.ndim == 4 else values.reshape(-1, 1)
    mins = np.full(flat.shape[1], np.inf, dtype=np.float64)
    maxs = np.full(flat.shape[1], -np.inf, dtype=np.float64)
    sums = np.zeros(flat.shape[1], dtype=np.float64)
    count = 0
    for start in range(0, flat.shape[0], chunk_rows):
        block = flat[start : start + chunk_rows]
        if not np.isfinite(block).all():
            bad = np.argwhere(~np.isfinite(block))[0]
            raise ValueError(
                f"Non-finite value at flattened row {start + int(bad[0])}, channel {int(bad[1])}"
            )
        mins = np.minimum(mins, block.min(axis=0))
        maxs = np.maximum(maxs, block.max(axis=0))
        sums += block.sum(axis=0, dtype=np.float64)
        count += block.shape[0]
    return {
        "element_rows": count,
        "min": mins.tolist(),
        "max": maxs.tolist(),
        "mean": (sums / count).tolist(),
        "nan_count": 0,
        "inf_count": 0,
    }


def run_inspect(args: argparse.Namespace) -> dict[str, Any]:
    reader = CombustionArrowReader(args.dataset_dir)
    payload = reader.metadata_summary()
    if args.scan_values:
        scans: dict[str, Any] = {}
        for sim_id in reader.simulation_ids:
            sim_scans: dict[str, Any] = {}
            if reader.has_observed and args.field in {"all", "observed"}:
                values = reader.load_observed(sim_id)
                sim_scans["observed"] = _scan_array(values, chunk_rows=args.chunk_rows)
                del values
            if reader.has_numerical and args.field in {"all", "numerical"}:
                values = reader.load_numerical(sim_id)
                stats = _scan_array(values, chunk_rows=args.chunk_rows)
                stats["channels"] = list(NUMERICAL_CHANNELS)
                sim_scans["numerical"] = stats
                del values
            scans[sim_id] = sim_scans
            gc.collect()
        payload["statistics"] = scans
    if args.output:
        output = Path(args.output).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        payload["summary_path"] = str(output)
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return payload


def _safe_sim_stem(sim_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", Path(sim_id).stem).strip("._") or "simulation"


def _default_render_output(sim_id: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return Path("runs") / "visualizations" / "combustion" / _safe_sim_stem(sim_id) / timestamp


def _prepare_render_output(output: Path, overwrite: bool) -> tuple[Path, Path, Path]:
    frames = output / "frames"
    video = output / "heatmap.mp4"
    manifest = output / "render_manifest.json"
    generated = list(frames.glob("frame_*.png")) if frames.is_dir() else []
    generated.extend(path for path in (video, manifest) if path.exists())
    if generated and not overwrite:
        raise FileExistsError(f"Render output is not empty: {output}; use --overwrite")
    if overwrite:
        for path in generated:
            path.unlink()
    frames.mkdir(parents=True, exist_ok=True)
    return frames, video, manifest


def _render_one(
    reader: CombustionArrowReader,
    sim_id: str,
    output: Path,
    *,
    frame_selection: str,
    vmin: float,
    vmax: float,
    scale: str,
    cmap: str,
    sampling_fps: float,
    video_fps: float,
    overwrite: bool,
) -> dict[str, Any]:
    try:
        import cv2
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "Rendering requires matplotlib and opencv-python in the active environment"
        ) from exc

    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        raise ValueError(f"Expected finite vmin < vmax, got {vmin}, {vmax}")
    if sampling_fps <= 0 or video_fps <= 0:
        raise ValueError("sampling-fps and video-fps must be positive")

    meta = reader.trajectory_meta(sim_id)
    selected = parse_frame_selection(frame_selection, meta.shape[0])
    frames_dir, video_path, manifest_path = _prepare_render_output(output, overwrite)
    temporary_video = output / "heatmap.tmp.mp4"
    if temporary_video.exists():
        temporary_video.unlink()

    trajectory = reader.load_observed(sim_id)
    fig, axis = plt.subplots(figsize=(8, 8), dpi=100, constrained_layout=True)
    image = axis.imshow(
        trajectory[selected[0]], origin="upper", interpolation="nearest", aspect="equal",
        cmap=cmap, vmin=vmin, vmax=vmax,
    )
    axis.set_xlabel("pixel x")
    axis.set_ylabel("pixel y")
    colorbar = fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    colorbar.set_label("OH* chemiluminescence intensity (a.u.)")

    writer = None
    image_size = None
    failed_frame = None
    completed = False
    try:
        for ordinal, frame_id in enumerate(selected, start=1):
            failed_frame = frame_id
            image.set_data(trajectory[frame_id])
            time_ms = frame_id / sampling_fps * 1000.0
            axis.set_title(
                f"{sim_id} | frame {frame_id:06d}/{meta.shape[0] - 1:06d} | "
                f"t={time_ms:.3f} ms @ {sampling_fps:g} fps"
            )
            frame_path = frames_dir / f"frame_{frame_id:06d}.png"
            fig.savefig(frame_path, format="png", dpi=100)
            rgba = np.asarray(fig.canvas.buffer_rgba())
            height, width = rgba.shape[:2]
            if writer is None:
                image_size = (width, height)
                writer = cv2.VideoWriter(
                    str(temporary_video), cv2.VideoWriter_fourcc(*"mp4v"), video_fps, image_size
                )
                if not writer.isOpened():
                    raise RuntimeError(f"OpenCV could not open MP4 writer: {temporary_video}")
            writer.write(cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR))
            if ordinal == 1 or ordinal % 100 == 0 or ordinal == len(selected):
                LOGGER.info("Rendered %s: %d/%d", sim_id, ordinal, len(selected))
        completed = True
    except Exception as exc:
        raise RuntimeError(f"Failed rendering {sim_id!r} at frame {failed_frame}: {exc}") from exc
    finally:
        if writer is not None:
            writer.release()
        plt.close(fig)
        del trajectory
        if not completed and temporary_video.exists():
            temporary_video.unlink()

    if image_size is None or not temporary_video.is_file() or temporary_video.stat().st_size == 0:
        if temporary_video.exists():
            temporary_video.unlink()
        raise RuntimeError(f"Rendering produced no MP4 for {sim_id!r}")
    temporary_video.replace(video_path)
    manifest = {
        "dataset_dir": str(reader.dataset_dir),
        "dataset_fingerprint": reader.dataset_fingerprint,
        "sim_id": sim_id,
        "trajectory_shape": list(meta.shape),
        "selected_frames": list(selected),
        "vmin": float(vmin),
        "vmax": float(vmax),
        "scale": scale,
        "cmap": cmap,
        "sampling_fps": sampling_fps,
        "video_fps": video_fps,
        "image_size": list(image_size),
        "frames_dir": str(frames_dir),
        "video_path": str(video_path),
        "frame_count": len(selected),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest


def run_render(args: argparse.Namespace) -> dict[str, Any]:
    reader = CombustionArrowReader(args.dataset_dir)
    if not reader.has_observed:
        raise ValueError("The selected Arrow dataset has no observed field to render")
    sim_ids = reader.simulation_ids if args.sim_id.lower() == "all" else tuple(
        dict.fromkeys(part.strip() for part in args.sim_id.split(",") if part.strip())
    )
    for sim_id in sim_ids:
        reader.trajectory_meta(sim_id)
    if not sim_ids:
        raise ValueError("--sim-id selection is empty")

    if (args.vmin is None) != (args.vmax is None):
        raise ValueError("--vmin and --vmax must be supplied together")
    if args.vmin is not None:
        vmin, vmax, scale = float(args.vmin), float(args.vmax), "manual"
    else:
        if args.scale == "manual":
            raise ValueError("--scale manual requires --vmin and --vmax")
        scan_ids = reader.simulation_ids if args.scale == "global-minmax" else sim_ids
        vmin, vmax = np.inf, -np.inf
        for sim_id in scan_ids:
            values = reader.load_observed(sim_id)
            if not np.isfinite(values).all():
                raise ValueError(f"Observed trajectory {sim_id!r} contains NaN or Inf")
            vmin = min(vmin, float(values.min()))
            vmax = max(vmax, float(values.max()))
            del values
            gc.collect()
        scale = args.scale

    output_root = Path(args.output).expanduser().resolve() if args.output else (
        _default_render_output(sim_ids[0]).resolve()
        if len(sim_ids) == 1
        else (_default_render_output("all")).resolve()
    )
    manifests = []
    for sim_id in sim_ids:
        output = output_root if len(sim_ids) == 1 else output_root / _safe_sim_stem(sim_id)
        manifests.append(
            _render_one(
                reader, sim_id, output, frame_selection=args.frames, vmin=float(vmin),
                vmax=float(vmax), scale=scale, cmap=args.cmap,
                sampling_fps=args.sampling_fps, video_fps=args.video_fps,
                overwrite=args.overwrite,
            )
        )
    result = {"output_root": str(output_root), "vmin": vmin, "vmax": vmax, "renders": manifests}
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return result


def _coordinate_summary(values: np.ndarray | None) -> dict[str, Any] | None:
    if values is None:
        return None
    if not np.isfinite(values).all():
        raise ValueError("Coordinate array contains NaN or Inf")
    diffs = np.diff(values)
    return {
        "size": int(values.size),
        "dtype": str(values.dtype),
        "min": float(values.min()),
        "max": float(values.max()),
        "first": float(values[0]),
        "last": float(values[-1]),
        "uniform": bool(values.size <= 2 or np.allclose(diffs, diffs[0])),
        "step": None if values.size <= 1 else float(diffs[0]),
    }


def _normalization(values: np.ndarray, low: float, high: float) -> np.ndarray:
    if high == low:
        return np.zeros(values.shape, dtype=np.float32)
    scale = np.float32(2.0 / (high - low))
    return ((values.astype(np.float32, copy=False) - np.float32(low)) * scale - np.float32(1.0)).astype(
        np.float32, copy=False
    )


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


def run_export_volume(args: argparse.Namespace) -> dict[str, Any]:
    reader = CombustionArrowReader(args.dataset_dir)
    meta = reader.trajectory_meta(args.sim_id)
    if not reader.has_numerical:
        raise ValueError("The selected Arrow dataset has no numerical payload")
    actual_shape = (*meta.shape, int(meta.numerical_channels or 0))
    if actual_shape != EXPECTED_EXPORT_SHAPE:
        raise ValueError(
            f"Expected {args.sim_id!r} numerical shape {EXPECTED_EXPORT_SHAPE}, got {actual_shape}"
        )

    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"Output directory already exists: {output}; use --overwrite")
    staging = output.parent / f".{output.name}.staging-{uuid.uuid4().hex}"
    staging.mkdir(parents=False, exist_ok=False)

    values: np.ndarray | None = None
    memmaps: dict[str, np.memmap] = {}
    try:
        values = reader.load_numerical(args.sim_id)
        flat = values.reshape(-1, len(NUMERICAL_CHANNELS))
        stats = _scan_array(values, chunk_rows=args.chunk_rows)
        mins = np.asarray(stats["min"], dtype=np.float64)
        maxs = np.asarray(stats["max"], dtype=np.float64)
        velocity_min = float(mins[list(VELOCITY_INDICES)].min())
        velocity_max = float(maxs[list(VELOCITY_INDICES)].max())

        outputs: list[tuple[str, tuple[int, ...], tuple[int, ...], float, float]] = []
        for index, name in enumerate(NUMERICAL_CHANNELS):
            if index in VELOCITY_INDICES:
                continue
            outputs.append((name, (flat.shape[0], 1), (index,), float(mins[index]), float(maxs[index])))
        outputs.insert(
            -1,
            ("Velocity", (flat.shape[0], 3), VELOCITY_INDICES, velocity_min, velocity_max),
        )

        target_manifest: dict[str, Any] = {}
        for name, shape, indices, low, high in outputs:
            filename = f"target_{name}.npy"
            memmaps[name] = np.lib.format.open_memmap(
                staging / filename, mode="w+", dtype=np.float32, shape=shape
            )
            target_manifest[name] = {
                "file": filename,
                "shape": list(shape),
                "dtype": "float32",
                "source_channel_indices": list(indices),
                "source_channel_names": [NUMERICAL_CHANNELS[index] for index in indices],
                "normalization": {
                    "method": "joint_minmax_to_minus_one_one" if name == "Velocity" else "minmax_to_minus_one_one",
                    "source_min": low,
                    "source_max": high,
                    "inverse": "raw = (normalized + 1) * (source_max - source_min) / 2 + source_min",
                },
            }

        for start in range(0, flat.shape[0], args.chunk_rows):
            stop = min(start + args.chunk_rows, flat.shape[0])
            block = flat[start:stop]
            for name, _, indices, low, high in outputs:
                selected = block[:, list(indices)]
                memmaps[name][start:stop] = _normalization(selected, low, high)
            if start == 0 or stop == flat.shape[0] or stop % (args.chunk_rows * 10) == 0:
                LOGGER.info("Exported %d/%d rows", stop, flat.shape[0])

        for array in memmaps.values():
            array.flush()
        if memmaps:
            del array
        memmaps.clear()
        gc.collect()

        coordinate_summary = {
            axis: _coordinate_summary(reader.load_coordinate(args.sim_id, axis))
            for axis in ("x", "y", "t")
        }
        manifest = {
            "format_version": 1,
            "source": {
                "dataset_dir": str(reader.dataset_dir),
                "dataset_fingerprint": reader.dataset_fingerprint,
                "sim_id": args.sim_id,
                "shape": list(actual_shape),
                "dtype": "little-endian float32",
                "channel_names": list(NUMERICAL_CHANNELS),
                "channel_min": stats["min"],
                "channel_max": stats["max"],
                "nan_count": 0,
                "inf_count": 0,
            },
            "structured_volume": {
                "volume_shape": {"X": 128, "Y": 128, "Z": 1, "T": 2001},
                "coordinate_axes": ["x", "y", "t"],
                "source_array_order": ["t", "y", "x", "channel"],
                "flatten_order": "C",
                "flat_row_formula": "((t * Y) + y) * X + x",
                "sample_count": int(flat.shape[0]),
                "coordinates": coordinate_summary,
            },
            "targets": target_manifest,
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8"
        )

        for name, details in target_manifest.items():
            array = np.load(staging / details["file"], mmap_mode="r")
            if list(array.shape) != details["shape"] or array.dtype != np.float32:
                raise RuntimeError(f"Output validation failed for {name}")
            del array
        _publish_staging(staging, output, args.overwrite)
    except Exception:
        memmaps.clear()
        values = None
        gc.collect()
        if staging.exists():
            shutil.rmtree(staging)
        raise

    result = {
        "output_dir": str(output),
        "manifest_path": str(output / "manifest.json"),
        "target_count": 13,
        "sample_count": EXPECTED_EXPORT_SHAPE[0] * EXPECTED_EXPORT_SHAPE[1] * EXPECTED_EXPORT_SHAPE[2],
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser("inspect", help="Inspect real or numerical Arrow data")
    inspect_parser.add_argument("--dataset-dir", required=True)
    inspect_parser.add_argument("--scan-values", action="store_true")
    inspect_parser.add_argument("--field", choices=("all", "observed", "numerical"), default="all")
    inspect_parser.add_argument("--chunk-rows", type=int, default=250_000)
    inspect_parser.add_argument("--output", default=None)

    render_parser = subparsers.add_parser("render", help="Render observed heatmaps to PNG and MP4")
    render_parser.add_argument("--dataset-dir", required=True)
    render_parser.add_argument("--sim-id", required=True)
    render_parser.add_argument("--frames", default="all")
    render_parser.add_argument(
        "--scale", choices=("global-minmax", "trajectory-minmax", "manual"), default="global-minmax"
    )
    render_parser.add_argument("--vmin", type=float, default=None)
    render_parser.add_argument("--vmax", type=float, default=None)
    render_parser.add_argument("--cmap", default="inferno")
    render_parser.add_argument("--sampling-fps", type=float, default=4000.0)
    render_parser.add_argument("--video-fps", type=float, default=30.0)
    render_parser.add_argument("--output", default=None)
    render_parser.add_argument("--overwrite", action="store_true")

    export_parser = subparsers.add_parser("export-volume", help="Export 40NH3_1 numerical fields")
    export_parser.add_argument("--dataset-dir", required=True)
    export_parser.add_argument("--sim-id", default="40NH3_1.h5")
    export_parser.add_argument("--output", default=str(Path("data") / "Volume" / "Combustion"))
    export_parser.add_argument("--chunk-rows", type=int, default=250_000)
    export_parser.add_argument("--overwrite", action="store_true")

    args = parser.parse_args(argv)
    if hasattr(args, "chunk_rows") and args.chunk_rows <= 0:
        parser.error("--chunk-rows must be positive")
    return args


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    args = parse_args()
    if args.command == "inspect":
        run_inspect(args)
    elif args.command == "render":
        run_render(args)
    else:
        run_export_volume(args)


if __name__ == "__main__":
    main()
