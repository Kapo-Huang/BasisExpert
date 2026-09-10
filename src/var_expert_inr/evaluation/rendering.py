from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .metrics import validate_error_bounds


TARGET_PRESET_ALIASES = {"h_plus": "H+", "h+": "H+"}
DATASET_PROFILE_ALIASES = {"bathymetry": "redsea"}
COORDINATE_AXES = {"x": 0, "y": 1, "z": 2}
_LPIPS_MODELS: dict[tuple[str, str], Any] = {}
_MESH_CACHE: dict[tuple[Any, ...], tuple[Any, Path]] = {}
LPIPS_MAX_BATCH_SIZE = 8
LPIPS_MAX_BATCH_PIXELS = 4 * 1024 * 1024



@lru_cache(maxsize=1)
def _white_to_red_colormap():
    try:
        from matplotlib.colors import LinearSegmentedColormap
    except ImportError:
        return "Reds"
    return LinearSegmentedColormap.from_list(
        "evaluation_error_white_to_red",
        ("#ffffff", "#ff0000"),
    )


def _resolved_colormap(name: str):
    normalized = str(name).strip().lower()
    if normalized in {"error_white_to_red", "white_to_red"}:
        return _white_to_red_colormap()
    return name

def _finalize_node_render_image(
    output: Path,
    *,
    profile: dict[str, Any],
    original_size: tuple[int, int],
) -> dict[str, Any]:
    """Optionally crop a mesh render to its non-background bounding box."""
    base_info: dict[str, Any] = {
        "crop_bbox": None,
        "original_size": [int(original_size[0]), int(original_size[1])],
        "output_size": [int(original_size[0]), int(original_size[1])],
    }
    if not bool(profile.get("crop_to_nonwhite_bbox", False)):
        return base_info

    try:
        from PIL import Image, ImageColor
    except ImportError as exc:
        raise RuntimeError(
            "Cropping rendered images requires Pillow; install .[evaluation]"
        ) from exc

    background = str(profile.get("background", "white"))
    background_rgb = np.asarray(ImageColor.getrgb(background), dtype=np.uint8).reshape(-1)[:3]
    with Image.open(output) as image:
        rgb = image.convert("RGB")
        pixels = np.asarray(rgb)
        content = np.any(pixels != background_rgb, axis=-1)
        rows, columns = np.nonzero(content)
        if rows.size == 0:
            raise ValueError(
                f"Cannot crop all-background node render {output}: background={background!r}"
            )
        left = int(columns.min())
        top = int(rows.min())
        right = int(columns.max()) + 1
        bottom = int(rows.max()) + 1
        cropped = rgb.crop((left, top, right, bottom))
        actual_original_size = [int(rgb.width), int(rgb.height)]
        output_size = [int(cropped.width), int(cropped.height)]

    cropped.save(output, format="PNG")
    return {
        "crop_bbox": [left, top, right, bottom],
        "original_size": actual_original_size,
        "output_size": output_size,
    }



def error_transfer_function() -> dict[str, list[dict[str, Any]]]:
    """VolumeVis transfer function for a pre-normalized error volume."""
    return {
        "colorNodes": [
            {"r": 255, "g": 255, "b": 255, "cx": 7, "color": "#ffffff"},
            {"r": 255, "g": 0, "b": 0, "cx": 505, "color": "#ff0000"},
        ],
        "opacityNodes": [
            {"opacity": 0.0, "cx": 7, "cy": 143},
            {"opacity": 1.0, "cx": 505, "cy": 0},
        ],
    }


def normalize_error_for_volume(
    error_values: np.ndarray,
    *,
    error_vmin: float,
    error_vmax: float,
) -> np.ndarray:
    lo, hi = validate_error_bounds(error_vmin, error_vmax)
    clipped = np.clip(np.asarray(error_values), lo, hi)
    normalized_zero_one = (clipped - lo) / (hi - lo)
    return np.asarray(normalized_zero_one * 2.0 - 1.0, dtype=np.float32)


def _error_render_profile(
    profile: dict[str, Any],
    *,
    error_vmin: float,
    error_vmax: float,
) -> dict[str, Any]:
    lo, hi = validate_error_bounds(error_vmin, error_vmax)
    return {
        **profile,
        "cmap": "error_white_to_red",
        "clim": [lo, hi],
        "crop_to_nonwhite_bbox": False,
        "target_clims": {},
        "clip_to_clim": True,
        "values_are_scalar": True,
    }


def load_render_profile(
    dataset_name: str | None,
    profile: str | Path | None,
    *,
    repo_root: Path,
) -> dict[str, Any]:
    if profile and str(profile).strip().lower() != "auto":
        path = Path(profile).expanduser().resolve()
    else:
        name = str(dataset_name or "").strip().lower()
        name = DATASET_PROFILE_ALIASES.get(name, name)
        path = Path(__file__).resolve().parent / "profiles" / f"{name}.yaml"
    if not path.is_file():
        raise FileNotFoundError(
            f"Render profile does not exist for dataset {dataset_name!r}: {path}. "
            "Provide --eval-config or evaluation.render_profile."
        )
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Render profile must be a mapping: {path}")
    for key in (
        "mesh_path", "mesh_path_template", "vertices_path", "vertices_path_template",
        "cells_path", "cells_path_template",
    ):
        if payload.get(key):
            candidate = Path(str(payload[key]))
            if not candidate.is_absolute():
                payload[key] = str((repo_root / candidate).resolve())
    payload["_path"] = str(path)
    return payload


def renderer_name(profile: dict[str, Any], *, dataset_kind: str) -> str:
    default = "volume" if str(dataset_kind).lower() == "volume" else "mesh"
    renderer = str(profile.get("renderer", default)).strip().lower()
    aliases = {"image": "image2d", "node": "mesh"}
    renderer = aliases.get(renderer, renderer)
    if renderer not in {"volume", "image2d", "mesh"}:
        raise ValueError("render profile renderer must be volume, image2d, or mesh")
    return renderer


def visual_scalar(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim in {2, 4} and array.shape[-1] in {2, 3}:
        return np.linalg.norm(array, axis=-1)
    if array.ndim in {2, 4} and array.shape[-1] == 1:
        return array[..., 0]
    return array


def resolve_clim(profile: dict[str, Any], gt: np.ndarray | None, *, target: str | None = None) -> tuple[float, float]:
    target_clims = {str(k).lower(): v for k, v in (profile.get("target_clims") or {}).items()}
    configured = target_clims.get(str(target).lower()) if target is not None else None
    if configured is None:
        configured = profile.get("clim")
    if configured is not None:
        if not isinstance(configured, (list, tuple)) or len(configured) != 2:
            raise ValueError("render profile clim must contain [minimum, maximum]")
        lo, hi = float(configured[0]), float(configured[1])
    elif gt is not None:
        finite = np.asarray(gt, dtype=np.float32).reshape(-1)
        finite = finite[np.isfinite(finite)]
        if finite.size == 0:
            raise ValueError("Ground Truth contains no finite values for render color limits")
        lo, hi = float(finite.min()), float(finite.max())
    else:
        raise ValueError(
            "Prediction-only rendering requires a fixed 'clim: [min, max]' in the render profile"
        )
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        raise ValueError(f"invalid render color limits: {(lo, hi)}")
    return lo, hi


def _preset_name(target: str, profile: dict[str, Any]) -> str:
    mappings = {str(k).lower(): str(v) for k, v in (profile.get("target_presets") or {}).items()}
    lowered = str(target).lower()
    return mappings.get(lowered, TARGET_PRESET_ALIASES.get(lowered, str(target)))


@dataclass
class VolumeRenderSession:
    profile: dict[str, Any]

    def __enter__(self) -> "VolumeRenderSession":
        try:
            from volume_vis import RenderOptions, VolumeRenderer
        except ImportError as exc:
            raise RuntimeError(
                "Volume rendering requires the sibling VolumeVis package. Install it with "
                "pip install -e <path-to-Vis>[lpips]."
            ) from exc
        options_payload = dict(self.profile.get("options") or {})
        self._renderer = VolumeRenderer(RenderOptions(**options_payload))
        self._renderer.open()
        return self

    def __exit__(self, *_: object) -> None:
        self._renderer.close()

    def render(self, values: np.ndarray, output: Path, *, target: str) -> dict[str, Any]:
        from volume_vis import load_preset

        namespace = str(self.profile.get("preset_namespace", "ionization"))
        preset = load_preset(_preset_name(target, self.profile), namespace=namespace)
        result = self._renderer.render(
            np.asarray(visual_scalar(values)),
            output,
            transfer_function=preset.transfer_function,
            viewport=preset.viewport,
            layout=str(self.profile.get("layout", "zyx")),
        )
        return {
            "path": str(result.output_path),
            "gpu_mode": result.gpu_mode_used,
            "source_min": result.source_min,
            "source_max": result.source_max,
            "clipped_voxel_count": result.clipped_voxel_count,
            "clipped_ratio": result.clipped_ratio,
        }

    def render_error(
        self,
        values: np.ndarray,
        output: Path,
        *,
        target: str,
        error_vmin: float,
        error_vmax: float,
    ) -> dict[str, Any]:
        from volume_vis import load_preset

        lo, hi = validate_error_bounds(error_vmin, error_vmax)
        scalar = np.asarray(visual_scalar(values))
        below_count = int(np.count_nonzero(scalar < lo))
        above_count = int(np.count_nonzero(scalar > hi))
        namespace = str(self.profile.get("preset_namespace", "ionization"))
        preset = load_preset(_preset_name(target, self.profile), namespace=namespace)
        result = self._renderer.render(
            normalize_error_for_volume(
                scalar,
                error_vmin=lo,
                error_vmax=hi,
            ),
            output,
            transfer_function=error_transfer_function(),
            viewport=preset.viewport,
            layout=str(self.profile.get("layout", "zyx")),
        )
        return {
            "path": str(result.output_path),
            "renderer": "volume_error",
            "gpu_mode": result.gpu_mode_used,
            "error_clim": [lo, hi],
            "below_error_vmin_count": below_count,
            "above_error_vmax_count": above_count,
        }


def compare_rendered_images(
    gt_path: Path,
    pred_path: Path,
    metrics: tuple[str, ...],
    *,
    device: str = "auto",
) -> dict[str, float | None]:
    requested = tuple(name for name in metrics if name in {"ssim", "lpips"})
    if not requested:
        return {}
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Rendered-image metrics require Pillow; install .[evaluation]") from exc

    def load_rgb(path: Path) -> np.ndarray:
        with Image.open(path) as image:
            return np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0

    gt = load_rgb(gt_path)
    pred = load_rgb(pred_path)
    if pred.shape != gt.shape:
        resampling = getattr(Image, "Resampling", Image)
        resized = Image.fromarray(np.clip(pred * 255.0, 0, 255).astype(np.uint8)).resize(
            (gt.shape[1], gt.shape[0]), resampling.BILINEAR
        )
        pred = np.asarray(resized, dtype=np.float32) / 255.0

    result: dict[str, float | None] = {}
    if "ssim" in requested:
        from skimage.metrics import structural_similarity

        result["ssim"] = float(
            structural_similarity(gt, pred, data_range=1.0, channel_axis=2)
        )
    if "lpips" in requested:
        import lpips
        import torch

        resolved_device = device
        if device == "auto":
            resolved_device = "cuda" if torch.cuda.is_available() else "cpu"
        key = ("alex", resolved_device)
        model = _LPIPS_MODELS.get(key)
        if model is None:
            model = lpips.LPIPS(net="alex", verbose=False).to(resolved_device)
            _LPIPS_MODELS[key] = model
        gt_tensor = torch.from_numpy(gt.transpose(2, 0, 1)).float().unsqueeze(0).to(resolved_device) * 2 - 1
        pred_tensor = torch.from_numpy(pred.transpose(2, 0, 1)).float().unsqueeze(0).to(resolved_device) * 2 - 1
        with torch.no_grad():
            result["lpips"] = float(model(gt_tensor, pred_tensor).item())
    return result


def compare_rendered_image_pairs(
    pairs: Sequence[tuple[Path, Path]],
    metrics: tuple[str, ...],
    *,
    device: str = "auto",
) -> list[dict[str, float | None]]:
    """Compare rendered pairs while batching LPIPS work by image shape."""
    requested = tuple(name for name in metrics if name in {"ssim", "lpips"})
    if not requested:
        return [{} for _ in pairs]
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Rendered-image metrics require Pillow; install .[evaluation]") from exc

    def load_rgb(path: Path) -> np.ndarray:
        with Image.open(path) as image:
            return np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0

    structural_similarity = None
    if "ssim" in requested:
        from skimage.metrics import structural_similarity

    results: list[dict[str, float | None]] = [{} for _ in pairs]
    lpips_batches: dict[tuple[int, int], list[tuple[int, np.ndarray, np.ndarray]]] = {}
    for index, (gt_path, pred_path) in enumerate(pairs):
        gt = load_rgb(gt_path)
        pred = load_rgb(pred_path)
        if pred.shape != gt.shape:
            resampling = getattr(Image, "Resampling", Image)
            resized = Image.fromarray(np.clip(pred * 255.0, 0, 255).astype(np.uint8)).resize(
                (gt.shape[1], gt.shape[0]), resampling.BILINEAR
            )
            pred = np.asarray(resized, dtype=np.float32) / 255.0
        if structural_similarity is not None:
            results[index]["ssim"] = float(
                structural_similarity(gt, pred, data_range=1.0, channel_axis=2)
            )
        if "lpips" in requested:
            shape = (int(gt.shape[0]), int(gt.shape[1]))
            batch = lpips_batches.setdefault(shape, [])
            batch.append((index, gt, pred))
            pixel_limit = max(1, LPIPS_MAX_BATCH_PIXELS // max(shape[0] * shape[1], 1))
            if len(batch) >= min(LPIPS_MAX_BATCH_SIZE, pixel_limit):
                _evaluate_lpips_batch(batch, results, device=device)
                batch.clear()

    for batch in lpips_batches.values():
        if batch:
            _evaluate_lpips_batch(batch, results, device=device)
    return results


def _evaluate_lpips_batch(
    batch: list[tuple[int, np.ndarray, np.ndarray]],
    results: list[dict[str, float | None]],
    *,
    device: str,
) -> None:
    import lpips
    import torch

    resolved_device = device
    if device == "auto":
        resolved_device = "cuda" if torch.cuda.is_available() else "cpu"
    key = ("alex", resolved_device)
    model = _LPIPS_MODELS.get(key)
    if model is None:
        model = lpips.LPIPS(net="alex", verbose=False).to(resolved_device).eval()
        _LPIPS_MODELS[key] = model
    gt_array = np.stack([item[1].transpose(2, 0, 1) for item in batch])
    pred_array = np.stack([item[2].transpose(2, 0, 1) for item in batch])
    gt_tensor = torch.from_numpy(gt_array).to(resolved_device) * 2 - 1
    pred_tensor = torch.from_numpy(pred_array).to(resolved_device) * 2 - 1
    with torch.inference_mode():
        values = model(gt_tensor, pred_tensor).reshape(-1).detach().cpu().numpy()
    for (index, _, _), value in zip(batch, values):
        results[index]["lpips"] = float(value)


def _coordinate_slice_mask(coordinates: np.ndarray, profile: dict[str, Any]) -> np.ndarray:
    coords = np.asarray(coordinates)
    if coords.ndim != 2:
        raise ValueError(f"render coordinates must have shape (N, D), got {coords.shape}")
    selection = profile.get("coordinate_slice")
    if selection is None:
        return np.ones((coords.shape[0],), dtype=bool)
    if not isinstance(selection, dict):
        raise ValueError("coordinate_slice must be a mapping with axis and index")
    raw_axis = selection.get("axis")
    if isinstance(raw_axis, str):
        axis_name = raw_axis.strip().lower()
        if axis_name not in COORDINATE_AXES:
            raise ValueError("coordinate_slice.axis must be x, y, z, or an integer")
        axis = COORDINATE_AXES[axis_name]
    else:
        axis = int(raw_axis)
    if axis < 0 or axis >= coords.shape[1] - 1:
        raise ValueError(
            f"coordinate_slice axis {axis} is outside the spatial coordinate columns {coords.shape[1] - 1}"
        )
    levels = np.unique(coords[:, axis])
    index = int(selection.get("index", 0))
    if index < 0:
        index += int(levels.size)
    if index < 0 or index >= int(levels.size):
        raise ValueError(
            f"coordinate_slice index {selection.get('index', 0)} is outside {levels.size} available levels"
        )
    return coords[:, axis] == levels[index]


def _mesh_scalar_values(
    mesh,
    values: np.ndarray,
    *,
    profile: dict[str, Any],
    coordinates: np.ndarray | None,
    association: str,
) -> tuple[np.ndarray, int, int | None]:
    scalar_values = values if bool(profile.get("values_are_scalar", False)) else visual_scalar(values)
    scalar = np.asarray(scalar_values).reshape(-1)
    if coordinates is not None and int(np.asarray(coordinates).shape[0]) != int(scalar.size):
        raise ValueError(
            f"Coordinate/value size mismatch: coordinates={np.asarray(coordinates).shape[0]}, values={scalar.size}"
        )
    if profile.get("coordinate_slice") is not None and coordinates is not None:
        selection = _coordinate_slice_mask(coordinates, profile)
        scalar = scalar[selection]
    elif profile.get("coordinate_slice") is not None:
        raise ValueError("coordinate_slice rendering requires frame coordinates")

    observed = int(mesh.n_points if association == "point" else mesh.n_cells)
    mask_name = profile.get("mesh_mask_array")
    if not mask_name:
        if observed != int(scalar.size):
            raise ValueError(
                f"{association.title()} mesh size mismatch: mesh={observed}, values={scalar.size}"
            )
        return scalar, int(scalar.size), None

    data = mesh.point_data if association == "point" else mesh.cell_data
    if str(mask_name) not in data:
        raise KeyError(
            f"Mesh {association} data does not contain mask array {mask_name!r}; "
            f"available={list(data.keys())}"
        )
    mask = np.asarray(data[str(mask_name)]).reshape(-1).astype(bool)
    if int(mask.size) != observed:
        raise ValueError(
            f"Mesh mask size mismatch: mask={mask.size}, mesh_{association}s={observed}"
        )
    selected = int(np.count_nonzero(mask))
    if selected != int(scalar.size):
        raise ValueError(
            f"Mesh mask/value size mismatch: mask={selected}, values={scalar.size}"
        )
    expanded = np.full((observed,), np.nan, dtype=np.result_type(scalar.dtype, np.float32))
    expanded[mask] = scalar
    return expanded, int(scalar.size), selected


def render_image_frame(
    values: np.ndarray,
    output: Path,
    *,
    profile: dict[str, Any],
    gt_values: np.ndarray | None,
    target: str | None = None,
) -> dict[str, Any]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("2D rendering requires matplotlib; install .[evaluation]") from exc

    scalar_values = values if bool(profile.get("values_are_scalar", False)) else visual_scalar(values)
    scalar = np.squeeze(np.asarray(scalar_values))
    if scalar.ndim != 2:
        raise ValueError(f"image2d renderer requires a 2D scalar frame, got {scalar.shape}")
    gt_scalar = None if gt_values is None else np.squeeze(np.asarray(visual_scalar(gt_values)))
    clim = resolve_clim(profile, gt_scalar, target=target)
    if bool(profile.get("clip_to_clim", False)):
        scalar = np.clip(scalar, clim[0], clim[1])
    size = tuple(int(item) for item in profile.get("window_size", [1024, 1024]))
    if len(size) != 2 or min(size) <= 0:
        raise ValueError("image2d window_size must contain two positive integers")
    dpi = int(profile.get("dpi", 100))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(size[0] / dpi, size[1] / dpi), dpi=dpi)
    try:
        image = axis.imshow(
            scalar,
            origin=str(profile.get("origin", "lower")),
            interpolation=str(profile.get("interpolation", "nearest")),
            aspect=str(profile.get("aspect", "equal")),
            cmap=_resolved_colormap(str(profile.get("cmap", "viridis"))),
            vmin=clim[0],
            vmax=clim[1],
        )
        if bool(profile.get("show_scalar_bar", False)):
            fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
        if not bool(profile.get("show_axes", False)):
            axis.set_axis_off()
        fig.tight_layout(pad=0)
        fig.savefig(output, format="png", dpi=dpi)
    finally:
        plt.close(fig)
    return {
        "path": str(output.resolve()),
        "renderer": "image2d",
        "shape": list(scalar.shape),
        "cmap": str(profile.get("cmap", "viridis")),
        "clim": list(clim),
    }


def render_error_image_frame(
    values: np.ndarray,
    output: Path,
    *,
    profile: dict[str, Any],
    error_vmin: float,
    error_vmax: float,
) -> dict[str, Any]:
    info = render_image_frame(
        values,
        output,
        profile=_error_render_profile(
            profile,
            error_vmin=error_vmin,
            error_vmax=error_vmax,
        ),
        gt_values=None,
        target=None,
    )
    info["renderer"] = "image2d_error"
    info["cmap"] = "white_to_red"
    return info


def _read_fort14(path: Path):
    import pyvista as pv

    with path.open("r", encoding="utf-8", errors="replace") as handle:
        handle.readline()
        counts = handle.readline().split()
        if len(counts) < 2:
            raise ValueError(f"Invalid fort.14 header: {path}")
        element_count, node_count = int(counts[0]), int(counts[1])
        points = np.empty((node_count, 3), dtype=np.float64)
        for row in range(node_count):
            fields = handle.readline().split()
            if len(fields) < 4:
                raise ValueError(f"Invalid fort.14 node row {row + 1}: {path}")
            points[row] = (float(fields[1]), float(fields[2]), float(fields[3]))
        faces = np.empty((element_count, 4), dtype=np.int64)
        for row in range(element_count):
            fields = handle.readline().split()
            if len(fields) < 5 or int(fields[1]) != 3:
                raise ValueError("Only triangular fort.14 elements are supported")
            faces[row] = (3, int(fields[2]) - 1, int(fields[3]) - 1, int(fields[4]) - 1)
    return pv.PolyData(points, faces.reshape(-1))


def _load_mesh(profile: dict[str, Any], *, time_index: int):
    try:
        import pyvista as pv
    except ImportError as exc:
        raise RuntimeError("Node rendering requires pyvista and vtk; install .[evaluation]") from exc
    raw = profile.get("mesh_path") or profile.get("mesh_path_template")
    vertices_raw = profile.get("vertices_path") or profile.get("vertices_path_template")
    cells_raw = profile.get("cells_path") or profile.get("cells_path_template")
    if not raw and (not vertices_raw or not cells_raw):
        raise FileNotFoundError(
            "Node render profile must provide mesh_path/mesh_path_template or both vertices and cells arrays"
        )
    if not raw:
        format_values = {"time_index": int(time_index), "timestep": int(time_index), "t": int(time_index)}
        vertices_path = Path(str(vertices_raw).format(**format_values)).expanduser().resolve()
        cells_path = Path(str(cells_raw).format(**format_values)).expanduser().resolve()
        if not vertices_path.is_file() or not cells_path.is_file():
            raise FileNotFoundError(
                f"Node mesh arrays do not exist: vertices={vertices_path}, cells={cells_path}"
            )
        cache_key = (
            "arrays",
            str(vertices_path),
            str(cells_path),
            str(profile.get("cell_type", "")),
            bool(profile.get("planarize_z", False)),
        )
        cached = _MESH_CACHE.get(cache_key)
        if cached is not None:
            return cached
        points = np.asarray(np.load(vertices_path, allow_pickle=False), dtype=np.float64)
        cells = np.asarray(np.load(cells_path, allow_pickle=False), dtype=np.int64)
        if points.ndim != 2 or points.shape[1] not in {2, 3}:
            raise ValueError(f"vertices array must have shape (N, 2|3), got {points.shape}")
        if points.shape[1] == 2:
            points = np.column_stack([points, np.zeros((points.shape[0],), dtype=points.dtype)])
        if cells.ndim != 2 or cells.shape[1] not in {3, 4}:
            raise ValueError(f"cells array must have shape (M, 3|4), got {cells.shape}")
        if cells.size and (int(cells.min()) < 0 or int(cells.max()) >= int(points.shape[0])):
            raise ValueError("cells array contains an out-of-range vertex index")
        requested_type = str(profile.get("cell_type", "triangle" if cells.shape[1] == 3 else "tetra")).lower()
        cell_types = {"triangle": pv.CellType.TRIANGLE, "quad": pv.CellType.QUAD, "tetra": pv.CellType.TETRA}
        if requested_type not in cell_types:
            raise ValueError("cell_type must be triangle, quad, or tetra")
        expected_width = 4 if requested_type in {"quad", "tetra"} else 3
        if int(cells.shape[1]) != expected_width:
            raise ValueError(f"cell_type {requested_type!r} requires {expected_width} indices per cell")
        connectivity = np.column_stack(
            [np.full((cells.shape[0],), cells.shape[1], dtype=np.int64), cells]
        ).reshape(-1)
        types = np.full((cells.shape[0],), cell_types[requested_type], dtype=np.uint8)
        mesh = pv.UnstructuredGrid(connectivity, types, points)
        result = (_apply_mesh_render_transforms(mesh, profile), vertices_path)
        _MESH_CACHE[cache_key] = result
        return result
    path = Path(str(raw).format(time_index=int(time_index), timestep=int(time_index), t=int(time_index))).expanduser().resolve()
    if not path.is_file():
        source_hint = profile.get("mesh_source_hint")
        copy_hint = f" Copy it from: {source_hint}" if source_hint else ""
        raise FileNotFoundError(f"Node mesh does not exist: {path}.{copy_hint}")
    cache_key = (
        "mesh",
        str(path),
        bool(profile.get("planarize_z", False)),
    )
    cached = _MESH_CACHE.get(cache_key)
    if cached is not None:
        return cached

    if path.name.lower() == "fort.14":
        mesh = _read_fort14(path)
    else:
        mesh = pv.read(str(path))
    result = (_apply_mesh_render_transforms(mesh, profile), path)
    _MESH_CACHE[cache_key] = result
    return result


def _apply_mesh_render_transforms(mesh: Any, profile: dict[str, Any]) -> Any:
    """Apply display-only mesh transforms requested by an evaluation profile."""
    if not bool(profile.get("planarize_z", False)):
        return mesh

    points = np.asarray(mesh.points)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("planarize_z requires a mesh with Nx3 point coordinates")

    rendered_mesh = mesh.copy(deep=True)
    rendered_points = np.array(rendered_mesh.points, dtype=np.float64, copy=True)
    rendered_points[:, 2] = 0.0
    rendered_mesh.points = rendered_points
    return rendered_mesh


def preflight_rendering(
    profile: dict[str, Any],
    *,
    dataset_kind: str,
    targets: tuple[str, ...],
    timesteps: tuple[int, ...],
    frame_sizes: dict[int, int] | None,
    prediction_only: bool,
    metrics: tuple[str, ...],
    frame_coordinates: dict[int, np.ndarray] | None = None,
    spatial_shape: tuple[int, int, int] | None = None,
) -> None:
    """Validate renderer inputs and optional dependencies before model decoding."""
    if "ssim" in metrics:
        try:
            import skimage.metrics  # noqa: F401
        except ImportError as exc:
            raise RuntimeError("SSIM evaluation requires scikit-image; install .[evaluation]") from exc
    if "lpips" in metrics:
        try:
            import lpips  # noqa: F401
        except ImportError as exc:
            raise RuntimeError("LPIPS evaluation requires lpips; install .[evaluation]") from exc
    renderer = renderer_name(profile, dataset_kind=dataset_kind)
    if renderer == "volume":
        if str(dataset_kind).lower() != "volume":
            raise ValueError("volume renderer requires a volume dataset")
        try:
            from volume_vis import VolumeRenderer, load_preset  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "Volume rendering requires the sibling VolumeVis package; install it in editable mode"
            ) from exc
        namespace = str(profile.get("preset_namespace", "ionization"))
        for target in targets:
            load_preset(_preset_name(target, profile), namespace=namespace)
        return

    if renderer == "image2d":
        if str(dataset_kind).lower() != "volume":
            raise ValueError("image2d renderer requires a volume dataset")
        if spatial_shape is None or sum(int(item) > 1 for item in spatial_shape) != 2:
            raise ValueError(
                f"image2d renderer requires exactly two non-singleton spatial axes, got {spatial_shape}"
            )
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot  # noqa: F401
        except ImportError as exc:
            raise RuntimeError("2D rendering requires matplotlib; install .[evaluation]") from exc
        for target in targets:
            if prediction_only:
                resolve_clim(profile, None, target=target)
        return

    association = str(profile.get("association", "point")).lower()
    if association not in {"point", "cell"}:
        raise ValueError("node render association must be 'point' or 'cell'")
    for target in targets:
        if prediction_only:
            resolve_clim(profile, None, target=target)
    for timestep in timesteps:
        mesh, _ = _load_mesh(profile, time_index=timestep)
        if frame_sizes is None:
            continue
        expected = int(frame_sizes[timestep])
        coordinates = None if frame_coordinates is None else frame_coordinates.get(timestep)
        dummy = np.zeros((expected, 1), dtype=np.float32)
        try:
            _mesh_scalar_values(
                mesh,
                dummy,
                profile=profile,
                coordinates=coordinates,
                association=association,
            )
        except (KeyError, ValueError) as exc:
            raise ValueError(f"Node render preflight failed at timestep {timestep}: {exc}") from exc


def render_node_frame(
    values: np.ndarray,
    output: Path,
    *,
    profile: dict[str, Any],
    time_index: int,
    gt_values: np.ndarray | None,
    coordinates: np.ndarray | None = None,
    target: str | None = None,
) -> dict[str, Any]:
    import pyvista as pv

    mesh, mesh_path = _load_mesh(profile, time_index=time_index)
    association = str(profile.get("association", "point")).lower()
    scalar, selected_value_count, mask_value_count = _mesh_scalar_values(
        mesh,
        values,
        profile=profile,
        coordinates=coordinates,
        association=association,
    )
    if association not in {"point", "cell"}:
        raise ValueError("node render association must be 'point' or 'cell'")
    gt_scalar = None
    if gt_values is not None:
        gt_scalar, _, _ = _mesh_scalar_values(
            mesh,
            gt_values,
            profile=profile,
            coordinates=coordinates,
            association=association,
        )
    clim = resolve_clim(profile, gt_scalar, target=target)
    if bool(profile.get("clip_to_clim", False)):
        scalar = np.clip(scalar, clim[0], clim[1])
    if association == "point":
        mesh.point_data["evaluation_scalar"] = scalar
    else:
        mesh.cell_data["evaluation_scalar"] = scalar
    output.parent.mkdir(parents=True, exist_ok=True)
    size = tuple(int(item) for item in profile.get("window_size", [1800, 1400]))
    plotter = pv.Plotter(off_screen=True, window_size=size)
    try:
        plotter.set_background(str(profile.get("background", "white")))
        cmap = _resolved_colormap(str(profile.get("cmap", "viridis")))
        plotter.add_mesh(
            mesh,
            scalars="evaluation_scalar",
            cmap=cmap,
            clim=list(clim),
            show_edges=False,
            show_scalar_bar=bool(profile.get("show_scalar_bar", False)),
            nan_color="white",
        )
        camera = profile.get("camera_position")
        if camera:
            plotter.camera_position = camera
        else:
            plotter.reset_camera()
            plotter.camera.zoom(float(profile.get("zoom", 1.35)))
        plotter.render()
        plotter.screenshot(filename=str(output), return_img=False)
    finally:
        plotter.close()
    image_info = _finalize_node_render_image(
        output,
        profile=profile,
        original_size=size,
    )
    return {
        "path": str(output.resolve()),
        "renderer": "mesh",
        "mesh_path": str(mesh_path),
        "association": association,
        "clim": list(clim),
        "selected_value_count": selected_value_count,
        "mesh_mask_value_count": mask_value_count,
        **image_info,
    }


def render_error_node_frame(
    values: np.ndarray,
    output: Path,
    *,
    profile: dict[str, Any],
    time_index: int,
    coordinates: np.ndarray | None,
    error_vmin: float,
    error_vmax: float,
) -> dict[str, Any]:
    info = render_node_frame(
        values,
        output,
        profile=_error_render_profile(
            profile,
            error_vmin=error_vmin,
            error_vmax=error_vmax,
        ),
        time_index=time_index,
        gt_values=None,
        coordinates=coordinates,
        target=None,
    )
    info["renderer"] = "mesh_error"
    info["cmap"] = "white_to_red"
    return info


def profile_fingerprint(profile: dict[str, Any]) -> str:
    from hashlib import sha256

    payload = {key: value for key, value in profile.items() if key != "_path"}
    return sha256(json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()
