from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import yaml


TARGET_PRESET_ALIASES = {"h_plus": "H+", "h+": "H+"}


@lru_cache(maxsize=1)
def _yellow_biased_viridis():
    try:
        from matplotlib import cm
        from matplotlib.colors import ListedColormap
    except ImportError:
        return "viridis"
    samples = np.linspace(0.0, 1.0, 256, dtype=np.float64)
    remapped = 1.0 - np.power(1.0 - samples, 1.18)
    return ListedColormap(cm.get_cmap("viridis", 256)(remapped), name="viridis_yellow_biased")


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


def compare_rendered_images(
    gt_path: Path,
    pred_path: Path,
    metrics: tuple[str, ...],
    *,
    device: str = "auto",
) -> dict[str, float | None]:
    from volume_vis import compare_images

    requested = tuple(name for name in metrics if name in {"ssim", "lpips"})
    if not requested:
        return {}
    return compare_images(gt_path, pred_path, metrics=requested, device=device).as_dict()


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
        return pv.UnstructuredGrid(connectivity, types, points), vertices_path
    path = Path(str(raw).format(time_index=int(time_index), timestep=int(time_index), t=int(time_index))).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Node mesh does not exist: {path}")
    if path.name.lower() == "fort.14":
        return _read_fort14(path), path
    return pv.read(str(path)), path


def preflight_rendering(
    profile: dict[str, Any],
    *,
    dataset_kind: str,
    targets: tuple[str, ...],
    timesteps: tuple[int, ...],
    frame_sizes: dict[int, int] | None,
    prediction_only: bool,
    metrics: tuple[str, ...],
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
    if str(dataset_kind).lower() == "volume":
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
        observed = int(mesh.n_points if association == "point" else mesh.n_cells)
        if observed != expected:
            raise ValueError(
                f"Node mesh size mismatch at timestep {timestep}: "
                f"association={association}, mesh={observed}, values={expected}"
            )


def render_node_frame(
    values: np.ndarray,
    output: Path,
    *,
    profile: dict[str, Any],
    time_index: int,
    gt_values: np.ndarray | None,
    target: str | None = None,
) -> dict[str, Any]:
    import pyvista as pv

    mesh, mesh_path = _load_mesh(profile, time_index=time_index)
    association = str(profile.get("association", "point")).lower()
    scalar = np.asarray(visual_scalar(values)).reshape(-1)
    if association == "point":
        if int(mesh.n_points) != int(scalar.size):
            raise ValueError(f"Point mesh size mismatch: mesh={mesh.n_points}, values={scalar.size}")
        mesh.point_data["evaluation_scalar"] = scalar
    elif association == "cell":
        if int(mesh.n_cells) != int(scalar.size):
            raise ValueError(f"Cell mesh size mismatch: mesh={mesh.n_cells}, values={scalar.size}")
        mesh.cell_data["evaluation_scalar"] = scalar
    else:
        raise ValueError("node render association must be 'point' or 'cell'")
    gt_scalar = None if gt_values is None else visual_scalar(gt_values)
    clim = resolve_clim(profile, gt_scalar, target=target)
    output.parent.mkdir(parents=True, exist_ok=True)
    size = tuple(int(item) for item in profile.get("window_size", [1800, 1400]))
    plotter = pv.Plotter(off_screen=True, window_size=size)
    try:
        plotter.set_background(str(profile.get("background", "white")))
        cmap_name = str(profile.get("cmap", "yellow_biased_viridis")).lower()
        cmap = _yellow_biased_viridis() if cmap_name in {"yellow_biased_viridis", "viridis_yellow_biased"} else cmap_name
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
    return {"path": str(output.resolve()), "mesh_path": str(mesh_path), "association": association, "clim": list(clim)}


def profile_fingerprint(profile: dict[str, Any]) -> str:
    from hashlib import sha256

    payload = {key: value for key, value in profile.items() if key != "_path"}
    return sha256(json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()
