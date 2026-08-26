from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from var_expert_inr.config.schema import VolumeShape
from var_expert_inr.evaluation.ground_truth import portable_data_path
from var_expert_inr.evaluation.rendering import (
    _load_mesh,
    _mesh_scalar_values,
    compare_rendered_images,
    load_render_profile,
    preflight_rendering,
    render_image_frame,
    renderer_name,
)
from var_expert_inr.evaluation.service import _InferenceOnlyDataset


def test_combustion_image2d_renders_scalar_and_vector(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    profile = {
        "kind": "volume",
        "renderer": "image2d",
        "cmap": "viridis",
        "clim": [-1.0, 1.0],
        "target_clims": {"Velocity": [0.0, np.sqrt(3.0)]},
        "window_size": [96, 80],
    }
    scalar = np.linspace(-1.0, 1.0, 20, dtype=np.float32).reshape(1, 4, 5)
    scalar_path = tmp_path / "scalar.png"
    scalar_info = render_image_frame(
        scalar,
        scalar_path,
        profile=profile,
        gt_values=scalar,
        target="Temperature",
    )
    assert scalar_path.is_file()
    assert scalar_info["shape"] == [4, 5]
    assert scalar_info["clim"] == [-1.0, 1.0]

    vector = np.ones((1, 4, 5, 3), dtype=np.float32)
    vector_path = tmp_path / "vector.png"
    vector_info = render_image_frame(
        vector,
        vector_path,
        profile=profile,
        gt_values=vector,
        target="Velocity",
    )
    assert vector_path.is_file()
    assert vector_info["shape"] == [4, 5]
    assert vector_info["clim"] == [0.0, np.sqrt(3.0)]


def test_mesh_coordinate_slice_maps_through_point_mask() -> None:
    pv = pytest.importorskip("pyvista")
    mesh = pv.PolyData(
        np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]],
            dtype=np.float32,
        )
    )
    mesh.point_data["wet"] = np.array([1, 0, 1, 0], dtype=np.uint8)
    coordinates = np.array(
        [
            [-1.0, -1.0, -1.0, 0.0],
            [1.0, -1.0, -1.0, 0.0],
            [-1.0, -1.0, 1.0, 0.0],
            [1.0, -1.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    values = np.array([[0.25], [0.75], [-0.25], [-0.75]], dtype=np.float32)
    profile = {
        "coordinate_slice": {"axis": "z", "index": 0},
        "mesh_mask_array": "wet",
    }
    scalar, selected, masked = _mesh_scalar_values(
        mesh,
        values,
        profile=profile,
        coordinates=coordinates,
        association="point",
    )
    np.testing.assert_allclose(scalar[[0, 2]], [0.25, 0.75])
    assert np.isnan(scalar[[1, 3]]).all()
    assert selected == 2
    assert masked == 2

    mesh.point_data["wet"] = np.array([1, 0, 0, 0], dtype=np.uint8)
    with pytest.raises(ValueError, match="Mesh mask/value size mismatch"):
        _mesh_scalar_values(
            mesh,
            values,
            profile=profile,
            coordinates=coordinates,
            association="point",
        )


def test_missing_mesh_reports_expected_path_and_copy_source(tmp_path: Path) -> None:
    pytest.importorskip("pyvista")
    missing = tmp_path / "surface_0000.vtp"
    source = "original/redsea/surface_0000.vtp"
    with pytest.raises(FileNotFoundError) as error:
        _load_mesh(
            {"mesh_path": str(missing), "mesh_source_hint": source},
            time_index=0,
        )
    assert str(missing) in str(error.value)
    assert source in str(error.value)


def test_profiles_cover_all_datasets_and_legacy_alias() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    expected = {
        "ionization": "volume",
        "combustion_40NH3_1": "image2d",
        "redsea": "mesh",
        "bathymetry": "mesh",
        "katrina": "mesh",
    }
    for dataset_name, renderer in expected.items():
        profile = load_render_profile(dataset_name, None, repo_root=repo_root)
        assert renderer_name(profile, dataset_kind=profile["kind"]) == renderer
    bathymetry = load_render_profile("bathymetry", None, repo_root=repo_root)
    assert Path(bathymetry["_path"]).name == "redsea.yaml"


def test_legacy_bathymetry_data_path_uses_redsea_folder(tmp_path: Path) -> None:
    repo_root = tmp_path / "project"
    target = repo_root / "data" / "Mesh" / "RedSea" / "target_TEMP.npy"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"test")
    resolved = portable_data_path(
        "/old/server/data/Mesh/Bathymetry/target_TEMP.npy",
        dataset_name="bathymetry",
        repo_root=repo_root,
    )
    assert resolved == target


def test_image_metrics_do_not_require_volume_vis(tmp_path: Path) -> None:
    pytest.importorskip("PIL")
    pytest.importorskip("skimage")
    from PIL import Image

    pixels = np.zeros((16, 16, 3), dtype=np.uint8)
    first = tmp_path / "first.png"
    second = tmp_path / "second.png"
    Image.fromarray(pixels).save(first)
    Image.fromarray(pixels).save(second)
    result = compare_rendered_images(first, second, ("ssim",), device="cpu")
    assert result["ssim"] == pytest.approx(1.0)


def test_combustion_preflight_accepts_singleton_z() -> None:
    pytest.importorskip("matplotlib")
    preflight_rendering(
        {"kind": "volume", "renderer": "image2d", "clim": [-1.0, 1.0]},
        dataset_kind="volume",
        targets=("Temperature",),
        timesteps=(0,),
        frame_sizes={0: 128 * 128},
        prediction_only=True,
        metrics=(),
        spatial_shape=(1, 128, 128),
    )


def test_inference_dataset_uses_coordinate_axes_and_real_target_dims(tmp_path: Path) -> None:
    shape = VolumeShape(X=2, Y=2, Z=1, T=2)
    scalar_path = tmp_path / "scalar.npy"
    vector_path = tmp_path / "vector.npy"
    np.save(scalar_path, np.zeros((shape.N, 1), dtype=np.float32))
    np.save(vector_path, np.zeros((shape.N, 3), dtype=np.float32))
    config = SimpleNamespace(
        data=SimpleNamespace(
            volume_shape=shape,
            coordinate_axes=("x", "y", "t"),
            kind="volume",
            coords_path=None,
            targets={"scalar": str(scalar_path), "vector": str(vector_path)},
            target_path=None,
            target=None,
        ),
        model=SimpleNamespace(params={}),
    )
    dataset = _InferenceOnlyDataset(config, ("scalar", "vector"))
    assert dataset.meta.input_dim == 3
    assert dataset.meta.target_dims == {"scalar": 1, "vector": 3}
    batch = dataset.fetch_batch([0, shape.N - 1], include_targets=False)
    assert tuple(batch.coords.shape) == (2, 3)
