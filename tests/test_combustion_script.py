import argparse
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "tools" / "combustion.py"
SPEC = importlib.util.spec_from_file_location("standalone_combustion", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
combustion = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = combustion
SPEC.loader.exec_module(combustion)


class FakeArrowDataset:
    def __init__(self, rows, columns=None):
        self.rows = [dict(row) for row in rows]
        self.column_names = list(columns or self.rows[0].keys())
        self._fingerprint = "synthetic-combustion"

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        if isinstance(index, str):
            return [row[index] for row in self.rows]
        return {name: self.rows[index][name] for name in self.column_names}

    def select_columns(self, columns):
        return FakeArrowDataset(self.rows, columns)


def numerical_row(values, sim_id="40NH3_1.h5"):
    t, h, w, channels = values.shape
    return {
        "sim_id": sim_id,
        "shape_t": t,
        "shape_h": h,
        "shape_w": w,
        "numerical_channels": channels,
        "numerical": values.astype("<f4").tobytes(),
        "observed": values[..., 0].astype("<f4").tobytes(),
        "x": np.linspace(-2.0, 2.0, w, dtype="<f8").tobytes(),
        "y": np.linspace(1.0, 3.0, h, dtype="<f8").tobytes(),
        "t": np.linspace(1.5, 2.0, t, dtype="<f8").tobytes(),
    }


class CombustionScriptTestCase(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tmpdir.name)

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_reader_decodes_numerical_and_rejects_bad_byte_count(self):
        values = np.arange(2 * 2 * 3 * 15, dtype=np.float32).reshape(2, 2, 3, 15)
        reader = combustion.CombustionArrowReader(self.root, dataset=FakeArrowDataset([numerical_row(values)]))
        np.testing.assert_array_equal(reader.load_numerical("40NH3_1.h5"), values)

        bad = numerical_row(values)
        bad["numerical"] = bad["numerical"][:-4]
        reader = combustion.CombustionArrowReader(self.root, dataset=FakeArrowDataset([bad]))
        with self.assertRaisesRegex(ValueError, "byte-size mismatch"):
            reader.load_numerical("40NH3_1.h5")

    def test_export_preserves_t_y_x_order_and_combines_velocity(self):
        values = np.arange(2 * 2 * 2 * 15, dtype=np.float32).reshape(2, 2, 2, 15)
        dataset = FakeArrowDataset([numerical_row(values)])
        output = self.root / "Combustion"
        args = argparse.Namespace(
            dataset_dir=str(self.root), sim_id="40NH3_1.h5", output=str(output),
            chunk_rows=3, overwrite=False,
        )
        with mock.patch.object(combustion, "EXPECTED_EXPORT_SHAPE", values.shape), mock.patch.object(
            combustion, "CombustionArrowReader", return_value=combustion.CombustionArrowReader(self.root, dataset=dataset)
        ):
            result = combustion.run_export_volume(args)

        self.assertEqual(result["target_count"], 13)
        pressure = np.load(output / "target_Absolute_Pressure.npy")
        raw_pressure = values[..., 0].reshape(-1, 1)
        expected_pressure = combustion._normalization(raw_pressure, raw_pressure.min(), raw_pressure.max())
        np.testing.assert_allclose(pressure, expected_pressure)

        velocity = np.load(output / "target_Velocity.npy")
        raw_velocity = values[..., 11:14].reshape(-1, 3)
        expected_velocity = combustion._normalization(raw_velocity, raw_velocity.min(), raw_velocity.max())
        np.testing.assert_allclose(velocity, expected_velocity)
        self.assertEqual(velocity.shape, (8, 3))

    def test_export_rejects_nonfinite_values_without_publishing_output(self):
        values = np.arange(2 * 2 * 2 * 15, dtype=np.float32).reshape(2, 2, 2, 15)
        values[1, 0, 1, 4] = np.nan
        dataset = FakeArrowDataset([numerical_row(values)])
        output = self.root / "Combustion"
        args = argparse.Namespace(
            dataset_dir=str(self.root), sim_id="40NH3_1.h5", output=str(output),
            chunk_rows=3, overwrite=False,
        )
        with mock.patch.object(combustion, "EXPECTED_EXPORT_SHAPE", values.shape), mock.patch.object(
            combustion, "CombustionArrowReader", return_value=combustion.CombustionArrowReader(self.root, dataset=dataset)
        ):
            with self.assertRaisesRegex(ValueError, "Non-finite"):
                combustion.run_export_volume(args)
        self.assertFalse(output.exists())

    def test_frame_selection_uses_inclusive_ranges(self):
        self.assertEqual(combustion.parse_frame_selection("0,2:5:2", 8), (0, 2, 4))
        with self.assertRaises(IndexError):
            combustion.parse_frame_selection("8", 8)


try:
    import cv2  # noqa: F401
    import matplotlib  # noqa: F401
except ImportError:
    RENDERING_AVAILABLE = False
else:
    RENDERING_AVAILABLE = True


@unittest.skipUnless(RENDERING_AVAILABLE, "rendering dependencies are not installed")
class CombustionRenderingScriptTestCase(unittest.TestCase):
    def test_render_writes_selected_pngs_mp4_and_manifest(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            values = np.linspace(-1.0, 1.0, 4 * 3 * 3, dtype=np.float32).reshape(4, 3, 3)
            row = {
                "sim_id": "tiny.h5", "shape_t": 4, "shape_h": 3, "shape_w": 3,
                "observed": values.astype("<f4").tobytes(),
            }
            reader = combustion.CombustionArrowReader(root, dataset=FakeArrowDataset([row]))
            manifest = combustion._render_one(
                reader, "tiny.h5", root / "render", frame_selection="0,2,3",
                vmin=-1.0, vmax=1.0, scale="manual", cmap="inferno",
                sampling_fps=4000.0, video_fps=30.0, overwrite=False,
            )
            self.assertEqual(manifest["selected_frames"], [0, 2, 3])
            self.assertEqual(manifest["frame_count"], 3)
            self.assertEqual(
                sorted(path.name for path in (root / "render" / "frames").glob("*.png")),
                ["frame_000000.png", "frame_000002.png", "frame_000003.png"],
            )
            capture = cv2.VideoCapture(str(root / "render" / "heatmap.mp4"))
            self.assertTrue(capture.isOpened())
            self.assertEqual(int(capture.get(cv2.CAP_PROP_FRAME_COUNT)), 3)
            self.assertAlmostEqual(capture.get(cv2.CAP_PROP_FPS), 30.0, places=1)
            capture.release()


if __name__ == "__main__":
    unittest.main()
