import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "katrina_wet.py"
SPEC = importlib.util.spec_from_file_location("standalone_katrina_wet", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
katrina_wet = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = katrina_wet
SPEC.loader.exec_module(katrina_wet)


class KatrinaWetScriptTestCase(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tmpdir.name)
        self.input_dir = self.root / "input"
        self.input_dir.mkdir()
        self.output_dir = self.root / "Katrina_Wet"

    def tearDown(self):
        self.tmpdir.cleanup()

    def _write_inputs(self, *, nonfinite_speed: bool = False) -> dict[str, np.ndarray]:
        xyz = np.array(
            [[0.0, 10.0, 5.0], [1.0, 20.0, 5.0], [2.0, 30.0, 5.0], [3.0, 40.0, 5.0]],
            dtype=np.float64,
        )
        coordinates = np.concatenate(
            [np.column_stack((xyz, np.full(4, time))) for time in (10.0, 20.0, 30.0)]
        )
        fort63 = np.array(
            [
                1.0, -99999.0, 3.0, 5.0,
                -99999.0, 2.0, 4.0, -99999.0,
                0.0, 1.0, -99999.0, 3.0,
            ],
            dtype=np.float64,
        ).reshape(-1, 1)
        row_ids = np.arange(12, dtype=np.float64).reshape(-1, 1)
        arrays = {
            "source_XYZT.npy": coordinates,
            "target_fort63.npy": fort63,
            "target_fort64.npy": row_ids + 100.0,
            "target_fort73.npy": np.full((12, 1), 7.0, dtype=np.float64),
            "target_speed.npy": row_ids - 6.0,
            "target_v.npy": np.column_stack((row_ids[:, 0], -row_ids[:, 0], np.zeros(12))),
        }
        if nonfinite_speed:
            arrays["target_speed.npy"][2, 0] = np.nan
        for filename, values in arrays.items():
            np.save(self.input_dir / filename, values)
        return arrays

    def test_export_filters_dynamic_wet_points_and_normalizes_outputs(self):
        raw = self._write_inputs()
        result = katrina_wet.export_dataset(self.input_dir, self.output_dir)

        self.assertEqual(result["sample_count"], 8)
        np.testing.assert_array_equal(
            np.load(self.output_dir / "frame_offsets.npy"), [0, 3, 5, 8]
        )
        expected_indices = np.array([0, 2, 3, 1, 2, 0, 1, 3], dtype=np.int32)
        np.testing.assert_array_equal(
            np.load(self.output_dir / "wet_node_indices.npy"), expected_indices
        )

        wet_rows = np.array([0, 2, 3, 5, 6, 8, 9, 11])
        expected_coords = katrina_wet.normalize_minmax(
            raw["source_XYZT.npy"][wet_rows],
            raw["source_XYZT.npy"][wet_rows].min(axis=0),
            raw["source_XYZT.npy"][wet_rows].max(axis=0),
        )
        np.testing.assert_allclose(
            np.load(self.output_dir / "source_XYZT.npy"), expected_coords
        )
        self.assertTrue(np.all(np.load(self.output_dir / "source_XYZT.npy")[:, 2] == 0.0))

        expected_fort64_raw = raw["target_fort64.npy"][wet_rows]
        expected_fort64 = katrina_wet.normalize_minmax(
            expected_fort64_raw, expected_fort64_raw.min(), expected_fort64_raw.max()
        )
        np.testing.assert_allclose(
            np.load(self.output_dir / "target_fort64.npy"), expected_fort64
        )
        self.assertTrue(np.all(np.load(self.output_dir / "target_fort73.npy") == 0.0))

        expected_v_raw = raw["target_v.npy"][wet_rows]
        expected_v = katrina_wet.normalize_minmax(
            expected_v_raw, expected_v_raw.min(), expected_v_raw.max()
        )
        np.testing.assert_allclose(np.load(self.output_dir / "target_v.npy"), expected_v)

        manifest = json.loads((self.output_dir / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["dataset_name"], "Katrina_Wet")
        self.assertEqual(manifest["wet_selection"]["frame_wet_counts"], [3, 2, 3])
        self.assertEqual(
            manifest["coordinates"]["normalization"]["constant_components"],
            [False, False, True, False],
        )
        self.assertEqual(
            manifest["targets"]["v"]["normalization"]["method"],
            "joint_component_minmax_to_minus_one_one",
        )

    def test_nonfinite_wet_target_fails_without_publishing_output(self):
        self._write_inputs(nonfinite_speed=True)
        with self.assertRaisesRegex(ValueError, "non-finite wet values"):
            katrina_wet.export_dataset(self.input_dir, self.output_dir)
        self.assertFalse(self.output_dir.exists())

    def test_rejects_misaligned_target_shape(self):
        self._write_inputs()
        np.save(self.input_dir / "target_fort64.npy", np.zeros((11, 1), dtype=np.float64))
        with self.assertRaisesRegex(ValueError, "must have shape"):
            katrina_wet.KatrinaReader(self.input_dir)


if __name__ == "__main__":
    unittest.main()
