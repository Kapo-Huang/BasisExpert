import tempfile
import unittest
from pathlib import Path

import numpy as np

from var_expert_inr.evaluation.ground_truth import validate_ground_truth_paths


class EvaluationGroundTruthTestCase(unittest.TestCase):
    def test_missing_ground_truth_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            missing = Path(tmpdir) / "missing.npy"
            with self.assertRaisesRegex(FileNotFoundError, "Ground Truth"):
                validate_ground_truth_paths({"GT": missing}, ("GT",), volume_shape=(1, 1, 1, 2))

    def test_corrupt_ground_truth_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "bad.npy"
            path.write_text("not a numpy file", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "unreadable"):
                validate_ground_truth_paths({"GT": path}, ("GT",), volume_shape=(1, 1, 1, 2))

    def test_shape_mismatch_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "gt.npy"
            np.save(path, np.zeros((2, 2), dtype=np.float32))
            with self.assertRaisesRegex(ValueError, "shape mismatch"):
                validate_ground_truth_paths({"GT": path}, ("GT",), volume_shape=(2, 1, 1, 3))


if __name__ == "__main__":
    unittest.main()
