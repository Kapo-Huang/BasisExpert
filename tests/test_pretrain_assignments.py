import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

import var_expert_inr.pretrain.assignments as assignment_module
from var_expert_inr.config.schema import VolumeShape
from var_expert_inr.data.node import NodeFieldDataset
from var_expert_inr.data.volume import VolumeFieldDataset
from var_expert_inr.pretrain.assignments import PretrainAssignmentConfig, compute_pretrain_assignments


class PretrainAssignmentsTestCase(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tmpdir.name)

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_node_pretrain_is_rejected(self):
        coords = np.array(
            [[0, 0, 0, 0], [0, 0, 0, 1], [1, 1, 1, 0], [1, 1, 1, 1]],
            dtype=np.float32,
        )
        targets = np.array([[-0.9], [-0.8], [0.8], [0.9]], dtype=np.float32)
        coords_path = self.root / "coords.npy"
        target_path = self.root / "target.npy"
        np.save(coords_path, coords)
        np.save(target_path, targets)

        dataset = NodeFieldDataset(coords_path=str(coords_path), target_path=str(target_path))
        cfg = PretrainAssignmentConfig(seed=42)
        with self.assertRaisesRegex(ValueError, "volume dataset"):
            compute_pretrain_assignments(dataset, 2, cfg)

    def test_volume_voxel_clustering_uses_cache(self):
        volume = np.zeros((2, 1, 1, 4), dtype=np.float32)
        volume[0, 0, 0, :2] = 0.0
        volume[1, 0, 0, :2] = 0.0
        volume[0, 0, 0, 2:] = 0.9
        volume[1, 0, 0, 2:] = 0.9
        path = self.root / "target.npy"
        np.save(path, volume)
        dataset = VolumeFieldDataset(
            target_path=str(path),
            volume_shape=VolumeShape(X=4, Y=1, Z=1, T=2),
        )
        cache_path = self.root / "assignments.npy"
        cfg = PretrainAssignmentConfig(seed=123, cache_path=str(cache_path))
        assignments = compute_pretrain_assignments(dataset, 2, cfg)
        cached = compute_pretrain_assignments(dataset, 2, cfg)
        self.assertTrue(cache_path.exists())
        self.assertEqual(assignments.shape, (4,))
        np.testing.assert_array_equal(assignments, cached)

    def test_volume_time_sample_count_tracks_num_experts(self):
        volume = np.linspace(-1.0, 1.0, 12, dtype=np.float32).reshape(4, 1, 1, 3)
        path = self.root / "target.npy"
        np.save(path, volume)
        dataset = VolumeFieldDataset(
            target_path=str(path),
            volume_shape=VolumeShape(X=3, Y=1, Z=1, T=4),
        )
        with mock.patch.object(
            assignment_module,
            "_sample_time_indices",
            wraps=assignment_module._sample_time_indices,
        ) as sample_mock:
            assignments = compute_pretrain_assignments(dataset, 3, PretrainAssignmentConfig(seed=123))
        self.assertEqual(assignments.shape, (3,))
        self.assertGreaterEqual(int(assignments.min()), 0)
        self.assertLess(int(assignments.max()), 3)
        sample_mock.assert_called_once_with(4, 3, 123)


if __name__ == "__main__":
    unittest.main()


