import tempfile
import unittest
from pathlib import Path

import numpy as np

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

    def test_node_sample_clustering_groups_similar_samples(self):
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
        cfg = PretrainAssignmentConfig(method="sample_clustering", seed=42)
        assignments = compute_pretrain_assignments(dataset, 2, cfg)
        self.assertEqual(assignments.shape, (4,))
        self.assertEqual(assignments[0], assignments[1])
        self.assertEqual(assignments[2], assignments[3])
        self.assertNotEqual(assignments[0], assignments[2])

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
        cfg = PretrainAssignmentConfig(
            method="voxel_clustering",
            seed=123,
            cache_path=str(cache_path),
            cluster_num_time_samples=2,
        )
        assignments = compute_pretrain_assignments(dataset, 2, cfg)
        cached = compute_pretrain_assignments(dataset, 2, cfg)
        self.assertTrue(cache_path.exists())
        self.assertEqual(assignments.shape, (4,))
        np.testing.assert_array_equal(assignments, cached)


if __name__ == "__main__":
    unittest.main()
