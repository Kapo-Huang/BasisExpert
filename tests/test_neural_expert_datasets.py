import tempfile
import unittest
from pathlib import Path

import numpy as np

from var_expert_inr.methods.neural_expert.ionization.datasets import IonizationINRDataset
from var_expert_inr.methods.neural_expert.mesh.datasets import MeshAttributeDataset


class NeuralExpertDatasetTestCase(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tmpdir.name)

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_ionization_dataset_supports_t2_and_segmentation(self):
        target = np.linspace(-1.0, 1.0, 8, dtype=np.float32).reshape(8, 1)
        target_path = self.root / "target.npy"
        np.save(target_path, target)
        cfg = {
            "seed": 0,
            "DATA": {
                "target_path": str(target_path),
                "target_stats_path": str(self.root / "stats.npz"),
                "volume_shape": {"X": 2, "Y": 2, "Z": 1, "T": 2},
                "normalize_inputs": True,
                "normalize_targets": False,
                "n_segments": 4,
                "grid_patch_size": 1,
                "segmentation_type": "random_balanced",
                "attr_name": "H+",
            },
            "TRAINING": {
                "n_points": 4,
                "segmentation_mode": True,
            },
        }
        dataset = IonizationINRDataset(cfg, "H+")
        batch = dataset[0]
        self.assertEqual(tuple(batch["nonmnfld_points"].shape), (4, 4))
        self.assertEqual(tuple(batch["nonmnfld_segments_gt"].shape), (4,))

    def test_mesh_dataset_reads_normalized_arrays(self):
        coords = np.array(
            [
                [-1.0, -1.0, -1.0, -1.0],
                [-0.5, -0.5, -0.5, -0.5],
                [0.5, 0.5, 0.5, 0.5],
                [1.0, 1.0, 1.0, 1.0],
            ],
            dtype=np.float32,
        )
        target = np.array([[-1.0], [-0.5], [0.5], [1.0]], dtype=np.float32)
        coords_path = self.root / "coords.npy"
        target_path = self.root / "target.npy"
        np.save(coords_path, coords)
        np.save(target_path, target)

        cfg = {
            "seed": 0,
            "DATA": {
                "dataset_name": "linkage_p",
                "association": "point",
                "source_path": str(coords_path),
                "target_path": str(target_path),
                "target_stats_path": str(self.root / "stats.npz"),
                "stats_key": "point_RF",
                "normalize_inputs": False,
                "normalize_targets": False,
                "attr_name": "point_RF",
            },
            "MODEL": {
                "in_dim": 4,
                "out_dim": 1,
                "n_experts": 2,
            },
            "TRAINING": {
                "n_points": 3,
                "segmentation_mode": False,
                "pretrain_assignment": {"method": "coord_kmeans"},
            },
        }
        dataset = MeshAttributeDataset(cfg, "point_RF")
        batch = dataset[0]
        self.assertEqual(tuple(batch["nonmnfld_points"].shape), (3, 4))
        self.assertEqual(tuple(batch["nonmnfld_val"].shape), (3, 1))


if __name__ == "__main__":
    unittest.main()


