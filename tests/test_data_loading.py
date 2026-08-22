import tempfile
import unittest
from pathlib import Path

import numpy as np

from var_expert_inr.config.schema import DataConfig, VolumeShape
from var_expert_inr.data import build_dataset
from var_expert_inr.data.node import NodeFieldDataset
from var_expert_inr.data.volume import VolumeFieldDataset


class DataLoadingTestCase(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tmpdir.name)

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_node_single_target_fetches_expected_values(self):
        coords = np.array(
            [[-1.0, -0.5, 0.0, -1.0], [0.0, 0.5, 0.75, 0.0], [1.0, 0.25, -0.25, 1.0]],
            dtype=np.float32,
        )
        target = np.array([[-0.8], [0.0], [0.8]], dtype=np.float32)
        coords_path = self.root / "coords.npy"
        target_path = self.root / "target.npy"
        np.save(coords_path, coords)
        np.save(target_path, target)

        dataset = NodeFieldDataset(coords_path=str(coords_path), target_path=str(target_path))
        batch = dataset.fetch_batch([0, 2])
        np.testing.assert_allclose(batch.targets.numpy(), target[[0, 2]])

    def test_node_multi_target_has_stable_sorted_target_order(self):
        coords = np.array(
            [
                [-1.0, -1.0, -1.0, -1.0],
                [-0.5, 0.0, 0.5, -0.5],
                [0.0, 0.5, -0.5, 0.0],
                [1.0, 1.0, 1.0, 1.0],
            ],
            dtype=np.float32,
        )
        a = np.array([[-1.0], [-0.5], [0.5], [1.0]], dtype=np.float32)
        b = np.array([[-1.0, 0.0], [-0.5, 0.5], [0.5, -0.5], [1.0, 1.0]], dtype=np.float32)
        coords_path = self.root / "coords.npy"
        a_path = self.root / "a.npy"
        b_path = self.root / "b.npy"
        np.save(coords_path, coords)
        np.save(a_path, a)
        np.save(b_path, b)

        dataset = NodeFieldDataset(
            coords_path=str(coords_path),
            targets={"b": str(b_path), "a": str(a_path)},
        )
        self.assertEqual(dataset.target_names(), ("a", "b"))
        batch = dataset.fetch_batch([1, 3])
        self.assertEqual(sorted(batch.targets.keys()), ["a", "b"])

    def test_node_coordinates_can_use_saved_mean_and_std(self):
        coords = np.array(
            [[10.0, 20.0], [12.0, 24.0], [14.0, 28.0]],
            dtype=np.float32,
        )
        target = np.array([[-1.0], [0.0], [1.0]], dtype=np.float32)
        coords_path = self.root / "raw_coords.npy"
        target_path = self.root / "target.npy"
        stats_path = self.root / "stats.npz"
        np.save(coords_path, coords)
        np.save(target_path, target)
        np.savez(
            stats_path,
            x_mean=np.array([[12.0, 24.0]], dtype=np.float32),
            x_std=np.array([[2.0, 4.0]], dtype=np.float32),
        )

        dataset = NodeFieldDataset(
            coords_path=str(coords_path),
            coordinate_stats_path=str(stats_path),
            target_path=str(target_path),
        )
        batch = dataset.fetch_batch([0, 1, 2])
        np.testing.assert_allclose(
            batch.coords.numpy(),
            np.array([[-1.0, -1.0], [0.0, 0.0], [1.0, 1.0]], dtype=np.float32),
        )

    def test_volume_single_target_fetches_expected_values(self):
        volume = np.linspace(-1.0, 1.0, 8, dtype=np.float32).reshape(2, 1, 2, 2)
        volume_path = self.root / "volume.npy"
        np.save(volume_path, volume)

        dataset = VolumeFieldDataset(target_path=str(volume_path))
        batch = dataset.fetch_batch([0, 3, 7])
        self.assertEqual(tuple(batch.coords.shape), (3, 4))
        np.testing.assert_allclose(batch.targets.numpy(), volume.reshape(-1, 1)[[0, 3, 7]])

    def test_volume_multi_target_preserves_target_shapes(self):
        a = np.linspace(-1.0, 1.0, 8, dtype=np.float32).reshape(2, 1, 2, 2)
        b = np.linspace(-1.0, 1.0, 16, dtype=np.float32).reshape(2, 1, 2, 2, 2)
        a_path = self.root / "target_a.npy"
        b_path = self.root / "target_b.npy"
        np.save(a_path, a)
        np.save(b_path, b)

        dataset = VolumeFieldDataset(
            targets={"b": str(b_path), "a": str(a_path)},
            volume_shape=VolumeShape(X=2, Y=2, Z=1, T=2),
        )
        self.assertEqual(dataset.target_names(), ("a", "b"))
        batch = dataset.fetch_batch([0, 1, 7])
        self.assertEqual(tuple(batch.targets["a"].shape), (3, 1))
        self.assertEqual(tuple(batch.targets["b"].shape), (3, 2))

    def test_volume_can_omit_singleton_z_from_coordinates(self):
        volume = np.linspace(-1.0, 1.0, 16, dtype=np.float32).reshape(2, 1, 2, 4)
        volume_path = self.root / "volume_xyt.npy"
        np.save(volume_path, volume)

        dataset = VolumeFieldDataset(
            target_path=str(volume_path),
            coordinate_axes=("x", "y", "t"),
        )
        self.assertEqual(dataset.meta.input_dim, 3)
        batch = dataset.fetch_batch([0, 3, 4, 15])
        np.testing.assert_allclose(
            batch.coords.numpy(),
            np.array(
                [
                    [-1.0, -1.0, -1.0],
                    [1.0, -1.0, -1.0],
                    [-1.0, 1.0, -1.0],
                    [1.0, 1.0, 1.0],
                ],
                dtype=np.float32,
            ),
        )

    def test_volume_rejects_omitting_non_singleton_axis(self):
        volume = np.zeros((2, 2, 2, 2), dtype=np.float32)
        volume_path = self.root / "volume_xyz.npy"
        np.save(volume_path, volume)
        with self.assertRaisesRegex(ValueError, "non-singleton axes omitted: z"):
            VolumeFieldDataset(
                target_path=str(volume_path),
                coordinate_axes=("x", "y", "t"),
            )

    def test_build_dataset_selects_named_target_for_single_target_node_model(self):
        coords = np.array(
            [
                [-1.0, -1.0, -1.0, -1.0],
                [-0.5, 0.0, 0.5, -0.5],
                [0.0, 0.5, -0.5, 0.0],
                [1.0, 1.0, 1.0, 1.0],
            ],
            dtype=np.float32,
        )
        a = np.array([[-1.0], [-0.5], [0.5], [1.0]], dtype=np.float32)
        b = np.array([[-1.0, 0.0], [-0.5, 0.5], [0.5, -0.5], [1.0, 1.0]], dtype=np.float32)
        coords_path = self.root / "coords.npy"
        a_path = self.root / "a.npy"
        b_path = self.root / "b.npy"
        np.save(coords_path, coords)
        np.save(a_path, a)
        np.save(b_path, b)

        cfg = DataConfig(
            kind="node",
            coords_path=str(coords_path),
            targets={"a": str(a_path), "b": str(b_path)},
            target="b",
        )
        dataset = build_dataset(cfg, model_name="siren")
        self.assertEqual(dataset.target_names(), ("b",))
        batch = dataset.fetch_batch([1, 3])
        np.testing.assert_allclose(batch.targets.numpy(), b[[1, 3]])

    def test_build_dataset_selects_named_target_for_single_target_volume_model(self):
        a = np.linspace(-1.0, 1.0, 8, dtype=np.float32).reshape(2, 1, 2, 2)
        b = np.linspace(-1.0, 1.0, 16, dtype=np.float32).reshape(2, 1, 2, 2, 2)
        a_path = self.root / "target_a.npy"
        b_path = self.root / "target_b.npy"
        np.save(a_path, a)
        np.save(b_path, b)

        cfg = DataConfig(
            kind="volume",
            targets={"a": str(a_path), "b": str(b_path)},
            target="b",
            volume_shape=VolumeShape(X=2, Y=2, Z=1, T=2),
        )
        dataset = build_dataset(cfg, model_name="siren")
        self.assertEqual(dataset.target_names(), ("b",))
        batch = dataset.fetch_batch([0, 1, 7])
        self.assertEqual(tuple(batch.targets.shape), (3, 2))

    def test_build_dataset_rejects_target_selector_without_model_name(self):
        coords = np.array(
            [[-1.0, -1.0, -1.0, -1.0], [1.0, 1.0, 1.0, 1.0]],
            dtype=np.float32,
        )
        target = np.array([[-1.0], [1.0]], dtype=np.float32)
        coords_path = self.root / "coords.npy"
        target_path = self.root / "target.npy"
        np.save(coords_path, coords)
        np.save(target_path, target)

        cfg = DataConfig(
            kind="node",
            coords_path=str(coords_path),
            targets={"a": str(target_path)},
            target="a",
        )
        with self.assertRaisesRegex(ValueError, "requires build_dataset"):
            build_dataset(cfg)

    def test_build_dataset_rejects_target_selector_for_multi_target_model(self):
        coords = np.array(
            [[-1.0, -1.0, -1.0, -1.0], [1.0, 1.0, 1.0, 1.0]],
            dtype=np.float32,
        )
        target = np.array([[-1.0], [1.0]], dtype=np.float32)
        coords_path = self.root / "coords.npy"
        target_path = self.root / "target.npy"
        np.save(coords_path, coords)
        np.save(target_path, target)

        cfg = DataConfig(
            kind="node",
            coords_path=str(coords_path),
            targets={"a": str(target_path)},
            target="a",
        )
        with self.assertRaisesRegex(ValueError, "only supported for single-target models"):
            build_dataset(cfg, model_name="var_expert")

    def test_node_dataset_rejects_out_of_range_values(self):
        coords = np.array([[0.0, 0.0, 0.0, 0.0], [1.2, 0.0, 0.0, 0.0]], dtype=np.float32)
        target = np.array([[0.0], [0.1]], dtype=np.float32)
        coords_path = self.root / "bad_coords.npy"
        target_path = self.root / "target.npy"
        np.save(coords_path, coords)
        np.save(target_path, target)

        with self.assertRaises(ValueError):
            NodeFieldDataset(coords_path=str(coords_path), target_path=str(target_path))

    def test_volume_dataset_rejects_out_of_range_values(self):
        volume = np.array([[[[-1.0, 0.0], [0.5, 1.5]]]], dtype=np.float32)
        volume_path = self.root / "bad_volume.npy"
        np.save(volume_path, volume)

        with self.assertRaises(ValueError):
            VolumeFieldDataset(target_path=str(volume_path))


if __name__ == "__main__":
    unittest.main()


