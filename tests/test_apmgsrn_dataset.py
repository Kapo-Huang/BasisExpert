import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from var_expert_inr.methods.apmgsrn.dataset import IonizationTargetReader, IonizationTimestepDataset


class APMGSRNDatasetTestCase(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tmpdir.name)

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_timestep_sampling_and_reshape_match_expected_values(self):
        volume = np.linspace(-1.0, 1.0, 16, dtype=np.float32).reshape(2, 2, 2, 2)
        volume_path = self.root / "target.npy"
        np.save(volume_path, volume)

        reader = IonizationTargetReader(volume_path, {"X": 2, "Y": 2, "Z": 2, "T": 2})
        dataset = IonizationTimestepDataset(reader, time_index=1, align_corners=True, device="cpu")

        coords = torch.tensor(
            [
                [-1.0, -1.0, -1.0],
                [1.0, 1.0, 1.0],
                [1.0, -1.0, -1.0],
                [-1.0, 1.0, 1.0],
            ],
            dtype=torch.float32,
        )
        sampled = dataset.sample_points(coords).numpy()
        expected = np.array(
            [
                [volume[1, 0, 0, 0]],
                [volume[1, 1, 1, 1]],
                [volume[1, 0, 0, 1]],
                [volume[1, 1, 1, 0]],
            ],
            dtype=np.float32,
        )
        np.testing.assert_allclose(sampled, expected, atol=1.0e-6)

        reshaped = dataset.reshape_flat_predictions(np.arange(8, dtype=np.float32).reshape(8, 1))
        self.assertEqual(reshaped.shape, (2, 2, 2))
        self.assertEqual(float(reshaped[0, 0, 0]), 0.0)
        self.assertEqual(float(reshaped[0, 0, 1]), 1.0)
        self.assertEqual(float(reshaped[1, 1, 1]), 7.0)

    def test_flat_n_by_one_input_matches_dense_timestep_values(self):
        volume = np.linspace(-1.0, 1.0, 24, dtype=np.float32).reshape(3, 2, 2, 2)
        volume_path = self.root / "target_flat_2d.npy"
        np.save(volume_path, volume.reshape(-1, 1))

        reader = IonizationTargetReader(volume_path, {"X": 2, "Y": 2, "Z": 2, "T": 3})
        np.testing.assert_allclose(reader.timestep_array(1), volume[1], atol=1.0e-6)

        dataset = IonizationTimestepDataset(reader, time_index=1, align_corners=True, device="cpu")
        coords = torch.tensor(
            [
                [-1.0, -1.0, -1.0],
                [1.0, 1.0, 1.0],
            ],
            dtype=torch.float32,
        )
        sampled = dataset.sample_points(coords).numpy()
        expected = np.array(
            [
                [volume[1, 0, 0, 0]],
                [volume[1, 1, 1, 1]],
            ],
            dtype=np.float32,
        )
        np.testing.assert_allclose(sampled, expected, atol=1.0e-6)

    def test_flat_n_input_matches_dense_timestep_values(self):
        volume = np.linspace(-1.0, 1.0, 24, dtype=np.float32).reshape(3, 2, 2, 2)
        volume_path = self.root / "target_flat_1d.npy"
        np.save(volume_path, volume.reshape(-1))

        reader = IonizationTargetReader(volume_path, {"X": 2, "Y": 2, "Z": 2, "T": 3})
        np.testing.assert_allclose(reader.timestep_array(2), volume[2], atol=1.0e-6)

    def test_reader_rejects_volume_shape_mismatch(self):
        volume = np.zeros((2, 2, 2, 2), dtype=np.float32)
        volume_path = self.root / "bad_target.npy"
        np.save(volume_path, volume)

        with self.assertRaisesRegex(ValueError, "does not match target array shape"):
            IonizationTargetReader(volume_path, {"X": 2, "Y": 2, "Z": 2, "T": 3})

    def test_reader_rejects_invalid_flat_size(self):
        volume = np.zeros((23, 1), dtype=np.float32)
        volume_path = self.root / "bad_flat_size.npy"
        np.save(volume_path, volume)

        with self.assertRaisesRegex(ValueError, "does not match flat target size"):
            IonizationTargetReader(volume_path, {"X": 2, "Y": 2, "Z": 2, "T": 3})

    def test_reader_rejects_invalid_flat_channel_dimension(self):
        volume = np.zeros((24, 2), dtype=np.float32)
        volume_path = self.root / "bad_flat_channels.npy"
        np.save(volume_path, volume)

        with self.assertRaisesRegex(ValueError, "must be scalar"):
            IonizationTargetReader(volume_path, {"X": 2, "Y": 2, "Z": 2, "T": 3})


if __name__ == "__main__":
    unittest.main()
