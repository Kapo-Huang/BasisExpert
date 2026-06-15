import tempfile
import unittest
from pathlib import Path

import numpy as np

try:
    import h5py
except ImportError:  # pragma: no cover - depends on local environment.
    h5py = None

if h5py is not None:
    from var_expert_inr.utils.hdf5_conversion import DatasetConversionSpec, convert_dataset


@unittest.skipIf(h5py is None, "h5py is not installed")
class HDF5ConversionTestCase(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tmpdir.name)

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_node_conversion_writes_expected_hdf5_structure(self):
        input_dir = self.root / "node"
        input_dir.mkdir()
        coords = np.array(
            [[-1.0, -0.5, 0.0, 0.5], [0.0, 0.25, 0.5, 0.75], [1.0, 0.5, 0.0, -0.5]],
            dtype=np.float64,
        )
        pressure = np.array([[0.1], [0.2], [0.3]], dtype=np.float64)
        velocity = np.array([[0.0, 0.1], [0.2, 0.3], [0.4, 0.5]], dtype=np.float64)
        np.save(input_dir / "coords.npy", coords)
        np.save(input_dir / "pressure.npy", pressure)
        np.save(input_dir / "velocity.npy", velocity)

        spec = DatasetConversionSpec(
            dataset_name="toy-node",
            dataset_kind="node",
            default_input_dir=input_dir,
            default_output_filename="toy_node.h5",
            coords_file="coords.npy",
            targets={"pressure": "pressure.npy", "velocity": "velocity.npy"},
        )
        output_path = self.root / "toy_node.h5"

        dataset_paths = convert_dataset(
            spec,
            input_dir=input_dir,
            output_path=output_path,
            chunk_rows=2,
            overwrite=False,
        )

        self.assertEqual(dataset_paths, ["/coords", "/targets/pressure", "/targets/velocity"])
        with h5py.File(output_path, "r") as h5_file:
            self.assertEqual(h5_file.attrs["dataset_name"], "toy-node")
            self.assertEqual(h5_file.attrs["dataset_kind"], "node")
            self.assertEqual(h5_file.attrs["source_format"], "npy")
            self.assertEqual(h5_file.attrs["stored_dtype"], "float32")
            self.assertEqual(h5_file.attrs["compression"], "lzf")

            self.assertEqual(h5_file["coords"].shape, coords.shape)
            self.assertEqual(h5_file["coords"].dtype, np.dtype(np.float32))
            self.assertEqual(h5_file["targets/pressure"].shape, pressure.shape)
            self.assertEqual(h5_file["targets/velocity"].shape, velocity.shape)
            self.assertEqual(h5_file["targets/pressure"].attrs["source_file"], "pressure.npy")
            np.testing.assert_array_equal(
                h5_file["targets/velocity"].attrs["original_shape"],
                np.asarray(velocity.shape, dtype=np.int64),
            )

            np.testing.assert_allclose(h5_file["coords"][...], coords.astype(np.float32))
            np.testing.assert_allclose(h5_file["targets/pressure"][...], pressure.astype(np.float32))
            np.testing.assert_allclose(h5_file["targets/velocity"][...], velocity.astype(np.float32))

    def test_volume_conversion_writes_extra_root_attributes(self):
        input_dir = self.root / "volume"
        input_dir.mkdir()
        target_a = np.linspace(-1.0, 1.0, 6, dtype=np.float32).reshape(3, 2)
        target_b = np.linspace(1.0, -1.0, 6, dtype=np.float32).reshape(3, 2)
        np.save(input_dir / "a.npy", target_a)
        np.save(input_dir / "b.npy", target_b)

        spec = DatasetConversionSpec(
            dataset_name="toy-volume",
            dataset_kind="volume",
            default_input_dir=input_dir,
            default_output_filename="toy_volume.h5",
            targets={"A": "a.npy", "B": "b.npy"},
            extra_root_attrs={
                "volume_shape_X": 2,
                "volume_shape_Y": 1,
                "volume_shape_Z": 1,
                "volume_shape_T": 3,
            },
        )
        output_path = self.root / "toy_volume.h5"

        dataset_paths = convert_dataset(
            spec,
            input_dir=input_dir,
            output_path=output_path,
            chunk_rows=2,
            overwrite=False,
        )

        self.assertEqual(dataset_paths, ["/targets/A", "/targets/B"])
        with h5py.File(output_path, "r") as h5_file:
            self.assertEqual(h5_file.attrs["dataset_name"], "toy-volume")
            self.assertEqual(h5_file.attrs["dataset_kind"], "volume")
            self.assertEqual(h5_file.attrs["volume_shape_X"], 2)
            self.assertEqual(h5_file.attrs["volume_shape_T"], 3)
            self.assertEqual(h5_file["targets/A"].dtype, np.dtype(np.float32))
            np.testing.assert_allclose(h5_file["targets/A"][...], target_a.astype(np.float32))
            np.testing.assert_allclose(h5_file["targets/B"][...], target_b.astype(np.float32))

    def test_conversion_rejects_existing_output_without_overwrite(self):
        input_dir = self.root / "node_existing"
        input_dir.mkdir()
        coords = np.array([[0.0, 0.1], [0.2, 0.3]], dtype=np.float32)
        target = np.array([[0.4], [0.5]], dtype=np.float32)
        np.save(input_dir / "coords.npy", coords)
        np.save(input_dir / "target.npy", target)

        spec = DatasetConversionSpec(
            dataset_name="toy-existing",
            dataset_kind="node",
            default_input_dir=input_dir,
            default_output_filename="toy_existing.h5",
            coords_file="coords.npy",
            targets={"target": "target.npy"},
        )
        output_path = self.root / "toy_existing.h5"
        output_path.write_bytes(b"placeholder")

        with self.assertRaisesRegex(FileExistsError, "Pass --overwrite"):
            convert_dataset(
                spec,
                input_dir=input_dir,
                output_path=output_path,
                chunk_rows=2,
                overwrite=False,
            )

    def test_node_conversion_rejects_missing_target_file(self):
        input_dir = self.root / "node_missing"
        input_dir.mkdir()
        coords = np.array([[0.0, 0.1], [0.2, 0.3]], dtype=np.float32)
        np.save(input_dir / "coords.npy", coords)

        spec = DatasetConversionSpec(
            dataset_name="toy-missing",
            dataset_kind="node",
            default_input_dir=input_dir,
            default_output_filename="toy_missing.h5",
            coords_file="coords.npy",
            targets={"target": "missing.npy"},
        )

        with self.assertRaisesRegex(FileNotFoundError, "Required source file is missing"):
            convert_dataset(
                spec,
                input_dir=input_dir,
                output_path=self.root / "toy_missing.h5",
                chunk_rows=2,
                overwrite=False,
            )


if __name__ == "__main__":
    unittest.main()
