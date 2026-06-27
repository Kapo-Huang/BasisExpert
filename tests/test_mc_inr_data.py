import unittest

import numpy as np

from var_expert_inr.config.schema import VolumeShape
from var_expert_inr.data.base import DatasetMeta
from var_expert_inr.mc_inr.data import (
    cluster_ids_for_rows,
    sample_volume_rows_from_cluster,
    sample_volume_rows_global,
    volume_rows_from_voxels_and_times,
    volume_voxel_count,
)


class MCINRDataTestCase(unittest.TestCase):
    def setUp(self):
        self.meta = DatasetMeta(
            kind="volume",
            n_samples=12,
            input_dim=4,
            target_names=("target",),
            target_dims={"target": 1},
            volume_shape=VolumeShape(X=2, Y=2, Z=1, T=3),
        )
        self.assignments = np.asarray([0, 0, 1, 1], dtype=np.int64)

    def test_volume_voxel_count_and_row_encoding(self):
        self.assertEqual(volume_voxel_count(self.meta), 4)
        rows = volume_rows_from_voxels_and_times(
            np.asarray([0, 3, 1], dtype=np.int64),
            np.asarray([0, 2, 1], dtype=np.int64),
            self.meta,
        )
        np.testing.assert_array_equal(rows, np.asarray([0, 11, 5], dtype=np.int64))

    def test_volume_rows_from_voxels_and_times_rejects_out_of_range_indices(self):
        with self.assertRaisesRegex(ValueError, "voxel_ids out of valid range"):
            volume_rows_from_voxels_and_times(
                np.asarray([4], dtype=np.int64),
                np.asarray([0], dtype=np.int64),
                self.meta,
            )

        with self.assertRaisesRegex(ValueError, "time_ids out of valid range"):
            volume_rows_from_voxels_and_times(
                np.asarray([0], dtype=np.int64),
                np.asarray([3], dtype=np.int64),
                self.meta,
            )

    def test_sample_volume_rows_from_cluster_and_global_use_valid_row_ranges(self):
        rng = np.random.default_rng(0)
        cluster_rows = sample_volume_rows_from_cluster(
            self.assignments,
            1,
            self.meta,
            row_count=5,
            rng=rng,
        )
        self.assertGreater(cluster_rows.size, 0)
        voxel_ids = cluster_rows % volume_voxel_count(self.meta)
        time_ids = cluster_rows // volume_voxel_count(self.meta)
        self.assertTrue(np.all(self.assignments[voxel_ids] == 1))
        self.assertTrue(np.all((0 <= time_ids) & (time_ids < int(self.meta.volume_shape.T))))

        global_rows = sample_volume_rows_global(
            self.assignments,
            self.meta,
            row_count=7,
            rng=np.random.default_rng(1),
        )
        self.assertTrue(np.all((0 <= global_rows) & (global_rows < int(self.meta.n_samples))))

    def test_cluster_ids_for_rows_requires_voxel_level_assignments_for_volume(self):
        rows = np.asarray([0, 5, 10], dtype=np.int64)
        cluster_ids = cluster_ids_for_rows(rows, self.assignments, self.meta)
        np.testing.assert_array_equal(cluster_ids, np.asarray([0, 0, 1], dtype=np.int64))

        with self.assertRaisesRegex(ValueError, "expected \\(4,\\)"):
            cluster_ids_for_rows(rows, np.asarray([0, 1, 2], dtype=np.int64), self.meta)


if __name__ == "__main__":
    unittest.main()
