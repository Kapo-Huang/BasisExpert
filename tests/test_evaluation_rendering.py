import unittest
from unittest.mock import patch

import numpy as np

from var_expert_inr.evaluation.rendering import preflight_rendering, resolve_clim, visual_scalar


class EvaluationRenderingTestCase(unittest.TestCase):
    def test_vector_rendering_uses_l2_magnitude(self):
        values = np.array([[3.0, 4.0], [0.0, 2.0]], dtype=np.float32)
        np.testing.assert_allclose(visual_scalar(values), [5.0, 2.0])

    def test_prediction_only_requires_fixed_clim(self):
        with self.assertRaisesRegex(ValueError, "Prediction-only"):
            resolve_clim({}, None)
        self.assertEqual(resolve_clim({"clim": [-1, 1]}, None), (-1.0, 1.0))

    def test_target_specific_clim_overrides_default(self):
        profile = {"clim": [-1, 1], "target_clims": {"v": [0, 2]}}
        self.assertEqual(resolve_clim(profile, None, target="v"), (0.0, 2.0))

    def test_node_preflight_rejects_mesh_association_size_mismatch(self):
        mesh = type("Mesh", (), {"n_points": 3, "n_cells": 1})()
        with patch("var_expert_inr.evaluation.rendering._load_mesh", return_value=(mesh, "mesh.vtu")):
            with self.assertRaisesRegex(ValueError, "mesh size mismatch"):
                preflight_rendering(
                    {"association": "point", "clim": [-1, 1]},
                    dataset_kind="node",
                    targets=("v",),
                    timesteps=(0,),
                    frame_sizes={0: 4},
                    prediction_only=True,
                    metrics=(),
                )


if __name__ == "__main__":
    unittest.main()
