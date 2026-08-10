import unittest

import torch

from var_expert_inr.config.schema import ModelConfig
from var_expert_inr.data.base import DatasetMeta
from var_expert_inr.models import build_model
from var_expert_inr.models.sota.moe_inr import (
    MoEINR,
    SharedSirenEncoder,
    build_moe_inr_from_config,
)


class MoEINRTestCase(unittest.TestCase):
    @staticmethod
    def _meta() -> DatasetMeta:
        return DatasetMeta(
            kind="node",
            n_samples=8,
            input_dim=4,
            target_names=("target",),
            target_dims={"target": 1},
            volume_shape=None,
        )

    def test_materialized_small_base_dim_has_consistent_feature_widths(self):
        model = build_model(
            ModelConfig(
                name="moe_inr",
                params={"in_features": 4, "num_experts": 7, "base_dim": 14},
            ),
            self._meta(),
        )
        backbone = model.backbone

        self.assertEqual(backbone.encoder.out_dim, 112)
        self.assertEqual(backbone.policy.gate.in_features, 126)
        self.assertTrue(all(expert.mlp.in_features == 112 for expert in backbone.experts))
        prediction = model(torch.randn(8, 4))
        self.assertEqual(prediction.shape, (8, 1))

        prediction.square().mean().backward()
        self.assertIsNotNone(backbone.policy.gate.weight.grad)

    def test_explicit_base_dim_uses_encoder_output_for_consumers(self):
        model = MoEINR(
            in_features=4,
            encoder_feature_dim=256,
            base_dim=14,
            policy_hidden_dim=14,
        )

        self.assertEqual(model.policy.gate.in_features, model.encoder.out_dim + 14)
        self.assertTrue(
            all(expert.mlp.in_features == model.encoder.out_dim for expert in model.experts)
        )
        self.assertEqual(model(torch.randn(3, 4)).shape, (3, 1))

    def test_implicit_feature_dim_must_match_encoder_width_granularity(self):
        with self.assertRaisesRegex(ValueError, "positive multiple of 8"):
            SharedSirenEncoder(feature_dim=113)

    def test_config_rejects_conflicting_base_and_encoder_widths(self):
        with self.assertRaisesRegex(ValueError, "must equal 8 \\* base_dim"):
            build_moe_inr_from_config(
                {"base_dim": 14, "encoder_feature_dim": 256}
            )

    def test_config_honors_explicit_policy_width_with_base_dim(self):
        model = build_moe_inr_from_config(
            {
                "base_dim": 14,
                "encoder_feature_dim": 112,
                "policy_hidden_dim": 20,
            }
        )

        self.assertEqual(model.encoder.out_dim, 112)
        self.assertEqual(model.policy.gate.in_features, 132)


if __name__ == "__main__":
    unittest.main()
