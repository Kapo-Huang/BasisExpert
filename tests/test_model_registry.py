import unittest

from var_expert_inr.config.schema import ModelConfig
from var_expert_inr.data.base import DatasetMeta
from var_expert_inr.models import build_model, materialize_model_config


class ModelRegistryTestCase(unittest.TestCase):
    def test_materialize_model_config_applies_defaults(self):
        meta = DatasetMeta(
            kind="node",
            n_samples=8,
            input_dim=4,
            target_names=("target",),
            target_dims={"target": 1},
            volume_shape=None,
        )
        payload = materialize_model_config(
            ModelConfig(name="siren", params={"hidden_features": 8, "hidden_layers": 1}),
            meta,
        )
        self.assertEqual(payload["name"], "siren")
        self.assertEqual(payload["in_features"], 4)
        self.assertEqual(payload["out_features"], 1)
        self.assertTrue(payload["outermost_linear"])

    def test_build_native_models(self):
        multi_meta = DatasetMeta(
            kind="node",
            n_samples=8,
            input_dim=4,
            target_names=("a", "b"),
            target_dims={"a": 1, "b": 2},
            volume_shape=None,
        )
        single_meta = DatasetMeta(
            kind="node",
            n_samples=8,
            input_dim=4,
            target_names=("target",),
            target_dims={"target": 1},
            volume_shape=None,
        )
        light_basis_model = build_model(
            ModelConfig(name="light_basis_expert", params={"in_features": 4, "num_experts": 2, "base_dim": 2}),
            multi_meta,
        )
        native_model = build_model(
            ModelConfig(name="coordnet", params={"in_features": 4, "init_features": 4, "num_res": 1}),
            single_meta,
        )
        self.assertEqual(type(light_basis_model.backbone).__name__, "LightBasisExpert")
        self.assertEqual(type(native_model.backbone).__name__, "CoordNet")

    def test_legacy_model_alias_is_rejected(self):
        meta = DatasetMeta(
            kind="node",
            n_samples=8,
            input_dim=4,
            target_names=("a", "b"),
            target_dims={"a": 1, "b": 2},
            volume_shape=None,
        )
        with self.assertRaises(ValueError):
            build_model(
                ModelConfig(name="basisExpert_simple_concat", params={"in_features": 4, "num_experts": 2, "base_dim": 2}),
                meta,
            )

    def test_unknown_model_key_is_rejected(self):
        meta = DatasetMeta(
            kind="node",
            n_samples=8,
            input_dim=4,
            target_names=("target",),
            target_dims={"target": 1},
            volume_shape=None,
        )
        with self.assertRaises(ValueError):
            materialize_model_config(
                ModelConfig(name="coordnet", params={"in_features": 4, "init_features": 4, "num_res": 1, "legacy_flag": True}),
                meta,
            )


if __name__ == "__main__":
    unittest.main()
