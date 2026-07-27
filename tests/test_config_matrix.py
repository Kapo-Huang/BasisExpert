import unittest
from pathlib import Path

import numpy as np
import yaml

from var_expert_inr.apmgsrn.model import APMGSRN
from var_expert_inr.config.schema import ModelConfig
from var_expert_inr.data.base import DatasetMeta
from var_expert_inr.fv_srn.model import TemporalFVSRN
from var_expert_inr.mc_inr.data import TargetLayoutEntry
from var_expert_inr.mc_inr.model import MCINR
from var_expert_inr.models import build_model
from var_expert_inr.neural_expert.ionization.model_registry import build_model as build_neural_model
from var_expert_inr.rmdsrn.model import RMDSRN


DATASET_TARGETS = {
    "bathymetry": {"SALT", "TEMP", "U", "V"},
    "katrina": {"fort63", "fort64", "fort73", "speed", "v"},
    "ionization": {"GT", "H_plus", "H2", "He", "PD"},
}
SIZES = {
    "Size082": 0.82,
    "Size163": 1.63,
    "Size326": 3.26,
    "Size652": 6.52,
    "Size1304": 13.04,
}


class ConfigMatrixTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.repo_root = Path(__file__).resolve().parents[1]
        cls.config_root = cls.repo_root / "configs"
        cls.paths = sorted(cls.config_root.rglob("*.yaml"))

    def test_matrix_contains_exactly_352_configs_and_no_removed_datasets(self):
        self.assertEqual(len(self.paths), 352)
        relative_names = [str(path.relative_to(self.config_root)).lower() for path in self.paths]
        self.assertFalse(any("car" in name or "linkage" in name for name in relative_names))

    def test_single_target_default_and_size_coverage(self):
        for family in ("SIREN", "CoordNet", "MoE-INR", "NeuralExpert"):
            for dataset, targets in DATASET_TARGETS.items():
                actual = {
                    path.stem.split("__")[1]
                    for path in (self.config_root / family).glob(f"{dataset}__*.yaml")
                    if "managerpretrain" not in path.stem
                }
                self.assertEqual(actual, targets, (family, dataset))
            for size in SIZES:
                actual = {
                    path.stem.split("__")[1]
                    for path in (self.config_root / family / size).glob("ionization__*.yaml")
                    if "managerpretrain" not in path.stem
                }
                self.assertEqual(actual, DATASET_TARGETS["ionization"], (family, size))

        for family in ("APMGSRN", "DC-INR", "fV-SRN", "RMDSRN"):
            actual = {path.stem.split("__")[1] for path in (self.config_root / family).glob("ionization__*.yaml")}
            self.assertEqual(actual, DATASET_TARGETS["ionization"])
            for size in SIZES:
                sized = {
                    path.stem.split("__")[1]
                    for path in (self.config_root / family / size).glob("ionization__*.yaml")
                }
                self.assertEqual(sized, DATASET_TARGETS["ionization"])

        compact_targets = {
            path.stem.split("__")[1]
            for path in (self.config_root / "CompactNGP").glob("ionization__*.yaml")
        }
        self.assertEqual(compact_targets, DATASET_TARGETS["ionization"])
        self.assertEqual(
            list((self.config_root / "CompactNGP").glob("bathymetry__*.yaml")), []
        )
        self.assertEqual(
            list((self.config_root / "CompactNGP").glob("katrina__*.yaml")), []
        )
        fa_tr_targets = {
            path.stem.split("__")[1]
            for path in (self.config_root / "FA-TR-INR").glob(
                "ionization__*.yaml"
            )
        }
        self.assertEqual(fa_tr_targets, DATASET_TARGETS["ionization"])
        self.assertEqual(
            list(
                (self.config_root / "FA-TR-INR").glob(
                    "bathymetry__*.yaml"
                )
            ),
            [],
        )
        self.assertEqual(
            list(
                (self.config_root / "FA-TR-INR").glob(
                    "katrina__*.yaml"
                )
            ),
            [],
        )
        instant_targets = {
            path.stem.split("__")[1]
            for path in (self.config_root / "InstantNGP").glob(
                "ionization__*.yaml"
            )
        }
        self.assertEqual(instant_targets, DATASET_TARGETS["ionization"])
        self.assertEqual(
            list(
                (self.config_root / "InstantNGP").glob(
                    "bathymetry__*.yaml"
                )
            ),
            [],
        )
        self.assertEqual(
            list(
                (self.config_root / "InstantNGP").glob(
                    "katrina__*.yaml"
                )
            ),
            [],
        )

    def test_neural_expert_has_matching_manager_pretrains(self):
        root = self.config_root / "NeuralExpert"
        main_configs = [path for path in root.rglob("*.yaml") if "managerpretrain" not in path.stem]
        self.assertEqual(len(main_configs), 39)
        for main in main_configs:
            manager = main.with_name(f"{main.stem}__managerpretrain.yaml")
            self.assertTrue(manager.exists(), main)

    def test_all_ionization_configs_use_100_timesteps(self):
        for path in self.paths:
            payload = yaml.safe_load(path.read_text(encoding="utf-8"))
            data = payload.get("data") or payload.get("DATA")
            if data.get("dataset_name") == "ionization":
                self.assertEqual(data["volume_shape"]["T"], 100, path)

    def test_generated_paths_are_repo_root_relative(self):
        for path in self.paths:
            payload = yaml.safe_load(path.read_text(encoding="utf-8"))
            for value_path, value in self._walk_strings(payload):
                if "../" in value or "..\\" in value:
                    self.fail(f"Generated config contains ambiguous relative path at {path}:{value_path}: {value}")
                if self._looks_like_generated_path(value_path, value):
                    self.assertTrue(
                        value.startswith("${REPO_ROOT}/"),
                        f"Expected repo-root path at {path}:{value_path}, got {value!r}",
                    )

    def test_generated_configs_disable_training_predictions_and_step_windows(self):
        for path in self.paths:
            payload = yaml.safe_load(path.read_text(encoding="utf-8"))
            family = path.relative_to(self.config_root).parts[0]
            if family in {"VarExpert", "SIREN", "CoordNet", "MoE-INR", "CompactNGP", "FA-TR-INR", "InstantNGP", "MC-INR", "DC-INR", "fV-SRN"}:
                self.assertFalse(payload["evaluation"]["save_predictions"], path)
            if family in {"fV-SRN", "RMDSRN"}:
                self.assertFalse(payload["evaluation"]["run_after_training"], path)
            if family == "APMGSRN":
                self.assertFalse(payload["EVALUATION"]["run_after_training"], path)
            timing = (payload.get("log") or {}).get("timing")
            if timing is not None:
                self.assertFalse(timing["step_window"], path)

    def test_var_expert_alpha_is_five(self):
        for path in (self.config_root / "VarExpert").rglob("*.yaml"):
            payload = yaml.safe_load(path.read_text(encoding="utf-8"))
            self.assertEqual(payload["training"]["multiview_ema_loss"]["alpha"], 5.0, path)

    def test_fa_tr_inr_uses_fixed_model_and_optimizer(self):
        for path in (self.config_root / "FA-TR-INR").glob("*.yaml"):
            payload = yaml.safe_load(path.read_text(encoding="utf-8"))
            model = payload["model"]
            self.assertEqual(model["frequency_coordinates"], [1.0, 2.0, 3.0])
            self.assertEqual(model["omega"], 19.0)
            self.assertEqual(model["factor_mlp_depth"], 4)
            self.assertEqual(model["factor_hidden_width"], 128)
            self.assertEqual(model["integration_mlp_depth"], 2)
            self.assertEqual(model["tensor_ring_ranks"], [22, 88, 3, 3, 5])

            training = payload["training"]
            self.assertEqual(training["loss_type"], "mse")
            self.assertEqual(training["lr"], 1.0e-4)
            self.assertEqual(training["beta_1"], 0.9)
            self.assertEqual(training["beta_2"], 0.999)
            self.assertEqual(training["epsilon"], 1.0e-8)
            self.assertEqual(training["weight_decay"], 0.0)
            self.assertFalse(training["scheduler"]["enabled"])

    def test_instant_ngp_uses_fixed_model_optimizer_and_single_target(self):
        expected_targets = DATASET_TARGETS["ionization"]
        actual_targets = set()
        for path in (self.config_root / "InstantNGP").glob("*.yaml"):
            payload = yaml.safe_load(path.read_text(encoding="utf-8"))
            actual_targets.update(payload["data"]["targets"])
            self.assertEqual(len(payload["data"]["targets"]), 1, path)
            self.assertNotIn("target", payload["data"])

            model = payload["model"]
            self.assertEqual(model["name"], "instant_ngp")
            self.assertEqual(model["in_features"], 4)
            self.assertEqual(model["out_features"], 1)
            self.assertEqual(model["n_levels"], 16)
            self.assertEqual(model["n_features_per_level"], 2)
            self.assertEqual(model["base_resolution"], 16)
            self.assertEqual(model["finest_resolution"], 600)
            self.assertEqual(model["log2_hashmap_size"], 19)
            self.assertEqual(model["hidden_features"], 64)
            self.assertEqual(model["hidden_layers"], 2)

            training = payload["training"]
            self.assertEqual(training["epochs"], 600)
            self.assertEqual(training["batch_size"], 16_000)
            self.assertEqual(
                training["batches_per_epoch_budget"], 1_500
            )
            self.assertEqual(
                training["gradient_accumulation_steps"], 16
            )
            self.assertEqual(training["lr"], 1.0e-2)
            self.assertEqual(training["beta_1"], 0.9)
            self.assertEqual(training["beta_2"], 0.99)
            self.assertEqual(training["epsilon"], 1.0e-15)
            self.assertEqual(training["weight_decay"], 0.0)
            self.assertFalse(training["pretrain"]["enabled"])
            self.assertNotIn("max_steps", training)
            scheduler = training["scheduler"]
            self.assertTrue(scheduler["enabled"])
            self.assertEqual(scheduler["interval"], "optimizer_step")
            self.assertEqual(scheduler["milestones"], [20_480, 30_720])
            self.assertEqual(scheduler["gamma"], 0.33)
        self.assertEqual(actual_targets, expected_targets)

    def test_var_expert_ionization_dwa_config_uses_dwa_not_ema(self):
        payload = yaml.safe_load((self.config_root / "VarExpert" / "ionization_dwa.yaml").read_text(encoding="utf-8"))
        self.assertFalse(payload["training"]["multiview_ema_loss"]["enabled"])
        self.assertTrue(payload["training"]["multiview_dwa_loss"]["enabled"])
        self.assertEqual(payload["training"]["multiview_dwa_loss"]["temperature"], 2.0)
        self.assertEqual(payload["training"]["multiview_dwa_loss"]["eps"], 1.0e-12)

    def test_primary_training_budget_is_exact(self):
        expected = 16_000 * 1_500 * 600
        for path in self.paths:
            if "managerpretrain" in path.stem:
                continue
            payload = yaml.safe_load(path.read_text(encoding="utf-8"))
            family = path.relative_to(self.config_root).parts[0]
            if family in {"VarExpert", "SIREN", "CoordNet", "MoE-INR", "CompactNGP", "FA-TR-INR", "InstantNGP"}:
                training = payload["training"]
                total = training["batch_size"] * training["batches_per_epoch_budget"] * training["epochs"]
            elif family == "NeuralExpert":
                total = payload["TRAINING"]["n_points"] * payload["TRAINING"]["num_epochs"]
            elif family == "MC-INR":
                training = payload["training"]
                total = training["batch_size"] * training["batches_per_epoch_budget"] * training["finetune_epochs"]
            elif family == "APMGSRN":
                total = (
                    payload["DATA"]["volume_shape"]["T"]
                    * payload["TRAINING"]["iterations"]
                    * payload["TRAINING"]["points_per_iteration"]
                )
            elif family == "DC-INR":
                total = payload["training"]["total_steps"] * payload["training"]["batch_size"]
            elif family == "fV-SRN":
                training = payload["training"]
                total = (
                    training["epochs"]
                    * payload["data"]["volume_shape"]["T"]
                    * training["samples_per_timestep"]
                )
            elif family == "RMDSRN":
                total = payload["training"]["steps"] * payload["training"]["batch_size"]
            else:
                raise AssertionError(f"Unhandled family: {family}")
            self.assertEqual(total, expected, path)

    def test_static_size_tiers_are_within_five_percent(self):
        for family in ("SIREN", "CoordNet", "MoE-INR", "VarExpert"):
            for size, target_mib in SIZES.items():
                filename = "ionization.yaml" if family == "VarExpert" else "ionization__GT.yaml"
                payload = yaml.safe_load((self.config_root / family / size / filename).read_text(encoding="utf-8"))
                model_payload = payload["model"]
                target_names = tuple(payload["data"]["targets"]) if family == "VarExpert" else ("GT",)
                meta = DatasetMeta(
                    kind="volume",
                    n_samples=1,
                    input_dim=4,
                    target_names=target_names,
                    target_dims={name: 1 for name in target_names},
                    volume_shape={"X": 1, "Y": 1, "Z": 1, "T": 1},
                )
                model = build_model(
                    ModelConfig(
                        name=model_payload["name"],
                        params={key: value for key, value in model_payload.items() if key != "name"},
                    ),
                    meta,
                )
                self._assert_size(model, target_mib, family, size)

        for size, target_mib in SIZES.items():
            payload = yaml.safe_load(
                (self.config_root / "NeuralExpert" / size / "ionization__GT.yaml").read_text(encoding="utf-8")
            )
            model, _ = build_neural_model(payload, payload["LOSS"])
            self._assert_size(model, target_mib, "NeuralExpert", size)

            payload = yaml.safe_load(
                (self.config_root / "MC-INR" / size / "ionization.yaml").read_text(encoding="utf-8")
            )
            layout = tuple(
                TargetLayoutEntry(name, index, index + 1, 1)
                for index, name in enumerate(payload["data"]["targets"])
            )
            model = MCINR(
                centroids=np.zeros((12, 3), dtype=np.float32),
                target_layout=layout,
                in_features=4,
                hidden_features=payload["model"]["hidden_features"],
                gfe_layers=payload["model"]["gfe_layers"],
                lfe_layers=payload["model"]["lfe_layers"],
            )
            self._assert_size(model, target_mib, "MC-INR", size)

            payload = yaml.safe_load(
                (self.config_root / "APMGSRN" / size / "ionization__GT.yaml").read_text(encoding="utf-8")
            )
            model = APMGSRN(payload["MODEL"], data_min=-1.0, data_max=1.0, use_tcnn=False)
            self._assert_size(model, target_mib, "APMGSRN", size, multiplier=100)

            payload = yaml.safe_load(
                (self.config_root / "fV-SRN" / size / "ionization__GT.yaml").read_text(encoding="utf-8")
            )
            self._assert_size(TemporalFVSRN(payload["model"]), target_mib, "fV-SRN", size)

            payload = yaml.safe_load(
                (self.config_root / "RMDSRN" / size / "ionization__GT.yaml").read_text(encoding="utf-8")
            )
            self._assert_size(RMDSRN(payload["model"]), target_mib, "RMDSRN", size)

    def _assert_size(self, model, target_mib, family, size, multiplier=1):
        actual_mib = sum(parameter.numel() for parameter in model.parameters()) * 2 * multiplier / (1024**2)
        self.assertLessEqual(abs(actual_mib - target_mib) / target_mib, 0.05, (family, size, actual_mib))

    def _walk_strings(self, value, prefix=""):
        if isinstance(value, dict):
            for key, item in value.items():
                yield from self._walk_strings(item, f"{prefix}.{key}" if prefix else str(key))
        elif isinstance(value, list):
            for index, item in enumerate(value):
                yield from self._walk_strings(item, f"{prefix}[{index}]")
        elif isinstance(value, str):
            yield prefix, value

    def _looks_like_generated_path(self, value_path: str, value: str) -> bool:
        key = value_path.rsplit(".", 1)[-1]
        return (
            key.endswith("_path")
            or key == "experiment_root"
            or ".targets." in value_path
            or value.startswith("${REPO_ROOT}/")
        )


if __name__ == "__main__":
    unittest.main()
