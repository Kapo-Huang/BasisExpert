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

    def test_matrix_contains_exactly_336_configs_and_no_removed_datasets(self):
        self.assertEqual(len(self.paths), 336)
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

    def test_primary_training_budget_is_exact(self):
        expected = 16_000 * 1_500 * 600
        for path in self.paths:
            if "managerpretrain" in path.stem:
                continue
            payload = yaml.safe_load(path.read_text(encoding="utf-8"))
            family = path.relative_to(self.config_root).parts[0]
            if family in {"VarExpert", "SIREN", "CoordNet", "MoE-INR"}:
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


if __name__ == "__main__":
    unittest.main()
