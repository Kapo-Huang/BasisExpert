import tempfile
import unittest
from pathlib import Path
from unittest import mock

import yaml

from scripts import generate_exploration_v3_configs as generator
from var_expert_inr.apmgsrn.config import load_config as load_apmg_config
from var_expert_inr.config import load_experiment_config
from var_expert_inr.dc_inr.config import load_config as load_dc_config
from var_expert_inr.fv_srn.config import load_config as load_fv_config
from var_expert_inr.mc_inr.config import load_config as load_mc_config
from var_expert_inr.neural_expert.config import load_config as load_neural_config
from var_expert_inr.rmdsrn.config import load_config as load_rm_config


EXPECTED_COUNTS = {
    "APMGSRN": 25,
    "CoordNet": 25,
    "DC-INR": 25,
    "fV-SRN": 25,
    "MC-INR": 5,
    "MoE-INR": 25,
    "NeuralExpert": 50,
    "RMDSRN": 25,
    "SIREN": 25,
    "VarExpert": 5,
}
SIZES = {"Size082", "Size163", "Size326", "Size652", "Size1304"}


class ExplorationV3ConfigMatrixTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.repo_root = Path(__file__).resolve().parents[1]
        cls.formal_root = cls.repo_root / "configs"
        cls.root = cls.repo_root / "configs_exploration_v3"
        cls.paths = sorted(cls.root.rglob("*.yaml"))

    def test_exact_formal_size_mirror(self):
        self.assertEqual(len(self.paths), generator.EXPECTED_TOTAL)
        formal_relatives = {
            path.relative_to(self.formal_root) for path in generator.formal_config_paths()
        }
        generated_relatives = {path.relative_to(self.root) for path in self.paths}
        self.assertEqual(generated_relatives, formal_relatives)
        counts: dict[str, int] = {}
        for path in self.paths:
            family = path.relative_to(self.root).parts[0]
            counts[family] = counts.get(family, 0) + 1
        self.assertEqual(counts, EXPECTED_COUNTS)
        self.assertEqual({path.relative_to(self.root).parts[1] for path in self.paths}, SIZES)

    def test_generator_rebuilds_in_isolated_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            generated_root = Path(tmpdir) / "configs_exploration_v3"
            with mock.patch.object(generator, "CONFIG_ROOT", generated_root):
                counts = generator.generate()
            self.assertEqual(counts, EXPECTED_COUNTS)
            self.assertEqual(len(list(generated_root.rglob("*.yaml"))), generator.EXPECTED_TOTAL)

    def test_roots_ids_probes_and_architectures(self):
        exp_ids: set[str] = set()
        for path in self.paths:
            relative = path.relative_to(self.root)
            family = relative.parts[0]
            generated = yaml.safe_load(path.read_text(encoding="utf-8"))
            formal = yaml.safe_load((self.formal_root / relative).read_text(encoding="utf-8"))
            self.assertEqual(generated["experiment_root"], "${REPO_ROOT}/runs/exploration_v3", path)
            self.assertEqual(generated["exploration_probe"], generator.PROBE, path)
            self.assertTrue(generated["exp_id"].startswith("explore-v3-"), path)
            self.assertNotIn(generated["exp_id"], exp_ids, path)
            exp_ids.add(generated["exp_id"])

            model_key = "MODEL" if family in {"APMGSRN", "NeuralExpert"} else "model"
            generated_model = dict(generated[model_key])
            formal_model = dict(formal[model_key])
            if family == "NeuralExpert":
                generated_model.pop("manager_pt_path")
                formal_model.pop("manager_pt_path")
            self.assertEqual(generated_model, formal_model, path)
            if family == "DC-INR":
                self.assertEqual(generated["partition"], formal["partition"], path)
                self.assertEqual(generated["compression"], formal["compression"], path)

    def test_every_config_loads_with_its_runner(self):
        loaders = {
            "APMGSRN": load_apmg_config,
            "MC-INR": load_mc_config,
            "DC-INR": load_dc_config,
            "fV-SRN": load_fv_config,
            "NeuralExpert": load_neural_config,
            "RMDSRN": load_rm_config,
        }
        for path in self.paths:
            family = path.relative_to(self.root).parts[0]
            loaded = loaders.get(family, load_experiment_config)(path)
            probe = loaded.get("exploration_probe") if isinstance(loaded, dict) else loaded.exploration_probe
            enabled = probe["enabled"] if isinstance(probe, dict) else probe.enabled
            self.assertTrue(enabled, path)

    def test_fifty_epoch_equivalent_training_lengths(self):
        for path in self.paths:
            family = path.relative_to(self.root).parts[0]
            payload = yaml.safe_load(path.read_text(encoding="utf-8"))
            if family in generator.GENERIC_FAMILIES:
                self.assertEqual(payload["training"]["epochs"], 50, path)
                self.assertEqual(payload["training"]["log_psnr_every"], 5, path)
            elif family == "MC-INR":
                self.assertEqual(payload["training"]["meta_iterations"], 5, path)
                self.assertEqual(payload["training"]["finetune_epochs"], 50, path)
            elif family == "NeuralExpert":
                manager = path.stem.endswith("__managerpretrain")
                self.assertEqual(payload["TRAINING"]["num_epochs"], 2_500 if manager else 75_000, path)
            elif family == "APMGSRN":
                self.assertEqual(payload["TRAINING"]["iterations"], 750, path)
            elif family == "DC-INR":
                self.assertEqual(payload["training"]["total_steps"], 75_000, path)
            elif family == "fV-SRN":
                self.assertEqual(payload["training"]["epochs"], 50, path)
            elif family == "RMDSRN":
                self.assertEqual(payload["training"]["steps"], 75_000, path)

    def test_neural_manager_dependencies_are_isolated_and_paired(self):
        for size in SIZES:
            size_root = self.root / "NeuralExpert" / size
            mains = sorted(path for path in size_root.glob("*.yaml") if "managerpretrain" not in path.stem)
            managers = sorted(path for path in size_root.glob("*__managerpretrain.yaml"))
            self.assertEqual(len(mains), 5)
            self.assertEqual(len(managers), 5)
            for main_path in mains:
                manager_path = main_path.with_name(f"{main_path.stem}__managerpretrain.yaml")
                self.assertTrue(manager_path.exists(), manager_path)
                main = yaml.safe_load(main_path.read_text(encoding="utf-8"))
                manager = yaml.safe_load(manager_path.read_text(encoding="utf-8"))
                expected_prefix = f"${{REPO_ROOT}}/runs/exploration_v3/neural_expert/pretrained_managers/ionization/{size.lower()}/"
                self.assertEqual(main["MODEL"]["manager_pt_path"], manager["MODEL"]["manager_pt_path"])
                self.assertTrue(main["MODEL"]["manager_pt_path"].startswith(expected_prefix))


if __name__ == "__main__":
    unittest.main()
