import tempfile
import unittest
from pathlib import Path

import yaml

from var_expert_inr.neural_expert.config import load_config


class NeuralExpertConfigTestCase(unittest.TestCase):
    def test_repo_configs_load(self):
        repo_root = Path(__file__).resolve().parents[1]
        config_paths = sorted((repo_root / "configs" / "NeuralExpert").rglob("*.yaml"))
        self.assertTrue(any(path.parts[-2] == "Size163" and path.name == "ionization.yaml" for path in config_paths))
        for path in config_paths:
            cfg = load_config(path)
            self.assertIn("DATA", cfg)
            self.assertIn("MODEL", cfg)
            self.assertIn("TRAINING", cfg)
            self.assertTrue(Path(cfg["DATA"]["target_path"]).exists())
            if "source_path" in cfg["DATA"]:
                self.assertTrue(Path(cfg["DATA"]["source_path"]).exists())

    def test_target_placeholder_and_identifier_override(self):
        payload = {
            "experiment": "demo-{target}",
            "exp_id": "demo-{target}",
            "experiment_root": "./runs",
            "MODEL": {"manager_pt_path": "./pt_{target}.pth"},
            "LOSS": {"loss_type": "1000valrecon"},
            "DATA": {
                "dataset_name": "linkage_p",
                "source_path": "./coords.npy",
                "target": "point_RF",
                "targets": {
                    "point_RF": "./rf.npy",
                    "point_U": "./u.npy",
                },
                "target_stats_path": "./stats_{target}.npz",
                "stats_key": "{target}",
            },
            "TRAINING": {"pretrain_assignment": {"cache_path": "./cache_{target}.npz"}},
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.yaml"
            config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
            cfg = load_config(config_path, target_override="point_U", identifier="manual-id")
            self.assertEqual(cfg["DATA"]["target"], "point_U")
            self.assertEqual(cfg["DATA"]["attr_name"], "point_U")
            self.assertEqual(cfg["exp_id"], "manual-id")
            self.assertTrue(str(cfg["MODEL"]["manager_pt_path"]).endswith("pt_point_U.pth"))
            self.assertTrue(str(cfg["TRAINING"]["pretrain_assignment"]["cache_path"]).endswith("cache_point_U.npz"))


if __name__ == "__main__":
    unittest.main()


