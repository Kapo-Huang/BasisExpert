import tempfile
import unittest
from pathlib import Path

import numpy as np
import yaml

from var_expert_inr.apmgsrn.config import load_config


class APMGSRNConfigTestCase(unittest.TestCase):
    def _write_yaml(self, path: Path, payload) -> Path:
        path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
        return path

    def test_target_override_identifier_and_all_time_indices(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            np.save(root / "target_GT.npy", np.zeros((3, 2, 2, 2), dtype=np.float32))
            np.save(root / "target_H2.npy", np.zeros((3, 2, 2, 2), dtype=np.float32))
            payload = {
                "experiment": "demo-{target}",
                "exp_id": "demo-{target}",
                "experiment_root": "./runs/apmgsrn",
                "MODEL": {
                    "feature_grid_shape": [2, 2, 2],
                },
                "DATA": {
                    "dataset_name": "ionization",
                    "target": "GT",
                    "targets": {
                        "GT": "./target_GT.npy",
                        "H2": "./target_H2.npy",
                    },
                    "volume_shape": {"X": 2, "Y": 2, "Z": 2, "T": 3},
                },
                "TRAINING": {
                    "time_indices": "all",
                },
            }
            config_path = self._write_yaml(root / "config.yaml", payload)
            cfg = load_config(config_path, target_override="H2", identifier="manual-id")
            self.assertEqual(cfg["DATA"]["target"], "H2")
            self.assertEqual(cfg["DATA"]["attr_name"], "H2")
            self.assertEqual(cfg["exp_id"], "manual-id")
            self.assertEqual(cfg["TRAINING"]["time_indices"], [0, 1, 2])
            self.assertTrue(str(cfg["DATA"]["target_path"]).endswith("target_H2.npy"))

    def test_explicit_time_indices_are_validated_and_preserved(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            np.save(root / "target.npy", np.zeros((4, 2, 2, 2), dtype=np.float32))
            payload = {
                "MODEL": {
                    "feature_grid_shape": [2, 2, 2],
                },
                "DATA": {
                    "dataset_name": "ionization",
                    "target_path": "./target.npy",
                    "volume_shape": {"X": 2, "Y": 2, "Z": 2, "T": 4},
                },
                "TRAINING": {
                    "time_indices": [3, 1],
                },
            }
            config_path = self._write_yaml(root / "config.yaml", payload)
            cfg = load_config(config_path)
            self.assertEqual(cfg["TRAINING"]["time_indices"], [3, 1])


if __name__ == "__main__":
    unittest.main()
