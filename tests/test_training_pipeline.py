import tempfile
import unittest
from pathlib import Path

import numpy as np
import yaml

from var_expert_inr.cli import run_evaluate, run_predict, run_train


class TrainingPipelineTestCase(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tmpdir.name)

    def tearDown(self):
        self.tmpdir.cleanup()

    def _write_yaml(self, path: Path, payload) -> Path:
        path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
        return path

    def test_node_multitarget_train_predict_evaluate(self):
        coords = np.array(
            [
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 1.0],
                [1.0, 1.0, 0.0, 1.0],
                [0.5, 0.5, 0.0, 0.5],
                [0.2, 0.8, 0.0, 0.5],
            ],
            dtype=np.float32,
        )
        target_a = coords[:, :1] * 0.5
        target_b = np.concatenate([coords[:, 1:2], coords[:, 3:4]], axis=1)
        coords_path = self.root / "coords.npy"
        a_path = self.root / "a.npy"
        b_path = self.root / "b.npy"
        np.save(coords_path, coords)
        np.save(a_path, target_a)
        np.save(b_path, target_b)

        config = {
            "experiment": "node-pipeline",
            "exp_id": "node-pipeline",
            "experiment_root": str(self.root / "runs"),
            "data": {
                "kind": "node",
                "coords_path": str(coords_path),
                "targets": {"a": str(a_path), "b": str(b_path)},
            },
            "model": {
                "name": "light_basis_expert",
                "in_features": 4,
                "num_experts": 2,
                "base_dim": 2,
                "top_k": 1,
                "expert_num_layers": 2,
                "gate_num_layers": 2,
                "decoder_num_layers": 2,
                "head_num_layers": 2,
            },
            "training": {
                "epochs": 2,
                "batch_size": 3,
                "pred_batch_size": 3,
                "num_workers": 0,
                "lr": 1.0e-3,
                "device": "cpu",
                "seed": 1,
                "val_split": 0.0,
                "log_every": 1,
                "save_every": 1,
                "sampler": "uniform_random",
            },
            "evaluation": {"batch_size": 3},
        }
        config_path = self._write_yaml(self.root / "node.yaml", config)
        train_result = run_train(config_path)
        self.assertTrue(Path(train_result["checkpoint_path"]).exists())
        predict_result = run_predict(config_path)
        self.assertTrue(all(path.exists() for path in predict_result["prediction_paths"].values()))
        eval_result = run_evaluate(config_path)
        self.assertTrue(Path(eval_result["metrics_path"]).exists())

    def test_volume_single_target_train_predict_evaluate(self):
        volume = np.linspace(-1.0, 1.0, 8, dtype=np.float32).reshape(2, 1, 2, 2)
        volume_path = self.root / "volume.npy"
        np.save(volume_path, volume)
        config = {
            "experiment": "volume-pipeline",
            "exp_id": "volume-pipeline",
            "experiment_root": str(self.root / "runs"),
            "data": {
                "kind": "volume",
                "target_path": str(volume_path),
            },
            "model": {
                "name": "siren",
                "in_features": 4,
                "hidden_features": 8,
                "hidden_layers": 1,
            },
            "training": {
                "epochs": 2,
                "batch_size": 4,
                "pred_batch_size": 4,
                "num_workers": 0,
                "lr": 1.0e-3,
                "device": "cpu",
                "seed": 2,
                "val_split": 0.0,
                "log_every": 1,
                "save_every": 1,
                "sampler": "uniform_random",
            },
            "evaluation": {"batch_size": 4},
        }
        config_path = self._write_yaml(self.root / "volume.yaml", config)
        train_result = run_train(config_path)
        self.assertTrue(Path(train_result["checkpoint_path"]).exists())
        eval_result = run_evaluate(config_path)
        self.assertTrue(Path(eval_result["metrics_path"]).exists())


if __name__ == "__main__":
    unittest.main()
