import tempfile
import unittest
from pathlib import Path

import numpy as np
import yaml

from var_expert_inr.cli import run_predict, run_train


class CheckpointIntegrityTestCase(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tmpdir.name)

    def tearDown(self):
        self.tmpdir.cleanup()

    def _write_yaml(self, path: Path, payload) -> Path:
        path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
        return path

    def test_checkpoint_target_order_mismatch_raises(self):
        coords = np.array(
            [[0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 0.0], [1.0, 1.0, 0.0, 1.0]],
            dtype=np.float32,
        )
        a = coords[:, :1]
        b = coords[:, 1:2]
        coords_path = self.root / "coords.npy"
        a_path = self.root / "a.npy"
        b_path = self.root / "b.npy"
        np.save(coords_path, coords)
        np.save(a_path, a)
        np.save(b_path, b)

        base_config = {
            "experiment": "checkpoint-test",
            "exp_id": "checkpoint-test",
            "experiment_root": str(self.root / "runs"),
            "data": {
                "kind": "node",
                "coords_path": str(coords_path),
                "targets": {"a": str(a_path), "b": str(b_path)},
            },
            "model": {
                "name": "var_expert",
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
                "epochs": 1,
                "batch_size": 2,
                "pred_batch_size": 2,
                "num_workers": 0,
                "lr": 1.0e-3,
                "device": "cpu",
                "seed": 0,
                "val_split": 0.0,
                "log_every": 1,
                "sampler": "uniform_random",
            },
            "evaluation": {"batch_size": 2},
        }
        config_path = self._write_yaml(self.root / "base.yaml", base_config)
        train_result = run_train(config_path)
        self.assertTrue(Path(train_result["checkpoint_path"]).exists())

        mismatched = dict(base_config)
        mismatched["data"] = dict(base_config["data"])
        mismatched["data"]["targets"] = {"a": str(a_path), "c": str(b_path)}
        mismatch_path = self._write_yaml(self.root / "mismatch.yaml", mismatched)

        with self.assertRaises(ValueError):
            run_predict(mismatch_path, checkpoint_path=train_result["checkpoint_path"])

    def test_checkpoint_reload_preserves_predictions(self):
        coords = np.array(
            [[0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [1.0, 0.0, 0.0, 1.0], [1.0, 1.0, 0.0, 1.0]],
            dtype=np.float32,
        )
        target = ((coords[:, :1] + coords[:, 1:2]) * 0.5).astype(np.float32)
        coords_path = self.root / "coords_single.npy"
        target_path = self.root / "target_single.npy"
        np.save(coords_path, coords)
        np.save(target_path, target)

        config = {
            "experiment": "checkpoint-single",
            "exp_id": "checkpoint-single",
            "experiment_root": str(self.root / "runs_single"),
            "data": {
                "kind": "node",
                "coords_path": str(coords_path),
                "target_path": str(target_path),
            },
            "model": {
                "name": "coordnet",
                "in_features": 4,
                "init_features": 4,
                "num_res": 1,
            },
            "training": {
                "epochs": 1,
                "batch_size": 2,
                "pred_batch_size": 2,
                "num_workers": 0,
                "lr": 1.0e-3,
                "device": "cpu",
                "seed": 3,
                "val_split": 0.0,
                "log_every": 1,
                "sampler": "uniform_random",
            },
            "evaluation": {"batch_size": 2},
        }
        config_path = self._write_yaml(self.root / "single.yaml", config)
        train_result = run_train(config_path)
        first_predict = run_predict(config_path, checkpoint_path=train_result["checkpoint_path"])
        second_predict = run_predict(config_path, checkpoint_path=train_result["checkpoint_path"])
        first_name = next(iter(first_predict["predictions"].keys()))
        np.testing.assert_allclose(
            first_predict["predictions"][first_name],
            second_predict["predictions"][first_name],
        )


if __name__ == "__main__":
    unittest.main()


