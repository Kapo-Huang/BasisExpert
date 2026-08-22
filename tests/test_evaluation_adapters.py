import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

from scripts.tools import evaluate_neural_expert_config as neural_eval

from var_expert_inr.methods.apmgsrn.model import APMGSRN
from var_expert_inr.evaluation.standalone import (
    _decode_apmgsrn_frames,
    _decode_neural_expert_frames,
)
from var_expert_inr.methods.neural_expert.ionization.inr import INR
from var_expert_inr.methods.neural_expert.ionization.inr_moe import INR_MoE


class StandaloneAdapterContractTestCase(unittest.TestCase):
    def test_neural_expert_evaluator_bootstraps_src_when_invoked_directly(self):
        script = Path(neural_eval.__file__).resolve()
        with tempfile.TemporaryDirectory() as tmpdir:
            completed = subprocess.run(
                [sys.executable, "-I", str(script), "--help"],
                cwd=tmpdir,
                check=False,
                capture_output=True,
                text=True,
            )

        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        self.assertIn("Evaluate one completed NeuralExpert main config", completed.stdout)

    def test_apmgsrn_selected_checkpoint_decodes_one_frame(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "timesteps"
            checkpoint_dir = root / "t000"
            checkpoint_dir.mkdir(parents=True)
            model_config = {
                "n_dims": 3,
                "n_outputs": 1,
                "n_grids": 1,
                "n_features": 1,
                "feature_grid_shape": [2, 2, 2],
                "grid_initialization": "default",
                "nodes_per_layer": 4,
                "n_layers": 1,
                "use_bias": True,
                "requires_padded_feats": False,
            }
            model = APMGSRN(model_config, data_min=-1.0, data_max=1.0, use_tcnn=False)
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "model_config": model_config,
                    "data_min": -1.0,
                    "data_max": 1.0,
                    "time_index": 0,
                    "target_name": "GT",
                },
                checkpoint_dir / "checkpoint.pth",
            )
            frames = _decode_apmgsrn_frames(
                root,
                {"TRAINING": {"prediction_points_per_batch": 3}},
                timesteps=(0,),
                targets=("GT",),
                shape_tzyx=(1, 2, 2, 2),
                device=torch.device("cpu"),
            )
            self.assertEqual(frames[("GT", 0)].shape, (2, 2, 2))

    def test_neural_expert_checkpoint_decodes_selected_volume_frame(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            raw = {
                "MODEL": {
                    "model_name": "inr_ionization",
                    "in_dim": 4,
                    "out_dim": 1,
                    "decoder_hidden_dim": 4,
                    "decoder_n_hidden_layers": 1,
                    "decoder_input_encoding": "none",
                    "decoder_nl": "relu",
                    "decoder_init_type": "normal",
                    "decoder_freqs": 30.0,
                    "decoder_trainable_freqs": False,
                    "outermost_linear": True,
                },
                "DATA": {
                    "dataset_name": "ionization",
                    "normalize_inputs": True,
                    "normalize_targets": False,
                },
                "TRAINING": {"n_points": 3},
            }
            model = INR(raw)
            checkpoint = root / "neural.pth"
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "x_mean": np.zeros((1, 4), dtype=np.float32),
                    "x_std": np.ones((1, 4), dtype=np.float32),
                    "y_mean": np.zeros((1, 1), dtype=np.float32),
                    "y_std": np.ones((1, 1), dtype=np.float32),
                },
                checkpoint,
            )
            frames = _decode_neural_expert_frames(
                checkpoint,
                raw,
                timesteps=(1,),
                targets=("GT",),
                indexers=[slice(0, 8), slice(8, 16)],
                shape_tzyx=(2, 2, 2, 2),
                coords=None,
                repo_root=root,
                config_path=root / "config.yaml",
                device=torch.device("cpu"),
            )
            self.assertEqual(frames[("GT", 1)].shape, (2, 2, 2))

            raw["DATA"]["dataset_name"] = "combustion_40nh3_1"
            combustion_frames = _decode_neural_expert_frames(
                checkpoint,
                raw,
                timesteps=(0,),
                targets=("Temperature",),
                indexers=[slice(0, 8), slice(8, 16)],
                shape_tzyx=(2, 2, 2, 2),
                coords=None,
                repo_root=root,
                config_path=root / "config.yaml",
                device=torch.device("cpu"),
            )
            self.assertEqual(combustion_frames[("Temperature", 0)].shape, (2, 2, 2))

    def test_neural_expert_moe_decodes_uneven_prediction_chunks(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            raw = {
                "MODEL": {
                    "model_name": "inr_moe_ionization",
                    "in_dim": 4,
                    "out_dim": 1,
                    "decoder_hidden_dim": 4,
                    "decoder_n_hidden_layers": 1,
                    "decoder_input_encoding": "none",
                    "decoder_nl": "relu",
                    "decoder_init_type": "normal",
                    "decoder_freqs": 30.0,
                    "n_experts": 2,
                    "shared_encoder": True,
                    "manager_conditioning": "none",
                    "manager_type": "standard",
                    "manager_hidden_dim": 4,
                    "manager_n_hidden_layers": 1,
                    "manager_input_encoding": "none",
                    "manager_nl": "relu",
                    "manager_init": "normal",
                    "manager_q_activation": "softmax",
                    "manager_softmax_temperature": 1.0,
                    "manager_softmax_temp_trainable": False,
                    "manager_clamp_q": 0.0,
                },
                "DATA": {
                    "dataset_name": "combustion_40nh3_1",
                    "normalize_inputs": False,
                    "normalize_targets": False,
                },
                "TRAINING": {"n_points": 3},
            }
            model = INR_MoE(raw)
            checkpoint = root / "neural-moe.pth"
            torch.save({"model_state": model.state_dict()}, checkpoint)

            frames = _decode_neural_expert_frames(
                checkpoint,
                raw,
                timesteps=(0,),
                targets=("Temperature",),
                indexers=[slice(0, 8)],
                shape_tzyx=(1, 2, 2, 2),
                coords=None,
                repo_root=root,
                config_path=root / "config.yaml",
                device=torch.device("cpu"),
            )

            self.assertEqual(frames[("Temperature", 0)].shape, (2, 2, 2))

    def test_neural_expert_config_evaluator_selects_full_single_target_psnr(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config_path = root / "config.yaml"
            config_path.write_text("placeholder: true\n", encoding="utf-8")
            metrics_path = root / "evaluations" / "metrics.json"
            cfg = {
                "experiment_root": str(root / "runs"),
                "exp_id": "neural-expert-combustion-Temperature",
                "DATA": {"target": "Temperature"},
                "TRAINING": {"segmentation_mode": False},
            }
            evaluation_result = {
                "metrics": {"status": "complete", "aggregate": {"psnr": 42.25}},
                "metrics_path": metrics_path,
            }
            with mock.patch.object(neural_eval, "load_config", return_value=cfg), mock.patch.object(
                neural_eval, "evaluate_run", return_value=evaluation_result
            ) as evaluate_mock:
                record = neural_eval.evaluate_config(config_path, device="cpu")

            expected_run = (root / "runs" / cfg["exp_id"]).resolve()
            evaluate_mock.assert_called_once_with(
                expected_run,
                metrics="psnr",
                timesteps="all",
                targets="Temperature",
                source="auto",
                device="cpu",
            )
            self.assertEqual(record["target"], "Temperature")
            self.assertEqual(record["psnr"], 42.25)
            self.assertEqual(record["run_dir"], expected_run)

    def test_neural_expert_config_evaluator_rejects_nonfinite_psnr(self):
        cfg = {
            "experiment_root": ".",
            "exp_id": "run",
            "DATA": {"target": "Temperature"},
            "TRAINING": {"segmentation_mode": False},
        }
        result = {
            "metrics": {"status": "complete", "aggregate": {"psnr": float("nan")}},
            "metrics_path": "metrics.json",
        }
        with mock.patch.object(neural_eval, "load_config", return_value=cfg), mock.patch.object(
            neural_eval, "evaluate_run", return_value=result
        ):
            with self.assertRaisesRegex(RuntimeError, "non-finite PSNR"):
                neural_eval.evaluate_config("config.yaml")


if __name__ == "__main__":
    unittest.main()
