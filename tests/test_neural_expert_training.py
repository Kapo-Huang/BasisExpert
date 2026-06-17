import tempfile
import unittest
from pathlib import Path

import numpy as np
import yaml

from var_expert_inr.neural_expert.cli import run_train


class NeuralExpertTrainingTestCase(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tmpdir.name)

    def tearDown(self):
        self.tmpdir.cleanup()

    def _write_yaml(self, path: Path, payload) -> Path:
        path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
        return path

    def test_ionization_manager_pretrain_smoke(self):
        target = np.linspace(-1.0, 1.0, 8, dtype=np.float32).reshape(8, 1)
        target_path = self.root / "target.npy"
        np.save(target_path, target)
        config = {
            "seed": 0,
            "wandb_project": "test_ionization",
            "experiment": "ionization-manager-pretrain",
            "exp_id": "ionization-manager-pretrain",
            "experiment_root": str(self.root / "runs"),
            "MODEL": {
                "model_name": "inr_moe_ionization",
                "in_dim": 4,
                "out_dim": 1,
                "decoder_hidden_dim": 8,
                "decoder_n_hidden_layers": 1,
                "decoder_input_encoding": "learned_8_1_sine_siren_none",
                "decoder_nl": "sine",
                "decoder_init_type": "siren",
                "n_experts": 2,
                "outermost_linear": True,
                "input_encoding": "none",
                "decoder_freqs": 30.0,
                "decoder_trainable_freqs": False,
                "top_k": 1,
                "manager_hidden_dim": 8,
                "manager_n_hidden_layers": 1,
                "manager_input_encoding": "learned_8_1_sine_siren_none",
                "manager_nl": "sine",
                "manager_init": "siren",
                "manager_type": "standard",
                "experts_bias_std": 0.1,
                "experts_bias_weight": 1.0,
                "manager_softmax_temperature": 1.0,
                "manager_softmax_temp_trainable": False,
                "manager_q_activation": "softmax",
                "manager_clamp_q": 0.0,
                "manager_conditioning": "cat",
                "manager_pt_path": str(self.root / "manager_pretrain.pth"),
                "load_pt_manager": False,
                "shared_encoder": False,
            },
            "LOSS": {
                "scale_by_q_grad": False,
                "loss_type": "1000segmentation",
                "segmentation_type": "both",
                "sample_bias_correction": False,
                "entropy_metric": "kl",
            },
            "DATA": {
                "dataset_name": "ionization",
                "target": "H+",
                "targets": {"H+": str(target_path)},
                "target_stats_path": str(self.root / "ion_stats.npz"),
                "volume_shape": {"X": 2, "Y": 2, "Z": 1, "T": 2},
                "normalize_inputs": True,
                "normalize_targets": False,
                "segmentation_type": "random_balanced",
                "grid_patch_size": 1,
                "n_segments": 2,
            },
            "TRAINING": {
                "n_points": 4,
                "lr": 1.0e-3,
                "lr_gamma": 0.99,
                "lr_scheduler": "ExponentialLR",
                "num_epochs": 1,
                "batch_size": 1,
                "num_workers": 0,
                "save_every": 1,
                "segmentation_mode": True,
                "timing": {"enabled": False},
                "stages": [{"end_iteration_frac": 1.0, "params": "all", "loss_type": "1000segmentation"}],
            },
        }
        config_path = self._write_yaml(self.root / "ionization.yaml", config)
        result = run_train(config_path)
        self.assertTrue(Path(result["checkpoint_path"]).exists())
        self.assertTrue(Path(result["validate_config_path"]).exists())
        self.assertTrue(Path(result["validate_checkpoint_path"]).exists())
        self.assertTrue((self.root / "manager_pretrain.pth").exists())

    def test_linkage_p_main_run_loads_pretrained_manager(self):
        coords = np.array(
            [
                [-1.0, -1.0, -1.0, -1.0],
                [-0.5, -0.5, -0.5, -0.5],
                [0.5, 0.5, 0.5, 0.5],
                [1.0, 1.0, 1.0, 1.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.25, -0.25, 0.25, -0.25],
            ],
            dtype=np.float32,
        )
        rf = np.array([[-1.0], [-0.5], [0.5], [1.0], [0.0], [0.25]], dtype=np.float32)
        u = np.array([[1.0], [0.5], [-0.5], [-1.0], [0.0], [-0.25]], dtype=np.float32)
        coords_path = self.root / "coords.npy"
        rf_path = self.root / "rf.npy"
        u_path = self.root / "u.npy"
        np.save(coords_path, coords)
        np.save(rf_path, rf)
        np.save(u_path, u)
        manager_path = self.root / "mesh_manager_pretrain.pth"
        cache_path = self.root / "assignments.npz"

        base_model = {
            "model_name": "inr_moe_mesh",
            "in_dim": 4,
            "out_dim": 1,
            "decoder_hidden_dim": 8,
            "decoder_n_hidden_layers": 1,
            "decoder_input_encoding": "learned_8_1_sine_siren_none",
            "decoder_nl": "sine",
            "decoder_init_type": "siren",
            "n_experts": 2,
            "outermost_linear": True,
            "input_encoding": "none",
            "decoder_freqs": 30.0,
            "decoder_trainable_freqs": False,
            "top_k": 1,
            "manager_hidden_dim": 8,
            "manager_n_hidden_layers": 1,
            "manager_input_encoding": "learned_8_1_sine_siren_none",
            "manager_nl": "sine",
            "manager_init": "siren",
            "manager_type": "standard",
            "experts_bias_std": 0.1,
            "experts_bias_weight": 1.0,
            "manager_softmax_temperature": 1.0,
            "manager_softmax_temp_trainable": False,
            "manager_q_activation": "softmax",
            "manager_clamp_q": 0.0,
            "manager_conditioning": "cat",
            "manager_pt_path": str(manager_path),
            "shared_encoder": False,
        }
        base_data = {
            "dataset_name": "linkage_p",
            "association": "point",
            "source_path": str(coords_path),
            "target": "point_RF",
            "targets": {
                "point_RF": str(rf_path),
                "point_U": str(u_path),
            },
            "target_stats_path": str(self.root / "mesh_stats_{target}.npz"),
            "stats_key": "{target}",
            "normalize_inputs": False,
            "normalize_targets": False,
        }
        base_training = {
            "n_points": 4,
            "lr": 1.0e-3,
            "lr_gamma": 0.99,
            "lr_scheduler": "ExponentialLR",
            "num_epochs": 1,
            "batch_size": 1,
            "num_workers": 0,
            "grad_clip_norm": 1.0,
            "save_every": 1,
            "log_every": 1,
            "pretrain_assignment": {
                "method": "coord_kmeans",
                "fit_samples": 4,
                "cache_path": str(cache_path),
                "normalize_features": False,
                "random_seed": 0,
                "chunk_size": 4,
            },
        }
        pretrain_config = {
            "seed": 0,
            "wandb_project": "test_mesh",
            "experiment": "mesh-pretrain",
            "exp_id": "mesh-pretrain",
            "experiment_root": str(self.root / "runs"),
            "MODEL": dict(base_model, load_pt_manager=False),
            "LOSS": {
                "scale_by_q_grad": False,
                "loss_type": "1000segmentation",
                "segmentation_type": "ce",
                "sample_bias_correction": False,
                "entropy_metric": "kl",
            },
            "DATA": base_data,
            "TRAINING": dict(
                base_training,
                segmentation_mode=True,
                stages=[{"end_iteration_frac": 1.0, "params": "all", "loss_type": "1000segmentation"}],
            ),
        }
        main_config = {
            "seed": 0,
            "wandb_project": "test_mesh",
            "experiment": "mesh-main",
            "exp_id": "mesh-main",
            "experiment_root": str(self.root / "runs"),
            "MODEL": dict(base_model, load_pt_manager=True),
            "LOSS": {
                "scale_by_q_grad": False,
                "loss_type": "1000valrecon",
                "segmentation_type": "ce",
                "sample_bias_correction": False,
                "entropy_metric": "kl",
            },
            "DATA": base_data,
            "TRAINING": dict(
                base_training,
                segmentation_mode=False,
                stages=[{"end_iteration_frac": 1.0, "params": "all", "loss_type": "1000valrecon"}],
            ),
        }

        pretrain_path = self._write_yaml(self.root / "mesh_pretrain.yaml", pretrain_config)
        main_path = self._write_yaml(self.root / "mesh_main.yaml", main_config)
        pretrain_result = run_train(pretrain_path, target="point_RF")
        self.assertTrue(manager_path.exists())
        self.assertTrue(Path(pretrain_result["validate_checkpoint_path"]).exists())
        main_result = run_train(main_path, target="point_RF")
        self.assertTrue(Path(main_result["checkpoint_path"]).exists())
        self.assertTrue(Path(main_result["validate_config_path"]).exists())
        self.assertTrue(Path(main_result["validate_checkpoint_path"]).exists())


if __name__ == "__main__":
    unittest.main()
