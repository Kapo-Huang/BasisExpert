import tempfile
import unittest
import importlib.util
from pathlib import Path

import numpy as np
import yaml

from var_expert_inr.cli import run_train
from var_expert_inr.evaluation.service import evaluate_run


class EvaluationServiceTestCase(unittest.TestCase):
    def _trained_run(self, root: Path):
        target_path = root / "target.npy"
        np.save(target_path, np.linspace(-1, 1, 8, dtype=np.float32).reshape(2, 1, 2, 2))
        config = {
            "exp_id": "evaluation-smoke",
            "experiment_root": str(root / "runs"),
            "data": {"kind": "volume", "dataset_name": "tiny", "target_path": str(target_path)},
            "model": {"name": "siren", "in_features": 4, "hidden_features": 4, "hidden_layers": 1},
            "training": {
                "epochs": 1, "batch_size": 4, "pred_batch_size": 4, "num_workers": 0,
                "lr": 1e-3, "device": "cpu", "val_split": 0.0, "log_every": 1, "save_every": 1,
            },
            "evaluation": {"batch_size": 4, "save_predictions": False},
        }
        config_path = root / "config.yaml"
        config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        result = run_train(config_path)
        return Path(result["checkpoint_path"]).parent.parent, target_path

    def test_selected_psnr_writes_standard_reports(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir, _ = self._trained_run(Path(tmpdir))
            result = evaluate_run(run_dir, metrics="psnr", timesteps="1")
            self.assertTrue(Path(result["manifest_path"]).is_file())
            self.assertTrue(Path(result["metrics_path"]).is_file())
            self.assertEqual([row["timestep"] for row in result["metrics"]["per_timestep"]], [1])
            self.assertIn("psnr", result["metrics"]["aggregate"])

    def test_quality_cache_uses_source_and_ground_truth_fingerprints(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir, target_path = self._trained_run(Path(tmpdir))
            first = evaluate_run(run_dir, metrics="psnr", timesteps="0")
            second = evaluate_run(run_dir, metrics="psnr", timesteps="0")
            self.assertEqual(first["output_dir"], second["output_dir"])
            self.assertTrue(second["cache_hit"])
            target = np.array(np.load(target_path), copy=True)
            np.save(target_path, target * np.float32(0.9))
            third = evaluate_run(run_dir, metrics="psnr", timesteps="0")
            self.assertNotEqual(first["output_dir"], third["output_dir"])

    def test_performance_only_succeeds_after_ground_truth_removed(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir, target_path = self._trained_run(Path(tmpdir))
            target_path.unlink()
            result = evaluate_run(run_dir, metrics="decode_time,memory", timesteps="0")
            self.assertFalse(result["metrics"]["aggregate"])
            self.assertIn("total_decode_seconds", result["metrics"]["performance"])
            self.assertNotIn("psnr", result["metrics"]["performance"])

    def test_psnr_fails_before_decode_when_ground_truth_removed(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir, target_path = self._trained_run(Path(tmpdir))
            target_path.unlink()
            with self.assertRaises(FileNotFoundError):
                evaluate_run(run_dir, metrics="psnr", timesteps="0")

    @unittest.skipIf(importlib.util.find_spec("volume_vis") is None, "VolumeVis is not installed")
    def test_prediction_only_volume_render_does_not_require_ground_truth(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            run_dir, target_path = self._trained_run(root)
            profile_path = root / "render.yaml"
            profile_path.write_text(
                yaml.safe_dump(
                    {
                        "kind": "volume",
                        "preset_namespace": "ionization",
                        "target_presets": {"target": "GT"},
                        "options": {"width": 64, "height": 64, "settle_frames": 1, "sample_rate": 1.0},
                    },
                    sort_keys=False,
                ),
                encoding="utf-8",
            )
            target_path.unlink()
            result = evaluate_run(
                run_dir,
                timesteps="0",
                render=True,
                render_profile=profile_path,
            )
            row = result["metrics"]["per_timestep"][0]
            self.assertTrue(Path(row["pred_render_path"]).is_file())
            self.assertNotIn("gt_render_path", row)
            self.assertFalse(result["metrics"]["performance"])

    @unittest.skipUnless(
        importlib.util.find_spec("volume_vis") is not None and importlib.util.find_spec("skimage") is not None,
        "VolumeVis and scikit-image are required",
    )
    def test_ssim_renders_matching_gt_and_prediction_frames(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            run_dir, _ = self._trained_run(root)
            profile_path = root / "render.yaml"
            profile_path.write_text(
                yaml.safe_dump(
                    {
                        "kind": "volume",
                        "preset_namespace": "ionization",
                        "target_presets": {"target": "GT"},
                        "options": {"width": 64, "height": 64, "settle_frames": 1, "sample_rate": 1.0},
                    },
                    sort_keys=False,
                ),
                encoding="utf-8",
            )
            result = evaluate_run(
                run_dir, metrics="ssim", timesteps="0", render_profile=profile_path
            )
            row = result["metrics"]["per_timestep"][0]
            self.assertIn("ssim", row)
            self.assertTrue(Path(row["gt_render_path"]).is_file())
            self.assertIn("ssim", result["metrics"]["aggregate"])


if __name__ == "__main__":
    unittest.main()
