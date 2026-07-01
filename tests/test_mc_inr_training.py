import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import yaml

from var_expert_inr.data import build_dataset
from var_expert_inr.mc_inr.checkpoint import load_mc_checkpoint
from var_expert_inr.mc_inr.cli import run_evaluate, run_predict, run_train
from var_expert_inr.mc_inr.config import load_config
from var_expert_inr.mc_inr.data import target_layout_from_dataset
from var_expert_inr.mc_inr.runner import (
    _load_cached_assignments,
    _save_cached_assignments,
    _stage_epoch_batches,
)


class MCINRTrainingTestCase(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tmpdir.name)

    def tearDown(self):
        self.tmpdir.cleanup()

    def _write_yaml(self, path: Path, payload) -> Path:
        path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
        return path

    def _run_dir_from_checkpoint(self, checkpoint_path: str | Path) -> Path:
        return Path(checkpoint_path).resolve().parent.parent

    def _checkpoint_dir_from_checkpoint(self, checkpoint_path: str | Path) -> Path:
        return Path(checkpoint_path).resolve().parent

    def _log_text_from_checkpoint(self, checkpoint_path: str | Path) -> str:
        run_dir = self._run_dir_from_checkpoint(checkpoint_path)
        log_path = next((run_dir / "logs").glob("run_*.log"))
        return log_path.read_text(encoding="utf-8")

    def _write_node_data(self, *, variant: str = "smooth"):
        coords = np.array(
            [
                [-1.0, -0.9, -0.8, -1.0],
                [-0.8, -0.6, -0.4, -0.8],
                [-0.6, -0.4, -0.2, -0.6],
                [-0.4, -0.2, 0.0, -0.4],
                [-0.2, 0.0, 0.2, -0.2],
                [0.0, 0.2, 0.4, 0.0],
                [0.2, 0.4, 0.6, 0.2],
                [0.4, 0.6, 0.8, 0.4],
                [0.6, 0.3, 0.0, 0.6],
                [0.8, 0.0, -0.3, 0.8],
                [0.9, -0.2, -0.5, 0.9],
                [1.0, -0.4, -0.7, 1.0],
            ],
            dtype=np.float32,
        )
        if variant == "split":
            target_a = np.array(
                [[-0.8], [0.2], [-0.5], [0.7], [-0.2], [0.5], [-0.7], [0.9], [-0.1], [0.8], [-0.6], [0.4]],
                dtype=np.float32,
            )
            target_b = np.stack(
                [
                    np.array([0.9, -0.9, 0.7, -0.7, 0.5, -0.5, 0.3, -0.3, 0.1, -0.1, 0.8, -0.8], dtype=np.float32),
                    np.array([-0.4, 0.6, -0.8, 0.2, -0.1, 0.9, -0.7, 0.4, -0.2, 0.5, -0.9, 0.3], dtype=np.float32),
                ],
                axis=1,
            )
        else:
            target_a = (0.35 * coords[:, :1]) + (0.15 * coords[:, 3:4])
            target_b = np.concatenate(
                [
                    0.5 * coords[:, 1:2],
                    0.25 * (coords[:, 2:3] - coords[:, 1:2] + coords[:, 3:4]),
                ],
                axis=1,
            )
        coords_path = self.root / f"coords_{variant}.npy"
        a_path = self.root / f"a_{variant}.npy"
        b_path = self.root / f"b_{variant}.npy"
        np.save(coords_path, coords)
        np.save(a_path, target_a.astype(np.float32))
        np.save(b_path, target_b.astype(np.float32))
        return coords, coords_path, a_path, b_path

    def _node_config(self, *, variant: str = "smooth", exp_id: str = "mc-node", **training_overrides):
        coords, coords_path, a_path, b_path = self._write_node_data(variant=variant)
        training = {
            "epochs": 3,
            "batch_size": 4,
            "pred_batch_size": 4,
            "num_workers": 0,
            "lr": 1.0e-3,
            "weight_decay": 0.0,
            "loss_type": "mse",
            "log_every": 1,
            "save_every": 1,
            "seed": 7,
            "device": "cpu",
            "initial_k": 2,
            "cluster_init_method": "coord_kmeans",
            "assignments_cache_path": str(self.root / f"{exp_id}_assignments.npy"),
            "meta_iterations": 3,
            "meta_inner_steps": 2,
            "meta_inner_batch_size": 3,
            "meta_inner_lr": 1.0e-3,
            "meta_batch_clusters": 2,
            "meta_support_max_rows": 6,
            "meta_outer_lr": 1.0e-3,
            "convergence_patience": 10,
            "convergence_delta": 0.0,
            "finetune_epochs": 1,
            "finetune_lr": 1.0e-3,
            "finetune_sampling_ratio": 0.5,
            "recluster_after_finetune": False,
            "split_threshold": 5.0e-4,
            "min_split_points": 2,
            "max_recluster_rounds": 1,
            "cluster_aware_batches": True,
            "scheduler": {
                "enabled": False,
                "step_size": 0,
                "gamma": 1.0,
            },
        }
        training.update(training_overrides)
        config = {
            "experiment": exp_id,
            "exp_id": exp_id,
            "experiment_root": str(self.root / "runs"),
            "data": {
                "kind": "node",
                "coords_path": str(coords_path),
                "targets": {"b": str(b_path), "a": str(a_path)},
            },
            "model": {
                "name": "mc_inr",
                "hidden_features": 8,
                "gfe_layers": 2,
                "lfe_layers": 2,
            },
            "training": training,
            "evaluation": {"batch_size": 4},
            "log": {
                "timing": {
                    "enabled": False,
                    "epoch_breakdown": False,
                    "step_window": False,
                    "step_window_every_steps": 100,
                    "cuda_sync": False,
                },
            },
        }
        return coords, self._write_yaml(self.root / f"{exp_id}.yaml", config)

    def _volume_config(
        self,
        *,
        exp_id: str = "mc-volume",
        time_steps: int = 2,
        batch_size: int = 4,
        finetune_sampling_ratio: float = 1.0,
        cluster_aware_batches: bool = True,
    ):
        a = np.linspace(-1.0, 1.0, time_steps * 4, dtype=np.float32).reshape(time_steps, 1, 2, 2)
        b = np.linspace(-1.0, 1.0, time_steps * 8, dtype=np.float32).reshape(time_steps, 1, 2, 2, 2)
        a_path = self.root / f"{exp_id}_target_a.npy"
        b_path = self.root / f"{exp_id}_target_b.npy"
        np.save(a_path, a)
        np.save(b_path, b)
        config = {
            "experiment": exp_id,
            "exp_id": exp_id,
            "experiment_root": str(self.root / "runs"),
            "data": {
                "kind": "volume",
                "targets": {"b": str(b_path), "a": str(a_path)},
                "volume_shape": {"X": 2, "Y": 2, "Z": 1, "T": time_steps},
            },
            "model": {
                "name": "mc_inr",
                "hidden_features": 8,
                "gfe_layers": 2,
                "lfe_layers": 2,
            },
            "training": {
                "epochs": 2,
                "batch_size": batch_size,
                "pred_batch_size": batch_size,
                "num_workers": 0,
                "lr": 1.0e-3,
                "weight_decay": 0.0,
                "loss_type": "mse",
                "log_every": 1,
                "save_every": 1,
                "seed": 9,
                "device": "cpu",
                "initial_k": 2,
                "cluster_init_method": "voxel_clustering",
                "assignments_cache_path": str(self.root / f"{exp_id}_volume_assignments.npy"),
                "meta_iterations": 1,
                "meta_inner_steps": 2,
                "meta_inner_batch_size": 2,
                "meta_inner_lr": 1.0e-3,
                "meta_batch_clusters": 2,
                "meta_support_max_rows": 6,
                "meta_outer_lr": 1.0e-3,
                "convergence_patience": 5,
                "convergence_delta": 0.0,
                "finetune_epochs": 1,
                "finetune_lr": 1.0e-3,
                "finetune_sampling_ratio": finetune_sampling_ratio,
                "cluster_aware_batches": cluster_aware_batches,
                "scheduler": {
                    "enabled": False,
                    "step_size": 0,
                    "gamma": 1.0,
                },
            },
            "evaluation": {"batch_size": batch_size},
            "log": {
                "timing": {
                    "enabled": False,
                    "epoch_breakdown": False,
                    "step_window": False,
                    "step_window_every_steps": 100,
                    "cuda_sync": False,
                },
            },
        }
        return self._write_yaml(self.root / f"{exp_id}.yaml", config)

    def test_node_streaming_train_predict_evaluate_and_meta_resume(self):
        coords, config_path = self._node_config(exp_id="mc-node-main")
        train_result = run_train(config_path)
        self.assertTrue(Path(train_result["checkpoint_path"]).exists())
        self.assertTrue(Path(train_result["metrics_path"]).exists())
        self.assertTrue(Path(train_result["stage_checkpoints"]["meta_init"]).exists())
        self.assertTrue(Path(train_result["stage_checkpoints"]["finetune"]).exists())
        self.assertEqual(train_result["training_summary"]["meta_init"]["last_iteration"], 3)
        self.assertLess(train_result["training_summary"]["finetune_rounds"][0]["sample_counts"][0], len(coords))
        self.assertIn("max_error", train_result["metrics"]["aggregate"])

        run_dir = self._run_dir_from_checkpoint(train_result["checkpoint_path"])
        pred_a = np.load(run_dir / "predictions" / "mc-node-main_a.npy")
        pred_b = np.load(run_dir / "predictions" / "mc-node-main_b.npy")
        self.assertEqual(pred_a.shape, (len(coords), 1))
        self.assertEqual(pred_b.shape, (len(coords), 2))

        log_text = self._log_text_from_checkpoint(train_result["checkpoint_path"])
        self.assertIn("PHASE 1: META INITIALIZATION", log_text)
        self.assertIn("PHASE 2: CLUSTER-SPECIFIC FINE-TUNING", log_text)

        predict_result = run_predict(config_path)
        self.assertTrue(all(Path(path).exists() for path in predict_result["prediction_paths"].values()))
        eval_result = run_evaluate(config_path)
        self.assertTrue(Path(eval_result["metrics_path"]).exists())
        self.assertIn("max_error", eval_result["metrics"]["aggregate"])

        checkpoint_dir = self._checkpoint_dir_from_checkpoint(train_result["checkpoint_path"])
        meta_iter_ckpt = checkpoint_dir / "mc-node-main_meta_init_iteration1.pth"
        self.assertTrue(meta_iter_ckpt.exists())
        resume_meta_result = run_train(config_path, resume_path=meta_iter_ckpt)
        self.assertTrue(Path(resume_meta_result["checkpoint_path"]).exists())
        self.assertEqual(resume_meta_result["training_summary"]["meta_init"]["last_iteration"], 3)

    def test_resume_finetune_continues_epoch(self):
        _, config_path = self._node_config(exp_id="mc-node-ft", finetune_epochs=1)
        first_run = run_train(config_path)
        payload = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
        payload["training"]["finetune_epochs"] = 2
        updated_config = self._write_yaml(self.root / "mc-node-ft-resume.yaml", payload)
        resumed = run_train(updated_config, resume_path=first_run["stage_checkpoints"]["finetune"])
        self.assertEqual(resumed["training_summary"]["finetune_rounds"][0]["last_epoch"], 2)

    def test_cluster_aware_batches_and_legacy_batches(self):
        _, config_path = self._node_config(exp_id="mc-node-batches", finetune_sampling_ratio=1.0)
        cfg = load_config(config_path)
        dataset = build_dataset(cfg.data)
        layout = target_layout_from_dataset(dataset)
        assignments = np.asarray([0, 0, 0, 1, 1, 1, 2, 2, 2, 0, 1, 2], dtype=np.int32)

        _, aware_batches = _stage_epoch_batches(
            dataset,
            layout,
            assignments,
            batch_size=4,
            sampling_ratio=1.0,
            rng=np.random.default_rng(0),
            cluster_aware_batches=True,
        )
        aware_unique_counts = [int(np.unique(batch.cluster_ids.numpy()).size) for batch in aware_batches]
        self.assertTrue(aware_unique_counts)
        self.assertTrue(all(count == 1 for count in aware_unique_counts))

        _, legacy_batches = _stage_epoch_batches(
            dataset,
            layout,
            assignments,
            batch_size=4,
            sampling_ratio=1.0,
            rng=np.random.default_rng(0),
            cluster_aware_batches=False,
        )
        legacy_unique_counts = [int(np.unique(batch.cluster_ids.numpy()).size) for batch in legacy_batches]
        self.assertTrue(any(count > 1 for count in legacy_unique_counts))

    def test_budgeted_batches_return_exact_full_batches_with_replacement(self):
        _, config_path = self._node_config(exp_id="mc-node-budget")
        cfg = load_config(config_path)
        dataset = build_dataset(cfg.data)
        layout = target_layout_from_dataset(dataset)
        assignments = np.asarray([0, 0, 0, 1, 1, 1, 2, 2, 2, 0, 1, 2], dtype=np.int32)
        sample_count, batches = _stage_epoch_batches(
            dataset,
            layout,
            assignments,
            batch_size=4,
            sampling_ratio=1.0,
            rng=np.random.default_rng(0),
            cluster_aware_batches=False,
            sample_count_override=20,
        )
        materialized = list(batches)
        self.assertEqual(sample_count, 20)
        self.assertEqual(len(materialized), 5)
        self.assertTrue(all(batch.coords.shape[0] == 4 for batch in materialized))

    def test_assignment_cache_metadata_mismatch_invalidates_cache(self):
        _, config_path = self._node_config(exp_id="mc-node-cache")
        cfg = load_config(config_path)
        dataset = build_dataset(cfg.data)
        assignments = np.asarray([0, 1] * 6, dtype=np.int32)
        _save_cached_assignments(cfg.training.assignments_cache_path, assignments, dataset=dataset, cfg=cfg.training)
        cached = _load_cached_assignments(cfg.training.assignments_cache_path, dataset=dataset, cfg=cfg.training)
        self.assertIsNotNone(cached)

        meta_path = Path(cfg.training.assignments_cache_path).with_suffix(".json")
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        metadata["dataset_fingerprint"] = "mismatch"
        meta_path.write_text(json.dumps(metadata, ensure_ascii=True, indent=2), encoding="utf-8")
        invalidated = _load_cached_assignments(cfg.training.assignments_cache_path, dataset=dataset, cfg=cfg.training)
        self.assertIsNone(invalidated)

    def test_split_retrain_and_split_resume(self):
        _, config_path = self._node_config(
            variant="split",
            exp_id="mc-node-split",
            finetune_epochs=1,
            finetune_sampling_ratio=1.0,
            recluster_after_finetune=True,
            split_threshold=0.0,
            max_recluster_rounds=1,
            cluster_aware_batches=True,
        )
        split_run = run_train(config_path)
        self.assertTrue(split_run["stage_checkpoints"]["split"])
        split_ckpt = Path(split_run["stage_checkpoints"]["split"][0])
        self.assertTrue(split_ckpt.exists())

        final_payload = load_mc_checkpoint(split_run["checkpoint_path"])
        self.assertGreater(final_payload["centroids"].shape[0], 2)
        split_payload = load_mc_checkpoint(split_ckpt)
        self.assertEqual(split_payload["mc_stage"], "split")
        self.assertEqual(split_payload["split_round"], 1)

        resumed = run_train(config_path, resume_path=split_ckpt)
        self.assertTrue(Path(resumed["checkpoint_path"]).exists())
        self.assertEqual(resumed["training_summary"]["finetune_rounds"][0]["split_round"], 1)

    def test_volume_smoke_and_streaming_prediction(self):
        config_path = self._volume_config()
        train_result = run_train(config_path)
        self.assertEqual(train_result["training_summary"]["meta_init"]["last_iteration"], 1)
        run_dir = self._run_dir_from_checkpoint(train_result["checkpoint_path"])
        pred_a = np.load(run_dir / "predictions" / "mc-volume_a.npy")
        pred_b = np.load(run_dir / "predictions" / "mc-volume_b.npy")
        self.assertEqual(pred_a.shape, (2, 1, 2, 2))
        self.assertEqual(pred_b.shape, (2, 1, 2, 2, 2))
        eval_result = run_evaluate(config_path)
        self.assertTrue(Path(eval_result["metrics_path"]).exists())

    def test_volume_stage_epoch_batches_use_row_level_streaming(self):
        config_path = self._volume_config(
            exp_id="mc-volume-batches",
            time_steps=5,
            batch_size=3,
            finetune_sampling_ratio=0.3,
        )
        cfg = load_config(config_path)
        dataset = build_dataset(cfg.data)
        layout = target_layout_from_dataset(dataset)
        assignments = np.asarray([0, 0, 1, 1], dtype=np.int32)

        expected_sample_count = int(np.ceil((2 * 2 * 1 * 5) * 0.3))

        sample_count, aware_batches = _stage_epoch_batches(
            dataset,
            layout,
            assignments,
            batch_size=3,
            sampling_ratio=0.3,
            rng=np.random.default_rng(0),
            cluster_aware_batches=True,
        )
        aware_batches = list(aware_batches)
        self.assertEqual(sample_count, expected_sample_count)
        aware_batch_sizes = [int(batch.coords.shape[0]) for batch in aware_batches]
        self.assertTrue(aware_batch_sizes)
        self.assertTrue(all(size <= 3 for size in aware_batch_sizes))
        aware_unique_counts = [int(np.unique(batch.cluster_ids.numpy()).size) for batch in aware_batches]
        self.assertTrue(all(count == 1 for count in aware_unique_counts))

        sample_count, mixed_batches = _stage_epoch_batches(
            dataset,
            layout,
            assignments,
            batch_size=3,
            sampling_ratio=0.3,
            rng=np.random.default_rng(1),
            cluster_aware_batches=False,
        )
        mixed_batches = list(mixed_batches)
        self.assertEqual(sample_count, expected_sample_count)
        mixed_batch_sizes = [int(batch.coords.shape[0]) for batch in mixed_batches]
        self.assertTrue(mixed_batch_sizes)
        self.assertTrue(all(size <= 3 for size in mixed_batch_sizes))


if __name__ == "__main__":
    unittest.main()
