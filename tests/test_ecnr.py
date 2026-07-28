from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
import yaml

from var_expert_inr.cli import run_evaluate as unified_run_evaluate
from var_expert_inr.cli import run_predict as unified_run_predict
from var_expert_inr.cli import run_train as unified_run_train
from var_expert_inr.ecnr.artifact import load_artifact
from var_expert_inr.ecnr.blocks import (
    attach_clustering,
    build_training_targets,
    prepare_scale_blocks,
    reconstruct_from_normalized_blocks,
)
from var_expert_inr.ecnr.clustering import balanced_kmeans
from var_expert_inr.ecnr.cnn import BoundaryCNN, forward_tiled
from var_expert_inr.ecnr.config import load_config
from var_expert_inr.ecnr.huffman import decode as huffman_decode
from var_expert_inr.ecnr.huffman import encode as huffman_encode
from var_expert_inr.ecnr.model import PackedSiren, local_coordinate_grid
from var_expert_inr.ecnr.pruning import (
    BIAS_CANDIDATES,
    WEIGHT_CANDIDATES,
    apply_cumulative_pruning,
    family_sparsity,
    initial_pruning_masks,
)
from var_expert_inr.ecnr.pyramid import (
    build_three_scale_pyramid,
    retained_time_positions,
    upsample_to_scale,
)


class ECNRTestCase(unittest.TestCase):
    def test_balanced_kmeans_is_deterministic_and_assigns_ordered_slots(self):
        rng = np.random.default_rng(4)
        blocks = rng.normal(size=(17, 8)).astype(np.float32)
        first = balanced_kmeans(blocks, target_blocks_per_mlp=4, seed=42)
        second = balanced_kmeans(blocks, target_blocks_per_mlp=4, seed=42)
        np.testing.assert_array_equal(first.labels, second.labels)
        np.testing.assert_array_equal(first.slots, second.slots)
        self.assertEqual(first.centroids.dtype, np.float32)
        self.assertLessEqual(int(first.cluster_sizes.max() - first.cluster_sizes.min()), 1)
        for cluster in range(len(first.cluster_sizes)):
            positions = np.flatnonzero(first.labels == cluster)
            np.testing.assert_array_equal(first.slots[positions], np.arange(len(positions)))

    def test_siren_uses_three_distinct_formal_initialization_ranges(self):
        torch.manual_seed(2)
        model = PackedSiren(mlp_count=3, max_slots=2)
        first_bound = 1.0 / 11.0
        sine_bound = math.sqrt(6.0 / 24.0) / 30.0
        for parameter in (model.layers[0].weight, model.layers[0].bias):
            self.assertLessEqual(float(parameter.abs().max()), first_bound)
        for layer in model.layers[1:]:
            self.assertLessEqual(float(layer.weight.abs().max()), sine_bound)
            self.assertLessEqual(float(layer.bias.abs().max()), sine_bound)
        coords = local_coordinate_grid((2, 2, 2))
        result = model(coords, torch.zeros(8, dtype=torch.long))
        self.assertEqual(tuple(result.shape), (3, 8))

    def test_pruning_candidates_and_cumulative_targets_are_exact(self):
        model = PackedSiren(mlp_count=2, max_slots=2)
        masks = initial_pruning_masks(model)
        self.assertEqual(set(masks), set(WEIGHT_CANDIDATES + BIAS_CANDIDATES))
        apply_cumulative_pruning(
            model,
            masks,
            mlp_losses=np.array([0.1, 0.2]),
            target_sparsity=0.30,
        )
        self.assertAlmostEqual(family_sparsity(masks, WEIGHT_CANDIDATES), 0.30, places=3)
        first_weight_zeros = sum(int((~masks[name]).sum()) for name in WEIGHT_CANDIDATES)
        apply_cumulative_pruning(
            model,
            masks,
            mlp_losses=np.array([0.1, 0.2]),
            target_sparsity=0.40,
        )
        total = sum(masks[name].numel() for name in WEIGHT_CANDIDATES)
        self.assertEqual(sum(int((~masks[name]).sum()) for name in WEIGHT_CANDIDATES), math.floor(0.4 * total))
        self.assertGreater(sum(int((~masks[name]).sum()) for name in WEIGHT_CANDIDATES), first_weight_zeros)
        self.assertNotIn("layers.3.weight", masks)
        self.assertNotIn("layers.0.bias", masks)
        self.assertNotIn("layers.3.bias", masks)
        self.assertNotIn("latent", masks)

    def test_pyramid_blocks_and_reconstruction_roundtrip(self):
        values = np.linspace(-1, 1, 5 * 5 * 7 * 9, dtype=np.float32).reshape(5, 5, 7, 9)
        self.assertEqual(retained_time_positions(5).tolist(), [0, 2, 4])
        scales = build_three_scale_pyramid(values)
        self.assertEqual([scale.values.shape[0] for scale in scales], [5, 3, 2])
        coarse = scales[2]
        restored = upsample_to_scale(
            coarse.values,
            coarse.time_indices,
            fine_shape_tzyx=scales[1].values.shape,
            fine_time_indices=scales[1].time_indices,
        )
        self.assertEqual(restored.shape, scales[1].values.shape)

        blocks = prepare_scale_blocks(
            values[:2, :3, :4, :5],
            block_shape_xyz=(2, 2, 2),
            residual_threshold=0,
            keep_all=True,
        )
        clustering = balanced_kmeans(blocks.normalized_blocks, target_blocks_per_mlp=4, seed=42)
        attach_clustering(blocks, clustering)
        targets = build_training_targets(blocks)
        decoded = np.empty_like(blocks.normalized_blocks)
        for block_index in range(blocks.effective_count):
            decoded[block_index] = targets[
                blocks.block_to_mlp[block_index],
                blocks.block_to_slot[block_index],
            ]
        reconstructed = reconstruct_from_normalized_blocks(blocks, decoded)
        np.testing.assert_allclose(reconstructed, values[:2, :3, :4, :5], atol=1.0e-6)

    def test_huffman_roundtrip_supports_nine_bit_symbols(self):
        values = np.array([511, 0, 7, 511, 255, 7, 7, 1], dtype=np.uint16).reshape(2, 4)
        stream = huffman_encode(values)
        np.testing.assert_array_equal(huffman_decode(stream), values)

    def test_tiled_cnn_matches_direct_full_volume(self):
        torch.manual_seed(5)
        model = BoundaryCNN(hidden_channels=32).eval()
        frame = torch.randn(7, 8, 9)
        with torch.no_grad():
            direct = model(frame[None, None])[0, 0]
            tiled = forward_tiled(
                model,
                frame,
                core_shape_zyx=(3, 4, 5),
                halo=5,
                device=torch.device("cpu"),
            )
        torch.testing.assert_close(tiled, direct, atol=2.0e-6, rtol=2.0e-5)

    def test_formal_config_has_equal_primary_budget(self):
        root = Path(__file__).resolve().parents[1]
        cfg = load_config(root / "configs" / "ECNR" / "ionization__GT.yaml")
        training = cfg["training"]
        self.assertEqual(
            3 * training["epochs_per_scale"] * training["batch_size"] * training["batches_per_epoch_budget"],
            14_400_000_000,
        )


class ECNREndToEndTestCase(unittest.TestCase):
    def test_tiny_three_scale_train_artifact_and_evaluate(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            z, y, x = np.meshgrid(
                np.linspace(-1, 1, 4),
                np.linspace(-1, 1, 4),
                np.linspace(-1, 1, 4),
                indexing="ij",
            )
            values = np.stack(
                [(x + y + z) / 3, (x - y + z) / 3, (-x + y + z) / 3]
            ).astype(np.float32)
            data_path = root / "data.npy"
            np.save(data_path, values)
            payload = {
                "exp_id": "ecnr-smoke",
                "experiment_root": str(root / "runs"),
                "data": {
                    "kind": "volume",
                    "target": "GT",
                    "target_path": str(data_path),
                    "volume_shape": {"T": 3, "Z": 4, "Y": 4, "X": 4},
                },
                "model": {"name": "ecnr", "block_shape_xyz": [2, 2, 2]},
                "training": {
                    "epochs_per_scale": 1,
                    "batch_size": 8,
                    "batches_per_epoch_budget": 1,
                    "primary_sample_budget": 24,
                    "pruning_epochs": [],
                    "pruning_sparsities": [],
                    "quantization_finetune_epochs": 1,
                    "quantization_finetune_batches_per_epoch": 1,
                    "log_every": 1,
                    "device": "cpu",
                },
                "cnn": {
                    "epochs": 0,
                    "tile_core_shape_zyx": [4, 4, 4],
                },
                "evaluation": {
                    "batch_size": 8,
                    "save_predictions": True,
                    "run_after_training": False,
                    "default_model": "artifact",
                },
            }
            config_path = root / "config.yaml"
            config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
            result = unified_run_train(config_path)
            self.assertTrue(Path(result["checkpoint_path"]).exists())
            self.assertTrue(Path(result["artifact_path"]).exists())
            compact = load_artifact(result["artifact_path"])
            for scale in compact["scales"]:
                if scale["empty"]:
                    continue
                for item in scale["model"]["quantization"]["parameters"].values():
                    self.assertEqual(item["labels"].dtype, np.dtype(np.uint8))
            for item in compact["cnn"]["parameters"].values():
                self.assertEqual(item["labels"].dtype, np.dtype(np.uint16))
            scale_checkpoint = Path(result["checkpoint_path"]).parent / "scale_2_complete.pth"
            self.assertTrue(scale_checkpoint.exists())

            resumed = unified_run_train(config_path, resume_path=scale_checkpoint)
            self.assertTrue(Path(resumed["checkpoint_path"]).exists())
            resumed_cost = json.loads(
                Path(resumed["training_cost_path"]).read_text(encoding="utf-8")
            )
            self.assertEqual(resumed_cost["primary_logical_sample_budget"], 24)
            self.assertEqual(resumed_cost["primary_logical_samples_executed"], 24)
            self.assertEqual(
                [item["level"] for item in resumed_cost["scales"]],
                [2, 1, 0],
            )

            checkpoint_prediction = unified_run_predict(
                config_path,
                checkpoint_path=result["checkpoint_path"],
            )
            self.assertEqual(
                np.load(checkpoint_prediction["prediction_path"], mmap_mode="r").shape,
                values.shape,
            )
            evaluated = unified_run_evaluate(
                config_path,
                artifact_path=result["artifact_path"],
            )
            prediction = np.load(evaluated["prediction_path"])
            self.assertEqual(prediction.shape, values.shape)
            self.assertIn("psnr", evaluated["metrics"]["aggregate"])


if __name__ == "__main__":
    unittest.main()
