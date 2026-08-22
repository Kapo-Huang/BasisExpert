from __future__ import annotations

import csv
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "tools" / "organize_runs_summary.py"
SPEC = importlib.util.spec_from_file_location("organize_runs_summary", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
organizer = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = organizer
SPEC.loader.exec_module(organizer)


def _write(path: Path, content: str | bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(content, bytes):
        path.write_bytes(content)
    else:
        path.write_text(content, encoding="utf-8")


def _make_run(
    repo: Path,
    experiment: str,
    run_id: str,
    model: str,
    log_text: str,
    *,
    nested_prefix: str = "",
) -> Path:
    root = repo / "runs"
    if nested_prefix:
        root /= nested_prefix
    root = root / experiment / run_id
    config = (
        f"experiment: fixture\n"
        f"exp_id: {experiment}\n"
        f"model:\n  name: {model}\n"
        f"training:\n  epochs: 600\n"
    )
    _write(root / "configs" / "config.yaml", config)
    _write(root / "logs" / f"run_{run_id}.log", log_text)
    _write(root / "checkpoints" / "model.pth", b"checkpoint-data")
    _write(root / "metrics" / "metrics.json", '{"ok": true}\n')
    return root


def _snapshot(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    }


class CompletionAndClassificationTests(unittest.TestCase):
    def test_terminal_and_native_completion(self) -> None:
        self.assertEqual(
            organizer.completion_from_log("Epoch 600/600 train=1"),
            (True, "terminal-count:600/600"),
        )
        self.assertEqual(
            organizer.completion_from_log("Epoch 50/50 train=1"),
            (True, "terminal-count:50/50"),
        )
        self.assertEqual(
            organizer.completion_from_log("Epoch 300/300 train=1"),
            (True, "terminal-count:300/300"),
        )
        self.assertEqual(
            organizer.completion_from_log("Epoch 600/600\nEpoch 599/600"),
            (False, "incomplete-count:599/600"),
        )
        self.assertEqual(
            organizer.completion_from_log("Training loop finished: steps=30000"),
            (True, "native:training-loop-finished"),
        )
        self.assertEqual(organizer.completion_from_log(""), (False, "empty-log"))

    def test_psnr_threshold_and_deprecated_exclusivity(self) -> None:
        finite, state = organizer.final_psnr_from_log(
            "PSNR epoch 50/50: aggregate=20.00 GT=20"
        )
        self.assertEqual((finite, state), (20.0, "finite"))
        nonfinite, state = organizer.final_psnr_from_log(
            "Exploration PSNR progress=50/50 aggregate=NaN"
        )
        self.assertTrue(nonfinite is not None and not organizer.math.isfinite(nonfinite))
        self.assertEqual(state, "nonfinite")

        deprecated = organizer.Record(
            record_id="deprecated",
            source_kind="batch-exploration",
            source_path=Path("batch"),
            log_path=Path("batch/log.log"),
            model="DC-INR",
            experiment="fixture",
            run_id="run",
            experiment_group="Size",
            complete=True,
            completion_reason="batch-status:ok",
            final_psnr=1.0,
            psnr_state="finite",
        )
        self.assertEqual(organizer.classify_record(deprecated), ["deprecate"])

        successful_exploration = organizer.Record(
            record_id="exploration",
            source_kind="batch-exploration",
            source_path=Path("batch"),
            log_path=Path("batch/log.log"),
            model="SIREN",
            experiment="fixture",
            run_id="run",
            experiment_group="Size",
            complete=True,
            completion_reason="batch-status:ok",
            final_psnr=30.0,
            psnr_state="finite",
        )
        self.assertEqual(organizer.classify_record(successful_exploration), ["exploration"])


class EndToEndTests(unittest.TestCase):
    def test_apply_builds_verified_tree_and_filtered_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            repo = Path(temp_dir)
            (repo / "runs").mkdir()
            (repo / "batch_logs").mkdir()

            _make_run(
                repo,
                "siren-ionization-GT",
                "20260101_000000_1",
                "siren",
                "Epoch 600/600 train=1\nPSNR epoch 600/600: aggregate=20.00\n",
            )
            _make_run(
                repo,
                "coordnet-ionization-size163-GT",
                "20260101_000001_1",
                "coordnet",
                "Epoch 50/50 train=1\nPSNR epoch 50/50: aggregate=19.99\n",
            )
            _make_run(
                repo,
                "fa-tr-inr-ionization-GT",
                "20260101_000002_1",
                "fa_tr_inr",
                "Epoch 600/600 train=1\nPSNR epoch 600/600: aggregate=5.0\n",
            )
            _make_run(
                repo,
                "mvnet-bathymetry",
                "20260101_000003_1",
                "mvnet",
                "Epoch 300/300 train=1\n",
            )
            _make_run(
                repo,
                "siren-incomplete",
                "20260101_000004_1",
                "siren",
                "Epoch 599/600 train=1\n",
            )
            _make_run(
                repo,
                "siren-empty",
                "20260101_000005_1",
                "siren",
                "",
            )
            _make_run(
                repo,
                "var-expert-backup",
                "20260101_000006_1",
                "var_expert",
                "Epoch 600/600 train=1\nPSNR epoch 600/600: aggregate=30\n",
                nested_prefix="backup",
            )
            _make_run(
                repo,
                "explore-v3-dc-inr-ionization-size163-GT",
                "20260101_000007_1",
                "dcinr",
                "Epoch 50/50 train=1\nPSNR epoch 50/50: aggregate=30\n",
                nested_prefix="exploration_v3",
            )
            manager = repo / "runs" / "neural_expert" / "neural-expert-managerpretrain"
            _write(manager / "config.yaml", "exp_id: neural-expert-managerpretrain\n")
            _write(manager / "out.log", "Training loop finished: steps=30000\n")
            _write(manager / "trained_models" / "final.pth", b"manager-checkpoint")
            _write(repo / "runs" / "model_size_catalog.csv", "model,size\nSIREN,1\n")
            _write(repo / "runs" / "visualizations" / "ignored.png", b"png")
            (repo / "runs" / "exploration_v3").mkdir(exist_ok=True)

            batch = repo / "batch_logs" / "exploration_v2" / "20260102_000000"
            dc_config = "configs/sensitivity/routing_and_depth/DC-INR/Size163/test/ionization__GT.yaml"
            siren_config = "configs/sensitivity/routing_and_depth/SIREN/Size163/test/ionization__GT.yaml"
            failed_config = "configs/sensitivity/routing_and_depth/SIREN/Size163/test/ionization__PD.yaml"
            dc_log = "configs__sensitivity__routing_and_depth__DC-INR__Size163__test__ionization__GT.log"
            siren_log = "configs__sensitivity__routing_and_depth__SIREN__Size163__test__ionization__GT.log"
            failed_log = "configs__sensitivity__routing_and_depth__SIREN__Size163__test__ionization__PD.log"
            _write(batch / "logs" / dc_log, "Exploration PSNR progress=50/50 aggregate=5.0\n")
            _write(batch / "logs" / siren_log, "Exploration PSNR progress=50/50 aggregate=NaN\n")
            _write(batch / "logs" / failed_log, "Exploration PSNR progress=45/50 aggregate=30\n")
            status_header = "config\tstatus\texit_code\tlog\n"
            status = (
                status_header
                + f"{dc_config}\tok\t0\t/work/{dc_log}\n"
                + f"{siren_config}\tok\t0\t/work/{siren_log}\n"
                + f"{failed_config}\tfailed\t1\t/work/{failed_log}\n"
            )
            _write(batch / "status.tsv", status)
            summary_header = "config\tfinal_psnr\thas_nonfinite\ttraining_status\n"
            summary = (
                summary_header
                + f"{dc_config}\t5\tfalse\tok\n"
                + f"{siren_config}\t\ttrue\tok\n"
            )
            _write(batch / "exploration_summary.tsv", summary)

            before_runs = _snapshot(repo / "runs")
            before_batch = _snapshot(repo / "batch_logs")
            output = repo / "runs_summary"
            inventory = organizer.organize(repo, output=output, apply=True, verify_hashes=True)

            self.assertTrue(output.is_dir())
            self.assertFalse((repo / "runs_summary.staging").exists())
            self.assertEqual(before_runs, _snapshot(repo / "runs"))
            self.assertEqual(before_batch, _snapshot(repo / "batch_logs"))
            self.assertTrue((output / "raw" / "model_size_catalog.csv").is_file())
            self.assertFalse((output / "raw" / "visualizations").exists())

            records = {record.record_id: record for record in inventory.records}
            fa = next(record for record in records.values() if record.model == "FA-TR-INR")
            self.assertEqual(fa.labels, ["deprecate"])
            dc = next(
                record
                for record in records.values()
                if record.source_kind == "batch-exploration" and record.model == "DC-INR"
            )
            self.assertEqual(dc.labels, ["deprecate"])
            runs_dc = next(
                record
                for record in records.values()
                if record.source_kind == "runs"
                and record.model == "DC-INR"
                and "exploration_v3" in record.source_path.parts
            )
            self.assertEqual(runs_dc.labels, ["deprecate"])
            siren_exploration = next(
                record
                for record in records.values()
                if record.source_kind == "batch-exploration" and record.model == "SIREN"
            )
            self.assertEqual(siren_exploration.labels, ["fail", "exploration"])

            success_main = output / "success" / "Main" / "SIREN"
            fail_size = output / "fail" / "Size" / "CoordNet"
            self.assertTrue(success_main.is_dir())
            self.assertTrue(fail_size.is_dir())
            self.assertTrue((output / "deprecate" / "FA-TR-INR").is_dir())
            self.assertTrue((output / "deprecate" / "DC-INR").is_dir())

            filtered = (
                output
                / "exploration"
                / "exploration_v2"
                / "20260102_000000"
                / "status.tsv"
            )
            with filtered.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle, delimiter="\t"))
            self.assertEqual([row["config"] for row in rows], [siren_config])

            excluded_reasons = {item.reason for item in inventory.excluded}
            self.assertIn("incomplete-count:599/600", excluded_reasons)
            self.assertIn("empty-log", excluded_reasons)
            self.assertIn("batch-status:failed", excluded_reasons)
            self.assertIn("auxiliary-no-training-log", excluded_reasons)
            self.assertFalse((output / "raw" / "backup").exists())
            with (output / "classification_manifest.tsv").open(
                "r", encoding="utf-8", newline=""
            ) as handle:
                classification_text = handle.read()
            with (output / "excluded_manifest.tsv").open(
                "r", encoding="utf-8", newline=""
            ) as handle:
                excluded_text = handle.read()
            self.assertNotIn("runs/backup/", classification_text)
            self.assertNotIn("runs/backup/", excluded_text)
            self.assertTrue((output / "classification_manifest.tsv").is_file())
            self.assertTrue((output / "excluded_manifest.tsv").is_file())
            self.assertTrue((output / "README.md").is_file())


if __name__ == "__main__":
    unittest.main()
