from __future__ import annotations

import json
from pathlib import Path

from scripts.evaluation import run_batch as exploration
from var_expert_inr.evaluation.artifacts import ArtifactStore
from var_expert_inr.evaluation.reporting import evaluation_output_dir
from var_expert_inr.evaluation.selection import parse_timestep_selection


def _existing_state(
    tmp_path: Path,
    stored_timesteps: tuple[int, ...],
    *,
    missing_psnr_timestep: int | None = None,
    timesteps: str = "uniform:10",
    stored_evaluation_id: str = "incremental",
    requested_evaluation_id: str = "incremental",
):
    run_dir = tmp_path / "Result" / "Main" / "Model" / "Dataset" / "Run"
    checkpoint = run_dir / "checkpoints" / "model.pth"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"reconstruction")
    output_root = tmp_path / "EvalResult"
    output_dir = evaluation_output_dir(
        run_dir,
        repo_root=exploration.REPO_ROOT,
        result_root=output_root,
        evaluation_id=stored_evaluation_id,
    )
    output_dir.mkdir(parents=True)
    (output_dir / "manifest.json").write_text(
        json.dumps({
            "source_kind": "checkpoint",
            "source_path": str(checkpoint.resolve()),
            "source_fingerprint": ArtifactStore(repo_root=exploration.REPO_ROOT, result_root=output_root).content_fingerprint(checkpoint),
            "timesteps": list(stored_timesteps),
            "targets": ["field"],
            "render_requested": False,
        }),
        encoding="utf-8",
    )
    rows = [
        {
            "target": "field",
            "timestep": timestep,
            "psnr": None if timestep == missing_psnr_timestep else 42.0,
        }
        for timestep in stored_timesteps
    ]
    (output_dir / "metrics.json").write_text(
        json.dumps({"status": "complete", "per_timestep": rows}),
        encoding="utf-8",
    )
    (output_dir / "metrics.csv").write_text("target,timestep,psnr\n", encoding="utf-8")
    return exploration._existing_evaluation_state(
        run_dir,
        raw={"data": {"targets": {"field": "unused"}}},
        target="all",
        timesteps=timesteps,
        requested_metrics=("psnr",),
        render=False,
        current_profile_fingerprint=None,
        result_root=output_root,
        evaluation_id=requested_evaluation_id,
        error_vmin=0.0,
        error_vmax=5.0,
    )


def test_denser_uniform_evaluation_is_reused(tmp_path: Path) -> None:
    stored = parse_timestep_selection("uniform:20", 100)

    state = _existing_state(tmp_path, stored)

    assert state is not None
    assert state["reuse_reason"] == "uniform-coverage"
    assert state["completed_metrics"] == {"psnr"}


def test_complete_metric_is_reused_across_evaluation_namespaces(tmp_path: Path) -> None:
    stored = parse_timestep_selection("uniform:20", 100)

    state = _existing_state(
        tmp_path,
        stored,
        stored_evaluation_id="legacy_metrics",
        requested_evaluation_id="result_psnr",
    )

    assert state is not None
    assert state["reuse_reason"] == "cross-namespace-uniform-coverage"
    assert state["completed_metrics"] == {"psnr"}


def test_sparser_uniform_evaluation_is_not_reused(tmp_path: Path) -> None:
    stored = parse_timestep_selection("uniform:10", 100)
    run_dir = tmp_path / "Result" / "Main" / "Model" / "Dataset" / "Run"
    checkpoint = run_dir / "checkpoints" / "model.pth"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"reconstruction")
    output_root = tmp_path / "EvalResult"
    output_dir = evaluation_output_dir(
        run_dir,
        repo_root=exploration.REPO_ROOT,
        result_root=output_root,
        evaluation_id="incremental",
    )
    output_dir.mkdir(parents=True)
    (output_dir / "manifest.json").write_text(json.dumps({
        "source_kind": "checkpoint",
        "source_path": str(checkpoint.resolve()),
        "source_fingerprint": ArtifactStore(repo_root=exploration.REPO_ROOT, result_root=output_root).content_fingerprint(checkpoint),
        "timesteps": list(stored),
        "targets": ["field"],
    }), encoding="utf-8")
    (output_dir / "metrics.json").write_text(json.dumps({
        "status": "complete",
        "per_timestep": [{"target": "field", "timestep": step, "psnr": 42.0} for step in stored],
    }), encoding="utf-8")
    (output_dir / "metrics.csv").write_text("target,timestep,psnr\n", encoding="utf-8")

    state = exploration._existing_evaluation_state(
        run_dir,
        raw={"data": {"targets": {"field": "unused"}}},
        target="all",
        timesteps="uniform:20",
        requested_metrics=("psnr",),
        render=False,
        current_profile_fingerprint=None,
        result_root=output_root,
        evaluation_id="incremental",
        error_vmin=0.0,
        error_vmax=5.0,
    )

    assert state is None


def test_equal_uniform_evaluation_is_reused(tmp_path: Path) -> None:
    stored = parse_timestep_selection("uniform:10", 100)

    state = _existing_state(tmp_path, stored)

    assert state is not None
    assert state["reuse_reason"] == "exact-timestep-match"
    assert state["completed_metrics"] == {"psnr"}


def test_uniform_request_larger_than_sequence_reuses_all_frames(tmp_path: Path) -> None:
    stored = parse_timestep_selection("uniform:200", 7)

    state = _existing_state(tmp_path, stored, timesteps="uniform:200")

    assert state is not None
    assert state["reuse_reason"] == "exact-timestep-match"
    assert state["completed_metrics"] == {"psnr"}


def test_non_uniform_selection_still_requires_exact_timesteps() -> None:
    assert exploration._matching_stored_timesteps(
        "0,10",
        stored_timesteps=(0, 10, 20),
    ) is None
    assert exploration._matching_stored_timesteps(
        "0,10",
        stored_timesteps=(0, 10),
    ) == ((0, 10), "exact-timestep-match")


def test_incomplete_denser_uniform_psnr_is_not_skipped(tmp_path: Path) -> None:
    stored = parse_timestep_selection("uniform:20", 100)

    state = _existing_state(tmp_path, stored, missing_psnr_timestep=stored[-1])

    assert state is not None
    assert "psnr" not in state["completed_metrics"]
