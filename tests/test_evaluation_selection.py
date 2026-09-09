from __future__ import annotations

from pathlib import Path

import pytest

from var_expert_inr.evaluation.reporting import (
    evaluation_output_dir,
)
from var_expert_inr.evaluation.selection import parse_timestep_selection


@pytest.mark.parametrize("total", [100, 2001])
def test_uniform_timestep_selection_spans_full_range(total: int) -> None:
    selected = parse_timestep_selection("uniform:20", total)
    assert len(selected) == 20
    assert selected[0] == 0
    assert selected[-1] == total - 1
    assert all(left < right for left, right in zip(selected, selected[1:]))


def test_uniform_timestep_selection_uses_all_short_sequences() -> None:
    assert parse_timestep_selection("uniform:20", 7) == tuple(range(7))


@pytest.mark.parametrize(
    ("selection", "message"),
    [
        ("uniform:1", "at least 2"),
        ("uniform:nope", "must be an integer"),
        ("uniform:20,3", "cannot be combined"),
    ],
)
def test_invalid_uniform_timestep_selection(selection: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        parse_timestep_selection(selection, 100)


def test_evaluation_id_isolates_schema_v2_outputs(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    run_dir = repo_root / "Result" / "Main" / "Model" / "Dataset" / "Run"
    default_output = evaluation_output_dir(run_dir, repo_root=repo_root)
    figure_output = evaluation_output_dir(
        run_dir,
        repo_root=repo_root,
        evaluation_id="figures",
    )
    expected = Path("Main/Model/Dataset/Run")
    assert default_output == repo_root / "EvalResult" / "evaluations" / "default" / expected
    assert figure_output == repo_root / "EvalResult" / "evaluations" / "figures" / expected


def test_external_result_paths_preserve_full_result_hierarchy(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    external_result = tmp_path / "autodl-tmp" / "Result"
    main_run = external_result / "Main" / "CoordNet" / "Ionization" / "GT"
    rd_run = external_result / "RD Curve" / "CoordNet" / "Ionization" / "0.41" / "GT"

    main_output = evaluation_output_dir(main_run, repo_root=repo_root)
    rd_output = evaluation_output_dir(rd_run, repo_root=repo_root)

    evaluation_root = repo_root / "EvalResult" / "evaluations" / "default"
    assert main_output == evaluation_root / "Main" / "CoordNet" / "Ionization" / "GT"
    assert rd_output == evaluation_root / "RD Curve" / "CoordNet" / "Ionization" / "0.41" / "GT"
    assert main_output != rd_output


def test_unrelated_external_runs_with_same_leaf_are_hashed(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    first = tmp_path / "external-a" / "Run"
    second = tmp_path / "external-b" / "Run"

    first_output = evaluation_output_dir(first, repo_root=repo_root)
    second_output = evaluation_output_dir(second, repo_root=repo_root)

    assert first_output.parent.name == "_external"
    assert second_output.parent.name == "_external"
    assert first_output.name.startswith("Run-")
    assert second_output.name.startswith("Run-")
    assert first_output != second_output
