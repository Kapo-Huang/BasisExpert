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
