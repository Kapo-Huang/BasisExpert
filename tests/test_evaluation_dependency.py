from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from var_expert_inr.evaluation.dependency import (
    aggregate_dependency_rows,
    compute_dependency_statistics,
    dependency_errors,
    sample_index_digest,
    sampled_channel,
    stable_sample_indices,
)
from var_expert_inr.evaluation.dependency_cache import (
    build_dependency_caches,
    load_dependency_configuration,
)
from var_expert_inr.evaluation.dependency_service import (
    canonical_dependency_run_dirs,
    resolve_dependency_group,
    evaluate_dependency_run,
)
from var_expert_inr.evaluation.selection import parse_metric_selection


def test_dependency_metric_tokens_are_supported() -> None:
    assert parse_metric_selection("pearson_error,mi_error") == (
        "pearson_error",
        "mi_error",
    )


def test_stable_sampling_reuses_exact_indices() -> None:
    first = stable_sample_indices(100_000, timestep=7, sample_ratio=0.2, seed=42)
    second = stable_sample_indices(100_000, timestep=7, sample_ratio=0.2, seed=42)
    assert np.array_equal(first, second)
    assert sample_index_digest(first) == sample_index_digest(second)
    assert 0.19 < first.size / 100_000 < 0.21
    assert not np.array_equal(first, stable_sample_indices(100_000, timestep=8))


def test_identical_dependency_statistics_have_zero_error() -> None:
    x = np.linspace(-1.0, 1.0, 10_000)
    samples = np.column_stack((x, -x, np.sin(4.0 * x)))
    gt = compute_dependency_statistics(samples)
    reconstruction = compute_dependency_statistics(
        samples,
        bin_edges=gt.bin_edges,
        pearson_valid_pairs=gt.pearson_valid_pairs,
        mi_valid_pairs=gt.mi_valid_pairs,
    )
    errors = dependency_errors(gt, reconstruction)
    assert errors["pearson_error"] == pytest.approx(0.0)
    assert errors["mi_error"] == pytest.approx(0.0)
    assert gt.pearson[0, 1] == pytest.approx(-1.0)
    assert gt.mutual_info[0, 1] > 0.0


def test_constant_reconstruction_is_penalized_and_gt_constant_pairs_are_skipped() -> None:
    x = np.linspace(-1.0, 1.0, 2000)
    gt_values = np.column_stack((x, x**2, np.ones_like(x)))
    gt = compute_dependency_statistics(gt_values)
    assert not gt.pearson_valid_pairs[0, 2]
    reconstruction_values = np.column_stack((np.zeros_like(x), x**2, np.ones_like(x)))
    reconstruction = compute_dependency_statistics(
        reconstruction_values,
        bin_edges=gt.bin_edges,
        pearson_valid_pairs=gt.pearson_valid_pairs,
        mi_valid_pairs=gt.mi_valid_pairs,
    )
    errors = dependency_errors(gt, reconstruction)
    assert errors["pearson_valid_pair_count"] == 1
    assert errors["pearson_error"] >= 0.0


def test_magnitude_transform_and_out_of_range_histogram_values() -> None:
    vector = np.array([[3.0, 4.0], [5.0, 12.0], [8.0, 15.0]])
    magnitude = sampled_channel(
        vector,
        np.arange(3),
        frame_size=3,
        transform="magnitude",
    )
    assert magnitude.tolist() == pytest.approx([5.0, 13.0, 17.0])
    gt = compute_dependency_statistics(np.column_stack((magnitude, [0.0, 0.5, 1.0])))
    reconstruction = compute_dependency_statistics(
        np.column_stack((magnitude * 100.0, [-100.0, 0.5, 100.0])),
        bin_edges=gt.bin_edges,
        pearson_valid_pairs=gt.pearson_valid_pairs,
        mi_valid_pairs=gt.mi_valid_pairs,
    )
    assert np.all(np.isfinite(reconstruction.mutual_info))


def test_duplicate_quantiles_and_no_valid_pairs_are_handled() -> None:
    duplicated = np.repeat([0.0, 1.0, 2.0], [900, 90, 10])
    stats = compute_dependency_statistics(
        np.column_stack((duplicated, np.ones_like(duplicated)))
    )
    assert np.all(np.diff(stats.bin_edges[0]) > 0.0)
    errors = dependency_errors(stats, stats)
    assert errors["pearson_valid_pair_count"] == 0
    assert errors["mi_valid_pair_count"] == 0
    aggregate = aggregate_dependency_rows([errors])
    assert aggregate == {"pearson_error": None, "mi_error": None}


def _experiment_payload(tmp_path: Path, targets: dict[str, Path], *, selected: str | None = None) -> dict:
    data = {
        "kind": "volume",
        "dataset_name": "synthetic",
        "volume_shape": {"X": 50, "Y": 1, "Z": 1, "T": 2},
        "targets": {name: str(path) for name, path in targets.items()},
    }
    if selected is not None:
        data["target"] = selected
    return {
        "experiment": "dependency-test",
        "exp_id": "dependency-test",
        "experiment_root": str(tmp_path / "runs"),
        "data": data,
        "model": {"name": "siren", "in_features": 4, "hidden_features": 8},
        "training": {"epochs": 1, "batch_size": 16},
        "evaluation": {"batch_size": 16},
    }


def _dependency_fixture(tmp_path: Path) -> tuple[Path, Path, dict[str, Path]]:
    values = np.linspace(-1.0, 1.0, 100, dtype=np.float32).reshape(2, 1, 1, 50)
    targets = {
        "a": tmp_path / "target_a.npy",
        "b": tmp_path / "target_b.npy",
    }
    np.save(targets["a"], values)
    np.save(targets["b"], values**2)
    main_config = tmp_path / "configs" / "main" / "synthetic.yaml"
    main_config.parent.mkdir(parents=True)
    main_config.write_text(
        yaml.safe_dump(_experiment_payload(tmp_path, targets), sort_keys=False),
        encoding="utf-8",
    )
    dependency_config = tmp_path / "configs" / "evaluation" / "dependency.yaml"
    dependency_config.parent.mkdir(parents=True)
    dependency_config.write_text(
        yaml.safe_dump(
            {
                "cache_root": "data/cache/evaluation/dependency",
                "settings": {"sample_ratio": 0.2, "seed": 42, "max_bins": 16, "variance_eps": 1e-12},
                "datasets": {
                    "synthetic": {
                        "config": "configs/main/synthetic.yaml",
                        "targets": ["a", "b"],
                    }
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return dependency_config, main_config, targets


def test_joint_prediction_dependency_integration_and_cache_reuse(tmp_path: Path) -> None:
    dependency_config, _, targets = _dependency_fixture(tmp_path)
    cache_path = build_dependency_caches(dependency_config, dataset="all")[0]
    assert cache_path.is_file()
    run_dir = tmp_path / "Result" / "Main" / "Joint" / "Synthetic" / "Joint"
    (run_dir / "configs").mkdir(parents=True)
    (run_dir / "predictions").mkdir()
    (run_dir / "configs" / "config.yaml").write_text(
        yaml.safe_dump(_experiment_payload(tmp_path, targets), sort_keys=False),
        encoding="utf-8",
    )
    for name, path in targets.items():
        values = np.load(path)
        np.save(run_dir / "predictions" / f"dependency-test_{name}.npy", values)
    result = evaluate_dependency_run(
        run_dir,
        source="prediction",
        result_root=tmp_path / "EvalResult",
        evaluation_id="dependency",
        dependency_config=dependency_config,
    )
    assert result["metrics"]["aggregate"]["pearson_error"] == pytest.approx(0.0)
    assert result["metrics"]["aggregate"]["mi_error"] == pytest.approx(0.0)
    assert (Path(result["output_dir"]) / "dependency_metrics.npz").is_file()
    cached = evaluate_dependency_run(
        run_dir,
        source="prediction",
        result_root=tmp_path / "EvalResult",
        evaluation_id="dependency",
        dependency_config=dependency_config,
    )
    assert cached["cache_hit"] is True

    arrays = dict(np.load(cache_path, allow_pickle=False))
    arrays["target_names"] = arrays["target_names"][::-1]
    np.savez(cache_path, **arrays)
    with pytest.raises(ValueError, match="target layout mismatch"):
        evaluate_dependency_run(
            run_dir,
            source="prediction",
            result_root=tmp_path / "EvalResult",
            evaluation_id="dependency",
            dependency_config=dependency_config,
        )

    arrays["target_names"] = arrays["target_names"][::-1]
    np.savez(cache_path, **arrays)
    metadata_path = cache_path.with_suffix(".json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["ground_truth"]["a"]["size"] += 1
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="cache is stale"):
        evaluate_dependency_run(
            run_dir,
            source="prediction",
            result_root=tmp_path / "EvalResult",
            evaluation_id="dependency",
            dependency_config=dependency_config,
        )
    with pytest.raises(ValueError, match="rerun with --overwrite"):
        build_dependency_caches(dependency_config, dataset="synthetic")


def test_sibling_group_requires_complete_canonical_targets(tmp_path: Path) -> None:
    dependency_config, _, targets = _dependency_fixture(tmp_path)
    configuration = load_dependency_configuration(dependency_config)
    parent = tmp_path / "Result" / "Main" / "Baseline" / "Synthetic"
    for name in ("a", "b"):
        run_dir = parent / name
        (run_dir / "configs").mkdir(parents=True)
        (run_dir / "configs" / "config.yaml").write_text(
            yaml.safe_dump(_experiment_payload(tmp_path, targets, selected=name), sort_keys=False),
            encoding="utf-8",
        )
    group = resolve_dependency_group(parent / "a", configuration)
    assert group.joint_run is None
    assert tuple(group.run_by_target) == ("a", "b")
    missing_parent = tmp_path / "Result" / "Main" / "Incomplete" / "Synthetic"
    run_dir = missing_parent / "a"
    (run_dir / "configs").mkdir(parents=True)
    (run_dir / "configs" / "config.yaml").write_text(
        yaml.safe_dump(_experiment_payload(tmp_path, targets, selected="a"), sort_keys=False),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="missing"):
        resolve_dependency_group(run_dir, configuration)


def test_sibling_prediction_dependency_integration(tmp_path: Path) -> None:
    dependency_config, _, targets = _dependency_fixture(tmp_path)
    build_dependency_caches(dependency_config, dataset="all")
    parent = tmp_path / "Result" / "Main" / "Baseline" / "Synthetic"
    for name, target_path in targets.items():
        run_dir = parent / name
        (run_dir / "configs").mkdir(parents=True)
        (run_dir / "predictions").mkdir()
        (run_dir / "configs" / "config.yaml").write_text(
            yaml.safe_dump(
                _experiment_payload(tmp_path, targets, selected=name),
                sort_keys=False,
            ),
            encoding="utf-8",
        )
        np.save(
            run_dir / "predictions" / f"dependency-test_{name}.npy",
            np.load(target_path),
        )

    result = evaluate_dependency_run(
        parent / "a",
        source="prediction",
        result_root=tmp_path / "EvalResult",
        evaluation_id="dependency",
        dependency_config=dependency_config,
    )

    assert result["metrics"]["aggregate"]["pearson_error"] == pytest.approx(0.0)
    assert result["metrics"]["aggregate"]["mi_error"] == pytest.approx(0.0)


def test_dependency_sibling_runs_are_collapsed_to_canonical_target(tmp_path: Path) -> None:
    dependency_config, _, targets = _dependency_fixture(tmp_path)
    configuration = load_dependency_configuration(dependency_config)
    parent = tmp_path / "Result" / "Main" / "Baseline" / "Synthetic"
    extra_path = tmp_path / "target_extra.npy"
    np.save(extra_path, np.zeros(100, dtype=np.float32))
    all_targets = {**targets, "extra": extra_path}
    for name in all_targets:
        run_dir = parent / name
        (run_dir / "configs").mkdir(parents=True)
        (run_dir / "configs" / "config.yaml").write_text(
            yaml.safe_dump(
                _experiment_payload(tmp_path, all_targets, selected=name),
                sort_keys=False,
            ),
            encoding="utf-8",
        )

    canonical = canonical_dependency_run_dirs(
        (parent / "extra", parent / "b", parent / "a"),
        configuration,
    )

    assert canonical == ((parent / "a").resolve(),)
