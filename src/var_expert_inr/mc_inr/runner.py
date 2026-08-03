from __future__ import annotations

import copy
import json
import logging
import math
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from numpy.lib.format import open_memmap
from sklearn.cluster import MiniBatchKMeans

from ..data import build_dataset
from ..evaluation.metrics import EPS, save_metrics
from ..pretrain.assignments import PretrainAssignmentConfig, compute_pretrain_assignments
from ..utils.io import resolve_path, sha256_payload
from ..utils.logging_utils import close_file_handlers, setup_logging
from ..utils.model_stats import collect_model_statistics, format_fp16_size_megabytes, format_param_count
from ..utils.runtime import apply_runtime_thread_limits, set_random_seed
from .checkpoint import load_mc_checkpoint, restore_target_layout, save_mc_checkpoint, validate_mc_checkpoint
from .config import MCExperimentConfig, MCTrainingConfig, load_config, save_config
from .data import (
    TargetLayoutEntry,
    compute_volume_centroids,
    fetch_mc_batch,
    prediction_shape,
    sample_node_rows_from_cluster,
    sample_node_rows,
    sample_volume_rows_from_cluster,
    sample_volume_rows_global,
    target_layout_from_dataset,
    volume_voxel_count,
    volume_spatial_coords_for_voxels,
)
from .model import ClusterCoordNet, MCINR

logger = logging.getLogger(__name__)
TIMESTAMP_RUN_PATTERN = re.compile(r"^\d{8}_\d{6}_\d{6}$")
NODE_CLUSTER_CHUNK_SIZE = 50_000
VOLUME_CENTROID_CHUNK_SIZE = 1_000_000


def _resolve_device(requested: str) -> torch.device:
    requested_norm = str(requested).strip().lower()
    if requested_norm.startswith("cuda") and not torch.cuda.is_available():
        logger.warning("CUDA requested but unavailable. Falling back to CPU.")
        return torch.device("cpu")
    return torch.device(requested)


def _experiment_dir(config: MCExperimentConfig) -> Path:
    return Path(config.experiment_root) / config.exp_id


def _is_timestamped_run_dir(path: Path) -> bool:
    return path.is_dir() and bool(TIMESTAMP_RUN_PATTERN.fullmatch(path.name))


def _build_run_dirs(run_dir: Path) -> dict[str, Path | str]:
    checkpoint_dir = run_dir / "checkpoints"
    config_dir = run_dir / "configs"
    prediction_dir = run_dir / "predictions"
    metrics_dir = run_dir / "metrics"
    logs_dir = run_dir / "logs"
    return {
        "experiment_dir": run_dir.parent,
        "run_dir": run_dir,
        "run_token": run_dir.name,
        "checkpoint_dir": checkpoint_dir,
        "config_dir": config_dir,
        "prediction_dir": prediction_dir,
        "metrics_dir": metrics_dir,
        "logs_dir": logs_dir,
    }


def _ensure_run_dirs(run_dir: Path) -> dict[str, Path | str]:
    dirs = _build_run_dirs(run_dir)
    for path in (
        dirs["run_dir"],
        dirs["checkpoint_dir"],
        dirs["config_dir"],
        dirs["prediction_dir"],
        dirs["metrics_dir"],
        dirs["logs_dir"],
    ):
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def _create_train_run_dirs(config: MCExperimentConfig):
    experiment_dir = _experiment_dir(config)
    experiment_dir.mkdir(parents=True, exist_ok=True)
    while True:
        run_token = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        run_dir = experiment_dir / run_token
        if not run_dir.exists():
            return _ensure_run_dirs(run_dir)
        time.sleep(0.001)


def _resolve_checkpoint_run_dir(checkpoint_path: str | Path) -> Path | None:
    resolved = Path(checkpoint_path).resolve()
    if resolved.parent.name != "checkpoints":
        return None
    run_dir = resolved.parent.parent
    if not _is_timestamped_run_dir(run_dir):
        raise FileNotFoundError(
            f"Checkpoint path must live under runs/<exp_id>/<timestamp>/checkpoints: {resolved}"
        )
    return run_dir


def _resolve_latest_run_dir(config: MCExperimentConfig) -> Path:
    experiment_dir = _experiment_dir(config)
    candidates = []
    if experiment_dir.exists():
        candidates = sorted(path for path in experiment_dir.iterdir() if _is_timestamped_run_dir(path))
    if not candidates:
        raise FileNotFoundError(
            f"No timestamped run directory found for exp_id '{config.exp_id}' under '{experiment_dir}'."
        )
    return candidates[-1]


def _resolve_existing_run_dirs(config: MCExperimentConfig, checkpoint_path: str | Path | None = None):
    run_dir = None
    if checkpoint_path is not None:
        run_dir = _resolve_checkpoint_run_dir(checkpoint_path)
    if run_dir is None:
        run_dir = _resolve_latest_run_dir(config)
    return _ensure_run_dirs(run_dir)


def _prepare_runtime(
    config_path: str | Path,
    *,
    create_run: bool,
    checkpoint_path: str | Path | None = None,
):
    config_started_at = time.perf_counter()
    config = load_config(config_path)
    config_load_seconds = time.perf_counter() - config_started_at

    dirs_started_at = time.perf_counter()
    if create_run:
        dirs = _create_train_run_dirs(config)
    else:
        dirs = _resolve_existing_run_dirs(config, checkpoint_path=checkpoint_path)
    run_dir_prepare_seconds = time.perf_counter() - dirs_started_at

    setup_logging(log_dir=dirs["logs_dir"], log_file=f"run_{dirs['run_token']}.log")

    dataset_started_at = time.perf_counter()
    dataset = build_dataset(config.data)
    dataset_init_seconds = time.perf_counter() - dataset_started_at

    config_snapshot_path = save_config(config, Path(dirs["config_dir"]) / "config.yaml")
    if config.log.effective_config:
        logger.info("Using config source: %s", config.source_config_path or "<memory>")
        logger.info("Effective config:\n%s", config_snapshot_path.read_text(encoding="utf-8").rstrip())
    if config.log.startup_timing:
        logger.info("Config load: %.2fs", config_load_seconds)
        logger.info("Run dir prepare: %.2fs", run_dir_prepare_seconds)
        logger.info("Dataset init: %.2fs", dataset_init_seconds)
    device = _resolve_device(config.training.device)
    return config, dirs, dataset, device


def _criterion(predictions: torch.Tensor, targets: torch.Tensor, loss_type: str) -> torch.Tensor:
    if loss_type == "mse":
        return F.mse_loss(predictions, targets)
    if loss_type == "l1":
        return F.l1_loss(predictions, targets)
    raise ValueError(f"Unsupported loss_type: {loss_type}")


def _assignment_expected_shape(dataset) -> tuple[int, ...]:
    if dataset.meta.kind == "node":
        return (int(len(dataset)),)
    volume_shape = dataset.meta.volume_shape
    if volume_shape is None:
        raise ValueError("Volume dataset is missing volume_shape")
    return (int(volume_shape.X) * int(volume_shape.Y) * int(volume_shape.Z),)


def _expected_spatial_dims(dataset) -> int:
    if dataset.meta.kind == "node":
        return min(3, int(dataset.meta.input_dim))
    return 3


def _output_dim_from_layout(layout: tuple[TargetLayoutEntry, ...]) -> int:
    return sum(int(entry.dim) for entry in layout)


def _iter_dataset_row_chunks(total_size: int, batch_size: int):
    chunk = max(1, int(batch_size))
    for start in range(0, int(total_size), chunk):
        stop = min(start + chunk, int(total_size))
        yield np.arange(start, stop, dtype=np.int64)


def _iter_node_coord_chunks(dataset, *, spatial_dims: int, chunk_size: int = NODE_CLUSTER_CHUNK_SIZE):
    total = len(dataset)
    for start in range(0, total, int(chunk_size)):
        stop = min(start + int(chunk_size), total)
        rows = np.arange(start, stop, dtype=np.int64)
        batch = dataset.fetch_batch(rows.tolist(), include_targets=False)
        yield start, stop, batch.coords.numpy()[:, :spatial_dims]


def _dataset_path_payload(dataset) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "kind": str(dataset.meta.kind),
        "input_dim": int(dataset.meta.input_dim),
        "n_samples": int(dataset.meta.n_samples),
        "target_names": list(dataset.target_names()),
        "target_dims": {str(name): int(dataset.meta.target_dims[name]) for name in dataset.target_names()},
    }
    if dataset.meta.volume_shape is not None:
        payload["volume_shape"] = dataset.meta.volume_shape.to_dict()
    if hasattr(dataset, "coords_path"):
        payload["coords_path"] = str(getattr(dataset, "coords_path"))
    if hasattr(dataset, "target_path"):
        payload["target_path"] = str(getattr(dataset, "target_path"))
    if hasattr(dataset, "targets_map"):
        payload["targets"] = {
            str(name): str(path) for name, path in sorted(getattr(dataset, "targets_map").items())
        }
    return payload


def _dataset_fingerprint(dataset) -> str:
    return sha256_payload(_dataset_path_payload(dataset))


def _assignment_cache_json_path(cache_path: str | Path) -> Path:
    path = Path(cache_path)
    return path.with_suffix(".json")


def _load_cached_assignments(cache_path: str, *, dataset, cfg: MCTrainingConfig) -> np.ndarray | None:
    if not cache_path:
        return None
    cache = Path(cache_path)
    meta_path = _assignment_cache_json_path(cache)
    if not cache.exists() or not meta_path.exists():
        return None
    try:
        cached = np.asarray(np.load(cache), dtype=np.int64)
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("Ignoring unreadable MC-INR assignment cache at %s: %s", cache, exc)
        return None

    expected_shape = _assignment_expected_shape(dataset)
    expected_spatial_dims = _expected_spatial_dims(dataset)
    expected_fingerprint = _dataset_fingerprint(dataset)
    checks = {
        "dataset_kind": str(metadata.get("dataset_kind")) == str(dataset.meta.kind),
        "dataset_length": int(metadata.get("dataset_length", -1)) == int(dataset.meta.n_samples),
        "input_dim": int(metadata.get("input_dim", -1)) == int(dataset.meta.input_dim),
        "spatial_dims": int(metadata.get("spatial_dims", -1)) == int(expected_spatial_dims),
        "initial_k": int(metadata.get("initial_k", -1)) == int(cfg.initial_k),
        "cluster_init_method": str(metadata.get("cluster_init_method", "")) == str(cfg.cluster_init_method),
        "seed": int(metadata.get("seed", -1)) == int(cfg.seed),
        "assignment_shape": tuple(metadata.get("assignment_shape", ())) == tuple(expected_shape),
        "fingerprint": str(metadata.get("dataset_fingerprint", "")) == expected_fingerprint,
    }
    if tuple(cached.shape) != tuple(expected_shape):
        logger.warning("Ignoring stale MC-INR assignment cache at %s due to shape mismatch", cache)
        return None
    if not all(checks.values()):
        logger.warning("Ignoring stale MC-INR assignment cache at %s due to metadata mismatch", cache)
        return None
    if cached.size > 0 and (int(cached.min()) < 0 or int(cached.max()) >= int(cfg.initial_k)):
        logger.warning("Ignoring stale MC-INR assignment cache at %s due to id range mismatch", cache)
        return None
    logger.info("Using cached MC-INR assignments: %s shape=%s", cache, tuple(cached.shape))
    return cached.astype(np.int32)


def _save_cached_assignments(cache_path: str, assignments: np.ndarray, *, dataset, cfg: MCTrainingConfig) -> None:
    if not cache_path:
        return
    cache = Path(cache_path)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache, np.asarray(assignments, dtype=np.int32))
    metadata = {
        "dataset_kind": str(dataset.meta.kind),
        "dataset_length": int(dataset.meta.n_samples),
        "input_dim": int(dataset.meta.input_dim),
        "spatial_dims": int(_expected_spatial_dims(dataset)),
        "initial_k": int(cfg.initial_k),
        "cluster_init_method": str(cfg.cluster_init_method),
        "seed": int(cfg.seed),
        "assignment_shape": list(np.asarray(assignments).shape),
        "dataset_fingerprint": _dataset_fingerprint(dataset),
        "data_payload": _dataset_path_payload(dataset),
        "created_at": datetime.utcnow().isoformat() + "Z",
    }
    _assignment_cache_json_path(cache).write_text(json.dumps(metadata, ensure_ascii=True, indent=2), encoding="utf-8")


def _compute_node_centroids(dataset, assignments: np.ndarray, *, num_clusters: int, spatial_dims: int) -> np.ndarray:
    sums = np.zeros((num_clusters, spatial_dims), dtype=np.float64)
    counts = np.zeros((num_clusters,), dtype=np.float64)
    for start, stop, coords in _iter_node_coord_chunks(dataset, spatial_dims=spatial_dims):
        cids = np.asarray(assignments[start:stop], dtype=np.int64)
        counts += np.bincount(cids, minlength=num_clusters).astype(np.float64)
        for dim in range(spatial_dims):
            sums[:, dim] += np.bincount(cids, weights=coords[:, dim], minlength=num_clusters).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    return (sums / counts[:, None]).astype(np.float32)


def _initialize_node_clusters(dataset, cfg: MCTrainingConfig) -> tuple[np.ndarray, np.ndarray]:
    method = str(cfg.cluster_init_method).strip().lower()
    if method not in {"auto", "coord_kmeans"}:
        raise ValueError(f"Unsupported node cluster_init_method for mc_inr: {cfg.cluster_init_method!r}")

    cached = _load_cached_assignments(cfg.assignments_cache_path, dataset=dataset, cfg=cfg)
    spatial_dims = _expected_spatial_dims(dataset)
    if cached is not None:
        return cached, _compute_node_centroids(dataset, cached, num_clusters=int(cfg.initial_k), spatial_dims=spatial_dims)

    logger.info(
        "Initializing MC-INR node clusters with MiniBatchKMeans: clusters=%d spatial_dims=%d",
        int(cfg.initial_k),
        int(spatial_dims),
    )
    batch_size = max(256, min(NODE_CLUSTER_CHUNK_SIZE, len(dataset)))
    kmeans = MiniBatchKMeans(
        n_clusters=int(cfg.initial_k),
        random_state=int(cfg.seed),
        batch_size=int(batch_size),
        n_init=3,
    )
    for _, _, coords in _iter_node_coord_chunks(dataset, spatial_dims=spatial_dims):
        kmeans.partial_fit(coords)

    assignments = np.empty((len(dataset),), dtype=np.int32)
    for start, stop, coords in _iter_node_coord_chunks(dataset, spatial_dims=spatial_dims):
        assignments[start:stop] = kmeans.predict(coords).astype(np.int32)
    _save_cached_assignments(cfg.assignments_cache_path, assignments, dataset=dataset, cfg=cfg)
    return assignments, np.asarray(kmeans.cluster_centers_, dtype=np.float32)


def _initialize_volume_clusters(dataset, cfg: MCTrainingConfig) -> tuple[np.ndarray, np.ndarray]:
    method = str(cfg.cluster_init_method).strip().lower()
    if method not in {"auto", "voxel_clustering"}:
        raise ValueError(f"Unsupported volume cluster_init_method for mc_inr: {cfg.cluster_init_method!r}")

    cached = _load_cached_assignments(cfg.assignments_cache_path, dataset=dataset, cfg=cfg)
    if cached is None:
        assignments = compute_pretrain_assignments(
            dataset,
            int(cfg.initial_k),
            PretrainAssignmentConfig(seed=int(cfg.seed), cache_path=""),
        ).astype(np.int32)
        _save_cached_assignments(cfg.assignments_cache_path, assignments, dataset=dataset, cfg=cfg)
    else:
        assignments = cached
    centroids = compute_volume_centroids(
        dataset.meta,
        assignments,
        int(cfg.initial_k),
        chunk_size=VOLUME_CENTROID_CHUNK_SIZE,
    )
    return assignments, centroids


def _initialize_clusters(dataset, cfg: MCTrainingConfig) -> tuple[np.ndarray, np.ndarray]:
    if dataset.meta.kind == "node":
        return _initialize_node_clusters(dataset, cfg)
    return _initialize_volume_clusters(dataset, cfg)


def _build_model(
    config: MCExperimentConfig,
    dataset,
    *,
    centroids: np.ndarray,
    target_layout: tuple[TargetLayoutEntry, ...],
) -> MCINR:
    return MCINR(
        centroids=centroids,
        target_layout=target_layout,
        in_features=int(dataset.meta.input_dim),
        hidden_features=int(config.model.hidden_features),
        gfe_layers=int(config.model.gfe_layers),
        lfe_layers=int(config.model.lfe_layers),
    )


def _build_template_model(
    config: MCExperimentConfig,
    dataset,
    *,
    target_layout: tuple[TargetLayoutEntry, ...],
) -> ClusterCoordNet:
    return ClusterCoordNet(
        in_features=int(dataset.meta.input_dim),
        out_features=_output_dim_from_layout(target_layout),
        hidden_features=int(config.model.hidden_features),
        gfe_layers=int(config.model.gfe_layers),
        lfe_layers=int(config.model.lfe_layers),
    )


def _copy_template_to_all_clusters(model: MCINR, template_model: ClusterCoordNet) -> None:
    template_state = template_model.state_dict()
    for cluster_network in model.cluster_networks:
        cluster_network.load_state_dict(template_state)


def _build_scheduler(optimizer: torch.optim.Optimizer, config: MCTrainingConfig):
    scheduler_cfg = config.scheduler
    if not scheduler_cfg.enabled or int(scheduler_cfg.step_size) <= 0:
        return None
    return torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=int(scheduler_cfg.step_size),
        gamma=float(scheduler_cfg.gamma),
    )


def _yield_cluster_grouped_rows(
    sampled_rows: np.ndarray,
    cluster_ids: np.ndarray,
    *,
    batch_size: int,
    rng: np.random.Generator,
):
    rows_by_cluster: dict[int, np.ndarray] = {}
    for cluster_id in np.unique(cluster_ids):
        mask = cluster_ids == int(cluster_id)
        rows = np.asarray(sampled_rows[mask], dtype=np.int64)
        if rows.size == 0:
            continue
        rows = rows[rng.permutation(rows.size)]
        rows_by_cluster[int(cluster_id)] = rows

    cluster_order = np.asarray(list(rows_by_cluster.keys()), dtype=np.int64)
    if cluster_order.size > 0:
        cluster_order = cluster_order[rng.permutation(cluster_order.size)]
    for cluster_id in cluster_order.tolist():
        rows = rows_by_cluster[int(cluster_id)]
        for start in range(0, int(rows.size), int(batch_size)):
            stop = min(start + int(batch_size), int(rows.size))
            yield rows[start:stop]


def _stage_epoch_batches(
    dataset,
    layout: tuple[TargetLayoutEntry, ...],
    assignments: np.ndarray,
    *,
    batch_size: int,
    sampling_ratio: float,
    rng: np.random.Generator,
    cluster_aware_batches: bool,
    sample_count_override: int = 0,
):
    if int(sample_count_override) > 0:
        sample_count = int(sample_count_override)
        batch_size_int = max(1, int(batch_size))
        total_rows = int(len(dataset))

        def _budget_generator():
            remaining = sample_count
            while remaining > 0:
                current = min(batch_size_int, remaining)
                rows = rng.integers(0, total_rows, size=current, dtype=np.int64)
                remaining -= current
                yield fetch_mc_batch(dataset, rows, layout, assignments)

        return sample_count, _budget_generator()

    if dataset.meta.kind == "node":
        sampled_rows = sample_node_rows(assignments, sampling_ratio, rng)
        if sampled_rows.size == 0:
            raise RuntimeError("No sampled rows available for MC-INR node stage")
        sampled_rows = np.asarray(sampled_rows, dtype=np.int64)
        sample_count = int(sampled_rows.size)

        def _generator():
            if cluster_aware_batches:
                sampled_cluster_ids = np.asarray(assignments[sampled_rows], dtype=np.int64)
                for rows in _yield_cluster_grouped_rows(
                    sampled_rows,
                    sampled_cluster_ids,
                    batch_size=int(batch_size),
                    rng=rng,
                ):
                    yield fetch_mc_batch(dataset, rows, layout, assignments)
                return

            shuffled_rows = sampled_rows[rng.permutation(sampled_rows.size)]
            for start in range(0, sample_count, int(batch_size)):
                stop = min(start + int(batch_size), sample_count)
                yield fetch_mc_batch(dataset, shuffled_rows[start:stop], layout, assignments)

        return sample_count, _generator()

    volume_shape = dataset.meta.volume_shape
    if volume_shape is None:
        raise ValueError("Volume metadata is required for MC-INR volume batching")
    voxel_count = volume_voxel_count(dataset.meta)
    time_count = int(volume_shape.T)
    total_rows = int(voxel_count) * int(time_count)
    sample_count = max(1, int(math.ceil(total_rows * float(sampling_ratio))))
    batch_size_int = max(1, int(batch_size))

    def _generator():
        if cluster_aware_batches:
            assignments_np = np.asarray(assignments, dtype=np.int64)
            cluster_ids = np.unique(assignments_np)
            cluster_ids = cluster_ids[cluster_ids >= 0]
            if cluster_ids.size == 0:
                raise RuntimeError("No valid volume clusters available")

            cluster_voxel_counts = np.bincount(assignments_np, minlength=int(cluster_ids.max()) + 1)
            cluster_row_counts = cluster_voxel_counts.astype(np.int64) * int(time_count)
            positive_clusters = cluster_ids[cluster_row_counts[cluster_ids] > 0]
            if positive_clusters.size == 0:
                raise RuntimeError("No non-empty volume clusters available")

            weights = cluster_row_counts[positive_clusters].astype(np.float64)
            weights /= float(weights.sum())
            cluster_sample_counts = rng.multinomial(sample_count, weights)
            sample_count_by_cluster = {
                int(cluster_id): int(cluster_sample_counts[index])
                for index, cluster_id in enumerate(positive_clusters.tolist())
            }
            cluster_order = positive_clusters[rng.permutation(positive_clusters.size)]

            for cluster_id in cluster_order.tolist():
                cluster_remaining = sample_count_by_cluster[int(cluster_id)]
                while cluster_remaining > 0:
                    current = min(batch_size_int, cluster_remaining)
                    rows = sample_volume_rows_from_cluster(
                        assignments,
                        int(cluster_id),
                        dataset.meta,
                        current,
                        rng,
                    )
                    if rows.size == 0:
                        break
                    cluster_remaining -= int(rows.size)
                    yield fetch_mc_batch(dataset, rows, layout, assignments)
            return

        remaining = sample_count
        while remaining > 0:
            current = min(batch_size_int, remaining)
            rows = sample_volume_rows_global(assignments, dataset.meta, current, rng)
            if rows.size == 0:
                break
            remaining -= int(rows.size)
            yield fetch_mc_batch(dataset, rows, layout, assignments)

    return sample_count, _generator()


def _sample_meta_task_rows(
    dataset,
    assignments: np.ndarray,
    cluster_id: int,
    *,
    max_rows: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if dataset.meta.kind == "node":
        return sample_node_rows_from_cluster(
            assignments,
            cluster_id,
            max_rows,
            rng,
        )
    return sample_volume_rows_from_cluster(
        assignments,
        cluster_id,
        dataset.meta,
        max_rows,
        rng,
    )


def _run_meta_initialization(
    *,
    dataset,
    layout: tuple[TargetLayoutEntry, ...],
    assignments: np.ndarray,
    centroids: np.ndarray,
    config: MCExperimentConfig,
    device: torch.device,
    config_hash: str,
    checkpoint_dir: Path,
    resume_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    logger.info("%s", "#" * 60)
    logger.info("PHASE 1: META INITIALIZATION")
    logger.info("%s", "#" * 60)

    template_model = _build_template_model(config, dataset, target_layout=layout).to(device)
    start_iteration = 1
    best_loss = float("inf")
    epochs_no_improve = 0
    if resume_payload is not None and str(resume_payload.get("mc_stage", "")) == "meta_init":
        template_model.load_state_dict(resume_payload["template_model_state"])
        start_iteration = int(resume_payload.get("iteration", resume_payload.get("epoch", 0))) + 1
        best_loss = float(resume_payload.get("best_loss", best_loss))
        epochs_no_improve = int(resume_payload.get("epochs_no_improve", 0))

    if start_iteration > int(config.training.meta_iterations):
        logger.info(
            "Meta initialization already reached iteration %d (target=%d). Skipping.",
            start_iteration - 1,
            int(config.training.meta_iterations),
        )
        return {
            "template_model": template_model,
            "last_iteration": start_iteration - 1,
            "stage_checkpoint_path": None,
            "last_interval_checkpoint_path": None,
            "best_loss": best_loss,
            "epochs_no_improve": epochs_no_improve,
            "loss_history": [],
        }

    started_at = time.time()
    cluster_ids = np.unique(np.asarray(assignments, dtype=np.int64))
    cluster_ids = cluster_ids[cluster_ids >= 0]
    if cluster_ids.size == 0:
        raise RuntimeError("Meta initialization requires at least one non-empty cluster")

    loss_history: list[float] = []
    last_interval_checkpoint_path: Path | None = None
    final_iteration = start_iteration - 1
    for iteration in range(start_iteration, int(config.training.meta_iterations) + 1):
        final_iteration = iteration
        rng = np.random.default_rng(int(config.training.seed) + iteration)
        task_count = min(int(config.training.meta_batch_clusters), int(cluster_ids.size))
        selected_clusters = (
            cluster_ids
            if task_count >= int(cluster_ids.size)
            else rng.choice(cluster_ids, size=task_count, replace=False).astype(np.int64)
        )
        deltas = [torch.zeros_like(param, device=device) for param in template_model.parameters()]
        support_losses: list[float] = []
        tasks_used = 0
        inner_batch_size = int(config.training.meta_inner_batch_size)
        support_max_rows = int(config.training.meta_support_max_rows)

        for cluster_id in selected_clusters.tolist():
            inner_model = copy.deepcopy(template_model).to(device)
            inner_optimizer = torch.optim.Adam(
                inner_model.parameters(),
                lr=float(config.training.meta_inner_lr),
                weight_decay=float(config.training.weight_decay),
            )
            inner_model.train()
            task_loss_total = 0.0
            task_steps = 0
            for _ in range(int(config.training.meta_inner_steps)):
                support_rows = _sample_meta_task_rows(
                    dataset,
                    assignments,
                    int(cluster_id),
                    max_rows=support_max_rows,
                    rng=rng,
                )
                if support_rows.size == 0:
                    continue

                support_rows = support_rows[rng.permutation(support_rows.size)]
                for start in range(0, int(support_rows.size), inner_batch_size):
                    stop = min(start + inner_batch_size, int(support_rows.size))
                    rows = support_rows[start:stop]
                    batch = fetch_mc_batch(dataset, rows, layout, assignments)
                    coords = batch.coords.to(device, non_blocking=True)
                    targets = batch.targets_concat.to(device, non_blocking=True)

                    preds = inner_model(coords)
                    loss = _criterion(preds, targets, config.training.loss_type)
                    inner_optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    inner_optimizer.step()

                    task_loss_total += float(loss.item())
                    task_steps += 1

            if task_steps <= 0:
                continue

            support_losses.append(task_loss_total / float(task_steps))
            tasks_used += 1
            with torch.no_grad():
                for delta_tensor, template_param, inner_param in zip(
                    deltas,
                    template_model.parameters(),
                    inner_model.parameters(),
                ):
                    delta_tensor.add_(inner_param.detach() - template_param.detach())

        if tasks_used <= 0:
            raise RuntimeError("Meta initialization could not sample any support tasks")

        with torch.no_grad():
            outer_lr = float(config.training.meta_outer_lr)
            for template_param, delta_tensor in zip(template_model.parameters(), deltas):
                template_param.add_(outer_lr * (delta_tensor / float(tasks_used)))

        avg_loss = float(np.mean(support_losses))
        loss_history.append(avg_loss)
        if best_loss - avg_loss > float(config.training.convergence_delta):
            best_loss = avg_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if config.log.epoch_summary and config.training.log_every > 0 and (
            iteration % int(config.training.log_every) == 0 or iteration == start_iteration
        ):
            logger.info(
                "Meta_init iteration %d/%d loss=%.6e best=%.6e tasks=%d inner_steps=%d outer_lr=%.2e time=%.1fs",
                iteration,
                int(config.training.meta_iterations),
                avg_loss,
                best_loss,
                int(tasks_used),
                int(config.training.meta_inner_steps),
                float(config.training.meta_outer_lr),
                time.time() - started_at,
            )

        if config.training.save_every > 0 and iteration % int(config.training.save_every) == 0:
            last_interval_checkpoint_path = save_mc_checkpoint(
                path=checkpoint_dir / f"{config.exp_id}_meta_init_iteration{iteration}.pth",
                model=None,
                optimizer=None,
                scheduler=None,
                epoch=iteration,
                stage="meta_init",
                config_hash=config_hash,
                target_names=dataset.target_names(),
                target_layout=layout,
                assignments=assignments,
                centroids=centroids,
                best_loss=best_loss,
                epochs_no_improve=epochs_no_improve,
                extra_payload={
                    "template_model_state": template_model.state_dict(),
                    "iteration": int(iteration),
                },
            )

        if int(config.training.convergence_patience) > 0 and epochs_no_improve >= int(config.training.convergence_patience):
            logger.info(
                "Meta initialization early stop at iteration %d after %d unimproved iterations.",
                iteration,
                epochs_no_improve,
            )
            break

    stage_checkpoint_path = save_mc_checkpoint(
        path=checkpoint_dir / f"{config.exp_id}_meta_init.pth",
        model=None,
        optimizer=None,
        scheduler=None,
        epoch=final_iteration,
        stage="meta_init",
        config_hash=config_hash,
        target_names=dataset.target_names(),
        target_layout=layout,
        assignments=assignments,
        centroids=centroids,
        best_loss=best_loss,
        epochs_no_improve=epochs_no_improve,
        extra_payload={
            "template_model_state": template_model.state_dict(),
            "iteration": int(final_iteration),
        },
    )
    return {
        "template_model": template_model,
        "last_iteration": final_iteration,
        "stage_checkpoint_path": stage_checkpoint_path,
        "last_interval_checkpoint_path": last_interval_checkpoint_path,
        "best_loss": best_loss,
        "epochs_no_improve": epochs_no_improve,
        "loss_history": loss_history,
    }


def _run_stage(
    *,
    stage_name: str,
    model: MCINR,
    dataset,
    layout: tuple[TargetLayoutEntry, ...],
    assignments: np.ndarray,
    config: MCExperimentConfig,
    device: torch.device,
    epochs: int,
    sampling_ratio: float,
    lr: float,
    config_hash: str,
    checkpoint_dir: Path,
    split_round: int,
    resume_payload: dict[str, Any] | None = None,
    metrics_dir: Path | None = None,
) -> dict[str, Any]:
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(lr),
        weight_decay=float(config.training.weight_decay),
    )
    scheduler = _build_scheduler(optimizer, config.training)
    stage_display_name = "CLUSTER-SPECIFIC FINE-TUNING" if stage_name == "finetune" else stage_name.upper()
    logger.info("%s", "#" * 60)
    logger.info("PHASE 2: %s", stage_display_name)
    logger.info("%s", "#" * 60)

    start_epoch = 1
    best_loss = float("inf")
    epochs_no_improve = 0
    if resume_payload is not None and str(resume_payload.get("mc_stage", "")) == stage_name:
        if resume_payload.get("optimizer_state") is not None:
            optimizer.load_state_dict(resume_payload["optimizer_state"])
        if scheduler is not None and resume_payload.get("scheduler_state") is not None:
            scheduler.load_state_dict(resume_payload["scheduler_state"])
        start_epoch = int(resume_payload.get("epoch", 0)) + 1
        best_loss = float(resume_payload.get("best_loss", best_loss))
        epochs_no_improve = int(resume_payload.get("epochs_no_improve", 0))

    if start_epoch > int(epochs):
        logger.info(
            "%s already reached epoch %d (target=%d). Skipping.",
            stage_display_name,
            start_epoch - 1,
            int(epochs),
        )
        return {
            "last_epoch": start_epoch - 1,
            "stage_checkpoint_path": None,
            "best_loss": best_loss,
            "epochs_no_improve": epochs_no_improve,
            "optimizer": optimizer,
            "scheduler": scheduler,
            "sample_counts": [],
            "batch_cluster_unique_counts": [],
        }

    normalized_sampling_ratio = float(sampling_ratio)
    if normalized_sampling_ratio <= 0.0:
        raise ValueError(f"{stage_name} sampling_ratio must be positive")
    normalized_sampling_ratio = min(normalized_sampling_ratio, 1.0)

    model.to(device)
    started_at = time.time()
    last_checkpoint_path: Path | None = None
    final_epoch = start_epoch - 1
    sample_counts: list[int] = []
    batch_cluster_unique_counts: list[int] = []
    from ..utils.exploration_probe import (
        ExplorationProbeRecorder,
        fixed_sample_indices,
        normalize_probe,
        probe_due,
        probe_progress,
        psnr_from_arrays,
    )

    probe_cfg = normalize_probe(config.exploration_probe)
    probe_rows = fixed_sample_indices(len(dataset), probe_cfg) if probe_cfg.enabled and stage_name == "finetune" else None
    probe_recorder = (
        ExplorationProbeRecorder(metrics_dir, probe_cfg)
        if probe_rows is not None and metrics_dir is not None
        else None
    )
    for epoch in range(start_epoch, int(epochs) + 1):
        final_epoch = epoch
        rng = np.random.default_rng(int(config.training.seed) + epoch + 10_000 + int(split_round) * 100_000)
        sample_count, batches = _stage_epoch_batches(
            dataset,
            layout,
            assignments,
            batch_size=int(config.training.batch_size),
            sampling_ratio=float(normalized_sampling_ratio),
            rng=rng,
            cluster_aware_batches=bool(config.training.cluster_aware_batches),
            sample_count_override=(
                int(config.training.batches_per_epoch_budget) * int(config.training.batch_size)
                if int(config.training.batches_per_epoch_budget) > 0
                else 0
            ),
        )
        sample_counts.append(int(sample_count))

        model.train()
        epoch_loss_sum = 0.0
        total_samples = 0
        for batch in batches:
            coords = batch.coords.to(device, non_blocking=True)
            targets = batch.targets_concat.to(device, non_blocking=True)
            cluster_ids = batch.cluster_ids.to(device, non_blocking=True)
            batch_cluster_unique_counts.append(int(torch.unique(cluster_ids).numel()))

            preds = model(coords, cluster_ids, return_concat=True)
            loss = _criterion(preds, targets, config.training.loss_type)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_size_current = int(coords.shape[0])
            epoch_loss_sum += float(loss.item()) * batch_size_current
            total_samples += batch_size_current

        avg_loss = epoch_loss_sum / max(total_samples, 1)
        if scheduler is not None:
            scheduler.step()

        if best_loss - avg_loss > float(config.training.convergence_delta):
            best_loss = avg_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if config.log.epoch_summary and config.training.log_every > 0 and (
            epoch % int(config.training.log_every) == 0 or epoch == start_epoch
        ):
            logger.info(
                "Finetune epoch %d/%d loss=%.6e best=%.6e sampled_points=%d sampling_ratio=%.3f lr=%.2e split_round=%d time=%.1fs",
                epoch,
                int(epochs),
                avg_loss,
                best_loss,
                int(sample_count),
                float(normalized_sampling_ratio),
                float(optimizer.param_groups[0]["lr"]),
                int(split_round),
                time.time() - started_at,
            )

        if config.training.save_every > 0 and epoch % int(config.training.save_every) == 0:
            last_checkpoint_path = save_mc_checkpoint(
                path=checkpoint_dir / f"{config.exp_id}_{stage_name}_round{split_round}_epoch{epoch}.pth",
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                stage=stage_name,
                config_hash=config_hash,
                target_names=dataset.target_names(),
                target_layout=layout,
                assignments=assignments,
                best_loss=best_loss,
                epochs_no_improve=epochs_no_improve,
                extra_payload={"split_round": int(split_round)},
            )

        if probe_recorder is not None and probe_due(epoch, int(epochs), probe_cfg):
            probe_started = time.perf_counter()
            target_parts = {entry.name: [] for entry in layout}
            prediction_parts = {entry.name: [] for entry in layout}
            model.eval()
            with torch.no_grad():
                for offset in range(0, int(probe_rows.size), int(config.training.pred_batch_size)):
                    rows = probe_rows[offset : offset + int(config.training.pred_batch_size)]
                    batch = fetch_mc_batch(dataset, rows, layout, assignments)
                    coords = batch.coords.to(device, non_blocking=True)
                    cluster_ids = batch.cluster_ids.to(device, non_blocking=True)
                    predictions = model(coords, cluster_ids, return_concat=True).detach().cpu().numpy()
                    targets = batch.targets_concat.numpy()
                    for entry in layout:
                        target_parts[entry.name].append(targets[:, entry.start : entry.stop])
                        prediction_parts[entry.name].append(predictions[:, entry.start : entry.stop])
            per_target = {
                entry.name: psnr_from_arrays(
                    np.concatenate(target_parts[entry.name], axis=0),
                    np.concatenate(prediction_parts[entry.name], axis=0),
                )
                for entry in layout
            }
            probe_recorder.record(
                progress=probe_progress(epoch, int(epochs), probe_cfg),
                scope="aggregate",
                aggregate_psnr=float(np.mean(list(per_target.values()))),
                sample_count=int(probe_rows.size),
                elapsed_seconds=time.perf_counter() - probe_started,
                details=per_target,
            )
            model.train()

        if int(config.training.convergence_patience) > 0 and epochs_no_improve >= int(config.training.convergence_patience):
            logger.info(
                "Finetune early stop at epoch %d after %d unimproved epochs.",
                epoch,
                epochs_no_improve,
            )
            break

    stage_checkpoint_path = save_mc_checkpoint(
        path=checkpoint_dir / f"{config.exp_id}_{stage_name}_round{split_round}.pth",
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=final_epoch,
        stage=stage_name,
        config_hash=config_hash,
        target_names=dataset.target_names(),
        target_layout=layout,
        assignments=assignments,
        best_loss=best_loss,
        epochs_no_improve=epochs_no_improve,
        extra_payload={"split_round": int(split_round)},
    )
    return {
        "last_epoch": final_epoch,
        "stage_checkpoint_path": stage_checkpoint_path,
        "best_loss": best_loss,
        "epochs_no_improve": epochs_no_improve,
        "optimizer": optimizer,
        "scheduler": scheduler,
        "last_interval_checkpoint_path": last_checkpoint_path,
        "sample_counts": sample_counts,
        "batch_cluster_unique_counts": batch_cluster_unique_counts,
    }


def _build_model_from_payload(
    config: MCExperimentConfig,
    dataset,
    payload: dict[str, Any],
) -> tuple[MCINR, tuple[TargetLayoutEntry, ...], np.ndarray]:
    layout = restore_target_layout(payload)
    centroids = np.asarray(payload["centroids"], dtype=np.float32)
    model = _build_model(config, dataset, centroids=centroids, target_layout=layout)
    model_state = payload.get("model_state")
    if model_state is None:
        raise ValueError("Checkpoint payload is missing model_state")
    model.load_state_dict(model_state, strict=False)
    assignments = np.asarray(payload["cluster_assignments"], dtype=np.int32)
    return model, layout, assignments


def _compute_cluster_residuals_streaming(
    *,
    model: MCINR,
    dataset,
    layout: tuple[TargetLayoutEntry, ...],
    assignments: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> dict[int, dict[str, float | int]]:
    sse = np.zeros((model.cluster_count,), dtype=np.float64)
    counts = np.zeros((model.cluster_count,), dtype=np.int64)
    model.eval()
    with torch.no_grad():
        for rows in _iter_dataset_row_chunks(len(dataset), int(batch_size)):
            batch = fetch_mc_batch(dataset, rows, layout, assignments)
            coords = batch.coords.to(device, non_blocking=True)
            targets = batch.targets_concat.to(device, non_blocking=True)
            cluster_ids = batch.cluster_ids.to(device, non_blocking=True)
            preds = model(coords, cluster_ids, return_concat=True)
            point_mse = torch.mean((preds - targets) ** 2, dim=1)
            for cluster_id in torch.unique(cluster_ids).tolist():
                mask = cluster_ids == int(cluster_id)
                counts[int(cluster_id)] += int(mask.sum().item())
                sse[int(cluster_id)] += float(point_mse[mask].sum().item())

    residuals: dict[int, dict[str, float | int]] = {}
    for cluster_id in range(int(model.cluster_count)):
        count = int(counts[cluster_id])
        mse = float(sse[cluster_id] / max(count, 1))
        residuals[int(cluster_id)] = {
            "count": count,
            "sse": float(sse[cluster_id]),
            "mse": mse,
        }
    return residuals


def _node_spatial_coords_for_rows(dataset, rows: np.ndarray) -> np.ndarray:
    batch = dataset.fetch_batch(np.asarray(rows, dtype=np.int64).tolist(), include_targets=False)
    return np.asarray(batch.coords.numpy()[:, : _expected_spatial_dims(dataset)], dtype=np.float32)


def _append_split_cluster_networks(
    *,
    model: MCINR,
    assignments: np.ndarray,
    parent_id: int,
    member_indices: np.ndarray,
    child_membership: np.ndarray,
    child_centroids: np.ndarray,
) -> tuple[int, int]:
    child_id_1, child_id_2 = model.split_cluster(parent_id, child_centroids)
    assignments[np.asarray(member_indices[child_membership == 0], dtype=np.int64)] = int(child_id_1)
    assignments[np.asarray(member_indices[child_membership == 1], dtype=np.int64)] = int(child_id_2)
    return int(child_id_1), int(child_id_2)


def _split_high_residual_clusters(
    *,
    model: MCINR,
    dataset,
    assignments: np.ndarray,
    residuals: dict[int, dict[str, float | int]],
    config: MCExperimentConfig,
) -> list[dict[str, Any]]:
    split_results: list[dict[str, Any]] = []
    for parent_id, cluster_stats in sorted(residuals.items()):
        if int(cluster_stats["count"]) < int(config.training.min_split_points):
            continue
        if float(cluster_stats["mse"]) <= float(config.training.split_threshold):
            continue
        if parent_id >= model.cluster_count or not bool(model.routing_active[parent_id].item()):
            continue

        member_indices = np.flatnonzero(np.asarray(assignments, dtype=np.int64) == int(parent_id))
        if member_indices.size < max(2, int(config.training.min_split_points)):
            continue

        if dataset.meta.kind == "node":
            spatial_coords = _node_spatial_coords_for_rows(dataset, member_indices)
        else:
            spatial_coords = volume_spatial_coords_for_voxels(member_indices, dataset.meta)
        if spatial_coords.shape[0] < 2:
            continue

        kmeans = MiniBatchKMeans(
            n_clusters=2,
            random_state=int(config.training.seed) + int(parent_id),
            batch_size=max(32, min(50_000, int(spatial_coords.shape[0]))),
            n_init=3,
        )
        child_membership = kmeans.fit_predict(spatial_coords).astype(np.int64)
        child_centroids = np.asarray(kmeans.cluster_centers_, dtype=np.float32)
        child_ids = _append_split_cluster_networks(
            model=model,
            assignments=assignments,
            parent_id=int(parent_id),
            member_indices=member_indices,
            child_membership=child_membership,
            child_centroids=child_centroids,
        )
        split_results.append(
            {
                "parent_id": int(parent_id),
                "child_ids": tuple(int(cid) for cid in child_ids),
                "count": int(member_indices.size),
                "mse": float(cluster_stats["mse"]),
            }
        )
    return split_results


def _prediction_path(output_dir: Path, exp_id: str, target_name: str) -> Path:
    return output_dir / (f"{exp_id}.npy" if target_name == "target" else f"{exp_id}_{target_name}.npy")


def _finalize_streaming_metrics(
    dataset,
    metrics_state: dict[str, Any],
    *,
    checkpoint_path: str | Path | None = None,
) -> dict[str, Any]:
    results: dict[str, Any] = {"targets": {}, "aggregate": {}}
    mses = []
    maes = []
    psnrs = []
    max_errors = []

    def _build_result(state: dict[str, Any]) -> dict[str, Any]:
        count = int(state["count"])
        mse = float(state["sse"] / max(count, 1))
        mae = float(state["sae"] / max(count, 1))
        gt_min = float(state["gt_min"])
        gt_max = float(state["gt_max"])
        data_range = float(gt_max - gt_min)
        if not np.isfinite(data_range) or data_range <= 0.0:
            data_range = max(abs(gt_min), abs(gt_max)) + EPS
        psnr = float("inf") if mse <= 0.0 else 10.0 * math.log10((data_range ** 2) / (mse + EPS))
        return {
            "mse": mse,
            "mae": mae,
            "max_error": float(state["max_abs_error"]),
            "psnr": psnr,
        }

    for name in dataset.target_names():
        state = metrics_state[name]
        target_result = _build_result(state)
        if dataset.meta.volume_shape is not None:
            per_time = []
            for t, time_state in enumerate(state["per_time"]):
                time_result = _build_result(time_state)
                time_result["t"] = int(t)
                per_time.append(time_result)
            target_result["per_time"] = per_time
        results["targets"][name] = target_result
        mses.append(target_result["mse"])
        maes.append(target_result["mae"])
        psnrs.append(target_result["psnr"])
        max_errors.append(target_result["max_error"])

    results["aggregate"] = {
        "mse": float(np.mean(mses)) if mses else float("nan"),
        "mae": float(np.mean(maes)) if maes else float("nan"),
        "psnr": float(np.mean(psnrs)) if psnrs else float("nan"),
        "max_error": float(np.max(max_errors)) if max_errors else float("nan"),
    }
    if checkpoint_path is not None:
        ckpt_size = Path(checkpoint_path).stat().st_size
        raw_bytes = sum(int(dataset.meta.n_samples) * int(dataset.meta.target_dims[name]) * 4 for name in dataset.target_names())
        results["aggregate"]["cr"] = float(raw_bytes / ckpt_size) if ckpt_size > 0 else float("nan")
        results["aggregate"]["checkpoint_bytes"] = int(ckpt_size)
        results["aggregate"]["raw_target_bytes"] = int(raw_bytes)
    return results


def _predict_and_save_streaming(
    *,
    model: MCINR,
    dataset,
    layout: tuple[TargetLayoutEntry, ...],
    device: torch.device,
    batch_size: int,
    prediction_dir: Path,
    exp_id: str,
    checkpoint_path: str | Path | None = None,
    compute_metrics: bool = False,
) -> dict[str, Any]:
    prediction_dir.mkdir(parents=True, exist_ok=True)
    prediction_paths: dict[str, Path] = {}
    prediction_memmaps: dict[str, np.memmap] = {}
    flat_prediction_views: dict[str, np.memmap] = {}

    metrics_state: dict[str, Any] | None = None
    if compute_metrics:
        metrics_state = {}
        for entry in layout:
            state: dict[str, Any] = {
                "sse": 0.0,
                "sae": 0.0,
                "max_abs_error": 0.0,
                "count": 0,
                "gt_min": float("inf"),
                "gt_max": float("-inf"),
            }
            if dataset.meta.volume_shape is not None:
                state["per_time"] = [
                    {
                        "sse": 0.0,
                        "sae": 0.0,
                        "max_abs_error": 0.0,
                        "count": 0,
                        "gt_min": float("inf"),
                        "gt_max": float("-inf"),
                    }
                    for _ in range(int(dataset.meta.volume_shape.T))
                ]
            metrics_state[entry.name] = state

    for entry in layout:
        path = _prediction_path(prediction_dir, exp_id, entry.name)
        final_shape = prediction_shape(dataset.meta, int(entry.dim))
        mmap = open_memmap(path, mode="w+", dtype=np.float32, shape=final_shape)
        prediction_paths[entry.name] = path
        prediction_memmaps[entry.name] = mmap
        flat_prediction_views[entry.name] = mmap.reshape(int(dataset.meta.n_samples), int(entry.dim))

    model.eval()
    with torch.no_grad():
        for rows in _iter_dataset_row_chunks(len(dataset), int(batch_size)):
            batch = dataset.fetch_batch(rows.tolist(), include_targets=compute_metrics)
            coords = batch.coords.to(device, non_blocking=True)
            preds = model(coords)
            for entry in layout:
                pred_np = np.asarray(preds[entry.name].detach().cpu().numpy(), dtype=np.float32)
                flat_prediction_views[entry.name][rows] = pred_np
                if metrics_state is None:
                    continue
                if isinstance(batch.targets, torch.Tensor):
                    gt_np = np.asarray(batch.targets.detach().cpu().numpy(), dtype=np.float32)
                else:
                    gt_np = np.asarray(batch.targets[entry.name].detach().cpu().numpy(), dtype=np.float32)
                diff = pred_np.astype(np.float64) - gt_np.astype(np.float64)
                abs_diff = np.abs(diff)
                state = metrics_state[entry.name]
                state["sse"] += float(np.sum(diff ** 2))
                state["sae"] += float(np.sum(abs_diff))
                state["max_abs_error"] = max(float(state["max_abs_error"]), float(np.max(abs_diff)))
                state["count"] += int(gt_np.size)
                state["gt_min"] = min(float(state["gt_min"]), float(np.min(gt_np)))
                state["gt_max"] = max(float(state["gt_max"]), float(np.max(gt_np)))
                if dataset.meta.volume_shape is not None:
                    voxel_count = int(dataset.meta.volume_shape.X) * int(dataset.meta.volume_shape.Y) * int(dataset.meta.volume_shape.Z)
                    time_ids = rows // voxel_count
                    for t in np.unique(time_ids).tolist():
                        time_mask = time_ids == int(t)
                        time_gt = gt_np[time_mask]
                        time_diff = diff[time_mask]
                        time_abs_diff = abs_diff[time_mask]
                        time_state = state["per_time"][int(t)]
                        time_state["sse"] += float(np.sum(time_diff ** 2))
                        time_state["sae"] += float(np.sum(time_abs_diff))
                        time_state["max_abs_error"] = max(
                            float(time_state["max_abs_error"]),
                            float(np.max(time_abs_diff)),
                        )
                        time_state["count"] += int(time_gt.size)
                        time_state["gt_min"] = min(float(time_state["gt_min"]), float(np.min(time_gt)))
                        time_state["gt_max"] = max(float(time_state["gt_max"]), float(np.max(time_gt)))

    for mmap in prediction_memmaps.values():
        mmap.flush()
    prediction_memmaps.clear()
    flat_prediction_views.clear()

    result = {"prediction_paths": prediction_paths}
    if metrics_state is not None:
        result["metrics"] = _finalize_streaming_metrics(dataset, metrics_state, checkpoint_path=checkpoint_path)
    return result


def _load_resume_payload(
    *,
    config: MCExperimentConfig,
    dataset,
    config_hash: str,
    checkpoint_path: str | Path,
) -> dict[str, Any]:
    payload = load_mc_checkpoint(checkpoint_path)
    validate_mc_checkpoint(
        payload,
        dataset.target_names(),
        expected_assignment_shape=_assignment_expected_shape(dataset),
        expected_output_dim=_output_dim_from_layout(target_layout_from_dataset(dataset)),
        expected_spatial_dims=_expected_spatial_dims(dataset),
        expected_config_hash=config_hash,
    )
    return payload


def run_train(config_path: str | Path, *, resume_path: str | Path | None = None) -> dict[str, Any]:
    apply_runtime_thread_limits()
    try:
        config, dirs, dataset, device = _prepare_runtime(config_path, create_run=True)
        set_random_seed(int(config.training.seed))
        config_hash = sha256_payload(config.to_dict())
        layout = target_layout_from_dataset(dataset)

        resolved_resume = (
            resolve_path(str(resume_path), base_dir=Path(config_path).resolve().parent)
            if resume_path is not None
            else config.training.resume_path
        )

        resume_payload = None
        model: MCINR | None = None
        template_model: ClusterCoordNet | None = None
        assignments: np.ndarray
        centroids: np.ndarray
        resume_stage = ""
        current_split_round = 0

        if resolved_resume:
            resume_payload = _load_resume_payload(
                config=config,
                dataset=dataset,
                config_hash=config_hash,
                checkpoint_path=resolved_resume,
            )
            resume_stage = str(resume_payload.get("mc_stage", "") or "")
            logger.info("Resuming MC-INR training from %s (stage=%s)", resolved_resume, resume_stage)
            assignments = np.asarray(resume_payload["cluster_assignments"], dtype=np.int32)
            centroids = np.asarray(resume_payload["centroids"], dtype=np.float32)
            layout = restore_target_layout(resume_payload)
            if resume_stage == "meta_init":
                template_model = _build_template_model(config, dataset, target_layout=layout).to(device)
                template_model.load_state_dict(resume_payload["template_model_state"])
            else:
                model, layout, assignments = _build_model_from_payload(config, dataset, resume_payload)
                centroids = np.asarray(model.centroids.detach().cpu().numpy(), dtype=np.float32)
                current_split_round = int(resume_payload.get("split_round", 0))
        else:
            assignments, centroids = _initialize_clusters(dataset, config.training)

        stage_checkpoints: dict[str, Any] = {"meta_init": None, "finetune": None, "split": []}
        training_summary: dict[str, Any] = {
            "meta_init": None,
            "finetune_rounds": [],
            "split_rounds": [],
        }

        if model is None:
            meta_result = _run_meta_initialization(
                dataset=dataset,
                layout=layout,
                assignments=assignments,
                centroids=centroids,
                config=config,
                device=device,
                config_hash=config_hash,
                checkpoint_dir=Path(dirs["checkpoint_dir"]),
                resume_payload=resume_payload if resume_stage == "meta_init" else None,
            )
            template_model = meta_result["template_model"]
            stage_checkpoints["meta_init"] = (
                str(meta_result["stage_checkpoint_path"]) if meta_result["stage_checkpoint_path"] is not None else None
            )
            training_summary["meta_init"] = {
                "last_iteration": int(meta_result["last_iteration"]),
                "best_loss": float(meta_result["best_loss"]),
                "loss_history": [float(loss) for loss in meta_result["loss_history"]],
            }

            model = _build_model(config, dataset, centroids=centroids, target_layout=layout)
            _copy_template_to_all_clusters(model, template_model)

        stats = collect_model_statistics(model)
        if config.log.model_stats:
            logger.info(
                "MC-INR model size: params=%s trainable=%s size(fp16, all parameters)=%s",
                format_param_count(int(stats["param_count"])),
                format_param_count(int(stats["trainable_param_count"])),
                format_fp16_size_megabytes(int(stats["fp16_size_bytes"])),
            )

        finetune_resume_payload = resume_payload if resume_stage == "finetune" else None
        if resume_stage == "split":
            logger.info(
                "Resuming from split checkpoint at split_round=%d; starting next fine-tune round.",
                current_split_round,
            )

        finetune_result = _run_stage(
            stage_name="finetune",
            model=model,
            dataset=dataset,
            layout=layout,
            assignments=assignments,
            config=config,
            device=device,
            epochs=int(config.training.finetune_epochs),
            sampling_ratio=float(config.training.finetune_sampling_ratio),
            lr=float(config.training.finetune_lr if config.training.finetune_lr is not None else config.training.lr),
            config_hash=config_hash,
            checkpoint_dir=Path(dirs["checkpoint_dir"]),
            split_round=int(current_split_round),
            resume_payload=finetune_resume_payload,
            metrics_dir=Path(dirs["metrics_dir"]),
        )
        stage_checkpoints["finetune"] = (
            str(finetune_result["stage_checkpoint_path"]) if finetune_result["stage_checkpoint_path"] is not None else None
        )
        training_summary["finetune_rounds"].append(
            {
                "split_round": int(current_split_round),
                "last_epoch": int(finetune_result["last_epoch"]),
                "sample_counts": [int(value) for value in finetune_result["sample_counts"]],
                "batch_cluster_unique_counts": [int(value) for value in finetune_result["batch_cluster_unique_counts"]],
            }
        )

        while bool(config.training.recluster_after_finetune) and int(current_split_round) < int(config.training.max_recluster_rounds):
            residuals = _compute_cluster_residuals_streaming(
                model=model,
                dataset=dataset,
                layout=layout,
                assignments=assignments,
                device=device,
                batch_size=int(config.training.pred_batch_size),
            )
            split_results = _split_high_residual_clusters(
                model=model,
                dataset=dataset,
                assignments=assignments,
                residuals=residuals,
                config=config,
            )
            if not split_results:
                break

            current_split_round += 1
            split_checkpoint = save_mc_checkpoint(
                path=Path(dirs["checkpoint_dir"]) / f"{config.exp_id}_split_round{current_split_round}.pth",
                model=model,
                optimizer=None,
                scheduler=None,
                epoch=int(finetune_result["last_epoch"]),
                stage="split",
                config_hash=config_hash,
                target_names=dataset.target_names(),
                target_layout=layout,
                assignments=assignments,
                best_loss=float(finetune_result["best_loss"]),
                epochs_no_improve=int(finetune_result["epochs_no_improve"]),
                extra_payload={
                    "split_round": int(current_split_round),
                    "split_results": split_results,
                },
            )
            stage_checkpoints["split"].append(str(split_checkpoint))
            training_summary["split_rounds"].append(
                {
                    "split_round": int(current_split_round),
                    "split_results": split_results,
                }
            )

            finetune_result = _run_stage(
                stage_name="finetune",
                model=model,
                dataset=dataset,
                layout=layout,
                assignments=assignments,
                config=config,
                device=device,
                epochs=int(config.training.finetune_epochs),
                sampling_ratio=float(config.training.finetune_sampling_ratio),
                lr=float(config.training.finetune_lr if config.training.finetune_lr is not None else config.training.lr),
                config_hash=config_hash,
                checkpoint_dir=Path(dirs["checkpoint_dir"]),
                split_round=int(current_split_round),
                resume_payload=None,
            )
            stage_checkpoints["finetune"] = (
                str(finetune_result["stage_checkpoint_path"]) if finetune_result["stage_checkpoint_path"] is not None else None
            )
            training_summary["finetune_rounds"].append(
                {
                    "split_round": int(current_split_round),
                    "last_epoch": int(finetune_result["last_epoch"]),
                    "sample_counts": [int(value) for value in finetune_result["sample_counts"]],
                    "batch_cluster_unique_counts": [int(value) for value in finetune_result["batch_cluster_unique_counts"]],
                }
            )

        final_checkpoint = save_mc_checkpoint(
            path=Path(dirs["checkpoint_dir"]) / f"{config.exp_id}.pth",
            model=model,
            optimizer=finetune_result["optimizer"],
            scheduler=finetune_result["scheduler"],
            epoch=int(finetune_result["last_epoch"]),
            stage="finetune",
            config_hash=config_hash,
            target_names=dataset.target_names(),
            target_layout=layout,
            assignments=assignments,
            best_loss=float(finetune_result["best_loss"]),
            epochs_no_improve=int(finetune_result["epochs_no_improve"]),
            extra_payload={"split_round": int(current_split_round)},
        )
        if not bool(config.evaluation.save_predictions):
            logger.info("Skipping automatic prediction/evaluation after training.")
            return {
                "checkpoint_path": str(final_checkpoint),
                "stage_checkpoints": stage_checkpoints,
                "training_summary": training_summary,
            }

        prediction_result = _predict_and_save_streaming(
            model=model,
            dataset=dataset,
            layout=layout,
            device=device,
            batch_size=int(config.training.pred_batch_size),
            prediction_dir=Path(dirs["prediction_dir"]),
            exp_id=config.exp_id,
            checkpoint_path=final_checkpoint,
            compute_metrics=True,
        )
        metrics_path = save_metrics(Path(dirs["metrics_dir"]) / f"{config.exp_id}.json", prediction_result["metrics"])
        return {
            "checkpoint_path": str(final_checkpoint),
            "prediction_paths": prediction_result["prediction_paths"],
            "metrics_path": str(metrics_path),
            "metrics": prediction_result["metrics"],
            "stage_checkpoints": stage_checkpoints,
            "training_summary": training_summary,
        }
    finally:
        close_file_handlers()


def _load_checkpoint_model_for_inference(
    *,
    config: MCExperimentConfig,
    dataset,
    checkpoint_path: str | Path,
    config_hash: str,
) -> tuple[MCINR, tuple[TargetLayoutEntry, ...]]:
    payload = _load_resume_payload(
        config=config,
        dataset=dataset,
        config_hash=config_hash,
        checkpoint_path=checkpoint_path,
    )
    if str(payload.get("mc_stage", "")) == "meta_init":
        raise ValueError("meta_init checkpoints cannot be used directly for prediction/evaluation")
    model, layout, _ = _build_model_from_payload(config, dataset, payload)
    return model, layout


def run_predict(config_path: str | Path, *, checkpoint_path: str | Path | None = None) -> dict[str, Any]:
    apply_runtime_thread_limits()
    try:
        config, dirs, dataset, device = _prepare_runtime(
            config_path,
            create_run=False,
            checkpoint_path=checkpoint_path,
        )
        resolved_checkpoint = (
            Path(checkpoint_path)
            if checkpoint_path is not None
            else Path(dirs["checkpoint_dir"]) / f"{config.exp_id}.pth"
        )
        config_hash = sha256_payload(config.to_dict())
        model, layout = _load_checkpoint_model_for_inference(
            config=config,
            dataset=dataset,
            checkpoint_path=resolved_checkpoint,
            config_hash=config_hash,
        )
        model.to(device)
        prediction_result = _predict_and_save_streaming(
            model=model,
            dataset=dataset,
            layout=layout,
            device=device,
            batch_size=int(config.evaluation.batch_size or config.training.pred_batch_size),
            prediction_dir=Path(dirs["prediction_dir"]),
            exp_id=config.exp_id,
            checkpoint_path=resolved_checkpoint,
            compute_metrics=False,
        )
        return {
            "checkpoint_path": str(resolved_checkpoint),
            "prediction_paths": prediction_result["prediction_paths"],
        }
    finally:
        close_file_handlers()


def run_evaluate(config_path: str | Path, *, checkpoint_path: str | Path | None = None) -> dict[str, Any]:
    apply_runtime_thread_limits()
    try:
        config, dirs, dataset, device = _prepare_runtime(
            config_path,
            create_run=False,
            checkpoint_path=checkpoint_path,
        )
        resolved_checkpoint = (
            Path(checkpoint_path)
            if checkpoint_path is not None
            else Path(dirs["checkpoint_dir"]) / f"{config.exp_id}.pth"
        )
        config_hash = sha256_payload(config.to_dict())
        model, layout = _load_checkpoint_model_for_inference(
            config=config,
            dataset=dataset,
            checkpoint_path=resolved_checkpoint,
            config_hash=config_hash,
        )
        model.to(device)
        prediction_result = _predict_and_save_streaming(
            model=model,
            dataset=dataset,
            layout=layout,
            device=device,
            batch_size=int(config.evaluation.batch_size or config.training.pred_batch_size),
            prediction_dir=Path(dirs["prediction_dir"]),
            exp_id=config.exp_id,
            checkpoint_path=resolved_checkpoint,
            compute_metrics=True,
        )
        metrics_path = save_metrics(Path(dirs["metrics_dir"]) / f"{config.exp_id}.json", prediction_result["metrics"])
        return {
            "checkpoint_path": str(resolved_checkpoint),
            "prediction_paths": prediction_result["prediction_paths"],
            "metrics": prediction_result["metrics"],
            "metrics_path": str(metrics_path),
        }
    finally:
        close_file_handlers()
