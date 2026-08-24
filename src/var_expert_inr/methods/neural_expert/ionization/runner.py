from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
import yaml

from ..common import dump_config, estimate_model_size_fp16, format_size_bytes, load_state_dict_payload, seed_everything, to_device
from ..config import run_dir_from_config
from ..train_utils import count_parameters, log_losses_wandb, log_string
from ..wandb_stub import get_wandb
from ....utils.exploration_probe import (
    ExplorationProbeRecorder,
    fixed_sample_indices,
    normalize_probe,
    probe_due,
    probe_progress,
    psnr_from_arrays,
)
from .datasets import build_dataloader
from .model_registry import build_model
from .stage_handler import TrainingStageHandler

wandb = get_wandb()


def lossdict2str(loss_dict):
    string = ""
    for key, value in loss_dict.items():
        val = value.item() if torch.is_tensor(value) else value
        if val == 0.0:
            continue
        if key == "lr" or abs(val) < 1.0e-4:
            string += f"{key}: {val:.4e}, "
        else:
            string += f"{key}: {val:.8f}, "
    return string


def _format_duration(seconds):
    total_seconds = max(0.0, float(seconds))
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    secs = total_seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"


def _log_timing_window(log_file, start_epoch, end_epoch, max_epochs, elapsed_seconds):
    window_epochs = max(1, int(end_epoch) - int(start_epoch) + 1)
    avg_epoch_time = float(elapsed_seconds) / window_epochs
    log_string(
        f"[timing] epochs {start_epoch:05d}-{end_epoch:05d}/{max_epochs:05d}: "
        f"total={elapsed_seconds:.3f}s ({_format_duration(elapsed_seconds)}), "
        f"avg_epoch={avg_epoch_time:.3f}s",
        log_file,
    )


def _save_inference_checkpoint(model, dataset, out_path):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": "neural_expert_inference_v1",
        "model_state": model.state_dict(),
        "x_mean": dataset.x_mean.cpu().numpy(),
        "x_std": dataset.x_std.cpu().numpy(),
        "y_mean": dataset.y_mean.cpu().numpy(),
        "y_std": dataset.y_std.cpu().numpy(),
    }
    torch.save(payload, str(out_path))


def run_train(cfg: dict, *, gpu: int = 0) -> dict:
    run_dir = run_dir_from_config(cfg)
    if normalize_probe(cfg.get("exploration_probe")).enabled:
        experiment_dir = run_dir
        run_token = time.strftime("%Y%m%d_%H%M%S")
        run_dir = experiment_dir / run_token
        collision = 1
        while run_dir.exists():
            run_dir = experiment_dir / f"{run_token}_{collision:02d}"
            collision += 1
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "wandb").mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    dump_config(cfg, run_dir / "config.yaml")

    cfg = yaml.safe_load(yaml.safe_dump(cfg))
    cfg["TRAINING"]["n_samples"] = int(cfg["TRAINING"]["num_epochs"])
    seed_everything(int(cfg["seed"]))

    train_dataloader, train_set = build_dataloader(cfg, cfg["DATA"]["attr_name"], training=True)
    cfg["MODEL"]["out_dim"] = int(train_set.target_dim)
    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    cfg["device"] = device

    wandb_run = wandb.init(
        project=f"{cfg.get('wandb_project', 'inr_moe_ionization')}_{cfg['DATA']['attr_name']}",
        entity="anu-cvml",
        save_code=True,
        dir=str(run_dir / "wandb"),
        mode="disabled",
    )
    cfg["WANDB"] = {"id": wandb_run.id, "project": wandb_run.project, "entity": wandb_run.entity}
    wandb_run.name = cfg["exp_id"]
    wandb.config.update(cfg)
    if hasattr(wandb, "run") and hasattr(wandb.run, "log_code"):
        wandb.run.log_code(".")
    wandb.define_metric("train/step")
    wandb.define_metric("train/*", step_metric="train/step")

    log_file = open(run_dir / "out.log", "w", encoding="utf-8")
    timing_cfg = cfg.get("TRAINING", {}).get("timing", {})
    timing_enabled = bool(timing_cfg.get("enabled", True))
    timing_log_interval = max(1, int(timing_cfg.get("log_every", timing_cfg.get("interval", 100)) or 100))
    timing_start = time.perf_counter()
    completed_epochs = 0
    timing_window_elapsed = 0.0
    timing_window_start_epoch = 1
    max_epochs = int(cfg["TRAINING"]["num_epochs"])
    print(f"torch version: {torch.__version__}")

    try:
        model, _ = build_model(cfg, cfg["LOSS"])
        n_parameters = count_parameters(model)
        total_parameters, model_size_bytes = estimate_model_size_fp16(model)
        wandb.log(
            {
                "number of paramters": n_parameters,
                "model_size_bytes_fp16": model_size_bytes,
                "model_size_mib_fp16": model_size_bytes / (1024**2),
            }
        )
        log_string(f"Number of parameters in the current model:{n_parameters}", log_file)
        log_string(
            f"Model size assuming float16 parameters: {format_size_bytes(model_size_bytes)} "
            f"(total parameters: {total_parameters})",
            log_file,
        )

        training_stage_handler = TrainingStageHandler(cfg["TRAINING"]["stages"], model, cfg)
        criterion = training_stage_handler.criterion
        lr = cfg["TRAINING"]["lr"] if isinstance(cfg["TRAINING"]["lr"], float) else cfg["TRAINING"]["lr"]["all"]
        optimizer = optim.Adam(training_stage_handler.get_trainable_params(), lr=lr, betas=(0.9, 0.999))
        if "moe" in cfg["MODEL"]["model_name"]:
            training_stage_handler.freeze_params()
        scheduler = training_stage_handler.get_scheduler(optimizer)

        if cfg["MODEL"].get("load_pt_manager", False):
            manager_pt_checkpoint_path = Path(cfg["MODEL"]["manager_pt_path"])
            if not manager_pt_checkpoint_path.exists():
                raise FileNotFoundError(f"Missing manager pretrain checkpoint: {manager_pt_checkpoint_path}")
            model.load_state_dict(load_state_dict_payload(manager_pt_checkpoint_path, device), strict=True)
            log_string(f"Loaded pretrained manager from {manager_pt_checkpoint_path}", log_file)

        model.to(device)
        model_outdir = run_dir / "checkpoints"
        save_interval = int(cfg["TRAINING"].get("save_every", 100))
        probe_cfg = normalize_probe(cfg.get("exploration_probe"))
        probe_recorder = None
        probe_points = None
        probe_values = None
        if probe_cfg.enabled and not cfg["TRAINING"].get("segmentation_mode", False):
            probe_indices = fixed_sample_indices(train_set.total_size, probe_cfg)
            probe_coords, _ = train_set._flat_to_coords(probe_indices)
            if train_set.normalize_inputs:
                probe_coords = (probe_coords - train_set.x_mean.numpy()) / train_set.x_std.numpy()
            probe_target = np.asarray(
                train_set.target[probe_indices], dtype=np.float32
            ).reshape(-1, train_set.target_dim)
            if train_set.normalize_targets:
                probe_target = (probe_target - train_set.y_mean.numpy()) / train_set.y_std.numpy()
            probe_points = torch.from_numpy(np.asarray(probe_coords, dtype=np.float32))
            probe_values = np.asarray(probe_target, dtype=np.float32).reshape(-1)
            probe_recorder = ExplorationProbeRecorder(run_dir / "metrics", probe_cfg)

        for step, data in enumerate(train_dataloader):
            if step >= max_epochs:
                break
            step_timer_start = time.perf_counter() if timing_enabled else None
            if save_interval > 0 and step % save_interval == 0:
                _save_inference_checkpoint(
                    model, train_set, model_outdir / f"{cfg['MODEL']['model_name']}_model_{step}.pth"
                )

            data = to_device(data, device)
            optimizer.zero_grad(set_to_none=True)
            model.train()
            output_pred = model(data["nonmnfld_points"])
            output_pred["step"] = step
            output_pred["logdir"] = str(run_dir)
            loss_dict = criterion(output_pred=output_pred, data=data, dataset=train_set)

            loss_dict["lr"] = torch.tensor(optimizer.param_groups[0]["lr"])
            if "moe" in cfg["MODEL"]["model_name"] and cfg["MODEL"]["manager_q_activation"] == "softmax" and cfg["MODEL"]["manager_softmax_temp_trainable"]:
                loss_dict["softmax_temp"] = model.manager_net.q_activation.temperature.item()

            log_losses_wandb(step, -1, 1, loss_dict, 1, criterion.weight_dict)
            if step % 100 == 0:
                log_string(f"{step:05d} " + lossdict2str(loss_dict), log_file)

            loss_dict["loss"].backward()
            optimizer.step()
            scheduler.step()

            current_step = step + 1
            if (
                probe_recorder is not None
                and probe_points is not None
                and probe_values is not None
                and probe_due(current_step, max_epochs, probe_cfg)
            ):
                probe_started = time.perf_counter()
                predictions = []
                model.eval()
                with torch.no_grad():
                    for offset in range(0, probe_points.shape[0], int(cfg["TRAINING"].get("batch_size", 16_000))):
                        points = probe_points[offset : offset + int(cfg["TRAINING"].get("batch_size", 16_000))].unsqueeze(0).to(device)
                        predictions.append(
                            model(points)["selected_nonmanifold_pnts_pred"].detach().cpu().numpy().reshape(-1)
                        )
                aggregate_psnr = psnr_from_arrays(probe_values, np.concatenate(predictions))
                progress = probe_progress(current_step, max_epochs, probe_cfg)
                elapsed = time.perf_counter() - probe_started
                probe_recorder.record(
                    progress=progress,
                    scope=f"variable:{cfg['DATA']['attr_name']}",
                    aggregate_psnr=aggregate_psnr,
                    sample_count=probe_values.size,
                    elapsed_seconds=elapsed,
                    details={"training_step": current_step},
                )
                log_string(
                    f"[exploration] progress={progress}/{probe_cfg.total_epoch_equivalents} "
                    f"aggregate_psnr={aggregate_psnr:.6f} variable={cfg['DATA']['attr_name']} "
                    f"samples={probe_values.size} elapsed={elapsed:.3f}s",
                    log_file,
                )

            if step > training_stage_handler.get_end_iteration():
                log_string("Moved to the next training stage...", log_file)
                training_stage_handler.move_to_the_next_training_stage(optimizer, scheduler)
                criterion = training_stage_handler.criterion

            completed_epochs = step + 1
            if timing_enabled and step_timer_start is not None:
                timing_window_elapsed += time.perf_counter() - step_timer_start
                if completed_epochs % timing_log_interval == 0:
                    _log_timing_window(log_file, timing_window_start_epoch, completed_epochs, max_epochs, timing_window_elapsed)
                    timing_window_elapsed = 0.0
                    timing_window_start_epoch = completed_epochs + 1

        final_state_path = (
            Path(cfg["MODEL"]["manager_pt_path"])
            if cfg["TRAINING"].get("segmentation_mode", False) and cfg["MODEL"].get("manager_pt_path")
            else model_outdir / f"{cfg['exp_id']}.pth"
        )
        _save_inference_checkpoint(model, train_set, final_state_path)
        log_string(f"Saved final state dict to {final_state_path}", log_file)
        checkpoint_bytes = int(final_state_path.stat().st_size)
        raw_target_bytes = int(np.asarray(train_set.target).nbytes)
        return {
            "run_dir": run_dir,
            "checkpoint_path": final_state_path,
            "checkpoint_bytes": checkpoint_bytes,
            "raw_target_bytes": raw_target_bytes,
            "cr": float(raw_target_bytes / max(checkpoint_bytes, 1)),
        }
    finally:
        if timing_enabled:
            if completed_epochs >= timing_window_start_epoch and timing_window_elapsed > 0.0:
                _log_timing_window(log_file, timing_window_start_epoch, completed_epochs, max_epochs, timing_window_elapsed)
            total_elapsed = time.perf_counter() - timing_start
            avg_epoch_time = total_elapsed / completed_epochs if completed_epochs > 0 else 0.0
            log_string(
                f"[timing] training summary: epochs={completed_epochs}, "
                f"total={total_elapsed:.3f}s ({_format_duration(total_elapsed)}), "
                f"avg_epoch={avg_epoch_time:.3f}s",
                log_file,
            )
        log_file.close()
