from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
import yaml

from ..common import dump_config, estimate_model_size_fp32, format_size_bytes, load_state_dict_payload, seed_everything, to_device
from ..config import run_dir_from_config
from ..train_utils import count_parameters, log_losses_wandb, log_string
from ..wandb_stub import get_wandb
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


def _save_validate_checkpoint(model, dataset, out_path):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state": model.state_dict(),
        "x_mean": dataset.x_mean.cpu().numpy(),
        "x_std": dataset.x_std.cpu().numpy(),
        "y_mean": dataset.y_mean.cpu().numpy(),
        "y_std": dataset.y_std.cpu().numpy(),
    }
    torch.save(payload, str(out_path))


def _build_validate_config(cfg, validate_ckpt_path):
    data_cfg = cfg["DATA"]
    model_cfg = cfg["MODEL"]
    volume_shape = data_cfg["volume_shape"]
    return {
        "experiment": f"exp_data_ionization_neural_expert_{data_cfg['attr_name']}",
        "exp_id": f"neural-expert-ionization-{data_cfg['attr_name']}",
        "experiment_root": "experiments",
        "data": {
            "dataset_name": "ionization",
            "split": "train",
            "data_root": "./data",
            "target_path": data_cfg["target_path"],
            "target_stats_path": data_cfg["target_stats_path"],
            "compute_target_stats": False,
            "volume_shape": {
                "X": int(volume_shape["X"]),
                "Y": int(volume_shape["Y"]),
                "Z": int(volume_shape["Z"]),
                "T": int(volume_shape["T"]),
            },
            "normalize_inputs": bool(data_cfg.get("normalize_inputs", True)),
            "normalize_targets": bool(data_cfg.get("normalize_targets", False)),
        },
        "model": {
            "name": "neural_expert",
            "in_features": int(model_cfg["in_dim"]),
            "out_features": int(model_cfg["out_dim"]),
            "num_experts": int(model_cfg["n_experts"]),
            "top_k": int(model_cfg["top_k"]),
            "decoder_hidden_dim": int(model_cfg["decoder_hidden_dim"]),
            "decoder_n_hidden_layers": int(model_cfg["decoder_n_hidden_layers"]),
            "decoder_input_encoding": str(model_cfg["decoder_input_encoding"]),
            "decoder_nl": str(model_cfg["decoder_nl"]),
            "decoder_init_type": str(model_cfg["decoder_init_type"]),
            "decoder_freqs": float(model_cfg["decoder_freqs"]),
            "decoder_trainable_freqs": bool(model_cfg.get("decoder_trainable_freqs", False)),
            "manager_hidden_dim": int(model_cfg["manager_hidden_dim"]),
            "manager_n_hidden_layers": int(model_cfg["manager_n_hidden_layers"]),
            "manager_input_encoding": str(model_cfg["manager_input_encoding"]),
            "manager_nl": str(model_cfg["manager_nl"]),
            "manager_init": str(model_cfg["manager_init"]),
            "manager_softmax_temperature": float(model_cfg["manager_softmax_temperature"]),
            "manager_softmax_temp_trainable": bool(model_cfg["manager_softmax_temp_trainable"]),
            "manager_q_activation": str(model_cfg["manager_q_activation"]),
            "manager_clamp_q": float(model_cfg["manager_clamp_q"]),
            "manager_conditioning": str(model_cfg["manager_conditioning"]),
            "manager_type": str(model_cfg.get("manager_type", "standard")),
            "shared_encoder": bool(model_cfg.get("shared_encoder", False)),
        },
        "training": {
            "epochs": int(cfg["TRAINING"]["num_epochs"]),
            "batch_size": int(cfg["TRAINING"]["n_points"]),
            "pred_batch_size": int(cfg["TRAINING"]["n_points"]),
            "num_workers": 0,
            "lr": float(cfg["TRAINING"]["lr"]),
            "save_model": str(Path(validate_ckpt_path).as_posix()),
        },
    }


def _export_validate_artifacts(cfg, run_dir: Path, dataset, model):
    validate_dir = run_dir / "validate_artifacts"
    validate_dir.mkdir(parents=True, exist_ok=True)
    validate_ckpt_path = validate_dir / f"{cfg['exp_id']}.pth"
    validate_cfg_path = validate_dir / "config.yaml"
    _save_validate_checkpoint(model, dataset, validate_ckpt_path)
    dump_config(_build_validate_config(cfg, validate_ckpt_path), validate_cfg_path)
    return validate_cfg_path, validate_ckpt_path


def run_train(cfg: dict, *, gpu: int = 0) -> dict:
    run_dir = run_dir_from_config(cfg)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "wandb").mkdir(parents=True, exist_ok=True)
    (run_dir / "trained_models").mkdir(parents=True, exist_ok=True)
    dump_config(cfg, run_dir / "config.yaml")

    cfg = yaml.safe_load(yaml.safe_dump(cfg))
    cfg["TRAINING"]["n_samples"] = int(cfg["TRAINING"]["num_epochs"])
    seed_everything(int(cfg["seed"]))

    train_dataloader, train_set = build_dataloader(cfg, cfg["DATA"]["attr_name"], training=True)
    cfg["MODEL"]["out_dim"] = 1
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
        total_parameters, model_size_bytes = estimate_model_size_fp32(model)
        wandb.log(
            {
                "number of paramters": n_parameters,
                "model_size_bytes_fp32": model_size_bytes,
                "model_size_mib_fp32": model_size_bytes / (1024**2),
            }
        )
        log_string(f"Number of parameters in the current model:{n_parameters}", log_file)
        log_string(
            f"Model size assuming float32 parameters: {format_size_bytes(model_size_bytes)} "
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
        model_outdir = run_dir / "trained_models"
        save_interval = int(cfg["TRAINING"].get("save_every", 100) or 100)

        for step, data in enumerate(train_dataloader):
            if step >= max_epochs:
                break
            step_timer_start = time.perf_counter() if timing_enabled else None
            if step % save_interval == 0:
                torch.save(model.state_dict(), str(model_outdir / f"{cfg['MODEL']['model_name']}_model_{step}.pth"))

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

        final_state_path = model_outdir / f"{cfg['MODEL']['model_name']}_model_final.pth"
        torch.save(model.state_dict(), str(final_state_path))
        log_string(f"Saved final state dict to {final_state_path}", log_file)

        if cfg["TRAINING"].get("segmentation_mode", False) and cfg["MODEL"].get("manager_pt_path"):
            manager_pt_path = Path(cfg["MODEL"]["manager_pt_path"])
            manager_pt_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), str(manager_pt_path))
            log_string(f"Exported manager pretrain checkpoint to {manager_pt_path}", log_file)

        validate_cfg_path, validate_ckpt_path = _export_validate_artifacts(cfg, run_dir, train_set, model)
        log_string(f"Exported validate config to {validate_cfg_path}", log_file)
        log_string(f"Exported validate checkpoint to {validate_ckpt_path}", log_file)
        return {
            "run_dir": run_dir,
            "checkpoint_path": final_state_path,
            "validate_config_path": validate_cfg_path,
            "validate_checkpoint_path": validate_ckpt_path,
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
