from __future__ import annotations

import time
from pathlib import Path

import torch
import torch.nn.utils as nn_utils
import torch.optim as optim
import yaml

from ..common import dump_config, estimate_model_size_fp32, format_size_bytes, load_state_dict_payload, seed_everything, to_device
from ..config import run_dir_from_config
from ..runtime_limits import apply_runtime_thread_limits, configure_threading_env
from ..train_utils import count_parameters, log_losses_wandb, log_string
from ..wandb_stub import get_wandb
from .datasets import build_dataloader
from .model_registry import build_model
from .stage_handler import TrainingStageHandler

configure_threading_env()
THREAD_LIMITS = apply_runtime_thread_limits()
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


def _sync_device_for_timing(device: torch.device) -> None:
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)


def _format_elapsed(seconds: float) -> str:
    seconds = max(float(seconds), 0.0)
    hours, rem = divmod(seconds, 3600.0)
    minutes, secs = divmod(rem, 60.0)
    if hours >= 1.0:
        return f"{int(hours):02d}:{int(minutes):02d}:{secs:05.2f}"
    return f"{int(minutes):02d}:{secs:05.2f}"


def _save_validate_checkpoint(model, dataset, out_path):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state": model.state_dict(),
        "x_mean": dataset.x_mean.cpu().numpy(),
        "x_std": dataset.x_std.cpu().numpy(),
        "y_mean": dataset.y_mean.cpu().numpy(),
        "y_std": dataset.y_std.cpu().numpy(),
        "attr_name": dataset.attr_name,
        "association": dataset.association,
    }
    torch.save(payload, str(out_path))


def _export_validate_artifacts(cfg, run_dir: Path, dataset, model):
    validate_dir = run_dir / "validate_artifacts"
    validate_dir.mkdir(parents=True, exist_ok=True)
    validate_ckpt_path = validate_dir / f"{cfg['exp_id']}.pth"
    validate_cfg_path = validate_dir / "config.yaml"
    _save_validate_checkpoint(model, dataset, validate_ckpt_path)
    serializable_cfg = dict(cfg)
    if "device" in serializable_cfg:
        serializable_cfg["device"] = str(serializable_cfg["device"])
    validate_cfg = yaml.safe_load(yaml.safe_dump(serializable_cfg))
    validate_cfg["MODEL"]["in_dim"] = int(dataset.input_dim)
    validate_cfg["MODEL"]["out_dim"] = int(dataset.target_dim)
    validate_cfg["VALIDATION"] = {
        "checkpoint_path": str(validate_ckpt_path),
        "attr_name": dataset.attr_name,
        "association": dataset.association,
    }
    dump_config(validate_cfg, validate_cfg_path)
    return validate_cfg_path, validate_ckpt_path


def run_train(cfg: dict, *, gpu: int = 0) -> dict:
    run_dir = run_dir_from_config(cfg)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "wandb").mkdir(parents=True, exist_ok=True)
    (run_dir / "trained_models").mkdir(parents=True, exist_ok=True)

    cfg = yaml.safe_load(yaml.safe_dump(cfg))
    cfg["TRAINING"]["n_samples"] = int(cfg["TRAINING"]["num_epochs"])
    seed_everything(int(cfg["seed"]))

    train_dataloader, train_set = build_dataloader(cfg, cfg["DATA"]["attr_name"], training=True)
    cfg["MODEL"]["in_dim"] = int(train_set.input_dim)
    cfg["MODEL"]["out_dim"] = int(train_set.target_dim)
    dump_config(cfg, run_dir / "config.yaml")

    wandb_run = wandb.init(
        project=f"{cfg.get('wandb_project', 'inr_moe_mesh')}_{cfg['DATA']['dataset_name']}_{cfg['DATA']['attr_name']}",
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
    print(f"torch version: {torch.__version__}")
    print(f"thread limits: intra_op={THREAD_LIMITS[0]}, inter_op={THREAD_LIMITS[1]}")

    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    cfg["device"] = device
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
    log_string(
        f"Thread limits active: intra_op={THREAD_LIMITS[0]}, inter_op={THREAD_LIMITS[1]}",
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
    log_interval = int(cfg["TRAINING"].get("log_every", 100) or 100)
    grad_clip_norm = float(cfg["TRAINING"].get("grad_clip_norm", 0.0) or 0.0)
    train_started_at = time.perf_counter()
    last_log_at = train_started_at
    last_logged_step = -1

    for step, data in enumerate(train_dataloader):
        if step >= int(cfg["TRAINING"]["num_epochs"]):
            break
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
        loss_dict["loss"].backward()
        if grad_clip_norm > 0:
            nn_utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
        optimizer.step()
        scheduler.step()

        if step % log_interval == 0:
            _sync_device_for_timing(device)
            logged_at = time.perf_counter()
            total_elapsed = logged_at - train_started_at
            window_elapsed = logged_at - last_log_at
            steps_since_last_log = step - last_logged_step
            avg_step_time = window_elapsed / max(steps_since_last_log, 1)
            log_string(
                f"{step:05d} "
                + lossdict2str(loss_dict)
                + (
                    f"time_total: {_format_elapsed(total_elapsed)}, "
                    f"time_window: {window_elapsed:.2f}s, "
                    f"time_per_step: {avg_step_time:.4f}s"
                ),
                log_file,
            )
            last_log_at = logged_at
            last_logged_step = step

        if step > training_stage_handler.get_end_iteration():
            log_string("Moved to the next training stage...", log_file)
            training_stage_handler.move_to_the_next_training_stage(optimizer, scheduler)
            criterion = training_stage_handler.criterion

    _sync_device_for_timing(device)
    training_elapsed = time.perf_counter() - train_started_at
    total_steps = min(int(cfg["TRAINING"]["num_epochs"]), step + 1 if "step" in locals() else 0)
    avg_step_time = training_elapsed / max(total_steps, 1)
    log_string(
        f"Training loop finished: steps={total_steps}, total_time={_format_elapsed(training_elapsed)}, "
        f"avg_step_time={avg_step_time:.4f}s",
        log_file,
    )

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
    log_file.close()
    return {
        "run_dir": run_dir,
        "checkpoint_path": final_state_path,
        "validate_config_path": validate_cfg_path,
        "validate_checkpoint_path": validate_ckpt_path,
    }
