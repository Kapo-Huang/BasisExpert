from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import shutil

import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs"
REPO_ROOT_TOKEN = "${REPO_ROOT}"
SIZES = {
    "Size082": 0.82,
    "Size163": 1.63,
    "Size326": 3.26,
    "Size652": 6.52,
    "Size1304": 13.04,
}
IONIZATION_VARIABLE_COUNT = 5
SINGLE_TARGET_SIZES = {
    name: size_mib / IONIZATION_VARIABLE_COUNT for name, size_mib in SIZES.items()
}
ION_SHAPE = {"X": 600, "Y": 248, "Z": 248, "T": 100}
DATASETS = {
    "bathymetry": {
        "kind": "node",
        "dir": "Mesh/Bathymetry",
        "coords": "source_XYZT.npy",
        "targets": ["SALT", "TEMP", "U", "V"],
    },
    "katrina": {
        "kind": "node",
        "dir": "Mesh/Katrina",
        "coords": "source_XYZT.npy",
        "targets": ["fort63", "fort64", "fort73", "speed", "v"],
    },
    "ionization": {
        "kind": "volume",
        "dir": "Volume/Ionization",
        "targets": ["GT", "H_plus", "H2", "He", "PD"],
    },
}
COMBUSTION_DATASET = {
    "name": "combustion_40NH3_1",
    "volume_shape": {"X": 128, "Y": 128, "Z": 1, "T": 2001},
    "coordinate_axes": ["x", "y", "t"],
    "targets": [
        "Absolute_Pressure",
        "Chemistry_Heat_Release_Rate",
        "Mole_Fraction_of_CH4",
        "Mole_Fraction_of_CO",
        "Mole_Fraction_of_CO2",
        "Mole_Fraction_of_H2O",
        "Mole_Fraction_of_NH2",
        "Mole_Fraction_of_NH3",
        "Mole_Fraction_of_OH",
        "Pressure",
        "Temperature",
        "Velocity",
        "Velocity_Magnitude",
    ],
}
TARGET_FILES = {"H_plus": "H+"}
UNIFIED_SIZE_MODELS = {
    "SIREN": {
        "Size082": {"name": "siren", "in_features": 4, "hidden_features": 168, "hidden_layers": 3, "first_omega_0": 30.0, "hidden_omega_0": 30.0, "outermost_linear": True},
        "Size163": {"name": "siren", "in_features": 4, "hidden_features": 237, "hidden_layers": 3, "first_omega_0": 30.0, "hidden_omega_0": 30.0, "outermost_linear": True},
        "Size326": {"name": "siren", "in_features": 4, "hidden_features": 336, "hidden_layers": 3, "first_omega_0": 30.0, "hidden_omega_0": 30.0, "outermost_linear": True},
        "Size652": {"name": "siren", "in_features": 4, "hidden_features": 412, "hidden_layers": 4, "first_omega_0": 30.0, "hidden_omega_0": 30.0, "outermost_linear": True},
        "Size1304": {"name": "siren", "in_features": 4, "hidden_features": 522, "hidden_layers": 5, "first_omega_0": 30.0, "hidden_omega_0": 30.0, "outermost_linear": True},
    },
    "CoordNet": {
        "Size082": {"name": "coordnet", "in_features": 4, "init_features": 15, "num_res": 10},
        "Size163": {"name": "coordnet", "in_features": 4, "init_features": 22, "num_res": 10},
        "Size326": {"name": "coordnet", "in_features": 4, "init_features": 31, "num_res": 10},
        "Size652": {"name": "coordnet", "in_features": 4, "init_features": 43, "num_res": 10},
        "Size1304": {"name": "coordnet", "in_features": 4, "init_features": 61, "num_res": 10},
    },
    "MoE-INR": {
        "Size082": {"name": "moe_inr", "in_features": 4, "num_experts": 7, "base_dim": 32, "policy_num_layers": 3},
        "Size163": {"name": "moe_inr", "in_features": 4, "num_experts": 7, "base_dim": 46, "policy_num_layers": 3},
        "Size326": {"name": "moe_inr", "in_features": 4, "num_experts": 7, "base_dim": 66, "policy_num_layers": 3},
        "Size652": {"name": "moe_inr", "in_features": 4, "num_experts": 7, "base_dim": 93, "policy_num_layers": 3},
        "Size1304": {"name": "moe_inr", "in_features": 4, "num_experts": 7, "base_dim": 132, "policy_num_layers": 3},
    },
}
DEFAULT_MODELS = {
    "SIREN": {
        "node": {"name": "siren", "in_features": 4, "hidden_features": 56, "hidden_layers": 5, "first_omega_0": 30.0, "hidden_omega_0": 30.0, "outermost_linear": True},
        "volume": {"name": "siren", "in_features": 4, "hidden_features": 256, "hidden_layers": 3, "first_omega_0": 30.0, "hidden_omega_0": 30.0, "outermost_linear": True},
    },
    "CoordNet": {
        "node": {"name": "coordnet", "in_features": 4, "init_features": 7, "num_res": 8},
        "volume": {"name": "coordnet", "in_features": 4, "init_features": 21, "num_res": 10},
    },
    "MoE-INR": {
        "node": {"name": "moe_inr", "in_features": 4, "num_experts": 7, "base_dim": 14, "policy_num_layers": 3},
        "volume": {"name": "moe_inr", "in_features": 4, "num_experts": 7, "base_dim": 45, "policy_num_layers": 3},
    },
}
VAR_SIZE_DIMS = {"Size082": 17, "Size163": 24, "Size326": 34, "Size652": 49, "Size1304": 70}
NEURAL_SIZE_DIMS = {"Size082": 16, "Size163": 23, "Size326": 33, "Size652": 47, "Size1304": 66}
MC_SIZE_DIMS = {"Size082": 23, "Size163": 33, "Size326": 48, "Size652": 68, "Size1304": 97}
APMG_SIZE = {
    "Size082": {"feature_grid_shape": [7, 7, 7], "n_grids": 1, "n_features": 1, "nodes_per_layer": 16, "n_layers": 2},
    "Size163": {"feature_grid_shape": [5, 5, 5], "n_grids": 6, "n_features": 1, "nodes_per_layer": 16, "n_layers": 3},
    "Size326": {"feature_grid_shape": [5, 5, 5], "n_grids": 12, "n_features": 1, "nodes_per_layer": 32, "n_layers": 2},
    "Size652": {"feature_grid_shape": [4, 4, 4], "n_grids": 44, "n_features": 1, "nodes_per_layer": 32, "n_layers": 3},
    "Size1304": {"feature_grid_shape": [4, 4, 4], "n_grids": 20, "n_features": 5, "nodes_per_layer": 32, "n_layers": 4},
}
FV_SIZE = {
    "Size082": (5, 54), "Size163": (6, 64), "Size326": (8, 55),
    "Size652": (8, 110), "Size1304": (11, 85),
}
RM_SIZE = {
    "Size082": (5, 30), "Size163": (6, 48), "Size326": (7, 70),
    "Size652": (17, 11), "Size1304": (11, 82),
}


def dump(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")


def rel_prefix(nested: bool) -> str:
    del nested
    return f"{REPO_ROOT_TOKEN}/"


def repo_path(relative: str) -> str:
    return f"{REPO_ROOT_TOKEN}/{relative.strip('/')}"


def targets_for(dataset: str, nested: bool) -> dict[str, str]:
    prefix = rel_prefix(nested)
    meta = DATASETS[dataset]
    return {
        target: f"{prefix}data/{meta['dir']}/target_{TARGET_FILES.get(target, target)}.npy"
        for target in meta["targets"]
    }


def unified_data(dataset: str, nested: bool, target: str | None = None) -> dict:
    meta = DATASETS[dataset]
    payload = {"kind": meta["kind"], "dataset_name": dataset, "split": "train"}
    if meta["kind"] == "node":
        payload["coords_path"] = f"{rel_prefix(nested)}data/{meta['dir']}/{meta['coords']}"
    else:
        payload["volume_shape"] = deepcopy(ION_SHAPE)
    payload["targets"] = targets_for(dataset, nested)
    if target is not None:
        payload["target"] = target
    return payload


def combustion_data() -> dict:
    dataset_name = COMBUSTION_DATASET["name"]
    return {
        "kind": "volume",
        "dataset_name": dataset_name,
        "split": "train",
        "volume_shape": deepcopy(COMBUSTION_DATASET["volume_shape"]),
        "coordinate_axes": list(COMBUSTION_DATASET["coordinate_axes"]),
        "targets": {
            target: repo_path(f"data/Volume/Combustion/target_{target}.npy")
            for target in COMBUSTION_DATASET["targets"]
        },
    }


def common_training() -> dict:
    return {
        "epochs": 600, "batch_size": 16000, "pred_batch_size": 16000,
        "num_workers": 0, "lr": 5.0e-5, "val_split": 0.0,
        "log_every": 10, "log_psnr_every": 100, "psnr_sample_ratio": 0.1,
        "save_every": 600, "early_stop_patience": 0, "loss_type": "mse",
        "seed": 42, "sampler": "budgeted_random", "batches_per_epoch_budget": 1500,
        "scheduler": {"enabled": True, "step_size": 40, "gamma": 0.92},
    }


def compact_ngp_model() -> dict:
    return {
        "name": "compact_ngp",
        "in_features": 4,
        "out_features": 1,
        "num_levels": 16,
        "features_per_level": 2,
        "feature_table_size": 1024,
        "index_table_size": 65536,
        "num_probes": 4,
        "base_resolution": 16,
        "max_resolution": 2048,
        "hidden_features": 64,
        "hidden_layers": 2,
    }


def compact_ngp_training() -> dict:
    payload = common_training()
    payload.update(
        {
            "lr": 1.0e-2,
            "beta_1": 0.9,
            "beta_2": 0.99,
            "epsilon": 1.0e-15,
            "weight_decay": 1.0e-6,
        }
    )
    return payload


def generate_compact_ngp() -> int:
    count = 0
    for target in DATASETS["ionization"]["targets"]:
        payload = {
            "experiment": f"ionization_compact-ngp_{target}",
            "exp_id": f"compact-ngp-ionization-{target}",
            "experiment_root": repo_path("runs"),
            "data": unified_data("ionization", False, target),
            "model": compact_ngp_model(),
            "training": compact_ngp_training(),
            "evaluation": evaluation(),
            "log": log_config(),
        }
        dump(CONFIGS / "CompactNGP" / f"ionization__{target}.yaml", payload)
        count += 1
    return count


def instant_ngp_model() -> dict:
    return {
        "name": "instant_ngp",
        "in_features": 4,
        "out_features": 1,
        "n_levels": 16,
        "n_features_per_level": 2,
        "base_resolution": 16,
        "finest_resolution": 600,
        "log2_hashmap_size": 19,
        "hidden_features": 64,
        "hidden_layers": 2,
    }


def instant_ngp_training() -> dict:
    payload = common_training()
    payload.update(
        {
            "gradient_accumulation_steps": 16,
            "lr": 1.0e-2,
            "beta_1": 0.9,
            "beta_2": 0.99,
            "epsilon": 1.0e-15,
            "weight_decay": 0.0,
            "scheduler": {
                "enabled": True,
                "interval": "optimizer_step",
                "milestones": [20480, 30720],
                "gamma": 0.33,
            },
            "pretrain": {"enabled": False},
        }
    )
    return payload


def generate_instant_ngp() -> int:
    count = 0
    all_targets = targets_for("ionization", False)
    for target in DATASETS["ionization"]["targets"]:
        data = {
            "kind": "volume",
            "dataset_name": "ionization",
            "split": "train",
            "volume_shape": deepcopy(ION_SHAPE),
            "targets": {target: all_targets[target]},
        }
        payload = {
            "experiment": f"ionization_instant_ngp_{target}",
            "exp_id": f"instant-ngp-ionization-{target}",
            "experiment_root": repo_path("runs"),
            "data": data,
            "model": instant_ngp_model(),
            "training": instant_ngp_training(),
            "evaluation": evaluation(),
            "log": log_config(),
        }
        dump(
            CONFIGS / "InstantNGP" / f"ionization__{target}.yaml",
            payload,
        )
        count += 1
    return count


def mvnet_model(num_variables: int) -> dict:
    return {
        "name": "mvnet",
        "in_features": 4,
        "out_features": int(num_variables),
        "hidden_features": 120,
        "num_residual_blocks": 10,
        "omega_0": 30.0,
        "bias": True,
    }


def mvnet_training() -> dict:
    return {
        "epochs": 300,
        "batch_size": 2048,
        "pred_batch_size": 16000,
        "gradient_accumulation_steps": 1,
        "num_workers": 0,
        "lr": 1.0e-4,
        "beta_1": 0.9,
        "beta_2": 0.999,
        "epsilon": 1.0e-8,
        "weight_decay": 0.0,
        "val_split": 0.0,
        "log_every": 10,
        "log_psnr_every": 0,
        "psnr_sample_ratio": 0.1,
        "save_every": 300,
        "early_stop_patience": 0,
        "loss_type": "mse",
        "seed": 42,
        "sampler": "budgeted_random",
        "batches_per_epoch_budget": 1500,
        "scheduler": {
            "enabled": True,
            "interval": "epoch",
            "step_size": 15,
            "gamma": 0.8,
        },
        "pretrain": {"enabled": False},
    }


def generate_mvnet() -> int:
    count = 0
    for dataset, meta in DATASETS.items():
        data = unified_data(dataset, False)
        data["targets"] = {
            name: data["targets"][name]
            for name in sorted(data["targets"])
        }
        payload = {
            "experiment": f"{dataset}_mvnet",
            "exp_id": f"mvnet-{dataset}",
            "experiment_root": repo_path("runs"),
            "data": data,
            "model": mvnet_model(len(meta["targets"])),
            "training": mvnet_training(),
            "evaluation": evaluation(),
            "log": log_config(),
        }
        dump(CONFIGS / "MVNet" / f"{dataset}.yaml", payload)
        count += 1
    return count


def fa_tr_inr_model() -> dict:
    return {
        "name": "fa_tr_inr",
        "in_features": 4,
        "out_features": 1,
        "frequency_coordinates": [1.0, 2.0, 3.0],
        "omega": 19.0,
        "factor_mlp_depth": 4,
        "factor_hidden_width": 128,
        "integration_mlp_depth": 2,
        "tensor_ring_ranks": [22, 88, 3, 3, 5],
    }


def fa_tr_inr_training() -> dict:
    payload = common_training()
    payload.update(
        {
            "lr": 1.0e-4,
            "beta_1": 0.9,
            "beta_2": 0.999,
            "epsilon": 1.0e-8,
            "weight_decay": 0.0,
            "scheduler": {
                "enabled": False,
                "step_size": 0,
                "gamma": 1.0,
            },
        }
    )
    return payload


def generate_fa_tr_inr() -> int:
    count = 0
    for target in DATASETS["ionization"]["targets"]:
        payload = {
            "experiment": f"ionization_fa-tr-inr_{target}",
            "exp_id": f"fa-tr-inr-ionization-{target}",
            "experiment_root": repo_path("runs"),
            "data": unified_data("ionization", False, target),
            "model": fa_tr_inr_model(),
            "training": fa_tr_inr_training(),
            "evaluation": evaluation(),
            "log": log_config(),
        }
        dump(
            CONFIGS / "FA-TR-INR" / f"ionization__{target}.yaml",
            payload,
        )
        count += 1
    return count


def evaluation() -> dict:
    return {"batch_size": 16000, "save_predictions": False}


def log_config() -> dict:
    return {
        "timing": {
            "enabled": True,
            "epoch_breakdown": True,
            "step_window": False,
            "step_window_every_steps": 100,
            "cuda_sync": False,
        }
    }


def generate_unified_single() -> int:
    count = 0
    for family, defaults in DEFAULT_MODELS.items():
        model_slug = defaults["volume"]["name"].replace("_", "-")
        for dataset, meta in DATASETS.items():
            for target in meta["targets"]:
                payload = {
                    "experiment": f"{dataset}_{model_slug}_{target}",
                    "exp_id": f"{model_slug}-{dataset}-{target}",
                    "experiment_root": repo_path("runs"),
                    "data": unified_data(dataset, False, target),
                    "model": deepcopy(defaults[meta["kind"]]),
                    "training": common_training(),
                    "evaluation": evaluation(),
                    "log": log_config(),
                }
                dump(CONFIGS / family / f"{dataset}__{target}.yaml", payload)
                count += 1
        for size in SIZES:
            for target in DATASETS["ionization"]["targets"]:
                payload = {
                    "experiment": f"ionization_{model_slug}_{size.lower()}_{target}",
                    "exp_id": f"{model_slug}-ionization-{size.lower()}-{target}",
                    "experiment_root": repo_path("runs"),
                    "data": unified_data("ionization", True, target),
                    "model": deepcopy(UNIFIED_SIZE_MODELS[family][size]),
                    "training": common_training(),
                    "evaluation": evaluation(),
                    "log": log_config(),
                }
                dump(CONFIGS / family / size / f"ionization__{target}.yaml", payload)
                count += 1
    return count


def var_training(dataset: str, nested: bool, experts: int = 6) -> dict:
    pretrain_enabled = dataset in {"ionization", COMBUSTION_DATASET["name"]}
    payload = common_training()
    payload["multiview_ema_loss"] = {
        "enabled": True, "beta": 0.95, "eps": 1.0e-8,
        "w_min": 0.2, "w_max": 5.0, "warmup_steps": 45000,
        "alpha": 5.0,
    }
    payload["pretrain"] = {
        "enabled": pretrain_enabled,
        "epochs": 5 if pretrain_enabled else 0,
        "lr": 5.0e-5,
    }
    if pretrain_enabled:
        payload["pretrain"]["cluster_seed"] = 42
        payload["pretrain"]["assignments_cache_path"] = (
            f"{rel_prefix(nested)}data/cache/{dataset}_voxel_assignments_{experts}.npy"
        )
    return payload


def generate_var_expert() -> int:
    count = 0
    for dataset, meta in DATASETS.items():
        base_dim = 24 if dataset == "ionization" else 8
        payload = {
            "experiment": f"{dataset}_var_expert",
            "exp_id": f"var-expert-{dataset}",
            "experiment_root": repo_path("runs"),
            "data": unified_data(dataset, False),
            "model": {"name": "var_expert", "in_features": 4, "num_experts": 6, "base_dim": base_dim, "top_k": 3},
            "training": var_training(dataset, False),
            "evaluation": evaluation(),
            "log": log_config(),
        }
        dump(CONFIGS / "VarExpert" / f"{dataset}.yaml", payload)
        count += 1
        if dataset == "ionization":
            dwa_payload = deepcopy(payload)
            dwa_payload["experiment"] = "ionization_var_expert_dwa"
            dwa_payload["exp_id"] = "var-expert-ionization-dwa"
            dwa_payload["training"]["multiview_ema_loss"]["enabled"] = False
            dwa_payload["training"]["multiview_dwa_loss"] = {
                "enabled": True,
                "temperature": 0.2,
                "eps": 1.0e-12,
                "warmup_epochs": 2,
                "max_factor_max": 1.25,
                "max_factor_min": 1.05,
                "update_schedule": "cosine",
            }
            dump(CONFIGS / "VarExpert" / "ionization_dwa.yaml", dwa_payload)
            count += 1
    combustion_name = COMBUSTION_DATASET["name"]
    combustion_payload = {
        "experiment": f"{combustion_name}_var_expert",
        "exp_id": "var-expert-combustion-40NH3-1",
        "experiment_root": repo_path("runs"),
        "data": combustion_data(),
        "model": {
            "name": "var_expert",
            "in_features": 3,
            "num_experts": 6,
            "base_dim": 24,
            "top_k": 3,
        },
        "training": var_training(combustion_name, False),
        "evaluation": evaluation(),
        "log": log_config(),
    }
    dump(CONFIGS / "VarExpert" / f"{combustion_name}.yaml", combustion_payload)
    count += 1
    for size, dim in VAR_SIZE_DIMS.items():
        payload = {
            "experiment": f"ionization_var_expert_{size.lower()}",
            "exp_id": f"var-expert-ionization-{size.lower()}",
            "experiment_root": repo_path("runs"),
            "data": unified_data("ionization", True),
            "model": {"name": "var_expert", "in_features": 4, "num_experts": 6, "base_dim": dim, "top_k": 3},
            "training": var_training("ionization", True),
            "evaluation": evaluation(),
            "log": log_config(),
        }
        dump(CONFIGS / "VarExpert" / size / "ionization.yaml", payload)
        count += 1
    return count


def mc_payload(dataset: str, nested: bool, hidden: int) -> dict:
    meta = DATASETS[dataset]
    return {
        "experiment": f"{dataset}_mc_inr",
        "exp_id": f"mc-inr-{dataset}" + (f"-h{hidden}" if nested else ""),
        "experiment_root": repo_path("runs"),
        "data": unified_data(dataset, nested),
        "model": {"name": "mc_inr", "hidden_features": hidden, "gfe_layers": 5, "lfe_layers": 6},
        "training": {
            "epochs": 60, "batch_size": 16000, "pred_batch_size": 16000,
            "num_workers": 0, "lr": 5.0e-5, "weight_decay": 0.0,
            "loss_type": "mse", "log_every": 10, "save_every": 0,
            "seed": 42, "device": "cuda", "initial_k": 12,
            "cluster_init_method": "voxel_clustering" if meta["kind"] == "volume" else "coord_kmeans",
            "assignments_cache_path": f"{rel_prefix(nested)}data/cache/mc_inr/{dataset}_assignments_k12.npy",
            "meta_iterations": 60, "meta_inner_steps": 5, "meta_inner_batch_size": 16000,
            "meta_inner_lr": 1.0e-4, "meta_batch_clusters": 4,
            "meta_support_max_rows": 32000, "meta_outer_lr": 1.0e-3,
            "convergence_patience": 0, "convergence_delta": 0.0,
            "finetune_epochs": 600, "batches_per_epoch_budget": 1500,
            "finetune_lr": 5.0e-5, "finetune_sampling_ratio": 1.0,
            "recluster_after_finetune": False, "split_threshold": 5.0e-4,
            "min_split_points": 32, "max_recluster_rounds": 0,
            "cluster_aware_batches": False,
            "scheduler": {"enabled": True, "step_size": 40, "gamma": 0.92},
        },
        "evaluation": evaluation(),
        "log": log_config(),
    }


def generate_mc() -> int:
    count = 0
    for dataset in DATASETS:
        dump(CONFIGS / "MC-INR" / f"{dataset}.yaml", mc_payload(dataset, False, 128 if dataset == "ionization" else 96))
        count += 1
    for size, hidden in MC_SIZE_DIMS.items():
        payload = mc_payload("ionization", True, hidden)
        payload["experiment"] = f"ionization_mc_inr_{size.lower()}"
        payload["exp_id"] = f"mc-inr-ionization-{size.lower()}"
        dump(CONFIGS / "MC-INR" / size / "ionization.yaml", payload)
        count += 1
    return count


def neural_model(dataset: str, dim: int, nested: bool, target: str, manager_pretrain: bool, size: str | None) -> dict:
    model_name = "inr_moe_ionization" if dataset == "ionization" else "inr_moe_mesh"
    size_token = size.lower() if size else "default"
    manager_path = (
        f"{rel_prefix(nested)}runs/neural_expert/pretrained_managers/{dataset}/{size_token}/"
        f"pt_{model_name}_{target}_managerpretraining.pth"
    )
    return {
        "model_name": model_name, "in_dim": 4, "out_dim": 1,
        "decoder_hidden_dim": dim, "decoder_n_hidden_layers": 2,
        "decoder_input_encoding": f"learned_{dim * 8}_2_sine_siren_none",
        "decoder_nl": "sine", "decoder_init_type": "siren", "n_experts": 8,
        "outermost_linear": True, "input_encoding": "none", "decoder_freqs": 30.0,
        "decoder_trainable_freqs": False, "top_k": 1,
        "manager_hidden_dim": dim * 2, "manager_n_hidden_layers": 2,
        "manager_input_encoding": f"learned_{dim * 2}_2_sine_siren_none",
        "manager_nl": "sine", "manager_init": "siren", "manager_type": "standard",
        "experts_bias_std": 0.1, "experts_bias_weight": 1.0,
        "manager_softmax_temperature": 1.0, "manager_softmax_temp_trainable": False,
        "manager_q_activation": "softmax", "manager_clamp_q": 0.0,
        "manager_conditioning": "cat", "manager_pt_path": manager_path,
        "load_pt_manager": not manager_pretrain, "shared_encoder": False,
    }


def neural_data(dataset: str, target: str, nested: bool) -> dict:
    data = {
        "dataset_name": dataset, "target": target, "targets": targets_for(dataset, nested),
        "target_stats_path": f"{rel_prefix(nested)}data/cache/neural_expert/{dataset}/target_stats_{target}.npz",
        "normalize_inputs": dataset == "ionization", "normalize_targets": False,
    }
    if dataset == "ionization":
        data.update({"volume_shape": deepcopy(ION_SHAPE), "segmentation_type": "random_balanced", "grid_patch_size": 4, "n_segments": 8})
    else:
        meta = DATASETS[dataset]
        data.update({
            "association": "point",
            "source_path": f"{rel_prefix(nested)}data/{meta['dir']}/{meta['coords']}",
            "stats_key": target,
        })
    return data


def neural_payload(dataset: str, target: str, nested: bool, manager_pretrain: bool, dim: int, size: str | None) -> dict:
    suffix = "-managerpretrain" if manager_pretrain else ""
    exp_size = f"-{size.lower()}" if size else ""
    loss_name = "1000segmentation" if manager_pretrain else "1000valrecon"
    training = {
        "n_points": 16000, "lr": 3.0e-5, "lr_gamma": 0.9999,
        "lr_scheduler": "ExponentialLR",
        "num_epochs": 30000 if manager_pretrain else 900000,
        "batch_size": 1, "num_workers": 0, "grad_clip_norm": 10.0,
        "save_every": 500, "segmentation_mode": manager_pretrain,
        "log_every": 100,
        "stages": [{"end_iteration_frac": 1.0, "params": "all", "loss_type": loss_name}],
    }
    if dataset != "ionization":
        training["pretrain_assignment"] = {
            "method": "coord_kmeans", "fit_samples": 50000,
            "cache_path": f"{rel_prefix(nested)}data/cache/neural_expert/{dataset}/coord_kmeans_{target}.npz",
            "normalize_features": False, "random_seed": 0, "chunk_size": 65536,
        }
    return {
        "seed": 0, "wandb_project": f"inr_moe_{dataset}",
        "experiment": f"neural_expert_{dataset}{exp_size}_{target}{suffix}",
        "exp_id": f"neural-expert-{dataset}{exp_size}-{target}{suffix}",
        "experiment_root": f"{rel_prefix(nested)}runs/neural_expert",
        "MODEL": neural_model(dataset, dim, nested, target, manager_pretrain, size),
        "LOSS": {
            "scale_by_q_grad": False, "loss_type": loss_name,
            "segmentation_type": "both" if dataset == "ionization" else "ce",
            "sample_bias_correction": False, "entropy_metric": "kl",
        },
        "DATA": neural_data(dataset, target, nested),
        "TRAINING": training,
    }


def generate_neural() -> int:
    count = 0
    for dataset, meta in DATASETS.items():
        for target in meta["targets"]:
            for pretrain in (True, False):
                suffix = "__managerpretrain" if pretrain else ""
                dump(
                    CONFIGS / "NeuralExpert" / f"{dataset}__{target}{suffix}.yaml",
                    neural_payload(dataset, target, False, pretrain, 64, None),
                )
                count += 1
    for size, dim in NEURAL_SIZE_DIMS.items():
        for target in DATASETS["ionization"]["targets"]:
            for pretrain in (True, False):
                suffix = "__managerpretrain" if pretrain else ""
                dump(
                    CONFIGS / "NeuralExpert" / size / f"ionization__{target}{suffix}.yaml",
                    neural_payload("ionization", target, True, pretrain, dim, size),
                )
                count += 1
    return count


def volume_data(target: str, nested: bool, upper: bool = False) -> dict:
    key = "DATA" if upper else "data"
    del key
    return {
        "kind": "volume", "dataset_name": "ionization", "split": "train",
        "target": target, "targets": targets_for("ionization", nested),
        "volume_shape": deepcopy(ION_SHAPE),
    }


def apmg_payload(target: str, nested: bool, size: str | None) -> dict:
    sizing = APMG_SIZE[size] if size else {
        "feature_grid_shape": [8, 8, 8], "n_grids": 64, "n_features": 2,
        "nodes_per_layer": 64, "n_layers": 2,
    }
    tag = f"-{size.lower()}" if size else ""
    return {
        "experiment": f"apmgsrn_ionization{tag}_{target}",
        "exp_id": f"apmgsrn-ionization{tag}-{target}",
        "experiment_root": f"{rel_prefix(nested)}runs",
        "MODEL": {
            "model_name": "apmgsrn", "n_dims": 3, "n_outputs": 1,
            "feature_grid_shape": deepcopy(sizing["feature_grid_shape"]),
            "n_features": sizing["n_features"],
            "n_grids": sizing["n_grids"],
            "nodes_per_layer": sizing["nodes_per_layer"],
            "n_layers": sizing["n_layers"],
            "use_bias": False, "use_tcnn_if_available": True,
            "grid_initialization": "default",
            "requires_padded_feats": True if size else None,
        },
        "DATA": {
            "dataset_name": "ionization", "target": target,
            "targets": targets_for("ionization", nested),
            "volume_shape": deepcopy(ION_SHAPE), "align_corners": True,
        },
        "TRAINING": {
            "iterations": 9000, "points_per_iteration": 16000,
            "prediction_points_per_batch": 16000, "lr": 0.01,
            "beta_1": 0.9, "beta_2": 0.99, "device": "cuda:0",
            "data_device": "same", "save_every": 0, "log_every": 100,
            "time_indices": "all", "seed": 42, "early_stopping": False,
        },
        "EVALUATION": {"run_after_training": False},
    }


def dc_payload(target: str, nested: bool, size: str | None) -> dict:
    tag = f"-{size.lower()}" if size else ""
    compression = {"max_initial_neurons": 2048, "min_initial_neurons": 4}
    if size:
        compression["target_size_mib"] = SINGLE_TARGET_SIZES[size]
    else:
        compression["target_cr"] = 20.0
    return {
        "experiment": f"dc_inr_ionization{tag}_{target}",
        "exp_id": f"dc-inr-ionization{tag}-{target}",
        "experiment_root": repo_path("runs/dc_inr"),
        "data": volume_data(target, nested),
        "model": {"name": "dc_inr"},
        "partition": {
            "candidate_block_shapes": [
                {"sx": 150, "sy": 8, "sz": 124}, {"sx": 150, "sy": 124, "sz": 8},
                {"sx": 300, "sy": 4, "sz": 124}, {"sx": 300, "sy": 124, "sz": 4},
            ],
            "dbscan_eps": 1.0e-2, "dbscan_min_samples": 1,
            "entropy_bins": 256, "distance_matrix_max_bytes": 1073741824,
        },
        "compression": compression,
        "training": {
            "epochs": 300, "total_steps": 900000, "batch_size": 16000,
            "lr": 1.0e-4, "beta_1": 0.9, "beta_2": 0.999,
            "points_per_timestep": 160, "prediction_batch_size": 16000,
            "lr_milestones": [450000, 675000], "lr_gamma": 0.5,
            "log_every": 100, "seed": 42, "device": "cuda",
        },
        "evaluation": evaluation(),
        "log": log_config(),
    }


def fv_payload(target: str, nested: bool, size: str | None) -> dict:
    resolution, channels = FV_SIZE[size] if size else (32, 16)
    tag = f"-{size.lower()}" if size else ""
    return {
        "experiment": f"fv_srn_ionization{tag}_{target}",
        "exp_id": f"fv-srn-ionization{tag}-{target}",
        "experiment_root": repo_path("runs/fv_srn"),
        "data": volume_data(target, nested),
        "model": {
            "name": "fv_srn", "grid_resolution": resolution,
            "grid_channels": channels, "grid_init_std": 0.01,
            "keyframe_indices": [0, 9, 18, 27, 36, 45, 54, 63, 72, 81, 90, 99],
            "fourier_features": 14, "fourier_mode": "nerf",
            "hidden_features": 32, "hidden_layers": 3,
            "activation": "snake_alt", "activation_frequency": 1.0,
            "time_encoding": "none",
        },
        "training": {
            "epochs": 600, "samples_per_timestep": 240000,
            "validation_fraction": 0.0, "batch_size": 16000,
            "prediction_batch_size": 16000, "lr": 0.01,
            "beta_1": 0.9, "beta_2": 0.999, "lr_step": 100,
            "lr_gamma": 0.5, "l1_weight": 1.0, "l2_weight": 0.0,
            "importance_floor": 0.01, "rebuild_every": 51,
            "rebuild_grid_size": 32, "rebuild_samples_per_cell": 2,
            "save_every": 20, "log_every": 1, "seed": 42, "device": "cuda",
        },
        "evaluation": {**evaluation(), "run_after_training": False, "default_model": "compact"},
    }


def ecnr_payload(target: str) -> dict:
    return {
        "experiment": f"ecnr_ionization_{target}",
        "exp_id": f"ecnr-ionization-{target}",
        "experiment_root": repo_path("runs/ecnr"),
        "data": volume_data(target, False),
        "model": {
            "name": "ecnr", "scales": 3,
            "block_shape_xyz": [25, 31, 31],
            "residual_threshold": 1.0e-4,
            "gaussian_kernel_size": 5, "gaussian_sigma": 1.0,
            "gaussian_padding": "reflect", "latent_dim": 8,
            "hidden_features": 24, "hidden_layers": 3, "omega_0": 30.0,
            "target_blocks_per_mlp": [8, 16, 32],
        },
        "clustering": {
            "distance": "squared_euclidean",
            "input": "normalized_block_values",
            "initialization": "kmeans_pp", "seed": 42,
            "balancing_passes": 1, "centroid_dtype": "float32",
            "tie_break": "lowest_cluster_index", "n_init": 1,
            "max_iter": 300, "tol": 1.0e-4, "algorithm": "lloyd",
        },
        "training": {
            "epochs_per_scale": 500, "batch_size": 3200,
            "batches_per_epoch_budget": 3000,
            "primary_sample_budget": 14_400_000_000,
            "lr": 1.0e-3, "beta_1": 0.9, "beta_2": 0.999,
            "weight_decay": 2.0e-5,
            "pruning_epochs": [150, 225, 300, 375],
            "pruning_sparsities": [0.30, 0.40, 0.45, 0.50],
            "pruning_loss_weight": 0.1, "pruning_lr_gamma": 0.75,
            "quantization_finetune_epochs": 75,
            "quantization_finetune_batches_per_epoch": 0,
            "quantization_finetune_lr": 1.0e-5,
            "save_every": 0, "log_every": 10, "seed": 42, "device": "cuda",
        },
        "quantization": {
            "mlp_weight_bits": 8, "mlp_bias_bits": 8,
            "latent_bits": 0, "cnn_bits": 9, "entropy": "huffman",
        },
        "cnn": {
            "implementation_choice": "fixed_for_underspecified_method_detail",
            "dimensionality": "3d", "layers": 5, "hidden_channels": 32,
            "kernel_size": 3, "stride": 1, "padding": 1, "dilation": 1,
            "bias": True, "hidden_activation": "relu",
            "output_activation": "none", "halo": 5,
            "tile_core_shape_zyx": [32, 64, 64], "epochs": 100, "lr": 1.0e-5,
        },
        "evaluation": {
            "batch_size": 3200, "save_predictions": False,
            "run_after_training": False, "default_model": "artifact",
        },
        "log": {
            "effective_config": True, "epoch_summary": True,
            "startup_timing": True,
        },
    }


def generate_ecnr() -> int:
    for target in DATASETS["ionization"]["targets"]:
        dump(CONFIGS / "ECNR" / f"ionization__{target}.yaml", ecnr_payload(target))
    return len(DATASETS["ionization"]["targets"])


def rm_payload(target: str, nested: bool, size: str | None) -> dict:
    resolution, channels = RM_SIZE[size] if size else (32, 16)
    tag = f"-{size.lower()}" if size else ""
    return {
        "experiment": f"rmdsrn_ionization{tag}_{target}",
        "exp_id": f"rmdsrn-ionization{tag}-{target}",
        "experiment_root": repo_path("runs/rmdsrn"),
        "data": volume_data(target, nested),
        "model": {
            "name": "rmdsrn", "base_encoder": "temporal_fv_srn",
            "grid_resolution": resolution, "grid_channels": channels,
            "grid_init_std": 0.01,
            "keyframe_indices": [0, 9, 18, 27, 36, 45, 54, 63, 72, 81, 90, 99],
            "fourier_features": 14, "fourier_mode": "nerf",
            "decoder_count": 5, "decoder_hidden_features": 64,
            "decoder_hidden_layers": 2, "activation": "snake_alt",
            "activation_frequency": 1.0,
        },
        "training": {
            "steps": 900000, "batch_size": 16000, "lr": 5.0e-3,
            "beta_1": 0.9, "beta_2": 0.999, "min_lr": 1.0e-7,
            "lambda_min": 0.0, "lambda_max": 10.0,
            "lambda_growth_rate": 500.0, "epsilon": 1.0e-12,
            "save_every": 5000, "log_every": 100, "seed": 42, "device": "cuda",
        },
        "evaluation": {
            "batch_size": 16000, "save_mean": True, "save_variance": True,
            "run_after_training": False, "default_model": "artifact",
            "uncertainty_sample_size": 1000000, "topk_fractions": [0.01, 0.05],
            "seed": 42,
        },
    }


def generate_volume_only() -> int:
    count = 0
    builders = {
        "APMGSRN": apmg_payload, "DC-INR": dc_payload,
        "fV-SRN": fv_payload, "RMDSRN": rm_payload,
    }
    for family, builder in builders.items():
        for target in DATASETS["ionization"]["targets"]:
            dump(CONFIGS / family / f"ionization__{target}.yaml", builder(target, False, None))
            count += 1
        for size in SIZES:
            for target in DATASETS["ionization"]["targets"]:
                dump(CONFIGS / family / size / f"ionization__{target}.yaml", builder(target, True, size))
                count += 1
    return count


def main() -> None:
    if CONFIGS.exists():
        shutil.rmtree(CONFIGS)
    CONFIGS.mkdir(parents=True)
    counts = {
        "unified_single": generate_unified_single(),
        "compact_ngp": generate_compact_ngp(),
        "instant_ngp": generate_instant_ngp(),
        "mvnet": generate_mvnet(),
        "fa_tr_inr": generate_fa_tr_inr(),
        "var_expert": generate_var_expert(),
        "mc_inr": generate_mc(),
        "neural_expert": generate_neural(),
        "volume_only": generate_volume_only(),
        "ecnr": generate_ecnr(),
    }
    total = sum(counts.values())
    if total != 356:
        raise RuntimeError(f"Expected 356 configs, generated {total}: {counts}")
    print(f"Generated {total} configs: {counts}")


if __name__ == "__main__":
    main()
