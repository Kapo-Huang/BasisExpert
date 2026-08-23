from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import shutil

import yaml


ROOT = Path(__file__).resolve().parents[2]
CONFIGS_ROOT = ROOT / "configs"
MAIN_CONFIGS = CONFIGS_ROOT / "main"
RD_CURVE_CONFIGS = CONFIGS_ROOT / "rd_curve"
REPO_ROOT_TOKEN = "${REPO_ROOT}"
DATASET_ROOT_TOKENS = {
    "redsea": "${REDSEA_ROOT}",
    "katrina": "${KATRINA_ROOT}",
    "ionization": "${IONIZATION_ROOT}",
    "combustion_40NH3_1": "${COMBUSTION_ROOT}",
}
SIZES = {
    "Size082": 0.82,
    "Size163": 1.63,
    "Size326": 3.26,
    "Size652": 6.52,
}
IONIZATION_VARIABLE_COUNT = 5
SINGLE_TARGET_SIZES = {
    name: size_mib / IONIZATION_VARIABLE_COUNT for name, size_mib in SIZES.items()
}
ION_SHAPE = {"X": 600, "Y": 248, "Z": 248, "T": 100}
DATASETS = {
    "redsea": {
        "kind": "node",
        "dir": "Mesh/RedSea",
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
COMBUSTION_SCALAR_TARGETS = [
    target for target in COMBUSTION_DATASET["targets"] if target != "Velocity"
]
COMBUSTION_KEYFRAMES = [
    0,
    182,
    364,
    545,
    727,
    909,
    1091,
    1273,
    1455,
    1636,
    1818,
    2000,
]
TARGET_FILES = {"H_plus": "H+"}
UNIFIED_SIZE_MODELS = {
    "SIREN": {
        "Size082": {"name": "siren", "in_features": 4, "hidden_features": 168, "hidden_layers": 3, "first_omega_0": 30.0, "hidden_omega_0": 30.0, "outermost_linear": True},
        "Size163": {"name": "siren", "in_features": 4, "hidden_features": 237, "hidden_layers": 3, "first_omega_0": 30.0, "hidden_omega_0": 30.0, "outermost_linear": True},
        "Size326": {"name": "siren", "in_features": 4, "hidden_features": 336, "hidden_layers": 3, "first_omega_0": 30.0, "hidden_omega_0": 30.0, "outermost_linear": True},
        "Size652": {"name": "siren", "in_features": 4, "hidden_features": 412, "hidden_layers": 4, "first_omega_0": 30.0, "hidden_omega_0": 30.0, "outermost_linear": True},
    },
    "CoordNet": {
        "Size082": {"name": "coordnet", "in_features": 4, "init_features": 15, "num_res": 10},
        "Size163": {"name": "coordnet", "in_features": 4, "init_features": 21, "num_res": 10},
        "Size326": {"name": "coordnet", "in_features": 4, "init_features": 31, "num_res": 10},
        "Size652": {"name": "coordnet", "in_features": 4, "init_features": 43, "num_res": 10},
    },
    "MoE-INR": {
        "Size082": {"name": "moe_inr", "in_features": 4, "num_experts": 7, "base_dim": 32, "encoder_feature_dim": 256, "policy_hidden_dim": 32, "policy_num_layers": 3},
        "Size163": {"name": "moe_inr", "in_features": 4, "num_experts": 7, "base_dim": 45, "encoder_feature_dim": 360, "policy_hidden_dim": 45, "policy_num_layers": 3},
        "Size326": {"name": "moe_inr", "in_features": 4, "num_experts": 7, "base_dim": 66, "encoder_feature_dim": 528, "policy_hidden_dim": 66, "policy_num_layers": 3},
        "Size652": {"name": "moe_inr", "in_features": 4, "num_experts": 7, "base_dim": 93, "encoder_feature_dim": 744, "policy_hidden_dim": 93, "policy_num_layers": 3},
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
        "node": {"name": "moe_inr", "in_features": 4, "num_experts": 7, "base_dim": 14, "encoder_feature_dim": 112, "policy_hidden_dim": 14, "policy_num_layers": 3},
        "volume": {"name": "moe_inr", "in_features": 4, "num_experts": 7, "base_dim": 45, "encoder_feature_dim": 360, "policy_hidden_dim": 45, "policy_num_layers": 3},
    },
}
MOE_MAIN_BASE_DIMS = {
    "redsea": {"default": 18},
    "katrina": {"default": 16},
    "ionization": {"default": 46},
    COMBUSTION_DATASET["name"]: {"default": 33},
}
MAIN_SINGLE_MODEL_PROFILES = {
    "SIREN": {
        "redsea": {"hidden_features": 72, "hidden_layers": 5},
        "katrina": {"hidden_features": 66, "hidden_layers": 5},
        "ionization": {"hidden_features": 237, "hidden_layers": 3},
        COMBUSTION_DATASET["name"]: {
            "hidden_features": 169,
            "hidden_layers": 3,
        },
    },
    "CoordNet": {
        "redsea": {"init_features": 10, "num_res": 7},
        "katrina": {"init_features": 9, "num_res": 7},
        "ionization": {"init_features": 29, "num_res": 5},
        COMBUSTION_DATASET["name"]: {"init_features": 18, "num_res": 7},
    },
}
INSTANT_VNR_MAIN_PROFILES = {
    "ionization": {"log2_hashmap_size": 11, "hidden_features": 105},
    COMBUSTION_DATASET["name"]: {
        "log2_hashmap_size": 10,
        "hidden_features": 88,
    },
}
MVNET_MAIN_HIDDEN_FEATURES = {
    "redsea": 73,
    "katrina": 74,
    "ionization": 206,
    COMBUSTION_DATASET["name"]: 237,
}
STSR_MAIN_PROFILES = {
    "redsea": {"init_features": 12, "embedding_dims": 655},
    "katrina": {"init_features": 12, "embedding_dims": 36},
    "ionization": {"init_features": 33, "embedding_dims": 635},
    COMBUSTION_DATASET["name"]: {
        "init_features": 28,
        "embedding_dims": 120,
    },
}
NEURAL_MAIN_PROFILES = {
    "redsea": {
        "decoder_hidden_dim": 8,
        "decoder_encoding_dim": 76,
        "manager_hidden_dim": 16,
        "manager_encoding_dim": 15,
    },
    "katrina": {
        "decoder_hidden_dim": 7,
        "decoder_encoding_dim": 69,
        "manager_hidden_dim": 14,
        "manager_encoding_dim": 17,
    },
    "ionization": {
        "decoder_hidden_dim": 31,
        "decoder_encoding_dim": 159,
        "manager_hidden_dim": 62,
        "manager_encoding_dim": 71,
    },
    COMBUSTION_DATASET["name"]: {
        "decoder_hidden_dim": 16,
        "decoder_encoding_dim": 126,
        "manager_hidden_dim": 32,
        "manager_encoding_dim": 56,
    },
}
FV_MAIN_PROFILES = {
    "ionization": (6, 64),
    COMBUSTION_DATASET["name"]: (5, 60),
}
VAR_SIZE_PROFILES = {
    "Size082": {"num_experts": 8, "base_dim": 15, "top_k": 4},
    "Size163": {"num_experts": 8, "base_dim": 22, "top_k": 4},
    "Size326": {"num_experts": 8, "base_dim": 31, "top_k": 4},
    "Size652": {"num_experts": 8, "base_dim": 45, "top_k": 4},
}
STSR_SIZE_PROFILES = {
    "Size082": {"init_features": 20, "embedding_dims": 80},
    "Size163": {"init_features": 64, "embedding_dims": 256},
    "Size326": {"init_features": 40, "embedding_dims": 160},
    "Size652": {"init_features": 56, "embedding_dims": 224},
}
MINER_SIZE_HIDDEN_FEATURES = {
    "Size082": 3,
    "Size163": 20,
    "Size326": 6,
    "Size652": 9,
}
NEURAL_SIZE_DIMS = {"Size082": 17, "Size163": 24, "Size326": 34, "Size652": 48}
MC_SIZE_DIMS = {"Size082": 30, "Size163": 43, "Size326": 62, "Size652": 88}
APMG_MAIN_MODEL = {
    "feature_grid_shape": [4, 4, 4],
    "n_grids": 1,
    "n_features": 14,
    "nodes_per_layer": 16,
    "n_layers": 3,
}
APMG_SIZE = {
    "Size082": {"feature_grid_shape": [7, 7, 7], "n_grids": 1, "n_features": 1, "nodes_per_layer": 16, "n_layers": 2},
    "Size163": {"feature_grid_shape": [5, 5, 5], "n_grids": 6, "n_features": 1, "nodes_per_layer": 16, "n_layers": 3},
    "Size326": {"feature_grid_shape": [5, 5, 5], "n_grids": 12, "n_features": 1, "nodes_per_layer": 32, "n_layers": 2},
    "Size652": {"feature_grid_shape": [4, 4, 4], "n_grids": 44, "n_features": 1, "nodes_per_layer": 32, "n_layers": 3},
}
APMG_MAIN_PROFILES = {
    "ionization": APMG_MAIN_MODEL,
    COMBUSTION_DATASET["name"]: APMG_SIZE["Size082"],
}
FV_SIZE = {
    "Size082": (5, 54), "Size163": (32, 16), "Size326": (8, 55),
    "Size652": (8, 110),
}
RM_SIZE = {
    "Size082": (5, 30), "Size163": (6, 48), "Size326": (7, 70),
    "Size652": (17, 11),
}


def dump(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")


def _formal_sort_key(path: Path) -> tuple[str, str]:
    category_root = MAIN_CONFIGS if path.is_relative_to(MAIN_CONFIGS) else RD_CURVE_CONFIGS
    relative = path.relative_to(category_root)
    return relative.parts[0], relative.as_posix()


def write_run_lists() -> None:
    main_paths = sorted(MAIN_CONFIGS.rglob("*.yaml"), key=_formal_sort_key)
    rd_curve_paths = sorted(RD_CURVE_CONFIGS.rglob("*.yaml"), key=_formal_sort_key)
    main = [f"configs/main/{path.relative_to(MAIN_CONFIGS).as_posix()}" for path in main_paths]
    rd_curve = [f"configs/rd_curve/{path.relative_to(RD_CURVE_CONFIGS).as_posix()}" for path in rd_curve_paths]
    relative = [*main, *rd_curve]
    headers = {
        "all_configs.list": [
            "# Formal training selection: one repository-relative YAML path per line.",
            "# Comment out or delete entries to run only a subset.",
            "# Execution order and parallel grouping remain defined by scripts/main/run_all.sh.",
            f"# Generated formal matrix: {len(relative)} configs.",
        ],
        "configs.list": [
            "# Main-experiment selection: all formal configs without a Size tier.",
            "# Use with: bash scripts/main/run_all.sh",
            "# Comment out or delete entries to select a smaller main-experiment subset.",
        ],
        "rd_curve_configs.list": [
            "# RD-curve selection: all formal Size-tier configs.",
            "# Use with: bash scripts/rd_curve/run.sh",
            "# Comment out or delete entries to select a smaller RD-curve subset.",
        ],
    }
    selections = {
        "all_configs.list": relative,
        "configs.list": main,
        "rd_curve_configs.list": rd_curve,
    }
    for filename, selection in selections.items():
        content = "\n".join([*headers[filename], "", *selection, ""])
        destination = (
            ROOT / "scripts" / "rd_curve" / "configs.list"
            if filename == "rd_curve_configs.list"
            else ROOT / "scripts" / "main" / filename
        )
        destination.write_text(content, encoding="utf-8", newline="\n")


def rel_prefix(nested: bool) -> str:
    del nested
    return f"{REPO_ROOT_TOKEN}/"


def repo_path(relative: str) -> str:
    return f"{REPO_ROOT_TOKEN}/{relative.strip('/')}"


def targets_for(dataset: str, nested: bool) -> dict[str, str]:
    del nested
    meta = DATASETS[dataset]
    return {
        target: f"{DATASET_ROOT_TOKENS[dataset]}/target_{TARGET_FILES.get(target, target)}.npy"
        for target in meta["targets"]
    }


def unified_data(dataset: str, nested: bool, target: str | None = None) -> dict:
    meta = DATASETS[dataset]
    payload = {"kind": meta["kind"], "dataset_name": dataset, "split": "train"}
    if meta["kind"] == "node":
        payload["coords_path"] = f"{DATASET_ROOT_TOKENS[dataset]}/{meta['coords']}"
    else:
        payload["volume_shape"] = deepcopy(ION_SHAPE)
    payload["targets"] = targets_for(dataset, nested)
    if target is not None:
        payload["target"] = target
    return payload


def combustion_targets() -> dict[str, str]:
    return {
        target: f"{DATASET_ROOT_TOKENS[COMBUSTION_DATASET['name']]}/target_{target}.npy"
        for target in COMBUSTION_DATASET["targets"]
    }


def target_dimension(dataset: str, target: str) -> int:
    if (dataset, target) in {
        ("katrina", "v"),
        (COMBUSTION_DATASET["name"], "Velocity"),
    }:
        return 3
    return 1


def total_target_channels(dataset: str, targets: list[str]) -> int:
    return sum(target_dimension(dataset, target) for target in targets)


def combustion_data(
    target: str | None = None,
    *,
    include_vector: bool = True,
    four_coordinates: bool = False,
    selected_only: bool = False,
) -> dict:
    dataset_name = COMBUSTION_DATASET["name"]
    targets = combustion_targets()
    if not include_vector:
        targets = {
            name: path for name, path in targets.items()
            if name in COMBUSTION_SCALAR_TARGETS
        }
    if target is not None and target not in targets:
        raise ValueError(f"Unknown Combustion target: {target}")
    if selected_only:
        if target is None:
            raise ValueError("selected_only Combustion data requires a target")
        targets = {target: targets[target]}
    payload = {
        "kind": "volume",
        "dataset_name": dataset_name,
        "split": "train",
        "volume_shape": deepcopy(COMBUSTION_DATASET["volume_shape"]),
        "targets": targets,
    }
    if not four_coordinates:
        payload["coordinate_axes"] = list(COMBUSTION_DATASET["coordinate_axes"])
    if target is not None and not selected_only:
        payload["target"] = target
    return payload


def common_training() -> dict:
    return {
        "epochs": 600, "batch_size": 16000, "pred_batch_size": 16000,
        "num_workers": 0, "lr": 5.0e-5, "val_split": 0.0,
        "log_every": 10, "log_psnr_every": 100, "psnr_sample_ratio": 0.1,
        "save_every": 600, "early_stop_patience": 0, "loss_type": "mse",
        "seed": 42, "sampler": "budgeted_random", "batches_per_epoch_budget": 1500,
        "scheduler": {"enabled": True, "step_size": 40, "gamma": 0.92},
    }


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
            MAIN_CONFIGS / "InstantNGP" / f"ionization__{target}.yaml",
            payload,
        )
        count += 1
    for target in COMBUSTION_SCALAR_TARGETS:
        payload = {
            "experiment": f"{COMBUSTION_DATASET['name']}_instant_ngp_{target}",
            "exp_id": f"instant-ngp-{COMBUSTION_DATASET['name']}-{target}",
            "experiment_root": repo_path("runs"),
            "data": combustion_data(
                target,
                include_vector=False,
                four_coordinates=True,
                selected_only=True,
            ),
            "model": instant_ngp_model(),
            "training": instant_ngp_training(),
            "evaluation": evaluation(),
            "log": log_config(),
        }
        dump(
            MAIN_CONFIGS / "InstantNGP" / f"{COMBUSTION_DATASET['name']}__{target}.yaml",
            payload,
        )
        count += 1
    return count


def instant_vnr_model(dataset: str) -> dict:
    payload = {
        "name": "instant_vnr",
        "in_features": 4,
        "out_features": 1,
        "n_levels": 8,
        "n_features_per_level": 8,
        "base_resolution": 16,
        "per_level_scale": 2.0,
        "log2_hashmap_size": 19,
        "hidden_features": 64,
        "hidden_layers": 4,
    }
    payload.update(deepcopy(INSTANT_VNR_MAIN_PROFILES[dataset]))
    return payload


def instant_vnr_training() -> dict:
    payload = common_training()
    payload.update(
        {
            "gradient_accumulation_steps": 4,
            "lr": 5.0e-5,
            "beta_1": 0.9,
            "beta_2": 0.999,
            "epsilon": 1.0e-8,
            "weight_decay": 0.0,
            "loss_type": "mse",
            "scheduler": {
                "enabled": True,
                "interval": "epoch",
                "step_size": 40,
                "gamma": 0.92,
            },
            "pretrain": {"enabled": False},
        }
    )
    return payload


def generate_instant_vnr() -> int:
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
            "experiment": f"ionization_instant_vnr_{target}",
            "exp_id": f"instant-vnr-ionization-{target}",
            "experiment_root": repo_path("runs"),
            "data": data,
            "model": instant_vnr_model("ionization"),
            "training": instant_vnr_training(),
            "evaluation": evaluation(),
            "log": log_config(),
        }
        dump(MAIN_CONFIGS / "InstantVNR" / f"ionization__{target}.yaml", payload)
        count += 1
    for target in COMBUSTION_SCALAR_TARGETS:
        payload = {
            "experiment": f"{COMBUSTION_DATASET['name']}_instant_vnr_{target}",
            "exp_id": f"instant-vnr-{COMBUSTION_DATASET['name']}-{target}",
            "experiment_root": repo_path("runs"),
            "data": combustion_data(
                target,
                include_vector=False,
                four_coordinates=True,
                selected_only=True,
            ),
            "model": instant_vnr_model(COMBUSTION_DATASET["name"]),
            "training": instant_vnr_training(),
            "evaluation": evaluation(),
            "log": log_config(),
        }
        dump(
            MAIN_CONFIGS / "InstantVNR" / f"{COMBUSTION_DATASET['name']}__{target}.yaml",
            payload,
        )
        count += 1
    return count


def mvnet_model(dataset: str, num_variables: int) -> dict:
    return {
        "name": "mvnet",
        "in_features": 4,
        "out_features": int(num_variables),
        "hidden_features": MVNET_MAIN_HIDDEN_FEATURES[dataset],
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
        "lr": 1.0e-5,
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
            "step_size": 40,
            "gamma": 0.92,
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
            "model": mvnet_model(
                dataset,
                total_target_channels(dataset, meta["targets"])
            ),
            "training": mvnet_training(),
            "evaluation": evaluation(),
            "log": log_config(),
        }
        dump(MAIN_CONFIGS / "MVNet" / f"{dataset}.yaml", payload)
        count += 1
    combustion = combustion_data(four_coordinates=True)
    combustion["targets"] = {
        name: combustion["targets"][name]
        for name in sorted(combustion["targets"])
    }
    payload = {
        "experiment": f"{COMBUSTION_DATASET['name']}_mvnet",
        "exp_id": f"mvnet-{COMBUSTION_DATASET['name']}",
        "experiment_root": repo_path("runs"),
        "data": combustion,
        "model": mvnet_model(
            COMBUSTION_DATASET["name"],
            total_target_channels(
                COMBUSTION_DATASET["name"],
                COMBUSTION_DATASET["targets"],
            )
        ),
        "training": mvnet_training(),
        "evaluation": evaluation(),
        "log": log_config(),
    }
    dump(MAIN_CONFIGS / "MVNet" / f"{COMBUSTION_DATASET['name']}.yaml", payload)
    count += 1
    return count


def generate_stsr_inr() -> int:
    def main_model(dataset: str) -> dict:
        payload = {
            "name": "stsr_inr",
            "in_features": 4,
            "init_features": 64,
            "num_res": 5,
            "omega_0": 5.0,
            "embedding_dims": 256,
            "outermost_linear": True,
            "use_global_latent": True,
        }
        payload.update(deepcopy(STSR_MAIN_PROFILES[dataset]))
        return payload

    count = 0
    for dataset in DATASETS:
        payload = {
            "experiment": f"{dataset}_stsr_inr",
            "exp_id": f"stsr-inr-{dataset}",
            "experiment_root": repo_path("runs"),
            "data": unified_data(dataset, False),
            "model": main_model(dataset),
            "training": mvnet_training(),
            "evaluation": evaluation(),
            "log": log_config(),
        }
        dump(MAIN_CONFIGS / "STSR-INR" / f"{dataset}.yaml", payload)
        count += 1

    combustion_name = COMBUSTION_DATASET["name"]
    payload = {
        "experiment": f"{combustion_name}_stsr_inr",
        "exp_id": f"stsr-inr-{combustion_name}",
        "experiment_root": repo_path("runs"),
        "data": combustion_data(four_coordinates=True),
        "model": main_model(combustion_name),
        "training": mvnet_training(),
        "evaluation": evaluation(),
        "log": log_config(),
    }
    dump(MAIN_CONFIGS / "STSR-INR" / f"{combustion_name}.yaml", payload)
    count += 1
    for size, profile in STSR_SIZE_PROFILES.items():
        if size == "Size163":
            training = mvnet_training()
            model = {
                "name": "stsr_inr",
                "in_features": 4,
                "init_features": 64,
                "num_res": 5,
                "omega_0": 5.0,
                "embedding_dims": 256,
                "outermost_linear": True,
                "use_global_latent": True,
            }
        else:
            training = common_training()
            training["lr"] = 1.0e-5
            model = {
                "name": "stsr_inr",
                "in_features": 4,
                **profile,
                "num_res": 10,
                "omega_0": 5.0,
                "outermost_linear": True,
                "use_global_latent": True,
            }
        sized_payload = {
            "experiment": f"ionization_stsr_inr_{size.lower()}",
            "exp_id": f"stsr-inr-ionization-{size.lower()}",
            "experiment_root": repo_path("runs"),
            "data": unified_data("ionization", True),
            "model": model,
            "training": training,
            "evaluation": evaluation(),
            "log": log_config(),
        }
        dump(RD_CURVE_CONFIGS / "STSR-INR" / size / "ionization.yaml", sized_payload)
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


def moe_main_model(dataset: str, target: str) -> dict:
    widths = MOE_MAIN_BASE_DIMS[dataset]
    base_dim = int(widths.get(target, widths["default"]))
    in_features = (
        len(COMBUSTION_DATASET["coordinate_axes"])
        if dataset == COMBUSTION_DATASET["name"]
        else 4
    )
    return {
        "name": "moe_inr",
        "in_features": in_features,
        "num_experts": 7,
        "base_dim": base_dim,
        "encoder_feature_dim": 8 * base_dim,
        "policy_hidden_dim": base_dim,
        "policy_num_layers": 3,
    }


def generate_unified_single() -> int:
    count = 0
    for family, defaults in DEFAULT_MODELS.items():
        model_slug = defaults["volume"]["name"].replace("_", "-")
        for dataset, meta in DATASETS.items():
            for target in meta["targets"]:
                model = (
                    moe_main_model(dataset, target)
                    if family == "MoE-INR"
                    else deepcopy(defaults[meta["kind"]])
                )
                if family in MAIN_SINGLE_MODEL_PROFILES:
                    model.update(
                        deepcopy(MAIN_SINGLE_MODEL_PROFILES[family][dataset])
                    )
                training = common_training()
                if family == "CoordNet":
                    training["lr"] = 1.0e-5
                payload = {
                    "experiment": f"{dataset}_{model_slug}_{target}",
                    "exp_id": f"{model_slug}-{dataset}-{target}",
                    "experiment_root": repo_path("runs"),
                    "data": unified_data(dataset, False, target),
                    "model": model,
                    "training": training,
                    "evaluation": evaluation(),
                    "log": log_config(),
                }
                dump(MAIN_CONFIGS / family / f"{dataset}__{target}.yaml", payload)
                count += 1
        for target in COMBUSTION_DATASET["targets"]:
            combustion_model = (
                moe_main_model(COMBUSTION_DATASET["name"], target)
                if family == "MoE-INR"
                else deepcopy(defaults["volume"])
            )
            if family in MAIN_SINGLE_MODEL_PROFILES:
                combustion_model.update(
                    deepcopy(
                        MAIN_SINGLE_MODEL_PROFILES[family][
                            COMBUSTION_DATASET["name"]
                        ]
                    )
                )
            combustion_model["in_features"] = len(COMBUSTION_DATASET["coordinate_axes"])
            training = common_training()
            if family == "CoordNet":
                training["lr"] = 1.0e-5
            payload = {
                "experiment": f"{COMBUSTION_DATASET['name']}_{model_slug}_{target}",
                "exp_id": f"{model_slug}-{COMBUSTION_DATASET['name']}-{target}",
                "experiment_root": repo_path("runs"),
                "data": combustion_data(target),
                "model": deepcopy(combustion_model),
                "training": training,
                "evaluation": evaluation(),
                "log": log_config(),
            }
            dump(
                MAIN_CONFIGS / family / f"{COMBUSTION_DATASET['name']}__{target}.yaml",
                payload,
            )
            count += 1
        if family not in {"CoordNet", "MoE-INR"}:
            continue
        for size in SIZES:
            for target in DATASETS["ionization"]["targets"]:
                training = common_training()
                if family == "CoordNet":
                    training["lr"] = 1.0e-5
                payload = {
                    "experiment": f"ionization_{model_slug}_{size.lower()}_{target}",
                    "exp_id": f"{model_slug}-ionization-{size.lower()}-{target}",
                    "experiment_root": repo_path("runs"),
                    "data": unified_data("ionization", True, target),
                    "model": deepcopy(UNIFIED_SIZE_MODELS[family][size]),
                    "training": training,
                    "evaluation": evaluation(),
                    "log": log_config(),
                }
                dump(RD_CURVE_CONFIGS / family / size / f"ionization__{target}.yaml", payload)
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
        dump(MAIN_CONFIGS / "VarExpert" / f"{dataset}.yaml", payload)
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
            dump(MAIN_CONFIGS / "VarExpert" / "ionization_dwa.yaml", dwa_payload)
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
    dump(MAIN_CONFIGS / "VarExpert" / f"{combustion_name}.yaml", combustion_payload)
    count += 1
    for size, model_profile in VAR_SIZE_PROFILES.items():
        experts = int(model_profile["num_experts"])
        training = var_training("ionization", True, experts)
        # Exploration v2 showed late PSNR collapses when alpha=5 allowed the
        # per-target EMA weights to swing from about 0.2 to 3.8.  Keep the
        # formal size runs adaptive, but make the controller substantially
        # smoother and bounded, and retain checkpoints around any regression.
        training["multiview_ema_loss"].update(
            {"beta": 0.99, "w_min": 0.5, "w_max": 2.0, "warmup_steps": 75000, "alpha": 1.0}
        )
        training.update({"log_psnr_every": 25, "save_every": 100})
        payload = {
            "experiment": f"ionization_var_expert_{size.lower()}",
            "exp_id": f"var-expert-ionization-{size.lower()}",
            "experiment_root": repo_path("runs"),
            "data": unified_data("ionization", True),
            "model": {"name": "var_expert", "in_features": 4, **model_profile},
            "training": training,
            "evaluation": evaluation(),
            "log": log_config(),
        }
        dump(RD_CURVE_CONFIGS / "VarExpert" / size / "ionization.yaml", payload)
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
        dump(MAIN_CONFIGS / "MC-INR" / f"{dataset}.yaml", mc_payload(dataset, False, 128 if dataset == "ionization" else 96))
        count += 1
    combustion_name = COMBUSTION_DATASET["name"]
    payload = {
        "experiment": f"{combustion_name}_mc_inr",
        "exp_id": f"mc-inr-{combustion_name}",
        "experiment_root": repo_path("runs"),
        "data": combustion_data(),
        "model": {
            "name": "mc_inr",
            "hidden_features": 128,
            "gfe_layers": 5,
            "lfe_layers": 6,
        },
        "training": deepcopy(mc_payload("ionization", False, 128)["training"]),
        "evaluation": evaluation(),
        "log": log_config(),
    }
    payload["training"]["assignments_cache_path"] = repo_path(
        f"data/cache/mc_inr/{combustion_name}_assignments_k12.npy"
    )
    dump(MAIN_CONFIGS / "MC-INR" / f"{combustion_name}.yaml", payload)
    count += 1
    return count


def neural_model(
    dataset: str,
    dim: int,
    nested: bool,
    target: str,
    manager_pretrain: bool,
    size: str | None,
    profile: dict | None = None,
) -> dict:
    model_name = (
        "inr_moe_ionization"
        if dataset in {"ionization", COMBUSTION_DATASET["name"]}
        else "inr_moe_mesh"
    )
    size_token = size.lower() if size else "default"
    manager_path = (
        f"{rel_prefix(nested)}runs/neural_expert/pretrained_managers/{dataset}/{size_token}/"
        f"pt_{model_name}_{target}_managerpretraining.pth"
    )
    profile = profile or {}
    decoder_hidden_dim = int(profile.get("decoder_hidden_dim", dim))
    decoder_encoding_dim = int(
        profile.get("decoder_encoding_dim", decoder_hidden_dim * 8)
    )
    manager_hidden_dim = int(
        profile.get("manager_hidden_dim", decoder_hidden_dim * 2)
    )
    manager_encoding_dim = int(
        profile.get("manager_encoding_dim", decoder_hidden_dim * 2)
    )
    return {
        "model_name": model_name,
        "in_dim": 4,
        "out_dim": target_dimension(dataset, target),
        "decoder_hidden_dim": decoder_hidden_dim, "decoder_n_hidden_layers": 2,
        "decoder_input_encoding": f"learned_{decoder_encoding_dim}_2_sine_siren_none",
        "decoder_nl": "sine", "decoder_init_type": "siren", "n_experts": 8,
        "outermost_linear": True, "input_encoding": "none", "decoder_freqs": 30.0,
        "decoder_trainable_freqs": False, "top_k": 1,
        "manager_hidden_dim": manager_hidden_dim, "manager_n_hidden_layers": 2,
        "manager_input_encoding": f"learned_{manager_encoding_dim}_2_sine_siren_none",
        "manager_nl": "sine", "manager_init": "siren", "manager_type": "standard",
        "experts_bias_std": 0.1, "experts_bias_weight": 1.0,
        "manager_softmax_temperature": 1.0, "manager_softmax_temp_trainable": False,
        "manager_q_activation": "softmax", "manager_clamp_q": 0.0,
        "manager_conditioning": "cat", "manager_pt_path": manager_path,
        "load_pt_manager": not manager_pretrain, "shared_encoder": False,
    }


def neural_data(dataset: str, target: str, nested: bool) -> dict:
    is_volume = dataset in {"ionization", COMBUSTION_DATASET["name"]}
    target_mapping = (
        targets_for(dataset, nested)
        if dataset in DATASETS
        else combustion_targets()
    )
    data = {
        "dataset_name": dataset, "target": target, "targets": target_mapping,
        "target_stats_path": f"{rel_prefix(nested)}data/cache/neural_expert/{dataset}/target_stats_{target}.npz",
        "normalize_inputs": is_volume, "normalize_targets": False,
    }
    if is_volume:
        volume_shape = (
            deepcopy(ION_SHAPE)
            if dataset == "ionization"
            else deepcopy(COMBUSTION_DATASET["volume_shape"])
        )
        data.update({"volume_shape": volume_shape, "segmentation_type": "random_balanced", "grid_patch_size": 4, "n_segments": 8})
    else:
        meta = DATASETS[dataset]
        data.update({
            "association": "point",
            "source_path": f"{DATASET_ROOT_TOKENS[dataset]}/{meta['coords']}",
            "stats_key": target,
        })
    return data


def neural_payload(
    dataset: str,
    target: str,
    nested: bool,
    manager_pretrain: bool,
    dim: int,
    size: str | None,
    profile: dict | None = None,
) -> dict:
    suffix = "-managerpretrain" if manager_pretrain else ""
    exp_size = f"-{size.lower()}" if size else ""
    loss_name = "1000segmentation" if manager_pretrain else "1000valrecon"
    training = {
        "n_points": 16000, "lr": 3.0e-5, "lr_gamma": 0.9999,
        "lr_scheduler": "ExponentialLR",
        "num_epochs": 30000 if manager_pretrain else 60000,
        "batch_size": 1, "num_workers": 0, "grad_clip_norm": 10.0,
        "save_every": 500, "segmentation_mode": manager_pretrain,
        "log_every": 100,
        "stages": [{"end_iteration_frac": 1.0, "params": "all", "loss_type": loss_name}],
    }
    if dataset not in {"ionization", COMBUSTION_DATASET["name"]}:
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
        "MODEL": neural_model(
            dataset,
            dim,
            nested,
            target,
            manager_pretrain,
            size,
            profile,
        ),
        "LOSS": {
            "scale_by_q_grad": False, "loss_type": loss_name,
            "segmentation_type": (
                "both"
                if dataset in {"ionization", COMBUSTION_DATASET["name"]}
                else "ce"
            ),
            "sample_bias_correction": False, "entropy_metric": "kl",
        },
        "DATA": neural_data(dataset, target, nested),
        "TRAINING": training,
    }


def generate_neural() -> int:
    count = 0
    for dataset, meta in DATASETS.items():
        profile = NEURAL_MAIN_PROFILES[dataset]
        for target in meta["targets"]:
            for pretrain in (True, False):
                suffix = "__managerpretrain" if pretrain else ""
                dump(
                    MAIN_CONFIGS / "NeuralExpert" / f"{dataset}__{target}{suffix}.yaml",
                    neural_payload(
                        dataset,
                        target,
                        False,
                        pretrain,
                        int(profile["decoder_hidden_dim"]),
                        None,
                        profile,
                    ),
                )
                count += 1
    profile = NEURAL_MAIN_PROFILES[COMBUSTION_DATASET["name"]]
    for target in COMBUSTION_DATASET["targets"]:
        for pretrain in (True, False):
            suffix = "__managerpretrain" if pretrain else ""
            dump(
                MAIN_CONFIGS / "NeuralExpert" / f"{COMBUSTION_DATASET['name']}__{target}{suffix}.yaml",
                neural_payload(
                    COMBUSTION_DATASET["name"],
                    target,
                    False,
                    pretrain,
                    int(profile["decoder_hidden_dim"]),
                    None,
                    profile,
                ),
            )
            count += 1
    return count


def volume_data(
    target: str,
    nested: bool,
    upper: bool = False,
    dataset: str = "ionization",
) -> dict:
    key = "DATA" if upper else "data"
    del key
    if dataset == COMBUSTION_DATASET["name"]:
        return combustion_data(
            target,
            include_vector=False,
            four_coordinates=True,
        )
    return {
        "kind": "volume", "dataset_name": "ionization", "split": "train",
        "target": target, "targets": targets_for("ionization", nested),
        "volume_shape": deepcopy(ION_SHAPE),
    }


def apmg_payload(
    target: str,
    nested: bool,
    size: str | None,
    dataset: str = "ionization",
) -> dict:
    sizing = APMG_SIZE[size] if size else APMG_MAIN_PROFILES[dataset]
    tag = f"-{size.lower()}" if size else ""
    data = volume_data(target, nested, dataset=dataset)
    return {
        "experiment": f"apmgsrn_{dataset}{tag}_{target}",
        "exp_id": f"apmgsrn-{dataset}{tag}-{target}",
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
            "requires_padded_feats": True,
        },
        "DATA": {
            "dataset_name": dataset, "target": target,
            "targets": data["targets"],
            "volume_shape": data["volume_shape"], "align_corners": True,
        },
        "TRAINING": {
            "iterations": 450 if dataset == COMBUSTION_DATASET["name"] else 9000,
            "points_per_iteration": 16000,
            "prediction_points_per_batch": 16000, "lr": 5.0e-5,
            "beta_1": 0.9, "beta_2": 0.999, "eps": 1.0e-8,
            "weight_decay": 0.0,
            "lr_scheduler": "var_expert_progress",
            "lr_step": 40, "lr_gamma": 0.92,
            "scheduler_reference_epochs": 600,
            "device": "cuda:0",
            "data_device": "same", "save_every": 0, "log_every": 100,
            "time_indices": "all", "seed": 42, "early_stopping": False,
        },
        "EVALUATION": {"run_after_training": False},
    }


def fv_payload(
    target: str,
    nested: bool,
    size: str | None,
    dataset: str = "ionization",
) -> dict:
    resolution, channels = (
        FV_SIZE[size] if size else FV_MAIN_PROFILES[dataset]
    )
    tag = f"-{size.lower()}" if size else ""
    return {
        "experiment": f"fv_srn_{dataset}{tag}_{target}",
        "exp_id": f"fv-srn-{dataset}{tag}-{target}",
        "experiment_root": repo_path("runs/fv_srn"),
        "data": volume_data(target, nested, dataset=dataset),
        "model": {
            "name": "fv_srn", "grid_resolution": resolution,
            "grid_channels": channels, "grid_init_std": 0.01,
            "keyframe_indices": (
                deepcopy(COMBUSTION_KEYFRAMES)
                if dataset == COMBUSTION_DATASET["name"]
                else [0, 9, 18, 27, 36, 45, 54, 63, 72, 81, 90, 99]
            ),
            "fourier_features": 14, "fourier_mode": "nerf",
            "hidden_features": 32, "hidden_layers": 3,
            "activation": "snake_alt", "activation_frequency": 1.0,
            "time_encoding": "none",
        },
        "training": {
            "epochs": 600,
            "samples_per_timestep": (
                12000 if dataset == COMBUSTION_DATASET["name"] else 240000
            ),
            "validation_fraction": 0.0, "batch_size": 16000,
            "prediction_batch_size": 16000, "lr": 5.0e-5,
            "beta_1": 0.9, "beta_2": 0.999, "eps": 1.0e-8,
            "weight_decay": 0.0, "lr_scheduler": "step",
            "lr_step": 40, "lr_gamma": 0.92,
            "l1_weight": 1.0, "l2_weight": 0.0,
            "importance_floor": 0.01, "rebuild_every": 51,
            "rebuild_grid_size": 32, "rebuild_samples_per_cell": 2,
            "save_every": 300, "log_every": 1, "seed": 42, "device": "cuda",
        },
        "evaluation": {**evaluation(), "run_after_training": False, "default_model": "compact"},
    }


def ecnr_payload(target: str, dataset: str = "ionization") -> dict:
    return {
        "experiment": f"ecnr_{dataset}_{target}",
        "exp_id": f"ecnr-{dataset}-{target}",
        "experiment_root": repo_path("runs/ecnr"),
        "data": volume_data(target, False, dataset=dataset),
        "model": {
            "name": "ecnr", "scales": 3,
            "block_shape_xyz": (
                [16, 16, 1]
                if dataset == COMBUSTION_DATASET["name"]
                else [25, 31, 31]
            ),
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
            "passes_per_epoch": 1,
            "lr": 1.0e-3, "beta_1": 0.9, "beta_2": 0.999,
            "weight_decay": 2.0e-5,
            "pruning_epochs": [150, 225, 300, 375],
            "pruning_sparsities": [0.30, 0.40, 0.45, 0.50],
            "pruning_loss_weight": 0.1, "pruning_lr_gamma": 0.75,
            "quantization_finetune_epochs": 75,
            "quantization_finetune_passes_per_epoch": 1,
            "quantization_finetune_lr": 1.0e-5,
            "save_every": 0, "log_every": 1,
            "progress_log_seconds": 60, "seed": 42, "device": "cuda",
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
        dump(MAIN_CONFIGS / "ECNR" / f"ionization__{target}.yaml", ecnr_payload(target))
    for target in COMBUSTION_SCALAR_TARGETS:
        dump(
            MAIN_CONFIGS / "ECNR" / f"{COMBUSTION_DATASET['name']}__{target}.yaml",
            ecnr_payload(target, COMBUSTION_DATASET["name"]),
        )
    return len(DATASETS["ionization"]["targets"]) + len(COMBUSTION_SCALAR_TARGETS)


def miner_payload(target: str, dataset: str = "ionization") -> dict:
    is_2d = dataset == COMBUSTION_DATASET["name"]
    return {
        "experiment": f"miner_{dataset}_{target}",
        "exp_id": f"miner-{dataset}-{target}",
        "experiment_root": repo_path("runs/miner"),
        "data": volume_data(target, False, dataset=dataset),
        "model": {
            "name": "miner",
            "scales": 4,
            "block_size": 32 if is_2d else 16,
            "hidden_features": 18 if is_2d else 20,
            "hidden_layers": 2,
            "omega_0": 150.0 if is_2d else 30.0,
            "coordinate_type": "local",
            "propagation": "coarse_to_fine",
            "carry_start_scale": 2,
            "coarse_feature_multiplier": 4,
        },
        "training": {
            "epochs_per_scale": 500 if is_2d else 2000,
            "lr": 5.0e-4 if is_2d else 1.0e-3,
            "beta_1": 0.9,
            "beta_2": 0.999,
            "block_mse_threshold": 1.0e-4 if is_2d else 2.0e-4,
            "scale_convergence_delta": 5.0e-7 if is_2d else 2.0e-6,
            "global_mse_threshold": 1.0e-4 if is_2d else 0.0,
            "lr_decay": 0.999,
            "max_active_blocks_per_step": 16384 if is_2d else 2048,
            "time_indices": "all",
            "seed": 42,
            "device": "cuda",
            "log_every": 25,
        },
        "evaluation": {
            "save_predictions": False,
            "run_after_training": False,
            "default_model": "checkpoint",
        },
        "log": {
            "effective_config": True,
            "startup_timing": True,
            "epoch_summary": True,
        },
    }


def generate_miner() -> int:
    count = 0
    for target in DATASETS["ionization"]["targets"]:
        dump(MAIN_CONFIGS / "MINER" / f"ionization__{target}.yaml", miner_payload(target))
        count += 1
    for target in COMBUSTION_SCALAR_TARGETS:
        dump(
            MAIN_CONFIGS / "MINER" / f"{COMBUSTION_DATASET['name']}__{target}.yaml",
            miner_payload(target, COMBUSTION_DATASET["name"]),
        )
        count += 1
    for size, hidden_features in MINER_SIZE_HIDDEN_FEATURES.items():
        for target in DATASETS["ionization"]["targets"]:
            payload = miner_payload(target)
            payload["experiment"] = f"miner_ionization_{size.lower()}_{target}"
            payload["exp_id"] = f"miner-ionization-{size.lower()}-{target}"
            if size != "Size163":
                payload["model"].update(
                    {
                        "scales": 4,
                        "block_size": 40,
                        "hidden_features": hidden_features,
                        "hidden_layers": 2,
                        "carry_start_scale": 2,
                        "coarse_feature_multiplier": 4,
                    }
                )
            dump(RD_CURVE_CONFIGS / "MINER" / size / f"ionization__{target}.yaml", payload)
            count += 1
    return count


def rm_payload(
    target: str,
    nested: bool,
    size: str | None,
    dataset: str = "ionization",
) -> dict:
    resolution, channels = RM_SIZE[size] if size else (32, 16)
    tag = f"-{size.lower()}" if size else ""
    return {
        "experiment": f"rmdsrn_{dataset}{tag}_{target}",
        "exp_id": f"rmdsrn-{dataset}{tag}-{target}",
        "experiment_root": repo_path("runs/rmdsrn"),
        "data": volume_data(target, nested, dataset=dataset),
        "model": {
            "name": "rmdsrn", "base_encoder": "temporal_fv_srn",
            "grid_resolution": resolution, "grid_channels": channels,
            "grid_init_std": 0.01,
            "keyframe_indices": (
                deepcopy(COMBUSTION_KEYFRAMES)
                if dataset == COMBUSTION_DATASET["name"]
                else [0, 9, 18, 27, 36, 45, 54, 63, 72, 81, 90, 99]
            ),
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
        "APMGSRN": apmg_payload,
        "fV-SRN": fv_payload, "RMDSRN": rm_payload,
    }
    for family, builder in builders.items():
        for target in DATASETS["ionization"]["targets"]:
            dump(MAIN_CONFIGS / family / f"ionization__{target}.yaml", builder(target, False, None))
            count += 1
        for target in COMBUSTION_SCALAR_TARGETS:
            dump(
                MAIN_CONFIGS / family / f"{COMBUSTION_DATASET['name']}__{target}.yaml",
                builder(target, False, None, COMBUSTION_DATASET["name"]),
            )
            count += 1
        if family != "fV-SRN":
            continue
        for size in SIZES:
            for target in DATASETS["ionization"]["targets"]:
                payload = builder(target, True, size)
                dump(RD_CURVE_CONFIGS / family / size / f"ionization__{target}.yaml", payload)
                count += 1
    return count


def main() -> None:
    for category_root in (MAIN_CONFIGS, RD_CURVE_CONFIGS):
        if category_root.exists():
            shutil.rmtree(category_root)
        category_root.mkdir(parents=True)
    counts = {
        "unified_single": generate_unified_single(),
        "instant_ngp": generate_instant_ngp(),
        "instant_vnr": generate_instant_vnr(),
        "mvnet": generate_mvnet(),
        "stsr_inr": generate_stsr_inr(),
        "var_expert": generate_var_expert(),
        "mc_inr": generate_mc(),
        "neural_expert": generate_neural(),
        "volume_only": generate_volume_only(),
        "ecnr": generate_ecnr(),
        "miner": generate_miner(),
    }
    total = sum(counts.values())
    if CONFIGS_ROOT.resolve() == (ROOT / "configs").resolve():
        write_run_lists()
    print(f"Generated {total} configs: {counts}")


if __name__ == "__main__":
    main()
