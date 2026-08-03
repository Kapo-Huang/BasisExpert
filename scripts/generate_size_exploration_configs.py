from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import shutil

from generate_config_matrix import (
    DATASETS,
    REPO_ROOT_TOKEN,
    apmg_payload,
    common_training,
    dc_payload,
    dump,
    evaluation,
    fv_payload,
    log_config,
    mc_payload,
    neural_payload,
    rm_payload,
    unified_data,
    var_training,
)


ROOT = Path(__file__).resolve().parents[1]
EXPLORATION_CONFIGS = ROOT / "configs_exploration"
SIZE_NAME = "Size163"
EXPERIMENT_ROOT = f"{REPO_ROOT_TOKEN}/runs/exploration"
PROBE = {
    "enabled": True,
    "total_epoch_equivalents": 50,
    "every_epoch_equivalents": 5,
    "sample_ratio": 0.01,
    "max_samples": 100_000,
    "seed": 42,
}
TARGETS = tuple(DATASETS["ionization"]["targets"])

SINGLE_PROFILES = {
    "SIREN": {
        "depth2": {"name": "siren", "in_features": 4, "hidden_features": 290, "hidden_layers": 2, "first_omega_0": 30.0, "hidden_omega_0": 30.0, "outermost_linear": True},
        "depth3": {"name": "siren", "in_features": 4, "hidden_features": 237, "hidden_layers": 3, "first_omega_0": 30.0, "hidden_omega_0": 30.0, "outermost_linear": True},
        "depth5": {"name": "siren", "in_features": 4, "hidden_features": 184, "hidden_layers": 5, "first_omega_0": 30.0, "hidden_omega_0": 30.0, "outermost_linear": True},
    },
    "CoordNet": {
        "res5": {"name": "coordnet", "in_features": 4, "init_features": 29, "num_res": 5},
        "res10": {"name": "coordnet", "in_features": 4, "init_features": 22, "num_res": 10},
        "res15": {"name": "coordnet", "in_features": 4, "init_features": 18, "num_res": 15},
    },
    "MoE-INR": {
        "experts4": {"name": "moe_inr", "in_features": 4, "num_experts": 4, "base_dim": 46, "policy_num_layers": 3},
        "experts7": {"name": "moe_inr", "in_features": 4, "num_experts": 7, "base_dim": 46, "policy_num_layers": 3},
        "experts10": {"name": "moe_inr", "in_features": 4, "num_experts": 10, "base_dim": 46, "policy_num_layers": 3},
    },
}

VAR_PROFILES = {
    "experts4": {"num_experts": 4, "base_dim": 27, "top_k": 3},
    "experts6": {"num_experts": 6, "base_dim": 24, "top_k": 3},
    "experts8": {"num_experts": 8, "base_dim": 22, "top_k": 3},
}
MC_PROFILES = {
    "depth3_4": {"hidden_features": 43, "gfe_layers": 3, "lfe_layers": 4},
    "depth5_6": {"hidden_features": 33, "gfe_layers": 5, "lfe_layers": 6},
    "depth7_8": {"hidden_features": 28, "gfe_layers": 7, "lfe_layers": 8},
}
NEURAL_PROFILES = {
    "depth1": {"dim": 24, "depth": 1},
    "depth2": {"dim": 23, "depth": 2},
    "depth3": {"dim": 23, "depth": 3},
}
APMG_PROFILES = {
    "decoder_heavy": {"feature_grid_shape": [5, 5, 5], "n_grids": 1, "n_features": 1, "nodes_per_layer": 32, "n_layers": 2},
    "balanced": {"feature_grid_shape": [4, 4, 4], "n_grids": 1, "n_features": 14, "nodes_per_layer": 16, "n_layers": 3},
    "grid_heavy": {"feature_grid_shape": [7, 7, 7], "n_grids": 4, "n_features": 1, "nodes_per_layer": 16, "n_layers": 1},
}
FV_PROFILES = {
    "decoder_heavy": {"grid_resolution": 2, "grid_channels": 20, "hidden_features": 384, "hidden_layers": 2},
    "balanced": {"grid_resolution": 4, "grid_channels": 112, "hidden_features": 128, "hidden_layers": 5},
    "grid_heavy": {"grid_resolution": 7, "grid_channels": 41, "hidden_features": 16, "hidden_layers": 2},
}
RM_PROFILES = {
    "decoder_heavy": {"grid_resolution": 4, "grid_channels": 8, "decoder_count": 3, "decoder_hidden_features": 128, "decoder_hidden_layers": 4},
    "balanced": {"grid_resolution": 5, "grid_channels": 58, "decoder_count": 3, "decoder_hidden_features": 128, "decoder_hidden_layers": 2},
    "grid_heavy": {"grid_resolution": 10, "grid_channels": 14, "decoder_count": 3, "decoder_hidden_features": 16, "decoder_hidden_layers": 1},
}


def _tag(payload: dict, family: str, profile: str, target: str | None) -> dict:
    result = deepcopy(payload)
    slug = family.lower().replace("-", "_")
    target_suffix = f"-{target}" if target else ""
    result["experiment"] = f"exploration_{slug}_{profile}{target_suffix}"
    result["exp_id"] = f"explore-{slug}-size163-{profile}{target_suffix}"
    result["experiment_root"] = EXPERIMENT_ROOT
    result["exploration_probe"] = deepcopy(PROBE)
    return result


def _quick_unified_training() -> dict:
    training = common_training()
    training.update({"epochs": 50, "log_every": 1, "log_psnr_every": 5, "psnr_sample_ratio": 0.01, "save_every": 50})
    return training


def generate_unified_single() -> int:
    count = 0
    for family, profiles in SINGLE_PROFILES.items():
        for profile, model in profiles.items():
            for target in TARGETS:
                payload = {
                    "data": unified_data("ionization", True, target),
                    "model": deepcopy(model),
                    "training": _quick_unified_training(),
                    "evaluation": evaluation(),
                    "log": log_config(),
                }
                payload = _tag(payload, family, profile, target)
                dump(EXPLORATION_CONFIGS / family / SIZE_NAME / profile / f"ionization__{target}.yaml", payload)
                count += 1
    return count


def generate_var_expert() -> int:
    count = 0
    for profile, model in VAR_PROFILES.items():
        payload = {
            "data": unified_data("ionization", True),
            "model": {"name": "var_expert", "in_features": 4, **model},
            "training": var_training("ionization", True, model["num_experts"]),
            "evaluation": evaluation(),
            "log": log_config(),
        }
        payload["training"].update({"epochs": 50, "log_every": 1, "log_psnr_every": 5, "psnr_sample_ratio": 0.01, "save_every": 50})
        payload = _tag(payload, "VarExpert", profile, None)
        dump(EXPLORATION_CONFIGS / "VarExpert" / SIZE_NAME / profile / "ionization.yaml", payload)
        count += 1
    return count


def generate_mc_inr() -> int:
    count = 0
    for profile, model in MC_PROFILES.items():
        payload = mc_payload("ionization", True, model["hidden_features"])
        payload["model"].update(model)
        payload["training"].update({"meta_iterations": 5, "finetune_epochs": 50, "log_every": 1, "save_every": 50})
        payload = _tag(payload, "MC-INR", profile, None)
        dump(EXPLORATION_CONFIGS / "MC-INR" / SIZE_NAME / profile / "ionization.yaml", payload)
        count += 1
    return count


def generate_neural_expert() -> int:
    count = 0
    for profile, values in NEURAL_PROFILES.items():
        for target in TARGETS:
            manager_path = (
                f"{EXPERIMENT_ROOT}/neural_expert/pretrained_managers/ionization/size163/{profile}/"
                f"pt_inr_moe_ionization_{target}_managerpretraining.pth"
            )
            for pretrain in (True, False):
                payload = neural_payload("ionization", target, True, pretrain, values["dim"], SIZE_NAME)
                model = payload["MODEL"]
                model["decoder_n_hidden_layers"] = values["depth"]
                model["manager_n_hidden_layers"] = values["depth"]
                model["manager_pt_path"] = manager_path
                payload["TRAINING"]["num_epochs"] = 2_500 if pretrain else 75_000
                payload["TRAINING"]["log_every"] = 100 if pretrain else 7_500
                payload["TRAINING"]["save_every"] = payload["TRAINING"]["num_epochs"]
                payload = _tag(payload, "NeuralExpert", profile, target)
                if pretrain:
                    payload["experiment"] += "-managerpretrain"
                    payload["exp_id"] += "-managerpretrain"
                suffix = "__managerpretrain" if pretrain else ""
                dump(EXPLORATION_CONFIGS / "NeuralExpert" / SIZE_NAME / profile / f"ionization__{target}{suffix}.yaml", payload)
                count += 1
    return count


def generate_apmgsrn() -> int:
    count = 0
    for profile, model in APMG_PROFILES.items():
        for target in TARGETS:
            payload = apmg_payload(target, True, SIZE_NAME)
            payload["MODEL"].update(deepcopy(model))
            payload["TRAINING"].update({"iterations": 750, "log_every": 75, "save_every": 750})
            payload = _tag(payload, "APMGSRN", profile, target)
            dump(EXPLORATION_CONFIGS / "APMGSRN" / SIZE_NAME / profile / f"ionization__{target}.yaml", payload)
            count += 1
    return count


def generate_dc_inr() -> int:
    count = 0
    for maximum in (512, 1024, 2048):
        profile = f"max{maximum}"
        for target in TARGETS:
            payload = dc_payload(target, True, SIZE_NAME)
            payload["compression"]["max_initial_neurons"] = maximum
            payload["training"].update({"total_steps": 75_000, "log_every": 7_500})
            payload = _tag(payload, "DC-INR", profile, target)
            dump(EXPLORATION_CONFIGS / "DC-INR" / SIZE_NAME / profile / f"ionization__{target}.yaml", payload)
            count += 1
    return count


def _generate_temporal(family: str, builder, profiles: dict[str, dict]) -> int:
    count = 0
    for profile, model in profiles.items():
        for target in TARGETS:
            payload = builder(target, True, SIZE_NAME)
            payload["model"].update(deepcopy(model))
            if family == "fV-SRN":
                payload["training"].update({"epochs": 50, "log_every": 1, "save_every": 50})
            else:
                payload["training"].update({"steps": 75_000, "log_every": 7_500, "save_every": 75_000})
            payload = _tag(payload, family, profile, target)
            dump(EXPLORATION_CONFIGS / family / SIZE_NAME / profile / f"ionization__{target}.yaml", payload)
            count += 1
    return count


def main() -> None:
    if EXPLORATION_CONFIGS.exists():
        shutil.rmtree(EXPLORATION_CONFIGS)
    EXPLORATION_CONFIGS.mkdir(parents=True)
    counts = {
        "unified_single": generate_unified_single(),
        "var_expert": generate_var_expert(),
        "mc_inr": generate_mc_inr(),
        "neural_expert": generate_neural_expert(),
        "apmgsrn": generate_apmgsrn(),
        "dc_inr": generate_dc_inr(),
        "fv_srn": _generate_temporal("fV-SRN", fv_payload, FV_PROFILES),
        "rmdsrn": _generate_temporal("RMDSRN", rm_payload, RM_PROFILES),
    }
    total = sum(counts.values())
    if total != 141:
        raise RuntimeError(f"Expected 141 exploration configs, generated {total}: {counts}")
    print(f"Generated {total} exploration configs: {counts}")


if __name__ == "__main__":
    main()
