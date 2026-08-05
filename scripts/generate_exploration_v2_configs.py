from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import shutil

from generate_config_matrix import (
    DATASETS,
    REPO_ROOT_TOKEN,
    SINGLE_TARGET_SIZES,
    dc_payload,
    dump,
    evaluation,
    log_config,
    mc_payload,
    neural_payload,
    unified_data,
    var_training,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = ROOT / "configs_exploration_v2"
RUN_ROOT = f"{REPO_ROOT_TOKEN}/runs/exploration_v2"
TARGETS = tuple(DATASETS["ionization"]["targets"])
PROBE = {
    "enabled": True,
    "total_epoch_equivalents": 50,
    "every_epoch_equivalents": 5,
    "sample_ratio": 0.01,
    "max_samples": 100_000,
    "seed": 42,
}

# Size labels are total five-variable FP16 budgets.  A single-variable model
# receives one fifth of that budget; a multi-variable model receives all of it.
SINGLE_SIZE = "Size326"
MULTI_SIZE = "Size163"
NEURAL_PROFILES = {
    "depth1": {"dim": 34, "depth": 1},
    "depth2": {"dim": 33, "depth": 2},
    "depth3": {"dim": 32, "depth": 3},
}
MC_PROFILES = {
    "depth3_4": {"hidden_features": 43, "gfe_layers": 3, "lfe_layers": 4},
    "depth5_6": {"hidden_features": 33, "gfe_layers": 5, "lfe_layers": 6},
    "depth7_8": {"hidden_features": 28, "gfe_layers": 7, "lfe_layers": 8},
}
DC_EPS_PROFILES = {
    "eps0p01": 0.01,
    "eps0p05": 0.05,
    "eps0p10": 0.10,
}
VAR_BASE_DIMS = {8: 22, 9: 21, 10: 20}


def _tag(payload: dict, family: str, size: str, profile: str, target: str | None) -> dict:
    tagged = deepcopy(payload)
    slug = family.lower().replace("-", "_")
    target_suffix = f"-{target}" if target else ""
    tagged["experiment"] = f"exploration_v2_{slug}_{size.lower()}_{profile}{target_suffix}"
    tagged["exp_id"] = f"explore-v2-{slug}-{size.lower()}-{profile}{target_suffix}"
    tagged["experiment_root"] = RUN_ROOT
    tagged["exploration_probe"] = deepcopy(PROBE)
    return tagged


def generate_mc_inr() -> int:
    count = 0
    for profile, model in MC_PROFILES.items():
        payload = mc_payload("ionization", True, model["hidden_features"])
        payload["model"].update(model)
        payload["training"].update(
            {"meta_iterations": 5, "finetune_epochs": 50, "log_every": 1, "save_every": 50}
        )
        payload = _tag(payload, "MC-INR", MULTI_SIZE, profile, None)
        dump(CONFIG_ROOT / "MC-INR" / MULTI_SIZE / profile / "ionization.yaml", payload)
        count += 1
    return count


def generate_dc_inr() -> int:
    count = 0
    for profile, eps in DC_EPS_PROFILES.items():
        for target in TARGETS:
            payload = dc_payload(target, True, MULTI_SIZE)
            payload["partition"]["dbscan_eps"] = float(eps)
            payload["compression"]["max_initial_neurons"] = 512
            payload["training"].update({"total_steps": 75_000, "log_every": 7_500})
            payload = _tag(payload, "DC-INR", MULTI_SIZE, profile, target)
            dump(CONFIG_ROOT / "DC-INR" / MULTI_SIZE / profile / f"ionization__{target}.yaml", payload)
            count += 1
    return count


def generate_neural_expert() -> int:
    count = 0
    for profile, values in NEURAL_PROFILES.items():
        for target in TARGETS:
            manager_path = (
                f"{RUN_ROOT}/neural_expert/pretrained_managers/ionization/size326/{profile}/"
                f"pt_inr_moe_ionization_{target}_managerpretraining.pth"
            )
            for pretrain in (True, False):
                payload = neural_payload(
                    "ionization",
                    target,
                    True,
                    pretrain,
                    values["dim"],
                    SINGLE_SIZE,
                )
                model = payload["MODEL"]
                model["decoder_n_hidden_layers"] = values["depth"]
                model["manager_n_hidden_layers"] = values["depth"]
                model["manager_pt_path"] = manager_path
                payload["TRAINING"]["num_epochs"] = 2_500 if pretrain else 75_000
                payload["TRAINING"]["log_every"] = 100 if pretrain else 7_500
                payload["TRAINING"]["save_every"] = payload["TRAINING"]["num_epochs"]
                payload = _tag(payload, "NeuralExpert", SINGLE_SIZE, profile, target)
                if pretrain:
                    payload["experiment"] += "-managerpretrain"
                    payload["exp_id"] += "-managerpretrain"
                suffix = "__managerpretrain" if pretrain else ""
                dump(
                    CONFIG_ROOT
                    / "NeuralExpert"
                    / SINGLE_SIZE
                    / profile
                    / f"ionization__{target}{suffix}.yaml",
                    payload,
                )
                count += 1
    return count


def _var_profile(experts: int, top_k: int) -> str:
    return f"experts{int(experts)}_top{int(top_k)}"


def generate_var_expert() -> int:
    count = 0
    # experts8/top3 is a same-budget control for the previous exploration.
    combinations = [(8, 3)]
    combinations.extend((9, top_k) for top_k in range(1, 10))
    combinations.extend((10, top_k) for top_k in range(1, 11))
    for experts, top_k in combinations:
        profile = _var_profile(experts, top_k)
        payload = {
            "data": unified_data("ionization", True),
            "model": {
                "name": "var_expert",
                "in_features": 4,
                "num_experts": int(experts),
                "base_dim": int(VAR_BASE_DIMS[experts]),
                "top_k": int(top_k),
            },
            "training": var_training("ionization", True, experts),
            "evaluation": evaluation(),
            "log": log_config(),
        }
        payload["training"].update(
            {"epochs": 50, "log_every": 1, "log_psnr_every": 5, "psnr_sample_ratio": 0.01, "save_every": 50}
        )
        payload = _tag(payload, "VarExpert", MULTI_SIZE, profile, None)
        dump(CONFIG_ROOT / "VarExpert" / MULTI_SIZE / profile / "ionization.yaml", payload)
        count += 1
    return count


def main() -> None:
    if CONFIG_ROOT.exists():
        shutil.rmtree(CONFIG_ROOT)
    CONFIG_ROOT.mkdir(parents=True)
    counts = {
        "mc_inr": generate_mc_inr(),
        "dc_inr": generate_dc_inr(),
        "neural_expert": generate_neural_expert(),
        "var_expert": generate_var_expert(),
    }
    total = sum(counts.values())
    if total != 68:
        raise RuntimeError(f"Expected 68 exploration-v2 configs, generated {total}: {counts}")
    print(
        "Generated exploration-v2 with "
        f"{total} configs (single target {SINGLE_SIZE}={SINGLE_TARGET_SIZES[SINGLE_SIZE]:.3f}MiB, "
        f"multi target {MULTI_SIZE}=1.630MiB): {counts}"
    )


if __name__ == "__main__":
    main()
