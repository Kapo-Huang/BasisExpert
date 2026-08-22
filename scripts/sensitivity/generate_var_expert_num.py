from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import shutil
import sys


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.main.generate_configs import (
    REPO_ROOT_TOKEN,
    dump,
    evaluation,
    log_config,
    unified_data,
    var_training,
)


CONFIG_ROOT = ROOT / "configs/sensitivity/var_expert_num"
RUN_ROOT = f"{REPO_ROOT_TOKEN}/runs/sensitivity/var_expert_num"
SIZE = "Size163"
EXPERT_PROFILES = {
    1: {"name": "shared_enc_inr", "base_dim": 36},
    2: {"name": "var_expert", "base_dim": 31, "top_k": 2},
    3: {"name": "var_expert", "base_dim": 29, "top_k": 3},
    4: {"name": "var_expert", "base_dim": 27, "top_k": 3},
    5: {"name": "var_expert", "base_dim": 25, "top_k": 3},
    6: {"name": "var_expert", "base_dim": 24, "top_k": 3},
    7: {"name": "var_expert", "base_dim": 23, "top_k": 3},
    8: {"name": "var_expert", "base_dim": 22, "top_k": 3},
}
EXPECTED_TOTAL = len(EXPERT_PROFILES)
PROBE = {
    "enabled": True,
    "total_epoch_equivalents": 50,
    "every_epoch_equivalents": 5,
    "sample_ratio": 0.01,
    "max_samples": 100_000,
    "seed": 42,
}


def _profile(experts: int, top_k: int | None) -> str:
    if int(experts) == 1:
        return "experts1_shared_enc"
    if top_k is None:
        raise ValueError("VarExpert profiles require top_k")
    return f"experts{int(experts)}_top{int(top_k)}"


def build_payload(*, experts: int, model_name: str, base_dim: int, top_k: int | None) -> dict:
    profile = _profile(experts, top_k)
    if model_name == "shared_enc_inr":
        if int(experts) != 1 or top_k is not None:
            raise ValueError("shared_enc_inr is only valid for the one-expert control")
        model = {
            "name": "shared_enc_inr",
            "in_features": 4,
            "base_dim": int(base_dim),
        }
    elif model_name == "var_expert":
        if int(experts) < 2 or top_k is None:
            raise ValueError("var_expert sensitivity profiles require experts >= 2 and top_k")
        model = {
            "name": "var_expert",
            "in_features": 4,
            "num_experts": int(experts),
            "base_dim": int(base_dim),
            "top_k": int(top_k),
        }
    else:
        raise ValueError(f"Unsupported expert-count control model: {model_name}")

    training = var_training("ionization", True, experts)
    if model_name == "shared_enc_inr":
        training["pretrain"] = {"enabled": False, "epochs": 0, "lr": 5.0e-5}
    payload = {
        "data": unified_data("ionization", True),
        "model": model,
        "training": training,
        "evaluation": evaluation(),
        "log": log_config(),
        "experiment": f"sensitivity_varexpert_expert_num_{SIZE.lower()}_{profile}",
        "exp_id": f"sensitivity-varexpert-expert-num-{SIZE.lower()}-{profile}",
        "experiment_root": RUN_ROOT,
        "exploration_probe": deepcopy(PROBE),
    }
    payload["training"].update(
        {
            "epochs": 50,
            "log_every": 1,
            "log_psnr_every": 5,
            "psnr_sample_ratio": 0.01,
            "save_every": 50,
        }
    )
    return payload


def generate() -> int:
    if CONFIG_ROOT.exists():
        shutil.rmtree(CONFIG_ROOT)
    CONFIG_ROOT.mkdir(parents=True)

    exp_ids: set[str] = set()
    generated: set[tuple[int, str, int, int | None]] = set()
    for experts, values in EXPERT_PROFILES.items():
        model_name = str(values["name"])
        base_dim = int(values["base_dim"])
        top_k = int(values["top_k"]) if values.get("top_k") is not None else None
        payload = build_payload(
            experts=experts,
            model_name=model_name,
            base_dim=base_dim,
            top_k=top_k,
        )
        exp_id = str(payload["exp_id"])
        if exp_id in exp_ids:
            raise ValueError(f"Duplicate VarExpert expert-count exp_id: {exp_id}")
        exp_ids.add(exp_id)
        generated.add((int(experts), model_name, base_dim, top_k))
        dump(
            CONFIG_ROOT / "VarExpert" / SIZE / _profile(experts, top_k) / "ionization.yaml",
            payload,
        )

    expected = {
        (
            int(experts),
            str(values["name"]),
            int(values["base_dim"]),
            int(values["top_k"]) if values.get("top_k") is not None else None,
        )
        for experts, values in EXPERT_PROFILES.items()
    }
    if generated != expected or len(generated) != EXPECTED_TOTAL:
        raise RuntimeError(
            f"Expected {EXPECTED_TOTAL} VarExpert expert-count configs, generated {sorted(generated)}"
        )
    return len(generated)


def main() -> None:
    count = generate()
    print(f"Generated {count} VarExpert expert-count sensitivity configs under {CONFIG_ROOT}")


if __name__ == "__main__":
    main()
