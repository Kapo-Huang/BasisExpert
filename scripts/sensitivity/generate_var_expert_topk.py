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


CONFIG_ROOT = ROOT / "configs/sensitivity/var_expert_topk"
RUN_ROOT = f"{REPO_ROOT_TOKEN}/runs/sensitivity/var_expert_topk"
SIZE = "Size163"
NUM_EXPERTS = 7
BASE_DIM = 23
TOP_K_VALUES = tuple(range(1, NUM_EXPERTS + 1))
EXPECTED_TOTAL = len(TOP_K_VALUES)
PROBE = {
    "enabled": True,
    "total_epoch_equivalents": 50,
    "every_epoch_equivalents": 5,
    "sample_ratio": 0.01,
    "max_samples": 100_000,
    "seed": 42,
}


def _profile(top_k: int) -> str:
    return f"experts{NUM_EXPERTS}_top{int(top_k)}"


def build_payload(*, top_k: int) -> dict:
    profile = _profile(top_k)
    payload = {
        "data": unified_data("ionization", True),
        "model": {
            "name": "var_expert",
            "in_features": 4,
            "num_experts": NUM_EXPERTS,
            "base_dim": BASE_DIM,
            "top_k": int(top_k),
        },
        "training": var_training("ionization", True, NUM_EXPERTS),
        "evaluation": evaluation(),
        "log": log_config(),
        "experiment": f"sensitivity_varexpert_topk_{SIZE.lower()}_{profile}",
        "exp_id": f"sensitivity-varexpert-topk-{SIZE.lower()}-{profile}",
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
    generated: set[tuple[int, int, int]] = set()
    for top_k in TOP_K_VALUES:
        payload = build_payload(top_k=top_k)
        exp_id = str(payload["exp_id"])
        if exp_id in exp_ids:
            raise ValueError(f"Duplicate VarExpert top-k exp_id: {exp_id}")
        exp_ids.add(exp_id)
        generated.add((NUM_EXPERTS, BASE_DIM, int(top_k)))
        dump(
            CONFIG_ROOT / "VarExpert" / SIZE / _profile(top_k) / "ionization.yaml",
            payload,
        )

    expected = {(NUM_EXPERTS, BASE_DIM, int(top_k)) for top_k in TOP_K_VALUES}
    if generated != expected or len(generated) != EXPECTED_TOTAL:
        raise RuntimeError(
            f"Expected {EXPECTED_TOTAL} VarExpert top-k configs, generated {sorted(generated)}"
        )
    return len(generated)


def main() -> None:
    count = generate()
    print(f"Generated {count} VarExpert top-k sensitivity configs under {CONFIG_ROOT}")


if __name__ == "__main__":
    main()
