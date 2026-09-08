from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import shutil
import sys

import yaml


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.main.generate_configs import RUNS_ROOT_TOKEN, dump


MAIN_CONFIG = ROOT / "configs/main/VarExpert/ionization.yaml"
CONFIG_ROOT = ROOT / "configs/sensitivity/var_expert_seed"
CONFIG_LIST = ROOT / "scripts/sensitivity/var_expert_seed.list"
RUN_ROOT = f"{RUNS_ROOT_TOKEN}/sensitivity/var_expert_joint_seed"
SIZE = "Size163"
SEEDS = (43, 44, 45)


def _load_main_payload() -> dict:
    payload = yaml.safe_load(MAIN_CONFIG.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a mapping in {MAIN_CONFIG}")
    return payload


def build_payload(main_payload: dict, *, seed: int) -> dict:
    payload = deepcopy(main_payload)
    payload["experiment"] = f"sensitivity_varexpert_joint_seed_{SIZE.lower()}_seed{seed}"
    payload["exp_id"] = f"sensitivity-varexpert-joint-seed-{SIZE.lower()}-seed{seed}"
    payload["experiment_root"] = RUN_ROOT
    payload["training"]["seed"] = int(seed)
    payload["training"]["epochs"] = 100
    payload["training"]["pretrain"]["cluster_seed"] = int(seed)
    payload["training"]["pretrain"]["assignments_cache_path"] = (
        f"${{REPO_ROOT}}/data/cache/ionization_voxel_assignments_6_seed{seed}.npy"
    )
    return payload


def generate() -> int:
    if CONFIG_ROOT.exists():
        shutil.rmtree(CONFIG_ROOT)
    CONFIG_ROOT.mkdir(parents=True)

    main_payload = _load_main_payload()
    config_paths: list[str] = []
    for seed in SEEDS:
        payload = build_payload(main_payload, seed=seed)
        config_path = CONFIG_ROOT / "VarExpert" / SIZE / f"seed{seed}" / "ionization.yaml"
        dump(config_path, payload)
        config_paths.append(config_path.relative_to(ROOT).as_posix())

    CONFIG_LIST.write_text(
        "# VarExpert joint training/clustering seed sensitivity: main seed 42 is the baseline; run seeds 43-45.\n"
        + "\n".join(config_paths)
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return len(config_paths)


def main() -> None:
    count = generate()
    print(f"Generated {count} VarExpert random-seed sensitivity configs under {CONFIG_ROOT}")


if __name__ == "__main__":
    main()
