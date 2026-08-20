from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import shutil

import yaml

try:
    from scripts.generate_config_matrix import REPO_ROOT_TOKEN, dump
except ModuleNotFoundError:  # Direct execution
    from generate_config_matrix import REPO_ROOT_TOKEN, dump


ROOT = Path(__file__).resolve().parents[1]
FORMAL_ROOT = ROOT / "configs" / "CoordNet"
CONFIG_ROOT = ROOT / "configs_exploration_CoordNet"
RUN_ROOT = f"{REPO_ROOT_TOKEN}/runs/exploration_CoordNet"

SIZES = ("Size082", "Size163", "Size326", "Size652", "Size1304")
TARGETS = ("GT", "H_plus", "He")
LR_PROFILES = {
    "lr1e-5": 1.0e-5,
    "lr5e-6": 5.0e-6,
}
EXPECTED_TOTAL = len(SIZES) * len(TARGETS) * len(LR_PROFILES)

PROBE = {
    "enabled": True,
    "total_epoch_equivalents": 50,
    "every_epoch_equivalents": 5,
    "sample_ratio": 0.01,
    "max_samples": 100_000,
    "seed": 42,
    "retain_best_checkpoint": True,
}


def _load_formal(size: str, target: str) -> dict:
    path = FORMAL_ROOT / size / f"ionization__{target}.yaml"
    if not path.is_file():
        raise FileNotFoundError(path)
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def build_payload(*, size: str, target: str, profile: str, learning_rate: float) -> dict:
    payload = deepcopy(_load_formal(size, target))
    payload["experiment"] = f"exploration_CoordNet_{size.lower()}_{profile}_{target}"
    payload["exp_id"] = f"explore-coordnet-{size.lower()}-{profile}-{target}"
    payload["experiment_root"] = RUN_ROOT
    payload["model"]["num_res"] = 10
    payload["training"].update(
        {
            "epochs": 50,
            "lr": float(learning_rate),
            "log_every": 1,
            "log_psnr_every": 5,
            "psnr_sample_ratio": 0.01,
            "save_every": 50,
            "seed": 42,
        }
    )
    payload["exploration_probe"] = deepcopy(PROBE)
    return payload


def generate() -> int:
    if CONFIG_ROOT.exists():
        shutil.rmtree(CONFIG_ROOT)
    CONFIG_ROOT.mkdir(parents=True)

    exp_ids: set[str] = set()
    count = 0
    for size in SIZES:
        for profile, learning_rate in LR_PROFILES.items():
            for target in TARGETS:
                payload = build_payload(
                    size=size,
                    target=target,
                    profile=profile,
                    learning_rate=learning_rate,
                )
                exp_id = str(payload["exp_id"])
                if exp_id in exp_ids:
                    raise ValueError(f"Duplicate exploration_CoordNet exp_id: {exp_id}")
                exp_ids.add(exp_id)
                dump(
                    CONFIG_ROOT
                    / "CoordNet"
                    / size
                    / profile
                    / f"ionization__{target}.yaml",
                    payload,
                )
                count += 1

    if count != EXPECTED_TOTAL:
        raise RuntimeError(
            f"Expected {EXPECTED_TOTAL} exploration_CoordNet configs, generated {count}"
        )
    return count


def main() -> None:
    count = generate()
    print(
        f"Generated {count} exploration_CoordNet configs: "
        f"sizes={len(SIZES)}, targets={len(TARGETS)}, learning_rates={len(LR_PROFILES)}"
    )


if __name__ == "__main__":
    main()
