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

FORMAL_ROOT = ROOT / "configs" / "main" / "ECNR"
CONFIG_ROOT = ROOT / "configs/exploration/ecnr_tuning"
RUN_ROOT = f"{RUNS_ROOT_TOKEN}/exploration_v6"
TARGETS = ("GT", "H2", "H_plus")
STRUCTURE = "official_main"
SMOKE_TRAINING = {
    "epochs_per_scale": 50,
    "batch_size": 3_200,
    "passes_per_epoch": 1,
    "pruning_epochs": [15, 23, 30, 38],
    "pruning_sparsities": [0.30, 0.40, 0.45, 0.50],
    "quantization_finetune_epochs": 8,
    "quantization_finetune_passes_per_epoch": 1,
    "save_every": 0,
    "log_every": 1,
    "progress_log_seconds": 60,
    "seed": 42,
}

PROFILES = {
    "official_control": {},
    "lr5e4": {"lr": 5.0e-4},
    "lr2e3": {"lr": 2.0e-3},
    "no_weight_decay": {"weight_decay": 0.0},
    "pruning_gamma09": {"pruning_lr_gamma": 0.9},
    "quant_lr5e5": {"quantization_finetune_lr": 5.0e-5},
}


def _read(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Missing formal ECNR config: {path}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Config must contain a mapping: {path}")
    return payload


def _profile_payload(source: dict, *, target: str, profile: str) -> dict:
    payload = deepcopy(source)
    payload["experiment"] = f"exploration_v6_ecnr_{STRUCTURE}_{profile}_{target}"
    payload["exp_id"] = f"explore-v6-ecnr-{STRUCTURE.replace('_', '-')}-{profile.replace('_', '-')}-{target}"
    payload["experiment_root"] = RUN_ROOT
    payload["data"]["target"] = target
    payload["training"].update(deepcopy(SMOKE_TRAINING))
    payload["training"].update(deepcopy(PROFILES[profile]))
    payload["cnn"]["epochs"] = 10
    payload["evaluation"].update(
        {
            "save_predictions": False,
            "run_after_training": True,
            "default_model": "checkpoint",
        }
    )
    return payload


def generate() -> int:
    if CONFIG_ROOT.exists():
        shutil.rmtree(CONFIG_ROOT)
    count = 0
    for profile in PROFILES:
        for target in TARGETS:
            source = _read(FORMAL_ROOT / f"ionization__{target}.yaml")
            payload = _profile_payload(source, target=target, profile=profile)
            destination = (
                CONFIG_ROOT
                / "ECNR"
                / STRUCTURE
                / profile
                / f"ionization__{target}.yaml"
            )
            dump(destination, payload)
            count += 1
    return count


def main() -> None:
    count = generate()
    print(f"Generated {count} exploration-v6 ECNR configs")


if __name__ == "__main__":
    main()
