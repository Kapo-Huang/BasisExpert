from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import shutil
import sys

import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.main.generate_configs import REPO_ROOT_TOKEN, dump

MAIN_CONFIG_ROOT = ROOT / "configs" / "main"
RD_CURVE_CONFIG_ROOT = ROOT / "configs" / "rd_curve"
CONFIG_ROOT = ROOT / "configs/exploration/optimizer_tuning"
RUN_ROOT = f"{REPO_ROOT_TOKEN}/runs/exploration_v5"
EXPECTED_TOTAL = 42
TARGETS = ("GT", "H2", "H_plus")

PROBE = {
    "enabled": True,
    "total_epoch_equivalents": 50,
    "every_epoch_equivalents": 5,
    "sample_ratio": 0.01,
    "max_samples": 100_000,
    "seed": 42,
}

FV_STRUCTURES = {
    "formal_size163": {},
    "grid_heavy": {
        "grid_resolution": 7,
        "grid_channels": 41,
        "hidden_features": 16,
        "hidden_layers": 2,
    },
}
FV_PROFILES = {
    "lr1e2_step100": {"lr": 1.0e-2, "lr_step": 100, "lr_gamma": 0.5},
    "lr5e3_step100": {"lr": 5.0e-3, "lr_step": 100, "lr_gamma": 0.5},
    "lr1e2_step20": {"lr": 1.0e-2, "lr_step": 20, "lr_gamma": 0.5},
    "lr5e3_step20": {"lr": 5.0e-3, "lr_step": 20, "lr_gamma": 0.5},
}

INSTANT_STRUCTURE = "official_default"
INSTANT_SIZE_MATCHED_MODEL_UPDATE = {
    "log2_hashmap_size": 11,
    "hidden_features": 105,
}
INSTANT_PROFILES = {
    "official_control": {
        "lr": 5.0e-3,
        "loss_type": "l1",
        "scheduler_gamma": 0.99,
    },
    "lr1e3": {
        "lr": 1.0e-3,
        "loss_type": "l1",
        "scheduler_gamma": 0.99,
    },
    "lr5e4": {
        "lr": 5.0e-4,
        "loss_type": "l1",
        "scheduler_gamma": 0.99,
    },
    "lr1e3_fast_decay": {
        "lr": 1.0e-3,
        "loss_type": "l1",
        "scheduler_gamma": 0.9,
    },
    "lr1e3_clip1": {
        "lr": 1.0e-3,
        "loss_type": "l1",
        "scheduler_gamma": 0.99,
        "grad_clip_norm": 1.0,
    },
    "lr1e3_mse": {
        "lr": 1.0e-3,
        "loss_type": "mse",
        "scheduler_gamma": 0.99,
    },
}


def _read(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Missing source config: {path}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Config must contain a mapping: {path}")
    return payload


def _tag(payload: dict, *, family: str, structure: str, profile: str, target: str) -> dict:
    result = deepcopy(payload)
    slug = family.lower().replace("-", "_")
    result["experiment"] = f"exploration_v5_{slug}_{structure}_{profile}_{target}"
    result["exp_id"] = f"explore-v5-{slug.replace('_', '-')}-{structure.replace('_', '-')}-{profile.replace('_', '-')}-{target}"
    result["experiment_root"] = RUN_ROOT
    result["exploration_probe"] = deepcopy(PROBE)
    result.setdefault("data", {})["target"] = target
    return result


def generate_fv_srn() -> int:
    count = 0
    for target in TARGETS:
        source = _read(RD_CURVE_CONFIG_ROOT / "fV-SRN" / "Size163" / f"ionization__{target}.yaml")
        for structure, model_update in FV_STRUCTURES.items():
            for profile, training_update in FV_PROFILES.items():
                payload = _tag(
                    source,
                    family="fV-SRN",
                    structure=structure,
                    profile=profile,
                    target=target,
                )
                payload["model"].update(deepcopy(model_update))
                payload["training"].update(
                    {
                        "epochs": 50,
                        "save_every": 50,
                        "log_every": 1,
                        "seed": 42,
                        **deepcopy(training_update),
                    }
                )
                payload["evaluation"].update(
                    {"save_predictions": False, "run_after_training": False}
                )
                destination = (
                    CONFIG_ROOT
                    / "fV-SRN"
                    / structure
                    / profile
                    / f"ionization__{target}.yaml"
                )
                dump(destination, payload)
                count += 1
    return count


def generate_instant_vnr() -> int:
    count = 0
    for target in TARGETS:
        source = _read(MAIN_CONFIG_ROOT / "InstantVNR" / f"ionization__{target}.yaml")
        for profile, values in INSTANT_PROFILES.items():
            payload = _tag(
                source,
                family="InstantVNR",
                structure=INSTANT_STRUCTURE,
                profile=profile,
                target=target,
            )
            payload["model"].update(deepcopy(INSTANT_SIZE_MATCHED_MODEL_UPDATE))
            training = payload["training"]
            training.pop("grad_clip_norm", None)
            scheduler = deepcopy(training["scheduler"])
            scheduler["gamma"] = float(values["scheduler_gamma"])
            training.update(
                {
                    "epochs": 50,
                    "lr": float(values["lr"]),
                    "loss_type": str(values["loss_type"]),
                    "log_every": 1,
                    "log_psnr_every": 5,
                    "psnr_sample_ratio": 0.01,
                    "save_every": 50,
                    "seed": 42,
                    "scheduler": scheduler,
                }
            )
            if "grad_clip_norm" in values:
                training["grad_clip_norm"] = float(values["grad_clip_norm"])
            payload["evaluation"]["save_predictions"] = False
            destination = (
                CONFIG_ROOT
                / "InstantVNR"
                / INSTANT_STRUCTURE
                / profile
                / f"ionization__{target}.yaml"
            )
            dump(destination, payload)
            count += 1
    return count


def generate() -> dict[str, int]:
    if CONFIG_ROOT.exists():
        shutil.rmtree(CONFIG_ROOT)
    CONFIG_ROOT.mkdir(parents=True)
    counts = {
        "fV-SRN": generate_fv_srn(),
        "InstantVNR": generate_instant_vnr(),
    }
    total = sum(counts.values())
    if total != EXPECTED_TOTAL:
        raise RuntimeError(f"Expected {EXPECTED_TOTAL} configs, generated {total}: {counts}")
    return counts


def main() -> None:
    counts = generate()
    print(f"Generated {sum(counts.values())} exploration-v5 configs: {counts}")


if __name__ == "__main__":
    main()
