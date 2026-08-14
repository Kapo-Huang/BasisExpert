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
FORMAL_ROOT = ROOT / "configs"
CONFIG_ROOT = ROOT / "configs_exploration_v4"
RUN_ROOT = f"{REPO_ROOT_TOKEN}/runs/exploration_v4"
EXPECTED_TOTAL = 81
PROBE = {
    "enabled": True,
    "total_epoch_equivalents": 50,
    "every_epoch_equivalents": 5,
    "sample_ratio": 0.01,
    "max_samples": 100_000,
    "seed": 42,
    "retain_best_checkpoint": True,
}

COORD_SIZES = ("Size326", "Size652", "Size1304")
COORD_TARGETS = ("GT", "H_plus", "He")
COORD_DEPTHS = (2, 3, 5, 7, 10)
COORD_FORMAL_WIDTHS = {"Size326": 31, "Size652": 43, "Size1304": 61}
COORD_SCALED_LR = {"Size1304": 1.25e-5}
COORD_CONTROL_PROFILES = (
    "res10_scaled_lr",
    "res10_clip",
    "res5_scaled_lr_clip",
)

RMDSRN_SIZES = ("Size082", "Size163", "Size326", "Size652", "Size1304")
RMDSRN_TARGETS = ("GT", "H_plus", "PD")
RMDSRN_PROFILES = {
    "schedule900k_lambda10": 10.0,
    "schedule900k_lambda1": 1.0,
    "schedule900k_lambda0": 0.0,
}
RMDSRN_ABLATION_SIZES = {"Size082", "Size1304"}


def coordnet_param_count(init_features: int, num_res: int) -> int:
    width = int(init_features)
    residuals = int(num_res)
    if width <= 0 or residuals < 0:
        raise ValueError("CoordNet width must be positive and num_res non-negative")
    return (41 + 32 * residuals) * width * width + (37 + 8 * residuals) * width + 4


def solve_coordnet_width(target_params: int, num_res: int) -> int:
    return min(
        range(1, 257),
        key=lambda width: abs(coordnet_param_count(width, num_res) - int(target_params)),
    )


def coordnet_widths() -> dict[str, dict[int, int]]:
    result: dict[str, dict[int, int]] = {}
    for size, formal_width in COORD_FORMAL_WIDTHS.items():
        target_params = coordnet_param_count(formal_width, 10)
        result[size] = {
            depth: solve_coordnet_width(target_params, depth)
            for depth in COORD_DEPTHS
        }
    return result


def _load_formal(family: str, size: str, target: str) -> dict:
    path = FORMAL_ROOT / family / size / f"ionization__{target}.yaml"
    if not path.is_file():
        raise FileNotFoundError(path)
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _tag(payload: dict, *, family: str, size: str, profile: str, target: str) -> dict:
    result = deepcopy(payload)
    slug = family.lower().replace("-", "")
    result["experiment"] = f"exploration_v4_{slug}_{size.lower()}_{profile}_{target}"
    result["exp_id"] = f"explore-v4-{slug}-{size.lower()}-{profile}-{target}"
    result["experiment_root"] = RUN_ROOT
    result["exploration_probe"] = deepcopy(PROBE)
    return result


def _quick_coordnet(payload: dict) -> None:
    payload["training"].update(
        {
            "epochs": 50,
            "log_every": 1,
            "log_psnr_every": 5,
            "psnr_sample_ratio": 0.01,
            "save_every": 50,
        }
    )


def generate_coordnet() -> int:
    count = 0
    widths = coordnet_widths()
    for size in COORD_SIZES:
        for depth in COORD_DEPTHS:
            profile = f"res{depth}_base_lr"
            for target in COORD_TARGETS:
                payload = _tag(
                    _load_formal("CoordNet", size, target),
                    family="CoordNet",
                    size=size,
                    profile=profile,
                    target=target,
                )
                payload["model"].update(
                    {"init_features": widths[size][depth], "num_res": depth}
                )
                _quick_coordnet(payload)
                dump(
                    CONFIG_ROOT / "CoordNet" / size / profile / f"ionization__{target}.yaml",
                    payload,
                )
                count += 1

    size = "Size1304"
    for profile in COORD_CONTROL_PROFILES:
        for target in COORD_TARGETS:
            payload = _tag(
                _load_formal("CoordNet", size, target),
                family="CoordNet",
                size=size,
                profile=profile,
                target=target,
            )
            _quick_coordnet(payload)
            if profile == "res10_scaled_lr":
                payload["training"]["lr"] = COORD_SCALED_LR[size]
            elif profile == "res10_clip":
                payload["training"]["grad_clip_norm"] = 1.0
            elif profile == "res5_scaled_lr_clip":
                payload["model"].update(
                    {"init_features": widths[size][5], "num_res": 5}
                )
                payload["training"].update(
                    {"lr": COORD_SCALED_LR[size], "grad_clip_norm": 1.0}
                )
            dump(
                CONFIG_ROOT / "CoordNet" / size / profile / f"ionization__{target}.yaml",
                payload,
            )
            count += 1
    return count


def generate_rmdsrn() -> int:
    count = 0
    for profile, lambda_max in RMDSRN_PROFILES.items():
        sizes = (
            RMDSRN_SIZES
            if profile == "schedule900k_lambda10"
            else tuple(size for size in RMDSRN_SIZES if size in RMDSRN_ABLATION_SIZES)
        )
        for size in sizes:
            for target in RMDSRN_TARGETS:
                formal = _load_formal("RMDSRN", size, target)
                formal_steps = int(formal["training"]["steps"])
                payload = _tag(
                    formal,
                    family="RMDSRN",
                    size=size,
                    profile=profile,
                    target=target,
                )
                payload["training"].update(
                    {
                        "steps": 75_000,
                        "lr_schedule_steps": formal_steps,
                        "lambda_schedule_steps": formal_steps,
                        "lambda_max": float(lambda_max),
                        "log_every": 7_500,
                        "save_every": 75_000,
                    }
                )
                dump(
                    CONFIG_ROOT / "RMDSRN" / size / profile / f"ionization__{target}.yaml",
                    payload,
                )
                count += 1
    return count


def generate() -> dict[str, int]:
    if CONFIG_ROOT.exists():
        shutil.rmtree(CONFIG_ROOT)
    CONFIG_ROOT.mkdir(parents=True)
    counts = {"CoordNet": generate_coordnet(), "RMDSRN": generate_rmdsrn()}
    total = sum(counts.values())
    if total != EXPECTED_TOTAL:
        raise RuntimeError(f"Expected {EXPECTED_TOTAL} configs, generated {total}: {counts}")
    return counts


def main() -> None:
    counts = generate()
    print(f"Generated {sum(counts.values())} exploration-v4 configs: {counts}")


if __name__ == "__main__":
    main()
