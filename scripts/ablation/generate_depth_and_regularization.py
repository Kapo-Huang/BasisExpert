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

FORMAL_ROOT = ROOT / "configs" / "rd_curve"
CONFIG_ROOT = ROOT / "configs/ablation/depth_and_regularization"
RUN_ROOT = f"{REPO_ROOT_TOKEN}/runs/exploration_v4"
EXPECTED_TOTAL = 30
PROBE = {
    "enabled": True,
    "total_epoch_equivalents": 50,
    "every_epoch_equivalents": 5,
    "sample_ratio": 0.01,
    "max_samples": 100_000,
    "seed": 42,
    "retain_best_checkpoint": True,
}

COORD_SIZES = ("Size326", "Size652")
COORD_TARGETS = ("GT", "H_plus", "He")
COORD_DEPTHS = (2, 3, 5, 7, 10)
COORD_FORMAL_WIDTHS = {"Size326": 31, "Size652": 43}


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

    return count


def generate() -> dict[str, int]:
    if CONFIG_ROOT.exists():
        shutil.rmtree(CONFIG_ROOT)
    CONFIG_ROOT.mkdir(parents=True)
    counts = {"CoordNet": generate_coordnet()}
    total = sum(counts.values())
    if total != EXPECTED_TOTAL:
        raise RuntimeError(f"Expected {EXPECTED_TOTAL} configs, generated {total}: {counts}")
    return counts


def main() -> None:
    counts = generate()
    print(f"Generated {sum(counts.values())} exploration-v4 configs: {counts}")


if __name__ == "__main__":
    main()
