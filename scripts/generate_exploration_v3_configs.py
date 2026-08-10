from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import shutil

import yaml

try:
    from scripts.generate_config_matrix import REPO_ROOT_TOKEN, dump
except ModuleNotFoundError:  # Direct execution: python scripts/generate_exploration_v3_configs.py
    from generate_config_matrix import REPO_ROOT_TOKEN, dump


ROOT = Path(__file__).resolve().parents[1]
FORMAL_CONFIG_ROOT = ROOT / "configs"
CONFIG_LIST = ROOT / "scripts" / "run_rd_curve_configs.list"
CONFIG_ROOT = ROOT / "configs_exploration_v3"
RUN_ROOT = f"{REPO_ROOT_TOKEN}/runs/exploration_v3"
EXPECTED_TOTAL = 235
PROBE = {
    "enabled": True,
    "total_epoch_equivalents": 50,
    "every_epoch_equivalents": 5,
    "sample_ratio": 0.01,
    "max_samples": 100_000,
    "seed": 42,
}
GENERIC_FAMILIES = {"SIREN", "CoordNet", "MoE-INR", "VarExpert"}


def formal_config_paths() -> list[Path]:
    paths: list[Path] = []
    seen: set[Path] = set()
    for raw_line in CONFIG_LIST.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        path = (ROOT / line).resolve()
        try:
            relative = path.relative_to(FORMAL_CONFIG_ROOT.resolve())
        except ValueError as exc:
            raise ValueError(f"RD-curve config escapes the formal config root: {line}") from exc
        if len(relative.parts) != 3 or not relative.parts[1].startswith("Size"):
            raise ValueError(f"RD-curve entry is not a formal Size config: {line}")
        if path.suffix != ".yaml" or not path.is_file():
            raise FileNotFoundError(f"Missing formal Size config: {path}")
        if path in seen:
            raise ValueError(f"Duplicate RD-curve config: {line}")
        seen.add(path)
        paths.append(path)
    if len(paths) != EXPECTED_TOTAL:
        raise RuntimeError(f"Expected {EXPECTED_TOTAL} formal Size configs, found {len(paths)}")
    return paths


def _tag(payload: dict, family: str, size: str) -> dict:
    result = deepcopy(payload)
    result["experiment"] = f"exploration_v3_{result['experiment']}"
    result["exp_id"] = f"explore-v3-{result['exp_id']}"
    result["experiment_root"] = RUN_ROOT
    result["exploration_probe"] = deepcopy(PROBE)
    return result


def _quick_generic(payload: dict) -> None:
    payload["training"].update(
        {
            "epochs": 50,
            "log_every": 1,
            "log_psnr_every": 5,
            "psnr_sample_ratio": 0.01,
            "save_every": 50,
        }
    )


def _quick_mc_inr(payload: dict) -> None:
    payload["training"].update(
        {
            "epochs": 50,
            "meta_iterations": 5,
            "finetune_epochs": 50,
            "log_every": 1,
            "save_every": 50,
        }
    )


def _quick_neural_expert(payload: dict, *, size: str, manager_pretrain: bool) -> None:
    target = str(payload["DATA"]["target"])
    manager_path = (
        f"{RUN_ROOT}/neural_expert/pretrained_managers/ionization/{size.lower()}/"
        f"pt_inr_moe_ionization_{target}_managerpretraining.pth"
    )
    payload["MODEL"]["manager_pt_path"] = manager_path
    training = payload["TRAINING"]
    training["num_epochs"] = 2_500 if manager_pretrain else 75_000
    training["log_every"] = 100 if manager_pretrain else 7_500
    training["save_every"] = training["num_epochs"]


def _quick_apmgsrn(payload: dict) -> None:
    payload["TRAINING"].update({"iterations": 750, "log_every": 75, "save_every": 750})


def _quick_dc_inr(payload: dict) -> None:
    payload["training"].update({"total_steps": 75_000, "log_every": 7_500})


def _quick_fv_srn(payload: dict) -> None:
    payload["training"].update({"epochs": 50, "log_every": 1, "save_every": 50})


def _quick_rmdsrn(payload: dict) -> None:
    payload["training"].update({"steps": 75_000, "log_every": 7_500, "save_every": 75_000})


def quick_payload(formal_path: Path) -> dict:
    relative = formal_path.relative_to(FORMAL_CONFIG_ROOT)
    family, size, filename = relative.parts
    payload = _tag(yaml.safe_load(formal_path.read_text(encoding="utf-8")), family, size)
    if family in GENERIC_FAMILIES:
        _quick_generic(payload)
    elif family == "MC-INR":
        _quick_mc_inr(payload)
    elif family == "NeuralExpert":
        _quick_neural_expert(
            payload,
            size=size,
            manager_pretrain=filename.endswith("__managerpretrain.yaml"),
        )
    elif family == "APMGSRN":
        _quick_apmgsrn(payload)
    elif family == "DC-INR":
        _quick_dc_inr(payload)
    elif family == "fV-SRN":
        _quick_fv_srn(payload)
    elif family == "RMDSRN":
        _quick_rmdsrn(payload)
    else:
        raise ValueError(f"Unsupported exploration-v3 family: {family}")
    return payload


def generate() -> dict[str, int]:
    if CONFIG_ROOT.exists():
        shutil.rmtree(CONFIG_ROOT)
    CONFIG_ROOT.mkdir(parents=True)
    counts: dict[str, int] = {}
    exp_ids: set[str] = set()
    for formal_path in formal_config_paths():
        relative = formal_path.relative_to(FORMAL_CONFIG_ROOT)
        family = relative.parts[0]
        payload = quick_payload(formal_path)
        exp_id = str(payload["exp_id"])
        if exp_id in exp_ids:
            raise ValueError(f"Duplicate exploration-v3 exp_id: {exp_id}")
        exp_ids.add(exp_id)
        dump(CONFIG_ROOT / relative, payload)
        counts[family] = counts.get(family, 0) + 1
    total = sum(counts.values())
    if total != EXPECTED_TOTAL:
        raise RuntimeError(f"Expected {EXPECTED_TOTAL} exploration-v3 configs, generated {total}: {counts}")
    return counts


def main() -> None:
    counts = generate()
    print(f"Generated {sum(counts.values())} exploration-v3 Size configs: {counts}")


if __name__ == "__main__":
    main()
