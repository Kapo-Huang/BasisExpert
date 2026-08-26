#!/usr/bin/env python3
"""Build the paper-facing Result tree from completed runs.

The source ``runs`` tree is always read-only.  The command selects the newest
scientifically matching completed run for every requested experiment, removes
the timestamp layer, keeps only the final reconstruction checkpoint, and
emits an auditable manifest plus Chinese summary reports.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import os
import re
import shutil
import statistics
import sys
import zipfile
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import yaml


TIMESTAMP_RE = re.compile(r"^\d{8}_\d{6}(?:_\d+)?$")
TERMINAL_PAIR_RE = re.compile(r"(?i)\b(?:epoch|progress)\s*[=:]?\s*(\d+)\s*/\s*(\d+)")
PSNR_PATTERNS = (
    re.compile(r"(?i)PSNR[^\r\n]*?aggregate\s*[=:]\s*(nan|[-+]?inf(?:inity)?|[-+]?\d+(?:\.\d+)?(?:e[-+]?\d+)?)"),
    re.compile(r"(?i)aggregate_psnr\s*[=:]\s*(nan|[-+]?inf(?:inity)?|[-+]?\d+(?:\.\d+)?(?:e[-+]?\d+)?)"),
)
NATIVE_COMPLETION = (
    re.compile(r"(?i)Training loop finished:\s*steps=\d+"),
    re.compile(r"(?i)Training completed(?:\.|,)"),
    re.compile(r"(?i)Saved final state dict to\s+\S+"),
)
PARAM_PATTERNS = (
    re.compile(r"(?i)Model size:\s*params=([\d,]+)"),
    re.compile(r"(?i)Number of parameters in the current model:\s*([\d,]+)"),
    re.compile(r"(?i)total parameters:\s*([\d,]+)"),
)
PSNR_FAILURE_DB = 20.0
RD_SIZE_MIB = {"Size041": 0.41, "Size082": 0.82, "Size163": 1.63, "Size326": 3.26}
RD_LABEL = {key: f"{value:.2f}" for key, value in RD_SIZE_MIB.items()}
MAIN_PARAMETER_BUDGET = {
    "RedSea": 108_138,
    "Katrina": 112_501,
    "Ionization": 856_259,
    "Combustion": 1_137_357,
}
STATIC_MAIN_METHODS = {
    "Ours", "CoordNet", "SIREN", "Neural Experts", "MoE-INR",
    "fV-SRN", "InstantVNR", "STSR-INR", "MVNet",
}
RECOVERY_KEYS = {
    "optimizer", "optimizer_state", "optimizer_state_dict",
    "scheduler", "scheduler_state", "scheduler_state_dict",
    "lr_scheduler", "lr_scheduler_state", "grad_scaler", "amp_scaler",
    "scaler_state", "training_state", "rng_state",
}
TOP_LEVEL_RECOVERY_KEYS = {"epoch", "global_step", "optimizer_step"}


@dataclass(frozen=True)
class Spec:
    category: str
    method: str
    dataset: str
    item_parts: tuple[str, ...]
    config_path: Path
    exp_id: str
    group_variant: str = ""

    @property
    def destination_parts(self) -> tuple[str, ...]:
        return (self.category, self.method, self.dataset, *self.item_parts)

    @property
    def group_key(self) -> tuple[str, str, str, str]:
        return self.category, self.method, self.dataset, self.group_variant

    @property
    def label(self) -> str:
        return " / ".join(self.destination_parts)


@dataclass
class Candidate:
    exp_id: str
    run_dir: Path
    config_path: Path
    checkpoint_path: Path | None
    timestamp: str
    complete: bool
    completion_reason: str
    psnr: float | None
    parameter_count: int | None
    config_mismatches: list[str] = field(default_factory=list)


@dataclass
class Selection:
    spec: Spec
    selected: Candidate | None
    candidates: list[Candidate]
    missing_reason: str = ""
    checkpoint_audit: dict[str, Any] | None = None


def load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8", errors="replace"))
    return payload if isinstance(payload, dict) else {}


def exp_id_from_config(path: Path) -> str:
    value = load_yaml(path).get("exp_id")
    if not value:
        raise ValueError(f"Missing exp_id in {path}")
    return str(value)


def section(payload: dict[str, Any], name: str) -> dict[str, Any]:
    for key in (name, name.lower(), name.upper()):
        value = payload.get(key)
        if isinstance(value, dict):
            return value
    return {}


def target_label(config_path: Path) -> str:
    payload = load_yaml(config_path)
    data = section(payload, "data")
    target = data.get("target") or data.get("attr_name")
    if target:
        return str(target)
    targets = data.get("targets")
    if isinstance(targets, dict) and len(targets) == 1:
        return str(next(iter(targets)))
    return "Joint"


def files_for_dataset(root: Path, dataset_key: str) -> list[Path]:
    return sorted(
        path for path in root.glob(f"{dataset_key}*.yaml")
        if "managerpretrain" not in path.stem.lower() and not path.stem.lower().endswith("_dwa")
    )


def make_spec(
    repo: Path,
    category: str,
    method: str,
    dataset: str,
    item_parts: tuple[str, ...],
    config_path: Path,
    group_variant: str = "",
) -> Spec:
    resolved = repo / config_path
    return Spec(
        category=category,
        method=method,
        dataset=dataset,
        item_parts=item_parts,
        config_path=resolved,
        exp_id=exp_id_from_config(resolved),
        group_variant=group_variant,
    )


def build_specs(repo: Path) -> list[Spec]:
    specs: list[Spec] = []
    main_matrix = {
        "Ours": ("VarExpert", ("Ionization", "Combustion", "Katrina", "RedSea")),
        "CoordNet": ("CoordNet", ("Ionization", "Combustion", "Katrina", "RedSea")),
        "SIREN": ("SIREN", ("Ionization", "Combustion", "Katrina", "RedSea")),
        "Neural Experts": ("NeuralExpert", ("Ionization", "Combustion", "Katrina", "RedSea")),
        "MoE-INR": ("MoE-INR", ("Ionization", "Combustion", "Katrina", "RedSea")),
        "fV-SRN": ("fV-SRN", ("Ionization", "Combustion")),
        "APMGSRN": ("APMGSRN", ("Ionization", "Combustion")),
        "InstantVNR": ("InstantVNR", ("Ionization", "Combustion")),
        "MINER": ("MINER", ("Ionization", "Combustion")),
        "ECNR": ("ECNR", ("Ionization", "Combustion")),
        "STSR-INR": ("STSR-INR", ("Ionization", "Combustion", "Katrina", "RedSea")),
        "MVNet": ("MVNet", ("Ionization", "Combustion", "Katrina", "RedSea")),
    }
    dataset_key = {
        "Ionization": "ionization",
        "Combustion": "combustion_40NH3_1",
        "Katrina": "katrina",
        "RedSea": "redsea",
    }
    for display_method, (config_method, datasets) in main_matrix.items():
        for dataset in datasets:
            root = repo / "configs" / "main" / config_method
            for config_path in files_for_dataset(root, dataset_key[dataset]):
                specs.append(make_spec(
                    repo, "Main", display_method, dataset,
                    (target_label(config_path),), config_path.relative_to(repo),
                ))

    rd_methods = {
        "Ours": "VarExpert",
        "CoordNet": "CoordNet",
        "MoE-INR": "MoE-INR",
        "fV-SRN": "fV-SRN",
        "MINER": "MINER",
        "STSR-INR": "STSR-INR",
    }
    for display_method, config_method in rd_methods.items():
        for size_key, rate in RD_LABEL.items():
            root = repo / "configs" / "rd_curve" / config_method / size_key
            for config_path in sorted(root.glob("ionization*.yaml")):
                specs.append(make_spec(
                    repo, "RD Curve", display_method, "Ionization",
                    (rate, target_label(config_path)), config_path.relative_to(repo),
                    group_variant=rate,
                ))

    main_ion = repo / "configs" / "main" / "VarExpert" / "ionization.yaml"
    no_embedding = repo / "configs" / "ablation" / "variable_conditioning" / "VarExpertNoEmbedding" / "ionization.yaml"
    specs.append(make_spec(
        repo, "Ablation", "Ours", "Ionization", ("WithVariableEmbedding",),
        main_ion.relative_to(repo), group_variant="VariableEmbedding",
    ))
    specs.append(make_spec(
        repo, "Ablation", "Ours", "Ionization", ("WithoutVariableEmbedding",),
        no_embedding.relative_to(repo), group_variant="VariableEmbedding",
    ))

    sensitivity_root = repo / "configs" / "sensitivity"
    for config_path in sorted((sensitivity_root / "var_expert_num").rglob("ionization.yaml")):
        match = re.search(r"experts(\d+)", config_path.as_posix())
        if not match:
            continue
        value = f"E{int(match.group(1))}"
        specs.append(make_spec(
            repo, "Sensitivity", "Ours", "Ionization", ("ExpertNum", value),
            config_path.relative_to(repo), group_variant="ExpertNum",
        ))
    for config_path in sorted((sensitivity_root / "var_expert_topk").rglob("ionization.yaml")):
        match = re.search(r"top(\d+)", config_path.as_posix())
        if not match:
            continue
        value = f"K{int(match.group(1))}"
        specs.append(make_spec(
            repo, "Sensitivity", "Ours", "Ionization", ("TopK", value),
            config_path.relative_to(repo), group_variant="TopK",
        ))

    for count in (1, 2, 4, 8):
        config_path = repo / "configs" / "variable_scaling" / "VarExpert" / f"V{count:02d}" / "combustion_40NH3_1.yaml"
        specs.append(make_spec(
            repo, "Scaling", "Ours", "Combustion", (f"V{count}",),
            config_path.relative_to(repo), group_variant="VariableScaling",
        ))
    main_combustion = repo / "configs" / "main" / "VarExpert" / "combustion_40NH3_1.yaml"
    specs.append(make_spec(
        repo, "Scaling", "Ours", "Combustion", ("V13",),
        main_combustion.relative_to(repo), group_variant="VariableScaling",
    ))
    return specs


def find_checkpoint(run_dir: Path, exp_id: str) -> Path | None:
    preferred = (
        run_dir / "checkpoints" / f"{exp_id}.pth",
        run_dir / f"{exp_id}.pth",
    )
    for path in preferred:
        if path.is_file():
            return path
    trained = run_dir / "trained_models"
    if trained.is_dir():
        finals = sorted(trained.glob("*final*.pth"))
        if finals:
            return finals[-1]
    return None


def read_logs(run_dir: Path) -> str:
    log_files = sorted((run_dir / "logs").glob("*.log")) if (run_dir / "logs").is_dir() else []
    if not log_files:
        log_files = sorted(run_dir.glob("*.log"))
    chunks: list[str] = []
    for path in log_files:
        try:
            chunks.append(path.read_text(encoding="utf-8", errors="replace"))
        except OSError:
            continue
    return "\n".join(chunks)


def completion_status(run_dir: Path, checkpoint: Path | None, log_text: str) -> tuple[bool, str]:
    if checkpoint is None:
        return False, "missing-final-checkpoint"
    manifest = run_dir / "manifest.json"
    if manifest.is_file():
        try:
            if str(json.loads(manifest.read_text(encoding="utf-8")).get("status", "")).lower() == "complete":
                return True, "manifest-complete"
        except (OSError, json.JSONDecodeError):
            pass
    if (run_dir / "metrics" / "training_summary.json").is_file():
        return True, "training-summary-and-final-checkpoint"
    if checkpoint is not None and zipfile.is_zipfile(checkpoint):
        try:
            with zipfile.ZipFile(checkpoint, "r") as archive:
                if "metadata.json" in archive.namelist():
                    metadata = json.loads(archive.read("metadata.json").decode("utf-8"))
                    indices = metadata.get("time_indices", [])
                    members = [name for name in archive.namelist() if name.endswith(".pt")]
                    if indices and len(indices) == len(members):
                        return True, f"complete-temporal-bundle:{len(indices)}"
        except (OSError, ValueError, KeyError, json.JSONDecodeError, zipfile.BadZipFile):
            pass
    pairs = list(TERMINAL_PAIR_RE.finditer(log_text))
    if pairs:
        current, total = (int(value) for value in pairs[-1].groups())
        if total > 0 and current == total:
            return True, f"terminal-count:{current}/{total}"
        return False, f"incomplete-count:{current}/{total}"
    for pattern in NATIVE_COMPLETION:
        if pattern.search(log_text):
            return True, "native-completion-marker"
    return False, "missing-completion-evidence"


def extract_psnr(run_dir: Path, log_text: str) -> float | None:
    values: list[float] = []
    for pattern in PSNR_PATTERNS:
        for match in pattern.finditer(log_text):
            try:
                values.append(float(match.group(1)))
            except ValueError:
                continue
    if values:
        return values[-1]

    trajectory = run_dir / "metrics" / "exploration_psnr.tsv"
    if trajectory.is_file():
        try:
            with trajectory.open("r", encoding="utf-8", errors="replace", newline="") as handle:
                rows = list(csv.DictReader(handle, delimiter="\t"))
            if rows:
                for key in ("aggregate_psnr", "psnr", "final_psnr_db"):
                    if rows[-1].get(key):
                        return float(rows[-1][key])
        except (OSError, ValueError):
            pass

    manifest = run_dir / "manifest.json"
    if manifest.is_file():
        try:
            payload = json.loads(manifest.read_text(encoding="utf-8"))
            timestep_values = [
                float(item["psnr"])
                for item in payload.get("timesteps", {}).values()
                if isinstance(item, dict) and item.get("psnr") is not None
            ]
            if timestep_values:
                return statistics.fmean(timestep_values)
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            pass
    return None


def extract_parameter_count(run_dir: Path, log_text: str) -> int | None:
    for pattern in PARAM_PATTERNS:
        match = pattern.search(log_text)
        if match:
            return int(match.group(1).replace(",", ""))
    for relative in ("metrics/model_stats.json", "metrics/training_summary.json", "manifest.json"):
        path = run_dir / relative
        if not path.is_file():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        for key in ("parameter_count", "param_count", "total_parameters"):
            value = payload.get(key)
            if value is None and isinstance(payload.get("model_stats"), dict):
                value = payload["model_stats"].get(key)
            if value is not None:
                try:
                    return int(value)
                except (TypeError, ValueError):
                    pass
    return None


def scientific_view(payload: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name in ("data", "model", "training"):
        value = section(payload, name)
        if value:
            result[name] = value
    return result


def ignored_scientific_path(parts: tuple[str, ...], value: Any) -> bool:
    lower = tuple(part.lower() for part in parts)
    leaf = lower[-1] if lower else ""
    if any(token in leaf for token in ("path", "root", "file", "dir")):
        return True
    if lower and lower[0] == "training" and leaf in {
        "save_every", "save_intermediate_checkpoints", "log_every",
        "log_psnr_every", "psnr_sample_ratio", "num_workers", "device",
        "pred_batch_size",
    }:
        return True
    if isinstance(value, str) and ("${" in value or value.startswith(("/", "\\")) or re.match(r"^[A-Za-z]:[\\/]", value)):
        return True
    return False


def compare_scientific_config(expected: dict[str, Any], actual: dict[str, Any]) -> list[str]:
    mismatches: list[str] = []
    expected_view = scientific_view(expected)
    actual_view = scientific_view(actual)

    def walk(exp: Any, act: Any, parts: tuple[str, ...]) -> None:
        if ignored_scientific_path(parts, exp):
            return
        if isinstance(exp, dict):
            if not isinstance(act, dict):
                mismatches.append(f"{'.'.join(parts)}: missing section")
                return
            if parts and parts[-1].lower() == "targets":
                if set(map(str, exp)) != set(map(str, act)):
                    mismatches.append(
                        f"{'.'.join(parts)} keys: expected={sorted(map(str, exp))} actual={sorted(map(str, act))}"
                    )
                return
            for key, value in exp.items():
                actual_key = next((candidate for candidate in act if str(candidate).lower() == str(key).lower()), None)
                if actual_key is None:
                    mismatches.append(f"{'.'.join((*parts, str(key)))}: missing")
                else:
                    walk(value, act[actual_key], (*parts, str(key)))
            return
        if isinstance(exp, (list, tuple)):
            if list(exp) != list(act) if isinstance(act, (list, tuple)) else True:
                mismatches.append(f"{'.'.join(parts)}: expected={exp!r} actual={act!r}")
            return
        leaf = parts[-1].lower() if parts else ""
        if leaf == "time_indices" and isinstance(exp, str) and exp.lower() == "all":
            if isinstance(act, (list, tuple)) and act and list(act) == list(range(len(act))):
                return
        if leaf == "dataset_name" and isinstance(exp, str) and isinstance(act, str):
            if exp.lower() == act.lower():
                return
        if isinstance(exp, (int, float)) and isinstance(act, (int, float)):
            if not math.isclose(float(exp), float(act), rel_tol=1e-9, abs_tol=1e-12):
                mismatches.append(f"{'.'.join(parts)}: expected={exp!r} actual={act!r}")
            return
        if exp != act:
            mismatches.append(f"{'.'.join(parts)}: expected={exp!r} actual={act!r}")

    walk(expected_view, actual_view, ())
    return mismatches


def candidate_run_dir(config_path: Path) -> Path:
    if config_path.parent.name.lower() == "configs":
        return config_path.parent.parent
    return config_path.parent


def index_candidates(runs: Path, wanted_ids: set[str]) -> dict[str, list[Candidate]]:
    index: dict[str, list[Candidate]] = defaultdict(list)
    seen: set[tuple[str, Path]] = set()
    for config_path in runs.rglob("config.yaml"):
        try:
            payload = load_yaml(config_path)
        except (OSError, yaml.YAMLError):
            continue
        exp_id = str(payload.get("exp_id", ""))
        if exp_id not in wanted_ids:
            continue
        run_dir = candidate_run_dir(config_path)
        key = exp_id, run_dir.resolve()
        if key in seen:
            continue
        seen.add(key)
        checkpoint = find_checkpoint(run_dir, exp_id)
        log_text = read_logs(run_dir)
        complete, reason = completion_status(run_dir, checkpoint, log_text)
        timestamp = run_dir.name if TIMESTAMP_RE.match(run_dir.name) else ""
        index[exp_id].append(Candidate(
            exp_id=exp_id,
            run_dir=run_dir,
            config_path=config_path,
            checkpoint_path=checkpoint,
            timestamp=timestamp,
            complete=complete,
            completion_reason=reason,
            psnr=extract_psnr(run_dir, log_text),
            parameter_count=extract_parameter_count(run_dir, log_text),
        ))
    return index


def candidate_sort_key(candidate: Candidate) -> tuple[str, float]:
    timestamp = candidate.timestamp
    try:
        modified = candidate.run_dir.stat().st_mtime
    except OSError:
        modified = 0.0
    return timestamp, modified


def select_runs(specs: list[Spec], index: dict[str, list[Candidate]]) -> list[Selection]:
    expected_cache: dict[Path, dict[str, Any]] = {}
    selections: list[Selection] = []
    for spec in specs:
        expected = expected_cache.setdefault(spec.config_path, load_yaml(spec.config_path))
        candidates = index.get(spec.exp_id, [])
        for candidate in candidates:
            candidate.config_mismatches = compare_scientific_config(expected, load_yaml(candidate.config_path))
        eligible = [
            candidate for candidate in candidates
            if candidate.complete and candidate.checkpoint_path is not None
        ]
        selected = max(eligible, key=candidate_sort_key) if eligible else None
        if selected is not None:
            reason = ""
        elif not candidates:
            reason = "runs 中没有该 exp_id"
        elif not any(candidate.checkpoint_path for candidate in candidates):
            reason = "存在目录，但缺少最终 checkpoint"
        elif not any(candidate.complete for candidate in candidates):
            reasons = sorted({candidate.completion_reason for candidate in candidates})
            reason = "没有完成运行（" + ", ".join(reasons) + "）"
        else:
            reason = "没有同时满足完成状态与 checkpoint 条件的运行"
        selections.append(Selection(spec, selected, candidates, reason))
    return selections


def recovery_paths(payload: Any, path: tuple[str, ...] = ()) -> list[str]:
    hits: list[str] = []
    if isinstance(payload, dict):
        for key, value in payload.items():
            key_text = str(key)
            key_lower = key_text.lower()
            current = (*path, key_text)
            if key_lower in RECOVERY_KEYS or (not path and key_lower in TOP_LEVEL_RECOVERY_KEYS):
                hits.append("/" + "/".join(current))
            if isinstance(value, (dict, list, tuple)):
                hits.extend(recovery_paths(value, current))
    elif isinstance(payload, (list, tuple)):
        for index, value in enumerate(payload):
            if isinstance(value, (dict, list, tuple)):
                hits.extend(recovery_paths(value, (*path, str(index))))
    return hits


def tensor_parameter_count(payload: Any) -> int:
    if isinstance(payload, torch.Tensor):
        return int(payload.numel())
    if isinstance(payload, np.ndarray):
        return int(payload.size)
    if isinstance(payload, dict):
        return sum(tensor_parameter_count(value) for value in payload.values())
    if isinstance(payload, (list, tuple)):
        return sum(tensor_parameter_count(value) for value in payload)
    return 0


def load_torch(path_or_buffer: Any) -> Any:
    try:
        return torch.load(path_or_buffer, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path_or_buffer, map_location="cpu")


def temporal_metadata(path: Path) -> dict[str, Any] | None:
    if not zipfile.is_zipfile(path):
        return None
    with zipfile.ZipFile(path, "r") as archive:
        if "metadata.json" not in archive.namelist():
            return None
        return json.loads(archive.read("metadata.json").decode("utf-8"))


def audit_and_write_checkpoint(source: Path, destination: Path) -> dict[str, Any]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporal = temporal_metadata(source)
    if temporal is not None:
        source_hits = recovery_paths(temporal)
        member_count = 0
        parameter_count = 0
        with zipfile.ZipFile(source, "r") as archive:
            members = sorted(name for name in archive.namelist() if name.endswith(".pt"))
            for member in members:
                payload = load_torch(io.BytesIO(archive.read(member)))
                source_hits.extend(f"{member}:{hit}" for hit in recovery_paths(payload))
                parameter_count += tensor_parameter_count(payload)
                member_count += 1
        if source_hits:
            raise ValueError(f"Temporal checkpoint contains training recovery state: {source_hits[:10]}")
        shutil.copy2(source, destination)
        return {
            "source_format": str(temporal.get("format", "temporal-unknown")),
            "result_format": str(temporal.get("format", "temporal-unknown")),
            "source_recovery_keys": [],
            "purified": False,
            "member_count": member_count,
            "parameter_count": parameter_count or None,
        }

    payload = load_torch(source)
    if not isinstance(payload, dict):
        raise ValueError(f"Unsupported non-dict checkpoint payload: {source}")
    source_hits = recovery_paths(payload)
    source_format = str(payload.get("format", "legacy"))
    purified = False
    result_payload = payload
    if "model_state" in payload and (source_hits or payload.get("format") != "inference_checkpoint_v1"):
        result_payload = {
            "format": "inference_checkpoint_v1",
            "model_state": payload["model_state"],
            "target_names_order": list(payload.get("target_names_order", [])),
            "config_hash": str(payload.get("config_hash", "")),
        }
        if payload.get("target_dims_order") is not None:
            result_payload["target_dims_order"] = list(payload["target_dims_order"])
        purified = True
        torch.save(result_payload, destination)
    else:
        if source_hits:
            raise ValueError(f"Checkpoint contains unsupported recovery keys: {source_hits[:10]}")
        shutil.copy2(source, destination)

    verified = load_torch(destination)
    result_hits = recovery_paths(verified)
    if result_hits:
        raise ValueError(f"Result checkpoint still contains recovery state: {result_hits[:10]}")
    result_format = str(verified.get("format", "legacy")) if isinstance(verified, dict) else "invalid"
    if result_format == "inference_checkpoint_v1" and "model_state" not in verified:
        raise ValueError(f"Inference checkpoint is missing model_state: {destination}")
    if result_format == "fv_srn_inference_v1" and not {"mlp_state", "quantized_grids"}.issubset(verified):
        raise ValueError(f"Invalid fV-SRN reconstruction checkpoint: {destination}")
    if result_format == "ecnr_inference_v1" and "payload" not in verified:
        raise ValueError(f"Invalid ECNR reconstruction checkpoint: {destination}")
    parameter_payload = verified.get("model_state", verified)
    return {
        "source_format": source_format,
        "result_format": result_format,
        "source_recovery_keys": source_hits,
        "purified": purified,
        "member_count": 0,
        "parameter_count": tensor_parameter_count(parameter_payload) or None,
    }


def copy_run_without_checkpoints(source: Path, destination: Path) -> None:
    for root, dirnames, filenames in os.walk(source):
        root_path = Path(root)
        relative = root_path.relative_to(source)
        dirnames[:] = [
            name for name in dirnames
            if name not in {"checkpoints", "trained_models", "__pycache__"}
        ]
        target_root = destination / relative
        target_root.mkdir(parents=True, exist_ok=True)
        for filename in filenames:
            if filename.endswith(".tmp"):
                continue
            source_file = root_path / filename
            target_file = target_root / filename
            shutil.copy2(source_file, target_file)


def markdown_table(headers: list[str], rows: Iterable[Iterable[Any]]) -> str:
    output = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for row in rows:
        cells = [str(value).replace("|", "\\|").replace("\n", "<br>") for value in row]
        output.append("| " + " | ".join(cells) + " |")
    return "\n".join(output)


def rel(path: Path, repo: Path) -> str:
    try:
        return path.resolve().relative_to(repo.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def group_rows(selections: list[Selection]) -> list[tuple[Any, ...]]:
    grouped: dict[tuple[str, str, str, str], list[Selection]] = defaultdict(list)
    for selection in selections:
        grouped[selection.spec.group_key].append(selection)
    rows = []
    for key in sorted(grouped):
        items = grouped[key]
        rows.append((*key, len(items), sum(item.selected is not None for item in items), sum(item.selected is None for item in items)))
    return rows


def known_configuration_anomalies() -> list[tuple[str, str, str]]:
    return [
        ("RD Curve / fV-SRN / Ionization / 1.63", "模型大小异常", "正式配置的训练态五变量总量约 60.035 MiB，远高于 1.63 MiB 标称档（约 +3583%）。"),
        ("RD Curve / MINER / Ionization / 1.63", "结构性大小异常", "静态下界约 254.841 MiB，1.63 MiB 档无法成立；最终大小还依赖 active blocks。"),
        ("RD Curve / STSR-INR / Ionization / 1.63", "参数/大小异常", "正式配置约 5.995 MiB（约 +267.8%），且大于 3.26 MiB 档，破坏 RD 档位单调性。"),
    ]


def write_reports(repo: Path, result_root: Path, selections: list[Selection]) -> None:
    copied = [item for item in selections if item.selected is not None]
    missing = [item for item in selections if item.selected is None]
    groups = group_rows(selections)

    summary_lines = [
        "# Result 实验汇总",
        "",
        "本目录由 `scripts/tools/build_result.py` 从 `runs` 只读生成；已移除 Timestamp 层。",
        "同一 `exp_id` 要求训练完成且存在最终 checkpoint，再选择最新运行；参数或指标异常不阻止归档。",
        "",
        f"- 目标实验项：{len(selections)}",
        f"- 已复制：{len(copied)}",
        f"- 缺失或不可用：{len(missing)}",
        f"- PSNR 异常阈值：低于 {PSNR_FAILURE_DB:g} dB；NaN/Inf 也视为异常",
        "",
    ]
    section_order = ["Main", "RD Curve", "Ablation", "Sensitivity", "Scaling"]
    for category in section_order:
        category_items = [item for item in selections if item.spec.category == category]
        if not category_items:
            continue
        summary_lines.extend([
            f"## {category}",
            "",
            markdown_table(
                ["方法", "数据集", "实验项", "状态", "PSNR(dB)"],
                [
                    (
                        item.spec.method,
                        item.spec.dataset,
                        "/".join(item.spec.item_parts),
                        "已复制" if item.selected is not None else "缺失",
                        "-" if item.selected is None or item.selected.psnr is None else f"{item.selected.psnr:.4f}",
                    )
                    for item in category_items
                ],
            ),
            "",
        ])
    (result_root / "EXPERIMENT_SUMMARY.md").write_text("\n".join(summary_lines), encoding="utf-8")

    anomaly_rows: list[tuple[str, str, str]] = known_configuration_anomalies()
    checkpoint_rows: list[tuple[str, str, str]] = []
    psnr_unknown: list[tuple[str, str]] = []
    for item in selections:
        selected = item.selected
        if selected is not None:
            audit = item.checkpoint_audit or {}
            if selected.config_mismatches:
                anomaly_rows.append((
                    item.spec.label,
                    "参数异常（已允许归档）",
                    "; ".join(selected.config_mismatches[:8]),
                ))
            if audit.get("source_recovery_keys"):
                checkpoint_rows.append((
                    item.spec.label,
                    rel(selected.checkpoint_path, repo) if selected.checkpoint_path else "-",
                    ", ".join(audit["source_recovery_keys"][:8]),
                ))
            if selected.psnr is None:
                psnr_unknown.append((item.spec.label, "完成运行中没有可读取的最终 PSNR"))
            elif not math.isfinite(selected.psnr) or selected.psnr < PSNR_FAILURE_DB:
                anomaly_rows.append((
                    item.spec.label,
                    "PSNR 异常",
                    f"最终/汇总 PSNR={selected.psnr!r} dB",
                ))

    # Group-level main model-size validation when the complete group has usable counts.
    grouped_selected: dict[tuple[str, str, str, str], list[Selection]] = defaultdict(list)
    for item in copied:
        grouped_selected[item.spec.group_key].append(item)
    expected_group_sizes = {row[:4]: row[4] for row in groups}
    for key, items in grouped_selected.items():
        category, method, dataset, variant = key
        if category != "Main" or method not in STATIC_MAIN_METHODS:
            continue
        if len(items) != expected_group_sizes[key]:
            continue
        counts = [
            item.selected.parameter_count or (item.checkpoint_audit or {}).get("parameter_count")
            for item in items
        ]
        if any(value is None for value in counts):
            continue
        total = sum(int(value) for value in counts if value is not None)
        budget = MAIN_PARAMETER_BUDGET[dataset]
        deviation = (total - budget) / budget
        if abs(deviation) > 0.03:
            anomaly_rows.append((
                f"Main / {method} / {dataset}",
                "模型大小异常",
                f"汇总参数量 {total:,}，目标 {budget:,}，偏差 {deviation:+.2%}（容差 ±3%）。",
            ))

    anomaly_lines = [
        "# 异常实验与 Checkpoint 审计",
        "",
        "判定依据：科学参数与当前正式 YAML 对比；主实验静态模型按对应 Ours 参数预算 ±3%；",
        f"PSNR 低于 {PSNR_FAILURE_DB:g} dB 或非有限值视为异常。动态方法以训练后 artifact/manifest 为准。",
        "",
        "## 异常项",
        "",
        markdown_table(["实验项", "异常类型", "说明"], anomaly_rows) if anomaly_rows else "未发现异常。",
        "",
        "## 源 Checkpoint 中发现并已净化的训练恢复状态",
        "",
        markdown_table(["实验项", "源 Checkpoint", "已移除字段"], checkpoint_rows) if checkpoint_rows else "所有已选源 checkpoint 均已是重建专用格式。",
        "",
        "净化只发生在 `Result` 副本中：仅保留重建所需的模型状态、目标顺序与配置哈希；`runs` 未修改。",
        "",
        "## PSNR 无法判定",
        "",
        markdown_table(["实验项", "原因"], psnr_unknown) if psnr_unknown else "所有已复制实验均有可读取的 PSNR。",
        "",
    ]
    (result_root / "ANOMALIES.md").write_text("\n".join(anomaly_lines), encoding="utf-8")

    missing_lines = [
        "# 缺失实验组",
        "",
        "“缺失”包含：完全没有运行、运行未完成或缺少最终 checkpoint。参数与当前正式配置不一致不再阻止归档。",
        "",
        "## 分组缺失统计",
        "",
        markdown_table(
            ["分类", "方法", "数据集", "子组", "应有", "已复制", "缺失"],
            [(a, b, c, d or "-", e, f, g) for a, b, c, d, e, f, g in groups if g],
        ) if missing else "没有缺失实验。",
        "",
        "## 缺失明细",
        "",
        markdown_table(
            ["实验项", "exp_id", "原因"],
            [(item.spec.label, item.spec.exp_id, item.missing_reason) for item in missing],
        ) if missing else "没有缺失实验。",
        "",
    ]
    (result_root / "MISSING_EXPERIMENTS.md").write_text("\n".join(missing_lines), encoding="utf-8")

    readme = """# Result

目录层级为 `分类/方法/数据集/实验项`，实验目录直接包含原 Timestamp 目录中的内容。

- `EXPERIMENT_SUMMARY.md`：按 Main、RD Curve、Ablation、Sensitivity、Scaling 分区的结果与 PSNR 总览。
- `ANOMALIES.md`：参数、模型大小、PSNR 与 checkpoint 审计。
- `MISSING_EXPERIMENTS.md`：缺失或不可用实验。
- `MANIFEST.tsv`：逐实验项的机器可读来源与验证记录。

`checkpoints/` 中只保留最终重建 checkpoint；旧格式中的 optimizer、scheduler、epoch 等训练恢复状态已在副本中移除，源 `runs` 不变。
"""
    (result_root / "README.md").write_text(readme, encoding="utf-8")

    fields = [
        "category", "method", "dataset", "item", "exp_id", "status",
        "source_run", "source_timestamp", "source_checkpoint", "result_path",
        "completion_reason", "psnr_db", "parameter_count", "source_checkpoint_bytes",
        "result_checkpoint_bytes", "source_format", "result_format", "purified",
        "missing_reason",
    ]
    with (result_root / "MANIFEST.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for item in selections:
            selected = item.selected
            audit = item.checkpoint_audit or {}
            result_path = result_root.joinpath(*item.spec.destination_parts)
            result_checkpoint = result_path / "checkpoints" / f"{item.spec.exp_id}.pth"
            writer.writerow({
                "category": item.spec.category,
                "method": item.spec.method,
                "dataset": item.spec.dataset,
                "item": "/".join(item.spec.item_parts),
                "exp_id": item.spec.exp_id,
                "status": "copied" if selected else "missing",
                "source_run": rel(selected.run_dir, repo) if selected else "",
                "source_timestamp": selected.timestamp if selected else "",
                "source_checkpoint": rel(selected.checkpoint_path, repo) if selected and selected.checkpoint_path else "",
                "result_path": rel(result_path, repo) if selected else "",
                "completion_reason": selected.completion_reason if selected else "",
                "psnr_db": "" if not selected or selected.psnr is None else f"{selected.psnr:.10g}",
                "parameter_count": "" if not selected else selected.parameter_count or audit.get("parameter_count") or "",
                "source_checkpoint_bytes": "" if not selected or not selected.checkpoint_path else selected.checkpoint_path.stat().st_size,
                "result_checkpoint_bytes": result_checkpoint.stat().st_size if result_checkpoint.is_file() else "",
                "source_format": audit.get("source_format", ""),
                "result_format": audit.get("result_format", ""),
                "purified": str(bool(audit.get("purified"))).lower() if selected else "",
                "missing_reason": item.missing_reason,
            })


def validate_result(result_root: Path, selections: list[Selection]) -> None:
    for item in selections:
        destination = result_root.joinpath(*item.spec.destination_parts)
        if item.selected is None:
            if destination.exists() and any(destination.rglob("*")):
                raise RuntimeError(f"Missing experiment unexpectedly has files: {destination}")
            continue
        checkpoint = destination / "checkpoints" / f"{item.spec.exp_id}.pth"
        if not checkpoint.is_file():
            raise RuntimeError(f"Missing Result checkpoint: {checkpoint}")
        if any(TIMESTAMP_RE.match(part) for part in destination.relative_to(result_root).parts):
            raise RuntimeError(f"Timestamp leaked into Result path: {destination}")
        temporal = temporal_metadata(checkpoint)
        if temporal is None:
            payload = load_torch(checkpoint)
            hits = recovery_paths(payload)
            if hits:
                raise RuntimeError(f"Recovery state leaked into {checkpoint}: {hits[:10]}")
    for required in ("README.md", "EXPERIMENT_SUMMARY.md", "ANOMALIES.md", "MISSING_EXPERIMENTS.md", "MANIFEST.tsv"):
        if not (result_root / required).is_file():
            raise RuntimeError(f"Missing report: {required}")


def build(repo: Path, output: Path, dry_run: bool) -> int:
    specs = build_specs(repo)
    destinations = [spec.destination_parts for spec in specs]
    duplicates = sorted({parts for parts in destinations if destinations.count(parts) > 1})
    if duplicates:
        raise ValueError(f"Duplicate Result destinations: {duplicates[:10]}")
    index = index_candidates(repo / "runs", {spec.exp_id for spec in specs})
    selections = select_runs(specs, index)
    copied = sum(item.selected is not None for item in selections)
    print(f"target_items={len(selections)} selectable={copied} missing={len(selections) - copied}")
    for row in group_rows(selections):
        print("GROUP", " | ".join(map(str, row)))
    if dry_run:
        return 0

    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output}")
    staging = output.with_name(output.name + ".__staging__")
    if staging.exists():
        raise FileExistsError(f"Refusing to overwrite existing staging directory: {staging}")
    staging.mkdir(parents=True)
    try:
        completed_audits: dict[Path, dict[str, Any]] = {}
        for ordinal, item in enumerate(selections, start=1):
            selected = item.selected
            if selected is None:
                continue
            destination = staging.joinpath(*item.spec.destination_parts)
            print(f"COPY {ordinal}/{len(selections)} {item.spec.label}", flush=True)
            copy_run_without_checkpoints(selected.run_dir, destination)
            result_checkpoint = destination / "checkpoints" / f"{item.spec.exp_id}.pth"
            source_checkpoint = selected.checkpoint_path
            if source_checkpoint is None:
                raise RuntimeError(f"Selected run lost checkpoint: {selected.run_dir}")
            # Reused Main runs (Ablation with embedding and Scaling V13) are
            # independently copied but share an immutable source audit.
            audit = audit_and_write_checkpoint(source_checkpoint, result_checkpoint)
            item.checkpoint_audit = audit
            completed_audits[source_checkpoint.resolve()] = audit
        write_reports(repo, staging, selections)
        validate_result(staging, selections)
        staging.rename(output)
    except Exception:
        print(f"Build failed; staging preserved for inspection: {staging}", file=sys.stderr)
        raise
    print(f"Result created at {output}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo = args.repo.resolve()
    output = (args.output or repo / "Result").resolve()
    return build(repo, output, args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
