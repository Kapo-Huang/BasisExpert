from __future__ import annotations

from typing import Any, Callable

from ..config.schema import ModelConfig
from ..data.base import DatasetMeta
from .basis_expert.light_basis_expert import build_light_basis_expert_from_config
from .basis_expert.shared_enc_inr import build_shared_enc_inr_from_config
from .common import ModelAdapter, require_single_target, view_specs_from_meta
from .sota.coordnet import build_coordnet_from_config
from .sota.moe_inr import build_moe_inr_from_config
from .sota.siren import build_siren_from_config


ModelBuilder = Callable[[dict[str, Any], DatasetMeta], object]
ModelConfigMaterializer = Callable[[dict[str, Any], DatasetMeta], dict[str, Any]]


def _reject_unknown_model_keys(cfg: dict[str, Any], allowed: set[str], model_name: str) -> None:
    unknown = sorted(set(cfg).difference(allowed))
    if unknown:
        raise ValueError(f"Unknown {model_name} config keys: {', '.join(unknown)}")


def _resolve_in_features(cfg: dict[str, Any], meta: DatasetMeta, model_name: str) -> int:
    resolved = int(cfg.get("in_features", meta.input_dim))
    if resolved != int(meta.input_dim):
        raise ValueError(
            f"{model_name} in_features={resolved} does not match dataset input_dim={meta.input_dim}"
        )
    return resolved


def _resolve_single_target_out_features(cfg: dict[str, Any], meta: DatasetMeta, model_name: str) -> int:
    expected = int(require_single_target(meta, model_name))
    resolved = int(cfg.get("out_features", expected))
    if resolved != expected:
        raise ValueError(
            f"{model_name} out_features={resolved} does not match dataset target_dim={expected}"
        )
    return resolved


def _coerce_bool(value: Any, *, key: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in {0, 1}:
        return bool(value)
    raise TypeError(f"{key} must be a boolean, got {type(value).__name__}")


def _build_siren(cfg: dict, meta: DatasetMeta):
    return build_siren_from_config(cfg)


def _build_coordnet(cfg: dict, meta: DatasetMeta):
    return build_coordnet_from_config(cfg)


def _build_moe_inr(cfg: dict, meta: DatasetMeta):
    return build_moe_inr_from_config(cfg)


def _build_light_basis_expert(cfg: dict, meta: DatasetMeta):
    return build_light_basis_expert_from_config(cfg, view_specs_from_meta(meta))


def _build_shared_enc_inr(cfg: dict, meta: DatasetMeta):
    return build_shared_enc_inr_from_config(cfg, view_specs_from_meta(meta))


def _materialize_siren(cfg: dict[str, Any], meta: DatasetMeta) -> dict[str, Any]:
    _reject_unknown_model_keys(
        cfg,
        {
            "in_features",
            "out_features",
            "hidden_features",
            "hidden_layers",
            "first_omega_0",
            "hidden_omega_0",
            "outermost_linear",
        },
        "siren",
    )
    return {
        "in_features": _resolve_in_features(cfg, meta, "siren"),
        "out_features": _resolve_single_target_out_features(cfg, meta, "siren"),
        "hidden_features": int(cfg.get("hidden_features", 256)),
        "hidden_layers": int(cfg.get("hidden_layers", 3)),
        "first_omega_0": float(cfg.get("first_omega_0", 30.0)),
        "hidden_omega_0": float(cfg.get("hidden_omega_0", 30.0)),
        "outermost_linear": _coerce_bool(cfg.get("outermost_linear", True), key="siren.outermost_linear"),
    }


def _materialize_coordnet(cfg: dict[str, Any], meta: DatasetMeta) -> dict[str, Any]:
    _reject_unknown_model_keys(
        cfg,
        {"in_features", "out_features", "init_features", "num_res"},
        "coordnet",
    )
    return {
        "in_features": _resolve_in_features(cfg, meta, "coordnet"),
        "out_features": _resolve_single_target_out_features(cfg, meta, "coordnet"),
        "init_features": int(cfg.get("init_features", 64)),
        "num_res": int(cfg.get("num_res", 10)),
    }


def _materialize_moe_inr(cfg: dict[str, Any], meta: DatasetMeta) -> dict[str, Any]:
    _reject_unknown_model_keys(
        cfg,
        {
            "in_features",
            "out_features",
            "num_experts",
            "base_dim",
            "encoder_feature_dim",
            "encoder_first_omega_0",
            "encoder_hidden_omega_0",
            "policy_hidden_dim",
            "policy_num_layers",
            "policy_first_omega_0",
            "policy_hidden_omega_0",
        },
        "moe_inr",
    )
    base_dim = cfg.get("base_dim")
    if base_dim is not None:
        base_dim = int(base_dim)
        encoder_feature_dim = 8 * base_dim
        policy_hidden_dim = base_dim
    else:
        encoder_feature_dim = int(cfg.get("encoder_feature_dim", 256))
        policy_hidden_dim = int(cfg.get("policy_hidden_dim", 128))
    return {
        "in_features": _resolve_in_features(cfg, meta, "moe_inr"),
        "out_features": _resolve_single_target_out_features(cfg, meta, "moe_inr"),
        "num_experts": int(cfg.get("num_experts", 7)),
        "encoder_feature_dim": encoder_feature_dim,
        "policy_hidden_dim": policy_hidden_dim,
        "encoder_first_omega_0": float(cfg.get("encoder_first_omega_0", 30.0)),
        "encoder_hidden_omega_0": float(cfg.get("encoder_hidden_omega_0", 30.0)),
        "policy_num_layers": int(cfg.get("policy_num_layers", 3)),
        "policy_first_omega_0": float(cfg.get("policy_first_omega_0", 30.0)),
        "policy_hidden_omega_0": float(cfg.get("policy_hidden_omega_0", 30.0)),
    }


def _materialize_light_basis_expert(cfg: dict[str, Any], meta: DatasetMeta) -> dict[str, Any]:
    _reject_unknown_model_keys(
        cfg,
        {
            "in_features",
            "num_experts",
            "base_dim",
            "expert_feature_dim",
            "top_k",
            "view_embed_dim",
            "expert_num_frequencies",
            "expert_hidden_dim",
            "expert_num_layers",
            "gate_hidden_dim",
            "gate_num_layers",
            "decoder_feature_dim",
            "decoder_hidden_dim",
            "decoder_num_layers",
            "head_hidden_dim",
            "head_num_layers",
            "expert_first_omega_0",
            "expert_hidden_omega_0",
            "gate_first_omega_0",
            "gate_hidden_omega_0",
            "decoder_first_omega_0",
            "decoder_hidden_omega_0",
            "head_first_omega_0",
            "head_hidden_omega_0",
        },
        "light_basis_expert",
    )
    base_dim = cfg.get("base_dim")
    head_hidden_raw = cfg.get("head_hidden_dim")
    decoder_feature_raw = cfg.get("decoder_feature_dim")
    if base_dim is not None:
        base_dim = int(base_dim)
        expert_feature_dim = 8 * base_dim
        view_embed_dim = base_dim
        expert_hidden_dim = 8 * base_dim
        gate_hidden_dim = 8 * base_dim
        decoder_hidden_dim = 8 * base_dim
    else:
        expert_feature_dim = int(cfg.get("expert_feature_dim", 128))
        view_embed_dim = int(cfg.get("view_embed_dim", 16))
        expert_hidden_dim = int(cfg.get("expert_hidden_dim", 128))
        gate_hidden_dim = int(cfg.get("gate_hidden_dim", 128))
        decoder_hidden_dim = int(cfg.get("decoder_hidden_dim", 128))
    decoder_feature_dim = int(decoder_feature_raw) if decoder_feature_raw is not None else expert_feature_dim
    head_hidden_dim = int(head_hidden_raw) if head_hidden_raw is not None else decoder_feature_dim
    return {
        "in_features": _resolve_in_features(cfg, meta, "light_basis_expert"),
        "num_experts": int(cfg.get("num_experts", 7)),
        "expert_feature_dim": expert_feature_dim,
        "top_k": int(cfg.get("top_k", 3)),
        "view_embed_dim": view_embed_dim,
        "expert_num_frequencies": int(cfg.get("expert_num_frequencies", 6)),
        "expert_hidden_dim": expert_hidden_dim,
        "expert_num_layers": int(cfg.get("expert_num_layers", 3)),
        "gate_hidden_dim": gate_hidden_dim,
        "gate_num_layers": int(cfg.get("gate_num_layers", 3)),
        "decoder_feature_dim": decoder_feature_dim,
        "decoder_hidden_dim": decoder_hidden_dim,
        "decoder_num_layers": int(cfg.get("decoder_num_layers", 3)),
        "head_hidden_dim": head_hidden_dim,
        "head_num_layers": int(cfg.get("head_num_layers", 2)),
        "expert_first_omega_0": float(cfg.get("expert_first_omega_0", 30.0)),
        "expert_hidden_omega_0": float(cfg.get("expert_hidden_omega_0", 30.0)),
        "gate_first_omega_0": float(cfg.get("gate_first_omega_0", 30.0)),
        "gate_hidden_omega_0": float(cfg.get("gate_hidden_omega_0", 30.0)),
        "decoder_first_omega_0": float(cfg.get("decoder_first_omega_0", 30.0)),
        "decoder_hidden_omega_0": float(cfg.get("decoder_hidden_omega_0", 30.0)),
        "head_first_omega_0": float(cfg.get("head_first_omega_0", 30.0)),
        "head_hidden_omega_0": float(cfg.get("head_hidden_omega_0", 30.0)),
    }


def _materialize_shared_enc_inr(cfg: dict[str, Any], meta: DatasetMeta) -> dict[str, Any]:
    _reject_unknown_model_keys(
        cfg,
        {
            "in_features",
            "base_dim",
            "enc_base_dim",
            "dec_base_dim",
            "decoder_feature_dim",
            "head_hidden_dim",
            "expert_num_frequencies",
            "enc_num_frequencies",
            "enc_layer_num",
            "expert_num_layers",
            "decoder_num_layers",
            "head_num_layers",
            "expert_first_omega_0",
            "enc_first_omega_0",
            "expert_hidden_omega_0",
            "enc_hidden_omega_0",
            "decoder_first_omega_0",
            "decoder_hidden_omega_0",
            "head_first_omega_0",
            "head_hidden_omega_0",
        },
        "shared_enc_inr",
    )
    base_dim = cfg.get("base_dim")
    enc_base_dim_raw = cfg.get("enc_base_dim", base_dim)
    dec_base_dim_raw = cfg.get("dec_base_dim", base_dim)
    if enc_base_dim_raw is None or dec_base_dim_raw is None:
        raise ValueError("shared_enc_inr requires base_dim or enc_base_dim/dec_base_dim")
    enc_base_dim = int(enc_base_dim_raw)
    dec_base_dim = int(dec_base_dim_raw)
    decoder_feature_raw = cfg.get("decoder_feature_dim")
    head_hidden_raw = cfg.get("head_hidden_dim")
    decoder_feature_dim = int(decoder_feature_raw) if decoder_feature_raw is not None else 8 * dec_base_dim
    enc_feature_dim = decoder_feature_dim
    view_embed_dim = enc_base_dim
    enc_hidden_dim = 8 * enc_base_dim
    decoder_hidden_dim = 8 * dec_base_dim
    head_hidden_dim = int(head_hidden_raw) if head_hidden_raw is not None else decoder_feature_dim
    return {
        "in_features": _resolve_in_features(cfg, meta, "shared_enc_inr"),
        "enc_feature_dim": enc_feature_dim,
        "view_embed_dim": view_embed_dim,
        "enc_num_frequencies": int(cfg.get("enc_num_frequencies", cfg.get("expert_num_frequencies", 6))),
        "enc_hidden_dim": enc_hidden_dim,
        "enc_layer_num": int(cfg.get("enc_layer_num", cfg.get("expert_num_layers", 3))),
        "decoder_feature_dim": decoder_feature_dim,
        "decoder_hidden_dim": decoder_hidden_dim,
        "decoder_num_layers": int(cfg.get("decoder_num_layers", 3)),
        "head_hidden_dim": head_hidden_dim,
        "head_num_layers": int(cfg.get("head_num_layers", 2)),
        "enc_first_omega_0": float(cfg.get("enc_first_omega_0", cfg.get("expert_first_omega_0", 30.0))),
        "enc_hidden_omega_0": float(cfg.get("enc_hidden_omega_0", cfg.get("expert_hidden_omega_0", 30.0))),
        "decoder_first_omega_0": float(cfg.get("decoder_first_omega_0", 30.0)),
        "decoder_hidden_omega_0": float(cfg.get("decoder_hidden_omega_0", 30.0)),
        "head_first_omega_0": float(cfg.get("head_first_omega_0", 30.0)),
        "head_hidden_omega_0": float(cfg.get("head_hidden_omega_0", 30.0)),
    }


MODEL_BUILDERS: dict[str, ModelBuilder] = {
    "siren": _build_siren,
    "coordnet": _build_coordnet,
    "moe_inr": _build_moe_inr,
    "light_basis_expert": _build_light_basis_expert,
    "shared_enc_inr": _build_shared_enc_inr,
}

MODEL_CONFIG_MATERIALIZERS: dict[str, ModelConfigMaterializer] = {
    "siren": _materialize_siren,
    "coordnet": _materialize_coordnet,
    "moe_inr": _materialize_moe_inr,
    "light_basis_expert": _materialize_light_basis_expert,
    "shared_enc_inr": _materialize_shared_enc_inr,
}


def materialize_model_config(model_cfg: ModelConfig, meta: DatasetMeta) -> dict[str, Any]:
    model_name = str(model_cfg.name)
    if model_name not in MODEL_CONFIG_MATERIALIZERS:
        raise ValueError(f"Unknown model name: {model_cfg.name}")
    payload = MODEL_CONFIG_MATERIALIZERS[model_name](dict(model_cfg.params), meta)
    return {"name": model_name, **payload}


def build_model(model_cfg: ModelConfig, meta: DatasetMeta):
    payload = materialize_model_config(model_cfg, meta)
    model_name = str(payload.pop("name"))
    return ModelAdapter(MODEL_BUILDERS[model_name](payload, meta))
