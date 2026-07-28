from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from sklearn.cluster import KMeans

from .model import PackedSiren
from .pruning import BIAS_CANDIDATES, WEIGHT_CANDIDATES


@dataclass
class QuantizedParameter:
    name: str
    labels: np.ndarray
    codebook: torch.nn.Parameter
    mask: np.ndarray


@dataclass
class ModelQuantization:
    parameters: dict[str, QuantizedParameter]
    bits: int

    def materialize(self, model: PackedSiren) -> None:
        named = dict(model.named_parameters())
        with torch.no_grad():
            for name, item in self.parameters.items():
                labels = torch.from_numpy(item.labels).to(item.codebook.device, dtype=torch.long)
                mask = torch.from_numpy(item.mask).to(item.codebook.device, dtype=torch.bool)
                restored = torch.zeros(labels.shape, dtype=item.codebook.dtype, device=item.codebook.device)
                restored[mask] = item.codebook[labels[mask]]
                named[name].copy_(restored.to(named[name].device, dtype=named[name].dtype))

    def codebook_parameters(self) -> list[torch.nn.Parameter]:
        return [item.codebook for item in self.parameters.values()]

    def collect_codebook_gradients(self, model: PackedSiren) -> None:
        named = dict(model.named_parameters())
        for name, item in self.parameters.items():
            dense_grad = named[name].grad
            item.codebook.grad = torch.zeros_like(item.codebook)
            if dense_grad is None:
                continue
            labels = torch.from_numpy(item.labels).to(dense_grad.device, dtype=torch.long)
            mask = torch.from_numpy(item.mask).to(dense_grad.device, dtype=torch.bool)
            item.codebook.grad.scatter_add_(0, labels[mask], dense_grad[mask].to(item.codebook.dtype))
            dense_grad.zero_()

    def state_dict(self) -> dict:
        return {
            "bits": int(self.bits),
            "parameters": {
                name: {
                    "labels": item.labels,
                    "mask": item.mask,
                    "codebook": item.codebook.detach().cpu().numpy().astype(np.float32),
                }
                for name, item in self.parameters.items()
            },
        }


def _quantize_values(values: np.ndarray, *, bits: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    flat = np.asarray(values, dtype=np.float32).reshape(-1)
    if flat.size == 0:
        return np.empty(0, dtype=np.float32), np.empty(0, dtype=np.int64)
    cluster_count = min(2 ** int(bits), int(flat.size), int(np.unique(flat).size))
    if cluster_count <= 1:
        centers = np.array([float(flat.mean())], dtype=np.float32)
        labels = np.zeros(flat.size, dtype=np.int64)
    else:
        estimator = KMeans(
            n_clusters=cluster_count,
            init="k-means++",
            n_init=1,
            max_iter=300,
            tol=1.0e-4,
            algorithm="lloyd",
            random_state=int(seed),
        ).fit(flat[:, None])
        centers = estimator.cluster_centers_[:, 0].astype(np.float32)
        labels = estimator.labels_.astype(np.int64)
        order = np.argsort(centers, kind="stable")
        inverse = np.empty_like(order)
        inverse[order] = np.arange(order.size)
        centers = centers[order]
        labels = inverse[labels]
    return centers, labels


def quantize_array(values: np.ndarray, *, bits: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Public deterministic scalar K-means quantizer used by the CNN exporter."""
    array = np.asarray(values, dtype=np.float32)
    centers, labels = _quantize_values(array, bits=bits, seed=seed)
    return centers, labels.reshape(array.shape)


def quantize_model(
    model: PackedSiren,
    pruning_masks: dict[str, torch.Tensor],
    *,
    bits: int = 8,
    seed: int = 42,
) -> ModelQuantization:
    named = dict(model.named_parameters())
    parameters: dict[str, QuantizedParameter] = {}
    for offset, name in enumerate((*WEIGHT_CANDIDATES, *BIAS_CANDIDATES)):
        values = named[name].detach().cpu().numpy().astype(np.float32)
        mask = pruning_masks[name].detach().cpu().numpy().astype(bool)
        centers, active_labels = _quantize_values(values[mask], bits=bits, seed=seed + offset)
        labels = np.full(values.shape, -1, dtype=np.int64)
        labels[mask] = active_labels
        parameters[name] = QuantizedParameter(
            name=name,
            labels=labels,
            mask=mask,
            codebook=torch.nn.Parameter(torch.from_numpy(centers).to(named[name].device)),
        )
    state = ModelQuantization(parameters=parameters, bits=int(bits))
    state.materialize(model)
    return state


def unquantized_parameters(model: PackedSiren) -> list[torch.nn.Parameter]:
    quantized = set((*WEIGHT_CANDIDATES, *BIAS_CANDIDATES))
    return [parameter for name, parameter in model.named_parameters() if name not in quantized]
