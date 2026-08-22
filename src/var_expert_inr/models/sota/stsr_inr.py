from __future__ import annotations

import math

import torch
import torch.nn as nn


class SineLayer(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        is_first: bool = False,
        omega_0: float = 30.0,
    ) -> None:
        super().__init__()
        self.in_features = int(in_features)
        self.omega_0 = float(omega_0)
        self.linear = nn.Linear(self.in_features, int(out_features))
        with torch.no_grad():
            if is_first:
                bound = 1.0 / self.in_features
            else:
                bound = math.sqrt(6.0 / self.in_features) / self.omega_0
            self.linear.weight.uniform_(-bound, bound)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.omega_0 * self.linear(inputs))


class ResidualSineBlock(nn.Module):
    def __init__(self, features: int, *, omega_0: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            SineLayer(features, features, omega_0=omega_0),
            SineLayer(features, features, omega_0=omega_0),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return 0.5 * (self.net(inputs) + inputs)


class STSRBody(nn.Module):
    def __init__(
        self,
        in_features: int,
        *,
        init_features: int,
        num_res: int,
        omega_0: float,
    ) -> None:
        super().__init__()
        width = 4 * int(init_features)
        layers: list[nn.Module] = [
            SineLayer(in_features, init_features, omega_0=omega_0),
            SineLayer(init_features, 2 * init_features, omega_0=omega_0),
            SineLayer(2 * init_features, width, omega_0=omega_0),
        ]
        layers.extend(
            ResidualSineBlock(width, omega_0=omega_0)
            for _ in range(int(num_res))
        )
        self.layers = nn.ModuleList(layers)


class STSRHead(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        out_features: int,
        *,
        outermost_linear: bool,
        omega_0: float,
    ) -> None:
        super().__init__()
        synthesis = [
            ResidualSineBlock(feature_dim, omega_0=omega_0),
            SineLayer(feature_dim, feature_dim // 2, omega_0=omega_0),
            SineLayer(feature_dim // 2, feature_dim // 4, omega_0=omega_0),
        ]
        modulation = [
            ResidualSineBlock(feature_dim, omega_0=omega_0),
            SineLayer(feature_dim, feature_dim // 2, omega_0=omega_0),
            SineLayer(feature_dim // 2, feature_dim // 4, omega_0=omega_0),
        ]
        self.synthesis = nn.ModuleList(synthesis)
        self.modulation = nn.ModuleList(modulation)
        self.final = (
            nn.Sequential(
                nn.Linear(feature_dim // 4, int(out_features)),
                nn.Tanh(),
            )
            if outermost_linear
            else SineLayer(
                feature_dim // 4,
                int(out_features),
                omega_0=omega_0,
            )
        )

    def forward(self, features: torch.Tensor, latent: torch.Tensor) -> torch.Tensor:
        for synthesis, modulation in zip(self.synthesis, self.modulation):
            features = synthesis(features * latent)
            latent = modulation(latent)
        return self.final(features * latent)


class STSRINR(nn.Module):
    """Multi-attribute STSR-INR with a shared synthesis/modulation trunk."""

    def __init__(
        self,
        *,
        in_features: int,
        target_dims: dict[str, int],
        init_features: int = 64,
        num_res: int = 5,
        omega_0: float = 5.0,
        embedding_dims: int = 256,
        outermost_linear: bool = True,
        use_global_latent: bool = True,
    ) -> None:
        super().__init__()
        if not target_dims:
            raise ValueError("STSR-INR requires at least one target")
        if not use_global_latent:
            raise ValueError("The unified STSR-INR adapter requires use_global_latent=true")
        self.target_dims = dict(target_dims)
        self.synthesis_body = STSRBody(
            int(in_features),
            init_features=int(init_features),
            num_res=int(num_res),
            omega_0=float(omega_0),
        )
        self.modulation_body = STSRBody(
            int(embedding_dims),
            init_features=int(init_features),
            num_res=int(num_res),
            omega_0=float(omega_0),
        )
        feature_dim = 4 * int(init_features)
        self.heads = nn.ModuleDict(
            {
                name: STSRHead(
                    feature_dim,
                    int(out_dim),
                    outermost_linear=bool(outermost_linear),
                    omega_0=float(omega_0),
                )
                for name, out_dim in self.target_dims.items()
            }
        )
        self.global_latent = nn.Parameter(torch.zeros(1, int(embedding_dims)))

    def forward(
        self,
        coords: torch.Tensor,
        request: str | None = None,
        *,
        return_aux: bool = False,
    ):
        if request is not None and request not in self.heads:
            raise KeyError(
                f"Unknown STSR-INR target {request!r}; available={list(self.heads)}"
            )
        latent = self.global_latent.expand(coords.shape[0], -1)
        latent_features = self.modulation_body.layers[0](latent)
        coord_features = self.synthesis_body.layers[0](coords)
        for synthesis, modulation in zip(
            self.synthesis_body.layers[1:],
            self.modulation_body.layers[1:],
        ):
            coord_features = synthesis(coord_features * latent_features)
            latent_features = modulation(latent_features)
        predictions = {
            name: head(coord_features, latent_features)
            for name, head in self.heads.items()
        }
        output = predictions if request is None else predictions[request]
        if return_aux:
            return output, {}
        return output


def build_stsr_inr_from_config(
    cfg: dict,
    target_dims: dict[str, int],
) -> STSRINR:
    return STSRINR(target_dims=target_dims, **cfg)
