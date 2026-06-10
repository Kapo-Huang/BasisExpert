import numpy as np
import torch
from torch import nn


class Sine(nn.Module):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return torch.sin(30 * input)


class SineLayer(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        is_first: bool = False,
        omega_0: float = 30.0,
    ):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.init_weights()

    def init_weights(self) -> None:
        with torch.no_grad():
            if self.is_first:
                bound = 1.0 / self.in_features
            else:
                bound = np.sqrt(6.0 / self.in_features) / self.omega_0
            self.linear.weight.uniform_(-bound, bound)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.omega_0 * self.linear(input))


class LinearLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.init_weights()

    def init_weights(self) -> None:
        with torch.no_grad():
            self.linear.weight.uniform_(-1.0 / self.in_features, 1.0 / self.in_features)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.linear(input)


class ResBlock(nn.Module):
    def __init__(self, in_features: int, out_features: int, nonlinearity: str = "relu"):
        super().__init__()
        nls_and_inits = {
            "sine": Sine(),
            "relu": nn.ReLU(inplace=True),
            "sigmoid": nn.Sigmoid(),
            "tanh": nn.Tanh(),
            "selu": nn.SELU(inplace=True),
            "softplus": nn.Softplus(),
            "elu": nn.ELU(inplace=True),
        }
        self.nl = nls_and_inits[nonlinearity]
        self.net = nn.Sequential(
            SineLayer(in_features, out_features),
            SineLayer(out_features, out_features),
        )
        self.flag = in_features != out_features
        if self.flag:
            self.transform = SineLayer(in_features, out_features)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        outputs = self.net(features)
        if self.flag:
            features = self.transform(features)
        return 0.5 * (outputs + features)


class CoordNet(nn.Module):
    def __init__(self, in_features: int, out_features: int, init_features: int = 64, num_res: int = 10):
        super().__init__()
        layers = [
            ResBlock(in_features, init_features),
            ResBlock(init_features, 2 * init_features),
            ResBlock(2 * init_features, 4 * init_features),
        ]
        for _ in range(num_res):
            layers.append(ResBlock(4 * init_features, 4 * init_features))
        layers.append(ResBlock(4 * init_features, out_features))
        self.net = nn.Sequential(*layers)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        return self.net(coords)


def build_coordnet_from_config(model_cfg) -> CoordNet:
    return CoordNet(
        in_features=int(model_cfg.get("in_features", 4)),
        out_features=int(model_cfg.get("out_features", 24)),
        init_features=int(model_cfg.get("init_features", 64)),
        num_res=int(model_cfg.get("num_res", 10)),
    )
