from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import initializations as inits


class Sine(nn.Module):
    def __init__(self, freq=30, trainable=False):
        super().__init__()
        if trainable:
            self.freq = nn.Parameter(torch.tensor(freq))
        else:
            self.freq = float(freq)

    def forward(self, input):
        return torch.sin(self.freq * input)


class FINER(nn.Module):
    def __init__(self, freq=30, trainable=False):
        super().__init__()
        if trainable:
            self.freq = nn.Parameter(torch.tensor(freq))
        else:
            self.freq = float(freq)

    def forward(self, input):
        with torch.no_grad():
            scale = torch.abs(input) + 1
        return torch.sin(self.freq * scale * input)


class DummyModule(nn.Module):
    def forward(self, x):
        return x


class FullyConnectedNN(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        num_hidden_layers,
        hidden_features,
        outermost_linear=False,
        nonlinearity="sine",
        init_type="siren",
        input_encoding=None,
        sphere_init_params=(1.6, 1.0),
        init_r=0.5,
        freq=30,
        trainable_freqs=False,
        module_name="",
    ):
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.num_hidden_layers = int(num_hidden_layers)
        self.hidden_features = int(hidden_features)
        self.outermost_linear = bool(outermost_linear)
        self.init_type = str(init_type)
        self.input_encoding = input_encoding
        self.sphere_init_params = tuple(sphere_init_params)
        self.module_name = module_name

        self.weights_list = nn.ParameterList([])
        self.biases_list = nn.ParameterList([])
        self.weights_list.append(nn.Parameter(torch.zeros(self.in_features, self.hidden_features)))
        self.biases_list.append(nn.Parameter(torch.zeros(self.hidden_features)))
        for _ in range(self.num_hidden_layers):
            self.weights_list.append(nn.Parameter(torch.zeros(self.hidden_features, self.hidden_features)))
            self.biases_list.append(nn.Parameter(torch.zeros(self.hidden_features)))
        self.weights_list.append(nn.Parameter(torch.zeros(self.hidden_features, self.out_features)))
        self.biases_list.append(nn.Parameter(torch.zeros(self.out_features)))

        nl_dict = {
            "sine": Sine(freq, trainable_freqs),
            "relu": nn.ReLU(inplace=True),
            "softplus": nn.Softplus(beta=100),
            "tanh": nn.Tanh(),
            "sigmoid": nn.Sigmoid(),
            "finer": FINER(freq, trainable_freqs),
        }
        if nonlinearity not in nl_dict:
            raise ValueError(f"Unsupported nonlinearity: {nonlinearity}")
        self.nl = nl_dict[nonlinearity]

        init_dict: dict[str, Any] = {
            "siren": lambda: inits.sirenWeightInit(self),
            "finer": lambda: inits.finerWeightInit(self),
            "geometric_sine": lambda: inits.sirenGeomWeightInit(self, flip=False, r=init_r),
            "geometric_relu": lambda: inits.geomReluWeightInit(self, flip=False, r=init_r),
            "normal": lambda: inits.kaimingNormalWeightInit(self),
            "kaiminguniform": lambda: inits.kaimingUniformWeightInit(self),
        }
        init_dict[self.init_type]()

    def forward(self, input):
        x = torch.einsum("...nd,dh->...nh", input, self.weights_list[0]) + self.biases_list[0]
        x = self.nl(x)
        for layer_idx in range(self.num_hidden_layers):
            x = torch.einsum("...nd,dh->...nh", x, self.weights_list[layer_idx + 1]) + self.biases_list[layer_idx + 1]
            x = self.nl(x)
        x = torch.einsum("...nh,ho->...no", x, self.weights_list[-1]) + self.biases_list[-1]
        if not self.outermost_linear:
            x = self.nl(x)
        if self.init_type in {"mfgi", "geometric_sine"}:
            radius, scaling = self.sphere_init_params
            x = torch.sign(x) * torch.sqrt(x.abs() + 1.0e-8)
            x = (x - radius) * scaling
        return x


class ParallelFullyConnectedNN(nn.Module):
    def __init__(
        self,
        k,
        in_features,
        out_features,
        num_hidden_layers,
        hidden_features,
        outermost_linear=False,
        nonlinearity="sine",
        init_type="siren",
        input_encoding=None,
        sphere_init_params=(1.6, 1.0),
        init_r=0.5,
        freq=30,
        module_name="",
    ):
        super().__init__()
        self.k = int(k)
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.num_hidden_layers = int(num_hidden_layers)
        self.hidden_features = int(hidden_features)
        self.outermost_linear = bool(outermost_linear)
        self.init_type = str(init_type)
        self.input_encoding = input_encoding
        self.sphere_init_params = tuple(sphere_init_params)
        self.module_name = module_name

        self.weights_list = nn.ParameterList([])
        self.biases_list = nn.ParameterList([])
        self.weights_list.append(nn.Parameter(torch.zeros(self.k, self.in_features, self.hidden_features)))
        self.biases_list.append(nn.Parameter(torch.zeros(self.k, self.hidden_features)))
        for _ in range(self.num_hidden_layers):
            self.weights_list.append(nn.Parameter(torch.zeros(self.k, self.hidden_features, self.hidden_features)))
            self.biases_list.append(nn.Parameter(torch.zeros(self.k, self.hidden_features)))
        self.weights_list.append(nn.Parameter(torch.zeros(self.k, self.hidden_features, self.out_features)))
        self.biases_list.append(nn.Parameter(torch.zeros(self.k, self.out_features)))

        nl_dict = {
            "sine": Sine(freq),
            "relu": nn.ReLU(inplace=True),
            "softplus": nn.Softplus(beta=100),
            "tanh": nn.Tanh(),
            "sigmoid": nn.Sigmoid(),
            "finer": FINER(freq),
        }
        if nonlinearity not in nl_dict:
            raise ValueError(f"Unsupported nonlinearity: {nonlinearity}")
        self.nl = nl_dict[nonlinearity]

        init_dict = {
            "siren": lambda: inits.sirenWeightInit(self),
            "finer": lambda: inits.finerWeightInit(self),
            "sirensame": lambda: inits.sirenSameWeightInit(self),
            "geometric_sine": lambda: inits.sirenGeomWeightInit(self, flip=False, r=init_r),
            "geometric_relu": lambda: inits.geomReluWeightInit(self, flip=False, r=init_r),
            "normal": lambda: inits.kaimingNormalWeightInit(self),
            "kaiminguniform": lambda: inits.kaimingUniformWeightInit(self),
        }
        init_dict[self.init_type]()

    def forward(self, x):
        x = torch.einsum("...nd,kdh->...knh", x, self.weights_list[0]) + self.biases_list[0].unsqueeze(-2)
        x = self.nl(x)
        for layer_idx in range(self.num_hidden_layers):
            x = torch.einsum("...knd,kdh->...knh", x, self.weights_list[layer_idx + 1]) + self.biases_list[layer_idx + 1].unsqueeze(-2)
            x = self.nl(x)
        x = torch.einsum("...knh,kho->...kno", x, self.weights_list[-1]) + self.biases_list[-1].unsqueeze(-2)
        if not self.outermost_linear:
            x = self.nl(x)
        if self.init_type in {"mfgi", "geometric_sine"}:
            radius, scaling = self.sphere_init_params
            x = torch.sign(x) * torch.sqrt(x.abs() + 1.0e-8)
            x = (x - radius) * scaling
        return x


class InputEncoder(nn.Module):
    def __init__(self, cfg, input_encoding, hidden_dim, module_name=""):
        super().__init__()
        self.input_encoding = str(input_encoding)
        hidden_features = int(hidden_dim)
        in_features = int(cfg["in_dim"])
        self.first_layer_dim = in_features

        if "FF" in self.input_encoding:
            if hidden_features % 2 != 0:
                raise ValueError("Fourier feature hidden_dim must be even")
            self.bvals_size = hidden_features // 2
            bvals = torch.randn(size=[self.bvals_size, in_features], dtype=torch.float32)
            self.register_buffer("bvals", bvals)
            self.first_layer_dim = hidden_features + in_features
        elif "PE" in self.input_encoding:
            bvals = 2 ** torch.linspace(0.0, 5.0, 6)
            self.register_buffer("bvals", bvals)
            self.first_layer_dim = in_features * 6 * 2 + in_features
        elif "dino" in self.input_encoding:
            self.first_layer_dim = in_features + int(cfg["dino_dim"])

        if "learned" in self.input_encoding:
            parsed_str = self.input_encoding.split("_")
            enc_hidden_features = int(parsed_str[1])
            enc_n_layers = int(parsed_str[2])
            nl = parsed_str[3]
            init = parsed_str[4]
            self.encoder = FullyConnectedNN(
                self.first_layer_dim,
                enc_hidden_features,
                num_hidden_layers=enc_n_layers,
                hidden_features=enc_hidden_features,
                outermost_linear=False,
                nonlinearity=nl,
                init_type=init,
                module_name=module_name + ".encoder",
            )
            self.first_layer_dim = enc_hidden_features + self.first_layer_dim if "cat" in self.input_encoding else enc_hidden_features

    def forward(self, coords, **kwargs):
        if "FF" in self.input_encoding:
            x = (2 * np.pi * coords) @ self.bvals.T
            x = torch.cat([torch.sin(x), torch.cos(x)], axis=-1) / np.sqrt(self.bvals_size)
            x = torch.cat([coords, x], axis=-1)
        elif "PE" in self.input_encoding:
            x = coords[..., None] * self.bvals
            x = x.reshape(*x.shape[:-2], -1)
            x = torch.sin(torch.cat([x, x + np.pi / 2.0], dim=-1))
            x = torch.cat([coords, x], axis=-1)
        elif "dino" in self.input_encoding:
            x = torch.cat([coords, kwargs["dino"]], axis=-1)
        else:
            x = coords

        if "learned" in self.input_encoding:
            encoded = self.encoder(x)
            if "cat" in self.input_encoding:
                x = torch.cat([x, encoded], axis=-1)
            else:
                x = encoded
        return x


class tSoftMax(nn.Module):
    def __init__(self, temperature, dim=-1, trainable=False):
        super().__init__()
        if trainable:
            self.temperature = nn.Parameter(torch.tensor(temperature))
        else:
            self.temperature = float(temperature)
        self.dim = dim
        self.activation = nn.Softmax(dim=dim)

    def forward(self, x):
        return self.activation(x / self.temperature)
