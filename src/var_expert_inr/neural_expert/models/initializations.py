from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


def sirenWeightInit(decoder):
    for i in range(len(decoder.weights_list)):
        fan_in = decoder.weights_list[i].shape[-2]
        if i == 0 and fan_in in {2, 3}:
            bound = 1 / fan_in
        else:
            bound = np.sqrt(6 / fan_in) / 30
        nn.init.uniform_(decoder.weights_list[i], -bound, bound)
        nn.init.uniform_(decoder.biases_list[i], 0.0, 0.0)


def finerWeightInit(decoder):
    for i in range(len(decoder.weights_list)):
        fan_in = decoder.weights_list[i].shape[-2]
        if i == 0 and fan_in in {2, 3}:
            bound = 1 / fan_in
        else:
            bound = np.sqrt(6 / fan_in) / 30
        nn.init.uniform_(decoder.weights_list[i], -bound, bound)


def sirenSameWeightInit(decoder):
    for i in range(len(decoder.weights_list)):
        fan_in = decoder.weights_list[i].shape[-2]
        bound = np.sqrt(6 / fan_in) / 30
        n_experts = decoder.weights_list[i].shape[0]
        w_init = None
        for expert_idx in range(n_experts):
            if expert_idx == 0:
                w_init = torch.empty_like(decoder.weights_list[i][expert_idx]).uniform_(-bound, bound)
            decoder.weights_list[i][expert_idx] = w_init.detach()
        nn.init.uniform_(decoder.biases_list[i], 0.0, 0.0)


def sirenGeomWeightInit(decoder, flip=False, r=0.5, centroids=None):
    parallel = len(decoder.weights_list[0].shape) == 3
    num_experts = decoder.weights_list[0].shape[0]
    flip_val = -1 if flip else 1
    for i in range(len(decoder.weights_list)):
        fan_in = decoder.weights_list[i].shape[-2]
        fan_out = decoder.weights_list[i].shape[-1]
        if i == 0:
            nn.init.uniform_(decoder.weights_list[i], -np.sqrt(3 / fan_out), np.sqrt(3 / fan_out))
            nn.init.uniform_(decoder.biases_list[i], -(1 / (fan_out * 1000)), (1 / (fan_out * 1000)))
            if centroids is not None and parallel:
                for expert_idx in range(num_experts):
                    bias_term = -centroids[expert_idx, None] @ decoder.weights_list[i][expert_idx]
                    decoder.biases_list[i][expert_idx] = decoder.biases_list[i][expert_idx] + bias_term
            decoder.biases_list[i] /= 30
            decoder.weights_list[i] /= 30
        elif i == len(decoder.weights_list) - 2:
            if parallel:
                eye = torch.eye(fan_in, device=decoder.weights_list[i].device)
                noise = 0.001 * torch.randn_like(decoder.weights_list[i])
                decoder.weights_list[i].data = 0.5 * np.pi * eye.unsqueeze(0) + noise
                decoder.biases_list[i].data = 0.5 * np.pi * torch.ones_like(decoder.biases_list[i]) + 0.001 * torch.randn_like(decoder.biases_list[i])
            else:
                eye = torch.eye(fan_out, device=decoder.weights_list[i].device)
                decoder.weights_list[i].data = 0.5 * np.pi * eye + 0.001 * torch.randn_like(decoder.weights_list[i])
                decoder.biases_list[i].data = 0.5 * np.pi * torch.ones_like(decoder.biases_list[i]) + 0.001 * torch.randn_like(decoder.biases_list[i])
            decoder.weights_list[i] /= 30
            decoder.biases_list[i] /= 30
        elif i == len(decoder.weights_list) - 1:
            nn.init.ones_(decoder.weights_list[i])
            decoder.weights_list[i].data.mul_(-1.0 * flip_val)
            decoder.biases_list[i].data.zero_().add_(fan_in)
            decoder.biases_list[i].data = flip_val * decoder.biases_list[i].data + r
        else:
            nn.init.uniform_(decoder.weights_list[i], -np.sqrt(6 / fan_in) / 30, np.sqrt(6 / fan_in) / 30)
            nn.init.uniform_(decoder.biases_list[i], 0.0, 0.0)


def geomReluWeightInit(decoder, flip=False, r=0.5, centroids=None):
    parallel = len(decoder.weights_list[0].shape) == 3
    num_experts = decoder.weights_list[0].shape[0]
    flip_val = -1 if flip else 1
    for i in range(len(decoder.weights_list)):
        fan_in = decoder.weights_list[i].shape[-2]
        fan_out = decoder.weights_list[i].shape[-1]
        if i == 0:
            decoder.weights_list[i].data.normal_(mean=0.0, std=np.sqrt(2) / np.sqrt(fan_out))
            decoder.biases_list[i].data.zero_()
            if centroids is not None and parallel:
                for expert_idx in range(num_experts):
                    bias_term = -centroids[expert_idx, None] @ decoder.weights_list[i][expert_idx]
                    decoder.biases_list[i][expert_idx] = decoder.biases_list[i][expert_idx] + bias_term.squeeze()
        elif i == len(decoder.weights_list) - 1:
            decoder.weights_list[i].data.normal_(mean=np.sqrt(np.pi) / np.sqrt(fan_in), std=1.0e-5)
            decoder.biases_list[i].data.zero_()
            decoder.weights_list[i].data = flip_val * decoder.weights_list[i].data
            decoder.biases_list[i].data = decoder.biases_list[i].data - flip_val * r
        else:
            decoder.weights_list[i].data.normal_(mean=0.0, std=np.sqrt(2) / np.sqrt(fan_out))
            decoder.biases_list[i].data.zero_()


def kaimingNormalWeightInit(decoder):
    for i in range(len(decoder.weights_list)):
        nn.init.kaiming_normal_(decoder.weights_list[i], a=0.0, nonlinearity="relu", mode="fan_out")


def kaimingUniformWeightInit(decoder):
    for i in range(len(decoder.weights_list)):
        nn.init.kaiming_uniform_(decoder.weights_list[i], a=0.0, nonlinearity="relu", mode="fan_out")
