from __future__ import annotations

import re


def parse_loss_string(input_string: str, separator: str = "_"):
    parts = str(input_string).split(separator)
    numbers = []
    substrings = []
    for part in parts:
        num_match = re.match(r"([\d.]+)", part)
        if num_match:
            numbers.append(float(num_match.group(1)))
        substring_match = re.search(r"[a-zA-Z]+", part)
        if substring_match:
            substrings.append(substring_match.group(0))
    return substrings, numbers


def build_loss_dictionary(required_loss_list, weights, model_type, full_loss_list):
    required_loss_dict = {}
    weight_dict = {}
    for loss_name, weight in zip(required_loss_list, weights):
        weight_dict[loss_name] = weight
        required_loss_dict[loss_name] = model_type
    for loss_name in full_loss_list:
        if loss_name not in required_loss_list:
            required_loss_dict[loss_name] = "none"
            weight_dict[loss_name] = 0.0
    return required_loss_dict, weight_dict
