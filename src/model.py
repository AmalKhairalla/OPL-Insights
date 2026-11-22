"""Construct configurable PyTorch regression networks."""

import torch.nn as nn


def build_model(input_dim: int, hidden_layers: list[int], activation: str = "relu") -> nn.Module:
    if input_dim < 1:
        raise ValueError("input_dim must be positive")
    if any(not isinstance(n, int) or n < 1 for n in hidden_layers):
        raise ValueError("hidden_layers must contain positive integers")

    activations = {"relu": nn.ReLU, "tanh": nn.Tanh, "leaky_relu": nn.LeakyReLU}
    if activation.lower() not in activations:
        raise ValueError(f"Unsupported activation {activation!r}; choose {list(activations)}")

    layers = []
    width = input_dim
    for next_width in hidden_layers:
        layers.extend([nn.Linear(width, next_width), activations[activation.lower()]()])
        width = next_width
    layers.append(nn.Linear(width, 1))
    return nn.Sequential(*layers)
