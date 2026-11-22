"""Frozen inputs and reference settings for the DS330 project."""

NUMERIC_FEATURES = [
    "Age",
    "BodyweightKg",
    "Best3SquatKg",
    "Best3BenchKg",
]
CATEGORICAL_FEATURES = ["Sex"]
TARGET = "Best3DeadliftKg"
SPLIT_COLUMN = "Split"
NAME_COLUMN = "Name"

# Every controlled experiment changes ONE setting relative to this reference.
BASELINE_CONFIG = {
    "hidden_layers": [64, 64, 64],
    "activation": "relu",
    "optimizer": "SGD",
    "learning_rate": 0.01,
    "momentum": 0.0,
    "batch_size": 64,
    "loss": "MSE",
    "huber_delta": 1.0,
    "epochs": 30,
    "seed": 42,
}
