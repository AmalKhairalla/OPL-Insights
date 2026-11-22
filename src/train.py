"""One shared trainer for every controlled experiment.

Training loss uses standardized y as in the Colab baseline; reported train/validation
MSE is calculated after inverse-scaling predictions back to kilograms (kg²).
"""

from copy import deepcopy
import random
import time

import numpy as np
import torch
from sklearn.metrics import mean_squared_error
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .config import BASELINE_CONFIG
from .data import PreparedData
from .model import build_model


def _make_optimizer(name: str, parameters, config: dict):
    name = name.lower()
    lr = float(config["learning_rate"])
    if name == "sgd":
        return torch.optim.SGD(parameters, lr=lr, momentum=float(config["momentum"]))
    if name == "adam":
        return torch.optim.Adam(parameters, lr=lr)
    if name == "adamw":
        return torch.optim.AdamW(parameters, lr=lr)
    if name == "rmsprop":
        return torch.optim.RMSprop(parameters, lr=lr)
    raise ValueError(f"Unsupported optimizer: {name}")


def _make_loss(name: str, delta: float):
    name = name.lower()
    if name in {"mse", "mse_loss"}:
        return nn.MSELoss()
    if name in {"l1", "mae"}:
        return nn.L1Loss()
    if name == "huber":
        if delta <= 0:
            raise ValueError("Huber delta must be positive")
        return nn.HuberLoss(delta=delta)
    raise ValueError(f"Unsupported loss function: {name}")


def run_experiment(config: dict, data: PreparedData, device: str | None = None) -> dict:
    """Train from scratch and report final-epoch original-scale MSE and all curves.

    No test data is used. All experiments use an identical fixed split and budget.
    If the optimizer diverges, the run is marked failed rather than silently discarded.
    """
    settings = deepcopy(BASELINE_CONFIG)
    unknown = set(config) - set(settings)
    if unknown:
        raise ValueError(f"Unknown config settings: {sorted(unknown)}")
    settings.update(deepcopy(config))
    if int(settings["epochs"]) < 1 or int(settings["batch_size"]) < 1:
        raise ValueError("epochs and batch_size must be positive")

    seed = int(settings["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    resolved_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = build_model(
        input_dim=data.X_train.shape[1],
        hidden_layers=list(settings["hidden_layers"]),
        activation=settings["activation"],
    ).to(resolved_device)
    optimizer = _make_optimizer(settings["optimizer"], model.parameters(), settings)
    criterion = _make_loss(settings["loss"], float(settings["huber_delta"]))

    # CPU-backed DataLoader also works when the model itself is on GPU.
    x_train = torch.tensor(data.X_train, dtype=torch.float32)
    y_train_scaled = torch.tensor(data.y_train_scaled, dtype=torch.float32)
    loader = DataLoader(
        TensorDataset(x_train, y_train_scaled),
        batch_size=int(settings["batch_size"]),
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    x_train_eval = x_train.to(resolved_device)
    x_val_eval = torch.tensor(data.X_val, dtype=torch.float32, device=resolved_device)

    train_losses, val_losses = [], []
    status, error = "completed", ""
    start = time.perf_counter()

    for epoch in range(int(settings["epochs"])):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(resolved_device), yb.to(resolved_device)
            optimizer.zero_grad()
            prediction = model(xb)
            loss = criterion(prediction, yb)
            if not torch.isfinite(loss):
                status, error = "failed", "Non-finite training loss"
                break
            loss.backward()
            optimizer.step()
        if status == "failed":
            break

        model.eval()
        with torch.no_grad():
            train_scaled_pred = model(x_train_eval).cpu().numpy()
            val_scaled_pred = model(x_val_eval).cpu().numpy()
        if not (np.isfinite(train_scaled_pred).all() and np.isfinite(val_scaled_pred).all()):
            status, error = "failed", "Non-finite predictions"
            break
        train_pred = data.y_scaler.inverse_transform(train_scaled_pred).ravel()
        val_pred = data.y_scaler.inverse_transform(val_scaled_pred).ravel()
        train_mse = float(mean_squared_error(data.y_train, train_pred))
        val_mse = float(mean_squared_error(data.y_val, val_pred))
        if not (np.isfinite(train_mse) and np.isfinite(val_mse)):
            status, error = "failed", "Non-finite evaluation MSE"
            break
        train_losses.append(train_mse)
        val_losses.append(val_mse)

    elapsed = time.perf_counter() - start
    return {
        "config": settings,
        "status": status,
        "error": error,
        "train_mse": train_losses[-1] if train_losses else None,
        "val_mse": val_losses[-1] if val_losses else None,
        "best_val_mse": min(val_losses) if val_losses else None,
        "best_epoch": int(np.argmin(val_losses) + 1) if val_losses else None,
        "epochs_completed": len(val_losses),
        "time_seconds": elapsed,
        "device": str(resolved_device),
        "train_losses": train_losses,
        "val_losses": val_losses,
        "model": model,  # In memory only; results.py does not attempt to serialize it.
    }
