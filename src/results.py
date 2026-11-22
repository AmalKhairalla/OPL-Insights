"""Persist experiment metrics and learning curves, safely across Colab restarts."""

import json
from pathlib import Path

import pandas as pd


def save_result(run_id: str, topic: str, result: dict, folder: str | Path, observation: str = "") -> tuple[Path, Path]:
    """Append one unique run to experiments.csv and loss_histories.json.

    Use distinct result folders for teammates, e.g. results/A and results/B.
    Files in /content disappear when Colab restarts: use mounted Drive or upload to GitHub.
    """
    if not run_id or not run_id.strip():
        raise ValueError("run_id is required")
    output = Path(folder)
    output.mkdir(parents=True, exist_ok=True)
    csv_path = output / "experiments.csv"
    json_path = output / "loss_histories.json"

    existing = pd.read_csv(csv_path) if csv_path.exists() else pd.DataFrame()
    if not existing.empty and run_id in existing["run_id"].astype(str).values:
        raise ValueError(f"Run {run_id!r} is already saved; choose a new ID or review the result")

    histories = json.loads(json_path.read_text(encoding="utf-8")) if json_path.exists() else {}
    if run_id in histories:
        raise ValueError(f"Run {run_id!r} already exists in saved histories")

    settings = result["config"]
    row = {
        "run_id": run_id,
        "topic": topic,
        "status": result["status"],
        "error": result["error"],
        "train_mse_kg2": result["train_mse"],
        "val_mse_kg2": result["val_mse"],
        "best_val_mse_kg2": result["best_val_mse"],
        "best_epoch": result["best_epoch"],
        "epochs_completed": result["epochs_completed"],
        "time_seconds": result["time_seconds"],
        "device": result["device"],
        "hidden_layers": json.dumps(settings["hidden_layers"]),
        "activation": settings["activation"],
        "optimizer": settings["optimizer"],
        "learning_rate": settings["learning_rate"],
        "momentum": settings["momentum"],
        "batch_size": settings["batch_size"],
        "loss": settings["loss"],
        "huber_delta": settings["huber_delta"],
        "epochs": settings["epochs"],
        "seed": settings["seed"],
        "observation": observation,
    }
    # Writes the full CSV on each call, preserving previous rows after notebook restarts.
    updated = pd.concat([existing, pd.DataFrame([row])], ignore_index=True)
    updated.to_csv(csv_path, index=False)
    histories[run_id] = {
        "config": settings,
        "train_mse_kg2": result["train_losses"],
        "val_mse_kg2": result["val_losses"],
        "status": result["status"],
    }
    json_path.write_text(json.dumps(histories, indent=2), encoding="utf-8")
    return csv_path, json_path
