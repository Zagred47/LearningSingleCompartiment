"""Training and evaluation utilities shared by the CLI and Kaggle notebook."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, Sequence

import h5py
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .dataset import Normalization, SequenceWindowDataset
from .models import build_model
from .simulator import INPUT_NAMES, STATE_NAMES


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    # Voltage and calcium receive extra emphasis; all Markov variables remain targets.
    weights = torch.ones(target.shape[-1], device=target.device)
    weights[0] = 4.0
    weights[1] = 2.0
    return ((prediction - target).square() * weights).mean()


@torch.no_grad()
def evaluate_loader(model, loader, device: torch.device) -> float:
    model.eval()
    losses = []
    for features, target in loader:
        features, target = features.to(device), target.to(device)
        losses.append(float(_loss(model(features), target).cpu()))
    return float(np.mean(losses)) if losses else float("nan")


@torch.no_grad()
def one_step_metrics(
    model,
    dataset_path: str | Path,
    split: str,
    normalization: Normalization,
    device: str | torch.device = "cpu",
) -> Dict[str, object]:
    device = torch.device(device)
    model.eval()
    with h5py.File(dataset_path, "r") as handle:
        states = handle[f"{split}/states"][...].astype(np.float32)
        inputs = handle[f"{split}/inputs"][...].astype(np.float32)
    state_t = (states[:, :-1] - normalization.state_mean) / normalization.state_std
    inputs_n = (inputs - normalization.input_mean) / normalization.input_std
    features = np.concatenate([state_t, inputs_n], axis=-1).astype(np.float32)
    predictions = []
    for start in range(0, len(features), 8):
        batch = torch.from_numpy(features[start : start + 8]).to(device)
        predictions.append(model(batch).cpu().numpy())
    prediction_n = np.concatenate(predictions, axis=0)
    prediction = prediction_n * normalization.state_std + normalization.state_mean
    target = states[:, 1:]
    error = prediction - target
    rmse = np.sqrt(np.mean(error**2, axis=(0, 1)))
    persistence_error = states[:, :-1] - target
    persistence_rmse = np.sqrt(np.mean(persistence_error**2, axis=(0, 1)))
    return {
        "normalized_mse": float(np.mean((prediction_n - (target - normalization.state_mean) / normalization.state_std) ** 2)),
        "voltage_rmse_mv": float(rmse[0]),
        "calcium_rmse_mm": float(rmse[1]),
        "mean_normalized_rmse": float(np.mean(rmse / normalization.state_std)),
        "persistence_voltage_rmse_mv": float(persistence_rmse[0]),
        "per_state_rmse": {name: float(value) for name, value in zip(STATE_NAMES, rmse)},
    }


@torch.no_grad()
def rollout_trajectory(
    model,
    initial_state: np.ndarray,
    inputs: np.ndarray,
    normalization: Normalization,
    device: str | torch.device = "cpu",
) -> np.ndarray:
    """Autoregressively predict a complete state trajectory."""

    device = torch.device(device)
    model.eval()
    prediction = np.empty((len(inputs) + 1, len(STATE_NAMES)), dtype=np.float32)
    prediction[0] = initial_state
    hidden = None
    for index, input_row in enumerate(inputs):
        state_n = (prediction[index] - normalization.state_mean) / normalization.state_std
        input_n = (input_row - normalization.input_mean) / normalization.input_std
        features = torch.from_numpy(
            np.concatenate([state_n, input_n]).astype(np.float32)[None, None]
        ).to(device)
        next_n, hidden = model(features, hidden=hidden, return_hidden=True)
        next_state = next_n[0, 0].cpu().numpy() * normalization.state_std + normalization.state_mean
        next_state[1] = max(0.0, next_state[1])
        next_state[2:14] = np.clip(next_state[2:14], 0.0, 1.0)
        next_state[14:17] = np.maximum(next_state[14:17], 0.0)
        prediction[index + 1] = next_state
    return prediction


def train_model(
    dataset_path: str | Path,
    output_dir: str | Path,
    architecture: str,
    *,
    epochs: int = 12,
    sequence_length: int = 64,
    stride: int = 16,
    batch_size: int = 64,
    hidden_dim: int = 128,
    layers: int = 2,
    learning_rate: float = 1e-3,
    seed: int = 2026,
    device: str | None = None,
) -> Dict[str, object]:
    """Train one architecture with validation checkpointing."""

    seed_everything(seed)
    device_obj = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    normalization = Normalization.from_h5(dataset_path)
    train_data = SequenceWindowDataset(dataset_path, "train", normalization, sequence_length, stride)
    validation_data = SequenceWindowDataset(dataset_path, "validation", normalization, sequence_length, stride)
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, generator=generator)
    validation_loader = DataLoader(validation_data, batch_size=batch_size, shuffle=False)

    model = build_model(
        architecture,
        input_dim=len(STATE_NAMES) + len(INPUT_NAMES),
        state_dim=len(STATE_NAMES),
        hidden_dim=hidden_dim,
        layers=layers,
    ).to(device_obj)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    best_loss = float("inf")
    best_state = None
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []
        for features, target in train_loader:
            features, target = features.to(device_obj), target.to(device_obj)
            optimizer.zero_grad(set_to_none=True)
            loss = _loss(model(features), target)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))
        validation_loss = evaluate_loader(model, validation_loader, device_obj)
        history.append(
            {"epoch": epoch, "train_loss": float(np.mean(train_losses)), "validation_loss": validation_loss}
        )
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    if best_state is None:
        raise RuntimeError("training produced no checkpoint")
    model.load_state_dict(best_state)
    metrics = one_step_metrics(model, dataset_path, "test", normalization, device_obj)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / f"{architecture}.pt"
    torch.save(
        {
            "architecture": architecture,
            "model_state": best_state,
            "model_kwargs": {"hidden_dim": hidden_dim, "layers": layers},
            "normalization": normalization.to_dict(),
            "state_names": list(STATE_NAMES),
            "input_names": list(INPUT_NAMES),
        },
        checkpoint_path,
    )
    report: Dict[str, object] = {
        "architecture": architecture,
        "device": str(device_obj),
        "parameters": int(sum(parameter.numel() for parameter in model.parameters())),
        "best_validation_loss": best_loss,
        "test": metrics,
        "history": history,
        "checkpoint": str(checkpoint_path),
    }
    (output_dir / f"{architecture}.metrics.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report


def train_architectures(
    dataset_path: str | Path,
    output_dir: str | Path,
    architectures: Sequence[str] = ("mlp", "gru", "lstm"),
    **kwargs,
) -> list[Dict[str, object]]:
    return [
        train_model(dataset_path, output_dir, architecture, **kwargs)
        for architecture in architectures
    ]
