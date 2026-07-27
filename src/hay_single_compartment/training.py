"""Training and evaluation utilities shared by the CLI and Kaggle notebook."""

from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Dict, Sequence

import h5py
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .dataset import Normalization, SequenceWindowDataset
from .models import build_model
from .ontology import ONTOLOGY_GROUPS
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
        "per_group_normalized_rmse": {
            group.name: float(np.mean(rmse[list(group.output_indices)] / normalization.state_std[list(group.output_indices)]))
            for group in ONTOLOGY_GROUPS
        },
    }


@torch.no_grad()
def rollout_batch(
    model,
    initial_states: np.ndarray,
    inputs: np.ndarray,
    normalization: Normalization,
    device: str | torch.device = "cpu",
    progress: bool = False,
) -> np.ndarray:
    """Autoregressively predict multiple trajectories in one batched rollout."""

    device = torch.device(device)
    model.eval()
    initial_states = np.asarray(initial_states, dtype=np.float32)
    inputs = np.asarray(inputs, dtype=np.float32)
    if initial_states.ndim != 2 or initial_states.shape[1] != len(STATE_NAMES):
        raise ValueError("initial_states must have shape [batch, state_dim]")
    if inputs.ndim != 3 or inputs.shape[0] != initial_states.shape[0]:
        raise ValueError("inputs must have shape [batch, steps, input_dim]")

    state_mean = torch.as_tensor(normalization.state_mean, dtype=torch.float32, device=device)
    state_std = torch.as_tensor(normalization.state_std, dtype=torch.float32, device=device)
    input_mean = torch.as_tensor(normalization.input_mean, dtype=torch.float32, device=device)
    input_std = torch.as_tensor(normalization.input_std, dtype=torch.float32, device=device)
    inputs_tensor = torch.as_tensor(inputs, dtype=torch.float32, device=device)
    inputs_normalized = (inputs_tensor - input_mean) / input_std
    current = torch.as_tensor(initial_states, dtype=torch.float32, device=device)
    prediction = torch.empty(
        (len(initial_states), inputs.shape[1] + 1, len(STATE_NAMES)),
        dtype=torch.float32,
        device=device,
    )
    prediction[:, 0] = current
    hidden = None
    started_at = time.perf_counter()
    progress_interval = max(1, inputs.shape[1] // 20)
    for index in range(inputs.shape[1]):
        state_n = (current - state_mean) / state_std
        features = torch.cat([state_n, inputs_normalized[:, index]], dim=-1).unsqueeze(1)
        next_n, hidden = model(features, hidden=hidden, return_hidden=True)
        next_state = next_n[:, 0] * state_std + state_mean
        next_state[:, 1] = torch.clamp(next_state[:, 1], min=0.0)
        next_state[:, 2:14] = torch.clamp(next_state[:, 2:14], 0.0, 1.0)
        next_state[:, 14:17] = torch.clamp(next_state[:, 14:17], min=0.0)
        prediction[:, index + 1] = next_state
        current = next_state
        completed = index + 1
        if progress and (completed % progress_interval == 0 or completed == inputs.shape[1]):
            elapsed = time.perf_counter() - started_at
            rate = completed / max(elapsed, 1e-9)
            eta = (inputs.shape[1] - completed) / max(rate, 1e-9)
            percentage = 100.0 * completed / inputs.shape[1]
            print(
                f"[rollout] {completed:>5}/{inputs.shape[1]} steps ({percentage:5.1f}%) "
                f"| elapsed {elapsed:6.1f}s | ETA {eta:6.1f}s",
                flush=True,
            )
    return prediction.cpu().numpy()


@torch.no_grad()
def rollout_trajectory(
    model,
    initial_state: np.ndarray,
    inputs: np.ndarray,
    normalization: Normalization,
    device: str | torch.device = "cpu",
) -> np.ndarray:
    """Autoregressively predict one complete state trajectory."""

    return rollout_batch(
        model,
        np.asarray(initial_state)[None],
        np.asarray(inputs)[None],
        normalization,
        device,
    )[0]


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
    width_multiplier: int = 2,
    head_dim: int | None = None,
    receptor_hidden_dim: int = 32,
    receptor_layers: int = 1,
    dropout: float = 0.1,
    learning_rate: float = 1e-3,
    seed: int = 2026,
    device: str | None = None,
    run_name: str | None = None,
    patience: int | None = None,
    minimum_epochs: int = 1,
    use_amp: bool = True,
    verbose: bool = False,
    train_fraction: float = 1.0,
) -> Dict[str, object]:
    """Train one architecture with validation checkpointing."""

    seed_everything(seed)
    device_obj = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    normalization = Normalization.from_h5(dataset_path)
    train_data = SequenceWindowDataset(
        dataset_path,
        "train",
        normalization,
        sequence_length,
        stride,
        trajectory_fraction=train_fraction,
        selection_seed=seed,
    )
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
        width_multiplier=width_multiplier,
        head_dim=head_dim,
        receptor_hidden_dim=receptor_hidden_dim,
        receptor_layers=receptor_layers,
        dropout=dropout,
    ).to(device_obj)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, epochs), eta_min=learning_rate * 0.05
    )
    amp_enabled = bool(use_amp and device_obj.type == "cuda")
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    except (AttributeError, TypeError):  # PyTorch 2.0--2.2 compatibility.
        scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    best_loss = float("inf")
    best_state = None
    history = []
    stale_epochs = 0
    training_started_at = time.perf_counter()
    for epoch in range(1, epochs + 1):
        epoch_started_at = time.perf_counter()
        model.train()
        train_losses = []
        for features, target in train_loader:
            features, target = features.to(device_obj), target.to(device_obj)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device_obj.type,
                dtype=torch.float16,
                enabled=amp_enabled,
            ):
                loss = _loss(model(features), target)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            train_losses.append(float(loss.detach().cpu()))
        validation_loss = evaluate_loader(model, validation_loader, device_obj)
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(train_losses)),
                "validation_loss": validation_loss,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
            }
        )
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
        scheduler.step()
        if verbose:
            elapsed = time.perf_counter() - training_started_at
            epoch_seconds = time.perf_counter() - epoch_started_at
            eta = (epochs - epoch) * elapsed / epoch
            marker = " *" if validation_loss == best_loss else ""
            print(
                f"[training] epoch {epoch:>3}/{epochs} ({100.0 * epoch / epochs:5.1f}%) "
                f"| train {history[-1]['train_loss']:.6g} "
                f"| val {validation_loss:.6g}{marker} "
                f"| lr {history[-1]['learning_rate']:.2e} "
                f"| epoch {epoch_seconds:5.1f}s | ETA {eta:6.1f}s",
                flush=True,
            )
        if patience is not None and epoch >= minimum_epochs and stale_epochs >= patience:
            if verbose:
                print(
                    f"[training] early stopping after {epoch} epochs; "
                    f"best validation loss {best_loss:.6g}",
                    flush=True,
                )
            break
    if best_state is None:
        raise RuntimeError("training produced no checkpoint")
    model.load_state_dict(best_state)
    metrics = one_step_metrics(model, dataset_path, "test", normalization, device_obj)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_name = run_name or architecture
    checkpoint_path = output_dir / f"{artifact_name}.pt"
    torch.save(
        {
            "architecture": architecture,
            "model_state": best_state,
            "model_kwargs": {
                "hidden_dim": hidden_dim,
                "layers": layers,
                "width_multiplier": width_multiplier,
                "head_dim": head_dim,
                "receptor_hidden_dim": receptor_hidden_dim,
                "receptor_layers": receptor_layers,
                "dropout": dropout,
            },
            "normalization": normalization.to_dict(),
            "state_names": list(STATE_NAMES),
            "input_names": list(INPUT_NAMES),
        },
        checkpoint_path,
    )
    report: Dict[str, object] = {
        "architecture": architecture,
        "run_name": artifact_name,
        "device": str(device_obj),
        "amp": amp_enabled,
        "train_fraction": train_fraction,
        "train_trajectories": len(train_data.selected_trajectories),
        "train_windows": len(train_data),
        "parameters": int(sum(parameter.numel() for parameter in model.parameters())),
        "epochs_trained": len(history),
        "best_validation_loss": best_loss,
        "test": metrics,
        "history": history,
        "checkpoint": str(checkpoint_path),
    }
    (output_dir / f"{artifact_name}.metrics.json").write_text(
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
