"""Kaggle companion script: inspect a trained input-only GRU checkpoint."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# `%run` on Kaggle can execute before an editable install becomes visible to
# the active kernel.  Prefer the checked-out source tree deterministically.
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from hay_single_compartment import (
    MICRO_EVENT_NAMES,
    InputOnlyGRU,
    classify_micro_events,
    replay_gru_gates,
)


ROOT = Path(os.environ.get("HAY_GRU_ROOT", "/kaggle/working"))
DATASET = Path(os.environ.get("HAY_GRU_DATASET", ROOT / "hay_micro_4c_v1_2.h5"))
CHECKPOINT = Path(os.environ.get("HAY_GRU_CHECKPOINT", ROOT / "hay_micro_input_only_01/checkpoints/gru.pt"))
OUTPUT = Path(os.environ.get("HAY_GRU_DIAGNOSTICS", ROOT / "hay_micro_gru_diagnostics_01"))
TEMPORAL_BIN = int(os.environ.get("HAY_TEMPORAL_BIN", "5"))
MAX_TRAJECTORIES = int(os.environ.get("HAY_DIAGNOSTIC_TRAJECTORIES", "6"))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT.mkdir(parents=True, exist_ok=True)
print("dataset:", DATASET)
print("checkpoint:", CHECKPOINT)
print("device:", DEVICE)


def pack_spikes(values: np.ndarray, factor: int) -> np.ndarray:
    usable = values.shape[1] // factor * factor
    return values[:, :usable].reshape(values.shape[0], usable // factor, factor * values.shape[2]).astype(np.float32)


checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
weights = checkpoint.get("model_state_dict", checkpoint.get("model"))
state_names = list(checkpoint["state_names"])
input_names = list(checkpoint["input_names"])
hidden_dim = weights["recurrent.weight_hh_l0"].shape[1]
input_dim = weights["input_encoder.0.weight"].shape[1]
state_dim = weights["decoder.network.3.weight"].shape[0]
decoder_dim = weights["decoder.network.1.weight"].shape[0]
model = InputOnlyGRU(input_dim, state_dim, hidden_dim=hidden_dim, decoder_dim=decoder_dim).to(DEVICE)
model.load_state_dict(weights)
model.eval()
state_mean = np.asarray(checkpoint["state_mean"], dtype=np.float32)
state_std = np.asarray(checkpoint["state_std"], dtype=np.float32)

with h5py.File(DATASET, "r") as handle:
    if "test/burnin_inputs" not in handle:
        raise RuntimeError("Dataset schema 1.2 with spike-only burn-in is required")
    raw_dt_ms = float(json.loads(handle.attrs["config_json"])["dt_ms"])
    count = min(MAX_TRAJECTORIES, handle["test/inputs"].shape[0])
    burnin = pack_spikes(handle["test/burnin_inputs"][:count], TEMPORAL_BIN)
    inputs = pack_spikes(handle["test/inputs"][:count], TEMPORAL_BIN)
    states = handle["test/states"][:count, ::TEMPORAL_BIN].astype(np.float32)
    raw_spikes = handle["test/spikes"][:count]
    usable = raw_spikes.shape[1] // TEMPORAL_BIN * TEMPORAL_BIN
    spikes = raw_spikes[:, :usable].reshape(count, -1, TEMPORAL_BIN).max(-1)
states = states[:, : inputs.shape[1] + 1]
event_labels = classify_micro_events(states[:, 1:], spikes, state_names, raw_dt_ms * TEMPORAL_BIN)


all_records = {name: [] for name in ("encoded", "reset", "update", "candidate", "hidden", "decoder")}
max_replay_error = 0.0
with torch.no_grad():
    for trajectory in range(count):
        burn = torch.as_tensor(burnin[trajectory : trajectory + 1], device=DEVICE)
        sequence = torch.as_tensor(inputs[trajectory : trajectory + 1], device=DEVICE)
        _, hidden = model(burn)
        prediction, _ = model(sequence, hidden)
        replay = replay_gru_gates(model, sequence, hidden)
        fused, _ = model.recurrent(model.input_encoder(sequence), hidden)
        max_replay_error = max(max_replay_error, float((fused - replay["hidden"]).abs().max()))
        encoded = model.input_encoder(sequence)
        decoder_hidden = model.decoder.network[2](model.decoder.network[1](model.decoder.network[0](fused)))
        for name, value in (("encoded", encoded), ("decoder", decoder_hidden), *replay.items()):
            all_records[name].append(value.squeeze(0).cpu().numpy())
records = {name: np.stack(values) for name, values in all_records.items()}
if max_replay_error > 5e-5:
    raise RuntimeError(f"GRU gate replay mismatch: {max_replay_error:.3e}")
print(f"gate replay verified, max |difference|={max_replay_error:.3e}")


rows = []
for event_index, event_name in enumerate(MICRO_EVENT_NAMES):
    mask = event_labels[..., event_index].astype(bool)
    if not mask.any():
        continue
    for activation in ("encoded", "reset", "update", "candidate", "hidden", "decoder"):
        values = records[activation][mask]
        rows.append({
            "event": event_name,
            "activation": activation,
            "samples": int(values.shape[0]),
            "mean": float(values.mean()),
            "std": float(values.std()),
            "absolute_mean": float(np.abs(values).mean()),
            "near_zero_fraction": float((np.abs(values) < 0.05).mean()),
            "low_saturation_fraction": float((values < 0.05).mean()) if activation in {"reset", "update"} else np.nan,
            "high_saturation_fraction": float((values > 0.95).mean()) if activation in {"reset", "update"} else np.nan,
        })
summary = pd.DataFrame(rows)
summary.to_csv(OUTPUT / "activation_by_event.csv", index=False)

flat_hidden = records["hidden"].reshape(-1, hidden_dim)
rng = np.random.default_rng(7)
sample = flat_hidden[rng.choice(len(flat_hidden), min(20000, len(flat_hidden)), replace=False)]
singular = np.linalg.svd(sample - sample.mean(0), compute_uv=False)
variance = np.square(singular)
cumulative = np.cumsum(variance) / variance.sum()
rank_90 = int(np.searchsorted(cumulative, 0.90) + 1)
rank_99 = int(np.searchsorted(cumulative, 0.99) + 1)

figure, axes = plt.subplots(2, 3, figsize=(16, 8))
for axis, name in zip(axes.flat, ("encoded", "reset", "update", "candidate", "hidden", "decoder")):
    values = records[name].reshape(-1)
    if len(values) > 500_000:
        values = rng.choice(values, 500_000, replace=False)
    axis.hist(values, bins=80, density=True, alpha=0.85)
    axis.set_title(name)
    axis.grid(alpha=0.2)
figure.tight_layout()
figure.savefig(OUTPUT / "activation_distributions.png", dpi=160)
plt.close(figure)

gate_event = summary[summary.activation.isin(["reset", "update"])].copy()
figure, axes = plt.subplots(1, 2, figsize=(15, 5), sharey=True)
for axis, gate in zip(axes, ("reset", "update")):
    table = gate_event[gate_event.activation == gate].set_index("event")
    table[["low_saturation_fraction", "high_saturation_fraction"]].plot.bar(ax=axis)
    axis.set_title(f"{gate} gate saturation by event")
    axis.set_ylabel("fraction of units")
    axis.grid(axis="y", alpha=0.2)
figure.tight_layout()
figure.savefig(OUTPUT / "gate_saturation_by_event.png", dpi=160)
plt.close(figure)

report = {
    "dataset": str(DATASET),
    "checkpoint": str(CHECKPOINT),
    "trajectories": count,
    "model_dt_ms": raw_dt_ms * TEMPORAL_BIN,
    "hidden_dim": hidden_dim,
    "gate_replay_max_abs_error": max_replay_error,
    "hidden_effective_rank_90_percent": rank_90,
    "hidden_effective_rank_99_percent": rank_99,
    "event_samples": {name: int(event_labels[..., i].sum()) for i, name in enumerate(MICRO_EVENT_NAMES)},
}
(OUTPUT / "diagnostic_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
np.savez_compressed(
    OUTPUT / "activation_sample.npz",
    hidden=sample.astype(np.float32),
    singular_values=singular.astype(np.float32),
)
print(json.dumps(report, indent=2))
print("saved diagnostics to", OUTPUT)
