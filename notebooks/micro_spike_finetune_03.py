"""Conservative spike fine-tuning from the converged input-only GRU-MSE."""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))
for module_name in tuple(sys.modules):
    if module_name == "hay_single_compartment" or module_name.startswith("hay_single_compartment."):
        del sys.modules[module_name]

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
try:
    from tqdm.auto import tqdm
except ImportError:
    class _PlainProgress:
        def __init__(self, iterable, desc=""):
            self.iterable, self.desc = iterable, desc
        def __iter__(self):
            yield from self.iterable
        def set_postfix(self, **values):
            print(self.desc, values, flush=True)
    def tqdm(iterable, desc=""):
        return _PlainProgress(iterable, desc)

from hay_single_compartment import (
    MICRO_EVENT_NAMES,
    ConservativeSpikeFineTuneLoss,
    InputOnlyGRU,
    StratifiedWindowSampler,
    WaveformConstrainedFineTuneLoss,
    classify_micro_events,
    count_trainable_parameters,
)


@dataclass(frozen=True)
class FineTuneConfig:
    objective: str = "conservative_v1"
    epochs: int = 12
    patience: int = 4
    learning_rate: float = 1e-4
    weight_decay: float = 1e-6
    gradient_clip: float = 0.5
    batch_trajectories: int = 6
    chunk_steps: int = 256
    stratified_windows_per_epoch: int = 24
    stratified_window_steps: int = 256
    context_steps: int = 4000
    curriculum_epochs: int = 5
    seed: int = 20260731


WORKING = Path("/kaggle/working")


def discover(default: Path, filename: str, *, directory: bool = False) -> Path:
    if default.exists():
        return default
    input_root = Path("/kaggle/input")
    if input_root.exists():
        matches = sorted(
            path for path in input_root.rglob(filename)
            if path.is_dir() == directory
        )
        # Kaggle may append a suffix when an uploaded file name collides.
        if not matches and not directory:
            requested = Path(filename)
            pattern = f"{requested.stem}*{requested.suffix}"
            matches = sorted(
                path for path in input_root.rglob(pattern)
                if path.is_file()
            )
        if matches:
            print(f"auto-discovered {filename}: {matches[0]}")
            return matches[0]
    return default


DATASET = discover(
    Path(os.environ.get("HAY_FINETUNE_DATASET", WORKING / "hay_micro_4c_event_enriched_v2.h5")),
    "hay_micro_4c_event_enriched_v2.h5",
)
BASELINE_CHECKPOINT = discover(
    Path(os.environ.get(
        "HAY_FINETUNE_BASELINE",
        WORKING / "hay_micro_event_aware_02/checkpoints/gru_mse.pt",
    )),
    "gru_mse.pt",
)
OUTPUT = Path(os.environ.get("HAY_FINETUNE_OUTPUT", WORKING / "hay_micro_spike_finetune_03"))
CHECKPOINTS = OUTPUT / "checkpoints"
OUTPUT.mkdir(parents=True, exist_ok=True)
CHECKPOINTS.mkdir(exist_ok=True)
CFG = FineTuneConfig(
    objective=os.environ.get("HAY_FINETUNE_OBJECTIVE", "conservative_v1"),
    epochs=int(os.environ.get("HAY_FINETUNE_EPOCHS", "12")),
    patience=int(os.environ.get("HAY_FINETUNE_PATIENCE", "4")),
    learning_rate=float(os.environ.get("HAY_FINETUNE_LEARNING_RATE", "1e-4")),
    batch_trajectories=int(os.environ.get("HAY_FINETUNE_BATCH_TRAJECTORIES", "6")),
    chunk_steps=int(os.environ.get("HAY_FINETUNE_CHUNK_STEPS", "256")),
    stratified_windows_per_epoch=int(os.environ.get("HAY_FINETUNE_WINDOWS_PER_EPOCH", "24")),
    stratified_window_steps=int(os.environ.get("HAY_FINETUNE_WINDOW_STEPS", "256")),
    context_steps=int(os.environ.get("HAY_FINETUNE_CONTEXT_STEPS", "4000")),
    curriculum_epochs=int(os.environ.get("HAY_FINETUNE_CURRICULUM_EPOCHS", "5")),
)
if CFG.objective not in {"conservative_v1", "waveform_constrained_v2"}:
    raise ValueError(f"unknown fine-tuning objective: {CFG.objective}")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if not DATASET.exists():
    raise FileNotFoundError(f"dataset not found: {DATASET}")
if not BASELINE_CHECKPOINT.exists():
    raise FileNotFoundError(f"GRU-MSE checkpoint not found: {BASELINE_CHECKPOINT}")
print("device:", DEVICE)
print("dataset:", DATASET)
print("baseline:", BASELINE_CHECKPOINT)
print("output:", OUTPUT)


def pack_spikes(values: np.ndarray, factor: int) -> np.ndarray:
    usable = values.shape[1] // factor * factor
    return values[:, :usable].reshape(
        values.shape[0], usable // factor, factor * values.shape[2]
    ).astype(np.float32)


baseline_payload = torch.load(BASELINE_CHECKPOINT, map_location="cpu", weights_only=False)
baseline_weights = baseline_payload.get("model_state_dict", baseline_payload.get("model"))
state_names = list(baseline_payload["state_names"])
input_names = list(baseline_payload.get("input_names", []))
state_mean = np.asarray(baseline_payload["state_mean"], dtype=np.float32)
state_std = np.asarray(baseline_payload["state_std"], dtype=np.float32)
temporal_bin = int(baseline_payload.get("training_config", baseline_payload.get("config", {})).get("temporal_bin", 5))
hidden_dim = baseline_weights["recurrent.weight_hh_l0"].shape[1]
input_dim = baseline_weights["input_encoder.0.weight"].shape[1]
state_dim = baseline_weights["decoder.network.3.weight"].shape[0]
decoder_dim = baseline_weights["decoder.network.1.weight"].shape[0]

with h5py.File(DATASET, "r") as handle:
    dataset_config = json.loads(handle.attrs["config_json"])
    raw_dt_ms = float(dataset_config["dt_ms"])
    dataset_model = str(handle.attrs["model"])

    def load_split(name: str):
        burnin = pack_spikes(handle[f"{name}/burnin_inputs"][...], temporal_bin)
        inputs = pack_spikes(handle[f"{name}/inputs"][...], temporal_bin)
        burnin_states = handle[f"{name}/burnin_states"][:, ::temporal_bin].astype(np.float32)
        states = handle[f"{name}/states"][:, ::temporal_bin].astype(np.float32)
        raw_spikes = handle[f"{name}/spikes"][...]
        usable = raw_spikes.shape[1] // temporal_bin * temporal_bin
        spikes = raw_spikes[:, :usable].reshape(raw_spikes.shape[0], -1, temporal_bin).max(-1)
        states = states[:, : inputs.shape[1] + 1]
        burnin_states = burnin_states[:, : burnin.shape[1] + 1]
        events = classify_micro_events(
            states[:, 1:], spikes, state_names, raw_dt_ms * temporal_bin
        )
        return burnin, burnin_states, inputs, states, events

    train_burnin, train_burnin_y, train_x, train_y, train_events = load_split("train")
    val_burnin, val_burnin_y, val_x, val_y, val_events = load_split("validation")
    test_burnin, test_burnin_y, test_x, test_y, test_events = load_split("test")

train_burnin_y_n = (train_burnin_y - state_mean) / state_std
train_y_n = (train_y - state_mean) / state_std
val_y_n = (val_y - state_mean) / state_std
test_y_n = (test_y - state_mean) / state_std
MODEL_DT_MS = raw_dt_ms * temporal_bin
print("train:", train_x.shape, "validation:", val_x.shape, "test:", test_x.shape)
print("model dt:", MODEL_DT_MS, "ms | states:", len(state_names))


def make_model() -> InputOnlyGRU:
    model = InputOnlyGRU(input_dim, state_dim, hidden_dim=hidden_dim, decoder_dim=decoder_dim)
    model.load_state_dict(baseline_weights)
    return model


baseline_model = make_model().to(DEVICE).eval()
for parameter in baseline_model.parameters():
    parameter.requires_grad_(False)
finetune_model = make_model()
MODEL_PARAMETERS = count_trainable_parameters(finetune_model)
print("parameters:", MODEL_PARAMETERS)
print("objective:", CFG.objective)


def tensor(values, indices=None):
    if indices is not None:
        values = values[indices]
    return torch.as_tensor(values, device=DEVICE)


def detach(hidden):
    return None if hidden is None else hidden.detach()


def context_for_window(trajectory: int, start: int) -> np.ndarray:
    if start >= CFG.context_steps:
        return train_x[trajectory, start - CFG.context_steps : start]
    needed_burnin = CFG.context_steps - start
    if needed_burnin > train_burnin.shape[1]:
        raise ValueError("requested context exceeds available spike-only burn-in")
    return np.concatenate((
        train_burnin[trajectory, -needed_burnin:],
        train_x[trajectory, :start],
    ), axis=0)


def event_scale(epoch: int) -> float:
    if CFG.curriculum_epochs <= 1:
        return 1.0
    return float(np.clip((epoch - 1) / (CFG.curriculum_epochs - 1), 0.0, 1.0))


if CFG.objective == "waveform_constrained_v2":
    criterion = WaveformConstrainedFineTuneLoss(
        state_names, state_mean, state_std, event_radius_steps=20
    ).to(DEVICE)
else:
    criterion = ConservativeSpikeFineTuneLoss(state_names, state_mean, state_std).to(DEVICE)


def evaluate_loss(prediction, target, reference=None):
    if CFG.objective == "waveform_constrained_v2":
        return criterion(prediction, target, reference)
    return criterion(prediction, target)


@torch.no_grad()
def validation_metrics(model):
    model.eval()
    criterion.set_event_scale(1.0)
    totals, elements = {}, 0
    soma_squared, soma_elements = 0.0, 0
    subthreshold_squared, subthreshold_elements = 0.0, 0
    truth_above_steps = predicted_above_steps = 0
    truth_crossings = predicted_crossings = 0
    soma_index = state_names.index("soma.v_mV")
    for offset in range(0, len(val_x), CFG.batch_trajectories):
        indices = np.arange(offset, min(offset + CFG.batch_trajectories, len(val_x)))
        burnin, inputs, targets = tensor(val_burnin, indices), tensor(val_x, indices), tensor(val_y_n, indices)
        hidden = reference_hidden = None
        for start in range(0, burnin.shape[1], CFG.chunk_steps):
            burnin_chunk = burnin[:, start : start + CFG.chunk_steps]
            _, hidden = model(burnin_chunk, hidden)
            _, reference_hidden = baseline_model(burnin_chunk, reference_hidden)
            hidden = detach(hidden)
            reference_hidden = detach(reference_hidden)
        for start in range(0, inputs.shape[1], CFG.chunk_steps):
            chunk = inputs[:, start : start + CFG.chunk_steps]
            prediction, hidden = model(chunk, hidden)
            reference, reference_hidden = baseline_model(chunk, reference_hidden)
            target = targets[:, start + 1 : start + 1 + chunk.shape[1]]
            value, terms = evaluate_loss(prediction, target, reference)
            weight = target.shape[0] * target.shape[1]
            totals["selection"] = totals.get("selection", 0.0) + float(value) * weight
            for key, term in terms.items():
                totals[key] = totals.get(key, 0.0) + float(term) * weight
            elements += weight
            pv = prediction[..., soma_index] * float(state_std[soma_index]) + float(state_mean[soma_index])
            tv = target[..., soma_index] * float(state_std[soma_index]) + float(state_mean[soma_index])
            squared = (pv - tv).square()
            soma_squared += float(squared.sum())
            soma_elements += squared.numel()
            subthreshold = tv < -35.0
            subthreshold_squared += float((squared * subthreshold).sum())
            subthreshold_elements += int(subthreshold.sum())
            truth_above_steps += int((tv >= -20.0).sum())
            predicted_above_steps += int((pv >= -20.0).sum())
            truth_crossings += int(((tv[..., :-1] < -20.0) & (tv[..., 1:] >= -20.0)).sum())
            predicted_crossings += int(((pv[..., :-1] < -20.0) & (pv[..., 1:] >= -20.0)).sum())
            hidden = detach(hidden)
            reference_hidden = detach(reference_hidden)
    metrics = {key: value / elements for key, value in totals.items()}
    metrics.update({
        "soma_rmse_mV": float(np.sqrt(soma_squared / max(1, soma_elements))),
        "subthreshold_rmse_mV": float(np.sqrt(subthreshold_squared / max(1, subthreshold_elements))),
        "truth_above_steps": truth_above_steps,
        "predicted_above_steps": predicted_above_steps,
        "truth_crossings": truth_crossings,
        "predicted_crossings": predicted_crossings,
    })
    return metrics


RUN_NAME = (
    "gru_waveform_finetune" if CFG.objective == "waveform_constrained_v2"
    else "gru_spike_finetune"
)
LAST_CHECKPOINT = CHECKPOINTS / f"{RUN_NAME}_last.pt"
BEST_CHECKPOINT = CHECKPOINTS / f"{RUN_NAME}_best.pt"
optimizer = torch.optim.AdamW(
    finetune_model.parameters(), lr=CFG.learning_rate, weight_decay=CFG.weight_decay
)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CFG.epochs)
scaler = torch.amp.GradScaler("cuda", enabled=DEVICE.type == "cuda")
history, start_epoch, best_score, stale = [], 1, float("inf"), 0
finetune_model = finetune_model.to(DEVICE)

baseline_validation = validation_metrics(baseline_model)
print("baseline validation:", json.dumps(baseline_validation, indent=2))
validation_limits = {
    "global": baseline_validation["global"] * 1.05,
    "subthreshold_rmse_mV": baseline_validation["subthreshold_rmse_mV"] * 1.10,
    "predicted_above_steps": max(1, baseline_validation["truth_above_steps"] * 2),
    "predicted_crossings": max(1, int(np.ceil(baseline_validation["truth_crossings"] * 1.5))),
}


def save_best(model, epoch, score, admissible):
    torch.save({
        "format_version": 4,
        "model_name": RUN_NAME,
        "model_state_dict": model.state_dict(),
        "baseline_checkpoint": str(BASELINE_CHECKPOINT),
        "config": asdict(CFG),
        "state_mean": state_mean,
        "state_std": state_std,
        "state_names": state_names,
        "input_names": input_names,
        "best_score": score,
        "epoch": epoch,
        "admissible": admissible,
        "validation_limits": validation_limits,
    }, BEST_CHECKPOINT)


if CFG.objective == "waveform_constrained_v2":
    best_score = baseline_validation["selection"]

resumed = False
if LAST_CHECKPOINT.exists() and os.environ.get("HAY_FINETUNE_FORCE_RESTART", "0") != "1":
    resume = torch.load(LAST_CHECKPOINT, map_location=DEVICE, weights_only=False)
    if resume.get("config") == asdict(CFG):
        finetune_model.load_state_dict(resume["model_state_dict"])
        optimizer.load_state_dict(resume["optimizer_state_dict"])
        scheduler.load_state_dict(resume["scheduler_state_dict"])
        scaler.load_state_dict(resume["scaler_state_dict"])
        history = resume["history"]
        start_epoch = int(resume["epoch"]) + 1
        best_score = float(resume["best_score"])
        stale = int(resume["stale"])
        resumed = True
        print(f"resume fine-tuning from completed epoch {start_epoch - 1}")

if CFG.objective == "waveform_constrained_v2" and not resumed:
    save_best(baseline_model, 0, best_score, True)

def make_sampler(epoch: int) -> StratifiedWindowSampler:
    return StratifiedWindowSampler(
        train_events,
        CFG.stratified_window_steps,
        mixture={
            "subthreshold": 0.35,
            "isolated_spike": 0.20,
            "burst_spike": 0.15,
            "rapid_fire": 0.10,
            "tuft_plateau": 0.10,
            "spike_with_tuft_plateau": 0.10,
        },
        seed=CFG.seed + epoch,
    )
started = time.perf_counter()
epoch_bar = tqdm(range(start_epoch, CFG.epochs + 1), desc="fine-tune GRU spike")
for epoch in epoch_bar:
    finetune_model.train()
    scale = event_scale(epoch)
    criterion.set_event_scale(scale)
    order = np.random.default_rng(CFG.seed + epoch).permutation(len(train_x))
    running, updates = 0.0, 0
    for offset in range(0, len(order), CFG.batch_trajectories):
        indices = order[offset : offset + CFG.batch_trajectories]
        phases = (
            (tensor(train_burnin, indices), tensor(train_burnin_y_n, indices)),
            (tensor(train_x, indices), tensor(train_y_n, indices)),
        )
        hidden = reference_hidden = None
        for phase_x, phase_y in phases:
            for start in range(0, phase_x.shape[1], CFG.chunk_steps):
                chunk = phase_x[:, start : start + CFG.chunk_steps]
                target = phase_y[:, start + 1 : start + 1 + chunk.shape[1]]
                with torch.no_grad():
                    reference, next_reference_hidden = baseline_model(chunk, reference_hidden)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=torch.float16, enabled=DEVICE.type == "cuda"):
                    prediction, next_hidden = finetune_model(chunk, hidden)
                    loss, _ = evaluate_loss(prediction, target, reference)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(finetune_model.parameters(), CFG.gradient_clip)
                scaler.step(optimizer)
                scaler.update()
                hidden = detach(next_hidden)
                reference_hidden = detach(next_reference_hidden)
                running += float(loss.detach())
                updates += 1

    sampled = make_sampler(epoch).sample(CFG.stratified_windows_per_epoch)
    for offset in range(0, len(sampled), CFG.batch_trajectories):
        rows = sampled[offset : offset + CFG.batch_trajectories]
        contexts = np.stack([context_for_window(int(t), int(s)) for t, s in rows])
        windows = np.stack([
            train_x[int(t), int(s) : int(s) + CFG.stratified_window_steps] for t, s in rows
        ])
        targets = np.stack([
            train_y_n[int(t), int(s) + 1 : int(s) + 1 + CFG.stratified_window_steps] for t, s in rows
        ])
        context_t, window_t, target_t = tensor(contexts), tensor(windows), tensor(targets)
        hidden = reference_hidden = None
        with torch.no_grad():
            for start in range(0, context_t.shape[1], CFG.chunk_steps):
                context_chunk = context_t[:, start : start + CFG.chunk_steps]
                _, hidden = finetune_model(context_chunk, hidden)
                _, reference_hidden = baseline_model(context_chunk, reference_hidden)
                hidden = detach(hidden)
                reference_hidden = detach(reference_hidden)
            reference, _ = baseline_model(window_t, reference_hidden)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.float16, enabled=DEVICE.type == "cuda"):
            prediction, _ = finetune_model(window_t, hidden)
            loss, _ = evaluate_loss(prediction, target_t, reference)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(finetune_model.parameters(), CFG.gradient_clip)
        scaler.step(optimizer)
        scaler.update()
        running += float(loss.detach())
        updates += 1

    scheduler.step()
    validation = validation_metrics(finetune_model)
    score = validation["selection"]
    admissible = True
    if CFG.objective == "waveform_constrained_v2":
        admissible = (
            validation["global"] <= validation_limits["global"]
            and validation["subthreshold_rmse_mV"] <= validation_limits["subthreshold_rmse_mV"]
            and validation["predicted_above_steps"] <= validation_limits["predicted_above_steps"]
            and validation["predicted_crossings"] <= validation_limits["predicted_crossings"]
        )
    elapsed = time.perf_counter() - started
    eta = elapsed / max(1, epoch - start_epoch + 1) * (CFG.epochs - epoch)
    row = {
        "epoch": epoch,
        "event_scale": scale,
        "train_loss": running / updates,
        "checkpoint_admissible": admissible,
        **{f"validation_{key}": value for key, value in validation.items()},
        "elapsed_s": elapsed,
        "eta_s": eta,
    }
    history.append(row)
    if admissible and score < best_score:
        best_score, stale = score, 0
        save_best(finetune_model, epoch, best_score, admissible)
    else:
        stale += 1
    torch.save({
        "format_version": 3,
        "model_state_dict": finetune_model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "config": asdict(CFG),
        "history": history,
        "epoch": epoch,
        "best_score": best_score,
        "stale": stale,
    }, LAST_CHECKPOINT)
    pd.DataFrame(history).to_csv(OUTPUT / "finetune_history.csv", index=False)
    epoch_bar.set_postfix(
        train=f"{row['train_loss']:.3e}",
        val_global=f"{validation['global']:.3e}",
        val_event=f"{validation.get('event', validation.get('rare_event', 0.0)):.3e}",
        admissible=admissible,
        scale=f"{scale:.2f}",
        eta=f"{eta / 60:.1f}m",
    )
    if stale >= CFG.patience and epoch >= CFG.curriculum_epochs:
        print(f"early stop at epoch {epoch}; best selection={best_score:.6f}")
        break

best_payload = torch.load(BEST_CHECKPOINT, map_location=DEVICE, weights_only=False)
finetune_model.load_state_dict(best_payload["model_state_dict"])
baseline_model = baseline_model.to(DEVICE).eval()
finetune_model.eval()


@torch.no_grad()
def predict(model):
    predictions = []
    for trajectory in range(len(test_x)):
        burnin, inputs = tensor(test_burnin[trajectory : trajectory + 1]), tensor(test_x[trajectory : trajectory + 1])
        hidden = None
        for start in range(0, burnin.shape[1], CFG.chunk_steps):
            _, hidden = model(burnin[:, start : start + CFG.chunk_steps], hidden)
        chunks = []
        for start in range(0, inputs.shape[1], CFG.chunk_steps):
            output, hidden = model(inputs[:, start : start + CFG.chunk_steps], hidden)
            chunks.append(output)
        normalized = torch.cat(chunks, dim=1).squeeze(0).cpu().numpy()
        predictions.append(normalized * state_std + state_mean)
    return np.stack(predictions)


def match_spikes(truth_voltage, prediction_voltage, threshold=-20.0, tolerance_steps=4):
    truth = (truth_voltage[:, :-1] < threshold) & (truth_voltage[:, 1:] >= threshold)
    predicted = (prediction_voltage[:, :-1] < threshold) & (prediction_voltage[:, 1:] >= threshold)
    matched = 0
    timing_errors = []
    for trajectory in range(len(truth)):
        actual_indices = list(np.flatnonzero(truth[trajectory]))
        used = set()
        for candidate in np.flatnonzero(predicted[trajectory]):
            choices = [
                (abs(int(candidate) - int(actual)), index, int(candidate) - int(actual))
                for index, actual in enumerate(actual_indices)
                if index not in used and abs(int(candidate) - int(actual)) <= tolerance_steps
            ]
            if choices:
                _, index, error = min(choices)
                used.add(index)
                timing_errors.append(error * MODEL_DT_MS)
                matched += 1
    truth_count, predicted_count = int(truth.sum()), int(predicted.sum())
    return {
        "truth_spikes": truth_count,
        "predicted_spikes": predicted_count,
        "matched_spikes_2ms": matched,
        "spike_precision_2ms": matched / max(1, predicted_count),
        "spike_recall_2ms": matched / max(1, truth_count),
        "spike_f1_2ms": 2 * matched / max(1, truth_count + predicted_count),
        "false_spikes_per_second": (predicted_count - matched) / (len(truth) * truth.shape[1] * MODEL_DT_MS / 1000.0),
        "mean_timing_error_ms": float(np.mean(np.abs(timing_errors))) if timing_errors else np.nan,
    }


truth = test_y[:, 1:]
predictions = {
    "gru_mse": predict(baseline_model),
    RUN_NAME: predict(finetune_model),
}
rows = []
soma_index = state_names.index("soma.v_mV")
for name, prediction in predictions.items():
    error = prediction - truth
    row = {
        "model": name,
        "parameters": MODEL_PARAMETERS,
        "test_mean_normalized_rmse": float(np.mean(np.sqrt(np.mean(np.square(error), axis=(0, 1))) / state_std)),
        "test_soma_rmse_mV": float(np.sqrt(np.mean(np.square(error[..., soma_index])))),
        "time_above_threshold_s": float(
            np.sum(prediction[..., soma_index] >= -20.0) * MODEL_DT_MS / 1000.0
        ),
        "selected_epoch": 0 if name == "gru_mse" else int(best_payload.get("epoch", -1)),
        **match_spikes(truth[..., soma_index], prediction[..., soma_index]),
    }
    for event_index, event_name in enumerate(MICRO_EVENT_NAMES):
        mask = test_events[..., event_index].astype(bool)
        row[f"soma_rmse_{event_name}_mV"] = (
            float(np.sqrt(np.mean(np.square(error[..., soma_index][mask])))) if mask.any() else np.nan
        )
    rows.append(row)
    np.savez_compressed(
        OUTPUT / f"{name}_test_predictions.npz",
        prediction=prediction,
        truth=truth,
        events=test_events,
    )
comparison = pd.DataFrame(rows)
comparison.to_csv(OUTPUT / "comparison.csv", index=False)
print(comparison.T)

trajectory = int(np.argmax(test_events[..., 2:5].sum(axis=(1, 2))))
steps = min(test_x.shape[1], int(round(1000.0 / MODEL_DT_MS)))
time_ms = np.arange(steps) * MODEL_DT_MS
figure, axis = plt.subplots(figsize=(16, 5))
axis.plot(time_ms, truth[trajectory, :steps, soma_index], label="teacher", linewidth=1.4)
axis.plot(time_ms, predictions["gru_mse"][trajectory, :steps, soma_index], label="GRU-MSE", alpha=0.85)
axis.plot(time_ms, predictions[RUN_NAME][trajectory, :steps, soma_index], label=RUN_NAME, alpha=0.85)
axis.axhline(-20.0, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
axis.set(xlabel="time (ms)", ylabel="soma V (mV)", title="Natural held-out test rollout")
axis.legend()
axis.grid(alpha=0.2)
figure.tight_layout()
figure.savefig(OUTPUT / "soma_comparison.png", dpi=170)
plt.close(figure)

loss_description = (
    {
        "global_mse_always_active": True,
        "curriculum": "linear 0->1",
        "rare_terms": [
            "symmetric event-state regression",
            "symmetric soma waveform regression",
            "Sobolev first-derivative matching",
            "symmetric soft threshold occupancy",
            "frozen-baseline functional distillation outside spike windows",
        ],
        "excluded_shortcuts": ["positive-class-weighted BCE", "one-sided peak deficit"],
        "checkpoint_constraints": validation_limits,
    }
    if CFG.objective == "waveform_constrained_v2"
    else {
        "global_mse_always_active": True,
        "curriculum": "linear 0->1",
        "rare_terms": ["asymmetric peak deficit", "event derivative", "rapid gates", "balanced spike logits"],
    }
)
(OUTPUT / "experiment.json").write_text(json.dumps({
    "contract": f"GRU-MSE checkpoint -> {CFG.objective} fine-tuning",
    "dataset": str(DATASET),
    "dataset_model": dataset_model,
    "baseline_checkpoint": str(BASELINE_CHECKPOINT),
    "model_dt_ms": MODEL_DT_MS,
    "config": asdict(CFG),
    "baseline_validation": baseline_validation,
    "selected_epoch": int(best_payload.get("epoch", -1)),
    "loss": loss_description,
}, indent=2), encoding="utf-8")
print("saved:", OUTPUT)
