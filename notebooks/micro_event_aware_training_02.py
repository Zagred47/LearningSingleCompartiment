"""Kaggle companion script for event-aware input-only micro-Hay training."""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

# Make `%run` robust to Kaggle kernels that do not immediately observe an
# editable install performed in a previous cell.
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))
# `%run` executes in the existing Kaggle kernel.  After a git pull, Python can
# otherwise keep an older loss/model implementation alive in sys.modules even
# though tracebacks display the newly pulled source lines.
for module_name in tuple(sys.modules):
    if module_name == "hay_single_compartment" or module_name.startswith("hay_single_compartment."):
        del sys.modules[module_name]

import h5py
import numpy as np
import pandas as pd
import torch
from torch import nn
try:
    from tqdm.auto import tqdm
except ImportError:  # The Kaggle runtime provides tqdm; keep the script portable.
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
    MICRO_STATE_NAMES,
    EventAwareStateLoss,
    InputOnlyBranchELM,
    InputOnlyConvGRU,
    InputOnlyConvLSTM,
    InputOnlyGRU,
    MicroDatasetConfig,
    MicroDriveConfig,
    StratifiedWindowSampler,
    classify_micro_events,
    count_trainable_parameters,
    event_catalog,
    generate_micro_dataset,
    micro_input_names,
    move_hidden,
)


@dataclass(frozen=True)
class ExperimentConfig:
    temporal_bin: int = 5
    batch_trajectories: int = 6
    chunk_steps: int = 256
    epochs: int = 30
    patience: int = 7
    learning_rate: float = 5e-4
    weight_decay: float = 1e-5
    gradient_clip: float = 1.0
    event_replays: int = 2
    stratified_windows_per_epoch: int = 48
    stratified_window_steps: int = 256
    context_steps: int = 4000
    seed: int = 20260730


ROOT = Path(os.environ.get("HAY_EVENT_ROOT", "/kaggle/working"))
DATASET = Path(os.environ.get("HAY_EVENT_DATASET", ROOT / "hay_micro_4c_event_enriched_v2.h5"))
OUTPUT = Path(os.environ.get("HAY_EVENT_OUTPUT", ROOT / "hay_micro_event_aware_02"))
CHECKPOINTS = OUTPUT / "checkpoints"
OUTPUT.mkdir(parents=True, exist_ok=True)
CHECKPOINTS.mkdir(exist_ok=True)
CFG = ExperimentConfig(
    epochs=int(os.environ.get("HAY_EVENT_EPOCHS", "30")),
    event_replays=int(os.environ.get("HAY_EVENT_REPLAYS", "0")),
    stratified_windows_per_epoch=int(os.environ.get("HAY_EVENT_WINDOWS_PER_EPOCH", "48")),
)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(CFG.seed)
np.random.seed(CFG.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(CFG.seed)
print("device:", DEVICE, "| dataset:", DATASET, "| output:", OUTPUT)


# More trajectories create a richer response catalogue; the drive itself remains
# the declared physiological synaptic drive.  Selection happens on teacher output.
dataset_config = MicroDatasetConfig(
    duration_ms=float(os.environ.get("HAY_EVENT_DURATION_MS", "5000")),
    warmup_ms=float(os.environ.get("HAY_EVENT_WARMUP_MS", "2000")),
    train_trajectories=int(os.environ.get("HAY_EVENT_TRAIN_TRAJECTORIES", "48")),
    validation_trajectories=int(os.environ.get("HAY_EVENT_VALIDATION_TRAJECTORIES", "8")),
    test_trajectories=int(os.environ.get("HAY_EVENT_TEST_TRAJECTORIES", "12")),
    drive=MicroDriveConfig(),
)
dataset_report = generate_micro_dataset(
    DATASET,
    dataset_config,
    progress=True,
    reuse=True,
    workers=int(os.environ.get("HAY_EVENT_DATA_WORKERS", "4")),
)
print(json.dumps(dataset_report, indent=2))


def pack_spikes(values: np.ndarray, factor: int) -> np.ndarray:
    usable = values.shape[1] // factor * factor
    return values[:, :usable].reshape(values.shape[0], usable // factor, factor * values.shape[2]).astype(np.float32)


def load_split(name: str):
    with h5py.File(DATASET, "r") as handle:
        raw_dt = float(json.loads(handle.attrs["config_json"])["dt_ms"])
        burnin = pack_spikes(handle[f"{name}/burnin_inputs"][...], CFG.temporal_bin)
        inputs = pack_spikes(handle[f"{name}/inputs"][...], CFG.temporal_bin)
        burnin_states = handle[f"{name}/burnin_states"][:, :: CFG.temporal_bin].astype(np.float32)
        states = handle[f"{name}/states"][:, :: CFG.temporal_bin].astype(np.float32)
        raw_states = handle[f"{name}/states"][...].astype(np.float32)
        raw_spikes = handle[f"{name}/spikes"][...]
    states = states[:, : inputs.shape[1] + 1]
    burnin_states = burnin_states[:, : burnin.shape[1] + 1]
    raw_labels = classify_micro_events(raw_states[:, :-1], raw_spikes, MICRO_STATE_NAMES, raw_dt)
    usable = raw_labels.shape[1] // CFG.temporal_bin * CFG.temporal_bin
    labels = raw_labels[:, :usable].reshape(raw_labels.shape[0], -1, CFG.temporal_bin, len(MICRO_EVENT_NAMES)).max(2)
    return burnin, burnin_states, inputs, states, labels


train_burnin, train_burnin_y, train_x, train_y, train_events = load_split("train")
val_burnin, val_burnin_y, val_x, val_y, val_events = load_split("validation")
test_burnin, test_burnin_y, test_x, test_y, test_events = load_split("test")
normalization_source = np.concatenate((train_burnin_y, train_y[:, 1:]), axis=1)
state_mean = normalization_source.reshape(-1, len(MICRO_STATE_NAMES)).mean(0).astype(np.float32)
state_std = np.maximum(normalization_source.reshape(-1, len(MICRO_STATE_NAMES)).std(0), 1e-6).astype(np.float32)
train_burnin_y_n = (train_burnin_y - state_mean) / state_std
train_y_n = (train_y - state_mean) / state_std
val_y_n = (val_y - state_mean) / state_std
test_y_n = (test_y - state_mean) / state_std
catalog = {
    split: event_catalog(labels)
    for split, labels in (("train", train_events), ("validation", val_events), ("test", test_events))
}
(OUTPUT / "event_catalog.json").write_text(json.dumps(catalog, indent=2), encoding="utf-8")
print(pd.DataFrame(catalog).T)


INPUT_DIM, STATE_DIM = train_x.shape[-1], train_y.shape[-1]
TARGET_PARAMETERS = 318_261


def nearest_model(builder, candidates):
    models = [builder(value) for value in candidates]
    return min(models, key=lambda model: abs(count_trainable_parameters(model) - TARGET_PARAMETERS))


model_builders = {
    "gru_mse": lambda: InputOnlyGRU(INPUT_DIM, STATE_DIM, hidden_dim=200, decoder_dim=200),
    "gru_event": lambda: InputOnlyGRU(INPUT_DIM, STATE_DIM, hidden_dim=200, decoder_dim=200),
    "branch_elm": lambda: nearest_model(
        lambda memory: InputOnlyBranchELM(INPUT_DIM, STATE_DIM, num_branch=24, num_memory=memory, model_dt_ms=0.5),
        range(64, 257, 8),
    ),
    "conv_gru": lambda: nearest_model(
        lambda hidden: InputOnlyConvGRU(INPUT_DIM, STATE_DIM, hidden_dim=hidden, conv_channels=96, decoder_dim=hidden),
        range(96, 225, 8),
    ),
    "conv_lstm": lambda: nearest_model(
        lambda hidden: InputOnlyConvLSTM(INPUT_DIM, STATE_DIM, hidden_dim=hidden, conv_channels=96, decoder_dim=hidden),
        range(80, 209, 8),
    ),
}
requested = tuple(name.strip() for name in os.environ.get(
    "HAY_EVENT_MODELS", "gru_mse,gru_event,branch_elm,conv_gru,conv_lstm"
).split(",") if name.strip())
unknown = set(requested) - set(model_builders)
if unknown:
    raise ValueError(f"unknown models: {sorted(unknown)}")
REUSE_COMPLETED_MODELS = os.environ.get("HAY_EVENT_REUSE_MODELS", "1") == "1"


def tensor(values, indices=None):
    if indices is not None:
        values = values[indices]
    return torch.as_tensor(values, device=DEVICE)


def detach(hidden):
    if hidden is None:
        return None
    if isinstance(hidden, tuple):
        return tuple(detach(value) for value in hidden)
    return hidden.detach()


def make_loss(event_aware: bool):
    if event_aware:
        return EventAwareStateLoss(MICRO_STATE_NAMES, state_mean, state_std).to(DEVICE)
    return None


def model_spec(model):
    state_dim = model.decoder.network[-1].out_features
    decoder_dim = model.decoder.network[1].out_features
    if isinstance(model, InputOnlyGRU):
        return {
            "class": "InputOnlyGRU", "input_dim": INPUT_DIM, "state_dim": state_dim,
            "hidden_dim": model.hidden_dim, "layers": model.layers, "decoder_dim": decoder_dim,
        }
    if isinstance(model, InputOnlyBranchELM):
        return {
            "class": "InputOnlyBranchELM", "input_dim": model.input_dim, "state_dim": state_dim,
            "num_branch": model.num_branch, "num_memory": model.num_memory,
            "mlp_hidden_dim": model.update[0].out_features, "decoder_dim": decoder_dim,
        }
    if isinstance(model, (InputOnlyConvGRU, InputOnlyConvLSTM)):
        return {
            "class": type(model).__name__, "input_dim": INPUT_DIM, "state_dim": state_dim,
            "hidden_dim": model.hidden_dim, "conv_channels": model.frontend[0].out_channels,
            "dilations": list(model.dilations), "kernel_size": model.kernel_size,
            "decoder_dim": decoder_dim, "receptive_field": model.receptive_field,
        }
    raise TypeError(f"unsupported checkpoint model {type(model).__name__}")


def loss_value(criterion, prediction, target):
    if criterion is None:
        value = torch.mean(torch.square(prediction - target))
        return value, {"global": value}
    return criterion(prediction, target)


@torch.no_grad()
def evaluate_loss(model, criterion, burnin_np, x_np, y_np):
    model.eval()
    totals, elements = {}, 0
    for offset in range(0, len(x_np), CFG.batch_trajectories):
        indices = np.arange(offset, min(offset + CFG.batch_trajectories, len(x_np)))
        burnin, x, y = tensor(burnin_np, indices), tensor(x_np, indices), tensor(y_np, indices)
        hidden = None
        for start in range(0, burnin.shape[1], CFG.chunk_steps):
            _, hidden = model(burnin[:, start : start + CFG.chunk_steps], hidden)
            hidden = detach(hidden)
        for start in range(0, x.shape[1], CFG.chunk_steps):
            chunk = x[:, start : start + CFG.chunk_steps]
            prediction, hidden = model(chunk, hidden)
            target = y[:, start + 1 : start + 1 + chunk.shape[1]]
            value, terms = loss_value(criterion, prediction, target)
            weight = target.shape[0] * target.shape[1]
            totals["selection"] = totals.get("selection", 0.0) + float(value) * weight
            for key, term in terms.items():
                totals[key] = totals.get(key, 0.0) + float(term) * weight
            elements += weight
            hidden = detach(hidden)
    return {key: value / elements for key, value in totals.items()}


def train_model(name, model):
    event_aware = name != "gru_mse"
    model = model.to(DEVICE)
    criterion = make_loss(event_aware)
    optimizer = torch.optim.AdamW(model.parameters(), lr=CFG.learning_rate, weight_decay=CFG.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CFG.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=DEVICE.type == "cuda")
    history, best, stale = [], float("inf"), 0
    started = time.perf_counter()
    sampler = StratifiedWindowSampler(
        train_events,
        CFG.stratified_window_steps,
        seed=CFG.seed,
    )
    complete_inputs = np.concatenate((train_burnin, train_x), axis=1)
    epoch_bar = tqdm(range(1, CFG.epochs + 1), desc=f"train {name}")
    for epoch in epoch_bar:
        model.train()
        order = np.random.permutation(len(train_x))
        running, updates = 0.0, 0
        for offset in range(0, len(order), CFG.batch_trajectories):
            indices = order[offset : offset + CFG.batch_trajectories]
            burnin, burnin_y = tensor(train_burnin, indices), tensor(train_burnin_y_n, indices)
            x, y = tensor(train_x, indices), tensor(train_y_n, indices)
            event = train_events[indices]
            hidden = None
            phases = ((burnin, burnin_y, None), (x, y, event))
            for phase_x, phase_y, phase_events in phases:
                for start in range(0, phase_x.shape[1], CFG.chunk_steps):
                    chunk = phase_x[:, start : start + CFG.chunk_steps]
                    target = phase_y[:, start + 1 : start + 1 + chunk.shape[1]]
                    hidden_before = detach(hidden)
                    repeats = 1
                    if phase_events is not None:
                        event_chunk = phase_events[:, start : start + chunk.shape[1], 2:7]
                        if event_chunk.any():
                            repeats += CFG.event_replays
                    for _ in range(repeats):
                        optimizer.zero_grad(set_to_none=True)
                        with torch.autocast("cuda", dtype=torch.float16, enabled=DEVICE.type == "cuda"):
                            prediction, candidate_hidden = model(chunk, hidden_before)
                            loss, _ = loss_value(criterion, prediction, target)
                        scaler.scale(loss).backward()
                        scaler.unscale_(optimizer)
                        nn.utils.clip_grad_norm_(model.parameters(), CFG.gradient_clip)
                        scaler.step(optimizer)
                        scaler.update()
                        running += float(loss.detach())
                        updates += 1
                    with torch.no_grad():
                        _, hidden = model(chunk, hidden_before)
                    hidden = detach(hidden)
        # A second, explicitly stratified view.  Every selected window receives
        # a spike-only context, so no teacher state is needed to initialize it.
        sampled_windows = sampler.sample(CFG.stratified_windows_per_epoch)
        for window_offset in range(0, len(sampled_windows), CFG.batch_trajectories):
            rows = sampled_windows[window_offset : window_offset + CFG.batch_trajectories]
            contexts, windows, targets = [], [], []
            for trajectory, start in rows:
                absolute_start = train_burnin.shape[1] + int(start)
                context_start = max(0, absolute_start - CFG.context_steps)
                context = complete_inputs[int(trajectory), context_start:absolute_start]
                if len(context) < CFG.context_steps:
                    context = np.pad(context, ((CFG.context_steps - len(context), 0), (0, 0)))
                contexts.append(context)
                windows.append(train_x[int(trajectory), start : start + CFG.stratified_window_steps])
                targets.append(train_y_n[int(trajectory), start + 1 : start + 1 + CFG.stratified_window_steps])
            context_t = tensor(np.stack(contexts))
            window_t = tensor(np.stack(windows))
            target_t = tensor(np.stack(targets))
            hidden = None
            with torch.no_grad():
                for start in range(0, context_t.shape[1], CFG.chunk_steps):
                    _, hidden = model(context_t[:, start : start + CFG.chunk_steps], hidden)
                    hidden = detach(hidden)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16, enabled=DEVICE.type == "cuda"):
                prediction, _ = model(window_t, hidden)
                loss, _ = loss_value(criterion, prediction, target_t)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), CFG.gradient_clip)
            scaler.step(optimizer)
            scaler.update()
            running += float(loss.detach())
            updates += 1
        scheduler.step()
        validation = evaluate_loss(model, criterion, val_burnin, val_x, val_y_n)
        row = {
            "epoch": epoch,
            "train_loss": running / updates,
            **{f"validation_{key}": value for key, value in validation.items()},
            "elapsed_s": time.perf_counter() - started,
        }
        history.append(row)
        score = validation["selection"]
        epoch_bar.set_postfix(train=f"{row['train_loss']:.3e}", val=f"{score:.3e}")
        if score < best:
            best, stale = score, 0
            payload = {
                "format_version": 2,
                "model_name": name,
                "architecture": model.architecture,
                "model_state_dict": model.state_dict(),
                "model_spec": {**model_spec(model), "parameters": count_trainable_parameters(model)},
                "state_mean": state_mean,
                "state_std": state_std,
                "state_names": list(MICRO_STATE_NAMES),
                "input_names": list(micro_input_names(dataset_config)),
                "input_dim": INPUT_DIM,
                "training_config": asdict(CFG),
                "dataset_report": dataset_report,
                "event_catalog": catalog,
            }
            torch.save(payload, CHECKPOINTS / f"{name}.pt")
        else:
            stale += 1
            if stale >= CFG.patience:
                print(f"{name}: early stop at epoch {epoch}")
                break
    checkpoint = torch.load(CHECKPOINTS / f"{name}.pt", map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    pd.DataFrame(history).to_csv(OUTPUT / f"{name}_history.csv", index=False)
    return model, best, history


def load_completed_model(name, model):
    checkpoint_path = CHECKPOINTS / f"{name}.pt"
    history_path = OUTPUT / f"{name}_history.csv"
    if not REUSE_COMPLETED_MODELS or not checkpoint_path.exists() or not history_path.exists():
        return None
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    same_dataset = checkpoint.get("dataset_report", {}).get("sha256") == dataset_report.get("sha256")
    same_training = checkpoint.get("training_config") == asdict(CFG)
    if not (same_dataset and same_training):
        print(f"{name}: checkpoint presente ma incompatibile; training da zero")
        return None
    model = model.to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    history_frame = pd.read_csv(history_path)
    history = history_frame.to_dict("records")
    best = float(history_frame["validation_selection"].min())
    print(f"{name}: checkpoint completo riutilizzato ({len(history)} epoche), nessun retraining")
    return model, best, history


@torch.no_grad()
def predict(model):
    model.eval()
    predictions = []
    for trajectory in range(len(test_x)):
        burnin = tensor(test_burnin[trajectory : trajectory + 1])
        inputs = tensor(test_x[trajectory : trajectory + 1])
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


def spike_metrics(truth_voltage, prediction_voltage, threshold=-20.0, tolerance_steps=4):
    truth = (truth_voltage[:, :-1] < threshold) & (truth_voltage[:, 1:] >= threshold)
    prediction = (prediction_voltage[:, :-1] < threshold) & (prediction_voltage[:, 1:] >= threshold)
    exact_tp = int((truth & prediction).sum())
    matched = 0
    for trajectory in range(len(truth)):
        truth_indices = list(np.flatnonzero(truth[trajectory]))
        prediction_indices = list(np.flatnonzero(prediction[trajectory]))
        used = set()
        for predicted in prediction_indices:
            candidates = [
                (abs(predicted - actual), index)
                for index, actual in enumerate(truth_indices)
                if index not in used and abs(predicted - actual) <= tolerance_steps
            ]
            if candidates:
                _, index = min(candidates)
                used.add(index)
                matched += 1
    fp = int(prediction.sum()) - matched
    fn = int(truth.sum()) - matched
    return {
        "truth_spikes": int(truth.sum()), "predicted_spikes": int(prediction.sum()),
        "spike_exact_matches": exact_tp,
        "spike_matches_tolerance_2ms": matched,
        "spike_precision_tolerance_2ms": matched / max(1, matched + fp),
        "spike_recall_tolerance_2ms": matched / max(1, matched + fn),
    }


comparison = []
all_predictions = {}
for model_index, name in enumerate(requested):
    torch.manual_seed(CFG.seed + model_index)
    model = model_builders[name]()
    print(name, "parameters:", count_trainable_parameters(model))
    completed = load_completed_model(name, model)
    if completed is None:
        model, best_validation, history = train_model(name, model)
    else:
        model, best_validation, history = completed
    prediction = predict(model)
    all_predictions[name] = prediction
    truth = test_y[:, 1:]
    error = prediction - truth
    row = {
        "model": name,
        "parameters": count_trainable_parameters(model),
        "epochs_trained": len(history),
        "best_validation_selection_loss": best_validation,
        "test_mean_normalized_rmse": float(np.mean(np.sqrt(np.mean(np.square(error), axis=(0, 1))) / state_std)),
        "test_soma_rmse_mV": float(np.sqrt(np.mean(np.square(error[..., MICRO_STATE_NAMES.index('soma.v_mV')])))),
        **spike_metrics(truth[..., MICRO_STATE_NAMES.index("soma.v_mV")], prediction[..., MICRO_STATE_NAMES.index("soma.v_mV")]),
    }
    for event_index, event_name in enumerate(MICRO_EVENT_NAMES):
        mask = test_events[..., event_index].astype(bool)
        row[f"soma_rmse_{event_name}_mV"] = float(np.sqrt(np.mean(np.square(error[..., 0][mask])))) if mask.any() else np.nan
    comparison.append(row)
    np.savez_compressed(OUTPUT / f"{name}_test_predictions.npz", prediction=prediction, truth=truth)
    pd.DataFrame(comparison).to_csv(OUTPUT / "comparison.csv", index=False)
    print(pd.DataFrame(comparison).sort_values("best_validation_selection_loss"))

(OUTPUT / "experiment.json").write_text(json.dumps({
    "contract": "spike-only input; no teacher state feedback",
    "dataset": dataset_report,
    "event_catalog": catalog,
    "training_config": asdict(CFG),
    "models": requested,
    "evaluation_views": ["natural overall test", "event-stratified test", "burst/rapid/plateau challenge"],
}, indent=2), encoding="utf-8")
print("complete results:", OUTPUT)
