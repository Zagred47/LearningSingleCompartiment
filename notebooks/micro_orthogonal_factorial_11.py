"""Orthogonal 2x2 training: recurrent scaffold x broadband objective.

Factors
-------
architecture: standard GRU vs causal dilated Conv1d followed by standard GRU
objective: normalized-state MSE vs MSE + multi-resolution STFT

The script trains from scratch, uses validation for selection, and never opens
the test split.  It is designed for Kaggle resume and complete artifact export.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import csv
import json
import math
import os
import shutil
import subprocess
import sys
import time

import h5py
import numpy as np
import torch
from torch import nn
try:
    from tqdm.auto import tqdm
except ImportError:
    class _PlainProgress:
        def __init__(self, total=None, desc="", **kwargs):
            self.total, self.desc, self.count = total, desc, 0
        def update(self, value=1):
            self.count += value
            print(f"{self.desc}: {self.count}/{self.total}", flush=True)
        def set_postfix(self, **values):
            print(self.desc, values, flush=True)
        def close(self):
            return None

    def tqdm(*args, **kwargs):
        return _PlainProgress(*args, **kwargs)


def _find_repository() -> Path:
    candidates = [Path(__file__).resolve().parents[1]]
    candidates.extend(path.parent for path in Path("/kaggle/working").glob("**/pyproject.toml"))
    candidates.extend(path.parent for path in Path("/kaggle/input").glob("**/pyproject.toml"))
    for root in candidates:
        if (root / "src/hay_single_compartment").is_dir():
            return root.resolve()
    raise FileNotFoundError("LearningSingleCompartiment repository not found")


REPO_ROOT = _find_repository()
SOURCE_ROOT = REPO_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))
for module_name in tuple(sys.modules):
    if module_name == "hay_single_compartment" or module_name.startswith("hay_single_compartment."):
        del sys.modules[module_name]

from hay_single_compartment import (  # noqa: E402
    FailureAtlasConfig,
    InputOnlyConvGRU,
    InputOnlyGRU,
    MICRO_EVENT_NAMES,
    MICRO_REGIME_NAMES,
    MICRO_STATE_NAMES,
    StateMSEMultiResolutionSTFTLoss,
    StratifiedWindowSampler,
    build_failure_atlas,
    classify_micro_events,
    count_trainable_parameters,
    write_failure_atlas,
)
from hay_single_compartment.failure_atlas import is_slow_state, match_spikes, sha256_file  # noqa: E402


@dataclass(frozen=True)
class FactorialConfig:
    temporal_bin: int = 5
    batch_trajectories: int = 6
    chunk_steps: int = 256
    epochs: int = 30
    minimum_epochs: int = 10
    patience: int = 8
    learning_rate: float = 5e-4
    weight_decay: float = 1e-5
    gradient_clip: float = 1.0
    stratified_windows_per_epoch: int = 48
    stratified_window_steps: int = 256
    context_steps: int = 4000
    spectral_weight: float = 0.10
    spectral_warmup_epochs: int = 3
    spectral_curriculum_epochs: int = 5
    seed: int = 20260802


CFG = FactorialConfig(
    epochs=int(os.environ.get("HAY_FACTORIAL_EPOCHS", "30")),
    minimum_epochs=int(os.environ.get("HAY_FACTORIAL_MINIMUM_EPOCHS", "10")),
    patience=int(os.environ.get("HAY_FACTORIAL_PATIENCE", "8")),
    batch_trajectories=int(os.environ.get("HAY_FACTORIAL_BATCH_TRAJECTORIES", "6")),
    stratified_windows_per_epoch=int(os.environ.get("HAY_FACTORIAL_WINDOWS_PER_EPOCH", "48")),
    context_steps=int(os.environ.get("HAY_FACTORIAL_CONTEXT_STEPS", "4000")),
    spectral_weight=float(os.environ.get("HAY_FACTORIAL_SPECTRAL_WEIGHT", "0.10")),
    spectral_warmup_epochs=int(
        os.environ.get("HAY_FACTORIAL_SPECTRAL_WARMUP_EPOCHS", "3")
    ),
    spectral_curriculum_epochs=int(
        os.environ.get("HAY_FACTORIAL_SPECTRAL_CURRICULUM_EPOCHS", "5")
    ),
)
if CFG.epochs < 1 or CFG.minimum_epochs < 1 or CFG.patience < 1:
    raise ValueError("epochs, minimum_epochs and patience must be positive")
if CFG.spectral_warmup_epochs < 0 or CFG.spectral_curriculum_epochs < 1:
    raise ValueError("spectral warmup must be non-negative and curriculum must be positive")

OUTPUT = Path(os.environ.get("HAY_FACTORIAL_OUTPUT", "/kaggle/working/hay_micro_orthogonal_factorial_11"))
if not Path("/kaggle").exists() and "HAY_FACTORIAL_OUTPUT" not in os.environ:
    OUTPUT = REPO_ROOT / "artifacts/orthogonal_factorial_11"
CHECKPOINTS = OUTPUT / "checkpoints"
ATLASES = OUTPUT / "failure_atlases"
OUTPUT.mkdir(parents=True, exist_ok=True)
CHECKPOINTS.mkdir(exist_ok=True)
ATLASES.mkdir(exist_ok=True)


def _valid_dataset(path: Path) -> bool:
    try:
        with h5py.File(path, "r") as handle:
            return all(
                key in handle
                for key in (
                    "train/inputs", "train/states", "train/burnin_inputs", "train/burnin_states",
                    "validation/inputs", "validation/states", "validation/burnin_inputs",
                    "validation/spikes", "validation/regimes",
                )
            ) and all(
                attribute in handle.attrs
                for attribute in ("config_json", "state_names_json", "input_names_json")
            )
    except (OSError, KeyError, ValueError):
        return False


def discover_dataset() -> Path:
    override = os.environ.get("HAY_FACTORIAL_DATASET")
    if override:
        path = Path(override).expanduser().resolve()
        if not _valid_dataset(path):
            raise ValueError(f"HAY_FACTORIAL_DATASET is not a compatible dataset: {path}")
        return path
    roots = [path for path in (Path("/kaggle/input"), Path("/kaggle/working"), REPO_ROOT.parent) if path.exists()]
    candidates: dict[str, Path] = {}
    for root in roots:
        for path in root.glob("**/*.h5"):
            if _valid_dataset(path):
                candidates[str(path.resolve())] = path.resolve()
    if not candidates:
        raise FileNotFoundError("Attach hay_micro_4c_event_enriched_v2.h5 or set HAY_FACTORIAL_DATASET")
    ranked = sorted(
        candidates.values(),
        key=lambda path: (
            0 if "event_enriched_v2" in path.name.lower() else 1,
            0 if "/kaggle/input/" in path.as_posix() else 1,
            str(path),
        ),
    )
    return ranked[0]


DATASET = discover_dataset()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("repository:", REPO_ROOT)
print("device    :", DEVICE)
print("dataset   :", DATASET)
print("output    :", OUTPUT)
hash_started = time.perf_counter()
print("[integrity] calculating dataset SHA-256 ...", flush=True)
DATASET_SHA256 = sha256_file(DATASET)
print(f"[integrity] {DATASET_SHA256} ({time.perf_counter() - hash_started:.1f}s)", flush=True)


def pack_spikes(values: np.ndarray, factor: int) -> np.ndarray:
    usable = values.shape[1] // factor * factor
    return values[:, :usable].reshape(
        values.shape[0], usable // factor, factor * values.shape[2]
    ).astype(np.float32)


def resample_binary(values: np.ndarray, factor: int) -> np.ndarray:
    usable = values.shape[1] // factor * factor
    return values[:, :usable].reshape(values.shape[0], usable // factor, factor).max(axis=2)


def resample_regimes(values: np.ndarray, factor: int) -> np.ndarray:
    usable = values.shape[1] // factor * factor
    # Drive regimes last far longer than a packed bin; the boundary label is
    # taken from the final raw microstep to preserve causal alignment.
    return values[:, :usable].reshape(values.shape[0], usable // factor, factor)[:, :, -1]


def load_split(name: str, maximum_trajectories: int | None = None) -> dict[str, np.ndarray]:
    with h5py.File(DATASET, "r") as handle:
        config = json.loads(handle.attrs["config_json"])
        raw_dt_ms = float(config["dt_ms"])
        count = int(handle[f"{name}/inputs"].shape[0])
        if maximum_trajectories is not None:
            count = min(count, maximum_trajectories)
        burnin = pack_spikes(handle[f"{name}/burnin_inputs"][:count], CFG.temporal_bin)
        inputs = pack_spikes(handle[f"{name}/inputs"][:count], CFG.temporal_bin)
        burnin_states = handle[f"{name}/burnin_states"][:count, :: CFG.temporal_bin].astype(np.float32)
        states = handle[f"{name}/states"][:count, :: CFG.temporal_bin].astype(np.float32)
        spikes = resample_binary(handle[f"{name}/spikes"][:count], CFG.temporal_bin)
        regimes = resample_regimes(handle[f"{name}/regimes"][:count], CFG.temporal_bin)
    states = states[:, : inputs.shape[1] + 1]
    burnin_states = burnin_states[:, : burnin.shape[1] + 1]
    labels = classify_micro_events(
        states[:, 1 : inputs.shape[1] + 1], spikes, MICRO_STATE_NAMES, raw_dt_ms * CFG.temporal_bin
    )
    return {
        "burnin": burnin,
        "burnin_states": burnin_states,
        "inputs": inputs,
        "states": states,
        "spikes": spikes,
        "regimes": regimes,
        "events": labels,
    }


max_train = int(os.environ.get("HAY_FACTORIAL_MAX_TRAIN_TRAJECTORIES", "0")) or None
max_validation = int(os.environ.get("HAY_FACTORIAL_MAX_VALIDATION_TRAJECTORIES", "0")) or None
print("[data] loading train and validation only; test remains unopened", flush=True)
train = load_split("train", max_train)
validation = load_split("validation", max_validation)
STATE_NAMES = list(MICRO_STATE_NAMES)
with h5py.File(DATASET, "r") as handle:
    stored_state_names = json.loads(handle.attrs["state_names_json"])
    input_names = json.loads(handle.attrs["input_names_json"])
    dataset_config = json.loads(handle.attrs["config_json"])
if stored_state_names != STATE_NAMES:
    raise ValueError("dataset state order differs from MICRO_STATE_NAMES")

normalization_source = np.concatenate((train["burnin_states"], train["states"][:, 1:]), axis=1)
state_mean = normalization_source.reshape(-1, len(STATE_NAMES)).mean(0).astype(np.float32)
state_std = np.maximum(
    normalization_source.reshape(-1, len(STATE_NAMES)).std(0), 1e-6
).astype(np.float32)
del normalization_source
train_burnin_y_n = (train["burnin_states"] - state_mean) / state_std
train_y_n = (train["states"] - state_mean) / state_std
model_dt_ms = float(dataset_config["dt_ms"]) * CFG.temporal_bin
INPUT_DIM = int(train["inputs"].shape[-1])
STATE_DIM = int(train["states"].shape[-1])
FAST_EVENT_NAMES = (
    "isolated_spike",
    "burst_spike",
    "rapid_fire",
    "tuft_plateau",
    "spike_with_tuft_plateau",
)


RUN_SPECS = {
    "gru_mse": {"architecture": "GRU", "objective": "MSE"},
    "gru_mrstft": {"architecture": "GRU", "objective": "MSE+MRSTFT"},
    "causal_conv_gru_mse": {"architecture": "CausalConv1d+GRU", "objective": "MSE"},
    "causal_conv_gru_mrstft": {"architecture": "CausalConv1d+GRU", "objective": "MSE+MRSTFT"},
}
requested = tuple(
    name.strip()
    for name in os.environ.get("HAY_FACTORIAL_RUNS", ",".join(RUN_SPECS)).split(",")
    if name.strip()
)
unknown = set(requested) - set(RUN_SPECS)
if unknown:
    raise ValueError(f"unknown factorial runs: {sorted(unknown)}")


def build_model(name: str) -> nn.Module:
    if RUN_SPECS[name]["architecture"] == "GRU":
        return InputOnlyGRU(INPUT_DIM, STATE_DIM, hidden_dim=200, decoder_dim=200)
    return InputOnlyConvGRU(
        INPUT_DIM, STATE_DIM, hidden_dim=200, conv_channels=96,
        dilations=(1, 2, 4), kernel_size=3, decoder_dim=200,
    )


def build_criterion(name: str) -> StateMSEMultiResolutionSTFTLoss | None:
    if RUN_SPECS[name]["objective"] == "MSE":
        return None
    return StateMSEMultiResolutionSTFTLoss(
        spectral_weight=CFG.spectral_weight,
        fft_sizes=(64, 32, 16), hop_sizes=(16, 8, 4), win_lengths=(64, 32, 16),
    ).to(DEVICE)


parameter_counts = {
    name: count_trainable_parameters(build_model(name)) for name in RUN_SPECS
}
parameter_budget_ratio = max(parameter_counts.values()) / min(parameter_counts.values())
if parameter_budget_ratio > 1.05:
    raise RuntimeError(
        f"factorial parameter-budget mismatch: ratio={parameter_budget_ratio:.4f} > 1.05"
    )
print(
    "[control] parameter counts:", parameter_counts,
    f"| max/min={parameter_budget_ratio:.4f}", flush=True,
)


def model_spec(model: nn.Module) -> dict[str, Any]:
    common = {
        "input_dim": INPUT_DIM,
        "state_dim": STATE_DIM,
        "parameters": count_trainable_parameters(model),
        "decoder_dim": int(model.decoder.network[1].out_features),
    }
    if isinstance(model, InputOnlyGRU):
        return {"class": "InputOnlyGRU", **common, "hidden_dim": model.hidden_dim, "layers": model.layers}
    return {
        "class": "InputOnlyConvGRU",
        **common,
        "scientific_label": "CausalConv1d+GRU",
        "hidden_dim": model.hidden_dim,
        "conv_channels": int(model.frontend[0].out_channels),
        "dilations": list(model.dilations),
        "kernel_size": model.kernel_size,
        "receptive_field_steps": model.receptive_field,
        "receptive_field_ms": model.receptive_field * model_dt_ms,
    }


def detach(hidden: Any) -> Any:
    if hidden is None:
        return None
    if isinstance(hidden, tuple):
        return tuple(detach(value) for value in hidden)
    return hidden.detach()


def tensor(values: np.ndarray, indices: np.ndarray | None = None) -> torch.Tensor:
    if indices is not None:
        values = values[indices]
    return torch.as_tensor(values, device=DEVICE)


def spectral_scale(epoch: int) -> float:
    if epoch <= CFG.spectral_warmup_epochs:
        return 0.0
    if CFG.spectral_curriculum_epochs <= 1:
        return 1.0
    return float(np.clip(
        (epoch - CFG.spectral_warmup_epochs) / CFG.spectral_curriculum_epochs, 0.0, 1.0
    ))


def loss_value(
    criterion: StateMSEMultiResolutionSTFTLoss | None,
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if criterion is None:
        mse = torch.mean(torch.square(prediction - target))
        return mse, {"mse": mse}
    return criterion(prediction, target)


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


@torch.no_grad()
def predict_validation(model: nn.Module) -> np.ndarray:
    model.eval()
    predictions = []
    for offset in range(0, len(validation["inputs"]), CFG.batch_trajectories):
        stop = min(offset + CFG.batch_trajectories, len(validation["inputs"]))
        burnin = tensor(validation["burnin"][offset:stop])
        inputs = tensor(validation["inputs"][offset:stop])
        hidden = None
        for start in range(0, burnin.shape[1], CFG.chunk_steps):
            _, hidden = model(burnin[:, start : start + CFG.chunk_steps], hidden)
        chunks = []
        for start in range(0, inputs.shape[1], CFG.chunk_steps):
            output, hidden = model(inputs[:, start : start + CFG.chunk_steps], hidden)
            chunks.append(output)
        normalized = torch.cat(chunks, dim=1).cpu().numpy()
        predictions.append(normalized * state_std + state_mean)
    return np.concatenate(predictions, axis=0)


def common_validation_metrics(prediction: np.ndarray) -> dict[str, float | int]:
    truth = validation["states"][:, 1 : prediction.shape[1] + 1]
    error = prediction - truth
    normalized = error / state_std
    state_nrmse = np.sqrt(np.mean(np.square(normalized), axis=(0, 1)))
    soma = STATE_NAMES.index("soma.v_mV")
    event_indices = [MICRO_EVENT_NAMES.index(name) for name in FAST_EVENT_NAMES]
    event_mask = validation["events"][..., event_indices].any(axis=-1)
    subthreshold = validation["events"][..., 0].astype(bool)
    spike_report, _ = match_spikes(truth[..., soma], prediction[..., soma], model_dt_ms)
    metrics: dict[str, float | int] = {
        "mean_state_nrmse": float(state_nrmse.mean()),
        "median_state_nrmse": float(np.median(state_nrmse)),
        "slow_state_nrmse": float(np.mean([
            state_nrmse[index] for index, name in enumerate(STATE_NAMES) if is_slow_state(name)
        ])),
        "soma_rmse_mV": float(np.sqrt(np.mean(np.square(error[..., soma])))),
        "event_soma_rmse_mV": float(np.sqrt(np.mean(np.square(error[..., soma][event_mask])))),
        "subthreshold_soma_rmse_mV": float(np.sqrt(np.mean(np.square(error[..., soma][subthreshold])))),
        "truth_spikes": int(spike_report["truth_spikes"]),
        "predicted_spikes": int(spike_report["predicted_spikes"]),
        "spike_precision": float(spike_report["precision"]),
        "spike_recall": float(spike_report["recall"]),
        "spike_f1": float(spike_report["f1"]),
    }
    for event_index, event_name in enumerate(MICRO_EVENT_NAMES):
        mask = validation["events"][..., event_index].astype(bool)
        metrics[f"soma_rmse_{event_name}_mV"] = (
            float(np.sqrt(np.mean(np.square(error[..., soma][mask])))) if mask.any() else float("nan")
        )
    return metrics


def natural_updates_per_epoch() -> int:
    batches = math.ceil(len(train["inputs"]) / CFG.batch_trajectories)
    burn_chunks = math.ceil(train["burnin"].shape[1] / CFG.chunk_steps)
    input_chunks = math.ceil(train["inputs"].shape[1] / CFG.chunk_steps)
    sampled_batches = math.ceil(CFG.stratified_windows_per_epoch / CFG.batch_trajectories)
    return batches * (burn_chunks + input_chunks) + sampled_batches


def train_model(name: str) -> tuple[nn.Module, list[dict[str, Any]], dict[str, Any]]:
    torch.manual_seed(CFG.seed)
    np.random.seed(CFG.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(CFG.seed)
    model = build_model(name).to(DEVICE)
    criterion = build_criterion(name)
    optimizer = torch.optim.AdamW(model.parameters(), lr=CFG.learning_rate, weight_decay=CFG.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CFG.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=DEVICE.type == "cuda")
    last_path = CHECKPOINTS / f"{name}_last.pt"
    best_path = CHECKPOINTS / f"{name}.pt"
    summary_path = OUTPUT / f"{name}_summary.json"
    history: list[dict[str, Any]] = []
    best_score, stale, start_epoch = float("inf"), 0, 1
    if last_path.exists():
        resume = torch.load(last_path, map_location=DEVICE, weights_only=False)
        compatible = (
            resume.get("dataset_sha256") == DATASET_SHA256
            and resume.get("training_config") == asdict(CFG)
            and resume.get("run_name") == name
        )
        if compatible:
            model.load_state_dict(resume["model_state_dict"])
            optimizer.load_state_dict(resume["optimizer_state_dict"])
            scheduler.load_state_dict(resume["scheduler_state_dict"])
            scaler.load_state_dict(resume["scaler_state_dict"])
            history = list(resume["history"])
            best_score = float(resume["best_score"])
            stale = int(resume["stale"])
            start_epoch = int(resume["epoch"]) + 1
            print(f"[{name}] resume after epoch {start_epoch - 1}", flush=True)
    if summary_path.exists() and best_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("dataset_sha256") == DATASET_SHA256 and summary.get("training_config") == asdict(CFG):
            checkpoint = torch.load(best_path, map_location=DEVICE, weights_only=False)
            model.load_state_dict(checkpoint["model_state_dict"])
            print(f"[{name}] completed run reused", flush=True)
            return model, history or list(summary["history"]), summary["best_validation"]

    complete_inputs = np.concatenate((train["burnin"], train["inputs"]), axis=1)
    run_started = time.perf_counter()
    total_updates = natural_updates_per_epoch()
    for epoch in range(start_epoch, CFG.epochs + 1):
        model.train()
        scale = spectral_scale(epoch) if criterion is not None else 0.0
        if criterion is not None:
            criterion.set_spectral_scale(scale)
        progress = tqdm(total=total_updates, desc=f"{name} epoch {epoch}/{CFG.epochs}", leave=True)
        order = np.random.default_rng(CFG.seed + epoch).permutation(len(train["inputs"]))
        train_totals: dict[str, float] = {}
        updates = 0
        for offset in range(0, len(order), CFG.batch_trajectories):
            indices = order[offset : offset + CFG.batch_trajectories]
            burnin = tensor(train["burnin"], indices)
            burnin_y = tensor(train_burnin_y_n, indices)
            inputs = tensor(train["inputs"], indices)
            targets = tensor(train_y_n, indices)
            hidden = None
            for phase_x, phase_y in ((burnin, burnin_y), (inputs, targets)):
                for start in range(0, phase_x.shape[1], CFG.chunk_steps):
                    chunk = phase_x[:, start : start + CFG.chunk_steps]
                    target = phase_y[:, start + 1 : start + 1 + chunk.shape[1]]
                    hidden_before = detach(hidden)
                    optimizer.zero_grad(set_to_none=True)
                    with torch.autocast("cuda", dtype=torch.float16, enabled=DEVICE.type == "cuda"):
                        prediction, _ = model(chunk, hidden_before)
                        loss, terms = loss_value(criterion, prediction, target)
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), CFG.gradient_clip)
                    scaler.step(optimizer)
                    scaler.update()
                    with torch.no_grad():
                        _, hidden = model(chunk, hidden_before)
                    hidden = detach(hidden)
                    train_totals["total"] = train_totals.get("total", 0.0) + float(loss.detach())
                    for key, value in terms.items():
                        train_totals[key] = train_totals.get(key, 0.0) + float(value.detach())
                    updates += 1
                    progress.update()

        sampler = StratifiedWindowSampler(
            train["events"], CFG.stratified_window_steps, seed=CFG.seed + epoch
        )
        sampled = sampler.sample(CFG.stratified_windows_per_epoch)
        for window_offset in range(0, len(sampled), CFG.batch_trajectories):
            rows = sampled[window_offset : window_offset + CFG.batch_trajectories]
            contexts, windows, targets = [], [], []
            for trajectory, start in rows:
                trajectory, start = int(trajectory), int(start)
                absolute = train["burnin"].shape[1] + start
                context_start = max(0, absolute - CFG.context_steps)
                context = complete_inputs[trajectory, context_start:absolute]
                if len(context) < CFG.context_steps:
                    context = np.pad(context, ((CFG.context_steps - len(context), 0), (0, 0)))
                contexts.append(context)
                windows.append(train["inputs"][trajectory, start : start + CFG.stratified_window_steps])
                targets.append(train_y_n[trajectory, start + 1 : start + 1 + CFG.stratified_window_steps])
            context_t, window_t, target_t = tensor(np.stack(contexts)), tensor(np.stack(windows)), tensor(np.stack(targets))
            hidden = None
            with torch.no_grad():
                for start in range(0, context_t.shape[1], CFG.chunk_steps):
                    _, hidden = model(context_t[:, start : start + CFG.chunk_steps], hidden)
                    hidden = detach(hidden)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16, enabled=DEVICE.type == "cuda"):
                prediction, _ = model(window_t, hidden)
                loss, terms = loss_value(criterion, prediction, target_t)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), CFG.gradient_clip)
            scaler.step(optimizer)
            scaler.update()
            train_totals["total"] = train_totals.get("total", 0.0) + float(loss.detach())
            for key, value in terms.items():
                train_totals[key] = train_totals.get(key, 0.0) + float(value.detach())
            updates += 1
            progress.update()
        scheduler.step()
        progress.close()

        validation_prediction = predict_validation(model)
        metrics = common_validation_metrics(validation_prediction)
        elapsed = time.perf_counter() - run_started
        completed_epochs = epoch - start_epoch + 1
        eta = elapsed / completed_epochs * max(0, CFG.epochs - epoch)
        row: dict[str, Any] = {
            "epoch": epoch,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "spectral_scale": scale,
            **{f"train_{key}": value / max(1, updates) for key, value in train_totals.items()},
            **{f"validation_{key}": value for key, value in metrics.items()},
            "elapsed_s": elapsed,
            "eta_s": eta,
        }
        history.append(row)
        score = float(metrics["event_soma_rmse_mV"])
        improved = score < best_score
        if improved:
            best_score, stale = score, 0
            torch.save({
                "format_version": 1,
                "experiment_id": "LO-01_SC-01_INT-01",
                "run_name": name,
                **RUN_SPECS[name],
                "model_state_dict": model.state_dict(),
                "model_spec": model_spec(model),
                "state_mean": state_mean,
                "state_std": state_std,
                "state_names": STATE_NAMES,
                "input_names": input_names,
                "dataset_sha256": DATASET_SHA256,
                "training_config": asdict(CFG),
                "selection_metric": "validation_event_soma_rmse_mV",
                "best_validation": metrics,
                "epoch": epoch,
            }, best_path)
        else:
            stale += 1
        torch.save({
            "format_version": 1,
            "run_name": name,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "dataset_sha256": DATASET_SHA256,
            "training_config": asdict(CFG),
            "history": history,
            "best_score": best_score,
            "stale": stale,
            "epoch": epoch,
        }, last_path)
        write_rows(OUTPUT / f"{name}_history.csv", history)
        print(
            f"[{name}] epoch {epoch:02d} | event {score:.4f} mV | global {metrics['mean_state_nrmse']:.4f} "
            f"| spikes {metrics['predicted_spikes']}/{metrics['truth_spikes']} | ETA {eta / 60:.1f} min",
            flush=True,
        )
        if epoch >= min(CFG.minimum_epochs, CFG.epochs) and stale >= CFG.patience:
            print(f"[{name}] early stop at epoch {epoch}", flush=True)
            break

    checkpoint = torch.load(best_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    summary = {
        "run_name": name,
        **RUN_SPECS[name],
        "dataset_sha256": DATASET_SHA256,
        "training_config": asdict(CFG),
        "parameters": count_trainable_parameters(model),
        "epochs_trained": len(history),
        "best_epoch": int(checkpoint["epoch"]),
        "best_validation": checkpoint["best_validation"],
        "history": history,
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return model, history, checkpoint["best_validation"]


def factorial_effects(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_name = {row["run"]: row for row in rows}
    required = set(RUN_SPECS)
    if not required <= set(by_name):
        return []
    effects = []
    metrics = (
        "event_soma_rmse_mV", "mean_state_nrmse", "slow_state_nrmse",
        "subthreshold_soma_rmse_mV", "spike_recall",
    )
    for metric in metrics:
        gm = float(by_name["gru_mse"][metric])
        gs = float(by_name["gru_mrstft"][metric])
        cm = float(by_name["causal_conv_gru_mse"][metric])
        cs = float(by_name["causal_conv_gru_mrstft"][metric])
        effects.append({
            "metric": metric,
            "loss_main_effect_mrstft_minus_mse": 0.5 * ((gs - gm) + (cs - cm)),
            "architecture_main_effect_conv_minus_gru": 0.5 * ((cm - gm) + (cs - gs)),
            "interaction": (cs - cm) - (gs - gm),
            "lower_is_better": metric != "spike_recall",
        })
    return effects


comparison: list[dict[str, Any]] = []
truth_validation = validation["states"][:, 1:]
np.savez_compressed(
    OUTPUT / "validation_reference.npz",
    truth=truth_validation,
    events=validation["events"],
    spikes=validation["spikes"],
    regimes=validation["regimes"],
)
experiment_started = time.perf_counter()
for run_index, name in enumerate(requested, start=1):
    model_preview = build_model(name)
    print(
        f"\n[factorial] run {run_index}/{len(requested)}: {name} | "
        f"parameters {count_trainable_parameters(model_preview):,}", flush=True,
    )
    del model_preview
    model, history, best_validation = train_model(name)
    prediction = predict_validation(model)
    np.savez_compressed(
        OUTPUT / f"{name}_validation_predictions.npz",
        prediction=prediction,
    )
    report, tables = build_failure_atlas(
        truth_validation,
        prediction,
        STATE_NAMES,
        model_dt_ms,
        event_masks=validation["events"],
        regimes=validation["regimes"],
        regime_names=MICRO_REGIME_NAMES,
        teacher_spikes=validation["spikes"],
        state_scale=state_std,
        config=FailureAtlasConfig(),
    )
    write_failure_atlas(ATLASES / name, report, tables, truth_validation, prediction, STATE_NAMES, name)
    row = {
        "run": name,
        **RUN_SPECS[name],
        "parameters": count_trainable_parameters(model),
        "epochs_trained": len(history),
        **best_validation,
    }
    comparison.append(row)
    write_rows(OUTPUT / "validation_comparison.csv", comparison)
    effects = factorial_effects(comparison)
    if effects:
        write_rows(OUTPUT / "factorial_effects.csv", effects)


def relative_improvement(reference: float, candidate: float) -> float:
    return (reference - candidate) / max(abs(reference), 1e-12)


by_name = {row["run"]: row for row in comparison}
decision: dict[str, Any] = {
    "status": "single_seed_preflight_only",
    "test_split_opened": False,
    "promotion_requires": "repeat the promoted contrast with at least three seeds",
}
if set(RUN_SPECS) <= set(by_name):
    baseline = by_name["gru_mse"]
    loss_candidate = by_name["gru_mrstft"]
    architecture_candidate = by_name["causal_conv_gru_mse"]
    interaction_candidate = by_name["causal_conv_gru_mrstft"]

    def contrast(candidate: dict[str, Any], reference: dict[str, Any]) -> dict[str, Any]:
        improvement = relative_improvement(
            reference["event_soma_rmse_mV"], candidate["event_soma_rmse_mV"]
        )
        ratios = {
            "global_mean_normalized_rmse": (
                candidate["mean_state_nrmse"] / reference["mean_state_nrmse"]
            ),
            "slow_state_mean_normalized_rmse": (
                candidate["slow_state_nrmse"] / reference["slow_state_nrmse"]
            ),
            "subthreshold_soma_rmse_mV": (
                candidate["subthreshold_soma_rmse_mV"]
                / reference["subthreshold_soma_rmse_mV"]
            ),
        }
        guardrails_pass = (
            ratios["global_mean_normalized_rmse"] <= 1.10
            and ratios["slow_state_mean_normalized_rmse"] <= 1.10
            and ratios["subthreshold_soma_rmse_mV"] <= 1.25
        )
        return {
            "event_relative_improvement": improvement,
            "guardrail_ratios": ratios,
            "guardrails_pass": guardrails_pass,
            "promote_to_three_seed_replication": improvement >= 0.15 and guardrails_pass,
        }

    decision.update({
        "loss_contrast": contrast(loss_candidate, baseline),
        "architecture_contrast_under_mse": contrast(architecture_candidate, baseline),
        "joint_candidate": contrast(interaction_candidate, baseline),
        "preflight_threshold": "at least 15% event-RMSE improvement plus all guardrails",
    })
(OUTPUT / "preflight_decision.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")

provenance = {
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "experiment_ids": ["LO-01", "SC-01", "INT-01"],
    "repository": str(REPO_ROOT),
    "git_commit": subprocess.check_output(
        ["git", "-c", f"safe.directory={REPO_ROOT.as_posix()}", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
        text=True,
    ).strip(),
    "dataset": str(DATASET),
    "dataset_sha256": DATASET_SHA256,
    "test_split_opened": False,
    "device": str(DEVICE),
    "torch": torch.__version__,
    "numpy": np.__version__,
    "training_config": asdict(CFG),
    "primary_event_set": FAST_EVENT_NAMES,
    "parameter_counts": parameter_counts,
    "parameter_budget_max_to_min_ratio": parameter_budget_ratio,
    "runs": requested,
    "elapsed_s": time.perf_counter() - experiment_started,
}
(OUTPUT / "provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
shutil.copy2(REPO_ROOT / "configs/research_contract_v1.json", OUTPUT / "research_contract_v1.json")
shutil.copy2(
    REPO_ROOT / "research/orthogonal_factorial_11_preregistration.json",
    OUTPUT / "orthogonal_factorial_11_preregistration.json",
)
shutil.copy2(REPO_ROOT / "research/literature_evidence.csv", OUTPUT / "literature_evidence.csv")
include_last = os.environ.get("HAY_FACTORIAL_DOWNLOAD_LAST_CHECKPOINTS", "0") == "1"
archive_source = OUTPUT
if not include_last:
    staging_root = OUTPUT.parent / f"{OUTPUT.name}_archive_staging"
    if staging_root.exists():
        shutil.rmtree(staging_root)
    staging_root.mkdir()
    staging = staging_root / OUTPUT.name
    shutil.copytree(
        OUTPUT,
        staging,
        ignore=lambda _path, names: {name for name in names if name.endswith("_last.pt")},
    )
    archive_source = staging
zip_base = OUTPUT.parent / f"{OUTPUT.name}_complete"
zip_path = Path(shutil.make_archive(str(zip_base), "zip", archive_source.parent, archive_source.name))
ZIP_PATH = zip_path
print("\nFactorial preflight complete. Test split was never opened.")
print("Results:", OUTPUT)
print(
    "ZIP    :", zip_path, f"({zip_path.stat().st_size / 2**20:.1f} MiB)",
    "| last checkpoints included:", include_last,
)
