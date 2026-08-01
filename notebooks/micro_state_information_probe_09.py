"""Diagnostic probe: frozen-GRU latent versus previous physical state.

This is not a deployable surrogate.  Teacher state is used only in the oracle
feature arm to test whether explicit state feedback is the missing information.
Both arms use the same standard linear/MLP probes and are evaluated on natural,
untouched validation and test distributions.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

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
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

from hay_single_compartment import InputOnlyGRU


def precision_recall_curve(labels: np.ndarray, scores: np.ndarray):
    labels = np.asarray(labels, dtype=bool)
    order = np.argsort(-np.asarray(scores), kind="stable")
    ordered_labels = labels[order]
    true_positive = np.cumsum(ordered_labels)
    false_positive = np.cumsum(~ordered_labels)
    precision = true_positive / np.maximum(1, true_positive + false_positive)
    recall = true_positive / max(1, int(labels.sum()))
    return precision, recall, np.asarray(scores)[order]


def average_precision_score(labels: np.ndarray, scores: np.ndarray) -> float:
    precision, recall, _ = precision_recall_curve(labels, scores)
    recall_increment = np.diff(np.concatenate(([0.0], recall)))
    return float(np.sum(recall_increment * precision))


def roc_auc_score(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=bool)
    order = np.argsort(-np.asarray(scores), kind="stable")
    ordered_labels = labels[order]
    true_positive = np.cumsum(ordered_labels) / max(1, int(labels.sum()))
    false_positive = np.cumsum(~ordered_labels) / max(1, int((~labels).sum()))
    true_positive = np.concatenate(([0.0], true_positive, [1.0]))
    false_positive = np.concatenate(([0.0], false_positive, [1.0]))
    return float(np.trapz(true_positive, false_positive))


WORKING = Path("/kaggle/working")


def discover(default: Path, pattern: str) -> Path:
    if default.exists():
        return default
    root = Path("/kaggle/input")
    matches = sorted(root.rglob(pattern)) if root.exists() else []
    if not matches:
        raise FileNotFoundError(f"input not found: {pattern}")
    print(f"auto-discovered {pattern}: {matches[0]}")
    return matches[0]


DATASET = discover(
    Path(os.environ.get("HAY_PROBE_DATASET", WORKING / "hay_micro_4c_event_enriched_v2.h5")),
    "hay_micro_4c_event_enriched_v2*.h5",
)
BASELINE = discover(
    Path(os.environ.get("HAY_PROBE_BASELINE", WORKING / "gru_mse.pt")),
    "gru_mse.pt",
)
OUTPUT = Path(os.environ.get("HAY_PROBE_OUTPUT", WORKING / "hay_micro_state_information_probe_09"))
OUTPUT.mkdir(parents=True, exist_ok=True)
EPOCHS = int(os.environ.get("HAY_PROBE_EPOCHS", "15"))
NEGATIVE_RATIO = int(os.environ.get("HAY_PROBE_NEGATIVE_RATIO", "10"))
BATCH_SIZE = int(os.environ.get("HAY_PROBE_BATCH_SIZE", "2048"))
SEED = 20260801
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", DEVICE, "| dataset:", DATASET, "| baseline:", BASELINE)


payload = torch.load(BASELINE, map_location="cpu", weights_only=False)
weights = payload.get("model_state_dict", payload.get("model"))
state_names = list(payload["state_names"])
state_mean = np.asarray(payload["state_mean"], np.float32)
state_std = np.asarray(payload["state_std"], np.float32)
temporal_bin = int(payload.get("training_config", payload.get("config", {})).get("temporal_bin", 5))
hidden_dim = weights["recurrent.weight_hh_l0"].shape[1]
input_dim = weights["input_encoder.0.weight"].shape[1]
state_dim = weights["decoder.network.3.weight"].shape[0]
decoder_dim = weights["decoder.network.1.weight"].shape[0]
soma_index = state_names.index("soma.v_mV")
model = InputOnlyGRU(input_dim, state_dim, hidden_dim=hidden_dim, decoder_dim=decoder_dim)
model.load_state_dict(weights)
model = model.to(DEVICE).eval()
for parameter in model.parameters():
    parameter.requires_grad_(False)


def pack_row(values: np.ndarray) -> np.ndarray:
    usable = len(values) // temporal_bin * temporal_bin
    return values[:usable].reshape(-1, temporal_bin * values.shape[-1]).astype(np.float32)


def dilate(core: np.ndarray, radius: int = 2) -> np.ndarray:
    return np.convolve(core.astype(np.int16), np.ones(2 * radius + 1, np.int16), mode="same") > 0


@torch.no_grad()
def extract_split(split: str, natural: bool) -> dict[str, np.ndarray]:
    records = {key: [] for key in (
        "latent", "state", "label", "core", "truth_voltage_n", "baseline_voltage_n"
    )}
    rng = np.random.default_rng(SEED + {"train": 1, "validation": 2, "test": 3}[split])
    with h5py.File(DATASET, "r") as handle:
        trajectories = handle[f"{split}/inputs"].shape[0]
        for trajectory in tqdm(range(trajectories), desc=f"extract {split}"):
            burnin = pack_row(handle[f"{split}/burnin_inputs"][trajectory])
            inputs = pack_row(handle[f"{split}/inputs"][trajectory])
            states = handle[f"{split}/states"][trajectory, ::temporal_bin][: len(inputs) + 1].astype(np.float32)
            states_n = (states - state_mean) / state_std
            voltage = states[1:, soma_index]
            core = voltage >= -35.0
            support = dilate(core, radius=2)
            if natural:
                selected = np.arange(len(inputs))
            else:
                positive = np.flatnonzero(support)
                negative = np.flatnonzero(~support)
                count = min(len(negative), max(1, len(positive) * NEGATIVE_RATIO))
                sampled_negative = rng.choice(negative, size=count, replace=False)
                selected = np.sort(np.concatenate((positive, sampled_negative)))

            hidden = None
            for start in range(0, len(burnin), 512):
                chunk = torch.from_numpy(burnin[start:start + 512]).unsqueeze(0).to(DEVICE)
                _, hidden = model(chunk, hidden)
            hidden_rows, baseline_rows = [], []
            for start in range(0, len(inputs), 512):
                chunk = torch.from_numpy(inputs[start:start + 512]).unsqueeze(0).to(DEVICE)
                encoded = model.input_encoder(chunk)
                sequence, hidden = model.recurrent(encoded, hidden)
                prediction = model.decoder(sequence)
                hidden_rows.append(sequence.squeeze(0).cpu().numpy())
                baseline_rows.append(prediction.squeeze(0).cpu().numpy())
            hidden_sequence = np.concatenate(hidden_rows)
            baseline_sequence = np.concatenate(baseline_rows)
            records["latent"].append(np.concatenate((hidden_sequence[selected], inputs[selected]), axis=1))
            records["state"].append(np.concatenate((states_n[:-1][selected], inputs[selected]), axis=1))
            records["label"].append(support[selected].astype(np.float32))
            records["core"].append(core[selected].astype(np.uint8))
            records["truth_voltage_n"].append(states_n[1:, soma_index][selected])
            records["baseline_voltage_n"].append(baseline_sequence[selected, soma_index])
    result = {key: np.concatenate(value) for key, value in records.items()}
    print(split, {key: value.shape for key, value in result.items()}, "prevalence", result["label"].mean())
    return result


train = extract_split("train", natural=False)
validation = extract_split("validation", natural=True)
test = extract_split("test", natural=True)


scalers = {}
for representation in ("latent", "state"):
    mean = train[representation].mean(0).astype(np.float32)
    std = train[representation].std(0).astype(np.float32)
    std = np.maximum(std, 1e-4)
    scalers[representation] = (mean, std)
    for split in (train, validation, test):
        split[representation] = ((split[representation] - mean) / std).astype(np.float32)


class Probe(nn.Module):
    def __init__(self, input_width: int, nonlinear: bool) -> None:
        super().__init__()
        self.network = (
            nn.Sequential(nn.Linear(input_width, 128), nn.SiLU(), nn.Linear(128, 1))
            if nonlinear else nn.Linear(input_width, 1)
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.network(values).squeeze(-1)


def fit_probe(
    features: np.ndarray,
    target: np.ndarray,
    *,
    nonlinear: bool,
    regression: bool,
    name: str,
    importance: np.ndarray | None = None,
):
    torch.manual_seed(SEED)
    probe = Probe(features.shape[1], nonlinear).to(DEVICE)
    if importance is None:
        importance = np.ones(len(target), dtype=np.float32)
    dataset = TensorDataset(
        torch.from_numpy(features),
        torch.from_numpy(target.astype(np.float32)),
        torch.from_numpy(importance.astype(np.float32)),
    )
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=DEVICE.type == "cuda")
    optimizer = torch.optim.AdamW(probe.parameters(), lr=1e-3, weight_decay=1e-5)
    positive_weight = (
        float((target <= 0.5).sum() / max(1, (target > 0.5).sum()))
        if not regression else 1.0
    )
    started = time.perf_counter()
    bar = tqdm(range(1, EPOCHS + 1), desc=name)
    for epoch in bar:
        running, count = 0.0, 0
        for batch_x, batch_y, batch_importance in loader:
            batch_x = batch_x.to(DEVICE)
            batch_y = batch_y.to(DEVICE)
            batch_importance = batch_importance.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            output = probe(batch_x)
            if regression:
                loss = (
                    (output - batch_y).square() * batch_importance
                ).sum() / batch_importance.sum().clamp_min(1.0)
            else:
                loss = nn.functional.binary_cross_entropy_with_logits(
                    output, batch_y,
                    pos_weight=torch.tensor(positive_weight, device=DEVICE),
                )
            loss.backward()
            optimizer.step()
            running += float(loss.detach()) * len(batch_x)
            count += len(batch_x)
        elapsed = time.perf_counter() - started
        eta = elapsed / epoch * (EPOCHS - epoch)
        bar.set_postfix(loss=f"{running/count:.4g}", eta=f"{eta/60:.1f}m")
    return probe.eval()


def infer(probe: nn.Module, features: np.ndarray) -> np.ndarray:
    values = []
    with torch.no_grad():
        for start in range(0, len(features), 8192):
            batch = torch.from_numpy(features[start:start + 8192]).to(DEVICE)
            values.append(probe(batch).cpu().numpy())
    return np.concatenate(values)


classification_rows = []
precision_recall_curves = {}
classification_models = {}
for representation in ("latent", "state"):
    for nonlinear, architecture in ((False, "linear"), (True, "mlp")):
        name = f"{representation}_{architecture}"
        probe = fit_probe(
            train[representation], train["label"], nonlinear=nonlinear,
            regression=False, name=f"classify {name}",
        )
        classification_models[name] = probe
        validation_probability = torch.sigmoid(torch.from_numpy(infer(probe, validation[representation]))).numpy()
        test_probability = torch.sigmoid(torch.from_numpy(infer(probe, test[representation]))).numpy()
        precision, recall, thresholds = precision_recall_curve(validation["label"], validation_probability)
        f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
        best = int(np.nanargmax(f1))
        threshold = float(thresholds[best])
        for split_name, split, probability in (
            ("validation", validation, validation_probability),
            ("test", test, test_probability),
        ):
            predicted = probability >= threshold
            label = split["label"] > 0.5
            tp = int((predicted & label).sum())
            fp = int((predicted & ~label).sum())
            fn = int((~predicted & label).sum())
            classification_rows.append({
                "representation": representation,
                "architecture": architecture,
                "split": split_name,
                "average_precision": average_precision_score(label, probability),
                "roc_auc": roc_auc_score(label, probability),
                "validation_selected_threshold": threshold,
                "precision": tp / max(1, tp + fp),
                "recall": tp / max(1, tp + fn),
                "f1": 2 * tp / max(1, 2 * tp + fp + fn),
                "predicted_fraction": float(predicted.mean()),
                "target_fraction": float(label.mean()),
            })
        precision_recall_curves[name] = (recall, precision)


classification = pd.DataFrame(classification_rows)
classification.to_csv(OUTPUT / "classification_probe.csv", index=False)
print(classification)


regression_rows = []
regression_models = {}
train_residual = train["truth_voltage_n"] - train["baseline_voltage_n"]
regression_importance = np.where(
    train["label"] > 0.5,
    (train["label"] <= 0.5).sum() / max(1, (train["label"] > 0.5).sum()),
    1.0,
).astype(np.float32)
for representation in ("latent", "state"):
    name = f"{representation}_mlp"
    probe = fit_probe(
        train[representation], train_residual, nonlinear=True,
        regression=True, name=f"regress {name}", importance=regression_importance,
    )
    regression_models[name] = probe
    for split_name, split in (("validation", validation), ("test", test)):
        residual = infer(probe, split[representation])
        prediction_n = split["baseline_voltage_n"] + residual
        prediction = prediction_n * state_std[soma_index] + state_mean[soma_index]
        truth_voltage = split["truth_voltage_n"] * state_std[soma_index] + state_mean[soma_index]
        baseline_voltage = split["baseline_voltage_n"] * state_std[soma_index] + state_mean[soma_index]
        masks = {
            "all": np.ones(len(prediction), dtype=bool),
            "support": split["label"] > 0.5,
            "core": split["core"] > 0,
            "subthreshold": truth_voltage < -35.0,
        }
        for region, mask in masks.items():
            regression_rows.append({
                "representation": representation,
                "split": split_name,
                "region": region,
                "samples": int(mask.sum()),
                "baseline_rmse_mV": float(np.sqrt(np.mean((baseline_voltage[mask] - truth_voltage[mask]) ** 2))),
                "probe_rmse_mV": float(np.sqrt(np.mean((prediction[mask] - truth_voltage[mask]) ** 2))),
            })


regression = pd.DataFrame(regression_rows)
regression.to_csv(OUTPUT / "voltage_residual_probe.csv", index=False)
print(regression)


figure, axis = plt.subplots(figsize=(8, 6))
for name, (recall, precision) in precision_recall_curves.items():
    axis.plot(recall, precision, label=name)
axis.set(xlabel="recall", ylabel="precision", title="Natural validation spike-support probes", xlim=(0, 1), ylim=(0, 1))
axis.grid(alpha=0.2)
axis.legend()
figure.tight_layout()
figure.savefig(OUTPUT / "precision_recall.png", dpi=170)
plt.close(figure)


torch.save({
    "classification": {name: model.state_dict() for name, model in classification_models.items()},
    "regression": {name: model.state_dict() for name, model in regression_models.items()},
    "scalers": scalers,
    "state_names": state_names,
    "state_mean": state_mean,
    "state_std": state_std,
}, OUTPUT / "probe_checkpoints.pt")
(OUTPUT / "experiment.json").write_text(json.dumps({
    "purpose": "diagnose whether previous physical state resolves the input-only spike bottleneck",
    "teacher_state_is_inference_input": False,
    "teacher_state_role": "oracle diagnostic arm only",
    "dataset": str(DATASET),
    "baseline": str(BASELINE),
    "temporal_bin": temporal_bin,
    "model_dt_ms": temporal_bin * 0.1,
    "epochs": EPOCHS,
    "negative_ratio": NEGATIVE_RATIO,
    "representations": {
        "latent": "frozen GRU hidden after current packed input + current packed input",
        "state": "previous true normalized 61-state vector + current packed input",
    },
}, indent=2), encoding="utf-8")
print("saved:", OUTPUT)
