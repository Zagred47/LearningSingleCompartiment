"""CB-01: controlled fast/slow threshold capability screen.

This is a synthetic architecture screen, not a Hay-teacher benchmark.  Ten
established sequence families are compared across parameter budget and task
difficulty.  One seed screens all sketches; only three mechanistically diverse
leads receive two additional paired seeds.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from shutil import make_archive
from typing import Any

os.environ.setdefault(
    "MPLCONFIGDIR",
    str((Path.cwd() / "artifacts/.matplotlib").resolve()),
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
try:
    from tqdm.auto import tqdm
except ImportError:  # The Kaggle notebook installs tqdm; this keeps local smoke tests portable.
    class tqdm:  # type: ignore[no-redef]
        def __init__(self, total: int, desc: str = "", unit: str = "") -> None:
            self.total, self.desc, self.completed = total, desc, 0
            print(f"{desc}: 0/{total} {unit}", flush=True)

        def update(self, amount: int = 1) -> None:
            self.completed += amount

        def set_postfix_str(self, value: str) -> None:
            print(f"{self.desc}: {self.completed}/{self.total} | {value}", flush=True)

        def close(self) -> None:
            pass

from hay_single_compartment import (
    CAPABILITY_ARCHITECTURES,
    FAST_SLOW_DIFFICULTIES,
    build_capability_model,
    fast_slow_metrics,
    generate_fast_slow_sequences,
    width_for_budget,
)


@dataclass(frozen=True)
class BenchConfig:
    train_sequences: int = 256
    validation_sequences: int = 128
    train_steps: int = 256
    validation_steps: int = 512
    batch_size: int = 32
    epochs: int = 8
    budgets: tuple[int, ...] = (8_000, 32_000)
    learning_rates: tuple[float, ...] = (3e-4, 1e-3)
    difficulties: tuple[str, ...] = ("easy", "medium", "hard")
    screen_seed: int = 20260804
    replication_seeds: tuple[int, ...] = (20260805, 20260806)
    data_seed: int = 20260840
    weight_decay: float = 1e-5
    gradient_clip: float = 1.0
    maximum_hits: int = 3


CFG = BenchConfig(
    train_sequences=int(os.environ.get("HAY_CB01_TRAIN_SEQUENCES", "256")),
    validation_sequences=int(os.environ.get("HAY_CB01_VALIDATION_SEQUENCES", "128")),
    train_steps=int(os.environ.get("HAY_CB01_TRAIN_STEPS", "256")),
    validation_steps=int(os.environ.get("HAY_CB01_VALIDATION_STEPS", "512")),
    batch_size=int(os.environ.get("HAY_CB01_BATCH_SIZE", "32")),
    epochs=int(os.environ.get("HAY_CB01_EPOCHS", "8")),
)
requested = os.environ.get("HAY_CB01_ARCHITECTURES", "").strip()
ARCHITECTURES = tuple(item.strip() for item in requested.split(",") if item.strip()) if requested else CAPABILITY_ARCHITECTURES
unknown = sorted(set(ARCHITECTURES) - set(CAPABILITY_ARCHITECTURES))
if unknown:
    raise ValueError(f"unknown HAY_CB01_ARCHITECTURES: {unknown}")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT = Path(os.environ.get("HAY_CB01_OUTPUT", "/kaggle/working/hay_micro_fast_slow_capability_15" if Path("/kaggle/working").exists() else "artifacts/cb01_fast_slow"))
FIGURES = OUTPUT / "figures"
CHECKPOINTS = OUTPUT / "checkpoints"
for directory in (OUTPUT, FIGURES, CHECKPOINTS):
    directory.mkdir(parents=True, exist_ok=True)

print("device       :", DEVICE)
print("output       :", OUTPUT.resolve())
print("architectures:", ", ".join(ARCHITECTURES))
print("[contract] synthetic train/validation only; Hay dataset and test are never opened", flush=True)


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class RunTracker:
    def __init__(self, total: int) -> None:
        self.total = total
        self.completed = 0
        self.started = time.perf_counter()
        self.bar = tqdm(total=total, desc="CB-01 model fits", unit="fit")

    def finish(self, label: str, metrics: dict[str, Any]) -> None:
        self.completed += 1
        elapsed = time.perf_counter() - self.started
        eta = elapsed / self.completed * max(0, self.total - self.completed)
        self.bar.update(1)
        self.bar.set_postfix_str(
            f"{label} | F1={float(metrics['spike_f1']):.3f} "
            f"event={float(metrics['event_voltage_rmse']):.3f} ETA={eta / 60:.1f}m"
        )

    def close(self) -> None:
        self.bar.close()


DATA: dict[str, dict[str, Any]] = {}
for index, difficulty in enumerate(CFG.difficulties):
    train = generate_fast_slow_sequences(CFG.train_sequences, CFG.train_steps, difficulty, CFG.data_seed + index)
    validation = generate_fast_slow_sequences(CFG.validation_sequences, CFG.validation_steps, difficulty, CFG.data_seed + 100 + index)
    train_targets = np.asarray(train["targets"], dtype=np.float32)
    mean = train_targets.reshape(-1, 3).mean(0).astype(np.float32)
    std = np.maximum(train_targets.reshape(-1, 3).std(0), 1e-5).astype(np.float32)
    DATA[difficulty] = {
        "train": train,
        "validation": validation,
        "mean": mean,
        "std": std,
        "train_normalized": ((train_targets - mean) / std).astype(np.float32),
    }
    print(
        f"[data:{difficulty}] train spikes={int(np.asarray(train['spikes']).sum())} "
        f"validation spikes={int(np.asarray(validation['spikes']).sum())}",
        flush=True,
    )


SIZES: dict[tuple[str, int], tuple[int, int]] = {}
for architecture in ARCHITECTURES:
    for budget in CFG.budgets:
        SIZES[(architecture, budget)] = width_for_budget(architecture, budget)
print("[capacity]", {f"{a}:{b}": {"width": SIZES[(a,b)][0], "parameters": SIZES[(a,b)][1]} for a in ARCHITECTURES for b in CFG.budgets}, flush=True)

hit_count = min(CFG.maximum_hits, len(ARCHITECTURES))
total_fits = (
    len(ARCHITECTURES) * len(CFG.learning_rates)
    + len(ARCHITECTURES) * len(CFG.budgets) * len(CFG.difficulties)
    + hit_count * len(CFG.replication_seeds) * len(CFG.budgets) * len(CFG.difficulties)
)
tracker = RunTracker(total_fits)
history_rows: list[dict[str, Any]] = []


def seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def predict(model: nn.Module, values: np.ndarray, batch_size: int) -> np.ndarray:
    model.eval()
    results = []
    for offset in range(0, len(values), batch_size):
        inputs = torch.as_tensor(values[offset:offset + batch_size], device=DEVICE)
        results.append(model(inputs).cpu().numpy())
    return np.concatenate(results)


def fit(
    architecture: str,
    budget: int,
    difficulty: str,
    seed: int,
    learning_rate: float,
    stage: str,
    save_checkpoint: bool = False,
) -> tuple[dict[str, Any], np.ndarray]:
    width, parameters = SIZES[(architecture, budget)]
    seed_everything(seed)
    model = build_capability_model(architecture, 3, 3, width).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=CFG.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CFG.epochs)
    train = DATA[difficulty]["train"]
    train_x = np.asarray(train["inputs"], dtype=np.float32)
    train_y = np.asarray(DATA[difficulty]["train_normalized"], dtype=np.float32)
    mean, std = DATA[difficulty]["mean"], DATA[difficulty]["std"]
    elapsed_start = time.perf_counter()
    for epoch in range(1, CFG.epochs + 1):
        model.train()
        order = np.random.default_rng(seed + epoch).permutation(len(train_x))
        total_loss = 0.0
        updates = 0
        for offset in range(0, len(order), CFG.batch_size):
            indices = order[offset:offset + CFG.batch_size]
            inputs = torch.as_tensor(train_x[indices], device=DEVICE)
            target = torch.as_tensor(train_y[indices], device=DEVICE)
            optimizer.zero_grad(set_to_none=True)
            output = model(inputs)
            loss = torch.mean(torch.square(output - target))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), CFG.gradient_clip)
            optimizer.step()
            total_loss += float(loss.detach())
            updates += 1
        scheduler.step()
        history_rows.append({
            "stage": stage,
            "architecture": architecture,
            "budget": budget,
            "width": width,
            "parameters": parameters,
            "difficulty": difficulty,
            "seed": seed,
            "learning_rate": learning_rate,
            "epoch": epoch,
            "train_normalized_mse": total_loss / max(1, updates),
            "elapsed_s": time.perf_counter() - elapsed_start,
        })
    validation = DATA[difficulty]["validation"]
    normalized = predict(model, np.asarray(validation["inputs"], dtype=np.float32), CFG.batch_size)
    physical = normalized * std + mean
    metrics = fast_slow_metrics(physical, validation, std)
    objective = float(np.mean(np.square((physical - np.asarray(validation["targets"])) / std)))
    row = {
        "stage": stage,
        "architecture": architecture,
        "budget": budget,
        "width": width,
        "parameters": parameters,
        "budget_relative_error": (parameters - budget) / budget,
        "difficulty": difficulty,
        "seed": seed,
        "learning_rate": learning_rate,
        "validation_normalized_mse": objective,
        **metrics,
        "fit_elapsed_s": time.perf_counter() - elapsed_start,
    }
    if save_checkpoint:
        path = CHECKPOINTS / f"{architecture}_b{budget}_{difficulty}_seed{seed}.pt"
        torch.save({
            "format_version": 1,
            "experiment_id": "CB-01",
            "architecture": architecture,
            "budget": budget,
            "width": width,
            "parameters": parameters,
            "difficulty": difficulty,
            "seed": seed,
            "learning_rate": learning_rate,
            "model_state_dict": model.state_dict(),
            "normalization_mean": mean,
            "normalization_std": std,
            "config": asdict(CFG),
        }, path)
    tracker.finish(f"{stage}:{architecture}/{budget}/{difficulty}/s{seed}", metrics)
    return row, physical


print("\n[1/3] equal-cost learning-rate calibration on medium difficulty", flush=True)
tuning_rows: list[dict[str, Any]] = []
for architecture in ARCHITECTURES:
    for learning_rate in CFG.learning_rates:
        row, _ = fit(architecture, CFG.budgets[-1], "medium", CFG.screen_seed, learning_rate, "lr_tuning")
        tuning_rows.append(row)
write_rows(OUTPUT / "learning_rate_tuning.csv", tuning_rows)

LEARNING_RATES = {}
for architecture in ARCHITECTURES:
    candidates = [row for row in tuning_rows if row["architecture"] == architecture]
    LEARNING_RATES[architecture] = float(min(candidates, key=lambda row: row["validation_normalized_mse"])["learning_rate"])
print("[learning rates]", LEARNING_RATES, flush=True)


print("\n[2/3] one-seed capacity x difficulty screen", flush=True)
screen_rows: list[dict[str, Any]] = []
for architecture in ARCHITECTURES:
    for budget in CFG.budgets:
        for difficulty in CFG.difficulties:
            row, _ = fit(architecture, budget, difficulty, CFG.screen_seed, LEARNING_RATES[architecture], "screen")
            screen_rows.append(row)
write_rows(OUTPUT / "screen_results.csv", screen_rows)
write_rows(OUTPUT / "training_history.csv", history_rows)


MECHANISM = {
    "mlp": "memoryless",
    "rnn": "discrete_recurrence", "gru": "discrete_recurrence", "lstm": "discrete_recurrence",
    "tcn": "causal_convolution", "transformer": "causal_attention",
    "cfc": "continuous_time", "ltc": "continuous_time",
    "conv_gru": "local_plus_recurrence", "conv_lstm": "local_plus_recurrence",
}

frame = pd.DataFrame(screen_rows)
large = frame[frame["budget"] == CFG.budgets[-1]].copy()
rank_columns = {
    "spike_f1": False,
    "event_voltage_rmse": True,
    "peak_amplitude_mae": True,
    "slow_nrmse": True,
    "false_spikes_per_1000_steps": True,
}
ranked = []
for architecture in ARCHITECTURES:
    rows = large[large["architecture"] == architecture]
    score = 0.0
    for metric, ascending in rank_columns.items():
        difficulty_means = large.groupby("architecture")[metric].mean()
        ranks = difficulty_means.rank(ascending=ascending, method="average")
        score += float(ranks[architecture])
    easy_f1 = float(rows[rows["difficulty"] == "easy"]["spike_f1"].iloc[0])
    medium_f1 = float(rows[rows["difficulty"] == "medium"]["spike_f1"].iloc[0])
    hard_f1 = float(rows[rows["difficulty"] == "hard"]["spike_f1"].iloc[0])
    ranked.append({
        "architecture": architecture,
        "mechanism": MECHANISM[architecture],
        "mean_rank_score": score / len(rank_columns),
        "easy_f1": easy_f1,
        "medium_f1": medium_f1,
        "hard_f1": hard_f1,
    })

group_winners = []
for mechanism in sorted({row["mechanism"] for row in ranked}):
    values = [row for row in ranked if row["mechanism"] == mechanism]
    group_winners.append(min(values, key=lambda row: row["mean_rank_score"]))
selected_rows = sorted(group_winners, key=lambda row: row["mean_rank_score"])[:hit_count]
HITS = tuple(row["architecture"] for row in selected_rows)
write_rows(OUTPUT / "hit_selection.csv", sorted(ranked, key=lambda row: row["mean_rank_score"]))
print("[hits]", HITS, flush=True)


print("\n[3/3] two additional paired seeds for the three diverse leads", flush=True)
replication_rows: list[dict[str, Any]] = []
hard_predictions: dict[str, np.ndarray] = {}
for architecture in HITS:
    for seed in CFG.replication_seeds:
        for budget in CFG.budgets:
            for difficulty in CFG.difficulties:
                save = budget == CFG.budgets[-1] and difficulty == "hard"
                row, prediction = fit(architecture, budget, difficulty, seed, LEARNING_RATES[architecture], "replication", save)
                replication_rows.append(row)
                if save and seed == CFG.replication_seeds[-1]:
                    hard_predictions[architecture] = prediction
write_rows(OUTPUT / "replication_results.csv", replication_rows)
write_rows(OUTPUT / "training_history.csv", history_rows)
tracker.close()


combined = pd.concat([
    frame[frame["architecture"].isin(HITS)],
    pd.DataFrame(replication_rows),
], ignore_index=True)
summary_rows = []
for architecture in HITS:
    for budget in CFG.budgets:
        for difficulty in CFG.difficulties:
            rows = combined[(combined.architecture == architecture) & (combined.budget == budget) & (combined.difficulty == difficulty)]
            summary_rows.append({
                "architecture": architecture,
                "mechanism": MECHANISM[architecture],
                "budget": budget,
                "difficulty": difficulty,
                "seeds": int(len(rows)),
                "spike_f1_mean": float(rows.spike_f1.mean()),
                "spike_f1_std": float(rows.spike_f1.std(ddof=1)),
                "event_voltage_rmse_mean": float(rows.event_voltage_rmse.mean()),
                "event_voltage_rmse_std": float(rows.event_voltage_rmse.std(ddof=1)),
                "slow_nrmse_mean": float(rows.slow_nrmse.mean()),
                "slow_nrmse_std": float(rows.slow_nrmse.std(ddof=1)),
                "false_spikes_per_1000_mean": float(rows.false_spikes_per_1000_steps.mean()),
                "peak_amplitude_mae_mean": float(rows.peak_amplitude_mae.mean()),
            })
write_rows(OUTPUT / "replicated_capability_surface.csv", summary_rows)


large_hard = [row for row in summary_rows if row["budget"] == CFG.budgets[-1] and row["difficulty"] == "hard"]
promoted = [
    row["architecture"] for row in large_hard
    if row["spike_f1_mean"] >= 0.20
    and row["slow_nrmse_mean"] <= 0.50
    and row["false_spikes_per_1000_mean"] <= 1.0
]
decision = {
    "experiment_id": "CB-01",
    "decision": "deconvolve_promoted_hits" if promoted else "no_hard_capability_hit",
    "screened_architectures": list(ARCHITECTURES),
    "selected_diverse_leads": list(HITS),
    "promoted_hits": promoted,
    "promotion_rule": {
        "large_budget_hard_spike_f1_mean_min": 0.20,
        "large_budget_hard_slow_nrmse_mean_max": 0.50,
        "large_budget_hard_false_spikes_per_1000_mean_max": 1.0,
    },
    "seeds_for_selected_leads": 3,
    "hay_dataset_opened": False,
    "hay_test_opened": False,
}
(OUTPUT / "decision.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")


fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for architecture in ARCHITECTURES:
    rows = large[large.architecture == architecture].set_index("difficulty").loc[list(CFG.difficulties)]
    axes[0].plot(CFG.difficulties, rows.spike_f1, marker="o", label=architecture)
    axes[1].plot(CFG.difficulties, rows.event_voltage_rmse, marker="o", label=architecture)
axes[0].set(title="one-seed large-budget spike F1", ylabel="F1")
axes[1].set(title="one-seed large-budget event voltage RMSE", ylabel="RMSE")
for axis in axes:
    axis.grid(alpha=0.25); axis.set_xlabel("difficulty")
axes[0].legend(ncol=2, fontsize=8)
fig.tight_layout(); fig.savefig(FIGURES / "screen_capability_surface.png", dpi=180); plt.close(fig)

fig, axis = plt.subplots(figsize=(8, 6))
hard = large[large.difficulty == "hard"]
for _, row in hard.iterrows():
    axis.scatter(row.slow_nrmse, row.event_voltage_rmse, s=40 + 180 * row.spike_f1, label=row.architecture)
    axis.annotate(row.architecture, (row.slow_nrmse, row.event_voltage_rmse), xytext=(4, 3), textcoords="offset points", fontsize=8)
axis.set(xlabel="slow NRMSE (lower is better)", ylabel="event voltage RMSE (lower is better)", title="hard-task Pareto view; marker size = spike F1")
axis.grid(alpha=0.25); fig.tight_layout(); fig.savefig(FIGURES / "hard_pareto.png", dpi=180); plt.close(fig)

truth = np.asarray(DATA["hard"]["validation"]["targets"])
truth_spikes = np.asarray(DATA["hard"]["validation"]["spikes"])
locations = np.argwhere(truth_spikes)
if len(locations) and hard_predictions:
    trajectory, center = locations[0]
    low, high = max(0, center - 80), min(truth.shape[1], center + 80)
    fig, axis = plt.subplots(figsize=(14, 5))
    axis.plot(np.arange(low, high), truth[trajectory, low:high, 1], color="black", linewidth=2, label="teacher")
    for architecture, prediction in hard_predictions.items():
        axis.plot(np.arange(low, high), prediction[trajectory, low:high, 1], label=architecture)
    axis.axvline(center, color="gray", linestyle="--", alpha=0.5)
    axis.set(xlabel="step", ylabel="fast voltage", title="hard-task first validation event")
    axis.grid(alpha=0.2); axis.legend(); fig.tight_layout(); fig.savefig(FIGURES / "hard_event_trace.png", dpi=180); plt.close(fig)

checkpoint_rows = []
for path in sorted(CHECKPOINTS.glob("*.pt")):
    checkpoint_rows.append({"file": str(path.relative_to(OUTPUT)).replace("\\", "/"), "sha256": file_sha256(path), "size_bytes": path.stat().st_size})
write_rows(OUTPUT / "checkpoint_manifest.csv", checkpoint_rows)

provenance = {
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "experiment_id": "CB-01",
    "device": str(DEVICE),
    "config": asdict(CFG),
    "architectures": list(ARCHITECTURES),
    "mechanism_groups": MECHANISM,
    "selected_learning_rates": LEARNING_RATES,
    "capacity_map": {f"{a}:{b}": {"width": SIZES[(a,b)][0], "parameters": SIZES[(a,b)][1]} for a in ARCHITECTURES for b in CFG.budgets},
    "difficulty_contracts": {name: asdict(spec) for name, spec in FAST_SLOW_DIFFICULTIES.items()},
    "data_counts": {name: {"train_spikes": int(np.asarray(DATA[name]["train"]["spikes"]).sum()), "validation_spikes": int(np.asarray(DATA[name]["validation"]["spikes"]).sum())} for name in CFG.difficulties},
    **decision,
}
(OUTPUT / "provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
(OUTPUT / "README.md").write_text(
    "# CB-01 fast/slow threshold capability bench\n\n"
    "Synthetic causal train/validation only; no Hay data or test split. Start with `decision.json`, "
    "`replicated_capability_surface.csv`, `hit_selection.csv`, and `figures/`.\n",
    encoding="utf-8",
)

zip_path = Path(make_archive(str(OUTPUT) + "_complete", "zip", root_dir=OUTPUT.parent, base_dir=OUTPUT.name))
ZIP_PATH = zip_path
print("decision:", json.dumps(decision, indent=2))
print("archive :", zip_path, f"({zip_path.stat().st_size / 2**20:.1f} MiB)")
