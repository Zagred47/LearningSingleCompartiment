"""HDF5 generation, validation, normalization, and sequence windows."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping, Sequence

import h5py
import numpy as np

from .config import SimulationConfig
from .protocols import REGIME_NAMES, RandomDrive
from .simulator import CURRENT_NAMES, INPUT_NAMES, STATE_NAMES, SingleCompartmentHay


SCHEMA_VERSION = "1.0.0"
SPLIT_OFFSETS = {"train": 0, "validation": 100_000, "test": 200_000}


def _json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _run_with_warmup(
    simulator: SingleCompartmentHay,
    drive_generator: RandomDrive,
    config: SimulationConfig,
    seed: int,
) -> Dict[str, np.ndarray]:
    warmup_steps = int(round(config.warmup_ms / config.dt_ms))
    data_steps = int(round(config.duration_ms / config.dt_ms))
    inputs, regimes = drive_generator.sample(warmup_steps + data_steps, config.dt_ms, seed)
    full = simulator.simulate(inputs, config.dt_ms, config.internal_dt_ms)
    start = warmup_steps
    return {
        "states": full["states"][start : start + data_steps + 1],
        "currents": full["currents"][start : start + data_steps + 1],
        "spikes": full["spikes"][start : start + data_steps],
        "inputs": inputs[start : start + data_steps],
        "regimes": regimes[start : start + data_steps],
    }


def generate_dataset(
    output_path: str | Path,
    config: SimulationConfig | None = None,
    *,
    progress: bool = False,
) -> Dict[str, object]:
    """Generate all splits into one portable, compressed HDF5 artifact."""

    config = config or SimulationConfig()
    config.validate()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    simulator = SingleCompartmentHay(config.membrane)
    drive_generator = RandomDrive(config.protocol)
    split_counts = {
        "train": config.train_trajectories,
        "validation": config.validation_trajectories,
        "test": config.test_trajectories,
    }
    steps = int(round(config.duration_ms / config.dt_ms))
    total_trajectories = sum(split_counts.values())
    completed_trajectories = 0
    started_at = time.perf_counter()

    def show_progress(split: str, index: int) -> None:
        if not progress:
            return
        elapsed = time.perf_counter() - started_at
        rate = completed_trajectories / max(elapsed, 1e-9)
        remaining = total_trajectories - completed_trajectories
        eta = remaining / max(rate, 1e-9)
        percentage = 100.0 * completed_trajectories / total_trajectories
        print(
            f"[dataset] {completed_trajectories:>3}/{total_trajectories} "
            f"({percentage:5.1f}%) | {split} trajectory {index + 1}/{split_counts[split]} "
            f"| elapsed {elapsed:6.1f}s | ETA {eta:6.1f}s",
            flush=True,
        )

    with h5py.File(output_path, "w") as handle:
        handle.attrs["schema_version"] = SCHEMA_VERSION
        handle.attrs["model"] = "reduced_hay_single_compartment"
        handle.attrs["config_json"] = _json(config.to_dict())
        handle.attrs["state_names_json"] = _json(STATE_NAMES)
        handle.attrs["input_names_json"] = _json(INPUT_NAMES)
        handle.attrs["current_names_json"] = _json(CURRENT_NAMES)
        handle.attrs["regime_names_json"] = _json(REGIME_NAMES)
        handle.create_dataset("time_ms", data=np.arange(steps + 1) * config.dt_ms)

        for split, count in split_counts.items():
            group = handle.create_group(split)
            arrays = []
            for index in range(count):
                arrays.append(_run_with_warmup(
                    simulator,
                    drive_generator,
                    config,
                    config.seed + SPLIT_OFFSETS[split] + index,
                ))
                completed_trajectories += 1
                show_progress(split, index)
            group.create_dataset(
                "states", data=np.stack([x["states"] for x in arrays]).astype("f4"),
                compression="gzip", shuffle=True,
            )
            group.create_dataset(
                "inputs", data=np.stack([x["inputs"] for x in arrays]).astype("f4"),
                compression="gzip", shuffle=True,
            )
            group.create_dataset(
                "currents", data=np.stack([x["currents"] for x in arrays]).astype("f4"),
                compression="gzip", shuffle=True,
            )
            group.create_dataset(
                "spikes", data=np.stack([x["spikes"] for x in arrays]),
                compression="gzip", shuffle=True,
            )
            group.create_dataset(
                "regimes", data=np.stack([x["regimes"] for x in arrays]),
                compression="gzip", shuffle=True,
            )
            group.create_dataset(
                "trajectory_seeds",
                data=np.arange(count, dtype=np.int64) + config.seed + SPLIT_OFFSETS[split],
            )

    report = validate_dataset(output_path)
    if not report["valid"]:
        raise ValueError(f"generated an invalid dataset: {report['issues']}")
    report["sha256"] = _sha256(output_path)
    report["path"] = str(output_path)
    manifest_path = output_path.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report


def validate_dataset(path: str | Path) -> Dict[str, object]:
    """Validate shape, finiteness, state bounds, and seed isolation."""

    path = Path(path)
    issues = []
    summary: Dict[str, object] = {"schema_version": SCHEMA_VERSION, "splits": {}}
    all_seeds = []
    try:
        with h5py.File(path, "r") as handle:
            if handle.attrs.get("schema_version") != SCHEMA_VERSION:
                issues.append("unexpected schema version")
            for split in SPLIT_OFFSETS:
                if split not in handle:
                    issues.append(f"missing split {split}")
                    continue
                group = handle[split]
                states = group["states"]
                inputs = group["inputs"]
                currents = group["currents"]
                spikes = group["spikes"]
                seeds = group["trajectory_seeds"][...]
                all_seeds.extend(int(seed) for seed in seeds)
                if states.shape[0] != inputs.shape[0] or states.shape[1] != inputs.shape[1] + 1:
                    issues.append(f"{split}: incompatible state/input shape")
                if states.shape[2] != len(STATE_NAMES) or inputs.shape[2] != len(INPUT_NAMES):
                    issues.append(f"{split}: feature width mismatch")
                if currents.shape != states.shape[:2] + (len(CURRENT_NAMES),):
                    issues.append(f"{split}: current shape mismatch")
                if spikes.shape != inputs.shape[:2]:
                    issues.append(f"{split}: spike shape mismatch")
                for name in ("states", "inputs", "currents"):
                    if not np.isfinite(group[name][...]).all():
                        issues.append(f"{split}/{name} contains NaN or Inf")
                gates = states[..., 2:14]
                if float(gates[:].min()) < 0.0 or float(gates[:].max()) > 1.0:
                    issues.append(f"{split}: gate outside [0, 1]")
                summary["splits"][split] = {
                    "trajectories": int(states.shape[0]),
                    "steps": int(inputs.shape[1]),
                    "spikes": int(spikes[...].sum()),
                    "voltage_min_mv": float(states[..., 0].min()),
                    "voltage_max_mv": float(states[..., 0].max()),
                }
    except (OSError, KeyError) as error:
        issues.append(str(error))
    if len(all_seeds) != len(set(all_seeds)):
        issues.append("trajectory seeds overlap across splits")
    summary["valid"] = not issues
    summary["issues"] = issues
    return summary


@dataclass
class Normalization:
    state_mean: np.ndarray
    state_std: np.ndarray
    input_mean: np.ndarray
    input_std: np.ndarray

    @classmethod
    def from_h5(cls, path: str | Path) -> "Normalization":
        with h5py.File(path, "r") as handle:
            states = handle["train/states"][...].astype(np.float64)
            inputs = handle["train/inputs"][...].astype(np.float64)
        state_flat = states.reshape(-1, states.shape[-1])
        input_flat = inputs.reshape(-1, inputs.shape[-1])
        return cls(
            state_mean=state_flat.mean(0),
            state_std=np.maximum(state_flat.std(0), 1e-7),
            input_mean=input_flat.mean(0),
            input_std=np.maximum(input_flat.std(0), 1e-7),
        )

    def to_dict(self) -> Dict[str, list[float]]:
        return {name: getattr(self, name).tolist() for name in self.__dataclass_fields__}

    @classmethod
    def from_dict(cls, values: Mapping[str, Sequence[float]]) -> "Normalization":
        return cls(**{name: np.asarray(values[name]) for name in cls.__dataclass_fields__})


class SequenceWindowDataset:
    """Lazy PyTorch dataset of non-crossing trajectory windows."""

    def __init__(
        self,
        path: str | Path,
        split: str,
        normalization: Normalization,
        sequence_length: int = 64,
        stride: int = 16,
    ) -> None:
        import torch

        self.torch = torch
        with h5py.File(path, "r") as handle:
            self.states = handle[f"{split}/states"][...]
            self.inputs = handle[f"{split}/inputs"][...]
        if sequence_length < 1 or sequence_length > self.inputs.shape[1]:
            raise ValueError("invalid sequence length")
        self.normalization = normalization
        self.sequence_length = sequence_length
        self.indices = [
            (trajectory, start)
            for trajectory in range(self.inputs.shape[0])
            for start in range(0, self.inputs.shape[1] - sequence_length + 1, stride)
        ]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int):
        trajectory, start = self.indices[index]
        end = start + self.sequence_length
        state_t = (self.states[trajectory, start:end] - self.normalization.state_mean) / self.normalization.state_std
        state_next = (self.states[trajectory, start + 1 : end + 1] - self.normalization.state_mean) / self.normalization.state_std
        inputs = (self.inputs[trajectory, start:end] - self.normalization.input_mean) / self.normalization.input_std
        features = np.concatenate([state_t, inputs], axis=-1).astype(np.float32)
        return self.torch.from_numpy(features), self.torch.from_numpy(state_next.astype(np.float32))
