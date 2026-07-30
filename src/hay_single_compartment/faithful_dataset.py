"""Balanced protocols and cacheable HDF5 datasets for the faithful Hay soma."""

from __future__ import annotations

import hashlib
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict

import h5py
import numpy as np

from .faithful import (
    FAITHFUL_CURRENT_NAMES,
    FAITHFUL_INPUT_NAMES,
    FAITHFUL_STATE_NAMES,
    FaithfulHaySoma,
    FaithfulProtocolConfig,
    FaithfulSimulationConfig,
)
from .protocols import REGIME_NAMES, REGIME_RATES_HZ


FAITHFUL_SCHEMA_VERSION = "2.0.0"
FAITHFUL_MODEL_ID = "hay_2011_figure4_faithful_soma_plus_synapses"
FAITHFUL_SPLIT_OFFSETS = {"train": 0, "validation": 100_000, "test": 200_000}


def _json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class BalancedFaithfulDrive:
    """Random drive with shuffled cycles that contain every regime once."""

    regime_bias_uA_cm2 = np.asarray((-0.40, 0.25, 1.60, -0.40, 3.10))
    pulse_rate_hz = np.asarray((2.0, 10.0, 35.0, 2.0, 80.0))

    def __init__(self, config: FaithfulProtocolConfig | None = None) -> None:
        self.config = config or FaithfulProtocolConfig()

    def _regime_schedule(self, steps: int, dt_ms: float, rng: np.random.Generator) -> np.ndarray:
        schedule = []
        while sum(len(block) for block in schedule) < steps:
            for regime in rng.permutation(len(REGIME_NAMES)):
                duration_ms = rng.uniform(self.config.min_regime_ms, self.config.max_regime_ms)
                count = max(1, int(round(duration_ms / dt_ms)))
                schedule.append(np.full(count, regime, dtype=np.int8))
        return np.concatenate(schedule)[:steps]

    def sample(self, steps: int, dt_ms: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
        if steps < 1 or dt_ms <= 0.0:
            raise ValueError("steps and dt_ms must be positive")
        rng = np.random.default_rng(seed)
        regimes = self._regime_schedule(steps, dt_ms, rng)
        result = np.zeros((steps, len(FAITHFUL_INPUT_NAMES)), dtype=np.float64)
        current = self.config.ou_mean_uA_cm2
        ou_decay = np.exp(-dt_ms / self.config.ou_tau_ms)
        ou_scale = self.config.ou_sigma_uA_cm2 * np.sqrt(1.0 - ou_decay**2)
        pulse_steps_remaining = 0
        pulse_amplitude = 0.0

        for index, regime in enumerate(regimes):
            target = self.config.ou_mean_uA_cm2 + self.regime_bias_uA_cm2[regime]
            current = target + (current - target) * ou_decay + ou_scale * rng.normal()
            if (
                pulse_steps_remaining == 0
                and rng.random() < self.pulse_rate_hz[regime] * dt_ms / 1000.0
            ):
                pulse_steps_remaining = max(1, int(round(rng.uniform(6.0, 14.0) / dt_ms)))
                pulse_amplitude = float(rng.uniform(4.0, 6.0))
            pulse = pulse_amplitude if pulse_steps_remaining > 0 else 0.0
            pulse_steps_remaining = max(0, pulse_steps_remaining - 1)
            result[index, 0] = np.clip(
                current + pulse,
                self.config.current_min_uA_cm2,
                self.config.current_max_uA_cm2,
            )
            result[index, 1:] = rng.poisson(REGIME_RATES_HZ[regime] * dt_ms / 1000.0)
        return result, regimes


def _run_with_warmup(
    simulator: FaithfulHaySoma,
    drive_generator: BalancedFaithfulDrive,
    config: FaithfulSimulationConfig,
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


def _run_seed(config: FaithfulSimulationConfig, seed: int) -> Dict[str, np.ndarray]:
    """Pickle-friendly trajectory entry point for parallel dataset generation."""

    return _run_with_warmup(
        FaithfulHaySoma(config.membrane),
        BalancedFaithfulDrive(config.protocol),
        config,
        seed,
    )


def validate_faithful_dataset(path: str | Path) -> Dict[str, object]:
    path = Path(path)
    issues = []
    summary: Dict[str, object] = {
        "schema_version": FAITHFUL_SCHEMA_VERSION,
        "model": FAITHFUL_MODEL_ID,
        "splits": {},
    }
    all_seeds = []
    try:
        with h5py.File(path, "r") as handle:
            if handle.attrs.get("schema_version") != FAITHFUL_SCHEMA_VERSION:
                issues.append("unexpected schema version")
            if handle.attrs.get("model") != FAITHFUL_MODEL_ID:
                issues.append("unexpected model")
            for split in FAITHFUL_SPLIT_OFFSETS:
                if split not in handle:
                    issues.append(f"missing split {split}")
                    continue
                group = handle[split]
                states = group["states"]
                inputs = group["inputs"]
                currents = group["currents"]
                spikes = group["spikes"]
                regimes = group["regimes"]
                seeds = group["trajectory_seeds"][...]
                all_seeds.extend(int(seed) for seed in seeds)
                if states.shape[:2] != (inputs.shape[0], inputs.shape[1] + 1):
                    issues.append(f"{split}: incompatible state/input shape")
                if states.shape[2] != len(FAITHFUL_STATE_NAMES):
                    issues.append(f"{split}: state width mismatch")
                if inputs.shape[2] != len(FAITHFUL_INPUT_NAMES):
                    issues.append(f"{split}: input width mismatch")
                if currents.shape != states.shape[:2] + (len(FAITHFUL_CURRENT_NAMES),):
                    issues.append(f"{split}: current shape mismatch")
                if spikes.shape != inputs.shape[:2] or regimes.shape != inputs.shape[:2]:
                    issues.append(f"{split}: event/regime shape mismatch")
                for name in ("states", "inputs", "currents"):
                    if not np.isfinite(group[name][...]).all():
                        issues.append(f"{split}/{name} contains NaN or Inf")
                gate_min = float(states[..., 2:17].min())
                gate_max = float(states[..., 2:17].max())
                if gate_min < 0.0 or gate_max > 1.0:
                    issues.append(f"{split}: gate outside [0, 1]")
                regime_counts = np.bincount(regimes[...].reshape(-1), minlength=len(REGIME_NAMES))
                if np.any(regime_counts == 0):
                    issues.append(f"{split}: missing drive regime")
                summary["splits"][split] = {
                    "trajectories": int(states.shape[0]),
                    "steps": int(inputs.shape[1]),
                    "spikes": int(spikes[...].sum()),
                    "voltage_min_mv": float(states[..., 0].min()),
                    "voltage_max_mv": float(states[..., 0].max()),
                    "regime_counts": {
                        name: int(value) for name, value in zip(REGIME_NAMES, regime_counts)
                    },
                    "regime_fractions": {
                        name: float(value / regime_counts.sum())
                        for name, value in zip(REGIME_NAMES, regime_counts)
                    },
                }
    except (OSError, KeyError) as error:
        issues.append(str(error))
    if len(all_seeds) != len(set(all_seeds)):
        issues.append("trajectory seeds overlap across splits")
    summary["valid"] = not issues
    summary["issues"] = issues
    return summary


def generate_faithful_dataset(
    output_path: str | Path,
    config: FaithfulSimulationConfig | None = None,
    *,
    progress: bool = False,
    reuse: bool = True,
    force: bool = False,
    workers: int = 1,
) -> Dict[str, object]:
    """Generate, validate and cache a portable faithful-soma HDF5 dataset.

    A compatible existing file is reused.  A configuration mismatch raises
    unless ``force=True`` so an expensive dataset is never silently replaced.
    """

    config = config or FaithfulSimulationConfig()
    config.validate()
    if workers < 1:
        raise ValueError("workers must be positive")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    config_json = _json(config.to_dict())

    if output_path.exists() and reuse and not force:
        try:
            with h5py.File(output_path, "r") as handle:
                compatible = (
                    handle.attrs.get("schema_version") == FAITHFUL_SCHEMA_VERSION
                    and handle.attrs.get("model") == FAITHFUL_MODEL_ID
                    and handle.attrs.get("config_json") == config_json
                )
        except OSError:
            compatible = False
        if not compatible:
            raise ValueError(
                f"cached dataset {output_path} is incompatible; use a new path or force=True"
            )
        report = validate_faithful_dataset(output_path)
        if not report["valid"]:
            raise ValueError(f"cached dataset is invalid: {report['issues']}")
        report.update({"cache_hit": True, "path": str(output_path), "sha256": _sha256(output_path)})
        if progress:
            print(f"[dataset] cache hit: {output_path}", flush=True)
        return report

    split_counts = {
        "train": config.train_trajectories,
        "validation": config.validation_trajectories,
        "test": config.test_trajectories,
    }
    steps = int(round(config.duration_ms / config.dt_ms))
    total = sum(split_counts.values())
    completed = 0
    started_at = time.perf_counter()
    temporary_path = output_path.with_suffix(output_path.suffix + ".partial")

    with h5py.File(temporary_path, "w") as handle:
        handle.attrs["schema_version"] = FAITHFUL_SCHEMA_VERSION
        handle.attrs["model"] = FAITHFUL_MODEL_ID
        handle.attrs["config_json"] = config_json
        handle.attrs["state_names_json"] = _json(FAITHFUL_STATE_NAMES)
        handle.attrs["input_names_json"] = _json(FAITHFUL_INPUT_NAMES)
        handle.attrs["current_names_json"] = _json(FAITHFUL_CURRENT_NAMES)
        handle.attrs["regime_names_json"] = _json(REGIME_NAMES)
        handle.attrs["intrinsic_source"] = "Hay et al. 2011 ModelDB 139653 L5PCbiophys3 soma"
        handle.create_dataset("time_ms", data=np.arange(steps + 1) * config.dt_ms)

        for split, count in split_counts.items():
            group = handle.create_group(split)
            arrays: list[Dict[str, np.ndarray] | None] = [None] * count

            def trajectory_finished(index: int, result: Dict[str, np.ndarray]) -> None:
                nonlocal completed
                arrays[index] = result
                completed += 1
                if progress:
                    elapsed = time.perf_counter() - started_at
                    eta = (total - completed) * elapsed / completed
                    print(
                        f"[dataset] {completed:>3}/{total} ({100.0 * completed / total:5.1f}%) "
                        f"| {split} {index + 1}/{count} | elapsed {elapsed:6.1f}s | ETA {eta:6.1f}s",
                        flush=True,
                    )

            if workers == 1:
                for index in range(count):
                    seed = config.seed + FAITHFUL_SPLIT_OFFSETS[split] + index
                    trajectory_finished(index, _run_seed(config, seed))
            else:
                with ProcessPoolExecutor(max_workers=workers) as executor:
                    futures = {
                        executor.submit(
                            _run_seed,
                            config,
                            config.seed + FAITHFUL_SPLIT_OFFSETS[split] + index,
                        ): index
                        for index in range(count)
                    }
                    for future in as_completed(futures):
                        trajectory_finished(futures[future], future.result())
            completed_arrays = [item for item in arrays if item is not None]
            if len(completed_arrays) != count:
                raise RuntimeError(f"{split}: incomplete trajectory generation")
            for name, dtype in (
                ("states", "f4"), ("inputs", "f4"), ("currents", "f4"),
                ("spikes", "u1"), ("regimes", "i1"),
            ):
                group.create_dataset(
                    name,
                    data=np.stack([item[name] for item in completed_arrays]).astype(dtype),
                    compression="gzip",
                    shuffle=True,
                )
            group.create_dataset(
                "trajectory_seeds",
                data=np.arange(count, dtype=np.int64)
                + config.seed + FAITHFUL_SPLIT_OFFSETS[split],
            )

    temporary_path.replace(output_path)
    report = validate_faithful_dataset(output_path)
    if not report["valid"]:
        raise ValueError(f"generated an invalid dataset: {report['issues']}")
    report.update({"cache_hit": False, "path": str(output_path), "sha256": _sha256(output_path)})
    output_path.with_suffix(".manifest.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    return report
