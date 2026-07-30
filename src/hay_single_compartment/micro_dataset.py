"""Balanced, cacheable spike-only datasets for the four-compartment Hay teacher."""

from __future__ import annotations

import hashlib
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Mapping, Sequence

import h5py
import numpy as np

from .micro_neuron import (
    MICRO_STATE_NAMES,
    SYNAPTIC_REGIONS,
    FourCompartmentHay,
    MicroHayConfig,
)


MICRO_SCHEMA_VERSION = "1.2.0"
MICRO_MODEL_ID = "hay_2011_four_compartment_spike_driven_v1"
MICRO_REGIME_NAMES = (
    "quiet_recovery",
    "sparse_subthreshold",
    "balanced_async",
    "inhibition_dominant",
    "excitation_dominant",
    "basal_burst",
    "trunk_burst",
    "tuft_nmda_burst",
    "global_correlated_burst",
)
MICRO_SPLIT_OFFSETS = {"train": 0, "validation": 100_000, "test": 200_000}


@dataclass(frozen=True)
class MicroDriveConfig:
    excitatory_synapses: int = 16
    inhibitory_synapses: int = 8
    min_regime_ms: float = 60.0
    max_regime_ms: float = 160.0
    smoothing_tau_min_ms: float = 10.0
    smoothing_tau_max_ms: float = 120.0
    shared_rate_modulation_sigma: float = 0.25
    private_rate_jitter_sigma: float = 0.12


@dataclass(frozen=True)
class MicroDatasetConfig:
    dt_ms: float = 0.1
    internal_dt_ms: float = 0.025
    duration_ms: float = 5000.0
    warmup_ms: float = 2000.0
    seed: int = 314159
    train_trajectories: int = 20
    validation_trajectories: int = 4
    test_trajectories: int = 6
    teacher: MicroHayConfig = field(default_factory=MicroHayConfig)
    drive: MicroDriveConfig = field(default_factory=MicroDriveConfig)

    def validate(self) -> None:
        if self.dt_ms <= 0.0 or self.internal_dt_ms <= 0.0:
            raise ValueError("time steps must be positive")
        ratio = self.dt_ms / self.internal_dt_ms
        if abs(ratio - round(ratio)) > 1.0e-12:
            raise ValueError("dt_ms must be an integer multiple of internal_dt_ms")
        if self.duration_ms <= 0.0 or self.warmup_ms < 0.0:
            raise ValueError("duration must be positive and warmup non-negative")
        if min(self.train_trajectories, self.validation_trajectories, self.test_trajectories) < 1:
            raise ValueError("every split needs at least one trajectory")
        if min(self.drive.excitatory_synapses, self.drive.inhibitory_synapses) < len(SYNAPTIC_REGIONS):
            raise ValueError("each dendritic region needs at least one synapse of each type")

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def _json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _allocate_by_length(total: int, lengths: np.ndarray) -> np.ndarray:
    """Largest-remainder allocation with at least one item per region."""

    if total < len(lengths):
        raise ValueError("total must allow one synapse per region")
    remaining = total - len(lengths)
    ideal = remaining * lengths / lengths.sum()
    allocated = np.ones(len(lengths), dtype=int) + np.floor(ideal).astype(int)
    leftovers = total - int(allocated.sum())
    if leftovers:
        order = np.argsort(-(ideal - np.floor(ideal)))
        allocated[order[:leftovers]] += 1
    return allocated


def build_micro_synapse_metadata(config: MicroDatasetConfig | None = None) -> tuple[dict[str, object], ...]:
    config = config or MicroDatasetConfig()
    lengths = np.asarray([
        getattr(config.teacher.geometry, region).represented_length_um
        for region in SYNAPTIC_REGIONS
    ], dtype=float)
    rows = []
    next_id = 0
    for kind, total in (
        ("excitatory", config.drive.excitatory_synapses),
        ("inhibitory", config.drive.inhibitory_synapses),
    ):
        allocation = _allocate_by_length(total, lengths)
        for region, count, represented_length in zip(SYNAPTIC_REGIONS, allocation, lengths):
            for local_index in range(int(count)):
                rows.append({
                    "synapse_id": next_id,
                    "name": f"spike.{kind}.{region}.{local_index:02d}",
                    "kind": kind,
                    "region": region,
                    "compartment": region,
                    "local_index": local_index,
                    "represented_length_um": float(represented_length),
                })
                next_id += 1
    return tuple(rows)


def micro_input_names(config: MicroDatasetConfig | None = None) -> tuple[str, ...]:
    return tuple(str(row["name"]) for row in build_micro_synapse_metadata(config))


class BalancedSpatialSpikeDrive:
    """Binary presynaptic spike matrix with length-weighted fixed locations."""

    # Per-synapse firing rates in Hz before regional multipliers.
    _base_e = np.asarray((0.2, 3.0, 14.0, 8.0, 38.0, 22.0, 22.0, 24.0, 65.0))
    _base_i = np.asarray((0.2, 3.0, 16.0, 42.0, 8.0, 16.0, 16.0, 14.0, 38.0))
    _regional_e = np.asarray([
        (1.0, 1.0, 1.0),
        (1.0, 1.0, 1.0),
        (1.0, 1.0, 1.0),
        (1.0, 1.0, 1.0),
        (1.0, 1.0, 1.0),
        (4.0, 0.35, 0.25),
        (0.35, 4.0, 0.35),
        (0.20, 0.55, 5.0),
        (1.0, 1.0, 1.0),
    ])
    _regional_i = np.asarray([
        (1.0, 1.0, 1.0),
        (1.0, 1.0, 1.0),
        (1.0, 1.0, 1.0),
        (1.0, 1.0, 1.0),
        (1.0, 1.0, 1.0),
        (1.6, 0.6, 0.5),
        (0.6, 1.6, 0.6),
        (0.5, 0.7, 1.4),
        (1.0, 1.0, 1.0),
    ])

    def __init__(self, config: MicroDatasetConfig | None = None) -> None:
        self.config = config or MicroDatasetConfig()
        self.metadata = build_micro_synapse_metadata(self.config)

    def _schedule(self, steps: int, dt_ms: float, rng: np.random.Generator) -> np.ndarray:
        blocks = []
        filled = 0
        while filled < steps:
            for regime in rng.permutation(len(MICRO_REGIME_NAMES)):
                duration = rng.uniform(self.config.drive.min_regime_ms, self.config.drive.max_regime_ms)
                count = max(1, int(round(duration / dt_ms)))
                blocks.append(np.full(count, regime, dtype=np.int8))
                filled += count
                if filled >= steps:
                    break
        return np.concatenate(blocks)[:steps]

    def sample(self, steps: int, dt_ms: float, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if steps < 1 or dt_ms <= 0.0:
            raise ValueError("steps and dt_ms must be positive")
        rng = np.random.default_rng(seed)
        regimes = self._schedule(steps, dt_ms, rng)
        target_rates = np.empty((steps, len(self.metadata)), dtype=np.float64)
        for synapse_index, row in enumerate(self.metadata):
            region_index = SYNAPTIC_REGIONS.index(str(row["region"]))
            if row["kind"] == "excitatory":
                target_rates[:, synapse_index] = self._base_e[regimes] * self._regional_e[regimes, region_index]
            else:
                target_rates[:, synapse_index] = self._base_i[regimes] * self._regional_i[regimes, region_index]

        tau = rng.uniform(self.config.drive.smoothing_tau_min_ms, self.config.drive.smoothing_tau_max_ms)
        decay = np.exp(-dt_ms / tau)
        rates = np.empty_like(target_rates)
        smoothed_rates = target_rates[0].copy()
        rates[0] = smoothed_rates
        shared = 0.0
        shared_decay = np.exp(-dt_ms / rng.uniform(50.0, 500.0))
        shared_scale = self.config.drive.shared_rate_modulation_sigma * np.sqrt(1.0 - shared_decay**2)
        for index in range(1, steps):
            shared = shared_decay * shared + shared_scale * rng.normal()
            private = self.config.drive.private_rate_jitter_sigma * rng.normal(size=len(self.metadata))
            smoothed_rates = target_rates[index] + (smoothed_rates - target_rates[index]) * decay
            rates[index] = smoothed_rates * np.exp(np.clip(shared + private, -1.5, 1.5))
        rates = np.clip(rates, 0.0, 500.0)
        probability = 1.0 - np.exp(-rates * dt_ms / 1000.0)
        spikes = (rng.random(probability.shape) < probability).astype(np.uint8)

        # The global burst contains a weak common event source while remaining binary.
        global_mask = regimes == MICRO_REGIME_NAMES.index("global_correlated_burst")
        common = (rng.random(steps) < (1.0 - np.exp(-18.0 * dt_ms / 1000.0))) & global_mask
        if common.any():
            participation = rng.random((int(common.sum()), len(self.metadata))) < 0.45
            spikes[common] = np.maximum(spikes[common], participation.astype(np.uint8))
        return spikes, regimes, rates.astype(np.float32)


def _run_seed(config: MicroDatasetConfig, seed: int) -> Dict[str, np.ndarray]:
    teacher = FourCompartmentHay(config.teacher)
    drive = BalancedSpatialSpikeDrive(config)
    warmup_steps = int(round(config.warmup_ms / config.dt_ms))
    data_steps = int(round(config.duration_ms / config.dt_ms))
    binary, regimes, rates = drive.sample(warmup_steps + data_steps, config.dt_ms, seed)
    simulation = teacher.simulate(binary, drive.metadata, config.dt_ms, config.internal_dt_ms)
    start = warmup_steps
    return {
        "burnin_inputs": binary[:start],
        "burnin_regimes": regimes[:start],
        "burnin_states": simulation["states"][: start + 1],
        "states": simulation["states"][start : start + data_steps + 1],
        "currents": simulation["currents"][start : start + data_steps + 1],
        "inputs": binary[start : start + data_steps],
        "event_counts": simulation["event_counts"][start : start + data_steps],
        "spikes": simulation["spikes"][start : start + data_steps],
        "regimes": regimes[start : start + data_steps],
        "instantaneous_rates_hz": rates[start : start + data_steps],
    }


def validate_micro_dataset(path: str | Path) -> Dict[str, object]:
    path = Path(path)
    issues = []
    summary: Dict[str, object] = {"schema_version": MICRO_SCHEMA_VERSION, "model": MICRO_MODEL_ID, "splits": {}}
    all_seeds = []
    try:
        with h5py.File(path, "r") as handle:
            if handle.attrs.get("schema_version") != MICRO_SCHEMA_VERSION:
                issues.append("unexpected schema version")
            if handle.attrs.get("model") != MICRO_MODEL_ID:
                issues.append("unexpected model")
            state_names = json.loads(handle.attrs["state_names_json"])
            input_metadata = json.loads(handle.attrs["input_metadata_json"])
            stored_config = json.loads(handle.attrs["config_json"])
            expected_burnin = int(round(stored_config["warmup_ms"] / stored_config["dt_ms"]))
            for split in MICRO_SPLIT_OFFSETS:
                if split not in handle:
                    issues.append(f"missing split {split}")
                    continue
                group = handle[split]
                states, inputs = group["states"], group["inputs"]
                burnin_inputs = group["burnin_inputs"]
                burnin_states = group["burnin_states"]
                burnin_regimes = group["burnin_regimes"]
                currents, counts = group["currents"], group["event_counts"]
                regimes, spikes = group["regimes"], group["spikes"]
                seeds = group["trajectory_seeds"][...]
                all_seeds.extend(map(int, seeds))
                if states.shape[:2] != (inputs.shape[0], inputs.shape[1] + 1):
                    issues.append(f"{split}: incompatible state/input shape")
                if states.shape[2] != len(state_names) or inputs.shape[2] != len(input_metadata):
                    issues.append(f"{split}: schema width mismatch")
                if counts.shape != inputs.shape[:2] + (6,):
                    issues.append(f"{split}: event count shape mismatch")
                if regimes.shape != inputs.shape[:2] or spikes.shape != inputs.shape[:2]:
                    issues.append(f"{split}: label shape mismatch")
                if burnin_inputs.shape != (inputs.shape[0], expected_burnin, inputs.shape[2]):
                    issues.append(f"{split}: burn-in input shape mismatch")
                if burnin_states.shape != (inputs.shape[0], expected_burnin + 1, states.shape[2]):
                    issues.append(f"{split}: burn-in state shape mismatch")
                if burnin_regimes.shape != burnin_inputs.shape[:2]:
                    issues.append(f"{split}: burn-in regime shape mismatch")
                if not np.isin(inputs[...], (0, 1)).all():
                    issues.append(f"{split}: inputs are not binary")
                if not np.isin(burnin_inputs[...], (0, 1)).all():
                    issues.append(f"{split}: burn-in inputs are not binary")
                for name in ("burnin_states", "states", "currents", "instantaneous_rates_hz"):
                    if not np.isfinite(group[name][...]).all():
                        issues.append(f"{split}/{name} contains NaN or Inf")
                regime_counts = np.bincount(regimes[...].reshape(-1), minlength=len(MICRO_REGIME_NAMES))
                if np.any(regime_counts == 0):
                    issues.append(f"{split}: missing drive regime")
                voltage_indices = [i for i, name in enumerate(state_names) if name.endswith(".v_mV")]
                gate_indices = [i for i, name in enumerate(state_names) if any(token in name for token in (".m_", ".h_", ".z_"))]
                gate_values = states[..., gate_indices]
                if float(gate_values.min()) < 0.0 or float(gate_values.max()) > 1.0:
                    issues.append(f"{split}: gate outside [0,1]")
                summary["splits"][split] = {
                    "trajectories": int(inputs.shape[0]),
                    "steps": int(inputs.shape[1]),
                    "presynaptic_spikes": int(inputs[...].sum()),
                    "somatic_spikes": int(spikes[...].sum()),
                    "voltage_min_mV": float(states[..., voltage_indices].min()),
                    "voltage_max_mV": float(states[..., voltage_indices].max()),
                    "regime_counts": {name: int(value) for name, value in zip(MICRO_REGIME_NAMES, regime_counts)},
                }
    except (OSError, KeyError, ValueError) as error:
        issues.append(str(error))
    if len(all_seeds) != len(set(all_seeds)):
        issues.append("trajectory seeds overlap across splits")
    summary["valid"] = not issues
    summary["issues"] = issues
    return summary


def generate_micro_dataset(
    output_path: str | Path,
    config: MicroDatasetConfig | None = None,
    *,
    progress: bool = False,
    reuse: bool = True,
    force: bool = False,
    workers: int = 1,
) -> Dict[str, object]:
    """Generate, validate and cache the complete four-compartment dataset."""

    config = config or MicroDatasetConfig()
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
                    handle.attrs.get("schema_version") == MICRO_SCHEMA_VERSION
                    and handle.attrs.get("model") == MICRO_MODEL_ID
                    and handle.attrs.get("config_json") == config_json
                )
        except OSError:
            compatible = False
        if not compatible:
            raise ValueError(f"cached dataset {output_path} is incompatible; use a new path or force=True")
        report = validate_micro_dataset(output_path)
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
    metadata = build_micro_synapse_metadata(config)
    teacher = FourCompartmentHay(config.teacher)
    steps = int(round(config.duration_ms / config.dt_ms))
    total = sum(split_counts.values())
    completed = 0
    started_at = time.perf_counter()
    partial = output_path.with_suffix(output_path.suffix + ".partial")
    with h5py.File(partial, "w") as handle:
        handle.attrs["schema_version"] = MICRO_SCHEMA_VERSION
        handle.attrs["model"] = MICRO_MODEL_ID
        handle.attrs["config_json"] = config_json
        handle.attrs["state_names_json"] = _json(MICRO_STATE_NAMES)
        handle.attrs["current_names_json"] = _json(teacher.current_names)
        handle.attrs["input_names_json"] = _json(tuple(row["name"] for row in metadata))
        handle.attrs["input_metadata_json"] = _json(metadata)
        handle.attrs["regime_names_json"] = _json(MICRO_REGIME_NAMES)
        handle.attrs["teacher_source"] = "Hay et al. 2011 ModelDB 139653; four-compartment reduction"
        handle.attrs["external_current_injection"] = False
        handle.create_dataset("time_ms", data=np.arange(steps + 1) * config.dt_ms)
        for split, count in split_counts.items():
            group = handle.create_group(split)
            arrays: list[Dict[str, np.ndarray] | None] = [None] * count

            def finished(index: int, result: Dict[str, np.ndarray]) -> None:
                nonlocal completed
                arrays[index] = result
                completed += 1
                if progress:
                    elapsed = time.perf_counter() - started_at
                    eta = (total - completed) * elapsed / completed
                    print(
                        f"[dataset] {completed:>3}/{total} ({100.0 * completed / total:5.1f}%) "
                        f"| {split} {index + 1}/{count} | elapsed {elapsed:7.1f}s | ETA {eta:7.1f}s",
                        flush=True,
                    )

            if workers == 1:
                for index in range(count):
                    finished(index, _run_seed(config, config.seed + MICRO_SPLIT_OFFSETS[split] + index))
            else:
                with ProcessPoolExecutor(max_workers=workers) as executor:
                    futures = {
                        executor.submit(_run_seed, config, config.seed + MICRO_SPLIT_OFFSETS[split] + index): index
                        for index in range(count)
                    }
                    for future in as_completed(futures):
                        finished(futures[future], future.result())
            completed_arrays = [item for item in arrays if item is not None]
            if len(completed_arrays) != count:
                raise RuntimeError(f"{split}: incomplete trajectory generation")
            for name, dtype in (
                ("burnin_inputs", "u1"), ("burnin_regimes", "i1"),
                ("burnin_states", "f4"),
                ("states", "f4"), ("currents", "f4"), ("inputs", "u1"),
                ("event_counts", "u2"), ("spikes", "u1"), ("regimes", "i1"),
                ("instantaneous_rates_hz", "f4"),
            ):
                group.create_dataset(
                    name,
                    data=np.stack([item[name] for item in completed_arrays]).astype(dtype),
                    compression="gzip",
                    shuffle=True,
                )
            group.create_dataset(
                "trajectory_seeds",
                data=np.arange(count, dtype=np.int64) + config.seed + MICRO_SPLIT_OFFSETS[split],
            )
    partial.replace(output_path)
    report = validate_micro_dataset(output_path)
    if not report["valid"]:
        raise ValueError(f"generated an invalid dataset: {report['issues']}")
    report.update({"cache_hit": False, "path": str(output_path), "sha256": _sha256(output_path)})
    output_path.with_suffix(".manifest.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    return report
