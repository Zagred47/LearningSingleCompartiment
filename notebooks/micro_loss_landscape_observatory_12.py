"""DG-01: read-only loss-landscape observatory for factorial-11 checkpoints.

The experiment opens only the validation split, performs no optimizer step and
never reads test trajectories. It consumes the event-enriched HDF5 dataset and
the complete factorial-11 ZIP (or its extracted directory).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping
import csv
import hashlib
import io
import json
import math
import os
import shutil
import sys
import time
import zipfile

import h5py
os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(os.environ.get("HAY_DG01_OUTPUT", "/tmp/hay_micro_loss_landscape_12")).parent / ".matplotlib"),
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
try:
    from tqdm.auto import tqdm
except ImportError:
    class _PlainProgress:
        def __init__(self, values=None, total=None, desc="", **kwargs):
            del kwargs
            self.values, self.total, self.desc, self.count = values, total, desc, 0

        def __iter__(self):
            if self.values is None:
                return iter(())
            for value in self.values:
                self.update()
                yield value

        def update(self, amount=1):
            self.count += amount
            if self.total:
                print(f"{self.desc}: {self.count}/{self.total}", flush=True)

        def close(self):
            return None

    def tqdm(values=None, **kwargs):
        return _PlainProgress(values, **kwargs)


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

from hay_single_compartment import InputOnlyConvGRU, InputOnlyGRU, MICRO_STATE_NAMES  # noqa: E402
from hay_single_compartment.failure_atlas import is_slow_state, sha256_file  # noqa: E402
from hay_single_compartment.loss_landscape import (  # noqa: E402
    clone_hidden,
    cosine,
    filter_normalized_direction,
    flatten_gradients,
    gradient_snr,
    hutchinson_trace,
    parameter_snapshot,
    rank_correlation,
    set_parameter_point,
    top_hessian_eigenvalue,
    trainable_named_parameters,
)


RUNS = (
    "gru_mse",
    "gru_mrstft",
    "causal_conv_gru_mse",
    "causal_conv_gru_mrstft",
)
VIEW_NAMES = (
    "natural_all",
    "event_soma",
    "subthreshold_soma",
    "slow_states",
    "synaptic_states",
)


@dataclass(frozen=True)
class ObservatoryConfig:
    temporal_bin: int = 5
    window_steps: int = 128
    windows_per_view: int = 6
    context_chunk_steps: int = 256
    landscape_context_steps: int = 512
    local_radius: float = 0.02
    local_grid_points: int = 5
    interpolation_points: int = 7
    hessian_windows: int = 2
    hessian_iterations: int = 6
    hutchinson_probes: int = 4
    seed: int = 20260802


CFG = ObservatoryConfig(
    window_steps=int(os.environ.get("HAY_DG01_WINDOW_STEPS", "128")),
    windows_per_view=int(os.environ.get("HAY_DG01_WINDOWS_PER_VIEW", "6")),
    landscape_context_steps=int(os.environ.get("HAY_DG01_LANDSCAPE_CONTEXT_STEPS", "512")),
    local_grid_points=int(os.environ.get("HAY_DG01_LOCAL_GRID_POINTS", "5")),
    interpolation_points=int(os.environ.get("HAY_DG01_INTERPOLATION_POINTS", "7")),
    hessian_windows=int(os.environ.get("HAY_DG01_HESSIAN_WINDOWS", "2")),
    hessian_iterations=int(os.environ.get("HAY_DG01_HESSIAN_ITERATIONS", "6")),
    hutchinson_probes=int(os.environ.get("HAY_DG01_HUTCHINSON_PROBES", "4")),
)
if CFG.window_steps < 16 or CFG.windows_per_view < 2:
    raise ValueError("DG-01 requires window_steps >= 16 and windows_per_view >= 2")
if CFG.local_grid_points < 3 or CFG.interpolation_points < 3:
    raise ValueError("landscape grids require at least three points")

requested_runs = tuple(
    name.strip() for name in os.environ.get("HAY_DG01_RUNS", ",".join(RUNS)).split(",")
    if name.strip()
)
if not requested_runs or set(requested_runs) - set(RUNS):
    raise ValueError(f"HAY_DG01_RUNS must be a nonempty subset of {RUNS}")

OUTPUT = Path(os.environ.get("HAY_DG01_OUTPUT", "/kaggle/working/hay_micro_loss_landscape_12"))
if not Path("/kaggle").exists() and "HAY_DG01_OUTPUT" not in os.environ:
    OUTPUT = REPO_ROOT / "artifacts/loss_landscape_12"
OUTPUT.mkdir(parents=True, exist_ok=True)
FIGURES = OUTPUT / "figures"
FIGURES.mkdir(exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _valid_dataset(path: Path) -> bool:
    try:
        with h5py.File(path, "r") as handle:
            return all(key in handle for key in (
                "validation/inputs", "validation/burnin_inputs", "validation/states",
                "validation/spikes", "validation/regimes",
            )) and "state_names_json" in handle.attrs
    except (OSError, ValueError, KeyError):
        return False


def discover_dataset() -> Path:
    override = os.environ.get("HAY_DG01_DATASET")
    if override:
        path = Path(override).expanduser().resolve()
        if not _valid_dataset(path):
            raise ValueError(f"incompatible HAY_DG01_DATASET: {path}")
        return path
    roots = [path for path in (Path("/kaggle/input"), Path("/kaggle/working"), REPO_ROOT.parent) if path.exists()]
    candidates = {str(path.resolve()): path.resolve() for root in roots for path in root.glob("**/*.h5") if _valid_dataset(path)}
    if not candidates:
        raise FileNotFoundError("Attach hay_micro_4c_event_enriched_v2.h5 or set HAY_DG01_DATASET")
    return sorted(candidates.values(), key=lambda path: (0 if "event_enriched_v2" in path.name else 1, str(path)))[0]


def _valid_factorial_zip(path: Path) -> bool:
    try:
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
            return all(any(name.endswith(f"checkpoints/{run}.pt") for name in names) for run in RUNS)
    except (OSError, zipfile.BadZipFile):
        return False


def _valid_factorial_directory(path: Path) -> bool:
    return all((path / "checkpoints" / f"{run}.pt").is_file() for run in RUNS)


def discover_factorial_source() -> Path:
    override = os.environ.get("HAY_DG01_FACTORIAL")
    if override:
        path = Path(override).expanduser().resolve()
        if not (_valid_factorial_zip(path) if path.is_file() else _valid_factorial_directory(path)):
            raise ValueError(f"incompatible HAY_DG01_FACTORIAL: {path}")
        return path
    roots = [path for path in (Path("/kaggle/input"), Path("/kaggle/working"), REPO_ROOT.parent) if path.exists()]
    candidates: dict[str, Path] = {}
    for root in roots:
        for path in root.glob("**/*.zip"):
            if _valid_factorial_zip(path):
                candidates[str(path.resolve())] = path.resolve()
        for path in root.glob("**/hay_micro_orthogonal_factorial_11"):
            if _valid_factorial_directory(path):
                candidates[str(path.resolve())] = path.resolve()
    if not candidates:
        raise FileNotFoundError("Attach hay_micro_orthogonal_factorial_11_complete.zip or set HAY_DG01_FACTORIAL")
    return sorted(candidates.values(), key=lambda path: (0 if path.suffix.lower() == ".zip" else 1, str(path)))[0]


DATASET = discover_dataset()
FACTORIAL_SOURCE = discover_factorial_source()
print("repository :", REPO_ROOT)
print("device     :", DEVICE)
print("dataset    :", DATASET)
print("factorial  :", FACTORIAL_SOURCE)
print("output     :", OUTPUT)
print("[contract] validation only; no training; test remains unopened", flush=True)


def source_bytes(suffix: str) -> bytes:
    if FACTORIAL_SOURCE.is_dir():
        path = FACTORIAL_SOURCE / suffix
        if not path.is_file():
            raise FileNotFoundError(path)
        return path.read_bytes()
    with zipfile.ZipFile(FACTORIAL_SOURCE) as archive:
        normalized = suffix.replace("\\", "/").lstrip("/")
        matches = [
            name for name in archive.namelist()
            if name == normalized or name.endswith("/" + normalized)
        ]
        if len(matches) != 1:
            raise FileNotFoundError(f"expected one {suffix} in {FACTORIAL_SOURCE}, found {matches}")
        return archive.read(matches[0])


def load_checkpoint(run: str) -> dict[str, Any]:
    return torch.load(io.BytesIO(source_bytes(f"checkpoints/{run}.pt")), map_location="cpu", weights_only=False)


def load_npz(suffix: str) -> dict[str, np.ndarray]:
    with np.load(io.BytesIO(source_bytes(suffix)), allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def pack_spikes(values: np.ndarray, factor: int) -> np.ndarray:
    usable = values.shape[1] // factor * factor
    return values[:, :usable].reshape(values.shape[0], usable // factor, factor * values.shape[2]).astype(np.float32)


print("[integrity] hashing the dataset and factorial source ...", flush=True)
DATASET_SHA256 = sha256_file(DATASET)
FACTORIAL_SHA256 = sha256_file(FACTORIAL_SOURCE) if FACTORIAL_SOURCE.is_file() else "extracted_directory"
checkpoints = {run: load_checkpoint(run) for run in requested_runs}
expected_hashes = {checkpoint["dataset_sha256"] for checkpoint in checkpoints.values()}
if expected_hashes != {DATASET_SHA256}:
    raise RuntimeError(f"dataset/checkpoint SHA-256 mismatch: dataset={DATASET_SHA256}, checkpoints={expected_hashes}")

reference = load_npz("validation_reference.npz")
with h5py.File(DATASET, "r") as handle:
    stored_names = json.loads(handle.attrs["state_names_json"])
    if stored_names != list(MICRO_STATE_NAMES):
        raise RuntimeError("dataset state order differs from MICRO_STATE_NAMES")
    validation_inputs = pack_spikes(handle["validation/inputs"][:], CFG.temporal_bin)
    validation_burnin = pack_spikes(handle["validation/burnin_inputs"][:], CFG.temporal_bin)
    validation_state_sample = handle["validation/states"][0, CFG.temporal_bin::CFG.temporal_bin][: validation_inputs.shape[1]].astype(np.float32)
    validation_count = int(handle["validation/inputs"].shape[0])
    train_count = int(handle["train/inputs"].shape[0])
if validation_count != reference["truth"].shape[0]:
    raise RuntimeError("factorial validation reference and dataset trajectory counts differ")
if not np.allclose(validation_state_sample, reference["truth"][0], rtol=1e-5, atol=1e-6):
    raise RuntimeError("factorial validation reference does not match the attached dataset")

truth = reference["truth"].astype(np.float32)
teacher_spikes = reference["spikes"].astype(bool)
STATE_NAMES = list(MICRO_STATE_NAMES)
SOMA_INDEX = STATE_NAMES.index("soma.v_mV")
SLOW_INDICES = np.array([index for index, name in enumerate(STATE_NAMES) if is_slow_state(name)], dtype=np.int64)
SYNAPTIC_INDICES = np.array([
    index for index, name in enumerate(STATE_NAMES)
    if any(token in name for token in ("ampa_", "nmda_", "gabaa_"))
], dtype=np.int64)
if len(SLOW_INDICES) == 0 or len(SYNAPTIC_INDICES) == 0:
    raise RuntimeError("state-family detection failed")


def choose_rows() -> list[dict[str, Any]]:
    rng = np.random.default_rng(CFG.seed)
    maximum_start = validation_inputs.shape[1] - CFG.window_steps
    natural_candidates = [(trajectory, start) for trajectory in range(len(validation_inputs)) for start in range(0, maximum_start + 1, CFG.window_steps)]
    event_candidates = []
    for trajectory, center in np.argwhere(teacher_spikes):
        start = int(np.clip(center - CFG.window_steps // 3, 0, maximum_start))
        event_candidates.append((int(trajectory), start))
    event_candidates = list(dict.fromkeys(event_candidates))
    subthreshold_candidates = []
    stride = max(1, CFG.window_steps // 2)
    for trajectory in range(len(validation_inputs)):
        for start in range(0, maximum_start + 1, stride):
            stop = start + CFG.window_steps
            if not teacher_spikes[trajectory, start:stop].any() and truth[trajectory, start:stop, SOMA_INDEX].max() < -40.0:
                subthreshold_candidates.append((trajectory, start))
    groups = {
        "natural": natural_candidates,
        "event": event_candidates,
        "subthreshold": subthreshold_candidates,
    }
    rows: list[dict[str, Any]] = []
    for view, candidates in groups.items():
        if len(candidates) < CFG.windows_per_view:
            raise RuntimeError(f"not enough {view} windows: {len(candidates)}")
        selected = rng.choice(len(candidates), size=CFG.windows_per_view, replace=False)
        for number, candidate_index in enumerate(selected):
            trajectory, start = candidates[int(candidate_index)]
            stop = start + CFG.window_steps
            rows.append({
                "window_id": f"{view}_{number:02d}",
                "sampling_view": view,
                "trajectory": int(trajectory),
                "start_step": int(start),
                "stop_step": int(stop),
                "event_bins": int(teacher_spikes[trajectory, start:stop].sum()),
                "maximum_soma_mV": float(truth[trajectory, start:stop, SOMA_INDEX].max()),
            })
    return rows


MANIFEST = choose_rows()
write_rows(OUTPUT / "validation_window_manifest.csv", MANIFEST)
manifest_by_id = {row["window_id"]: row for row in MANIFEST}
rows_by_sampling_view = {
    name: [row for row in MANIFEST if row["sampling_view"] == name]
    for name in ("natural", "event", "subthreshold")
}
view_contract = {
    "natural_all": (rows_by_sampling_view["natural"], np.arange(len(STATE_NAMES))),
    "event_soma": (rows_by_sampling_view["event"], np.array([SOMA_INDEX])),
    "subthreshold_soma": (rows_by_sampling_view["subthreshold"], np.array([SOMA_INDEX])),
    "slow_states": (rows_by_sampling_view["natural"], SLOW_INDICES),
    "synaptic_states": (rows_by_sampling_view["natural"], SYNAPTIC_INDICES),
}
manifest_sha256 = hashlib.sha256((OUTPUT / "validation_window_manifest.csv").read_bytes()).hexdigest()


def build_model(checkpoint: Mapping[str, Any], *, load_weights: bool = True) -> nn.Module:
    spec = checkpoint["model_spec"]
    if spec["class"] == "InputOnlyGRU":
        model = InputOnlyGRU(
            spec["input_dim"], spec["state_dim"], hidden_dim=spec["hidden_dim"],
            layers=spec["layers"], decoder_dim=spec["decoder_dim"],
        )
    elif spec["class"] == "InputOnlyConvGRU":
        model = InputOnlyConvGRU(
            spec["input_dim"], spec["state_dim"], hidden_dim=spec["hidden_dim"],
            conv_channels=spec["conv_channels"], dilations=tuple(spec["dilations"]),
            kernel_size=spec["kernel_size"], decoder_dim=spec["decoder_dim"],
        )
    else:
        raise ValueError(f"unsupported checkpoint model: {spec['class']}")
    if load_weights:
        model.load_state_dict(checkpoint["model_state_dict"])
    return model.to(DEVICE).eval()


def advance(model: nn.Module, sequence: np.ndarray, hidden: Any = None) -> Any:
    with torch.no_grad():
        for start in range(0, len(sequence), CFG.context_chunk_steps):
            values = torch.as_tensor(sequence[None, start:start + CFG.context_chunk_steps], device=DEVICE)
            _, hidden = model(values, hidden)
            hidden = clone_hidden(hidden)
    return hidden


def full_history_cache(model: nn.Module, rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    requested: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        requested.setdefault(int(row["trajectory"]), []).append(row)
    result: dict[str, Any] = {}
    for trajectory, trajectory_rows in tqdm(sorted(requested.items()), desc="history contexts", leave=False):
        hidden = advance(model, validation_burnin[trajectory])
        position = 0
        for row in sorted(trajectory_rows, key=lambda value: int(value["start_step"])):
            start = int(row["start_step"])
            hidden = advance(model, validation_inputs[trajectory, position:start], hidden)
            result[row["window_id"]] = clone_hidden(hidden)
            position = start
    return result


def truncated_context_cache(model: nn.Module, rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for row in rows:
        trajectory, start = int(row["trajectory"]), int(row["start_step"])
        combined = np.concatenate((validation_burnin[trajectory], validation_inputs[trajectory, :start]), axis=0)
        context = combined[-CFG.landscape_context_steps:]
        result[row["window_id"]] = advance(model, context)
    return result


def window_loss(
    model: nn.Module,
    row: Mapping[str, Any],
    hidden: Any,
    state_indices: np.ndarray,
) -> torch.Tensor:
    trajectory, start, stop = int(row["trajectory"]), int(row["start_step"]), int(row["stop_step"])
    inputs = torch.as_tensor(validation_inputs[trajectory:trajectory + 1, start:stop], device=DEVICE)
    mean = np.asarray(next(iter(checkpoints.values()))["state_mean"], dtype=np.float32)
    std = np.asarray(next(iter(checkpoints.values()))["state_std"], dtype=np.float32)
    normalized = (truth[trajectory:trajectory + 1, start:stop] - mean) / std
    targets = torch.as_tensor(normalized, device=DEVICE)
    prediction, _ = model(inputs, clone_hidden(hidden))
    indices = torch.as_tensor(state_indices, device=DEVICE, dtype=torch.long)
    return torch.mean(torch.square(prediction.index_select(-1, indices) - targets.index_select(-1, indices)))


def evaluate_views(model: nn.Module, cache: Mapping[str, Any]) -> dict[str, float]:
    values: dict[str, float] = {}
    with torch.no_grad():
        for view, (rows, indices) in view_contract.items():
            losses = [float(window_loss(model, row, cache[row["window_id"]], indices)) for row in rows]
            values[view] = float(np.mean(losses))
    return values


gradient_window_rows: list[dict[str, Any]] = []
gradient_summary_rows: list[dict[str, Any]] = []
cosine_rows: list[dict[str, Any]] = []
local_slice_rows: list[dict[str, Any]] = []
interpolation_rows: list[dict[str, Any]] = []
hessian_rows: list[dict[str, Any]] = []
density_rows: list[dict[str, Any]] = []
model_results: dict[str, dict[str, Any]] = {}
skip_hessian = os.environ.get("HAY_DG01_SKIP_HESSIAN", "0") == "1"
skip_landscape = os.environ.get("HAY_DG01_SKIP_LANDSCAPE", "0") == "1"
experiment_started = time.perf_counter()


def collect_gradient_geometry(run: str, model: nn.Module, cache: Mapping[str, Any]):
    # cuDNN intentionally disables the RNN backward path after an eval-mode
    # forward. These checkpoints contain no stochastic training-time layer
    # (GRU dropout is zero), so train mode changes no numerical forward
    # behavior; it only enables the read-only gradient diagnostic on CUDA.
    model.train()
    named = trainable_named_parameters(model)
    view_vectors: dict[str, dict[str, Any]] = {}
    total_windows = sum(len(rows) for rows, _ in view_contract.values())
    progress = tqdm(total=total_windows, desc=f"{run} gradients", leave=True)
    for view, (rows, indices) in view_contract.items():
        vectors: list[torch.Tensor] = []
        block_vectors: dict[str, list[torch.Tensor]] = {}
        losses: list[float] = []
        for row in rows:
            model.zero_grad(set_to_none=True)
            loss = window_loss(model, row, cache[row["window_id"]], indices)
            loss.backward()
            vector, blocks = flatten_gradients(named)
            vector = vector.cpu()
            blocks = {name: value.cpu() for name, value in blocks.items()}
            vectors.append(vector)
            for block, value in blocks.items():
                block_vectors.setdefault(block, []).append(value)
            losses.append(float(loss.detach()))
            gradient_window_rows.append({
                "run": run, "view": view, "window_id": row["window_id"],
                "loss": losses[-1], "gradient_norm": float(torch.linalg.vector_norm(vector)),
            })
            progress.update()
        mean_vector = torch.stack(vectors).mean(0)
        mean_blocks = {block: torch.stack(values).mean(0) for block, values in block_vectors.items()}
        view_vectors[view] = {"complete": mean_vector, "blocks": mean_blocks}
        gradient_summary_rows.append({
            "run": run, "view": view, "parameter_block": "all",
            "mean_loss": float(np.mean(losses)),
            "mean_gradient_norm": float(torch.linalg.vector_norm(mean_vector)),
            "window_gradient_norm_mean": float(np.mean([torch.linalg.vector_norm(value) for value in vectors])),
            "window_gradient_norm_std": float(np.std([torch.linalg.vector_norm(value) for value in vectors])),
            "gradient_snr": gradient_snr(vectors), "windows": len(vectors),
        })
        for block, values in block_vectors.items():
            gradient_summary_rows.append({
                "run": run, "view": view, "parameter_block": block,
                "mean_loss": float(np.mean(losses)),
                "mean_gradient_norm": float(torch.linalg.vector_norm(mean_blocks[block])),
                "window_gradient_norm_mean": float(np.mean([torch.linalg.vector_norm(value) for value in values])),
                "window_gradient_norm_std": float(np.std([torch.linalg.vector_norm(value) for value in values])),
                "gradient_snr": gradient_snr(values), "windows": len(values),
            })
    progress.close()
    for left_index, left in enumerate(VIEW_NAMES):
        for right in VIEW_NAMES[left_index:]:
            cosine_rows.append({
                "run": run, "left_view": left, "right_view": right,
                "parameter_block": "all",
                "cosine": cosine(view_vectors[left]["complete"], view_vectors[right]["complete"]),
            })
            for block in sorted(set(view_vectors[left]["blocks"]) & set(view_vectors[right]["blocks"])):
                cosine_rows.append({
                    "run": run, "left_view": left, "right_view": right,
                    "parameter_block": block,
                    "cosine": cosine(view_vectors[left]["blocks"][block], view_vectors[right]["blocks"][block]),
                })
    model.eval()
    return view_vectors


for run_index, run in enumerate(requested_runs, start=1):
    run_started = time.perf_counter()
    checkpoint = checkpoints[run]
    torch.manual_seed(int(checkpoint["training_config"]["seed"]))
    initial_model = build_model(checkpoint, load_weights=False)
    initial_state = parameter_snapshot(initial_model)
    del initial_model
    model = build_model(checkpoint)
    final_state = parameter_snapshot(model)
    print(f"\n[{run_index}/{len(requested_runs)}] {run}: full-history contexts", flush=True)
    history_cache = full_history_cache(model, MANIFEST)
    view_vectors = collect_gradient_geometry(run, model, history_cache)
    model_result = {"vectors": view_vectors, "initial_state": initial_state, "final_state": final_state}

    if not skip_landscape:
        print(f"[{run}] filter-normalized local surface", flush=True)
        generator = torch.Generator(device=DEVICE).manual_seed(CFG.seed + run_index * 101)
        first_direction = filter_normalized_direction(model, generator=generator)
        second_direction = filter_normalized_direction(model, generator=generator)
        coordinates = np.linspace(-CFG.local_radius, CFG.local_radius, CFG.local_grid_points)
        for alpha in tqdm(coordinates, desc=f"{run} local-x", leave=False):
            for beta in coordinates:
                set_parameter_point(model, final_state, [(float(alpha), first_direction), (float(beta), second_direction)])
                values = evaluate_views(model, history_cache)
                for view, value in values.items():
                    local_slice_rows.append({
                        "run": run, "view": view, "alpha": float(alpha), "beta": float(beta),
                        "loss": value, "radius": CFG.local_radius,
                        "context_policy": "frozen_full_history_TBPTT_boundary",
                    })
        set_parameter_point(model, final_state)

        print(f"[{run}] initial-to-final interpolation", flush=True)
        delta = {name: final_state[name] - initial_state[name] for name in final_state}
        for alpha in tqdm(np.linspace(0.0, 1.0, CFG.interpolation_points), desc=f"{run} interpolation", leave=False):
            set_parameter_point(model, initial_state, [(float(alpha), delta)])
            context = truncated_context_cache(model, MANIFEST)
            values = evaluate_views(model, context)
            for view, value in values.items():
                interpolation_rows.append({
                    "run": run, "path": "initial_to_final", "view": view,
                    "alpha": float(alpha), "loss": value,
                    "context_policy": f"recomputed_last_{CFG.landscape_context_steps}_steps",
                })
        set_parameter_point(model, final_state)

    if not skip_hessian:
        print(f"[{run}] Hessian estimators", flush=True)
        # Hessian-vector products require backward-through-backward. As above,
        # train mode only enables cuDNN autograd and performs no parameter or
        # buffer update for these architectures.
        model.train()
        parameters = [parameter for _, parameter in trainable_named_parameters(model)]
        hessian_views = ("event_soma", "slow_states", "synaptic_states")
        for view_number, view in enumerate(hessian_views):
            rows, indices = view_contract[view]
            selected_rows = rows[:CFG.hessian_windows]

            def closure(rows=selected_rows, indices=indices):
                return torch.stack([
                    window_loss(model, row, history_cache[row["window_id"]], indices)
                    for row in rows
                ]).mean()

            generator = torch.Generator(device=DEVICE).manual_seed(CFG.seed + run_index * 1000 + view_number)
            eigenvalue, residual, eigen_history = top_hessian_eigenvalue(
                closure, parameters, iterations=CFG.hessian_iterations, generator=generator,
            )
            trace, trace_se, trace_samples = hutchinson_trace(
                closure, parameters, probes=CFG.hutchinson_probes, generator=generator,
            )
            hessian_rows.append({
                "run": run, "view": view, "top_eigenvalue": eigenvalue,
                "power_residual": residual, "hessian_trace": trace,
                "trace_standard_error": trace_se,
                "power_history_json": json.dumps(eigen_history),
                "trace_samples_json": json.dumps(trace_samples),
                "windows": len(selected_rows),
                "context_policy": "frozen_full_history_TBPTT_boundary",
            })
        model.eval()
    set_parameter_point(model, final_state)
    model_results[run] = model_result
    elapsed = time.perf_counter() - run_started
    total_elapsed = time.perf_counter() - experiment_started
    eta = total_elapsed / run_index * (len(requested_runs) - run_index)
    print(f"[{run}] complete in {elapsed / 60:.1f} min | experiment ETA {eta / 60:.1f} min", flush=True)
    del model, history_cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if not skip_landscape:
    print("\n[landscape] paired MSE-to-MR-STFT paths", flush=True)
    pairs = (("gru_mse", "gru_mrstft"), ("causal_conv_gru_mse", "causal_conv_gru_mrstft"))
    for left_run, right_run in pairs:
        if left_run not in checkpoints or right_run not in checkpoints:
            continue
        model = build_model(checkpoints[left_run])
        left_state = parameter_snapshot(model)
        right_model = build_model(checkpoints[right_run])
        right_state = parameter_snapshot(right_model)
        del right_model
        delta = {name: right_state[name] - left_state[name] for name in left_state}
        for alpha in tqdm(np.linspace(0.0, 1.0, CFG.interpolation_points), desc=f"{left_run}->{right_run}", leave=False):
            set_parameter_point(model, left_state, [(float(alpha), delta)])
            context = truncated_context_cache(model, MANIFEST)
            values = evaluate_views(model, context)
            for view, value in values.items():
                interpolation_rows.append({
                    "run": f"{left_run}_to_{right_run}", "path": "paired_objective_checkpoints",
                    "view": view, "alpha": float(alpha), "loss": value,
                    "context_policy": f"recomputed_last_{CFG.landscape_context_steps}_steps",
                })
        del model


print("[density] target-density/error relation from frozen validation predictions", flush=True)
voltage_edges = np.array([-120, -90, -82, -78, -74, -70, -66, -62, -58, -50, -40, -20, 0, 20, 60], dtype=np.float32)
teacher_voltage = truth[..., SOMA_INDEX].reshape(-1)
for run in requested_runs:
    prediction = load_npz(f"{run}_validation_predictions.npz")["prediction"][..., SOMA_INDEX].reshape(-1)
    for left, right in zip(voltage_edges[:-1], voltage_edges[1:]):
        mask = (teacher_voltage >= left) & (teacher_voltage < right)
        if not mask.any():
            continue
        error = prediction[mask] - teacher_voltage[mask]
        density_rows.append({
            "run": run, "voltage_left_mV": float(left), "voltage_right_mV": float(right),
            "count": int(mask.sum()), "fraction": float(mask.mean()),
            "mae_mV": float(np.mean(np.abs(error))), "rmse_mV": float(np.sqrt(np.mean(np.square(error)))),
            "bias_mV": float(np.mean(error)),
        })


training_config = next(iter(checkpoints.values()))["training_config"]
packed_burnin_steps = validation_burnin.shape[1]
packed_input_steps = validation_inputs.shape[1]
natural_batches = math.ceil(train_count / training_config["batch_trajectories"])
natural_updates = natural_batches * (
    math.ceil(packed_burnin_steps / training_config["chunk_steps"])
    + math.ceil(packed_input_steps / training_config["chunk_steps"])
)
stratified_updates = math.ceil(training_config["stratified_windows_per_epoch"] / training_config["batch_trajectories"])
event_update_share = stratified_updates / (natural_updates + stratified_updates)
event_bin_fraction = float(teacher_spikes.mean())

write_rows(OUTPUT / "window_loss_gradients.csv", gradient_window_rows)
write_rows(OUTPUT / "gradient_summary.csv", gradient_summary_rows)
write_rows(OUTPUT / "gradient_cosines.csv", cosine_rows)
write_rows(OUTPUT / "local_landscape.csv", local_slice_rows)
write_rows(OUTPUT / "interpolation_paths.csv", interpolation_rows)
write_rows(OUTPUT / "hessian_summary.csv", hessian_rows)
write_rows(OUTPUT / "target_density_error.csv", density_rows)


def _metric(run: str, view: str, key: str) -> float:
    row = next(value for value in gradient_summary_rows if value["run"] == run and value["view"] == view and value["parameter_block"] == "all")
    return float(row[key])


def _cosine(run: str, other: str) -> float:
    row = next(value for value in cosine_rows if value["run"] == run and value["parameter_block"] == "all" and {value["left_view"], value["right_view"]} == {"event_soma", other})
    return float(row["cosine"])


density_correlations = []
for run in requested_runs:
    rows = [row for row in density_rows if row["run"] == run]
    density_correlations.append(rank_correlation([math.log10(row["count"] + 1) for row in rows], [row["rmse_mV"] for row in rows]))
event_snr = float(np.median([_metric(run, "event_soma", "gradient_snr") for run in requested_runs]))
event_gradient_ratio = float(np.median([
    _metric(run, "event_soma", "mean_gradient_norm") / max(_metric(run, "natural_all", "mean_gradient_norm"), 1e-12)
    for run in requested_runs
]))
event_slow_cosine = float(np.median([_cosine(run, "slow_states") for run in requested_runs]))
event_synaptic_cosine = float(np.median([_cosine(run, "synaptic_states") for run in requested_runs]))
density_error_correlation = float(np.median(density_correlations))


def clipped(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


scores = {
    "TR-01_exposure": 0.5 * clipped((0.05 - event_update_share) / 0.05) + 0.5 * clipped(event_snr / 0.5),
    "LO-04_gradient_conflict": clipped(-min(event_slow_cosine, event_synaptic_cosine) / 0.30),
    "LO-02_imbalanced_regression": 0.6 * clipped(-density_error_correlation / 0.60) + 0.4 * clipped(event_gradient_ratio / 0.50),
    "DG-02_representation": 0.6 * clipped((0.15 - event_snr) / 0.15) + 0.4 * clipped((0.10 - event_gradient_ratio) / 0.10),
    "OP-01_optimizer_basin": 0.5 * clipped(event_snr / 0.5) + 0.5 * clipped(min(event_slow_cosine, event_synaptic_cosine) / 0.30),
}
ranking = sorted(scores.items(), key=lambda item: item[1], reverse=True)
winner, winner_score = ranking[0]
margin = winner_score - ranking[1][1]
provisional = winner if winner_score >= 0.50 and margin >= 0.10 else "inconclusive_run_DG-02"
decision = {
    "experiment_id": "DG-01",
    "status": "completed_read_only",
    "decision": provisional,
    "requires_human_review": True,
    "warning": "The scorecard is a preregistered triage aid, not an automatic causal claim.",
    "measured_signature": {
        "event_bin_fraction": event_bin_fraction,
        "event_targeted_optimizer_update_share": event_update_share,
        "median_event_gradient_snr": event_snr,
        "median_event_to_natural_gradient_norm_ratio": event_gradient_ratio,
        "median_event_vs_slow_gradient_cosine": event_slow_cosine,
        "median_event_vs_synaptic_gradient_cosine": event_synaptic_cosine,
        "median_target_density_vs_error_rank_correlation": density_error_correlation,
    },
    "branch_scores": scores,
    "score_margin": margin,
    "next_required_action": "Review plots and raw batch stability, then run DG-02 on the exact same manifest before authorizing training.",
    "manifest_sha256": manifest_sha256,
    "dataset_sha256": DATASET_SHA256,
    "factorial_source_sha256": FACTORIAL_SHA256,
}
(OUTPUT / "decision.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")


def plot_cosines() -> None:
    runs = list(requested_runs)
    figure, axes = plt.subplots(1, len(runs), figsize=(5 * len(runs), 4.5), squeeze=False)
    for axis, run in zip(axes[0], runs):
        matrix = np.eye(len(VIEW_NAMES))
        for row in cosine_rows:
            if row["run"] != run or row["parameter_block"] != "all":
                continue
            left, right = VIEW_NAMES.index(row["left_view"]), VIEW_NAMES.index(row["right_view"])
            matrix[left, right] = matrix[right, left] = row["cosine"]
        image = axis.imshow(matrix, vmin=-1, vmax=1, cmap="coolwarm")
        axis.set_xticks(range(len(VIEW_NAMES)), VIEW_NAMES, rotation=45, ha="right")
        axis.set_yticks(range(len(VIEW_NAMES)), VIEW_NAMES)
        axis.set_title(run)
        for y in range(len(VIEW_NAMES)):
            for x in range(len(VIEW_NAMES)):
                axis.text(x, y, f"{matrix[y, x]:.2f}", ha="center", va="center", fontsize=7)
    figure.colorbar(image, ax=axes.ravel().tolist(), shrink=0.75, label="gradient cosine")
    figure.suptitle("Regime-conditioned gradient alignment")
    figure.savefig(FIGURES / "gradient_cosine_heatmaps.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_gradient_summary() -> None:
    rows = [row for row in gradient_summary_rows if row["parameter_block"] == "all"]
    figure, axes = plt.subplots(1, 2, figsize=(14, 5))
    x = np.arange(len(VIEW_NAMES))
    width = 0.8 / len(requested_runs)
    for index, run in enumerate(requested_runs):
        selected = {row["view"]: row for row in rows if row["run"] == run}
        axes[0].bar(x + (index - (len(requested_runs) - 1) / 2) * width, [selected[view]["mean_gradient_norm"] for view in VIEW_NAMES], width, label=run)
        axes[1].bar(x + (index - (len(requested_runs) - 1) / 2) * width, [selected[view]["gradient_snr"] for view in VIEW_NAMES], width, label=run)
    for axis, title, ylabel in zip(axes, ("Mean gradient norm", "Across-window gradient SNR"), ("L2 norm", "||mean g|| / RMS deviation")):
        axis.set_xticks(x, VIEW_NAMES, rotation=30, ha="right")
        axis.set_title(title); axis.set_ylabel(ylabel); axis.grid(axis="y", alpha=0.25)
    axes[0].set_yscale("log")
    axes[0].legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(FIGURES / "gradient_norm_and_snr.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_local_surfaces() -> None:
    if not local_slice_rows:
        return
    figure, axes = plt.subplots(1, len(requested_runs), figsize=(5 * len(requested_runs), 4.3), squeeze=False)
    for axis, run in zip(axes[0], requested_runs):
        rows = [row for row in local_slice_rows if row["run"] == run and row["view"] == "event_soma"]
        coordinates = sorted({row["alpha"] for row in rows})
        values = np.array([[next(row["loss"] for row in rows if row["alpha"] == alpha and row["beta"] == beta) for beta in coordinates] for alpha in coordinates])
        contour = axis.contourf(coordinates, coordinates, np.log10(np.maximum(values, 1e-12)), levels=16, cmap="viridis")
        axis.scatter([0], [0], color="red", s=25, label="checkpoint")
        axis.set_title(run); axis.set_xlabel("direction 1"); axis.set_ylabel("direction 2")
        figure.colorbar(contour, ax=axis, label="log10 event loss")
    figure.suptitle("Filter-normalized local event-loss surfaces")
    figure.tight_layout()
    figure.savefig(FIGURES / "local_event_landscapes.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_interpolations() -> None:
    if not interpolation_rows:
        return
    paths = list(dict.fromkeys(row["run"] for row in interpolation_rows))
    figure, axes = plt.subplots(math.ceil(len(paths) / 2), 2, figsize=(13, 4 * math.ceil(len(paths) / 2)), squeeze=False)
    for axis, path in zip(axes.ravel(), paths):
        rows = [row for row in interpolation_rows if row["run"] == path]
        for view in VIEW_NAMES:
            selected = sorted((row for row in rows if row["view"] == view), key=lambda row: row["alpha"])
            axis.plot([row["alpha"] for row in selected], [row["loss"] for row in selected], marker="o", label=view)
        axis.set_yscale("log"); axis.set_title(path); axis.set_xlabel("interpolation alpha"); axis.set_ylabel("normalized MSE")
        axis.grid(alpha=0.25); axis.legend(fontsize=7)
    for axis in axes.ravel()[len(paths):]:
        axis.axis("off")
    figure.tight_layout()
    figure.savefig(FIGURES / "interpolation_profiles.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_density() -> None:
    figure, axes = plt.subplots(1, len(requested_runs), figsize=(5 * len(requested_runs), 4.3), squeeze=False)
    for axis, run in zip(axes[0], requested_runs):
        rows = [row for row in density_rows if row["run"] == run]
        centers = [(row["voltage_left_mV"] + row["voltage_right_mV"]) / 2 for row in rows]
        axis.bar(centers, [row["fraction"] for row in rows], width=4, alpha=0.35, label="target fraction")
        twin = axis.twinx()
        twin.plot(centers, [row["rmse_mV"] for row in rows], color="tab:red", marker="o", label="RMSE")
        axis.set_yscale("log"); axis.set_title(run); axis.set_xlabel("teacher soma voltage bin (mV)")
        axis.set_ylabel("target fraction"); twin.set_ylabel("RMSE (mV)")
    figure.suptitle("Target density versus conditional error")
    figure.tight_layout()
    figure.savefig(FIGURES / "target_density_vs_error.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


plot_cosines()
plot_gradient_summary()
plot_local_surfaces()
plot_interpolations()
plot_density()

provenance = {
    "experiment_id": "DG-01",
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "repository": str(REPO_ROOT),
    "dataset": str(DATASET),
    "dataset_sha256": DATASET_SHA256,
    "factorial_source": str(FACTORIAL_SOURCE),
    "factorial_source_sha256": FACTORIAL_SHA256,
    "validation_manifest_sha256": manifest_sha256,
    "device": str(DEVICE),
    "torch_version": torch.__version__,
    "numpy_version": np.__version__,
    "config": asdict(CFG),
    "runs": list(requested_runs),
    "test_split_opened": False,
    "optimizer_steps": 0,
    "surface_scope": {
        "local": "filter-normalized, conditional on frozen full-history TBPTT boundary states",
        "interpolation": f"hidden state recomputed from the last {CFG.landscape_context_steps} packed steps at every point",
    },
}
(OUTPUT / "provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
(OUTPUT / "README.md").write_text(
    "# DG-01 Loss Landscape Observatory\n\n"
    "Read-only validation diagnostics for the four factorial-11 checkpoints. "
    "Start with `decision.json`, then inspect gradient cosines, per-window stability, "
    "Hessian convergence, local surfaces, interpolation paths and density-conditioned error. "
    "No test trajectory was opened and no optimizer step was performed.\n",
    encoding="utf-8",
)

zip_base = OUTPUT.parent / "hay_micro_loss_landscape_12_complete"
ZIP_PATH = Path(shutil.make_archive(str(zip_base), "zip", root_dir=OUTPUT.parent, base_dir=OUTPUT.name))
print("\nDG-01 complete")
print(json.dumps(decision, indent=2))
print("archive:", ZIP_PATH, f"({ZIP_PATH.stat().st_size / 2**20:.1f} MiB)")
