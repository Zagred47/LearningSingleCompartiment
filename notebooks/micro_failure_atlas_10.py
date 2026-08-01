"""Read-only Failure Atlas for a trained micro-Hay surrogate.

This companion script is intentionally training-free.  It discovers Kaggle
inputs, reconstructs a one-layer InputOnlyGRU when possible, and otherwise
analyses an exported prediction archive.  All results are written to one
portable directory and ZIP file.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import json
import os
import shutil
import subprocess
import sys

import h5py
import numpy as np
import torch


def _find_repository() -> Path:
    candidates = [Path(__file__).resolve().parents[1]]
    candidates.extend(Path("/kaggle/working").glob("**/pyproject.toml"))
    candidates.extend(Path("/kaggle/input").glob("**/pyproject.toml"))
    for candidate in candidates:
        root = candidate if candidate.is_dir() else candidate.parent
        if (root / "src/hay_single_compartment").is_dir():
            return root
    raise FileNotFoundError(
        "Repository not found. Run the setup cell in kaggle_micro_failure_atlas_10.ipynb first."
    )


REPO_ROOT = _find_repository()
sys.path.insert(0, str(REPO_ROOT / "src"))

from hay_single_compartment import (  # noqa: E402
    FailureAtlasConfig,
    build_failure_atlas,
    checkpoint_gru_spec,
    inspect_gru_checkpoint,
    load_dataset_context,
    load_prediction_archive,
    write_activation_diagnostics,
    write_failure_atlas,
)
from hay_single_compartment.event_aware import MICRO_EVENT_NAMES  # noqa: E402
from hay_single_compartment.failure_atlas import sha256_file  # noqa: E402


SEARCH_ROOTS = tuple(
    path for path in (Path("/kaggle/input"), Path("/kaggle/working"), REPO_ROOT.parent)
    if path.exists()
)
OUTPUT_DIR = Path(os.environ.get("HAY_ATLAS_OUTPUT", "/kaggle/working/hay_micro_failure_atlas_10"))
if not Path("/kaggle").exists() and "HAY_ATLAS_OUTPUT" not in os.environ:
    OUTPUT_DIR = REPO_ROOT / "artifacts/failure_atlas_10"
MODEL_KEY = os.environ.get("HAY_ATLAS_MODEL_KEY") or None
MAX_TRAJECTORIES = int(os.environ.get("HAY_ATLAS_MAX_TRAJECTORIES", "0")) or None
REPLAY_SPLIT = os.environ.get("HAY_ATLAS_SPLIT", "validation")
PREDICTION_SPLIT = os.environ.get("HAY_ATLAS_PREDICTION_SPLIT", "test")


def _override(name: str) -> Path | None:
    value = os.environ.get(name)
    if not value:
        return None
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{name} points to a missing file: {path}")
    return path


def _all_files(pattern: str) -> list[Path]:
    found: dict[str, Path] = {}
    for root in SEARCH_ROOTS:
        for path in root.glob(pattern):
            if path.is_file() and OUTPUT_DIR not in path.parents:
                found[str(path.resolve())] = path.resolve()
    return list(found.values())


def _valid_dataset(path: Path) -> bool:
    try:
        with h5py.File(path, "r") as handle:
            return (
                "state_names_json" in handle.attrs
                and "config_json" in handle.attrs
                and "test/inputs" in handle
                and "test/states" in handle
                and "test/spikes" in handle
                and "validation/inputs" in handle
            )
    except (OSError, KeyError, ValueError):
        return False


def _valid_checkpoint(path: Path) -> bool:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        checkpoint_gru_spec(payload)
        return "state_mean" in payload and "state_std" in payload
    except (OSError, KeyError, TypeError, ValueError, RuntimeError):
        return False


def _valid_predictions(path: Path) -> bool:
    try:
        with np.load(path, allow_pickle=False) as archive:
            keys = set(archive.files)
            return {"truth", "prediction"} <= keys or "teacher" in keys
    except (OSError, ValueError):
        return False


def _rank(path: Path, preferences: tuple[str, ...]) -> tuple[int, int, str]:
    lower = path.name.lower()
    preferred = next((index for index, token in enumerate(preferences) if token in lower), len(preferences))
    # Prefer Kaggle input over transient working artifacts when both exist.
    location = 0 if "/kaggle/input/" in path.as_posix() else 1
    return preferred, location, str(path)


def discover_artifacts() -> tuple[Path | None, Path | None, Path | None]:
    dataset = _override("HAY_ATLAS_DATASET")
    checkpoint = _override("HAY_ATLAS_CHECKPOINT")
    predictions = _override("HAY_ATLAS_PREDICTIONS")
    if dataset is None:
        candidates = [path for path in _all_files("**/*.h5") if _valid_dataset(path)]
        candidates.sort(key=lambda path: _rank(path, ("event_enriched_v2", "micro_4c_v1", "micro")))
        dataset = candidates[0] if candidates else None
    if checkpoint is None:
        candidates = [path for path in _all_files("**/*.pt") if _valid_checkpoint(path)]
        candidates.sort(key=lambda path: _rank(path, ("gru_mse", "gru.pt", "gru_conservative", "gru")))
        checkpoint = candidates[0] if candidates else None
    if predictions is None:
        candidates = [path for path in _all_files("**/*.npz") if _valid_predictions(path)]
        candidates.sort(key=lambda path: _rank(path, ("test_predictions", "predictions")))
        predictions = candidates[0] if candidates else None
    return dataset, checkpoint, predictions


def _checkpoint_metadata(path: Path | None) -> dict:
    if path is None:
        return {}
    return torch.load(path, map_location="cpu", weights_only=False)


def _dataset_steps(path: Path, split: str) -> int:
    with h5py.File(path, "r") as handle:
        return int(handle[f"{split}/inputs"].shape[1])


def _align_historical_initial_state(
    truth: np.ndarray,
    prediction: np.ndarray,
    events: np.ndarray | None,
    dataset: Path | None,
    split: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    if dataset is None:
        return truth, prediction, events
    raw_steps = _dataset_steps(dataset, split)
    target_steps = truth.shape[1]
    if raw_steps % target_steps == 0:
        return truth, prediction, events
    if target_steps > 1 and raw_steps % (target_steps - 1) == 0:
        truth, prediction = truth[:, 1:], prediction[:, 1:]
        if events is not None and events.shape[1] == target_steps:
            events = events[:, 1:]
        print("[atlas] historical archive includes t=0; removed the initial boundary")
        return truth, prediction, events
    raise ValueError(
        f"Prediction length {target_steps} is incompatible with dataset length {raw_steps}."
    )


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-c", f"safe.directory={REPO_ROOT.as_posix()}", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


dataset_path, checkpoint_path, prediction_path = discover_artifacts()
print("Repository :", REPO_ROOT)
print("Dataset    :", dataset_path)
print("Checkpoint :", checkpoint_path)
print("Predictions:", prediction_path)
print("Output     :", OUTPUT_DIR)

if dataset_path is None:
    raise FileNotFoundError(
        "No compatible HDF5 dataset found. Attach it to Kaggle or set HAY_ATLAS_DATASET."
    )

checkpoint = _checkpoint_metadata(checkpoint_path)
activation_diagnostics = None
activation_rows = None
analysis_mode = "prediction_archive"

schema_has_burnin = False
with h5py.File(dataset_path, "r") as handle:
    schema_has_burnin = f"{REPLAY_SPLIT}/burnin_inputs" in handle

force_predictions = os.environ.get("HAY_ATLAS_FORCE_PREDICTIONS", "0") == "1"
if checkpoint_path is not None and schema_has_burnin and not force_predictions:
    analysis_mode = "checkpoint_replay"
    analysis_split = REPLAY_SPLIT
    truth, prediction, state_names, dt_ms, activation_diagnostics, activation_rows = inspect_gru_checkpoint(
        checkpoint_path,
        dataset_path,
        split=analysis_split,
        max_trajectories=MAX_TRAJECTORIES,
        chunk_steps=int(os.environ.get("HAY_ATLAS_CHUNK_STEPS", "512")),
        activation_timepoints=int(os.environ.get("HAY_ATLAS_ACTIVATION_POINTS", "4096")),
    )
    model_name = str(checkpoint.get("model_name", "gru"))
    if any(str(key).startswith("baseline.") for key in checkpoint.get("model_state_dict", {})):
        model_name = f"baseline_gru_from_{model_name}"
    archived_events = None
else:
    analysis_split = PREDICTION_SPLIT
    if prediction_path is None:
        reason = "checkpoint replay requires a schema-1.2 dataset with test/burnin_inputs"
        raise FileNotFoundError(f"No usable prediction archive found; {reason}.")
    truth, prediction, model_name, archived_events = load_prediction_archive(prediction_path, MODEL_KEY)
    if MAX_TRAJECTORIES is not None:
        truth = truth[:MAX_TRAJECTORIES]
        prediction = prediction[:MAX_TRAJECTORIES]
        if archived_events is not None:
            archived_events = archived_events[:MAX_TRAJECTORIES]
    truth, prediction, archived_events = _align_historical_initial_state(
        truth, prediction, archived_events, dataset_path, analysis_split
    )
    state_names = list(checkpoint.get("state_names", []))
    if not state_names:
        with h5py.File(dataset_path, "r") as handle:
            state_names = json.loads(handle.attrs["state_names_json"])

context = load_dataset_context(
    dataset_path, truth.shape[0], truth.shape[1], split=analysis_split
)
if analysis_split == "test":
    print(
        "[atlas] WARNING: this archive is evaluated on test; treat the report as exploratory "
        "and use a fresh holdout for future confirmatory claims."
    )
dt_ms = float(context["model_dt_ms"])
if state_names != list(context["state_names"]):
    raise ValueError("Checkpoint/prediction state order does not match the dataset schema")
if archived_events is not None and archived_events.shape != truth.shape[:2] + (len(MICRO_EVENT_NAMES),):
    print(f"[atlas] ignoring incompatible archived event masks {archived_events.shape}")
    archived_events = None

state_scale = checkpoint.get("state_std")
if state_scale is not None:
    state_scale = np.asarray(state_scale)
    if state_scale.shape != (truth.shape[-1],):
        raise ValueError("checkpoint state_std does not match prediction state width")

report, tables = build_failure_atlas(
    truth,
    prediction,
    state_names,
    dt_ms,
    event_masks=archived_events,
    regimes=context["regimes"],
    regime_names=context["regime_names"],
    teacher_spikes=context["spikes"],
    state_scale=state_scale,
    config=FailureAtlasConfig(),
)

if OUTPUT_DIR.exists():
    shutil.rmtree(OUTPUT_DIR)
OUTPUT_DIR.mkdir(parents=True)
write_failure_atlas(OUTPUT_DIR, report, tables, truth, prediction, state_names, model_name)
if activation_diagnostics is not None and activation_rows is not None:
    write_activation_diagnostics(OUTPUT_DIR, activation_diagnostics, activation_rows)

shutil.copy2(REPO_ROOT / "configs/research_contract_v1.json", OUTPUT_DIR / "research_contract_v1.json")
shutil.copy2(REPO_ROOT / "research/literature_evidence.csv", OUTPUT_DIR / "literature_evidence.csv")
experiment_card = {
    "experiment_id": "FA-00",
    "status": "completed_read_only_diagnosis",
    "question": "Where and how does the frozen input-only GRU fail?",
    "primary_factor": "measurement only",
    "training_performed": False,
    "analysis_split": analysis_split,
    "confirmatory_test_consumed": analysis_split == "test",
    "decision": "review_atlas_before_preregistering_the_next_ablation",
}
(OUTPUT_DIR / "experiment_card_FA-00.json").write_text(
    json.dumps(experiment_card, indent=2), encoding="utf-8"
)

provenance = {
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "analysis_mode": analysis_mode,
    "model_name": model_name,
    "analysis_split": analysis_split,
    "confirmatory_test_consumed": analysis_split == "test",
    "repository": str(REPO_ROOT),
    "git_commit": _git_commit(),
    "dataset": str(dataset_path),
    "dataset_sha256": sha256_file(dataset_path),
    "checkpoint": None if checkpoint_path is None else str(checkpoint_path),
    "checkpoint_sha256": None if checkpoint_path is None else sha256_file(checkpoint_path),
    "prediction_archive": str(prediction_path) if analysis_mode == "prediction_archive" else None,
    "prediction_archive_sha256": sha256_file(prediction_path)
    if analysis_mode == "prediction_archive" and prediction_path is not None else None,
    "dataset_schema": context["schema_version"],
    "dataset_model": context["model"],
    "shape": list(truth.shape),
    "dt_ms": dt_ms,
    "max_trajectories": MAX_TRAJECTORIES,
    "python": sys.version,
    "torch": torch.__version__,
    "numpy": np.__version__,
}
(OUTPUT_DIR / "provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")

zip_path = Path(shutil.make_archive(str(OUTPUT_DIR), "zip", OUTPUT_DIR.parent, OUTPUT_DIR.name))
print("\nFailure Atlas complete")
print(json.dumps(report["summary"], indent=2))
print("Directory:", OUTPUT_DIR)
print("ZIP      :", zip_path, f"({zip_path.stat().st_size / 2**20:.1f} MiB)")
