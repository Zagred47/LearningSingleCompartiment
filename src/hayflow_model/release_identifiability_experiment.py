"""Release-identifiability and full-state flow-map experiment for notebook 04.

This is a diagnostic runner, not the HayFlow-Hines architecture.  It freezes
the B3 structured shared residual backbone from notebook 02b and varies only
the causal input view and explicit state ablations.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from ..hayflow_data.composite_flowmap import (
    DYNAMIC_CATEGORIES,
    EVENT_KINDS,
    INPUT_VIEWS,
    CompositeFlowmapBundle,
    CompositeTransitionStore,
    classify_regime,
    input_view_schema,
)
from ..hayflow_data.reconditioned_flowmap import (
    ReconditionedAuxiliaryNormalizer,
    ReconditionedStateNormalizer,
    ReconditioningConfig,
)
from ..hayflow_eval.flowmap_metrics import write_parquet
from ..hayflow_eval.release_flowmap_metrics import (
    branching_metrics,
    episode_bootstrap,
    episode_bootstrap_event_f1,
    identifiability_summary,
    macro_event_summary,
    pooled_event_metrics,
    release_flowmap_decision,
    voltage_fidelity_rows,
)
from .full_state_flowmap import (
    DualRidgeBaseline,
    FlowmapModelConfig,
    PersistenceBaseline,
    parameter_count,
    require_torch,
    ridge_design_matrix,
    structured_arrays,
)
from .reconditioned_full_state import ReconditionedStructuredResidual


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    def safe(value: Any) -> Any:
        if isinstance(value, Mapping):
            return {str(key): safe(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [safe(item) for item in value]
        if isinstance(value, np.generic):
            value = value.item()
        if isinstance(value, float) and not math.isfinite(value):
            return None
        return value

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(safe(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _stable_hash(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[2],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


class Progress:
    def __init__(self, label: str, total: int) -> None:
        self.label = label
        self.total = max(1, int(total))
        self.started = time.monotonic()

    def update(self, value: int, detail: str = "") -> None:
        elapsed = time.monotonic() - self.started
        rate = value / elapsed if value and elapsed else 0.0
        eta = (self.total - value) / rate if rate else math.inf
        eta_text = "?" if not math.isfinite(eta) else f"{eta / 60:.1f} min"
        print(
            f"[HayFlow 04][{self.label}] {value}/{self.total} "
            f"({100.0 * value / self.total:.1f}%) ETA {eta_text} {detail}",
            flush=True,
        )


@dataclass(frozen=True)
class ReleaseExperimentConfig:
    profile: str = "diagnostic_full"
    initialization_seeds: Tuple[int, ...] = (17, 29, 43)
    maximum_epochs: int = 40
    early_stopping_patience: int = 7
    batch_size: int = 2
    evaluation_batch_size: int = 2
    learning_rate: float = 2e-4
    weight_decay: float = 1e-5
    gradient_clip_norm: float = 1.0
    ridge_alpha: float = 10.0
    normalization_transition_limit: int = 2048
    ridge_transition_limit: int = 768
    evaluation_transition_limit: int = 0
    transitions_per_episode_per_epoch: int = 2
    rollout_horizons_ms: Tuple[int, ...] = (2, 4, 8, 16, 32)
    rollout_windows_per_split: int = 32
    activity_epsilon: float = 1e-9
    sparse_update_fraction: float = 0.10
    minimum_scale: float = 1e-8
    lambda_voltage: float = 1.0
    lambda_mechanism: float = 0.20
    lambda_calcium: float = 0.30
    lambda_synapse: float = 0.05
    lambda_activity: float = 0.10
    lambda_event: float = 0.30
    lambda_timing: float = 0.03
    lambda_region: float = 0.01
    lambda_privileged: float = 0.003
    branching_teacher_distance_floor_mv: float = 0.05
    catastrophic_drift_mv: float = 5.0
    bootstrap_replicates: int = 2000

    def effective(self) -> "ReleaseExperimentConfig":
        if self.profile not in {"smoke", "diagnostic_full"}:
            raise ValueError("profile must be smoke or diagnostic_full")
        if self.profile == "diagnostic_full":
            if len(self.initialization_seeds) < 3:
                raise ValueError("diagnostic_full requires at least three common seeds")
            return self
        values = asdict(self)
        values.update(
            initialization_seeds=(self.initialization_seeds[0],),
            maximum_epochs=2,
            early_stopping_patience=1,
            normalization_transition_limit=64,
            ridge_transition_limit=32,
            evaluation_transition_limit=64,
            transitions_per_episode_per_epoch=1,
            rollout_horizons_ms=(2,),
            rollout_windows_per_split=2,
            bootstrap_replicates=100,
        )
        return ReleaseExperimentConfig(**values)


@dataclass(frozen=True)
class ReleaseRunSpec:
    model: str
    input_view: str
    state_ablation: str = "C_core_full"
    privileged: bool = False

    def validate(self) -> None:
        if self.model not in {"B1_ridge", "B3_structured"}:
            raise ValueError(self.model)
        if self.input_view not in INPUT_VIEWS:
            raise ValueError(self.input_view)
        if self.state_ablation not in {
            "A_voltage", "B_voltage_calcium", "C_core_full", "D_full_privileged"
        }:
            raise ValueError(self.state_ablation)

    def identifier(self, seed: Optional[int] = None) -> str:
        self.validate()
        suffix = f"-seed{int(seed)}" if seed is not None else ""
        return f"{self.model}-{self.input_view}-{self.state_ablation}{suffix}"


class ReleaseIdentifiabilityExperiment:
    """End-to-end runner with resumable, fingerprinted B1/B3 checkpoints."""

    def __init__(
        self,
        bundle: CompositeFlowmapBundle,
        output_dir: Path,
        config: ReleaseExperimentConfig,
    ) -> None:
        self.bundle = bundle
        self.output_dir = Path(output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.figure_dir = self.output_dir / "figures"
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.figure_dir.mkdir(exist_ok=True)
        self.config = config.effective()
        self.store = CompositeTransitionStore(bundle)
        self.layout = self.store.layout
        self.code_commit = _git_commit()
        self.normalizer: Optional[ReconditionedStateNormalizer] = None
        self.aux_normalizer: Optional[ReconditionedAuxiliaryNormalizer] = None
        self.rows: Dict[str, List[Dict[str, Any]]] = {
            "seed": [], "one_step": [], "rollout": [], "events": [],
            "drift": [], "attenuation": [], "branching": [], "recovery": [],
            "ablation": [],
        }
        self.registry: Dict[str, Dict[str, Any]] = {}

    def _episode_id(self, index: int) -> str:
        trajectory = str(self.store.metadata["trajectory_id"][int(index)])
        row = self.store.episode_by_trajectory.get(trajectory, {})
        return str(row.get("episode_id", trajectory))

    def _regime(self, index: int) -> str:
        trajectory = str(self.store.metadata["trajectory_id"][int(index)])
        return classify_regime(
            self.store.episode_by_trajectory.get(trajectory, {}),
            self.store.metadata["category"][int(index)],
        )

    @staticmethod
    def _episode_regimes(episode: Mapping[str, Any], category: Any) -> List[str]:
        split = str(episode.get("split", ""))
        if split == "recovery_test":
            return ["recovery"]
        if split == "branching_near_test":
            return ["branching_near"]
        if split == "branching_far_test":
            return ["branching_far"]
        hard = episode.get("hard_negative_for", ())
        if isinstance(hard, str):
            hard_present = hard not in {"", "[]", "nan"}
        else:
            hard_present = len(hard) > 0 if hard is not None else False
        labels = episode.get("event_labels", ())
        if isinstance(labels, str):
            present = [kind for kind in EVENT_KINDS if kind in labels]
        else:
            label_set = set(labels) if labels is not None else set()
            present = [kind for kind in EVENT_KINDS if kind in label_set]
        regimes = (["hard_negative"] if hard_present else []) + present
        return regimes or ["rest_subthreshold" if "rest" in str(category) else "other"]

    def _state_mask(self, ablation: str) -> np.ndarray:
        mask = np.zeros(self.layout.state_width, dtype=np.float32)
        mask[self.layout.category_slices["voltage"]] = 1.0
        if ablation == "B_voltage_calcium":
            mask[self.layout.category_slices["calcium_ions"]] = 1.0
        elif ablation in {"C_core_full", "D_full_privileged"}:
            mask[:] = 1.0
        return mask

    def _stratified_indices(
        self,
        split: str,
        *,
        limit: int,
        seed: int,
        per_episode: Optional[int] = None,
        allow_duplicates: bool = False,
    ) -> np.ndarray:
        rng = np.random.default_rng(int(seed))
        groups: Dict[str, List[np.ndarray]] = {}
        accepted = (
            [name for name in self.store.split_indices if name not in {"train", "validation", "test"}]
            if split == "test" else [split]
        )
        for name in accepted:
            for trajectory in self.store.episode_indices(split=name):
                episode = self.store.episode_by_trajectory.get(
                    str(self.store.metadata["trajectory_id"][trajectory[0]]), {}
                )
                for regime in self._episode_regimes(
                    episode, self.store.metadata["category"][trajectory[0]]
                ):
                    groups.setdefault(regime, []).append(trajectory)
        selected: List[int] = []
        episodes = [(regime, values) for regime, values in sorted(groups.items())]
        if not episodes:
            return np.empty(0, dtype=np.int64)
        target_per_regime = max(1, int(math.ceil(limit / len(episodes)))) if limit else 0
        for _, trajectories in episodes:
            order = rng.permutation(len(trajectories))
            group_values: List[int] = []
            for position in order:
                indices = np.asarray(trajectories[int(position)], dtype=np.int64)
                count = per_episode or len(indices)
                if count < len(indices):
                    picks = np.linspace(0, len(indices) - 1, count, dtype=int)
                    indices = indices[picks]
                group_values.extend(indices.tolist())
                if target_per_regime and len(group_values) >= target_per_regime:
                    break
            if target_per_regime:
                group_values = group_values[:target_per_regime]
            selected.extend(group_values)
        values = np.asarray(
            selected if allow_duplicates else sorted(set(selected)), dtype=np.int64
        )
        if limit and len(values) > limit:
            values = np.sort(rng.choice(values, size=limit, replace=False))
        return values

    def prepare(self) -> Dict[str, Any]:
        loader_report = self.store.report()
        _write_json(self.output_dir / "composite_loader_report.json", loader_report)
        _write_json(self.output_dir / "input_view_schema.json", input_view_schema())
        sample = self._stratified_indices(
            "train", limit=self.config.normalization_transition_limit,
            seed=1103, per_episode=2,
        )
        state_t = self.store.read_state(sample, "t")
        state_t1 = self.store.read_state(sample, "t_plus_1")
        self.normalizer = ReconditionedStateNormalizer(
            self.layout,
            ReconditioningConfig(
                activity_epsilon=self.config.activity_epsilon,
                sparse_update_fraction=self.config.sparse_update_fraction,
                minimum_scale=self.config.minimum_scale,
                gate_transform="logit",
            ),
        ).fit(state_t, state_t1)
        aux = self.store.read_privileged(sample)
        self.aux_normalizer = ReconditionedAuxiliaryNormalizer(
            self.config.minimum_scale
        ).fit(
            aux,
            {"currents_conductances_t_plus_1": slice(0, aux.shape[1])},
            privileged_records=self.layout.privileged_records,
        )
        normalization = {
            "schema_version": "04-normalization-v1",
            "fit_split": "train",
            "fit_transition_count": len(sample),
            "sampling": "episode-aware regime-stratified",
            "sample_logical_indices_sha256": hashlib.sha256(sample.tobytes()).hexdigest(),
            "state": self.normalizer.to_dict(),
            "privileged": self.aux_normalizer.to_dict(),
        }
        _write_json(self.output_dir / "normalization_schema.json", normalization)
        model_configs = {
            "schema_version": "04-model-config-v1",
            "architecture_frozen_from_02b": True,
            "primary": "B3 structured shared residual, P0, hurdle, gate logit, U2/event-aware",
            "config": asdict(self.config),
            "views": list(INPUT_VIEWS),
            "common_seeds": list(self.config.initialization_seeds),
            "state_ablations": ["A_voltage", "B_voltage_calcium", "C_core_full", "D_full_privileged"],
            "phase_c_rollout_aware_finetuning": "not_run_optional; pre-finetuning results only",
            "ridge_seed_policy": "deterministic closed-form fit; identical stratified rows across input views",
            "not_implemented": ["Hines", "latent state", "morphology reduction", "S4", "Mamba", "mixed precision"],
        }
        _write_json(self.output_dir / "model_configs.json", model_configs)
        return {
            "loader": loader_report,
            "normalizer_fingerprint": self.normalizer.fingerprint(),
            "normalization_transition_count": len(sample),
        }

    @staticmethod
    def _action_signature(actions: Sequence[Mapping[str, Any]]) -> str:
        return _stable_hash(actions)

    @staticmethod
    def _release_regimes(releases: Sequence[Mapping[str, Any]]) -> List[str]:
        if not releases or not any(bool(row.get("release_success")) for row in releases):
            return ["no_release"]
        success = [row for row in releases if bool(row.get("release_success"))]
        excitatory = any(float(row.get("ampa_state_increment", 0.0)) or float(row.get("nmda_state_increment", 0.0)) for row in success)
        inhibitory = any(float(row.get("inhibitory_state_increment", 0.0)) for row in success)
        regimes: List[str] = []
        if len(success) < len(releases):
            regimes.append("partial_release")
        if excitatory:
            regimes.append("excitatory_release")
            if any(float(row.get("nmda_state_increment", 0.0)) for row in success):
                regimes.append("ampa_nmda_combined")
        if inhibitory:
            regimes.append("inhibitory_release")
        return regimes or ["release_other"]

    def analyze_identifiability(self) -> Dict[str, Any]:
        trajectories = [
            values for values in self.store.episode_indices(split="release_identifiability_test")
        ]
        candidates: Dict[Tuple[str, int, str], List[int]] = {}
        for indices in trajectories:
            trajectory = str(self.store.metadata["trajectory_id"][indices[0]])
            episode = self.store.episode_by_trajectory.get(trajectory, {})
            snapshot = str(episode.get("snapshot_id", episode.get("snapshot_source", "unknown")))
            for index in indices:
                scheduled = self.store.actions(int(index), "U_scheduled")
                key = (snapshot, int(self.store.metadata["step_index"][int(index)]), self._action_signature(scheduled))
                candidates.setdefault(key, []).append(int(index))
        pair_rows: List[Dict[str, Any]] = []
        visual_pairs: List[Dict[str, Any]] = []
        for key, indices in candidates.items():
            for left_position, left in enumerate(indices):
                for right in indices[left_position + 1:]:
                    left_release = self.store.releases(left)
                    right_release = self.store.releases(right)
                    if self._action_signature(left_release) == self._action_signature(right_release):
                        continue
                    initial_error = float(np.max(np.abs(self.store.read_state([left], "t") - self.store.read_state([right], "t"))))
                    if initial_error > 1e-5:
                        continue
                    targets = self.store.read_state([left, right], "t_plus_1")[:, : self.layout.segment_count]
                    distance = float(np.sqrt(np.mean((targets[0] - targets[1]) ** 2)))
                    scheduled_variance = 0.5 * distance ** 2
                    rng_differs = self._action_signature(self.store.actions(left, "U_rng")) != self._action_signature(self.store.actions(right, "U_rng"))
                    realized_differs = self._action_signature(self.store.actions(left, "U_realized")) != self._action_signature(self.store.actions(right, "U_realized"))
                    event_near = bool(self.store.events(left) or self.store.events(right))
                    release_regimes = self._release_regimes([*left_release, *right_release])
                    row = {
                        "pair_id": _stable_hash([left, right])[:16],
                        "left_logical_index": left, "right_logical_index": right,
                        "initial_max_state_error": initial_error,
                        "target_distance": distance,
                        "scheduled_residual_variance": scheduled_variance,
                        "rng_residual_variance": 0.0 if rng_differs else scheduled_variance,
                        "realized_residual_variance": 0.0 if realized_differs else scheduled_variance,
                        "rng_disambiguates": bool(rng_differs),
                        "realized_disambiguates": bool(realized_differs),
                        "release_regime": release_regimes[0],
                        "release_regimes": release_regimes,
                        "near_dendritic_event": event_near,
                    }
                    pair_rows.append(row)
                    if len(visual_pairs) < 8:
                        visual_pairs.append({**row, "scheduled_signature": key[2], "left_releases": left_release, "right_releases": right_release})
        report = identifiability_summary(pair_rows)
        report.update(
            {
                "schema_version": "04-identifiability-v1",
                "valid": bool(pair_rows),
                "pair_definition": "same full S_t, step and U_scheduled; different exact causal release outcome",
                "variance_estimator": "paired conditional lower bound 0.5 * squared voltage RMSE",
                "pair_rows": pair_rows,
                "visual_pair_examples": visual_pairs,
                "future_state_used_to_construct_inputs": False,
            }
        )
        _write_json(self.output_dir / "identifiability_report.json", report)
        return report

    def _batch(
        self,
        indices: Sequence[int],
        view: str,
        ablation: str,
        *,
        raw_override: Optional[np.ndarray] = None,
        include_target: bool = True,
    ) -> Dict[str, Any]:
        if self.normalizer is None:
            raise RuntimeError("prepare must be called first")
        indices = np.asarray(indices, dtype=np.int64)
        raw_t = self.store.read_state(indices, "t") if raw_override is None else np.asarray(raw_override, dtype=np.float64)
        state = self.normalizer.normalize_state(raw_t).astype(np.float32)
        state *= self._state_mask(ablation)[None, :]
        result: Dict[str, Any] = {"indices": indices, "raw_state_t": raw_t, "state_t": state}
        result.update(self.store.encode_inputs(indices, view))
        result.update(self.store.event_targets(indices))
        if include_target:
            raw_t1 = self.store.read_state(indices, "t_plus_1")
            delta, activity = self.normalizer.delta_and_activity(raw_t, raw_t1)
            result.update(raw_state_t_plus_1=raw_t1, delta_target=delta, activity_target=activity)
        return result

    def _torch_batch(self, raw: Mapping[str, Any], device: Any) -> Dict[str, Any]:
        import torch
        integer = {"indices", "u2_segment_ids", "event_region"}
        boolean = {"u2_mask", "event_timing_mask", "event_region_mask", "activity_target"}
        excluded = {"raw_state_t", "raw_state_t_plus_1"}
        result: Dict[str, Any] = {}
        for key, value in raw.items():
            if key in excluded:
                result[key] = value
            elif isinstance(value, np.ndarray):
                dtype = torch.long if key in integer else torch.bool if key in boolean else torch.float32
                result[key] = torch.as_tensor(value, dtype=dtype, device=device)
            else:
                result[key] = value
        return result

    def _model(self, privileged: bool, device: Any) -> Any:
        config = FlowmapModelConfig(
            "B3_structured", "full", "U2", privileged_loss=privileged,
            # The dense auxiliary output is intentionally unused; width one
            # avoids a zero-width Linear warning while the selective P1 loss
            # supervises only the per-variable current/conductance decoder.
            auxiliary_dense_dim=1 if privileged else 0,
        )
        return ReconditionedStructuredResidual(
            config, self.layout.to_model_metadata(), structured_arrays(self.layout)
        ).to(device)

    def _loss(self, output: Mapping[str, Any], batch: Mapping[str, Any], privileged: bool) -> Any:
        import torch
        import torch.nn.functional as functional
        sparse = torch.as_tensor(self.normalizer.sparse_mask, dtype=torch.bool, device=output["delta"].device)
        delta_error = functional.smooth_l1_loss(output["delta"], batch["delta_target"], reduction="none")
        activity = functional.binary_cross_entropy_with_logits(
            output["activity_logits"][:, sparse], batch["activity_target"][:, sparse].float()
        ) if bool(sparse.any()) else output["delta"].new_tensor(0.0)
        total = output["delta"].new_tensor(0.0)
        weights = {
            "voltage": self.config.lambda_voltage,
            "mechanism_states": self.config.lambda_mechanism,
            "calcium_ions": self.config.lambda_calcium,
            "synapse_states": self.config.lambda_synapse,
        }
        for category, weight in weights.items():
            total = total + float(weight) * delta_error[:, self.layout.category_slices[category]].mean()
        event = functional.binary_cross_entropy_with_logits(output["event_logits"], batch["event_presence"])
        timing_mask = batch["event_timing_mask"]
        timing = torch.abs(output["event_timing"] - batch["event_timing"])[timing_mask].mean() if bool(timing_mask.any()) else total.new_tensor(0.0)
        region_mask = batch["event_region_mask"]
        region = functional.cross_entropy(output["event_region_logits"][region_mask], batch["event_region"][region_mask]) if bool(region_mask.any()) else total.new_tensor(0.0)
        total = total + self.config.lambda_activity * activity + self.config.lambda_event * event + self.config.lambda_timing * timing + self.config.lambda_region * region
        if privileged:
            target_raw = self.store.read_privileged(batch["indices"].detach().cpu().numpy())
            target, mask = self.aux_normalizer.transform(target_raw)
            target_tensor = torch.as_tensor(target, dtype=torch.float32, device=total.device)
            mask_tensor = torch.as_tensor(mask, dtype=torch.bool, device=total.device)
            total = total + self.config.lambda_privileged * functional.smooth_l1_loss(output["privileged_current"][mask_tensor], target_tensor[mask_tensor])
        return total

    def _epoch_indices(self, seed: int) -> np.ndarray:
        return self._stratified_indices(
            "train", limit=0, seed=seed,
            per_episode=self.config.transitions_per_episode_per_epoch,
            allow_duplicates=True,
        )

    def _iter_batches(self, indices: Sequence[int], size: int) -> Iterable[np.ndarray]:
        values = np.asarray(indices, dtype=np.int64)
        for start in range(0, len(values), int(size)):
            yield values[start:start + int(size)]

    def _validation_score(self, model: Any, spec: ReleaseRunSpec, device: Any) -> Dict[str, float]:
        import torch
        indices = self._stratified_indices("validation", limit=512, seed=991, per_episode=4)
        voltage_error: List[np.ndarray] = []
        probabilities: List[np.ndarray] = []
        targets: List[np.ndarray] = []
        model.eval()
        with torch.no_grad():
            for batch_indices in self._iter_batches(indices, self.config.evaluation_batch_size):
                raw = self._batch(batch_indices, spec.input_view, spec.state_ablation)
                batch = self._torch_batch(raw, device)
                output = model(batch)
                activity = torch.sigmoid(output["activity_logits"]).cpu().numpy()
                prediction = self.normalizer.reconstruct(raw["raw_state_t"], output["delta"].cpu().numpy(), activity_probability=activity)
                voltage_error.append(prediction[:, :self.layout.segment_count] - raw["raw_state_t_plus_1"][:, :self.layout.segment_count])
                probabilities.append(torch.sigmoid(output["event_logits"]).cpu().numpy())
                targets.append(raw["event_presence"])
        error = np.concatenate(voltage_error)
        probability = np.concatenate(probabilities)
        truth = np.concatenate(targets).astype(bool)
        f1s = []
        for column in range(len(EVENT_KINDS)):
            pred = probability[:, column] >= 0.5
            tp = np.sum(pred & truth[:, column]); fp = np.sum(pred & ~truth[:, column]); fn = np.sum(~pred & truth[:, column])
            f1s.append(float(2 * tp / max(1, 2 * tp + fp + fn)))
        return {
            "voltage_rmse_mv": float(np.sqrt(np.mean(error ** 2))),
            "drift_mv": float(np.mean(error)),
            "macro_f1": float(np.mean(f1s)),
            "selection_score": float(np.sqrt(np.mean(error ** 2)) + abs(np.mean(error)) + 2 * (1 - np.mean(f1s))),
        }

    def train_b3(self, spec: ReleaseRunSpec, seed: int) -> Any:
        require_torch()
        import torch
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = self._model(spec.privileged, device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.config.learning_rate, weight_decay=self.config.weight_decay)
        contract = {
            "schema_version": "04-checkpoint-v1", "dataset_fingerprint": self.bundle.fingerprint,
            "normalizer_fingerprint": self.normalizer.fingerprint(), "code_commit": self.code_commit,
            "run_spec": asdict(spec), "seed": int(seed), "config": asdict(self.config),
        }
        contract["fingerprint"] = _stable_hash(contract)
        run_dir = self.checkpoint_dir / spec.identifier(seed) / contract["fingerprint"][:16]
        run_dir.mkdir(parents=True, exist_ok=True)
        _write_json(run_dir / "fingerprint.json", contract)
        best_path = run_dir / "best_selection.pt"
        last_path = run_dir / "last.pt"
        history: List[Dict[str, Any]] = []
        best = math.inf; patience = 0; start_epoch = 0
        if last_path.is_file():
            saved = torch.load(last_path, map_location=device)
            if saved.get("fingerprint") != contract["fingerprint"]:
                raise RuntimeError("stale checkpoint fingerprint refused")
            model.load_state_dict(saved["model"]); optimizer.load_state_dict(saved["optimizer"])
            history = list(saved["history"]); start_epoch = int(saved["epoch"]) + 1
            best = min((row["validation"]["selection_score"] for row in history), default=math.inf)
            patience = int(saved.get("patience", 0))
        progress = Progress(spec.identifier(seed), self.config.maximum_epochs)
        for epoch in range(start_epoch, self.config.maximum_epochs):
            indices = self._epoch_indices(seed + epoch)
            np.random.default_rng(seed + epoch).shuffle(indices)
            model.train(); losses = []
            for batch_indices in self._iter_batches(indices, self.config.batch_size):
                raw = self._batch(batch_indices, spec.input_view, spec.state_ablation)
                batch = self._torch_batch(raw, device)
                optimizer.zero_grad(set_to_none=True)
                output = model(batch)
                loss = self._loss(output, batch, spec.privileged)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.gradient_clip_norm)
                optimizer.step(); losses.append(float(loss.detach()))
            validation = self._validation_score(model, spec, device)
            history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "validation": validation})
            if validation["selection_score"] < best - 1e-6:
                best = validation["selection_score"]; patience = 0
                torch.save({"model": model.state_dict(), "fingerprint": contract["fingerprint"], "epoch": epoch}, best_path)
            else:
                patience += 1
            torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "fingerprint": contract["fingerprint"], "epoch": epoch, "history": history, "patience": patience}, last_path)
            progress.update(epoch + 1, f"loss={np.mean(losses):.4g} score={validation['selection_score']:.4g} V={validation['voltage_rmse_mv']:.3g} F1={validation['macro_f1']:.3f} drift={validation['drift_mv']:.3g}")
            if patience >= self.config.early_stopping_patience:
                break
        saved = torch.load(best_path, map_location=device)
        model.load_state_dict(saved["model"])
        self.registry[spec.identifier(seed)] = {
            "fingerprint": contract["fingerprint"], "checkpoint": str(best_path.relative_to(self.output_dir)),
            "parameter_count": parameter_count(model), "epochs_completed": len(history),
            "history": history,
        }
        return model

    def _calibrate_thresholds(self, probabilities: np.ndarray, targets: np.ndarray) -> np.ndarray:
        result = np.full(len(EVENT_KINDS), 0.5, dtype=float)
        for column in range(len(EVENT_KINDS)):
            best = (-1.0, 0.5)
            for threshold in np.linspace(0.05, 0.95, 19):
                pred = probabilities[:, column] >= threshold; truth = targets[:, column] > 0.5
                tp = np.sum(pred & truth); fp = np.sum(pred & ~truth); fn = np.sum(~pred & truth)
                f1 = 2 * tp / max(1, 2 * tp + fp + fn)
                if f1 > best[0]: best = (float(f1), float(threshold))
            result[column] = best[1]
        return result

    def _predict_b3(self, model: Any, spec: ReleaseRunSpec, indices: Sequence[int], raw_override: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
        import torch
        device = next(model.parameters()).device
        parts: Dict[str, List[np.ndarray]] = {key: [] for key in ("state", "event_probability", "event_timing", "event_region")}
        model.eval()
        with torch.no_grad():
            for start in range(0, len(indices), self.config.evaluation_batch_size):
                batch_indices = np.asarray(indices[start:start + self.config.evaluation_batch_size])
                override = None if raw_override is None else raw_override[start:start + len(batch_indices)]
                raw = self._batch(batch_indices, spec.input_view, spec.state_ablation, raw_override=override, include_target=False)
                batch = self._torch_batch(raw, device); output = model(batch)
                activity = torch.sigmoid(output["activity_logits"]).cpu().numpy()
                state = self.normalizer.reconstruct(raw["raw_state_t"], output["delta"].cpu().numpy(), activity_probability=activity)
                parts["state"].append(state)
                parts["event_probability"].append(torch.sigmoid(output["event_logits"]).cpu().numpy())
                parts["event_timing"].append(output["event_timing"].cpu().numpy())
                parts["event_region"].append(output["event_region_logits"].argmax(-1).cpu().numpy())
        return {key: np.concatenate(value) for key, value in parts.items()}

    def _evaluation_indices(self, split: str) -> np.ndarray:
        return self._stratified_indices(split, limit=self.config.evaluation_transition_limit, seed=7721, per_episode=None)

    def evaluate_one_step(self, model: Any, spec: ReleaseRunSpec, seed: int) -> np.ndarray:
        validation = self._evaluation_indices("validation")
        validation_prediction = self._predict_b3(model, spec, validation)
        thresholds = self._calibrate_thresholds(validation_prediction["event_probability"], self.store.event_targets(validation)["event_presence"])
        for split in ("validation", "test"):
            indices = self._evaluation_indices(split)
            predicted = self._predict_b3(model, spec, indices)
            target_state = self.store.read_state(indices, "t_plus_1")
            segment_regions = [str(row["region"]) for row in self.layout.segments]
            voltage_rows = voltage_fidelity_rows(
                predicted["state"][:, :self.layout.segment_count], target_state[:, :self.layout.segment_count],
                model=spec.identifier(seed), split=split, horizon_ms=1,
                segment_regions=segment_regions, regimes=[self._regime(int(index)) for index in indices],
            )
            for row in voltage_rows:
                row["seed"] = seed; row["input_view"] = spec.input_view; row["state_ablation"] = spec.state_ablation
                self.rows["one_step"].append(row)
                if "baseline_drift_mv" in row: self.rows["drift"].append(dict(row))
                if "peak_attenuation_mv" in row: self.rows["attenuation"].append(dict(row))
            target_events = self.store.event_targets(indices)
            event_rows = pooled_event_metrics(
                predicted["event_probability"], target_events["event_presence"],
                [self._episode_id(int(index)) for index in indices], model=spec.identifier(seed), split=split,
                thresholds=thresholds, timing_prediction=predicted["event_timing"], timing_target=target_events["event_timing"],
                timing_mask=target_events["event_timing_mask"], region_prediction=predicted["event_region"],
                region_target=target_events["event_region"], region_mask=target_events["event_region_mask"],
            )
            for row in event_rows:
                row.update(seed=seed, input_view=spec.input_view, state_ablation=spec.state_ablation, subset="all", evaluation="one_step", horizon_ms=1)
            self.rows["events"].extend(event_rows)
            if split == "test":
                bootstrap_rows = episode_bootstrap_event_f1(
                    predicted["event_probability"],
                    target_events["event_presence"],
                    [self._episode_id(int(index)) for index in indices],
                    thresholds=thresholds,
                    replicates=self.config.bootstrap_replicates,
                    seed=seed,
                )
                for row in bootstrap_rows:
                    row.update(
                        model=spec.identifier(seed), seed=seed,
                        input_view=spec.input_view,
                        state_ablation=spec.state_ablation, split=split,
                        horizon_ms=1,
                        metric="episode_bootstrap_event_f1",
                    )
                self.rows["seed"].extend(bootstrap_rows)
            regime_values = np.asarray([self._regime(int(index)) for index in indices], dtype=object)
            for regime in sorted(set(regime_values.tolist())):
                selected = regime_values == regime
                if not selected.any():
                    continue
                subset_rows = pooled_event_metrics(
                    predicted["event_probability"][selected],
                    target_events["event_presence"][selected],
                    [self._episode_id(int(index)) for index in indices[selected]],
                    model=spec.identifier(seed), split=split,
                    thresholds=thresholds,
                    timing_prediction=predicted["event_timing"][selected],
                    timing_target=target_events["event_timing"][selected],
                    timing_mask=target_events["event_timing_mask"][selected],
                    region_prediction=predicted["event_region"][selected],
                    region_target=target_events["event_region"][selected],
                    region_mask=target_events["event_region_mask"][selected],
                )
                for row in subset_rows:
                    row.update(
                        seed=seed, input_view=spec.input_view,
                        state_ablation=spec.state_ablation, subset=regime,
                        evaluation="one_step", horizon_ms=1,
                    )
                self.rows["events"].extend(subset_rows)
        return thresholds

    def _rollout_outputs(
        self, model: Any, spec: ReleaseRunSpec, indices: Sequence[int]
    ) -> Tuple[np.ndarray, np.ndarray]:
        state = self.store.read_state([int(indices[0])], "t")
        trace = []
        event_probabilities = []
        for index in indices:
            output = self._predict_b3(model, spec, [int(index)], raw_override=state)
            state = output["state"]
            trace.append(state[0].copy())
            event_probabilities.append(output["event_probability"][0].copy())
        return np.asarray(trace), np.asarray(event_probabilities)

    def _rollout_trajectory(self, model: Any, spec: ReleaseRunSpec, indices: Sequence[int]) -> np.ndarray:
        return self._rollout_outputs(model, spec, indices)[0]

    def evaluate_rollouts(
        self, model: Any, spec: ReleaseRunSpec, seed: int,
        thresholds: np.ndarray,
    ) -> None:
        segment_regions = [str(row["region"]) for row in self.layout.segments]
        for split in ("validation", "test"):
            for horizon in self.config.rollout_horizons_ms:
                windows = self.store.rollout_windows(split, horizon)[:self.config.rollout_windows_per_split]
                if not windows: continue
                prediction = []; target = []; episodes = []; regimes = []
                event_probability = []; event_target = []; event_episode = []
                for window in windows:
                    state_trace, probability_trace = self._rollout_outputs(model, spec, window)
                    prediction.append(state_trace[-1])
                    target.append(self.store.read_state([int(window[-1])], "t_plus_1")[0])
                    episodes.append(self._episode_id(int(window[0]))); regimes.append(self._regime(int(window[0])))
                    event_probability.append(probability_trace)
                    event_target.append(self.store.event_targets(window)["event_presence"])
                    event_episode.extend([self._episode_id(int(window[0]))] * len(window))
                pred = np.asarray(prediction); truth = np.asarray(target)
                rows = voltage_fidelity_rows(
                    pred[:, :self.layout.segment_count], truth[:, :self.layout.segment_count],
                    model=spec.identifier(seed), split=split, horizon_ms=horizon,
                    segment_regions=segment_regions, regimes=regimes,
                )
                for row in rows:
                    row.update(seed=seed, input_view=spec.input_view, state_ablation=spec.state_ablation)
                    self.rows["rollout"].append(row)
                    if "baseline_drift_mv" in row:
                        self.rows["drift"].append(dict(row))
                    if "peak_attenuation_mv" in row:
                        self.rows["attenuation"].append(dict(row))
                episode_rows = [
                    {"episode_id": episode, "rmse_mv": float(np.sqrt(np.mean((p[:self.layout.segment_count] - t[:self.layout.segment_count]) ** 2)))}
                    for episode, p, t in zip(episodes, pred, truth)
                ]
                bootstrap = episode_bootstrap(episode_rows, value_key="rmse_mv", replicates=self.config.bootstrap_replicates, seed=seed)
                self.rows["seed"].append({"model": spec.identifier(seed), "seed": seed, "input_view": spec.input_view, "state_ablation": spec.state_ablation, "split": split, "horizon_ms": horizon, "metric": "episode_bootstrap_voltage_rmse_mv", **bootstrap})
                rollout_event_rows = pooled_event_metrics(
                    np.concatenate(event_probability), np.concatenate(event_target),
                    event_episode, model=spec.identifier(seed), split=split,
                    thresholds=thresholds,
                )
                for row in rollout_event_rows:
                    row.update(
                        seed=seed, input_view=spec.input_view,
                        state_ablation=spec.state_ablation, subset="all",
                        evaluation="rollout", horizon_ms=horizon,
                    )
                self.rows["events"].extend(rollout_event_rows)

    def evaluate_branching_and_recovery(self, model: Any, spec: ReleaseRunSpec, seed: int) -> None:
        trajectories: Dict[str, Dict[str, Any]] = {}
        for split, kind in (("branching_near_test", "near"), ("branching_far_test", "far"), ("release_identifiability_test", "release_identifiability")):
            for indices in self.store.episode_indices(split=split):
                horizon = min(32, len(indices)); window = indices[:horizon]
                trajectory = str(self.store.metadata["trajectory_id"][window[0]])
                episode = self.store.episode_by_trajectory.get(trajectory, {})
                pair_id = str(episode.get("branch_pair_id", episode.get("release_pair_id", episode.get("snapshot_id", "unknown"))))
                predicted_trace, predicted_events = self._rollout_outputs(model, spec, window)
                teacher_events = self.store.event_targets(window)["event_presence"] > 0.5
                trajectories[trajectory] = {"pair_id": pair_id, "kind": kind, "indices": window, "initial": self.store.read_state([int(window[0])], "t")[0], "teacher": np.asarray([self.store.read_state([int(i)], "t_plus_1")[0, :self.layout.segment_count] for i in window]), "prediction": predicted_trace[:, :self.layout.segment_count], "teacher_events": teacher_events, "predicted_events": predicted_events >= 0.5}
        pair_inputs = []
        grouped: Dict[Tuple[str, str], List[Mapping[str, Any]]] = {}
        for row in trajectories.values(): grouped.setdefault((row["kind"], row["pair_id"]), []).append(row)
        for (kind, pair_id), values in grouped.items():
            if len(values) != 2: continue
            left, right = values
            if float(np.max(np.abs(left["initial"] - right["initial"]))) > 1e-5: continue
            teacher_event_divergence = not np.array_equal(left["teacher_events"], right["teacher_events"])
            predicted_event_divergence = not np.array_equal(left["predicted_events"], right["predicted_events"])
            pair_inputs.append({"pair_id": pair_id, "branching_kind": kind, "horizon_ms": len(left["indices"]), "teacher_a": left["teacher"], "teacher_b": right["teacher"], "prediction_a": left["prediction"], "prediction_b": right["prediction"], "divergent_event_correct": teacher_event_divergence == predicted_event_divergence})
        branch_rows, summary = branching_metrics(pair_inputs, teacher_distance_floor=self.config.branching_teacher_distance_floor_mv)
        for row in branch_rows: row.update(model=spec.identifier(seed), seed=seed, input_view=spec.input_view)
        self.rows["branching"].extend(branch_rows)
        self.registry[spec.identifier(seed)]["branching_summary"] = summary

        calcium = self.layout.category_slices["calcium_ions"]
        synapse = self.layout.category_slices["synapse_states"]
        for indices in self.store.episode_indices(split="recovery_test"):
            horizon = min(32, len(indices)); window = indices[:horizon]
            prediction, predicted_event_probability = self._rollout_outputs(model, spec, window)
            teacher = np.asarray([self.store.read_state([int(i)], "t_plus_1")[0] for i in window])
            voltage_error = np.sqrt(np.mean((prediction[:, :self.layout.segment_count] - teacher[:, :self.layout.segment_count]) ** 2, axis=1))
            threshold = 1.0
            recovered = np.flatnonzero(voltage_error <= threshold)
            target_events = self.store.event_targets(window)["event_presence"]
            self.rows["recovery"].append({
                "model": spec.identifier(seed), "seed": seed, "input_view": spec.input_view,
                "episode_id": self._episode_id(int(window[0])), "horizon_ms": horizon,
                "voltage_rmse_mv": float(np.sqrt(np.mean((prediction[:, :self.layout.segment_count] - teacher[:, :self.layout.segment_count]) ** 2))),
                "calcium_rmse": float(np.sqrt(np.mean((prediction[:, calcium] - teacher[:, calcium]) ** 2))),
                "synapse_state_rmse": float(np.sqrt(np.mean((prediction[:, synapse] - teacher[:, synapse]) ** 2))),
                "recovery_time_error_proxy_ms": int(recovered[0]) + 1 if len(recovered) else math.nan,
                "post_event_excitability_teacher_event_count": int(target_events[horizon // 2:].sum()),
                "post_event_excitability_predicted_event_count": int((predicted_event_probability[horizon // 2:] >= 0.5).sum()),
                "post_event_excitability_count_error": int(abs((predicted_event_probability[horizon // 2:] >= 0.5).sum() - target_events[horizon // 2:].sum())),
                "similar_voltage_different_history_test": "recovery episode family",
            })

    def run_ridge(self, view: str) -> None:
        sample = self._stratified_indices("train", limit=self.config.ridge_transition_limit, seed=221, per_episode=2)
        raw = self._batch(sample, view, "C_core_full")
        target = np.concatenate([raw["delta_target"], raw["event_presence"]], axis=1)
        features = ridge_design_matrix(raw, voltage_width=self.layout.segment_count, state_mode="full", input_encoding="U2")
        model = DualRidgeBaseline(self.config.ridge_alpha).fit(features, target)
        model_id = ReleaseRunSpec("B1_ridge", view).identifier()
        model.save(self.checkpoint_dir / f"{model_id}.npz")
        ridge_contract = {
            "schema_version": "04-ridge-checkpoint-v1",
            "dataset_fingerprint": self.bundle.fingerprint,
            "normalizer_fingerprint": self.normalizer.fingerprint(),
            "code_commit": self.code_commit,
            "input_view": view,
            "ridge_alpha": self.config.ridge_alpha,
            "training_logical_indices_sha256": hashlib.sha256(sample.tobytes()).hexdigest(),
        }
        ridge_contract["fingerprint"] = _stable_hash(ridge_contract)
        _write_json(self.checkpoint_dir / f"{model_id}.fingerprint.json", ridge_contract)
        validation = self._evaluation_indices("validation")
        val_raw = self._batch(validation, view, "C_core_full")
        val_output = model.predict(ridge_design_matrix(val_raw, voltage_width=self.layout.segment_count, state_mode="full", input_encoding="U2"))
        thresholds = self._calibrate_thresholds(np.clip(val_output[:, self.layout.state_width:], 0, 1), val_raw["event_presence"])
        for split in ("validation", "test"):
            indices = self._evaluation_indices(split); batch = self._batch(indices, view, "C_core_full")
            output = model.predict(ridge_design_matrix(batch, voltage_width=self.layout.segment_count, state_mode="full", input_encoding="U2"))
            state = self.normalizer.reconstruct(batch["raw_state_t"], output[:, :self.layout.state_width], apply_hurdle=False)
            voltage_rows = voltage_fidelity_rows(state[:, :self.layout.segment_count], batch["raw_state_t_plus_1"][:, :self.layout.segment_count], model=model_id, split=split, horizon_ms=1, segment_regions=[str(row["region"]) for row in self.layout.segments], regimes=[self._regime(int(index)) for index in indices])
            for row in voltage_rows: row.update(input_view=view, state_ablation="C_core_full", seed=-1)
            self.rows["one_step"].extend(voltage_rows)
            self.rows["drift"].extend(
                dict(row) for row in voltage_rows if "baseline_drift_mv" in row
            )
            self.rows["attenuation"].extend(
                dict(row) for row in voltage_rows if "peak_attenuation_mv" in row
            )
            event_rows = pooled_event_metrics(np.clip(output[:, self.layout.state_width:], 0, 1), batch["event_presence"], [self._episode_id(int(index)) for index in indices], model=model_id, split=split, thresholds=thresholds)
            for row in event_rows: row.update(input_view=view, state_ablation="C_core_full", seed=-1, subset="all", evaluation="one_step", horizon_ms=1)
            self.rows["events"].extend(event_rows)
        for split in ("validation", "test"):
            for horizon in self.config.rollout_horizons_ms:
                windows = self.store.rollout_windows(split, horizon)[:self.config.rollout_windows_per_split]
                prediction = []; target_state = []; episodes = []; regimes = []
                event_probability = []; event_target = []; event_episode = []
                for window in windows:
                    state = self.store.read_state([int(window[0])], "t")
                    window_probabilities = []
                    for index in window:
                        batch = self._batch([int(index)], view, "C_core_full", raw_override=state, include_target=False)
                        output = model.predict(
                            ridge_design_matrix(
                                batch, voltage_width=self.layout.segment_count,
                                state_mode="full", input_encoding="U2",
                            )
                        )
                        state = self.normalizer.reconstruct(
                            state, output[:, :self.layout.state_width], apply_hurdle=False
                        )
                        window_probabilities.append(
                            np.clip(output[0, self.layout.state_width:], 0, 1)
                        )
                    prediction.append(state[0])
                    target_state.append(self.store.read_state([int(window[-1])], "t_plus_1")[0])
                    episodes.append(self._episode_id(int(window[0])))
                    regimes.append(self._regime(int(window[0])))
                    event_probability.append(np.asarray(window_probabilities))
                    event_target.append(self.store.event_targets(window)["event_presence"])
                    event_episode.extend([self._episode_id(int(window[0]))] * len(window))
                if not prediction:
                    continue
                rows = voltage_fidelity_rows(
                    np.asarray(prediction)[:, :self.layout.segment_count],
                    np.asarray(target_state)[:, :self.layout.segment_count],
                    model=model_id, split=split, horizon_ms=horizon,
                    segment_regions=[str(row["region"]) for row in self.layout.segments],
                    regimes=regimes,
                )
                for row in rows:
                    row.update(input_view=view, state_ablation="C_core_full", seed=-1)
                self.rows["rollout"].extend(rows)
                self.rows["drift"].extend(
                    dict(row) for row in rows if "baseline_drift_mv" in row
                )
                self.rows["attenuation"].extend(
                    dict(row) for row in rows if "peak_attenuation_mv" in row
                )
                rollout_event_rows = pooled_event_metrics(
                    np.concatenate(event_probability), np.concatenate(event_target),
                    event_episode, model=model_id, split=split,
                    thresholds=thresholds,
                )
                for row in rollout_event_rows:
                    row.update(
                        input_view=view, state_ablation="C_core_full", seed=-1,
                        subset="all", evaluation="rollout", horizon_ms=horizon,
                    )
                self.rows["events"].extend(rollout_event_rows)
        self.registry[model_id] = {
            "training_transition_count": len(sample),
            "ridge_alpha": self.config.ridge_alpha,
            "checkpoint": f"checkpoints/{model_id}.npz",
            "fingerprint": ridge_contract["fingerprint"],
        }

    def run_persistence(self) -> None:
        for split in ("validation", "test"):
            indices = self._evaluation_indices(split)
            state = self.store.read_state(indices, "t"); target = self.store.read_state(indices, "t_plus_1")
            rows = voltage_fidelity_rows(state[:, :self.layout.segment_count], target[:, :self.layout.segment_count], model=PersistenceBaseline.name, split=split, horizon_ms=1, segment_regions=[str(row["region"]) for row in self.layout.segments], regimes=[self._regime(int(index)) for index in indices])
            self.rows["one_step"].extend(rows)
            self.rows["drift"].extend(
                dict(row) for row in rows if "baseline_drift_mv" in row
            )
            self.rows["attenuation"].extend(
                dict(row) for row in rows if "peak_attenuation_mv" in row
            )
            event_target = self.store.event_targets(indices)
            persistence_events = pooled_event_metrics(
                np.zeros_like(event_target["event_presence"]),
                event_target["event_presence"],
                [self._episode_id(int(index)) for index in indices],
                model=PersistenceBaseline.name, split=split, thresholds=0.5,
            )
            for row in persistence_events:
                row.update(input_view="none", state_ablation="C_core_full", seed=-1, subset="all", evaluation="one_step", horizon_ms=1)
            self.rows["events"].extend(persistence_events)
            for horizon in self.config.rollout_horizons_ms:
                windows = self.store.rollout_windows(split, horizon)[:self.config.rollout_windows_per_split]
                if not windows:
                    continue
                initial = np.asarray([self.store.read_state([int(window[0])], "t")[0, :self.layout.segment_count] for window in windows])
                future = np.asarray([self.store.read_state([int(window[-1])], "t_plus_1")[0, :self.layout.segment_count] for window in windows])
                rollout_rows = voltage_fidelity_rows(
                    initial, future, model=PersistenceBaseline.name, split=split,
                    horizon_ms=horizon,
                    segment_regions=[str(row["region"]) for row in self.layout.segments],
                    regimes=[self._regime(int(window[0])) for window in windows],
                )
                self.rows["rollout"].extend(rollout_rows)
                self.rows["drift"].extend(
                    dict(row) for row in rollout_rows if "baseline_drift_mv" in row
                )
                self.rows["attenuation"].extend(
                    dict(row) for row in rollout_rows if "peak_attenuation_mv" in row
                )

    def run(self) -> Dict[str, Any]:
        if self.normalizer is None: self.prepare()
        identifiability = self.analyze_identifiability()
        self.run_persistence()
        for view in INPUT_VIEWS: self.run_ridge(view)
        total = len(INPUT_VIEWS) * len(self.config.initialization_seeds)
        outer = Progress("primary B3 runs", total); complete = 0
        for view in INPUT_VIEWS:
            for seed in self.config.initialization_seeds:
                spec = ReleaseRunSpec("B3_structured", view)
                model = self.train_b3(spec, seed)
                thresholds = self.evaluate_one_step(model, spec, seed)
                self.evaluate_rollouts(model, spec, seed, thresholds)
                self.evaluate_branching_and_recovery(model, spec, seed)
                complete += 1; outer.update(complete, spec.identifier(seed))
                del model
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except ImportError:
                    pass
        # State sufficiency ablations use the first common seed and U_realized.
        ablation_seed = self.config.initialization_seeds[0]
        for name in ("A_voltage", "B_voltage_calcium", "D_full_privileged"):
            spec = ReleaseRunSpec("B3_structured", "U_realized", name, name == "D_full_privileged")
            model = self.train_b3(spec, ablation_seed)
            thresholds = self.evaluate_one_step(model, spec, ablation_seed)
            self.evaluate_rollouts(model, spec, ablation_seed, thresholds)
            del model
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass
        return self.finalize(identifiability)

    def _write_tables(self) -> None:
        outputs = {
            "seed": "seed_metrics.parquet", "one_step": "one_step_metrics.parquet",
            "rollout": "rollout_metrics.parquet", "events": "event_metrics_pooled.parquet",
            "drift": "regional_drift.parquet", "attenuation": "peak_attenuation.parquet",
            "branching": "branching_metrics.parquet", "recovery": "recovery_metrics.parquet",
            "ablation": "state_ablation_metrics.parquet",
        }
        # State-ablation rows are the relevant subset of one-step/rollout rows.
        self.rows["ablation"] = [row for row in [*self.rows["one_step"], *self.rows["rollout"]] if str(row.get("state_ablation", "")).startswith(("A_", "B_", "C_", "D_")) and row.get("input_view") == "U_realized"]
        for key, filename in outputs.items():
            write_parquet(self.output_dir / filename, self.rows[key])

    @staticmethod
    def _mean(rows: Sequence[Mapping[str, Any]], key: str, **filters: Any) -> float:
        values = [float(row[key]) for row in rows if key in row and "regime" not in row and all(row.get(name) == value for name, value in filters.items()) and math.isfinite(float(row[key]))]
        return float(np.mean(values)) if values else math.nan

    def _make_figures(self, identifiability: Mapping[str, Any]) -> None:
        import matplotlib.pyplot as plt

        primary = [
            row for row in self.rows["rollout"]
            if row.get("split") == "test" and row.get("region") == "all"
            and "regime" not in row and row.get("state_ablation") == "C_core_full"
        ]
        figure, axis = plt.subplots(figsize=(8, 4.5))
        for model_prefix in ("B1_ridge", "B3_structured"):
            for view in INPUT_VIEWS:
                values = [row for row in primary if str(row.get("model", "")).startswith(model_prefix) and row.get("input_view") == view]
                horizons = sorted({int(row["horizon_ms"]) for row in values})
                means = [np.mean([float(row["rmse_mv"]) for row in values if int(row["horizon_ms"]) == horizon]) for horizon in horizons]
                if horizons:
                    axis.plot(horizons, means, marker="o", label=f"{model_prefix.split('_')[0]} {view[2:]}")
        axis.set(xlabel="rollout (ms)", ylabel="voltage RMSE (mV)", title="Release-aware flow-map rollout")
        axis.legend(ncol=2, fontsize=8); axis.grid(alpha=0.25)
        figure.tight_layout(); figure.savefig(self.figure_dir / "rollout_by_input_view.png", dpi=180); plt.close(figure)

        overall = identifiability.get("overall", {})
        figure, axis = plt.subplots(figsize=(6, 4))
        names = ("scheduled", "rng", "realized")
        values = [float(overall.get(f"{name}_residual_variance", math.nan)) for name in names]
        axis.bar(names, values, color=("#777777", "#4c78a8", "#f58518"))
        axis.set(ylabel="paired residual variance lower bound (mV²)", title="Synaptic identifiability")
        figure.tight_layout(); figure.savefig(self.figure_dir / "identifiability_variance.png", dpi=180); plt.close(figure)

        pair_examples = list(identifiability.get("visual_pair_examples", ()))[:4]
        if pair_examples:
            figure, axes = plt.subplots(
                len(pair_examples), 1, figsize=(8, 2.6 * len(pair_examples)),
                squeeze=False,
            )
            for axis, pair in zip(axes[:, 0], pair_examples):
                left = self.store.microtrace(
                    int(pair["left_logical_index"]), "probe_voltage"
                )
                right = self.store.microtrace(
                    int(pair["right_logical_index"]), "probe_voltage"
                )
                probe = int(
                    np.unravel_index(np.argmax(np.abs(left - right)), left.shape)[1]
                )
                time_ms = np.linspace(0.0, 1.0, left.shape[0])
                axis.plot(time_ms, left[:, probe], label="future A")
                axis.plot(time_ms, right[:, probe], label="future B")
                axis.set(
                    ylabel="voltage (mV)",
                    title=(
                        f"same S_t/U_scheduled, different release — probe {probe}, "
                        f"Δtarget={float(pair['target_distance']):.3g} mV"
                    ),
                )
                axis.grid(alpha=0.25); axis.legend(fontsize=8)
            axes[-1, 0].set_xlabel("time inside macro-step (ms)")
            figure.tight_layout()
            figure.savefig(
                self.figure_dir / "scheduled_identical_realized_different_pairs.png",
                dpi=180,
            )
            plt.close(figure)

        event_rows = [row for row in self.rows["events"] if row.get("split") == "test" and row.get("input_view") == "U_realized" and str(row.get("model", "")).startswith("B3_structured") and row.get("state_ablation") == "C_core_full" and row.get("subset") == "all" and row.get("evaluation") == "one_step"]
        figure, axis = plt.subplots(figsize=(8, 4))
        kinds = list(EVENT_KINDS)
        f1 = [np.mean([float(row["f1"]) for row in event_rows if row["event_kind"] == kind]) if any(row["event_kind"] == kind for row in event_rows) else 0.0 for kind in kinds]
        axis.bar(range(len(kinds)), f1, color="#54a24b")
        axis.set_xticks(range(len(kinds)), [name.replace("_", "\n") for name in kinds])
        axis.set(ylabel="pooled F1", ylim=(0, 1), title="B3-U_realized event fidelity")
        figure.tight_layout(); figure.savefig(self.figure_dir / "event_f1_realized.png", dpi=180); plt.close(figure)

    def finalize(self, identifiability: Mapping[str, Any]) -> Dict[str, Any]:
        self._write_tables()
        event_test = [row for row in self.rows["events"] if row.get("split") == "test" and str(row.get("model", "")).startswith("B3_structured-U_realized-C_core_full") and row.get("subset") == "all" and row.get("evaluation") == "one_step"]
        if event_test:
            aggregate_event_rows = [
                {
                    "event_kind": kind,
                    "f1": float(np.mean([float(row["f1"]) for row in event_test if row["event_kind"] == kind])),
                }
                for kind in EVENT_KINDS
            ]
            event_summary = macro_event_summary(aggregate_event_rows)
            dendritic = {
                "backpropagating_ap", "calcium_spike", "nmda_spike", "nmda_plateau"
            }
            event_summary["macro_pr_auc_dendritic"] = float(
                np.mean([float(row["pr_auc"]) for row in event_test if row["event_kind"] in dendritic and math.isfinite(float(row["pr_auc"]))])
            )
            event_summary["macro_prevalence_dendritic"] = float(
                np.mean([float(row["positive_prevalence"]) for row in event_test if row["event_kind"] in dendritic])
            )
        else:
            event_summary = {"macro_f1_dendritic": 0.0, "macro_f1_overall": 0.0, "macro_f1_somatic_axonal": 0.0}
        realized_b3 = self._mean(self.rows["rollout"], "rmse_mv", input_view="U_realized", state_ablation="C_core_full", split="test", region="all")
        realized_b1 = self._mean(self.rows["rollout"], "rmse_mv", input_view="U_realized", state_ablation="C_core_full", split="test", region="all", seed=-1)
        drifts = [abs(float(row["baseline_drift_mv"])) for row in self.rows["rollout"] if row.get("input_view") == "U_realized" and row.get("state_ablation") == "C_core_full" and row.get("region") != "all" and "baseline_drift_mv" in row]
        branch_values = [float(row["divergence_retention"]) for row in self.rows["branching"] if row.get("input_view") == "U_realized" and row.get("eligible")]
        recovery_values = [float(row["voltage_rmse_mv"]) for row in self.rows["recovery"] if row.get("input_view") == "U_realized"]
        recovery_event_errors = [float(row["post_event_excitability_count_error"]) for row in self.rows["recovery"] if row.get("input_view") == "U_realized"]
        ambiguity = identifiability.get("overall", {})
        seed_rollout = {}
        for seed in self.config.initialization_seeds:
            seed_rollout[seed] = self._mean(self.rows["rollout"], "rmse_mv", input_view="U_realized", state_ablation="C_core_full", split="test", region="all", seed=seed)
        finite_seed = [value for value in seed_rollout.values() if math.isfinite(value)]
        b3_consistently_better = bool(
            math.isfinite(realized_b1)
            and len(finite_seed) == len(self.config.initialization_seeds)
            and all(value < realized_b1 for value in finite_seed)
        )
        criteria = {
            "b3_realized_beats_b1_consistently": b3_consistently_better,
            "realized_reduces_scheduled_ambiguity": float(ambiguity.get("realized_variance_reduction_fraction", 0.0)) > 0.5,
            "dendritic_events_useful": bool(
                float(event_summary["macro_f1_dendritic"]) > 0.30
                and float(event_summary.get("macro_pr_auc_dendritic", 0.0))
                > float(event_summary.get("macro_prevalence_dendritic", 1.0)) + 0.05
            ),
            "no_catastrophic_regional_drift": bool(drifts and max(drifts) < self.config.catastrophic_drift_mv),
            "peak_attenuation_reduced": abs(self._mean(self.rows["attenuation"], "peak_attenuation_mv", input_view="U_realized", state_ablation="C_core_full", split="test")) < abs(self._mean(self.rows["attenuation"], "peak_attenuation_mv", model=PersistenceBaseline.name, split="test")),
            "divergence_retention_above_0_30": bool(branch_values and np.median(branch_values) > 0.30),
            "recovery_distinguishable": bool(
                recovery_values and recovery_event_errors
                and np.median(recovery_values) < self.config.catastrophic_drift_mv
                and np.median(recovery_event_errors) <= 1.0
            ),
            "stable_across_seeds": bool(len(finite_seed) >= len(self.config.initialization_seeds) and np.std(finite_seed) / max(np.mean(finite_seed), 1e-12) < 0.25),
        }
        decision = release_flowmap_decision(criteria)
        if self.config.profile != "diagnostic_full":
            decision = {
                "decision": "NON_DECISIONAL_SMOKE",
                "criteria": criteria,
                "blockers": ["smoke profile cannot support a scientific GO/NO-GO decision"],
            }
        report = {
            "schema_version": "04-final-report-v1",
            "valid": True,
            "decision_grade": self.config.profile == "diagnostic_full",
            "methodological_validity": {"composite_loader": self.store.report(), "checkpoint_fingerprints_complete": all("fingerprint" in row for key, row in self.registry.items() if key.startswith("B3_")), "profile": self.config.profile},
            "synaptic_identifiability": ambiguity,
            "flowmap_learnability": {"b3_realized_rollout_rmse_mv": realized_b3, "b1_realized_one_step_rmse_mv": realized_b1, "per_seed_rollout_rmse_mv": seed_rollout},
            "event_fidelity": event_summary,
            "decision": decision,
            "answers": {
                "1_realized_vs_scheduled": ambiguity,
                "2_rng_equals_realized": bool(ambiguity.get("rng_residual_variance") == ambiguity.get("realized_residual_variance")),
                "3_b3_vs_b1": criteria["b3_realized_beats_b1_consistently"],
                "4_advantage_with_horizon": "see rollout_metrics.parquet",
                "5_dendritic_events": event_summary,
                "6_positive_vs_hard_negative": "see event_metrics_pooled.parquet and regime rows",
                "7_peak_and_drift": "see peak_attenuation.parquet and regional_drift.parquet",
                "8_divergence_retention_above_0_30": criteria["divergence_retention_above_0_30"],
                "9_recovery": "see recovery_metrics.parquet",
                "10_full_state_gain": "see state_ablation_metrics.parquet",
            },
            "registry": self.registry,
        }
        self._make_figures(identifiability)
        _write_json(self.output_dir / "final_report.json", report)
        return report
