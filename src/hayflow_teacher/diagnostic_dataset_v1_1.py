"""Targeted HayFlow diagnostic transition dataset, schema 1.1.2."""

from __future__ import annotations

import json
import time
import hashlib
import math
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from ..hayflow_data import (
    TARGETED_DATASET_SCHEMA_VERSION,
    RELEASE_SCHEMA_VERSION,
    CausalReleaseOutcome,
    InputAction,
    ProtocolTrajectory,
    build_input_views,
    write_json,
    summarize_independent_support,
    validate_hdf5_store,
    validate_minimum_support,
    TargetedRecipe,
    action_schedule_from_json,
    select_adaptive_recipe_brackets,
)
from .audit import sha256_file
from .audit_runtime import PINNED_TEACHER_COMMIT
from .causal_release import CausalReleaseRecorder
from .diagnostic_dataset import (
    DEFAULT_MICROTRACE_STEP_MS,
    DiagnosticDatasetSession,
    _ConsoleProgress,
)
from .diagnostic_dataset_v1 import (
    DiagnosticDatasetV1Session,
    canonical_json_sha256,
)
from .event_extractor import annotate_backpropagation, extract_events
from .dendritic_calibration import DendriticCandidate, DendriticProtocolCalibrator


BAP_CONFOUNDING_EVENT_KINDS = frozenset(
    {
        "axonal_spike",
        "somatic_spike",
        "backpropagating_ap",
        "calcium_spike",
        "nmda_spike",
        "nmda_plateau",
    }
)
BAP_REGENERATIVE_EVENT_KINDS = frozenset(
    {"calcium_spike", "nmda_spike", "nmda_plateau"}
)


def _first_threshold_time(
    time_ms: Sequence[float],
    values: Sequence[float],
    *,
    start_ms: float,
    stop_ms: float,
    threshold_mv: float,
) -> Optional[float]:
    """Return the first threshold sample in a closed diagnostic window."""

    if len(time_ms) != len(values):
        raise ValueError("bAP diagnostic trace does not match the time grid")
    for time_value, signal_value in zip(time_ms, values):
        time_value = float(time_value)
        if time_value < float(start_ms) - 1.0e-12:
            continue
        if time_value > float(stop_ms) + 1.0e-12:
            break
        if float(signal_value) >= float(threshold_mv):
            return time_value
    return None


def assess_causal_bap(
    time_ms: Sequence[float],
    traces: Mapping[str, Sequence[float]],
    events: Sequence[Mapping[str, Any]],
    *,
    assist_only_events: Sequence[Mapping[str, Any]] = (),
    assist_only_peak_voltage_mv: Optional[float] = None,
    require_subthreshold_assist: bool = False,
    threshold_mv: float = -20.0,
    maximum_delay_ms: float = 3.0,
) -> Dict[str, Any]:
    """Validate a bAP label against temporal and counterfactual evidence.

    A combined somatic+dendritic protocol is eligible only when its dendritic
    arm is event-free in isolation.  In every arm the accepted event must be
    linked to a somatic spike, reach the trunk inside the configured delay and
    must not be preceded by a regenerative distal event or distal threshold
    crossing.  This prevents a locally initiated Ca/NMDA event from being
    relabelled as a backpropagating action potential.
    """

    if len(time_ms) == 0:
        raise ValueError("bAP assessment requires a non-empty time grid")
    if float(maximum_delay_ms) <= 0.0:
        raise ValueError("maximum bAP delay must be positive")

    assist_kinds = sorted({str(row["kind"]) for row in assist_only_events})
    assist_confounds = sorted(BAP_CONFOUNDING_EVENT_KINDS & set(assist_kinds))
    assist_peak_subthreshold = (
        assist_only_peak_voltage_mv is None
        or float(assist_only_peak_voltage_mv) < float(threshold_mv)
    )
    assist_subthreshold = not assist_confounds and assist_peak_subthreshold
    raw_baps = [
        (index, row)
        for index, row in enumerate(events)
        if str(row.get("kind")) == "backpropagating_ap"
    ]
    axonal_events = [
        row for row in events if str(row.get("kind")) == "axonal_spike"
    ]
    accepted_indices: List[int] = []
    candidate_reports: List[Dict[str, Any]] = []

    for event_index, event in raw_baps:
        reasons: List[str] = []
        trunk_onset = float(event["onset_ms"])
        linked_delay = event.get("linked_delay_ms")
        if linked_delay is None:
            reasons.append("missing_somatic_link")
            soma_onset = trunk_onset
        else:
            linked_delay = float(linked_delay)
            soma_onset = trunk_onset - linked_delay
            if linked_delay < -1.0e-12:
                reasons.append("trunk_precedes_soma")
            if linked_delay > float(maximum_delay_ms) + 1.0e-12:
                reasons.append("trunk_delay_exceeds_limit")

        if not any(
            soma_onset - float(maximum_delay_ms) - 1.0e-12
            <= float(row["onset_ms"])
            <= trunk_onset + 1.0e-12
            for row in axonal_events
        ):
            reasons.append("missing_nearby_axonal_spike")

        early_regenerative = sorted(
            {
                str(row["kind"])
                for row in events
                if str(row.get("kind")) in BAP_REGENERATIVE_EVENT_KINDS
                and float(row["onset_ms"]) < trunk_onset - 1.0e-12
            }
        )
        if early_regenerative:
            reasons.append("regenerative_dendritic_event_precedes_trunk")

        early_distal_crossings: Dict[str, float] = {}
        for label in ("nexus", "tuft", "voltage_event_probe_mv"):
            if label not in traces:
                continue
            crossing = _first_threshold_time(
                time_ms,
                traces[label],
                start_ms=soma_onset,
                stop_ms=trunk_onset,
                threshold_mv=threshold_mv,
            )
            if crossing is not None and crossing < trunk_onset - 1.0e-12:
                early_distal_crossings[label] = float(crossing)
        if early_distal_crossings:
            reasons.append("distal_threshold_crossing_precedes_trunk")

        if require_subthreshold_assist and not assist_subthreshold:
            reasons.append("assist_only_is_not_subthreshold")

        accepted = not reasons
        if accepted:
            accepted_indices.append(int(event_index))
        candidate_reports.append(
            {
                "event_index": int(event_index),
                "accepted": bool(accepted),
                "soma_onset_ms": float(soma_onset),
                "trunk_onset_ms": float(trunk_onset),
                "trunk_delay_ms": float(trunk_onset - soma_onset),
                "early_regenerative_event_kinds": early_regenerative,
                "early_distal_crossings_ms": early_distal_crossings,
                "rejection_reasons": reasons,
            }
        )

    return {
        "valid": bool(accepted_indices),
        "contract": "counterfactual_soma_origin_and_ordered_trunk_arrival",
        "threshold_mv": float(threshold_mv),
        "maximum_delay_ms": float(maximum_delay_ms),
        "require_subthreshold_assist": bool(require_subthreshold_assist),
        "assist_only_subthreshold": bool(assist_subthreshold),
        "assist_only_peak_voltage_mv": (
            None
            if assist_only_peak_voltage_mv is None
            else float(assist_only_peak_voltage_mv)
        ),
        "assist_only_peak_below_threshold": bool(assist_peak_subthreshold),
        "assist_only_event_kinds": assist_kinds,
        "assist_only_confounding_event_kinds": assist_confounds,
        "raw_bap_count": len(raw_baps),
        "accepted_bap_count": len(accepted_indices),
        "accepted_event_indices": accepted_indices,
        "candidates": candidate_reports,
    }


def _causal_event_kinds(
    events: Sequence[Mapping[str, Any]], assessment: Mapping[str, Any]
) -> List[str]:
    """Expose bAP to adaptive selection only after the causal gate passes."""

    kinds = {str(row["kind"]) for row in events}
    if not bool(assessment.get("valid")):
        kinds.discard("backpropagating_ap")
    return sorted(kinds)


def select_causal_bap_assist(
    trials: Sequence[Mapping[str, Any]],
    seeds: Sequence[int],
    *,
    threshold_mv: float = -20.0,
    subthreshold_margin_mv: float = 2.0,
) -> Dict[str, Any]:
    """Select the strongest dendritic arm that is subthreshold on every seed."""

    required_seeds = set(map(int, seeds))
    if not required_seeds:
        raise ValueError("BAP assist selection requires at least one seed")
    if float(subthreshold_margin_mv) < 0.0:
        raise ValueError("BAP assist subthreshold margin cannot be negative")
    ceiling = float(threshold_mv) - float(subthreshold_margin_mv)
    grouped: Dict[str, List[Mapping[str, Any]]] = {}
    for row in trials:
        if row.get("family") != "targeted_calcium":
            continue
        if bool(row.get("pair_with_somatic_spike", False)):
            continue
        grouped.setdefault(str(row["candidate_id"]), []).append(row)

    candidates = []
    for candidate_id, rows in sorted(grouped.items()):
        observed_seeds = {int(row["seed"]) for row in rows}
        confounds = sorted(
            BAP_CONFOUNDING_EVENT_KINDS
            & {
                str(kind)
                for row in rows
                for kind in row.get("event_kinds", ())
            }
        )
        peaks = [float(row["event_probe_peak_voltage_mv"]) for row in rows]
        reasons = []
        if observed_seeds != required_seeds:
            reasons.append("missing_required_seed")
        if confounds:
            reasons.append("event_detected_in_assist_only_arm")
        if peaks and max(peaks) > ceiling + 1.0e-12:
            reasons.append("assist_peak_exceeds_subthreshold_ceiling")
        if not peaks:
            reasons.append("missing_assist_peak_voltage")
        candidates.append(
            {
                "candidate_id": candidate_id,
                "eligible": not reasons,
                "event_probe_peak_voltage_mv_by_seed": {
                    str(row["seed"]): float(row["event_probe_peak_voltage_mv"])
                    for row in rows
                },
                "worst_seed_peak_voltage_mv": min(peaks) if peaks else None,
                "highest_peak_voltage_mv": max(peaks) if peaks else None,
                "confounding_event_kinds": confounds,
                "rejection_reasons": reasons,
            }
        )
    eligible = [row for row in candidates if row["eligible"]]
    selected = (
        max(
            eligible,
            key=lambda row: (
                float(row["worst_seed_peak_voltage_mv"]),
                float(row["highest_peak_voltage_mv"]),
                str(row["candidate_id"]),
            ),
        )
        if eligible
        else None
    )
    return {
        "valid": selected is not None,
        "policy": "strongest_robust_event_free_subthreshold_assist",
        "threshold_mv": float(threshold_mv),
        "subthreshold_margin_mv": float(subthreshold_margin_mv),
        "subthreshold_ceiling_mv": ceiling,
        "selected_candidate_id": (
            None if selected is None else str(selected["candidate_id"])
        ),
        "eligible_candidate_ids": [
            str(row["candidate_id"]) for row in eligible
        ],
        "candidates": candidates,
    }


class TargetedDiagnosticDatasetSession(DiagnosticDatasetV1Session):
    """Version 1.1 session with causal release logging and pilot gates."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        if kwargs.get("output_dir") is None:
            elm_repo = Path(args[0] if args else kwargs["elm_repo"])
            kwargs["output_dir"] = (
                elm_repo / "artifacts" / "diagnostic_dataset_v1_1"
            )
        super().__init__(*args, **kwargs)
        self.active_random123_seed = int(self.seed)
        self._active_transition_id = -1
        self._last_release_outcomes: List[CausalReleaseOutcome] = []
        self._last_release_verification: Dict[str, Any] = {}
        self._collect_release_rows = False
        self.release_rows: List[Dict[str, Any]] = []
        self.causal_release_pilot_report: Dict[str, Any] = {}
        self.targeted_preflight_report: Dict[str, Any] = {}
        self.targeted_recipe_catalog: List[TargetedRecipe] = []
        self.snapshot_bank: Dict[str, Dict[str, Any]] = {}
        self._preserve_raw_bap_events = False

    @staticmethod
    def _protocol_plan_sha256(
        protocols: Sequence[ProtocolTrajectory],
    ) -> str:
        """Bind actions, labels, split, branch, snapshot and recovery metadata."""

        payload = {
            "schema_version": TARGETED_DATASET_SCHEMA_VERSION,
            "trajectories": [
                {
                    "trajectory_id": row.trajectory_id,
                    "category": row.category,
                    "protocol": row.protocol,
                    "protocol_id": row.protocol_id,
                    "protocol_variant": row.protocol_variant,
                    "seed": int(row.seed),
                    "duration_ms": int(row.duration_ms),
                    "split": row.split,
                    "stimulus_onset_step": int(row.stimulus_onset_step),
                    "required_event_kinds": list(row.required_event_kinds),
                    "negative_control": bool(row.negative_control),
                    "snapshot_source": row.snapshot_source,
                    "metadata": dict(row.metadata),
                    "actions": {
                        str(step): [action.to_dict() for action in actions]
                        for step, actions in sorted(row.actions_by_step.items())
                    },
                }
                for row in sorted(protocols, key=lambda item: item.trajectory_id)
            ],
        }
        return canonical_json_sha256(payload)

    def prepare_targeted_contract(self) -> Dict[str, Any]:
        """Prepare v1.0.1 first, then add only versioned v1.1 metadata."""

        base = self.prepare_v1_contract()
        # NMDA event classes are hierarchical.  A sustained local regenerative
        # event is both an NMDA spike and, when it lasts at least 10 ms, an
        # NMDA plateau.  The initial mutually-exclusive <=9.975 ms rule was
        # rejected empirically: the canonical slow-NMDA teacher moved directly
        # from subthreshold responses to sustained events in the targeted
        # sweep, leaving no positive short-spike support.
        base_layout = {
            "index_contract": self.state_schema["index_contract"],
            "categories": self.state_schema["categories"],
            "rng_state": self.state_schema["rng_state"],
            "variables": self.state_schema["variables"],
        }
        base_layout_sha256 = canonical_json_sha256(base_layout)
        self.state_schema.update(
            {
                "schema_version": TARGETED_DATASET_SCHEMA_VERSION,
                "compatible_base_schema_version": "1.0.1",
                "canonical_state_layout_sha256": base_layout_sha256,
                "v1_1_extension_groups": (
                    "release_outcome,event_labels,input_views"
                ),
            }
        )
        write_json(self.output_dir / "state_schema.json", self.state_schema)
        release_schema = {
            "schema_version": RELEASE_SCHEMA_VERSION,
            "dataset_schema_version": TARGETED_DATASET_SCHEMA_VERSION,
            "teacher_commit": PINNED_TEACHER_COMMIT,
            "teacher_mechanisms_unchanged": True,
            "mechanisms": {
                "ProbAMPANMDA2": {
                    "random_distribution": "negexp(1)",
                    "release_rule": "erand() < Pr",
                    "causal_state_increments": [
                        "A_AMPA",
                        "B_AMPA",
                        "A_NMDA",
                        "B_NMDA",
                    ],
                },
                "ProbUDFsyn2": {
                    "random_distribution": "negexp(1)",
                    "release_rule": "erand() < Pr",
                    "causal_state_increments": ["A", "B"],
                },
            },
            "causal_boundary": {
                "pre": (
                    "boundary synapse state, ordered event schedule and cloned "
                    "Random123 position"
                ),
                "decision": (
                    "exact causal replay of the unchanged original NET_RECEIVE "
                    "equations before membrane integration"
                ),
                "post": (
                    "predicted same-timestamp synapse state, independently checked "
                    "against authentic teacher state and RNG at the 1 ms boundary"
                ),
                "forbidden_source": "S_(t+1) or any future membrane state",
            },
            "random_preview": {
                "purpose": (
                    "evaluate the causal front-end and validate it against the "
                    "authentic teacher without advancing the teacher RNG"
                ),
                "method": (
                    "independent Random123 with identical seed, stream and seq; "
                    "the teacher RNG is never advanced by instrumentation"
                ),
            },
            "boundary_verification": {
                "point_state_absolute_error": (
                    "reported diagnostically because CVODE numerically integrates "
                    "the fast A/B states"
                ),
                "release_state_gate": (
                    "for every event the observed boundary state must be closer "
                    "to the RNG-selected path than to the path obtained by "
                    "flipping that release outcome"
                ),
                "netcon_state_atol": CausalReleaseRecorder.NETCON_STATE_ATOL,
                "rng_sequence": "exact equality",
                "role": (
                    "validation only; S_(t+1) is never used to construct "
                    "U_realized"
                ),
            },
            "input_views": {
                "U_scheduled": "ordered presynaptic schedule only",
                "U_rng": "schedule plus causal Random123 identity and position",
                "U_realized": (
                    "successful releases and component increments causally "
                    "computed from the authentic front-end contract and verified "
                    "against the unchanged teacher"
                ),
            },
            "deployment_boundary": (
                "HayFlow retains the authentic synaptic front-end. U_realized is "
                "therefore available before the learned membrane core update and "
                "is not target leakage."
            ),
            "fields": list(CausalReleaseOutcome.__dataclass_fields__),
        }
        write_json(self.output_dir / "release_schema.json", release_schema)
        write_json(
            self.output_dir / "event_definition_config.json",
            {
                "schema_version": TARGETED_DATASET_SCHEMA_VERSION,
                "status": "targeted_configurable_thresholds",
                "definitions": [row.to_dict() for row in self.event_definitions],
                "nmda_regime_policy": (
                    "hierarchical labels: nmda_spike duration >= 1 ms; "
                    "nmda_plateau is the sustained subset with duration >= 10 ms"
                ),
                "bap_policy": (
                    "counterfactual soma-only/assist-only/combined calibration; "
                    "axonal and somatic origin; trunk arrival within 3 ms; reject "
                    "regenerative events or distal threshold crossings before trunk"
                ),
                "right_censoring_policy": (
                    "retain censored labels; exclude them from duration/offset targets"
                ),
            },
        )
        return {
            **base,
            "schema_version": TARGETED_DATASET_SCHEMA_VERSION,
            "canonical_state_layout_sha256": base_layout_sha256,
            "release_schema": "release_schema.json",
        }

    def _extract_events(
        self,
        time_ms: Sequence[float],
        traces: Mapping[str, Sequence[float]],
    ) -> List[Dict[str, Any]]:
        definitions = list(self.event_definitions)
        trajectory = self._active_trajectory
        if trajectory is not None and "voltage_event_probe_mv" in traces:
            probe_segment_id = trajectory.metadata.get("event_probe_segment_id")
            probe_kinds = set(
                trajectory.metadata.get(
                    "event_probe_kinds", ("nmda_spike", "nmda_plateau")
                )
            )
            if probe_segment_id is not None:
                definitions = [
                    replace(
                        definition,
                        signal="voltage_event_probe_mv",
                        segment_id=int(probe_segment_id),
                        region=str(
                            trajectory.metadata.get(
                                "event_probe_region", "stimulus_cluster"
                            )
                        ),
                    )
                    if definition.kind in probe_kinds
                    else definition
                    for definition in definitions
                ]
        events = self._enforce_nmda_event_hierarchy(
            extract_events(time_ms, traces, definitions)
        )
        distances = {}
        for label in ("soma", "basal", "trunk", "nexus", "tuft"):
            segment_id = self.audit.representatives.get(label)
            if segment_id is None:
                continue
            row = self.audit.segment_df[
                self.audit.segment_df["segment_id"] == int(segment_id)
            ]
            distances[label] = float(row.iloc[0]["distance_from_soma_um"])
        annotated = annotate_backpropagation(
            time_ms,
            traces,
            events,
            regional_distances_um=distances,
        )
        if self._preserve_raw_bap_events:
            return annotated
        assessment = assess_causal_bap(time_ms, traces, annotated)
        accepted = set(map(int, assessment["accepted_event_indices"]))
        filtered = []
        for event_index, event in enumerate(annotated):
            if str(event.get("kind")) != "backpropagating_ap":
                filtered.append(event)
                continue
            if event_index not in accepted:
                continue
            causal = dict(event)
            causal["causal_validation"] = assessment["contract"]
            causal["causal_validation_passed"] = True
            filtered.append(causal)
        return filtered

    @staticmethod
    def _enforce_nmda_event_hierarchy(
        events: Sequence[Mapping[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Guarantee that every plateau also carries its NMDA-spike parent label."""

        rows = [dict(row) for row in events]
        spikes = [row for row in rows if row.get("kind") == "nmda_spike"]
        for plateau in [row for row in rows if row.get("kind") == "nmda_plateau"]:
            matched = any(
                int(spike["segment_id"]) == int(plateau["segment_id"])
                and float(spike["onset_ms"]) <= float(plateau["offset_ms"])
                and float(plateau["onset_ms"]) <= float(spike["offset_ms"])
                for spike in spikes
            )
            if matched:
                continue
            parent = dict(plateau)
            parent["kind"] = "nmda_spike"
            parent["rule"] = "hierarchical_parent_of_nmda_plateau"
            parent["derived_from_event_kind"] = "nmda_plateau"
            parameters = dict(parent.get("parameters", {}))
            parameters.update(
                {
                    "kind": "nmda_spike",
                    "min_duration_ms": 1.0,
                    "maximum_event_duration_ms": None,
                }
            )
            parent["parameters"] = parameters
            rows.append(parent)
            spikes.append(parent)
        return rows

    def _extract_trajectory_events(
        self,
        trajectory: ProtocolTrajectory,
        time_ms: Sequence[float],
        traces: Mapping[str, Sequence[float]],
        *,
        preserve_raw_bap: bool = False,
    ) -> List[Dict[str, Any]]:
        """Extract with the trajectory-local probe contract still in scope."""

        previous = self._active_trajectory
        previous_raw_policy = self._preserve_raw_bap_events
        self._active_trajectory = trajectory
        self._preserve_raw_bap_events = bool(preserve_raw_bap)
        try:
            return self._extract_events(time_ms, traces)
        finally:
            self._active_trajectory = previous
            self._preserve_raw_bap_events = previous_raw_policy

    def _configure_rngs(
        self, seed: int, sequences: Sequence[float]
    ) -> None:
        super()._configure_rngs(seed, sequences)
        self.active_random123_seed = int(seed)

    @staticmethod
    def _available_pv_at_event(
        stored_pv: float, elapsed_ms: float, depression_tau_ms: float
    ) -> float:
        """Evaluate the canonical lazy STP recovery immediately before an event."""

        elapsed = float(elapsed_ms)
        depression_tau = float(depression_tau_ms)
        if elapsed < 0.0:
            raise ValueError("prospective synaptic event precedes stored tsyn")
        if depression_tau == 0.0:
            if elapsed <= 0.0:
                raise ValueError("Dep=0 recovery requires a positive event interval")
            return 1.0
        return 1.0 - (1.0 - float(stored_pv)) * math.exp(
            -elapsed / depression_tau
        )

    def _trajectory_initial_state(
        self, trajectory: ProtocolTrajectory
    ) -> Tuple[Path, Sequence[float], int]:
        snapshot_id = str(
            trajectory.metadata.get("snapshot_id", trajectory.snapshot_source)
        )
        if snapshot_id not in self.snapshot_bank:
            return super()._trajectory_initial_state(trajectory)
        record = self.snapshot_bank[snapshot_id]
        return (
            self.output_dir / str(record["native_snapshot"]),
            list(record["rng_sequences"]),
            int(record["random123_seed"]),
        )

    def prepare_snapshot_bank(
        self,
        protocols: Sequence[ProtocolTrajectory],
        *,
        conditioning_ms: int = 4,
    ) -> Dict[str, Any]:
        """Create real split-specific subthreshold teacher snapshots."""

        self._require_equilibrium()
        if int(conditioning_ms) <= 0:
            raise ValueError("snapshot conditioning duration must be positive")
        snapshot_splits: Dict[str, set] = {}
        for trajectory in protocols:
            snapshot_id = str(
                trajectory.metadata.get(
                    "snapshot_id", trajectory.snapshot_source
                )
            )
            snapshot_splits.setdefault(snapshot_id, set()).add(trajectory.split)
        leaking = {
            key: sorted(values)
            for key, values in snapshot_splits.items()
            if len(values) > 1
        }
        if leaking:
            raise ValueError(f"snapshot ids leak across splits: {leaking}")
        snapshot_ids = sorted(snapshot_splits)
        equilibrium_rng = json.loads(
            self.equilibrium_rng_path.read_text(encoding="utf-8")
        )
        excitatory = [
            int(row["synapse_id"])
            for row in self.audit.synapse_records
            if row["class_name"] == "ProbAMPANMDA2"
        ]
        inhibitory = [
            int(row["synapse_id"])
            for row in self.audit.synapse_records
            if row["class_name"] == "ProbUDFsyn2"
        ]
        destination = self.snapshots_dir / "initial_states"
        destination.mkdir(parents=True, exist_ok=True)
        progress = _ConsoleProgress("snapshot bank", len(snapshot_ids))
        bank = {}
        for index, snapshot_id in enumerate(snapshot_ids, start=1):
            digest = hashlib.sha256(snapshot_id.encode("utf-8")).digest()
            bank_seed = 800_000 + int.from_bytes(digest[:4], "big") % 100_000
            ex_id = excitatory[int.from_bytes(digest[4:8], "big") % len(excitatory)]
            inh_id = inhibitory[int.from_bytes(digest[8:12], "big") % len(inhibitory)]
            self._restore_native_snapshot(
                self.equilibrium_snapshot_path,
                equilibrium_rng["sequences"],
                equilibrium_rng.get("random123_seed", self.seed),
            )
            self._rekey_rngs(bank_seed)
            dummy = ProtocolTrajectory(
                trajectory_id=f"snapshot-conditioning-{snapshot_id}",
                category="rest_subthreshold",
                protocol="snapshot_conditioning",
                seed=bank_seed,
                duration_ms=int(conditioning_ms),
                split=next(iter(snapshot_splits[snapshot_id])),
                stimulus_onset_step=0,
            )
            peak_soma = -1.0e300
            for step in range(conditioning_ms):
                actions = (
                    InputAction(
                        "synaptic_event", 0.25, synapse_id=ex_id
                    ),
                    InputAction(
                        "synaptic_event", 0.65, synapse_id=inh_id
                    ),
                )
                row = self._run_transition(
                    -1, dummy, step, actions, snapshot_path=None
                )
                soma_index = list(self.audit.representatives).index("soma")
                peak_soma = max(
                    peak_soma,
                    float(self.np.max(row["micro_probe_voltage"][:, soma_index])),
                )
            if peak_soma >= -20.0:
                raise RuntimeError(
                    f"snapshot conditioning became suprathreshold for {snapshot_id}"
                )
            safe_name = hashlib.sha256(snapshot_id.encode("utf-8")).hexdigest()[:20]
            native_path = destination / f"{safe_name}.neuron.bin"
            self._write_native_snapshot(native_path)
            sequences = self.audit._snapshot_rng_sequences()
            bank[snapshot_id] = {
                "snapshot_id": snapshot_id,
                "splits": sorted(snapshot_splits[snapshot_id]),
                "native_snapshot": native_path.relative_to(self.output_dir).as_posix(),
                "native_snapshot_sha256": sha256_file(native_path),
                "random123_seed": bank_seed,
                "rng_sequences": list(map(float, sequences)),
                "conditioning_ms": int(conditioning_ms),
                "conditioning_synapse_ids": [ex_id, inh_id],
                "peak_soma_mv": peak_soma,
                "suprathreshold": False,
            }
            progress.update(
                index,
                detail=f"{snapshot_id}; peak soma={peak_soma:.2f} mV",
            )
        self.snapshot_bank = bank
        report = {
            "schema_version": TARGETED_DATASET_SCHEMA_VERSION,
            "valid": len(bank) == len(snapshot_ids) and not leaking,
            "snapshot_count": len(bank),
            "conditioning_ms": int(conditioning_ms),
            "split_specific": True,
            "shared_only_within_split": True,
            "snapshots": bank,
        }
        write_json(self.output_dir / "snapshot_bank.json", report)
        return report

    def _make_transition_writer(self, *args: Any, **kwargs: Any) -> Any:
        kwargs.update(
            {
                "input_view_names": ("U_scheduled", "U_rng", "U_realized"),
                "store_release_outcomes": True,
            }
        )
        return super()._make_transition_writer(*args, **kwargs)

    def _drive_one_ms(
        self,
        start_time: float,
        actions: Sequence[InputAction],
        observer: Any,
        *,
        sample_interval_ms: float = DEFAULT_MICROTRACE_STEP_MS,
    ) -> Tuple[Any, List[Dict[str, Any]], List[Any]]:
        """Canonical driver plus a causal, boundary-verified shadow front-end."""

        interval = float(sample_interval_ms)
        sample_count = int(round(1.0 / interval)) + 1
        if interval <= 0.0 or abs((sample_count - 1) * interval - 1.0) > 1e-9:
            raise ValueError("sample interval must divide one millisecond")
        self._disable_somatic_clamp()
        self._configure_somatic_current(start_time, actions)
        self.cvode.re_init()
        recorder = CausalReleaseRecorder(
            self,
            transition_id=self._active_transition_id,
            random123_seed=self.active_random123_seed,
        )
        public_actions = recorder.schedule(start_time, actions)
        times = self.np.linspace(
            float(start_time), float(start_time) + 1.0, sample_count
        )
        samples = []
        for sample_time in times:
            self.audit._advance_exact(float(sample_time))
            samples.append(observer())
        self._last_release_verification = recorder.verify_boundary(
            float(start_time) + 1.0
        )
        self._last_release_outcomes = recorder.outcomes()
        return self.np.asarray(times, dtype=float), public_actions, samples

    def _run_transition(
        self,
        transition_id: int,
        trajectory: ProtocolTrajectory,
        step_index: int,
        actions: Sequence[InputAction],
        snapshot_path: Optional[Path],
    ) -> Dict[str, Any]:
        self._active_transition_id = int(transition_id)
        self._last_release_outcomes = []
        self._last_release_verification = {}
        row = super()._run_transition(
            transition_id,
            trajectory,
            step_index,
            actions,
            snapshot_path,
        )
        views = build_input_views(row["inputs"], self._last_release_outcomes)
        releases = [outcome.to_dict() for outcome in self._last_release_outcomes]
        row["inputs"] = views["U_scheduled"]
        row["input_views"] = views
        row["release_outcomes"] = releases
        row["metadata"]["causal_release_boundary_verification"] = dict(
            self._last_release_verification
        )
        if self._collect_release_rows and int(transition_id) >= 0:
            for release in releases:
                self.release_rows.append(
                    {
                        "trajectory_id": trajectory.trajectory_id,
                        "protocol_id": trajectory.protocol_id,
                        "split": trajectory.split,
                        "seed": int(trajectory.seed),
                        "step_index": int(step_index),
                        **release,
                    }
                )
        return row

    @staticmethod
    def _outcome_signature(rows: Sequence[Mapping[str, Any]]) -> str:
        payload = []
        for row in rows:
            payload.append(
                {
                    key: value
                    for key, value in row.items()
                    if key not in {"transition_id"}
                }
            )
        return canonical_json_sha256({"outcomes": payload})

    def run_causal_release_pilot(
        self, *, transition_count: int = 8, seed: int = 410001
    ) -> Dict[str, Any]:
        """Prove causal logging and exact replay before dataset generation."""

        self._require_equilibrium()
        if int(transition_count) < 2:
            raise ValueError("causal release pilot needs at least two transitions")
        excitatory = [
            int(row["synapse_id"])
            for row in self.audit.synapse_records
            if row["class_name"] == "ProbAMPANMDA2"
        ][:4]
        inhibitory = [
            int(row["synapse_id"])
            for row in self.audit.synapse_records
            if row["class_name"] == "ProbUDFsyn2"
        ][:4]
        if len(excitatory) != 4 or len(inhibitory) != 4:
            raise RuntimeError("canonical release pilot synapse pool is incomplete")
        offsets = self.np.linspace(0.1, 0.8, 8)
        actions = tuple(
            InputAction("synaptic_event", float(offset), synapse_id=synapse_id)
            for offset, synapse_id in zip(offsets, excitatory + inhibitory)
        )
        trajectory = ProtocolTrajectory(
            trajectory_id=f"causal-release-pilot-seed{seed}",
            category="local_synaptic",
            protocol="causal_release_pilot",
            protocol_id="causal_release_pilot",
            protocol_variant="excitatory_and_inhibitory_repeated",
            seed=int(seed),
            duration_ms=int(transition_count),
            split="release_identifiability_test",
            actions_by_step={step: actions for step in range(transition_count)},
            stimulus_onset_step=0,
        )
        equilibrium_rng = json.loads(
            self.equilibrium_rng_path.read_text(encoding="utf-8")
        )

        def run_once(run_seed: int) -> Tuple[List[Dict[str, Any]], Dict[str, Any], float]:
            self._restore_native_snapshot(
                self.equilibrium_snapshot_path,
                equilibrium_rng["sequences"],
                equilibrium_rng.get("random123_seed", self.seed),
            )
            self._rekey_rngs(run_seed)
            rows = []
            started = time.perf_counter()
            for step_index in range(transition_count):
                rows.append(
                    self._run_transition(
                        -1,
                        trajectory,
                        step_index,
                        actions,
                        snapshot_path=None,
                    )
                )
            elapsed = time.perf_counter() - started
            final = {
                "state": self.capture_boundary_state(),
                "rng": self.np.asarray(
                    self.audit._snapshot_rng_sequences(), dtype=float
                ),
            }
            return rows, final, elapsed

        first, first_final, elapsed = run_once(int(seed))
        second, second_final, _ = run_once(int(seed))
        different_seed_rows, _, _ = run_once(int(seed) + 1)

        self._restore_native_snapshot(
            self.equilibrium_snapshot_path,
            equilibrium_rng["sequences"],
            equilibrium_rng.get("random123_seed", self.seed),
        )
        self._rekey_rngs(int(seed))
        for _ in range(transition_count):
            DiagnosticDatasetSession._drive_one_ms(
                self,
                float(self.h.t),
                actions,
                self._sample_transition_point,
                sample_interval_ms=DEFAULT_MICROTRACE_STEP_MS,
            )
        uninstrumented_final = {
            "state": self.capture_boundary_state(),
            "rng": self.np.asarray(
                self.audit._snapshot_rng_sequences(), dtype=float
            ),
        }
        first_releases = [
            release for row in first for release in row["release_outcomes"]
        ]
        second_releases = [
            release for row in second for release in row["release_outcomes"]
        ]
        different_releases = [
            release
            for row in different_seed_rows
            for release in row["release_outcomes"]
        ]
        state_error = max(
            float(
                self.np.max(
                    self.np.abs(
                        first_final["state"][category]
                        - second_final["state"][category]
                    )
                )
            )
            for category in self.state_variables
        )
        rng_error = float(
            self.np.max(self.np.abs(first_final["rng"] - second_final["rng"]))
        )
        uninstrumented_state_error = max(
            float(
                self.np.max(
                    self.np.abs(
                        first_final["state"][category]
                        - uninstrumented_final["state"][category]
                    )
                )
            )
            for category in self.state_variables
        )
        uninstrumented_rng_error = float(
            self.np.max(
                self.np.abs(
                    first_final["rng"] - uninstrumented_final["rng"]
                )
            )
        )
        same_release = self._outcome_signature(first_releases) == self._outcome_signature(
            second_releases
        )
        different_rng = any(
            abs(float(left["rng_preview_value"]) - float(right["rng_preview_value"]))
            > 0.0
            for left, right in zip(first_releases, different_releases)
        )

        pilot_dir = self.output_dir / "pilot"
        pilot_dir.mkdir(parents=True, exist_ok=True)
        pilot_path = pilot_dir / "causal_release_pilot.h5"
        widths = {
            category: len(variables)
            for category, variables in self.state_variables.items()
        }
        widths["rng_state"] = len(self.audit.synapse_rngs)
        with self._make_transition_writer(
            pilot_path,
            widths,
            41,
            len(self.micro_variables),
            len(self.audit.live_segments),
            len(self.audit.representatives),
            micro_observable_names=self.micro_observable_ids,
        ) as writer:
            writer.set_microtrace_grid(self.np.linspace(0.0, 1.0, 41))
            for index, row in enumerate(first):
                row["metadata"]["transition_id"] = index
                writer.append(row)
        release_counts = Counter(
            (row["synapse_type"], bool(row["release_success"]))
            for row in first_releases
        )
        byte_per_transition = int(pilot_path.stat().st_size / len(first))
        valid = bool(
            first_releases
            and same_release
            and state_error == 0.0
            and rng_error == 0.0
            and uninstrumented_state_error == 0.0
            and uninstrumented_rng_error == 0.0
            and different_rng
            and any(row["release_success"] for row in first_releases)
            and any(not row["release_success"] for row in first_releases)
        )
        report = {
            "schema_version": TARGETED_DATASET_SCHEMA_VERSION,
            "valid": valid,
            "teacher_commit": PINNED_TEACHER_COMMIT,
            "transition_count": int(transition_count),
            "scheduled_release_count": len(first_releases),
            "release_counts": {
                f"{kind}:{'success' if success else 'failure'}": count
                for (kind, success), count in sorted(release_counts.items())
            },
            "same_snapshot_rng_schedule_same_release_outcome": same_release,
            "same_snapshot_rng_schedule_same_transition": state_error == 0.0,
            "maximum_final_state_error": state_error,
            "maximum_final_rng_error": rng_error,
            "instrumentation_vs_canonical_driver_maximum_state_error": (
                uninstrumented_state_error
            ),
            "instrumentation_vs_canonical_driver_maximum_rng_error": (
                uninstrumented_rng_error
            ),
            "instrumentation_changes_teacher_dynamics": bool(
                uninstrumented_state_error != 0.0
                or uninstrumented_rng_error != 0.0
            ),
            "different_seed_changes_causal_rng_stream": different_rng,
            "teacher_rng_advanced_only_by_original_net_receive": True,
            "release_source": "exact causal replay of original NET_RECEIVE",
            "boundary_verification": {
                "maximum_point_state_error": max(
                    float(row["metadata"]["causal_release_boundary_verification"].get(
                        "maximum_point_state_error", 0.0
                    ))
                    for row in first
                ),
                "maximum_netcon_state_error": max(
                    float(row["metadata"]["causal_release_boundary_verification"].get(
                        "maximum_netcon_state_error", 0.0
                    ))
                    for row in first
                ),
                "all_transitions_valid": all(
                    bool(row["metadata"]["causal_release_boundary_verification"].get(
                        "valid", False
                    ))
                    for row in first
                ),
            },
            "future_state_used": False,
            "elapsed_seconds": elapsed,
            "seconds_per_transition": elapsed / len(first),
            "observed_hdf5_bytes_per_transition": byte_per_transition,
            "projected": {
                str(count): {
                    "generation_hours": elapsed / len(first) * count / 3600.0,
                    "hdf5_gib": byte_per_transition * count / (1024.0**3),
                }
                for count in (10_000, 30_000)
            },
            "pilot_store": str(pilot_path.relative_to(self.output_dir)),
            "pilot_store_sha256": sha256_file(pilot_path),
        }
        self.causal_release_pilot_report = report
        write_json(self.output_dir / "causal_release_pilot.json", report)
        if not valid:
            raise RuntimeError(
                "causal release pilot failed; do not generate dataset v1.1: "
                f"{report}"
            )
        return report

    def run_targeted_biological_pilot(
        self, config: Mapping[str, Any]
    ) -> Dict[str, Any]:
        """Coarse-to-boundary pilot for separable event-regime recipes."""

        if not self.causal_release_pilot_report.get("valid"):
            raise RuntimeError("causal release pilot must pass first")
        seeds = list(map(int, config["seeds"]))
        if len(seeds) < 2:
            raise ValueError("biological pilot requires at least two seeds")
        duration_ms = int(config.get("dendritic_duration_ms", 80))
        if duration_ms < 40:
            raise ValueError("dendritic pilot duration is too short")
        calibrator = DendriticProtocolCalibrator(
            self,
            output_dir=self.output_dir / "targeted_pilot",
            sample_interval_ms=0.025,
        )
        trials: List[Dict[str, Any]] = []
        nmda_cfg = config["nmda_sweep"]
        calcium_cfg = config["calcium_sweep"]
        bap_cfg = config.get("bap_sweep", {})
        assist_selection: Dict[str, Any] = {}
        base_trial_total = len(seeds) * (
            len(nmda_cfg["synapse_counts"])
            * len(nmda_cfg["burst_counts"])
            * len(nmda_cfg["event_windows_ms"])
            + len(calcium_cfg["synapse_counts"])
            * len(calcium_cfg["paired"])
            * len(calcium_cfg["event_windows_ms"])
            + len(config["somatic_current_factors"])
            + 2
            * len(bap_cfg.get("current_factors", ()))
            * len(bap_cfg.get("pulse_counts", ()))
        )
        pilot_progress = _ConsoleProgress("pilot biologico", base_trial_total)

        def run_candidate(candidate: DendriticCandidate) -> None:
            for seed in seeds:
                trial = calibrator.run_trial(candidate, seed, duration_ms)
                trial["event_kinds"] = sorted(
                    {str(row["kind"]) for row in trial["events"]}
                )
                trial["stimulus_scalar"] = float(
                    candidate.event_cost / candidate.event_window_ms
                    + (candidate.synapse_count if candidate.pair_with_somatic_spike else 0)
                )
                trial["duration_ms"] = duration_ms
                trial["branch_id"] = f"segment-{trial['event_probe_segment_id']}"
                trials.append(trial)
                pilot_progress.update(
                    len(trials),
                    detail=(
                        f"{candidate.candidate_id}; seed={seed}; "
                        f"eventi={','.join(trial['event_kinds']) or 'nessuno'}"
                    ),
                )

        nmda = config["nmda_sweep"]
        for count in map(int, nmda["synapse_counts"]):
            for bursts in map(int, nmda["burst_counts"]):
                for window in map(float, nmda["event_windows_ms"]):
                    run_candidate(
                        DendriticCandidate(
                            family="targeted_nmda",
                            target="tuft",
                            required_event_kinds=("nmda_spike",),
                            forbidden_event_kinds=(),
                            synapse_count=count,
                            burst_count=bursts,
                            burst_start_ms=int(nmda.get("burst_start_ms", 3)),
                            burst_interval_ms=int(nmda.get("burst_interval_ms", 1)),
                            pair_with_somatic_spike=False,
                            selection_mode="branch_cluster",
                            event_probe_mode="cluster_center",
                            event_probe_kinds=("nmda_spike", "nmda_plateau"),
                            event_window_ms=window,
                        )
                    )

        calcium = config["calcium_sweep"]
        for count in map(int, calcium["synapse_counts"]):
            for paired in map(bool, calcium["paired"]):
                for window in map(float, calcium["event_windows_ms"]):
                    run_candidate(
                        DendriticCandidate(
                            family="targeted_calcium",
                            target="hot_zone",
                            required_event_kinds=("calcium_spike",),
                            forbidden_event_kinds=(),
                            synapse_count=count,
                            burst_count=int(calcium.get("burst_count", 3)),
                            burst_start_ms=int(calcium.get("burst_start_ms", 3)),
                            burst_interval_ms=int(calcium.get("burst_interval_ms", 1)),
                            pair_with_somatic_spike=paired,
                            selection_mode="target_nearest",
                            event_probe_mode="target_representative",
                            event_window_ms=window,
                        )
                    )

        if self.calibrated_somatic_single_spike_current_na is None:
            self.calibrate_somatic_single_spike_current()
        base_current = float(self.calibrated_somatic_single_spike_current_na)
        somatic_duration = int(config.get("somatic_duration_ms", 20))
        for factor in map(float, config["somatic_current_factors"]):
            amplitude = base_current * factor
            candidate_id = f"targeted_somatic-current-factor{factor:.4f}"
            schedule = {
                3: (
                    InputAction(
                        "somatic_current",
                        0.05,
                        duration_ms=0.9,
                        amplitude_na=amplitude,
                    ),
                )
            }
            for seed in seeds:
                trajectory = ProtocolTrajectory(
                    trajectory_id=f"pilot-{candidate_id}-seed{seed}",
                    category="somatic_events",
                    protocol="targeted_somatic_bap",
                    protocol_id=candidate_id,
                    protocol_variant="current_sweep",
                    seed=seed,
                    duration_ms=somatic_duration,
                    split="event_boundary_test",
                    actions_by_step=schedule,
                    stimulus_onset_step=3,
                )
                times, traces = self._run_trajectory_prefix_in_memory(
                    trajectory, somatic_duration
                )
                events = self._extract_trajectory_events(trajectory, times, traces)
                trials.append(
                    {
                        "candidate_id": candidate_id,
                        "family": "targeted_somatic_bap",
                        "seed": seed,
                        "event_kinds": sorted(
                            {str(row["kind"]) for row in events}
                        ),
                        "events": events,
                        "stimulus_scalar": amplitude,
                        "duration_ms": somatic_duration,
                        "branch_id": "soma-to-apical-and-basal",
                        "input_schedule": {
                            str(step): [action.to_dict() for action in actions]
                            for step, actions in schedule.items()
                        },
                    }
                )
                pilot_progress.update(
                    len(trials),
                    detail=(
                        f"{candidate_id}; seed={seed}; "
                        f"eventi={','.join(trials[-1]['event_kinds']) or 'nessuno'}"
                    ),
                )

        if bap_cfg:
            if bap_cfg.get("assist_selection_policy") != (
                "strongest_robust_event_free_subthreshold_assist"
            ):
                raise ValueError("unsupported BAP assist selection policy")
            if set(bap_cfg.get("counterfactual_arms", ())) != {
                "soma_only",
                "assist_only",
                "combined",
            }:
                raise ValueError(
                    "BAP calibration requires soma-only, assist-only and combined arms"
                )
            if not bool(bap_cfg.get("require_subthreshold_assist", False)):
                raise ValueError("BAP assist must be counterfactually subthreshold")
            if not bool(
                bap_cfg.get("reject_distal_crossing_before_trunk", False)
            ):
                raise ValueError(
                    "BAP calibration must reject distal crossings before trunk"
                )
            bap_threshold_mv = float(bap_cfg.get("threshold_mv", -20.0))
            bap_maximum_delay_ms = float(
                bap_cfg.get("maximum_delay_ms", 3.0)
            )
            assist_selection = select_causal_bap_assist(
                trials,
                seeds,
                threshold_mv=bap_threshold_mv,
                subthreshold_margin_mv=float(
                    bap_cfg.get("assist_subthreshold_margin_mv", 2.0)
                ),
            )
            if not assist_selection["valid"]:
                raise RuntimeError(
                    "no robust subthreshold dendritic BAP assist candidate; "
                    f"selection={assist_selection}"
                )
            assist_id = str(assist_selection["selected_candidate_id"])
            assist_by_seed = {
                int(row["seed"]): row
                for row in trials
                if row.get("candidate_id") == assist_id
            }
            if set(assist_by_seed) != set(seeds):
                raise RuntimeError(
                    "BAP assist candidate is missing one or more biological-pilot seeds"
                )
            if self.calibrated_somatic_current_na is None:
                self.calibrate_somatic_spike_current()
            bap_base_current = float(self.calibrated_somatic_current_na)
            bap_duration = int(bap_cfg.get("duration_ms", duration_ms))
            for pulse_count in map(int, bap_cfg["pulse_counts"]):
                if pulse_count <= 0:
                    raise ValueError("BAP pulse count must be positive")
                for factor in map(float, bap_cfg["current_factors"]):
                    for seed in seeds:
                        source = assist_by_seed[seed]
                        assist_schedule = {
                            step: list(actions)
                            for step, actions in action_schedule_from_json(
                                source["input_schedule"]
                            ).items()
                        }
                        first_synaptic_step = min(assist_schedule)
                        current_schedule: Dict[int, List[InputAction]] = {}
                        if factor > 0.0:
                            first_current_step = max(0, first_synaptic_step - 1)
                            for pulse_index in range(pulse_count):
                                step = first_current_step + pulse_index
                                current_schedule.setdefault(step, []).append(
                                    InputAction(
                                        "somatic_current",
                                        0.05,
                                        duration_ms=0.9,
                                        amplitude_na=bap_base_current * factor,
                                    )
                                )

                        soma_only_schedule = {
                            int(step): tuple(
                                sorted(
                                    actions,
                                    key=lambda action: (
                                        float(action.offset_ms),
                                        action.kind,
                                        -1
                                        if action.synapse_id is None
                                        else int(action.synapse_id),
                                    ),
                                )
                            )
                            for step, actions in current_schedule.items()
                        }
                        probe_segment_id = int(source["event_probe_segment_id"])
                        soma_only_id = (
                            f"targeted_bap-soma-only-p{pulse_count}-"
                            f"factor{factor:.4f}"
                        )
                        soma_only_trajectory = ProtocolTrajectory(
                            trajectory_id=f"pilot-{soma_only_id}-seed{seed}",
                            category="dendritic_events",
                            protocol="targeted_bap",
                            protocol_id=soma_only_id,
                            protocol_variant="somatic_only_counterfactual",
                            seed=int(seed),
                            duration_ms=bap_duration,
                            split="event_boundary_test",
                            actions_by_step=soma_only_schedule,
                            stimulus_onset_step=min(soma_only_schedule, default=0),
                            metadata={
                                "event_probe_segment_id": probe_segment_id,
                                "event_probe_kinds": ["calcium_spike"],
                                "event_probe_region": str(
                                    source.get("event_probe_region", "hot_zone")
                                ),
                            },
                        )
                        soma_times, soma_traces = (
                            self._run_trajectory_prefix_in_memory(
                                soma_only_trajectory, bap_duration
                            )
                        )
                        soma_events = self._extract_trajectory_events(
                            soma_only_trajectory,
                            soma_times,
                            soma_traces,
                            preserve_raw_bap=True,
                        )
                        soma_assessment = assess_causal_bap(
                            soma_times,
                            soma_traces,
                            soma_events,
                            threshold_mv=bap_threshold_mv,
                            maximum_delay_ms=bap_maximum_delay_ms,
                        )
                        soma_raw_kinds = sorted(
                            {str(row["kind"]) for row in soma_events}
                        )
                        trials.append(
                            {
                                "candidate_id": soma_only_id,
                                "family": "targeted_bap_soma_only",
                                "seed": int(seed),
                                "event_kinds": _causal_event_kinds(
                                    soma_events, soma_assessment
                                ),
                                "raw_event_kinds": soma_raw_kinds,
                                "events": soma_events,
                                "bap_causal_assessment": soma_assessment,
                                "stimulus_scalar": float(factor * pulse_count),
                                "duration_ms": bap_duration,
                                "branch_id": "soma-to-trunk",
                                "event_probe_segment_id": probe_segment_id,
                                "event_probe_region": str(
                                    source.get("event_probe_region", "hot_zone")
                                ),
                                "event_probe_kinds": ["calcium_spike"],
                                "selected_synapse_ids": [],
                                "input_schedule": {
                                    str(step): [
                                        action.to_dict() for action in actions
                                    ]
                                    for step, actions in soma_only_schedule.items()
                                },
                            }
                        )
                        pilot_progress.update(
                            len(trials),
                            detail=(
                                f"{soma_only_id}; seed={seed}; "
                                "bAP causale="
                                f"{'si' if soma_assessment['valid'] else 'no'}"
                            ),
                        )

                        combined_schedule = {
                            step: list(actions)
                            for step, actions in assist_schedule.items()
                        }
                        for step, actions in current_schedule.items():
                            combined_schedule.setdefault(step, []).extend(actions)
                        ordered_schedule = {
                            int(step): tuple(
                                sorted(
                                    actions,
                                    key=lambda action: (
                                        float(action.offset_ms),
                                        action.kind,
                                        -1
                                        if action.synapse_id is None
                                        else int(action.synapse_id),
                                    ),
                                )
                            )
                            for step, actions in combined_schedule.items()
                        }
                        candidate_id = (
                            f"targeted_bap-assisted-p{pulse_count}-"
                            f"factor{factor:.4f}"
                        )
                        trajectory = ProtocolTrajectory(
                            trajectory_id=f"pilot-{candidate_id}-seed{seed}",
                            category="dendritic_events",
                            protocol="targeted_bap",
                            protocol_id=candidate_id,
                            protocol_variant=(
                                "somatic_spikes_with_counterfactual_hot_zone_assist"
                            ),
                            seed=int(seed),
                            duration_ms=bap_duration,
                            split="event_boundary_test",
                            actions_by_step=ordered_schedule,
                            stimulus_onset_step=min(ordered_schedule),
                            metadata={
                                "event_probe_segment_id": probe_segment_id,
                                "event_probe_kinds": ["calcium_spike"],
                                "event_probe_region": str(
                                    source.get("event_probe_region", "hot_zone")
                                ),
                            },
                        )
                        times, traces = self._run_trajectory_prefix_in_memory(
                            trajectory, bap_duration
                        )
                        events = self._extract_trajectory_events(
                            trajectory,
                            times,
                            traces,
                            preserve_raw_bap=True,
                        )
                        assessment = assess_causal_bap(
                            times,
                            traces,
                            events,
                            assist_only_events=source["events"],
                            assist_only_peak_voltage_mv=float(
                                source["event_probe_peak_voltage_mv"]
                            ),
                            require_subthreshold_assist=True,
                            threshold_mv=bap_threshold_mv,
                            maximum_delay_ms=bap_maximum_delay_ms,
                        )
                        raw_kinds = sorted(
                            {str(row["kind"]) for row in events}
                        )
                        trials.append(
                            {
                                "candidate_id": candidate_id,
                                "family": "targeted_bap",
                                "seed": int(seed),
                                "event_kinds": _causal_event_kinds(
                                    events, assessment
                                ),
                                "raw_event_kinds": raw_kinds,
                                "events": events,
                                "bap_causal_assessment": assessment,
                                "counterfactual_soma_only_candidate_id": (
                                    soma_only_id
                                ),
                                "counterfactual_assist_only_candidate_id": (
                                    assist_id
                                ),
                                "stimulus_scalar": float(factor * pulse_count),
                                "duration_ms": bap_duration,
                                "branch_id": (
                                    "soma-assisted-" + str(source["branch_id"])
                                ),
                                "event_probe_segment_id": probe_segment_id,
                                "event_probe_region": str(
                                    source.get("event_probe_region", "hot_zone")
                                ),
                                "event_probe_kinds": ["calcium_spike"],
                                "selected_synapse_ids": list(
                                    source["selected_synapse_ids"]
                                ),
                                "input_schedule": {
                                    str(step): [
                                        action.to_dict() for action in actions
                                    ]
                                    for step, actions in ordered_schedule.items()
                                },
                            }
                        )
                        pilot_progress.update(
                            len(trials),
                            detail=(
                                f"{candidate_id}; seed={seed}; "
                                "assist subthreshold="
                                f"{'si' if assessment['assist_only_subthreshold'] else 'no'}; "
                                "bAP causale="
                                f"{'si' if assessment['valid'] else 'no'}"
                            ),
                        )

        pilot_progress.update(
            min(len(trials), base_trial_total),
            detail="sweep primario completato",
            force=True,
        )

        # A second tuft branch is pilot-tested but explicitly excluded from
        # train, so the held-out-branch split is a real morphology holdout.
        primary_nmda = [
            row
            for row in trials
            if row.get("family") == "targeted_nmda"
            and int(row.get("synapse_count", 0))
            <= len(self.alternate_tuft_selection["synapse_ids"])
        ]
        if primary_nmda:
            source_id = max(
                primary_nmda, key=lambda row: float(row["stimulus_scalar"])
            )["candidate_id"]
            alternate_ids = list(
                map(int, self.alternate_tuft_selection["synapse_ids"])
            )
            for source in [
                row for row in primary_nmda if row["candidate_id"] == source_id
            ]:
                old_ids = list(map(int, source["selected_synapse_ids"]))
                replacement = {
                    old: new
                    for old, new in zip(old_ids, alternate_ids[: len(old_ids)])
                }
                schedule = action_schedule_from_json(source["input_schedule"])
                replaced = {
                    step: tuple(
                        InputAction(
                            action.kind,
                            action.offset_ms,
                            synapse_id=(
                                replacement[int(action.synapse_id)]
                                if action.kind == "synaptic_event"
                                else None
                            ),
                            weight_multiplier=action.weight_multiplier,
                            duration_ms=action.duration_ms,
                            amplitude_na=action.amplitude_na,
                            metadata=action.metadata,
                        )
                        for action in actions
                    )
                    for step, actions in schedule.items()
                }
                candidate_id = f"{source_id}-held-out-tuft-branch"
                trajectory = ProtocolTrajectory(
                    trajectory_id=f"pilot-{candidate_id}-seed{source['seed']}",
                    category="dendritic_events",
                    protocol="targeted_nmda_heldout_branch",
                    protocol_id=candidate_id,
                    protocol_variant="held_out_morphological_branch",
                    seed=int(source["seed"]),
                    duration_ms=duration_ms,
                    split="held_out_branch_test",
                    actions_by_step=replaced,
                    stimulus_onset_step=min(replaced),
                    metadata={
                        "event_probe_segment_id": int(
                            self.alternate_tuft_selection["center_segment_id"]
                        ),
                        "event_probe_region": "held_out_tuft_stimulus_cluster",
                        "event_probe_kinds": ["nmda_spike", "nmda_plateau"],
                    },
                )
                times, traces = self._run_trajectory_prefix_in_memory(
                    trajectory, duration_ms
                )
                events = self._extract_trajectory_events(trajectory, times, traces)
                trials.append(
                    {
                        "candidate_id": candidate_id,
                        "family": "targeted_nmda_heldout_branch",
                        "seed": int(source["seed"]),
                        "event_kinds": sorted(
                            {str(row["kind"]) for row in events}
                        ),
                        "events": events,
                        "stimulus_scalar": float(source["stimulus_scalar"]),
                        "duration_ms": duration_ms,
                        "branch_id": (
                            "segment-"
                            f"{self.alternate_tuft_selection['center_segment_id']}"
                        ),
                        "event_probe_segment_id": int(
                            self.alternate_tuft_selection["center_segment_id"]
                        ),
                        "event_probe_region": "held_out_tuft_stimulus_cluster",
                        "event_probe_kinds": ["nmda_spike", "nmda_plateau"],
                        "selected_synapse_ids": list(replacement.values()),
                        "input_schedule": {
                            str(step): [action.to_dict() for action in actions]
                            for step, actions in replaced.items()
                        },
                        "train_eligible": False,
                    }
                )

        brackets = select_adaptive_recipe_brackets(
            [row for row in trials if row.get("train_eligible", True)],
            required_seed_count=len(seeds),
        )
        by_candidate: Dict[str, List[Mapping[str, Any]]] = {}
        for trial in trials:
            by_candidate.setdefault(str(trial["candidate_id"]), []).append(trial)
        roles: Dict[str, Dict[str, Any]] = {}
        for event_class, selection in brackets["selections"].items():
            for role in ("positive", "negative"):
                candidate_id = selection[f"{role}_candidate_id"]
                record = roles.setdefault(
                    candidate_id,
                    {
                        "positive_for": set(),
                        "hard_negative_for": set(),
                        "distances": {},
                    },
                )
                key = "positive_for" if role == "positive" else "hard_negative_for"
                record[key].add(event_class)
                record["distances"][event_class] = selection[
                    f"{role}_boundary_distance"
                ]
        recipes = []
        for candidate_id, role in sorted(roles.items()):
            reference = by_candidate[candidate_id][0]
            distances = dict(role["distances"])
            recipes.append(
                TargetedRecipe(
                    recipe_id=candidate_id,
                    family=str(reference["family"]),
                    protocol_variant=(
                        "adaptive_positive"
                        if role["positive_for"]
                        else "adaptive_hard_negative"
                    ),
                    duration_ms=int(reference["duration_ms"]),
                    actions_by_step=action_schedule_from_json(
                        reference["input_schedule"]
                    ),
                    positive_for=tuple(sorted(role["positive_for"])),
                    hard_negative_for=tuple(
                        sorted(role["hard_negative_for"] - role["positive_for"])
                    ),
                    branch_id=str(reference["branch_id"]),
                    boundary_distance=float(
                        min(distances.values(), key=abs)
                    ),
                    metadata={
                        "boundary_distance_by_class": distances,
                        "pilot_seed_count": len(seeds),
                        "event_probe_segment_id": reference.get(
                            "event_probe_segment_id"
                        ),
                        "event_probe_region": reference.get(
                            "event_probe_region"
                        ),
                        "event_probe_kinds": list(
                            reference.get("event_probe_kinds", ())
                        ),
                        "selected_synapse_ids": list(
                            reference.get("selected_synapse_ids", ())
                        ),
                        "bap_causal_assessments_by_seed": {
                            str(row["seed"]): row["bap_causal_assessment"]
                            for row in by_candidate[candidate_id]
                            if row.get("bap_causal_assessment") is not None
                        },
                        "counterfactual_soma_only_candidate_id": reference.get(
                            "counterfactual_soma_only_candidate_id"
                        ),
                        "counterfactual_assist_only_candidate_id": reference.get(
                            "counterfactual_assist_only_candidate_id"
                        ),
                    },
                )
            )

        heldout_trials = [
            row for row in trials if not row.get("train_eligible", True)
        ]
        if heldout_trials:
            heldout_id = str(heldout_trials[0]["candidate_id"])
            heldout_group = [
                row for row in heldout_trials if row["candidate_id"] == heldout_id
            ]
            event_sets = [set(row["event_kinds"]) for row in heldout_group]
            positive_for = tuple(
                kind
                for kind in (
                    "axonal_spike",
                    "somatic_spike",
                    "backpropagating_ap",
                    "calcium_spike",
                    "nmda_spike",
                    "nmda_plateau",
                )
                if event_sets and all(kind in observed for observed in event_sets)
            )
            hard_negative_for = tuple(
                kind
                for kind in (
                    "axonal_spike",
                    "somatic_spike",
                    "backpropagating_ap",
                    "calcium_spike",
                    "nmda_spike",
                    "nmda_plateau",
                )
                if event_sets and all(kind not in observed for observed in event_sets)
            )
            reference = heldout_group[0]
            recipes.append(
                TargetedRecipe(
                    recipe_id=heldout_id,
                    family=str(reference["family"]),
                    protocol_variant="held_out_morphological_branch",
                    duration_ms=int(reference["duration_ms"]),
                    actions_by_step=action_schedule_from_json(
                        reference["input_schedule"]
                    ),
                    positive_for=positive_for,
                    hard_negative_for=hard_negative_for,
                    branch_id=str(reference["branch_id"]),
                    metadata={
                        "train_eligible": False,
                        "pilot_seed_count": len(heldout_group),
                        "event_probe_segment_id": reference.get(
                            "event_probe_segment_id"
                        ),
                        "event_probe_region": reference.get(
                            "event_probe_region", "held_out_tuft_stimulus_cluster"
                        ),
                        "event_probe_kinds": list(
                            reference.get(
                                "event_probe_kinds",
                                ("nmda_spike", "nmda_plateau"),
                            )
                        ),
                        "selected_synapse_ids": list(
                            reference.get("selected_synapse_ids", ())
                        ),
                    },
                )
            )

        # Pilot recovery probes by replaying the selected stimulus after a
        # configurable interval. Only empirically recovered variants enter the
        # recipe catalog.
        recovery_recipes: List[TargetedRecipe] = []
        recovery_trials: List[Dict[str, Any]] = []
        recovery_duration = int(config.get("recovery_duration_ms", 100))
        recovery_candidates = [row for row in recipes if row.positive_for]
        recovery_delays = list(
            map(int, config.get("recovery_probe_delays_ms", (20, 50, 80)))
        )
        recovery_progress = _ConsoleProgress(
            "pilot recovery", len(recovery_candidates) * len(recovery_delays)
        )
        recovery_trial_index = 0
        for recipe in recovery_candidates:
            first_step = min(recipe.actions_by_step, default=0)
            final_step = max(recipe.actions_by_step, default=first_step)
            for delay in recovery_delays:
                recovery_trial_index += 1
                shift = int(delay)
                if final_step + shift >= recovery_duration:
                    recovery_progress.update(
                        recovery_trial_index,
                        detail=f"{recipe.recipe_id}; delay {delay} fuori finestra",
                    )
                    continue
                schedule = {step: tuple(actions) for step, actions in recipe.actions_by_step.items()}
                for step, actions in recipe.actions_by_step.items():
                    schedule[int(step) + shift] = tuple(actions)
                candidate_id = f"{recipe.recipe_id}-recovery-{delay}ms"
                trajectory_metadata = {
                    key: value
                    for key, value in dict(recipe.metadata).items()
                    if value is not None
                }
                selected_synapses = sorted(
                    {
                        int(action.synapse_id)
                        for actions in recipe.actions_by_step.values()
                        for action in actions
                        if action.kind == "synaptic_event"
                    }
                )

                # Run only the first stimulus up to the boundary immediately
                # preceding the repeated probe.  Pv in these MOD mechanisms is
                # lazily recovered inside NET_RECEIVE, so reading it after the
                # second event would incorrectly see the newly depleted value.
                pre_probe_step = first_step + shift
                pre_probe_trajectory = ProtocolTrajectory(
                    trajectory_id=f"pilot-{candidate_id}-pre-probe-seed{seeds[0]}",
                    category=(
                        "somatic_events"
                        if recipe.family == "targeted_somatic_bap"
                        else "dendritic_events"
                    ),
                    protocol=recipe.family,
                    protocol_id=f"{candidate_id}-pre-probe",
                    protocol_variant="recovery_pre_probe_state",
                    seed=seeds[0],
                    duration_ms=pre_probe_step,
                    split="recovery_test",
                    actions_by_step=recipe.actions_by_step,
                    stimulus_onset_step=first_step,
                    metadata=trajectory_metadata,
                )
                self._run_trajectory_prefix_in_memory(
                    pre_probe_trajectory, pre_probe_step
                )
                pre_probe_boundary_time = float(self.h.t)
                first_repeated_event: Dict[int, Tuple[int, float]] = {}
                for step, actions in recipe.actions_by_step.items():
                    repeated_step = int(step) + shift
                    for action in actions:
                        if action.kind != "synaptic_event":
                            continue
                        synapse_id = int(action.synapse_id)
                        candidate_time = (repeated_step, float(action.offset_ms))
                        if (
                            synapse_id not in first_repeated_event
                            or candidate_time < first_repeated_event[synapse_id]
                        ):
                            first_repeated_event[synapse_id] = candidate_time
                available_pv = []
                for synapse_id in selected_synapses:
                    record = self.audit.synapse_records[synapse_id]
                    class_name = str(record["class_name"])
                    pv_index = 3 if class_name == "ProbAMPANMDA2" else 1
                    tsyn_index = 6 if class_name == "ProbAMPANMDA2" else 4
                    repeated_step, offset_ms = first_repeated_event[synapse_id]
                    event_time = (
                        pre_probe_boundary_time
                        + float(repeated_step - pre_probe_step)
                        + offset_ms
                    )
                    available_pv.append(
                        self._available_pv_at_event(
                            float(record["netcon"].weight[pv_index]),
                            event_time
                            - float(record["netcon"].weight[tsyn_index]),
                            float(record["point_process"].Dep),
                        )
                    )
                minimum_available_pv = min(available_pv, default=1.0)
                synapse_recovered = minimum_available_pv >= float(
                    config.get("recovery_min_pv", 0.75)
                )

                trajectory = ProtocolTrajectory(
                    trajectory_id=f"pilot-{candidate_id}-seed{seeds[0]}",
                    category=(
                        "somatic_events"
                        if recipe.family == "targeted_somatic_bap"
                        else "dendritic_events"
                    ),
                    protocol=recipe.family,
                    protocol_id=candidate_id,
                    protocol_variant="recovery_probe",
                    seed=seeds[0],
                    duration_ms=recovery_duration,
                    split="recovery_test",
                    actions_by_step=schedule,
                    stimulus_onset_step=first_step,
                    metadata=trajectory_metadata,
                )
                times, traces = self._run_trajectory_prefix_in_memory(
                    trajectory, recovery_duration
                )
                events = self._extract_trajectory_events(trajectory, times, traces)
                observed = {str(row["kind"]) for row in events}
                event_counts = Counter(str(row["kind"]) for row in events)
                uncensored = all(not bool(row["right_censored"]) for row in events)
                voltage_recovered = all(
                    abs(float(values[-1]) - float(values[0]))
                    <= float(config.get("recovery_voltage_tolerance_mv", 5.0))
                    for label, values in traces.items()
                    if label in self.audit.representatives
                )
                calcium_trace = self.np.asarray(
                    traces.get("cai_event_probe_mM", ()), dtype=float
                )
                if calcium_trace.size:
                    calcium_excursion = float(
                        self.np.max(self.np.abs(calcium_trace - calcium_trace[0]))
                    )
                    calcium_tolerance = max(
                        1.0e-8,
                        float(config.get("recovery_calcium_fraction", 0.10))
                        * calcium_excursion,
                    )
                    calcium_recovered = (
                        abs(float(calcium_trace[-1] - calcium_trace[0]))
                        <= calcium_tolerance
                    )
                else:
                    calcium_recovered = False
                repeated_response = all(
                    event_counts.get(kind, 0) >= 2
                    for kind in recipe.positive_for
                )
                criteria = {
                    "required_events_observed": set(recipe.positive_for).issubset(
                        observed
                    ),
                    "events_uncensored": uncensored,
                    "voltage_recovered": voltage_recovered,
                    "calcium_recovered": calcium_recovered,
                    "synapse_recovered_before_second_probe": synapse_recovered,
                    "repeated_probe_response": repeated_response,
                }
                recovery_valid = all(criteria.values())
                recovery_trials.append(
                    {
                        "candidate_id": candidate_id,
                        "source_recipe_id": recipe.recipe_id,
                        "delay_ms": delay,
                        "positive_for": list(recipe.positive_for),
                        "event_counts": dict(sorted(event_counts.items())),
                        "minimum_available_pv_before_second_probe": (
                            minimum_available_pv
                        ),
                        "criteria": criteria,
                        "valid": recovery_valid,
                    }
                )
                if not recovery_valid:
                    failed = [name for name, passed in criteria.items() if not passed]
                    recovery_progress.update(
                        recovery_trial_index,
                        detail=f"{candidate_id}; fail={','.join(failed)}",
                    )
                    continue
                if not (
                    set(recipe.positive_for).issubset(observed)
                    and uncensored
                    and voltage_recovered
                    and calcium_recovered
                    and synapse_recovered
                    and repeated_response
                ):
                    raise AssertionError("recovery criteria aggregation is inconsistent")
                recovery_recipes.append(
                    TargetedRecipe(
                        recipe_id=candidate_id,
                        family=recipe.family,
                        protocol_variant="pilot_validated_recovery_probe",
                        duration_ms=recovery_duration,
                        actions_by_step=schedule,
                        positive_for=recipe.positive_for,
                        hard_negative_for=recipe.hard_negative_for,
                        branch_id=recipe.branch_id,
                        boundary_distance=recipe.boundary_distance,
                        recovery_probe_delay_ms=float(delay),
                        metadata={
                            **dict(recipe.metadata),
                            "train_eligible": False,
                            "recovery_probe": True,
                            "pilot_validated": True,
                            "voltage_recovered": True,
                            "calcium_recovered": True,
                            "synapse_recovered": True,
                            "repeated_probe_response": True,
                            "minimum_available_pv_before_second_probe": (
                                minimum_available_pv
                            ),
                            "events_uncensored": True,
                        },
                    )
                )
                recovery_progress.update(
                    recovery_trial_index,
                    detail=f"{candidate_id}; valido",
                )
        if recovery_candidates:
            recovery_progress.update(
                recovery_trial_index,
                detail=f"ricette valide={len(recovery_recipes)}",
                force=True,
            )
        recipes.extend(recovery_recipes)
        self.targeted_recipe_catalog = recipes
        bap_counterfactual_trials = [
            row
            for row in trials
            if row.get("family") in {"targeted_bap", "targeted_bap_soma_only"}
        ]
        bap_candidate_groups: Dict[str, List[Mapping[str, Any]]] = {}
        for row in bap_counterfactual_trials:
            bap_candidate_groups.setdefault(
                str(row["candidate_id"]), []
            ).append(row)
        bap_causal_summary = {
            "contract": "counterfactual_soma_origin_and_ordered_trunk_arrival",
            "threshold_mv": float(bap_cfg.get("threshold_mv", -20.0))
            if bap_cfg
            else -20.0,
            "maximum_delay_ms": float(
                bap_cfg.get("maximum_delay_ms", 3.0)
            )
            if bap_cfg
            else 3.0,
            "soma_only_arm": True,
            "assist_only_arm": True,
            "combined_arm": True,
            "assist_selection": assist_selection,
            "raw_bap_trial_count": sum(
                "backpropagating_ap" in row.get("raw_event_kinds", ())
                for row in bap_counterfactual_trials
            ),
            "causal_bap_trial_count": sum(
                bool(row.get("bap_causal_assessment", {}).get("valid"))
                for row in bap_counterfactual_trials
            ),
            "robust_positive_candidate_ids": sorted(
                candidate_id
                for candidate_id, rows in bap_candidate_groups.items()
                if len({int(row["seed"]) for row in rows}) == len(seeds)
                and all(
                    bool(row.get("bap_causal_assessment", {}).get("valid"))
                    for row in rows
                )
            ),
            "combined_assist_subthreshold_seed_count": sum(
                bool(
                    row.get("bap_causal_assessment", {}).get(
                        "assist_only_subthreshold"
                    )
                )
                for row in bap_counterfactual_trials
                if row.get("family") == "targeted_bap"
            ),
        }
        report = {
            "schema_version": TARGETED_DATASET_SCHEMA_VERSION,
            "valid": brackets["valid"]
            and bool(recipes)
            and bool(heldout_trials)
            and bool(recovery_recipes),
            "trial_count": len(trials),
            "candidate_count": len(by_candidate),
            "seeds": seeds,
            "nmda_label_hierarchy": {
                "nmda_spike_minimum_duration_ms": 1.0,
                "nmda_plateau_minimum_duration_ms": 10.0,
                "plateau_is_nmda_spike_subset": True,
            },
            "bap_sweep": dict(bap_cfg),
            "bap_causal_validation": bap_causal_summary,
            "adaptive_brackets": brackets,
            "recipe_count": len(recipes),
            "heldout_branch_recipe_count": sum(
                not bool(row.metadata.get("train_eligible", True))
                and not bool(row.metadata.get("recovery_probe", False))
                for row in recipes
            ),
            "recovery_recipe_count": len(recovery_recipes),
            "recovery_trials": recovery_trials,
            "recipes": [
                {
                    "recipe_id": row.recipe_id,
                    "family": row.family,
                    "positive_for": list(row.positive_for),
                    "hard_negative_for": list(row.hard_negative_for),
                    "branch_id": row.branch_id,
                    "duration_ms": row.duration_ms,
                    "boundary_distance": row.boundary_distance,
                }
                for row in recipes
            ],
        }
        pilot_dir = self.output_dir / "targeted_pilot"
        pilot_dir.mkdir(parents=True, exist_ok=True)
        self.pd.DataFrame(self._parquet_safe_rows(trials)).to_parquet(
            pilot_dir / "candidate_trials.parquet", index=False
        )
        write_json(pilot_dir / "pilot_report.json", report)
        write_json(
            pilot_dir / "recipe_catalog.json",
            {
                "schema_version": TARGETED_DATASET_SCHEMA_VERSION,
                "recipes": [
                    {
                        "recipe_id": row.recipe_id,
                        "family": row.family,
                        "protocol_variant": row.protocol_variant,
                        "duration_ms": row.duration_ms,
                        "actions_by_step": {
                            str(step): [action.to_dict() for action in actions]
                            for step, actions in row.actions_by_step.items()
                        },
                        "positive_for": list(row.positive_for),
                        "hard_negative_for": list(row.hard_negative_for),
                        "branch_id": row.branch_id,
                        "snapshot_id": row.snapshot_id,
                        "boundary_distance": row.boundary_distance,
                        "metadata": dict(row.metadata),
                    }
                    for row in recipes
                ],
            },
        )
        if not report["valid"]:
            raise RuntimeError(
                "targeted biological pilot is not valid; "
                f"adaptive_blockers={brackets['blockers']}; "
                f"valid_recovery_recipes={len(recovery_recipes)}; "
                f"inspect {pilot_dir / 'pilot_report.json'}"
            )
        return report

    def generate_dataset(
        self, protocols: Optional[Sequence[ProtocolTrajectory]] = None
    ) -> Dict[str, Any]:
        """Generate only after the release and targeted-plan gates pass."""

        if not self.causal_release_pilot_report.get("valid"):
            raise RuntimeError("run_causal_release_pilot() must pass first")
        protocols = list(protocols or ())
        if not protocols:
            raise ValueError("an explicit targeted protocol plan is required")
        expected_hash = self.targeted_preflight_report.get("protocol_plan_sha256")
        observed_hash = self._protocol_plan_sha256(protocols)
        if not self.targeted_preflight_report.get("valid") or expected_hash != observed_hash:
            raise RuntimeError(
                "targeted preflight must pass for the exact protocol plan before generation"
            )
        required_snapshot_ids = {
            str(row.metadata.get("snapshot_id", row.snapshot_source))
            for row in protocols
        }
        missing_snapshots = required_snapshot_ids - set(self.snapshot_bank)
        if missing_snapshots:
            raise RuntimeError(
                f"snapshot bank is incomplete: {sorted(missing_snapshots)}"
            )
        self.release_rows = []
        self._bind_protocol_registry(protocols)
        self._collect_release_rows = True
        try:
            manifest = DiagnosticDatasetSession.generate_dataset(self, protocols)
        finally:
            self._collect_release_rows = False
        manifest.update(
            {
                "schema_version": TARGETED_DATASET_SCHEMA_VERSION,
                "compatible_base_schema_version": "1.0.1",
                "canonical_state_layout_sha256": self.state_schema[
                    "canonical_state_layout_sha256"
                ],
                "release_schema": "release_schema.json",
                "input_views": ["U_scheduled", "U_rng", "U_realized"],
                "release_outcomes": "release_outcomes.parquet",
                "causal_release_pilot": "causal_release_pilot.json",
                "targeted_preflight": "targeted_preflight_report.json",
                "snapshot_bank": "snapshot_bank.json",
            }
        )
        self.dataset_manifest = manifest
        write_json(self.output_dir / "dataset_manifest.json", manifest)
        self.pd.DataFrame(self._parquet_safe_rows(self.release_rows)).to_parquet(
            self.output_dir / "release_outcomes.parquet", index=False
        )
        table_report = self._write_targeted_tables(protocols)
        manifest["indices"] = {
            "protocols": "protocols.parquet",
            "episodes": "episodes.parquet",
            "transitions": "transition_index.parquet",
            "events": "events.parquet",
            "release_outcomes": "release_outcomes.parquet",
            "branching_pairs": "branching_pairs.parquet",
            "splits": "splits.json",
        }
        manifest["dataset_card"] = "dataset_card.json"
        manifest["storage_report"] = "storage_report.json"
        manifest["table_report"] = table_report
        write_json(self.output_dir / "dataset_manifest.json", manifest)
        return manifest

    def _bind_protocol_registry(
        self, protocols: Sequence[ProtocolTrajectory]
    ) -> None:
        self.protocol_registry = {}
        self.protocol_rows = []
        for row in protocols:
            metadata = dict(row.metadata)
            selected_synapses = sorted(
                {
                    int(action.synapse_id)
                    for actions in row.actions_by_step.values()
                    for action in actions
                    if action.kind == "synaptic_event"
                }
            )
            contract = {
                "trajectory_id": row.trajectory_id,
                "category": row.category,
                "protocol": row.protocol,
                "protocol_id": row.protocol_id or row.protocol,
                "protocol_variant": row.protocol_variant,
                "seed": int(row.seed),
                "duration_ms": int(row.duration_ms),
                "split": row.split,
                "stimulus_onset_step": int(row.stimulus_onset_step),
                "required_event_kinds": list(row.required_event_kinds),
                "negative_control": bool(row.negative_control),
                "event_probe_label": row.event_probe_label,
                "event_probe_segment_id": metadata.get(
                    "event_probe_segment_id"
                ),
                "selected_synapse_ids": selected_synapses,
                "snapshot_source": row.snapshot_source,
                **metadata,
            }
            self.protocol_registry[row.trajectory_id] = contract
            self.protocol_rows.append(contract)

    def _write_targeted_tables(
        self, protocols: Sequence[ProtocolTrajectory]
    ) -> Dict[str, Any]:
        try:
            import h5py
        except ImportError as error:
            raise RuntimeError("targeted tables require h5py") from error

        transition_rows = []
        event_rows = []
        actual_events: Dict[str, set] = {}
        with h5py.File(self.transition_path, "r") as handle:
            count = int(handle.attrs["transition_count"])
            for index in range(count):
                trajectory_id = self._decode(
                    handle["metadata/trajectory_id"][index]
                )
                contract = self.protocol_registry[trajectory_id]
                labels = json.loads(handle["events/labels_json"][index])
                event_flags = sorted({str(row["kind"]) for row in labels})
                actual_events.setdefault(trajectory_id, set()).update(event_flags)
                transition_rows.append(
                    {
                        "transition_id": index,
                        "trajectory_id": trajectory_id,
                        "episode_id": contract.get("episode_id", trajectory_id),
                        "protocol_id": contract["protocol_id"],
                        "protocol_variant": contract["protocol_variant"],
                        "split": contract["split"],
                        "seed": contract["seed"],
                        "branch_id": contract.get("branch_id", "unknown"),
                        "snapshot_id": contract.get(
                            "snapshot_id", contract["snapshot_source"]
                        ),
                        "step_index": int(handle["metadata/step_index"][index]),
                        "absolute_time_ms": float(
                            handle["metadata/start_time_ms"][index]
                        ),
                        "event_flags": event_flags,
                        "release_count": len(
                            json.loads(
                                handle["release_outcomes/records_json"][index]
                            )
                        ),
                        "microtrace_mode": self._decode(
                            handle["metadata/microtrace_mode"][index]
                        ),
                    }
                )
                for event_index, event in enumerate(labels):
                    event_rows.append(
                        {
                            "trajectory_id": trajectory_id,
                            "episode_id": contract.get("episode_id", trajectory_id),
                            "transition_id": index,
                            "trajectory_event_id": event.get(
                                "trajectory_event_id", event_index
                            ),
                            "split": contract["split"],
                            "seed": contract["seed"],
                            **event,
                        }
                    )

        episodes = []
        for contract in self.protocol_rows:
            observed = sorted(actual_events.get(contract["trajectory_id"], set()))
            intended_negatives = list(contract.get("hard_negative_for", ()))
            valid_negatives = [
                kind for kind in intended_negatives if kind not in observed
            ]
            episodes.append(
                {
                    **contract,
                    "episode_id": contract.get(
                        "episode_id", contract["trajectory_id"]
                    ),
                    "snapshot_id": contract.get(
                        "snapshot_id", contract["snapshot_source"]
                    ),
                    "branch_id": contract.get("branch_id", "unknown"),
                    "event_labels": observed,
                    "hard_negative_for": valid_negatives,
                    "hard_negative_failures": sorted(
                        set(intended_negatives) - set(valid_negatives)
                    ),
                }
            )
        support = summarize_independent_support(episodes)
        effective_positive_targets = self.targeted_preflight_report.get(
            "positive_support_targets"
        )
        effective_negative_targets = self.targeted_preflight_report.get(
            "hard_negative_support_targets"
        )
        support_validation = validate_minimum_support(
            support,
            positive_targets=effective_positive_targets,
            hard_negative_targets=effective_negative_targets,
        )
        release_success = Counter(
            (
                str(row["synapse_type"]),
                "success" if bool(row["release_success"]) else "failure",
            )
            for row in self.release_rows
        )
        branching_contracts: Dict[str, List[Mapping[str, Any]]] = {}
        for row in episodes:
            pair_id = row.get("branch_pair_id")
            if pair_id:
                branching_contracts.setdefault(str(pair_id), []).append(row)
        branching_rows = []
        release_pair_rows = []
        transition_ids_by_trajectory: Dict[str, List[int]] = {}
        for row in transition_rows:
            transition_ids_by_trajectory.setdefault(
                row["trajectory_id"], []
            ).append(int(row["transition_id"]))
        with h5py.File(self.transition_path, "r") as handle:
            for pair_id, pair in sorted(branching_contracts.items()):
                if len(pair) != 2:
                    raise RuntimeError(
                        f"branch pair {pair_id} must contain exactly two futures"
                    )
                left, right = pair
                left_ids = sorted(
                    transition_ids_by_trajectory[left["trajectory_id"]]
                )
                right_ids = sorted(
                    transition_ids_by_trajectory[right["trajectory_id"]]
                )
                horizon_count = min(len(left_ids), len(right_ids))
                distances = []
                for horizon in range(horizon_count):
                    left_v = handle["states/voltage/t_plus_1"][
                        left_ids[horizon], :
                    ]
                    right_v = handle["states/voltage/t_plus_1"][
                        right_ids[horizon], :
                    ]
                    distances.append(
                        float(self.np.max(self.np.abs(left_v - right_v)))
                    )
                initial_errors = []
                for category in self.state_variables:
                    initial_errors.append(
                        float(
                            self.np.max(
                                self.np.abs(
                                    handle[f"states/{category}/t"][left_ids[0], :]
                                    - handle[f"states/{category}/t"][right_ids[0], :]
                                )
                            )
                        )
                    )
                initial_errors.append(
                    float(
                        self.np.max(
                            self.np.abs(
                                handle["rng_state/t"][left_ids[0], :]
                                - handle["rng_state/t"][right_ids[0], :]
                            )
                        )
                    )
                )
                first_divergence = next(
                    (
                        index + 1
                        for index, value in enumerate(distances)
                        if value > 1.0e-3
                    ),
                    None,
                )
                regional_max = {}
                for label, segment_id in self.audit.representatives.items():
                    regional_max[label] = max(
                        (
                            abs(
                                float(
                                    handle["states/voltage/t_plus_1"][
                                        left_ids[horizon], int(segment_id)
                                    ]
                                )
                                - float(
                                    handle["states/voltage/t_plus_1"][
                                        right_ids[horizon], int(segment_id)
                                    ]
                                )
                            )
                            for horizon in range(horizon_count)
                        ),
                        default=0.0,
                    )
                release_left = [
                    json.loads(handle["release_outcomes/records_json"][index])
                    for index in left_ids
                ]
                release_right = [
                    json.loads(handle["release_outcomes/records_json"][index])
                    for index in right_ids
                ]
                branching_rows.append(
                    {
                        "branch_pair_id": pair_id,
                        "branching_distance": left.get(
                            "branching_distance", right.get("branching_distance")
                        ),
                        "left_trajectory_id": left["trajectory_id"],
                        "right_trajectory_id": right["trajectory_id"],
                        "snapshot_id": left["snapshot_id"],
                        "seed": left["seed"],
                        "same_initial_state_max_error": max(initial_errors),
                        "distance_teacher_by_horizon_mv": distances,
                        "maximum_teacher_distance_mv": max(distances, default=0.0),
                        "first_divergence_ms": first_divergence,
                        "regional_max_distance_mv": regional_max,
                        "teacher_quasi_identical": max(distances, default=0.0)
                        <= 1.0e-6,
                        "left_event_labels": left["event_labels"],
                        "right_event_labels": right["event_labels"],
                        "event_labels_differ": set(left["event_labels"])
                        != set(right["event_labels"]),
                        "release_outcomes_differ": self._outcome_signature(
                            [item for rows in release_left for item in rows]
                        )
                        != self._outcome_signature(
                            [item for rows in release_right for item in rows]
                        ),
                        "modified_parameter": (
                            f"{left['protocol_id']} vs {right['protocol_id']}"
                        ),
                    }
                )
            release_contracts: Dict[str, List[Mapping[str, Any]]] = {}
            for row in episodes:
                pair_id = row.get("release_pair_id")
                if pair_id:
                    release_contracts.setdefault(str(pair_id), []).append(row)
            for pair_id, pair in sorted(release_contracts.items()):
                if len(pair) != 2:
                    raise RuntimeError(
                        f"release pair {pair_id} must contain exactly two futures"
                    )
                left, right = pair
                left_ids = sorted(
                    transition_ids_by_trajectory[left["trajectory_id"]]
                )
                right_ids = sorted(
                    transition_ids_by_trajectory[right["trajectory_id"]]
                )
                left_inputs = [
                    json.loads(handle["inputs/U_scheduled_json"][index])
                    for index in left_ids
                ]
                right_inputs = [
                    json.loads(handle["inputs/U_scheduled_json"][index])
                    for index in right_ids
                ]
                left_realized = [
                    json.loads(handle["inputs/U_realized_json"][index])
                    for index in left_ids
                ]
                right_realized = [
                    json.loads(handle["inputs/U_realized_json"][index])
                    for index in right_ids
                ]
                release_pair_rows.append(
                    {
                        "release_pair_id": pair_id,
                        "same_snapshot": left["snapshot_id"] == right["snapshot_id"],
                        "different_seed": int(left["seed"]) != int(right["seed"]),
                        "same_scheduled_input": canonical_json_sha256(
                            {"rows": left_inputs}
                        )
                        == canonical_json_sha256({"rows": right_inputs}),
                        "release_outcomes_differ": canonical_json_sha256(
                            {"rows": left_realized}
                        )
                        != canonical_json_sha256({"rows": right_realized}),
                    }
                )
        self.pd.DataFrame(
            self._parquet_safe_rows(self.protocol_rows)
        ).to_parquet(self.output_dir / "protocols.parquet", index=False)
        self.pd.DataFrame(self._parquet_safe_rows(episodes)).to_parquet(
            self.output_dir / "episodes.parquet", index=False
        )
        self.pd.DataFrame(self._parquet_safe_rows(transition_rows)).to_parquet(
            self.output_dir / "transition_index.parquet", index=False
        )
        self.pd.DataFrame(self._parquet_safe_rows(event_rows)).to_parquet(
            self.output_dir / "events.parquet", index=False
        )
        branching_columns = [
            "branch_pair_id",
            "branching_distance",
            "left_trajectory_id",
            "right_trajectory_id",
        ]
        branching_frame = self.pd.DataFrame(
            self._parquet_safe_rows(branching_rows)
        )
        if branching_frame.empty:
            branching_frame = self.pd.DataFrame(columns=branching_columns)
        branching_frame.to_parquet(
            self.output_dir / "branching_pairs.parquet", index=False
        )
        split_payload = {
            split: sorted(
                row["trajectory_id"]
                for row in self.protocol_rows
                if row["split"] == split
            )
            for split in sorted({row["split"] for row in self.protocol_rows})
        }
        write_json(self.output_dir / "splits.json", split_payload)
        dataset_card = {
            "schema_version": TARGETED_DATASET_SCHEMA_VERSION,
            "transition_count": len(transition_rows),
            "episode_count": len(episodes),
            "support": support,
            "support_validation": support_validation,
            "release_outcomes": {
                f"{kind}:{outcome}": count
                for (kind, outcome), count in sorted(release_success.items())
            },
            "release_identifiability_pairs": release_pair_rows,
            "event_count": len(event_rows),
            "event_cooccurrence": dict(
                Counter(
                    "+".join(row["event_labels"]) or "none" for row in episodes
                )
            ),
            "branch_count": len({row["branch_id"] for row in episodes}),
            "seed_count": len({int(row["seed"]) for row in episodes}),
            "snapshot_count": len({row["snapshot_id"] for row in episodes}),
            "protocol_variant_count": len(
                {row["protocol_variant"] for row in episodes}
            ),
            "episodes_by_seed": dict(
                Counter(str(int(row["seed"])) for row in episodes)
            ),
            "episodes_by_branch": dict(
                Counter(str(row["branch_id"]) for row in episodes)
            ),
            "episodes_by_protocol_variant": dict(
                Counter(str(row["protocol_variant"]) for row in episodes)
            ),
            "episodes_by_split": dict(
                Counter(str(row["split"]) for row in episodes)
            ),
            "boundary_distance": {
                "minimum": min(
                    (float(row.get("boundary_distance", 0.0)) for row in episodes),
                    default=0.0,
                ),
                "maximum": max(
                    (float(row.get("boundary_distance", 0.0)) for row in episodes),
                    default=0.0,
                ),
                "values": [
                    float(row.get("boundary_distance", 0.0)) for row in episodes
                ],
            },
            "event_duration_ms_by_kind": {
                kind: [
                    float(row["duration_ms"])
                    for row in event_rows
                    if row["kind"] == kind
                ]
                for kind in sorted({row["kind"] for row in event_rows})
            },
            "recovery_probe_delays_ms": sorted(
                {
                    float(row["recovery_probe_delay_ms"])
                    for row in episodes
                    if row.get("recovery_probe_delay_ms") is not None
                    and not self.pd.isna(row.get("recovery_probe_delay_ms"))
                }
            ),
            "branching_pair_count": len(branching_rows),
            "branching_pairs": branching_rows,
            "hdf5_bytes": self.transition_path.stat().st_size,
        }
        write_json(self.output_dir / "dataset_card.json", dataset_card)
        write_json(
            self.output_dir / "storage_report.json",
            {
                "schema_version": TARGETED_DATASET_SCHEMA_VERSION,
                "hdf5_bytes": self.transition_path.stat().st_size,
                "hdf5_bytes_per_transition": int(
                    self.transition_path.stat().st_size / len(transition_rows)
                ),
                "release_table_bytes": (
                    self.output_dir / "release_outcomes.parquet"
                ).stat().st_size,
                "microtrace_interval_ms": 0.025,
                "canonical_boundary_state_unchanged": True,
            },
        )
        self._plot_targeted_figures(dataset_card)
        return {
            "transition_count": len(transition_rows),
            "episode_count": len(episodes),
            "event_count": len(event_rows),
            "release_count": len(self.release_rows),
            "support_valid": support_validation["valid"],
        }

    def _plot_targeted_figures(self, card: Mapping[str, Any]) -> None:
        import matplotlib.pyplot as plt

        figures = self.output_dir / "figures"
        figures.mkdir(parents=True, exist_ok=True)
        classes = [
            "axonal_spike",
            "somatic_spike",
            "backpropagating_ap",
            "calcium_spike",
            "nmda_spike",
            "nmda_plateau",
        ]
        support = card["support"]
        positives = [
            int(support.get(kind, {}).get("train", {}).get("positive_episode_count", 0))
            for kind in classes
        ]
        negatives = [
            int(
                support.get(kind, {})
                .get("train", {})
                .get("hard_negative_episode_count", 0)
            )
            for kind in classes
        ]
        x = self.np.arange(len(classes))
        figure, axis = plt.subplots(figsize=(11, 4.5))
        axis.bar(x - 0.2, positives, width=0.4, label="positive episodes")
        axis.bar(x + 0.2, negatives, width=0.4, label="hard negatives")
        axis.set_xticks(x, classes, rotation=30, ha="right")
        axis.set_ylabel("independent train episodes")
        axis.legend()
        figure.tight_layout()
        figure.savefig(figures / "independent_support.png", dpi=180)
        plt.close(figure)

        release_counts = card["release_outcomes"]
        labels = sorted(release_counts)
        figure, axis = plt.subplots(figsize=(9, 4))
        axis.bar(labels, [release_counts[label] for label in labels])
        axis.set_ylabel("presynaptic events")
        axis.tick_params(axis="x", rotation=25)
        figure.tight_layout()
        figure.savefig(figures / "release_success_failure.png", dpi=180)
        plt.close(figure)

        figure, axis = plt.subplots(figsize=(8, 4))
        axis.hist(card["boundary_distance"]["values"], bins=31)
        axis.axvline(0.0, color="black", linewidth=1)
        axis.set_xlabel("normalized boundary_distance")
        axis.set_ylabel("episodes")
        figure.tight_layout()
        figure.savefig(figures / "boundary_distance.png", dpi=180)
        plt.close(figure)

        figure, axis = plt.subplots(figsize=(8, 4))
        for row in card["branching_pairs"]:
            axis.plot(
                self.np.arange(1, len(row["distance_teacher_by_horizon_mv"]) + 1),
                row["distance_teacher_by_horizon_mv"],
                label=f"{row['branching_distance']}:{row['branch_pair_id']}",
            )
        axis.set_xlabel("horizon (ms)")
        axis.set_ylabel("teacher max |ΔV| (mV)")
        if card["branching_pairs"]:
            axis.legend()
        figure.tight_layout()
        figure.savefig(figures / "branching_teacher_distance.png", dpi=180)
        plt.close(figure)

    def validate_dataset_v1_1(self) -> Dict[str, Any]:
        """Run structural, exhaustive replay, causal, support and split gates."""

        structural = validate_hdf5_store(self.transition_path)
        exhaustive = self._exhaustive_sequential_replay()
        card = json.loads(
            (self.output_dir / "dataset_card.json").read_text(encoding="utf-8")
        )
        observed_splits = set(
            json.loads((self.output_dir / "splits.json").read_text(encoding="utf-8"))
        )
        required_splits = {
            "train",
            "validation",
            "deterministic_test",
            "event_boundary_test",
            "held_out_branch_test",
            "held_out_seed_test",
            "branching_near_test",
            "branching_far_test",
            "release_identifiability_test",
            "recovery_test",
        }
        release_rows = self.release_rows
        if not release_rows:
            release_rows = self.pd.read_parquet(
                self.output_dir / "release_outcomes.parquet"
            ).to_dict("records")
        release_valid = bool(release_rows) and all(
            not bool(row.get("future_state_used", False))
            and row["source"] == "exact_causal_replay_of_original_net_receive"
            for row in release_rows
        )
        episode_rows = self.pd.read_parquet(
            self.output_dir / "episodes.parquet"
        ).to_dict("records")
        seed_splits: Dict[int, set] = {}
        snapshot_splits: Dict[str, set] = {}
        for row in episode_rows:
            seed_splits.setdefault(int(row["seed"]), set()).add(str(row["split"]))
            snapshot_splits.setdefault(str(row["snapshot_id"]), set()).add(
                str(row["split"])
            )
        leaking_seeds = {
            seed: sorted(splits)
            for seed, splits in seed_splits.items()
            if len(splits) > 1
        }
        leaking_snapshots = {
            snapshot: sorted(splits)
            for snapshot, splits in snapshot_splits.items()
            if len(splits) > 1
        }
        train_branches = {
            str(row["branch_id"])
            for row in episode_rows
            if row["split"] == "train"
        }
        heldout_branches = {
            str(row["branch_id"])
            for row in episode_rows
            if row["split"] == "held_out_branch_test"
        }
        heldout_branch_valid = bool(heldout_branches) and not (
            train_branches & heldout_branches
        )
        branching_rows = self.pd.read_parquet(
            self.output_dir / "branching_pairs.parquet"
        ).to_dict("records")
        branching_valid = (
            {str(row["branching_distance"]) for row in branching_rows}
            == {"near", "far"}
            and all(
                float(row["same_initial_state_max_error"]) == 0.0
                for row in branching_rows
            )
        )
        recovery_rows = [
            row for row in episode_rows if row["split"] == "recovery_test"
        ]
        recovery_valid = bool(recovery_rows) and all(
            bool(row.get("pilot_validated"))
            and bool(row.get("recovery_probe"))
            for row in recovery_rows
        )
        release_pair_rows = card.get("release_identifiability_pairs", [])
        release_identifiability_valid = bool(release_pair_rows) and all(
            bool(row["same_snapshot"])
            and bool(row["different_seed"])
            and bool(row["same_scheduled_input"])
            and bool(row["release_outcomes_differ"])
            for row in release_pair_rows
        )
        uncensored_required = all(
            not bool(row.get("right_censored"))
            for row in self.pd.read_parquet(self.output_dir / "events.parquet").to_dict(
                "records"
            )
            if row.get("kind") in {"calcium_spike", "nmda_plateau"}
        )
        blockers = []
        if not structural["valid"]:
            blockers.append("HDF5 structural validation failed")
        if not exhaustive["valid"]:
            blockers.append("exhaustive transition/release replay failed")
        if not release_valid:
            blockers.append("causal release contract failed")
        if not card["support_validation"]["valid"]:
            blockers.append("minimum independent support is not satisfied")
        if observed_splits != required_splits:
            blockers.append("required targeted split set is incomplete")
        if leaking_seeds:
            blockers.append("Random123 seed leakage across splits")
        if leaking_snapshots:
            blockers.append("initial snapshot leakage across splits")
        if not heldout_branch_valid:
            blockers.append("held-out branch is present in train")
        if not branching_valid:
            blockers.append("near/far branching contract failed")
        if not recovery_valid:
            blockers.append("recovery probes are missing or not pilot-validated")
        if not release_identifiability_valid:
            blockers.append("release-identifiability pair did not isolate RNG outcome")
        if not uncensored_required:
            blockers.append("a required duration event is right censored")
        if len(self.audit.live_segments) != 642:
            blockers.append("canonical segment mapping changed")
        if self.state_schema["canonical_state_layout_sha256"] != canonical_json_sha256(
            {
                "index_contract": self.state_schema["index_contract"],
                "categories": self.state_schema["categories"],
                "rng_state": self.state_schema["rng_state"],
                "variables": self.state_schema["variables"],
            }
        ):
            blockers.append("canonical state layout changed")
        report = {
            "schema_version": TARGETED_DATASET_SCHEMA_VERSION,
            "valid": not blockers,
            "blockers": blockers,
            "structural": structural,
            "exhaustive_replay": exhaustive,
            "causal_release_valid": release_valid,
            "support": card["support_validation"],
            "required_splits": sorted(required_splits),
            "observed_splits": sorted(observed_splits),
            "seed_split_leaks": leaking_seeds,
            "snapshot_split_leaks": leaking_snapshots,
            "heldout_branch_valid": heldout_branch_valid,
            "branching_valid": branching_valid,
            "recovery_valid": recovery_valid,
            "release_identifiability_valid": release_identifiability_valid,
            "uncensored_required_events": uncensored_required,
            "teacher_commit": PINNED_TEACHER_COMMIT,
            "segment_count": len(self.audit.live_segments),
            "state_layout_unchanged": "canonical state layout changed" not in blockers,
        }
        write_json(self.output_dir / "validation_report.json", report)
        self._write_artifact_index()
        if blockers:
            raise RuntimeError(f"diagnostic dataset v1.1 validation failed: {blockers}")
        return report

    def accept_targeted_protocol_plan(
        self,
        protocols: Sequence[ProtocolTrajectory],
        *,
        pilot_report: Mapping[str, Any],
        positive_support_targets: Optional[Mapping[str, int]] = None,
        hard_negative_support_targets: Optional[Mapping[str, int]] = None,
        require_snapshot_bank: bool = True,
    ) -> Dict[str, Any]:
        """Bind generation to a reviewed pilot and exact protocol hash."""

        transition_count = sum(int(row.duration_ms) for row in protocols)
        blockers = []
        if not pilot_report.get("valid"):
            blockers.append("targeted biological pilot is not valid")
        if not 10_000 <= transition_count <= 30_000:
            blockers.append("planned transition count is outside 10k-30k")
        snapshot_ids = {
            str(row.metadata.get("snapshot_id", row.snapshot_source))
            for row in protocols
        }
        if require_snapshot_bank and snapshot_ids - set(self.snapshot_bank):
            blockers.append("split-specific snapshot bank is incomplete")
        required_splits = {
            "train",
            "validation",
            "deterministic_test",
            "event_boundary_test",
            "held_out_branch_test",
            "held_out_seed_test",
            "branching_near_test",
            "branching_far_test",
            "release_identifiability_test",
            "recovery_test",
        }
        observed_splits = {row.split for row in protocols}
        if observed_splits != required_splits:
            blockers.append("targeted split set is incomplete")
        seed_splits: Dict[int, set] = {}
        snapshot_splits: Dict[str, set] = {}
        planned_episode_rows = []
        for row in protocols:
            metadata = dict(row.metadata)
            seed_splits.setdefault(int(row.seed), set()).add(row.split)
            snapshot_id = str(metadata.get("snapshot_id", row.snapshot_source))
            snapshot_splits.setdefault(snapshot_id, set()).add(row.split)
            planned_episode_rows.append(
                {
                    "episode_id": metadata.get("episode_id", row.trajectory_id),
                    "trajectory_id": row.trajectory_id,
                    "split": row.split,
                    "seed": int(row.seed),
                    "snapshot_id": snapshot_id,
                    "branch_id": str(metadata.get("branch_id", "unknown")),
                    "protocol_variant": row.protocol_variant,
                    "event_labels": list(metadata.get("positive_for", ())),
                    "hard_negative_for": list(
                        metadata.get("hard_negative_for", ())
                    ),
                }
            )
        leaking_seeds = {
            seed: sorted(splits)
            for seed, splits in seed_splits.items()
            if len(splits) > 1
        }
        leaking_snapshots = {
            snapshot: sorted(splits)
            for snapshot, splits in snapshot_splits.items()
            if len(splits) > 1
        }
        if leaking_seeds:
            blockers.append("Random123 seeds leak across splits")
        if leaking_snapshots:
            blockers.append("initial snapshots leak across splits")
        train_branches = {
            row["branch_id"]
            for row in planned_episode_rows
            if row["split"] == "train"
        }
        heldout_branches = {
            row["branch_id"]
            for row in planned_episode_rows
            if row["split"] == "held_out_branch_test"
        }
        if not heldout_branches or train_branches & heldout_branches:
            blockers.append("held-out branch is present in train")
        planned_support = summarize_independent_support(planned_episode_rows)
        planned_support_validation = validate_minimum_support(
            planned_support,
            positive_targets=positive_support_targets,
            hard_negative_targets=hard_negative_support_targets,
        )
        if not planned_support_validation["valid"]:
            blockers.append("planned independent support is insufficient")
        recovery_rows = [
            row for row in protocols if row.split == "recovery_test"
        ]
        if not recovery_rows or not all(
            bool(row.metadata.get("pilot_validated"))
            and bool(row.metadata.get("recovery_probe"))
            for row in recovery_rows
        ):
            blockers.append("recovery split is not pilot-validated")
        report = {
            "schema_version": TARGETED_DATASET_SCHEMA_VERSION,
            "valid": not blockers,
            "blockers": blockers,
            "protocol_plan_sha256": self._protocol_plan_sha256(protocols),
            "trajectory_count": len(protocols),
            "transition_count": transition_count,
            "snapshot_count": len(snapshot_ids),
            "required_splits": sorted(required_splits),
            "observed_splits": sorted(observed_splits),
            "seed_split_leaks": leaking_seeds,
            "snapshot_split_leaks": leaking_snapshots,
            "heldout_branches": sorted(heldout_branches),
            "planned_support": planned_support,
            "planned_support_validation": planned_support_validation,
            "positive_support_targets": (
                None
                if positive_support_targets is None
                else dict(positive_support_targets)
            ),
            "hard_negative_support_targets": (
                None
                if hard_negative_support_targets is None
                else dict(hard_negative_support_targets)
            ),
            "snapshot_bank_required": bool(require_snapshot_bank),
            "pilot_report": dict(pilot_report),
        }
        if require_snapshot_bank:
            self.targeted_preflight_report = report
            destination = self.output_dir / "targeted_preflight_report.json"
        else:
            destination = self.output_dir / "targeted_plan_review.json"
        write_json(destination, report)
        return report
