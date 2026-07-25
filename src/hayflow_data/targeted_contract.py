"""Data contracts for the targeted HayFlow diagnostic dataset v1.1.

The release contract deliberately separates the presynaptic schedule from the
causal output of the authentic stochastic synapse front-end.  No field in a
``CausalReleaseOutcome`` may be inferred from the membrane state at the end of
the one-millisecond transition.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple


TARGETED_DATASET_SCHEMA_VERSION = "1.1.2"
RELEASE_SCHEMA_VERSION = "1.2.0"
TARGETED_EVENT_CLASSES: Tuple[str, ...] = (
    "axonal_spike",
    "somatic_spike",
    "backpropagating_ap",
    "calcium_spike",
    "nmda_spike",
    "nmda_plateau",
)
TARGETED_SPLITS: Tuple[str, ...] = (
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
)

CAUSAL_OBSERVATION_PHASE = "presynaptic_frontend_before_membrane_advance"


@dataclass(frozen=True)
class CausalReleaseOutcome:
    """One release decision observed at the synaptic event boundary.

    ``pre_synapse_state`` and ``post_synapse_state`` are evaluated from the
    boundary state by an independent, exact replay of the canonical
    ``NET_RECEIVE`` equations and a cloned Random123 stream.  This causal
    prediction is made before membrane integration and is verified against the
    authentic teacher synapse states and RNG position at the next boundary.
    """

    transition_id: int
    event_index: int
    synapse_id: int
    scheduled_time_ms: float
    offset_ms: float
    synapse_type: str
    functional_type: str
    weight: float
    random123_seed: int
    random123_stream_id: int
    random123_global_index: int
    rng_sequence_before: float
    rng_sequence_after: float
    rng_distribution: str
    rng_preview_value: float
    release_probability: float
    release_success: bool
    released_quantity: float
    ampa_state_increment: float
    nmda_state_increment: float
    inhibitory_state_increment: float
    pre_synapse_state: Mapping[str, float]
    post_synapse_state: Mapping[str, float]
    observation_phase: str = CAUSAL_OBSERVATION_PHASE
    source: str = "exact_causal_replay_of_original_net_receive"

    def validate(self) -> None:
        if self.transition_id < -1:
            raise ValueError("transition_id must be at least -1")
        if self.event_index < 0 or self.synapse_id < 0:
            raise ValueError("event and synapse ids must be non-negative")
        if not 0.0 <= self.offset_ms < 1.0:
            raise ValueError("release offset must lie in [0, 1 ms)")
        if self.random123_stream_id < 0 or self.random123_global_index < 0:
            raise ValueError("Random123 identifiers must be non-negative")
        if self.rng_distribution != "negexp(1)":
            raise ValueError("canonical Hay synapses require negexp(1)")
        if self.observation_phase != CAUSAL_OBSERVATION_PHASE:
            raise ValueError("release outcome was not observed at the causal boundary")
        if "t_plus_1" in self.source or "future" in self.source:
            raise ValueError("release source may not refer to future state")
        if not self.pre_synapse_state or not self.post_synapse_state:
            raise ValueError("pre/post synapse state is required")
        increments = (
            self.ampa_state_increment,
            self.nmda_state_increment,
            self.inhibitory_state_increment,
        )
        observed_success = any(abs(value) > 1.0e-12 for value in increments)
        if bool(self.release_success) != observed_success:
            raise ValueError("release flag disagrees with causal state increment")
        if self.release_success and self.released_quantity <= 0.0:
            raise ValueError("successful release requires positive released quantity")
        if not self.release_success and abs(self.released_quantity) > 1.0e-12:
            raise ValueError("failed release must have zero released quantity")

    def to_dict(self) -> Dict[str, Any]:
        self.validate()
        return dict(asdict(self))


def build_input_views(
    scheduled_actions: Sequence[Mapping[str, Any]],
    outcomes: Sequence[CausalReleaseOutcome],
) -> Dict[str, List[Dict[str, Any]]]:
    """Build scheduled, RNG-aware, and realized causal input views."""

    outcome_by_event = {int(row.event_index): row for row in outcomes}
    scheduled: List[Dict[str, Any]] = []
    rng_view: List[Dict[str, Any]] = []
    realized: List[Dict[str, Any]] = []
    synaptic_index = 0
    for raw in scheduled_actions:
        action = dict(raw)
        scheduled.append(action)
        if action.get("kind") != "synaptic_event":
            rng_view.append(dict(action))
            realized.append(dict(action))
            continue
        if synaptic_index not in outcome_by_event:
            raise ValueError(f"missing causal outcome for synaptic event {synaptic_index}")
        outcome = outcome_by_event[synaptic_index]
        rng_action = dict(action)
        rng_action.update(
            {
                "random123_seed": outcome.random123_seed,
                "random123_stream_id": outcome.random123_stream_id,
                "random123_global_index": outcome.random123_global_index,
                "rng_sequence_before": outcome.rng_sequence_before,
                "rng_distribution": outcome.rng_distribution,
            }
        )
        rng_view.append(rng_action)
        if outcome.release_success:
            realized.append(
                {
                    **action,
                    "release_success": True,
                    "released_quantity": outcome.released_quantity,
                    "ampa_state_increment": outcome.ampa_state_increment,
                    "nmda_state_increment": outcome.nmda_state_increment,
                    "inhibitory_state_increment": outcome.inhibitory_state_increment,
                }
            )
        synaptic_index += 1
    if synaptic_index != len(outcomes):
        raise ValueError("causal outcomes do not match scheduled synaptic events")
    return {
        "U_scheduled": scheduled,
        "U_rng": rng_view,
        "U_realized": realized,
    }


def summarize_independent_support(
    episodes: Iterable[Mapping[str, Any]],
) -> Dict[str, Dict[str, Dict[str, int]]]:
    """Count support without treating consecutive transitions as episodes."""

    result: Dict[str, Dict[str, Dict[str, set]]] = {}
    for row in episodes:
        split = str(row["split"])
        labels = set(row.get("event_labels", ()))
        hard_negative_for = set(row.get("hard_negative_for", ()))
        for event_class in TARGETED_EVENT_CLASSES:
            role = (
                "positive"
                if event_class in labels
                else "hard_negative"
                if event_class in hard_negative_for
                else None
            )
            if role is None:
                continue
            bucket = result.setdefault(event_class, {}).setdefault(
                split,
                {
                    "positive_episode_ids": set(),
                    "hard_negative_episode_ids": set(),
                    "seed_ids": set(),
                    "snapshot_ids": set(),
                    "branch_ids": set(),
                    "protocol_variants": set(),
                    "trajectory_ids": set(),
                },
            )
            bucket[f"{role}_episode_ids"].add(str(row["episode_id"]))
            bucket["seed_ids"].add(int(row["seed"]))
            bucket["snapshot_ids"].add(str(row["snapshot_id"]))
            bucket["branch_ids"].add(str(row["branch_id"]))
            bucket["protocol_variants"].add(str(row["protocol_variant"]))
            bucket["trajectory_ids"].add(str(row["trajectory_id"]))

    counts: Dict[str, Dict[str, Dict[str, int]]] = {}
    for event_class, by_split in result.items():
        counts[event_class] = {}
        for split, bucket in by_split.items():
            counts[event_class][split] = {
                name.replace("_ids", "_count"): len(values)
                for name, values in bucket.items()
            }
    return counts


def validate_minimum_support(
    support: Mapping[str, Mapping[str, Mapping[str, int]]],
    *,
    positive_targets: Mapping[str, int] | None = None,
    hard_negative_targets: Mapping[str, int] | None = None,
) -> Dict[str, Any]:
    """Apply the v1.1 independent-episode acceptance targets."""

    if positive_targets is None and hard_negative_targets is None:
        targets = {
            "train": (64, 128),
            "validation": (16, 32),
            "test": (16, 32),
        }
    else:
        positive = dict(positive_targets or {})
        negative = dict(hard_negative_targets or {})
        if not positive or set(positive) != set(negative):
            raise ValueError(
                "positive and hard-negative support targets must share splits"
            )
        targets = {
            str(split): (int(positive[split]), int(negative[split]))
            for split in positive
        }
    failures = []
    for event_class in TARGETED_EVENT_CLASSES:
        by_split = support.get(event_class, {})
        for split, (positive_target, negative_target) in targets.items():
            if split == "test":
                test_rows = [
                    values
                    for name, values in by_split.items()
                    if name.endswith("_test") or name == "deterministic_test"
                ]
                positive = sum(
                    int(row.get("positive_episode_count", 0)) for row in test_rows
                )
                negative = sum(
                    int(row.get("hard_negative_episode_count", 0))
                    for row in test_rows
                )
            else:
                row = by_split.get(split, {})
                positive = int(row.get("positive_episode_count", 0))
                negative = int(row.get("hard_negative_episode_count", 0))
            if positive < positive_target or negative < negative_target:
                failures.append(
                    {
                        "event_class": event_class,
                        "split": split,
                        "positive": positive,
                        "positive_target": positive_target,
                        "hard_negative": negative,
                        "hard_negative_target": negative_target,
                    }
                )
    return {"valid": not failures, "targets": targets, "failures": failures}


def validate_support_contract(
    support: Mapping[str, Mapping[str, Mapping[str, int]]],
    *,
    minimum_positive_targets: Mapping[str, int],
    minimum_hard_negative_targets: Mapping[str, int],
    planned_positive_targets: Mapping[str, int],
    planned_hard_negative_targets: Mapping[str, int],
) -> Dict[str, Any]:
    """Separate the acceptance floor from the planner's nominal quota.

    Biological recipes are selected from a finite pilot and their realized
    labels can change under new Random123 seeds.  Consequently, the quota used
    to size a plan is an attainment target, while ``minimum_*`` is the hard
    scientific acceptance gate.  Treating the two as the same contract makes
    an otherwise valid diagnostic dataset fail after a single stochastic
    label flip.
    """

    minimum = validate_minimum_support(
        support,
        positive_targets=minimum_positive_targets,
        hard_negative_targets=minimum_hard_negative_targets,
    )
    planned = validate_minimum_support(
        support,
        positive_targets=planned_positive_targets,
        hard_negative_targets=planned_hard_negative_targets,
    )
    return {
        "valid": bool(minimum["valid"]),
        "minimum_support_validation": minimum,
        "planned_target_attainment": planned,
    }
