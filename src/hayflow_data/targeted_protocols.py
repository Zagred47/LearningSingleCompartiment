"""Adaptive recipe selection and episode planning for dataset v1.1."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

from .diagnostic_contract import InputAction, ProtocolTrajectory
from .targeted_contract import TARGETED_EVENT_CLASSES


@dataclass(frozen=True)
class TargetedRecipe:
    recipe_id: str
    family: str
    protocol_variant: str
    duration_ms: int
    actions_by_step: Mapping[int, Tuple[InputAction, ...]]
    positive_for: Tuple[str, ...] = ()
    hard_negative_for: Tuple[str, ...] = ()
    branch_id: str = "soma"
    snapshot_id: str = "equilibrium"
    boundary_distance: float = 0.0
    recovery_probe_delay_ms: float | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.recipe_id or not self.family or not self.protocol_variant:
            raise ValueError("recipe identity fields are required")
        if self.duration_ms <= 0:
            raise ValueError("recipe duration must be positive")
        positive = set(self.positive_for)
        negative = set(self.hard_negative_for)
        if positive & negative:
            raise ValueError("a recipe cannot be positive and negative for one class")
        unknown = (positive | negative) - set(TARGETED_EVENT_CLASSES)
        if unknown:
            raise ValueError(f"unknown targeted event classes: {sorted(unknown)}")
        if not positive and not negative:
            raise ValueError("recipe must contribute positive or hard-negative support")
        for step, actions in self.actions_by_step.items():
            if not 0 <= int(step) < self.duration_ms:
                raise ValueError("recipe action step is outside the episode")
            offsets = []
            for action in actions:
                action.validate()
                offsets.append(action.offset_ms)
            if offsets != sorted(offsets):
                raise ValueError("recipe actions must be ordered")


def action_schedule_from_json(
    schedule: Mapping[str, Sequence[Mapping[str, Any]]],
) -> Dict[int, Tuple[InputAction, ...]]:
    result = {}
    for step, rows in schedule.items():
        actions = []
        for row in rows:
            metadata = row.get("metadata", {})
            if not isinstance(metadata, Mapping):
                metadata = {}
            actions.append(
                InputAction(
                    kind=str(row["kind"]),
                    offset_ms=float(row["offset_ms"]),
                    synapse_id=(
                        None
                        if row.get("synapse_id") is None
                        else int(row["synapse_id"])
                    ),
                    weight_multiplier=float(row.get("weight_multiplier", 1.0)),
                    duration_ms=(
                        None
                        if row.get("duration_ms") is None
                        else float(row["duration_ms"])
                    ),
                    amplitude_na=(
                        None
                        if row.get("amplitude_na") is None
                        else float(row["amplitude_na"])
                    ),
                    metadata=dict(metadata),
                )
            )
        result[int(step)] = tuple(actions)
    return result


def select_adaptive_recipe_brackets(
    trials: Iterable[Mapping[str, Any]],
    *,
    required_seed_count: int,
) -> Dict[str, Any]:
    """Select robust event/non-event recipes nearest each observed boundary."""

    grouped: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in trials:
        grouped[str(row["candidate_id"])].append(row)
    candidates = []
    for candidate_id, rows in grouped.items():
        if len({int(row["seed"]) for row in rows}) < int(required_seed_count):
            continue
        first = rows[0]
        event_sets = [set(row.get("event_kinds", ())) for row in rows]
        candidates.append(
            {
                "candidate_id": candidate_id,
                "stimulus_scalar": float(first["stimulus_scalar"]),
                "family": str(first.get("family", "default")),
                "branch_id": str(first.get("branch_id", "default")),
                "rows": rows,
                "robust_positive_for": [
                    kind
                    for kind in TARGETED_EVENT_CLASSES
                    if all(kind in events for events in event_sets)
                ],
                "robust_negative_for": [
                    kind
                    for kind in TARGETED_EVENT_CLASSES
                    if all(kind not in events for events in event_sets)
                ],
            }
        )
    selections = {}
    blockers = []
    for event_class in TARGETED_EVENT_CLASSES:
        positives = [
            row for row in candidates if event_class in row["robust_positive_for"]
        ]
        negatives = [
            row for row in candidates if event_class in row["robust_negative_for"]
        ]
        compatible_pairs = []
        for positive in positives:
            compatible_negatives = [
                row
                for row in negatives
                if row["family"] == positive["family"]
                and row["branch_id"] == positive["branch_id"]
            ]
            if not compatible_negatives:
                continue
            below = [
                row
                for row in compatible_negatives
                if row["stimulus_scalar"] < positive["stimulus_scalar"]
            ]
            negative = max(
                below or compatible_negatives,
                key=lambda row: row["stimulus_scalar"]
                if row in below
                else -abs(
                    row["stimulus_scalar"] - positive["stimulus_scalar"]
                ),
            )
            scale = max(abs(float(positive["stimulus_scalar"])), 1.0e-12)
            compatible_pairs.append(
                (
                    abs(
                        float(positive["stimulus_scalar"])
                        - float(negative["stimulus_scalar"])
                    )
                    / scale,
                    positive,
                    negative,
                )
            )
        if not positives or not negatives or not compatible_pairs:
            blockers.append(
                {
                    "event_class": event_class,
                    "positive_candidate_count": len(positives),
                    "negative_candidate_count": len(negatives),
                    "compatible_boundary_pair_count": len(compatible_pairs),
                }
            )
            continue
        _, positive, negative = min(
            compatible_pairs,
            key=lambda row: (
                row[0],
                row[1]["stimulus_scalar"],
                row[1]["candidate_id"],
            ),
        )
        threshold = 0.5 * (
            float(positive["stimulus_scalar"])
            + float(negative["stimulus_scalar"])
        )
        scale = max(abs(threshold), 1.0e-12)
        selections[event_class] = {
            "estimated_critical_value": threshold,
            "positive_candidate_id": positive["candidate_id"],
            "negative_candidate_id": negative["candidate_id"],
            "positive_boundary_distance": (
                float(positive["stimulus_scalar"]) - threshold
            )
            / scale,
            "negative_boundary_distance": (
                float(negative["stimulus_scalar"]) - threshold
            )
            / scale,
            "boundary_family": positive["family"],
            "branch_id": positive["branch_id"],
        }
    return {
        "valid": not blockers,
        "candidate_count": len(candidates),
        "selections": selections,
        "blockers": blockers,
    }


def _seed_for(split: str, index: int) -> int:
    prefixes = {
        "train": 610_000,
        "validation": 620_000,
        "deterministic_test": 630_000,
    }
    return prefixes[split] + int(index) + 1


def build_balanced_episode_plan(
    recipes: Sequence[TargetedRecipe],
    *,
    positive_targets: Mapping[str, int] = None,
    hard_negative_targets: Mapping[str, int] = None,
) -> Tuple[List[ProtocolTrajectory], List[Dict[str, Any]]]:
    """Greedily share multilabel episodes while satisfying split targets."""

    positive_targets = dict(
        positive_targets or {"train": 64, "validation": 16, "deterministic_test": 16}
    )
    hard_negative_targets = dict(
        hard_negative_targets
        or {"train": 128, "validation": 32, "deterministic_test": 32}
    )
    recipes = list(recipes)
    for recipe in recipes:
        recipe.validate()
    missing_positive = [
        kind
        for kind in TARGETED_EVENT_CLASSES
        if not any(
            kind in recipe.positive_for
            and bool(recipe.metadata.get("train_eligible", True))
            for recipe in recipes
        )
    ]
    missing_negative = [
        kind
        for kind in TARGETED_EVENT_CLASSES
        if not any(
            kind in recipe.hard_negative_for
            and bool(recipe.metadata.get("train_eligible", True))
            for recipe in recipes
        )
    ]
    if missing_positive or missing_negative:
        raise ValueError(
            "recipe catalog is incomplete: "
            f"missing positives={missing_positive}, negatives={missing_negative}"
        )

    trajectories: List[ProtocolTrajectory] = []
    episode_rows: List[Dict[str, Any]] = []
    for split in positive_targets:
        deficits = {
            ("positive", kind): int(positive_targets[split])
            for kind in TARGETED_EVENT_CLASSES
        }
        deficits.update(
            {
                ("hard_negative", kind): int(hard_negative_targets[split])
                for kind in TARGETED_EVENT_CLASSES
            }
        )
        episode_index = 0
        while any(value > 0 for value in deficits.values()):
            eligible = [
                recipe
                for recipe in recipes
                if bool(recipe.metadata.get("train_eligible", True))
            ]
            unresolved = [key for key, value in deficits.items() if value > 0]

            def covers(recipe: TargetedRecipe, key: Tuple[str, str]) -> bool:
                role, kind = key
                return kind in (
                    recipe.positive_for
                    if role == "positive"
                    else recipe.hard_negative_for
                )

            focus = min(
                unresolved,
                key=lambda key: (
                    sum(covers(recipe, key) for recipe in eligible),
                    -min(
                        (
                            recipe.duration_ms
                            for recipe in eligible
                            if covers(recipe, key)
                        ),
                        default=0,
                    ),
                    key,
                ),
            )
            scored = []
            for recipe in eligible:
                if not covers(recipe, focus):
                    continue
                gain = sum(
                    deficits[("positive", kind)] > 0
                    for kind in recipe.positive_for
                ) + sum(
                    deficits[("hard_negative", kind)] > 0
                    for kind in recipe.hard_negative_for
                )
                if gain:
                    scored.append((gain / recipe.duration_ms, gain, recipe))
            if not scored:
                raise RuntimeError("support planner stalled with unresolved deficits")
            _, _, recipe = max(
                scored,
                key=lambda row: (row[0], row[1], -row[2].duration_ms, row[2].recipe_id),
            )
            seed = _seed_for(split, episode_index)
            episode_id = f"{split}-{recipe.recipe_id}-episode{episode_index:04d}"
            snapshot_id = f"{split}-snapshot-{episode_index % 8:02d}"
            metadata = {
                **dict(recipe.metadata),
                "episode_id": episode_id,
                "recipe_id": recipe.recipe_id,
                "branch_id": recipe.branch_id,
                "snapshot_id": snapshot_id,
                "positive_for": list(recipe.positive_for),
                "hard_negative_for": list(recipe.hard_negative_for),
                "boundary_distance": float(recipe.boundary_distance),
                "recovery_probe_delay_ms": recipe.recovery_probe_delay_ms,
            }
            category = (
                "somatic_events"
                if recipe.family
                in {"somatic", "bap", "targeted_somatic_bap", "targeted_bap_soma_only"}
                else "dendritic_events"
            )
            trajectory = ProtocolTrajectory(
                trajectory_id=episode_id,
                category=category,
                protocol=recipe.family,
                protocol_id=recipe.recipe_id,
                protocol_variant=recipe.protocol_variant,
                seed=seed,
                duration_ms=recipe.duration_ms,
                split=split,
                actions_by_step=recipe.actions_by_step,
                event_enriched=bool(recipe.positive_for),
                stimulus_onset_step=min(recipe.actions_by_step, default=0),
                required_event_kinds=recipe.positive_for,
                negative_control=not bool(recipe.positive_for),
                snapshot_source=snapshot_id,
                metadata=metadata,
            )
            trajectory.validate()
            trajectories.append(trajectory)
            episode_rows.append(
                {
                    "episode_id": episode_id,
                    "trajectory_id": episode_id,
                    "split": split,
                    "seed": seed,
                    "snapshot_id": snapshot_id,
                    "branch_id": recipe.branch_id,
                    "protocol_variant": recipe.protocol_variant,
                    "event_labels": list(recipe.positive_for),
                    "hard_negative_for": list(recipe.hard_negative_for),
                    "boundary_distance": float(recipe.boundary_distance),
                    "duration_ms": recipe.duration_ms,
                    "recipe_id": recipe.recipe_id,
                }
            )
            for kind in recipe.positive_for:
                deficits[("positive", kind)] = max(
                    0, deficits[("positive", kind)] - 1
                )
            for kind in recipe.hard_negative_for:
                deficits[("hard_negative", kind)] = max(
                    0, deficits[("hard_negative", kind)] - 1
                )
            episode_index += 1
    return trajectories, episode_rows


def split_assignment_key(row: Mapping[str, Any]) -> str:
    """Stable identity used to audit seed/branch/snapshot-family isolation."""

    fields = (
        str(row["seed"]),
        str(row["branch_id"]),
        str(row["snapshot_id"]),
        str(row.get("protocol_family", row.get("protocol", ""))),
    )
    return hashlib.sha256(":".join(fields).encode("utf-8")).hexdigest()


def append_specialized_test_episodes(
    trajectories: Sequence[ProtocolTrajectory],
    episode_rows: Sequence[Mapping[str, Any]],
    recipes: Sequence[TargetedRecipe],
) -> Tuple[List[ProtocolTrajectory], List[Dict[str, Any]]]:
    """Add held-out, branching, release-identifiability, and recovery episodes."""

    plans = list(trajectories)
    rows = [dict(row) for row in episode_rows]
    train_branches = {
        str(row["branch_id"]) for row in rows if row["split"] == "train"
    }
    heldout = [
        recipe
        for recipe in recipes
        if not bool(recipe.metadata.get("train_eligible", True))
        and recipe.branch_id not in train_branches
    ]
    recovery = [
        recipe for recipe in recipes if bool(recipe.metadata.get("recovery_probe", False))
    ]
    if not heldout:
        raise ValueError("recipe catalog has no truly held-out morphological branch")
    if not recovery:
        raise ValueError("recipe catalog has no pilot-validated recovery probe")
    positive = sorted(
        [recipe for recipe in recipes if recipe.positive_for],
        key=lambda recipe: abs(recipe.boundary_distance),
    )
    negative = sorted(
        [recipe for recipe in recipes if recipe.hard_negative_for],
        key=lambda recipe: abs(recipe.boundary_distance),
    )
    if not positive or not negative:
        raise ValueError("specialized tests require positive and hard-negative recipes")
    near_candidates = [
        (left, right)
        for left in positive
        for right in negative
        if left.family == right.family
        and left.branch_id == right.branch_id
        and set(left.positive_for) & set(right.hard_negative_for)
    ]
    if not near_candidates:
        raise ValueError(
            "near branching requires a same-family, same-branch boundary pair"
        )
    near_positive, near_negative = min(
        near_candidates,
        key=lambda pair: (
            abs(pair[0].boundary_distance) + abs(pair[1].boundary_distance),
            pair[0].duration_ms + pair[1].duration_ms,
        ),
    )
    release_recipes = [
        recipe
        for recipe in positive
        if any(
            action.kind == "synaptic_event"
            for actions in recipe.actions_by_step.values()
            for action in actions
        )
    ]
    if not release_recipes:
        raise ValueError("release-identifiability test requires synaptic inputs")
    release_recipe = release_recipes[0]

    counter = 0

    def add(
        recipe: TargetedRecipe,
        split: str,
        *,
        seed: int,
        suffix: str,
        metadata: Mapping[str, Any] = None,
    ) -> None:
        nonlocal counter
        episode_id = f"{split}-{recipe.recipe_id}-{suffix}-{counter:03d}"
        pair_id = str(
            (metadata or {}).get("branch_pair_id")
            or (metadata or {}).get("release_pair_id")
            or ""
        )
        snapshot_id = (
            f"{split}-{pair_id}-snapshot"
            if pair_id
            else f"{split}-snapshot-{counter:02d}"
        )
        extra = dict(metadata or {})
        contract = {
            **dict(recipe.metadata),
            **extra,
            "episode_id": episode_id,
            "recipe_id": recipe.recipe_id,
            "branch_id": recipe.branch_id,
            "snapshot_id": snapshot_id,
            "positive_for": list(recipe.positive_for),
            "hard_negative_for": list(recipe.hard_negative_for),
            "boundary_distance": recipe.boundary_distance,
            "recovery_probe_delay_ms": recipe.recovery_probe_delay_ms,
        }
        trajectory = ProtocolTrajectory(
            trajectory_id=episode_id,
            category=(
                "branching"
                if split in {"branching_near_test", "branching_far_test"}
                else "somatic_events"
                if recipe.family
                in {
                    "somatic",
                    "bap",
                    "targeted_somatic_bap",
                    "targeted_bap_soma_only",
                }
                else "dendritic_events"
            ),
            protocol=recipe.family,
            protocol_id=recipe.recipe_id,
            protocol_variant=recipe.protocol_variant,
            seed=int(seed),
            duration_ms=recipe.duration_ms,
            split=split,
            actions_by_step=recipe.actions_by_step,
            event_enriched=bool(recipe.positive_for),
            stimulus_onset_step=min(recipe.actions_by_step, default=0),
            required_event_kinds=recipe.positive_for,
            negative_control=not bool(recipe.positive_for),
            snapshot_source=snapshot_id,
            metadata=contract,
        )
        trajectory.validate()
        plans.append(trajectory)
        rows.append(
            {
                "episode_id": episode_id,
                "trajectory_id": episode_id,
                "split": split,
                "seed": int(seed),
                "snapshot_id": snapshot_id,
                "branch_id": recipe.branch_id,
                "protocol_variant": recipe.protocol_variant,
                "event_labels": list(recipe.positive_for),
                "hard_negative_for": list(recipe.hard_negative_for),
                "boundary_distance": recipe.boundary_distance,
                "duration_ms": recipe.duration_ms,
                "recipe_id": recipe.recipe_id,
                **extra,
            }
        )
        counter += 1

    add(near_positive, "event_boundary_test", seed=710001, suffix="above")
    add(near_negative, "event_boundary_test", seed=710002, suffix="below")
    add(heldout[0], "held_out_branch_test", seed=720001, suffix="branch")
    add(positive[0], "held_out_seed_test", seed=730001, suffix="seed")

    near_pair = "near-boundary-001"
    add(
        near_negative,
        "branching_near_test",
        seed=740001,
        suffix="negative",
        metadata={"branch_pair_id": near_pair, "branching_distance": "near"},
    )
    add(
        near_positive,
        "branching_near_test",
        seed=740001,
        suffix="positive",
        metadata={"branch_pair_id": near_pair, "branching_distance": "near"},
    )
    far_pair = "far-regime-001"
    add(
        negative[-1],
        "branching_far_test",
        seed=750001,
        suffix="control",
        metadata={"branch_pair_id": far_pair, "branching_distance": "far"},
    )
    add(
        positive[-1],
        "branching_far_test",
        seed=750001,
        suffix="event",
        metadata={"branch_pair_id": far_pair, "branching_distance": "far"},
    )
    add(
        release_recipe,
        "release_identifiability_test",
        seed=760001,
        suffix="rng-a",
        metadata={"release_pair_id": "same-schedule-different-rng-001"},
    )
    add(
        release_recipe,
        "release_identifiability_test",
        seed=760002,
        suffix="rng-b",
        metadata={"release_pair_id": "same-schedule-different-rng-001"},
    )
    for index, recipe in enumerate(recovery[:3]):
        add(
            recipe,
            "recovery_test",
            seed=770001 + index,
            suffix=f"delay-{recipe.recovery_probe_delay_ms}",
        )
    return plans, rows


def build_budgeted_episode_plan(
    recipes: Sequence[TargetedRecipe],
    *,
    preferred_positive_targets: Mapping[str, int],
    preferred_hard_negative_targets: Mapping[str, int],
    minimum_positive_targets: Mapping[str, int],
    minimum_hard_negative_targets: Mapping[str, int],
    minimum_transition_count: int = 10_000,
    maximum_transition_count: int = 30_000,
    search_steps: int = 100,
) -> Tuple[List[ProtocolTrajectory], List[Dict[str, Any]], Dict[str, Any]]:
    """Maximize independent support inside an explicit transition budget.

    Preferred support is the nominal planning target, while the minimum maps
    define the pre-registered acceptance floor for a diagnostic dataset.  The
    planner searches deterministically between them and includes all
    specialized test episodes in every budget calculation.  Realized labels
    are validated separately because new Random123 seeds can move observed
    support below the nominal quota.
    """

    preferred_positive = dict(preferred_positive_targets)
    preferred_negative = dict(preferred_hard_negative_targets)
    minimum_positive = dict(minimum_positive_targets)
    minimum_negative = dict(minimum_hard_negative_targets)
    split_names = set(preferred_positive)
    if not split_names or not (
        split_names
        == set(preferred_negative)
        == set(minimum_positive)
        == set(minimum_negative)
    ):
        raise ValueError("budgeted support maps must use the same non-empty splits")
    if int(minimum_transition_count) <= 0:
        raise ValueError("minimum transition budget must be positive")
    if int(maximum_transition_count) < int(minimum_transition_count):
        raise ValueError("maximum transition budget is below its minimum")
    if int(search_steps) <= 0:
        raise ValueError("budget search steps must be positive")
    for split in split_names:
        if not (
            0 <= int(minimum_positive[split]) <= int(preferred_positive[split])
            and 0
            <= int(minimum_negative[split])
            <= int(preferred_negative[split])
        ):
            raise ValueError(
                f"invalid preferred/minimum support ordering for {split}"
            )

    attempts = []
    observed_targets = set()
    selected = None
    for step in range(int(search_steps), -1, -1):
        fraction = step / float(search_steps)

        def interpolate(
            preferred: Mapping[str, int], minimum: Mapping[str, int]
        ) -> Dict[str, int]:
            return {
                split: int(minimum[split])
                + int(
                    round(
                        (int(preferred[split]) - int(minimum[split]))
                        * fraction
                    )
                )
                for split in sorted(split_names)
            }

        positive = interpolate(preferred_positive, minimum_positive)
        negative = interpolate(preferred_negative, minimum_negative)
        target_key = (
            tuple(sorted(positive.items())),
            tuple(sorted(negative.items())),
        )
        if target_key in observed_targets:
            continue
        observed_targets.add(target_key)
        trajectories, episode_rows = build_balanced_episode_plan(
            recipes,
            positive_targets=positive,
            hard_negative_targets=negative,
        )
        trajectories, episode_rows = append_specialized_test_episodes(
            trajectories, episode_rows, recipes
        )
        transition_count = sum(
            int(row.duration_ms) for row in trajectories
        )
        attempt = {
            "scale": fraction,
            "positive_targets": positive,
            "hard_negative_targets": negative,
            "trajectory_count": len(trajectories),
            "transition_count": transition_count,
        }
        attempts.append(attempt)
        if (
            int(minimum_transition_count)
            <= transition_count
            <= int(maximum_transition_count)
        ):
            selected = (trajectories, episode_rows, attempt)
            break

    if selected is None:
        smallest = min(attempts, key=lambda row: row["transition_count"])
        largest = max(attempts, key=lambda row: row["transition_count"])
        raise ValueError(
            "no support plan fits the transition budget; "
            f"observed range={smallest['transition_count']}-"
            f"{largest['transition_count']}, requested="
            f"{int(minimum_transition_count)}-{int(maximum_transition_count)}"
        )

    trajectories, episode_rows, effective = selected
    report = {
        "valid": True,
        "policy": "maximum_independent_support_within_transition_budget",
        "transition_budget": {
            "minimum": int(minimum_transition_count),
            "maximum": int(maximum_transition_count),
        },
        "preferred_positive_targets": preferred_positive,
        "preferred_hard_negative_targets": preferred_negative,
        "minimum_positive_targets": minimum_positive,
        "minimum_hard_negative_targets": minimum_negative,
        "effective_positive_targets": dict(effective["positive_targets"]),
        "effective_hard_negative_targets": dict(
            effective["hard_negative_targets"]
        ),
        "selected_scale": float(effective["scale"]),
        "trajectory_count": int(effective["trajectory_count"]),
        "transition_count": int(effective["transition_count"]),
        "attempt_count": len(attempts),
    }
    return trajectories, episode_rows, report
