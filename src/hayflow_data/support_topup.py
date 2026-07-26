"""Pre-registered support supplement for the targeted HayFlow dataset."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from .diagnostic_contract import ProtocolTrajectory
from .targeted_contract import summarize_independent_support, validate_minimum_support
from .targeted_protocols import TargetedRecipe, action_schedule_from_json


BAP_SUPPORT_TOPUP_SCHEMA_VERSION = "1.0.0"
BAP_SUPPORT_TOPUP_EPISODE_COUNT = 8
BAP_SUPPORT_TOPUP_SEED_START = 720_001


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def select_bap_positive_recipe(
    pilot_report: Mapping[str, Any],
) -> TargetedRecipe:
    """Recover the prevalidated positive BAP recipe from the base pilot."""

    candidates = [
        dict(row)
        for row in pilot_report.get("recipes", [])
        if "backpropagating_ap" in row.get("positive_for", [])
        and str(row.get("family")) == "targeted_bap_soma_only"
        and not bool(row.get("metadata", {}).get("recovery_probe", False))
    ]
    if len(candidates) != 1:
        raise RuntimeError(
            "expected exactly one prevalidated soma-only positive BAP recipe; "
            f"found {[row.get('recipe_id') for row in candidates]}"
        )
    row = candidates[0]
    recipe = TargetedRecipe(
        recipe_id=str(row["recipe_id"]),
        family=str(row["family"]),
        protocol_variant=str(row["protocol_variant"]),
        duration_ms=int(row["duration_ms"]),
        actions_by_step=action_schedule_from_json(row["actions_by_step"]),
        positive_for=tuple(map(str, row.get("positive_for", ()))),
        hard_negative_for=tuple(map(str, row.get("hard_negative_for", ()))),
        branch_id=str(row.get("branch_id", "soma")),
        snapshot_id=str(row.get("snapshot_id", "equilibrium")),
        boundary_distance=float(row.get("boundary_distance", 0.0)),
        recovery_probe_delay_ms=(
            None
            if row.get("recovery_probe_delay_ms") is None
            else float(row["recovery_probe_delay_ms"])
        ),
        metadata=dict(row.get("metadata", {})),
    )
    recipe.validate()
    return recipe


def build_bap_validation_topup_plan(
    recipe: TargetedRecipe,
    *,
    episode_count: int = BAP_SUPPORT_TOPUP_EPISODE_COUNT,
    seed_start: int = BAP_SUPPORT_TOPUP_SEED_START,
) -> Tuple[List[ProtocolTrajectory], Dict[str, Any]]:
    """Build one fixed, fully retained validation-only acquisition batch."""

    if "backpropagating_ap" not in recipe.positive_for:
        raise ValueError("the top-up recipe must be positive for backpropagating_ap")
    if int(episode_count) <= 0:
        raise ValueError("episode_count must be positive")
    trajectories = []
    plan_rows = []
    for index in range(int(episode_count)):
        seed = int(seed_start) + index
        episode_id = f"validation-topup-bap-episode{index:04d}"
        snapshot_id = f"validation-topup-bap-snapshot-{index:02d}"
        metadata = {
            **dict(recipe.metadata),
            "episode_id": episode_id,
            "recipe_id": recipe.recipe_id,
            "branch_id": recipe.branch_id,
            "snapshot_id": snapshot_id,
            "positive_for": ["backpropagating_ap"],
            "hard_negative_for": [],
            "boundary_distance": float(recipe.boundary_distance),
            "support_topup": True,
            "support_topup_policy": "fixed_batch_all_episodes_retained",
        }
        trajectory = ProtocolTrajectory(
            trajectory_id=episode_id,
            category="somatic_events",
            protocol=recipe.family,
            protocol_id=recipe.recipe_id,
            protocol_variant=recipe.protocol_variant,
            seed=seed,
            duration_ms=recipe.duration_ms,
            split="validation",
            actions_by_step=recipe.actions_by_step,
            event_enriched=True,
            stimulus_onset_step=min(recipe.actions_by_step, default=0),
            required_event_kinds=("backpropagating_ap",),
            negative_control=False,
            snapshot_source=snapshot_id,
            metadata=metadata,
        )
        trajectory.validate()
        trajectories.append(trajectory)
        plan_rows.append(
            {
                "episode_id": episode_id,
                "trajectory_id": episode_id,
                "split": "validation",
                "seed": seed,
                "snapshot_id": snapshot_id,
                "recipe_id": recipe.recipe_id,
                "duration_ms": recipe.duration_ms,
                "positive_intent": ["backpropagating_ap"],
            }
        )
    plan_payload = {
        "schema_version": BAP_SUPPORT_TOPUP_SCHEMA_VERSION,
        "policy": "fixed_batch_all_episodes_retained",
        "selection_was_outcome_blind": True,
        "target_event_class": "backpropagating_ap",
        "target_split": "validation",
        "required_positive_increment": 1,
        "episode_count": len(trajectories),
        "transition_count": sum(row.duration_ms for row in trajectories),
        "seed_start": int(seed_start),
        "recipe_id": recipe.recipe_id,
        "recipe_contract": {
            "recipe_id": recipe.recipe_id,
            "family": recipe.family,
            "protocol_variant": recipe.protocol_variant,
            "duration_ms": int(recipe.duration_ms),
            "actions_by_step": {
                str(step): [action.to_dict() for action in actions]
                for step, actions in sorted(recipe.actions_by_step.items())
            },
            "positive_for": list(recipe.positive_for),
            "hard_negative_for": list(recipe.hard_negative_for),
            "branch_id": recipe.branch_id,
            "boundary_distance": float(recipe.boundary_distance),
            "metadata": dict(recipe.metadata),
        },
        "episodes": plan_rows,
    }
    plan_payload["protocol_plan_sha256"] = _canonical_sha256(
        {"plan": plan_payload}
    )
    return trajectories, plan_payload


def validate_composite_support(
    base_episodes: Sequence[Mapping[str, Any]],
    topup_episodes: Sequence[Mapping[str, Any]],
    *,
    minimum_positive_targets: Mapping[str, int],
    minimum_hard_negative_targets: Mapping[str, int],
    expected_topup_episode_count: int = BAP_SUPPORT_TOPUP_EPISODE_COUNT,
) -> Dict[str, Any]:
    """Validate the logical union without mutating either physical shard."""

    base = [dict(row) for row in base_episodes]
    topup = [dict(row) for row in topup_episodes]
    combined = [*base, *topup]
    support = summarize_independent_support(combined)
    minimum = validate_minimum_support(
        support,
        positive_targets=minimum_positive_targets,
        hard_negative_targets=minimum_hard_negative_targets,
    )
    base_seeds = {int(row["seed"]) for row in base}
    topup_seeds = {int(row["seed"]) for row in topup}
    base_snapshots = {str(row["snapshot_id"]) for row in base}
    topup_snapshots = {str(row["snapshot_id"]) for row in topup}
    seed_overlap = sorted(base_seeds & topup_seeds)
    snapshot_overlap = sorted(base_snapshots & topup_snapshots)
    topup_bap_positives = sum(
        "backpropagating_ap" in set(map(str, row.get("event_labels", ())))
        for row in topup
    )
    blockers = []
    if len(topup) != int(expected_topup_episode_count):
        blockers.append("the complete pre-registered top-up batch was not retained")
    if any(str(row.get("split")) != "validation" for row in topup):
        blockers.append("a top-up episode is outside the validation split")
    if seed_overlap:
        blockers.append("top-up Random123 seeds overlap the base dataset")
    if snapshot_overlap:
        blockers.append("top-up snapshots overlap the base dataset")
    if topup_bap_positives < 1:
        blockers.append("the top-up produced no positive validation BAP episode")
    if not minimum["valid"]:
        blockers.append("combined dataset still fails minimum independent support")
    return {
        "schema_version": BAP_SUPPORT_TOPUP_SCHEMA_VERSION,
        "valid": not blockers,
        "blockers": blockers,
        "base_episode_count": len(base),
        "topup_episode_count": len(topup),
        "combined_episode_count": len(combined),
        "topup_bap_positive_count": int(topup_bap_positives),
        "seed_overlap": seed_overlap,
        "snapshot_overlap": snapshot_overlap,
        "minimum_support_validation": minimum,
        "combined_support": support,
    }
