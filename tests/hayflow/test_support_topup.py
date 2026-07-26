import copy

import pytest

from src.hayflow_data import (
    BAP_SUPPORT_TOPUP_EPISODE_COUNT,
    build_bap_validation_topup_plan,
    select_bap_positive_recipe,
    validate_composite_support,
)


def _pilot():
    return {
        "recipes": [
            {
                "recipe_id": "bap-positive",
                "family": "targeted_bap_soma_only",
                "protocol_variant": "p3-factor3",
                "duration_ms": 80,
                "actions_by_step": {
                    "3": [
                        {
                            "kind": "somatic_current",
                            "offset_ms": 0.0,
                            "amplitude_na": 1.2,
                            "duration_ms": 0.9,
                            "weight_multiplier": 1.0,
                            "synapse_id": None,
                            "release_observed": None,
                            "rng_sequence_before": None,
                            "metadata": {},
                        }
                    ]
                },
                "positive_for": ["backpropagating_ap"],
                "hard_negative_for": [],
                "branch_id": "segment-387",
                "snapshot_id": "pilot",
                "boundary_distance": 0.01,
                "metadata": {},
            }
        ]
    }


def _episode(episode_id, seed, snapshot_id, labels=(), negatives=()):
    return {
        "episode_id": episode_id,
        "trajectory_id": episode_id,
        "split": "validation",
        "seed": seed,
        "snapshot_id": snapshot_id,
        "branch_id": "segment-387",
        "protocol_variant": "p3-factor3",
        "event_labels": list(labels),
        "hard_negative_for": list(negatives),
    }


def _targets():
    positive = {
        "train": 0,
        "validation": 4,
        "deterministic_test": 0,
    }
    negative = {key: 0 for key in positive}
    return positive, negative


def test_fixed_topup_plan_is_outcome_blind_and_isolated():
    recipe = select_bap_positive_recipe(_pilot())
    protocols, plan = build_bap_validation_topup_plan(recipe)

    assert len(protocols) == BAP_SUPPORT_TOPUP_EPISODE_COUNT == 8
    assert [row.seed for row in protocols] == list(range(720001, 720009))
    assert len({row.snapshot_source for row in protocols}) == 8
    assert {row.split for row in protocols} == {"validation"}
    assert all(row.metadata["hard_negative_for"] == [] for row in protocols)
    assert plan["selection_was_outcome_blind"] is True
    assert plan["transition_count"] == 640


def test_recipe_selection_rejects_ambiguity():
    pilot = _pilot()
    pilot["recipes"].append(copy.deepcopy(pilot["recipes"][0]))
    pilot["recipes"][1]["recipe_id"] = "another"
    with pytest.raises(RuntimeError, match="exactly one"):
        select_bap_positive_recipe(pilot)


def test_plan_hash_binds_actions():
    first = select_bap_positive_recipe(_pilot())
    _, first_plan = build_bap_validation_topup_plan(first)
    changed = _pilot()
    changed["recipes"][0]["actions_by_step"]["3"][0]["amplitude_na"] = 1.3
    second = select_bap_positive_recipe(changed)
    _, second_plan = build_bap_validation_topup_plan(second)
    assert first_plan["protocol_plan_sha256"] != second_plan["protocol_plan_sha256"]


def test_composite_accepts_one_new_positive_without_posthoc_negatives():
    all_labels = [
        "axonal_spike",
        "somatic_spike",
        "backpropagating_ap",
        "calcium_spike",
        "nmda_spike",
        "nmda_plateau",
    ]
    base = [
        _episode(f"base-{index}", 1 + index, f"base-s{index}", all_labels)
        for index in range(3)
    ]
    base.append(_episode("base-other", 4, "base-s3", [label for label in all_labels if label != "backpropagating_ap"]))
    topup = [
        _episode(
            f"topup-{index}",
            720001 + index,
            f"topup-s{index}",
            ["backpropagating_ap"] if index == 0 else [],
        )
        for index in range(8)
    ]
    positive, negative = _targets()
    report = validate_composite_support(
        base,
        topup,
        minimum_positive_targets=positive,
        minimum_hard_negative_targets=negative,
    )
    assert report["valid"]
    assert report["topup_bap_positive_count"] == 1
    bap = report["combined_support"]["backpropagating_ap"]["validation"]
    assert bap["positive_episode_count"] == 4
    assert bap.get("hard_negative_episode_count", 0) == 0


@pytest.mark.parametrize("fault", ["no_positive", "seed_overlap", "snapshot_overlap", "missing"])
def test_composite_rejects_invalid_topup(fault):
    base = [
        _episode(f"base-{index}", 1 + index, f"base-s{index}", ["backpropagating_ap"])
        for index in range(3)
    ]
    topup = [
        _episode(
            f"topup-{index}",
            720001 + index,
            f"topup-s{index}",
            ["backpropagating_ap"] if index == 0 else [],
        )
        for index in range(8)
    ]
    if fault == "no_positive":
        for row in topup:
            row["event_labels"] = []
    elif fault == "seed_overlap":
        topup[0]["seed"] = base[0]["seed"]
    elif fault == "snapshot_overlap":
        topup[0]["snapshot_id"] = base[0]["snapshot_id"]
    else:
        topup.pop()
    positive, negative = _targets()
    report = validate_composite_support(
        base,
        topup,
        minimum_positive_targets=positive,
        minimum_hard_negative_targets=negative,
    )
    assert not report["valid"]
    assert report["blockers"]
