import math
import unittest

import numpy as np

from src.hayflow_data.composite_flowmap import (
    EXPECTED_BASE_HDF_SHA256,
    EXPECTED_EPISODES,
    EXPECTED_PROTOCOL_PLAN_SHA256,
    EXPECTED_TOPUP_CONTRACT_SHA256,
    EXPECTED_TOPUP_HDF_SHA256,
    EXPECTED_TRANSITIONS,
    _canonical_release_rows,
    input_view_schema,
)
from src.hayflow_eval.release_flowmap_metrics import (
    branching_metrics,
    episode_bootstrap,
    episode_bootstrap_event_f1,
    identifiability_summary,
    macro_event_summary,
    pooled_event_metrics,
    release_flowmap_decision,
)
from src.hayflow_data.flowmap_dataset import EVENT_KINDS


class CompositeContractTests(unittest.TestCase):
    def test_preregistered_fingerprints_and_cardinality_are_frozen(self):
        self.assertEqual(EXPECTED_EPISODES, 369)
        self.assertEqual(EXPECTED_TRANSITIONS, 29_880)
        self.assertEqual(len(EXPECTED_BASE_HDF_SHA256), 64)
        self.assertEqual(len(EXPECTED_TOPUP_HDF_SHA256), 64)
        self.assertEqual(len(EXPECTED_PROTOCOL_PLAN_SHA256), 64)
        self.assertEqual(len(EXPECTED_TOPUP_CONTRACT_SHA256), 64)

    def test_realized_join_retains_failed_release(self):
        scheduled = [
            {"kind": "synaptic_event", "synapse_id": 7, "offset_ms": 0.2},
            {"kind": "synaptic_event", "synapse_id": 8, "offset_ms": 0.4},
        ]
        releases = [
            {
                "event_index": 0, "synapse_id": 7, "release_success": False,
                "released_quantity": 0.0, "ampa_state_increment": 0.0,
                "nmda_state_increment": 0.0, "inhibitory_state_increment": 0.0,
                "release_probability": 0.3, "rng_preview_value": 0.8,
                "rng_sequence_before": 4.0,
            },
            {
                "event_index": 1, "synapse_id": 8, "release_success": True,
                "released_quantity": 1.0, "ampa_state_increment": 0.2,
                "nmda_state_increment": 0.1, "inhibitory_state_increment": 0.0,
                "release_probability": 0.7, "rng_preview_value": 0.2,
                "rng_sequence_before": 5.0,
            },
        ]
        joined = _canonical_release_rows(scheduled, releases)
        self.assertEqual(len(joined), 2)
        self.assertFalse(joined[0]["release_success"])
        self.assertTrue(joined[1]["release_success"])

    def test_rng_schema_excludes_memorization_keys(self):
        schema = input_view_schema()
        excluded = set(schema["views"]["U_rng"]["excluded_memorization_keys"])
        self.assertTrue({"episode_id", "trajectory_id", "snapshot_id"} <= excluded)
        self.assertFalse(schema["future_state_used"])


class ReleaseMetricTests(unittest.TestCase):
    def test_positive_class_never_detected_has_zero_f1(self):
        targets = np.zeros((4, len(EVENT_KINDS)), dtype=float)
        targets[:2, 0] = 1.0
        probabilities = np.zeros_like(targets)
        rows = pooled_event_metrics(
            probabilities, targets, ["a", "b", "c", "d"],
            model="test", split="test", thresholds=0.5,
        )
        self.assertEqual(rows[0]["support"], 2)
        self.assertEqual(rows[0]["true_positive"], 0)
        self.assertEqual(rows[0]["f1"], 0.0)
        self.assertTrue(math.isfinite(rows[0]["pr_auc"]))

    def test_macro_f1_uses_all_six_classes(self):
        rows = [
            {"event_kind": kind, "f1": float(index) / 5.0}
            for index, kind in enumerate(EVENT_KINDS)
        ]
        summary = macro_event_summary(rows)
        self.assertAlmostEqual(summary["macro_f1_overall"], 0.5)

    def test_episode_bootstrap_does_not_treat_rows_as_independent(self):
        rows = [
            {"episode_id": "a", "value": 0.0},
            {"episode_id": "a", "value": 2.0},
            {"episode_id": "b", "value": 5.0},
        ]
        result = episode_bootstrap(rows, value_key="value", replicates=100, seed=1)
        self.assertEqual(result["episode_count"], 2)
        self.assertAlmostEqual(result["estimate"], 3.0)

    def test_event_bootstrap_resamples_complete_episodes(self):
        target = np.zeros((4, len(EVENT_KINDS)), dtype=float)
        target[[0, 2], 0] = 1.0
        probability = target.copy()
        rows = episode_bootstrap_event_f1(
            probability, target, ["a", "a", "b", "b"],
            replicates=50, seed=2,
        )
        self.assertEqual(rows[0]["episode_count"], 2)
        self.assertEqual(rows[0]["estimate"], 1.0)

    def test_branching_retention_is_not_clipped_at_one(self):
        rows, summary = branching_metrics(
            [
                {
                    "pair_id": "p", "branching_kind": "far", "horizon_ms": 2,
                    "teacher_a": np.asarray([[0.0], [1.0]]),
                    "teacher_b": np.asarray([[0.0], [0.0]]),
                    "prediction_a": np.asarray([[0.0], [2.0]]),
                    "prediction_b": np.asarray([[0.0], [0.0]]),
                    "divergent_event_correct": True,
                }
            ],
            teacher_distance_floor=0.1,
        )
        self.assertGreater(rows[0]["divergence_retention"], 1.0)
        self.assertTrue(rows[0]["over_divergence"])
        self.assertGreater(summary["over_divergence_rate"], 0.0)

    def test_identifiability_variance_reduction(self):
        report = identifiability_summary(
            [
                {
                    "release_regime": "no_release", "scheduled_residual_variance": 2.0,
                    "rng_residual_variance": 0.5, "realized_residual_variance": 0.0,
                    "target_distance": 2.0,
                }
            ]
        )
        self.assertAlmostEqual(report["overall"]["rng_variance_reduction_fraction"], 0.75)
        self.assertAlmostEqual(report["overall"]["realized_variance_reduction_fraction"], 1.0)

    def test_conditional_go_is_distinct_from_go(self):
        decision = release_flowmap_decision(
            {
                "b3_realized_beats_b1_consistently": True,
                "realized_reduces_scheduled_ambiguity": True,
                "stable_across_seeds": True,
            }
        )
        self.assertEqual(decision["decision"], "CONDITIONAL_GO")


if __name__ == "__main__":
    unittest.main()
