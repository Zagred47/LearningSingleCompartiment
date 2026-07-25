import unittest
import math

from src.hayflow_data import (
    CAUSAL_OBSERVATION_PHASE,
    CausalReleaseOutcome,
    InputAction,
    ProtocolTrajectory,
    TargetedRecipe,
    build_balanced_episode_plan,
    append_specialized_test_episodes,
    build_input_views,
    select_adaptive_recipe_brackets,
    summarize_independent_support,
)
from src.hayflow_teacher.event_extractor import (
    EventDefinition,
    annotate_backpropagation,
    extract_events,
)
from src.hayflow_teacher import (
    TargetedDiagnosticDatasetSession,
    assess_causal_bap,
    select_causal_bap_assist,
)
from src.hayflow_teacher.causal_release import CausalReleaseRecorder


def outcome(success=True):
    increment = 2.0 if success else 0.0
    return CausalReleaseOutcome(
        transition_id=3,
        event_index=0,
        synapse_id=7,
        scheduled_time_ms=100.25,
        offset_ms=0.25,
        synapse_type="ProbAMPANMDA2",
        functional_type="excitatory_AMPA+NMDA",
        weight=1.0,
        random123_seed=11,
        random123_stream_id=7,
        random123_global_index=0,
        rng_sequence_before=0.0,
        rng_sequence_after=1.0,
        rng_distribution="negexp(1)",
        rng_preview_value=0.2,
        release_probability=0.4,
        release_success=success,
        released_quantity=1.0 if success else 0.0,
        ampa_state_increment=increment,
        nmda_state_increment=increment,
        inhibitory_state_increment=0.0,
        pre_synapse_state={"point_process.A_AMPA": 0.0},
        post_synapse_state={"point_process.A_AMPA": increment},
        observation_phase=CAUSAL_OBSERVATION_PHASE,
    )


class TargetedReleaseContractTest(unittest.TestCase):
    def test_hidden_mod_normalization_factor_is_reconstructed_from_range_taus(self):
        factor = CausalReleaseRecorder._dual_exp_factor(0.2, 1.7)
        time_to_peak = 0.2 * 1.7 / 1.5 * math.log(1.7 / 0.2)
        expected = 1.0 / (
            -math.exp(-time_to_peak / 0.2) + math.exp(-time_to_peak / 1.7)
        )
        self.assertAlmostEqual(factor, expected)

    def test_shadow_frontend_matches_canonical_excitatory_net_receive_equations(self):
        shadow = {
            "class_name": "ProbAMPANMDA2",
            "time_ms": 5.0,
            "point": {
                "A_AMPA": 2.0,
                "B_AMPA": 3.0,
                "A_NMDA": 4.0,
                "B_NMDA": 5.0,
                "factor_AMPA": 2.5,
                "factor_NMDA": 4.0,
                "tau_r_AMPA": 0.2,
                "tau_d_AMPA": 1.7,
                "tau_r_NMDA": 0.29,
                "tau_d_NMDA": 43.0,
                "Use": 0.4,
                "Dep": 100.0,
                "Fac": 10.0,
                "u0": 0.0,
            },
            "weights": [1.5, 0.0, 0.0, 0.7, 0.0, 0.2, 4.0],
        }
        CausalReleaseRecorder._advance_point(shadow, 6.0)
        self.assertAlmostEqual(shadow["point"]["A_AMPA"], 2.0 * math.exp(-5.0))
        probability = CausalReleaseRecorder._apply_short_term_plasticity(
            shadow, 6.0
        )
        expected_u = 0.2 * math.exp(-0.2) + 0.4 * (
            1.0 - 0.2 * math.exp(-0.2)
        )
        expected_pv = 1.0 - 0.3 * math.exp(-0.02)
        self.assertAlmostEqual(probability, expected_u * expected_pv)
        self.assertEqual(shadow["weights"][1:3], [1.5, 1.5])
        ampa, nmda, inhibitory = CausalReleaseRecorder._apply_release(shadow, True)
        self.assertAlmostEqual(ampa, 3.75)
        self.assertAlmostEqual(nmda, 6.0)
        self.assertEqual(inhibitory, 0.0)

    def test_shadow_frontend_matches_canonical_inhibitory_release_increment(self):
        shadow = {
            "class_name": "ProbUDFsyn2",
            "point": {"A": 0.0, "B": 0.0, "factor": 2.25},
            "weights": [0.8, 1.0, 0.0, 0.0, 0.0],
        }
        ampa, nmda, inhibitory = CausalReleaseRecorder._apply_release(shadow, True)
        self.assertEqual((ampa, nmda), (0.0, 0.0))
        self.assertAlmostEqual(inhibitory, 1.8)
        self.assertAlmostEqual(shadow["point"]["A"], 1.8)
        self.assertAlmostEqual(shadow["point"]["B"], 1.8)

    def test_canonical_zero_dep_and_fac_use_the_positive_time_limit(self):
        shadow = {
            "class_name": "ProbUDFsyn2",
            "point": {"Use": 1.0, "Dep": 0.0, "Fac": 0.0},
            "weights": [1.0, 0.0, 0.0, 0.0, 10.0],
        }
        first = CausalReleaseRecorder._apply_short_term_plasticity(shadow, 10.2)
        self.assertEqual(first, 1.0)
        self.assertEqual(shadow["weights"][1:], [0.0, 1.0, 1.0, 10.2])
        second = CausalReleaseRecorder._apply_short_term_plasticity(shadow, 11.2)
        self.assertEqual(second, 1.0)
        self.assertEqual(shadow["weights"][1:], [0.0, 1.0, 1.0, 11.2])

    def test_zero_dep_rejects_same_synapse_at_identical_time(self):
        shadow = {
            "class_name": "ProbUDFsyn2",
            "point": {"Use": 1.0, "Dep": 0.0, "Fac": 0.0},
            "weights": [1.0, 1.0, 0.0, 0.0, 10.0],
        }
        with self.assertRaisesRegex(RuntimeError, "strictly positive"):
            CausalReleaseRecorder._apply_short_term_plasticity(shadow, 10.0)

    def test_release_gate_prefers_selected_path_over_flipped_counterfactual(self):
        observed = {"A": 0.1012498441, "B": 1.0591680520}
        predicted = {"A": 0.0925417357, "B": 1.0590840816}
        contribution = {"A": 0.0925417357, "B": 1.0590840816}
        selected, flipped = CausalReleaseRecorder._counterfactual_errors(
            observed, predicted, contribution, True
        )
        self.assertLess(selected, flipped)

    def test_release_gate_detects_wrong_selected_outcome(self):
        observed = {"A": 0.0, "B": 0.0}
        predicted = {"A": 0.5, "B": 1.0}
        contribution = {"A": 0.5, "B": 1.0}
        selected, flipped = CausalReleaseRecorder._counterfactual_errors(
            observed, predicted, contribution, True
        )
        self.assertGreater(selected, flipped)

    def test_release_gate_rejects_non_finite_boundary_state(self):
        with self.assertRaisesRegex(RuntimeError, "non-finite"):
            CausalReleaseRecorder._counterfactual_errors(
                {"A": float("nan")}, {"A": 0.0}, {"A": 1.0}, True
            )

    def test_three_input_views_keep_realized_release_causal(self):
        actions = [
            {
                "kind": "synaptic_event",
                "offset_ms": 0.25,
                "synapse_id": 7,
                "weight_multiplier": 1.0,
                "metadata": {},
            }
        ]
        views = build_input_views(actions, [outcome(True)])
        self.assertNotIn("release_success", views["U_scheduled"][0])
        self.assertIn("random123_stream_id", views["U_rng"][0])
        self.assertNotIn("release_success", views["U_rng"][0])
        self.assertTrue(views["U_realized"][0]["release_success"])
        self.assertEqual(
            outcome(True).source, "exact_causal_replay_of_original_net_receive"
        )

        failed = build_input_views(actions, [outcome(False)])
        self.assertEqual(failed["U_realized"], [])

    def test_release_contract_rejects_future_or_inconsistent_labels(self):
        valid = outcome(True)
        valid.validate()
        invalid = CausalReleaseOutcome(
            **{
                **valid.__dict__,
                "source": "inferred_from_t_plus_1_future",
            }
        )
        with self.assertRaisesRegex(ValueError, "future"):
            invalid.validate()

    def test_nmda_spike_can_be_separated_from_plateau_by_duration(self):
        definition = EventDefinition(
            "nmda_spike",
            "tuft",
            1,
            "tuft",
            -40.0,
            -50.0,
            min_duration_ms=1.0,
            maximum_event_duration_ms=9.0,
        )
        time = list(range(13))
        short = [-70, -35, -35, -55] + [-70] * 9
        long = [-70, -35] + [-35] * 10 + [-55]
        self.assertEqual(len(extract_events(time, {"tuft": short}, [definition])), 1)
        self.assertEqual(len(extract_events(time, {"tuft": long}, [definition])), 0)

    def test_nmda_plateau_is_a_hierarchical_nmda_spike(self):
        spike = EventDefinition(
            "nmda_spike", "tuft", 1, "tuft", -40.0, -50.0, min_duration_ms=1.0
        )
        plateau = EventDefinition(
            "nmda_plateau", "tuft", 1, "tuft", -40.0, -50.0, min_duration_ms=10.0
        )
        time = list(range(13))
        long = [-70, -35] + [-35] * 10 + [-55]
        kinds = {
            row["kind"]
            for row in extract_events(time, {"tuft": long}, [spike, plateau])
        }
        self.assertEqual(kinds, {"nmda_spike", "nmda_plateau"})

    def test_plateau_from_a_distinct_probe_gets_parent_spike_label(self):
        plateau = {
            "kind": "nmda_plateau",
            "segment_id": 9,
            "onset_ms": 3.0,
            "offset_ms": 20.0,
            "parameters": {"kind": "nmda_plateau", "min_duration_ms": 10.0},
        }
        rows = TargetedDiagnosticDatasetSession._enforce_nmda_event_hierarchy(
            [plateau]
        )
        self.assertEqual({row["kind"] for row in rows}, {"nmda_spike", "nmda_plateau"})
        parent = next(row for row in rows if row["kind"] == "nmda_spike")
        self.assertEqual(parent["derived_from_event_kind"], "nmda_plateau")

    def test_recovery_uses_lazy_pv_available_at_next_event(self):
        available = TargetedDiagnosticDatasetSession._available_pv_at_event(
            stored_pv=0.0, elapsed_ms=20.0, depression_tau_ms=0.0
        )
        self.assertEqual(available, 1.0)
        partial = TargetedDiagnosticDatasetSession._available_pv_at_event(
            stored_pv=0.0, elapsed_ms=10.0, depression_tau_ms=100.0
        )
        self.assertAlmostEqual(partial, 1.0 - math.exp(-0.1))

    def test_bap_annotation_requires_outward_temporal_order(self):
        time = [0.0, 0.5, 1.0, 1.5, 2.0]
        event = {
            "kind": "backpropagating_ap",
            "onset_ms": 1.0,
            "linked_delay_ms": 0.5,
        }
        traces = {
            "soma": [-70, 10, -30, -60, -70],
            "trunk": [-70, -50, 0, -40, -70],
            "nexus": [-70, -70, -50, -10, -70],
            "tuft": [-70, -70, -60, -40, -10],
            "basal": [-70, -50, -10, -50, -70],
        }
        rows = annotate_backpropagation(
            time,
            traces,
            [event],
            regional_distances_um={
                "soma": 0,
                "trunk": 400,
                "nexus": 700,
                "tuft": 900,
                "basal": 150,
            },
        )
        self.assertEqual(rows[0]["origin"], "soma")
        self.assertEqual(rows[0]["maximum_distance_um"], 900)

    def test_causal_bap_accepts_soma_then_trunk_without_distal_preemption(self):
        time = [0.0, 0.5, 1.0, 1.5, 2.0]
        events = [
            {"kind": "axonal_spike", "onset_ms": 0.5},
            {"kind": "somatic_spike", "onset_ms": 0.5},
            {
                "kind": "backpropagating_ap",
                "onset_ms": 1.0,
                "linked_delay_ms": 0.5,
            },
        ]
        traces = {
            "nexus": [-70, -60, -50, -10, -60],
            "tuft": [-70, -70, -60, -50, -10],
            "voltage_event_probe_mv": [-70, -65, -50, -15, -60],
        }
        report = assess_causal_bap(time, traces, events)
        self.assertTrue(report["valid"])
        self.assertEqual(report["accepted_bap_count"], 1)

    def test_causal_bap_rejects_distal_crossing_before_trunk(self):
        time = [0.0, 0.5, 1.0, 1.5]
        events = [
            {"kind": "axonal_spike", "onset_ms": 0.5},
            {"kind": "somatic_spike", "onset_ms": 0.5},
            {
                "kind": "backpropagating_ap",
                "onset_ms": 1.0,
                "linked_delay_ms": 0.5,
            },
        ]
        traces = {"nexus": [-70, -10, -5, -60]}
        report = assess_causal_bap(time, traces, events)
        self.assertFalse(report["valid"])
        self.assertIn(
            "distal_threshold_crossing_precedes_trunk",
            report["candidates"][0]["rejection_reasons"],
        )

    def test_causal_bap_rejects_regenerative_assist_counterfactual(self):
        time = [0.0, 0.5, 1.0, 1.5]
        events = [
            {"kind": "axonal_spike", "onset_ms": 0.5},
            {"kind": "somatic_spike", "onset_ms": 0.5},
            {
                "kind": "backpropagating_ap",
                "onset_ms": 1.0,
                "linked_delay_ms": 0.5,
            },
        ]
        report = assess_causal_bap(
            time,
            {},
            events,
            assist_only_events=[{"kind": "nmda_plateau"}],
            assist_only_peak_voltage_mv=-10.0,
            require_subthreshold_assist=True,
        )
        self.assertFalse(report["valid"])
        self.assertFalse(report["assist_only_subthreshold"])
        self.assertFalse(report["assist_only_peak_below_threshold"])
        self.assertIn(
            "assist_only_is_not_subthreshold",
            report["candidates"][0]["rejection_reasons"],
        )

    def test_bap_assist_selection_uses_strongest_robust_subthreshold_arm(self):
        trials = []
        for candidate_id, peaks, kinds in (
            ("weak", (-45.0, -44.0), ()),
            ("strong", (-24.0, -23.0), ()),
            ("too_close", (-21.0, -21.5), ()),
            ("regenerative", (-30.0, -29.0), ("nmda_spike",)),
        ):
            for seed, peak in zip((1, 2), peaks):
                trials.append(
                    {
                        "candidate_id": candidate_id,
                        "family": "targeted_calcium",
                        "pair_with_somatic_spike": False,
                        "seed": seed,
                        "event_kinds": list(kinds),
                        "event_probe_peak_voltage_mv": peak,
                    }
                )
        report = select_causal_bap_assist(
            trials,
            (1, 2),
            threshold_mv=-20.0,
            subthreshold_margin_mv=2.0,
        )
        self.assertTrue(report["valid"])
        self.assertEqual(report["selected_candidate_id"], "strong")
        rejected = {
            row["candidate_id"]: row["rejection_reasons"]
            for row in report["candidates"]
        }
        self.assertIn(
            "assist_peak_exceeds_subthreshold_ceiling", rejected["too_close"]
        )
        self.assertIn(
            "event_detected_in_assist_only_arm", rejected["regenerative"]
        )


class TargetedProtocolPlannerTest(unittest.TestCase):
    def test_v1_1_plan_hash_binds_snapshot_and_branch_metadata(self):
        first = ProtocolTrajectory(
            trajectory_id="episode",
            category="dendritic_events",
            protocol="ca",
            seed=1,
            duration_ms=2,
            split="train",
            snapshot_source="train-snapshot-a",
            metadata={"snapshot_id": "train-snapshot-a", "branch_id": "a"},
        )
        second = ProtocolTrajectory(
            trajectory_id="episode",
            category="dendritic_events",
            protocol="ca",
            seed=1,
            duration_ms=2,
            split="train",
            snapshot_source="train-snapshot-b",
            metadata={"snapshot_id": "train-snapshot-b", "branch_id": "b"},
        )
        self.assertNotEqual(
            TargetedDiagnosticDatasetSession._protocol_plan_sha256([first]),
            TargetedDiagnosticDatasetSession._protocol_plan_sha256([second]),
        )

    def test_adaptive_brackets_require_seed_robust_positive_and_negative(self):
        trials = []
        for candidate, stimulus, kinds in (
            ("below", 0.9, []),
            ("above", 1.1, ["calcium_spike"]),
        ):
            for seed in (1, 2):
                trials.append(
                    {
                        "candidate_id": candidate,
                        "stimulus_scalar": stimulus,
                        "seed": seed,
                        "event_kinds": kinds,
                    }
                )
        report = select_adaptive_recipe_brackets(trials, required_seed_count=2)
        selection = report["selections"]["calcium_spike"]
        self.assertEqual(selection["positive_candidate_id"], "above")
        self.assertEqual(selection["negative_candidate_id"], "below")
        self.assertFalse(report["valid"])  # other required classes are absent

    def test_episode_planner_counts_independent_episodes_not_transitions(self):
        action = InputAction(
            "somatic_current", 0.05, duration_ms=0.9, amplitude_na=1.0
        )
        all_classes = (
            "axonal_spike",
            "somatic_spike",
            "backpropagating_ap",
            "calcium_spike",
            "nmda_spike",
            "nmda_plateau",
        )
        recipes = [
            TargetedRecipe(
                "positive",
                "dendritic",
                "positive",
                2,
                {0: (action,)},
                positive_for=all_classes,
            ),
            TargetedRecipe(
                "negative",
                "dendritic",
                "hard_negative",
                2,
                {0: (action,)},
                hard_negative_for=all_classes,
            ),
        ]
        plans, rows = build_balanced_episode_plan(
            recipes,
            positive_targets={"train": 2},
            hard_negative_targets={"train": 3},
        )
        self.assertEqual(len(plans), 5)
        support = summarize_independent_support(rows)
        self.assertEqual(
            support["calcium_spike"]["train"]["positive_episode_count"], 2
        )
        self.assertEqual(
            support["calcium_spike"]["train"][
                "hard_negative_episode_count"
            ],
            3,
        )

    def test_specialized_pairs_share_only_the_intended_snapshot(self):
        synaptic = InputAction(
            "synaptic_event", 0.25, synapse_id=4
        )
        classes = (
            "axonal_spike",
            "somatic_spike",
            "backpropagating_ap",
            "calcium_spike",
            "nmda_spike",
            "nmda_plateau",
        )
        positive = TargetedRecipe(
            "positive",
            "dendritic",
            "positive",
            4,
            {1: (synaptic,)},
            positive_for=classes,
            branch_id="train-branch",
        )
        negative = TargetedRecipe(
            "negative",
            "dendritic",
            "negative",
            4,
            {1: (synaptic,)},
            hard_negative_for=classes,
            branch_id="train-branch",
        )
        heldout = TargetedRecipe(
            "heldout",
            "dendritic",
            "heldout",
            4,
            {1: (synaptic,)},
            hard_negative_for=classes,
            branch_id="heldout-branch",
            metadata={"train_eligible": False},
        )
        recovery = TargetedRecipe(
            "recovery",
            "dendritic",
            "recovery",
            6,
            {1: (synaptic,), 3: (synaptic,)},
            positive_for=classes,
            branch_id="train-branch",
            recovery_probe_delay_ms=2.0,
            metadata={
                "train_eligible": False,
                "recovery_probe": True,
                "pilot_validated": True,
            },
        )
        plans, rows = build_balanced_episode_plan(
            [positive, negative, heldout, recovery],
            positive_targets={"train": 1, "validation": 1, "deterministic_test": 1},
            hard_negative_targets={"train": 1, "validation": 1, "deterministic_test": 1},
        )
        plans, rows = append_specialized_test_episodes(
            plans, rows, [positive, negative, heldout, recovery]
        )
        near = [row for row in plans if row.split == "branching_near_test"]
        self.assertEqual(len({row.seed for row in near}), 1)
        self.assertEqual(
            len({row.metadata["snapshot_id"] for row in near}), 1
        )
        release = [
            row for row in plans if row.split == "release_identifiability_test"
        ]
        self.assertEqual(len({row.seed for row in release}), 2)
        self.assertEqual(
            len({row.metadata["snapshot_id"] for row in release}), 1
        )
        train_branches = {
            row.metadata["branch_id"] for row in plans if row.split == "train"
        }
        heldout_branches = {
            row.metadata["branch_id"]
            for row in plans
            if row.split == "held_out_branch_test"
        }
        self.assertTrue(train_branches.isdisjoint(heldout_branches))


if __name__ == "__main__":
    unittest.main()
