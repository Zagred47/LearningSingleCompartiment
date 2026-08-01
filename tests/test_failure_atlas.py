import json

import h5py
import numpy as np
import torch

from hay_single_compartment import (
    FailureAtlasConfig,
    InputOnlyGRU,
    MICRO_STATE_NAMES,
    build_failure_atlas,
    checkpoint_gru_spec,
    load_dataset_context,
    load_prediction_archive,
    teacher_centered_waveform_metrics,
)


def _toy_trajectories(steps=240):
    names = list(MICRO_STATE_NAMES)
    truth = np.zeros((2, steps, len(names)), dtype=np.float32)
    time = np.linspace(0.0, 6.0 * np.pi, steps)
    for state, name in enumerate(names):
        if name.endswith(".v_mV"):
            truth[..., state] = -70.0 + 8.0 * np.sin(time + 0.1 * state)
        elif name.endswith(".ca_i_mM"):
            truth[..., state] = 1e-4 + 2e-5 * np.cos(time + 0.1 * state)
        else:
            truth[..., state] = 0.5 + 0.1 * np.sin(time + 0.1 * state)
    prediction = truth + 0.01
    return truth, prediction, names


def test_failure_atlas_covers_orthogonal_views():
    truth, prediction, names = _toy_trajectories()
    report, tables = build_failure_atlas(
        truth,
        prediction,
        names,
        0.5,
        config=FailureAtlasConfig(recurrence_points=80, horizons_ms=(1.0, 10.0, 100.0)),
    )
    assert report["shape"] == list(truth.shape)
    assert report["summary"]["soma_rmse_mV"] > 0
    assert len(tables["statewise"]) == len(names)
    assert {row["family"] for row in tables["families"]} >= {"voltage", "calcium"}
    assert len(tables["events"]) == 8
    assert report["recurrence"]["teacher"]["recurrence_rate"] > 0
    assert tables["recurrence_teacher"].shape == (80, 80)
    assert tables["takens_teacher"].shape[1] == 3
    assert report["takens"]["delay_steps"] >= 1


def test_checkpoint_spec_accepts_plain_and_wrapped_gru():
    model = InputOnlyGRU(120, len(MICRO_STATE_NAMES), hidden_dim=17, decoder_dim=23)
    plain_spec, _ = checkpoint_gru_spec({"model": model.state_dict()})
    wrapped_spec, wrapped = checkpoint_gru_spec({
        "model_state_dict": {f"baseline.{key}": value for key, value in model.state_dict().items()}
    })
    assert plain_spec == wrapped_spec == {
        "input_dim": 120,
        "state_dim": len(MICRO_STATE_NAMES),
        "hidden_dim": 17,
        "decoder_dim": 23,
    }
    assert "recurrent.weight_hh_l0" in wrapped


def test_prediction_archive_supports_historical_formats(tmp_path):
    truth, prediction, _ = _toy_trajectories(12)
    modern = tmp_path / "modern.npz"
    historical = tmp_path / "historical.npz"
    np.savez(modern, truth=truth, prediction=prediction)
    np.savez(historical, teacher=truth, gru=prediction, cfc=prediction + 1)
    loaded_truth, loaded_prediction, name, events = load_prediction_archive(modern)
    np.testing.assert_array_equal(loaded_truth, truth)
    np.testing.assert_array_equal(loaded_prediction, prediction)
    assert name == "prediction" and events is None
    _, selected, name, _ = load_prediction_archive(historical, "gru")
    np.testing.assert_array_equal(selected, prediction)
    assert name == "gru"


def test_dataset_context_reads_schema_and_resamples(tmp_path):
    path = tmp_path / "micro.h5"
    with h5py.File(path, "w") as handle:
        handle.attrs["schema_version"] = "1.2.0"
        handle.attrs["model"] = "hay_micro_4c"
        handle.attrs["state_names_json"] = json.dumps(list(MICRO_STATE_NAMES))
        handle.attrs["input_names_json"] = json.dumps(["a", "b"])
        handle.attrs["regime_names_json"] = json.dumps(["quiet", "active"])
        handle.attrs["config_json"] = json.dumps({"dt_ms": 0.1})
        group = handle.create_group("test")
        group.create_dataset("inputs", data=np.zeros((2, 20, 2), dtype=np.uint8))
        spikes = np.zeros((2, 20), dtype=np.uint8)
        spikes[:, 3] = 1
        group.create_dataset("spikes", data=spikes)
        group.create_dataset("regimes", data=np.tile(np.arange(20) % 2, (2, 1)))
    context = load_dataset_context(path, 2, 10)
    assert context["schema_version"] == "1.2.0"
    assert context["model"] == "hay_micro_4c"
    assert context["model_dt_ms"] == 0.2
    assert context["spikes"].shape == (2, 10)


def test_teacher_centered_waveform_measures_completely_missed_spike():
    truth = np.full((1, 80), -70.0)
    truth[0, 38:43] = np.asarray([-35.0, -10.0, 35.0, -5.0, -45.0])
    prediction = np.full_like(truth, -68.0)
    report = teacher_centered_waveform_metrics(
        truth, prediction, dt_ms=0.5, before_ms=2.0, after_ms=3.0
    )
    assert report["teacher_spike_windows"] == 1
    assert report["predicted_peak_above_threshold_fraction"] == 0.0
    assert report["peak_amplitude_bias_mV"] < -90.0
