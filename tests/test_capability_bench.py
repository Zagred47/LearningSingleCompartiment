import numpy as np
import pytest
import torch

from hay_single_compartment import (
    CAPABILITY_ARCHITECTURES,
    build_capability_model,
    fast_slow_metrics,
    generate_fast_slow_sequences,
    width_for_budget,
)


def test_fast_slow_generator_is_deterministic_and_causal():
    first = generate_fast_slow_sequences(64, 256, "easy", 7)
    second = generate_fast_slow_sequences(64, 256, "easy", 7)
    for key in ("inputs", "targets", "triggers", "spikes", "eligible"):
        np.testing.assert_array_equal(first[key], second[key])
    assert first["inputs"].shape == (64, 256, 3)
    assert first["targets"].shape == (64, 256, 3)
    assert int(first["spikes"].sum()) > 0
    assert np.all(first["spikes"] <= first["triggers"])
    assert np.all(first["spikes"] <= first["eligible"])


def test_fast_slow_perfect_prediction_has_perfect_event_metrics():
    reference = generate_fast_slow_sequences(64, 256, "easy", 13)
    target = reference["targets"]
    metrics = fast_slow_metrics(target, reference, target.reshape(-1, 3).std(0))
    assert metrics["spike_f1"] == pytest.approx(1.0)
    assert metrics["spike_recall"] == pytest.approx(1.0)
    assert metrics["peak_amplitude_mae"] == pytest.approx(0.0)
    assert metrics["event_voltage_rmse"] == pytest.approx(0.0)


@pytest.mark.parametrize("name", ["mlp", "rnn", "gru", "lstm", "tcn", "transformer", "conv_gru", "conv_lstm"])
def test_built_in_capability_models_are_causal_and_shape_preserving(name):
    torch.manual_seed(2)
    model = build_capability_model(name, 3, 3, 8)
    inputs = torch.randn(2, 24, 3)
    changed = inputs.clone()
    changed[:, 16:] = torch.randn_like(changed[:, 16:]) * 5
    original = model(inputs)
    altered = model(changed)
    assert original.shape == (2, 24, 3)
    torch.testing.assert_close(original[:, :16], altered[:, :16], atol=2e-5, rtol=2e-5)


def test_width_selection_returns_nearest_available_model():
    for name in CAPABILITY_ARCHITECTURES:
        width, parameters = width_for_budget(name, 8_000)
        assert width > 0 and parameters > 0
