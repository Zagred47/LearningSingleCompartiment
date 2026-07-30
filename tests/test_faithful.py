import h5py
import numpy as np
import pytest
import torch

from hay_single_compartment import (
    FAITHFUL_INPUT_NAMES,
    FAITHFUL_STATE_NAMES,
    BalancedFaithfulDrive,
    FaithfulHaySoma,
    FaithfulProtocolConfig,
    FaithfulSimulationConfig,
    generate_faithful_dataset,
    validate_faithful_dataset,
)
from hay_single_compartment.models import build_model


def test_faithful_hay_soma_retains_original_slow_gate_and_bounded_state():
    simulator = FaithfulHaySoma()
    _, tau = simulator.gate_targets(-70.0, 1.0e-4)
    assert tau[3] == pytest.approx(2144.9, rel=2.0e-3)  # h_Nap_Et2
    assert tau[7] == pytest.approx(395.3, rel=2.0e-3)   # h_K_Pst
    assert tau[13] == pytest.approx(448.7, rel=2.0e-3) # h_Ca_HVA

    inputs, _ = BalancedFaithfulDrive().sample(300, 0.1, seed=11)
    result = simulator.simulate(inputs, 0.1, 0.025)
    assert result["states"].shape == (301, len(FAITHFUL_STATE_NAMES))
    assert np.isfinite(result["states"]).all()
    assert np.all((result["states"][:, 2:17] >= 0.0) & (result["states"][:, 2:17] <= 1.0))
    assert np.all(result["states"][:, 1] >= simulator.config.ca_min_mM)


def test_balanced_drive_contains_every_regime_and_is_reproducible():
    drive = BalancedFaithfulDrive()
    first, first_regimes = drive.sample(4000, 0.1, seed=7)
    second, second_regimes = drive.sample(4000, 0.1, seed=7)
    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(first_regimes, second_regimes)
    assert first.shape == (4000, len(FAITHFUL_INPUT_NAMES))
    assert set(first_regimes.tolist()) == set(range(5))


def test_faithful_dataset_cache_roundtrip(tmp_path, capsys):
    path = tmp_path / "faithful_tiny.h5"
    config = FaithfulSimulationConfig(
        duration_ms=2.0,
        warmup_ms=1.0,
        train_trajectories=1,
        validation_trajectories=1,
        test_trajectories=1,
        protocol=FaithfulProtocolConfig(min_regime_ms=0.2, max_regime_ms=0.3),
    )
    generated = generate_faithful_dataset(path, config, progress=True)
    assert not generated["cache_hit"]
    assert validate_faithful_dataset(path)["valid"]
    cached = generate_faithful_dataset(path, config, progress=True)
    assert cached["cache_hit"]
    assert "cache hit" in capsys.readouterr().out
    with h5py.File(path, "r") as handle:
        assert handle["train/states"].shape == (1, 21, 20)
        assert handle["test/inputs"].shape == (1, 20, 4)


def test_receptor_composite_supports_faithful_twenty_state_schema():
    torch.manual_seed(12)
    model = build_model(
        "conv_lstm_receptor_gru",
        input_dim=24,
        state_dim=20,
        hidden_dim=8,
        layers=1,
        receptor_hidden_dim=5,
    )
    features = torch.randn(2, 7, 24)
    full = model(features)
    assert full.shape == (2, 7, 20)
    hidden = None
    stepwise = []
    for step in range(features.shape[1]):
        prediction, hidden = model(
            features[:, step : step + 1], hidden=hidden, return_hidden=True
        )
        stepwise.append(prediction)
    torch.testing.assert_close(torch.cat(stepwise, dim=1), full, atol=1e-5, rtol=1e-5)
