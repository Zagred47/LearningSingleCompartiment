import h5py
import numpy as np
import pytest

from hay_single_compartment import (
    MICRO_REGIME_NAMES,
    MICRO_STATE_NAMES,
    BalancedSpatialSpikeDrive,
    FourCompartmentHay,
    MicroDatasetConfig,
    MicroDriveConfig,
    build_micro_synapse_metadata,
    generate_micro_dataset,
    validate_micro_dataset,
)


def tiny_config() -> MicroDatasetConfig:
    return MicroDatasetConfig(
        duration_ms=12.0,
        warmup_ms=4.0,
        train_trajectories=1,
        validation_trajectories=1,
        test_trajectories=1,
        drive=MicroDriveConfig(
            excitatory_synapses=6,
            inhibitory_synapses=3,
            min_regime_ms=0.1,
            max_regime_ms=0.2,
            smoothing_tau_min_ms=0.2,
            smoothing_tau_max_ms=0.4,
        ),
    )


def test_micro_teacher_is_spatial_state_complete_and_has_slow_gates():
    teacher = FourCompartmentHay()
    assert len(MICRO_STATE_NAMES) == 61
    state = teacher.initial_state()
    metadata = build_micro_synapse_metadata()
    inputs = np.zeros((50, len(metadata)), dtype=np.uint8)
    inputs[1, 0] = 1
    result = teacher.simulate(inputs, metadata, dt_ms=0.1, internal_dt_ms=0.025)
    assert result["states"].shape == (51, 61)
    assert result["event_counts"].shape == (50, 6)
    assert np.isfinite(result["states"]).all()
    assert teacher.axial_uS[("soma", "basal")] > 0.0
    _, tau = teacher.kinetics.gate_targets(-70.0, 1.0e-4)
    assert tau[3] == pytest.approx(2144.9, rel=2.0e-3)
    ampa_rise = teacher.index["basal.ampa_rise"]
    ampa_decay = teacher.index["basal.ampa_decay"]
    assert result["states"][2, ampa_decay] > result["states"][2, ampa_rise] > 0.0


def test_synapses_are_length_weighted_and_spikes_are_binary_reproducible():
    config = MicroDatasetConfig()
    metadata = build_micro_synapse_metadata(config)
    assert len(metadata) == config.drive.excitatory_synapses + config.drive.inhibitory_synapses
    excitatory_counts = {
        region: sum(row["kind"] == "excitatory" and row["region"] == region for row in metadata)
        for region in ("basal", "trunk", "tuft")
    }
    assert excitatory_counts == {"basal": 4, "trunk": 5, "tuft": 7}
    drive = BalancedSpatialSpikeDrive(config)
    first, regimes_a, _ = drive.sample(20_000, 0.1, 9)
    second, regimes_b, _ = drive.sample(20_000, 0.1, 9)
    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(regimes_a, regimes_b)
    assert np.isin(first, (0, 1)).all()
    assert set(regimes_a.tolist()) == set(range(len(MICRO_REGIME_NAMES)))


def test_micro_dataset_cache_roundtrip(tmp_path):
    path = tmp_path / "micro_tiny.h5"
    config = tiny_config()
    generated = generate_micro_dataset(path, config, progress=True)
    assert not generated["cache_hit"]
    assert validate_micro_dataset(path)["valid"]
    cached = generate_micro_dataset(path, config, progress=True)
    assert cached["cache_hit"]
    with h5py.File(path, "r") as handle:
        assert handle.attrs["external_current_injection"] == np.bool_(False)
        assert handle["train/states"].shape == (1, 121, 61)
        assert handle["test/inputs"].shape == (1, 120, 9)
        assert handle["test/burnin_inputs"].shape == (1, 40, 9)
        assert handle["test/burnin_states"].shape == (1, 41, 61)
        assert np.isin(handle["train/inputs"][...], (0, 1)).all()
