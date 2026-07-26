import numpy as np

from hay_single_compartment import INPUT_NAMES, STATE_NAMES, RandomDrive, SingleCompartmentHay


def test_random_drive_is_reproducible_and_nonnegative_for_events():
    generator = RandomDrive()
    first, first_regimes = generator.sample(200, 0.1, seed=7)
    second, second_regimes = generator.sample(200, 0.1, seed=7)
    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(first_regimes, second_regimes)
    assert first.shape == (200, len(INPUT_NAMES))
    assert np.all(first[:, 1:] >= 0.0)


def test_simulator_retains_full_bounded_state():
    inputs, _ = RandomDrive().sample(300, 0.1, seed=11)
    result = SingleCompartmentHay().simulate(inputs, 0.1, 0.025)
    states = result["states"]
    assert states.shape == (301, len(STATE_NAMES))
    assert np.isfinite(states).all()
    assert np.all((states[:, 2:14] >= 0.0) & (states[:, 2:14] <= 1.0))
    assert np.all(states[:, 1] >= 0.0)
    assert np.all(states[:, 14:17] >= 0.0)


def test_a_strong_pulse_activates_fast_nonlinearity():
    inputs = np.zeros((200, len(INPUT_NAMES)), dtype=float)
    inputs[20:120, 0] = 1.0
    result = SingleCompartmentHay().simulate(inputs, 0.1, 0.025)
    assert result["states"][:, 0].max() > -20.0
    assert result["spikes"].sum() >= 1
