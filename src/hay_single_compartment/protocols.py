"""Reproducible random current and synaptic-event protocols."""

from __future__ import annotations

import numpy as np

from .config import ProtocolConfig
from .simulator import INPUT_NAMES


REGIME_NAMES = ("quiet", "balanced", "excitatory", "inhibitory", "burst")
REGIME_RATES_HZ = np.asarray(
    [
        [20.0, 5.0, 15.0],
        [180.0, 70.0, 140.0],
        [420.0, 150.0, 70.0],
        [100.0, 35.0, 360.0],
        [750.0, 300.0, 180.0],
    ],
    dtype=np.float64,
)


class RandomDrive:
    """Generate diverse drive without leaking seeds across data splits."""

    def __init__(self, config: ProtocolConfig | None = None) -> None:
        self.config = config or ProtocolConfig()

    def sample(self, steps: int, dt_ms: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
        if steps < 1 or dt_ms <= 0.0:
            raise ValueError("steps and dt_ms must be positive")
        rng = np.random.default_rng(seed)
        result = np.zeros((steps, len(INPUT_NAMES)), dtype=np.float64)
        regimes = np.zeros(steps, dtype=np.int8)
        regime = int(rng.integers(len(REGIME_NAMES)))
        current = self.config.ou_mean_na
        ou_decay = np.exp(-dt_ms / self.config.ou_tau_ms)
        ou_scale = self.config.ou_sigma_na * np.sqrt(1.0 - ou_decay**2)
        switch_probability = min(1.0, dt_ms / self.config.mean_regime_ms)
        pulse_steps_remaining = 0
        pulse_amplitude = 0.0

        for index in range(steps):
            if rng.random() < switch_probability:
                choices = np.delete(np.arange(len(REGIME_NAMES)), regime)
                regime = int(rng.choice(choices))
            regimes[index] = regime
            regime_bias = (-0.08, 0.05, 0.32, -0.08, 0.62)[regime]
            target = self.config.ou_mean_na + regime_bias
            current = target + (current - target) * ou_decay + ou_scale * rng.normal()
            pulse_rate_hz = (2.0, 10.0, 35.0, 2.0, 80.0)[regime]
            if pulse_steps_remaining == 0 and rng.random() < pulse_rate_hz * dt_ms / 1000.0:
                pulse_steps_remaining = max(1, int(round(rng.uniform(6.0, 14.0) / dt_ms)))
                pulse_amplitude = float(rng.uniform(0.80, 1.20))
            pulse = pulse_amplitude if pulse_steps_remaining > 0 else 0.0
            pulse_steps_remaining = max(0, pulse_steps_remaining - 1)
            result[index, 0] = np.clip(
                current + pulse, self.config.current_min_na, self.config.current_max_na
            )
            result[index, 1:] = rng.poisson(REGIME_RATES_HZ[regime] * dt_ms / 1000.0)
        return result, regimes
