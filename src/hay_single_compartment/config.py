"""Configuration objects for the reduced Hay-inspired compartment."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict


@dataclass(frozen=True)
class MembraneConfig:
    """Biophysical parameters in nF, mV, uS, nA, ms, and mM."""

    capacitance_nf: float = 0.20
    e_leak_mv: float = -76.0
    e_na_mv: float = 55.0
    e_k_mv: float = -90.0
    e_h_mv: float = -45.0
    e_ca_mv: float = 120.0
    e_exc_mv: float = 0.0
    e_gaba_mv: float = -80.0

    g_leak_us: float = 0.010
    g_na_us: float = 2.80
    g_nap_us: float = 0.012
    g_kdr_us: float = 0.65
    g_kv3_us: float = 1.10
    g_m_us: float = 0.020
    g_h_us: float = 0.008
    g_ca_lva_us: float = 0.020
    g_ca_hva_us: float = 0.018
    g_sk_us: float = 0.050

    tau_ampa_ms: float = 2.0
    tau_nmda_ms: float = 70.0
    tau_gaba_ms: float = 8.0
    ampa_jump_us: float = 0.0018
    nmda_jump_us: float = 0.0010
    gaba_jump_us: float = 0.0022

    ca_rest_mm: float = 1.0e-4
    ca_tau_ms: float = 80.0
    ca_influx_mm_per_na_ms: float = 3.0e-4
    spike_threshold_mv: float = -20.0


@dataclass(frozen=True)
class ProtocolConfig:
    """Statistics of the random, piecewise-stationary drive."""

    ou_mean_na: float = 0.12
    ou_sigma_na: float = 0.18
    ou_tau_ms: float = 12.0
    current_min_na: float = -0.25
    current_max_na: float = 1.50
    mean_regime_ms: float = 35.0


@dataclass(frozen=True)
class SimulationConfig:
    """Complete reproducible experiment configuration."""

    dt_ms: float = 0.10
    internal_dt_ms: float = 0.025
    duration_ms: float = 500.0
    warmup_ms: float = 100.0
    seed: int = 2026
    train_trajectories: int = 12
    validation_trajectories: int = 3
    test_trajectories: int = 3
    membrane: MembraneConfig = field(default_factory=MembraneConfig)
    protocol: ProtocolConfig = field(default_factory=ProtocolConfig)

    def validate(self) -> None:
        if self.dt_ms <= 0.0 or self.internal_dt_ms <= 0.0:
            raise ValueError("time steps must be positive")
        ratio = self.dt_ms / self.internal_dt_ms
        if abs(ratio - round(ratio)) > 1e-12:
            raise ValueError("dt_ms must be an integer multiple of internal_dt_ms")
        if self.duration_ms <= 0.0 or self.warmup_ms < 0.0:
            raise ValueError("duration must be positive and warmup non-negative")
        if min(
            self.train_trajectories,
            self.validation_trajectories,
            self.test_trajectories,
        ) < 1:
            raise ValueError("every split needs at least one trajectory")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: Dict[str, Any]) -> "SimulationConfig":
        values = dict(values)
        values["membrane"] = MembraneConfig(**values.get("membrane", {}))
        values["protocol"] = ProtocolConfig(**values.get("protocol", {}))
        config = cls(**values)
        config.validate()
        return config
