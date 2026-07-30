"""Faithful single-compartment realization of the Hay et al. soma.

The intrinsic equations and parameters are transcribed from ModelDB 139653:
``L5PCbiophys3.hoc`` and the accompanying NMODL mechanisms.  Spatial cable
coupling is intentionally removed: this module represents one isopotential
somatic compartment, not the full reconstructed neuron.

AMPA, NMDA, and GABA-A conductances are explicit external-drive states added
for the learning experiments.  They are not claimed to be part of the
original Hay 2011 mechanism set.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, Mapping

import numpy as np


FAITHFUL_STATE_NAMES = (
    "v_mV",
    "ca_i_mM",
    "m_NaTa_t",
    "h_NaTa_t",
    "m_Nap_Et2",
    "h_Nap_Et2",
    "m_K_Tst",
    "h_K_Tst",
    "m_K_Pst",
    "h_K_Pst",
    "m_SKv3_1",
    "m_Ih",
    "m_Ca_LVAst",
    "h_Ca_LVAst",
    "m_Ca_HVA",
    "h_Ca_HVA",
    "z_SK_E2",
    "g_AMPA_S_cm2",
    "g_NMDA_S_cm2",
    "g_GABAA_S_cm2",
)

FAITHFUL_INPUT_NAMES = (
    "i_inj_uA_cm2",
    "ampa_event_count",
    "nmda_event_count",
    "gaba_event_count",
)

FAITHFUL_CURRENT_NAMES = (
    "i_pas_mA_cm2",
    "i_NaTa_t_mA_cm2",
    "i_Nap_Et2_mA_cm2",
    "i_K_Tst_mA_cm2",
    "i_K_Pst_mA_cm2",
    "i_SKv3_1_mA_cm2",
    "i_Ih_mA_cm2",
    "i_Ca_LVAst_mA_cm2",
    "i_Ca_HVA_mA_cm2",
    "i_SK_E2_mA_cm2",
    "i_AMPA_mA_cm2",
    "i_NMDA_mA_cm2",
    "i_GABAA_mA_cm2",
    "i_membrane_total_mA_cm2",
)


@dataclass(frozen=True)
class FaithfulMembraneConfig:
    """Hay Figure-4 somatic parameters in the original density units."""

    cm_uF_cm2: float = 1.0
    e_pas_mV: float = -90.0
    e_na_mV: float = 50.0
    e_k_mV: float = -85.0
    e_h_mV: float = -45.0
    ca_o_mM: float = 2.0
    celsius: float = 34.0

    g_pas_S_cm2: float = 0.0000338
    g_NaTa_tbar_S_cm2: float = 2.04
    g_Nap_Et2bar_S_cm2: float = 0.00172
    g_K_Tstbar_S_cm2: float = 0.0812
    g_K_Pstbar_S_cm2: float = 0.00223
    g_SKv3_1bar_S_cm2: float = 0.693
    g_Ihbar_S_cm2: float = 0.0002
    g_Ca_LVAstbar_S_cm2: float = 0.00343
    g_Ca_HVAbar_S_cm2: float = 0.000992
    g_SK_E2bar_S_cm2: float = 0.0441

    ca_decay_ms: float = 460.0
    ca_gamma: float = 0.000501
    ca_depth_um: float = 0.1
    ca_min_mM: float = 1.0e-4

    tau_ampa_ms: float = 3.0
    tau_nmda_ms: float = 70.0
    tau_gabaa_ms: float = 8.0
    ampa_jump_S_cm2: float = 9.0e-6
    nmda_jump_S_cm2: float = 5.0e-6
    gabaa_jump_S_cm2: float = 11.0e-6
    e_exc_mV: float = 0.0
    e_gabaa_mV: float = -80.0
    spike_threshold_mV: float = -20.0


@dataclass(frozen=True)
class FaithfulProtocolConfig:
    """Balanced piecewise-stationary current and synaptic drive."""

    ou_mean_uA_cm2: float = 0.60
    ou_sigma_uA_cm2: float = 0.90
    ou_tau_ms: float = 12.0
    current_min_uA_cm2: float = -1.25
    current_max_uA_cm2: float = 7.50
    min_regime_ms: float = 25.0
    max_regime_ms: float = 45.0


@dataclass(frozen=True)
class FaithfulSimulationConfig:
    """Reproducible configuration for the faithful single-compartment data."""

    dt_ms: float = 0.1
    internal_dt_ms: float = 0.025
    duration_ms: float = 2000.0
    warmup_ms: float = 500.0
    seed: int = 27182
    train_trajectories: int = 24
    validation_trajectories: int = 4
    test_trajectories: int = 6
    membrane: FaithfulMembraneConfig = field(default_factory=FaithfulMembraneConfig)
    protocol: FaithfulProtocolConfig = field(default_factory=FaithfulProtocolConfig)

    def validate(self) -> None:
        if self.dt_ms <= 0.0 or self.internal_dt_ms <= 0.0:
            raise ValueError("time steps must be positive")
        ratio = self.dt_ms / self.internal_dt_ms
        if abs(ratio - round(ratio)) > 1e-12:
            raise ValueError("dt_ms must be an integer multiple of internal_dt_ms")
        if self.duration_ms <= 0.0 or self.warmup_ms < 0.0:
            raise ValueError("duration must be positive and warmup non-negative")
        if min(self.train_trajectories, self.validation_trajectories, self.test_trajectories) < 1:
            raise ValueError("every split needs at least one trajectory")

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: Mapping[str, object]) -> "FaithfulSimulationConfig":
        values = dict(values)
        values["membrane"] = FaithfulMembraneConfig(**values.get("membrane", {}))
        values["protocol"] = FaithfulProtocolConfig(**values.get("protocol", {}))
        result = cls(**values)
        result.validate()
        return result


def _safe_rate(numerator: float, denominator: float, limit: float) -> float:
    return limit if abs(denominator) < 1.0e-10 else numerator / denominator


class FaithfulHaySoma:
    """One isopotential soma using the original Hay channel equations."""

    FARADAY = 96485.33212
    GAS_CONSTANT = 8.314462618

    def __init__(self, config: FaithfulMembraneConfig | None = None) -> None:
        self.config = config or FaithfulMembraneConfig()
        self.q10 = 2.3 ** ((self.config.celsius - 21.0) / 10.0)

    def calcium_reversal_mV(self, ca_i_mM: float) -> float:
        ca_i_mM = max(float(ca_i_mM), 1.0e-12)
        temperature_k = self.config.celsius + 273.15
        return float(
            1000.0 * self.GAS_CONSTANT * temperature_k
            / (2.0 * self.FARADAY)
            * np.log(self.config.ca_o_mM / ca_i_mM)
        )

    def gate_targets(self, voltage_mV: float, ca_i_mM: float) -> tuple[np.ndarray, np.ndarray]:
        """Return steady states and taus in FAITHFUL_STATE_NAMES gate order."""

        v = float(voltage_mV)
        q = self.q10

        # NaTa_t.
        x = v + 38.0
        m_alpha = _safe_rate(0.182 * x, 1.0 - np.exp(-x / 6.0), 0.182 * 6.0)
        x = -v - 38.0
        m_beta = _safe_rate(0.124 * x, 1.0 - np.exp(-x / 6.0), 0.124 * 6.0)
        nata_m_inf = m_alpha / (m_alpha + m_beta)
        nata_m_tau = 1.0 / (m_alpha + m_beta) / q
        x = v + 66.0
        h_alpha = _safe_rate(-0.015 * x, 1.0 - np.exp(x / 6.0), 0.015 * 6.0)
        x = -v - 66.0
        h_beta = _safe_rate(-0.015 * x, 1.0 - np.exp(x / 6.0), 0.015 * 6.0)
        nata_h_inf = h_alpha / (h_alpha + h_beta)
        nata_h_tau = 1.0 / (h_alpha + h_beta) / q

        # Nap_Et2 uses the NaTa activation rates but a six-times slower m tau.
        nap_m_inf = 1.0 / (1.0 + np.exp((v + 52.6) / -4.6))
        nap_m_tau = 6.0 / (m_alpha + m_beta) / q
        x = v + 17.0
        nap_h_alpha = _safe_rate(-2.88e-6 * x, 1.0 - np.exp(x / 4.63), 2.88e-6 * 4.63)
        x = v + 64.4
        nap_h_beta = _safe_rate(6.94e-6 * x, 1.0 - np.exp(-x / 2.63), 6.94e-6 * 2.63)
        nap_h_inf = 1.0 / (1.0 + np.exp((v + 48.8) / 10.0))
        nap_h_tau = 1.0 / (nap_h_alpha + nap_h_beta) / q

        # K_Tst and K_Pst include the original -10 mV junction correction.
        shifted = v + 10.0
        kt_m_inf = 1.0 / (1.0 + np.exp(-shifted / 19.0))
        kt_m_tau = (0.34 + 0.92 * np.exp(-((shifted + 71.0) / 59.0) ** 2)) / q
        kt_h_inf = 1.0 / (1.0 + np.exp((shifted + 66.0) / 10.0))
        kt_h_tau = (8.0 + 49.0 * np.exp(-((shifted + 73.0) / 23.0) ** 2)) / q
        kp_m_inf = 1.0 / (1.0 + np.exp(-(shifted + 1.0) / 12.0))
        if shifted < -50.0:
            kp_m_tau = (1.25 + 175.03 * np.exp(0.026 * shifted)) / q
        else:
            kp_m_tau = (1.25 + 13.0 * np.exp(-0.026 * shifted)) / q
        kp_h_inf = 1.0 / (1.0 + np.exp((shifted + 54.0) / 11.0))
        kp_h_tau = (
            360.0
            + (1010.0 + 24.0 * (shifted + 55.0))
            * np.exp(-((shifted + 75.0) / 48.0) ** 2)
        ) / q

        skv_m_inf = 1.0 / (1.0 + np.exp((v - 18.7) / -9.7))
        skv_m_tau = 4.0 / (1.0 + np.exp((v + 46.56) / -44.14))

        # Ih has no q10 correction in the original mechanism.
        x = v + 154.9
        ih_alpha = _safe_rate(0.001 * 6.43 * x, np.exp(x / 11.9) - 1.0, 0.001 * 6.43 * 11.9)
        ih_beta = 0.001 * 193.0 * np.exp(v / 33.1)
        ih_m_inf = ih_alpha / (ih_alpha + ih_beta)
        ih_m_tau = 1.0 / (ih_alpha + ih_beta)

        shifted = v + 10.0
        calva_m_inf = 1.0 / (1.0 + np.exp((shifted + 30.0) / -6.0))
        calva_m_tau = (5.0 + 20.0 / (1.0 + np.exp((shifted + 25.0) / 5.0))) / q
        calva_h_inf = 1.0 / (1.0 + np.exp((shifted + 80.0) / 6.4))
        calva_h_tau = (20.0 + 50.0 / (1.0 + np.exp((shifted + 40.0) / 7.0))) / q

        x = -27.0 - v
        cahva_m_alpha = _safe_rate(0.055 * x, np.exp(x / 3.8) - 1.0, 0.055 * 3.8)
        cahva_m_beta = 0.94 * np.exp((-75.0 - v) / 17.0)
        cahva_m_inf = cahva_m_alpha / (cahva_m_alpha + cahva_m_beta)
        cahva_m_tau = 1.0 / (cahva_m_alpha + cahva_m_beta)
        cahva_h_alpha = 0.000457 * np.exp((-13.0 - v) / 50.0)
        cahva_h_beta = 0.0065 / (np.exp((-v - 15.0) / 28.0) + 1.0)
        cahva_h_inf = cahva_h_alpha / (cahva_h_alpha + cahva_h_beta)
        cahva_h_tau = 1.0 / (cahva_h_alpha + cahva_h_beta)

        ca = max(float(ca_i_mM), 1.0e-7)
        sk_inf = 1.0 / (1.0 + (0.00043 / ca) ** 4.8)

        inf = np.asarray([
            nata_m_inf, nata_h_inf, nap_m_inf, nap_h_inf,
            kt_m_inf, kt_h_inf, kp_m_inf, kp_h_inf,
            skv_m_inf, ih_m_inf,
            calva_m_inf, calva_h_inf, cahva_m_inf, cahva_h_inf,
            sk_inf,
        ], dtype=np.float64)
        tau = np.asarray([
            nata_m_tau, nata_h_tau, nap_m_tau, nap_h_tau,
            kt_m_tau, kt_h_tau, kp_m_tau, kp_h_tau,
            skv_m_tau, ih_m_tau,
            calva_m_tau, calva_h_tau, cahva_m_tau, cahva_h_tau,
            1.0,
        ], dtype=np.float64)
        return inf, tau

    def initial_state(self, voltage_mV: float = -80.0) -> np.ndarray:
        result = np.zeros(len(FAITHFUL_STATE_NAMES), dtype=np.float64)
        result[0] = voltage_mV
        result[1] = self.config.ca_min_mM
        result[2:17] = self.gate_targets(voltage_mV, result[1])[0]
        return result

    @staticmethod
    def _nmda_block(voltage_mV: float) -> float:
        return float(1.0 / (1.0 + np.exp(-0.062 * voltage_mV) / 3.57))

    def _conductances_and_reversals(self, state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        c = self.config
        v, ca = float(state[0]), float(state[1])
        mna, hna, mnap, hnap, mkt, hkt, mkp, hkp, mskv, mih, ml, hl, mh, hh, zsk = state[2:17]
        conductances = np.asarray([
            c.g_pas_S_cm2,
            c.g_NaTa_tbar_S_cm2 * mna**3 * hna,
            c.g_Nap_Et2bar_S_cm2 * mnap**3 * hnap,
            c.g_K_Tstbar_S_cm2 * mkt**4 * hkt,
            c.g_K_Pstbar_S_cm2 * mkp**2 * hkp,
            c.g_SKv3_1bar_S_cm2 * mskv,
            c.g_Ihbar_S_cm2 * mih,
            c.g_Ca_LVAstbar_S_cm2 * ml**2 * hl,
            c.g_Ca_HVAbar_S_cm2 * mh**2 * hh,
            c.g_SK_E2bar_S_cm2 * zsk,
            state[17],
            state[18] * self._nmda_block(v),
            state[19],
        ], dtype=np.float64)
        reversals = np.asarray([
            c.e_pas_mV, c.e_na_mV, c.e_na_mV,
            c.e_k_mV, c.e_k_mV, c.e_k_mV, c.e_h_mV,
            self.calcium_reversal_mV(ca), self.calcium_reversal_mV(ca),
            c.e_k_mV, c.e_exc_mV, c.e_exc_mV, c.e_gabaa_mV,
        ], dtype=np.float64)
        return conductances, reversals

    def currents(self, state: np.ndarray) -> np.ndarray:
        conductances, reversals = self._conductances_and_reversals(state)
        currents = conductances * (float(state[0]) - reversals)
        return np.concatenate([currents, [currents.sum()]])

    def _substep(self, state: np.ndarray, i_inj_uA_cm2: float, dt_ms: float) -> np.ndarray:
        result = state.copy()
        v, ca = float(state[0]), float(state[1])
        inf, tau = self.gate_targets(v, ca)
        result[2:17] = inf + (state[2:17] - inf) * np.exp(-dt_ms / tau)

        conductances, reversals = self._conductances_and_reversals(result)
        total_g = float(conductances.sum())
        weighted_reversal = float(np.dot(conductances, reversals))
        injected_mA_cm2 = float(i_inj_uA_cm2) / 1000.0
        voltage_inf = (weighted_reversal + injected_mA_cm2) / total_g
        voltage_rate_per_ms = 1000.0 * total_g / self.config.cm_uF_cm2
        result[0] = voltage_inf + (v - voltage_inf) * np.exp(-voltage_rate_per_ms * dt_ms)

        currents = self.currents(result)
        calcium_current = float(currents[7] + currents[8])
        influx = 10000.0 * calcium_current * self.config.ca_gamma / (
            2.0 * self.FARADAY * self.config.ca_depth_um
        )
        calcium_inf = self.config.ca_min_mM - self.config.ca_decay_ms * influx
        result[1] = calcium_inf + (ca - calcium_inf) * np.exp(
            -dt_ms / self.config.ca_decay_ms
        )
        return result

    def step(
        self,
        state: np.ndarray,
        drive: Mapping[str, float] | np.ndarray,
        dt_ms: float,
        internal_dt_ms: float = 0.025,
    ) -> np.ndarray:
        if isinstance(drive, np.ndarray):
            values = {name: float(drive[index]) for index, name in enumerate(FAITHFUL_INPUT_NAMES)}
        else:
            values = {name: float(drive[name]) for name in FAITHFUL_INPUT_NAMES}
        ratio = dt_ms / internal_dt_ms
        if abs(ratio - round(ratio)) > 1.0e-12:
            raise ValueError("dt_ms must be an integer multiple of internal_dt_ms")

        result = np.asarray(state, dtype=np.float64).copy()
        result[17] += values["ampa_event_count"] * self.config.ampa_jump_S_cm2
        result[18] += values["nmda_event_count"] * self.config.nmda_jump_S_cm2
        result[19] += values["gaba_event_count"] * self.config.gabaa_jump_S_cm2
        for _ in range(int(round(ratio))):
            result = self._substep(result, values["i_inj_uA_cm2"], internal_dt_ms)
            result[17] *= np.exp(-internal_dt_ms / self.config.tau_ampa_ms)
            result[18] *= np.exp(-internal_dt_ms / self.config.tau_nmda_ms)
            result[19] *= np.exp(-internal_dt_ms / self.config.tau_gabaa_ms)
        if not np.isfinite(result).all():
            raise FloatingPointError("non-finite faithful compartment state")
        result[1] = max(result[1], self.config.ca_min_mM)
        result[2:17] = np.clip(result[2:17], 0.0, 1.0)
        result[17:20] = np.maximum(result[17:20], 0.0)
        return result

    def simulate(self, inputs: np.ndarray, dt_ms: float, internal_dt_ms: float) -> Dict[str, np.ndarray]:
        inputs = np.asarray(inputs, dtype=np.float64)
        if inputs.ndim != 2 or inputs.shape[1] != len(FAITHFUL_INPUT_NAMES):
            raise ValueError(f"inputs must have shape [steps, {len(FAITHFUL_INPUT_NAMES)}]")
        states = np.empty((len(inputs) + 1, len(FAITHFUL_STATE_NAMES)), dtype=np.float64)
        currents = np.empty((len(inputs) + 1, len(FAITHFUL_CURRENT_NAMES)), dtype=np.float64)
        spikes = np.zeros(len(inputs), dtype=np.uint8)
        states[0] = self.initial_state()
        currents[0] = self.currents(states[0])
        for index, drive in enumerate(inputs):
            states[index + 1] = self.step(states[index], drive, dt_ms, internal_dt_ms)
            currents[index + 1] = self.currents(states[index + 1])
            spikes[index] = bool(
                states[index, 0] < self.config.spike_threshold_mV
                and states[index + 1, 0] >= self.config.spike_threshold_mV
            )
        return {"states": states, "currents": currents, "spikes": spikes}

