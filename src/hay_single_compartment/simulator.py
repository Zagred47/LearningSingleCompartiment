"""Dependency-light simulator of one challenging Hay-inspired compartment.

The model deliberately keeps one electrical compartment while retaining the
slow and fast state that makes the original Hay cell interesting: transient
and persistent sodium, three potassium families, HCN, low/high-voltage
calcium, calcium-activated potassium, calcium concentration, and three
synaptic conductances.  It is a reduced model, not a claim of numerical
equivalence with the 642-segment Hay et al. neuron.
"""

from __future__ import annotations

from typing import Dict, Mapping, Tuple

import numpy as np

from .config import MembraneConfig


STATE_NAMES = (
    "v_mV",
    "ca_i_mM",
    "m_NaTa_t",
    "h_NaTa_t",
    "m_Nap_Et2",
    "n_Kdr",
    "m_SKv3_1",
    "p_Im",
    "m_Ih",
    "m_Ca_LVAst",
    "h_Ca_LVAst",
    "m_Ca_HVA",
    "h_Ca_HVA",
    "m_SK_E2",
    "g_AMPA_uS",
    "g_NMDA_uS",
    "g_GABAA_uS",
)

INPUT_NAMES = (
    "i_inj_nA",
    "ampa_event_count",
    "nmda_event_count",
    "gaba_event_count",
)

CURRENT_NAMES = (
    "i_leak_nA",
    "i_NaTa_t_nA",
    "i_Nap_Et2_nA",
    "i_Kdr_nA",
    "i_SKv3_1_nA",
    "i_Im_nA",
    "i_Ih_nA",
    "i_Ca_LVAst_nA",
    "i_Ca_HVA_nA",
    "i_SK_E2_nA",
    "i_AMPA_nA",
    "i_NMDA_nA",
    "i_GABAA_nA",
    "i_ionic_total_nA",
)


def _sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-np.clip(x, -60.0, 60.0))))


def _gate_targets(v: float, ca: float) -> Tuple[np.ndarray, np.ndarray]:
    """Steady states and time constants for the 12 channel gates."""

    inf = np.asarray(
        [
            _sigmoid((v + 38.0) / 6.0),
            _sigmoid(-(v + 58.0) / 6.5),
            _sigmoid((v + 52.0) / 5.0),
            _sigmoid((v + 30.0) / 10.0),
            _sigmoid((v + 12.0) / 9.0),
            _sigmoid((v + 40.0) / 8.0),
            _sigmoid(-(v + 82.0) / 8.0),
            _sigmoid((v + 55.0) / 6.2),
            _sigmoid(-(v + 80.0) / 6.0),
            _sigmoid((v + 27.0) / 6.5),
            _sigmoid(-(v + 45.0) / 7.0),
            ca**4 / (ca**4 + (3.5e-4) ** 4),
        ],
        dtype=np.float64,
    )
    tau = np.asarray(
        [
            0.04 + 0.15 * np.exp(-((v + 38.0) / 18.0) ** 2),
            0.35 + 2.0 * np.exp(-((v + 55.0) / 20.0) ** 2),
            1.0 + 4.0 * np.exp(-((v + 50.0) / 20.0) ** 2),
            0.7 + 3.0 * np.exp(-((v + 35.0) / 25.0) ** 2),
            0.25 + 1.0 * np.exp(-((v + 10.0) / 30.0) ** 2),
            20.0 + 80.0 * np.exp(-((v + 40.0) / 25.0) ** 2),
            25.0 + 120.0 * np.exp(-((v + 80.0) / 20.0) ** 2),
            1.2 + 5.0 * np.exp(-((v + 55.0) / 20.0) ** 2),
            12.0 + 35.0 * np.exp(-((v + 75.0) / 18.0) ** 2),
            0.4 + 1.5 * np.exp(-((v + 25.0) / 20.0) ** 2),
            8.0 + 25.0 * np.exp(-((v + 45.0) / 20.0) ** 2),
            2.0,
        ],
        dtype=np.float64,
    )
    return inf, tau


class SingleCompartmentHay:
    """Stable Rush-Larsen/exponential integrator for the reduced compartment."""

    def __init__(self, config: MembraneConfig | None = None) -> None:
        self.config = config or MembraneConfig()

    def initial_state(self, voltage_mv: float = -76.0) -> np.ndarray:
        state = np.zeros(len(STATE_NAMES), dtype=np.float64)
        state[0] = voltage_mv
        state[1] = self.config.ca_rest_mm
        state[2:14] = _gate_targets(voltage_mv, state[1])[0]
        return state

    @staticmethod
    def _nmda_block(voltage_mv: float) -> float:
        return float(1.0 / (1.0 + np.exp(-0.062 * voltage_mv) / 3.57))

    def _conductances(self, state: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        c = self.config
        v, _ca = state[:2]
        mna, hna, mnap, nk, mkv3, pim, mh, ml, hl, mhv, hhv, msk = state[2:14]
        g_nmda_effective = state[15] * self._nmda_block(v)
        conductances = np.asarray(
            [
                c.g_leak_us,
                c.g_na_us * mna**3 * hna,
                c.g_nap_us * mnap**3,
                c.g_kdr_us * nk**4,
                c.g_kv3_us * mkv3**2,
                c.g_m_us * pim,
                c.g_h_us * mh,
                c.g_ca_lva_us * ml**2 * hl,
                c.g_ca_hva_us * mhv**2 * hhv,
                c.g_sk_us * msk,
                state[14],
                g_nmda_effective,
                state[16],
            ],
            dtype=np.float64,
        )
        reversals = np.asarray(
            [
                c.e_leak_mv,
                c.e_na_mv,
                c.e_na_mv,
                c.e_k_mv,
                c.e_k_mv,
                c.e_k_mv,
                c.e_h_mv,
                c.e_ca_mv,
                c.e_ca_mv,
                c.e_k_mv,
                c.e_exc_mv,
                c.e_exc_mv,
                c.e_gaba_mv,
            ],
            dtype=np.float64,
        )
        return conductances, reversals

    def currents(self, state: np.ndarray) -> np.ndarray:
        conductances, reversals = self._conductances(state)
        currents = conductances * (float(state[0]) - reversals)
        return np.concatenate([currents, [currents.sum()]])

    def _substep(self, state: np.ndarray, i_inj_na: float, dt_ms: float) -> np.ndarray:
        c = self.config
        result = state.copy()
        v, ca = float(state[0]), float(state[1])

        inf, tau = _gate_targets(v, ca)
        result[2:14] = inf + (state[2:14] - inf) * np.exp(-dt_ms / tau)

        conductances, reversals = self._conductances(result)
        total_g = float(conductances.sum())
        weighted_e = float(np.dot(conductances, reversals))
        v_inf = (weighted_e + i_inj_na) / total_g
        result[0] = v_inf + (v - v_inf) * np.exp(
            -total_g * dt_ms / c.capacitance_nf
        )

        updated_currents = self.currents(result)
        inward_ca = max(0.0, -float(updated_currents[7] + updated_currents[8]))
        ca_target = c.ca_rest_mm + c.ca_tau_ms * c.ca_influx_mm_per_na_ms * inward_ca
        result[1] = ca_target + (ca - ca_target) * np.exp(-dt_ms / c.ca_tau_ms)
        return result

    def step(
        self,
        state: np.ndarray,
        drive: Mapping[str, float] | np.ndarray,
        dt_ms: float,
        internal_dt_ms: float = 0.025,
    ) -> np.ndarray:
        """Advance one observed step; event counts arrive at its left boundary."""

        if isinstance(drive, np.ndarray):
            values = {name: float(drive[i]) for i, name in enumerate(INPUT_NAMES)}
        else:
            values = {name: float(drive[name]) for name in INPUT_NAMES}
        ratio = dt_ms / internal_dt_ms
        if abs(ratio - round(ratio)) > 1e-12:
            raise ValueError("dt_ms must be an integer multiple of internal_dt_ms")

        result = np.asarray(state, dtype=np.float64).copy()
        result[14] += values["ampa_event_count"] * self.config.ampa_jump_us
        result[15] += values["nmda_event_count"] * self.config.nmda_jump_us
        result[16] += values["gaba_event_count"] * self.config.gaba_jump_us

        for _ in range(int(round(ratio))):
            result = self._substep(result, values["i_inj_nA"], internal_dt_ms)
            result[14] *= np.exp(-internal_dt_ms / self.config.tau_ampa_ms)
            result[15] *= np.exp(-internal_dt_ms / self.config.tau_nmda_ms)
            result[16] *= np.exp(-internal_dt_ms / self.config.tau_gaba_ms)

        if not np.isfinite(result).all():
            raise FloatingPointError("non-finite compartment state")
        result[1] = max(result[1], 0.0)
        result[2:14] = np.clip(result[2:14], 0.0, 1.0)
        result[14:17] = np.maximum(result[14:17], 0.0)
        return result

    def simulate(self, inputs: np.ndarray, dt_ms: float, internal_dt_ms: float) -> Dict[str, np.ndarray]:
        """Simulate one trajectory and retain every Markov state and current."""

        inputs = np.asarray(inputs, dtype=np.float64)
        if inputs.ndim != 2 or inputs.shape[1] != len(INPUT_NAMES):
            raise ValueError(f"inputs must have shape [steps, {len(INPUT_NAMES)}]")
        states = np.empty((len(inputs) + 1, len(STATE_NAMES)), dtype=np.float64)
        currents = np.empty((len(inputs) + 1, len(CURRENT_NAMES)), dtype=np.float64)
        spikes = np.zeros(len(inputs), dtype=np.uint8)
        states[0] = self.initial_state()
        currents[0] = self.currents(states[0])
        for index, drive in enumerate(inputs):
            states[index + 1] = self.step(states[index], drive, dt_ms, internal_dt_ms)
            currents[index + 1] = self.currents(states[index + 1])
            spikes[index] = bool(
                states[index, 0] < self.config.spike_threshold_mv
                and states[index + 1, 0] >= self.config.spike_threshold_mv
            )
        return {"states": states, "currents": currents, "spikes": spikes}
