"""Four-compartment reduction of the active Hay L5 pyramidal cell.

The reduction deliberately keeps the four electrically distinct roles that
cannot be represented by one isopotential compartment: basal dendrite, soma,
apical trunk and distal tuft.  Channel equations and density parameters are
transcribed from ModelDB 139653 ``L5PCbiophys3.hoc`` and its NMODL files.

This is a reduced teacher, not a claim that four cylinders reproduce the full
reconstruction.  The reduction contract is explicit: channel kinetics are not
compressed, synapses retain rise and decay states, and spatial interaction is
limited to three axial edges.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, Mapping, Sequence

import numpy as np

from .faithful import FaithfulHaySoma, FaithfulMembraneConfig


COMPARTMENT_NAMES = ("soma", "basal", "trunk", "tuft")
SYNAPTIC_REGIONS = ("basal", "trunk", "tuft")
RECEPTOR_NAMES = ("ampa", "nmda", "gabaa")

SOMA_GATES = (
    "m_NaTa_t", "h_NaTa_t", "m_Nap_Et2", "h_Nap_Et2",
    "m_K_Tst", "h_K_Tst", "m_K_Pst", "h_K_Pst", "m_SKv3_1",
    "m_Ih", "m_Ca_LVAst", "h_Ca_LVAst", "m_Ca_HVA",
    "h_Ca_HVA", "z_SK_E2",
)
APICAL_GATES = (
    "m_NaTa_t", "h_NaTa_t", "m_SKv3_1", "m_Ih",
    "m_Ca_LVAst", "h_Ca_LVAst", "m_Ca_HVA", "h_Ca_HVA",
    "z_SK_E2", "m_Im",
)


def _state_names() -> tuple[str, ...]:
    names = ["soma.v_mV", "soma.ca_i_mM"]
    names.extend(f"soma.{name}" for name in SOMA_GATES)
    names.extend(("basal.v_mV", "basal.m_Ih"))
    for region in ("trunk", "tuft"):
        names.extend((f"{region}.v_mV", f"{region}.ca_i_mM"))
        names.extend(f"{region}.{name}" for name in APICAL_GATES)
    for region in SYNAPTIC_REGIONS:
        for receptor in RECEPTOR_NAMES:
            names.extend((f"{region}.{receptor}_rise", f"{region}.{receptor}_decay"))
    return tuple(names)


MICRO_STATE_NAMES = _state_names()


@dataclass(frozen=True)
class ReducedCompartmentGeometry:
    """Electrical cylinder and represented cable length in micrometres."""

    length_um: float
    diameter_um: float
    represented_length_um: float
    distance_from_soma_um: float
    cm_uF_cm2: float


@dataclass(frozen=True)
class MicroGeometryConfig:
    """Transparent four-cylinder reduction of Hay morphology cell1.asc.

    ``represented_length_um`` controls synapse allocation; cylinder geometry
    controls area, capacitance and axial resistance.  Keeping these separate
    prevents a collapsed tree from acquiring an unphysical membrane area.
    """

    soma: ReducedCompartmentGeometry = field(default_factory=lambda: ReducedCompartmentGeometry(
        length_um=23.0, diameter_um=23.0, represented_length_um=23.0,
        distance_from_soma_um=0.0, cm_uF_cm2=1.0,
    ))
    basal: ReducedCompartmentGeometry = field(default_factory=lambda: ReducedCompartmentGeometry(
        length_um=180.0, diameter_um=2.4, represented_length_um=2600.0,
        distance_from_soma_um=90.0, cm_uF_cm2=2.0,
    ))
    trunk: ReducedCompartmentGeometry = field(default_factory=lambda: ReducedCompartmentGeometry(
        length_um=500.0, diameter_um=3.2, represented_length_um=3100.0,
        distance_from_soma_um=420.0, cm_uF_cm2=2.0,
    ))
    tuft: ReducedCompartmentGeometry = field(default_factory=lambda: ReducedCompartmentGeometry(
        length_um=320.0, diameter_um=1.7, represented_length_um=4300.0,
        distance_from_soma_um=760.0, cm_uF_cm2=2.0,
    ))
    axial_resistivity_ohm_cm: float = 100.0


@dataclass(frozen=True)
class MicroSynapseConfig:
    tau_ampa_rise_ms: float = 0.3
    tau_ampa_decay_ms: float = 3.0
    tau_nmda_rise_ms: float = 2.0
    tau_nmda_decay_ms: float = 70.0
    tau_gabaa_rise_ms: float = 0.2
    tau_gabaa_decay_ms: float = 8.0
    ampa_peak_nS: float = 0.4
    nmda_peak_nS: float = 0.4
    gabaa_peak_nS: float = 1.0
    e_exc_mV: float = 0.0
    e_gabaa_mV: float = -80.0


@dataclass(frozen=True)
class MicroHayConfig:
    geometry: MicroGeometryConfig = field(default_factory=MicroGeometryConfig)
    synapses: MicroSynapseConfig = field(default_factory=MicroSynapseConfig)
    celsius: float = 34.0
    e_pas_mV: float = -90.0
    e_na_mV: float = 50.0
    e_k_mV: float = -85.0
    e_h_mV: float = -45.0
    ca_o_mM: float = 2.0
    ca_min_mM: float = 1.0e-4
    spike_threshold_mV: float = -20.0

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def _double_exp_normalizer(tau_rise: float, tau_decay: float) -> float:
    if not 0.0 < tau_rise < tau_decay:
        raise ValueError("synaptic rise time must be positive and shorter than decay")
    peak_t = tau_rise * tau_decay / (tau_decay - tau_rise) * np.log(tau_decay / tau_rise)
    peak = np.exp(-peak_t / tau_decay) - np.exp(-peak_t / tau_rise)
    return float(1.0 / peak)


class FourCompartmentHay:
    """State-complete four-compartment Hay reduction driven only by spikes."""

    FARADAY = FaithfulHaySoma.FARADAY
    _faithful_gate_index = {name: index for index, name in enumerate(SOMA_GATES)}
    _edges = (("soma", "basal"), ("soma", "trunk"), ("trunk", "tuft"))

    def __init__(self, config: MicroHayConfig | None = None) -> None:
        self.config = config or MicroHayConfig()
        membrane = FaithfulMembraneConfig(celsius=self.config.celsius)
        self.kinetics = FaithfulHaySoma(membrane)
        self.state_names = MICRO_STATE_NAMES
        self.index = {name: i for i, name in enumerate(self.state_names)}
        self.geometry = {
            name: getattr(self.config.geometry, name) for name in COMPARTMENT_NAMES
        }
        self.area_cm2 = {
            name: np.pi * item.length_um * item.diameter_um * 1.0e-8
            for name, item in self.geometry.items()
        }
        self.capacitance_nF = {
            name: item.cm_uF_cm2 * self.area_cm2[name] * 1000.0
            for name, item in self.geometry.items()
        }
        self.axial_uS = {
            edge: self._axial_conductance_uS(*edge) for edge in self._edges
        }
        self._synapse_normalizers = {
            "ampa": _double_exp_normalizer(
                self.config.synapses.tau_ampa_rise_ms,
                self.config.synapses.tau_ampa_decay_ms,
            ),
            "nmda": _double_exp_normalizer(
                self.config.synapses.tau_nmda_rise_ms,
                self.config.synapses.tau_nmda_decay_ms,
            ),
            "gabaa": _double_exp_normalizer(
                self.config.synapses.tau_gabaa_rise_ms,
                self.config.synapses.tau_gabaa_decay_ms,
            ),
        }
        self.current_names = self._current_names()

    @staticmethod
    def _current_names() -> tuple[str, ...]:
        names = []
        mechanisms = {
            "soma": ("pas", "NaTa_t", "Nap_Et2", "K_Tst", "K_Pst", "SKv3_1", "Ih", "Ca_LVAst", "Ca_HVA", "SK_E2"),
            "basal": ("pas", "Ih", "AMPA", "NMDA", "GABAA"),
            "trunk": ("pas", "NaTa_t", "SKv3_1", "Ih", "Ca_LVAst", "Ca_HVA", "SK_E2", "Im", "AMPA", "NMDA", "GABAA"),
            "tuft": ("pas", "NaTa_t", "SKv3_1", "Ih", "Ca_LVAst", "Ca_HVA", "SK_E2", "Im", "AMPA", "NMDA", "GABAA"),
        }
        for region, items in mechanisms.items():
            names.extend(f"{region}.i_{item}_nA" for item in items)
            names.extend((f"{region}.i_axial_nA", f"{region}.i_total_nA"))
        return tuple(names)

    def _axial_conductance_uS(self, first: str, second: str) -> float:
        resistance = 0.0
        for name in (first, second):
            item = self.geometry[name]
            length_cm = 0.5 * item.length_um * 1.0e-4
            radius_cm = 0.5 * item.diameter_um * 1.0e-4
            resistance += self.config.geometry.axial_resistivity_ohm_cm * length_cm / (
                np.pi * radius_cm**2
            )
        return float(1.0e6 / resistance)

    def _apical_density(self, region: str) -> Dict[str, float]:
        distance = self.geometry[region].distance_from_soma_um
        normalized_distance = distance / max(self.geometry["tuft"].distance_from_soma_um, 1.0)
        ih = 0.0002 * (-0.8696 + 2.0870 * np.exp(3.6161 * normalized_distance))
        in_hot_zone = 685.0 < distance < 885.0
        return {
            "pas": 0.0000589,
            "NaTa_t": 0.0213,
            "SKv3_1": 0.000261,
            "Ih": max(float(ih), 0.0),
            "Ca_LVAst": 0.0187 * (1.0 if in_hot_zone else 0.01),
            "Ca_HVA": 0.000555 * (1.0 if in_hot_zone else 0.1),
            "SK_E2": 0.0012,
            "Im": 0.0000675,
        }

    def _state_gate_values(self, state: np.ndarray, region: str) -> Dict[str, float]:
        gates = SOMA_GATES if region == "soma" else APICAL_GATES
        return {name: float(state[self.index[f"{region}.{name}"]]) for name in gates}

    def _membrane_terms(self, state: np.ndarray, region: str) -> tuple[list[float], list[float], list[str]]:
        c = self.config
        v = float(state[self.index[f"{region}.v_mV"]])
        if region == "soma":
            gates = self._state_gate_values(state, region)
            ca = float(state[self.index["soma.ca_i_mM"]])
            eca = self.kinetics.calcium_reversal_mV(ca)
            densities = [
                0.0000338,
                2.04 * gates["m_NaTa_t"]**3 * gates["h_NaTa_t"],
                0.00172 * gates["m_Nap_Et2"]**3 * gates["h_Nap_Et2"],
                0.0812 * gates["m_K_Tst"]**4 * gates["h_K_Tst"],
                0.00223 * gates["m_K_Pst"]**2 * gates["h_K_Pst"],
                0.693 * gates["m_SKv3_1"],
                0.0002 * gates["m_Ih"],
                0.00343 * gates["m_Ca_LVAst"]**2 * gates["h_Ca_LVAst"],
                0.000992 * gates["m_Ca_HVA"]**2 * gates["h_Ca_HVA"],
                0.0441 * gates["z_SK_E2"],
            ]
            reversals = [c.e_pas_mV, c.e_na_mV, c.e_na_mV, c.e_k_mV, c.e_k_mV,
                         c.e_k_mV, c.e_h_mV, eca, eca, c.e_k_mV]
            labels = ["pas", "NaTa_t", "Nap_Et2", "K_Tst", "K_Pst", "SKv3_1",
                      "Ih", "Ca_LVAst", "Ca_HVA", "SK_E2"]
            return densities, reversals, labels
        if region == "basal":
            ih_gate = float(state[self.index["basal.m_Ih"]])
            densities = [0.0000467, 0.0002 * ih_gate]
            reversals = [c.e_pas_mV, c.e_h_mV]
            labels = ["pas", "Ih"]
        else:
            gates = self._state_gate_values(state, region)
            ca = float(state[self.index[f"{region}.ca_i_mM"]])
            eca = self.kinetics.calcium_reversal_mV(ca)
            bar = self._apical_density(region)
            densities = [
                bar["pas"],
                bar["NaTa_t"] * gates["m_NaTa_t"]**3 * gates["h_NaTa_t"],
                bar["SKv3_1"] * gates["m_SKv3_1"],
                bar["Ih"] * gates["m_Ih"],
                bar["Ca_LVAst"] * gates["m_Ca_LVAst"]**2 * gates["h_Ca_LVAst"],
                bar["Ca_HVA"] * gates["m_Ca_HVA"]**2 * gates["h_Ca_HVA"],
                bar["SK_E2"] * gates["z_SK_E2"],
                bar["Im"] * gates["m_Im"],
            ]
            reversals = [c.e_pas_mV, c.e_na_mV, c.e_k_mV, c.e_h_mV, eca, eca, c.e_k_mV, c.e_k_mV]
            labels = ["pas", "NaTa_t", "SKv3_1", "Ih", "Ca_LVAst", "Ca_HVA", "SK_E2", "Im"]

        if region in SYNAPTIC_REGIONS:
            syn = self.synaptic_conductances(state, region)
            densities.extend((syn["ampa"], syn["nmda"] * self._nmda_block(v), syn["gabaa"]))
            reversals.extend((c.synapses.e_exc_mV, c.synapses.e_exc_mV, c.synapses.e_gabaa_mV))
            labels.extend(("AMPA", "NMDA", "GABAA"))
        return densities, reversals, labels

    @staticmethod
    def _nmda_block(voltage_mV: float) -> float:
        return float(1.0 / (1.0 + np.exp(-0.062 * voltage_mV) / 3.57))

    def synaptic_conductances(self, state: np.ndarray, region: str) -> Dict[str, float]:
        return {
            receptor: max(0.0, float(
                state[self.index[f"{region}.{receptor}_decay"]]
                - state[self.index[f"{region}.{receptor}_rise"]]
            ))
            for receptor in RECEPTOR_NAMES
        }

    def initial_state(self, voltage_mV: float = -80.0) -> np.ndarray:
        state = np.zeros(len(self.state_names), dtype=np.float64)
        soma_targets, _ = self.kinetics.gate_targets(voltage_mV, self.config.ca_min_mM)
        state[self.index["soma.v_mV"]] = voltage_mV
        state[self.index["soma.ca_i_mM"]] = self.config.ca_min_mM
        for gate, value in zip(SOMA_GATES, soma_targets):
            state[self.index[f"soma.{gate}"]] = value
        state[self.index["basal.v_mV"]] = voltage_mV
        state[self.index["basal.m_Ih"]] = soma_targets[self._faithful_gate_index["m_Ih"]]
        for region in ("trunk", "tuft"):
            state[self.index[f"{region}.v_mV"]] = voltage_mV
            state[self.index[f"{region}.ca_i_mM"]] = self.config.ca_min_mM
            for gate in APICAL_GATES[:-1]:
                state[self.index[f"{region}.{gate}"]] = soma_targets[self._faithful_gate_index[gate]]
            im_inf, _ = self._im_target(voltage_mV)
            state[self.index[f"{region}.m_Im"]] = im_inf
        return state

    def _im_target(self, voltage_mV: float) -> tuple[float, float]:
        q10 = 2.3 ** ((34.0 - 21.0) / 10.0)
        alpha = 3.3e-3 * np.exp(0.1 * (voltage_mV + 35.0))
        beta = 3.3e-3 * np.exp(-0.1 * (voltage_mV + 35.0))
        return float(alpha / (alpha + beta)), float(1.0 / (alpha + beta) / q10)

    def _update_gates(self, old: np.ndarray, result: np.ndarray, dt_ms: float) -> None:
        for region in ("soma", "trunk", "tuft"):
            v = float(old[self.index[f"{region}.v_mV"]])
            ca = float(old[self.index[f"{region}.ca_i_mM"]])
            targets, taus = self.kinetics.gate_targets(v, ca)
            gates = SOMA_GATES if region == "soma" else APICAL_GATES[:-1]
            for gate in gates:
                source = self._faithful_gate_index[gate]
                index = self.index[f"{region}.{gate}"]
                result[index] = targets[source] + (old[index] - targets[source]) * np.exp(-dt_ms / taus[source])
            if region != "soma":
                target, tau = self._im_target(v)
                index = self.index[f"{region}.m_Im"]
                result[index] = target + (old[index] - target) * np.exp(-dt_ms / tau)
        v = float(old[self.index["basal.v_mV"]])
        target, tau = self.kinetics.gate_targets(v, self.config.ca_min_mM)
        index = self.index["basal.m_Ih"]
        source = self._faithful_gate_index["m_Ih"]
        result[index] = target[source] + (old[index] - target[source]) * np.exp(-dt_ms / tau[source])

    def _update_synapses(self, result: np.ndarray, dt_ms: float) -> None:
        s = self.config.synapses
        taus = {
            "ampa": (s.tau_ampa_rise_ms, s.tau_ampa_decay_ms),
            "nmda": (s.tau_nmda_rise_ms, s.tau_nmda_decay_ms),
            "gabaa": (s.tau_gabaa_rise_ms, s.tau_gabaa_decay_ms),
        }
        for region in SYNAPTIC_REGIONS:
            for receptor, (rise, decay) in taus.items():
                result[self.index[f"{region}.{receptor}_rise"]] *= np.exp(-dt_ms / rise)
                result[self.index[f"{region}.{receptor}_decay"]] *= np.exp(-dt_ms / decay)

    def _apply_events(self, state: np.ndarray, event_counts: Sequence[float]) -> None:
        if len(event_counts) != 2 * len(SYNAPTIC_REGIONS):
            raise ValueError("event_counts must be [E,I] for basal, trunk and tuft")
        s = self.config.synapses
        peaks = {"ampa": s.ampa_peak_nS, "nmda": s.nmda_peak_nS, "gabaa": s.gabaa_peak_nS}
        for region_index, region in enumerate(SYNAPTIC_REGIONS):
            counts = {"ampa": event_counts[2 * region_index], "nmda": event_counts[2 * region_index], "gabaa": event_counts[2 * region_index + 1]}
            for receptor, count in counts.items():
                density_jump = (
                    float(count) * peaks[receptor] * 1.0e-9 / self.area_cm2[region]
                    * self._synapse_normalizers[receptor]
                )
                state[self.index[f"{region}.{receptor}_rise"]] += density_jump
                state[self.index[f"{region}.{receptor}_decay"]] += density_jump

    def _axial_currents(self, state: np.ndarray) -> Dict[str, float]:
        currents = {name: 0.0 for name in COMPARTMENT_NAMES}
        for (first, second), conductance in self.axial_uS.items():
            first_v = float(state[self.index[f"{first}.v_mV"]])
            second_v = float(state[self.index[f"{second}.v_mV"]])
            outward = conductance * (first_v - second_v)
            currents[first] += outward
            currents[second] -= outward
        return currents

    def currents(self, state: np.ndarray) -> np.ndarray:
        axial = self._axial_currents(state)
        values = []
        for region in COMPARTMENT_NAMES:
            voltage = float(state[self.index[f"{region}.v_mV"]])
            densities, reversals, _ = self._membrane_terms(state, region)
            membrane = np.asarray(densities) * (voltage - np.asarray(reversals)) * self.area_cm2[region] * 1.0e6
            values.extend(membrane.tolist())
            values.extend((axial[region], float(membrane.sum() + axial[region])))
        return np.asarray(values, dtype=np.float64)

    def _substep(self, state: np.ndarray, dt_ms: float) -> np.ndarray:
        result = state.copy()
        self._update_gates(state, result, dt_ms)
        axial = self._axial_currents(state)
        calcium_currents: Dict[str, float] = {}
        next_voltages: Dict[str, float] = {}
        for region in COMPARTMENT_NAMES:
            voltage = float(state[self.index[f"{region}.v_mV"]])
            densities, reversals, labels = self._membrane_terms(result, region)
            area = self.area_cm2[region]
            membrane_g_uS = np.asarray(densities) * area * 1.0e6
            total_g = float(membrane_g_uS.sum())
            weighted_reversal = float(np.dot(membrane_g_uS, reversals))
            for neighbour_edge, axial_g in self.axial_uS.items():
                if region not in neighbour_edge:
                    continue
                neighbour = neighbour_edge[1] if neighbour_edge[0] == region else neighbour_edge[0]
                total_g += axial_g
                weighted_reversal += axial_g * float(state[self.index[f"{neighbour}.v_mV"]])
            v_inf = weighted_reversal / total_g
            rate_per_ms = total_g / self.capacitance_nF[region]
            next_voltages[region] = v_inf + (voltage - v_inf) * np.exp(-rate_per_ms * dt_ms)
            if region in ("soma", "trunk", "tuft"):
                current_density = np.asarray(densities) * (voltage - np.asarray(reversals))
                calcium_currents[region] = float(sum(
                    current_density[index] for index, label in enumerate(labels)
                    if label in ("Ca_LVAst", "Ca_HVA")
                ))
        for region, voltage in next_voltages.items():
            result[self.index[f"{region}.v_mV"]] = voltage
        for region, calcium_current in calcium_currents.items():
            ca_index = self.index[f"{region}.ca_i_mM"]
            old_ca = float(state[ca_index])
            decay = 460.0 if region == "soma" else 122.0
            gamma = 0.000501 if region == "soma" else 0.000509
            influx = 10000.0 * calcium_current * gamma / (2.0 * self.FARADAY * 0.1)
            ca_inf = self.config.ca_min_mM - decay * influx
            result[ca_index] = max(
                self.config.ca_min_mM,
                ca_inf + (old_ca - ca_inf) * np.exp(-dt_ms / decay),
            )
        self._update_synapses(result, dt_ms)
        return result

    def step(self, state: np.ndarray, event_counts: Sequence[float], dt_ms: float, internal_dt_ms: float = 0.025) -> np.ndarray:
        ratio = dt_ms / internal_dt_ms
        if abs(ratio - round(ratio)) > 1.0e-12:
            raise ValueError("dt_ms must be an integer multiple of internal_dt_ms")
        result = np.asarray(state, dtype=np.float64).copy()
        self._apply_events(result, event_counts)
        for _ in range(int(round(ratio))):
            result = self._substep(result, internal_dt_ms)
        if not np.isfinite(result).all():
            raise FloatingPointError("non-finite four-compartment state")
        for name in self.state_names:
            if any(token in name for token in (".m_", ".h_", ".z_")):
                result[self.index[name]] = np.clip(result[self.index[name]], 0.0, 1.0)
        return result

    def simulate(self, binary_spikes: np.ndarray, input_metadata: Sequence[Mapping[str, object]], dt_ms: float, internal_dt_ms: float = 0.025) -> Dict[str, np.ndarray]:
        binary_spikes = np.asarray(binary_spikes, dtype=np.uint8)
        if binary_spikes.ndim != 2 or binary_spikes.shape[1] != len(input_metadata):
            raise ValueError("binary_spikes must be [time, synapse] and match metadata")
        event_map = {(region, kind): [] for region in SYNAPTIC_REGIONS for kind in ("excitatory", "inhibitory")}
        for index, metadata in enumerate(input_metadata):
            event_map[(str(metadata["region"]), str(metadata["kind"]))].append(index)
        states = np.empty((len(binary_spikes) + 1, len(self.state_names)), dtype=np.float64)
        currents = np.empty((len(binary_spikes) + 1, len(self.current_names)), dtype=np.float64)
        somatic_spikes = np.zeros(len(binary_spikes), dtype=np.uint8)
        event_counts = np.zeros((len(binary_spikes), 2 * len(SYNAPTIC_REGIONS)), dtype=np.uint16)
        states[0] = self.initial_state()
        currents[0] = self.currents(states[0])
        for time_index, row in enumerate(binary_spikes):
            for region_index, region in enumerate(SYNAPTIC_REGIONS):
                event_counts[time_index, 2 * region_index] = int(row[event_map[(region, "excitatory")]].sum())
                event_counts[time_index, 2 * region_index + 1] = int(row[event_map[(region, "inhibitory")]].sum())
            states[time_index + 1] = self.step(states[time_index], event_counts[time_index], dt_ms, internal_dt_ms)
            currents[time_index + 1] = self.currents(states[time_index + 1])
            somatic_spikes[time_index] = bool(
                states[time_index, self.index["soma.v_mV"]] < self.config.spike_threshold_mV
                and states[time_index + 1, self.index["soma.v_mV"]] >= self.config.spike_threshold_mV
            )
        return {"states": states, "currents": currents, "spikes": somatic_spikes, "event_counts": event_counts}
