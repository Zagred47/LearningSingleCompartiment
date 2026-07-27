"""Conceptual schema and causal interfaces for the single compartment.

The schema is deliberately separate from the neural implementation.  It says
which observed variables belong to an entity and which observed causes that
entity may read during a one-step flow-map prediction.
"""

from __future__ import annotations

from dataclasses import dataclass

from .simulator import INPUT_NAMES, STATE_NAMES


@dataclass(frozen=True)
class OntologyGroup:
    """One predicted subsystem and its permitted normalized input features."""

    name: str
    state_names: tuple[str, ...]
    dependency_names: tuple[str, ...]

    @property
    def output_indices(self) -> tuple[int, ...]:
        return tuple(STATE_NAMES.index(name) for name in self.state_names)

    @property
    def feature_indices(self) -> tuple[int, ...]:
        all_features = STATE_NAMES + INPUT_NAMES
        return tuple(all_features.index(name) for name in self.dependency_names)


EXTERNAL_INPUTS = tuple(INPUT_NAMES)

# The input dependencies are the causal closure over one observed interval.
# Thus a voltage-gated channel may read external drive: during the interval the
# drive changes V, which in turn changes the gate.  Synaptic conductance states
# remain stricter and read only their matching event stream.
ONTOLOGY_GROUPS = (
    OntologyGroup("membrane", ("v_mV",), STATE_NAMES + INPUT_NAMES),
    OntologyGroup(
        "calcium_pool",
        ("ca_i_mM",),
        ("v_mV", "ca_i_mM", "m_Ca_LVAst", "h_Ca_LVAst", "m_Ca_HVA", "h_Ca_HVA")
        + EXTERNAL_INPUTS,
    ),
    OntologyGroup(
        "sodium_channels",
        ("m_NaTa_t", "h_NaTa_t", "m_Nap_Et2"),
        ("v_mV", "m_NaTa_t", "h_NaTa_t", "m_Nap_Et2") + EXTERNAL_INPUTS,
    ),
    OntologyGroup(
        "voltage_gated_potassium",
        ("n_Kdr", "m_SKv3_1", "p_Im"),
        ("v_mV", "n_Kdr", "m_SKv3_1", "p_Im") + EXTERNAL_INPUTS,
    ),
    OntologyGroup(
        "hcn_channel",
        ("m_Ih",),
        ("v_mV", "m_Ih") + EXTERNAL_INPUTS,
    ),
    OntologyGroup(
        "calcium_channels",
        ("m_Ca_LVAst", "h_Ca_LVAst", "m_Ca_HVA", "h_Ca_HVA"),
        ("v_mV", "ca_i_mM", "m_Ca_LVAst", "h_Ca_LVAst", "m_Ca_HVA", "h_Ca_HVA")
        + EXTERNAL_INPUTS,
    ),
    OntologyGroup(
        "calcium_activated_potassium",
        ("m_SK_E2",),
        ("v_mV", "ca_i_mM", "m_SK_E2") + EXTERNAL_INPUTS,
    ),
    OntologyGroup(
        "ampa_receptor",
        ("g_AMPA_uS",),
        ("g_AMPA_uS", "ampa_event_count"),
    ),
    OntologyGroup(
        "nmda_receptor",
        ("g_NMDA_uS",),
        ("g_NMDA_uS", "nmda_event_count"),
    ),
    OntologyGroup(
        "gabaa_receptor",
        ("g_GABAA_uS",),
        ("g_GABAA_uS", "gaba_event_count"),
    ),
)


def validate_ontology() -> None:
    """Raise when the schema does not predict every state exactly once."""

    outputs = [name for group in ONTOLOGY_GROUPS for name in group.state_names]
    if len(outputs) != len(set(outputs)):
        raise ValueError("ontology predicts at least one state more than once")
    if set(outputs) != set(STATE_NAMES):
        missing = sorted(set(STATE_NAMES) - set(outputs))
        extra = sorted(set(outputs) - set(STATE_NAMES))
        raise ValueError(f"ontology/state mismatch; missing={missing}, extra={extra}")


validate_ontology()
