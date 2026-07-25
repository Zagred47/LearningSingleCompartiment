"""NEURON-facing adapters used only for teacher instrumentation."""

from .backend import BoundaryState, TeacherBackend, TeacherSnapshot
from .audit import (
    detect_spikes,
    git_commit,
    load_source_functions,
    repository_file_record,
    sha256_file,
    validate_parent_tree,
    write_json,
)
from .neuron_manifest import (
    NeuronManifestConfig,
    NeuronManifestExtractor,
    NeuronSynapseBinding,
)
from .audit_runtime import TeacherAuditSession, resolve_audit_repositories
from .event_extractor import (
    EVENT_DETECTOR_VERSION,
    EventDefinition,
    default_event_definitions,
    event_ids_by_transition,
    extract_events,
    annotate_backpropagation,
)
from .diagnostic_dataset import (
    DiagnosticDatasetSession,
    expected_audit_hashes,
)
from .dendritic_calibration import (
    DENDRITIC_CALIBRATION_SCHEMA_VERSION,
    DendriticCandidate,
    DendriticProtocolCalibrator,
    InsufficientCanonicalSynapsesError,
    SynapseSelection,
    build_candidate_actions,
    candidate_from_mapping,
    candidate_from_selected_protocol,
    evenly_spaced_offsets,
)
from .diagnostic_dataset_v1 import (
    CALCIUM_PROTOCOL_ID,
    CONFIRMED_PROTOCOL_IDS,
    CONFIRMED_SEEDS,
    PLATEAU_PROTOCOL_ID,
    DiagnosticDatasetV1Session,
    actions_from_selected_protocol,
    canonical_json_sha256,
    filter_synaptic_actions,
    validate_calibration_artifacts,
)
from .causal_release import CausalReleaseRecorder
from .diagnostic_dataset_v1_1 import (
    TargetedDiagnosticDatasetSession,
    assess_causal_bap,
    select_causal_bap_assist,
)

__all__ = [
    "CausalReleaseRecorder",
    "TargetedDiagnosticDatasetSession",
    "assess_causal_bap",
    "select_causal_bap_assist",
    "BoundaryState",
    "default_event_definitions",
    "detect_spikes",
    "DiagnosticDatasetSession",
    "DiagnosticDatasetV1Session",
    "DENDRITIC_CALIBRATION_SCHEMA_VERSION",
    "DendriticCandidate",
    "DendriticProtocolCalibrator",
    "InsufficientCanonicalSynapsesError",
    "SynapseSelection",
    "EVENT_DETECTOR_VERSION",
    "EventDefinition",
    "event_ids_by_transition",
    "CALCIUM_PROTOCOL_ID",
    "CONFIRMED_PROTOCOL_IDS",
    "CONFIRMED_SEEDS",
    "PLATEAU_PROTOCOL_ID",
    "actions_from_selected_protocol",
    "canonical_json_sha256",
    "filter_synaptic_actions",
    "validate_calibration_artifacts",
    "build_candidate_actions",
    "candidate_from_mapping",
    "candidate_from_selected_protocol",
    "evenly_spaced_offsets",
    "extract_events",
    "annotate_backpropagation",
    "expected_audit_hashes",
    "git_commit",
    "load_source_functions",
    "NeuronManifestConfig",
    "NeuronManifestExtractor",
    "NeuronSynapseBinding",
    "repository_file_record",
    "sha256_file",
    "TeacherBackend",
    "TeacherAuditSession",
    "TeacherSnapshot",
    "validate_parent_tree",
    "write_json",
    "resolve_audit_repositories",
]
