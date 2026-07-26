"""Full-state flow-map baselines and the future latent HayFlow model."""

from .full_state_flowmap import (
    DualRidgeBaseline,
    FlatResidualMLP,
    FlowmapModelConfig,
    PersistenceBaseline,
    StructuredSharedResidual,
    parameter_count,
    require_torch,
    ridge_design_matrix,
    structured_arrays,
)
from .flowmap_experiment import (
    AuxiliaryNormalizer,
    FlowmapExperimentConfig,
    FullStateFlowmapExperiment,
)
from .reconditioned_full_state import ReconditionedStructuredResidual
from .reconditioned_experiment import (
    ReconditionedExperimentConfig,
    ReconditionedFlowmapExperiment,
    ReconditionedRunSpec,
)
from .release_identifiability_experiment import (
    ReleaseExperimentConfig,
    ReleaseIdentifiabilityExperiment,
    ReleaseRunSpec,
)

__all__ = [
    "DualRidgeBaseline",
    "FlatResidualMLP",
    "FlowmapModelConfig",
    "PersistenceBaseline",
    "StructuredSharedResidual",
    "parameter_count",
    "require_torch",
    "ridge_design_matrix",
    "structured_arrays",
    "AuxiliaryNormalizer",
    "FlowmapExperimentConfig",
    "FullStateFlowmapExperiment",
    "ReconditionedStructuredResidual",
    "ReconditionedExperimentConfig",
    "ReconditionedFlowmapExperiment",
    "ReconditionedRunSpec",
    "ReleaseExperimentConfig",
    "ReleaseIdentifiabilityExperiment",
    "ReleaseRunSpec",
]
