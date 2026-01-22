"""
Core Types - Re-export from src.core.types module and src.contracts.

This subpackage consolidates type definitions from:
- src.core.types (enums, type aliases)
- src.contracts (data contracts, model contracts)

New import path:
    from src.core.types import DataRank, ModelContract, DataContract

Legacy import paths (still work):
    from src.core.types import DataRank
    from src.contracts import ModelContract, DataContract
"""

# Re-export core types
from src.core.types import (
    DataRank,
    ModelFamily,
    FeatureFamily,
    TrainingMode,
    CVMethod,
    AdapterType,
    LabelingMethod,
    Features,
    Labels,
    ModelType,
    Array1D,
    Array2D,
    Array3D,
    Array4D,
    DatetimeIndex,
    Index,
)

# Re-export from contracts
from src.core.contracts import (
    FeatureMode,
    MTFMode,
    DataContractSchema,
    DATA_SCHEMA,
    DataContract as ContractDataContract,
    ModelContract,
    MODEL_CONTRACTS,
    get_model_contract,
    list_model_contracts,
    get_models_by_rank,
    get_models_requiring_scaling,
    get_models_by_mtf_mode,
    ArtifactManifest,
)

__all__ = [
    # Core types (enums)
    "DataRank",
    "ModelFamily",
    "FeatureFamily",
    "TrainingMode",
    "CVMethod",
    "AdapterType",
    "LabelingMethod",
    # Type aliases
    "Features",
    "Labels",
    "ModelType",
    "Array1D",
    "Array2D",
    "Array3D",
    "Array4D",
    "DatetimeIndex",
    "Index",
    # From contracts
    "FeatureMode",
    "MTFMode",
    "DataContractSchema",
    "DATA_SCHEMA",
    "ContractDataContract",
    "ModelContract",
    "MODEL_CONTRACTS",
    "get_model_contract",
    "list_model_contracts",
    "get_models_by_rank",
    "get_models_requiring_scaling",
    "get_models_by_mtf_mode",
    "ArtifactManifest",
]
