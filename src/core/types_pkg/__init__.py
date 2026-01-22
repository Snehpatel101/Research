"""
Core Types - Re-export from src.core.types module and src.contracts.

This subpackage consolidates type definitions from:
- src.core.types (enums, type aliases)
- src.contracts (data contracts, model contracts)

New import path:
    from src.core.types import DataRank, ModelContract, DataContract

Legacy import paths (still work):
    from src.core.types import DataRank
    from src.core.contracts import ModelContract, DataContract
"""

# Re-export core types
# Re-export from contracts
from src.core.contracts import (
    DATA_SCHEMA,
    MODEL_CONTRACTS,
    ArtifactManifest,
    DataContractSchema,
    FeatureMode,
    ModelContract,
    MTFMode,
    get_model_contract,
    get_models_by_mtf_mode,
    get_models_by_rank,
    get_models_requiring_scaling,
    list_model_contracts,
)
from src.core.contracts import (
    DataContract as ContractDataContract,
)
from src.core.types import (
    AdapterType,
    Array1D,
    Array2D,
    Array3D,
    Array4D,
    CVMethod,
    DataRank,
    DatetimeIndex,
    FeatureFamily,
    Features,
    Index,
    LabelingMethod,
    Labels,
    ModelFamily,
    ModelType,
    TrainingMode,
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
