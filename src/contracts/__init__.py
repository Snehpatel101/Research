"""
Contracts package - Canonical contracts for data, models, and artifacts.

This package defines the foundational contracts that ensure
compatibility and reproducibility across the ML pipeline.

Phase 0 of the SNwH (Unified Multi-Timeframe Model Factory) implementation.

Usage:
    from src.contracts import (
        # Data contract
        DataRank,
        FeatureMode,
        MTFMode,
        DataContract,
        DATA_SCHEMA,

        # Model contract
        ModelContract,
        MODEL_CONTRACTS,
        get_model_contract,

        # Artifact manifest
        ArtifactManifest,
    )

Example:
    >>> from src.contracts import get_model_contract, DataContract, DataRank
    >>>
    >>> # Get contract for a model
    >>> contract = get_model_contract("xgboost")
    >>> print(contract.input_rank)  # DataRank.TABULAR_2D
    >>> print(contract.requires_scaling)  # False
    >>>
    >>> # Create data contract from array
    >>> import numpy as np
    >>> X = np.random.randn(1000, 100)
    >>> data_contract = DataContract.from_array(
    ...     X, symbol="MES", timeframe="15min", horizon=20, split="train"
    ... )
    >>>
    >>> # Validate compatibility
    >>> is_valid, issues = contract.validate_data_contract(data_contract)
"""

from .data_contract import (
    DataRank,
    FeatureMode,
    MTFMode,
    DataContractSchema,
    DATA_SCHEMA,
    DataContract,
)
from .model_contract import (
    ModelContract,
    MODEL_CONTRACTS,
    get_model_contract,
    list_model_contracts,
    get_models_by_rank,
    get_models_requiring_scaling,
    get_models_by_mtf_mode,
)
from .artifact_manifest import (
    ArtifactManifest,
)

__all__ = [
    # Data contract
    "DataRank",
    "FeatureMode",
    "MTFMode",
    "DataContractSchema",
    "DATA_SCHEMA",
    "DataContract",
    # Model contract
    "ModelContract",
    "MODEL_CONTRACTS",
    "get_model_contract",
    "list_model_contracts",
    "get_models_by_rank",
    "get_models_requiring_scaling",
    "get_models_by_mtf_mode",
    # Artifact manifest
    "ArtifactManifest",
]
