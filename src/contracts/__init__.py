"""
DEPRECATED: Import from 'src.core.contracts' instead.

This module will be removed in v2.0.0.

Migration:
    # Old (deprecated)
    from src.contracts import DataContract, ModelContract

    # New (preferred)
    from src.core.contracts import DataContract, ModelContract
"""

import warnings

_CONTRACTS_EXPORTS = [
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


def __getattr__(name: str):
    """Lazy import to avoid circular imports."""
    if name in _CONTRACTS_EXPORTS:
        warnings.warn(
            f"Importing '{name}' from 'src.contracts' is deprecated. "
            f"Use 'from src.core.contracts import {name}' instead. "
            "This will be removed in v2.0.0.",
            DeprecationWarning,
            stacklevel=2,
        )
        from src.core import contracts as contracts_module

        return getattr(contracts_module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return _CONTRACTS_EXPORTS
