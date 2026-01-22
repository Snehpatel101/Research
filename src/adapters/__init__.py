"""
DEPRECATED: Import from 'src.data.adapters' instead.

This module will be removed in v2.0.0.

Migration:
    # Old (deprecated)
    from src.adapters import TabularAdapter, get_adapter

    # New (preferred)
    from src.data.adapters import TabularAdapter, get_adapter
"""

import warnings

_ADAPTERS_EXPORTS = [
    "AdapterRegistry",
    "get_adapter",
    "BaseAdapter",
    "AdapterResult",
    "TabularAdapter",
    "SequenceAdapter",
    "MultiStreamAdapter",
    "MultiResolution4DAdapter",
    "MultiResolution4DConfig",
    "MultiResolution4DDataset",
    "create_multi_resolution_dataset",
    "DEFAULT_MTF_FEATURES",
    "DEFAULT_MTF_TIMEFRAMES",
    "AdapterFactory",
    "create_adapter_factory",
    "AdapterScaler",
    "ScalerConfig",
    "create_scaler",
    "PreparedData",
    "UnifiedDataPreparation",
    "prepare_for_model",
    "AlignedOOFResult",
    "OOFAligner",
    "align_oof_predictions",
    "compute_coverage_stats",
    "validate_oof_results",
]


def __getattr__(name: str):
    """Lazy import to avoid circular imports."""
    if name in _ADAPTERS_EXPORTS:
        warnings.warn(
            f"Importing '{name}' from 'src.adapters' is deprecated. "
            f"Use 'from src.data.adapters import {name}' instead. "
            "This will be removed in v2.0.0.",
            DeprecationWarning,
            stacklevel=2,
        )
        from src.data import adapters as adapters_module

        return getattr(adapters_module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return _ADAPTERS_EXPORTS
