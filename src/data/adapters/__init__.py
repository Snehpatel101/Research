"""
Data Adapters - Re-export from src.adapters.

New import path:
    from src.data.adapters import TabularAdapter, SequenceAdapter

Legacy import path (still works):
    from src.adapters import TabularAdapter, SequenceAdapter
"""

from src.adapters import (
    AdapterRegistry,
    get_adapter,
    BaseAdapter,
    AdapterResult,
    TabularAdapter,
    SequenceAdapter,
    MultiStreamAdapter,
    MultiResolution4DAdapter,
    MultiResolution4DConfig,
    MultiResolution4DDataset,
    create_multi_resolution_dataset,
    DEFAULT_MTF_FEATURES,
    DEFAULT_MTF_TIMEFRAMES,
    AdapterFactory,
    create_adapter_factory,
    AdapterScaler,
    ScalerConfig,
    create_scaler,
    PreparedData,
    UnifiedDataPreparation,
    prepare_for_model,
    AlignedOOFResult,
    OOFAligner,
    align_oof_predictions,
    compute_coverage_stats,
    validate_oof_results,
)

__all__ = [
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
