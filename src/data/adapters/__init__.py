"""
Adapters package - Convert canonical data to model-specific formats.

This package provides adapters that transform canonical DataFrames
into model-specific numpy array formats:
- TabularAdapter: 2D arrays (n_samples, n_features) for boosting/classical
- SequenceAdapter: 3D arrays (n_samples, seq_len, n_features) for RNN/TCN
- MultiStreamAdapter: 4D arrays (n_samples, n_tfs, seq_len, n_features) for transformers

Also includes:
- AdapterFactory: Single entry point for adapter creation with PipelineConfig
- AdapterScaler: Fit-on-train scaling for adapter outputs (handles 2D/3D/4D)
- UnifiedDataPreparation: Complete data prep pipeline (split + transform + scale)

Usage:
    from src.data.adapters import get_adapter, AdapterRegistry

    adapter = get_adapter(model_name="xgboost")
    result = adapter.transform(df)
"""

# PHASE_2: OOF alignment utilities for heterogeneous ensembles
from .alignment import (
    AlignedOOFResult,
    OOFAligner,
    align_oof_predictions,
    compute_coverage_stats,
    validate_oof_results,
)
from .base import AdapterResult, BaseAdapter

# PHASE_2: AdapterFactory for config-driven adapter creation
from .factory import AdapterFactory, create_adapter_factory

# Multi-Resolution 4D Adapter (migrated from phase1)
from .multi_resolution import (
    MultiResolution4DAdapter,
    MultiResolution4DConfig,
    MultiResolution4DDataset,
    create_multi_resolution_dataset,
)
from .multi_resolution_utils import (
    DEFAULT_MTF_FEATURES,
    DEFAULT_MTF_TIMEFRAMES,
)
from .multi_stream import MTF_TIMEFRAMES, MultiStreamAdapter

# PHASE_2: Unified data preparation (split + transform + scale)
from .preparation import PreparedData, UnifiedDataPreparation, prepare_for_model
from .registry import AdapterRegistry, get_adapter

# PHASE_2: Scaling for adapter outputs
from .scaling import AdapterScaler, ScalerConfig, create_scaler
from .sequence import SequenceAdapter

# Import adapters to register them
from .tabular import TabularAdapter

__all__ = [
    # Registry
    "AdapterRegistry",
    "get_adapter",
    # Base classes
    "BaseAdapter",
    "AdapterResult",
    # Adapters
    "TabularAdapter",
    "SequenceAdapter",
    "MultiStreamAdapter",
    # Multi-Resolution 4D
    "MultiResolution4DAdapter",
    "MultiResolution4DConfig",
    "MultiResolution4DDataset",
    "create_multi_resolution_dataset",
    "DEFAULT_MTF_FEATURES",
    "DEFAULT_MTF_TIMEFRAMES",
    # Multi-Stream 4D (from raw MTF store)
    "MTF_TIMEFRAMES",
    # PHASE_2: Factory
    "AdapterFactory",
    "create_adapter_factory",
    # PHASE_2: Scaling
    "AdapterScaler",
    "ScalerConfig",
    "create_scaler",
    # PHASE_2: Unified Preparation
    "PreparedData",
    "UnifiedDataPreparation",
    "prepare_for_model",
    # PHASE_2: OOF Alignment
    "AlignedOOFResult",
    "OOFAligner",
    "align_oof_predictions",
    "compute_coverage_stats",
    "validate_oof_results",
]
