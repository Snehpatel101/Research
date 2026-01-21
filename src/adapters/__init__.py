"""
Adapters package - Convert canonical data to model-specific formats.

Phase 2 SNwH Implementation.

This package provides adapters that transform canonical DataFrames
into model-specific numpy array formats:
- TabularAdapter: 2D arrays (n_samples, n_features) for boosting/classical
- SequenceAdapter: 3D arrays (n_samples, seq_len, n_features) for RNN/TCN
- MultiStreamAdapter: 4D arrays (n_samples, n_tfs, seq_len, n_features) for transformers

PHASE_2 adds:
- AdapterFactory: Single entry point for adapter creation with PipelineConfig
- AdapterScaler: Fit-on-train scaling for adapter outputs (handles 2D/3D/4D)
- UnifiedDataPreparation: Complete data prep pipeline (split + transform + scale)

ALL data preparation MUST go through adapters. No bypass paths.

Usage:
    from src.adapters import get_adapter, AdapterRegistry

    # Get adapter for a model (uses contract)
    adapter = get_adapter(model_name="xgboost")
    result = adapter.transform(df)
    # result.X.shape = (n_samples, n_features)

    # Get adapter for sequence model
    adapter = get_adapter(model_name="lstm", sequence_length=60)
    result = adapter.transform(df)
    # result.X.shape = (n_sequences, 60, n_features)

    # Get adapter by ID directly
    adapter = get_adapter(adapter_id="tabular")
    result = adapter.transform(df)

    # PHASE_2: Use AdapterFactory with PipelineConfig
    from src.adapters import AdapterFactory, create_adapter_factory
    from src.core.config import PipelineConfig

    config = PipelineConfig(symbol="MES", data_path="./data", output_dir="./out")
    factory = AdapterFactory(config)

    # Single model
    result = factory.prepare_data("xgboost", df)

    # Heterogeneous ensemble
    results = factory.prepare_heterogeneous(
        models=["xgboost", "lstm", "patchtst"],
        df=df,
        additional_dfs={"5min": df_5min}
    )

    # PHASE_2: Complete data preparation with splitting and scaling
    from src.adapters import UnifiedDataPreparation, PreparedData

    prep = UnifiedDataPreparation(config)
    data = prep.prepare(df, model_name="xgboost")
    # data.X_train, data.X_val, data.X_test are scaled and ready

    # Or for heterogeneous ensemble
    all_data = prep.prepare_heterogeneous(df, models=["xgboost", "lstm"])
"""

from .base import AdapterResult, BaseAdapter
from .registry import AdapterRegistry, get_adapter

# Import adapters to register them
from .tabular import TabularAdapter
from .sequence import SequenceAdapter
from .multi_stream import MultiStreamAdapter

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

# PHASE_2: AdapterFactory for config-driven adapter creation
from .factory import AdapterFactory, create_adapter_factory

# PHASE_2: Scaling for adapter outputs
from .scaling import AdapterScaler, ScalerConfig, create_scaler

# PHASE_2: Unified data preparation (split + transform + scale)
from .preparation import PreparedData, UnifiedDataPreparation, prepare_for_model

# PHASE_2: OOF alignment utilities for heterogeneous ensembles
from .alignment import (
    AlignedOOFResult,
    OOFAligner,
    align_oof_predictions,
    compute_coverage_stats,
    validate_oof_results,
)

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
