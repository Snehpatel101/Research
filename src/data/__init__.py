"""
Data Domain - Consolidates data lifecycle modules.

This domain provides a unified interface for data processing:
- pipeline: Data preparation pipeline stages (the live feature engine is
  src/data/pipeline/stages/features/)
- adapters: Model-specific data format adapters
- features: Per-model feature-set resolution (strategies) + event utilities
- labeling: Label generation (triple-barrier, etc.)
- store: Feature storage and versioning

Import paths:
    from src.data.pipeline import PipelineRunner, DataConfig
    from src.data.adapters import TabularAdapter, SequenceAdapter
    from src.data.features import get_features_for_model
    from src.data.labeling import TripleBarrierLabeler
    from src.data.store import FeatureStore
"""

# Re-export from pipeline (now in data/pipeline)
# Re-export from adapters (now in data/adapters)
from .adapters import (
    DEFAULT_MTF_FEATURES,
    DEFAULT_MTF_TIMEFRAMES,
    AdapterFactory,
    AdapterRegistry,
    AdapterResult,
    AdapterScaler,
    AlignedOOFResult,
    BaseAdapter,
    MultiResolution4DAdapter,
    MultiResolution4DConfig,
    MultiResolution4DDataset,
    MultiStreamAdapter,
    OOFAligner,
    PreparedData,
    ScalerConfig,
    SequenceAdapter,
    TabularAdapter,
    UnifiedDataPreparation,
    align_oof_predictions,
    compute_coverage_stats,
    create_adapter_factory,
    create_multi_resolution_dataset,
    create_scaler,
    get_adapter,
    prepare_for_model,
    validate_oof_results,
)

# Re-export from features (now in data/features)
from .features import (
    MODEL_FEATURE_STRATEGIES,
    FeatureStrategyManager,
    ModelFeatureStrategy,
    ResolvedFeatureSet,
    get_baseline_features,
    get_features_for_model,
    get_strategy_for_model,
)

# Re-export from labeling (now in data/labeling)
from .labeling import (
    LabelingResult,
    LabelingStrategy,
    LabelingType,
    TripleBarrierConfig,
    TripleBarrierLabeler,
    triple_barrier_numba,
    triple_barrier_numba_with_costs,
)
from .pipeline import (
    DataConfig,
    PipelineRunner,
    PipelineStage,
    StageResult,
    StageStatus,
    get_stage_definitions,
    get_stage_order,
)

# Re-export from feature_store (now in data/store)
from .store import (
    CacheMetadata,
    DataSource,
    FeatureCache,
    FeatureIntegrityError,
    FeatureLineage,
    FeatureNotFoundError,
    FeatureStore,
    FeatureStoreError,
    LineageTracker,
    SemanticVersion,
    Transformation,
    TransformationType,
    VersionInfo,
    VersionManager,
    compute_config_hash,
    compute_dataframe_checksum,
    compute_file_checksum,
    compute_schema_hash,
)

__all__ = [
    # Pipeline
    "PipelineRunner",
    "DataConfig",
    "PipelineStage",
    "StageResult",
    "StageStatus",
    "get_stage_definitions",
    "get_stage_order",
    # Adapters
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
    # Features
    "ModelFeatureStrategy",
    "MODEL_FEATURE_STRATEGIES",
    "get_strategy_for_model",
    "get_baseline_features",
    "ResolvedFeatureSet",
    "FeatureStrategyManager",
    "get_features_for_model",
    # Labeling
    "LabelingResult",
    "LabelingStrategy",
    "LabelingType",
    "TripleBarrierConfig",
    "TripleBarrierLabeler",
    "triple_barrier_numba",
    "triple_barrier_numba_with_costs",
    # Feature Store
    "FeatureStore",
    "FeatureStoreError",
    "FeatureNotFoundError",
    "FeatureIntegrityError",
    "FeatureCache",
    "CacheMetadata",
    "compute_file_checksum",
    "compute_dataframe_checksum",
    "LineageTracker",
    "FeatureLineage",
    "DataSource",
    "Transformation",
    "TransformationType",
    "SemanticVersion",
    "VersionInfo",
    "VersionManager",
    "compute_schema_hash",
    "compute_config_hash",
]
