"""
Data Domain - Consolidates data lifecycle modules.

This domain provides a unified interface for data processing:
- pipeline: Data preparation pipeline stages
- adapters: Model-specific data format adapters
- features: Feature engineering and computation
- labeling: Label generation (triple-barrier, etc.)
- store: Feature storage and versioning

New import paths (preferred):
    from src.data.pipeline import PipelineRunner, DataConfig
    from src.data.adapters import TabularAdapter, SequenceAdapter
    from src.data.features import compute_all_features
    from src.data.labeling import TripleBarrierLabeler
    from src.data.store import FeatureStore

Legacy import paths (still work, will be deprecated in v2.0):
    from src.pipeline import PipelineRunner
    from src.adapters import TabularAdapter
    from src.features import compute_all_features
    from src.labeling import TripleBarrierLabeler
    from src.feature_store import FeatureStore
"""

# Re-export from pipeline
from src.pipeline import (
    PipelineRunner,
    DataConfig,
    PipelineStage,
    StageResult,
    StageStatus,
    get_stage_definitions,
    get_stage_order,
)

# Re-export from adapters
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

# Re-export from features
from src.features import (
    FeatureDefinition,
    FEATURE_REGISTRY,
    get_features_by_families,
    get_features_by_model,
    get_feature_families,
    compute_all_features,
    compute_features_by_family,
    compute_features_by_names,
    compute_single_feature,
    FEATURE_COMPUTE_MAP,
    get_all_feature_names,
    get_features_in_family,
    MTFConfig,
    MTFFeatureComputer,
    compute_mtf_features,
    get_mtf_feature_names,
    validate_mtf_config,
    resample_ohlcv,
    ModelFeatureStrategy,
    MODEL_FEATURE_STRATEGIES,
    get_strategy_for_model,
    get_baseline_features,
    OptimizationResult,
    FeatureOptimizer,
    optimize_features_for_model,
    suggest_features,
    ResolvedFeatureSet,
    FeatureStrategyManager,
    get_features_for_model,
    FeatureSelectionResult,
    FeatureSelector,
    select_features,
    FeaturePruningResult,
    FeaturePruner,
    prune_features,
    prune_correlated_features,
)

# Re-export from labeling
from src.labeling import (
    LabelingResult,
    LabelingStrategy,
    LabelingType,
    TripleBarrierConfig,
    TripleBarrierLabeler,
    triple_barrier_numba,
    triple_barrier_numba_with_costs,
    LabelOptimizationResult,
    LabelOptimizer,
    optimize_labels,
)

# Re-export from feature_store
from src.feature_store import (
    FeatureStore,
    FeatureStoreError,
    FeatureNotFoundError,
    FeatureIntegrityError,
    FeatureCache,
    CacheMetadata,
    compute_file_checksum,
    compute_dataframe_checksum,
    LineageTracker,
    FeatureLineage,
    DataSource,
    Transformation,
    TransformationType,
    SemanticVersion,
    VersionInfo,
    VersionManager,
    compute_schema_hash,
    compute_config_hash,
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
    "FeatureDefinition",
    "FEATURE_REGISTRY",
    "get_features_by_families",
    "get_features_by_model",
    "get_feature_families",
    "compute_all_features",
    "compute_features_by_family",
    "compute_features_by_names",
    "compute_single_feature",
    "FEATURE_COMPUTE_MAP",
    "get_all_feature_names",
    "get_features_in_family",
    "MTFConfig",
    "MTFFeatureComputer",
    "compute_mtf_features",
    "get_mtf_feature_names",
    "validate_mtf_config",
    "resample_ohlcv",
    "ModelFeatureStrategy",
    "MODEL_FEATURE_STRATEGIES",
    "get_strategy_for_model",
    "get_baseline_features",
    "OptimizationResult",
    "FeatureOptimizer",
    "optimize_features_for_model",
    "suggest_features",
    "ResolvedFeatureSet",
    "FeatureStrategyManager",
    "get_features_for_model",
    "FeatureSelectionResult",
    "FeatureSelector",
    "select_features",
    "FeaturePruningResult",
    "FeaturePruner",
    "prune_features",
    "prune_correlated_features",
    # Labeling
    "LabelingResult",
    "LabelingStrategy",
    "LabelingType",
    "TripleBarrierConfig",
    "TripleBarrierLabeler",
    "triple_barrier_numba",
    "triple_barrier_numba_with_costs",
    "LabelOptimizationResult",
    "LabelOptimizer",
    "optimize_labels",
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
