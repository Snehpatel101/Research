"""
Core package - Foundational interfaces, types, and constants.

PHASE_0: Clean ML Factory Foundation.

This package is the SINGLE SOURCE OF TRUTH for:
- Abstract interfaces (ModelContract, AdapterContract, DataContract)
- Type definitions (DataRank, ModelFamily, FeatureFamily, etc.)
- Constants (CANONICAL_TIMEFRAMES, MODEL_FAMILIES, etc.)
- Validation utilities
- PipelineConfig (centralized configuration)
- Pipeline lineage tracking (PipelineLineage, DatasetChecksum)
- TimeSeriesDataContainer (unified data container for ML)

Usage:
    from src.core import (
        # Config
        PipelineConfig,
        quick_config,
        production_config,

        # Types
        DataRank,
        ModelFamily,
        FeatureFamily,
        TrainingMode,
        CVMethod,

        # Interfaces
        ModelContract,
        AdapterContract,
        DataContract,
        AdapterResult,
        TrainingResult,
        OOFResult,

        # Constants
        CANONICAL_TIMEFRAMES,
        MODEL_FAMILIES,
        MODEL_DATA_RANKS,
        ALL_MODELS,

        # Validation
        ValidationError,
        validate_input_shape,
        validate_dataframe,

        # Lineage
        PipelineLineage,
        DatasetChecksum,
        validate_dataset_checksum,

        # Container
        TimeSeriesDataContainer,
        DataContainerConfig,
        SplitData,

        # Data Contract
        DatasetContract,
        SplitDatasetContract,
    )
"""

# =============================================================================
# TYPES - Enums and type aliases
# =============================================================================
from src.core.types import (
    DataRank,
    ModelFamily,
    FeatureFamily,
    TrainingMode,
    CVMethod,
    AdapterType,
    LabelingMethod,
    # Type aliases
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

# =============================================================================
# CONSTANTS - Canonical values
# =============================================================================
from src.core.constants import (
    # Timeframes
    CANONICAL_TIMEFRAMES,
    BASE_TIMEFRAME,
    DEFAULT_MTF_TIMEFRAMES,
    # Horizons
    DEFAULT_HORIZONS,
    DEFAULT_HORIZON,
    # Splits
    DEFAULT_SPLIT_RATIOS,
    # Purge/Embargo
    DEFAULT_PURGE_BARS,
    DEFAULT_EMBARGO_BARS,
    # Models
    MODEL_FAMILIES,
    ALL_MODELS,
    MODEL_TO_FAMILY,
    MODEL_DATA_RANKS,
    MODEL_ADAPTER_MAP,
    # Features
    FEATURE_FAMILY_COUNTS,
    TOTAL_BASE_FEATURES,
    MTF_FEATURES_PER_TF,
    MTF_TOTAL_FEATURES,
    # Sequence defaults
    DEFAULT_SEQUENCE_LENGTH,
    DEFAULT_HIDDEN_SIZE,
    DEFAULT_NUM_LAYERS,
    DEFAULT_DROPOUT,
    # Training defaults
    DEFAULT_BATCH_SIZE,
    DEFAULT_MAX_EPOCHS,
    DEFAULT_LEARNING_RATE,
    DEFAULT_EARLY_STOPPING_PATIENCE,
    DEFAULT_N_SPLITS,
    # Optuna defaults
    DEFAULT_LABEL_OPTIMIZATION_TRIALS,
    DEFAULT_FEATURE_SELECTION_TRIALS,
    DEFAULT_FEATURE_PRUNING_TRIALS,
    DEFAULT_HYPERPARAM_TRIALS,
    DEFAULT_OPTUNA_RANDOM_STATE,
    DEFAULT_MIN_FEATURES,
    # OHLCV
    OHLCV_COLUMNS,
    REQUIRED_COLUMNS,
    # Labels
    LABEL_CLASSES,
    N_CLASSES,
    # Helper functions
    get_models_for_family,
    get_models_for_rank,
    get_adapter_for_model,
)

# =============================================================================
# INTERFACES - Abstract contracts
# =============================================================================
from src.core.interfaces import (
    # Result types
    AdapterResult,
    PredictionResult,
    TrainingResult,
    OOFResult,
    # Contracts
    DataContract,
    ModelContract,
    AdapterContract,
)

# =============================================================================
# VALIDATION - Input validation
# =============================================================================
from src.core.validation import (
    # Exception
    ValidationError,
    # Array validation
    validate_input_shape,
    validate_labels,
    validate_probabilities,
    # Feature validation
    validate_features,
    # DataFrame validation
    validate_dataframe,
    validate_ohlcv,
    # Model validation
    validate_model_name,
    validate_model_list,
    # Path validation
    validate_path_exists,
    # Timeframe validation
    validate_timeframe,
    validate_timeframe_list,
)

# =============================================================================
# CONFIG - Centralized pipeline configuration
# =============================================================================
from src.core.config import (
    PipelineConfig,
    quick_config,
    production_config,
    research_config,
)

# =============================================================================
# LINEAGE - Pipeline lineage tracking
# =============================================================================
from src.core.lineage import (
    DatasetChecksum,
    PipelineLineage,
    compute_dataframe_checksum,
    compute_file_checksum,
    create_dataset_checksum,
    validate_dataset_checksum,
)

# =============================================================================
# EXISTING EXPORTS (preserved from original)
# =============================================================================
from src.core.defaults import DEFAULTS, GlobalDefaults, as_dict, get_default
from src.core.paths import (
    CONFIG_DIR,
    CONFIG_MODELS_DIR,
    CONFIG_PIPELINE_DIR,
    CONFIG_ROOT,
    CV_CONFIG_PATH,
    DATA_DIR,
    EXPERIMENTS_DIR,
    PROJECT_ROOT,
    RAW_DATA_DIR,
    RESULTS_DIR,
    RUNS_DIR,
    TRAINING_CONFIG_PATH,
)
from src.core.reproducibility import (
    ReproducibilityConfig,
    ReproducibilityInfo,
    ensure_reproducibility,
    get_reproducibility_info,
    get_worker_init_fn,
    set_all_seeds,
)

# =============================================================================
# CONTAINER - TimeSeriesDataContainer (unified data container)
# =============================================================================
from src.core.container import (
    TimeSeriesDataContainer,
    DataContainerConfig,
    SplitData,
    VALID_SPLITS,
    INVALID_LABEL,
)

# =============================================================================
# DATA CONTRACT - Explicit data passing between pipeline stages
# =============================================================================
from src.core.data_contract import (
    DatasetContract,
    SplitDatasetContract,
)

# =============================================================================
# ABSORBED: src/contracts - Model and data contracts
# =============================================================================
from src.contracts import (
    FeatureMode,
    MTFMode,
    DataContractSchema,
    DATA_SCHEMA,
    MODEL_CONTRACTS,
    get_model_contract,
    list_model_contracts,
    get_models_by_rank,
    get_models_requiring_scaling,
    get_models_by_mtf_mode,
    ArtifactManifest,
)

# =============================================================================
# ABSORBED: src/config - Unified configuration
# =============================================================================
from src.config import (
    UnifiedConfig,
    get_unified_config,
    set_unified_config,
    reset_unified_config,
    get_config_value,
    get_config_value_strict,
    validate_config,
    validate_config_file,
    GlobalConfig,
    load_global_config,
    get_global_config,
    TrainerConfig,
    detect_environment,
)

# =============================================================================
# ABSORBED: src/common - Timeframes, horizons, split ratios
# =============================================================================
from src.common import (
    HorizonConfig,
    HORIZONS,
    SUPPORTED_HORIZONS,
    ACTIVE_HORIZONS,
    LABEL_HORIZONS,
    LOOKBACK_HORIZONS,
    validate_horizons,
    get_scaled_horizons,
    auto_scale_purge_embargo,
    get_default_barrier_params_for_horizon,
    ALL_CANONICAL_TIMEFRAMES,
    EXTENDED_TIMEFRAMES,
    FULL_9TF_LADDER,
    SUPPORTED_TIMEFRAMES,
    TIMEFRAME_ALIASES,
    TIMEFRAME_TO_FREQ,
    TIMEFRAME_TO_MINUTES,
    get_timeframe_minutes,
    is_valid_timeframe,
    normalize_timeframe,
    normalize_timeframe_list,
    timeframe_to_minutes,
    DEFAULT_TRAIN_RATIO,
    DEFAULT_VAL_RATIO,
    DEFAULT_TEST_RATIO,
    validate_split_ratios,
)

# =============================================================================
# ABSORBED: src/utils - Memory, cache, notebook utilities
# =============================================================================
from src.utils import (
    CacheManager,
    CacheConfig,
    CacheEntry,
    CacheStats,
    MemoryInfo,
    check_available_memory,
    check_memory_sufficient,
    estimate_array_size,
    estimate_object_size,
    get_memory_info,
    log_memory_usage,
    memory_logged,
    get_global_cache,
    DataCache,
    DataCacheConfig,
    cached_result,
    get_global_data_cache,
    CheckpointManager,
    is_colab,
    setup_environment,
    setup_colab_environment,
)

# =============================================================================
# ABSORBED: src/coordination - Temporal alignment utilities
# =============================================================================
from src.coordination import (
    align_to_anchor,
    apply_mtf_lag,
    compute_sequence_offset,
    validate_timestamp_alignment,
    TimeframeCoordinator,
    TimeframeData,
)

# =============================================================================
# ALL EXPORTS
# =============================================================================
__all__ = [
    # =========================================================================
    # CONFIG (THE main entry point)
    # =========================================================================
    "PipelineConfig",
    "quick_config",
    "production_config",
    "research_config",

    # =========================================================================
    # TYPES - Enums
    # =========================================================================
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

    # =========================================================================
    # INTERFACES - Contracts
    # =========================================================================
    "DataContract",
    "ModelContract",
    "AdapterContract",

    # Result types
    "AdapterResult",
    "PredictionResult",
    "TrainingResult",
    "OOFResult",

    # =========================================================================
    # CONSTANTS
    # =========================================================================
    # Timeframes
    "CANONICAL_TIMEFRAMES",
    "BASE_TIMEFRAME",
    "DEFAULT_MTF_TIMEFRAMES",

    # Horizons
    "DEFAULT_HORIZONS",
    "DEFAULT_HORIZON",

    # Splits
    "DEFAULT_SPLIT_RATIOS",

    # Purge/Embargo
    "DEFAULT_PURGE_BARS",
    "DEFAULT_EMBARGO_BARS",

    # Models
    "MODEL_FAMILIES",
    "ALL_MODELS",
    "MODEL_TO_FAMILY",
    "MODEL_DATA_RANKS",
    "MODEL_ADAPTER_MAP",

    # Features
    "FEATURE_FAMILY_COUNTS",
    "TOTAL_BASE_FEATURES",
    "MTF_FEATURES_PER_TF",
    "MTF_TOTAL_FEATURES",

    # Sequence defaults
    "DEFAULT_SEQUENCE_LENGTH",
    "DEFAULT_HIDDEN_SIZE",
    "DEFAULT_NUM_LAYERS",
    "DEFAULT_DROPOUT",

    # Training defaults
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_MAX_EPOCHS",
    "DEFAULT_LEARNING_RATE",
    "DEFAULT_EARLY_STOPPING_PATIENCE",
    "DEFAULT_N_SPLITS",

    # Optuna defaults
    "DEFAULT_LABEL_OPTIMIZATION_TRIALS",
    "DEFAULT_FEATURE_SELECTION_TRIALS",
    "DEFAULT_FEATURE_PRUNING_TRIALS",
    "DEFAULT_HYPERPARAM_TRIALS",
    "DEFAULT_OPTUNA_RANDOM_STATE",
    "DEFAULT_MIN_FEATURES",

    # OHLCV
    "OHLCV_COLUMNS",
    "REQUIRED_COLUMNS",

    # Labels
    "LABEL_CLASSES",
    "N_CLASSES",

    # Helper functions
    "get_models_for_family",
    "get_models_for_rank",
    "get_adapter_for_model",

    # =========================================================================
    # VALIDATION
    # =========================================================================
    "ValidationError",
    "validate_input_shape",
    "validate_labels",
    "validate_probabilities",
    "validate_features",
    "validate_dataframe",
    "validate_ohlcv",
    "validate_model_name",
    "validate_model_list",
    "validate_path_exists",
    "validate_timeframe",
    "validate_timeframe_list",

    # =========================================================================
    # LINEAGE
    # =========================================================================
    "DatasetChecksum",
    "PipelineLineage",
    "compute_dataframe_checksum",
    "compute_file_checksum",
    "create_dataset_checksum",
    "validate_dataset_checksum",

    # =========================================================================
    # LEGACY EXPORTS (from original core package)
    # =========================================================================
    # Paths
    "PROJECT_ROOT",
    "DATA_DIR",
    "RAW_DATA_DIR",
    "RESULTS_DIR",
    "RUNS_DIR",
    "EXPERIMENTS_DIR",
    "CONFIG_ROOT",
    "CONFIG_MODELS_DIR",
    "CONFIG_PIPELINE_DIR",
    "CONFIG_DIR",
    "TRAINING_CONFIG_PATH",
    "CV_CONFIG_PATH",

    # Defaults
    "DEFAULTS",
    "GlobalDefaults",
    "get_default",
    "as_dict",

    # Reproducibility
    "ReproducibilityConfig",
    "ReproducibilityInfo",
    "set_all_seeds",
    "get_reproducibility_info",
    "ensure_reproducibility",
    "get_worker_init_fn",

    # =========================================================================
    # CONTAINER - TimeSeriesDataContainer
    # =========================================================================
    "TimeSeriesDataContainer",
    "DataContainerConfig",
    "SplitData",
    "VALID_SPLITS",
    "INVALID_LABEL",

    # =========================================================================
    # DATA CONTRACT - Explicit data passing between pipeline stages
    # =========================================================================
    "DatasetContract",
    "SplitDatasetContract",

    # =========================================================================
    # ABSORBED: src/contracts
    # =========================================================================
    "FeatureMode",
    "MTFMode",
    "DataContractSchema",
    "DATA_SCHEMA",
    "MODEL_CONTRACTS",
    "get_model_contract",
    "list_model_contracts",
    "get_models_by_rank",
    "get_models_requiring_scaling",
    "get_models_by_mtf_mode",
    "ArtifactManifest",

    # =========================================================================
    # ABSORBED: src/config
    # =========================================================================
    "UnifiedConfig",
    "get_unified_config",
    "set_unified_config",
    "reset_unified_config",
    "get_config_value",
    "get_config_value_strict",
    "validate_config",
    "validate_config_file",
    "GlobalConfig",
    "load_global_config",
    "get_global_config",
    "TrainerConfig",
    "detect_environment",

    # =========================================================================
    # ABSORBED: src/common
    # =========================================================================
    "HorizonConfig",
    "HORIZONS",
    "SUPPORTED_HORIZONS",
    "ACTIVE_HORIZONS",
    "LABEL_HORIZONS",
    "LOOKBACK_HORIZONS",
    "validate_horizons",
    "get_scaled_horizons",
    "auto_scale_purge_embargo",
    "get_default_barrier_params_for_horizon",
    "ALL_CANONICAL_TIMEFRAMES",
    "EXTENDED_TIMEFRAMES",
    "FULL_9TF_LADDER",
    "SUPPORTED_TIMEFRAMES",
    "TIMEFRAME_ALIASES",
    "TIMEFRAME_TO_FREQ",
    "TIMEFRAME_TO_MINUTES",
    "get_timeframe_minutes",
    "is_valid_timeframe",
    "normalize_timeframe",
    "normalize_timeframe_list",
    "timeframe_to_minutes",
    "DEFAULT_TRAIN_RATIO",
    "DEFAULT_VAL_RATIO",
    "DEFAULT_TEST_RATIO",
    "validate_split_ratios",

    # =========================================================================
    # ABSORBED: src/utils
    # =========================================================================
    "CacheManager",
    "CacheConfig",
    "CacheEntry",
    "CacheStats",
    "MemoryInfo",
    "check_available_memory",
    "check_memory_sufficient",
    "estimate_array_size",
    "estimate_object_size",
    "get_memory_info",
    "log_memory_usage",
    "memory_logged",
    "get_global_cache",
    "DataCache",
    "DataCacheConfig",
    "cached_result",
    "get_global_data_cache",
    "CheckpointManager",
    "is_colab",
    "setup_environment",
    "setup_colab_environment",

    # =========================================================================
    # ABSORBED: src/coordination
    # =========================================================================
    "align_to_anchor",
    "apply_mtf_lag",
    "compute_sequence_offset",
    "validate_timestamp_alignment",
    "TimeframeCoordinator",
    "TimeframeData",
]
