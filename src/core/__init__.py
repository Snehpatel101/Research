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
# NOTE: src.config imports are NOT included here to avoid circular imports.
# Import directly from src.config when needed:
#   from src.config import GlobalConfig, TrainerConfig, etc.

# =============================================================================
# ABSORBED: src/common - Timeframes, horizons, split ratios
# =============================================================================
from src.core.common import (
    ACTIVE_HORIZONS,
    ALL_CANONICAL_TIMEFRAMES,
    DEFAULT_TEST_RATIO,
    DEFAULT_TRAIN_RATIO,
    DEFAULT_VAL_RATIO,
    EXTENDED_TIMEFRAMES,
    FULL_9TF_LADDER,
    HORIZONS,
    LABEL_HORIZONS,
    LOOKBACK_HORIZONS,
    SUPPORTED_HORIZONS,
    SUPPORTED_TIMEFRAMES,
    TIMEFRAME_ALIASES,
    TIMEFRAME_TO_FREQ,
    TIMEFRAME_TO_MINUTES,
    HorizonConfig,
    auto_scale_purge_embargo,
    get_default_barrier_params_for_horizon,
    get_scaled_horizons,
    get_timeframe_minutes,
    is_valid_timeframe,
    normalize_timeframe,
    normalize_timeframe_list,
    timeframe_to_minutes,
    validate_horizons,
    validate_split_ratios,
)

# =============================================================================
# CONFIG - Centralized pipeline configuration
# =============================================================================
from src.core.config import (
    PipelineConfig,
    production_config,
    quick_config,
    research_config,
)

# =============================================================================
# CONSTANTS - Canonical values
# =============================================================================
from src.core.constants import (
    ALL_MODELS,
    BASE_TIMEFRAME,
    # Timeframes
    CANONICAL_TIMEFRAMES,
    # Training defaults
    DEFAULT_BATCH_SIZE,
    DEFAULT_DROPOUT,
    DEFAULT_EARLY_STOPPING_PATIENCE,
    DEFAULT_EMBARGO_BARS,
    DEFAULT_FEATURE_PRUNING_TRIALS,
    DEFAULT_FEATURE_SELECTION_TRIALS,
    DEFAULT_HIDDEN_SIZE,
    DEFAULT_HORIZON,
    # Horizons
    DEFAULT_HORIZONS,
    DEFAULT_HYPERPARAM_TRIALS,
    # Optuna defaults
    DEFAULT_LABEL_OPTIMIZATION_TRIALS,
    DEFAULT_LEARNING_RATE,
    DEFAULT_MAX_EPOCHS,
    DEFAULT_MIN_FEATURES,
    DEFAULT_MTF_TIMEFRAMES,
    DEFAULT_N_SPLITS,
    DEFAULT_NUM_LAYERS,
    DEFAULT_OPTUNA_RANDOM_STATE,
    # Purge/Embargo
    DEFAULT_PURGE_BARS,
    # Sequence defaults
    DEFAULT_SEQUENCE_LENGTH,
    # Splits
    DEFAULT_SPLIT_RATIOS,
    # Features
    FEATURE_FAMILY_COUNTS,
    # Labels
    LABEL_CLASSES,
    MODEL_ADAPTER_MAP,
    MODEL_DATA_RANKS,
    # Models
    MODEL_FAMILIES,
    MODEL_TO_FAMILY,
    MTF_FEATURES_PER_TF,
    MTF_TOTAL_FEATURES,
    N_CLASSES,
    # OHLCV
    OHLCV_COLUMNS,
    REQUIRED_COLUMNS,
    TOTAL_BASE_FEATURES,
    get_adapter_for_model,
    # Helper functions
    get_models_for_family,
    get_models_for_rank,
)

# =============================================================================
# CONTAINER - TimeSeriesDataContainer (unified data container)
# =============================================================================
from src.core.container import (
    INVALID_LABEL,
    VALID_SPLITS,
    DataContainerConfig,
    SplitData,
    TimeSeriesDataContainer,
)

# =============================================================================
# ABSORBED: src/contracts - Model and data contracts
# =============================================================================
# Re-export DataContract from contracts for backward compatibility
from src.core.contracts import (
    DATA_SCHEMA,
    MODEL_CONTRACTS,
    ArtifactManifest,
    DataContract,
    DataContractSchema,
    FeatureMode,
    MTFMode,
    get_model_contract,
    get_models_by_mtf_mode,
    get_models_by_rank,
    get_models_requiring_scaling,
    list_model_contracts,
)

# =============================================================================
# ABSORBED: src/coordination - Temporal alignment utilities
# =============================================================================
from src.core.coordination import (
    TimeframeCoordinator,
    TimeframeData,
    align_to_anchor,
    apply_mtf_lag,
    compute_sequence_offset,
    validate_timestamp_alignment,
)

# =============================================================================
# DATA CONTRACT - Explicit data passing between pipeline stages
# =============================================================================
from src.core.data_contract import (
    DatasetContract,
    SplitDatasetContract,
)

# =============================================================================
# EXISTING EXPORTS (preserved from original)
# =============================================================================
from src.core.defaults import DEFAULTS, GlobalDefaults, as_dict, get_default

# =============================================================================
# EXCEPTIONS - Unified exception hierarchy (Phase 8B)
# =============================================================================
from src.core.exceptions import (
    ConfigError,
    ContractViolation,
    DataContractViolation,
    DataError,
    InferenceError,
    LeakageError,
    LookaheadError,
    MLFactoryError,
    ModelContractViolation,
    TrainingError,
)

# =============================================================================
# INTERFACES - Abstract contracts
# =============================================================================
from src.core.interfaces import (
    AdapterContract,
    # Result types
    AdapterResult,
    # Contracts (Note: DataContract is from contracts/, not interfaces - Phase 27)
    ModelContract,
    OOFResult,
    PredictionResult,
    TrainingResult,
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

# =============================================================================
# PROTOCOLS - Structural typing contracts for inference
# =============================================================================
from src.core.protocols import (
    InferenceBundle,
    TrainerProtocol,
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
# RESILIENCE - Timeout protection, circuit breakers, retry (Phase 17B-E)
# =============================================================================
from src.core.resilience import (
    GPU_OOM_RETRY,
    NETWORK_RETRY,
    TRANSIENT_RETRY,
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerRegistry,
    CircuitBreakerStats,
    CircuitOpenError,
    CircuitState,
    ResilienceError,
    ResilienceTimeoutError,
    RetryConfig,
    RetryExhaustedError,
    TimeoutError,
    get_model_breaker_registry,
    retry,
    retry_with_config,
    run_with_timeout,
    timeout,
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
    # Type aliases
    Features,
    Index,
    LabelingMethod,
    Labels,
    ModelFamily,
    ModelType,
    TrainingMode,
)

# =============================================================================
# ABSORBED: src/utils - Memory, cache, notebook utilities
# =============================================================================
from src.core.utils import (
    CacheConfig,
    CacheEntry,
    CacheManager,
    CacheStats,
    CheckpointManager,
    DataCache,
    DataCacheConfig,
    MemoryInfo,
    cached_result,
    check_available_memory,
    check_memory_sufficient,
    estimate_array_size,
    estimate_object_size,
    get_global_cache,
    get_global_data_cache,
    get_memory_info,
    is_colab,
    log_memory_usage,
    memory_logged,
    setup_colab_environment,
    setup_environment,
)

# =============================================================================
# VALIDATION - Input validation
# =============================================================================
from src.core.validation import (
    # Exception
    ValidationError,
    # DataFrame validation
    validate_dataframe,
    # Feature validation
    validate_features,
    # Array validation
    validate_input_shape,
    validate_labels,
    validate_model_list,
    # Model validation
    validate_model_name,
    validate_ohlcv,
    # Path validation
    validate_path_exists,
    validate_probabilities,
    # Timeframe validation
    validate_timeframe,
    validate_timeframe_list,
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
    # NOTE: src.config items NOT re-exported to avoid circular imports.
    # Import directly: from src.config import GlobalConfig, TrainerConfig, etc.
    # =========================================================================
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
    # =========================================================================
    # EXCEPTIONS - Unified exception hierarchy (Phase 8B)
    # =========================================================================
    "MLFactoryError",
    "ConfigError",
    "ContractViolation",
    "DataContractViolation",
    "ModelContractViolation",
    "DataError",
    "TrainingError",
    "InferenceError",
    "LeakageError",
    "LookaheadError",
    # =========================================================================
    # PROTOCOLS - Structural typing contracts for inference
    # =========================================================================
    "TrainerProtocol",
    "InferenceBundle",
    # =========================================================================
    # RESILIENCE - Timeout protection, circuit breakers, retry (Phase 17B-E)
    # =========================================================================
    # Exceptions
    "ResilienceError",
    "ResilienceTimeoutError",
    "TimeoutError",
    "CircuitOpenError",
    "RetryExhaustedError",
    # Timeout
    "timeout",
    "run_with_timeout",
    # Retry
    "RetryConfig",
    "retry",
    "retry_with_config",
    "GPU_OOM_RETRY",
    "NETWORK_RETRY",
    "TRANSIENT_RETRY",
    # Circuit Breaker
    "CircuitState",
    "CircuitBreakerConfig",
    "CircuitBreakerStats",
    "CircuitBreaker",
    "CircuitBreakerRegistry",
    "get_model_breaker_registry",
]
