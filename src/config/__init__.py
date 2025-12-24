"""
Configuration package for ensemble trading pipeline.

This package provides modular configuration for all pipeline components.
All exports are re-exported here for backward compatibility with imports
from the main config module.

Example usage (backward compatible):
    from config import BARRIER_PARAMS, SYMBOLS, validate_config
    from config import TRAIN_RATIO, PURGE_BARS, EMBARGO_BARS

Module structure:
    - constants.py: Core constants, paths, symbols, timeframes
    - horizons.py: Horizon configuration (re-exports from horizon_config)
    - barriers_config.py: Barrier parameters and transaction costs
    - splits.py: Train/val/test split configuration
    - features.py: Feature selection and MTF configuration
    - sessions.py: Trading session configuration
    - regime_config.py: Regime detection configuration
    - labeling_config.py: Labeling strategy configuration
    - validation.py: Configuration validation
"""

# =============================================================================
# CONSTANTS - Core constants, paths, symbols, timeframes
# =============================================================================
from .constants import (
    # Reproducibility
    RANDOM_SEED,
    set_global_seeds,
    # Paths
    PROJECT_ROOT,
    DATA_DIR,
    SRC_DIR,
    CONFIG_DIR,
    RAW_DATA_DIR,
    CLEAN_DATA_DIR,
    FEATURES_DIR,
    PROCESSED_DATA_DIR,
    FINAL_DATA_DIR,
    SPLITS_DIR,
    MODELS_DIR,
    BASE_MODELS_DIR,
    ENSEMBLE_MODELS_DIR,
    RESULTS_DIR,
    REPORTS_DIR,
    LOGS_DIR,
    ensure_directories,
    # Symbols
    SYMBOLS,
    # Timeframes
    SUPPORTED_TIMEFRAMES,
    TARGET_TIMEFRAME,
    BAR_RESOLUTION,
    parse_timeframe_to_minutes,
    validate_timeframe,
    get_timeframe_metadata,
)

# =============================================================================
# HORIZONS - Dynamic horizon configuration
# =============================================================================
from .horizons import (
    # Horizon lists
    SUPPORTED_HORIZONS,
    HORIZONS,
    LOOKBACK_HORIZONS,
    ACTIVE_HORIZONS,
    # Timeframe configuration
    HORIZON_TIMEFRAME_MINUTES,
    HORIZON_TIMEFRAME_SCALING,
    # Multipliers
    PURGE_MULTIPLIER,
    EMBARGO_MULTIPLIER,
    # Utility functions
    validate_horizons,
    get_scaled_horizons,
    auto_scale_purge_embargo,
    get_default_barrier_params_for_horizon,
    # Dataclass
    HorizonConfig,
)

# =============================================================================
# BARRIERS - Barrier parameters and transaction costs
# =============================================================================
from .barriers_config import (
    BARRIER_PARAMS,
    BARRIER_PARAMS_DEFAULT,
    PERCENTAGE_BARRIER_PARAMS,
    TRANSACTION_COSTS,
    SLIPPAGE_TICKS,
    TICK_VALUES,
    get_barrier_params,
    get_slippage_ticks,
    get_total_trade_cost,
    get_max_bars_across_all_params,
    validate_barrier_params,
)

# =============================================================================
# SPLITS - Train/val/test split configuration
# =============================================================================
from .splits import (
    TRAIN_RATIO,
    VAL_RATIO,
    TEST_RATIO,
    PURGE_BARS,
    EMBARGO_BARS,
    validate_splits_config,
    get_splits_config,
)

# =============================================================================
# FEATURES - Feature selection and MTF configuration
# =============================================================================
from .features import (
    CORRELATION_THRESHOLD,
    VARIANCE_THRESHOLD,
    CROSS_ASSET_FEATURES,
    MTF_CONFIG,
    get_mtf_config,
    validate_mtf_config,
    get_cross_asset_config,
    validate_feature_thresholds,
    STATIONARITY_TESTS,
    get_stationarity_config,
    validate_stationarity_config,
    DRIFT_CONFIG,
    get_drift_config,
    validate_drift_config,
)

# =============================================================================
# FEATURE SETS - Named feature set definitions
# =============================================================================
from .feature_sets import (
    FeatureSetDefinition,
    FEATURE_SET_DEFINITIONS,
    FEATURE_SET_ALIASES,
    get_feature_set_definitions,
    resolve_feature_set_name,
    resolve_feature_set_names,
    validate_feature_set_config,
)

# =============================================================================
# SESSIONS - Trading session configuration
# =============================================================================
from .sessions import (
    SESSIONS_CONFIG,
    get_sessions_config,
    validate_sessions_config,
)

# =============================================================================
# REGIME - Regime detection configuration
# =============================================================================
from .regime_config import (
    REGIME_CONFIG,
    REGIME_BARRIER_ADJUSTMENTS,
    get_regime_adjusted_barriers,
    get_regime_config,
    validate_regime_config,
)

# =============================================================================
# LABELING - Labeling strategy configuration
# =============================================================================
from .labeling_config import (
    LabelingStrategyType,
    DEFAULT_LABELING_STRATEGY,
    LABELING_STRATEGY_CONFIGS,
    LABEL_BALANCE_CONSTRAINTS,
    MULTI_LABEL_CONFIG,
    get_labeling_strategy_config,
    get_multi_label_config,
    validate_labeling_config,
)

# =============================================================================
# LABELS - Label column definitions and templates
# =============================================================================
from .labels import (
    REQUIRED_LABEL_TEMPLATES,
    OPTIONAL_LABEL_TEMPLATES,
    ALL_LABEL_TEMPLATES,
    LABEL_COLUMN_METADATA,
    get_required_label_columns,
    get_optional_label_columns,
    get_all_label_columns,
    is_label_column,
    get_label_metadata,
)

# =============================================================================
# VALIDATION - Configuration validation
# =============================================================================
from .validation import (
    validate_config,
    validate_config_silent,
)

# =============================================================================
# PRESETS - Trading presets (re-exported from presets module)
# =============================================================================
from src.presets import (
    TradingPreset,
    PRESET_CONFIGS,
    get_preset,
    apply_preset_to_config,
    validate_preset,
    list_available_presets,
    get_preset_summary,
    get_adjusted_barrier_params,
)


# =============================================================================
# ALL EXPORTS
# =============================================================================
__all__ = [
    # Constants
    'RANDOM_SEED',
    'set_global_seeds',
    'PROJECT_ROOT',
    'DATA_DIR',
    'SRC_DIR',
    'CONFIG_DIR',
    'RAW_DATA_DIR',
    'CLEAN_DATA_DIR',
    'FEATURES_DIR',
    'PROCESSED_DATA_DIR',
    'FINAL_DATA_DIR',
    'SPLITS_DIR',
    'MODELS_DIR',
    'BASE_MODELS_DIR',
    'ENSEMBLE_MODELS_DIR',
    'RESULTS_DIR',
    'REPORTS_DIR',
    'LOGS_DIR',
    'ensure_directories',
    'SYMBOLS',
    'SUPPORTED_TIMEFRAMES',
    'TARGET_TIMEFRAME',
    'BAR_RESOLUTION',
    'parse_timeframe_to_minutes',
    'validate_timeframe',
    'get_timeframe_metadata',
    # Horizons
    'SUPPORTED_HORIZONS',
    'HORIZONS',
    'LOOKBACK_HORIZONS',
    'ACTIVE_HORIZONS',
    'HORIZON_TIMEFRAME_MINUTES',
    'HORIZON_TIMEFRAME_SCALING',
    'PURGE_MULTIPLIER',
    'EMBARGO_MULTIPLIER',
    'validate_horizons',
    'get_scaled_horizons',
    'auto_scale_purge_embargo',
    'get_default_barrier_params_for_horizon',
    'HorizonConfig',
    # Barriers
    'BARRIER_PARAMS',
    'BARRIER_PARAMS_DEFAULT',
    'PERCENTAGE_BARRIER_PARAMS',
    'TRANSACTION_COSTS',
    'SLIPPAGE_TICKS',
    'TICK_VALUES',
    'get_barrier_params',
    'get_slippage_ticks',
    'get_total_trade_cost',
    'get_max_bars_across_all_params',
    'validate_barrier_params',
    # Splits
    'TRAIN_RATIO',
    'VAL_RATIO',
    'TEST_RATIO',
    'PURGE_BARS',
    'EMBARGO_BARS',
    'validate_splits_config',
    'get_splits_config',
    # Features
    'CORRELATION_THRESHOLD',
    'VARIANCE_THRESHOLD',
    'CROSS_ASSET_FEATURES',
    'MTF_CONFIG',
    'get_mtf_config',
    'validate_mtf_config',
    'get_cross_asset_config',
    'validate_feature_thresholds',
    'STATIONARITY_TESTS',
    'get_stationarity_config',
    'validate_stationarity_config',
    'DRIFT_CONFIG',
    'get_drift_config',
    'validate_drift_config',
    'FeatureSetDefinition',
    'FEATURE_SET_DEFINITIONS',
    'FEATURE_SET_ALIASES',
    'get_feature_set_definitions',
    'resolve_feature_set_name',
    'resolve_feature_set_names',
    'validate_feature_set_config',
    # Sessions
    'SESSIONS_CONFIG',
    'get_sessions_config',
    'validate_sessions_config',
    # Regime
    'REGIME_CONFIG',
    'REGIME_BARRIER_ADJUSTMENTS',
    'get_regime_adjusted_barriers',
    'get_regime_config',
    'validate_regime_config',
    # Labeling
    'LabelingStrategyType',
    'DEFAULT_LABELING_STRATEGY',
    'LABELING_STRATEGY_CONFIGS',
    'LABEL_BALANCE_CONSTRAINTS',
    'MULTI_LABEL_CONFIG',
    'get_labeling_strategy_config',
    'get_multi_label_config',
    'validate_labeling_config',
    # Labels
    'REQUIRED_LABEL_TEMPLATES',
    'OPTIONAL_LABEL_TEMPLATES',
    'ALL_LABEL_TEMPLATES',
    'LABEL_COLUMN_METADATA',
    'get_required_label_columns',
    'get_optional_label_columns',
    'get_all_label_columns',
    'is_label_column',
    'get_label_metadata',
    # Validation
    'validate_config',
    'validate_config_silent',
    # Presets
    'TradingPreset',
    'PRESET_CONFIGS',
    'get_preset',
    'apply_preset_to_config',
    'validate_preset',
    'list_available_presets',
    'get_preset_summary',
    'get_adjusted_barrier_params',
]


# =============================================================================
# RUN VALIDATION AT IMPORT TIME
# =============================================================================
# This ensures configuration is valid when the module is imported.
# Note: This was moved from the old config.py to maintain the same behavior.
validate_config()
