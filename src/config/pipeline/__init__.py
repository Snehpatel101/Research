"""
Pipeline configuration - re-exports from src.phase1.config.

This module provides unified access to pipeline configuration.
All config modules remain in their original locations; this is a facade.

Usage:
    from src.config.pipeline import MODEL_DATA_REQUIREMENTS, ModelFamily
    from src.config.pipeline import BARRIER_PARAMS, get_barrier_params
"""

# =============================================================================
# MODEL CONFIG (canonical location: src.models.config.data_requirements)
# =============================================================================
from src.models.config.data_requirements import (
    ENSEMBLE_CONFIGS,
    MODEL_DATA_REQUIREMENTS,
    EnsembleConfig,
    ModelDataRequirements,
    ModelFamily,
    ScalerType,
    get_all_ensemble_names,
    get_all_model_names,
    get_combined_requirements,
    get_ensemble_config,
    get_model_requirements,
    get_models_by_family,
    validate_model_config,
)

# =============================================================================
# BARRIERS CONFIG
# =============================================================================
from src.pipeline._phase1_impl.config.barriers_config import (
    BARRIER_PARAMS,
    BARRIER_PARAMS_DEFAULT,
    PERCENTAGE_BARRIER_PARAMS,
    SLIPPAGE_TICKS,
    TICK_VALUES,
    TRANSACTION_COSTS,
    get_barrier_params,
    get_max_bars_across_all_params,
    get_slippage_ticks,
    get_total_trade_cost,
    validate_barrier_params,
)

# =============================================================================
# LABELING CONFIG
# =============================================================================
from src.pipeline._phase1_impl.config.labeling_config import (
    DEFAULT_LABELING_STRATEGY,
    LABEL_BALANCE_CONSTRAINTS,
    LABELING_STRATEGY_CONFIGS,
    MULTI_LABEL_CONFIG,
    LabelingStrategyType,
    get_labeling_strategy_config,
    get_multi_label_config,
    validate_labeling_config,
)

# =============================================================================
# LABELS CONFIG
# =============================================================================
from src.pipeline._phase1_impl.config.labels import (
    ALL_LABEL_TEMPLATES,
    LABEL_COLUMN_METADATA,
    OPTIONAL_LABEL_TEMPLATES,
    REQUIRED_LABEL_TEMPLATES,
    get_all_label_columns,
    get_label_metadata,
    get_optional_label_columns,
    get_required_label_columns,
    is_label_column,
)

# =============================================================================
# FEATURE SETS CONFIG
# =============================================================================
from src.pipeline._phase1_impl.config.feature_sets import (
    FEATURE_SET_ALIASES,
    FEATURE_SET_DEFINITIONS,
    FeatureSetDefinition,
    get_feature_set_definitions,
    resolve_feature_set_name,
    resolve_feature_set_names,
    validate_feature_set_config,
)

# =============================================================================
# FEATURES CONFIG
# =============================================================================
from src.pipeline._phase1_impl.config.features import (
    CORRELATION_THRESHOLD,
    DRIFT_CONFIG,
    MTF_CONFIG,
    STATIONARITY_TESTS,
    SUPPORTED_TIMEFRAMES as PHASE1_SUPPORTED_TIMEFRAMES,
    TIMEFRAME_TO_FREQ as PHASE1_TIMEFRAME_TO_FREQ,
    VARIANCE_THRESHOLD,
    get_drift_config,
    get_mtf_config,
    get_stationarity_config,
    parse_timeframe_to_minutes,
    validate_drift_config,
    validate_feature_thresholds,
    validate_mtf_config,
    validate_stationarity_config,
    validate_timeframe as validate_phase1_timeframe,
)

# =============================================================================
# REGIME CONFIG
# =============================================================================
from src.pipeline._phase1_impl.config.regime_config import (
    REGIME_BARRIER_ADJUSTMENTS,
    REGIME_CONFIG,
    get_regime_adjusted_barriers,
)

# =============================================================================
# RUNTIME CONFIG
# =============================================================================
from src.pipeline._phase1_impl.config.runtime import (
    CONFIG_DIR as PHASE1_CONFIG_DIR,
    DATA_DIR,
    EMBARGO_BARS,
    PROJECT_ROOT,
    PURGE_BARS,
    RANDOM_SEED,
    RAW_DATA_DIR,
    RESULTS_DIR,
    RUNS_DIR,
    SYMBOLS,
    TARGET_TIMEFRAME,
    TEST_RATIO,
    TRAIN_RATIO,
    VAL_RATIO,
    get_timeframe_metadata,
    set_global_seeds,
    validate_config as validate_runtime_config,
)

# =============================================================================
# MULTI-MODEL CONFIG
# =============================================================================
from src.pipeline._phase1_impl.config.multi_model import (
    MultiModelPipelineConfig,
    build_multi_model_config,
    expand_ensemble_models,
    get_recommended_feature_set,
    validate_multi_model_setup,
)

__all__ = [
    # Model config
    "ModelFamily",
    "ScalerType",
    "ModelDataRequirements",
    "EnsembleConfig",
    "MODEL_DATA_REQUIREMENTS",
    "ENSEMBLE_CONFIGS",
    "get_model_requirements",
    "get_ensemble_config",
    "get_models_by_family",
    "get_combined_requirements",
    "validate_model_config",
    "get_all_model_names",
    "get_all_ensemble_names",
    # Barriers config
    "BARRIER_PARAMS",
    "BARRIER_PARAMS_DEFAULT",
    "PERCENTAGE_BARRIER_PARAMS",
    "TRANSACTION_COSTS",
    "SLIPPAGE_TICKS",
    "TICK_VALUES",
    "get_barrier_params",
    "get_slippage_ticks",
    "get_total_trade_cost",
    "get_max_bars_across_all_params",
    "validate_barrier_params",
    # Labeling config
    "LabelingStrategyType",
    "DEFAULT_LABELING_STRATEGY",
    "LABELING_STRATEGY_CONFIGS",
    "LABEL_BALANCE_CONSTRAINTS",
    "MULTI_LABEL_CONFIG",
    "get_labeling_strategy_config",
    "get_multi_label_config",
    "validate_labeling_config",
    # Labels config
    "REQUIRED_LABEL_TEMPLATES",
    "OPTIONAL_LABEL_TEMPLATES",
    "ALL_LABEL_TEMPLATES",
    "LABEL_COLUMN_METADATA",
    "get_required_label_columns",
    "get_optional_label_columns",
    "get_all_label_columns",
    "is_label_column",
    "get_label_metadata",
    # Feature sets config
    "FeatureSetDefinition",
    "FEATURE_SET_DEFINITIONS",
    "FEATURE_SET_ALIASES",
    "get_feature_set_definitions",
    "resolve_feature_set_name",
    "resolve_feature_set_names",
    "validate_feature_set_config",
    # Features config
    "CORRELATION_THRESHOLD",
    "VARIANCE_THRESHOLD",
    "MTF_CONFIG",
    "STATIONARITY_TESTS",
    "DRIFT_CONFIG",
    "PHASE1_SUPPORTED_TIMEFRAMES",
    "PHASE1_TIMEFRAME_TO_FREQ",
    "get_mtf_config",
    "validate_mtf_config",
    "validate_feature_thresholds",
    "get_stationarity_config",
    "validate_stationarity_config",
    "get_drift_config",
    "validate_drift_config",
    "parse_timeframe_to_minutes",
    "validate_phase1_timeframe",
    # Regime config
    "REGIME_CONFIG",
    "REGIME_BARRIER_ADJUSTMENTS",
    "get_regime_adjusted_barriers",
    # Runtime config
    "PROJECT_ROOT",
    "DATA_DIR",
    "RAW_DATA_DIR",
    "RESULTS_DIR",
    "RUNS_DIR",
    "PHASE1_CONFIG_DIR",
    "SYMBOLS",
    "TARGET_TIMEFRAME",
    "TRAIN_RATIO",
    "VAL_RATIO",
    "TEST_RATIO",
    "RANDOM_SEED",
    "PURGE_BARS",
    "EMBARGO_BARS",
    "set_global_seeds",
    "validate_runtime_config",
    "get_timeframe_metadata",
    # Multi-model config
    "MultiModelPipelineConfig",
    "build_multi_model_config",
    "expand_ensemble_models",
    "get_recommended_feature_set",
    "validate_multi_model_setup",
]
