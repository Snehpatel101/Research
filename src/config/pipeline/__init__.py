"""
Pipeline configuration - re-exports from src.data.pipeline.config.

This module provides unified access to pipeline configuration.
All config modules remain in their original locations; this is a facade.

Usage:
    from src.config.pipeline import MODEL_DATA_REQUIREMENTS, ModelFamily
    from src.config.pipeline import BARRIER_PARAMS, get_barrier_params
"""

# =============================================================================
# MODEL CONFIG (canonical location: src.models.config.data_requirements)
# =============================================================================
# Timeframe config - import from common module (single source of truth)
from src.core.common.timeframes import (
    SUPPORTED_TIMEFRAMES as PHASE1_SUPPORTED_TIMEFRAMES,
)
from src.core.common.timeframes import (
    TIMEFRAME_TO_FREQ as PHASE1_TIMEFRAME_TO_FREQ,
)
from src.core.common.timeframes import (
    get_timeframe_minutes as parse_timeframe_to_minutes,
)
from src.core.common.timeframes import (
    validate_timeframe as validate_phase1_timeframe,
)

# =============================================================================
# BARRIERS CONFIG
# =============================================================================
from src.data.pipeline.config.barriers_config import (
    BARRIER_PARAMS,
    BARRIER_PARAMS_DEFAULT,
    PERCENTAGE_BARRIER_PARAMS,
    SLIPPAGE_TICKS,
    TICK_VALUES,
    TRANSACTION_COSTS,
    get_barrier_params,
    get_slippage_ticks,
    get_total_trade_cost,
)

# =============================================================================
# FEATURE SETS CONFIG
# =============================================================================
from src.data.pipeline.config.feature_sets import (
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
from src.data.pipeline.config.features import (
    CORRELATION_THRESHOLD,
    DRIFT_CONFIG,
    MTF_CONFIG,
    STATIONARITY_TESTS,
    VARIANCE_THRESHOLD,
    get_drift_config,
    validate_mtf_config,
)

# =============================================================================
# LABELING CONFIG
# =============================================================================
from src.data.pipeline.config.labeling_config import LABEL_BALANCE_CONSTRAINTS

# =============================================================================
# LABELS CONFIG
# =============================================================================
from src.data.pipeline.config.labels import (
    OPTIONAL_LABEL_TEMPLATES,
    REQUIRED_LABEL_TEMPLATES,
)

# =============================================================================
# RUNTIME CONFIG
# =============================================================================
from src.data.pipeline.config.runtime import (
    CONFIG_DIR as PHASE1_CONFIG_DIR,
)
from src.data.pipeline.config.runtime import (
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
)
from src.data.pipeline.config.runtime import (
    validate_config as validate_runtime_config,
)
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
    # Labeling config
    "LABEL_BALANCE_CONSTRAINTS",
    # Labels config
    "REQUIRED_LABEL_TEMPLATES",
    "OPTIONAL_LABEL_TEMPLATES",
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
    "validate_mtf_config",
    "get_drift_config",
    "parse_timeframe_to_minutes",
    "validate_phase1_timeframe",
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
]
