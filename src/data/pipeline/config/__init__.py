"""Phase 1 Configuration.

Provides barrier, labeling, and feature set configurations.
"""

from src.core.common.horizon_config import (
    ACTIVE_HORIZONS,
    HORIZON_TIMEFRAME_MINUTES,
    HORIZONS,
    LABEL_HORIZONS,
    LOOKBACK_HORIZONS,
    SUPPORTED_HORIZONS,
    auto_scale_purge_embargo,
    get_scaled_horizons,
    validate_horizons,
)

# Timeframe config - now in src.core.common.timeframes
from src.core.common.timeframes import (
    SUPPORTED_TIMEFRAMES,
    TIMEFRAME_TO_FREQ,
    validate_timeframe,
)
from src.core.common.timeframes import (
    timeframe_to_minutes as parse_timeframe_to_minutes,
)
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
from src.data.pipeline.config.feature_sets import (
    FEATURE_SET_ALIASES,
    FEATURE_SET_DEFINITIONS,
    FeatureSetDefinition,
    get_feature_set_definitions,
    resolve_feature_set_name,
    resolve_feature_set_names,
    validate_feature_set_config,
)
from src.data.pipeline.config.features import (
    CORRELATION_THRESHOLD,
    DRIFT_CONFIG,
    MTF_CONFIG,
    STATIONARITY_TESTS,
    VARIANCE_THRESHOLD,
    get_drift_config,
    validate_mtf_config,
)
from src.data.pipeline.config.labeling_config import LABEL_BALANCE_CONSTRAINTS
from src.data.pipeline.config.labels import (
    OPTIONAL_LABEL_TEMPLATES,
    REQUIRED_LABEL_TEMPLATES,
)
from src.data.pipeline.config.runtime import (
    CONFIG_DIR,
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
    validate_config,
)

# Model config - canonical location: src.models.config.data_requirements
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
    # barriers_config
    "BARRIER_PARAMS",
    "BARRIER_PARAMS_DEFAULT",
    "PERCENTAGE_BARRIER_PARAMS",
    "TRANSACTION_COSTS",
    "SLIPPAGE_TICKS",
    "TICK_VALUES",
    "get_barrier_params",
    "get_slippage_ticks",
    "get_total_trade_cost",
    # labeling_config
    "LABEL_BALANCE_CONSTRAINTS",
    # feature_sets
    "FeatureSetDefinition",
    "FEATURE_SET_DEFINITIONS",
    "FEATURE_SET_ALIASES",
    "get_feature_set_definitions",
    "resolve_feature_set_name",
    "resolve_feature_set_names",
    "validate_feature_set_config",
    # timeframe config
    "SUPPORTED_TIMEFRAMES",
    "TIMEFRAME_TO_FREQ",
    "HORIZONS",
    "SUPPORTED_HORIZONS",
    "ACTIVE_HORIZONS",
    "LABEL_HORIZONS",
    "LOOKBACK_HORIZONS",
    "HORIZON_TIMEFRAME_MINUTES",
    "validate_timeframe",
    "parse_timeframe_to_minutes",
    "auto_scale_purge_embargo",
    "validate_horizons",
    "get_scaled_horizons",
    # runtime defaults
    "PROJECT_ROOT",
    "DATA_DIR",
    "RAW_DATA_DIR",
    "RESULTS_DIR",
    "RUNS_DIR",
    "CONFIG_DIR",
    "SYMBOLS",
    "TARGET_TIMEFRAME",
    "TRAIN_RATIO",
    "VAL_RATIO",
    "TEST_RATIO",
    "RANDOM_SEED",
    "PURGE_BARS",
    "EMBARGO_BARS",
    "set_global_seeds",
    "validate_config",
    "get_timeframe_metadata",
    # features
    "CORRELATION_THRESHOLD",
    "VARIANCE_THRESHOLD",
    "MTF_CONFIG",
    "STATIONARITY_TESTS",
    "DRIFT_CONFIG",
    "validate_mtf_config",
    "get_drift_config",
    # labels
    "REQUIRED_LABEL_TEMPLATES",
    "OPTIONAL_LABEL_TEMPLATES",
    # model_config
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
]
