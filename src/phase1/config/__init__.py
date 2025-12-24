"""Phase 1 Configuration.

Provides barrier, labeling, and feature set configurations.
"""
from src.phase1.config.barriers_config import (
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

from src.phase1.config.labeling_config import (
    LabelingStrategyType,
    DEFAULT_LABELING_STRATEGY,
    LABELING_STRATEGY_CONFIGS,
    LABEL_BALANCE_CONSTRAINTS,
    MULTI_LABEL_CONFIG,
    get_labeling_strategy_config,
    get_multi_label_config,
    validate_labeling_config,
)

from src.phase1.config.feature_sets import (
    FeatureSetDefinition,
    FEATURE_SET_DEFINITIONS,
    FEATURE_SET_ALIASES,
    get_feature_set_definitions,
    resolve_feature_set_name,
    resolve_feature_set_names,
    validate_feature_set_config,
)

from src.phase1.config.features import (
    # Timeframe config
    SUPPORTED_TIMEFRAMES,
    TIMEFRAME_TO_FREQ,
    SUPPORTED_HORIZONS,
    validate_timeframe,
    parse_timeframe_to_minutes,
    auto_scale_purge_embargo,
    validate_horizons,
    # Feature thresholds
    CORRELATION_THRESHOLD,
    VARIANCE_THRESHOLD,
    CROSS_ASSET_FEATURES,
    MTF_CONFIG,
    STATIONARITY_TESTS,
    DRIFT_CONFIG,
    get_mtf_config,
    validate_mtf_config,
    get_cross_asset_config,
    validate_feature_thresholds,
    get_stationarity_config,
    validate_stationarity_config,
    get_drift_config,
    validate_drift_config,
    get_cross_asset_feature_names,
    is_cross_asset_feature,
)

from src.phase1.config.labels import (
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

__all__ = [
    # barriers_config
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
    # labeling_config
    'LabelingStrategyType',
    'DEFAULT_LABELING_STRATEGY',
    'LABELING_STRATEGY_CONFIGS',
    'LABEL_BALANCE_CONSTRAINTS',
    'MULTI_LABEL_CONFIG',
    'get_labeling_strategy_config',
    'get_multi_label_config',
    'validate_labeling_config',
    # feature_sets
    'FeatureSetDefinition',
    'FEATURE_SET_DEFINITIONS',
    'FEATURE_SET_ALIASES',
    'get_feature_set_definitions',
    'resolve_feature_set_name',
    'resolve_feature_set_names',
    'validate_feature_set_config',
    # timeframe config
    'SUPPORTED_TIMEFRAMES',
    'TIMEFRAME_TO_FREQ',
    'SUPPORTED_HORIZONS',
    'validate_timeframe',
    'parse_timeframe_to_minutes',
    'auto_scale_purge_embargo',
    'validate_horizons',
    # features
    'CORRELATION_THRESHOLD',
    'VARIANCE_THRESHOLD',
    'CROSS_ASSET_FEATURES',
    'MTF_CONFIG',
    'STATIONARITY_TESTS',
    'DRIFT_CONFIG',
    'get_mtf_config',
    'validate_mtf_config',
    'get_cross_asset_config',
    'validate_feature_thresholds',
    'get_stationarity_config',
    'validate_stationarity_config',
    'get_drift_config',
    'validate_drift_config',
    'get_cross_asset_feature_names',
    'is_cross_asset_feature',
    # labels
    'REQUIRED_LABEL_TEMPLATES',
    'OPTIONAL_LABEL_TEMPLATES',
    'ALL_LABEL_TEMPLATES',
    'LABEL_COLUMN_METADATA',
    'get_required_label_columns',
    'get_optional_label_columns',
    'get_all_label_columns',
    'is_label_column',
    'get_label_metadata',
]
