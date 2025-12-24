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
]
