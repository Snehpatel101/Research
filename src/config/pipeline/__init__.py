"""
Pipeline configuration facade - barriers, feature sets and model data requirements.

This module provides unified access to pipeline configuration.
All config modules remain in their original locations; this is a facade.

Usage:
    from src.config.pipeline import MODEL_DATA_REQUIREMENTS, ModelFamily
    from src.config.pipeline import BARRIER_PARAMS, get_barrier_params
"""

# =============================================================================
# MODEL CONFIG (canonical location: src.models.config.data_requirements)
# =============================================================================
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
    # Feature sets config
    "FeatureSetDefinition",
    "FEATURE_SET_DEFINITIONS",
    "FEATURE_SET_ALIASES",
    "get_feature_set_definitions",
    "resolve_feature_set_name",
    "resolve_feature_set_names",
    "validate_feature_set_config",
]
