"""
Hyperparameter search spaces for Optuna tuning.

Defines search spaces for all supported model types.

IMPORTANT - LightGBM Constraint:
    num_leaves must be <= 2^max_depth. For example:
    - max_depth=3 -> max num_leaves = 8
    - max_depth=6 -> max num_leaves = 64
    - max_depth=10 -> max num_leaves = 1024

    The static param space uses conservative bounds. Dynamic constraint
    enforcement is applied during Optuna tuning via validate_lightgbm_params()
    and constrained sampling in _sample_params_with_constraints().
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)

# =============================================================================
# LIGHTGBM CONSTRAINT VALIDATION
# =============================================================================


def validate_lightgbm_params(params: dict[str, Any]) -> dict[str, Any]:
    """
    Validate and fix LightGBM parameter constraints.

    LightGBM requires: num_leaves <= 2^max_depth

    If num_leaves exceeds the maximum for the given max_depth,
    it will be capped with a warning.

    Args:
        params: LightGBM parameters dictionary

    Returns:
        Validated parameters dictionary (modified in place and returned)
    """
    max_depth = params.get("max_depth", 6)
    num_leaves = params.get("num_leaves", 31)

    # max_depth=-1 means unlimited in LightGBM
    if max_depth > 0:
        max_valid_leaves = 2**max_depth
        if num_leaves > max_valid_leaves:
            logger.warning(
                f"LightGBM num_leaves ({num_leaves}) exceeds max for max_depth={max_depth} "
                f"(max={max_valid_leaves}). Capping to {max_valid_leaves}."
            )
            params["num_leaves"] = max_valid_leaves

    return params


def get_max_leaves_for_depth(max_depth: int) -> int:
    """
    Get maximum valid num_leaves for a given max_depth.

    Args:
        max_depth: Tree depth limit (positive integer)

    Returns:
        Maximum valid num_leaves value
    """
    if max_depth <= 0:
        return 1024  # Reasonable default for unlimited depth
    return 2**max_depth


# =============================================================================
# HYPERPARAMETER SEARCH SPACES
# =============================================================================

PARAM_SPACES: dict[str, dict[str, dict[str, Any]]] = {
    # --- BOOSTING MODELS ---
    "xgboost": {
        "n_estimators": {"type": "int", "low": 100, "high": 500},
        "max_depth": {"type": "int", "low": 3, "high": 10},
        "learning_rate": {"type": "float", "low": 0.01, "high": 0.3, "log": True},
        "subsample": {"type": "float", "low": 0.6, "high": 1.0},
        "colsample_bytree": {"type": "float", "low": 0.6, "high": 1.0},
        "min_child_weight": {"type": "int", "low": 1, "high": 20},
        "gamma": {"type": "float", "low": 0.0, "high": 5.0},
        "reg_alpha": {"type": "float", "low": 0.0, "high": 10.0},
        "reg_lambda": {"type": "float", "low": 0.0, "high": 10.0},
    },
    # LightGBM: Use conservative num_leaves bounds that are valid for all max_depth values.
    # For max_depth=3, max valid leaves = 8. We use 8-64 as safe static range.
    # Dynamic constraint enforcement via validate_lightgbm_params() is applied during tuning.
    "lightgbm": {
        "n_estimators": {"type": "int", "low": 100, "high": 500},
        "max_depth": {"type": "int", "low": 3, "high": 10},
        # num_leaves capped at 64 as safe default; dynamic tuning enforces 2^max_depth constraint
        "num_leaves": {"type": "int", "low": 8, "high": 64},
        "learning_rate": {"type": "float", "low": 0.01, "high": 0.3, "log": True},
        "subsample": {"type": "float", "low": 0.6, "high": 1.0},
        "colsample_bytree": {"type": "float", "low": 0.6, "high": 1.0},
        "min_child_samples": {"type": "int", "low": 5, "high": 50},
        "reg_alpha": {"type": "float", "low": 0.0, "high": 10.0},
        "reg_lambda": {"type": "float", "low": 0.0, "high": 10.0},
    },
    "catboost": {
        "iterations": {"type": "int", "low": 100, "high": 500},
        "depth": {"type": "int", "low": 4, "high": 10},
        "learning_rate": {"type": "float", "low": 0.01, "high": 0.3, "log": True},
        "l2_leaf_reg": {"type": "float", "low": 1.0, "high": 10.0},
        "bagging_temperature": {"type": "float", "low": 0.0, "high": 1.0},
        "random_strength": {"type": "float", "low": 0.0, "high": 1.0},
    },
    # --- NEURAL MODELS ---
    "lstm": {
        "hidden_size": {"type": "categorical", "choices": [64, 128, 256]},
        "num_layers": {"type": "int", "low": 1, "high": 3},
        "dropout": {"type": "float", "low": 0.1, "high": 0.5},
        "learning_rate": {"type": "float", "low": 1e-4, "high": 1e-2, "log": True},
        "batch_size": {"type": "categorical", "choices": [64, 128, 256]},
    },
    "gru": {
        "hidden_size": {"type": "categorical", "choices": [64, 128, 256]},
        "num_layers": {"type": "int", "low": 1, "high": 3},
        "dropout": {"type": "float", "low": 0.1, "high": 0.5},
        "learning_rate": {"type": "float", "low": 1e-4, "high": 1e-2, "log": True},
        "batch_size": {"type": "categorical", "choices": [64, 128, 256]},
    },
    "tcn": {
        "kernel_size": {"type": "int", "low": 2, "high": 7},
        "dropout": {"type": "float", "low": 0.1, "high": 0.4},
        "learning_rate": {"type": "float", "low": 1e-4, "high": 1e-2, "log": True},
        "batch_size": {"type": "categorical", "choices": [64, 128, 256]},
    },
    # --- CLASSICAL MODELS ---
    "random_forest": {
        "n_estimators": {"type": "int", "low": 100, "high": 500},
        "max_depth": {"type": "int", "low": 5, "high": 20},
        "min_samples_split": {"type": "int", "low": 5, "high": 50},
        "min_samples_leaf": {"type": "int", "low": 2, "high": 20},
    },
}


def get_param_space(model_name: str) -> dict[str, dict[str, Any]]:
    """Get hyperparameter search space for a model."""
    return PARAM_SPACES.get(model_name.lower(), {})


__all__ = [
    "PARAM_SPACES",
    "get_param_space",
    "validate_lightgbm_params",
    "get_max_leaves_for_depth",
]
