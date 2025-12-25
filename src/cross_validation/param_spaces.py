"""
Hyperparameter search spaces for Optuna tuning.

Defines search spaces for all supported model types.
"""
from typing import Any, Dict

# =============================================================================
# HYPERPARAMETER SEARCH SPACES
# =============================================================================

PARAM_SPACES: Dict[str, Dict[str, Dict[str, Any]]] = {
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
    "lightgbm": {
        "n_estimators": {"type": "int", "low": 100, "high": 500},
        "max_depth": {"type": "int", "low": 3, "high": 10},
        "num_leaves": {"type": "int", "low": 20, "high": 100},
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


def get_param_space(model_name: str) -> Dict[str, Dict[str, Any]]:
    """Get hyperparameter search space for a model."""
    return PARAM_SPACES.get(model_name.lower(), {})


__all__ = ["PARAM_SPACES", "get_param_space"]
