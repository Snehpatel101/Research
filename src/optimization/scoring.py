"""
Shared scoring functions for optimization metrics.

Provides a unified `get_score_fn()` dispatcher used by HyperparameterOptimizer
and FeatureOptimizer to evaluate model quality during Optuna optimization.

Supports both classification metrics (sklearn) and trading proxy metrics
(sharpe_ratio, sortino_ratio, profit_factor) that simulate PnL from
classification predictions.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


def get_score_fn(metric_name: str) -> Callable[[np.ndarray, np.ndarray], float]:
    """
    Get a scoring function by metric name.

    Args:
        metric_name: One of the supported metric names.

    Returns:
        Callable taking (y_true, y_pred) and returning a float score.

    Raises:
        ValueError: If metric_name is not supported.
    """
    if metric_name == "accuracy":
        from sklearn.metrics import accuracy_score

        return lambda y_true, y_pred: float(accuracy_score(y_true, y_pred))

    if metric_name == "f1_weighted":
        from sklearn.metrics import f1_score

        return lambda y_true, y_pred: float(f1_score(y_true, y_pred, average="weighted"))

    if metric_name == "f1_macro":
        from sklearn.metrics import f1_score

        return lambda y_true, y_pred: float(f1_score(y_true, y_pred, average="macro"))

    if metric_name == "precision":
        from sklearn.metrics import precision_score

        return lambda y_true, y_pred: float(
            precision_score(y_true, y_pred, average="weighted", zero_division=0)
        )

    if metric_name == "recall":
        from sklearn.metrics import recall_score

        return lambda y_true, y_pred: float(
            recall_score(y_true, y_pred, average="weighted", zero_division=0)
        )

    if metric_name == "sharpe_ratio":
        return _proxy_sharpe

    if metric_name == "sortino_ratio":
        return _proxy_sortino

    if metric_name == "profit_factor":
        return _proxy_profit_factor

    if metric_name in ("roc_auc", "log_loss"):
        raise ValueError(
            f"Metric '{metric_name}' requires predicted probabilities, "
            "not class predictions. Use f1_weighted or sharpe_ratio instead."
        )

    raise ValueError(
        f"Unknown scoring metric: '{metric_name}'. "
        "Supported: accuracy, f1_weighted, f1_macro, precision, recall, "
        "sharpe_ratio, sortino_ratio, profit_factor"
    )


# =============================================================================
# Trading proxy metrics
# =============================================================================
# These simulate PnL from classification predictions:
#   correct prediction → +1 return, incorrect → -1 return.
# This is a PROXY — real Sharpe/Sortino come from backtesting with prices.
# Matches the logic in five_dimension_objective.py:441-487.
# =============================================================================

_ANNUALIZATION_FACTOR = np.sqrt(252 * 78)  # 1-min bars, 78 per session


def _proxy_sharpe(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Proxy Sharpe ratio from classification accuracy."""
    returns = np.where(y_pred == y_true, 1.0, -1.0)
    std = returns.std()
    if std < 1e-8:
        return 0.0
    return float((returns.mean() / std) * _ANNUALIZATION_FACTOR)


def _proxy_sortino(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Proxy Sortino ratio — penalizes downside deviation only."""
    returns = np.where(y_pred == y_true, 1.0, -1.0)
    downside = returns[returns < 0]
    if len(downside) == 0:
        return float(returns.mean() * _ANNUALIZATION_FACTOR)
    downside_std = downside.std()
    if downside_std < 1e-8:
        return 0.0
    return float((returns.mean() / downside_std) * _ANNUALIZATION_FACTOR)


def _proxy_profit_factor(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Proxy profit factor: gross wins / gross losses."""
    returns = np.where(y_pred == y_true, 1.0, -1.0)
    wins = returns[returns > 0].sum()
    losses = abs(returns[returns < 0].sum())
    if losses < 1e-8:
        return float(wins) if wins > 0 else 0.0
    return float(wins / losses)
