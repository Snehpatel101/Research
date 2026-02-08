"""
Shared scoring and splitting utilities for optimization.

Provides:
- `get_score_fn()` dispatcher for classification/trading metrics
- `purged_train_val_split()` for leakage-free train/val splitting

Used by HyperparameterOptimizer, FeatureOptimizer, and OptimizationPipeline.
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


# =============================================================================
# Purged train/val split
# =============================================================================

# Default purge/embargo values from horizon_config (static to avoid circular imports)
_DEFAULT_PURGE_BARS = 60  # 3 * max(default horizons=[5,10,15,20])
_DEFAULT_EMBARGO_BARS = 1440  # 5 days at 5-min bars


def purged_train_val_split(
    n_samples: int,
    train_ratio: float = 0.8,
    purge_bars: int | None = None,
    embargo_bars: int | None = None,
) -> tuple[int, int]:
    """
    Compute train/val boundary indices with a purge+embargo gap.

    Returns (train_end, val_start) where:
    - Train uses indices [0 : train_end]
    - Val uses indices   [val_start : n_samples]
    - The gap [train_end : val_start] is the purge+embargo buffer

    If the gap would leave fewer than 10% of samples for validation,
    the gap is clamped so val still gets at least 10%.

    Args:
        n_samples: Total number of samples.
        train_ratio: Fraction of data allocated to training (before gap).
        purge_bars: Bars to purge between train and val (default: 60).
        embargo_bars: Embargo bars after purge (default: 1440).

    Returns:
        (train_end, val_start) index pair.

    Raises:
        ValueError: If n_samples is too small for any meaningful split.
    """
    if purge_bars is None:
        purge_bars = _DEFAULT_PURGE_BARS
    if embargo_bars is None:
        embargo_bars = _DEFAULT_EMBARGO_BARS

    if n_samples < 20:
        raise ValueError(
            f"Cannot create purged split with {n_samples} samples (minimum 20)"
        )

    split_idx = int(n_samples * train_ratio)
    gap = purge_bars + embargo_bars

    # Ensure val gets at least 10% of samples
    min_val = max(1, n_samples // 10)
    max_gap = n_samples - split_idx - min_val
    gap = max(0, min(gap, max_gap))

    train_end = split_idx
    val_start = split_idx + gap

    # Safety: ensure at least 1 training sample and 1 val sample
    train_end = max(1, train_end)
    val_start = min(n_samples - 1, val_start)

    return train_end, val_start
