"""
Training and evaluation metrics for model training.

This module provides classification and trading-specific metrics
for evaluating model performance.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
) -> dict[str, Any]:
    """
    Compute classification metrics.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_proba: Predicted probabilities

    Returns:
        Dict with accuracy, F1 scores, confusion matrix, etc.
    """
    from sklearn.metrics import (
        accuracy_score,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
    )

    # Basic metrics
    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
    recall = recall_score(y_true, y_pred, average="macro", zero_division=0)

    # Per-class F1
    classes = sorted(np.unique(np.concatenate([y_true, y_pred])))
    per_class_f1 = f1_score(y_true, y_pred, average=None, labels=classes, zero_division=0)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=classes)

    # Class names for readability (trading labels: -1=short, 0=neutral, 1=long)
    class_names = {-1: "short", 0: "neutral", 1: "long"}

    return {
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "precision": float(precision),
        "recall": float(recall),
        "per_class_f1": {
            class_names.get(c, str(c)): float(f1)
            for c, f1 in zip(classes, per_class_f1, strict=False)
        },
        "confusion_matrix": cm.tolist(),
        "n_samples": len(y_true),
    }


def compute_trading_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, Any]:
    """
    Compute trading metrics for quick model comparison.

    Note: This is a simplified version for quick model evaluation.
    Full backtesting with realistic transaction costs, slippage, and
    position sizing is done in Phase 3+.

    Args:
        y_true: True labels (-1=short, 0=neutral, 1=long)
        y_pred: Predicted labels (-1=short, 0=neutral, 1=long)

    Returns:
        Dict with trading statistics
    """
    # Signal distribution
    long_signals = (y_pred == 1).sum()
    short_signals = (y_pred == -1).sum()
    neutral_signals = (y_pred == 0).sum()
    total_positions = long_signals + short_signals

    # Overall position accuracy
    position_mask = y_pred != 0
    if position_mask.sum() > 0:
        correct_positions = (y_pred[position_mask] == y_true[position_mask]).sum()
        position_win_rate = correct_positions / position_mask.sum()
    else:
        position_win_rate = 0.0

    # Long/short accuracy (directional edge)
    long_mask = y_pred == 1
    short_mask = y_pred == -1

    long_accuracy = 0.0
    if long_mask.sum() > 0:
        long_accuracy = (y_pred[long_mask] == y_true[long_mask]).sum() / long_mask.sum()

    short_accuracy = 0.0
    if short_mask.sum() > 0:
        short_accuracy = (y_pred[short_mask] == y_true[short_mask]).sum() / short_mask.sum()

    # Consecutive wins/losses (measure of streakiness)
    if position_mask.sum() > 0:
        position_correct = (y_pred[position_mask] == y_true[position_mask]).astype(int)

        # Find consecutive sequences
        max_consecutive_wins = 0
        max_consecutive_losses = 0
        current_wins = 0
        current_losses = 0

        for is_correct in position_correct:
            if is_correct:
                current_wins += 1
                current_losses = 0
                max_consecutive_wins = max(max_consecutive_wins, current_wins)
            else:
                current_losses += 1
                current_wins = 0
                max_consecutive_losses = max(max_consecutive_losses, current_losses)
    else:
        max_consecutive_wins = 0
        max_consecutive_losses = 0

    # Position-based Sharpe (simplified, assumes returns are correct predictions)
    # This is a proxy - real Sharpe requires actual returns
    if position_mask.sum() > 0:
        # Assume correct prediction = +1 return, incorrect = -1 return
        position_returns = np.where(y_pred[position_mask] == y_true[position_mask], 1.0, -1.0)
        position_sharpe = (
            position_returns.mean() / position_returns.std() if position_returns.std() > 0 else 0.0
        )
    else:
        position_sharpe = 0.0

    return {
        # Signal distribution
        "long_signals": int(long_signals),
        "short_signals": int(short_signals),
        "neutral_signals": int(neutral_signals),
        "total_positions": int(total_positions),
        "position_rate": float(total_positions / len(y_pred)) if len(y_pred) > 0 else 0.0,
        # Accuracy metrics
        "position_win_rate": float(position_win_rate),
        "long_accuracy": float(long_accuracy),
        "short_accuracy": float(short_accuracy),
        "directional_edge": float(abs(long_accuracy - short_accuracy)),  # Measures directional bias
        # Streak metrics
        "max_consecutive_wins": int(max_consecutive_wins),
        "max_consecutive_losses": int(max_consecutive_losses),
        # Risk metrics (simplified)
        "position_sharpe": float(position_sharpe),
        # Metadata
        "note": "Simplified metrics for quick comparison. Use Phase 3+ for full backtest.",
    }


__all__ = [
    "compute_classification_metrics",
    "compute_trading_metrics",
]
