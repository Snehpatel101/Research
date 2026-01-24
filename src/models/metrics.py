"""
Training and evaluation metrics for model training.

This module provides classification and trading-specific metrics
for evaluating model performance.

For production backtesting with real P&L, transaction costs, and
position sizing, use src.backtesting module.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


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
        matthews_corrcoef,
        precision_score,
        recall_score,
    )

    # Basic metrics
    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
    recall = recall_score(y_true, y_pred, average="macro", zero_division=0)

    # Matthews Correlation Coefficient - gold standard for imbalanced classification
    # MCC ranges from -1 (total disagreement) to +1 (perfect prediction)
    # MCC = 0 indicates random prediction
    mcc = matthews_corrcoef(y_true, y_pred)

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
        "mcc": float(mcc),  # Matthews Correlation Coefficient
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
    prices: pd.DataFrame | None = None,
    timestamps: np.ndarray | None = None,
    point_value: float = 5.0,
    commission_per_contract: float = 2.50,
    slippage_ticks: float = 1.0,
    tick_value: float = 1.25,
) -> dict[str, Any]:
    """
    Compute trading metrics with optional realistic P&L calculation.

    When prices are provided, calculates real P&L with transaction costs.
    Otherwise falls back to simplified directional metrics.

    Args:
        y_true: True labels (-1=short, 0=neutral, 1=long)
        y_pred: Predicted labels (-1=short, 0=neutral, 1=long)
        prices: Optional DataFrame with 'close' column for real P&L
        timestamps: Optional array of timestamps aligned with predictions
        point_value: Dollar value per point (5.0 for MES)
        commission_per_contract: Round-trip commission
        slippage_ticks: Expected slippage in ticks
        tick_value: Dollar value per tick

    Returns:
        Dict with trading statistics including real metrics when prices available
    """
    # Signal distribution
    long_signals = int((y_pred == 1).sum())
    short_signals = int((y_pred == -1).sum())
    neutral_signals = int((y_pred == 0).sum())
    total_positions = long_signals + short_signals

    # Overall position accuracy
    position_mask = y_pred != 0
    if position_mask.sum() > 0:
        correct_positions = (y_pred[position_mask] == y_true[position_mask]).sum()
        position_win_rate = float(correct_positions / position_mask.sum())
    else:
        position_win_rate = 0.0

    # Long/short accuracy (directional edge)
    long_mask = y_pred == 1
    short_mask = y_pred == -1

    long_accuracy = 0.0
    if long_mask.sum() > 0:
        long_accuracy = float((y_pred[long_mask] == y_true[long_mask]).sum() / long_mask.sum())

    short_accuracy = 0.0
    if short_mask.sum() > 0:
        short_accuracy = float((y_pred[short_mask] == y_true[short_mask]).sum() / short_mask.sum())

    # Consecutive wins/losses (measure of streakiness)
    if position_mask.sum() > 0:
        position_correct = (y_pred[position_mask] == y_true[position_mask]).astype(int)
        max_consecutive_wins, max_consecutive_losses = _calculate_streaks(position_correct)
    else:
        max_consecutive_wins = 0
        max_consecutive_losses = 0

    # Build base metrics
    metrics: dict[str, Any] = {
        "long_signals": long_signals,
        "short_signals": short_signals,
        "neutral_signals": neutral_signals,
        "total_positions": total_positions,
        "position_rate": float(total_positions / len(y_pred)) if len(y_pred) > 0 else 0.0,
        "position_win_rate": position_win_rate,
        "long_accuracy": long_accuracy,
        "short_accuracy": short_accuracy,
        "directional_edge": float(abs(long_accuracy - short_accuracy)),
        "max_consecutive_wins": max_consecutive_wins,
        "max_consecutive_losses": max_consecutive_losses,
    }

    # If prices provided, calculate real P&L metrics
    if prices is not None and "close" in prices.columns:
        real_metrics = _compute_real_pnl_metrics(
            y_pred=y_pred,
            prices=prices,
            point_value=point_value,
            commission_per_contract=commission_per_contract,
            slippage_ticks=slippage_ticks,
            tick_value=tick_value,
        )
        metrics.update(real_metrics)
        metrics["metrics_type"] = "real_pnl"
    else:
        # Simplified proxy metrics when no prices
        proxy_metrics = _compute_proxy_metrics(y_true, y_pred, position_mask)
        metrics.update(proxy_metrics)
        metrics["metrics_type"] = "proxy"
        metrics["note"] = "Proxy metrics. Provide prices for real P&L calculation."

    return metrics


def _calculate_streaks(position_correct: np.ndarray) -> tuple[int, int]:
    """Calculate max consecutive wins and losses."""
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

    return max_consecutive_wins, max_consecutive_losses


def _compute_proxy_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    position_mask: np.ndarray,
) -> dict[str, Any]:
    """Compute proxy metrics when real prices not available."""
    if position_mask.sum() > 0:
        # Proxy returns: correct = +1, incorrect = -1
        position_returns = np.where(y_pred[position_mask] == y_true[position_mask], 1.0, -1.0)
        proxy_sharpe = (
            float(position_returns.mean() / position_returns.std())
            if position_returns.std() > 0
            else 0.0
        )
        proxy_expectancy = float(position_returns.mean())
        float((position_returns > 0).sum() / len(position_returns))
    else:
        proxy_sharpe = 0.0
        proxy_expectancy = 0.0

    return {
        "sharpe_ratio": proxy_sharpe,
        "sortino_ratio": 0.0,
        "calmar_ratio": 0.0,
        "max_drawdown_pct": 0.0,
        "expectancy": proxy_expectancy,
        "profit_factor": 0.0,
        "total_pnl": 0.0,
        "gross_pnl": 0.0,
        "total_costs": 0.0,
    }


def _compute_real_pnl_metrics(
    y_pred: np.ndarray,
    prices: pd.DataFrame,
    point_value: float,
    commission_per_contract: float,
    slippage_ticks: float,
    tick_value: float,
) -> dict[str, Any]:
    """
    Compute real P&L metrics from predictions and prices.

    This simulates simple signal-following trading:
    - Enter on signal change (0->1 or 0->-1)
    - Exit on signal reversal or neutral
    - Calculate actual P&L with costs
    """
    from src.inference.backtesting.metrics import (
        calculate_calmar_ratio,
        calculate_expectancy,
        calculate_max_drawdown,
        calculate_profit_factor,
        calculate_sharpe_ratio,
        calculate_sortino_ratio,
        calculate_win_rate,
    )

    # Get close prices aligned with predictions
    if len(prices) != len(y_pred):
        # Truncate to match
        n = min(len(prices), len(y_pred))
        close_prices = prices["close"].values[:n]
        signals = y_pred[:n]
    else:
        close_prices = prices["close"].values
        signals = y_pred

    # Calculate trade returns
    trade_returns_list: list[float] = []
    gross_returns_list: list[float] = []
    costs_list: list[float] = []

    position = 0  # Current position: -1, 0, 1
    entry_price = 0.0

    cost_per_contract = commission_per_contract + 2 * slippage_ticks * tick_value

    for i in range(len(signals)):
        signal = int(signals[i])

        # Check for position change
        if signal != position:
            # Close existing position
            if position != 0:
                exit_price = close_prices[i]
                price_change = exit_price - entry_price
                gross_pnl = position * price_change * point_value
                net_pnl = gross_pnl - cost_per_contract

                trade_returns_list.append(net_pnl)
                gross_returns_list.append(gross_pnl)
                costs_list.append(cost_per_contract)

            # Open new position
            if signal != 0:
                entry_price = close_prices[i]
                position = signal
            else:
                position = 0

    # Close final position if open
    if position != 0 and len(close_prices) > 0:
        exit_price = close_prices[-1]
        price_change = exit_price - entry_price
        gross_pnl = position * price_change * point_value
        net_pnl = gross_pnl - cost_per_contract

        trade_returns_list.append(net_pnl)
        gross_returns_list.append(gross_pnl)
        costs_list.append(cost_per_contract)

    # Calculate metrics from trade returns
    trade_returns = np.array(trade_returns_list)
    gross_returns = np.array(gross_returns_list)

    if len(trade_returns) == 0:
        return {
            "sharpe_ratio": 0.0,
            "sortino_ratio": 0.0,
            "calmar_ratio": 0.0,
            "max_drawdown_pct": 0.0,
            "expectancy": 0.0,
            "profit_factor": 0.0,
            "total_pnl": 0.0,
            "gross_pnl": 0.0,
            "total_costs": 0.0,
            "n_trades": 0,
        }

    # Build equity curve
    initial_equity = 100000.0
    equity_curve = initial_equity + np.cumsum(trade_returns)
    equity_curve = np.insert(equity_curve, 0, initial_equity)

    # Calculate period returns from equity
    period_returns = np.diff(equity_curve) / equity_curve[:-1]

    # Risk metrics
    sharpe = calculate_sharpe_ratio(period_returns, periods_per_year=252)
    sortino = calculate_sortino_ratio(period_returns, periods_per_year=252)
    max_dd, _, _ = calculate_max_drawdown(equity_curve)
    calmar = calculate_calmar_ratio(period_returns, max_dd, periods_per_year=252)

    # Trade metrics
    win_rate = calculate_win_rate(trade_returns)
    profit_factor = calculate_profit_factor(trade_returns)
    expectancy = calculate_expectancy(trade_returns)

    return {
        "sharpe_ratio": float(sharpe),
        "sortino_ratio": float(sortino),
        "calmar_ratio": float(calmar),
        "max_drawdown_pct": float(max_dd * 100),
        "expectancy": float(expectancy),
        "profit_factor": float(profit_factor) if not np.isinf(profit_factor) else 999.0,
        "total_pnl": float(trade_returns.sum()),
        "gross_pnl": float(gross_returns.sum()),
        "total_costs": float(sum(costs_list)),
        "n_trades": len(trade_returns),
        "win_rate": float(win_rate),
        "avg_win": (
            float(trade_returns[trade_returns > 0].mean()) if (trade_returns > 0).any() else 0.0
        ),
        "avg_loss": (
            float(trade_returns[trade_returns < 0].mean()) if (trade_returns < 0).any() else 0.0
        ),
    }


def compute_backtest_metrics(
    predictions: pd.DataFrame,
    prices: pd.DataFrame,
    initial_equity: float = 100000.0,
    point_value: float = 5.0,
    commission_per_contract: float = 2.50,
    slippage_ticks: float = 1.0,
    tick_value: float = 1.25,
) -> dict[str, Any]:
    """
    Run full backtest and return comprehensive metrics.

    This is a convenience function that wraps the backtesting module
    for quick model evaluation.

    Args:
        predictions: DataFrame with 'prediction' column (-1, 0, 1)
        prices: DataFrame with OHLC columns
        initial_equity: Starting portfolio value
        point_value: Dollar value per point
        commission_per_contract: Round-trip commission
        slippage_ticks: Expected slippage
        tick_value: Dollar value per tick

    Returns:
        Dict with comprehensive backtest metrics
    """
    from src.inference.backtesting import BacktestConfig, Backtester

    config = BacktestConfig(
        initial_equity=initial_equity,
        position_sizing_method="fixed_contracts",
        fixed_contracts=1,
        point_value=point_value,
        commission_per_contract=commission_per_contract,
        slippage_ticks=slippage_ticks,
        tick_value=tick_value,
    )

    backtester = Backtester(predictions, prices, config)
    result = backtester.run()

    return result.summary()


def compute_regime_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    prices: pd.DataFrame,
    timestamps: pd.DatetimeIndex,
    classifier_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Compute regime-conditional metrics for model evaluation.

    This function provides a breakdown of model performance across different
    market regimes (volatility, trend, time-of-day), helping identify where
    models excel or underperform.

    Args:
        y_true: True labels (-1=short, 0=neutral, 1=long)
        y_pred: Predicted labels
        y_proba: Prediction probabilities (n_samples, n_classes)
        prices: OHLCV price DataFrame for regime classification and returns
        timestamps: DatetimeIndex aligned with predictions
        classifier_config: Optional dict to configure RegimeClassifier

    Returns:
        Dict with regime breakdowns:
        - overall: Overall metrics
        - volatility_breakdown: Performance by volatility regime (low/normal/high)
        - trend_breakdown: Performance by trend regime (trending/mean_reverting)
        - time_of_day_breakdown: Performance by trading session
        - weak_regimes: List of underperforming regimes
        - summary: Text summary of regime performance

    Example:
        >>> from src.models.metrics import compute_regime_metrics
        >>> regime_metrics = compute_regime_metrics(
        ...     y_true=y_test,
        ...     y_pred=predictions,
        ...     y_proba=probabilities,
        ...     prices=test_prices,
        ...     timestamps=test_timestamps,
        ... )
        >>> print(regime_metrics["summary"])
    """
    from .regime_evaluation import (
        RegimeClassifier,
        RegimeEvaluator,
        get_regime_summary,
    )

    # Create classifier and evaluator
    classifier_config = classifier_config or {}
    classifier = RegimeClassifier(**classifier_config)
    evaluator = RegimeEvaluator(classifier=classifier)

    # Run evaluation
    result = evaluator.evaluate(
        y_true=y_true,
        y_pred=y_pred,
        y_proba=y_proba,
        prices=prices,
        timestamps=timestamps,
    )

    # Convert to serializable dict
    output = result.to_dict()

    # Add weak regimes detection
    weak_regimes = evaluator.identify_weak_regimes(
        result, accuracy_threshold=0.5, sharpe_threshold=0.0
    )
    output["weak_regimes"] = weak_regimes

    # Add text summary
    output["summary"] = get_regime_summary(result)

    return output


def compute_metrics_with_regime_breakdown(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    prices: pd.DataFrame | None = None,
    timestamps: pd.DatetimeIndex | None = None,
    include_regime: bool = True,
) -> dict[str, Any]:
    """
    Compute comprehensive metrics including optional regime breakdown.

    This function combines classification metrics, trading metrics, and
    regime-conditional analysis into a single comprehensive result.

    Args:
        y_true: True labels (-1=short, 0=neutral, 1=long)
        y_pred: Predicted labels
        y_proba: Prediction probabilities
        prices: Optional OHLCV DataFrame for trading metrics and regime analysis
        timestamps: Optional DatetimeIndex for time-of-day analysis
        include_regime: Whether to include regime breakdown (requires prices/timestamps)

    Returns:
        Dict with:
        - classification: Classification metrics (accuracy, F1, etc.)
        - trading: Trading metrics (Sharpe, win rate, etc.)
        - regime: Regime breakdown (if include_regime=True and data available)

    Example:
        >>> metrics = compute_metrics_with_regime_breakdown(
        ...     y_true=y_test,
        ...     y_pred=predictions,
        ...     y_proba=probabilities,
        ...     prices=test_prices,
        ...     timestamps=test_timestamps,
        ...     include_regime=True,
        ... )
        >>> print(f"Accuracy: {metrics['classification']['accuracy']:.3f}")
        >>> if 'regime' in metrics:
        ...     print(metrics['regime']['summary'])
    """
    import logging

    logger = logging.getLogger(__name__)

    # Always compute classification metrics
    classification_metrics = compute_classification_metrics(y_true, y_pred, y_proba)

    # Compute trading metrics
    trading_metrics = compute_trading_metrics(
        y_true=y_true,
        y_pred=y_pred,
        prices=prices,
        timestamps=timestamps,
    )

    result: dict[str, Any] = {
        "classification": classification_metrics,
        "trading": trading_metrics,
    }

    # Add regime breakdown if requested and data available
    if include_regime and prices is not None and timestamps is not None:
        try:
            regime_metrics = compute_regime_metrics(
                y_true=y_true,
                y_pred=y_pred,
                y_proba=y_proba,
                prices=prices,
                timestamps=timestamps,
            )
            result["regime"] = regime_metrics
        except Exception as e:
            logger.warning(f"Failed to compute regime metrics: {e}")
            result["regime_error"] = str(e)

    return result


__all__ = [
    "compute_classification_metrics",
    "compute_trading_metrics",
    "compute_backtest_metrics",
    "compute_regime_metrics",
    "compute_metrics_with_regime_breakdown",
]
