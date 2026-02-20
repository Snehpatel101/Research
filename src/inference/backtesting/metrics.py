"""
Performance metrics for backtesting and strategy evaluation.

This module provides risk-adjusted performance metrics:
- Sharpe Ratio: Risk-adjusted return (excess return / volatility)
- Sortino Ratio: Downside risk-adjusted return
- Calmar Ratio: Return / Max Drawdown
- Max Drawdown: Largest peak-to-trough decline
- Win Rate, Profit Factor, Expectancy
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# ============================================================================
# Core Risk Metrics
# ============================================================================


def calculate_sharpe_ratio(
    returns: np.ndarray | pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """
    Calculate annualized Sharpe ratio.

    Sharpe = (Mean Return - Risk Free Rate) / Std Dev of Returns
           = Annualized Excess Return / Annualized Volatility

    Args:
        returns: Array of period returns (not cumulative)
        risk_free_rate: Annual risk-free rate (e.g., 0.05 for 5%)
        periods_per_year: Number of periods per year (252 for daily, 12 for monthly)

    Returns:
        Annualized Sharpe ratio
    """
    returns = np.asarray(returns)

    if len(returns) == 0:
        return 0.0

    # Convert annual risk-free to per-period
    rf_per_period = (1 + risk_free_rate) ** (1 / periods_per_year) - 1

    # Excess returns
    excess_returns = returns - rf_per_period

    # Mean and std of excess returns
    mean_excess = np.mean(excess_returns)
    std_returns = np.std(returns, ddof=1)

    if std_returns < 1e-12 or np.isnan(std_returns):
        return 0.0

    # Per-period Sharpe
    sharpe_per_period = mean_excess / std_returns

    # Annualize
    annualized_sharpe = sharpe_per_period * np.sqrt(periods_per_year)

    return float(annualized_sharpe)


def calculate_sortino_ratio(
    returns: np.ndarray | pd.Series,
    target_return: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """
    Calculate annualized Sortino ratio.

    Sortino = (Mean Return - Target Return) / Downside Deviation

    Unlike Sharpe, Sortino only penalizes downside volatility,
    making it more appropriate for asymmetric return distributions.

    Args:
        returns: Array of period returns
        target_return: Minimum acceptable return per period (default 0)
        periods_per_year: Number of periods per year

    Returns:
        Annualized Sortino ratio
    """
    returns = np.asarray(returns)

    if len(returns) == 0:
        return 0.0

    # Mean return
    mean_return = np.mean(returns)

    # Downside returns (only negative deviations from target)
    downside_returns = returns - target_return
    downside_returns = downside_returns[downside_returns < 0]

    if len(downside_returns) == 0:
        # No downside - infinite Sortino, cap at large number
        return 10.0 if mean_return > target_return else 0.0

    # Downside deviation (semi-deviation)
    downside_deviation = np.sqrt(np.mean(downside_returns**2))

    if downside_deviation <= 0:
        return 0.0

    # Per-period Sortino
    sortino_per_period = (mean_return - target_return) / downside_deviation

    # Annualize
    annualized_sortino = sortino_per_period * np.sqrt(periods_per_year)

    return float(annualized_sortino)


def calculate_max_drawdown(
    equity_curve: np.ndarray | pd.Series,
) -> tuple[float, int, int]:
    """
    Calculate maximum drawdown from an equity curve.

    Max Drawdown = (Trough - Peak) / Peak

    Args:
        equity_curve: Array of equity values over time

    Returns:
        Tuple of (max_drawdown_pct, peak_index, trough_index)
        max_drawdown_pct is negative (e.g., -0.25 for 25% drawdown)
    """
    equity = np.asarray(equity_curve)

    if len(equity) == 0:
        return 0.0, 0, 0

    # Running maximum (peak)
    running_max = np.maximum.accumulate(equity)

    # Drawdown at each point
    drawdowns = (equity - running_max) / running_max

    # Find maximum drawdown
    trough_idx = int(np.argmin(drawdowns))
    max_dd = float(drawdowns[trough_idx])

    # Find the peak that precedes this trough
    peak_idx = int(np.argmax(equity[: trough_idx + 1])) if trough_idx > 0 else 0

    return max_dd, peak_idx, trough_idx


def calculate_max_drawdown_duration(
    equity_curve: np.ndarray | pd.Series,
) -> tuple[int, int, int]:
    """
    Calculate maximum drawdown duration (time to recover).

    Args:
        equity_curve: Array of equity values

    Returns:
        Tuple of (max_duration, start_index, end_index)
    """
    equity = np.asarray(equity_curve)

    if len(equity) == 0:
        return 0, 0, 0

    running_max = np.maximum.accumulate(equity)
    in_drawdown = equity < running_max

    max_duration = 0
    current_duration = 0
    max_start = 0
    current_start = 0

    for i, is_dd in enumerate(in_drawdown):
        if is_dd:
            if current_duration == 0:
                current_start = i
            current_duration += 1
        else:
            if current_duration > max_duration:
                max_duration = current_duration
                max_start = current_start
            current_duration = 0

    # Check final drawdown
    if current_duration > max_duration:
        max_duration = current_duration
        max_start = current_start

    max_end = max_start + max_duration - 1 if max_duration > 0 else max_start

    return max_duration, max_start, max_end


def calculate_calmar_ratio(
    returns: np.ndarray | pd.Series,
    max_drawdown: float | None = None,
    equity_curve: np.ndarray | pd.Series | None = None,
    periods_per_year: int = 252,
) -> float:
    """
    Calculate Calmar ratio.

    Calmar = CAGR / |Max Drawdown|

    Args:
        returns: Array of period returns
        max_drawdown: Pre-calculated max drawdown (negative value)
        equity_curve: Equity curve for calculating max drawdown if not provided
        periods_per_year: Number of periods per year

    Returns:
        Calmar ratio
    """
    returns = np.asarray(returns)

    if len(returns) == 0:
        return 0.0

    # Calculate CAGR from returns
    total_return = np.prod(1 + returns) - 1
    n_periods = len(returns)
    years = n_periods / periods_per_year

    if years <= 0:
        return 0.0

    cagr = (1 + total_return) ** (1 / years) - 1

    # Get max drawdown
    if max_drawdown is None:
        if equity_curve is not None:
            max_drawdown, _, _ = calculate_max_drawdown(equity_curve)
        else:
            # Calculate equity curve from returns
            equity = np.cumprod(1 + returns)
            max_drawdown, _, _ = calculate_max_drawdown(equity)

    # Calmar ratio (use absolute value of drawdown)
    abs_max_dd = abs(max_drawdown)
    if abs_max_dd <= 0:
        return 10.0 if cagr > 0 else 0.0

    return float(cagr / abs_max_dd)


# ============================================================================
# Trade Statistics
# ============================================================================


def calculate_win_rate(
    trade_returns: np.ndarray | pd.Series,
) -> float:
    """
    Calculate win rate (percentage of winning trades).

    Args:
        trade_returns: Array of individual trade returns

    Returns:
        Win rate as decimal (0-1)
    """
    trade_returns = np.asarray(trade_returns)

    if len(trade_returns) == 0:
        return 0.0

    wins = np.sum(trade_returns > 0)
    return float(wins / len(trade_returns))


def calculate_profit_factor(
    trade_returns: np.ndarray | pd.Series,
) -> float:
    """
    Calculate profit factor (gross profits / gross losses).

    Profit Factor > 1 indicates profitable strategy
    Profit Factor > 2 is considered excellent

    Args:
        trade_returns: Array of individual trade returns

    Returns:
        Profit factor (returns inf if no losses)
    """
    trade_returns = np.asarray(trade_returns)

    if len(trade_returns) == 0:
        return 0.0

    gross_profit = np.sum(trade_returns[trade_returns > 0])
    gross_loss = abs(np.sum(trade_returns[trade_returns < 0]))

    if gross_loss <= 0:
        return float("inf") if gross_profit > 0 else 0.0

    return float(gross_profit / gross_loss)


def calculate_expectancy(
    trade_returns: np.ndarray | pd.Series,
) -> float:
    """
    Calculate expectancy (average profit per trade).

    Expectancy = (Win Rate * Avg Win) - (Loss Rate * Avg Loss)
              = Average return per trade

    Args:
        trade_returns: Array of individual trade returns

    Returns:
        Expected return per trade
    """
    trade_returns = np.asarray(trade_returns)

    if len(trade_returns) == 0:
        return 0.0

    return float(np.mean(trade_returns))


def calculate_expectancy_ratio(
    trade_returns: np.ndarray | pd.Series,
) -> float:
    """
    Calculate expectancy ratio (expectancy / average loss).

    Also known as "edge ratio" - normalizes expectancy by risk.

    Args:
        trade_returns: Array of individual trade returns

    Returns:
        Expectancy ratio
    """
    trade_returns = np.asarray(trade_returns)

    if len(trade_returns) == 0:
        return 0.0

    losses = trade_returns[trade_returns < 0]
    if len(losses) == 0:
        return float("inf") if np.mean(trade_returns) > 0 else 0.0

    avg_loss = abs(np.mean(losses))
    if avg_loss <= 0:
        return 0.0

    expectancy = calculate_expectancy(trade_returns)
    return float(expectancy / avg_loss)


def calculate_payoff_ratio(
    trade_returns: np.ndarray | pd.Series,
) -> float:
    """
    Calculate payoff ratio (average win / average loss).

    Also known as "reward-to-risk ratio".

    Args:
        trade_returns: Array of individual trade returns

    Returns:
        Payoff ratio
    """
    trade_returns = np.asarray(trade_returns)

    wins = trade_returns[trade_returns > 0]
    losses = trade_returns[trade_returns < 0]

    if len(wins) == 0 or len(losses) == 0:
        return 0.0

    avg_win = np.mean(wins)
    avg_loss = abs(np.mean(losses))

    if avg_loss <= 0:
        return float("inf")

    return float(avg_win / avg_loss)


# ============================================================================
# Additional Risk Metrics
# ============================================================================


def calculate_var(
    returns: np.ndarray | pd.Series,
    confidence_level: float = 0.95,
    method: str = "historical",
) -> float:
    """
    Calculate Value at Risk (VaR).

    VaR represents the maximum expected loss at a given confidence level.

    Args:
        returns: Array of returns
        confidence_level: Confidence level (e.g., 0.95 for 95%)
        method: 'historical' or 'parametric'

    Returns:
        VaR as a positive number (potential loss)
    """
    returns = np.asarray(returns)

    if len(returns) == 0:
        return 0.0

    if method == "historical":
        # Historical VaR: percentile of returns
        var = np.percentile(returns, (1 - confidence_level) * 100)
    else:
        # Parametric VaR: assume normal distribution
        from scipy import stats

        mean = np.mean(returns)
        std = np.std(returns, ddof=1)
        z_score = stats.norm.ppf(1 - confidence_level)
        var = mean + z_score * std

    return float(-var) if var < 0 else 0.0


def calculate_cvar(
    returns: np.ndarray | pd.Series,
    confidence_level: float = 0.95,
) -> float:
    """
    Calculate Conditional Value at Risk (CVaR / Expected Shortfall).

    CVaR is the expected loss given that the loss exceeds VaR.

    Args:
        returns: Array of returns
        confidence_level: Confidence level

    Returns:
        CVaR as a positive number
    """
    returns = np.asarray(returns)

    if len(returns) == 0:
        return 0.0

    # Threshold at VaR level
    var_threshold = np.percentile(returns, (1 - confidence_level) * 100)

    # Average of returns below threshold
    tail_returns = returns[returns <= var_threshold]

    if len(tail_returns) == 0:
        return 0.0

    cvar = np.mean(tail_returns)
    return float(-cvar) if cvar < 0 else 0.0


def calculate_ulcer_index(
    equity_curve: np.ndarray | pd.Series,
) -> float:
    """
    Calculate Ulcer Index (measure of drawdown severity).

    UI = sqrt(mean(drawdown^2))

    Lower is better. Measures both depth and duration of drawdowns.

    Args:
        equity_curve: Array of equity values

    Returns:
        Ulcer Index (always positive)
    """
    equity = np.asarray(equity_curve)

    if len(equity) == 0:
        return 0.0

    running_max = np.maximum.accumulate(equity)
    drawdowns = (equity - running_max) / running_max * 100  # Percentage

    ulcer_index = np.sqrt(np.mean(drawdowns**2))

    return float(ulcer_index)


def calculate_ulcer_performance_index(
    returns: np.ndarray | pd.Series,
    equity_curve: np.ndarray | pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """
    Calculate Ulcer Performance Index (UPI).

    UPI = (Return - Risk Free Rate) / Ulcer Index

    Similar to Sharpe but uses Ulcer Index instead of volatility.

    Args:
        returns: Array of returns
        equity_curve: Array of equity values
        risk_free_rate: Annual risk-free rate
        periods_per_year: Number of periods per year

    Returns:
        Ulcer Performance Index
    """
    ulcer = calculate_ulcer_index(equity_curve)

    if ulcer <= 0:
        return 0.0

    returns = np.asarray(returns)
    mean_return = np.mean(returns) * periods_per_year  # Annualize

    return float((mean_return - risk_free_rate) / ulcer)


# ============================================================================
# Comprehensive Metrics Dataclass
# ============================================================================


@dataclass
class PerformanceMetrics:
    """
    Comprehensive performance metrics for a backtest.

    All metrics are calculated from trade returns and equity curve.
    """

    # Return metrics
    total_return: float = 0.0
    cagr: float = 0.0
    mean_return: float = 0.0
    std_return: float = 0.0

    # Risk-adjusted metrics
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    ulcer_index: float = 0.0

    # Drawdown metrics
    max_drawdown: float = 0.0
    max_drawdown_duration: int = 0
    avg_drawdown: float = 0.0

    # Trade statistics
    total_trades: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    expectancy: float = 0.0
    payoff_ratio: float = 0.0

    # Risk metrics
    var_95: float = 0.0
    cvar_95: float = 0.0

    # Time metrics
    start_date: str | None = None
    end_date: str | None = None
    trading_days: int = 0


def calculate_all_metrics(
    trade_returns: np.ndarray | pd.Series,
    equity_curve: np.ndarray | pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> PerformanceMetrics:
    """
    Calculate all performance metrics.

    Args:
        trade_returns: Array of individual trade returns
        equity_curve: Array of equity values over time
        risk_free_rate: Annual risk-free rate
        periods_per_year: Number of periods per year

    Returns:
        PerformanceMetrics dataclass with all metrics
    """
    trade_returns = np.asarray(trade_returns)
    equity_curve = np.asarray(equity_curve)

    # Return metrics
    if len(equity_curve) > 0:
        total_return = (equity_curve[-1] / equity_curve[0]) - 1 if equity_curve[0] > 0 else 0.0
        years = len(equity_curve) / periods_per_year
        cagr = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0.0
    else:
        total_return = 0.0
        cagr = 0.0

    # Period returns from equity curve
    if len(equity_curve) > 1:
        period_returns = np.diff(equity_curve) / equity_curve[:-1]
        mean_return = float(np.mean(period_returns))
        std_return = float(np.std(period_returns, ddof=1))
    else:
        period_returns = np.array([])
        mean_return = 0.0
        std_return = 0.0

    # Risk-adjusted metrics
    sharpe = calculate_sharpe_ratio(period_returns, risk_free_rate, periods_per_year)
    sortino = calculate_sortino_ratio(period_returns, 0.0, periods_per_year)

    # Drawdown metrics
    max_dd, _, _ = calculate_max_drawdown(equity_curve)
    max_dd_duration, _, _ = calculate_max_drawdown_duration(equity_curve)

    # Calmar ratio
    calmar = calculate_calmar_ratio(period_returns, max_dd, equity_curve, periods_per_year)

    # Ulcer index
    ulcer = calculate_ulcer_index(equity_curve)

    # Trade statistics
    total_trades = len(trade_returns)
    win_rate = calculate_win_rate(trade_returns)
    profit_factor = calculate_profit_factor(trade_returns)
    expectancy = calculate_expectancy(trade_returns)
    payoff = calculate_payoff_ratio(trade_returns)

    # Risk metrics
    var_95 = calculate_var(period_returns, 0.95)
    cvar_95 = calculate_cvar(period_returns, 0.95)

    # Average drawdown
    if len(equity_curve) > 0:
        running_max = np.maximum.accumulate(equity_curve)
        drawdowns = (equity_curve - running_max) / running_max
        avg_drawdown = float(np.mean(drawdowns))
    else:
        avg_drawdown = 0.0

    return PerformanceMetrics(
        total_return=float(total_return),
        cagr=float(cagr),
        mean_return=mean_return,
        std_return=std_return,
        sharpe_ratio=sharpe,
        sortino_ratio=sortino,
        calmar_ratio=calmar,
        ulcer_index=ulcer,
        max_drawdown=float(max_dd),
        max_drawdown_duration=max_dd_duration,
        avg_drawdown=avg_drawdown,
        total_trades=total_trades,
        win_rate=win_rate,
        profit_factor=profit_factor,
        expectancy=expectancy,
        payoff_ratio=payoff,
        var_95=var_95,
        cvar_95=cvar_95,
        trading_days=len(equity_curve),
    )


__all__ = [
    # Core metrics
    "calculate_sharpe_ratio",
    "calculate_sortino_ratio",
    "calculate_max_drawdown",
    "calculate_max_drawdown_duration",
    "calculate_calmar_ratio",
    # Trade statistics
    "calculate_win_rate",
    "calculate_profit_factor",
    "calculate_expectancy",
    "calculate_expectancy_ratio",
    "calculate_payoff_ratio",
    # Risk metrics
    "calculate_var",
    "calculate_cvar",
    "calculate_ulcer_index",
    "calculate_ulcer_performance_index",
    # Comprehensive
    "PerformanceMetrics",
    "calculate_all_metrics",
]
