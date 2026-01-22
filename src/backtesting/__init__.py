"""
DEPRECATED: Import from 'src.inference.backtesting' instead.

This module will be removed in v2.0.0.

Migration:
    # Old (deprecated)
    from src.backtesting import Backtester, BacktestConfig

    # New (preferred)
    from src.inference.backtesting import Backtester, BacktestConfig
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

# Names that will be lazily loaded from src.inference.backtesting
_BACKTESTING_EXPORTS = [
    # Backtester
    "BacktestConfig",
    "BacktestResult",
    "Backtester",
    "ExecutionModel",
    "Position",
    "run_backtest",
    "run_walk_forward_backtest",
    # Transaction costs
    "BaseSlippageModel",
    "CostCalculator",
    "FixedSlippage",
    "LinearSlippage",
    "SlippageModel",
    "SquareRootSlippage",
    "TransactionCosts",
    "VolatilityScaledSlippage",
    "create_cost_calculator",
    # Equity curve
    "EquityCurve",
    "Trade",
    "create_equity_curve_from_trades",
    # Performance metrics
    "PerformanceMetrics",
    "calculate_all_metrics",
    "calculate_calmar_ratio",
    "calculate_cvar",
    "calculate_expectancy",
    "calculate_expectancy_ratio",
    "calculate_max_drawdown",
    "calculate_max_drawdown_duration",
    "calculate_payoff_ratio",
    "calculate_profit_factor",
    "calculate_sharpe_ratio",
    "calculate_sortino_ratio",
    "calculate_ulcer_index",
    "calculate_ulcer_performance_index",
    "calculate_var",
    "calculate_win_rate",
    # Position sizing
    "BasePositionSizer",
    "EqualWeight",
    "FixedContracts",
    "FixedFractional",
    "KellyCriterion",
    "PositionSizerConfig",
    "PositionSizingMethod",
    "VolatilityTargeted",
    "create_position_sizer",
]


def __getattr__(name: str):
    """Lazy import to avoid circular imports."""
    if name in _BACKTESTING_EXPORTS:
        # Show deprecation warning
        warnings.warn(
            f"Importing '{name}' from 'src.backtesting' is deprecated. "
            f"Use 'from src.inference.backtesting import {name}' instead. "
            "This will be removed in v2.0.0.",
            DeprecationWarning,
            stacklevel=2,
        )
        # Import from new location
        from src.inference import backtesting as backtesting_module
        return getattr(backtesting_module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    """Return list of available names."""
    return _BACKTESTING_EXPORTS + ["__version__"]


__version__ = "0.1.0"

__all__ = ["__version__"] + _BACKTESTING_EXPORTS
