"""
Backtesting module for strategy simulation and performance evaluation.

This module provides production-grade backtesting capabilities:
- Transaction costs and slippage modeling
- Position sizing (Kelly, fixed fractional, volatility-targeted)
- Risk-adjusted metrics (Sharpe, Sortino, Calmar, max drawdown)
- Equity curve simulation and visualization

Import paths:
    # New (preferred):
    from src.inference.backtesting import Backtester, BacktestConfig

    # Legacy (still works, deprecation warning):
    from src.inference.backtesting import Backtester, BacktestConfig

Quick Start:
-----------
    from src.inference.backtesting import Backtester, BacktestConfig

    # Configure backtest
    config = BacktestConfig.for_mes(
        initial_equity=100000.0,
        position_sizing_method="fixed_fractional",
        risk_per_trade=0.02,
    )

    # Run backtest
    backtester = Backtester(predictions, prices, config)
    result = backtester.run()

    # Print summary
    result.print_summary()

    # Plot equity curve
    result.equity_curve.plot()

Components:
----------
    costs.py: Transaction costs and slippage models
        - TransactionCosts: Commission, fees, slippage configuration
        - FixedSlippage, LinearSlippage, SquareRootSlippage: Slippage models
        - CostCalculator: Unified cost calculation

    position_sizing.py: Position sizing algorithms
        - KellyCriterion: Optimal growth-maximizing sizing
        - FixedFractional: Risk a fixed % per trade
        - VolatilityTargeted: Scale by inverse volatility
        - EqualWeight, FixedContracts: Simple methods

    metrics.py: Performance metrics
        - calculate_sharpe_ratio: Risk-adjusted return
        - calculate_sortino_ratio: Downside risk-adjusted return
        - calculate_max_drawdown: Peak-to-trough decline
        - calculate_calmar_ratio: CAGR / Max Drawdown
        - calculate_var, calculate_cvar: Risk metrics
        - PerformanceMetrics: Comprehensive metrics dataclass

    equity_curve.py: Equity tracking and visualization
        - Trade: Single trade record
        - EquityCurve: Equity tracking with plotting
        - create_equity_curve_from_trades: Factory function

    backtest.py: Main backtester
        - BacktestConfig: Backtest configuration
        - Backtester: Main simulation class
        - BacktestResult: Results container
        - run_backtest: Convenience function
"""

from __future__ import annotations

# Backtest core
from .backtest import (
    BacktestConfig,
    Backtester,
    BacktestResult,
    ExecutionModel,
    Position,
    run_backtest,
    run_walk_forward_backtest,
)

# Transaction costs
from .costs import (
    BaseSlippageModel,
    CostCalculator,
    FixedSlippage,
    LinearSlippage,
    SlippageModel,
    SquareRootSlippage,
    TransactionCosts,
    VolatilityScaledSlippage,
    create_cost_calculator,
)

# Equity curve
from .equity_curve import (
    EquityCurve,
    Trade,
    create_equity_curve_from_trades,
)

# Execution (Phase 12A-3: Market Hours Filtering)
from .execution import (
    NY_SESSION_END,
    NY_SESSION_START,
    MarketHoursFilter,
    create_market_hours_filter,
)

# Performance metrics
from .metrics import (
    PerformanceMetrics,
    calculate_all_metrics,
    calculate_calmar_ratio,
    calculate_cvar,
    calculate_expectancy,
    calculate_expectancy_ratio,
    calculate_max_drawdown,
    calculate_max_drawdown_duration,
    calculate_payoff_ratio,
    calculate_profit_factor,
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
    calculate_ulcer_index,
    calculate_ulcer_performance_index,
    calculate_var,
    calculate_win_rate,
)

# Position sizing
from .position_sizing import (
    BasePositionSizer,
    EqualWeight,
    FixedContracts,
    FixedFractional,
    KellyCriterion,
    PositionSizerConfig,
    PositionSizingMethod,
    VolatilityTargeted,
    create_position_sizer,
)

__version__ = "0.1.0"

__all__ = [
    # Version
    "__version__",
    # Backtester
    "Backtester",
    "BacktestConfig",
    "BacktestResult",
    "ExecutionModel",
    "Position",
    "run_backtest",
    "run_walk_forward_backtest",
    # Transaction costs
    "TransactionCosts",
    "SlippageModel",
    "BaseSlippageModel",
    "FixedSlippage",
    "LinearSlippage",
    "SquareRootSlippage",
    "VolatilityScaledSlippage",
    "CostCalculator",
    "create_cost_calculator",
    # Position sizing
    "PositionSizingMethod",
    "BasePositionSizer",
    "KellyCriterion",
    "FixedFractional",
    "VolatilityTargeted",
    "EqualWeight",
    "FixedContracts",
    "PositionSizerConfig",
    "create_position_sizer",
    # Metrics
    "calculate_sharpe_ratio",
    "calculate_sortino_ratio",
    "calculate_max_drawdown",
    "calculate_max_drawdown_duration",
    "calculate_calmar_ratio",
    "calculate_win_rate",
    "calculate_profit_factor",
    "calculate_expectancy",
    "calculate_expectancy_ratio",
    "calculate_payoff_ratio",
    "calculate_var",
    "calculate_cvar",
    "calculate_ulcer_index",
    "calculate_ulcer_performance_index",
    "PerformanceMetrics",
    "calculate_all_metrics",
    # Equity curve
    "Trade",
    "EquityCurve",
    "create_equity_curve_from_trades",
    # Execution (Phase 12A-3)
    "MarketHoursFilter",
    "create_market_hours_filter",
    "NY_SESSION_START",
    "NY_SESSION_END",
]
