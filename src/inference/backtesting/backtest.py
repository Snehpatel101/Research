"""
Main backtester implementation for strategy simulation.

This module provides the Backtester class that simulates trading
with realistic execution, transaction costs, and position sizing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd

from .costs import CostCalculator, TransactionCosts
from .equity_curve import EquityCurve, Trade
from .execution import MarketHoursFilter
from .metrics import PerformanceMetrics
from .position_sizing import BasePositionSizer, create_position_sizer


class ExecutionModel(Enum):
    """Order execution model types."""

    MARKET_ON_CLOSE = "market_on_close"
    MARKET_ON_OPEN = "market_on_open"
    FILL_AT_SIGNAL = "fill_at_signal"
    VWAP = "vwap"


@dataclass
class BacktestConfig:
    """
    Configuration for backtester.

    Attributes:
        initial_equity: Starting portfolio value
        position_sizing: Method for calculating position size
        execution_model: How orders are executed
        allow_short: Whether to allow short positions
        allow_pyramiding: Whether to allow adding to positions
        max_positions: Maximum concurrent positions
        commission_per_contract: Round-trip commission
        slippage_ticks: Expected slippage in ticks
        tick_value: Dollar value per tick
        tick_size: Minimum price increment
        point_value: Dollar value per point (full contract value factor)
        risk_per_trade: Risk per trade for position sizing
        kelly_fraction: Kelly fraction for Kelly sizing
        target_volatility: Target volatility for vol-targeted sizing
        min_holding_period: Minimum bars to hold position
        max_holding_period: Maximum bars to hold position (0 = no limit)
    """

    initial_equity: float = 100000.0
    position_sizing: str = "fixed_contracts"
    execution_model: ExecutionModel = ExecutionModel.MARKET_ON_CLOSE
    allow_short: bool = True
    allow_pyramiding: bool = False
    max_positions: int = 1
    commission_per_contract: float = 2.50
    slippage_ticks: float = 1.0
    tick_value: float = 1.25
    tick_size: float = 0.25
    point_value: float = 5.0
    risk_per_trade: float = 0.02
    kelly_fraction: float = 0.25
    target_volatility: float = 0.10
    fixed_contracts: int = 1
    min_holding_period: int = 1
    max_holding_period: int = 0
    max_drawdown_threshold: float = 0.10
    daily_loss_threshold: float = 0.02
    consecutive_loss_limit: int = 5
    enable_market_hours_filter: bool = True
    contract_symbol: str = "MES"

    @classmethod
    def for_mes(cls, **kwargs: Any) -> BacktestConfig:
        """Create config for Micro E-mini S&P 500."""
        defaults: dict[str, Any] = {
            "tick_value": 1.25,
            "tick_size": 0.25,
            "point_value": 5.0,
        }
        defaults.update(kwargs)
        return cls(**defaults)

    @classmethod
    def for_mgc(cls, **kwargs: Any) -> BacktestConfig:
        """Create config for Micro Gold."""
        defaults: dict[str, Any] = {
            "tick_value": 1.00,
            "tick_size": 0.10,
            "point_value": 10.0,
        }
        defaults.update(kwargs)
        return cls(**defaults)


@dataclass
class Position:
    """Represents an open position."""

    direction: int
    contracts: int
    entry_price: float
    entry_time: datetime
    entry_bar: int
    stop_loss: float | None = None
    take_profit: float | None = None
    label: int | None = None
    prediction: int | None = None
    confidence: float | None = None


@dataclass
class BacktestResult:
    """
    Results from a backtest run.

    Attributes:
        equity_curve: EquityCurve object with full history
        metrics: Performance metrics
        trades: List of completed trades
        config: Backtest configuration used
        stats: Additional statistics
    """

    equity_curve: EquityCurve
    metrics: PerformanceMetrics
    trades: list[Trade]
    config: BacktestConfig
    stats: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> dict[str, Any]:
        """Generate summary of backtest results."""
        return {
            "initial_equity": self.config.initial_equity,
            "final_equity": self.equity_curve.current_equity,
            "total_return_pct": self.equity_curve.total_return * 100,
            "total_pnl": self.equity_curve.total_pnl,
            "total_trades": self.metrics.total_trades,
            "win_rate_pct": self.metrics.win_rate * 100,
            "profit_factor": self.metrics.profit_factor,
            "sharpe_ratio": self.metrics.sharpe_ratio,
            "sortino_ratio": self.metrics.sortino_ratio,
            "calmar_ratio": self.metrics.calmar_ratio,
            "max_drawdown_pct": self.metrics.max_drawdown * 100,
            "expectancy": self.metrics.expectancy,
            "var_95": self.metrics.var_95,
            **self.stats,
        }

    def print_summary(self) -> None:
        """Print formatted summary."""
        summary = self.summary()
        print("\n" + "=" * 60)
        print("BACKTEST RESULTS")
        print("=" * 60)
        print(f"Initial Equity:     ${summary['initial_equity']:,.2f}")
        print(f"Final Equity:       ${summary['final_equity']:,.2f}")
        print(f"Total Return:       {summary['total_return_pct']:.2f}%")
        print(f"Total P&L:          ${summary['total_pnl']:,.2f}")
        print("-" * 60)
        print(f"Total Trades:       {summary['total_trades']}")
        print(f"Win Rate:           {summary['win_rate_pct']:.1f}%")
        print(f"Profit Factor:      {summary['profit_factor']:.2f}")
        print(f"Expectancy:         ${summary['expectancy']:.2f}")
        print("-" * 60)
        print(f"Sharpe Ratio:       {summary['sharpe_ratio']:.2f}")
        print(f"Sortino Ratio:      {summary['sortino_ratio']:.2f}")
        print(f"Calmar Ratio:       {summary['calmar_ratio']:.2f}")
        print(f"Max Drawdown:       {summary['max_drawdown_pct']:.2f}%")
        print(f"VaR (95%):          ${summary['var_95']:.2f}")
        print("=" * 60)


class Backtester:
    """
    Main backtester class for simulating trading strategies.

    This class takes model predictions and price data and simulates
    realistic trading with transaction costs and position sizing.

    Example:
        >>> backtester = Backtester(
        ...     predictions=predictions_df,
        ...     prices=prices_df,
        ...     config=BacktestConfig.for_mes(),
        ... )
        >>> result = backtester.run()
        >>> result.print_summary()
    """

    def __init__(
        self,
        predictions: pd.DataFrame,
        prices: pd.DataFrame,
        config: BacktestConfig | None = None,
        cost_calculator: CostCalculator | None = None,
        position_sizer: BasePositionSizer | None = None,
    ):
        """
        Initialize backtester.

        Args:
            predictions: DataFrame with columns:
                - timestamp or datetime index
                - prediction: Model prediction (-1, 0, 1)
                - probability or confidence (optional): Prediction confidence
                - label (optional): Actual label for analysis
            prices: DataFrame with columns:
                - timestamp or datetime index
                - open, high, low, close
                - volume (optional)
            config: Backtest configuration
            cost_calculator: Cost calculator (created from config if not provided)
            position_sizer: Position sizer (created from config if not provided)
        """
        self.predictions = self._validate_predictions(predictions)
        self.prices = self._validate_prices(prices)
        self.config = config or BacktestConfig()

        # Create cost calculator
        if cost_calculator is None:
            tx_costs = TransactionCosts(
                commission_per_contract=self.config.commission_per_contract,
                slippage_ticks=self.config.slippage_ticks,
                tick_value=self.config.tick_value,
                tick_size=self.config.tick_size,
            )
            self.cost_calculator = CostCalculator(transaction_costs=tx_costs)
        else:
            self.cost_calculator = cost_calculator

        # Create position sizer
        if position_sizer is None:
            self.position_sizer = create_position_sizer(
                method=self._resolve_sizing_method(self.config.position_sizing),
                risk_per_trade=self.config.risk_per_trade,
                kelly_fraction=self.config.kelly_fraction,
                target_volatility=self.config.target_volatility,
                point_value=self.config.point_value,
                contracts=self.config.fixed_contracts,
            )
        else:
            self.position_sizer = position_sizer

        # Create market hours filter
        self.market_hours_filter = MarketHoursFilter(
            contract=self.config.contract_symbol,
            enable_market_hours_filter=self.config.enable_market_hours_filter,
            enable_adverse_selection=True,
        )

        # State variables
        self._current_position: Position | None = None
        self._equity = self.config.initial_equity
        self._trades: list[Trade] = []
        self._equity_history: list[tuple[datetime, float]] = []
        self._halt_trading = False
        self._day_start_equity = self.config.initial_equity
        self._consecutive_losses = 0

    @staticmethod
    def _resolve_sizing_method(value: str) -> str:
        """Map canonical position_sizing values to local PositionSizingMethod values.

        The canonical config (src/config/inference.py) uses short names like
        "fixed", "kelly", "volatility", "confidence".  The local position sizer
        (position_sizing.py) expects "fixed_contracts", "kelly",
        "volatility_targeted", "bet_sizing", etc.  This method bridges the two.
        """
        mapping = {
            "fixed": "fixed_contracts",
            "volatility": "volatility_targeted",
            "confidence": "bet_sizing",
            # These already match:
            # "kelly" -> "kelly"
        }
        return mapping.get(value, value)

    def _validate_predictions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate and normalize predictions DataFrame."""
        df = df.copy()

        # Ensure we have required columns
        if "prediction" not in df.columns:
            if "pred" in df.columns:
                df["prediction"] = df["pred"]
            elif "signal" in df.columns:
                df["prediction"] = df["signal"]
            else:
                raise ValueError("predictions must have 'prediction' column")

        # Handle timestamp
        if "timestamp" not in df.columns:
            if isinstance(df.index, pd.DatetimeIndex):
                df["timestamp"] = df.index
            else:
                df["timestamp"] = pd.to_datetime(df.index)

        # Ensure prediction is in {-1, 0, 1}
        valid_preds = df["prediction"].isin([-1, 0, 1])
        if not valid_preds.all():
            raise ValueError("predictions must be in {-1, 0, 1}")

        # Optional columns
        if "confidence" not in df.columns and "probability" in df.columns:
            df["confidence"] = df["probability"]
        if "confidence" not in df.columns:
            df["confidence"] = 1.0

        if "label" not in df.columns:
            df["label"] = np.nan

        return df.reset_index(drop=True)

    def _validate_prices(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate and normalize prices DataFrame."""
        df = df.copy()

        # Required columns
        required = ["open", "high", "low", "close"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            # Try lowercase
            for col in missing:
                if col.upper() in df.columns:
                    df[col] = df[col.upper()]
                elif col.capitalize() in df.columns:
                    df[col] = df[col.capitalize()]

        # Re-check
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"prices must have columns: {missing}")

        # Handle timestamp
        if "timestamp" not in df.columns:
            if isinstance(df.index, pd.DatetimeIndex):
                df["timestamp"] = df.index
            else:
                df["timestamp"] = pd.to_datetime(df.index)

        return df.reset_index(drop=True)

    def _align_data(self) -> pd.DataFrame:
        """Align predictions with prices on timestamp."""
        # Merge on timestamp
        merged = pd.merge(
            self.predictions,
            self.prices,
            on="timestamp",
            how="inner",
            suffixes=("_pred", "_price"),
        )

        if len(merged) == 0:
            raise ValueError("No overlapping timestamps between predictions and prices")

        # Sort by timestamp
        merged = merged.sort_values("timestamp").reset_index(drop=True)

        return merged

    def _get_execution_price(
        self,
        bar: pd.Series,
        direction: int,
        is_entry: bool,
    ) -> float:
        """
        Get execution price based on execution model.

        Args:
            bar: Price bar (open, high, low, close)
            direction: 1 for buy, -1 for sell
            is_entry: True if this is entry, False if exit

        Returns:
            Execution price
        """
        model = self.config.execution_model

        if model == ExecutionModel.MARKET_ON_CLOSE:
            return float(bar["close"])
        elif model == ExecutionModel.MARKET_ON_OPEN:
            return float(bar["open"])
        elif model == ExecutionModel.FILL_AT_SIGNAL:
            return float(bar["close"])
        elif model == ExecutionModel.VWAP:
            # Approximate VWAP as midpoint
            return float((bar["high"] + bar["low"]) / 2)
        else:
            return float(bar["close"])

    def _calculate_position_size(
        self,
        current_price: float,
        stop_distance: float | None = None,
        volatility: float | None = None,
        win_rate: float = 0.5,
        avg_win: float = 100.0,
        avg_loss: float = 100.0,
        confidence: float | None = None,
    ) -> int:
        """Calculate position size using configured method."""
        # Provide reasonable defaults for position sizing
        if stop_distance is None:
            stop_distance = current_price * 0.02  # 2% default stop

        # Phase 15E: Pass confidence/probability to position sizer for bet sizing
        return self.position_sizer.calculate_position_size(
            account_equity=self._equity,
            current_price=current_price,
            stop_distance=stop_distance,
            current_volatility=volatility or 0.15,
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            probability=confidence or 0.5,  # BetSizingPositioner uses probability parameter
        )

    def _open_position(
        self,
        direction: int,
        price: float,
        timestamp: datetime,
        bar_idx: int,
        label: int | None = None,
        prediction: int | None = None,
        confidence: float | None = None,
        atr: float | None = None,
    ) -> None:
        """Open a new position."""
        # Phase 15E: Pass confidence for bet sizing integration
        contracts = self._calculate_position_size(price, confidence=confidence)

        if contracts <= 0:
            return

        # Calculate stop loss (2% default or 2 ATR if available)
        stop_loss = None
        if atr is not None and atr > 0:
            stop_distance_atr = 2.0
            stop_loss = price - direction * stop_distance_atr * atr
        else:
            # Default 2% stop
            stop_distance_pct = 0.02
            stop_loss = price * (1 - direction * stop_distance_pct)

        self._current_position = Position(
            direction=direction,
            contracts=contracts,
            entry_price=price,
            entry_time=timestamp,
            entry_bar=bar_idx,
            stop_loss=stop_loss,
            label=label,
            prediction=prediction,
            confidence=confidence,
        )

    def _close_position(
        self,
        price: float,
        timestamp: datetime,
    ) -> Trade | None:
        """Close current position and record trade."""
        if self._current_position is None:
            return None

        pos = self._current_position

        # Calculate P&L
        pnl_result = self.cost_calculator.calculate_pnl(
            contracts=pos.contracts,
            entry_price=pos.entry_price,
            exit_price=price,
            direction=pos.direction,
            point_value=self.config.point_value,
            include_costs=True,
        )

        # Create trade record
        trade = Trade(
            entry_time=pos.entry_time,
            exit_time=timestamp,
            direction=pos.direction,
            contracts=pos.contracts,
            entry_price=pos.entry_price,
            exit_price=price,
            gross_pnl=pnl_result["gross_pnl"],
            costs=pnl_result["costs"],
            net_pnl=pnl_result["net_pnl"],
            return_pct=pnl_result["return_pct"],
            label=pos.label,
            prediction=pos.prediction,
            confidence=pos.confidence,
            stop_loss_price=pos.stop_loss,
        )

        # Calculate R-multiple
        trade.calculate_r_multiple(point_value=self.config.point_value)

        # Update equity
        self._equity += trade.net_pnl
        self._trades.append(trade)

        # Track consecutive losses for circuit breaker
        if trade.net_pnl <= 0:
            self._consecutive_losses += 1
        else:
            self._consecutive_losses = 0

        self._current_position = None

        return trade

    def _should_exit(
        self,
        position: Position,
        current_bar: pd.Series,
        current_bar_idx: int,
        new_signal: int,
    ) -> bool:
        """Determine if position should be closed."""
        # Check holding period constraints
        bars_held = current_bar_idx - position.entry_bar

        # Minimum holding period
        if bars_held < self.config.min_holding_period:
            return False

        # Maximum holding period
        if self.config.max_holding_period > 0 and bars_held >= self.config.max_holding_period:
            return True

        # Signal reversal (opposite direction or neutral)
        if new_signal == -position.direction:
            return True

        # Exit on neutral if we require direction
        if new_signal == 0:
            return True

        # Stop loss (if set)
        if position.stop_loss is not None:
            if position.direction == 1 and current_bar["low"] <= position.stop_loss:
                return True
            if position.direction == -1 and current_bar["high"] >= position.stop_loss:
                return True

        # Take profit (if set)
        if position.take_profit is not None:
            if position.direction == 1 and current_bar["high"] >= position.take_profit:
                return True
            if position.direction == -1 and current_bar["low"] <= position.take_profit:
                return True

        return False

    def run(self) -> BacktestResult:
        """
        Run the backtest simulation.

        Returns:
            BacktestResult with equity curve, metrics, and trades
        """
        # Align data
        data = self._align_data()

        # Initialize state
        self._equity = self.config.initial_equity
        self._trades = []
        self._equity_history = [(data.iloc[0]["timestamp"], self._equity)]
        self._current_position = None

        # Statistics
        signals_count = {"long": 0, "short": 0, "neutral": 0}

        # Iterate through bars
        for i in range(len(data)):
            bar = data.iloc[i]
            timestamp = bar["timestamp"]
            prediction = int(bar["prediction"])
            confidence = bar.get("confidence", 1.0)
            label = bar.get("label")

            # Count signals
            if prediction == 1:
                signals_count["long"] += 1
            elif prediction == -1:
                signals_count["short"] += 1
            else:
                signals_count["neutral"] += 1

            # Get execution prices
            close_price = self._get_execution_price(bar, 1, is_entry=False)

            # Check if we have a position and should exit
            if self._current_position is not None and self._should_exit(
                self._current_position, bar, i, prediction
            ):
                self._close_position(close_price, timestamp)

            # If no position, check for entry
            if self._current_position is None:
                # Check if tradeable time (market hours filter)
                if not self.market_hours_filter.is_tradeable_time(timestamp):
                    continue

                if prediction == 1:
                    # Long signal
                    entry_price = self._get_execution_price(bar, 1, is_entry=True)
                    self._open_position(
                        direction=1,
                        price=entry_price,
                        timestamp=timestamp,
                        bar_idx=i,
                        label=label if not pd.isna(label) else None,
                        prediction=prediction,
                        confidence=confidence,
                    )
                elif prediction == -1 and self.config.allow_short:
                    # Short signal
                    entry_price = self._get_execution_price(bar, -1, is_entry=True)
                    self._open_position(
                        direction=-1,
                        price=entry_price,
                        timestamp=timestamp,
                        bar_idx=i,
                        label=int(label) if not pd.isna(label) else None,
                        prediction=prediction,
                        confidence=confidence,
                    )

            # Record equity
            # Include unrealized P&L from open position
            current_equity = self._equity
            if self._current_position is not None:
                unrealized = self._calculate_unrealized_pnl(self._current_position, close_price)
                current_equity += unrealized

            self._equity_history.append((timestamp, current_equity))

            # Circuit Breaker: Check max drawdown.
            # Note: current_equity includes unrealized P&L (mark-to-market).
            # This is intentional — circuit breakers protect against adverse
            # moves in open positions, not just realized losses.
            if len(self._equity_history) > 1:
                equity_values = [e for _, e in self._equity_history]
                running_max = max(equity_values)
                current_drawdown = (
                    (current_equity - running_max) / running_max if running_max > 0 else 0
                )

                if abs(current_drawdown) > self.config.max_drawdown_threshold:
                    import logging

                    logger = logging.getLogger(__name__)
                    logger.critical(
                        f"CIRCUIT BREAKER TRIGGERED: "
                        f"Drawdown {current_drawdown:.2%} exceeds threshold {self.config.max_drawdown_threshold:.2%}"
                    )
                    self._halt_trading = True
                    if self._current_position is not None:
                        self._close_position(close_price, timestamp)
                    break

            # Circuit Breaker: Check daily loss limit
            if i > 0:
                prev_timestamp = data.iloc[i - 1]["timestamp"]
                if timestamp.date() != prev_timestamp.date():
                    # New day started
                    daily_return = (
                        (self._equity - self._day_start_equity) / self._day_start_equity
                        if self._day_start_equity > 0
                        else 0
                    )
                    if daily_return < -self.config.daily_loss_threshold:
                        import logging

                        logger = logging.getLogger(__name__)
                        logger.critical(f"DAILY LOSS LIMIT: {daily_return:.2%}")
                        self._halt_trading = True
                        if self._current_position is not None:
                            self._close_position(close_price, timestamp)
                        break
                    self._day_start_equity = self._equity

            # Circuit Breaker: Check consecutive losses
            if self._consecutive_losses >= self.config.consecutive_loss_limit:
                import logging

                logger = logging.getLogger(__name__)
                logger.critical(
                    f"CONSECUTIVE LOSS LIMIT: {self._consecutive_losses} losses in a row"
                )
                self._halt_trading = True
                if self._current_position is not None:
                    self._close_position(close_price, timestamp)
                break

            # Skip trading if halted
            if self._halt_trading:
                continue

        # Close any remaining position at the end
        if self._current_position is not None:
            final_bar = data.iloc[-1]
            final_price = final_bar["close"]
            self._close_position(final_price, final_bar["timestamp"])

        # Build equity curve
        equity_curve = EquityCurve(
            initial_equity=self.config.initial_equity,
            timestamps=[t for t, _ in self._equity_history],
            equity_values=[e for _, e in self._equity_history],
            trades=self._trades,
        )

        # Calculate metrics
        metrics = equity_curve.get_metrics()

        # Additional statistics
        stats = {
            "total_bars": len(data),
            "signals": signals_count,
            "position_rate": (signals_count["long"] + signals_count["short"]) / len(data),
            "long_trades": sum(1 for t in self._trades if t.direction == 1),
            "short_trades": sum(1 for t in self._trades if t.direction == -1),
        }

        return BacktestResult(
            equity_curve=equity_curve,
            metrics=metrics,
            trades=self._trades,
            config=self.config,
            stats=stats,
        )

    def _calculate_unrealized_pnl(self, position: Position, current_price: float) -> float:
        """Calculate unrealized P&L for open position.

        Deducts estimated entry costs (commission + slippage) so that
        unrealized equity is conservative rather than overstated.
        """
        price_change = current_price - position.entry_price
        gross = position.direction * position.contracts * price_change * self.config.point_value
        # Deduct estimated entry costs (commission + slippage per contract)
        entry_cost = (
            self.config.commission_per_contract
            + self.config.slippage_ticks * self.config.tick_value
        ) * position.contracts
        return gross - entry_cost


def run_backtest(
    predictions: pd.DataFrame,
    prices: pd.DataFrame,
    config: BacktestConfig | None = None,
    **kwargs: Any,
) -> BacktestResult:
    """
    Convenience function to run a backtest.

    Args:
        predictions: Predictions DataFrame
        prices: Prices DataFrame
        config: Backtest configuration
        **kwargs: Additional config parameters

    Returns:
        BacktestResult
    """
    if config is None:
        config = BacktestConfig(**kwargs)

    backtester = Backtester(predictions, prices, config)
    return backtester.run()


def run_walk_forward_backtest(
    predictions: pd.DataFrame,
    prices: pd.DataFrame,
    n_splits: int = 5,
    config: BacktestConfig | None = None,
) -> list[BacktestResult]:
    """
    Run walk-forward backtest with multiple out-of-sample periods.

    Args:
        predictions: Full predictions DataFrame
        prices: Full prices DataFrame
        n_splits: Number of walk-forward splits
        config: Backtest configuration

    Returns:
        List of BacktestResult for each split
    """
    results = []
    n_samples = len(predictions)
    split_size = n_samples // n_splits

    for i in range(n_splits):
        start_idx = i * split_size
        end_idx = (i + 1) * split_size if i < n_splits - 1 else n_samples

        split_predictions = predictions.iloc[start_idx:end_idx].copy()
        split_prices = prices.iloc[start_idx:end_idx].copy()

        result = run_backtest(split_predictions, split_prices, config)
        results.append(result)

    return results


__all__ = [
    "ExecutionModel",
    "BacktestConfig",
    "Position",
    "BacktestResult",
    "Backtester",
    "run_backtest",
    "run_walk_forward_backtest",
]
