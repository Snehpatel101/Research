"""
Main backtester implementation for strategy simulation.

This module provides the Backtester class that simulates trading
with realistic execution, transaction costs, and position sizing.

Timing contract (no same-bar lookahead)
---------------------------------------
A prediction at row ``i`` is computed from features that include bar ``i``'s
close, and its triple-barrier label is anchored at ``close[i]``. The signal is
therefore only *known* once bar ``i`` has closed, and it is acted on at bar
``j = i + signal_delay_bars`` at the execution model's fill price:

========================  ==============================  =========  ==================
Execution model           Fill price at bar ``j``         Min delay  Barriers watched
========================  ==============================  =========  ==================
MARKET_ON_OPEN (default)  ``open[j]``                     1          from bar ``j``
MIDPOINT                  ``(high[j] + low[j]) / 2``      1          from bar ``j + 1``
MARKET_ON_CLOSE           ``close[j]``                    0          from bar ``j + 1``
FILL_AT_SIGNAL            ``close[j]``                    0          from bar ``j + 1``
========================  ==============================  =========  ==================

- Close fills with delay 0 assume the decision and the fill both happen at the
  signal bar's close (no compute or order-routing latency) — the optimistic
  limit of a realistic simulation.
- An open fill happens at the start of bar ``j``, so bar ``j``'s high/low all
  trade after the fill and are watched for the stop / take-profit. A midpoint
  fill happens somewhere inside bar ``j``, so barriers are watched from bar
  ``j + 1``.
- Label vs backtest reference price: the label measures its barriers from
  ``close[i]``; the backtest measures the SAME barrier distances
  ``(k + cost_in_atr) * ATR[i]`` from its actual fill (``open[i + 1]`` by
  default). The two differ by the close-to-next-open gap (overnight / session
  gaps, and adverse selection) — that gap is real P&L the label never sees.
- Signal exits (reversal / neutral) and circuit-breaker flattening use the
  same delay and fill price as entries. Time exits (max holding period, session
  end, end of data) are scheduled in advance and fill at ``close[j]``; the max
  holding period counts from the signal bar, so the exit lands on the label's
  time barrier ``close[i + max_holding_period]``.
- Equity is marked to market at ``close[j]`` every bar.

Circuit breakers pause trading instead of ending the simulation: daily loss
and consecutive-loss halts resume on the next trading day, the drawdown halt
resumes after ``drawdown_cooloff_bars`` (or the next trading day) unless
``drawdown_halt_permanent``. Positions are flattened when a halt trips, the
equity curve stays flat while halted, and every halt is recorded in
``BacktestResult.halts``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from src.config.symbol import SymbolConfig

from .costs import CostCalculator, TransactionCosts
from .equity_curve import EquityCurve, Trade
from .execution import CONTRACT_SESSION_TIMES, DEFAULT_SESSION, MarketHoursFilter, to_eastern
from .metrics import PerformanceMetrics
from .position_sizing import BasePositionSizer, create_position_sizer

logger = logging.getLogger(__name__)


class ExecutionModel(StrEnum):
    """Order execution model types (see the module docstring for timing)."""

    MARKET_ON_CLOSE = "market_on_close"
    MARKET_ON_OPEN = "market_on_open"
    FILL_AT_SIGNAL = "fill_at_signal"
    MIDPOINT = "midpoint"
    # Backward compat alias — "vwap" was actually (H+L)/2 midpoint
    VWAP = "midpoint"


class FillTiming(StrEnum):
    """Where inside the action bar an execution model fills."""

    OPEN = "open"  # first print of the bar
    INTRABAR = "intrabar"  # somewhere inside the bar
    CLOSE = "close"  # last print of the bar


_FILL_TIMING: dict[ExecutionModel, FillTiming] = {
    ExecutionModel.MARKET_ON_OPEN: FillTiming.OPEN,
    ExecutionModel.MIDPOINT: FillTiming.INTRABAR,
    ExecutionModel.MARKET_ON_CLOSE: FillTiming.CLOSE,
    ExecutionModel.FILL_AT_SIGNAL: FillTiming.CLOSE,
}

# Smallest signal delay (bars) that keeps a fill at or after the signal bar's
# close: a close fill may happen on the signal bar itself, an open or
# intrabar fill of that bar would precede the close that produced the signal.
_MIN_SIGNAL_DELAY: dict[FillTiming, int] = {
    FillTiming.OPEN: 1,
    FillTiming.INTRABAR: 1,
    FillTiming.CLOSE: 0,
}


class ExitReason(StrEnum):
    """Why a position exit was triggered."""

    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"
    MAX_HOLDING = "max_holding"
    SIGNAL_REVERSAL = "signal_reversal"
    SIGNAL_NEUTRAL = "signal_neutral"
    SESSION_END = "session_end"
    CIRCUIT_BREAKER = "circuit_breaker"
    END_OF_DATA = "end_of_data"


class HaltReason(StrEnum):
    """Which circuit breaker paused trading."""

    MAX_DRAWDOWN = "max_drawdown"
    DAILY_LOSS = "daily_loss"
    CONSECUTIVE_LOSSES = "consecutive_losses"


@dataclass
class BacktestConfig:
    """
    Configuration for backtester.

    Attributes:
        initial_equity: Starting portfolio value
        position_sizing: Method for calculating position size
        execution_model: How orders are executed (fill price inside the bar)
        signal_delay_bars: Bars between the signal bar and the fill bar.
            None = the execution model's minimum (1 for open/midpoint fills,
            0 for close fills). Values below that minimum would fill before
            the signal exists and are rejected.
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
        min_holding_period: Minimum bars before a SIGNAL-driven exit (stops,
            take-profits, time and forced exits are always honored)
        max_holding_period: Maximum bars to hold, counted from the signal bar
            like the label's time barrier (0 = no limit)
        max_drawdown_threshold: Mark-to-market drawdown that pauses trading
        drawdown_cooloff_bars: Bars the drawdown halt lasts (0 = until the next
            trading day). The drawdown peak is re-baselined on resume.
        drawdown_halt_permanent: Drawdown halt never resumes (the equity curve
            still covers the whole period, flat after the halt)
        daily_loss_threshold: Intraday loss (vs the day's opening equity) that
            pauses trading until the next trading day
        consecutive_loss_limit: Losing trades in a row that pause trading until
            the next trading day (the streak resets on resume)
        barrier_k_up / barrier_k_down: Triple-barrier ATR multipliers from the
            training config (0.0 = legacy 2% stop, no take-profit)
        barrier_cost_in_atr: Cost term added to both multipliers, exactly as
            the labeler does. None = derive it with the labeler's helper
            (round-trip cost in price units / median ATR of the price data);
            0.0 = plain k * ATR barriers.
    """

    initial_equity: float = 100000.0
    position_sizing: str = "fixed_contracts"
    execution_model: ExecutionModel = ExecutionModel.MARKET_ON_OPEN
    signal_delay_bars: int | None = None
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
    drawdown_cooloff_bars: int = 0
    drawdown_halt_permanent: bool = False
    daily_loss_threshold: float = 0.02
    consecutive_loss_limit: int = 5
    enable_market_hours_filter: bool = True
    contract_symbol: str = "MES"

    # Alignment data loss warning threshold (percentage)
    alignment_loss_warn_pct: float = 5.0

    # Triple-barrier alignment (from training config)
    # 0.0 = not set, uses legacy hardcoded logic for backward compat
    barrier_k_up: float = 0.0  # Upper barrier ATR multiplier
    barrier_k_down: float = 0.0  # Lower barrier ATR multiplier
    barrier_cost_in_atr: float | None = None  # None = derive like the labeler

    # Session-end forced close: close all positions at session end
    # False = legacy behavior (positions can be held overnight)
    force_session_close: bool = False

    # Minimum completed trades before Kelly stats become active
    kelly_min_trades: int = 30

    def __post_init__(self) -> None:
        """Normalize the execution model and reject lookahead fills."""
        self.execution_model = ExecutionModel(self.execution_model)
        if self.signal_delay_bars is not None and self.signal_delay_bars < self.min_signal_delay:
            raise ValueError(
                f"signal_delay_bars={self.signal_delay_bars} would fill "
                f"{self.execution_model.value} orders before the signal bar's close "
                f"(minimum {self.min_signal_delay} for this execution model)"
            )
        if self.drawdown_cooloff_bars < 0:
            raise ValueError(
                f"drawdown_cooloff_bars must be >= 0, got {self.drawdown_cooloff_bars}"
            )

    @property
    def fill_timing(self) -> FillTiming:
        """Where inside the action bar orders fill."""
        return _FILL_TIMING[self.execution_model]

    @property
    def min_signal_delay(self) -> int:
        """Smallest lookahead-free delay for the execution model."""
        return _MIN_SIGNAL_DELAY[self.fill_timing]

    @property
    def resolved_signal_delay(self) -> int:
        """Signal delay actually used (explicit value or the model minimum)."""
        if self.signal_delay_bars is None:
            return self.min_signal_delay
        return self.signal_delay_bars

    @property
    def uses_barriers(self) -> bool:
        """Whether triple-barrier stop / take-profit levels are configured."""
        return self.barrier_k_up > 0 or self.barrier_k_down > 0

    @classmethod
    def from_symbol_config(cls, sym: SymbolConfig, **kwargs: Any) -> BacktestConfig:
        """Create BacktestConfig from a SymbolConfig.

        Args:
            sym: SymbolConfig with contract specifications
            **kwargs: Additional BacktestConfig overrides

        Returns:
            BacktestConfig populated with the symbol's tick/point values
        """
        defaults: dict[str, Any] = {
            "tick_value": sym.tick_value,
            "tick_size": sym.tick_size,
            "point_value": sym.point_value,
            "contract_symbol": sym.symbol,
        }
        defaults.update(kwargs)
        return cls(**defaults)

    @classmethod
    def for_mes(cls, **kwargs: Any) -> BacktestConfig:
        """Create config for Micro E-mini S&P 500."""
        from src.config.symbol import SymbolConfig

        return cls.from_symbol_config(SymbolConfig.for_mes(), **kwargs)

    @classmethod
    def for_mgc(cls, **kwargs: Any) -> BacktestConfig:
        """Create config for Micro Gold."""
        from src.config.symbol import SymbolConfig

        return cls.from_symbol_config(SymbolConfig.for_mgc(), **kwargs)


@dataclass
class Position:
    """Represents an open position.

    ``entry_bar`` is the fill bar; ``signal_bar`` the bar whose prediction
    opened it (defaults to ``entry_bar``); barriers are checked from
    ``monitor_from_bar`` (defaults to ``entry_bar + 1``).
    """

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
    entry_atr: float | None = None
    signal_bar: int | None = None
    monitor_from_bar: int | None = None

    def __post_init__(self) -> None:
        if self.signal_bar is None:
            self.signal_bar = self.entry_bar
        if self.monitor_from_bar is None:
            self.monitor_from_bar = self.entry_bar + 1


@dataclass
class HaltEvent:
    """One circuit-breaker pause.

    Attributes:
        bar: Bar index whose close tripped the breaker
        timestamp: Timestamp of that bar
        reason: Which breaker tripped
        value: The tripping measurement (drawdown / daily return as a
            fraction, or the losing-streak length)
        resume_bar: First bar trading was allowed again (None = never)
        resume_time: Timestamp of ``resume_bar``
    """

    bar: int
    timestamp: datetime
    reason: HaltReason
    value: float
    resume_bar: int | None = None
    resume_time: datetime | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serializable form."""
        return {
            "bar": self.bar,
            "timestamp": str(self.timestamp),
            "reason": self.reason.value,
            "value": self.value,
            "resume_bar": self.resume_bar,
            "resume_time": None if self.resume_time is None else str(self.resume_time),
        }


@dataclass
class BacktestResult:
    """
    Results from a backtest run.

    Attributes:
        equity_curve: EquityCurve object with full history (one point per bar)
        metrics: Performance metrics
        trades: List of completed trades
        config: Backtest configuration used
        stats: Additional statistics (incl. n_halts / halted_at)
        halts: Circuit-breaker pauses, in order
    """

    equity_curve: EquityCurve
    metrics: PerformanceMetrics
    trades: list[Trade]
    config: BacktestConfig
    stats: dict[str, Any] = field(default_factory=dict)
    halts: list[HaltEvent] = field(default_factory=list)

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
        print(f"Circuit Halts:      {summary.get('n_halts', 0)}")
        print("=" * 60)


@dataclass
class _Bars:
    """Aligned per-bar arrays the simulation loop reads."""

    timestamps: list[pd.Timestamp]  # boxed once; indexing a DatetimeIndex per bar is slow
    predictions: np.ndarray
    confidences: np.ndarray
    labels: np.ndarray
    opens: np.ndarray
    highs: np.ndarray
    lows: np.ndarray
    closes: np.ndarray
    atr: np.ndarray
    fills: np.ndarray
    can_enter: np.ndarray
    session_end: np.ndarray
    day_ids: np.ndarray


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
                - prediction: Model prediction (-1, 0, 1), computed from data
                  up to and including that bar's close
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

        # Barrier cost term, resolved once so labels and backtest agree
        self._barrier_cost_in_atr = self._resolve_barrier_cost_in_atr()

        # State variables (reset at the start of every run())
        self._current_position: Position | None = None
        self._equity = self.config.initial_equity
        self._trades: list[Trade] = []
        self._equity_history: list[tuple[datetime, float]] = []
        self._halt_trading = False
        self._halts: list[HaltEvent] = []
        self._active_halt: HaltEvent | None = None
        self._halt_resume_bar: int | None = None
        self._halt_until_next_day = False
        self._flatten_at_bar: int | None = None
        self._day_start_equity = self.config.initial_equity
        self._consecutive_losses = 0

        # Running Kelly statistics from completed trades
        self._kelly_win_rate: float = 0.5
        self._kelly_avg_win: float = 100.0
        self._kelly_avg_loss: float = 100.0
        self._kelly_active: bool = False
        self._n_wins = 0
        self._n_losses = 0
        self._sum_wins = 0.0
        self._sum_losses = 0.0

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

    def _resolve_barrier_cost_in_atr(self) -> float:
        """Cost term of the barrier multipliers, computed like the labeler's.

        An explicit ``barrier_cost_in_atr`` (the factory passes the labeling
        run's value) wins. Otherwise the labeler's own helper converts the
        symbol's round-trip cost (price units) into ATR units with the median
        ATR of the price data — the same global calibration the labeler
        applies to its dataset.
        """
        if not self.config.uses_barriers:
            return 0.0
        if self.config.barrier_cost_in_atr is not None:
            return float(self.config.barrier_cost_in_atr)
        from src.data.labeling.triple_barrier import compute_cost_in_atr

        prices = self.prices.sort_values("timestamp")
        atr = self._compute_atr(prices).to_numpy(dtype=float)
        return compute_cost_in_atr(self.config.contract_symbol, atr)

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
        rows_before = len(self.predictions)

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

        # Log alignment data loss
        rows_after = len(merged)
        rows_dropped = rows_before - rows_after
        if rows_dropped > 0:
            drop_pct = rows_dropped / rows_before * 100
            logger.info(f"Alignment dropped {rows_dropped} rows ({drop_pct:.1f}%)")
            if drop_pct > self.config.alignment_loss_warn_pct:
                logger.warning(
                    f"Significant data loss during alignment: {rows_dropped}/{rows_before} rows "
                    f"({drop_pct:.1f}%) dropped. Check timestamp consistency between predictions and market data."
                )

        return merged

    def _calculate_position_size(
        self,
        current_price: float,
        stop_distance: float | None = None,
        volatility: float | None = None,
        confidence: float | None = None,
    ) -> int:
        """Calculate position size using configured method.

        When Kelly sizing is active and enough trades have been completed
        (kelly_min_trades), running win_rate/avg_win/avg_loss from actual
        trades are passed to the sizer instead of defaults.
        """
        # Provide reasonable defaults for position sizing
        if stop_distance is None:
            stop_distance = current_price * 0.02  # 2% default stop

        # Use running Kelly stats if active, otherwise defaults
        win_rate = self._kelly_win_rate if self._kelly_active else 0.5
        avg_win = self._kelly_avg_win if self._kelly_active else 100.0
        avg_loss = self._kelly_avg_loss if self._kelly_active else 100.0

        return self.position_sizer.calculate_position_size(
            account_equity=self._equity,
            current_price=current_price,
            stop_distance=stop_distance,
            current_volatility=volatility or 0.15,
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            probability=confidence if confidence is not None else 0.5,
        )

    def _compute_atr(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Compute ATR(period) from aligned price data.

        Args:
            data: Merged DataFrame with high, low, close columns
            period: ATR lookback period (default 14)

        Returns:
            Series of ATR values aligned with data index
        """
        high = data["high"].astype(float)
        low = data["low"].astype(float)
        close = data["close"].astype(float)

        prev_close = close.shift(1)
        tr = pd.concat(
            [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
            axis=1,
        ).max(axis=1)
        # Wilder's EMA (alpha = 1/period) — matches labeling ATR
        return tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

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
        signal_bar: int | None = None,
        monitor_from_bar: int | None = None,
    ) -> None:
        """Open a new position filled at ``price`` on bar ``bar_idx``.

        Barrier distances come from ``barrier_distances`` — the labeler's
        helper — so the stop / take-profit sit ``(k + cost_in_atr) * ATR``
        from entry, exactly like the label's barriers.
        """
        from src.data.labeling.triple_barrier import barrier_distances

        k_up = self.config.barrier_k_up
        k_down = self.config.barrier_k_down
        use_barriers = self.config.uses_barriers and atr is not None and atr > 0

        # Compute stop distance for position sizing
        up_dist = down_dist = 0.0
        if use_barriers:
            assert atr is not None
            up_dist, down_dist = barrier_distances(atr, k_up, k_down, self._barrier_cost_in_atr)
            # Barrier-aware stop distance (matches training semantics)
            stop_dist = down_dist if direction == 1 else up_dist
        else:
            # Legacy: 2% of price
            stop_dist = price * 0.02

        contracts = self._calculate_position_size(
            price, stop_distance=stop_dist, confidence=confidence
        )

        if contracts <= 0:
            return

        # Calculate stop loss and take profit
        if use_barriers:
            # Barrier-aligned: matches training triple-barrier semantics
            if direction == 1:  # Long
                stop_loss = price - down_dist
                take_profit = price + up_dist if k_up > 0 else None
            else:  # Short
                stop_loss = price + up_dist
                take_profit = price - down_dist if k_down > 0 else None
        else:
            # Legacy: 2% stop, no take profit
            stop_loss = price * (1 - direction * 0.02)
            take_profit = None

        self._current_position = Position(
            direction=direction,
            contracts=contracts,
            entry_price=price,
            entry_time=timestamp,
            entry_bar=bar_idx,
            stop_loss=stop_loss,
            take_profit=take_profit,
            label=label,
            prediction=prediction,
            confidence=confidence,
            entry_atr=atr,
            signal_bar=signal_bar,
            monitor_from_bar=monitor_from_bar,
        )

    def _close_position(
        self,
        price: float,
        timestamp: datetime,
        current_atr: float | None = None,
        reason: ExitReason | None = None,
    ) -> Trade | None:
        """Close current position and record trade.

        Args:
            price: Exit price
            timestamp: Exit timestamp
            current_atr: Current ATR value for volatility-scaled slippage
            reason: Why the position is closed (recorded on the trade)
        """
        if self._current_position is None:
            return None

        pos = self._current_position

        # Compute volatility proxies (ATR / price) for slippage scaling
        entry_vol = None
        if pos.entry_atr is not None and pos.entry_price > 0:
            entry_vol = pos.entry_atr / pos.entry_price
        exit_vol = None
        if current_atr is not None and price > 0:
            exit_vol = current_atr / price

        # Calculate P&L
        pnl_result = self.cost_calculator.calculate_pnl(
            contracts=pos.contracts,
            entry_price=pos.entry_price,
            exit_price=price,
            direction=pos.direction,
            point_value=self.config.point_value,
            include_costs=True,
            entry_volatility=entry_vol,
            exit_volatility=exit_vol,
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
            exit_reason=None if reason is None else reason.value,
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

        # Update running Kelly statistics from completed trades
        self._update_kelly_stats()

        self._current_position = None

        return trade

    def _update_kelly_stats(self) -> None:
        """Fold the latest trade into running Kelly stats (O(1) per trade).

        Once kelly_min_trades are completed, win_rate/avg_win/avg_loss
        are derived from actual trade history and fed into position sizing.
        """
        pnl = self._trades[-1].net_pnl
        if pnl > 0:
            self._n_wins += 1
            self._sum_wins += pnl
        else:
            self._n_losses += 1
            self._sum_losses += pnl

        n = self._n_wins + self._n_losses
        if n < self.config.kelly_min_trades:
            return

        self._kelly_win_rate = self._n_wins / n
        self._kelly_avg_win = self._sum_wins / self._n_wins if self._n_wins else 0.0
        self._kelly_avg_loss = abs(self._sum_losses / self._n_losses) if self._n_losses else 1.0
        self._kelly_active = True

    def _session_end_mask(self, timestamps: pd.DatetimeIndex) -> np.ndarray:
        """Flag the last bar of each trading session (vectorized).

        A bar is session-end if it is at/after the contract's session end
        (ET), if the next bar falls on a different ET date or crosses the
        session end, or if it is the last bar of the data.
        """
        n = len(timestamps)
        if n == 0:
            return np.zeros(0, dtype=bool)
        session_end = CONTRACT_SESSION_TIMES.get(
            self.config.contract_symbol.upper(), DEFAULT_SESSION
        )[1]
        et = to_eastern(timestamps)
        seconds = np.asarray(et.hour * 3600 + et.minute * 60 + et.second)
        end_seconds = session_end.hour * 3600 + session_end.minute * 60
        at_or_after_end = seconds >= end_seconds

        dates = np.asarray(et.normalize().tz_localize(None))
        mask = at_or_after_end.copy()
        mask[:-1] |= dates[1:] != dates[:-1]
        mask[:-1] |= at_or_after_end[1:] & ~at_or_after_end[:-1]
        mask[-1] = True
        return mask

    @staticmethod
    def _trading_day_ids(timestamps: pd.DatetimeIndex) -> np.ndarray:
        """Integer id of each bar's ET calendar date (daily breaker scope)."""
        if len(timestamps) == 0:
            return np.zeros(0, dtype=np.int64)
        days = np.asarray(to_eastern(timestamps).normalize().tz_localize(None))
        return np.concatenate([[0], np.cumsum(days[1:] != days[:-1])])

    @staticmethod
    def _barrier_exit_reason(position: Position, high: float, low: float) -> ExitReason | None:
        """Stop / take-profit touched by a bar's range.

        When both trigger on the same bar, stop_loss wins (conservative).
        """
        stop_hit = False
        tp_hit = False

        if position.stop_loss is not None:
            if position.direction == 1 and low <= position.stop_loss:
                stop_hit = True
            if position.direction == -1 and high >= position.stop_loss:
                stop_hit = True

        if position.take_profit is not None:
            if position.direction == 1 and high >= position.take_profit:
                tp_hit = True
            if position.direction == -1 and low <= position.take_profit:
                tp_hit = True

        if stop_hit:
            return ExitReason.STOP_LOSS
        if tp_hit:
            return ExitReason.TAKE_PROFIT
        return None

    def _resolve_exit_price(
        self,
        reason: ExitReason,
        position: Position,
        close_price: float,
        bar_open: float | None = None,
    ) -> float:
        """Determine exit price based on exit reason.

        For barrier exits (stop_loss, take_profit), use the barrier price
        so P&L reflects the actual trigger level rather than the close.
        Stop-loss exits include additional slippage (stops slip in fast
        markets); when the bar opened through the stop (``bar_open`` beyond
        it) the stop fills at the open, not at the stop level. Take-profit
        fills never get gap price improvement (conservative).
        For signal-based and max-holding exits, use close price.

        Falls back to close_price when barrier levels are not set
        (legacy mode with barrier_k_up=0.0).
        """
        if reason == ExitReason.STOP_LOSS and position.stop_loss is not None:
            stop_price = position.stop_loss
            if bar_open is not None:
                if position.direction == 1:
                    stop_price = min(stop_price, bar_open)
                else:
                    stop_price = max(stop_price, bar_open)
            # Stop exits slip: price is worse than barrier by slippage amount
            slippage = self.config.slippage_ticks * self.config.tick_size
            if position.direction == 1:
                # Long stop: fill below the stop level
                stop_price -= slippage
            else:
                # Short stop: fill above the stop level
                stop_price += slippage
            return stop_price
        if reason == ExitReason.TAKE_PROFIT and position.take_profit is not None:
            return position.take_profit
        return close_price

    def _prepare_bars(self, data: pd.DataFrame) -> _Bars:
        """Extract aligned numpy arrays and per-bar masks for the loop."""
        timing = self.config.fill_timing
        timestamps = pd.DatetimeIndex(data["timestamp"])
        n = len(data)
        opens = data["open"].to_numpy(dtype=float)
        highs = data["high"].to_numpy(dtype=float)
        lows = data["low"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)

        if timing == FillTiming.OPEN:
            fills = opens
        elif timing == FillTiming.INTRABAR:
            # (H+L)/2 midpoint — previously labeled VWAP, renamed for accuracy
            fills = (highs + lows) / 2
        else:
            fills = closes

        session_end = (
            self._session_end_mask(timestamps)
            if self.config.force_session_close
            else np.zeros(n, dtype=bool)
        )
        can_enter = self.market_hours_filter.tradeable_mask(timestamps)
        if timing == FillTiming.CLOSE and n > 0:
            # A close fill on a bar that is force-closed at that same close
            # (session end, end of data) would be an empty round trip.
            can_enter = can_enter & ~session_end
            can_enter[-1] = False

        return _Bars(
            timestamps=list(timestamps),
            predictions=data["prediction"].to_numpy().astype(int),
            confidences=data["confidence"].to_numpy(dtype=float),
            labels=data["label"].to_numpy(),
            opens=opens,
            highs=highs,
            lows=lows,
            closes=closes,
            atr=self._compute_atr(data).to_numpy(dtype=float),
            fills=fills,
            can_enter=can_enter,
            session_end=session_end,
            day_ids=self._trading_day_ids(timestamps),
        )

    def _reset_state(self) -> None:
        """Reset all simulation state so run() is repeatable."""
        self._equity = self.config.initial_equity
        self._trades = []
        self._equity_history = []
        self._current_position = None
        self._halt_trading = False
        self._halts = []
        self._active_halt = None
        self._halt_resume_bar = None
        self._halt_until_next_day = False
        self._flatten_at_bar = None
        self._day_start_equity = self.config.initial_equity
        self._consecutive_losses = 0
        self._kelly_active = False
        self._kelly_win_rate = 0.5
        self._kelly_avg_win = 100.0
        self._kelly_avg_loss = 100.0
        self._n_wins = 0
        self._n_losses = 0
        self._sum_wins = 0.0
        self._sum_losses = 0.0

    @staticmethod
    def _finite_or_none(value: float) -> float | None:
        return float(value) if np.isfinite(value) else None

    def _check_barriers(self, bars: _Bars, j: int) -> None:
        """Exit on a stop / take-profit touched by bar ``j``'s range."""
        pos = self._current_position
        if pos is None or pos.monitor_from_bar is None or j < pos.monitor_from_bar:
            return
        reason = self._barrier_exit_reason(pos, bars.highs[j], bars.lows[j])
        if reason is None:
            return
        # A position already open at bar j's open can gap through its stop;
        # one filled AT that open cannot.
        opened_this_open = self.config.fill_timing == FillTiming.OPEN and pos.entry_bar == j
        bar_open = None if opened_this_open else float(bars.opens[j])
        price = self._resolve_exit_price(reason, pos, float(bars.closes[j]), bar_open=bar_open)
        self._close_position(
            price, bars.timestamps[j], self._finite_or_none(bars.atr[j]), reason=reason
        )

    def _check_time_exit(self, bars: _Bars, j: int) -> None:
        """Max-holding exit at close[j], counted from the signal bar."""
        pos = self._current_position
        max_hold = self.config.max_holding_period
        if pos is None or max_hold <= 0 or pos.signal_bar is None:
            return
        if j >= pos.signal_bar + max_hold:
            self._close_position(
                float(bars.closes[j]),
                bars.timestamps[j],
                self._finite_or_none(bars.atr[j]),
                reason=ExitReason.MAX_HOLDING,
            )

    def _execute_orders(self, bars: _Bars, j: int, delay: int) -> None:
        """Execute everything decided at close[j - delay] at bar j's fill price.

        Order: circuit-breaker flatten, signal exit (reversal / neutral), then
        a new entry from the signal of bar ``j - delay``.
        """
        fill = float(bars.fills[j])
        ts = bars.timestamps[j]
        bar_atr = self._finite_or_none(bars.atr[j])

        if self._flatten_at_bar is not None and j >= self._flatten_at_bar:
            self._flatten_at_bar = None
            self._close_position(fill, ts, bar_atr, reason=ExitReason.CIRCUIT_BREAKER)

        s = j - delay
        if s < 0:
            return
        signal = int(bars.predictions[s])

        pos = self._current_position
        if pos is not None and j - pos.entry_bar >= self.config.min_holding_period:
            if signal == -pos.direction:
                self._close_position(fill, ts, bar_atr, reason=ExitReason.SIGNAL_REVERSAL)
            elif signal == 0:
                self._close_position(fill, ts, bar_atr, reason=ExitReason.SIGNAL_NEUTRAL)

        if (
            self._current_position is not None
            or self._active_halt is not None
            or signal == 0
            or not bars.can_enter[j]
            or (signal == -1 and not self.config.allow_short)
        ):
            return

        # Barriers and adverse selection use what was known at the signal bar
        signal_atr = self._finite_or_none(bars.atr[s])
        if self.config.uses_barriers and (signal_atr is None or signal_atr <= 0):
            return  # labels are undefined without ATR; no barrier-less trade
        signal_close = bars.closes[s]
        if signal_atr is not None and signal_close > 0:
            realized_vol = signal_atr / signal_close
        else:
            realized_vol = 0.15

        entry_price = self.market_hours_filter.apply_adverse_selection(
            signal_price=fill,
            signal_direction=signal,
            realized_volatility=realized_vol,
        )
        label_val = bars.labels[s]
        monitor_from = j if self.config.fill_timing == FillTiming.OPEN else j + 1
        self._open_position(
            direction=signal,
            price=entry_price,
            timestamp=ts,
            bar_idx=j,
            label=None if pd.isna(label_val) else int(label_val),
            prediction=signal,
            confidence=float(bars.confidences[s]),
            atr=signal_atr,
            signal_bar=s,
            monitor_from_bar=monitor_from,
        )

    def _start_halt(self, j: int, ts: datetime, reason: HaltReason, value: float) -> None:
        """Pause trading, flatten, and schedule the resume."""
        cfg = self.config
        event = HaltEvent(bar=j, timestamp=ts, reason=reason, value=float(value))
        self._halts.append(event)
        self._active_halt = event
        self._halt_resume_bar = None
        self._halt_until_next_day = False
        if reason == HaltReason.MAX_DRAWDOWN:
            if not cfg.drawdown_halt_permanent:
                if cfg.drawdown_cooloff_bars > 0:
                    self._halt_resume_bar = j + cfg.drawdown_cooloff_bars
                else:
                    self._halt_until_next_day = True
        else:
            self._halt_until_next_day = True
        logger.info(f"Circuit breaker {reason.value} ({value:.4g}) at bar {j} ({ts}): paused")
        if self._current_position is not None:
            # Decided at close[j] -> executed like any other order decision
            self._flatten_at_bar = j + cfg.resolved_signal_delay

    def _end_halt(self, j: int, ts: datetime) -> HaltEvent | None:
        """Resume trading after a pause; returns the halt that ended."""
        event = self._active_halt
        if event is None:
            return None
        event.resume_bar = j
        event.resume_time = ts
        if event.reason == HaltReason.CONSECUTIVE_LOSSES:
            self._consecutive_losses = 0
        self._active_halt = None
        self._halt_resume_bar = None
        self._halt_until_next_day = False
        return event

    def _check_circuit_breakers(
        self, j: int, ts: datetime, equity: float, peak: float
    ) -> HaltReason | None:
        """Evaluate the breakers on the close-of-bar mark (None = no trip)."""
        cfg = self.config
        drawdown = (equity - peak) / peak if peak > 0 else 0.0
        if drawdown < -cfg.max_drawdown_threshold:
            self._start_halt(j, ts, HaltReason.MAX_DRAWDOWN, drawdown)
            return HaltReason.MAX_DRAWDOWN
        day_start = self._day_start_equity
        daily_return = (equity - day_start) / day_start if day_start > 0 else 0.0
        if daily_return < -cfg.daily_loss_threshold:
            self._start_halt(j, ts, HaltReason.DAILY_LOSS, daily_return)
            return HaltReason.DAILY_LOSS
        if self._consecutive_losses >= cfg.consecutive_loss_limit:
            self._start_halt(j, ts, HaltReason.CONSECUTIVE_LOSSES, self._consecutive_losses)
            return HaltReason.CONSECUTIVE_LOSSES
        return None

    def _mark_to_market(self, close_price: float) -> float:
        """Realized equity plus the open position's unrealized P&L."""
        if self._current_position is None:
            return self._equity
        return self._equity + self._calculate_unrealized_pnl(self._current_position, close_price)

    def run(self) -> BacktestResult:
        """
        Run the backtest simulation.

        Returns:
            BacktestResult with equity curve (one point per bar), metrics,
            trades and circuit-breaker halts
        """
        data = self._align_data()
        bars = self._prepare_bars(data)
        n_bars = len(data)
        cfg = self.config
        delay = cfg.resolved_signal_delay
        timing = cfg.fill_timing

        if cfg.uses_barriers:
            logger.info(
                f"Barrier-aligned backtest: k_up={cfg.barrier_k_up:.2f}, "
                f"k_down={cfg.barrier_k_down:.2f}, cost_in_atr={self._barrier_cost_in_atr:.4f}, "
                f"max_holding={cfg.max_holding_period}"
            )

        self._reset_state()
        peak = self._equity
        bars_halted = 0

        for j in range(n_bars):
            ts = bars.timestamps[j]

            # New trading day: reset the daily baseline, end day-scoped halts
            resumed: HaltEvent | None = None
            if j > 0 and bars.day_ids[j] != bars.day_ids[j - 1]:
                self._day_start_equity = self._equity_history[-1][1]
                if self._active_halt is not None and self._halt_until_next_day:
                    resumed = self._end_halt(j, ts)
            if (
                self._active_halt is not None
                and self._halt_resume_bar is not None
                and j >= self._halt_resume_bar
            ):
                resumed = self._end_halt(j, ts)
            if resumed is not None and resumed.reason == HaltReason.MAX_DRAWDOWN:
                # Flat while halted, so equity cannot recover: re-baseline the
                # peak or the breaker would re-trip on the first bar back
                peak = self._equity_history[-1][1]

            # Orders and barrier checks in intrabar chronological order
            if timing == FillTiming.OPEN:
                self._execute_orders(bars, j, delay)
                self._check_barriers(bars, j)
                self._check_time_exit(bars, j)
            elif timing == FillTiming.INTRABAR:
                self._check_barriers(bars, j)
                self._execute_orders(bars, j, delay)
                self._check_time_exit(bars, j)
            else:
                self._check_barriers(bars, j)
                self._check_time_exit(bars, j)
                self._execute_orders(bars, j, delay)

            # Scheduled closes at close[j]
            close_j = float(bars.closes[j])
            if self._current_position is not None and (bars.session_end[j] or j == n_bars - 1):
                reason = ExitReason.SESSION_END if bars.session_end[j] else ExitReason.END_OF_DATA
                self._close_position(close_j, ts, self._finite_or_none(bars.atr[j]), reason=reason)

            equity = self._mark_to_market(close_j)

            # Circuit breakers on the close-of-bar mark (mark-to-market, so
            # adverse moves in open positions count, not just realized losses)
            if self._active_halt is None:
                peak = max(peak, equity)
                self._check_circuit_breakers(j, ts, equity, peak)
                if (
                    self._flatten_at_bar is not None
                    and self._flatten_at_bar <= j
                    and self._current_position is not None
                ):
                    # Zero-delay close fills execute the flatten at this close
                    self._flatten_at_bar = None
                    self._close_position(
                        close_j,
                        ts,
                        self._finite_or_none(bars.atr[j]),
                        reason=ExitReason.CIRCUIT_BREAKER,
                    )
                    equity = self._mark_to_market(close_j)
            else:
                bars_halted += 1

            self._equity_history.append((ts, equity))

        self._halt_trading = self._active_halt is not None

        # Build equity curve
        equity_curve = EquityCurve(
            initial_equity=cfg.initial_equity,
            timestamps=[t for t, _ in self._equity_history],
            equity_values=[e for _, e in self._equity_history],
            trades=self._trades,
        )

        # Derive periods_per_year from data frequency
        periods_per_year = 252  # fallback
        if n_bars >= 2:
            span_days = (bars.timestamps[-1] - bars.timestamps[0]).total_seconds() / 86400.0
            if span_days > 0:
                years = span_days / 365.25
                periods_per_year = max(1, int(round(n_bars / years)))

        # Calculate metrics
        metrics = equity_curve.get_metrics(periods_per_year=periods_per_year)

        # Additional statistics
        preds = bars.predictions
        signals_long = int(np.sum(preds == 1))
        signals_short = int(np.sum(preds == -1))
        signals_count = {
            "long": signals_long,
            "short": signals_short,
            "neutral": int(n_bars - signals_long - signals_short),
        }
        halts_by_reason: dict[str, int] = {}
        for event in self._halts:
            halts_by_reason[event.reason.value] = halts_by_reason.get(event.reason.value, 0) + 1
        stats = {
            "total_bars": n_bars,
            "signals": signals_count,
            "position_rate": (signals_long + signals_short) / n_bars,
            "long_trades": sum(1 for t in self._trades if t.direction == 1),
            "short_trades": sum(1 for t in self._trades if t.direction == -1),
            "execution_model": cfg.execution_model.value,
            "signal_delay_bars": delay,
            "barrier_cost_in_atr": self._barrier_cost_in_atr,
            "n_halts": len(self._halts),
            "halted_at": str(self._halts[0].timestamp) if self._halts else None,
            "halts_by_reason": halts_by_reason,
            "bars_halted": bars_halted,
        }

        return BacktestResult(
            equity_curve=equity_curve,
            metrics=metrics,
            trades=self._trades,
            config=cfg,
            stats=stats,
            halts=list(self._halts),
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

    min_samples_per_fold = 2
    if split_size < min_samples_per_fold:
        raise ValueError(
            f"Walk-forward splits too small: {split_size} samples per fold "
            f"(minimum {min_samples_per_fold}). Have {n_samples} samples with "
            f"{n_splits} splits. Reduce n_splits or provide more data."
        )

    for i in range(n_splits):
        start_idx = i * split_size
        end_idx = (i + 1) * split_size if i < n_splits - 1 else n_samples

        split_predictions = predictions.iloc[start_idx:end_idx].copy()
        split_prices = prices.iloc[start_idx:end_idx].copy()

        if len(split_predictions) < min_samples_per_fold:
            raise ValueError(
                f"Walk-forward split {i + 1}/{n_splits} has only {len(split_predictions)} "
                f"samples (minimum {min_samples_per_fold}). Reduce n_splits or provide more data."
            )

        result = run_backtest(split_predictions, split_prices, config)
        results.append(result)

    return results


__all__ = [
    "ExecutionModel",
    "ExitReason",
    "FillTiming",
    "HaltEvent",
    "HaltReason",
    "BacktestConfig",
    "Position",
    "BacktestResult",
    "Backtester",
    "run_backtest",
    "run_walk_forward_backtest",
]
