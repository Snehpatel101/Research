"""
Triple-Barrier Labeling - Unified Implementation.

Consolidates both implementations:
- Numba-optimized core from phase1/stages/labeling/triple_barrier.py
- Configuration and adaptive barriers from src/labeling/triple_barrier.py

Implements the Lopez de Prado triple-barrier method with:
- ATR-based dynamic barriers
- Asymmetric barriers for bias correction
- Numba optimization for performance
- Transaction cost adjustment for realistic label accuracy
- Adaptive volatility scaling (optional)

Labels:
    -1 = Short (lower barrier hit first)
     0 = Neutral/Timeout (time barrier hit)
    +1 = Long (upper barrier hit first)
   -99 = Invalid (ambiguous or insufficient data)

CRITICAL FIX (2024-12): ASYMMETRIC BARRIERS TO CORRECT LONG BIAS
Previous symmetric barriers (k_up = k_down) in a historically bullish market
produced 87-91% long signals. New asymmetric barriers (k_up > k_down) make the
lower barrier easier to hit, targeting ~50/50 long/short distribution.

CRITICAL FIX (2025-12): TRANSACTION COST ADJUSTMENT
Labels now account for round-trip transaction costs (commission + slippage).
Both barriers are adjusted by cost_in_atr units symmetrically, ensuring a WIN label
(long or short) actually represents a profitable trade after costs.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.core.utils.atr import wilder_atr
from src.data.labeling.base import LabelingResult, LabelingStrategy, LabelingType

# Try to import numba, fall back to pure Python if unavailable
try:
    import numba as nb

    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    nb = None

logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================


@dataclass
class TripleBarrierConfig:
    """
    Configuration for triple-barrier labeling.

    The triple-barrier method creates labels based on which barrier
    (upper, lower, or time) is hit first from each entry point.

    Attributes:
        upper_mult: ATR multiplier for upper (profit) barrier (k_up).
            Higher values = wider profit targets = fewer long labels.
        lower_mult: ATR multiplier for lower (stop) barrier (k_down).
            Higher values = wider stops = fewer short labels.
        horizon: Maximum holding period in bars (time barrier / max_bars).
        atr_period: Window for Average True Range calculation.
        atr_column: Column name for pre-computed ATR (if available).
        use_adaptive_barriers: If True, scale barriers by recent volatility.
        vol_lookback: Lookback period for volatility scaling when adaptive.
        apply_transaction_costs: If True, adjust upper barrier for costs.
        symbol: Trading symbol for cost lookup (e.g., 'MES').
        volatility_regime: 'low_vol' or 'high_vol' for slippage estimation.
        cost_calibration_fraction: Leading share of the rows whose median ATR
            converts the price cost into ATR units. Pass the training split's
            share (``train_ratio``) so validation/test volatility never shapes
            the training labels; 1.0 uses every row.

    Example:
        config = TripleBarrierConfig(
            upper_mult=2.0,
            lower_mult=1.5,
            horizon=20,
            atr_period=14,
        )
        labeler = TripleBarrierLabeler(config)
        labels = labeler.create_labels(df)
    """

    upper_mult: float = 2.0
    lower_mult: float = 2.0
    horizon: int = 20
    atr_period: int = 14
    atr_column: str | None = "atr_14"  # None = compute Wilder ATR inline from OHLCV
    use_adaptive_barriers: bool = False
    vol_lookback: int = 60
    apply_transaction_costs: bool = True
    symbol: str = "MES"
    volatility_regime: str = "low_vol"
    cost_calibration_fraction: float = 1.0

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if self.upper_mult <= 0:
            raise ValueError(f"upper_mult must be positive, got {self.upper_mult}")
        if self.lower_mult <= 0:
            raise ValueError(f"lower_mult must be positive, got {self.lower_mult}")
        if self.horizon < 1:
            raise ValueError(f"horizon must be at least 1, got {self.horizon}")
        if self.atr_period < 1:
            raise ValueError(f"atr_period must be at least 1, got {self.atr_period}")
        if self.vol_lookback < 1:
            raise ValueError(f"vol_lookback must be at least 1, got {self.vol_lookback}")
        if not 0 < self.cost_calibration_fraction <= 1:
            raise ValueError(
                "cost_calibration_fraction must be in (0, 1], got "
                f"{self.cost_calibration_fraction}"
            )

    def to_dict(self) -> dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "upper_mult": self.upper_mult,
            "lower_mult": self.lower_mult,
            "horizon": self.horizon,
            "atr_period": self.atr_period,
            "atr_column": self.atr_column,
            "use_adaptive_barriers": self.use_adaptive_barriers,
            "vol_lookback": self.vol_lookback,
            "apply_transaction_costs": self.apply_transaction_costs,
            "symbol": self.symbol,
            "volatility_regime": self.volatility_regime,
            "cost_calibration_fraction": self.cost_calibration_fraction,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> TripleBarrierConfig:
        """Create config from dictionary."""
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# =============================================================================
# NUMBA-OPTIMIZED CORE FUNCTIONS
# =============================================================================


def _triple_barrier_python(
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    atr: np.ndarray,
    k_up: float,
    k_down: float,
    max_bars: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Pure Python fallback for triple-barrier labeling."""
    n = len(close)
    labels = np.zeros(n, dtype=np.int8)
    bars_to_hit = np.zeros(n, dtype=np.int32)
    mae = np.zeros(n, dtype=np.float32)
    mfe = np.zeros(n, dtype=np.float32)
    touch_type = np.zeros(n, dtype=np.int8)

    for i in range(n - 1):
        entry_price = close[i]
        entry_atr = atr[i]

        if np.isnan(entry_atr) or entry_atr <= 0:
            labels[i] = -99
            bars_to_hit[i] = max_bars
            continue

        upper_barrier = entry_price + k_up * entry_atr
        lower_barrier = entry_price - k_down * entry_atr

        max_adverse = 0.0
        max_favorable = 0.0
        hit = False

        for j in range(1, min(max_bars + 1, n - i)):
            idx = i + j
            bar_high = high[idx]
            bar_low = low[idx]

            upside = (bar_high - entry_price) / entry_price
            downside = (bar_low - entry_price) / entry_price

            if upside > max_favorable:
                max_favorable = upside
            if downside < max_adverse:
                max_adverse = downside

            upper_hit = bar_high >= upper_barrier
            lower_hit = bar_low <= lower_barrier

            if upper_hit and lower_hit:
                # Resolve by distance to barrier (match adaptive_barriers)
                dist_to_upper = abs(bar_high - upper_barrier)
                dist_to_lower = abs(bar_low - lower_barrier)
                if dist_to_upper <= dist_to_lower:
                    labels[i] = 1
                    touch_type[i] = 1
                else:
                    labels[i] = -1
                    touch_type[i] = -1
                bars_to_hit[i] = j
                hit = True
                break
            elif upper_hit:
                labels[i] = 1
                bars_to_hit[i] = j
                touch_type[i] = 1
                hit = True
                break
            elif lower_hit:
                labels[i] = -1
                bars_to_hit[i] = j
                touch_type[i] = -1
                hit = True
                break

        if not hit:
            labels[i] = 0
            bars_to_hit[i] = max_bars
            touch_type[i] = 0

        mae[i] = max_adverse
        mfe[i] = max_favorable

    # Mark last max_bars samples as invalid
    for i in range(max(0, n - max_bars), n):
        labels[i] = -99
        bars_to_hit[i] = 0
        mae[i] = 0.0
        mfe[i] = 0.0
        touch_type[i] = 0

    return labels, bars_to_hit, mae, mfe, touch_type


if NUMBA_AVAILABLE:

    @nb.jit(nopython=True, cache=True)
    def triple_barrier_numba(
        close: np.ndarray,
        high: np.ndarray,
        low: np.ndarray,
        atr: np.ndarray,
        k_up: float,
        k_down: float,
        max_bars: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Numba-optimized triple barrier labeling.

        Parameters
        ----------
        close : np.ndarray
            Array of close prices
        high : np.ndarray
            Array of high prices
        low : np.ndarray
            Array of low prices
        atr : np.ndarray
            Array of ATR values
        k_up : float
            Profit barrier multiplier (e.g., 2.0 means 2*ATR above entry)
        k_down : float
            Stop barrier multiplier (e.g., 1.0 means 1*ATR below entry)
        max_bars : int
            Maximum bars to hold before timeout

        Returns
        -------
        tuple
            labels : +1 (long win), -1 (short loss), 0 (timeout), -99 (invalid)
            bars_to_hit : number of bars until barrier was hit
            mae : maximum adverse excursion (as % of entry price)
            mfe : maximum favorable excursion (as % of entry price)
            touch_type : 1 (upper), -1 (lower), 0 (timeout)
        """
        n = len(close)
        labels = np.zeros(n, dtype=np.int8)
        bars_to_hit = np.zeros(n, dtype=np.int32)
        mae = np.zeros(n, dtype=np.float32)
        mfe = np.zeros(n, dtype=np.float32)
        touch_type = np.zeros(n, dtype=np.int8)

        for i in range(n - 1):
            entry_price = close[i]
            entry_atr = atr[i]

            if np.isnan(entry_atr) or entry_atr <= 0:
                labels[i] = -99
                bars_to_hit[i] = max_bars
                continue

            upper_barrier = entry_price + k_up * entry_atr
            lower_barrier = entry_price - k_down * entry_atr

            max_adverse = 0.0
            max_favorable = 0.0
            hit = False

            for j in range(1, min(max_bars + 1, n - i)):
                idx = i + j
                bar_high = high[idx]
                bar_low = low[idx]

                upside = (bar_high - entry_price) / entry_price
                downside = (bar_low - entry_price) / entry_price

                if upside > max_favorable:
                    max_favorable = upside
                if downside < max_adverse:
                    max_adverse = downside

                upper_hit = bar_high >= upper_barrier
                lower_hit = bar_low <= lower_barrier

                if upper_hit and lower_hit:
                    dist_to_upper = abs(bar_high - upper_barrier)
                    dist_to_lower = abs(bar_low - lower_barrier)
                    if dist_to_upper <= dist_to_lower:
                        labels[i] = 1
                        touch_type[i] = 1
                    else:
                        labels[i] = -1
                        touch_type[i] = -1
                    bars_to_hit[i] = j
                    hit = True
                    break
                elif upper_hit:
                    labels[i] = 1
                    bars_to_hit[i] = j
                    touch_type[i] = 1
                    hit = True
                    break
                elif lower_hit:
                    labels[i] = -1
                    bars_to_hit[i] = j
                    touch_type[i] = -1
                    hit = True
                    break

            if not hit:
                labels[i] = 0
                bars_to_hit[i] = max_bars
                touch_type[i] = 0

            mae[i] = max_adverse
            mfe[i] = max_favorable

        for i in range(max(0, n - max_bars), n):
            labels[i] = -99
            bars_to_hit[i] = 0
            mae[i] = 0.0
            mfe[i] = 0.0
            touch_type[i] = 0

        return labels, bars_to_hit, mae, mfe, touch_type

    @nb.jit(nopython=True, cache=True)
    def triple_barrier_numba_with_costs(
        close: np.ndarray,
        high: np.ndarray,
        low: np.ndarray,
        atr: np.ndarray,
        k_up: float,
        k_down: float,
        max_bars: int,
        cost_in_atr: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Numba-optimized triple barrier labeling with transaction cost adjustment.

        Transaction costs are added to both barrier requirements symmetrically.
        """
        n = len(close)
        labels = np.zeros(n, dtype=np.int8)
        bars_to_hit = np.zeros(n, dtype=np.int32)
        mae = np.zeros(n, dtype=np.float32)
        mfe = np.zeros(n, dtype=np.float32)
        touch_type = np.zeros(n, dtype=np.int8)

        k_up_effective = k_up + cost_in_atr
        k_down_effective = k_down + cost_in_atr

        for i in range(n - 1):
            entry_price = close[i]
            entry_atr = atr[i]

            if np.isnan(entry_atr) or entry_atr <= 0:
                labels[i] = -99
                bars_to_hit[i] = max_bars
                continue

            upper_barrier = entry_price + k_up_effective * entry_atr
            lower_barrier = entry_price - k_down_effective * entry_atr

            max_adverse = 0.0
            max_favorable = 0.0
            hit = False

            for j in range(1, min(max_bars + 1, n - i)):
                idx = i + j
                bar_high = high[idx]
                bar_low = low[idx]

                upside = (bar_high - entry_price) / entry_price
                downside = (bar_low - entry_price) / entry_price

                if upside > max_favorable:
                    max_favorable = upside
                if downside < max_adverse:
                    max_adverse = downside

                upper_hit = bar_high >= upper_barrier
                lower_hit = bar_low <= lower_barrier

                if upper_hit and lower_hit:
                    dist_to_upper = abs(bar_high - upper_barrier)
                    dist_to_lower = abs(bar_low - lower_barrier)
                    if dist_to_upper <= dist_to_lower:
                        labels[i] = 1
                        touch_type[i] = 1
                    else:
                        labels[i] = -1
                        touch_type[i] = -1
                    bars_to_hit[i] = j
                    hit = True
                    break
                elif upper_hit:
                    labels[i] = 1
                    bars_to_hit[i] = j
                    touch_type[i] = 1
                    hit = True
                    break
                elif lower_hit:
                    labels[i] = -1
                    bars_to_hit[i] = j
                    touch_type[i] = -1
                    hit = True
                    break

            if not hit:
                labels[i] = 0
                bars_to_hit[i] = max_bars
                touch_type[i] = 0

            mae[i] = max_adverse
            mfe[i] = max_favorable

        for i in range(max(0, n - max_bars), n):
            labels[i] = -99
            bars_to_hit[i] = 0
            mae[i] = 0.0
            mfe[i] = 0.0
            touch_type[i] = 0

        return labels, bars_to_hit, mae, mfe, touch_type

else:
    # Fallback to pure Python
    triple_barrier_numba = _triple_barrier_python

    def _triple_barrier_with_costs_python(
        close: np.ndarray,
        high: np.ndarray,
        low: np.ndarray,
        atr: np.ndarray,
        k_up: float,
        k_down: float,
        max_bars: int,
        cost_in_atr: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Pure Python fallback with transaction costs."""
        # Adjust both barriers to include costs (symmetric)
        return _triple_barrier_python(
            close, high, low, atr, k_up + cost_in_atr, k_down + cost_in_atr, max_bars
        )

    triple_barrier_numba_with_costs = _triple_barrier_with_costs_python


# =============================================================================
# TRANSACTION COST -> BARRIER DISTANCE (shared with the backtester)
# =============================================================================
# Single source of truth for the cost term of the barriers. The labeler and
# src/inference/backtesting/backtest.py both call these, so a label's barrier
# and the backtest's stop/take-profit are the same distance from entry:
#
#   distance_up   = (k_up   + cost_in_atr) * ATR
#   distance_down = (k_down + cost_in_atr) * ATR
#
# cost_in_atr is the round-trip cost in PRICE units (ticks * tick_size, not
# tick_value dollars) divided by the median ATR of the training rows.


def transaction_cost_in_price(symbol: str, volatility_regime: str = "low_vol") -> float:
    """Round-trip trade cost (commission + entry/exit slippage) in price units.

    ``get_total_trade_cost`` returns ticks; one tick is ``tick_size`` price
    points (MES: 0.25). Multiplying by ``tick_value`` (dollars per tick) would
    mix dollars into a price-space barrier (5x too wide for MES).
    """
    from src.config.symbol import SymbolConfig
    from src.data.pipeline.config.barriers_config import get_total_trade_cost

    cost_ticks = get_total_trade_cost(symbol, volatility_regime, include_slippage=True)
    tick_size = SymbolConfig.from_symbol_or_default(symbol).tick_size
    return float(cost_ticks * tick_size)


def cost_in_atr_units(cost_in_price: float, atr_values: np.ndarray) -> float:
    """Express a price-unit cost in ATR units using the median valid ATR.

    One median over the calibration rows is intentional: a training-time
    calibration that turns a fixed price cost into one ATR multiple for the
    whole dataset (an expanding median would bias early samples with larger
    cost adjustments). Pass only training rows' ATR (the labeler does, via
    ``cost_calibration_fraction``) so later volatility does not leak in.
    Returns 0.0 when no ATR value is valid.
    """
    atr = np.asarray(atr_values, dtype=float)
    valid = atr[np.isfinite(atr) & (atr > 0)]
    if len(valid) == 0:
        return 0.0
    return float(cost_in_price / np.median(valid))


def expanding_cost_in_atr(cost_in_price: float, atr_values: np.ndarray) -> np.ndarray:
    """Causal cost term per bar: price cost over the expanding median valid ATR.

    Element ``i`` uses only ``atr_values[: i + 1]`` (invalid ATR entries -
    NaN, non-positive - are skipped), so appending later bars never changes
    it. 0.0 until the first valid ATR. The backtester uses this when no
    labeling-run cost is passed; the labeler keeps its single training-rows
    median (``cost_in_atr_units``).

    Warm-up: the median needs one valid ATR (``min_periods=1``), not a full
    ATR period. The ATR itself is already an average over its period (NaN
    before it), and no trade opens on a bar without a valid ATR, so every
    trade gets a cost on the scale of the labels' cost. A longer warm-up
    would have to return 0.0 (barriers narrower than the labels') or block
    entries; the early values are noisier estimates of the same quantity.
    """
    atr = np.asarray(atr_values, dtype=float)
    valid = pd.Series(np.where(np.isfinite(atr) & (atr > 0), atr, np.nan))
    median = valid.expanding(min_periods=1).median().to_numpy()
    out = np.zeros(len(atr), dtype=float)
    known = np.isfinite(median) & (median > 0)
    out[known] = cost_in_price / median[known]
    return out


def compute_cost_in_atr(
    symbol: str, atr_values: np.ndarray, volatility_regime: str = "low_vol"
) -> float:
    """Round-trip cost of ``symbol`` in ATR units (labeler and backtester)."""
    return cost_in_atr_units(transaction_cost_in_price(symbol, volatility_regime), atr_values)


def barrier_distances(
    atr: float, k_up: float, k_down: float, cost_in_atr: float
) -> tuple[float, float]:
    """Price distance of the (upper, lower) barrier from the entry reference.

    Mirrors ``triple_barrier_numba_with_costs``: both multipliers are widened
    by ``cost_in_atr`` (pass 0.0 for the cost-free labeler).
    """
    return (k_up + cost_in_atr) * atr, (k_down + cost_in_atr) * atr


# =============================================================================
# TRIPLE BARRIER LABELER
# =============================================================================


class TripleBarrierLabeler(LabelingStrategy):
    """
    Triple-barrier labeling with ATR-based dynamic barriers.

    Creates labels based on which of three barriers is hit first:
        - Upper barrier: Entry price + (ATR * upper_mult) -> Long (+1)
        - Lower barrier: Entry price - (ATR * lower_mult) -> Short (-1)
        - Time barrier: horizon bars elapsed -> Neutral (0)

    The ATR (Average True Range) adapts barriers to current volatility,
    making labels more stable across different market regimes.

    Transaction Cost Adjustment (2025-12):
    When apply_transaction_costs=True, the upper barrier is adjusted by adding
    transaction costs (in ATR units). This ensures a WIN label represents a
    trade that is profitable AFTER commission and slippage.

    Example:
        >>> config = TripleBarrierConfig(upper_mult=2.0, lower_mult=1.5, horizon=20)
        >>> labeler = TripleBarrierLabeler(config)
        >>> result = labeler.compute_labels(ohlcv_df, horizon=20)
        >>> print(result.quality_metrics)
    """

    def __init__(self, config: TripleBarrierConfig | None = None) -> None:
        """
        Initialize the triple-barrier labeler.

        Args:
            config: TripleBarrierConfig instance. If None, uses defaults.
        """
        self.config = config or TripleBarrierConfig()

    @property
    def labeling_type(self) -> LabelingType:
        """Return the type of this labeling strategy."""
        return LabelingType.TRIPLE_BARRIER

    @property
    def required_columns(self) -> list[str]:
        """Return list of required DataFrame columns."""
        base_cols = ["close", "high", "low", "open"]
        # ATR column is required if not computing inline
        if self.config.atr_column:
            return base_cols + [self.config.atr_column]
        return base_cols

    def _compute_volatility_scaling(self, df: pd.DataFrame) -> np.ndarray:
        """
        Compute volatility scaling factor for adaptive barriers.
        """
        if not self.config.use_adaptive_barriers:
            return np.ones(len(df))

        returns = df["close"].pct_change().values
        vol_lookback = self.config.vol_lookback

        # Rolling std
        scaling = np.ones(len(df))
        for i in range(vol_lookback, len(df)):
            recent_vol = np.nanstd(returns[i - vol_lookback : i])
            long_term_vol = np.nanstd(returns[:i])
            if long_term_vol > 0:
                ratio = recent_vol / long_term_vol
                scaling[i] = np.clip(ratio, 0.5, 2.0)

        return scaling

    def _calculate_cost_in_atr(self, atr_values: np.ndarray) -> float:
        """
        Calculate transaction cost expressed in ATR units.

        Delegates to ``compute_cost_in_atr`` — the same helper the backtester
        uses for its stop/take-profit distances (label/backtest parity).
        """
        atr = np.asarray(atr_values, dtype=float)
        if not np.any(np.isfinite(atr) & (atr > 0)):
            logger.warning("No valid ATR values for cost calculation, using cost_in_atr=0")
            return 0.0
        return compute_cost_in_atr(self.config.symbol, atr, self.config.volatility_regime)

    def compute_labels(
        self,
        df: pd.DataFrame,
        horizon: int,
        k_up: float | None = None,
        k_down: float | None = None,
        max_bars: int | None = None,
        **kwargs: Any,
    ) -> LabelingResult:
        """
        Compute triple-barrier labels for the given horizon.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with OHLCV data and ATR
        horizon : int
            Horizon identifier (e.g., 1, 5, 20)
        k_up : float, optional
            Override for upper barrier multiplier
        k_down : float, optional
            Override for lower barrier multiplier
        max_bars : int, optional
            Override for maximum bars
        **kwargs : Any
            Additional parameters (ignored)

        Returns
        -------
        LabelingResult
            Container with labels and quality metrics
        """
        self.validate_inputs(df)

        if not isinstance(horizon, int) or horizon <= 0:
            raise ValueError(f"horizon must be a positive integer, got {horizon}")

        if len(df) < horizon:
            raise ValueError(
                f"DataFrame has {len(df)} rows but horizon={horizon}. "
                f"Need at least {horizon} rows for meaningful labeling."
            )

        # Resolve parameters (use `is not None` to allow explicit 0.0 values)
        k_up = k_up if k_up is not None else self.config.upper_mult
        k_down = k_down if k_down is not None else self.config.lower_mult
        max_bars = max_bars if max_bars is not None else self.config.horizon

        logger.info(f"Computing triple-barrier labels for horizon {horizon}")
        logger.info(f"  k_up={k_up:.3f}, k_down={k_down:.3f}, max_bars={max_bars}")

        # Extract arrays
        close = df["close"].values
        high = df["high"].values
        low = df["low"].values

        # Get or compute ATR
        if self.config.atr_column and self.config.atr_column in df.columns:
            atr = df[self.config.atr_column].values
        else:
            logger.info("  ATR column not found, computing inline")
            # Canonical Wilder ATR (the backtester's): NaN on the first
            # atr_period bars, which the barrier loop labels -99 (invalid)
            atr = wilder_atr(df["high"], df["low"], df["close"], self.config.atr_period)

        # Apply adaptive scaling if enabled
        if self.config.use_adaptive_barriers:
            vol_scaling = self._compute_volatility_scaling(df)
            atr = atr * vol_scaling
            logger.info("  Adaptive barriers enabled")

        # Calculate transaction cost adjustment
        cost_in_atr = 0.0
        if self.config.apply_transaction_costs:
            # Calibrated on the leading (training) rows only
            n_calibration = max(1, int(len(atr) * self.config.cost_calibration_fraction))
            cost_in_atr = self._calculate_cost_in_atr(atr[:n_calibration])
            logger.info(
                f"  Transaction costs applied: symbol={self.config.symbol}, "
                f"regime={self.config.volatility_regime}, cost_in_atr={cost_in_atr:.4f}"
            )

            labels, bars_to_hit, mae, mfe, touch_type = triple_barrier_numba_with_costs(
                close, high, low, atr, k_up, k_down, max_bars, cost_in_atr
            )
        else:
            logger.info("  Transaction costs NOT applied")
            labels, bars_to_hit, mae, mfe, touch_type = triple_barrier_numba(
                close, high, low, atr, k_up, k_down, max_bars
            )

        # Build metadata
        metadata = {
            "bars_to_hit": bars_to_hit,
            "mae": mae,
            "mfe": mfe,
            "touch_type": touch_type,
        }

        if self.config.apply_transaction_costs:
            metadata["transaction_cost_applied"] = np.array([True])
            metadata["cost_in_atr"] = np.array([cost_in_atr])

        result = LabelingResult(labels=labels, horizon=horizon, metadata=metadata)
        result.quality_metrics = self.get_quality_metrics(result)

        self._log_label_statistics(result, horizon)

        return result

    def create_labels(self, df: pd.DataFrame) -> pd.Series:
        """
        Simplified interface for creating labels using config horizon.

        Args:
            df: OHLCV DataFrame.

        Returns:
            pd.Series: Labels (-1, 0, +1) indexed like input DataFrame.
        """
        result = self.compute_labels(df, horizon=self.config.horizon)
        return pd.Series(result.labels, index=df.index, name="label")

    def create_labels_with_ends(self, df: pd.DataFrame) -> tuple[pd.Series, np.ndarray]:
        """
        Labels plus the bar position at which each label resolves.

        A label decided at row ``i`` is known once a barrier is touched,
        ``bars_to_hit`` bars later, so its span is ``[i, i + bars_to_hit]``
        (row positions of ``df``). CV purging and sample-uniqueness weights
        need that span; invalid labels (-99) get end position -1.

        Args:
            df: OHLCV DataFrame.

        Returns:
            (labels indexed like ``df``, int64 label-end row positions)
        """
        from src.core.label_spans import label_end_positions

        result = self.compute_labels(df, horizon=self.config.horizon)
        labels = pd.Series(result.labels, index=df.index, name="label")
        ends = label_end_positions(result.labels, result.metadata["bars_to_hit"])
        return labels, ends

    def _log_label_statistics(self, result: LabelingResult, horizon: int) -> None:
        """Log label distribution statistics."""
        labels = result.labels
        valid_mask = labels != -99
        valid_labels = labels[valid_mask]

        if len(valid_labels) == 0:
            logger.warning(f"No valid labels for horizon {horizon}")
            return

        total = len(valid_labels)

        logger.info(f"Label distribution for horizon {horizon}:")
        for label_val in [-1, 0, 1]:
            count = int((valid_labels == label_val).sum())
            pct = count / total * 100
            label_name = {-1: "Short/Loss", 0: "Neutral/Timeout", 1: "Long/Win"}[label_val]
            logger.info(f"  {label_name:20s}: {count:6d} ({pct:5.1f}%)")

        invalid_count = int((~valid_mask).sum())
        if invalid_count > 0:
            logger.info(f"  Invalid samples: {invalid_count} (excluded from training)")

    def get_quality_metrics(self, result: LabelingResult) -> dict[str, float]:
        """Compute quality metrics including MAE/MFE analysis."""
        metrics = super().get_quality_metrics(result)

        labels = result.labels
        valid_mask = labels != -99

        mae = result.metadata.get("mae", np.array([]))
        mfe = result.metadata.get("mfe", np.array([]))

        if len(mae) > 0 and len(mfe) > 0:
            valid_mae = mae[valid_mask]
            valid_mfe = mfe[valid_mask]

            if len(valid_mae) > 0:
                metrics["avg_mae"] = float(np.mean(valid_mae))
                metrics["avg_mfe"] = float(np.mean(valid_mfe))
                avg_mae_abs = abs(metrics["avg_mae"]) if metrics["avg_mae"] != 0 else 1e-6
                metrics["mfe_mae_ratio"] = abs(metrics["avg_mfe"]) / avg_mae_abs

        bars_to_hit = result.metadata.get("bars_to_hit", np.array([]))
        if len(bars_to_hit) > 0:
            valid_bars = bars_to_hit[valid_mask]
            if len(valid_bars) > 0:
                non_zero_bars = valid_bars[valid_bars > 0]
                if len(non_zero_bars) > 0:
                    metrics["avg_bars_to_hit"] = float(np.mean(non_zero_bars))

        return metrics


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "TripleBarrierConfig",
    "TripleBarrierLabeler",
    "barrier_distances",
    "compute_cost_in_atr",
    "cost_in_atr_units",
    "transaction_cost_in_price",
    "triple_barrier_numba",
    "triple_barrier_numba_with_costs",
    "LabelingResult",
    "LabelingStrategy",
    "LabelingType",
]
