"""
Adaptive Triple-Barrier Labeling Strategy.

Extends TripleBarrierLabeler with regime-adaptive barrier adjustments.
Barrier parameters (k_up, k_down, max_bars) are dynamically adjusted based on
detected market regimes (volatility, trend, structure).

This addresses a key limitation of static barriers: market conditions vary,
and optimal barrier parameters depend on the current market regime.

Example regime adjustments:
- High volatility: Widen barriers to avoid noise-triggered exits
- Trending market: Extend max_bars to capture momentum
- Mean-reverting: Tighten barriers for quick reversals
"""

import logging
from typing import Any

import numpy as np
import pandas as pd

from .base import LabelingResult, LabelingType
from .triple_barrier import TripleBarrierLabeler, triple_barrier_numba

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class AdaptiveTripleBarrierLabeler(TripleBarrierLabeler):
    """
    Triple-barrier labeler with regime-adaptive barrier parameters.

    This extends TripleBarrierLabeler to adjust k_up, k_down, and max_bars
    based on detected market regimes. The regime columns must be present
    in the input DataFrame or the labeler falls back to base parameters.

    Regime Columns Expected:
    - volatility_regime: 'low', 'normal', 'high'
    - trend_regime: 'uptrend', 'downtrend', 'sideways'
    - structure_regime: 'mean_reverting', 'random', 'trending'

    Parameters
    ----------
    k_up : float, optional
        Base multiplier for upper barrier (default from config)
    k_down : float, optional
        Base multiplier for lower barrier (default from config)
    max_bars : int, optional
        Base maximum bars before timeout (default from config)
    atr_column : str
        Column name for ATR values (default: 'atr_14')
    volatility_regime_col : str
        Column name for volatility regime (default: 'volatility_regime')
    trend_regime_col : str
        Column name for trend regime (default: 'trend_regime')
    structure_regime_col : str
        Column name for structure regime (default: 'structure_regime')
    symbol : str
        Symbol name for barrier param lookup (default: None, uses defaults)

    Example
    -------
    >>> from stages.labeling import AdaptiveTripleBarrierLabeler
    >>> from stages.regime import CompositeRegimeDetector
    >>>
    >>> # Add regime columns to data
    >>> detector = CompositeRegimeDetector.with_defaults()
    >>> df_with_regimes = detector.add_regime_columns(df)
    >>>
    >>> # Create adaptive labeler
    >>> labeler = AdaptiveTripleBarrierLabeler(symbol='MES')
    >>> result = labeler.compute_labels(df_with_regimes, horizon=5)
    """

    def __init__(
        self,
        k_up: float | None = None,
        k_down: float | None = None,
        max_bars: int | None = None,
        atr_column: str = 'atr_14',
        volatility_regime_col: str = 'volatility_regime',
        trend_regime_col: str = 'trend_regime',
        structure_regime_col: str = 'structure_regime',
        symbol: str | None = None
    ):
        super().__init__(
            k_up=k_up,
            k_down=k_down,
            max_bars=max_bars,
            atr_column=atr_column
        )
        self._volatility_regime_col = volatility_regime_col
        self._trend_regime_col = trend_regime_col
        self._structure_regime_col = structure_regime_col
        self._symbol = symbol

    @property
    def labeling_type(self) -> LabelingType:
        """Return the type of this labeling strategy."""
        # Reuse TRIPLE_BARRIER type since this is a variant
        return LabelingType.TRIPLE_BARRIER

    @property
    def regime_columns(self) -> list[str]:
        """Return list of regime column names."""
        return [
            self._volatility_regime_col,
            self._trend_regime_col,
            self._structure_regime_col
        ]

    def _get_regime_adjusted_params(
        self,
        symbol: str,
        horizon: int,
        volatility_regime: str,
        trend_regime: str,
        structure_regime: str
    ) -> dict[str, Any]:
        """
        Get barrier parameters adjusted for current market regime.

        Uses config.get_regime_adjusted_barriers() for the actual adjustment
        logic, falling back to base parameters if import fails.

        Parameters
        ----------
        symbol : str
            Symbol name (e.g., 'MES', 'MGC')
        horizon : int
            Labeling horizon
        volatility_regime : str
            Current volatility regime
        trend_regime : str
            Current trend regime
        structure_regime : str
            Current structure regime

        Returns
        -------
        dict : Adjusted barrier parameters with k_up, k_down, max_bars
        """
        try:
            from src.config import get_regime_adjusted_barriers

            return get_regime_adjusted_barriers(
                symbol=symbol,
                horizon=horizon,
                volatility_regime=volatility_regime,
                trend_regime=trend_regime,
                structure_regime=structure_regime
            )
        except ImportError:
            logger.warning(
                "Could not import get_regime_adjusted_barriers, "
                "using default parameters"
            )
            defaults = self._get_default_params(horizon)
            return {
                'k_up': defaults['k_up'],
                'k_down': defaults['k_down'],
                'max_bars': defaults['max_bars']
            }

    def _has_regime_columns(self, df: pd.DataFrame) -> dict[str, bool]:
        """
        Check which regime columns are present in the DataFrame.

        Returns
        -------
        dict : Mapping of regime type to presence boolean
        """
        return {
            'volatility': self._volatility_regime_col in df.columns,
            'trend': self._trend_regime_col in df.columns,
            'structure': self._structure_regime_col in df.columns
        }

    def _get_regime_value(
        self,
        df: pd.DataFrame,
        idx: int,
        regime_col: str,
        default: str
    ) -> str:
        """
        Get regime value at index, handling missing columns and NaN values.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame
        idx : int
            Row index
        regime_col : str
            Column name for regime
        default : str
            Default value if column missing or value is NaN

        Returns
        -------
        str : Regime value or default
        """
        if regime_col not in df.columns:
            return default

        val = df.iloc[idx][regime_col]
        if pd.isna(val):
            return default

        return str(val)

    def compute_labels(
        self,
        df: pd.DataFrame,
        horizon: int,
        k_up: float | None = None,
        k_down: float | None = None,
        max_bars: int | None = None,
        **kwargs: Any
    ) -> LabelingResult:
        """
        Compute triple-barrier labels with regime-adaptive parameters.

        For each row, the barrier parameters are adjusted based on the
        detected regime at that point. If regime columns are missing,
        falls back to static base parameters.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with OHLCV data, ATR, and regime columns
        horizon : int
            Horizon identifier (e.g., 5, 20)
        k_up : float, optional
            Override base upper barrier multiplier
        k_down : float, optional
            Override base lower barrier multiplier
        max_bars : int, optional
            Override base maximum bars
        **kwargs : Any
            Additional parameters (ignored)

        Returns
        -------
        LabelingResult
            Container with labels and quality metrics

        Notes
        -----
        The adaptive labeling is more computationally expensive than static
        labeling because it cannot use the vectorized numba function directly.
        For very large datasets, consider using static labeling or sampling.
        """
        # Validate basic inputs (OHLCV + ATR)
        self.validate_inputs(df)

        if not isinstance(horizon, int) or horizon <= 0:
            raise ValueError(f"horizon must be a positive integer, got {horizon}")

        # Check regime column availability
        regime_available = self._has_regime_columns(df)
        has_any_regime = any(regime_available.values())

        if not has_any_regime:
            # No regime columns - fall back to base implementation
            logger.info(
                "No regime columns found in DataFrame, "
                "falling back to static triple-barrier labeling"
            )
            return super().compute_labels(
                df, horizon, k_up=k_up, k_down=k_down, max_bars=max_bars
            )

        # Log which regimes are available
        available_regimes = [k for k, v in regime_available.items() if v]
        missing_regimes = [k for k, v in regime_available.items() if not v]
        logger.info(
            f"Adaptive labeling with regimes: {available_regimes}"
        )
        if missing_regimes:
            logger.info(
                f"Missing regime columns (using defaults): {missing_regimes}"
            )

        # Resolve base parameters
        defaults = self._get_default_params(horizon)
        base_k_up = k_up or self._k_up or defaults['k_up']
        base_k_down = k_down or self._k_down or defaults['k_down']
        base_max_bars = max_bars or self._max_bars or defaults['max_bars']

        # Validate base parameters
        if base_k_up <= 0:
            raise ValueError(f"k_up must be positive, got {base_k_up}")
        if base_k_down <= 0:
            raise ValueError(f"k_down must be positive, got {base_k_down}")
        if base_max_bars <= 0:
            raise ValueError(f"max_bars must be positive, got {base_max_bars}")

        # Determine symbol for param lookup
        symbol = self._symbol or 'MES'  # Default to MES if not specified

        logger.info(f"Computing adaptive triple-barrier labels for horizon {horizon}")
        logger.info(f"  Base params: k_up={base_k_up:.3f}, k_down={base_k_down:.3f}, max_bars={base_max_bars}")
        logger.info(f"  Symbol: {symbol}")

        # Extract arrays
        close = df['close'].values
        high = df['high'].values
        low = df['low'].values
        open_prices = df['open'].values
        atr = df[self._atr_column].values

        n = len(df)
        labels = np.zeros(n, dtype=np.int8)
        bars_to_hit = np.zeros(n, dtype=np.int32)
        mae = np.zeros(n, dtype=np.float32)
        mfe = np.zeros(n, dtype=np.float32)
        touch_type = np.zeros(n, dtype=np.int8)

        # Track regime distribution for logging
        regime_counts: dict[str, int] = {}

        # Process each bar with adaptive parameters
        for i in range(n - 1):
            entry_price = close[i]
            entry_atr = atr[i]

            # Get current regime values
            volatility_regime = self._get_regime_value(
                df, i, self._volatility_regime_col, 'normal'
            )
            trend_regime = self._get_regime_value(
                df, i, self._trend_regime_col, 'sideways'
            )
            structure_regime = self._get_regime_value(
                df, i, self._structure_regime_col, 'random'
            )

            # Track regime combination
            regime_key = f"{volatility_regime}_{trend_regime}_{structure_regime}"
            regime_counts[regime_key] = regime_counts.get(regime_key, 0) + 1

            # Get adjusted parameters for this regime
            adjusted = self._get_regime_adjusted_params(
                symbol=symbol,
                horizon=horizon,
                volatility_regime=volatility_regime,
                trend_regime=trend_regime,
                structure_regime=structure_regime
            )

            adj_k_up = adjusted['k_up']
            adj_k_down = adjusted['k_down']
            adj_max_bars = adjusted['max_bars']

            # Skip if ATR is invalid
            if np.isnan(entry_atr) or entry_atr <= 0:
                labels[i] = 0
                bars_to_hit[i] = adj_max_bars
                continue

            # Define barriers
            upper_barrier = entry_price + adj_k_up * entry_atr
            lower_barrier = entry_price - adj_k_down * entry_atr

            # Track excursions
            max_adverse = 0.0
            max_favorable = 0.0

            # Scan forward
            hit = False
            for j in range(1, min(adj_max_bars + 1, n - i)):
                idx = i + j
                bar_high = high[idx]
                bar_low = low[idx]
                bar_open = open_prices[idx]

                # Update excursions (for long position perspective)
                upside = (bar_high - entry_price) / entry_price
                downside = (bar_low - entry_price) / entry_price

                if upside > max_favorable:
                    max_favorable = upside
                if downside < max_adverse:
                    max_adverse = downside

                # Check barrier hits
                upper_hit = bar_high >= upper_barrier
                lower_hit = bar_low <= lower_barrier

                if upper_hit and lower_hit:
                    # Both barriers hit - use distance from open to determine order
                    dist_to_upper = abs(bar_open - upper_barrier)
                    dist_to_lower = abs(bar_open - lower_barrier)

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

            # Timeout case
            if not hit:
                labels[i] = 0
                bars_to_hit[i] = adj_max_bars
                touch_type[i] = 0

            mae[i] = max_adverse
            mfe[i] = max_favorable

        # Mark last max_bars samples as invalid
        # Use base_max_bars for consistency (max possible window)
        invalid_start = max(0, n - base_max_bars)
        for i in range(invalid_start, n):
            labels[i] = -99
            bars_to_hit[i] = 0
            mae[i] = 0.0
            mfe[i] = 0.0
            touch_type[i] = 0

        # Build result
        result = LabelingResult(
            labels=labels,
            horizon=horizon,
            metadata={
                'bars_to_hit': bars_to_hit,
                'mae': mae,
                'mfe': mfe,
                'touch_type': touch_type
            }
        )

        # Compute quality metrics
        result.quality_metrics = self.get_quality_metrics(result)

        # Add regime distribution to quality metrics
        total_counted = sum(regime_counts.values())
        if total_counted > 0:
            top_regimes = sorted(
                regime_counts.items(),
                key=lambda x: x[1],
                reverse=True
            )[:5]
            result.quality_metrics['regime_distribution'] = {
                k: v / total_counted * 100 for k, v in top_regimes
            }

        # Log statistics
        self._log_label_statistics(result, horizon)
        self._log_regime_distribution(regime_counts)

        return result

    def _log_regime_distribution(self, regime_counts: dict[str, int]) -> None:
        """Log regime distribution used during labeling."""
        total = sum(regime_counts.values())
        if total == 0:
            return

        logger.info("Regime distribution during labeling:")
        # Sort by count descending and show top 5
        sorted_regimes = sorted(
            regime_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]

        for regime, count in sorted_regimes:
            pct = count / total * 100
            logger.info(f"  {regime}: {count:,} ({pct:.1f}%)")

        if len(regime_counts) > 5:
            logger.info(f"  ... and {len(regime_counts) - 5} more regime combinations")


__all__ = ['AdaptiveTripleBarrierLabeler']
