"""
Triple-Barrier Labeling Strategy.

Implements the Lopez de Prado triple-barrier method with:
- ATR-based dynamic barriers
- Asymmetric barriers for bias correction
- Numba optimization for performance

CRITICAL FIX (2024-12): ASYMMETRIC BARRIERS TO CORRECT LONG BIAS
Previous symmetric barriers (k_up = k_down) in a historically bullish market
produced 87-91% long signals. New asymmetric barriers (k_up > k_down) make the
lower barrier easier to hit, targeting ~50/50 long/short distribution.
"""

import logging
from typing import Any

import numba as nb
import numpy as np
import pandas as pd

from .base import LabelingResult, LabelingStrategy, LabelingType

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


@nb.jit(nopython=True, cache=True)
def triple_barrier_numba(
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    open_prices: np.ndarray,
    atr: np.ndarray,
    k_up: float,
    k_down: float,
    max_bars: int
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
    open_prices : np.ndarray
        Array of open prices (used to resolve simultaneous barrier hits)
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

    Note
    ----
    When both upper and lower barriers are hit on the same bar, we use distance
    from the bar's open price to determine which barrier was likely hit first.
    This follows Lopez de Prado's methodology and eliminates long bias from
    always checking upper barrier first.
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

        # Skip if ATR is invalid
        if np.isnan(entry_atr) or entry_atr <= 0:
            labels[i] = 0
            bars_to_hit[i] = max_bars
            continue

        # Define barriers
        upper_barrier = entry_price + k_up * entry_atr
        lower_barrier = entry_price - k_down * entry_atr

        # Track excursions
        max_adverse = 0.0
        max_favorable = 0.0

        # Scan forward
        hit = False
        for j in range(1, min(max_bars + 1, n - i)):
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
                # BOTH barriers hit on same bar - determine which was hit first
                # Use distance from bar open as proxy for which barrier hit first
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
            bars_to_hit[i] = max_bars
            touch_type[i] = 0

        mae[i] = max_adverse
        mfe[i] = max_favorable

    # CRITICAL FIX (2025-12-21): Last max_bars samples should be excluded
    # to prevent edge case leakage.
    for i in range(max(0, n - max_bars), n):
        labels[i] = -99  # Sentinel: invalid label (to be removed)
        bars_to_hit[i] = 0
        mae[i] = 0.0
        mfe[i] = 0.0
        touch_type[i] = 0

    return labels, bars_to_hit, mae, mfe, touch_type


class TripleBarrierLabeler(LabelingStrategy):
    """
    Triple-barrier labeling strategy with ATR-based dynamic barriers.

    This strategy labels each bar based on which barrier is hit first:
    - Upper barrier (profit target): label = +1
    - Lower barrier (stop loss): label = -1
    - Timeout (max_bars exceeded): label = 0

    Parameters
    ----------
    k_up : float, optional
        Multiplier for upper barrier (default from config)
    k_down : float, optional
        Multiplier for lower barrier (default from config)
    max_bars : int, optional
        Maximum bars before timeout (default from config)
    atr_column : str
        Column name for ATR values (default: 'atr_14')
    """

    def __init__(
        self,
        k_up: float | None = None,
        k_down: float | None = None,
        max_bars: int | None = None,
        atr_column: str = 'atr_14'
    ):
        self._k_up = k_up
        self._k_down = k_down
        self._max_bars = max_bars
        self._atr_column = atr_column

    @property
    def labeling_type(self) -> LabelingType:
        """Return the type of this labeling strategy."""
        return LabelingType.TRIPLE_BARRIER

    @property
    def required_columns(self) -> list[str]:
        """Return list of required DataFrame columns."""
        return ['close', 'high', 'low', 'open', self._atr_column]

    def _get_default_params(self, horizon: int) -> dict[str, Any]:
        """Get default parameters for a given horizon."""
        # Import here to avoid circular imports
        from src.phase1.config import BARRIER_PARAMS_DEFAULT

        if horizon in BARRIER_PARAMS_DEFAULT:
            return BARRIER_PARAMS_DEFAULT[horizon]
        return {
            'k_up': 1.0,
            'k_down': 1.0,
            'max_bars': max(horizon * 3, 10)
        }

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
        # Validate inputs
        self.validate_inputs(df)

        if not isinstance(horizon, int) or horizon <= 0:
            raise ValueError(f"horizon must be a positive integer, got {horizon}")

        # Resolve parameters
        defaults = self._get_default_params(horizon)
        k_up = k_up or self._k_up or defaults['k_up']
        k_down = k_down or self._k_down or defaults['k_down']
        max_bars = max_bars or self._max_bars or defaults['max_bars']

        # Validate resolved parameters
        if k_up <= 0:
            raise ValueError(f"k_up must be positive, got {k_up}")
        if k_down <= 0:
            raise ValueError(f"k_down must be positive, got {k_down}")
        if max_bars <= 0:
            raise ValueError(f"max_bars must be positive, got {max_bars}")

        logger.info(f"Computing triple-barrier labels for horizon {horizon}")
        logger.info(f"  k_up={k_up:.3f}, k_down={k_down:.3f}, max_bars={max_bars}")

        # Extract arrays
        close = df['close'].values
        high = df['high'].values
        low = df['low'].values
        open_prices = df['open'].values
        atr = df[self._atr_column].values

        # Apply numba function
        labels, bars_to_hit, mae, mfe, touch_type = triple_barrier_numba(
            close, high, low, open_prices, atr, k_up, k_down, max_bars
        )

        # Build result
        result = LabelingResult(
            labels=labels,
            horizon=horizon,
            metadata={
                'bars_to_hit': bars_to_hit,
                'mae': mae,
                'mfe': mfe,
                'touch_type': touch_type,
                'max_bars': max_bars,
                'k_up': k_up,
                'k_down': k_down
            }
        )

        # Compute quality metrics
        result.quality_metrics = self.get_quality_metrics(result)

        # Log statistics
        self._log_label_statistics(result, horizon)

        return result

    def _log_label_statistics(self, result: LabelingResult, horizon: int) -> None:
        """Log label distribution statistics."""
        labels = result.labels
        valid_mask = labels != -99
        valid_labels = labels[valid_mask]
        total_samples = len(labels)

        if len(valid_labels) == 0:
            logger.warning(f"No valid labels for horizon {horizon}")
            return

        label_counts = pd.Series(valid_labels).value_counts().sort_index()
        total = len(valid_labels)

        logger.info(f"Label distribution for horizon {horizon}:")
        for label_val in [-1, 0, 1]:
            count = label_counts.get(label_val, 0)
            pct = count / total * 100
            label_name = {-1: "Short/Loss", 0: "Neutral/Timeout", 1: "Long/Win"}[label_val]
            logger.info(f"  {label_name:20s}: {count:6d} ({pct:5.1f}%)")

        # Calculate average bars to hit
        bars_to_hit = result.metadata.get('bars_to_hit', np.array([]))
        if len(bars_to_hit) > 0:
            valid_bars = bars_to_hit[valid_mask & (bars_to_hit > 0)]
            if len(valid_bars) > 0:
                avg_bars = valid_bars.mean()
                logger.info(f"  Avg bars to hit: {avg_bars:.1f}")

        # Log edge case exclusions with clear explanation
        invalid_count = (~valid_mask).sum()
        if invalid_count > 0:
            invalid_pct = invalid_count / total_samples * 100
            # Get max_bars from the result metadata or reconstruct from defaults
            max_bars = result.metadata.get('max_bars', horizon * 3)
            logger.info(
                f"  Edge case exclusions: {invalid_count} samples ({invalid_pct:.1f}%) "
                f"marked invalid at dataset end"
            )
            logger.info(
                f"  Reason: Insufficient horizon (last {max_bars} bars cannot look forward {max_bars} bars "
                f"to compute triple-barrier outcome)"
            )

    def get_quality_metrics(self, result: LabelingResult) -> dict[str, float]:
        """
        Compute quality metrics specific to triple-barrier labeling.

        Includes MAE/MFE analysis for label quality assessment.
        """
        metrics = super().get_quality_metrics(result)

        # Add triple-barrier specific metrics
        labels = result.labels
        valid_mask = labels != -99

        mae = result.metadata.get('mae', np.array([]))
        mfe = result.metadata.get('mfe', np.array([]))
        bars_to_hit = result.metadata.get('bars_to_hit', np.array([]))

        if len(mae) > 0 and len(mfe) > 0:
            valid_mae = mae[valid_mask]
            valid_mfe = mfe[valid_mask]

            if len(valid_mae) > 0:
                metrics['avg_mae'] = float(np.mean(valid_mae))
                metrics['avg_mfe'] = float(np.mean(valid_mfe))

                # Risk-reward ratio (MFE/|MAE|)
                avg_mae_abs = abs(metrics['avg_mae']) if metrics['avg_mae'] != 0 else 1e-6
                metrics['mfe_mae_ratio'] = abs(metrics['avg_mfe']) / avg_mae_abs

        if len(bars_to_hit) > 0:
            valid_bars = bars_to_hit[valid_mask]
            if len(valid_bars) > 0 and valid_bars.sum() > 0:
                non_zero_bars = valid_bars[valid_bars > 0]
                if len(non_zero_bars) > 0:
                    metrics['avg_bars_to_hit'] = float(np.mean(non_zero_bars))

        return metrics
