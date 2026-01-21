"""
Threshold-Based Labeling Strategy.

Labels based on whether price hits a percentage threshold before the opposite.
Similar to triple-barrier but uses percentage-based thresholds instead of ATR.

Label mapping:
- +1: Upper threshold (+X%) hit before lower threshold (-Y%)
- -1: Lower threshold (-Y%) hit before upper threshold (+X%)
- 0: Neither threshold hit within max_bars (timeout)
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
def threshold_labeling_numba(
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    open_prices: np.ndarray,
    pct_up: float,
    pct_down: float,
    max_bars: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Numba-optimized percentage threshold labeling.

    Parameters
    ----------
    close : np.ndarray
        Array of close prices
    high : np.ndarray
        Array of high prices
    low : np.ndarray
        Array of low prices
    open_prices : np.ndarray
        Array of open prices (for simultaneous hit resolution)
    pct_up : float
        Upper threshold as decimal (e.g., 0.01 for 1%)
    pct_down : float
        Lower threshold as decimal (e.g., 0.01 for 1%)
    max_bars : int
        Maximum bars before timeout

    Returns
    -------
    tuple
        labels : +1 (upper hit), -1 (lower hit), 0 (timeout), -99 (invalid)
        bars_to_hit : number of bars until threshold was hit
        max_gain : maximum upside reached (as decimal)
        max_loss : maximum downside reached (as decimal, negative)
    """
    n = len(close)
    labels = np.zeros(n, dtype=np.int8)
    bars_to_hit = np.zeros(n, dtype=np.int32)
    max_gain = np.zeros(n, dtype=np.float32)
    max_loss = np.zeros(n, dtype=np.float32)

    for i in range(n - 1):
        entry_price = close[i]

        if entry_price <= 0 or np.isnan(entry_price):
            labels[i] = 0
            bars_to_hit[i] = max_bars
            continue

        # Define thresholds
        upper_threshold = entry_price * (1 + pct_up)
        lower_threshold = entry_price * (1 - pct_down)

        # Track excursions
        running_max_gain = 0.0
        running_max_loss = 0.0

        # Scan forward
        hit = False
        for j in range(1, min(max_bars + 1, n - i)):
            idx = i + j
            bar_high = high[idx]
            bar_low = low[idx]
            bar_open = open_prices[idx]

            # Update excursions
            upside = (bar_high - entry_price) / entry_price
            downside = (bar_low - entry_price) / entry_price

            if upside > running_max_gain:
                running_max_gain = upside
            if downside < running_max_loss:
                running_max_loss = downside

            # Check threshold hits
            upper_hit = bar_high >= upper_threshold
            lower_hit = bar_low <= lower_threshold

            if upper_hit and lower_hit:
                # Both thresholds hit on same bar
                dist_to_upper = abs(bar_open - upper_threshold)
                dist_to_lower = abs(bar_open - lower_threshold)

                if dist_to_upper <= dist_to_lower:
                    labels[i] = 1
                else:
                    labels[i] = -1
                bars_to_hit[i] = j
                hit = True
                break
            elif upper_hit:
                labels[i] = 1
                bars_to_hit[i] = j
                hit = True
                break
            elif lower_hit:
                labels[i] = -1
                bars_to_hit[i] = j
                hit = True
                break

        # Timeout case
        if not hit:
            labels[i] = 0
            bars_to_hit[i] = max_bars

        max_gain[i] = running_max_gain
        max_loss[i] = running_max_loss

    # Mark last max_bars samples as invalid
    for i in range(max(0, n - max_bars), n):
        labels[i] = -99
        bars_to_hit[i] = 0
        max_gain[i] = 0.0
        max_loss[i] = 0.0

    return labels, bars_to_hit, max_gain, max_loss


class ThresholdLabeler(LabelingStrategy):
    """
    Percentage threshold labeling strategy.

    Labels each bar based on which percentage threshold is hit first:
    - Upper threshold (+X%): label = +1
    - Lower threshold (-Y%): label = -1
    - Timeout (max_bars exceeded): label = 0

    This is similar to triple-barrier but uses fixed percentage thresholds
    instead of ATR-based dynamic barriers, making it simpler and more
    interpretable.

    Parameters
    ----------
    pct_up : float
        Upper threshold as percentage (e.g., 0.01 for 1%)
    pct_down : float
        Lower threshold as percentage (e.g., 0.01 for 1%)
    max_bars : int
        Maximum bars before timeout (default: 20)
    """

    def __init__(self, pct_up: float = 0.01, pct_down: float = 0.01, max_bars: int = 20):
        if pct_up <= 0:
            raise ValueError(f"pct_up must be positive, got {pct_up}")
        if pct_down <= 0:
            raise ValueError(f"pct_down must be positive, got {pct_down}")
        if max_bars <= 0:
            raise ValueError(f"max_bars must be positive, got {max_bars}")

        self._pct_up = pct_up
        self._pct_down = pct_down
        self._max_bars = max_bars

    @property
    def labeling_type(self) -> LabelingType:
        """Return the type of this labeling strategy."""
        return LabelingType.THRESHOLD

    @property
    def required_columns(self) -> list[str]:
        """Return list of required DataFrame columns."""
        return ["close", "high", "low", "open"]

    def compute_labels(
        self,
        df: pd.DataFrame,
        horizon: int,
        pct_up: float | None = None,
        pct_down: float | None = None,
        max_bars: int | None = None,
        **kwargs: Any,
    ) -> LabelingResult:
        """
        Compute threshold labels for the given horizon.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with OHLC prices
        horizon : int
            Horizon identifier (used for naming, actual lookforward is max_bars)
        pct_up : float, optional
            Override for upper threshold
        pct_down : float, optional
            Override for lower threshold
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
        pct_up = pct_up if pct_up is not None else self._pct_up
        pct_down = pct_down if pct_down is not None else self._pct_down
        max_bars = max_bars if max_bars is not None else self._max_bars

        # Validate resolved parameters
        if pct_up <= 0:
            raise ValueError(f"pct_up must be positive, got {pct_up}")
        if pct_down <= 0:
            raise ValueError(f"pct_down must be positive, got {pct_down}")
        if max_bars <= 0:
            raise ValueError(f"max_bars must be positive, got {max_bars}")

        logger.info(f"Computing threshold labels for horizon {horizon}")
        logger.info(
            f"  pct_up={pct_up*100:.2f}%, pct_down={pct_down*100:.2f}%, max_bars={max_bars}"
        )

        # Extract arrays
        close = df["close"].values
        high = df["high"].values
        low = df["low"].values
        open_prices = df["open"].values

        # Apply numba function
        labels, bars_to_hit, max_gain, max_loss = threshold_labeling_numba(
            close, high, low, open_prices, pct_up, pct_down, max_bars
        )

        # Build result
        result = LabelingResult(
            labels=labels,
            horizon=horizon,
            metadata={"bars_to_hit": bars_to_hit, "max_gain": max_gain, "max_loss": max_loss},
        )

        # Compute quality metrics
        result.quality_metrics = self.get_quality_metrics(result)

        # Log statistics
        self._log_label_statistics(result, horizon, pct_up, pct_down)

        return result

    def _log_label_statistics(
        self, result: LabelingResult, horizon: int, pct_up: float, pct_down: float
    ) -> None:
        """Log label distribution statistics."""
        labels = result.labels
        valid_mask = labels != -99
        valid_labels = labels[valid_mask]

        if len(valid_labels) == 0:
            logger.warning(f"No valid labels for horizon {horizon}")
            return

        label_counts = pd.Series(valid_labels).value_counts().sort_index()
        total = len(valid_labels)

        logger.info(
            f"Label distribution for horizon {horizon} "
            f"(+{pct_up*100:.2f}%/-{pct_down*100:.2f}%):"
        )
        for label_val in [-1, 0, 1]:
            count = label_counts.get(label_val, 0)
            pct = count / total * 100
            label_name = {-1: "Lower Hit", 0: "Timeout", 1: "Upper Hit"}[label_val]
            logger.info(f"  {label_name:12s}: {count:6d} ({pct:5.1f}%)")

        # Log average bars to hit
        bars_to_hit = result.metadata.get("bars_to_hit", np.array([]))
        if len(bars_to_hit) > 0:
            valid_bars = bars_to_hit[valid_mask & (bars_to_hit > 0)]
            if len(valid_bars) > 0:
                avg_bars = valid_bars.mean()
                logger.info(f"  Avg bars to hit: {avg_bars:.1f}")

    def get_quality_metrics(self, result: LabelingResult) -> dict[str, float]:
        """
        Compute quality metrics for threshold labeling.

        Includes gain/loss analysis.
        """
        metrics = super().get_quality_metrics(result)

        labels = result.labels
        valid_mask = labels != -99

        max_gain = result.metadata.get("max_gain", np.array([]))
        max_loss = result.metadata.get("max_loss", np.array([]))
        bars_to_hit = result.metadata.get("bars_to_hit", np.array([]))

        if len(max_gain) > 0 and len(max_loss) > 0:
            valid_gain = max_gain[valid_mask]
            valid_loss = max_loss[valid_mask]

            if len(valid_gain) > 0:
                metrics["avg_max_gain"] = float(np.mean(valid_gain))
                metrics["avg_max_loss"] = float(np.mean(valid_loss))

                # Risk-reward ratio
                avg_loss_abs = (
                    abs(metrics["avg_max_loss"]) if metrics["avg_max_loss"] != 0 else 1e-6
                )
                metrics["gain_loss_ratio"] = metrics["avg_max_gain"] / avg_loss_abs

        if len(bars_to_hit) > 0:
            valid_bars = bars_to_hit[valid_mask]
            if len(valid_bars) > 0:
                non_zero_bars = valid_bars[valid_bars > 0]
                if len(non_zero_bars) > 0:
                    metrics["avg_bars_to_hit"] = float(np.mean(non_zero_bars))

        return metrics
