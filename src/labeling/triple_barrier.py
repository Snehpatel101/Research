"""
Triple-Barrier Labeling for PHASE_1B.

Implements the triple-barrier labeling method from Lopez de Prado's
"Advances in Financial Machine Learning" (2018).

Labels:
    -1 = Short (lower barrier hit first)
     0 = Neutral/Timeout (time barrier hit)
    +1 = Long (upper barrier hit first)

The triple-barrier method creates more informative labels than simple
directional returns by incorporating:
    1. Upper profit-taking barrier (scaled by ATR)
    2. Lower stop-loss barrier (scaled by ATR)
    3. Time/horizon barrier (max holding period)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from src.core.constants import (
    DEFAULT_HORIZON,
    DEFAULT_HORIZONS,
    N_CLASSES,
    OHLCV_COLUMNS,
)
from src.core.types import LabelingMethod
from src.core.validation import (
    ValidationError,
    validate_dataframe,
    validate_ohlcv,
)


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
        upper_mult: ATR multiplier for upper (profit) barrier.
            Higher values = wider profit targets = fewer long labels.
        lower_mult: ATR multiplier for lower (stop) barrier.
            Higher values = wider stops = fewer short labels.
        horizon: Maximum holding period in bars (time barrier).
            If neither price barrier is hit within this period,
            the label is neutral (0).
        atr_period: Window for Average True Range calculation.
        use_adaptive_barriers: If True, scale barriers by recent volatility.
        vol_lookback: Lookback period for volatility scaling when adaptive.
        target_long_pct: Target percentage for long labels (for reporting).
        target_short_pct: Target percentage for short labels (for reporting).
        target_neutral_pct: Target percentage for neutral labels (for reporting).

    Example:
        config = TripleBarrierConfig(
            upper_mult=2.0,
            lower_mult=2.0,
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
    use_adaptive_barriers: bool = False
    vol_lookback: int = 60
    target_long_pct: float = 0.33
    target_short_pct: float = 0.33
    target_neutral_pct: float = 0.34

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if self.upper_mult <= 0:
            raise ValidationError(
                "upper_mult must be positive",
                field="upper_mult",
                expected="> 0",
                actual=self.upper_mult,
            )
        if self.lower_mult <= 0:
            raise ValidationError(
                "lower_mult must be positive",
                field="lower_mult",
                expected="> 0",
                actual=self.lower_mult,
            )
        if self.horizon < 1:
            raise ValidationError(
                "horizon must be at least 1",
                field="horizon",
                expected=">= 1",
                actual=self.horizon,
            )
        if self.atr_period < 1:
            raise ValidationError(
                "atr_period must be at least 1",
                field="atr_period",
                expected=">= 1",
                actual=self.atr_period,
            )
        if self.vol_lookback < 1:
            raise ValidationError(
                "vol_lookback must be at least 1",
                field="vol_lookback",
                expected=">= 1",
                actual=self.vol_lookback,
            )

        # Validate target percentages
        total = self.target_long_pct + self.target_short_pct + self.target_neutral_pct
        if abs(total - 1.0) > 0.01:
            raise ValidationError(
                "target percentages must sum to 1.0",
                field="target_*_pct",
                expected=1.0,
                actual=total,
            )

    def to_dict(self) -> Dict[str, float]:
        """Convert config to dictionary."""
        return {
            "upper_mult": self.upper_mult,
            "lower_mult": self.lower_mult,
            "horizon": self.horizon,
            "atr_period": self.atr_period,
            "use_adaptive_barriers": self.use_adaptive_barriers,
            "vol_lookback": self.vol_lookback,
            "target_long_pct": self.target_long_pct,
            "target_short_pct": self.target_short_pct,
            "target_neutral_pct": self.target_neutral_pct,
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "TripleBarrierConfig":
        """Create config from dictionary."""
        return cls(**d)


# =============================================================================
# TRIPLE BARRIER LABELER
# =============================================================================


class TripleBarrierLabeler:
    """
    Triple-barrier labeling implementation (Lopez de Prado method).

    Creates labels based on which of three barriers is hit first:
        - Upper barrier: Entry price + (ATR * upper_mult) -> Long (+1)
        - Lower barrier: Entry price - (ATR * lower_mult) -> Short (-1)
        - Time barrier: horizon bars elapsed -> Neutral (0)

    The ATR (Average True Range) adapts barriers to current volatility,
    making labels more stable across different market regimes.

    Attributes:
        config: TripleBarrierConfig instance with labeling parameters.

    Example:
        >>> config = TripleBarrierConfig(upper_mult=2.0, lower_mult=2.0, horizon=20)
        >>> labeler = TripleBarrierLabeler(config)
        >>> labels = labeler.create_labels(ohlcv_df)
        >>> print(labeler.get_class_distribution(labels))
        {'long': 0.32, 'neutral': 0.36, 'short': 0.32}
    """

    def __init__(self, config: Optional[TripleBarrierConfig] = None) -> None:
        """
        Initialize the triple-barrier labeler.

        Args:
            config: TripleBarrierConfig instance. If None, uses defaults.
        """
        self.config = config or TripleBarrierConfig()
        self._atr_cache: Optional[pd.Series] = None
        self._vol_scaling_cache: Optional[pd.Series] = None

    def compute_atr(self, df: pd.DataFrame) -> pd.Series:
        """
        Compute Average True Range (ATR).

        ATR measures volatility as the moving average of True Range,
        where True Range is the maximum of:
            - High - Low (current bar range)
            - |High - Previous Close| (gap up)
            - |Low - Previous Close| (gap down)

        Args:
            df: OHLCV DataFrame with 'high', 'low', 'close' columns.

        Returns:
            pd.Series: ATR values indexed like input DataFrame.

        Raises:
            ValidationError: If required columns are missing.
        """
        required = ["high", "low", "close"]
        missing = set(required) - set(df.columns)
        if missing:
            raise ValidationError(
                f"Missing columns for ATR calculation: {missing}",
                field="columns",
                expected=required,
                actual=list(df.columns),
            )

        high = df["high"]
        low = df["low"]
        close = df["close"]
        prev_close = close.shift(1)

        # True Range components
        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()

        # True Range is the maximum of the three
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # ATR is the EMA of True Range
        atr = true_range.ewm(span=self.config.atr_period, adjust=False).mean()

        return atr

    def _compute_volatility_scaling(self, df: pd.DataFrame) -> pd.Series:
        """
        Compute volatility scaling factor for adaptive barriers.

        Scales barriers based on recent volatility relative to long-term average.
        Higher recent volatility = wider barriers.

        Args:
            df: OHLCV DataFrame.

        Returns:
            pd.Series: Scaling factors (typically 0.5 to 2.0).
        """
        if not self.config.use_adaptive_barriers:
            return pd.Series(1.0, index=df.index)

        # Use returns volatility for scaling
        returns = df["close"].pct_change()
        recent_vol = returns.rolling(window=self.config.vol_lookback).std()
        long_term_vol = returns.expanding(min_periods=self.config.vol_lookback).std()

        # Scaling factor: recent_vol / long_term_vol, bounded [0.5, 2.0]
        scaling = (recent_vol / long_term_vol).clip(0.5, 2.0)
        scaling = scaling.fillna(1.0)

        return scaling

    def create_labels(self, df: pd.DataFrame) -> pd.Series:
        """
        Create triple-barrier labels for the entire DataFrame.

        For each bar, determines which barrier is hit first within the horizon:
            - Upper barrier (+ATR * upper_mult) hit first -> +1 (Long)
            - Lower barrier (-ATR * lower_mult) hit first -> -1 (Short)
            - Neither hit within horizon bars -> 0 (Neutral)

        Args:
            df: OHLCV DataFrame with DatetimeIndex.
                Must contain: open, high, low, close, volume

        Returns:
            pd.Series: Labels (-1, 0, +1) indexed like input DataFrame.
                Last `horizon` rows will be NaN (no future data).

        Raises:
            ValidationError: If DataFrame is invalid or too small.

        Notes:
            - Uses vectorized operations for performance
            - Handles gaps in data gracefully
            - Labels are forward-looking (use with proper train/test splits)
        """
        # Validate input
        validate_ohlcv(df, context="TripleBarrierLabeler")

        n_rows = len(df)
        min_rows = self.config.atr_period + self.config.horizon + 1
        if n_rows < min_rows:
            raise ValidationError(
                f"DataFrame too small for labeling",
                field="n_rows",
                expected=f">= {min_rows}",
                actual=n_rows,
            )

        # Compute ATR and optional volatility scaling
        atr = self.compute_atr(df)
        vol_scaling = self._compute_volatility_scaling(df)

        # Barrier widths (scaled by volatility if adaptive)
        upper_barrier_width = atr * self.config.upper_mult * vol_scaling
        lower_barrier_width = atr * self.config.lower_mult * vol_scaling

        # Entry price is the close of each bar
        entry_price = df["close"]

        # Compute barriers
        upper_barrier = entry_price + upper_barrier_width
        lower_barrier = entry_price - lower_barrier_width

        # Initialize labels as NaN
        labels = pd.Series(np.nan, index=df.index, name="label")

        # Vectorized labeling using numpy for performance
        close_arr = df["close"].values
        high_arr = df["high"].values
        low_arr = df["low"].values
        upper_arr = upper_barrier.values
        lower_arr = lower_barrier.values
        horizon = self.config.horizon

        # Process each bar (except last `horizon` bars)
        labels_arr = np.full(n_rows, np.nan)

        for i in range(n_rows - horizon):
            entry = close_arr[i]
            upper = upper_arr[i]
            lower = lower_arr[i]

            # Look forward up to horizon bars
            label = 0  # Default: neutral (timeout)

            for j in range(1, horizon + 1):
                future_idx = i + j
                if future_idx >= n_rows:
                    break

                # Check if upper barrier hit (using high price)
                if high_arr[future_idx] >= upper:
                    # Check if lower also hit on same bar (use first touch logic)
                    if low_arr[future_idx] <= lower:
                        # Both barriers touched - use close to determine direction
                        # Or more conservatively, use neutral
                        label = 0
                    else:
                        label = 1  # Long
                    break

                # Check if lower barrier hit (using low price)
                if low_arr[future_idx] <= lower:
                    label = -1  # Short
                    break

            labels_arr[i] = label

        labels = pd.Series(labels_arr, index=df.index, name="label")

        logger.info(
            f"Created {n_rows - horizon} labels: "
            f"Long={int((labels == 1).sum())}, "
            f"Neutral={int((labels == 0).sum())}, "
            f"Short={int((labels == -1).sum())}"
        )

        return labels

    def create_labels_vectorized(self, df: pd.DataFrame) -> pd.Series:
        """
        Create labels using fully vectorized operations (faster for large data).

        This method uses NumPy broadcasting for better performance on large
        datasets, at the cost of higher memory usage.

        Args:
            df: OHLCV DataFrame.

        Returns:
            pd.Series: Labels (-1, 0, +1).
        """
        validate_ohlcv(df, context="TripleBarrierLabeler")

        n_rows = len(df)
        horizon = self.config.horizon

        if n_rows < self.config.atr_period + horizon + 1:
            raise ValidationError(
                f"DataFrame too small for labeling",
                field="n_rows",
                expected=f">= {self.config.atr_period + horizon + 1}",
                actual=n_rows,
            )

        # Compute barriers
        atr = self.compute_atr(df)
        vol_scaling = self._compute_volatility_scaling(df)

        entry_price = df["close"].values
        upper_barrier = entry_price + (atr.values * self.config.upper_mult * vol_scaling.values)
        lower_barrier = entry_price - (atr.values * self.config.lower_mult * vol_scaling.values)

        high_arr = df["high"].values
        low_arr = df["low"].values

        # Pre-allocate labels
        labels = np.full(n_rows, np.nan)

        # Create rolling future windows (memory-intensive but fast)
        # For each position, check future highs and lows
        for i in range(n_rows - horizon):
            future_slice = slice(i + 1, i + horizon + 1)
            future_highs = high_arr[future_slice]
            future_lows = low_arr[future_slice]

            upper = upper_barrier[i]
            lower = lower_barrier[i]

            # Find first touch of each barrier
            upper_touches = np.where(future_highs >= upper)[0]
            lower_touches = np.where(future_lows <= lower)[0]

            first_upper = upper_touches[0] if len(upper_touches) > 0 else horizon + 1
            first_lower = lower_touches[0] if len(lower_touches) > 0 else horizon + 1

            if first_upper < first_lower:
                labels[i] = 1  # Long
            elif first_lower < first_upper:
                labels[i] = -1  # Short
            else:
                labels[i] = 0  # Neutral (timeout or simultaneous touch)

        return pd.Series(labels, index=df.index, name="label")

    def get_class_distribution(self, labels: pd.Series) -> Dict[str, float]:
        """
        Compute class distribution from labels.

        Args:
            labels: Series of labels (-1, 0, +1).

        Returns:
            dict: Distribution with keys 'long', 'neutral', 'short'
                  and percentage values (0.0 to 1.0).

        Example:
            >>> dist = labeler.get_class_distribution(labels)
            >>> print(dist)
            {'long': 0.33, 'neutral': 0.34, 'short': 0.33, 'n_samples': 10000}
        """
        # Filter out NaN labels
        valid_labels = labels.dropna()
        n_total = len(valid_labels)

        if n_total == 0:
            return {
                "long": 0.0,
                "neutral": 0.0,
                "short": 0.0,
                "n_samples": 0,
            }

        n_long = int((valid_labels == 1).sum())
        n_neutral = int((valid_labels == 0).sum())
        n_short = int((valid_labels == -1).sum())

        return {
            "long": n_long / n_total,
            "neutral": n_neutral / n_total,
            "short": n_short / n_total,
            "n_samples": n_total,
            "n_long": n_long,
            "n_neutral": n_neutral,
            "n_short": n_short,
        }

    def get_label_quality_metrics(
        self, labels: pd.Series, df: pd.DataFrame
    ) -> Dict[str, float]:
        """
        Compute quality metrics for generated labels.

        Metrics include:
            - Class balance (entropy-based)
            - Average return per label class
            - Label persistence (autocorrelation)

        Args:
            labels: Generated labels.
            df: Original OHLCV DataFrame.

        Returns:
            dict: Quality metrics.
        """
        valid_mask = ~labels.isna()
        valid_labels = labels[valid_mask]
        valid_df = df.loc[valid_mask]

        if len(valid_labels) == 0:
            return {"balance_score": 0.0, "avg_return_long": 0.0, "avg_return_short": 0.0}

        # Class balance (normalized entropy)
        dist = self.get_class_distribution(labels)
        probs = [dist["long"], dist["neutral"], dist["short"]]
        probs = [p for p in probs if p > 0]

        if len(probs) > 1:
            entropy = -sum(p * np.log(p) for p in probs)
            max_entropy = np.log(N_CLASSES)
            balance_score = entropy / max_entropy
        else:
            balance_score = 0.0

        # Average forward returns per class
        returns = df["close"].pct_change(self.config.horizon).shift(-self.config.horizon)
        returns = returns.loc[valid_mask]

        avg_return_long = returns[valid_labels == 1].mean() if (valid_labels == 1).any() else 0.0
        avg_return_short = returns[valid_labels == -1].mean() if (valid_labels == -1).any() else 0.0
        avg_return_neutral = returns[valid_labels == 0].mean() if (valid_labels == 0).any() else 0.0

        # Label persistence (how often label matches previous)
        label_persistence = (valid_labels == valid_labels.shift(1)).mean()

        return {
            "balance_score": float(balance_score),
            "avg_return_long": float(avg_return_long) if not np.isnan(avg_return_long) else 0.0,
            "avg_return_short": float(avg_return_short) if not np.isnan(avg_return_short) else 0.0,
            "avg_return_neutral": float(avg_return_neutral) if not np.isnan(avg_return_neutral) else 0.0,
            "label_persistence": float(label_persistence),
            "class_distribution": dist,
        }


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "TripleBarrierConfig",
    "TripleBarrierLabeler",
]
