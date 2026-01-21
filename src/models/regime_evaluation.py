"""
Regime-conditional evaluation for model performance analysis.

This module provides tools for evaluating model performance across different
market regimes, including volatility regimes, trend regimes, and time-of-day
sessions. This helps identify model strengths and weaknesses in specific
market conditions.

Example:
    >>> from src.models.regime_evaluation import RegimeEvaluator
    >>> evaluator = RegimeEvaluator()
    >>> result = evaluator.evaluate(
    ...     y_true=y_test,
    ...     y_pred=predictions,
    ...     y_proba=probabilities,
    ...     prices=price_df,
    ...     timestamps=test_timestamps,
    ... )
    >>> print(result.to_dataframe())
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# =============================================================================
# Regime Enumerations
# =============================================================================


class VolatilityRegime(Enum):
    """Volatility regime classification based on rolling volatility percentiles."""

    LOW = "low"  # Bottom 25% of rolling volatility
    NORMAL = "normal"  # Middle 50% (25th-75th percentile)
    HIGH = "high"  # Top 25% of rolling volatility


class TrendRegime(Enum):
    """Trend regime classification based on ADX and directional indicators."""

    TRENDING_UP = "trending_up"  # ADX > 25 and +DI > -DI
    TRENDING_DOWN = "trending_down"  # ADX > 25 and -DI > +DI
    MEAN_REVERTING = "mean_reverting"  # ADX < 20 (range-bound)


class TimeOfDay(Enum):
    """Trading session classification based on Eastern Time."""

    ASIA = "asia"  # 18:00-02:00 ET (Sunday-Friday)
    LONDON = "london"  # 03:00-08:00 ET
    US_OPEN = "us_open"  # 09:30-12:00 ET (most volatile)
    US_MIDDAY = "us_midday"  # 12:00-15:00 ET
    US_CLOSE = "us_close"  # 15:00-16:00 ET


# =============================================================================
# Regime Metrics
# =============================================================================


@dataclass
class RegimeMetrics:
    """Performance metrics for a specific market regime.

    Attributes:
        regime_name: Name of the regime (e.g., "high", "trending_up", "us_open")
        n_samples: Number of samples in this regime
        accuracy: Classification accuracy in this regime
        f1_score: Macro F1 score in this regime
        sharpe_ratio: Sharpe ratio of returns in this regime
        win_rate: Win rate for position trades in this regime
        avg_return: Average return per trade in this regime
        max_drawdown: Maximum drawdown in this regime
        precision: Precision score in this regime
        recall: Recall score in this regime
    """

    regime_name: str
    n_samples: int
    accuracy: float
    f1_score: float
    sharpe_ratio: float
    win_rate: float
    avg_return: float
    max_drawdown: float
    precision: float = 0.0
    recall: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "regime_name": self.regime_name,
            "n_samples": self.n_samples,
            "accuracy": self.accuracy,
            "f1_score": self.f1_score,
            "sharpe_ratio": self.sharpe_ratio,
            "win_rate": self.win_rate,
            "avg_return": self.avg_return,
            "max_drawdown": self.max_drawdown,
            "precision": self.precision,
            "recall": self.recall,
        }


@dataclass
class RegimeEvaluationResult:
    """Complete regime-conditional evaluation results.

    Attributes:
        overall_metrics: Overall model performance metrics
        volatility_breakdown: Performance by volatility regime
        trend_breakdown: Performance by trend regime
        time_of_day_breakdown: Performance by trading session
        regime_correlations: Correlation between regime and performance
    """

    overall_metrics: dict[str, Any]
    volatility_breakdown: dict[str, RegimeMetrics]
    trend_breakdown: dict[str, RegimeMetrics]
    time_of_day_breakdown: dict[str, RegimeMetrics]
    regime_correlations: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to nested dictionary for serialization."""
        return {
            "overall_metrics": self.overall_metrics,
            "volatility_breakdown": {
                k: v.to_dict() for k, v in self.volatility_breakdown.items()
            },
            "trend_breakdown": {
                k: v.to_dict() for k, v in self.trend_breakdown.items()
            },
            "time_of_day_breakdown": {
                k: v.to_dict() for k, v in self.time_of_day_breakdown.items()
            },
            "regime_correlations": self.regime_correlations,
        }

    def to_dataframe(self) -> pd.DataFrame:
        """Convert regime breakdowns to comparison DataFrame.

        Returns:
            DataFrame with regime metrics for easy comparison
        """
        rows = []

        # Add volatility regimes
        for regime, metrics in self.volatility_breakdown.items():
            row = metrics.to_dict()
            row["regime_type"] = "volatility"
            rows.append(row)

        # Add trend regimes
        for regime, metrics in self.trend_breakdown.items():
            row = metrics.to_dict()
            row["regime_type"] = "trend"
            rows.append(row)

        # Add time-of-day regimes
        for regime, metrics in self.time_of_day_breakdown.items():
            row = metrics.to_dict()
            row["regime_type"] = "time_of_day"
            rows.append(row)

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        # Reorder columns for readability
        cols = [
            "regime_type",
            "regime_name",
            "n_samples",
            "accuracy",
            "f1_score",
            "sharpe_ratio",
            "win_rate",
            "avg_return",
            "max_drawdown",
        ]
        available_cols = [c for c in cols if c in df.columns]
        other_cols = [c for c in df.columns if c not in available_cols]
        return df[available_cols + other_cols]

    def get_weak_regimes(self, accuracy_threshold: float = 0.5) -> list[str]:
        """Identify regimes where model underperforms.

        Args:
            accuracy_threshold: Accuracy below this is considered weak

        Returns:
            List of regime names where model underperforms
        """
        weak = []

        for regime, metrics in self.volatility_breakdown.items():
            if metrics.accuracy < accuracy_threshold:
                weak.append(f"volatility:{regime}")

        for regime, metrics in self.trend_breakdown.items():
            if metrics.accuracy < accuracy_threshold:
                weak.append(f"trend:{regime}")

        for regime, metrics in self.time_of_day_breakdown.items():
            if metrics.accuracy < accuracy_threshold:
                weak.append(f"time_of_day:{regime}")

        return weak

    def plot_regime_comparison(
        self,
        save_path: Optional[Path] = None,
        figsize: tuple[int, int] = (14, 10),
    ) -> Any:
        """Visualize performance across regimes.

        Args:
            save_path: Optional path to save the figure
            figsize: Figure size (width, height)

        Returns:
            matplotlib figure object
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib not available for plotting")
            return None

        df = self.to_dataframe()
        if df.empty:
            logger.warning("No regime data to plot")
            return None

        fig, axes = plt.subplots(2, 2, figsize=figsize)

        # 1. Accuracy by regime
        ax1 = axes[0, 0]
        colors = {"volatility": "#2ecc71", "trend": "#3498db", "time_of_day": "#9b59b6"}
        for regime_type in df["regime_type"].unique():
            subset = df[df["regime_type"] == regime_type]
            ax1.barh(
                subset["regime_name"],
                subset["accuracy"],
                label=regime_type,
                color=colors.get(regime_type, "#95a5a6"),
                alpha=0.8,
            )
        ax1.set_xlabel("Accuracy")
        ax1.set_title("Accuracy by Regime")
        ax1.legend()
        ax1.axvline(x=0.5, color="red", linestyle="--", alpha=0.5, label="Random")

        # 2. Sharpe ratio by regime
        ax2 = axes[0, 1]
        for regime_type in df["regime_type"].unique():
            subset = df[df["regime_type"] == regime_type]
            ax2.barh(
                subset["regime_name"],
                subset["sharpe_ratio"],
                label=regime_type,
                color=colors.get(regime_type, "#95a5a6"),
                alpha=0.8,
            )
        ax2.set_xlabel("Sharpe Ratio")
        ax2.set_title("Sharpe Ratio by Regime")
        ax2.axvline(x=0, color="red", linestyle="--", alpha=0.5)

        # 3. Sample distribution
        ax3 = axes[1, 0]
        for regime_type in df["regime_type"].unique():
            subset = df[df["regime_type"] == regime_type]
            ax3.barh(
                subset["regime_name"],
                subset["n_samples"],
                label=regime_type,
                color=colors.get(regime_type, "#95a5a6"),
                alpha=0.8,
            )
        ax3.set_xlabel("Number of Samples")
        ax3.set_title("Sample Distribution by Regime")

        # 4. Win rate by regime
        ax4 = axes[1, 1]
        for regime_type in df["regime_type"].unique():
            subset = df[df["regime_type"] == regime_type]
            ax4.barh(
                subset["regime_name"],
                subset["win_rate"],
                label=regime_type,
                color=colors.get(regime_type, "#95a5a6"),
                alpha=0.8,
            )
        ax4.set_xlabel("Win Rate")
        ax4.set_title("Win Rate by Regime")
        ax4.axvline(x=0.5, color="red", linestyle="--", alpha=0.5)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info(f"Saved regime comparison plot to {save_path}")

        return fig


# =============================================================================
# Regime Classifier
# =============================================================================


class RegimeClassifier:
    """Classify market regimes from price data.

    This classifier detects three types of regimes:
    1. Volatility: Low/Normal/High based on rolling volatility percentiles
    2. Trend: Trending Up/Down/Mean Reverting based on ADX and DI
    3. Time of Day: Trading session based on timestamp

    The classifier reuses existing regime detection logic from the phase1 pipeline
    where available.

    Attributes:
        vol_window: Window for rolling volatility calculation
        trend_window: Window for ADX calculation
        vol_percentiles: Tuple of (low, high) percentiles for volatility classification
        adx_trending_threshold: ADX above this indicates trending
        adx_mean_reverting_threshold: ADX below this indicates mean reversion
    """

    def __init__(
        self,
        vol_window: int = 20,
        trend_window: int = 14,
        vol_percentiles: tuple[float, float] = (25.0, 75.0),
        adx_trending_threshold: float = 25.0,
        adx_mean_reverting_threshold: float = 20.0,
    ):
        """Initialize regime classifier.

        Args:
            vol_window: Rolling window for volatility calculation
            trend_window: Period for ADX calculation
            vol_percentiles: (low, high) percentiles for volatility buckets
            adx_trending_threshold: ADX value indicating strong trend
            adx_mean_reverting_threshold: ADX below this indicates range-bound
        """
        if vol_window < 2:
            raise ValueError(f"vol_window must be >= 2, got {vol_window}")
        if trend_window < 2:
            raise ValueError(f"trend_window must be >= 2, got {trend_window}")
        if not (0 < vol_percentiles[0] < vol_percentiles[1] < 100):
            raise ValueError(
                f"vol_percentiles must be (low, high) with 0 < low < high < 100, "
                f"got {vol_percentiles}"
            )

        self.vol_window = vol_window
        self.trend_window = trend_window
        self.vol_percentiles = vol_percentiles
        self.adx_trending_threshold = adx_trending_threshold
        self.adx_mean_reverting_threshold = adx_mean_reverting_threshold

    def classify_volatility(
        self,
        returns: pd.Series,
        lookback: int = 100,
    ) -> pd.Series:
        """Classify volatility regime for each timestamp.

        Uses rolling standard deviation of returns and compares against
        rolling percentile thresholds.

        Args:
            returns: Series of returns (e.g., log returns or simple returns)
            lookback: Lookback window for percentile calculation

        Returns:
            Series with VolatilityRegime values as strings
        """
        if len(returns) < self.vol_window:
            logger.warning(
                f"Insufficient data for volatility classification: "
                f"{len(returns)} < {self.vol_window}"
            )
            return pd.Series(
                VolatilityRegime.NORMAL.value, index=returns.index, dtype="object"
            )

        # Calculate rolling volatility
        rolling_vol = returns.rolling(window=self.vol_window, min_periods=2).std()

        # Calculate rolling percentile thresholds
        low_threshold = rolling_vol.rolling(
            window=lookback, min_periods=self.vol_window
        ).quantile(self.vol_percentiles[0] / 100.0)

        high_threshold = rolling_vol.rolling(
            window=lookback, min_periods=self.vol_window
        ).quantile(self.vol_percentiles[1] / 100.0)

        # Classify regime
        regimes = pd.Series(
            VolatilityRegime.NORMAL.value, index=returns.index, dtype="object"
        )
        regimes[rolling_vol < low_threshold] = VolatilityRegime.LOW.value
        regimes[rolling_vol > high_threshold] = VolatilityRegime.HIGH.value

        # Handle NaN (warmup period)
        regimes[rolling_vol.isna()] = np.nan

        return regimes

    def classify_trend(
        self,
        df: pd.DataFrame,
        adx_column: Optional[str] = None,
        plus_di_column: Optional[str] = None,
        minus_di_column: Optional[str] = None,
    ) -> pd.Series:
        """Classify trend regime using ADX and directional indicators.

        Args:
            df: DataFrame with OHLC data
            adx_column: Pre-computed ADX column name (optional)
            plus_di_column: Pre-computed +DI column name (optional)
            minus_di_column: Pre-computed -DI column name (optional)

        Returns:
            Series with TrendRegime values as strings
        """
        # Try to use pre-computed values first
        if adx_column and adx_column in df.columns:
            adx = df[adx_column].values
            plus_di = (
                df[plus_di_column].values
                if plus_di_column and plus_di_column in df.columns
                else None
            )
            minus_di = (
                df[minus_di_column].values
                if minus_di_column and minus_di_column in df.columns
                else None
            )
        else:
            # Compute ADX from OHLC
            adx, plus_di, minus_di = self._calculate_adx(
                df["high"].values, df["low"].values, df["close"].values
            )

        # Need directional indicators for trend direction
        if plus_di is None or minus_di is None:
            # Calculate if not provided
            _, plus_di, minus_di = self._calculate_adx(
                df["high"].values, df["low"].values, df["close"].values
            )

        adx_series = pd.Series(adx, index=df.index)
        plus_di_series = pd.Series(plus_di, index=df.index)
        minus_di_series = pd.Series(minus_di, index=df.index)

        # Classify regime
        regimes = pd.Series(
            TrendRegime.MEAN_REVERTING.value, index=df.index, dtype="object"
        )

        # Strong trend conditions
        is_trending = adx_series > self.adx_trending_threshold
        is_uptrend = plus_di_series > minus_di_series
        is_downtrend = minus_di_series > plus_di_series

        # Mean reverting (weak ADX)
        is_mean_reverting = adx_series < self.adx_mean_reverting_threshold

        # Apply classifications
        regimes[is_trending & is_uptrend] = TrendRegime.TRENDING_UP.value
        regimes[is_trending & is_downtrend] = TrendRegime.TRENDING_DOWN.value
        regimes[is_mean_reverting] = TrendRegime.MEAN_REVERTING.value

        # Handle NaN from warmup
        regimes[adx_series.isna()] = np.nan

        return regimes

    def classify_time_of_day(
        self,
        timestamps: pd.DatetimeIndex,
        timezone: str = "US/Eastern",
    ) -> pd.Series:
        """Classify trading session for each timestamp.

        Sessions (in Eastern Time):
        - Asia: 18:00-02:00 ET
        - London: 03:00-08:00 ET
        - US Open: 09:30-12:00 ET (most volatile)
        - US Midday: 12:00-15:00 ET
        - US Close: 15:00-16:00 ET

        Args:
            timestamps: DatetimeIndex of timestamps
            timezone: Timezone for session classification

        Returns:
            Series with TimeOfDay values as strings
        """
        # Convert to Eastern time if needed
        if timestamps.tz is None:
            # Assume UTC if no timezone
            ts_et = timestamps.tz_localize("UTC").tz_convert(timezone)
        else:
            ts_et = timestamps.tz_convert(timezone)

        # Extract hour for classification
        hours = ts_et.hour
        minutes = ts_et.minute

        # Convert to decimal hours for easier comparison
        decimal_hours = hours + minutes / 60.0

        # Initialize with a default (US Midday for gaps in coverage)
        regimes = pd.Series(TimeOfDay.US_MIDDAY.value, index=timestamps, dtype="object")

        # Asia session: 18:00-02:00 (wraps around midnight)
        asia_mask = (decimal_hours >= 18.0) | (decimal_hours < 2.0)
        regimes[asia_mask] = TimeOfDay.ASIA.value

        # London session: 03:00-08:00
        london_mask = (decimal_hours >= 3.0) & (decimal_hours < 8.0)
        regimes[london_mask] = TimeOfDay.LONDON.value

        # US Open: 09:30-12:00 (most volatile)
        us_open_mask = (decimal_hours >= 9.5) & (decimal_hours < 12.0)
        regimes[us_open_mask] = TimeOfDay.US_OPEN.value

        # US Midday: 12:00-15:00
        us_midday_mask = (decimal_hours >= 12.0) & (decimal_hours < 15.0)
        regimes[us_midday_mask] = TimeOfDay.US_MIDDAY.value

        # US Close: 15:00-16:00
        us_close_mask = (decimal_hours >= 15.0) & (decimal_hours < 16.0)
        regimes[us_close_mask] = TimeOfDay.US_CLOSE.value

        # Gap between sessions (08:00-09:30) - classify as US Open overlap
        gap_mask = (decimal_hours >= 8.0) & (decimal_hours < 9.5)
        regimes[gap_mask] = TimeOfDay.LONDON.value

        # After hours (16:00-18:00) - classify as transition to Asia
        after_hours_mask = (decimal_hours >= 16.0) & (decimal_hours < 18.0)
        regimes[after_hours_mask] = TimeOfDay.US_CLOSE.value

        return regimes

    def _calculate_adx(
        self,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculate ADX, +DI, -DI.

        Reuses logic from src/phase1/stages/regime/trend.py.

        Args:
            high: High prices
            low: Low prices
            close: Close prices

        Returns:
            Tuple of (ADX, +DI, -DI) arrays
        """
        # Import from existing implementation
        try:
            from src.pipeline.stages.regime.trend import calculate_adx

            return calculate_adx(high, low, close, self.trend_window)
        except ImportError:
            # Fallback implementation
            return self._adx_fallback(high, low, close)

    def _adx_fallback(
        self,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Fallback ADX calculation if import fails."""
        period = self.trend_window
        n = len(close)
        plus_di = np.full(n, np.nan)
        minus_di = np.full(n, np.nan)
        adx = np.full(n, np.nan)

        # Calculate True Range and Directional Movement
        tr = np.zeros(n)
        plus_dm = np.zeros(n)
        minus_dm = np.zeros(n)

        for i in range(1, n):
            hl = high[i] - low[i]
            hc = abs(high[i] - close[i - 1])
            lc = abs(low[i] - close[i - 1])
            tr[i] = max(hl, hc, lc)

            high_diff = high[i] - high[i - 1]
            low_diff = low[i - 1] - low[i]

            if high_diff > low_diff and high_diff > 0:
                plus_dm[i] = high_diff
            if low_diff > high_diff and low_diff > 0:
                minus_dm[i] = low_diff

        # Smooth TR and DM
        tr_smooth = np.zeros(n)
        plus_dm_smooth = np.zeros(n)
        minus_dm_smooth = np.zeros(n)

        if period < n:
            tr_smooth[period] = np.sum(tr[1 : period + 1])
            plus_dm_smooth[period] = np.sum(plus_dm[1 : period + 1])
            minus_dm_smooth[period] = np.sum(minus_dm[1 : period + 1])

            for i in range(period + 1, n):
                tr_smooth[i] = tr_smooth[i - 1] - (tr_smooth[i - 1] / period) + tr[i]
                plus_dm_smooth[i] = (
                    plus_dm_smooth[i - 1] - (plus_dm_smooth[i - 1] / period) + plus_dm[i]
                )
                minus_dm_smooth[i] = (
                    minus_dm_smooth[i - 1]
                    - (minus_dm_smooth[i - 1] / period)
                    + minus_dm[i]
                )

        # Calculate DI
        for i in range(period, n):
            if tr_smooth[i] != 0:
                plus_di[i] = 100 * plus_dm_smooth[i] / tr_smooth[i]
                minus_di[i] = 100 * minus_dm_smooth[i] / tr_smooth[i]

        # Calculate DX and ADX
        dx = np.full(n, np.nan)
        for i in range(period, n):
            di_sum = plus_di[i] + minus_di[i]
            if di_sum != 0:
                dx[i] = 100 * abs(plus_di[i] - minus_di[i]) / di_sum

        # ADX is smoothed DX
        if 2 * period - 1 < n:
            adx[2 * period - 1] = np.nanmean(dx[period : 2 * period])
            for i in range(2 * period, n):
                if not np.isnan(dx[i]):
                    adx[i] = (adx[i - 1] * (period - 1) + dx[i]) / period

        return adx, plus_di, minus_di


# =============================================================================
# Regime Evaluator
# =============================================================================


class RegimeEvaluator:
    """Evaluate model performance conditioned on market regime.

    This evaluator breaks down model performance by:
    1. Volatility regime (Low/Normal/High)
    2. Trend regime (Trending Up/Down/Mean Reverting)
    3. Time of day (Asia/London/US Open/US Midday/US Close)

    For each regime, it calculates:
    - Classification metrics (accuracy, F1, precision, recall)
    - Trading metrics (Sharpe ratio, win rate, avg return, max drawdown)

    Example:
        >>> evaluator = RegimeEvaluator()
        >>> result = evaluator.evaluate(
        ...     y_true=y_test,
        ...     y_pred=predictions,
        ...     y_proba=probabilities,
        ...     prices=price_df,
        ...     timestamps=test_timestamps,
        ... )
        >>> weak_regimes = evaluator.identify_weak_regimes(result)
        >>> print(f"Underperforming regimes: {weak_regimes}")
    """

    def __init__(
        self,
        classifier: Optional[RegimeClassifier] = None,
        min_samples_per_regime: int = 30,
    ):
        """Initialize regime evaluator.

        Args:
            classifier: RegimeClassifier instance (created with defaults if None)
            min_samples_per_regime: Minimum samples required for regime metrics
        """
        self.classifier = classifier or RegimeClassifier()
        self.min_samples_per_regime = min_samples_per_regime

    def evaluate(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: np.ndarray,
        prices: pd.DataFrame,
        timestamps: pd.DatetimeIndex,
    ) -> RegimeEvaluationResult:
        """Evaluate model performance broken down by regime.

        Args:
            y_true: True labels (-1=short, 0=neutral, 1=long)
            y_pred: Predicted labels
            y_proba: Prediction probabilities (n_samples, n_classes)
            prices: OHLCV price data for return and regime calculation
            timestamps: Timestamps aligned with predictions

        Returns:
            RegimeEvaluationResult with performance breakdown
        """
        if len(y_true) != len(y_pred):
            raise ValueError(
                f"Length mismatch: y_true ({len(y_true)}) != y_pred ({len(y_pred)})"
            )

        if len(timestamps) != len(y_true):
            raise ValueError(
                f"Length mismatch: timestamps ({len(timestamps)}) != y_true ({len(y_true)})"
            )

        # Calculate returns for Sharpe/win rate calculations
        if "close" in prices.columns:
            close_prices = prices["close"].values
            if len(close_prices) > len(y_true):
                # Trim to match predictions
                close_prices = close_prices[-len(y_true) :]
            elif len(close_prices) < len(y_true):
                raise ValueError(
                    f"Price data too short: {len(close_prices)} < {len(y_true)}"
                )
            returns = np.diff(np.log(close_prices))
            returns = np.insert(returns, 0, 0)  # Pad to match length
        else:
            raise ValueError("prices DataFrame must contain 'close' column")

        # Classify regimes
        returns_series = pd.Series(returns, index=timestamps)
        vol_regimes = self.classifier.classify_volatility(returns_series)

        # For trend classification, need OHLC
        if len(prices) > len(y_true):
            prices_aligned = prices.iloc[-len(y_true) :].copy()
            prices_aligned.index = timestamps
        elif len(prices) == len(y_true):
            prices_aligned = prices.copy()
            prices_aligned.index = timestamps
        else:
            raise ValueError(f"Price data too short: {len(prices)} < {len(y_true)}")

        trend_regimes = self.classifier.classify_trend(prices_aligned)
        tod_regimes = self.classifier.classify_time_of_day(timestamps)

        # Calculate overall metrics
        overall_metrics = self._calculate_metrics(
            y_true, y_pred, y_proba, returns, "overall"
        )

        # Calculate per-regime metrics
        volatility_breakdown = self._calculate_regime_breakdown(
            y_true,
            y_pred,
            y_proba,
            returns,
            vol_regimes,
            [r.value for r in VolatilityRegime],
        )

        trend_breakdown = self._calculate_regime_breakdown(
            y_true,
            y_pred,
            y_proba,
            returns,
            trend_regimes,
            [r.value for r in TrendRegime],
        )

        tod_breakdown = self._calculate_regime_breakdown(
            y_true,
            y_pred,
            y_proba,
            returns,
            tod_regimes,
            [r.value for r in TimeOfDay],
        )

        # Calculate regime-performance correlations
        regime_correlations = self._calculate_regime_correlations(
            y_true,
            y_pred,
            vol_regimes,
            trend_regimes,
            tod_regimes,
        )

        return RegimeEvaluationResult(
            overall_metrics=overall_metrics,
            volatility_breakdown=volatility_breakdown,
            trend_breakdown=trend_breakdown,
            time_of_day_breakdown=tod_breakdown,
            regime_correlations=regime_correlations,
        )

    def _calculate_regime_breakdown(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: np.ndarray,
        returns: np.ndarray,
        regimes: pd.Series,
        regime_values: list[str],
    ) -> dict[str, RegimeMetrics]:
        """Calculate metrics for each regime.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Prediction probabilities
            returns: Returns series
            regimes: Regime classification series
            regime_values: List of possible regime values

        Returns:
            Dict mapping regime name to RegimeMetrics
        """
        breakdown = {}

        for regime in regime_values:
            mask = regimes.values == regime

            if mask.sum() < self.min_samples_per_regime:
                logger.debug(
                    f"Skipping regime '{regime}' with {mask.sum()} samples "
                    f"(< {self.min_samples_per_regime})"
                )
                continue

            metrics = self._calculate_metrics(
                y_true[mask],
                y_pred[mask],
                y_proba[mask],
                returns[mask],
                regime,
            )

            breakdown[regime] = RegimeMetrics(**metrics)

        return breakdown

    def _calculate_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: np.ndarray,
        returns: np.ndarray,
        regime_name: str,
    ) -> dict[str, Any]:
        """Calculate classification and trading metrics.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Prediction probabilities
            returns: Returns series
            regime_name: Name of the regime for logging

        Returns:
            Dict with all metrics
        """
        from sklearn.metrics import (
            accuracy_score,
            f1_score,
            precision_score,
            recall_score,
        )

        n_samples = len(y_true)

        if n_samples == 0:
            return {
                "regime_name": regime_name,
                "n_samples": 0,
                "accuracy": 0.0,
                "f1_score": 0.0,
                "sharpe_ratio": 0.0,
                "win_rate": 0.0,
                "avg_return": 0.0,
                "max_drawdown": 0.0,
                "precision": 0.0,
                "recall": 0.0,
            }

        # Classification metrics
        accuracy = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
        precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
        recall = recall_score(y_true, y_pred, average="macro", zero_division=0)

        # Trading metrics
        # Strategy returns: long when pred=1, short when pred=-1, flat when pred=0
        strategy_returns = returns * y_pred

        # Sharpe ratio (annualized assuming 5-min bars, ~78 trading days per year)
        # 252 days * 6.5 hours * 12 bars/hour = ~19,656 bars/year
        # For simplicity, use sqrt(252) scaling from daily approximation
        if strategy_returns.std() > 0:
            sharpe = (
                strategy_returns.mean() / strategy_returns.std() * np.sqrt(252 * 78)
            )
        else:
            sharpe = 0.0

        # Win rate (only for non-zero positions)
        position_mask = y_pred != 0
        if position_mask.sum() > 0:
            position_returns = strategy_returns[position_mask]
            win_rate = (position_returns > 0).sum() / len(position_returns)
            avg_return = position_returns.mean()
        else:
            win_rate = 0.0
            avg_return = 0.0

        # Max drawdown from cumulative returns
        cumulative_returns = np.cumsum(strategy_returns)
        if len(cumulative_returns) > 0:
            running_max = np.maximum.accumulate(cumulative_returns)
            drawdowns = running_max - cumulative_returns
            max_drawdown = drawdowns.max() if len(drawdowns) > 0 else 0.0
        else:
            max_drawdown = 0.0

        return {
            "regime_name": regime_name,
            "n_samples": n_samples,
            "accuracy": float(accuracy),
            "f1_score": float(f1),
            "sharpe_ratio": float(sharpe),
            "win_rate": float(win_rate),
            "avg_return": float(avg_return),
            "max_drawdown": float(max_drawdown),
            "precision": float(precision),
            "recall": float(recall),
        }

    def _calculate_regime_correlations(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        vol_regimes: pd.Series,
        trend_regimes: pd.Series,
        tod_regimes: pd.Series,
    ) -> dict[str, float]:
        """Calculate correlation between regime and prediction correctness.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            vol_regimes: Volatility regime series
            trend_regimes: Trend regime series
            tod_regimes: Time-of-day regime series

        Returns:
            Dict with correlation values
        """
        correct = (y_true == y_pred).astype(float)

        correlations = {}

        # Encode regimes numerically for correlation
        for name, regimes in [
            ("volatility", vol_regimes),
            ("trend", trend_regimes),
            ("time_of_day", tod_regimes),
        ]:
            # Get unique values and encode
            unique_vals = regimes.dropna().unique()
            encoding = {v: i for i, v in enumerate(unique_vals)}
            encoded = regimes.map(encoding)

            # Remove NaN for correlation
            valid_mask = ~encoded.isna()
            if valid_mask.sum() > 10:
                corr = np.corrcoef(
                    correct[valid_mask.values], encoded[valid_mask].values
                )[0, 1]
                correlations[f"{name}_correlation"] = (
                    float(corr) if not np.isnan(corr) else 0.0
                )
            else:
                correlations[f"{name}_correlation"] = 0.0

        return correlations

    def identify_weak_regimes(
        self,
        result: RegimeEvaluationResult,
        accuracy_threshold: float = 0.5,
        sharpe_threshold: float = 0.0,
    ) -> list[dict[str, Any]]:
        """Identify regimes where model underperforms.

        Args:
            result: RegimeEvaluationResult from evaluate()
            accuracy_threshold: Accuracy below this is considered weak
            sharpe_threshold: Sharpe ratio below this is considered weak

        Returns:
            List of dicts with weak regime details
        """
        weak_regimes = []

        all_breakdowns = [
            ("volatility", result.volatility_breakdown),
            ("trend", result.trend_breakdown),
            ("time_of_day", result.time_of_day_breakdown),
        ]

        for regime_type, breakdown in all_breakdowns:
            for regime_name, metrics in breakdown.items():
                is_weak = (
                    metrics.accuracy < accuracy_threshold
                    or metrics.sharpe_ratio < sharpe_threshold
                )

                if is_weak:
                    weak_regimes.append(
                        {
                            "regime_type": regime_type,
                            "regime_name": regime_name,
                            "n_samples": metrics.n_samples,
                            "accuracy": metrics.accuracy,
                            "sharpe_ratio": metrics.sharpe_ratio,
                            "win_rate": metrics.win_rate,
                            "reason": self._get_weakness_reason(
                                metrics, accuracy_threshold, sharpe_threshold
                            ),
                        }
                    )

        return weak_regimes

    def _get_weakness_reason(
        self,
        metrics: RegimeMetrics,
        accuracy_threshold: float,
        sharpe_threshold: float,
    ) -> str:
        """Get human-readable reason for regime weakness."""
        reasons = []
        if metrics.accuracy < accuracy_threshold:
            reasons.append(f"low accuracy ({metrics.accuracy:.2%} < {accuracy_threshold:.2%})")
        if metrics.sharpe_ratio < sharpe_threshold:
            reasons.append(
                f"low Sharpe ({metrics.sharpe_ratio:.2f} < {sharpe_threshold:.2f})"
            )
        return "; ".join(reasons) if reasons else "unknown"


# =============================================================================
# Convenience Functions
# =============================================================================


def evaluate_regime_performance(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    prices: pd.DataFrame,
    timestamps: pd.DatetimeIndex,
    classifier_config: Optional[dict[str, Any]] = None,
) -> RegimeEvaluationResult:
    """Convenience function for regime-conditional evaluation.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_proba: Prediction probabilities
        prices: OHLCV price data
        timestamps: Timestamps aligned with predictions
        classifier_config: Optional config for RegimeClassifier

    Returns:
        RegimeEvaluationResult with full breakdown
    """
    classifier_config = classifier_config or {}
    classifier = RegimeClassifier(**classifier_config)
    evaluator = RegimeEvaluator(classifier=classifier)
    return evaluator.evaluate(y_true, y_pred, y_proba, prices, timestamps)


def get_regime_summary(result: RegimeEvaluationResult) -> str:
    """Generate a text summary of regime evaluation results.

    Args:
        result: RegimeEvaluationResult from evaluate()

    Returns:
        Human-readable summary string
    """
    lines = ["=" * 60, "REGIME-CONDITIONAL EVALUATION SUMMARY", "=" * 60, ""]

    # Overall metrics
    lines.append("OVERALL PERFORMANCE:")
    overall = result.overall_metrics
    lines.append(f"  Accuracy: {overall.get('accuracy', 0):.4f}")
    lines.append(f"  F1 Score: {overall.get('f1_score', 0):.4f}")
    lines.append(f"  Sharpe Ratio: {overall.get('sharpe_ratio', 0):.2f}")
    lines.append(f"  Win Rate: {overall.get('win_rate', 0):.2%}")
    lines.append("")

    # Volatility breakdown
    lines.append("VOLATILITY REGIME BREAKDOWN:")
    for regime, metrics in result.volatility_breakdown.items():
        lines.append(
            f"  {regime.upper():15} | n={metrics.n_samples:5} | "
            f"acc={metrics.accuracy:.3f} | sharpe={metrics.sharpe_ratio:6.2f} | "
            f"win={metrics.win_rate:.2%}"
        )
    lines.append("")

    # Trend breakdown
    lines.append("TREND REGIME BREAKDOWN:")
    for regime, metrics in result.trend_breakdown.items():
        lines.append(
            f"  {regime.upper():15} | n={metrics.n_samples:5} | "
            f"acc={metrics.accuracy:.3f} | sharpe={metrics.sharpe_ratio:6.2f} | "
            f"win={metrics.win_rate:.2%}"
        )
    lines.append("")

    # Time-of-day breakdown
    lines.append("TIME-OF-DAY BREAKDOWN:")
    for regime, metrics in result.time_of_day_breakdown.items():
        lines.append(
            f"  {regime.upper():15} | n={metrics.n_samples:5} | "
            f"acc={metrics.accuracy:.3f} | sharpe={metrics.sharpe_ratio:6.2f} | "
            f"win={metrics.win_rate:.2%}"
        )
    lines.append("")

    # Regime correlations
    if result.regime_correlations:
        lines.append("REGIME-PERFORMANCE CORRELATIONS:")
        for name, corr in result.regime_correlations.items():
            lines.append(f"  {name}: {corr:+.3f}")
        lines.append("")

    # Weak regimes warning
    weak = result.get_weak_regimes(accuracy_threshold=0.5)
    if weak:
        lines.append("WARNING - WEAK REGIMES DETECTED:")
        for regime in weak:
            lines.append(f"  - {regime}")
        lines.append("")

    lines.append("=" * 60)
    return "\n".join(lines)


__all__ = [
    # Enums
    "VolatilityRegime",
    "TrendRegime",
    "TimeOfDay",
    # Data classes
    "RegimeMetrics",
    "RegimeEvaluationResult",
    # Classes
    "RegimeClassifier",
    "RegimeEvaluator",
    # Functions
    "evaluate_regime_performance",
    "get_regime_summary",
]
