"""
Multi-Timeframe (MTF) Feature Computation - PHASE_1.

This module provides functionality to compute features across multiple timeframes
by resampling base 1-minute OHLCV data to higher timeframes and computing
specified features on each.

Key Features:
- Resamples 1-minute data to configurable higher timeframes (5min, 15min, 30min, 60min)
- Computes ~30 features per higher timeframe
- Applies shift(1) for anti-lookahead bias protection
- Returns features named as: {feature}_{timeframe} (e.g., rsi_14_5min)

Usage:
    from src.features.compute.mtf import (
        MTFConfig,
        MTFFeatureComputer,
        compute_mtf_features,
        resample_ohlcv,
    )

    # Quick usage with defaults
    mtf_df = compute_mtf_features(ohlcv_1min_df)

    # Configurable usage
    config = MTFConfig(
        timeframes=["5min", "15min", "60min"],
        features=["rsi_14", "macd_line", "atr_14"],
    )
    computer = MTFFeatureComputer(config)
    mtf_df = computer.compute(ohlcv_1min_df)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from src.core.constants import (
    DEFAULT_MTF_TIMEFRAMES,
    MTF_FEATURES_PER_TF,
    OHLCV_COLUMNS,
)


# =============================================================================
# MTF FEATURE SELECTION
# =============================================================================

# Default features to compute on higher timeframes (~30 features)
# These are selected for their relevance in multi-timeframe analysis
# All names must exist in FEATURE_COMPUTE_MAP (see src/features/compute/__init__.py)
DEFAULT_MTF_FEATURES: List[str] = [
    # Momentum (8)
    "rsi_14",
    "rsi_7",
    "macd_line",
    "macd_signal",
    "macd_histogram",
    "stoch_k",
    "stoch_d",
    "williams_r",
    # Moving Averages (6)
    "sma_10",
    "sma_20",
    "ema_12",
    "ema_21",
    "sma_cross_10_50",
    "price_to_sma_20",
    # Volatility (6)
    "atr_14",
    "bb_upper",
    "bb_middle",
    "bb_lower",
    "bb_width",
    "bb_position",
    # Volume (4)
    "volume_ratio",
    "obv",
    "vwap",
    "volume_zscore",
    # Trend (3)
    "adx_14",
    "plus_di_14",
    "minus_di_14",
    # Price (3)
    "return_1",
    "return_5",
    "range_pct",
]

# Validate default feature count matches expected
assert len(DEFAULT_MTF_FEATURES) == MTF_FEATURES_PER_TF, (
    f"DEFAULT_MTF_FEATURES has {len(DEFAULT_MTF_FEATURES)} features, "
    f"expected {MTF_FEATURES_PER_TF}"
)


# =============================================================================
# CONFIGURATION
# =============================================================================


@dataclass
class MTFConfig:
    """
    Configuration for Multi-Timeframe feature computation.

    Attributes:
        timeframes: Higher timeframes to resample to (e.g., ["5min", "15min", "60min"])
        features: Feature names to compute on each timeframe
        apply_shift: Whether to apply shift(1) for anti-lookahead bias (default: True)
        ffill_limit: Maximum periods to forward-fill after resampling (default: None = no limit)
        min_periods_ratio: Minimum valid data ratio required (default: 0.5)

    Example:
        config = MTFConfig(
            timeframes=["5min", "15min"],
            features=["rsi_14", "macd_line"],
        )
    """

    timeframes: List[str] = field(default_factory=lambda: DEFAULT_MTF_TIMEFRAMES.copy())
    features: List[str] = field(default_factory=lambda: DEFAULT_MTF_FEATURES.copy())
    apply_shift: bool = True  # Anti-lookahead protection
    ffill_limit: Optional[int] = None  # Forward fill limit after reindex
    min_periods_ratio: float = 0.5  # Minimum valid data ratio

    def __post_init__(self):
        """Validate configuration."""
        # Validate timeframes
        valid_timeframes = ["5min", "10min", "15min", "20min", "25min", "30min", "45min", "60min"]
        for tf in self.timeframes:
            if tf not in valid_timeframes:
                raise ValueError(
                    f"Invalid timeframe '{tf}'. Must be one of: {valid_timeframes}"
                )

        # Validate features list is not empty
        if not self.features:
            raise ValueError("features list cannot be empty")

        # Validate min_periods_ratio
        if not 0 < self.min_periods_ratio <= 1:
            raise ValueError("min_periods_ratio must be between 0 and 1")

    @property
    def n_timeframes(self) -> int:
        """Number of higher timeframes."""
        return len(self.timeframes)

    @property
    def n_features(self) -> int:
        """Number of features per timeframe."""
        return len(self.features)

    @property
    def total_features(self) -> int:
        """Total number of MTF features (n_timeframes * n_features)."""
        return self.n_timeframes * self.n_features


# =============================================================================
# OHLCV RESAMPLING
# =============================================================================


def resample_ohlcv(
    df: pd.DataFrame,
    timeframe: str,
    datetime_col: Optional[str] = None,
) -> pd.DataFrame:
    """
    Resample 1-minute OHLCV data to a higher timeframe.

    Uses standard OHLCV resampling rules:
    - open: first value in period
    - high: max value in period
    - low: min value in period
    - close: last value in period
    - volume: sum of values in period

    Args:
        df: DataFrame with OHLCV columns (open, high, low, close, volume)
        timeframe: Target timeframe string (e.g., "5min", "15min", "60min")
        datetime_col: Name of datetime column or None if index is DatetimeIndex

    Returns:
        Resampled DataFrame with same columns

    Raises:
        ValueError: If required columns are missing or timeframe is invalid

    Example:
        >>> df_1min = pd.read_parquet("data.parquet")
        >>> df_5min = resample_ohlcv(df_1min, "5min")
    """
    # Validate OHLCV columns exist
    missing_cols = [col for col in OHLCV_COLUMNS if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required OHLCV columns: {missing_cols}")

    # Convert timeframe string to pandas offset
    tf_map = {
        "5min": "5min",
        "10min": "10min",
        "15min": "15min",
        "20min": "20min",
        "25min": "25min",
        "30min": "30min",
        "45min": "45min",
        "60min": "60min",
        "1h": "60min",  # Alias
    }

    if timeframe not in tf_map:
        raise ValueError(
            f"Invalid timeframe '{timeframe}'. Valid options: {list(tf_map.keys())}"
        )

    offset = tf_map[timeframe]

    # Set up datetime index for resampling
    if datetime_col is not None:
        if datetime_col not in df.columns:
            raise ValueError(f"datetime_col '{datetime_col}' not found in DataFrame")
        df_work = df.set_index(datetime_col)
    else:
        df_work = df.copy()

    # Ensure index is DatetimeIndex
    if not isinstance(df_work.index, pd.DatetimeIndex):
        raise ValueError(
            "DataFrame index must be DatetimeIndex or datetime_col must be specified"
        )

    # Define resampling rules for OHLCV
    ohlcv_agg = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }

    # Resample using label='left' to mark candle at period start
    resampled = df_work[OHLCV_COLUMNS].resample(offset, label="left", closed="left").agg(ohlcv_agg)

    # Drop rows where we don't have complete candles (NaN in OHLC)
    resampled = resampled.dropna(subset=["open", "high", "low", "close"])

    return resampled


def _reindex_to_base(
    resampled_df: pd.DataFrame,
    base_index: pd.DatetimeIndex,
    ffill_limit: Optional[int] = None,
) -> pd.DataFrame:
    """
    Reindex resampled data back to base (1-minute) index with forward fill.

    This aligns higher timeframe features with the base 1-minute data,
    forward-filling values until the next higher-timeframe candle completes.

    Args:
        resampled_df: DataFrame with resampled data
        base_index: Original 1-minute DatetimeIndex
        ffill_limit: Max periods to forward fill (None = no limit)

    Returns:
        DataFrame aligned to base_index with forward-filled values
    """
    # Reindex to base frequency
    aligned = resampled_df.reindex(base_index)

    # Forward fill to propagate higher TF values
    aligned = aligned.ffill(limit=ffill_limit)

    return aligned


# =============================================================================
# FEATURE COMPUTATION FUNCTIONS
# =============================================================================


def _get_feature_compute_fn(feature_name: str) -> Callable[[pd.DataFrame], pd.Series]:
    """
    Get the compute function for a feature by name.

    This lazily imports from the main compute module to avoid circular imports.

    Args:
        feature_name: Name of the feature

    Returns:
        Callable that computes the feature

    Raises:
        ValueError: If feature is unknown
    """
    # Import here to avoid circular imports
    from src.features.compute import FEATURE_COMPUTE_MAP

    if feature_name not in FEATURE_COMPUTE_MAP:
        raise ValueError(
            f"Unknown feature '{feature_name}'. "
            f"Available features: {len(FEATURE_COMPUTE_MAP)}"
        )

    return FEATURE_COMPUTE_MAP[feature_name]


def _compute_features_on_resampled(
    resampled_df: pd.DataFrame,
    feature_names: List[str],
) -> pd.DataFrame:
    """
    Compute specified features on resampled OHLCV data.

    Args:
        resampled_df: Resampled OHLCV DataFrame
        feature_names: List of feature names to compute

    Returns:
        DataFrame with computed features
    """
    results = {}

    for feature_name in feature_names:
        try:
            compute_fn = _get_feature_compute_fn(feature_name)
            results[feature_name] = compute_fn(resampled_df)
        except Exception as e:
            import warnings
            warnings.warn(f"Error computing {feature_name}: {e}")
            results[feature_name] = pd.Series(np.nan, index=resampled_df.index)

    return pd.DataFrame(results, index=resampled_df.index)


# =============================================================================
# MTF FEATURE COMPUTER
# =============================================================================


class MTFFeatureComputer:
    """
    Multi-Timeframe Feature Computer.

    Computes features across multiple timeframes by:
    1. Resampling base 1-minute OHLCV data to higher timeframes
    2. Computing specified features on each resampled timeframe
    3. Reindexing back to base frequency with forward fill
    4. Applying shift(1) for anti-lookahead bias protection
    5. Returning DataFrame with features named: {feature}_{timeframe}

    Example:
        config = MTFConfig(
            timeframes=["5min", "15min", "60min"],
            features=["rsi_14", "macd_line", "atr_14"],
        )
        computer = MTFFeatureComputer(config)
        mtf_df = computer.compute(ohlcv_1min_df)

        # Result has columns like: rsi_14_5min, rsi_14_15min, rsi_14_60min, etc.
    """

    def __init__(self, config: Optional[MTFConfig] = None):
        """
        Initialize MTF Feature Computer.

        Args:
            config: MTFConfig instance. If None, uses defaults.
        """
        self.config = config or MTFConfig()
        self._feature_fns: Dict[str, Callable] = {}
        self._validate_features()

    def _validate_features(self) -> None:
        """Validate that all configured features exist in FEATURE_COMPUTE_MAP."""
        # Import here to avoid circular imports
        from src.features.compute import FEATURE_COMPUTE_MAP

        invalid_features = []
        for feature_name in self.config.features:
            if feature_name not in FEATURE_COMPUTE_MAP:
                invalid_features.append(feature_name)
            else:
                self._feature_fns[feature_name] = FEATURE_COMPUTE_MAP[feature_name]

        if invalid_features:
            raise ValueError(
                f"Unknown features: {invalid_features}. "
                f"Check FEATURE_COMPUTE_MAP for available features."
            )

    def compute(
        self,
        df: pd.DataFrame,
        datetime_col: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Compute MTF features for all configured timeframes.

        Args:
            df: DataFrame with 1-minute OHLCV data
            datetime_col: Name of datetime column (if not using DatetimeIndex)

        Returns:
            DataFrame with MTF features named: {feature}_{timeframe}
            Aligned to original index with shift(1) applied for anti-lookahead.

        Example:
            >>> mtf_df = computer.compute(ohlcv_1min)
            >>> print(mtf_df.columns.tolist()[:5])
            ['rsi_14_5min', 'rsi_14_15min', 'rsi_14_60min', 'macd_line_5min', ...]
        """
        # Set up working DataFrame with datetime index
        if datetime_col is not None:
            if datetime_col not in df.columns:
                raise ValueError(f"datetime_col '{datetime_col}' not in DataFrame")
            df_work = df.set_index(datetime_col)
        else:
            df_work = df.copy()

        # Validate datetime index
        if not isinstance(df_work.index, pd.DatetimeIndex):
            raise ValueError(
                "DataFrame index must be DatetimeIndex or datetime_col must be specified"
            )

        base_index = df_work.index
        all_mtf_features: Dict[str, pd.Series] = {}

        for timeframe in self.config.timeframes:
            # 1. Resample OHLCV to higher timeframe
            resampled = resample_ohlcv(df_work, timeframe)

            if len(resampled) == 0:
                import warnings
                warnings.warn(f"No data after resampling to {timeframe}, skipping")
                continue

            # 2. Compute features on resampled data
            tf_features = _compute_features_on_resampled(
                resampled, self.config.features
            )

            # 3. Reindex back to base frequency with forward fill
            tf_features_aligned = _reindex_to_base(
                tf_features,
                base_index,
                ffill_limit=self.config.ffill_limit,
            )

            # 4. Apply shift(1) for anti-lookahead bias
            # This ensures we only use information from completed candles
            if self.config.apply_shift:
                tf_features_aligned = tf_features_aligned.shift(1)

            # 5. Rename columns with timeframe suffix
            for feature_name in self.config.features:
                if feature_name in tf_features_aligned.columns:
                    new_name = f"{feature_name}_{timeframe}"
                    all_mtf_features[new_name] = tf_features_aligned[feature_name]

        # Combine all MTF features into single DataFrame
        result = pd.DataFrame(all_mtf_features, index=base_index)

        return result

    def compute_single_timeframe(
        self,
        df: pd.DataFrame,
        timeframe: str,
        datetime_col: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Compute MTF features for a single timeframe.

        Args:
            df: DataFrame with 1-minute OHLCV data
            timeframe: Target timeframe (e.g., "5min")
            datetime_col: Name of datetime column (if not using DatetimeIndex)

        Returns:
            DataFrame with features for single timeframe, named: {feature}_{timeframe}
        """
        if timeframe not in self.config.timeframes:
            # Temporarily add timeframe
            original_timeframes = self.config.timeframes
            self.config.timeframes = [timeframe]

        # Set up working DataFrame with datetime index
        if datetime_col is not None:
            df_work = df.set_index(datetime_col)
        else:
            df_work = df.copy()

        if not isinstance(df_work.index, pd.DatetimeIndex):
            raise ValueError(
                "DataFrame index must be DatetimeIndex or datetime_col must be specified"
            )

        base_index = df_work.index

        # Resample and compute
        resampled = resample_ohlcv(df_work, timeframe)
        tf_features = _compute_features_on_resampled(resampled, self.config.features)

        # Reindex and shift
        tf_features_aligned = _reindex_to_base(
            tf_features, base_index, self.config.ffill_limit
        )

        if self.config.apply_shift:
            tf_features_aligned = tf_features_aligned.shift(1)

        # Rename with timeframe suffix
        tf_features_aligned.columns = [
            f"{col}_{timeframe}" for col in tf_features_aligned.columns
        ]

        return tf_features_aligned

    def get_feature_names(self) -> List[str]:
        """
        Get list of all MTF feature names that will be generated.

        Returns:
            List of feature names in format: {feature}_{timeframe}
        """
        names = []
        for timeframe in self.config.timeframes:
            for feature in self.config.features:
                names.append(f"{feature}_{timeframe}")
        return names

    def __repr__(self) -> str:
        return (
            f"MTFFeatureComputer(\n"
            f"  timeframes={self.config.timeframes},\n"
            f"  n_features={self.config.n_features},\n"
            f"  total_features={self.config.total_features},\n"
            f"  apply_shift={self.config.apply_shift}\n"
            f")"
        )


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


def compute_mtf_features(
    df: pd.DataFrame,
    timeframes: Optional[List[str]] = None,
    features: Optional[List[str]] = None,
    apply_shift: bool = True,
    datetime_col: Optional[str] = None,
) -> pd.DataFrame:
    """
    Convenience function to compute MTF features with minimal configuration.

    This is the primary entry point for MTF feature computation.
    Uses sensible defaults that can be overridden as needed.

    Args:
        df: DataFrame with 1-minute OHLCV data
        timeframes: Higher timeframes (default: ["5min", "15min", "60min"])
        features: Features to compute (default: ~30 standard features)
        apply_shift: Apply shift(1) for anti-lookahead (default: True)
        datetime_col: Name of datetime column (if not using DatetimeIndex)

    Returns:
        DataFrame with MTF features, columns named: {feature}_{timeframe}

    Example:
        >>> mtf_df = compute_mtf_features(ohlcv_1min)
        >>> print(f"Generated {len(mtf_df.columns)} MTF features")
        Generated 90 MTF features  # 30 features * 3 timeframes

        >>> # Custom configuration
        >>> mtf_df = compute_mtf_features(
        ...     ohlcv_1min,
        ...     timeframes=["5min", "15min", "30min", "60min"],
        ...     features=["rsi_14", "macd_line", "atr_14"],
        ... )
    """
    config = MTFConfig(
        timeframes=timeframes or DEFAULT_MTF_TIMEFRAMES.copy(),
        features=features or DEFAULT_MTF_FEATURES.copy(),
        apply_shift=apply_shift,
    )

    computer = MTFFeatureComputer(config)
    return computer.compute(df, datetime_col=datetime_col)


def get_mtf_feature_names(
    timeframes: Optional[List[str]] = None,
    features: Optional[List[str]] = None,
) -> List[str]:
    """
    Get list of MTF feature names without computing.

    Useful for schema validation and documentation.

    Args:
        timeframes: Higher timeframes (default: ["5min", "15min", "60min"])
        features: Base features (default: ~30 standard features)

    Returns:
        List of feature names in format: {feature}_{timeframe}
    """
    timeframes = timeframes or DEFAULT_MTF_TIMEFRAMES
    features = features or DEFAULT_MTF_FEATURES

    names = []
    for tf in timeframes:
        for feat in features:
            names.append(f"{feat}_{tf}")
    return names


def validate_mtf_config(config: MTFConfig) -> Dict[str, any]:
    """
    Validate MTF configuration and return diagnostics.

    Args:
        config: MTFConfig to validate

    Returns:
        Dict with validation results:
        - valid: bool
        - errors: List of error messages
        - warnings: List of warning messages
        - stats: Dict with configuration statistics
    """
    from src.features.compute import FEATURE_COMPUTE_MAP

    errors = []
    warnings_list = []

    # Check features exist
    invalid_features = [
        f for f in config.features if f not in FEATURE_COMPUTE_MAP
    ]
    if invalid_features:
        errors.append(f"Unknown features: {invalid_features}")

    # Check timeframes
    valid_timeframes = ["5min", "10min", "15min", "20min", "25min", "30min", "45min", "60min"]
    invalid_timeframes = [
        tf for tf in config.timeframes if tf not in valid_timeframes
    ]
    if invalid_timeframes:
        errors.append(f"Invalid timeframes: {invalid_timeframes}")

    # Warnings
    if len(config.features) > 50:
        warnings_list.append(
            f"Large feature set ({len(config.features)}) may impact performance"
        )

    if len(config.timeframes) > 5:
        warnings_list.append(
            f"Many timeframes ({len(config.timeframes)}) will multiply feature count significantly"
        )

    stats = {
        "n_timeframes": config.n_timeframes,
        "n_features_per_tf": config.n_features,
        "total_features": config.total_features,
        "apply_shift": config.apply_shift,
    }

    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings_list,
        "stats": stats,
    }


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Configuration
    "MTFConfig",
    "DEFAULT_MTF_FEATURES",
    # Main class
    "MTFFeatureComputer",
    # Convenience functions
    "compute_mtf_features",
    "get_mtf_feature_names",
    "validate_mtf_config",
    # Helper functions
    "resample_ohlcv",
]
