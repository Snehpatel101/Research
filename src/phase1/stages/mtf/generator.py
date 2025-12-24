"""
MTFFeatureGenerator - Main class for Multi-Timeframe feature generation.

Computes features from higher timeframes and aligns them to the base timeframe
without lookahead bias.

Key Design Decisions:
1. ANTI-LOOKAHEAD: All MTF features use shift(1) on the higher TF data before
   alignment. This ensures we only use COMPLETED higher TF bars, not the
   current (incomplete) bar which would cause lookahead bias.

2. SUPPORTED TIMEFRAMES: 5min (base), 15min, 30min, 1h, 4h, daily.
   Only timeframes that are integer multiples of the base are supported.

3. FORWARD-FILL ALIGNMENT: Higher TF features are forward-filled to base TF,
   meaning each base TF bar uses the most recent completed higher TF bar's value.

4. MTF MODES:
   - BARS: Generate only OHLCV data at higher timeframes
   - INDICATORS: Generate only technical indicators at higher timeframes
   - BOTH: Generate both OHLCV bars and indicators
"""

import logging
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd

from .constants import (
    MTF_TIMEFRAMES,
    REQUIRED_OHLCV_COLS,
    DEFAULT_MTF_TIMEFRAMES,
    DEFAULT_MTF_MODE,
    MIN_BASE_BARS,
    MIN_MTF_BARS,
    PANDAS_FREQ_MAP,
    MTFMode,
)
from .validators import (
    validate_ohlcv_dataframe,
    validate_mtf_timeframes,
)

# Configure logging
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class MTFFeatureGenerator:
    """
    Generate features from multiple timeframes.

    This class resamples base timeframe data to higher timeframes,
    optionally computes technical indicators on those higher timeframes,
    and aligns them back to the base timeframe without lookahead bias.

    Parameters
    ----------
    base_timeframe : str, default '5min'
        Base timeframe of input data
    mtf_timeframes : List[str], optional
        List of higher timeframes to compute features for.
        Default: ['15min', '30min', '1h', '4h', 'daily']
    mode : MTFMode or str, default MTFMode.BOTH
        What to generate:
        - 'bars': Only OHLCV data at higher timeframes
        - 'indicators': Only technical indicators at higher timeframes
        - 'both': Both OHLCV bars and indicators
    include_ohlcv : bool, optional (deprecated)
        Use mode='bars' or mode='both' instead
    include_indicators : bool, optional (deprecated)
        Use mode='indicators' or mode='both' instead
    """

    def __init__(
        self,
        base_timeframe: str = '5min',
        mtf_timeframes: Optional[List[str]] = None,
        mode: Union[MTFMode, str] = DEFAULT_MTF_MODE,
        include_ohlcv: Optional[bool] = None,
        include_indicators: Optional[bool] = None
    ):
        """Initialize MTF feature generator with validation."""
        self.base_timeframe = base_timeframe
        self.mtf_timeframes = mtf_timeframes or DEFAULT_MTF_TIMEFRAMES

        # Handle mode parameter
        if isinstance(mode, str):
            mode = MTFMode(mode.lower())
        self.mode = mode

        # Handle legacy parameters (backward compatibility)
        if include_ohlcv is not None or include_indicators is not None:
            logger.warning(
                "include_ohlcv and include_indicators are deprecated. "
                "Use mode='bars', 'indicators', or 'both' instead."
            )
            # Convert legacy params to mode
            if include_ohlcv and include_indicators:
                self.mode = MTFMode.BOTH
            elif include_ohlcv:
                self.mode = MTFMode.BARS
            elif include_indicators:
                self.mode = MTFMode.INDICATORS
            else:
                self.mode = MTFMode.BOTH  # Default if both False (shouldn't happen)

        # Validate and parse timeframes
        validate_mtf_timeframes(base_timeframe, self.mtf_timeframes)
        self.base_minutes = MTF_TIMEFRAMES[base_timeframe]

        logger.info(
            f"Initialized MTFFeatureGenerator: base={base_timeframe}, "
            f"mtf={self.mtf_timeframes}, mode={self.mode.value}"
        )

    @property
    def include_ohlcv(self) -> bool:
        """Legacy property for backward compatibility."""
        return self.mode in (MTFMode.BARS, MTFMode.BOTH)

    @property
    def include_indicators(self) -> bool:
        """Legacy property for backward compatibility."""
        return self.mode in (MTFMode.INDICATORS, MTFMode.BOTH)

    def _get_tf_suffix(self, tf: str) -> str:
        """
        Get column suffix for a timeframe.

        Examples:
            '15min' -> '_15m'
            '1h' -> '_1h'
            '4h' -> '_4h'
            'daily' -> '_1d'
        """
        minutes = MTF_TIMEFRAMES[tf]
        if minutes >= 1440:  # Daily or longer
            days = minutes // 1440
            return f"_{days}d"
        elif minutes >= 60 and minutes % 60 == 0:
            hours = minutes // 60
            return f"_{hours}h"
        return f"_{minutes}m"

    def _get_pandas_freq(self, tf: str) -> str:
        """Convert timeframe string to pandas frequency string."""
        if tf in PANDAS_FREQ_MAP:
            return PANDAS_FREQ_MAP[tf]
        # Fallback to minute-based frequency
        minutes = MTF_TIMEFRAMES[tf]
        return f"{minutes}min"

    def resample_to_tf(self, df: pd.DataFrame, target_tf: str) -> pd.DataFrame:
        """
        Resample OHLCV data to target timeframe.

        Parameters
        ----------
        df : pd.DataFrame
            Base timeframe OHLCV data with 'datetime' column
        target_tf : str
            Target timeframe to resample to

        Returns
        -------
        pd.DataFrame
            Resampled OHLCV data with 'datetime' column
        """
        validate_ohlcv_dataframe(df)

        df_copy = df.copy()
        df_copy = df_copy.set_index('datetime')

        freq = self._get_pandas_freq(target_tf)

        agg_dict = {
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }

        resampled = df_copy.resample(freq).agg(agg_dict).dropna()
        return resampled.reset_index()

    def generate_mtf_bars(
        self,
        df_tf: pd.DataFrame,
        timeframe: str
    ) -> pd.DataFrame:
        """
        Generate MTF OHLCV bar columns for a specific timeframe.

        Parameters
        ----------
        df_tf : pd.DataFrame
            OHLCV data at the target timeframe
        timeframe : str
            Timeframe string for column naming

        Returns
        -------
        pd.DataFrame
            DataFrame with MTF bar columns added (open_4h, high_4h, etc.)
        """
        result = df_tf.copy()
        tf_suffix = self._get_tf_suffix(timeframe)

        # Add OHLCV columns with timeframe suffix
        for col in REQUIRED_OHLCV_COLS:
            result[f'{col}{tf_suffix}'] = result[col]

        return result

    def compute_mtf_indicators(
        self,
        df_tf: pd.DataFrame,
        timeframe: str
    ) -> pd.DataFrame:
        """
        Compute technical indicators for a specific timeframe.

        Parameters
        ----------
        df_tf : pd.DataFrame
            OHLCV data at the target timeframe
        timeframe : str
            Timeframe string for column naming

        Returns
        -------
        pd.DataFrame
            DataFrame with indicator columns added
        """
        result = df_tf.copy()
        tf_suffix = self._get_tf_suffix(timeframe)

        # Moving Averages
        for period in [20, 50]:
            result[f'sma_{period}{tf_suffix}'] = (
                result['close'].rolling(period, min_periods=period).mean()
            )
        for period in [9, 21]:
            result[f'ema_{period}{tf_suffix}'] = (
                result['close'].ewm(span=period, min_periods=period, adjust=False).mean()
            )

        # RSI
        delta = result['close'].diff()
        gain = delta.where(delta > 0, 0.0)
        loss = (-delta.where(delta < 0, 0.0))
        avg_gain = gain.rolling(14, min_periods=14).mean()
        avg_loss = loss.rolling(14, min_periods=14).mean()
        rs = avg_gain / avg_loss.replace(0, np.inf)
        result[f'rsi_14{tf_suffix}'] = 100 - (100 / (1 + rs))

        # ATR
        high_low = result['high'] - result['low']
        high_close = (result['high'] - result['close'].shift(1)).abs()
        low_close = (result['low'] - result['close'].shift(1)).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        result[f'atr_14{tf_suffix}'] = tr.rolling(14, min_periods=14).mean()

        # Bollinger Band Position
        bb_period = 20
        bb_middle = result['close'].rolling(bb_period, min_periods=bb_period).mean()
        bb_std = result['close'].rolling(bb_period, min_periods=bb_period).std()
        bb_upper = bb_middle + 2 * bb_std
        bb_lower = bb_middle - 2 * bb_std
        band_range = bb_upper - bb_lower
        band_range_safe = band_range.replace(0, np.nan)
        result[f'bb_position{tf_suffix}'] = (
            (result['close'] - bb_lower) / band_range_safe
        )

        # MACD Histogram
        ema_12 = result['close'].ewm(span=12, min_periods=12, adjust=False).mean()
        ema_26 = result['close'].ewm(span=26, min_periods=26, adjust=False).mean()
        macd_line = ema_12 - ema_26
        signal_line = macd_line.ewm(span=9, min_periods=9, adjust=False).mean()
        result[f'macd_hist{tf_suffix}'] = macd_line - signal_line

        # Price to SMA Ratio
        sma_20 = result[f'sma_20{tf_suffix}']
        sma_20_safe = sma_20.replace(0, np.nan)
        result[f'close_sma20_ratio{tf_suffix}'] = result['close'] / sma_20_safe

        return result

    def align_to_base_tf(
        self,
        df_base: pd.DataFrame,
        df_mtf: pd.DataFrame,
        mtf_columns: List[str]
    ) -> pd.DataFrame:
        """
        Align MTF features to base timeframe using forward-fill.

        CRITICAL: Uses shift(1) on MTF data to prevent lookahead bias.
        A 4h bar at 12:00 represents 08:00-12:00, so we must shift by 1
        to ensure 5min bars from 12:00-16:00 only see the completed
        08:00-12:00 bar, not the current in-progress 12:00-16:00 bar.

        Parameters
        ----------
        df_base : pd.DataFrame
            Base timeframe data with 'datetime' column
        df_mtf : pd.DataFrame
            MTF data with 'datetime' column and indicator columns
        mtf_columns : List[str]
            List of MTF column names to align

        Returns
        -------
        pd.DataFrame
            Base timeframe data with MTF features aligned
        """
        if 'datetime' not in df_base.columns:
            raise ValueError("df_base must have 'datetime' column")
        if 'datetime' not in df_mtf.columns:
            raise ValueError("df_mtf must have 'datetime' column")

        df_base_idx = df_base.set_index('datetime').copy()
        df_mtf_idx = df_mtf.set_index('datetime')[mtf_columns].copy()

        # ANTI-LOOKAHEAD: Shift MTF data by 1 period
        # This ensures we use COMPLETED higher TF bars only
        df_mtf_shifted = df_mtf_idx.shift(1)

        # Forward-fill to base timeframe
        aligned = df_mtf_shifted.reindex(df_base_idx.index, method='ffill')

        result = df_base_idx.join(aligned)
        return result.reset_index()

    def generate_mtf_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate all MTF features for the input data.

        Parameters
        ----------
        df : pd.DataFrame
            Base timeframe OHLCV data with 'datetime' column

        Returns
        -------
        pd.DataFrame
            DataFrame with base data + all MTF features
        """
        validate_ohlcv_dataframe(df)

        if len(df) < MIN_BASE_BARS:
            raise ValueError(
                f"Insufficient data for MTF features: {len(df)} rows. "
                f"Need at least {MIN_BASE_BARS} rows."
            )

        result = df.copy()
        logger.info(
            f"Generating MTF features for {len(df)} rows "
            f"(mode={self.mode.value})"
        )

        for tf in self.mtf_timeframes:
            logger.info(f"Processing timeframe: {tf}")

            df_tf = self.resample_to_tf(df, tf)
            logger.info(f"  Resampled to {len(df_tf)} {tf} bars")

            if len(df_tf) < MIN_MTF_BARS:
                logger.warning(
                    f"  Insufficient {tf} bars ({len(df_tf)}). Skipping this timeframe."
                )
                continue

            mtf_columns = []
            tf_suffix = self._get_tf_suffix(tf)

            # Generate MTF bars if requested
            if self.mode in (MTFMode.BARS, MTFMode.BOTH):
                df_tf = self.generate_mtf_bars(df_tf, tf)
                bar_cols = [f'{col}{tf_suffix}' for col in REQUIRED_OHLCV_COLS]
                mtf_columns.extend(bar_cols)
                logger.info(f"  Added {len(bar_cols)} bar columns")

            # Generate MTF indicators if requested
            if self.mode in (MTFMode.INDICATORS, MTFMode.BOTH):
                df_tf = self.compute_mtf_indicators(df_tf, tf)
                indicator_cols = [
                    c for c in df_tf.columns
                    if c.endswith(tf_suffix) and c not in mtf_columns
                ]
                mtf_columns.extend(indicator_cols)
                logger.info(f"  Added {len(indicator_cols)} indicator columns")

            result = self.align_to_base_tf(result, df_tf, mtf_columns)
            logger.info(f"  Aligned {len(mtf_columns)} columns to base TF")

        total_mtf_cols = len(result.columns) - len(df.columns)
        logger.info(f"Total MTF features added: {total_mtf_cols}")

        return result

    def validate_no_lookahead(self, df: pd.DataFrame, verbose: bool = False) -> bool:
        """
        Validate that MTF features have no lookahead bias.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with MTF features
        verbose : bool, default False
            If True, log details about each column

        Returns
        -------
        bool
            True if validation passes
        """
        mtf_cols = []
        for tf in self.mtf_timeframes:
            tf_suffix = self._get_tf_suffix(tf)
            tf_cols = [c for c in df.columns if c.endswith(tf_suffix)]
            mtf_cols.extend(tf_cols)

        if not mtf_cols:
            logger.warning("No MTF columns found in DataFrame")
            return True

        for col in mtf_cols:
            first_valid_idx = df[col].first_valid_index()

            if first_valid_idx is None:
                if verbose:
                    logger.warning(f"Column {col} is entirely NaN")
                continue

            if first_valid_idx == 0:
                raise ValueError(
                    f"Potential lookahead in {col}: first row is not NaN. "
                    f"MTF features should have initial NaNs due to shift(1)."
                )

            if verbose:
                logger.info(f"Column {col}: first valid at index {first_valid_idx}")

        logger.info(f"Lookahead validation passed for {len(mtf_cols)} MTF columns")
        return True

    def get_mtf_column_names(self) -> Dict[str, List[str]]:
        """
        Get expected MTF column names for each timeframe.

        Returns
        -------
        Dict[str, List[str]]
            Mapping from timeframe to list of column names
        """
        result = {}

        for tf in self.mtf_timeframes:
            tf_suffix = self._get_tf_suffix(tf)
            cols = []

            # Add bar columns if mode includes bars
            if self.mode in (MTFMode.BARS, MTFMode.BOTH):
                cols.extend([f'{col}{tf_suffix}' for col in REQUIRED_OHLCV_COLS])

            # Add indicator columns if mode includes indicators
            if self.mode in (MTFMode.INDICATORS, MTFMode.BOTH):
                indicator_names = [
                    'sma_20', 'sma_50', 'ema_9', 'ema_21',
                    'rsi_14', 'atr_14', 'bb_position', 'macd_hist',
                    'close_sma20_ratio'
                ]
                cols.extend([f'{name}{tf_suffix}' for name in indicator_names])

            result[tf] = cols

        return result

    def get_bar_column_names(self) -> Dict[str, List[str]]:
        """
        Get expected MTF bar column names for each timeframe.

        Returns
        -------
        Dict[str, List[str]]
            Mapping from timeframe to list of bar column names
        """
        result = {}
        for tf in self.mtf_timeframes:
            tf_suffix = self._get_tf_suffix(tf)
            result[tf] = [f'{col}{tf_suffix}' for col in REQUIRED_OHLCV_COLS]
        return result

    def get_indicator_column_names(self) -> Dict[str, List[str]]:
        """
        Get expected MTF indicator column names for each timeframe.

        Returns
        -------
        Dict[str, List[str]]
            Mapping from timeframe to list of indicator column names
        """
        indicator_names = [
            'sma_20', 'sma_50', 'ema_9', 'ema_21',
            'rsi_14', 'atr_14', 'bb_position', 'macd_hist',
            'close_sma20_ratio'
        ]
        result = {}
        for tf in self.mtf_timeframes:
            tf_suffix = self._get_tf_suffix(tf)
            result[tf] = [f'{name}{tf_suffix}' for name in indicator_names]
        return result
