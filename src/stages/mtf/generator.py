"""
MTFFeatureGenerator - Main class for Multi-Timeframe feature generation.

Computes features from higher timeframes and aligns them to the base timeframe
without lookahead bias.

Key Design Decisions:
1. ANTI-LOOKAHEAD: All MTF features use shift(1) on the higher TF data before
   alignment. This ensures we only use COMPLETED higher TF bars, not the
   current (incomplete) bar which would cause lookahead bias.

2. SUPPORTED TIMEFRAMES: 5min (base), 15min, 30min, 60min (1h).
   Only timeframes that are integer multiples of the base are supported.

3. FORWARD-FILL ALIGNMENT: Higher TF features are forward-filled to base TF,
   meaning each base TF bar uses the most recent completed higher TF bar's value.
"""

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .constants import (
    MTF_TIMEFRAMES,
    REQUIRED_OHLCV_COLS,
    DEFAULT_MTF_TIMEFRAMES,
    MIN_BASE_BARS,
    MIN_MTF_BARS,
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
    computes technical indicators on those higher timeframes,
    and aligns them back to the base timeframe without lookahead bias.

    Parameters
    ----------
    base_timeframe : str, default '5min'
        Base timeframe of input data
    mtf_timeframes : List[str], optional
        List of higher timeframes to compute features for.
        Default: ['15min', '60min']
    include_ohlcv : bool, default True
        Whether to include OHLCV data from higher TFs
    include_indicators : bool, default True
        Whether to compute indicators on higher TFs
    """

    def __init__(
        self,
        base_timeframe: str = '5min',
        mtf_timeframes: Optional[List[str]] = None,
        include_ohlcv: bool = True,
        include_indicators: bool = True
    ):
        """Initialize MTF feature generator with validation."""
        self.base_timeframe = base_timeframe
        self.mtf_timeframes = mtf_timeframes or DEFAULT_MTF_TIMEFRAMES
        self.include_ohlcv = include_ohlcv
        self.include_indicators = include_indicators

        # Validate and parse timeframes
        validate_mtf_timeframes(base_timeframe, self.mtf_timeframes)
        self.base_minutes = MTF_TIMEFRAMES[base_timeframe]

        logger.info(
            f"Initialized MTFFeatureGenerator: base={base_timeframe}, "
            f"mtf={self.mtf_timeframes}"
        )

    def _get_tf_suffix(self, tf: str) -> str:
        """Get column suffix for a timeframe."""
        minutes = MTF_TIMEFRAMES[tf]
        if minutes >= 60 and minutes % 60 == 0:
            hours = minutes // 60
            return f"_{hours}h"
        return f"_{minutes}m"

    def _get_pandas_freq(self, tf: str) -> str:
        """Convert timeframe string to pandas frequency string."""
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
        logger.info(f"Generating MTF features for {len(df)} rows")

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

            if self.include_ohlcv:
                for col in REQUIRED_OHLCV_COLS:
                    new_col = f'{col}{tf_suffix}'
                    df_tf[new_col] = df_tf[col]
                    mtf_columns.append(new_col)

            if self.include_indicators:
                df_tf = self.compute_mtf_indicators(df_tf, tf)
                indicator_cols = [
                    c for c in df_tf.columns
                    if c.endswith(tf_suffix) and c not in mtf_columns
                ]
                mtf_columns.extend(indicator_cols)

            result = self.align_to_base_tf(result, df_tf, mtf_columns)
            logger.info(f"  Aligned {len(mtf_columns)} columns")

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
        """Get expected MTF column names for each timeframe."""
        result = {}

        for tf in self.mtf_timeframes:
            tf_suffix = self._get_tf_suffix(tf)
            cols = []

            if self.include_ohlcv:
                cols.extend([f'{col}{tf_suffix}' for col in REQUIRED_OHLCV_COLS])

            if self.include_indicators:
                indicator_names = [
                    'sma_20', 'sma_50', 'ema_9', 'ema_21',
                    'rsi_14', 'atr_14', 'bb_position', 'macd_hist',
                    'close_sma20_ratio'
                ]
                cols.extend([f'{name}{tf_suffix}' for name in indicator_names])

            result[tf] = cols

        return result
