"""
Gap Handler - Gap detection and filling for financial time series.

This module provides comprehensive gap handling functionality for OHLCV data:
- Gap detection with detailed reporting
- Forward fill and interpolation gap filling
- Configurable maximum gap limits
- Synthetic bar flagging for downstream processing
- Calendar-aware gap handling for futures markets

The GapHandler class is used internally by DataCleaner but can also be
used independently for specialized gap processing workflows.

Author: ML Pipeline
Created: 2025-12-22
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


# CME Globex E-mini futures trading hours (Central Time):
# Sunday 5:00 PM - Friday 4:00 PM CT with daily maintenance break 4:00-5:00 PM CT
# Weekend closure: Friday 4:00 PM CT - Sunday 5:00 PM CT
CME_MAINTENANCE_START_HOUR = 16  # 4:00 PM CT
CME_MAINTENANCE_END_HOUR = 17    # 5:00 PM CT
CME_WEEKEND_CLOSE_DAY = 4        # Friday (0=Monday)
CME_WEEKEND_OPEN_DAY = 6         # Sunday (0=Monday)


def is_cme_market_closed(
    dt: pd.Timestamp,
    tz: str = 'America/Chicago',
    assume_utc: bool = True
) -> bool:
    """
    Check if CME Globex is closed at the given timestamp.

    CME E-mini futures (MES, MGC, etc.) trade nearly 24 hours with:
    - Daily maintenance: 4:00 PM - 5:00 PM CT (Mon-Thu)
    - Weekend closure: Friday 4:00 PM CT - Sunday 5:00 PM CT

    Parameters
    ----------
    dt : pd.Timestamp
        Timestamp to check (can be timezone-aware or naive)
    tz : str
        Target timezone for the check (default: America/Chicago for Central Time)
    assume_utc : bool
        If True and timestamp is tz-naive, assume UTC before converting.
        If False and timestamp is tz-naive, assume already in target timezone.

    Returns
    -------
    bool
        True if market is closed, False if open
    """
    # Handle timezone conversion explicitly
    if dt.tzinfo is None:
        if assume_utc:
            # Naive timestamp assumed to be UTC - localize then convert
            dt = dt.tz_localize('UTC').tz_convert(tz)
        else:
            # Naive timestamp assumed to be in target timezone already
            dt = dt.tz_localize(tz)
    elif str(dt.tzinfo) != tz:
        # Already has timezone but not the target - convert
        dt = dt.tz_convert(tz)

    hour = dt.hour
    weekday = dt.weekday()  # 0=Monday, 6=Sunday

    # Weekend closure check: Friday 4pm to Sunday 5pm
    if weekday == CME_WEEKEND_CLOSE_DAY and hour >= CME_MAINTENANCE_START_HOUR:
        # Friday after 4pm - market closed
        return True
    if weekday == 5:
        # Saturday - market closed all day
        return True
    if weekday == CME_WEEKEND_OPEN_DAY and hour < CME_MAINTENANCE_END_HOUR:
        # Sunday before 5pm - market closed
        return True

    # Daily maintenance break: 4:00 PM - 5:00 PM CT
    if hour == CME_MAINTENANCE_START_HOUR:
        return True

    return False


def is_expected_gap(
    gap_start: pd.Timestamp,
    gap_end: pd.Timestamp,
    tz: str = 'America/Chicago',
    assume_utc: bool = True
) -> bool:
    """
    Check if a gap is expected due to market closure.

    A gap is expected if it spans a known market closure period
    (weekend or daily maintenance).

    Parameters
    ----------
    gap_start : pd.Timestamp
        Start of the gap (last bar before gap)
    gap_end : pd.Timestamp
        End of the gap (first bar after gap)
    tz : str
        Target timezone for the check (default: America/Chicago for Central Time)
    assume_utc : bool
        If True and timestamps are tz-naive, assume UTC before converting.
        If False and timestamps are tz-naive, assume already in target timezone.

    Returns
    -------
    bool
        True if gap is expected (spans market closure), False otherwise
    """
    # Convert gap_start to target timezone
    if gap_start.tzinfo is None:
        if assume_utc:
            gap_start = gap_start.tz_localize('UTC').tz_convert(tz)
        else:
            gap_start = gap_start.tz_localize(tz)
    elif str(gap_start.tzinfo) != tz:
        gap_start = gap_start.tz_convert(tz)

    # Convert gap_end to target timezone
    if gap_end.tzinfo is None:
        if assume_utc:
            gap_end = gap_end.tz_localize('UTC').tz_convert(tz)
        else:
            gap_end = gap_end.tz_localize(tz)
    elif str(gap_end.tzinfo) != tz:
        gap_end = gap_end.tz_convert(tz)

    # Check if gap spans a weekend
    start_weekday = gap_start.weekday()
    end_weekday = gap_end.weekday()

    # Friday to Sunday/Monday gap
    if start_weekday == CME_WEEKEND_CLOSE_DAY and gap_start.hour >= CME_MAINTENANCE_START_HOUR:
        if end_weekday in (CME_WEEKEND_OPEN_DAY, 0):  # Sunday or Monday
            return True

    # Check if gap spans daily maintenance (4-5pm CT)
    if gap_start.hour <= CME_MAINTENANCE_START_HOUR and gap_end.hour >= CME_MAINTENANCE_END_HOUR:
        if gap_start.date() == gap_end.date():
            return True

    return False


class GapHandler:
    """
    Handles gap detection and filling for financial time series data.

    This class encapsulates all gap-related functionality, providing
    configurable gap detection and multiple filling strategies.

    Parameters:
    -----------
    freq_minutes : int
        Expected bar frequency in minutes (e.g., 1 for 1-minute bars)
    gap_fill_method : str
        Method for filling gaps: 'forward', 'interpolate', or 'none'
    max_gap_fill_minutes : int
        Maximum gap size in minutes that will be filled

    Examples:
    ---------
    >>> handler = GapHandler(freq_minutes=1, gap_fill_method='forward')
    >>> df, report = handler.detect_gaps(df)
    >>> df = handler.fill_gaps(df)
    """

    def __init__(
        self,
        freq_minutes: int = 1,
        gap_fill_method: str = 'forward',
        max_gap_fill_minutes: int = 5,
        calendar_aware: bool = True,
        timezone: str = 'America/Chicago'
    ):
        """
        Initialize the gap handler.

        Parameters:
        -----------
        freq_minutes : int
            Expected bar frequency in minutes
        gap_fill_method : str
            Gap filling method: 'forward', 'interpolate', or 'none'
        max_gap_fill_minutes : int
            Maximum gap to fill in minutes
        calendar_aware : bool
            If True, exclude expected market closures from gap detection
        timezone : str
            Timezone for calendar-aware gap detection (default: America/Chicago for CME)
        """
        self.freq_minutes = freq_minutes
        self.gap_fill_method = gap_fill_method
        self.max_gap_fill_minutes = max_gap_fill_minutes
        self.calendar_aware = calendar_aware
        self.timezone = timezone

    def detect_gaps(self, df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
        """
        Detect gaps in time series.

        Analyzes the datetime column to find missing bars based on the
        expected frequency. Returns both the processed DataFrame and a
        detailed report of gaps found.

        Parameters:
        -----------
        df : pd.DataFrame
            Input DataFrame with datetime index. Must have 'datetime' column.

        Returns:
        --------
        tuple : (DataFrame with gap info, gap report dict)
            - DataFrame: sorted by datetime with time_diff column removed
            - gap_report: dict containing:
                - total_gaps: number of gap periods found
                - total_missing_bars: estimated total missing bars
                - expected_bars: theoretical bars for date range
                - actual_bars: actual bars in dataset
                - completeness_pct: data completeness percentage
                - gaps: list of gap details (limited to first 100)

        Examples:
        ---------
        >>> handler = GapHandler(freq_minutes=1)
        >>> df, report = handler.detect_gaps(df)
        >>> print(f"Found {report['total_gaps']} gaps")
        >>> print(f"Completeness: {report['completeness_pct']:.2f}%")
        """
        logger.info("Detecting gaps in time series...")

        df = df.copy()
        df = df.sort_values('datetime').reset_index(drop=True)

        # Calculate time differences
        df['time_diff'] = df['datetime'].diff()

        # Expected frequency
        expected_freq = pd.Timedelta(minutes=self.freq_minutes)

        # Identify gaps (where time_diff > expected_freq)
        gap_mask = df['time_diff'] > expected_freq

        gaps = []
        unexpected_gaps = []
        expected_gap_count = 0

        if gap_mask.any():
            gap_indices = np.where(gap_mask)[0]

            for idx in gap_indices:
                gap_start = df.loc[idx - 1, 'datetime']
                gap_end = df.loc[idx, 'datetime']
                gap_duration = df.loc[idx, 'time_diff']
                missing_bars = int(gap_duration / expected_freq) - 1

                # Check if this gap is expected (market closure)
                gap_is_expected = False
                if self.calendar_aware:
                    gap_is_expected = is_expected_gap(gap_start, gap_end, self.timezone)

                gap_info = {
                    'gap_start': gap_start,
                    'gap_end': gap_end,
                    'duration': str(gap_duration),
                    'missing_bars': missing_bars,
                    'expected': gap_is_expected
                }

                gaps.append(gap_info)

                if gap_is_expected:
                    expected_gap_count += 1
                else:
                    unexpected_gaps.append(gap_info)

        # Gap report
        total_expected_bars = (df['datetime'].max() - df['datetime'].min()) / expected_freq
        total_actual_bars = len(df)
        total_missing_bars = int(total_expected_bars - total_actual_bars)

        gap_report = {
            'total_gaps': len(gaps),
            'expected_gaps': expected_gap_count,
            'unexpected_gaps': len(unexpected_gaps),
            'total_missing_bars': total_missing_bars,
            'expected_bars': int(total_expected_bars),
            'actual_bars': total_actual_bars,
            'completeness_pct': (total_actual_bars / total_expected_bars * 100) if total_expected_bars > 0 else 100,
            'gaps': gaps[:100],  # Limit to first 100 gaps for reporting
            'unexpected_gap_details': unexpected_gaps[:20]  # First 20 unexpected gaps
        }

        logger.info(f"Found {len(gaps)} total gaps ({expected_gap_count} expected, {len(unexpected_gaps)} unexpected)")
        logger.info(f"Total missing bars: {total_missing_bars:,}")
        logger.info(f"Completeness: {gap_report['completeness_pct']:.2f}%")

        # Drop temp column
        df = df.drop('time_diff', axis=1)

        return df, gap_report

    def fill_gaps(self, df: pd.DataFrame, max_fill_bars: int | None = None) -> pd.DataFrame:
        """
        Fill gaps in time series.

        Creates a complete datetime range and fills missing values using
        the configured method. Marks synthetic bars with a 'filled' column
        so downstream processing can distinguish real vs filled data.

        Parameters:
        -----------
        df : pd.DataFrame
            Input DataFrame with OHLCV data and 'datetime' column
        max_fill_bars : int, optional
            Maximum number of bars to fill. If None, uses max_gap_fill_minutes
            divided by freq_minutes.

        Returns:
        --------
        pd.DataFrame
            DataFrame with filled gaps. Includes:
            - 'filled' column: 1 for synthetic bars, 0 for original data
            - All original OHLCV columns
            - Rows with gaps larger than max_fill_bars are dropped

        Notes:
        ------
        - 'forward' method: Uses previous bar's OHLC values, volume set to 0
        - 'interpolate' method: Linear interpolation of OHLC, volume set to 0
        - 'none' method: Returns DataFrame unchanged

        Examples:
        ---------
        >>> handler = GapHandler(freq_minutes=1, gap_fill_method='forward')
        >>> df = handler.fill_gaps(df)
        >>> synthetic_count = df['filled'].sum()
        >>> print(f"Added {synthetic_count} synthetic bars")
        """
        if self.gap_fill_method == 'none':
            logger.info("Gap filling disabled")
            return df

        logger.info(f"Filling gaps using method: {self.gap_fill_method}")

        df = df.copy()
        df = df.sort_values('datetime').reset_index(drop=True)

        if max_fill_bars is None:
            max_fill_bars = max(1, self.max_gap_fill_minutes // self.freq_minutes)

        # Create complete datetime range
        date_range = pd.date_range(
            start=df['datetime'].min(),
            end=df['datetime'].max(),
            freq=f'{self.freq_minutes}min'
        )

        # Reindex to complete range
        df_complete = df.set_index('datetime').reindex(date_range).reset_index()
        df_complete = df_complete.rename(columns={'index': 'datetime'})

        # If calendar-aware, mark bars during market closure for removal
        # These should NOT be filled - they are expected to be missing
        if self.calendar_aware:
            market_closed_mask = df_complete['datetime'].apply(
                lambda dt: is_cme_market_closed(dt, self.timezone)
            )
            # Remove rows during market closure from the date range
            df_complete = df_complete[~market_closed_mask].reset_index(drop=True)

        # Identify which rows will be filled (before filling)
        # This flags synthetic bars so models can distinguish real vs filled data
        filled_mask = df_complete['close'].isna()
        n_filled = filled_mask.sum()

        # Store filled flag before any filling operations
        df_complete['filled'] = filled_mask.astype(int)

        if n_filled > 0:
            logger.info(f"Filling {n_filled:,} missing bars (within trading sessions only)...")

            if self.gap_fill_method == 'forward':
                # Forward fill with limit
                df_complete[['open', 'high', 'low', 'close']] = \
                    df_complete[['open', 'high', 'low', 'close']].ffill(limit=max_fill_bars)
                df_complete['volume'] = df_complete['volume'].fillna(0)

            elif self.gap_fill_method == 'interpolate':
                # Linear interpolation
                df_complete[['open', 'high', 'low', 'close']] = \
                    df_complete[['open', 'high', 'low', 'close']].interpolate(method='linear', limit=max_fill_bars)
                df_complete['volume'] = df_complete['volume'].fillna(0)

            # Copy symbol column if exists
            if 'symbol' in df_complete.columns:
                df_complete['symbol'] = df_complete['symbol'].ffill()

            # Drop any remaining NaN rows (gaps too large to fill)
            remaining_na = df_complete['close'].isna().sum()
            if remaining_na > 0:
                logger.info(f"Dropping {remaining_na} rows with gaps > {max_fill_bars} bars")
                df_complete = df_complete.dropna(subset=['close'])

        return df_complete


def create_gap_handler(
    freq_minutes: int = 1,
    gap_fill_method: str = 'forward',
    max_gap_fill_minutes: int = 5,
    calendar_aware: bool = True,
    timezone: str = 'America/Chicago'
) -> GapHandler:
    """
    Factory function to create a GapHandler instance.

    This provides a convenient way to create handlers with validated parameters.

    Parameters:
    -----------
    freq_minutes : int
        Expected bar frequency in minutes
    gap_fill_method : str
        Gap filling method: 'forward', 'interpolate', or 'none'
    max_gap_fill_minutes : int
        Maximum gap to fill in minutes
    calendar_aware : bool
        If True, exclude expected market closures from gap detection
    timezone : str
        Timezone for calendar-aware gap detection

    Returns:
    --------
    GapHandler : Configured gap handler instance

    Raises:
    -------
    ValueError : If gap_fill_method is not valid

    Examples:
    ---------
    >>> handler = create_gap_handler(freq_minutes=5, gap_fill_method='interpolate')
    >>> handler = create_gap_handler(calendar_aware=False)  # Disable calendar awareness
    """
    valid_methods = ('forward', 'interpolate', 'none')
    if gap_fill_method not in valid_methods:
        raise ValueError(
            f"Invalid gap_fill_method: '{gap_fill_method}'. "
            f"Must be one of: {valid_methods}"
        )

    return GapHandler(
        freq_minutes=freq_minutes,
        gap_fill_method=gap_fill_method,
        max_gap_fill_minutes=max_gap_fill_minutes,
        calendar_aware=calendar_aware,
        timezone=timezone
    )
