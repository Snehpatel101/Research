"""
CME Holiday Calendar and DST Handling

This module provides:
- CME market holiday dates
- DST transition detection for US Eastern time
- Partial trading day detection (early close days)
- Date validation utilities

CME Futures holidays (2024-2026):
- New Year's Day (or observed)
- Martin Luther King Jr. Day
- Presidents Day
- Good Friday
- Memorial Day
- Juneteenth (observed)
- Independence Day (or observed)
- Labor Day
- Thanksgiving Day
- Christmas Day (or observed)

Author: ML Pipeline
Created: 2025-12-22
"""

import logging
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

try:
    import pytz
    PYTZ_AVAILABLE = True
except ImportError:
    PYTZ_AVAILABLE = False

try:
    from zoneinfo import ZoneInfo
    ZONEINFO_AVAILABLE = True
except ImportError:
    ZONEINFO_AVAILABLE = False

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class TradingDayType(Enum):
    """Type of trading day."""
    REGULAR = 'regular'
    EARLY_CLOSE = 'early_close'
    HOLIDAY = 'holiday'
    WEEKEND = 'weekend'


@dataclass(frozen=True)
class TradingDay:
    """Information about a specific trading day."""
    date: date
    day_type: TradingDayType
    description: str = ""
    early_close_time: Optional[time] = None  # UTC time if early close


# =============================================================================
# CME HOLIDAYS (2024-2026)
# =============================================================================
# CME Globex is closed on these days (no trading)
# Dates are updated annually - check CME Group website for updates

CME_HOLIDAYS: Dict[int, List[date]] = {
    2024: [
        date(2024, 1, 1),   # New Year's Day
        date(2024, 1, 15),  # Martin Luther King Jr. Day
        date(2024, 2, 19),  # Presidents Day
        date(2024, 3, 29),  # Good Friday
        date(2024, 5, 27),  # Memorial Day
        date(2024, 6, 19),  # Juneteenth
        date(2024, 7, 4),   # Independence Day
        date(2024, 9, 2),   # Labor Day
        date(2024, 11, 28), # Thanksgiving Day
        date(2024, 12, 25), # Christmas Day
    ],
    2025: [
        date(2025, 1, 1),   # New Year's Day
        date(2025, 1, 20),  # Martin Luther King Jr. Day
        date(2025, 2, 17),  # Presidents Day
        date(2025, 4, 18),  # Good Friday
        date(2025, 5, 26),  # Memorial Day
        date(2025, 6, 19),  # Juneteenth
        date(2025, 7, 4),   # Independence Day
        date(2025, 9, 1),   # Labor Day
        date(2025, 11, 27), # Thanksgiving Day
        date(2025, 12, 25), # Christmas Day
    ],
    2026: [
        date(2026, 1, 1),   # New Year's Day
        date(2026, 1, 19),  # Martin Luther King Jr. Day
        date(2026, 2, 16),  # Presidents Day
        date(2026, 4, 3),   # Good Friday
        date(2026, 5, 25),  # Memorial Day
        date(2026, 6, 19),  # Juneteenth
        date(2026, 7, 3),   # Independence Day (observed, 7/4 is Saturday)
        date(2026, 9, 7),   # Labor Day
        date(2026, 11, 26), # Thanksgiving Day
        date(2026, 12, 25), # Christmas Day
    ],
}

# Early close days (trading ends early, typically 12:00 ET / 17:00 UTC)
CME_EARLY_CLOSE: Dict[int, List[date]] = {
    2024: [
        date(2024, 7, 3),   # Day before Independence Day
        date(2024, 11, 29), # Day after Thanksgiving
        date(2024, 12, 24), # Christmas Eve
        date(2024, 12, 31), # New Year's Eve
    ],
    2025: [
        date(2025, 7, 3),   # Day before Independence Day
        date(2025, 11, 28), # Day after Thanksgiving
        date(2025, 12, 24), # Christmas Eve
        date(2025, 12, 31), # New Year's Eve
    ],
    2026: [
        date(2026, 7, 2),   # Day before observed Independence Day
        date(2026, 11, 27), # Day after Thanksgiving
        date(2026, 12, 24), # Christmas Eve
        date(2026, 12, 31), # New Year's Eve
    ],
}


class CMECalendar:
    """
    CME market calendar for holiday and trading day detection.

    This class provides methods to:
    - Check if a date is a CME holiday
    - Check if a date is an early close day
    - Get trading day information
    - Filter DataFrames by trading days
    """

    def __init__(self):
        """Initialize CME calendar."""
        self._holiday_set: Set[date] = set()
        self._early_close_set: Set[date] = set()

        # Build lookup sets
        for year_holidays in CME_HOLIDAYS.values():
            self._holiday_set.update(year_holidays)

        for year_early_close in CME_EARLY_CLOSE.values():
            self._early_close_set.update(year_early_close)

    def is_holiday(self, dt: date) -> bool:
        """
        Check if a date is a CME holiday.

        Args:
            dt: Date to check (date or datetime)

        Returns:
            True if the date is a CME holiday
        """
        if isinstance(dt, datetime):
            dt = dt.date()
        return dt in self._holiday_set

    def is_early_close(self, dt: date) -> bool:
        """
        Check if a date is an early close day.

        Args:
            dt: Date to check (date or datetime)

        Returns:
            True if the date is an early close day
        """
        if isinstance(dt, datetime):
            dt = dt.date()
        return dt in self._early_close_set

    def is_weekend(self, dt: date) -> bool:
        """
        Check if a date is a weekend.

        Args:
            dt: Date to check (date or datetime)

        Returns:
            True if the date is Saturday or Sunday
        """
        if isinstance(dt, datetime):
            dt = dt.date()
        return dt.weekday() >= 5  # Saturday = 5, Sunday = 6

    def get_trading_day_type(self, dt: date) -> TradingDayType:
        """
        Get the trading day type for a date.

        Args:
            dt: Date to check

        Returns:
            TradingDayType enum value
        """
        if isinstance(dt, datetime):
            dt = dt.date()

        if self.is_weekend(dt):
            return TradingDayType.WEEKEND
        if self.is_holiday(dt):
            return TradingDayType.HOLIDAY
        if self.is_early_close(dt):
            return TradingDayType.EARLY_CLOSE
        return TradingDayType.REGULAR

    def is_trading_day(self, dt: date) -> bool:
        """
        Check if a date is a trading day (not holiday or weekend).

        Args:
            dt: Date to check

        Returns:
            True if trading is expected on this day
        """
        day_type = self.get_trading_day_type(dt)
        return day_type in (TradingDayType.REGULAR, TradingDayType.EARLY_CLOSE)

    def get_holidays_in_range(
        self,
        start: date,
        end: date
    ) -> List[date]:
        """
        Get all CME holidays in a date range.

        Args:
            start: Start date (inclusive)
            end: End date (inclusive)

        Returns:
            List of holiday dates in the range
        """
        return sorted([
            h for h in self._holiday_set
            if start <= h <= end
        ])

    def get_trading_days_in_range(
        self,
        start: date,
        end: date
    ) -> List[date]:
        """
        Get all trading days in a date range.

        Args:
            start: Start date (inclusive)
            end: End date (inclusive)

        Returns:
            List of trading day dates
        """
        current = start
        trading_days = []

        while current <= end:
            if self.is_trading_day(current):
                trading_days.append(current)
            current += timedelta(days=1)

        return trading_days

    def filter_holidays(
        self,
        df: pd.DataFrame,
        datetime_column: str = 'datetime'
    ) -> pd.DataFrame:
        """
        Filter out rows that fall on CME holidays.

        Args:
            df: Input DataFrame
            datetime_column: Name of datetime column

        Returns:
            DataFrame with holiday rows removed
        """
        if datetime_column not in df.columns:
            raise ValueError(f"Column '{datetime_column}' not found in DataFrame")

        dates = df[datetime_column].dt.date
        mask = ~dates.isin(self._holiday_set)

        original_len = len(df)
        result = df.loc[mask].copy()

        logger.info(
            f"Filtered holidays: {original_len} -> {len(result)} rows "
            f"({original_len - len(result)} holiday rows removed)"
        )

        return result

    def add_trading_day_features(
        self,
        df: pd.DataFrame,
        datetime_column: str = 'datetime',
        feature_metadata: Optional[Dict[str, str]] = None
    ) -> pd.DataFrame:
        """
        Add trading day type features to DataFrame.

        Args:
            df: Input DataFrame
            datetime_column: Name of datetime column
            feature_metadata: Optional dict to store feature descriptions

        Returns:
            DataFrame with trading day features added
        """
        if datetime_column not in df.columns:
            raise ValueError(f"Column '{datetime_column}' not found in DataFrame")

        df = df.copy()
        dates = df[datetime_column].dt.date

        # Add is_early_close flag
        df['is_early_close'] = dates.isin(self._early_close_set).astype(int)

        if feature_metadata is not None:
            feature_metadata['is_early_close'] = "CME early close trading day flag"

        return df


class DSTHandler:
    """
    Handler for Daylight Saving Time transitions.

    DST affects session timing because:
    - US Eastern Time shifts between EST (UTC-5) and EDT (UTC-4)
    - This changes the UTC times for NY session
    - London also has BST transitions

    This class detects DST transitions and adjusts session boundaries accordingly.
    """

    def __init__(self, timezone: str = 'America/New_York'):
        """
        Initialize DST handler.

        Args:
            timezone: IANA timezone string

        Raises:
            RuntimeError: If neither pytz nor zoneinfo is available
        """
        self.timezone_str = timezone
        self._tz = self._get_timezone(timezone)

    def _get_timezone(self, timezone: str):
        """Get timezone object from string."""
        if ZONEINFO_AVAILABLE:
            return ZoneInfo(timezone)
        elif PYTZ_AVAILABLE:
            return pytz.timezone(timezone)
        else:
            raise RuntimeError(
                "Neither zoneinfo nor pytz is available. "
                "Install pytz or upgrade to Python 3.9+."
            )

    def is_dst(self, dt: datetime) -> bool:
        """
        Check if DST is in effect for a datetime.

        Args:
            dt: Datetime to check (timezone-naive assumed UTC)

        Returns:
            True if DST is in effect
        """
        if dt.tzinfo is None:
            # Assume UTC, convert to local
            if ZONEINFO_AVAILABLE:
                from datetime import timezone as dt_tz
                dt_utc = dt.replace(tzinfo=dt_tz.utc)
                dt_local = dt_utc.astimezone(self._tz)
            elif PYTZ_AVAILABLE:
                dt_utc = pytz.UTC.localize(dt)
                dt_local = dt_utc.astimezone(self._tz)
            else:
                return False
        else:
            dt_local = dt.astimezone(self._tz)

        # Check if DST is active
        if PYTZ_AVAILABLE and hasattr(dt_local, 'dst'):
            dst_offset = dt_local.dst()
            return dst_offset is not None and dst_offset.total_seconds() > 0

        # For zoneinfo, check the tzname
        tzname = dt_local.strftime('%Z')
        # EDT, BST, etc. indicate DST
        return tzname in ('EDT', 'BST', 'CEST', 'CDT', 'PDT', 'MDT')

    def get_utc_offset_hours(self, dt: datetime) -> float:
        """
        Get the UTC offset in hours for a datetime.

        Args:
            dt: Datetime to check (timezone-naive assumed UTC)

        Returns:
            UTC offset in hours (e.g., -4 for EDT, -5 for EST)
        """
        if dt.tzinfo is None:
            if ZONEINFO_AVAILABLE:
                from datetime import timezone as dt_tz
                dt_utc = dt.replace(tzinfo=dt_tz.utc)
                dt_local = dt_utc.astimezone(self._tz)
            elif PYTZ_AVAILABLE:
                dt_utc = pytz.UTC.localize(dt)
                dt_local = dt_utc.astimezone(self._tz)
            else:
                return 0.0
        else:
            dt_local = dt.astimezone(self._tz)

        offset = dt_local.utcoffset()
        if offset is None:
            return 0.0
        return offset.total_seconds() / 3600

    def get_dst_transition_dates(
        self,
        year: int
    ) -> Tuple[Optional[date], Optional[date]]:
        """
        Get DST transition dates for a given year.

        For US Eastern Time:
        - Spring forward: Second Sunday of March
        - Fall back: First Sunday of November

        Args:
            year: Year to get transitions for

        Returns:
            Tuple of (spring_forward_date, fall_back_date)
        """
        if self.timezone_str != 'America/New_York':
            # Only US Eastern implemented for now
            return (None, None)

        # Spring forward: Second Sunday of March
        march_first = date(year, 3, 1)
        days_until_sunday = (6 - march_first.weekday()) % 7
        first_sunday = march_first + timedelta(days=days_until_sunday)
        spring_forward = first_sunday + timedelta(days=7)

        # Fall back: First Sunday of November
        nov_first = date(year, 11, 1)
        days_until_sunday = (6 - nov_first.weekday()) % 7
        fall_back = nov_first + timedelta(days=days_until_sunday)

        return (spring_forward, fall_back)

    def is_dst_transition_date(self, dt: date) -> bool:
        """
        Check if a date is a DST transition date.

        Args:
            dt: Date to check

        Returns:
            True if this is a DST transition date
        """
        if isinstance(dt, datetime):
            dt = dt.date()

        spring, fall = self.get_dst_transition_dates(dt.year)

        return dt == spring or dt == fall

    def adjust_session_times_for_dst(
        self,
        base_start_utc: Tuple[int, int],
        base_end_utc: Tuple[int, int],
        dt: datetime
    ) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """
        Adjust session times for DST.

        If DST is in effect, session times shift by 1 hour earlier in UTC.

        Args:
            base_start_utc: Base session start (hour, minute) in UTC (winter time)
            base_end_utc: Base session end (hour, minute) in UTC (winter time)
            dt: Datetime to check DST status

        Returns:
            Adjusted (start_utc, end_utc) tuple
        """
        if not self.is_dst(dt):
            return (base_start_utc, base_end_utc)

        # DST is active - shift 1 hour earlier
        adj_start = ((base_start_utc[0] - 1) % 24, base_start_utc[1])
        adj_end = ((base_end_utc[0] - 1) % 24, base_end_utc[1])

        return (adj_start, adj_end)

    def add_dst_features(
        self,
        df: pd.DataFrame,
        datetime_column: str = 'datetime',
        feature_metadata: Optional[Dict[str, str]] = None
    ) -> pd.DataFrame:
        """
        Add DST-related features to DataFrame.

        Args:
            df: Input DataFrame
            datetime_column: Name of datetime column
            feature_metadata: Optional dict to store feature descriptions

        Returns:
            DataFrame with DST features added
        """
        if datetime_column not in df.columns:
            raise ValueError(f"Column '{datetime_column}' not found in DataFrame")

        df = df.copy()

        # Vectorized DST detection
        is_dst_list = [self.is_dst(dt) for dt in df[datetime_column]]
        df['is_dst'] = np.array(is_dst_list, dtype=int)

        # DST transition detection
        dates = df[datetime_column].dt.date
        is_transition = [self.is_dst_transition_date(d) for d in dates]
        df['is_dst_transition'] = np.array(is_transition, dtype=int)

        if feature_metadata is not None:
            feature_metadata['is_dst'] = f"Daylight Saving Time active ({self.timezone_str})"
            feature_metadata['is_dst_transition'] = "DST transition date flag"

        return df


# Module-level calendar instance for convenience
_calendar: Optional[CMECalendar] = None


def get_calendar() -> CMECalendar:
    """Get the shared CME calendar instance."""
    global _calendar
    if _calendar is None:
        _calendar = CMECalendar()
    return _calendar


__all__ = [
    'TradingDayType',
    'TradingDay',
    'CME_HOLIDAYS',
    'CME_EARLY_CLOSE',
    'CMECalendar',
    'DSTHandler',
    'get_calendar',
]
