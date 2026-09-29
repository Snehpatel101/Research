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
from enum import StrEnum

import pandas as pd

try:
    import pytz

    PYTZ_AVAILABLE = True
except ImportError:
    pytz = None  # bound so guarded uses are well-defined
    PYTZ_AVAILABLE = False

try:
    from zoneinfo import ZoneInfo

    ZONEINFO_AVAILABLE = True
except ImportError:
    ZoneInfo = None  # type: ignore[assignment,misc]  # bound so guarded uses are well-defined
    ZONEINFO_AVAILABLE = False

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class TradingDayType(StrEnum):
    """Type of trading day."""

    REGULAR = "regular"
    EARLY_CLOSE = "early_close"
    HOLIDAY = "holiday"
    WEEKEND = "weekend"


@dataclass(frozen=True)
class TradingDay:
    """Information about a specific trading day."""

    date: date
    day_type: TradingDayType
    description: str = ""
    early_close_time: time | None = None  # UTC time if early close


# =============================================================================
# CME HOLIDAYS (2024-2026)
# =============================================================================
# CME Globex is closed on these days (no trading)
# Dates are updated annually - check CME Group website for updates

CME_HOLIDAYS: dict[int, list[date]] = {
    2024: [
        date(2024, 1, 1),  # New Year's Day
        date(2024, 1, 15),  # Martin Luther King Jr. Day
        date(2024, 2, 19),  # Presidents Day
        date(2024, 3, 29),  # Good Friday
        date(2024, 5, 27),  # Memorial Day
        date(2024, 6, 19),  # Juneteenth
        date(2024, 7, 4),  # Independence Day
        date(2024, 9, 2),  # Labor Day
        date(2024, 11, 28),  # Thanksgiving Day
        date(2024, 12, 25),  # Christmas Day
    ],
    2025: [
        date(2025, 1, 1),  # New Year's Day
        date(2025, 1, 20),  # Martin Luther King Jr. Day
        date(2025, 2, 17),  # Presidents Day
        date(2025, 4, 18),  # Good Friday
        date(2025, 5, 26),  # Memorial Day
        date(2025, 6, 19),  # Juneteenth
        date(2025, 7, 4),  # Independence Day
        date(2025, 9, 1),  # Labor Day
        date(2025, 11, 27),  # Thanksgiving Day
        date(2025, 12, 25),  # Christmas Day
    ],
    2026: [
        date(2026, 1, 1),  # New Year's Day
        date(2026, 1, 19),  # Martin Luther King Jr. Day
        date(2026, 2, 16),  # Presidents Day
        date(2026, 4, 3),  # Good Friday
        date(2026, 5, 25),  # Memorial Day
        date(2026, 6, 19),  # Juneteenth
        date(2026, 7, 3),  # Independence Day (observed, 7/4 is Saturday)
        date(2026, 9, 7),  # Labor Day
        date(2026, 11, 26),  # Thanksgiving Day
        date(2026, 12, 25),  # Christmas Day
    ],
}

# Early close days (trading ends early, typically 12:00 ET / 17:00 UTC)
CME_EARLY_CLOSE: dict[int, list[date]] = {
    2024: [
        date(2024, 7, 3),  # Day before Independence Day
        date(2024, 11, 29),  # Day after Thanksgiving
        date(2024, 12, 24),  # Christmas Eve
        date(2024, 12, 31),  # New Year's Eve
    ],
    2025: [
        date(2025, 7, 3),  # Day before Independence Day
        date(2025, 11, 28),  # Day after Thanksgiving
        date(2025, 12, 24),  # Christmas Eve
        date(2025, 12, 31),  # New Year's Eve
    ],
    2026: [
        date(2026, 7, 2),  # Day before observed Independence Day
        date(2026, 11, 27),  # Day after Thanksgiving
        date(2026, 12, 24),  # Christmas Eve
        date(2026, 12, 31),  # New Year's Eve
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
        self._holiday_set: set[date] = set()
        self._early_close_set: set[date] = set()

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

    def filter_holidays(self, df: pd.DataFrame, datetime_column: str = "datetime") -> pd.DataFrame:
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


class DSTHandler:
    """
    Handler for Daylight Saving Time transitions.

    DST affects session timing because:
    - US Eastern Time shifts between EST (UTC-5) and EDT (UTC-4)
    - This changes the UTC times for NY session
    - London also has BST transitions

    This class detects DST transitions and adjusts session boundaries accordingly.
    """

    def __init__(self, timezone: str = "America/New_York"):
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
                "Neither zoneinfo nor pytz is available. " "Install pytz or upgrade to Python 3.9+."
            )

    def get_dst_transition_dates(self, year: int) -> tuple[date | None, date | None]:
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
        if self.timezone_str != "America/New_York":
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


# Module-level calendar instance for convenience
_calendar: CMECalendar | None = None


def get_calendar() -> CMECalendar:
    """Get the shared CME calendar instance."""
    global _calendar
    if _calendar is None:
        _calendar = CMECalendar()
    return _calendar


__all__ = [
    "TradingDayType",
    "TradingDay",
    "CME_HOLIDAYS",
    "CME_EARLY_CLOSE",
    "CMECalendar",
    "DSTHandler",
    "get_calendar",
]
