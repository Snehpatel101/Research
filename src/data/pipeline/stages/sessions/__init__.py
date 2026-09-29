"""
Trading session configuration and the CME holiday calendar.

- Session definitions (New York, London, Asia) with DST handling
- CME holiday / early-close calendar (used by the backtester's execution model)

Usage:
    from src.data.pipeline.stages.sessions import CMECalendar, SessionName

    calendar = CMECalendar()
    df = calendar.filter_holidays(df)
"""

from .calendar import (
    CME_EARLY_CLOSE,
    CME_HOLIDAYS,
    CMECalendar,
    DSTHandler,
    TradingDay,
    TradingDayType,
    get_calendar,
)
from .config import (
    DEFAULT_SESSIONS_CONFIG,
    SESSION_OVERLAPS,
    SESSIONS,
    SessionConfig,
    SessionName,
    SessionOverlap,
    SessionsConfig,
    get_all_sessions,
    get_session_config,
)

__all__ = [
    # Config
    "SessionName",
    "SessionConfig",
    "SessionOverlap",
    "SessionsConfig",
    "SESSIONS",
    "SESSION_OVERLAPS",
    "DEFAULT_SESSIONS_CONFIG",
    "get_session_config",
    "get_all_sessions",
    # Calendar
    "TradingDayType",
    "TradingDay",
    "CME_HOLIDAYS",
    "CME_EARLY_CLOSE",
    "CMECalendar",
    "DSTHandler",
    "get_calendar",
]
