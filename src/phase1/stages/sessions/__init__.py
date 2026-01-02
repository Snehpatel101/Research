"""
Session Normalization System for Trading Sessions

This package provides comprehensive session handling for CME futures trading data:
- Session classification and filtering
- CME holiday calendar integration
- DST (Daylight Saving Time) handling
- Session-specific volatility normalization

Sessions:
- New York: 14:30-21:00 UTC (09:30-16:00 ET)
- London: 08:00-16:30 UTC
- Asia: 23:00-07:00 UTC (crosses midnight)

Usage:
    from stages.sessions import SessionFilter, SessionsConfig, CMECalendar

    # Create filter with configuration
    config = SessionsConfig(
        include_sessions=[SessionName.NEW_YORK, SessionName.LONDON],
        add_session_flags=True,
        add_overlap_flags=True,
        filter_holidays=True,
    )
    filter = SessionFilter(config)

    # Add session features
    df = filter.add_session_features(df)

    # Filter by session
    df = filter.filter_by_session(df)

    # Filter holidays
    calendar = CMECalendar()
    df = calendar.filter_holidays(df)

    # Normalize by session volatility
    from stages.sessions import SessionNormalizer
    normalizer = SessionNormalizer(config)
    df = normalizer.fit_transform(df, feature_cols)

Author: ML Pipeline
Created: 2025-12-22
"""

# Configuration
# Calendar
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

# Filter
from .filter import (
    SessionFilter,
    create_session_filter,
)

# Normalizer
from .normalizer import (
    NormalizationReport,
    SessionNormalizer,
    SessionVolatilityStats,
    get_session_volatility_ratios,
    normalize_by_session,
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
    # Filter
    "SessionFilter",
    "create_session_filter",
    # Calendar
    "TradingDayType",
    "TradingDay",
    "CME_HOLIDAYS",
    "CME_EARLY_CLOSE",
    "CMECalendar",
    "DSTHandler",
    "get_calendar",
    # Normalizer
    "SessionVolatilityStats",
    "NormalizationReport",
    "SessionNormalizer",
    "normalize_by_session",
    "get_session_volatility_ratios",
]
