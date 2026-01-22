"""
Canonical timeframe registry - single source of truth for all timeframe operations.

This module provides unified timeframe handling across the entire codebase:
- Supported timeframes for resampling
- Timeframe aliases (e.g., "1h" = "60min", "daily" = "1440min")
- Conversion utilities
- Normalization functions for consistent handling

All other modules should import from here instead of defining their own timeframe lists.

Usage:
    from src.core.common.timeframes import (
        normalize_timeframe,
        normalize_timeframe_list,
        get_timeframe_minutes,
        is_valid_timeframe,
        CANONICAL_TIMEFRAMES,
    )
"""

from __future__ import annotations

import warnings

# =============================================================================
# CANONICAL TIMEFRAME DEFINITIONS
# =============================================================================

# Primary 9-TF intraday ladder (canonical forms only, no aliases)
# This is the authoritative list for iteration and validation.
CANONICAL_TIMEFRAMES = [
    "1min",
    "5min",
    "10min",
    "15min",
    "20min",
    "25min",
    "30min",
    "45min",
    "60min",
]

# Full 9-TF ladder as used by --output-timeframes=9tf
FULL_9TF_LADDER = CANONICAL_TIMEFRAMES.copy()

# Extended timeframes (including multi-hour and daily)
EXTENDED_TIMEFRAMES = [
    "240min",  # 4h
    "1440min",  # daily
]

# All canonical timeframes (intraday + extended)
ALL_CANONICAL_TIMEFRAMES = CANONICAL_TIMEFRAMES + EXTENDED_TIMEFRAMES

# All supported timeframes (canonical + aliases for input validation)
# NOTE: Only canonical forms should be used for output/storage.
SUPPORTED_TIMEFRAMES = CANONICAL_TIMEFRAMES + [
    "1h",  # Alias for 60min (accepted on input, normalized internally)
    "4h",  # Alias for 240min
    "daily",  # Alias for 1440min
    "1d",  # Alias for 1440min
    "D",  # Pandas convention alias for daily
]

# =============================================================================
# ALIAS MAPPINGS (input alias -> canonical form)
# =============================================================================

# Aliases mapping to canonical form
# Keys are normalized to lowercase for matching
TIMEFRAME_ALIASES = {
    # Hourly aliases
    "1h": "60min",
    "60m": "60min",
    # Multi-hour aliases
    "4h": "240min",
    "240m": "240min",
    # Daily aliases
    "daily": "1440min",
    "1d": "1440min",
    "d": "1440min",
}

# =============================================================================
# TIMEFRAME TO MINUTES MAPPING
# =============================================================================

# Minutes for each timeframe (canonical forms)
TIMEFRAME_TO_MINUTES = {
    # Intraday (canonical 9-TF ladder)
    "1min": 1,
    "5min": 5,
    "10min": 10,
    "15min": 15,
    "20min": 20,
    "25min": 25,
    "30min": 30,
    "45min": 45,
    "60min": 60,
    # Extended
    "240min": 240,
    "1440min": 1440,
    # Aliases (for backward compatibility in lookups)
    "1h": 60,
    "4h": 240,
    "daily": 1440,
    "1d": 1440,
    "D": 1440,
}

# =============================================================================
# PANDAS FREQUENCY MAPPING
# =============================================================================

# Pandas frequency mapping (canonical forms)
TIMEFRAME_TO_FREQ = {
    "1min": "1min",
    "5min": "5min",
    "10min": "10min",
    "15min": "15min",
    "20min": "20min",
    "25min": "25min",
    "30min": "30min",
    "45min": "45min",
    "60min": "60min",
    "240min": "4h",
    "1440min": "D",
    # Aliases
    "1h": "1h",
    "4h": "4h",
    "daily": "D",
    "1d": "D",
    "D": "D",
}


def normalize_timeframe(timeframe: str, warn_on_alias: bool = False) -> str:
    """
    Normalize a timeframe string to its canonical form.

    This function converts any valid timeframe alias to its canonical
    representation. For example:
    - "1h" -> "60min"
    - "4h" -> "240min"
    - "daily", "1d", "D" -> "1440min"

    Parameters
    ----------
    timeframe : str
        Timeframe string (e.g., "1h", "60min", "daily")
    warn_on_alias : bool, default False
        If True, emit a deprecation warning when an alias is used

    Returns
    -------
    str
        Canonical timeframe (e.g., "60min" for "1h")

    Examples
    --------
    >>> normalize_timeframe("1h")
    '60min'
    >>> normalize_timeframe("daily")
    '1440min'
    >>> normalize_timeframe("15min")
    '15min'
    """
    tf = timeframe.lower().strip()

    # Check if it's an alias
    if tf in TIMEFRAME_ALIASES:
        canonical = TIMEFRAME_ALIASES[tf]
        if warn_on_alias:
            warnings.warn(
                f"Timeframe alias '{timeframe}' is deprecated. "
                f"Use canonical form '{canonical}' instead.",
                DeprecationWarning,
                stacklevel=2,
            )
        return canonical

    # Handle case variations (e.g., "1H" -> "1h" -> "60min")
    if tf != timeframe and tf in TIMEFRAME_ALIASES:
        return TIMEFRAME_ALIASES[tf]

    # Already canonical or unrecognized (return as-is for validation elsewhere)
    return timeframe


def normalize_timeframe_list(
    timeframes: list[str],
    deduplicate: bool = True,
    warn_on_alias: bool = False,
) -> list[str]:
    """
    Normalize a list of timeframes to canonical forms and optionally deduplicate.

    This function:
    1. Converts all aliases to canonical forms
    2. Optionally removes duplicates (preserving order)

    Parameters
    ----------
    timeframes : list[str]
        List of timeframe strings (may include aliases)
    deduplicate : bool, default True
        If True, remove duplicate timeframes after normalization
    warn_on_alias : bool, default False
        If True, emit deprecation warnings for aliases

    Returns
    -------
    list[str]
        List of canonical timeframes (deduplicated if requested)

    Examples
    --------
    >>> normalize_timeframe_list(["5min", "1h", "60min"])
    ['5min', '60min']
    >>> normalize_timeframe_list(["daily", "1d", "D"])
    ['1440min']
    >>> normalize_timeframe_list(["15min", "1h", "15min"])
    ['15min', '60min']
    """
    normalized = [normalize_timeframe(tf, warn_on_alias=warn_on_alias) for tf in timeframes]

    if deduplicate:
        # Preserve order while removing duplicates
        seen = set()
        result = []
        for tf in normalized:
            if tf not in seen:
                seen.add(tf)
                result.append(tf)
        return result

    return normalized


def get_timeframe_minutes(timeframe: str) -> int:
    """
    Get the number of minutes for a timeframe string.

    This is the centralized parser for timeframe-to-minutes conversion.
    It handles:
    - Canonical forms (e.g., "5min", "60min", "1440min")
    - Aliases (e.g., "1h", "4h", "daily")
    - Dynamic parsing (e.g., "2h", "120min")

    Parameters
    ----------
    timeframe : str
        Timeframe string (e.g., '5min', '1h', 'daily')

    Returns
    -------
    int
        Number of minutes

    Raises
    ------
    ValueError
        If the timeframe format is not recognized

    Examples
    --------
    >>> get_timeframe_minutes("5min")
    5
    >>> get_timeframe_minutes("1h")
    60
    >>> get_timeframe_minutes("daily")
    1440
    """
    # First try direct lookup (fastest path)
    if timeframe in TIMEFRAME_TO_MINUTES:
        return TIMEFRAME_TO_MINUTES[timeframe]

    # Normalize and try again
    tf = timeframe.lower().strip()
    if tf in TIMEFRAME_TO_MINUTES:
        return TIMEFRAME_TO_MINUTES[tf]

    # Try alias resolution
    if tf in TIMEFRAME_ALIASES:
        canonical = TIMEFRAME_ALIASES[tf]
        return TIMEFRAME_TO_MINUTES[canonical]

    # Dynamic parsing for non-standard formats
    if tf.endswith("min"):
        try:
            return int(tf[:-3])
        except ValueError:
            pass
    elif tf.endswith("m") and not tf.endswith("min"):
        # Handle "60m" style
        try:
            return int(tf[:-1])
        except ValueError:
            pass
    elif tf.endswith("h"):
        try:
            return int(tf[:-1]) * 60
        except ValueError:
            pass
    elif tf.endswith("d"):
        try:
            return int(tf[:-1]) * 1440
        except ValueError:
            pass

    raise ValueError(
        f"Unrecognized timeframe format: '{timeframe}'. "
        f"Expected formats: '5min', '1h', '4h', 'daily', etc."
    )


def is_valid_timeframe(timeframe: str, allow_extended: bool = True) -> bool:
    """
    Check if a timeframe string is valid (including aliases).

    Parameters
    ----------
    timeframe : str
        Timeframe string to check
    allow_extended : bool, default True
        If True, also accept extended timeframes (4h, daily)

    Returns
    -------
    bool
        True if valid timeframe

    Examples
    --------
    >>> is_valid_timeframe("5min")
    True
    >>> is_valid_timeframe("1h")
    True
    >>> is_valid_timeframe("invalid")
    False
    """
    if timeframe in SUPPORTED_TIMEFRAMES:
        return True

    # Check lowercase version
    tf_lower = timeframe.lower().strip()
    if tf_lower in SUPPORTED_TIMEFRAMES or tf_lower in TIMEFRAME_ALIASES:
        return True

    # Check if it can be parsed as a valid format
    try:
        minutes = get_timeframe_minutes(timeframe)
        # For intraday, must be <= 60min (or 240/1440 for extended)
        if allow_extended:
            return minutes > 0
        else:
            return minutes > 0 and minutes <= 60
    except ValueError:
        return False


def validate_timeframe(timeframe: str, allow_extended: bool = True) -> None:
    """
    Validate that a timeframe string is supported.

    Parameters
    ----------
    timeframe : str
        Timeframe string to validate (e.g., '5min', '15min', '1h')
    allow_extended : bool, default True
        If True, also accept extended timeframes (4h, daily)

    Raises
    ------
    ValueError
        If the timeframe is not supported
    """
    if not is_valid_timeframe(timeframe, allow_extended=allow_extended):
        raise ValueError(
            f"Unsupported timeframe: '{timeframe}'. "
            f"Supported timeframes: {SUPPORTED_TIMEFRAMES}"
        )


# Legacy alias for backward compatibility
def timeframe_to_minutes(timeframe: str) -> int:
    """
    Convert a timeframe string to minutes.

    DEPRECATED: Use get_timeframe_minutes() instead.

    Parameters
    ----------
    timeframe : str
        Timeframe string (e.g., '5min', '1h')

    Returns
    -------
    int
        Number of minutes
    """
    return get_timeframe_minutes(timeframe)


def get_timeframe_suffix(timeframe: str) -> str:
    """
    Get the standardized column suffix for a timeframe.

    This ensures consistent naming across the codebase:
    - "5min" -> "_5m"
    - "15min" -> "_15m"
    - "60min" or "1h" -> "_1h"
    - "240min" or "4h" -> "_4h"
    - "1440min" or "daily" -> "_1d"

    Parameters
    ----------
    timeframe : str
        Timeframe string (canonical or alias)

    Returns
    -------
    str
        Column suffix (e.g., "_5m", "_1h", "_1d")

    Examples
    --------
    >>> get_timeframe_suffix("15min")
    '_15m'
    >>> get_timeframe_suffix("1h")
    '_1h'
    >>> get_timeframe_suffix("daily")
    '_1d'
    """
    minutes = get_timeframe_minutes(timeframe)

    if minutes >= 1440:
        # Daily or longer
        days = minutes // 1440
        return f"_{days}d"
    elif minutes >= 60 and minutes % 60 == 0:
        # Hourly
        hours = minutes // 60
        return f"_{hours}h"
    else:
        # Minutes
        return f"_{minutes}m"


def get_canonical_from_suffix(suffix: str) -> str:
    """
    Get the canonical timeframe from a column suffix.

    Parameters
    ----------
    suffix : str
        Column suffix (e.g., "_5m", "_1h", "_1d")

    Returns
    -------
    str
        Canonical timeframe (e.g., "5min", "60min", "1440min")

    Examples
    --------
    >>> get_canonical_from_suffix("_15m")
    '15min'
    >>> get_canonical_from_suffix("_1h")
    '60min'
    >>> get_canonical_from_suffix("_1d")
    '1440min'
    """
    # Remove leading underscore if present
    s = suffix.lstrip("_")

    if s.endswith("d"):
        days = int(s[:-1])
        return f"{days * 1440}min"
    elif s.endswith("h"):
        hours = int(s[:-1])
        return f"{hours * 60}min"
    elif s.endswith("m"):
        minutes = int(s[:-1])
        return f"{minutes}min"
    else:
        raise ValueError(f"Unrecognized suffix format: '{suffix}'")
