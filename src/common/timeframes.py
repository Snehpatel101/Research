"""
Canonical timeframe registry - single source of truth for all timeframe operations.

This module provides unified timeframe handling across the entire codebase:
- Supported timeframes for resampling
- Timeframe aliases (e.g., "1h" = "60min")
- Conversion utilities

All other modules should import from here instead of defining their own timeframe lists.
"""

# =============================================================================
# CANONICAL TIMEFRAME DEFINITIONS
# =============================================================================

# All supported timeframes (unified vocabulary)
# These are the canonical representations used throughout the codebase.
SUPPORTED_TIMEFRAMES = [
    "1min",
    "5min",
    "10min",
    "15min",
    "20min",
    "25min",
    "30min",
    "45min",
    "60min",
    "1h",  # Alias for 60min, both are valid
]

# Primary 9-TF ladder (excludes alias duplicates for iteration)
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
FULL_9TF_LADDER = [
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

# Aliases mapping to canonical form
TIMEFRAME_ALIASES = {
    "1h": "60min",
    "4h": "240min",
}

# Minutes for each timeframe (includes aliases)
TIMEFRAME_TO_MINUTES = {
    "1min": 1,
    "5min": 5,
    "10min": 10,
    "15min": 15,
    "20min": 20,
    "25min": 25,
    "30min": 30,
    "45min": 45,
    "60min": 60,
    "1h": 60,  # Alias
}

# Pandas frequency mapping
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
    "1h": "1h",
}


def normalize_timeframe(timeframe: str) -> str:
    """
    Normalize a timeframe string to its canonical form.

    Parameters
    ----------
    timeframe : str
        Timeframe string (e.g., "1h", "60min")

    Returns
    -------
    str
        Canonical timeframe (e.g., "60min" for "1h")
    """
    tf = timeframe.lower().strip()
    return TIMEFRAME_ALIASES.get(tf, tf)


def is_valid_timeframe(timeframe: str) -> bool:
    """
    Check if a timeframe string is valid (including aliases).

    Parameters
    ----------
    timeframe : str
        Timeframe string to check

    Returns
    -------
    bool
        True if valid timeframe
    """
    return timeframe in SUPPORTED_TIMEFRAMES


def validate_timeframe(timeframe: str) -> None:
    """
    Validate that a timeframe string is supported.

    Parameters
    ----------
    timeframe : str
        Timeframe string to validate (e.g., '5min', '15min', '1h')

    Raises
    ------
    ValueError
        If the timeframe is not supported
    """
    if not is_valid_timeframe(timeframe):
        raise ValueError(
            f"Unsupported timeframe: '{timeframe}'. "
            f"Supported timeframes: {SUPPORTED_TIMEFRAMES}"
        )


def timeframe_to_minutes(timeframe: str) -> int:
    """
    Convert a timeframe string to minutes.

    Parameters
    ----------
    timeframe : str
        Timeframe string (e.g., '5min', '1h')

    Returns
    -------
    int
        Number of minutes

    Raises
    ------
    ValueError
        If the timeframe is not recognized
    """
    if timeframe in TIMEFRAME_TO_MINUTES:
        return TIMEFRAME_TO_MINUTES[timeframe]

    # Try parsing
    tf = timeframe.lower().strip()
    if tf.endswith("min"):
        try:
            return int(tf[:-3])
        except ValueError:
            pass
    elif tf.endswith("h"):
        try:
            return int(tf[:-1]) * 60
        except ValueError:
            pass

    raise ValueError(
        f"Unrecognized timeframe format: '{timeframe}'. "
        f"Expected format like '5min' or '1h'"
    )
