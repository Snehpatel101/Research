"""
Trading session configuration.

This module contains trading session definitions for CME futures markets,
including session times, DST handling, and session-based feature flags.

Sessions:
- New York: 14:30-21:00 UTC (09:30-16:00 ET during EST)
- London: 08:00-16:30 UTC
- Asia: 23:00-07:00 UTC (crosses midnight)

Session overlaps (e.g., London-NY: 14:30-16:30 UTC) often exhibit
higher volatility and liquidity.
"""

# =============================================================================
# SESSION CONFIGURATION
# =============================================================================
SESSIONS_CONFIG = {
    # Master enable/disable for session filtering
    'enabled': True,

    # Session definitions (UTC times)
    # These are the base times during winter (EST/GMT)
    # DST adjustments are handled automatically
    'sessions': {
        'new_york': {
            'name': 'New York',
            'start_utc': (14, 30),  # 09:30 ET
            'end_utc': (21, 0),     # 16:00 ET
            'timezone': 'America/New_York',
            'enabled': True,
        },
        'london': {
            'name': 'London',
            'start_utc': (8, 0),    # 08:00 GMT
            'end_utc': (16, 30),    # 16:30 GMT
            'timezone': 'Europe/London',
            'enabled': True,
        },
        'asia': {
            'name': 'Asia',
            'start_utc': (23, 0),   # 08:00 JST (next day)
            'end_utc': (7, 0),      # 16:00 JST
            'timezone': 'Asia/Tokyo',
            'enabled': True,
            'crosses_midnight': True,
        },
    },

    # Feature flags
    'add_session_flags': True,      # Add session_ny, session_london, session_asia columns
    'add_overlap_flags': True,      # Add overlap_london_ny column for overlapping hours
    'filter_holidays': True,        # Filter out CME holidays
    'handle_dst': True,             # Adjust session times for DST transitions

    # Session volatility normalization
    'session_volatility_normalization': False,  # Normalize features by session volatility
    'volatility_lookback': 288,                 # Lookback for volatility calc (~1 day for 5min)

    # Holiday calendar
    'holiday_calendar': 'CME',      # Use CME holiday calendar
}


def get_sessions_config() -> dict:
    """
    Get the sessions configuration dictionary.

    Returns
    -------
    dict
        Copy of SESSIONS_CONFIG
    """
    import copy
    return copy.deepcopy(SESSIONS_CONFIG)


def validate_sessions_config(config: dict = None) -> list[str]:
    """
    Validate sessions configuration values.

    Parameters
    ----------
    config : dict, optional
        Sessions config dict to validate. Uses SESSIONS_CONFIG if not provided.

    Returns
    -------
    list[str]
        List of validation error messages (empty if valid)
    """
    if config is None:
        config = SESSIONS_CONFIG

    errors = []

    # Validate session definitions
    sessions = config.get('sessions', {})
    if not sessions:
        errors.append("No sessions defined in SESSIONS_CONFIG['sessions']")

    for session_name, session_config in sessions.items():
        if not isinstance(session_config, dict):
            errors.append(f"Session '{session_name}' config must be a dict")
            continue

        # Validate required fields
        required_fields = ['name', 'start_utc', 'end_utc', 'timezone']
        for field in required_fields:
            if field not in session_config:
                errors.append(f"Session '{session_name}' missing required field '{field}'")

        # Validate time tuples
        for time_field in ['start_utc', 'end_utc']:
            time_val = session_config.get(time_field)
            if time_val is not None:
                if not isinstance(time_val, tuple) or len(time_val) != 2:
                    errors.append(
                        f"Session '{session_name}' {time_field} must be tuple (hour, minute)"
                    )
                else:
                    hour, minute = time_val
                    if not (0 <= hour <= 23):
                        errors.append(
                            f"Session '{session_name}' {time_field} hour must be 0-23"
                        )
                    if not (0 <= minute <= 59):
                        errors.append(
                            f"Session '{session_name}' {time_field} minute must be 0-59"
                        )

    # Validate volatility lookback
    lookback = config.get('volatility_lookback', 0)
    if lookback < 0:
        errors.append(f"volatility_lookback must be non-negative, got {lookback}")

    return errors
