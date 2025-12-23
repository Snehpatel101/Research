"""
Session Configuration and Definitions

This module defines trading session configurations in UTC for CME futures markets.
Sessions are defined with proper handling of:
- Session boundaries (start/end times in UTC)
- Timezone information for DST handling
- Session overlaps (e.g., London+NY overlap)

Sessions are based on actual market trading hours:
- New York: 09:30-16:00 ET (14:30-21:00 UTC during EST, 13:30-20:00 UTC during EDT)
- London: 08:00-16:30 GMT/BST
- Asia: Tokyo 09:00-15:00 JST (00:00-06:00 UTC)

Author: ML Pipeline
Created: 2025-12-22
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class SessionName(Enum):
    """Trading session names."""
    NEW_YORK = 'new_york'
    LONDON = 'london'
    ASIA = 'asia'


@dataclass(frozen=True)
class SessionConfig:
    """
    Configuration for a single trading session.

    Attributes:
        name: Human-readable session name
        session_id: Unique identifier for the session
        start_utc: Session start time in UTC (hour, minute)
        end_utc: Session end time in UTC (hour, minute)
        timezone: IANA timezone for the session's primary exchange
        description: Brief description of the session
        crosses_midnight: Whether the session spans midnight UTC
    """
    name: str
    session_id: SessionName
    start_utc: Tuple[int, int]  # (hour, minute)
    end_utc: Tuple[int, int]    # (hour, minute)
    timezone: str
    description: str = ""
    crosses_midnight: bool = False

    def __post_init__(self) -> None:
        """Validate session configuration."""
        self._validate_time(self.start_utc, "start_utc")
        self._validate_time(self.end_utc, "end_utc")

    def _validate_time(self, time_tuple: Tuple[int, int], field_name: str) -> None:
        """Validate time tuple format."""
        if not isinstance(time_tuple, tuple) or len(time_tuple) != 2:
            raise ValueError(f"{field_name} must be a tuple of (hour, minute)")
        hour, minute = time_tuple
        if not (0 <= hour <= 23):
            raise ValueError(f"{field_name} hour must be 0-23, got {hour}")
        if not (0 <= minute <= 59):
            raise ValueError(f"{field_name} minute must be 0-59, got {minute}")

    @property
    def start_minutes(self) -> int:
        """Start time as minutes since midnight UTC."""
        return self.start_utc[0] * 60 + self.start_utc[1]

    @property
    def end_minutes(self) -> int:
        """End time as minutes since midnight UTC."""
        return self.end_utc[0] * 60 + self.end_utc[1]

    @property
    def duration_minutes(self) -> int:
        """Session duration in minutes."""
        if self.crosses_midnight:
            # Session spans midnight: e.g., 23:00 to 07:00
            return (24 * 60 - self.start_minutes) + self.end_minutes
        else:
            return self.end_minutes - self.start_minutes


# =============================================================================
# SESSION DEFINITIONS (UTC times)
# =============================================================================
# These are the PRIMARY trading sessions for CME futures
# Times are in UTC to avoid DST complexity in core logic

SESSIONS: Dict[SessionName, SessionConfig] = {
    # New York Session: NYSE/CME primary hours
    # US Eastern: 09:30-16:00 ET
    # UTC: 14:30-21:00 (during EST) / 13:30-20:00 (during EDT)
    # Using EST (winter) times as base - DST handling is separate
    SessionName.NEW_YORK: SessionConfig(
        name='New York',
        session_id=SessionName.NEW_YORK,
        start_utc=(14, 30),
        end_utc=(21, 0),
        timezone='America/New_York',
        description='NYSE/CME primary trading hours (09:30-16:00 ET)',
        crosses_midnight=False
    ),

    # London Session: LSE primary hours
    # GMT/BST: 08:00-16:30 London time
    # UTC: 08:00-16:30 (during GMT) / 07:00-15:30 (during BST)
    # Using GMT (winter) times as base
    SessionName.LONDON: SessionConfig(
        name='London',
        session_id=SessionName.LONDON,
        start_utc=(8, 0),
        end_utc=(16, 30),
        timezone='Europe/London',
        description='LSE trading hours (08:00-16:30 London)',
        crosses_midnight=False
    ),

    # Asia Session: Tokyo + Hong Kong primary hours
    # Tokyo: 09:00-15:00 JST (lunch break 11:30-12:30 not modeled)
    # UTC: 00:00-06:00 (JST = UTC+9, so 09:00 JST = 00:00 UTC)
    # Extended to 07:00 to bridge to London
    SessionName.ASIA: SessionConfig(
        name='Asia',
        session_id=SessionName.ASIA,
        start_utc=(23, 0),
        end_utc=(7, 0),
        timezone='Asia/Tokyo',
        description='Tokyo/HK trading hours (23:00-07:00 UTC)',
        crosses_midnight=True
    ),
}


@dataclass
class SessionOverlap:
    """
    Definition of overlapping trading sessions.

    Overlaps often exhibit higher volatility and liquidity.
    """
    name: str
    sessions: Tuple[SessionName, SessionName]
    start_utc: Tuple[int, int]
    end_utc: Tuple[int, int]
    description: str = ""


# Session overlaps - periods when multiple major markets are open
SESSION_OVERLAPS: Dict[str, SessionOverlap] = {
    'london_ny': SessionOverlap(
        name='London-New York',
        sessions=(SessionName.LONDON, SessionName.NEW_YORK),
        start_utc=(14, 30),  # NY opens
        end_utc=(16, 30),    # London closes
        description='Highest liquidity period - both London and NY open'
    ),
    'asia_london': SessionOverlap(
        name='Asia-London',
        sessions=(SessionName.ASIA, SessionName.LONDON),
        start_utc=(8, 0),    # London opens
        end_utc=(7, 0),      # This doesn't actually overlap in current config
        description='Brief overlap as Asia closes and London opens'
    ),
}


@dataclass
class SessionsConfig:
    """
    Complete configuration for session filtering and normalization.

    This is the main configuration class used throughout the sessions module.
    """
    # Which sessions to include in analysis (None = all)
    include_sessions: Optional[List[SessionName]] = None

    # Which sessions to exclude from analysis
    exclude_sessions: Optional[List[SessionName]] = None

    # Whether to add session flags as features
    add_session_flags: bool = True

    # Whether to add overlap flags as features
    add_overlap_flags: bool = True

    # Whether to filter out CME holidays
    filter_holidays: bool = True

    # Whether to handle DST transitions
    handle_dst: bool = True

    # Whether to normalize volatility by session
    session_volatility_normalization: bool = False

    # Volatility lookback for normalization (in bars)
    volatility_lookback: int = 288  # ~1 day for 5-min bars

    def __post_init__(self) -> None:
        """Validate configuration."""
        self._validate_session_lists()

    def _validate_session_lists(self) -> None:
        """Validate include/exclude session lists."""
        if self.include_sessions is not None and self.exclude_sessions is not None:
            overlap = set(self.include_sessions) & set(self.exclude_sessions)
            if overlap:
                raise ValueError(
                    f"Sessions cannot be both included and excluded: {overlap}"
                )

    def get_active_sessions(self) -> List[SessionName]:
        """Get list of sessions that should be active based on config."""
        all_sessions = list(SessionName)

        if self.include_sessions is not None:
            active = [s for s in all_sessions if s in self.include_sessions]
        else:
            active = all_sessions

        if self.exclude_sessions is not None:
            active = [s for s in active if s not in self.exclude_sessions]

        return active

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            'include_sessions': [s.value for s in self.include_sessions] if self.include_sessions else None,
            'exclude_sessions': [s.value for s in self.exclude_sessions] if self.exclude_sessions else None,
            'add_session_flags': self.add_session_flags,
            'add_overlap_flags': self.add_overlap_flags,
            'filter_holidays': self.filter_holidays,
            'handle_dst': self.handle_dst,
            'session_volatility_normalization': self.session_volatility_normalization,
            'volatility_lookback': self.volatility_lookback,
        }

    @classmethod
    def from_dict(cls, d: Dict) -> 'SessionsConfig':
        """Create from dictionary."""
        include = None
        if d.get('include_sessions'):
            include = [SessionName(s) for s in d['include_sessions']]

        exclude = None
        if d.get('exclude_sessions'):
            exclude = [SessionName(s) for s in d['exclude_sessions']]

        return cls(
            include_sessions=include,
            exclude_sessions=exclude,
            add_session_flags=d.get('add_session_flags', True),
            add_overlap_flags=d.get('add_overlap_flags', True),
            filter_holidays=d.get('filter_holidays', True),
            handle_dst=d.get('handle_dst', True),
            session_volatility_normalization=d.get('session_volatility_normalization', False),
            volatility_lookback=d.get('volatility_lookback', 288),
        )


# Default configuration
DEFAULT_SESSIONS_CONFIG = SessionsConfig()


def get_session_config(session_name: SessionName) -> SessionConfig:
    """
    Get configuration for a specific session.

    Args:
        session_name: Session identifier

    Returns:
        SessionConfig for the requested session

    Raises:
        ValueError: If session_name is not a valid SessionName
    """
    if not isinstance(session_name, SessionName):
        raise ValueError(f"session_name must be SessionName enum, got {type(session_name)}")

    return SESSIONS[session_name]


def get_all_sessions() -> List[SessionConfig]:
    """Get all session configurations."""
    return list(SESSIONS.values())


__all__ = [
    'SessionName',
    'SessionConfig',
    'SessionOverlap',
    'SessionsConfig',
    'SESSIONS',
    'SESSION_OVERLAPS',
    'DEFAULT_SESSIONS_CONFIG',
    'get_session_config',
    'get_all_sessions',
]
