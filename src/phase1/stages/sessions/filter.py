"""
Session Filter for Trading Data

This module provides the SessionFilter class for:
- Filtering data by trading session
- Adding session flags as features
- Detecting session overlaps
- Handling DST transitions

Author: ML Pipeline
Created: 2025-12-22
"""

import logging
from datetime import datetime

import pandas as pd

from .config import (
    DEFAULT_SESSIONS_CONFIG,
    SESSION_OVERLAPS,
    SESSIONS,
    SessionConfig,
    SessionName,
    SessionsConfig,
)

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class SessionFilter:
    """
    Filter trading data by session and add session-related features.

    This class handles:
    - Session classification for each bar
    - Filtering to include/exclude specific sessions
    - Adding session flags as features
    - Detecting session overlaps
    - DST-aware session boundaries

    Usage:
        filter = SessionFilter(config)
        df = filter.add_session_features(df)
        df = filter.filter_by_session(df)
    """

    def __init__(
        self,
        config: SessionsConfig | None = None,
        datetime_column: str = 'datetime'
    ):
        """
        Initialize SessionFilter.

        Args:
            config: Session configuration. Uses defaults if None.
            datetime_column: Name of the datetime column in DataFrames.

        Raises:
            ValueError: If config is invalid.
        """
        self.config = config or DEFAULT_SESSIONS_CONFIG
        self.datetime_column = datetime_column
        self._validate_config()

    def _validate_config(self) -> None:
        """Validate configuration."""
        if not isinstance(self.config, SessionsConfig):
            raise ValueError(
                f"config must be SessionsConfig, got {type(self.config)}"
            )

    def _validate_dataframe(self, df: pd.DataFrame) -> None:
        """Validate input DataFrame."""
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"Expected DataFrame, got {type(df)}")

        if df.empty:
            raise ValueError("DataFrame is empty")

        if self.datetime_column not in df.columns:
            raise ValueError(
                f"DataFrame missing required column '{self.datetime_column}'. "
                f"Available columns: {list(df.columns)}"
            )

        # Check datetime column is datetime type
        if not pd.api.types.is_datetime64_any_dtype(df[self.datetime_column]):
            raise TypeError(
                f"Column '{self.datetime_column}' must be datetime type, "
                f"got {df[self.datetime_column].dtype}"
            )

    def _get_time_minutes(self, dt: datetime) -> int:
        """Get time of day as minutes since midnight UTC."""
        return dt.hour * 60 + dt.minute

    def _is_in_session(
        self,
        time_minutes: int,
        session: SessionConfig
    ) -> bool:
        """
        Check if a time is within a session.

        Args:
            time_minutes: Time as minutes since midnight UTC
            session: Session configuration

        Returns:
            True if time is within the session
        """
        start = session.start_minutes
        end = session.end_minutes

        if session.crosses_midnight:
            # Session spans midnight: e.g., 23:00 to 07:00
            # Time is in session if >= start OR < end
            return time_minutes >= start or time_minutes < end
        else:
            # Normal session: e.g., 08:00 to 16:30
            return start <= time_minutes < end

    def classify_session(self, dt: datetime) -> SessionName | None:
        """
        Classify which session a datetime belongs to.

        Args:
            dt: Datetime to classify (assumed UTC)

        Returns:
            SessionName if in a session, None otherwise
        """
        time_minutes = self._get_time_minutes(dt)

        for session_name, session_config in SESSIONS.items():
            if self._is_in_session(time_minutes, session_config):
                return session_name

        return None

    def classify_sessions_vectorized(
        self,
        datetimes: pd.Series
    ) -> pd.Series:
        """
        Classify sessions for a series of datetimes (vectorized).

        Args:
            datetimes: Series of datetime values (assumed UTC)

        Returns:
            Series of SessionName values (or None)
        """
        time_minutes = datetimes.dt.hour * 60 + datetimes.dt.minute

        # Initialize with None
        result = pd.Series(index=datetimes.index, dtype=object)

        for session_name, session_config in SESSIONS.items():
            start = session_config.start_minutes
            end = session_config.end_minutes

            if session_config.crosses_midnight:
                mask = (time_minutes >= start) | (time_minutes < end)
            else:
                mask = (time_minutes >= start) & (time_minutes < end)

            result.loc[mask] = session_name

        return result

    def get_session_flags(
        self,
        datetimes: pd.Series
    ) -> dict[str, pd.Series]:
        """
        Generate session flag columns for a series of datetimes.

        Args:
            datetimes: Series of datetime values

        Returns:
            Dictionary of session name -> binary flag Series
        """
        time_minutes = datetimes.dt.hour * 60 + datetimes.dt.minute
        flags = {}

        active_sessions = self.config.get_active_sessions()

        for session_name in active_sessions:
            session_config = SESSIONS[session_name]
            start = session_config.start_minutes
            end = session_config.end_minutes

            if session_config.crosses_midnight:
                mask = (time_minutes >= start) | (time_minutes < end)
            else:
                mask = (time_minutes >= start) & (time_minutes < end)

            col_name = f'session_{session_name.value}'
            flags[col_name] = mask.astype(int)

        return flags

    def get_overlap_flags(
        self,
        datetimes: pd.Series
    ) -> dict[str, pd.Series]:
        """
        Generate overlap flag columns for a series of datetimes.

        Args:
            datetimes: Series of datetime values

        Returns:
            Dictionary of overlap name -> binary flag Series
        """
        time_minutes = datetimes.dt.hour * 60 + datetimes.dt.minute
        flags = {}

        for overlap_name, overlap in SESSION_OVERLAPS.items():
            start = overlap.start_utc[0] * 60 + overlap.start_utc[1]
            end = overlap.end_utc[0] * 60 + overlap.end_utc[1]

            # Check if both sessions are active
            sessions_active = all(
                s in self.config.get_active_sessions()
                for s in overlap.sessions
            )

            if not sessions_active:
                continue

            if start < end:
                mask = (time_minutes >= start) & (time_minutes < end)
            else:
                # Crosses midnight
                mask = (time_minutes >= start) | (time_minutes < end)

            col_name = f'overlap_{overlap_name}'
            flags[col_name] = mask.astype(int)

        return flags

    def add_session_features(
        self,
        df: pd.DataFrame,
        feature_metadata: dict[str, str] | None = None
    ) -> pd.DataFrame:
        """
        Add session and overlap flags as features to DataFrame.

        Args:
            df: Input DataFrame with datetime column
            feature_metadata: Optional dict to store feature descriptions

        Returns:
            DataFrame with session features added

        Raises:
            ValueError: If DataFrame is invalid
        """
        self._validate_dataframe(df)
        df = df.copy()

        datetimes = df[self.datetime_column]

        # Add session flags
        if self.config.add_session_flags:
            session_flags = self.get_session_flags(datetimes)
            for col_name, values in session_flags.items():
                df[col_name] = values
                if feature_metadata is not None:
                    session_name = col_name.replace('session_', '')
                    session_config = SESSIONS.get(SessionName(session_name))
                    if session_config:
                        desc = (
                            f"{session_config.name} session "
                            f"({session_config.start_utc[0]:02d}:{session_config.start_utc[1]:02d}-"
                            f"{session_config.end_utc[0]:02d}:{session_config.end_utc[1]:02d} UTC)"
                        )
                        feature_metadata[col_name] = desc

        # Add overlap flags
        if self.config.add_overlap_flags:
            overlap_flags = self.get_overlap_flags(datetimes)
            for col_name, values in overlap_flags.items():
                df[col_name] = values
                if feature_metadata is not None:
                    overlap_name = col_name.replace('overlap_', '')
                    overlap = SESSION_OVERLAPS.get(overlap_name)
                    if overlap:
                        feature_metadata[col_name] = overlap.description

        logger.info(
            f"Added session features: {len(session_flags) if self.config.add_session_flags else 0} sessions, "
            f"{len(overlap_flags) if self.config.add_overlap_flags else 0} overlaps"
        )

        return df

    def filter_by_session(
        self,
        df: pd.DataFrame,
        inplace: bool = False
    ) -> pd.DataFrame:
        """
        Filter DataFrame to only include rows from active sessions.

        Args:
            df: Input DataFrame with datetime column
            inplace: If True, modify df in place

        Returns:
            Filtered DataFrame

        Raises:
            ValueError: If DataFrame is invalid
        """
        self._validate_dataframe(df)

        if not inplace:
            df = df.copy()

        datetimes = df[self.datetime_column]
        time_minutes = datetimes.dt.hour * 60 + datetimes.dt.minute

        active_sessions = self.config.get_active_sessions()

        if not active_sessions:
            logger.warning("No active sessions - returning empty DataFrame")
            return df.iloc[0:0]

        # Build mask for active sessions
        mask = pd.Series(False, index=df.index)

        for session_name in active_sessions:
            session_config = SESSIONS[session_name]
            start = session_config.start_minutes
            end = session_config.end_minutes

            if session_config.crosses_midnight:
                session_mask = (time_minutes >= start) | (time_minutes < end)
            else:
                session_mask = (time_minutes >= start) & (time_minutes < end)

            mask = mask | session_mask

        original_len = len(df)
        df = df.loc[mask]
        filtered_len = len(df)

        logger.info(
            f"Filtered by session: {original_len} -> {filtered_len} rows "
            f"({100 * filtered_len / original_len:.1f}% retained)"
        )

        return df

    def get_session_stats(self, df: pd.DataFrame) -> dict[str, dict]:
        """
        Calculate statistics for each session in the DataFrame.

        Args:
            df: Input DataFrame with datetime column

        Returns:
            Dictionary of session name -> statistics dict
        """
        self._validate_dataframe(df)

        datetimes = df[self.datetime_column]
        sessions = self.classify_sessions_vectorized(datetimes)

        stats = {}
        for session_name in SessionName:
            mask = sessions == session_name
            count = mask.sum()

            if count > 0:
                stats[session_name.value] = {
                    'count': int(count),
                    'percentage': float(100 * count / len(df)),
                    'first': df.loc[mask, self.datetime_column].min(),
                    'last': df.loc[mask, self.datetime_column].max(),
                }
            else:
                stats[session_name.value] = {
                    'count': 0,
                    'percentage': 0.0,
                    'first': None,
                    'last': None,
                }

        # Add None/gap stats
        mask = sessions.isna()
        stats['outside_sessions'] = {
            'count': int(mask.sum()),
            'percentage': float(100 * mask.sum() / len(df)) if len(df) > 0 else 0.0,
        }

        return stats


def create_session_filter(
    include: list[str] | None = None,
    exclude: list[str] | None = None,
    add_flags: bool = True,
    add_overlaps: bool = True
) -> SessionFilter:
    """
    Factory function to create a SessionFilter with common configurations.

    Args:
        include: List of session names to include (e.g., ['new_york', 'london'])
        exclude: List of session names to exclude
        add_flags: Whether to add session flags as features
        add_overlaps: Whether to add overlap flags as features

    Returns:
        Configured SessionFilter

    Raises:
        ValueError: If session names are invalid
    """
    include_sessions = None
    if include:
        include_sessions = []
        for name in include:
            try:
                include_sessions.append(SessionName(name.lower()))
            except ValueError:
                valid = [s.value for s in SessionName]
                raise ValueError(
                    f"Invalid session name '{name}'. Valid names: {valid}"
                )

    exclude_sessions = None
    if exclude:
        exclude_sessions = []
        for name in exclude:
            try:
                exclude_sessions.append(SessionName(name.lower()))
            except ValueError:
                valid = [s.value for s in SessionName]
                raise ValueError(
                    f"Invalid session name '{name}'. Valid names: {valid}"
                )

    config = SessionsConfig(
        include_sessions=include_sessions,
        exclude_sessions=exclude_sessions,
        add_session_flags=add_flags,
        add_overlap_flags=add_overlaps,
    )

    return SessionFilter(config)


__all__ = [
    'SessionFilter',
    'create_session_filter',
]
