"""
Comprehensive tests for Session/DST handling.

This module tests:
- DST boundary classification for trading sessions
- London+NY overlap detection across DST transitions
- Asia session midnight crossing behavior
- Session flag features correctness
- DSTHandler utility functions

The current session implementation uses REAL market hours:
- Asia:   23:00-07:00 UTC (crosses midnight, no DST - Japan doesn't observe DST)
- London: 08:00-16:30 UTC (fixed UTC hours)
- NY:     EST 14:30-21:00 UTC / EDT 13:30-20:00 UTC (DST-aware)

There is also a London+NY overlap period with DST-aware timing.

Author: ML Pipeline
Created: 2025-12-23
"""

import sys
from datetime import datetime, date, time, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Add src to path for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from stages.sessions import (
    SessionName,
    SessionConfig,
    SessionsConfig,
    SESSIONS,
    SESSION_OVERLAPS,
    get_session_for_hour,
    SessionFilter,
    DSTHandler,
    CMECalendar,
)
from stages.sessions.config import (
    is_in_overlap,
    is_in_overlap_dt,
    get_overlap_times,
    is_dst_active,
    is_in_session,
    get_session_for_time,
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def dst_handler_ny():
    """Create DSTHandler for America/New_York timezone."""
    return DSTHandler('America/New_York')


@pytest.fixture
def dst_handler_london():
    """Create DSTHandler for Europe/London timezone."""
    return DSTHandler('Europe/London')


@pytest.fixture
def session_filter():
    """Create default SessionFilter."""
    return SessionFilter()


@pytest.fixture
def sample_multi_session_df():
    """Create DataFrame spanning multiple sessions across DST boundaries."""
    timestamps = [
        # Summer (EDT) - July 2024
        datetime(2024, 7, 15, 2, 0),   # Asia session (23:00-07:00 UTC)
        datetime(2024, 7, 15, 10, 0),  # London session (08:00-16:30 UTC)
        datetime(2024, 7, 15, 15, 0),  # London+NY overlap (summer: 13:30-16:30)
        datetime(2024, 7, 15, 18, 0),  # NY session only (16:30-20:00 UTC summer)
        # Winter (EST) - January 2024
        datetime(2024, 1, 15, 2, 0),   # Asia session
        datetime(2024, 1, 15, 10, 0),  # London session
        datetime(2024, 1, 15, 15, 30), # London+NY overlap (winter: 14:30-16:30)
        datetime(2024, 1, 15, 18, 0),  # NY session only
    ]
    df = pd.DataFrame({
        'datetime': timestamps,
        'close': [100, 101, 102, 103, 104, 105, 106, 107],
    })
    return df


# =============================================================================
# DST BOUNDARY CLASSIFICATION TESTS
# =============================================================================

class TestDSTBoundaryClassification:
    """Tests for DST-aware session boundary classification."""

    def test_dst_handler_detects_summer_dst(self, dst_handler_ny):
        """DST should be active in July (EDT)."""
        dt_summer = datetime(2024, 7, 15, 13, 30)
        assert dst_handler_ny.is_dst(dt_summer) is True

    def test_dst_handler_detects_winter_no_dst(self, dst_handler_ny):
        """DST should NOT be active in January (EST)."""
        dt_winter = datetime(2024, 1, 15, 14, 30)
        assert dst_handler_ny.is_dst(dt_winter) is False

    def test_utc_offset_summer_edt(self, dst_handler_ny):
        """UTC offset should be -4 hours during EDT (summer)."""
        dt_summer = datetime(2024, 7, 15, 12, 0)
        offset = dst_handler_ny.get_utc_offset_hours(dt_summer)
        assert offset == -4.0

    def test_utc_offset_winter_est(self, dst_handler_ny):
        """UTC offset should be -5 hours during EST (winter)."""
        dt_winter = datetime(2024, 1, 15, 12, 0)
        offset = dst_handler_ny.get_utc_offset_hours(dt_winter)
        assert offset == -5.0

    def test_session_time_adjustment_for_dst_summer(self, dst_handler_ny):
        """Session times should shift 1 hour earlier in UTC during DST."""
        # Base times represent winter (EST) session
        base_start = (14, 30)  # 9:30 ET in EST = 14:30 UTC
        base_end = (21, 0)     # 16:00 ET in EST = 21:00 UTC

        dt_summer = datetime(2024, 7, 15, 12, 0)
        adj_start, adj_end = dst_handler_ny.adjust_session_times_for_dst(
            base_start, base_end, dt_summer
        )

        # In summer (EDT), 9:30 ET = 13:30 UTC (1 hour earlier)
        assert adj_start == (13, 30)
        assert adj_end == (20, 0)

    def test_session_time_adjustment_for_dst_winter(self, dst_handler_ny):
        """Session times should remain unchanged in winter (EST)."""
        base_start = (14, 30)  # 9:30 ET in EST = 14:30 UTC
        base_end = (21, 0)     # 16:00 ET in EST = 21:00 UTC

        dt_winter = datetime(2024, 1, 15, 12, 0)
        adj_start, adj_end = dst_handler_ny.adjust_session_times_for_dst(
            base_start, base_end, dt_winter
        )

        # In winter (EST), times unchanged
        assert adj_start == (14, 30)
        assert adj_end == (21, 0)


class TestNYSessionDSTBoundary:
    """
    Tests for NY session DST boundary classification.

    NY Session (real market hours):
    - 9:30 AM - 4:00 PM Eastern Time
    - Summer (EDT): 13:30-20:00 UTC
    - Winter (EST): 14:30-21:00 UTC
    """

    def test_ny_session_dst_boundary_summer(self, session_filter):
        """NY 9:30 ET should be 13:30 UTC in EDT (summer)."""
        # Summer (EDT) - July
        dt_summer = datetime(2024, 7, 15, 13, 30)  # 9:30 ET in EDT = 13:30 UTC
        assert session_filter.classify_session(dt_summer) == SessionName.NEW_YORK

    def test_ny_session_dst_boundary_winter_different_time(self, session_filter):
        """Same UTC time (13:30) in winter should NOT be NY (it's 8:30 ET)."""
        # Winter (EST) - January
        # 13:30 UTC = 8:30 ET in EST = NOT NY session (NY starts at 9:30)
        dt_winter = datetime(2024, 1, 15, 13, 30)  # 8:30 ET in EST = 13:30 UTC
        # This falls in London session (08:00-16:30 UTC)
        assert session_filter.classify_session(dt_winter) == SessionName.LONDON

    def test_ny_session_winter_starts_14_30(self, session_filter):
        """NY in winter starts at 14:30 UTC (9:30 ET in EST)."""
        dt_winter_ny = datetime(2024, 1, 15, 14, 30)  # 9:30 ET in EST = 14:30 UTC
        assert session_filter.classify_session(dt_winter_ny) == SessionName.NEW_YORK

    def test_ny_session_config_dst_aware(self):
        """NY session config should be DST-aware."""
        ny_config = SESSIONS[SessionName.NEW_YORK]
        assert ny_config.dst_aware is True
        assert ny_config.dst_start_utc == (13, 30)  # Summer: 9:30 ET = 13:30 UTC
        assert ny_config.dst_end_utc == (20, 0)     # Summer: 4:00 PM ET = 20:00 UTC

    def test_ny_session_winter_bounds(self):
        """NY session winter (EST) bounds should be 14:30-21:00 UTC."""
        ny_config = SESSIONS[SessionName.NEW_YORK]
        assert ny_config.start_utc == (14, 30)  # Winter: 9:30 ET = 14:30 UTC
        assert ny_config.end_utc == (21, 0)     # Winter: 4:00 PM ET = 21:00 UTC


class TestDSTTransitionDates:
    """Tests for DST transition date detection."""

    def test_spring_forward_2024(self, dst_handler_ny):
        """Spring forward 2024 is March 10."""
        spring, fall = dst_handler_ny.get_dst_transition_dates(2024)
        assert spring == date(2024, 3, 10)

    def test_fall_back_2024(self, dst_handler_ny):
        """Fall back 2024 is November 3."""
        spring, fall = dst_handler_ny.get_dst_transition_dates(2024)
        assert fall == date(2024, 11, 3)

    def test_spring_forward_2025(self, dst_handler_ny):
        """Spring forward 2025 is March 9."""
        spring, fall = dst_handler_ny.get_dst_transition_dates(2025)
        assert spring == date(2025, 3, 9)

    def test_fall_back_2025(self, dst_handler_ny):
        """Fall back 2025 is November 2."""
        spring, fall = dst_handler_ny.get_dst_transition_dates(2025)
        assert fall == date(2025, 11, 2)

    def test_is_dst_transition_date_spring(self, dst_handler_ny):
        """Correctly identifies spring DST transition date."""
        assert dst_handler_ny.is_dst_transition_date(date(2024, 3, 10)) is True
        assert dst_handler_ny.is_dst_transition_date(date(2024, 3, 9)) is False
        assert dst_handler_ny.is_dst_transition_date(date(2024, 3, 11)) is False

    def test_is_dst_transition_date_fall(self, dst_handler_ny):
        """Correctly identifies fall DST transition date."""
        assert dst_handler_ny.is_dst_transition_date(date(2024, 11, 3)) is True
        assert dst_handler_ny.is_dst_transition_date(date(2024, 11, 2)) is False
        assert dst_handler_ny.is_dst_transition_date(date(2024, 11, 4)) is False


class TestIsDstActiveFunction:
    """Tests for the is_dst_active function."""

    def test_is_dst_active_winter_months(self):
        """January, February, December are never DST."""
        assert is_dst_active(datetime(2024, 1, 15, 12), 'America/New_York') is False
        assert is_dst_active(datetime(2024, 2, 15, 12), 'America/New_York') is False
        assert is_dst_active(datetime(2024, 12, 15, 12), 'America/New_York') is False

    def test_is_dst_active_summer_months(self):
        """April through October are always DST."""
        for month in range(4, 11):  # April to October
            assert is_dst_active(datetime(2024, month, 15, 12), 'America/New_York') is True

    def test_is_dst_active_march_transition(self):
        """March around second Sunday shows DST transition."""
        # 2024: Second Sunday is March 10
        # Day before should be standard time
        assert is_dst_active(datetime(2024, 3, 9, 12), 'America/New_York') is False
        # Day after should be DST
        assert is_dst_active(datetime(2024, 3, 11, 12), 'America/New_York') is True

    def test_is_dst_active_november_transition(self):
        """November around first Sunday shows DST transition."""
        # 2024: First Sunday is November 3
        # Day before should be DST
        assert is_dst_active(datetime(2024, 11, 2, 12), 'America/New_York') is True
        # Day after should be standard time
        assert is_dst_active(datetime(2024, 11, 4, 12), 'America/New_York') is False


# =============================================================================
# LONDON+NY OVERLAP TESTS
# =============================================================================

class TestLondonNYOverlap:
    """
    Tests for London+NY session overlap detection.

    The overlap occurs when both London and NY are actively trading:
    - Winter (EST): 14:30-16:30 UTC (2 hours)
    - Summer (EDT): 13:30-16:30 UTC (3 hours - NY opens 1 hour earlier)
    """

    def test_london_ny_overlap_defined(self):
        """London+NY overlap should be defined."""
        assert 'london_ny' in SESSION_OVERLAPS
        overlap = SESSION_OVERLAPS['london_ny']
        assert overlap.sessions == (SessionName.LONDON, SessionName.NEW_YORK)

    def test_overlap_winter_in_overlap(self):
        """Winter: 15:00 UTC should be in London+NY overlap."""
        dt_overlap_winter = datetime(2024, 1, 15, 15, 0)  # 15:00 UTC
        assert is_in_overlap_dt(dt_overlap_winter, 'london_ny')

    def test_overlap_winter_before(self):
        """Winter: 12:00 UTC should NOT be in overlap (before NY opens)."""
        dt_before = datetime(2024, 1, 15, 12, 0)  # 12:00 UTC
        assert not is_in_overlap_dt(dt_before, 'london_ny')

    def test_overlap_summer_in_overlap(self):
        """Summer: 14:00 UTC should be in London+NY overlap."""
        dt_overlap_summer = datetime(2024, 7, 15, 14, 0)  # 14:00 UTC
        assert is_in_overlap_dt(dt_overlap_summer, 'london_ny')

    def test_overlap_summer_extended_start(self):
        """Summer: 13:45 UTC should be in overlap (NY opens at 13:30 in summer)."""
        dt_summer_early = datetime(2024, 7, 15, 13, 45)  # 13:45 UTC
        assert is_in_overlap_dt(dt_summer_early, 'london_ny')

    def test_overlap_winter_at_start_boundary(self):
        """Winter: Exactly at 14:30 UTC should be in overlap."""
        dt_at_start = datetime(2024, 1, 15, 14, 30)
        assert is_in_overlap_dt(dt_at_start, 'london_ny')

    def test_overlap_at_end_boundary(self):
        """At 16:30 UTC should NOT be in overlap (end is exclusive)."""
        dt_at_end = datetime(2024, 1, 15, 16, 30)
        assert not is_in_overlap_dt(dt_at_end, 'london_ny')

    def test_get_overlap_times_winter(self):
        """Get winter overlap times: 14:30-16:30 UTC."""
        dt_winter = datetime(2024, 1, 15, 12, 0)
        start, end = get_overlap_times('london_ny', dt_winter)
        assert start == (14, 30)
        assert end == (16, 30)

    def test_get_overlap_times_summer(self):
        """Get summer overlap times: 13:30-16:30 UTC."""
        dt_summer = datetime(2024, 7, 15, 12, 0)
        start, end = get_overlap_times('london_ny', dt_summer)
        assert start == (13, 30)  # 1 hour earlier than winter
        assert end == (16, 30)    # Same end time

    def test_is_in_overlap_function(self):
        """Test is_in_overlap returns overlap name or None."""
        # Winter overlap
        dt_winter = datetime(2024, 1, 15, 15, 0)
        result = is_in_overlap(15, 0, dt_winter)
        assert result == 'london_ny'

        # Outside overlap
        result_outside = is_in_overlap(12, 0, dt_winter)
        assert result_outside is None


# =============================================================================
# ASIA SESSION MIDNIGHT CROSSING TESTS
# =============================================================================

class TestAsiaSessionMidnightCrossing:
    """
    Tests for Asia session midnight crossing behavior.

    Asia session: 23:00-07:00 UTC (crosses midnight)
    Tokyo: 08:00-16:00 JST (Japan does not observe DST)
    """

    def test_asia_session_before_midnight(self, session_filter):
        """23:30 UTC should be Asia session."""
        dt_before_midnight = datetime(2024, 1, 15, 23, 30)
        assert session_filter.classify_session(dt_before_midnight) == SessionName.ASIA

    def test_asia_session_after_midnight(self, session_filter):
        """02:00 UTC should be Asia session."""
        dt_after_midnight = datetime(2024, 1, 16, 2, 0)
        assert session_filter.classify_session(dt_after_midnight) == SessionName.ASIA

    def test_asia_session_at_start(self, session_filter):
        """23:00 UTC should be Asia session start."""
        dt_start = datetime(2024, 1, 15, 23, 0)
        assert session_filter.classify_session(dt_start) == SessionName.ASIA

    def test_asia_session_before_end(self, session_filter):
        """06:59 UTC should still be Asia session."""
        dt_before_end = datetime(2024, 1, 16, 6, 59)
        assert session_filter.classify_session(dt_before_end) == SessionName.ASIA

    def test_outside_asia_session(self, session_filter):
        """10:00 UTC should be London (not Asia)."""
        dt_london = datetime(2024, 1, 15, 10, 0)
        assert session_filter.classify_session(dt_london) == SessionName.LONDON

    def test_asia_session_config_crosses_midnight(self):
        """Asia session config should have crosses_midnight=True."""
        asia_config = SESSIONS[SessionName.ASIA]
        assert asia_config.crosses_midnight is True

    def test_asia_session_not_dst_aware(self):
        """Asia session should NOT be DST-aware (Japan has no DST)."""
        asia_config = SESSIONS[SessionName.ASIA]
        assert asia_config.dst_aware is False


class TestMidnightCrossingSessionConfig:
    """Tests for session configurations that cross midnight."""

    def test_asia_crosses_midnight(self):
        """Asia session crosses midnight."""
        asia_config = SESSIONS[SessionName.ASIA]
        assert asia_config.crosses_midnight is True
        assert asia_config.start_utc == (23, 0)
        assert asia_config.end_utc == (7, 0)

    def test_london_does_not_cross_midnight(self):
        """London session does NOT cross midnight."""
        london_config = SESSIONS[SessionName.LONDON]
        assert london_config.crosses_midnight is False

    def test_ny_does_not_cross_midnight(self):
        """NY session does NOT cross midnight."""
        ny_config = SESSIONS[SessionName.NEW_YORK]
        assert ny_config.crosses_midnight is False


# =============================================================================
# SESSION FLAGS FEATURE TESTS
# =============================================================================

class TestSessionFlagsCorrect:
    """Tests for session flag features correctness."""

    def test_session_flags_assigned_correctly(self, session_filter, sample_multi_session_df):
        """Session flag features are correctly assigned."""
        df = session_filter.add_session_features(sample_multi_session_df)

        # Check column existence
        assert 'session_asia' in df.columns
        assert 'session_london' in df.columns
        assert 'session_new_york' in df.columns

        # Row 0: 02:00 UTC = Asia (23:00-07:00 UTC)
        assert df['session_asia'].iloc[0] == 1
        assert df['session_london'].iloc[0] == 0
        assert df['session_new_york'].iloc[0] == 0

        # Row 1: 10:00 UTC = London (08:00-16:30 UTC)
        assert df['session_asia'].iloc[1] == 0
        assert df['session_london'].iloc[1] == 1
        assert df['session_new_york'].iloc[1] == 0

    def test_session_flags_overlap_period(self, session_filter, sample_multi_session_df):
        """Session flags during overlap show both sessions active."""
        df = session_filter.add_session_features(sample_multi_session_df)

        # Row 2: 15:00 UTC = London+NY overlap (summer, NY active from 13:30)
        assert df['session_london'].iloc[2] == 1
        assert df['session_new_york'].iloc[2] == 1

        # Row 6: 15:30 UTC = London+NY overlap (winter, NY active from 14:30)
        assert df['session_london'].iloc[6] == 1
        assert df['session_new_york'].iloc[6] == 1

    def test_session_flags_ny_only(self, session_filter, sample_multi_session_df):
        """After London close, only NY is active."""
        df = session_filter.add_session_features(sample_multi_session_df)

        # Row 3: 18:00 UTC (summer) = NY only (London closes at 16:30)
        assert df['session_london'].iloc[3] == 0
        assert df['session_new_york'].iloc[3] == 1

        # Row 7: 18:00 UTC (winter) = NY only
        assert df['session_london'].iloc[7] == 0
        assert df['session_new_york'].iloc[7] == 1

    def test_session_flags_binary(self, session_filter, sample_multi_session_df):
        """Session flags are binary (0 or 1)."""
        df = session_filter.add_session_features(sample_multi_session_df)

        for col in ['session_asia', 'session_london', 'session_new_york']:
            assert df[col].isin([0, 1]).all(), f"{col} should be binary"


class TestSessionFlagsEdgeCases:
    """Edge case tests for session flags."""

    def test_session_flags_filter_subset(self):
        """Session flags only for specified sessions."""
        config = SessionsConfig(
            include_sessions=[SessionName.NEW_YORK, SessionName.LONDON]
        )
        filter = SessionFilter(config)

        timestamps = [
            datetime(2024, 6, 15, 2, 0),   # Asia
            datetime(2024, 6, 15, 10, 0),  # London
            datetime(2024, 6, 15, 18, 0),  # NY (summer)
        ]
        df = pd.DataFrame({'datetime': timestamps, 'close': [100, 101, 102]})

        df = filter.add_session_features(df)

        # Only NY and London flags should be added
        assert 'session_new_york' in df.columns
        assert 'session_london' in df.columns
        # Asia should NOT have a flag column
        assert 'session_asia' not in df.columns

    def test_overlap_flags_added(self, session_filter):
        """Overlap flags are added when configured."""
        config = SessionsConfig(add_overlap_flags=True)
        filter = SessionFilter(config)

        timestamps = [
            datetime(2024, 7, 15, 14, 0),  # In London+NY overlap (summer)
            datetime(2024, 7, 15, 18, 0),  # Not in overlap
        ]
        df = pd.DataFrame({'datetime': timestamps, 'close': [100, 101]})

        df = filter.add_session_features(df)

        # Should have overlap flag
        assert 'overlap_london_ny' in df.columns
        assert df['overlap_london_ny'].iloc[0] == 1
        assert df['overlap_london_ny'].iloc[1] == 0


# =============================================================================
# DST FEATURE TESTS
# =============================================================================

class TestDSTFeatures:
    """Tests for DST-related features."""

    def test_add_dst_features(self, dst_handler_ny, sample_multi_session_df):
        """DST features are correctly added."""
        df = dst_handler_ny.add_dst_features(sample_multi_session_df, 'datetime')

        assert 'is_dst' in df.columns
        assert 'is_dst_transition' in df.columns

        # Summer rows (July) should have is_dst = 1
        # Rows 0-3 are July
        assert df['is_dst'].iloc[0] == 1
        assert df['is_dst'].iloc[1] == 1
        assert df['is_dst'].iloc[2] == 1
        assert df['is_dst'].iloc[3] == 1

        # Winter rows (January) should have is_dst = 0
        # Rows 4-7 are January
        assert df['is_dst'].iloc[4] == 0
        assert df['is_dst'].iloc[5] == 0
        assert df['is_dst'].iloc[6] == 0
        assert df['is_dst'].iloc[7] == 0

    def test_dst_transition_dates_flagged(self, dst_handler_ny):
        """DST transition dates are correctly flagged."""
        # Create data around spring forward 2024 (March 10)
        timestamps = [
            datetime(2024, 3, 9, 12, 0),   # Day before
            datetime(2024, 3, 10, 12, 0),  # Transition day
            datetime(2024, 3, 11, 12, 0),  # Day after
        ]
        df = pd.DataFrame({'datetime': timestamps, 'close': [100, 101, 102]})

        df = dst_handler_ny.add_dst_features(df, 'datetime')

        assert df['is_dst_transition'].iloc[0] == 0
        assert df['is_dst_transition'].iloc[1] == 1
        assert df['is_dst_transition'].iloc[2] == 0


# =============================================================================
# get_session_for_hour TESTS
# =============================================================================

class TestGetSessionForHour:
    """
    Tests for get_session_for_hour function.

    This function uses REAL market hours with gap handling.
    """

    def test_asia_hours(self):
        """Late night and early morning hours map to Asia."""
        dt = datetime(2024, 1, 15, 12, 0)  # Use winter for consistent testing
        # Asia is 23:00-07:00 UTC
        for hour in [23, 0, 1, 2, 3, 4, 5, 6]:
            assert get_session_for_hour(hour, dt) == SessionName.ASIA, f"Hour {hour}"

    def test_london_hours(self):
        """Mid-day hours map to London."""
        dt = datetime(2024, 1, 15, 12, 0)  # Winter
        # London is 08:00-16:30 UTC
        for hour in [8, 9, 10, 11, 12, 13, 14, 15, 16]:
            assert get_session_for_hour(hour, dt) == SessionName.LONDON, f"Hour {hour}"

    def test_ny_hours_winter(self):
        """Afternoon hours in winter map to NY."""
        dt = datetime(2024, 1, 15, 12, 0)  # Winter
        # NY is 14:30-21:00 UTC in winter
        for hour in [17, 18, 19, 20]:
            assert get_session_for_hour(hour, dt) == SessionName.NEW_YORK, f"Hour {hour}"

    def test_gap_handling_hour_7(self):
        """Hour 7 (gap between Asia and London) maps to Asia."""
        dt = datetime(2024, 1, 15, 12, 0)
        # 7:00-8:00 is gap, should map to Asia
        assert get_session_for_hour(7, dt) == SessionName.ASIA

    def test_invalid_hour_raises(self):
        """Invalid hour should raise ValueError."""
        with pytest.raises(ValueError):
            get_session_for_hour(-1)
        with pytest.raises(ValueError):
            get_session_for_hour(24)


class TestGetSessionForTime:
    """Tests for get_session_for_time function."""

    def test_returns_none_in_gap(self):
        """Returns None when time is in a gap between sessions."""
        dt = datetime(2024, 1, 15, 12, 0)  # Winter
        # 7:30 is in gap between Asia (ends 7:00) and London (starts 8:00)
        result = get_session_for_time(7, 30, dt)
        assert result is None

    def test_returns_session_when_active(self):
        """Returns session name when time is in an active session."""
        dt = datetime(2024, 1, 15, 12, 0)  # Winter
        # 10:30 is clearly in London session
        result = get_session_for_time(10, 30, dt)
        assert result == SessionName.LONDON


class TestIsInSession:
    """Tests for is_in_session function."""

    def test_in_asia_session(self):
        """Check time in Asia session."""
        dt = datetime(2024, 1, 15, 2, 0)
        assert is_in_session(2, 0, SessionName.ASIA, dt) is True
        assert is_in_session(23, 30, SessionName.ASIA, dt) is True

    def test_not_in_asia_session(self):
        """Check time NOT in Asia session."""
        dt = datetime(2024, 1, 15, 12, 0)
        assert is_in_session(10, 0, SessionName.ASIA, dt) is False

    def test_ny_session_dst_aware(self):
        """NY session bounds change with DST."""
        # Summer - NY starts at 13:30
        dt_summer = datetime(2024, 7, 15, 13, 30)
        assert is_in_session(13, 30, SessionName.NEW_YORK, dt_summer) is True
        assert is_in_session(13, 0, SessionName.NEW_YORK, dt_summer) is False

        # Winter - NY starts at 14:30
        dt_winter = datetime(2024, 1, 15, 14, 30)
        assert is_in_session(14, 30, SessionName.NEW_YORK, dt_winter) is True
        assert is_in_session(13, 30, SessionName.NEW_YORK, dt_winter) is False


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestDSTSessionIntegration:
    """Integration tests combining DST and session handling."""

    def test_full_session_dst_pipeline(self, sample_multi_session_df):
        """Full pipeline with session flags and DST features."""
        # Add session features
        filter = SessionFilter()
        df = filter.add_session_features(sample_multi_session_df)

        # Add DST features
        handler = DSTHandler('America/New_York')
        df = handler.add_dst_features(df, 'datetime')

        # Verify all features present
        assert 'session_asia' in df.columns
        assert 'session_london' in df.columns
        assert 'session_new_york' in df.columns
        assert 'is_dst' in df.columns
        assert 'is_dst_transition' in df.columns

    def test_calendar_integration_with_sessions(self):
        """CME calendar integrates correctly with session handling."""
        # Create data spanning a holiday
        timestamps = [
            datetime(2024, 12, 24, 10, 0),  # Christmas Eve (early close)
            datetime(2024, 12, 25, 10, 0),  # Christmas (holiday)
            datetime(2024, 12, 26, 10, 0),  # Day after Christmas
        ]
        df = pd.DataFrame({'datetime': timestamps, 'close': [100, 101, 102]})

        # Add session features
        filter = SessionFilter()
        df = filter.add_session_features(df)

        # Filter holidays
        calendar = CMECalendar()
        df_filtered = calendar.filter_holidays(df)

        # Christmas should be filtered out
        assert len(df_filtered) == 2
        assert date(2024, 12, 25) not in df_filtered['datetime'].dt.date.values


# =============================================================================
# BOUNDARY CONDITION TESTS
# =============================================================================

class TestBoundaryConditions:
    """Tests for boundary conditions and edge cases."""

    def test_year_boundary_asia(self, session_filter):
        """Session classification works across year boundary (Asia crosses midnight)."""
        # New Year's Eve 23:30 = Asia (Asia starts at 23:00)
        dt_nye = datetime(2024, 12, 31, 23, 30)
        assert session_filter.classify_session(dt_nye) == SessionName.ASIA

        # New Year's Day 00:30 = Asia (continues after midnight)
        dt_nyd = datetime(2025, 1, 1, 0, 30)
        assert session_filter.classify_session(dt_nyd) == SessionName.ASIA

    def test_leap_year(self, session_filter):
        """Session classification works on leap year day."""
        # Feb 29, 2024 (leap year) 10:00 UTC = London
        dt_leap = datetime(2024, 2, 29, 10, 0)
        assert session_filter.classify_session(dt_leap) == SessionName.LONDON

    def test_asia_session_minute_boundary(self, session_filter):
        """Classification handles Asia minute boundaries correctly."""
        # 22:59 should be gap (Asia starts at 23:00)
        dt_just_before = datetime(2024, 6, 15, 22, 59)
        # This might be NY close or gap depending on implementation
        result = session_filter.classify_session(dt_just_before)
        assert result != SessionName.ASIA

        # 23:00 should be Asia
        dt_exactly = datetime(2024, 6, 15, 23, 0)
        assert session_filter.classify_session(dt_exactly) == SessionName.ASIA

    def test_london_session_boundaries(self, session_filter):
        """London session boundaries are correct."""
        # 08:00 = London start
        dt_london_start = datetime(2024, 6, 15, 8, 0)
        assert session_filter.classify_session(dt_london_start) == SessionName.LONDON

        # 16:29 = London (before 16:30 end)
        dt_london_end = datetime(2024, 6, 15, 16, 29)
        assert session_filter.classify_session(dt_london_end) == SessionName.LONDON


class TestGapsBetweenSessions:
    """Tests for gaps between sessions (7:00-8:00 UTC and NY close to Asia open)."""

    def test_gap_after_asia_before_london(self, session_filter):
        """07:00-08:00 UTC is a gap between Asia and London."""
        # Asia ends at 07:00, London starts at 08:00
        dt_gap = datetime(2024, 6, 15, 7, 30)
        result = session_filter.classify_session(dt_gap)
        # This is in a gap - current implementation may return None
        assert result is None

    def test_gap_after_ny_before_asia(self, session_filter):
        """Gap between NY close and Asia open."""
        # NY ends at 21:00 (winter) or 20:00 (summer)
        # Asia starts at 23:00
        dt_gap_summer = datetime(2024, 7, 15, 22, 0)  # After 20:00 NY close, before 23:00 Asia
        result = session_filter.classify_session(dt_gap_summer)
        # This is in a gap
        assert result is None


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
