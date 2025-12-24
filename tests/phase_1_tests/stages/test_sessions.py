"""
Comprehensive tests for the Session Normalization System.

Tests cover:
- Session configuration and validation
- Session filtering and classification
- CME holiday calendar
- DST handling
- Session-specific volatility normalization
- Edge cases and error handling

Author: ML Pipeline
Created: 2025-12-22
"""

import sys
from pathlib import Path
from datetime import datetime, date, timedelta

import pytest
import numpy as np
import pandas as pd

# Add src to path for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from src.phase1.stages.sessions import (
    # Config
    SessionName,
    SessionConfig,
    SessionOverlap,
    SessionsConfig,
    SESSIONS,
    SESSION_OVERLAPS,
    DEFAULT_SESSIONS_CONFIG,
    get_session_config,
    get_all_sessions,
    # Filter
    SessionFilter,
    create_session_filter,
    # Calendar
    TradingDayType,
    CME_HOLIDAYS,
    CME_EARLY_CLOSE,
    CMECalendar,
    DSTHandler,
    get_calendar,
    # Normalizer
    SessionVolatilityStats,
    SessionNormalizer,
    normalize_by_session,
    get_session_volatility_ratios,
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def sample_datetime_df():
    """Create sample DataFrame with datetime spanning all sessions."""
    np.random.seed(42)

    # Create 24 hours of 5-minute data covering all sessions
    start = datetime(2024, 6, 15, 0, 0)  # Saturday starts, but we'll use weekdays
    timestamps = []
    current = start

    # Generate 3 days of data
    for day in range(3):
        day_start = start + timedelta(days=day)
        for hour in range(24):
            for minute in range(0, 60, 5):
                timestamps.append(day_start + timedelta(hours=hour, minutes=minute))

    n = len(timestamps)
    df = pd.DataFrame({
        'datetime': timestamps,
        'close': 100.0 + np.cumsum(np.random.randn(n) * 0.1),
        'volume': np.random.randint(100, 1000, n),
        'atr_20': np.abs(np.random.randn(n) * 0.5) + 0.5,
    })

    return df


@pytest.fixture
def sample_ohlcv_with_sessions():
    """Create OHLCV DataFrame with data across all sessions."""
    np.random.seed(42)
    n = 1000

    # Start on Monday
    start = datetime(2024, 6, 17, 0, 0)  # Monday
    timestamps = [start + timedelta(minutes=5 * i) for i in range(n)]

    base_price = 4500.0
    returns = np.random.randn(n) * 0.002
    close = base_price * np.exp(np.cumsum(returns))

    df = pd.DataFrame({
        'datetime': timestamps,
        'symbol': 'MES',
        'open': close * (1 + np.random.randn(n) * 0.001),
        'high': close * (1 + np.abs(np.random.randn(n)) * 0.002),
        'low': close * (1 - np.abs(np.random.randn(n)) * 0.002),
        'close': close,
        'volume': np.random.randint(100, 1000, n).astype(float),
    })

    # Fix OHLC relationships
    df['high'] = df[['open', 'high', 'close']].max(axis=1)
    df['low'] = df[['open', 'low', 'close']].min(axis=1)

    # Add ATR
    df['atr_20'] = (df['high'] - df['low']).rolling(20).mean().fillna(1.0)

    return df


# =============================================================================
# SESSION CONFIGURATION TESTS
# =============================================================================

class TestSessionConfig:
    """Tests for SessionConfig and related configuration."""

    def test_session_config_creation(self):
        """Test creating a SessionConfig instance."""
        config = SessionConfig(
            name='Test Session',
            session_id=SessionName.NEW_YORK,
            start_utc=(14, 30),
            end_utc=(21, 0),
            timezone='America/New_York',
            description='Test session',
            crosses_midnight=False
        )

        assert config.name == 'Test Session'
        assert config.session_id == SessionName.NEW_YORK
        assert config.start_utc == (14, 30)
        assert config.end_utc == (21, 0)
        assert config.timezone == 'America/New_York'
        assert not config.crosses_midnight

    def test_session_config_time_validation_invalid_hour(self):
        """Test that invalid hour raises ValueError."""
        with pytest.raises(ValueError, match="hour must be 0-23"):
            SessionConfig(
                name='Invalid',
                session_id=SessionName.NEW_YORK,
                start_utc=(25, 0),  # Invalid hour
                end_utc=(21, 0),
                timezone='America/New_York'
            )

    def test_session_config_time_validation_invalid_minute(self):
        """Test that invalid minute raises ValueError."""
        with pytest.raises(ValueError, match="minute must be 0-59"):
            SessionConfig(
                name='Invalid',
                session_id=SessionName.NEW_YORK,
                start_utc=(14, 60),  # Invalid minute
                end_utc=(21, 0),
                timezone='America/New_York'
            )

    def test_session_config_start_minutes(self):
        """Test start_minutes property."""
        config = SESSIONS[SessionName.NEW_YORK]
        expected = 14 * 60 + 30  # 14:30 = 870 minutes
        assert config.start_minutes == expected

    def test_session_config_end_minutes(self):
        """Test end_minutes property."""
        config = SESSIONS[SessionName.NEW_YORK]
        expected = 21 * 60  # 21:00 = 1260 minutes
        assert config.end_minutes == expected

    def test_session_config_duration_minutes(self):
        """Test duration_minutes property for normal session."""
        config = SESSIONS[SessionName.NEW_YORK]
        expected = 21 * 60 - (14 * 60 + 30)  # 390 minutes = 6.5 hours
        assert config.duration_minutes == expected

    def test_session_config_duration_minutes_crosses_midnight(self):
        """Test duration_minutes for session crossing midnight."""
        config = SESSIONS[SessionName.ASIA]
        assert config.crosses_midnight
        # 23:00 to 07:00 = 8 hours = 480 minutes
        expected = (24 * 60 - 23 * 60) + 7 * 60  # 60 + 420 = 480
        assert config.duration_minutes == expected

    def test_get_session_config(self):
        """Test get_session_config function."""
        config = get_session_config(SessionName.LONDON)
        assert config.name == 'London'
        assert config.timezone == 'Europe/London'

    def test_get_session_config_invalid(self):
        """Test get_session_config with invalid input."""
        with pytest.raises(ValueError, match="must be SessionName enum"):
            get_session_config('london')  # String instead of enum

    def test_get_all_sessions(self):
        """Test get_all_sessions returns all sessions."""
        sessions = get_all_sessions()
        assert len(sessions) == 3
        names = [s.name for s in sessions]
        assert 'New York' in names
        assert 'London' in names
        assert 'Asia' in names

    def test_default_sessions_defined(self):
        """Test that all default sessions are defined."""
        assert SessionName.NEW_YORK in SESSIONS
        assert SessionName.LONDON in SESSIONS
        assert SessionName.ASIA in SESSIONS


class TestSessionsConfig:
    """Tests for SessionsConfig class."""

    def test_default_config(self):
        """Test default configuration."""
        config = SessionsConfig()
        assert config.include_sessions is None
        assert config.exclude_sessions is None
        assert config.add_session_flags is True
        assert config.add_overlap_flags is True
        assert config.filter_holidays is True

    def test_include_sessions(self):
        """Test include_sessions configuration."""
        config = SessionsConfig(
            include_sessions=[SessionName.NEW_YORK, SessionName.LONDON]
        )
        active = config.get_active_sessions()
        assert SessionName.NEW_YORK in active
        assert SessionName.LONDON in active
        assert SessionName.ASIA not in active

    def test_exclude_sessions(self):
        """Test exclude_sessions configuration."""
        config = SessionsConfig(
            exclude_sessions=[SessionName.ASIA]
        )
        active = config.get_active_sessions()
        assert SessionName.NEW_YORK in active
        assert SessionName.LONDON in active
        assert SessionName.ASIA not in active

    def test_include_exclude_conflict(self):
        """Test that including and excluding same session raises error."""
        with pytest.raises(ValueError, match="cannot be both included and excluded"):
            SessionsConfig(
                include_sessions=[SessionName.NEW_YORK],
                exclude_sessions=[SessionName.NEW_YORK]
            )

    def test_to_dict(self):
        """Test to_dict serialization."""
        config = SessionsConfig(
            include_sessions=[SessionName.NEW_YORK],
            add_session_flags=False
        )
        d = config.to_dict()

        assert d['include_sessions'] == ['new_york']
        assert d['add_session_flags'] is False

    def test_from_dict(self):
        """Test from_dict deserialization."""
        d = {
            'include_sessions': ['new_york', 'london'],
            'exclude_sessions': None,
            'add_session_flags': True,
            'add_overlap_flags': False,
        }
        config = SessionsConfig.from_dict(d)

        assert len(config.include_sessions) == 2
        assert SessionName.NEW_YORK in config.include_sessions
        assert config.add_overlap_flags is False


# =============================================================================
# SESSION FILTER TESTS
# =============================================================================

class TestSessionFilter:
    """Tests for SessionFilter class."""

    def test_filter_creation_default(self):
        """Test creating SessionFilter with defaults."""
        filter = SessionFilter()
        assert filter.config.add_session_flags is True
        assert filter.datetime_column == 'datetime'

    def test_filter_creation_custom_config(self):
        """Test creating SessionFilter with custom config."""
        config = SessionsConfig(add_session_flags=False)
        filter = SessionFilter(config)
        assert filter.config.add_session_flags is False

    def test_classify_session_new_york(self):
        """Test classifying a time in NY session."""
        filter = SessionFilter()
        # 15:00 UTC = 10:00 ET = NY session
        dt = datetime(2024, 6, 17, 15, 0)
        session = filter.classify_session(dt)
        assert session == SessionName.NEW_YORK

    def test_classify_session_london(self):
        """Test classifying a time in London session."""
        filter = SessionFilter()
        # 10:00 UTC = London session
        dt = datetime(2024, 6, 17, 10, 0)
        session = filter.classify_session(dt)
        assert session == SessionName.LONDON

    def test_classify_session_asia(self):
        """Test classifying a time in Asia session."""
        filter = SessionFilter()
        # 01:00 UTC = Asia session (crosses midnight)
        dt = datetime(2024, 6, 17, 1, 0)
        session = filter.classify_session(dt)
        assert session == SessionName.ASIA

    def test_classify_session_asia_before_midnight(self):
        """Test classifying a time in Asia session before midnight."""
        filter = SessionFilter()
        # 23:30 UTC = Asia session
        dt = datetime(2024, 6, 17, 23, 30)
        session = filter.classify_session(dt)
        assert session == SessionName.ASIA

    def test_add_session_features(self, sample_datetime_df):
        """Test adding session features to DataFrame."""
        filter = SessionFilter()
        feature_metadata = {}
        df = filter.add_session_features(sample_datetime_df, feature_metadata)

        # Check session flag columns exist
        assert 'session_new_york' in df.columns
        assert 'session_london' in df.columns
        assert 'session_asia' in df.columns

        # Check metadata was recorded
        assert 'session_new_york' in feature_metadata

        # Check flags are binary
        for col in ['session_new_york', 'session_london', 'session_asia']:
            assert df[col].isin([0, 1]).all()

    def test_add_overlap_features(self, sample_datetime_df):
        """Test adding overlap features to DataFrame."""
        config = SessionsConfig(add_overlap_flags=True)
        filter = SessionFilter(config)
        df = filter.add_session_features(sample_datetime_df)

        # Check overlap column exists
        assert 'overlap_london_ny' in df.columns

        # Check flags are binary
        assert df['overlap_london_ny'].isin([0, 1]).all()

    def test_filter_by_session(self, sample_datetime_df):
        """Test filtering DataFrame by session."""
        config = SessionsConfig(include_sessions=[SessionName.NEW_YORK])
        filter = SessionFilter(config)

        original_len = len(sample_datetime_df)
        df = filter.filter_by_session(sample_datetime_df)

        # Should have fewer rows
        assert len(df) < original_len
        assert len(df) > 0

    def test_filter_excludes_sessions(self, sample_datetime_df):
        """Test filtering excludes specified sessions."""
        config = SessionsConfig(exclude_sessions=[SessionName.ASIA])
        filter = SessionFilter(config)

        df = filter.add_session_features(sample_datetime_df)
        df = filter.filter_by_session(df)

        # Should not have any Asia session rows
        time_minutes = df['datetime'].dt.hour * 60 + df['datetime'].dt.minute
        asia_start = 23 * 60
        asia_end = 7 * 60

        # Check no rows in Asia session
        asia_mask = (time_minutes >= asia_start) | (time_minutes < asia_end)
        assert not asia_mask.any()

    def test_get_session_stats(self, sample_datetime_df):
        """Test getting session statistics."""
        filter = SessionFilter()
        stats = filter.get_session_stats(sample_datetime_df)

        # Should have stats for all sessions
        assert 'new_york' in stats
        assert 'london' in stats
        assert 'asia' in stats

        # Check stat structure
        for session_name, session_stats in stats.items():
            if session_name != 'outside_sessions':
                assert 'count' in session_stats
                assert 'percentage' in session_stats

    def test_validate_empty_dataframe(self):
        """Test that empty DataFrame raises error."""
        filter = SessionFilter()
        df = pd.DataFrame({'datetime': pd.Series([], dtype='datetime64[ns]')})

        with pytest.raises(ValueError, match="DataFrame is empty"):
            filter.add_session_features(df)

    def test_validate_missing_datetime_column(self):
        """Test that missing datetime column raises error."""
        filter = SessionFilter()
        df = pd.DataFrame({'close': [100, 101, 102]})

        with pytest.raises(ValueError, match="missing required column"):
            filter.add_session_features(df)

    def test_create_session_filter_factory(self):
        """Test create_session_filter factory function."""
        filter = create_session_filter(
            include=['new_york', 'london'],
            add_flags=True,
            add_overlaps=True
        )

        assert len(filter.config.include_sessions) == 2
        assert SessionName.NEW_YORK in filter.config.include_sessions

    def test_create_session_filter_invalid_session(self):
        """Test factory with invalid session name."""
        with pytest.raises(ValueError, match="Invalid session name"):
            create_session_filter(include=['invalid_session'])


# =============================================================================
# CME CALENDAR TESTS
# =============================================================================

class TestCMECalendar:
    """Tests for CMECalendar class."""

    def test_calendar_creation(self):
        """Test creating CMECalendar instance."""
        calendar = CMECalendar()
        assert calendar is not None

    def test_is_holiday_christmas(self):
        """Test Christmas is a holiday."""
        calendar = CMECalendar()
        assert calendar.is_holiday(date(2024, 12, 25))
        assert calendar.is_holiday(date(2025, 12, 25))

    def test_is_holiday_regular_day(self):
        """Test regular day is not a holiday."""
        calendar = CMECalendar()
        assert not calendar.is_holiday(date(2024, 6, 17))  # Monday

    def test_is_weekend(self):
        """Test weekend detection."""
        calendar = CMECalendar()
        assert calendar.is_weekend(date(2024, 6, 15))  # Saturday
        assert calendar.is_weekend(date(2024, 6, 16))  # Sunday
        assert not calendar.is_weekend(date(2024, 6, 17))  # Monday

    def test_is_early_close(self):
        """Test early close detection."""
        calendar = CMECalendar()
        assert calendar.is_early_close(date(2024, 12, 24))  # Christmas Eve
        assert not calendar.is_early_close(date(2024, 12, 25))  # Christmas (holiday)

    def test_get_trading_day_type_regular(self):
        """Test regular trading day type."""
        calendar = CMECalendar()
        day_type = calendar.get_trading_day_type(date(2024, 6, 17))
        assert day_type == TradingDayType.REGULAR

    def test_get_trading_day_type_weekend(self):
        """Test weekend trading day type."""
        calendar = CMECalendar()
        day_type = calendar.get_trading_day_type(date(2024, 6, 15))
        assert day_type == TradingDayType.WEEKEND

    def test_get_trading_day_type_holiday(self):
        """Test holiday trading day type."""
        calendar = CMECalendar()
        day_type = calendar.get_trading_day_type(date(2024, 12, 25))
        assert day_type == TradingDayType.HOLIDAY

    def test_get_trading_day_type_early_close(self):
        """Test early close trading day type."""
        calendar = CMECalendar()
        day_type = calendar.get_trading_day_type(date(2024, 12, 24))
        assert day_type == TradingDayType.EARLY_CLOSE

    def test_is_trading_day(self):
        """Test trading day detection."""
        calendar = CMECalendar()
        assert calendar.is_trading_day(date(2024, 6, 17))  # Monday
        assert calendar.is_trading_day(date(2024, 12, 24))  # Early close
        assert not calendar.is_trading_day(date(2024, 12, 25))  # Holiday
        assert not calendar.is_trading_day(date(2024, 6, 15))  # Weekend

    def test_get_holidays_in_range(self):
        """Test getting holidays in a date range."""
        calendar = CMECalendar()
        holidays = calendar.get_holidays_in_range(
            date(2024, 12, 1),
            date(2025, 1, 31)
        )

        # Should include Christmas 2024 and New Year 2025
        assert date(2024, 12, 25) in holidays
        assert date(2025, 1, 1) in holidays

    def test_get_trading_days_in_range(self):
        """Test getting trading days in a date range."""
        calendar = CMECalendar()
        trading_days = calendar.get_trading_days_in_range(
            date(2024, 12, 23),
            date(2024, 12, 27)
        )

        # Should exclude 12/25 (holiday), 12/21-22 (weekend)
        assert date(2024, 12, 23) in trading_days  # Monday
        assert date(2024, 12, 24) in trading_days  # Tuesday (early close)
        assert date(2024, 12, 25) not in trading_days  # Holiday
        assert date(2024, 12, 26) in trading_days  # Thursday
        assert date(2024, 12, 27) in trading_days  # Friday

    def test_filter_holidays(self, sample_datetime_df):
        """Test filtering holidays from DataFrame."""
        calendar = CMECalendar()

        # Add a holiday row
        holiday_row = pd.DataFrame({
            'datetime': [datetime(2024, 12, 25, 10, 0)],
            'close': [100.0],
            'volume': [500],
            'atr_20': [0.5],
        })
        df = pd.concat([sample_datetime_df, holiday_row], ignore_index=True)

        filtered = calendar.filter_holidays(df)
        # Holiday row should be removed
        assert len(filtered) == len(sample_datetime_df)

    def test_get_calendar_singleton(self):
        """Test get_calendar returns shared instance."""
        cal1 = get_calendar()
        cal2 = get_calendar()
        assert cal1 is cal2


class TestDSTHandler:
    """Tests for DSTHandler class."""

    def test_dst_handler_creation(self):
        """Test creating DSTHandler instance."""
        handler = DSTHandler('America/New_York')
        assert handler.timezone_str == 'America/New_York'

    def test_is_dst_summer(self):
        """Test DST detection in summer (EDT)."""
        handler = DSTHandler('America/New_York')
        # July is during DST
        dt = datetime(2024, 7, 15, 12, 0)
        assert handler.is_dst(dt)

    def test_is_dst_winter(self):
        """Test DST detection in winter (EST)."""
        handler = DSTHandler('America/New_York')
        # January is not during DST
        dt = datetime(2024, 1, 15, 12, 0)
        assert not handler.is_dst(dt)

    def test_get_dst_transition_dates(self):
        """Test getting DST transition dates."""
        handler = DSTHandler('America/New_York')
        spring, fall = handler.get_dst_transition_dates(2024)

        # 2024: Spring forward March 10, Fall back November 3
        assert spring == date(2024, 3, 10)
        assert fall == date(2024, 11, 3)

    def test_is_dst_transition_date(self):
        """Test DST transition date detection."""
        handler = DSTHandler('America/New_York')
        assert handler.is_dst_transition_date(date(2024, 3, 10))
        assert handler.is_dst_transition_date(date(2024, 11, 3))
        assert not handler.is_dst_transition_date(date(2024, 6, 15))

    def test_adjust_session_times_for_dst(self):
        """Test session time adjustment for DST."""
        handler = DSTHandler('America/New_York')

        base_start = (14, 30)  # NY open in EST
        base_end = (21, 0)     # NY close in EST

        # Winter (no adjustment)
        dt_winter = datetime(2024, 1, 15, 12, 0)
        adj_start, adj_end = handler.adjust_session_times_for_dst(
            base_start, base_end, dt_winter
        )
        assert adj_start == base_start
        assert adj_end == base_end

        # Summer (shift 1 hour earlier)
        dt_summer = datetime(2024, 7, 15, 12, 0)
        adj_start, adj_end = handler.adjust_session_times_for_dst(
            base_start, base_end, dt_summer
        )
        assert adj_start == (13, 30)  # 1 hour earlier
        assert adj_end == (20, 0)     # 1 hour earlier

    def test_add_dst_features(self, sample_datetime_df):
        """Test adding DST features to DataFrame."""
        handler = DSTHandler('America/New_York')
        feature_metadata = {}

        df = handler.add_dst_features(sample_datetime_df, 'datetime', feature_metadata)

        assert 'is_dst' in df.columns
        assert 'is_dst_transition' in df.columns
        assert df['is_dst'].isin([0, 1]).all()


# =============================================================================
# SESSION NORMALIZER TESTS
# =============================================================================

class TestSessionNormalizer:
    """Tests for SessionNormalizer class."""

    def test_normalizer_creation(self):
        """Test creating SessionNormalizer instance."""
        normalizer = SessionNormalizer()
        assert normalizer.method == 'zscore'
        assert not normalizer.is_fitted

    def test_normalizer_creation_invalid_method(self):
        """Test creating normalizer with invalid method."""
        with pytest.raises(ValueError, match="method must be one of"):
            SessionNormalizer(method='invalid')

    def test_fit(self, sample_ohlcv_with_sessions):
        """Test fitting normalizer."""
        normalizer = SessionNormalizer()
        df = sample_ohlcv_with_sessions
        feature_cols = ['close', 'volume']

        normalizer.fit(df, feature_cols)

        assert normalizer.is_fitted
        assert len(normalizer._feature_names) == 2

    def test_transform_before_fit(self, sample_ohlcv_with_sessions):
        """Test transform raises error if not fitted."""
        normalizer = SessionNormalizer()

        with pytest.raises(RuntimeError, match="must be fitted"):
            normalizer.transform(sample_ohlcv_with_sessions)

    def test_fit_transform_zscore(self, sample_ohlcv_with_sessions):
        """Test fit_transform with zscore method."""
        normalizer = SessionNormalizer(method='zscore')
        df = sample_ohlcv_with_sessions
        feature_cols = ['close', 'volume']

        result = normalizer.fit_transform(df, feature_cols)

        # Check normalized columns exist
        assert 'close_session_norm' in result.columns
        assert 'volume_session_norm' in result.columns

        # Normalized values should have reasonable range
        close_norm = result['close_session_norm'].dropna()
        assert close_norm.std() > 0  # Not all same value

    def test_fit_transform_robust(self, sample_ohlcv_with_sessions):
        """Test fit_transform with robust method."""
        normalizer = SessionNormalizer(method='robust')
        df = sample_ohlcv_with_sessions
        feature_cols = ['close']

        result = normalizer.fit_transform(df, feature_cols)
        assert 'close_session_norm' in result.columns

    def test_fit_transform_volatility(self, sample_ohlcv_with_sessions):
        """Test fit_transform with volatility method."""
        normalizer = SessionNormalizer(method='volatility')
        df = sample_ohlcv_with_sessions
        feature_cols = ['close']

        result = normalizer.fit_transform(df, feature_cols, volatility_col='atr_20')
        assert 'close_session_norm' in result.columns

    def test_volatility_method_requires_volatility_col(self, sample_ohlcv_with_sessions):
        """Test volatility method requires volatility_col."""
        normalizer = SessionNormalizer(method='volatility')
        df = sample_ohlcv_with_sessions

        with pytest.raises(ValueError, match="volatility_col is required"):
            normalizer.fit(df, ['close'])

    def test_get_session_stats(self, sample_ohlcv_with_sessions):
        """Test getting session volatility statistics."""
        normalizer = SessionNormalizer(method='volatility')
        df = sample_ohlcv_with_sessions

        normalizer.fit(df, ['close'], volatility_col='atr_20')
        stats = normalizer.get_session_stats()

        # Should have stats for sessions
        assert len(stats) > 0
        for session_name, session_stats in stats.items():
            assert 'mean_volatility' in session_stats
            assert 'n_samples' in session_stats

    def test_get_normalization_report(self, sample_ohlcv_with_sessions):
        """Test generating normalization report."""
        normalizer = SessionNormalizer()
        df = sample_ohlcv_with_sessions

        normalizer.fit(df, ['close', 'volume'])
        report = normalizer.get_normalization_report()

        assert report.n_features_normalized == 2
        assert isinstance(report.warnings, list)

    def test_custom_suffix(self, sample_ohlcv_with_sessions):
        """Test custom suffix for normalized columns."""
        normalizer = SessionNormalizer()
        df = sample_ohlcv_with_sessions

        result = normalizer.fit_transform(
            df, ['close'], suffix='_norm_by_session'
        )

        assert 'close_norm_by_session' in result.columns
        assert 'close_session_norm' not in result.columns


class TestNormalizerConvenienceFunctions:
    """Tests for normalizer convenience functions."""

    def test_normalize_by_session(self, sample_ohlcv_with_sessions):
        """Test normalize_by_session convenience function."""
        df, normalizer = normalize_by_session(
            sample_ohlcv_with_sessions,
            feature_cols=['close', 'volume'],
            method='zscore'
        )

        assert 'close_session_norm' in df.columns
        assert normalizer.is_fitted

    def test_get_session_volatility_ratios(self, sample_ohlcv_with_sessions):
        """Test get_session_volatility_ratios function."""
        ratios = get_session_volatility_ratios(
            sample_ohlcv_with_sessions,
            volatility_col='atr_20'
        )

        assert 'new_york' in ratios
        assert 'london' in ratios
        assert 'asia' in ratios

        # Ratios should be positive
        for session, ratio in ratios.items():
            assert ratio > 0


# =============================================================================
# EDGE CASES AND ERROR HANDLING
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_session_after_filter(self):
        """Test handling when filtering results in empty DataFrame."""
        # Create data only in NY session
        timestamps = [
            datetime(2024, 6, 17, 15, 0),  # NY session
            datetime(2024, 6, 17, 16, 0),  # NY session
        ]
        df = pd.DataFrame({
            'datetime': timestamps,
            'close': [100.0, 101.0]
        })

        # Try to filter to Asia only - should result in empty
        config = SessionsConfig(include_sessions=[SessionName.ASIA])
        filter = SessionFilter(config)

        result = filter.filter_by_session(df)
        assert len(result) == 0

    def test_all_data_in_single_session(self, sample_datetime_df):
        """Test with all data in single session."""
        # Filter to NY only
        config = SessionsConfig(include_sessions=[SessionName.NEW_YORK])
        filter = SessionFilter(config)

        df_filtered = filter.filter_by_session(sample_datetime_df)

        # Should still be able to normalize
        normalizer = SessionNormalizer()
        if len(df_filtered) > 10:  # Need some data
            result = normalizer.fit_transform(df_filtered, ['close'])
            assert 'close_session_norm' in result.columns

    def test_no_active_sessions(self):
        """Test with no active sessions (all excluded)."""
        config = SessionsConfig(
            exclude_sessions=[
                SessionName.NEW_YORK,
                SessionName.LONDON,
                SessionName.ASIA
            ]
        )

        active = config.get_active_sessions()
        assert len(active) == 0

    def test_invalid_datetime_type(self):
        """Test with non-datetime column."""
        df = pd.DataFrame({
            'datetime': ['2024-01-01', '2024-01-02'],  # Strings, not datetime
            'close': [100, 101]
        })

        filter = SessionFilter()
        with pytest.raises(TypeError, match="must be datetime type"):
            filter.add_session_features(df)

    def test_cme_holidays_future_year(self):
        """Test CME holidays for years not defined."""
        calendar = CMECalendar()

        # Year 2030 not defined - should return False
        assert not calendar.is_holiday(date(2030, 12, 25))

    def test_normalizer_no_valid_features(self):
        """Test normalizer with no valid feature columns."""
        df = pd.DataFrame({
            'datetime': pd.date_range('2024-01-01', periods=10, freq='5min'),
            'close': range(10)
        })

        normalizer = SessionNormalizer()
        with pytest.raises(ValueError, match="No valid feature columns"):
            normalizer.fit(df, ['nonexistent_column'])


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestIntegration:
    """Integration tests for the sessions module."""

    def test_full_pipeline(self, sample_ohlcv_with_sessions):
        """Test full session processing pipeline."""
        df = sample_ohlcv_with_sessions

        # 1. Add session features
        filter = SessionFilter()
        feature_metadata = {}
        df = filter.add_session_features(df, feature_metadata)

        # 2. Get session stats
        stats = filter.get_session_stats(df)
        assert len(stats) > 0

        # 3. Filter holidays
        calendar = CMECalendar()
        df = calendar.filter_holidays(df)

        # 4. Add DST features
        handler = DSTHandler('America/New_York')
        df = handler.add_dst_features(df, 'datetime', feature_metadata)

        # 5. Normalize by session
        normalizer = SessionNormalizer()
        df = normalizer.fit_transform(df, ['close', 'volume'])

        # Verify all features added
        assert 'session_new_york' in df.columns
        assert 'session_london' in df.columns
        assert 'session_asia' in df.columns
        assert 'is_dst' in df.columns
        assert 'close_session_norm' in df.columns
        assert 'volume_session_norm' in df.columns

    def test_session_filter_then_normalize(self, sample_ohlcv_with_sessions):
        """Test filtering sessions then normalizing."""
        df = sample_ohlcv_with_sessions

        # Filter to NY and London only
        config = SessionsConfig(
            include_sessions=[SessionName.NEW_YORK, SessionName.LONDON]
        )
        filter = SessionFilter(config)
        df = filter.add_session_features(df)
        df = filter.filter_by_session(df)

        # Should still be able to normalize
        normalizer = SessionNormalizer()
        df = normalizer.fit_transform(df, ['close'])

        # Verify normalization worked
        assert 'close_session_norm' in df.columns
        assert not df['close_session_norm'].isna().all()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
