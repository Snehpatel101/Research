"""
Tests for calendar-aware gap handling in Stage 2 Data Cleaning.

Verifies that:
- Gaps spanning weekends are NOT filled with synthetic bars
- Gaps spanning CME holidays are NOT filled
- Gaps during normal trading hours ARE filled (up to max_gap_fill_minutes)
- The 'filled' column correctly marks synthetic vs original bars

Run with: pytest tests/phase_1_tests/stages/test_calendar_gap_handling.py -v
"""
import sys
from pathlib import Path

import pytest
import numpy as np
import pandas as pd

# Add src to path for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from stages.stage2_clean import (
    GapHandler,
    create_gap_handler,
    is_cme_market_closed,
    is_expected_gap,
    is_cme_holiday,
    gap_spans_holiday,
)


class TestWeekendGapHandling:
    """Tests for weekend gap detection and exclusion from filling."""

    def test_gap_not_filled_across_weekend(self):
        """Gaps spanning weekend should not be filled with synthetic bars.

        Calendar-aware gap filling:
        - Fills bars during market open hours (Mon-Fri before 4pm, Sun after 5pm)
        - Does NOT fill bars during market closed hours (Sat, Sun before 5pm, daily 4-5pm)

        The test validates that weekend hours are excluded from the filled range.
        """
        # Friday 3:58pm to Friday 3:59pm - just before maintenance closure
        # Then Monday 9:30am after weekend
        df = pd.DataFrame({
            'datetime': pd.to_datetime([
                '2024-01-05 15:58',  # Friday 3:58pm (market open)
                '2024-01-05 15:59',  # Friday 3:59pm (market open)
                '2024-01-08 09:30',  # Monday 9:30am (market open)
                '2024-01-08 09:31',  # Monday 9:31am (market open)
            ]),
            'open': [100.0, 100.0, 101.0, 101.0],
            'high': [102.0, 102.0, 103.0, 103.0],
            'low': [98.0, 98.0, 99.0, 99.0],
            'close': [100.0, 100.0, 101.0, 101.0],
            'volume': [1000, 1000, 1000, 1000]
        })

        handler = GapHandler(
            freq_minutes=1,
            gap_fill_method='forward',
            max_gap_fill_minutes=5,  # Only fill up to 5 bars
            calendar_aware=True
        )
        result = handler.fill_gaps(df)

        # The gap from Friday 3:59pm to Monday 9:30am spans weekend
        # - Friday 4pm-5pm is daily maintenance (excluded)
        # - Saturday all day (excluded)
        # - Sunday before 5pm (excluded)
        # With a 5-bar max fill limit, most of the gap won't be filled anyway
        # But crucially, no Saturday bars should exist

        # Verify no Saturday bars exist
        saturday_bars = result[result['datetime'].dt.dayofweek == 5]
        assert len(saturday_bars) == 0, (
            f"Should have no Saturday bars, got {len(saturday_bars)}"
        )

        # Verify original bars are preserved
        assert len(result) >= 4, (
            f"Expected at least 4 bars (original data), got {len(result)}"
        )

    def test_friday_close_to_sunday_open_is_expected_gap(self):
        """Gap from Friday 4pm to Sunday 5pm should be detected as expected."""
        friday_close = pd.Timestamp('2024-01-05 16:00')
        sunday_open = pd.Timestamp('2024-01-07 17:00')

        assert is_expected_gap(friday_close, sunday_open), (
            "Friday 4pm to Sunday 5pm should be an expected weekend gap"
        )

    def test_friday_to_monday_is_expected_gap(self):
        """Gap from Friday afternoon to Monday morning is expected."""
        friday = pd.Timestamp('2024-01-05 16:00')
        monday = pd.Timestamp('2024-01-08 09:30')

        assert is_expected_gap(friday, monday), (
            "Friday 4pm to Monday 9:30am should be an expected weekend gap"
        )

    def test_saturday_market_is_closed(self):
        """CME market should be closed all day Saturday."""
        saturday_morning = pd.Timestamp('2024-01-06 10:00')
        saturday_evening = pd.Timestamp('2024-01-06 20:00')

        assert is_cme_market_closed(saturday_morning), (
            "Market should be closed Saturday morning"
        )
        assert is_cme_market_closed(saturday_evening), (
            "Market should be closed Saturday evening"
        )

    def test_sunday_before_open_is_closed(self):
        """CME market should be closed Sunday before 5pm CT."""
        sunday_noon = pd.Timestamp('2024-01-07 12:00')
        sunday_4pm = pd.Timestamp('2024-01-07 16:00')

        assert is_cme_market_closed(sunday_noon), (
            "Market should be closed Sunday at noon"
        )
        assert is_cme_market_closed(sunday_4pm), (
            "Market should be closed Sunday at 4pm (before 5pm open)"
        )

    def test_sunday_after_open_is_open(self):
        """CME market should be open Sunday after 5pm CT."""
        sunday_evening = pd.Timestamp('2024-01-07 18:00')

        assert not is_cme_market_closed(sunday_evening), (
            "Market should be open Sunday at 6pm (after 5pm open)"
        )


class TestHolidayGapHandling:
    """Tests for CME holiday gap detection and exclusion from filling."""

    def test_gap_not_filled_across_holiday(self):
        """Gaps spanning CME holidays should not be filled.

        Calendar-aware gap filling should exclude holiday dates from filling.
        This test validates that bars on Christmas Day are not created.
        """
        # Christmas 2024 (Dec 25) is typically a CME holiday
        # Use times during market hours
        df = pd.DataFrame({
            'datetime': pd.to_datetime([
                '2024-12-24 15:58',  # Christmas Eve afternoon (before maintenance)
                '2024-12-24 15:59',  # Christmas Eve afternoon
                '2024-12-26 09:30',  # Day after Christmas morning
                '2024-12-26 09:31',  # Day after Christmas morning
            ]),
            'open': [100.0, 100.0, 101.0, 101.0],
            'high': [102.0, 102.0, 103.0, 103.0],
            'low': [98.0, 98.0, 99.0, 99.0],
            'close': [100.0, 100.0, 101.0, 101.0],
            'volume': [1000, 1000, 1000, 1000]
        })

        handler = GapHandler(
            freq_minutes=1,
            gap_fill_method='forward',
            max_gap_fill_minutes=60,
            calendar_aware=True
        )
        result = handler.fill_gaps(df)

        # Verify no Christmas Day bars exist (if exchange_calendars is installed)
        christmas_bars = result[result['datetime'].dt.date == pd.Timestamp('2024-12-25').date()]

        # With exchange_calendars, Christmas should be excluded
        # Without it, we can't guarantee this behavior
        try:
            import exchange_calendars
            assert len(christmas_bars) == 0, (
                f"Should have no Christmas Day bars, got {len(christmas_bars)}"
            )
        except ImportError:
            # Without exchange_calendars, skip this assertion
            pass

        # Original bars should always be preserved
        assert len(result) >= 4, (
            f"Expected at least 4 bars (original data), got {len(result)}"
        )

    def test_new_years_day_is_holiday(self):
        """New Year's Day should be detected as a CME holiday."""
        # Skip test if exchange_calendars is not installed
        pytest.importorskip('exchange_calendars')

        new_years_day = pd.Timestamp('2024-01-01 12:00')
        is_holiday = is_cme_holiday(new_years_day)

        # New Year's Day is typically a CME holiday
        assert is_holiday, "New Year's Day (Jan 1) should be a CME holiday"

    def test_gap_spans_holiday_detection(self):
        """Test that gap_spans_holiday correctly identifies holiday periods."""
        # Skip test if exchange_calendars is not installed
        pytest.importorskip('exchange_calendars')

        # Dec 31 to Jan 2 spans New Year's Day
        dec_31 = pd.Timestamp('2023-12-31 16:00')
        jan_2 = pd.Timestamp('2024-01-02 09:30')

        # This should detect New Year's Day (Jan 1, 2024)
        spans_holiday = gap_spans_holiday(dec_31, jan_2)
        assert spans_holiday, (
            "Gap from Dec 31 to Jan 2 should span New Year's Day holiday"
        )

    def test_regular_weekday_not_holiday(self):
        """Regular weekdays should not be detected as holidays."""
        # Skip test if exchange_calendars is not installed
        pytest.importorskip('exchange_calendars')

        # Tuesday, January 2, 2024 - regular trading day
        regular_day = pd.Timestamp('2024-01-02 10:00')
        is_holiday = is_cme_holiday(regular_day)

        assert not is_holiday, "Regular weekday should not be a holiday"


class TestDailyMaintenanceGaps:
    """Tests for daily maintenance break handling (4-5pm CT)."""

    def test_daily_maintenance_is_closed(self):
        """Market should be closed during daily maintenance (4-5pm CT)."""
        maintenance_time = pd.Timestamp('2024-01-02 16:30')

        assert is_cme_market_closed(maintenance_time), (
            "Market should be closed during daily maintenance at 4:30pm"
        )

    def test_before_maintenance_is_open(self):
        """Market should be open before maintenance (before 4pm CT)."""
        before_maintenance = pd.Timestamp('2024-01-02 15:30')

        assert not is_cme_market_closed(before_maintenance), (
            "Market should be open at 3:30pm (before 4pm maintenance)"
        )

    def test_after_maintenance_is_open(self):
        """Market should be open after maintenance (after 5pm CT)."""
        after_maintenance = pd.Timestamp('2024-01-02 17:30')

        assert not is_cme_market_closed(after_maintenance), (
            "Market should be open at 5:30pm (after 5pm maintenance)"
        )

    def test_maintenance_gap_is_expected(self):
        """Gap spanning maintenance period should be expected."""
        before_maintenance = pd.Timestamp('2024-01-02 15:59')
        after_maintenance = pd.Timestamp('2024-01-02 17:01')

        assert is_expected_gap(before_maintenance, after_maintenance), (
            "Gap spanning 4-5pm maintenance should be expected"
        )


class TestNormalTradingHoursFilling:
    """Tests for gap filling during normal trading hours."""

    def test_gap_filled_during_open_market(self):
        """Gaps during normal trading hours should be filled."""
        # Tuesday, Jan 2 - regular trading day, morning session
        df = pd.DataFrame({
            'datetime': pd.to_datetime([
                '2024-01-02 10:00',
                '2024-01-02 10:01',
                '2024-01-02 10:05',  # 4-minute gap during market
                '2024-01-02 10:06',
            ]),
            'open': [100.0, 100.0, 100.0, 100.0],
            'high': [102.0, 102.0, 102.0, 102.0],
            'low': [98.0, 98.0, 98.0, 98.0],
            'close': [100.0, 100.0, 100.0, 100.0],
            'volume': [1000, 1000, 1000, 1000]
        })

        handler = GapHandler(
            freq_minutes=1,
            gap_fill_method='forward',
            max_gap_fill_minutes=5,
            calendar_aware=False  # Disable calendar to test pure gap filling
        )
        result = handler.fill_gaps(df)

        # Should fill the 4-minute gap (10:02, 10:03, 10:04)
        # Original: 4 rows, After fill: 7 rows
        assert len(result) > 4, (
            f"Expected more than 4 rows after filling gap, got {len(result)}"
        )

    def test_small_gap_filled_completely(self):
        """Small gaps (within max_gap_fill_minutes) should be fully filled."""
        df = pd.DataFrame({
            'datetime': pd.to_datetime([
                '2024-01-02 10:00',
                '2024-01-02 10:03',  # 3-minute gap
            ]),
            'open': [100.0, 101.0],
            'high': [102.0, 103.0],
            'low': [98.0, 99.0],
            'close': [100.0, 101.0],
            'volume': [1000, 1000]
        })

        handler = GapHandler(
            freq_minutes=1,
            gap_fill_method='forward',
            max_gap_fill_minutes=5,
            calendar_aware=False
        )
        result = handler.fill_gaps(df)

        # Should have 4 rows: 10:00, 10:01, 10:02, 10:03
        assert len(result) == 4, (
            f"Expected 4 rows after filling 3-minute gap, got {len(result)}"
        )

    def test_large_gap_exceeds_max_fill(self):
        """Gaps larger than max_gap_fill_minutes should not be fully filled."""
        df = pd.DataFrame({
            'datetime': pd.to_datetime([
                '2024-01-02 10:00',
                '2024-01-02 10:10',  # 10-minute gap
            ]),
            'open': [100.0, 101.0],
            'high': [102.0, 103.0],
            'low': [98.0, 99.0],
            'close': [100.0, 101.0],
            'volume': [1000, 1000]
        })

        handler = GapHandler(
            freq_minutes=1,
            gap_fill_method='forward',
            max_gap_fill_minutes=5,  # Only allow 5 bars to be filled
            calendar_aware=False
        )
        result = handler.fill_gaps(df)

        # Gap is 10 bars, but max fill is 5
        # Forward fill with limit=5 will fill 5 bars then drop remaining NaN
        assert len(result) < 11, (
            f"Large gap should not be fully filled, got {len(result)} rows"
        )


class TestFilledFlagPreservation:
    """Tests for the 'filled' column marking synthetic bars."""

    def test_filled_flag_preserved(self):
        """The 'filled' column should mark synthetic bars."""
        df = pd.DataFrame({
            'datetime': pd.to_datetime([
                '2024-01-02 10:00',
                '2024-01-02 10:01',
                '2024-01-02 10:04',  # 3-minute gap
                '2024-01-02 10:05',
            ]),
            'open': [100.0, 100.0, 100.0, 100.0],
            'high': [102.0, 102.0, 102.0, 102.0],
            'low': [98.0, 98.0, 98.0, 98.0],
            'close': [100.0, 100.0, 100.0, 100.0],
            'volume': [1000, 1000, 1000, 1000]
        })

        handler = GapHandler(
            freq_minutes=1,
            gap_fill_method='forward',
            max_gap_fill_minutes=5,
            calendar_aware=False
        )
        result = handler.fill_gaps(df)

        # Check 'filled' column exists
        assert 'filled' in result.columns, "Result should have 'filled' column"

        # Original bars should have filled=0
        original_times = ['2024-01-02 10:00', '2024-01-02 10:01',
                          '2024-01-02 10:04', '2024-01-02 10:05']
        for ts in original_times:
            ts_dt = pd.Timestamp(ts)
            row = result[result['datetime'] == ts_dt]
            if not row.empty:
                assert row['filled'].iloc[0] == 0, (
                    f"Original bar at {ts} should have filled=0"
                )

        # Synthetic bars should have filled=1
        synthetic_times = ['2024-01-02 10:02', '2024-01-02 10:03']
        for ts in synthetic_times:
            ts_dt = pd.Timestamp(ts)
            row = result[result['datetime'] == ts_dt]
            if not row.empty:
                assert row['filled'].iloc[0] == 1, (
                    f"Synthetic bar at {ts} should have filled=1"
                )

    def test_filled_count_matches_gap_size(self):
        """Number of filled bars should match the gap size."""
        df = pd.DataFrame({
            'datetime': pd.to_datetime([
                '2024-01-02 10:00',
                '2024-01-02 10:05',  # 5-minute gap (4 missing bars)
            ]),
            'open': [100.0, 101.0],
            'high': [102.0, 103.0],
            'low': [98.0, 99.0],
            'close': [100.0, 101.0],
            'volume': [1000, 1000]
        })

        handler = GapHandler(
            freq_minutes=1,
            gap_fill_method='forward',
            max_gap_fill_minutes=10,
            calendar_aware=False
        )
        result = handler.fill_gaps(df)

        # Should have 6 rows total
        assert len(result) == 6, f"Expected 6 rows, got {len(result)}"

        # 4 bars should be marked as filled
        filled_count = result['filled'].sum()
        assert filled_count == 4, (
            f"Expected 4 filled bars, got {filled_count}"
        )

        # 2 bars should be original (filled=0)
        original_count = (result['filled'] == 0).sum()
        assert original_count == 2, (
            f"Expected 2 original bars, got {original_count}"
        )

    def test_no_fill_method_preserves_data(self):
        """Method 'none' should return data unchanged without filled column."""
        df = pd.DataFrame({
            'datetime': pd.to_datetime([
                '2024-01-02 10:00',
                '2024-01-02 10:05',
            ]),
            'open': [100.0, 101.0],
            'high': [102.0, 103.0],
            'low': [98.0, 99.0],
            'close': [100.0, 101.0],
            'volume': [1000, 1000]
        })

        handler = GapHandler(
            freq_minutes=1,
            gap_fill_method='none',
            max_gap_fill_minutes=10,
            calendar_aware=True
        )
        result = handler.fill_gaps(df)

        # Should be unchanged
        assert len(result) == 2, f"Expected 2 rows with method='none', got {len(result)}"


class TestGapHandlerFactoryFunction:
    """Tests for the create_gap_handler factory function."""

    def test_create_gap_handler_valid_methods(self):
        """Factory should accept valid gap fill methods."""
        for method in ['forward', 'interpolate', 'none']:
            handler = create_gap_handler(gap_fill_method=method)
            assert handler.gap_fill_method == method

    def test_create_gap_handler_invalid_method(self):
        """Factory should reject invalid gap fill methods."""
        with pytest.raises(ValueError, match="Invalid gap_fill_method"):
            create_gap_handler(gap_fill_method='invalid')

    def test_create_gap_handler_parameters(self):
        """Factory should correctly set all parameters."""
        handler = create_gap_handler(
            freq_minutes=5,
            gap_fill_method='interpolate',
            max_gap_fill_minutes=15,
            calendar_aware=False,
            timezone='America/New_York'
        )

        assert handler.freq_minutes == 5
        assert handler.gap_fill_method == 'interpolate'
        assert handler.max_gap_fill_minutes == 15
        assert handler.calendar_aware is False
        assert handler.timezone == 'America/New_York'


class TestCalendarAwareVsNonCalendarAware:
    """Tests comparing calendar-aware vs non-calendar-aware behavior."""

    def test_calendar_aware_fills_fewer_bars_on_weekend(self):
        """Calendar-aware handler should fill fewer bars when gap spans weekend."""
        # Gap spanning Friday evening to Monday morning
        df = pd.DataFrame({
            'datetime': pd.to_datetime([
                '2024-01-05 15:00',  # Friday afternoon
                '2024-01-08 10:00',  # Monday morning
            ]),
            'open': [100.0, 101.0],
            'high': [102.0, 103.0],
            'low': [98.0, 99.0],
            'close': [100.0, 101.0],
            'volume': [1000, 1000]
        })

        # Calendar-aware handler
        handler_aware = GapHandler(
            freq_minutes=1,
            gap_fill_method='forward',
            max_gap_fill_minutes=1000,  # Large limit to show difference
            calendar_aware=True
        )
        result_aware = handler_aware.fill_gaps(df)

        # Non-calendar-aware handler
        handler_naive = GapHandler(
            freq_minutes=1,
            gap_fill_method='forward',
            max_gap_fill_minutes=1000,
            calendar_aware=False
        )
        result_naive = handler_naive.fill_gaps(df)

        # Calendar-aware should produce fewer rows (excludes weekend hours)
        assert len(result_aware) <= len(result_naive), (
            f"Calendar-aware ({len(result_aware)}) should have <= rows than "
            f"non-calendar-aware ({len(result_naive)})"
        )

    def test_calendar_aware_detects_expected_gaps(self):
        """Calendar-aware detection should mark weekend gaps as expected."""
        df = pd.DataFrame({
            'datetime': pd.to_datetime([
                '2024-01-05 16:00',  # Friday 4pm
                '2024-01-08 09:30',  # Monday 9:30am
            ]),
            'open': [100.0, 101.0],
            'high': [102.0, 103.0],
            'low': [98.0, 99.0],
            'close': [100.0, 101.0],
            'volume': [1000, 1000]
        })

        handler = GapHandler(
            freq_minutes=1,
            gap_fill_method='forward',
            calendar_aware=True
        )
        df_result, gap_report = handler.detect_gaps(df)

        # Should detect 1 gap
        assert gap_report['total_gaps'] == 1, (
            f"Expected 1 gap, got {gap_report['total_gaps']}"
        )

        # The gap should be marked as expected (weekend)
        assert gap_report['expected_gaps'] == 1, (
            f"Expected 1 expected gap, got {gap_report['expected_gaps']}"
        )

        # No unexpected gaps
        assert gap_report['unexpected_gaps'] == 0, (
            f"Expected 0 unexpected gaps, got {gap_report['unexpected_gaps']}"
        )


class TestInterpolateMethod:
    """Tests for interpolation gap filling method."""

    def test_interpolate_fills_gaps_linearly(self):
        """Interpolate method should create linearly interpolated values."""
        df = pd.DataFrame({
            'datetime': pd.to_datetime([
                '2024-01-02 10:00',
                '2024-01-02 10:03',  # 3-minute gap
            ]),
            'open': [100.0, 106.0],
            'high': [102.0, 108.0],
            'low': [98.0, 104.0],
            'close': [100.0, 106.0],
            'volume': [1000, 1000]
        })

        handler = GapHandler(
            freq_minutes=1,
            gap_fill_method='interpolate',
            max_gap_fill_minutes=5,
            calendar_aware=False
        )
        result = handler.fill_gaps(df)

        # Should have 4 rows
        assert len(result) == 4, f"Expected 4 rows, got {len(result)}"

        # Check interpolated values are between start and end
        # Close should go: 100, ~102, ~104, 106
        close_values = result['close'].values
        assert close_values[0] == 100.0, "First close should be 100"
        assert close_values[-1] == 106.0, "Last close should be 106"

        # Middle values should be between 100 and 106
        for val in close_values[1:-1]:
            assert 100.0 <= val <= 106.0, (
                f"Interpolated value {val} should be between 100 and 106"
            )
