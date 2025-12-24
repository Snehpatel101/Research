"""
Unit tests for timeframe-aware constants in feature engineering.

Tests the dynamic annualization factor and bars_per_day calculations
that support multiple timeframes.

Run with: pytest tests/phase_1_tests/stages/test_timeframe_constants.py -v
"""

import math
import sys
from pathlib import Path

import pytest
import numpy as np

# Add src to path for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from src.phase1.stages.features.constants import (
    ANNUALIZATION_FACTOR,
    ANNUALIZATION_FACTOR_MAP,
    BARS_PER_DAY,
    BARS_PER_DAY_MAP,
    TIMEFRAME_MINUTES,
    TRADING_DAYS_PER_YEAR,
    TRADING_HOURS_EXTENDED,
    TRADING_HOURS_REGULAR,
    get_annualization_factor,
    get_bars_per_day,
)


class TestBarsPerDay:
    """Tests for bars_per_day calculation."""

    def test_5min_bars_regular_hours(self):
        """5-minute bars should have 78 bars per day (6.5 hours)."""
        result = get_bars_per_day('5min')
        assert result == 78.0

    def test_15min_bars_regular_hours(self):
        """15-minute bars should have 26 bars per day."""
        result = get_bars_per_day('15min')
        assert result == 26.0

    def test_1h_bars_regular_hours(self):
        """1-hour bars should have 6.5 bars per day."""
        result = get_bars_per_day('1h')
        assert result == 6.5

    def test_60min_bars_equals_1h(self):
        """60min and 1h should return the same value."""
        assert get_bars_per_day('60min') == get_bars_per_day('1h')

    def test_1min_bars_regular_hours(self):
        """1-minute bars should have 390 bars per day."""
        result = get_bars_per_day('1min')
        assert result == 390.0

    def test_extended_hours_higher_count(self):
        """Extended hours (23h) should have more bars than regular hours."""
        regular = get_bars_per_day('5min', extended_hours=False)
        extended = get_bars_per_day('5min', extended_hours=True)
        assert extended > regular
        # 23 hours * 60 / 5 = 276 bars
        assert extended == 276.0

    def test_10min_bars(self):
        """10-minute bars should have 39 bars per day."""
        result = get_bars_per_day('10min')
        assert result == 39.0

    def test_20min_bars(self):
        """20-minute bars should have 19.5 bars per day."""
        result = get_bars_per_day('20min')
        assert result == 19.5

    def test_30min_bars(self):
        """30-minute bars should have 13 bars per day."""
        result = get_bars_per_day('30min')
        assert result == 13.0

    def test_45min_bars(self):
        """45-minute bars should have ~8.67 bars per day."""
        result = get_bars_per_day('45min')
        expected = 6.5 * 60 / 45  # 8.666...
        assert abs(result - expected) < 0.01

    def test_case_insensitivity(self):
        """Timeframe should be case-insensitive."""
        assert get_bars_per_day('5MIN') == get_bars_per_day('5min')
        assert get_bars_per_day('1H') == get_bars_per_day('1h')

    def test_invalid_timeframe_raises_error(self):
        """Invalid timeframe should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown timeframe"):
            get_bars_per_day('2min')

        with pytest.raises(ValueError, match="Unknown timeframe"):
            get_bars_per_day('invalid')


class TestAnnualizationFactor:
    """Tests for annualization factor calculation."""

    def test_5min_factor_matches_legacy(self):
        """5-minute annualization factor should match legacy constant."""
        factor = get_annualization_factor('5min')
        expected = math.sqrt(78 * 252)
        assert abs(factor - expected) < 0.01
        # Also check legacy constant
        assert abs(factor - ANNUALIZATION_FACTOR) < 0.01

    def test_15min_factor(self):
        """15-minute annualization factor should be sqrt(26 * 252)."""
        factor = get_annualization_factor('15min')
        expected = math.sqrt(26 * 252)
        assert abs(factor - expected) < 0.01

    def test_1h_factor(self):
        """1-hour annualization factor should be sqrt(6.5 * 252)."""
        factor = get_annualization_factor('1h')
        expected = math.sqrt(6.5 * 252)
        assert abs(factor - expected) < 0.01

    def test_different_timeframes_different_factors(self):
        """Different timeframes should have different annualization factors."""
        f1min = get_annualization_factor('1min')
        f5min = get_annualization_factor('5min')
        f15min = get_annualization_factor('15min')
        f1h = get_annualization_factor('1h')

        # More bars = higher factor
        assert f1min > f5min > f15min > f1h

    def test_factor_proportionality(self):
        """Factor should scale with sqrt of bars_per_day ratio."""
        f5min = get_annualization_factor('5min')
        f15min = get_annualization_factor('15min')

        # bars_per_day ratio: 78/26 = 3
        # factor ratio should be sqrt(3) ~ 1.732
        ratio = f5min / f15min
        expected_ratio = math.sqrt(78 / 26)
        assert abs(ratio - expected_ratio) < 0.01

    def test_extended_hours_factor(self):
        """Extended hours should have higher annualization factor."""
        regular = get_annualization_factor('5min', extended_hours=False)
        extended = get_annualization_factor('5min', extended_hours=True)
        assert extended > regular

    def test_invalid_timeframe_raises_error(self):
        """Invalid timeframe should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown timeframe"):
            get_annualization_factor('3min')


class TestPrecomputedMaps:
    """Tests for pre-computed lookup tables."""

    def test_bars_per_day_map_completeness(self):
        """BARS_PER_DAY_MAP should contain all supported timeframes."""
        for tf in TIMEFRAME_MINUTES.keys():
            assert tf in BARS_PER_DAY_MAP

    def test_annualization_factor_map_completeness(self):
        """ANNUALIZATION_FACTOR_MAP should contain all supported timeframes."""
        for tf in TIMEFRAME_MINUTES.keys():
            assert tf in ANNUALIZATION_FACTOR_MAP

    def test_bars_per_day_map_values_match_function(self):
        """Map values should match function output."""
        for tf, expected in BARS_PER_DAY_MAP.items():
            result = get_bars_per_day(tf)
            assert result == expected, f"Mismatch for {tf}: {result} != {expected}"

    def test_annualization_factor_map_values_match_function(self):
        """Map values should match function output."""
        for tf, expected in ANNUALIZATION_FACTOR_MAP.items():
            result = get_annualization_factor(tf)
            assert abs(result - expected) < 0.001, f"Mismatch for {tf}: {result} != {expected}"


class TestBackwardCompatibility:
    """Tests for backward compatibility with legacy constants."""

    def test_bars_per_day_constant(self):
        """Legacy BARS_PER_DAY should be 78 (5-minute default)."""
        assert BARS_PER_DAY == 78

    def test_annualization_factor_constant(self):
        """Legacy ANNUALIZATION_FACTOR should match 5-minute calculation."""
        expected = np.sqrt(252 * 78)
        assert abs(ANNUALIZATION_FACTOR - expected) < 0.01

    def test_trading_days_per_year(self):
        """TRADING_DAYS_PER_YEAR should be 252."""
        assert TRADING_DAYS_PER_YEAR == 252

    def test_trading_hours_regular(self):
        """Regular trading hours should be 6.5."""
        assert TRADING_HOURS_REGULAR == 6.5

    def test_trading_hours_extended(self):
        """Extended trading hours should be 23."""
        assert TRADING_HOURS_EXTENDED == 23.0


class TestTimeframeMinutesMapping:
    """Tests for TIMEFRAME_MINUTES mapping."""

    def test_all_expected_timeframes_present(self):
        """All expected timeframes should be in the mapping."""
        expected = ['1min', '5min', '10min', '15min', '20min', '30min', '45min', '60min', '1h']
        for tf in expected:
            assert tf in TIMEFRAME_MINUTES

    def test_correct_minute_values(self):
        """Minute values should be correct."""
        assert TIMEFRAME_MINUTES['1min'] == 1
        assert TIMEFRAME_MINUTES['5min'] == 5
        assert TIMEFRAME_MINUTES['10min'] == 10
        assert TIMEFRAME_MINUTES['15min'] == 15
        assert TIMEFRAME_MINUTES['60min'] == 60
        assert TIMEFRAME_MINUTES['1h'] == 60


class TestMathematicalConsistency:
    """Tests for mathematical consistency of calculations."""

    def test_bars_times_minutes_equals_trading_minutes(self):
        """bars_per_day * minutes_per_bar should equal trading minutes per day."""
        trading_minutes = TRADING_HOURS_REGULAR * 60  # 390 minutes

        for tf, minutes in TIMEFRAME_MINUTES.items():
            bars = get_bars_per_day(tf)
            calculated_minutes = bars * minutes
            assert abs(calculated_minutes - trading_minutes) < 0.01, \
                f"Mismatch for {tf}: {calculated_minutes} != {trading_minutes}"

    def test_annualization_squared_equals_bars_times_days(self):
        """Annualization factor squared should equal bars_per_day * 252."""
        for tf in TIMEFRAME_MINUTES.keys():
            factor = get_annualization_factor(tf)
            bars = get_bars_per_day(tf)
            expected = bars * TRADING_DAYS_PER_YEAR
            calculated = factor ** 2
            assert abs(calculated - expected) < 0.01, \
                f"Mismatch for {tf}: {calculated} != {expected}"
