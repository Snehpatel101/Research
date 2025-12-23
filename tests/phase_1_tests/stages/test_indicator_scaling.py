"""
Tests for indicator period scaling.

Validates that indicator periods scale correctly with timeframe changes
to maintain consistent lookback duration.
"""
import pytest
from stages.features.scaling import (
    scale_period,
    get_scaled_periods,
    get_unique_scaled_periods,
    create_period_config,
    get_timeframe_minutes,
    PeriodScaler,
    DEFAULT_BASE_PERIODS,
    TIMEFRAME_MINUTES,
)


class TestGetTimeframeMinutes:
    """Tests for get_timeframe_minutes function."""

    def test_common_timeframes(self):
        """Test all common timeframes are recognized."""
        assert get_timeframe_minutes('1min') == 1
        assert get_timeframe_minutes('5min') == 5
        assert get_timeframe_minutes('15min') == 15
        assert get_timeframe_minutes('30min') == 30
        assert get_timeframe_minutes('60min') == 60
        assert get_timeframe_minutes('1h') == 60

    def test_extended_timeframes(self):
        """Test extended timeframes."""
        assert get_timeframe_minutes('2h') == 120
        assert get_timeframe_minutes('4h') == 240
        assert get_timeframe_minutes('1d') == 1440

    def test_invalid_timeframe_raises(self):
        """Test that unknown timeframe raises ValueError."""
        with pytest.raises(ValueError, match="Unknown timeframe"):
            get_timeframe_minutes('invalid')

        with pytest.raises(ValueError, match="Unknown timeframe"):
            get_timeframe_minutes('3min')


class TestScalePeriod:
    """Tests for scale_period function."""

    def test_same_timeframe_no_change(self):
        """Test that same timeframe returns unchanged period."""
        assert scale_period(14, '5min', '5min') == 14
        assert scale_period(20, '1min', '1min') == 20
        assert scale_period(100, '15min', '15min') == 100

    def test_scale_up_to_higher_tf(self):
        """Test scaling from lower to higher timeframe reduces period."""
        # 14 bars @ 5min = 70min -> ~5 bars @ 15min (75min)
        assert scale_period(14, '5min', '15min') == 5

        # 20 bars @ 5min = 100min -> ~2 bars @ 60min (120min, but closer is 100)
        # 100/60 = 1.67, rounds to 2
        assert scale_period(20, '5min', '60min') == 2

        # 50 bars @ 5min = 250min -> ~8 bars @ 30min (240min)
        assert scale_period(50, '5min', '30min') == 8

    def test_scale_down_to_lower_tf(self):
        """Test scaling from higher to lower timeframe increases period."""
        # 14 bars @ 5min = 70min -> 70 bars @ 1min
        assert scale_period(14, '5min', '1min') == 70

        # 5 bars @ 15min = 75min -> 15 bars @ 5min
        assert scale_period(5, '15min', '5min') == 15

        # 2 bars @ 60min = 120min -> 24 bars @ 5min
        assert scale_period(2, '60min', '5min') == 24

    def test_minimum_period_enforced(self):
        """Test that minimum period of 2 is enforced."""
        # Very short period scaling to high TF
        # 2 bars @ 1min = 2min -> 0.033 bars @ 60min, but min is 2
        assert scale_period(2, '1min', '60min') >= 2

        # 3 bars @ 5min = 15min -> 0.25 bars @ 60min, but min is 2
        assert scale_period(3, '5min', '60min') >= 2

    def test_rounding_behavior(self):
        """Test that periods are properly rounded."""
        # 10 bars @ 5min = 50min -> 50/15 = 3.33 -> rounds to 3
        assert scale_period(10, '5min', '15min') == 3

        # 11 bars @ 5min = 55min -> 55/15 = 3.67 -> rounds to 4
        assert scale_period(11, '5min', '15min') == 4

    def test_invalid_source_tf_raises(self):
        """Test that invalid source timeframe raises ValueError."""
        with pytest.raises(ValueError, match="Unknown timeframe"):
            scale_period(14, 'invalid', '5min')

    def test_invalid_target_tf_raises(self):
        """Test that invalid target timeframe raises ValueError."""
        with pytest.raises(ValueError, match="Unknown timeframe"):
            scale_period(14, '5min', 'invalid')


class TestGetScaledPeriods:
    """Tests for get_scaled_periods function."""

    def test_scales_list_of_periods(self):
        """Test that all periods in list are scaled."""
        periods = [5, 10, 20]
        scaled = get_scaled_periods(periods, '15min', '5min')

        # 5 -> 2, 10 -> 3, 20 -> 7
        assert len(scaled) == 3
        assert scaled[0] == 2  # 5*5/15 = 1.67 -> 2
        assert scaled[1] == 3  # 10*5/15 = 3.33 -> 3
        assert scaled[2] == 7  # 20*5/15 = 6.67 -> 7

    def test_empty_list(self):
        """Test empty list returns empty."""
        assert get_scaled_periods([], '15min', '5min') == []

    def test_no_scaling_for_same_tf(self):
        """Test no change when source equals target."""
        periods = [10, 20, 50]
        scaled = get_scaled_periods(periods, '5min', '5min')
        assert scaled == periods


class TestGetUniqueScaledPeriods:
    """Tests for get_unique_scaled_periods function."""

    def test_removes_duplicates(self):
        """Test that duplicate periods are removed."""
        # When scaling down, multiple periods may collapse to same value
        periods = [5, 6, 7]  # These may all become 2 when scaled up
        scaled = get_unique_scaled_periods(periods, '15min', '5min')

        # All should be unique
        assert len(scaled) == len(set(scaled))

    def test_preserves_order(self):
        """Test that order is preserved after deduplication."""
        periods = [10, 20, 50, 100]
        scaled = get_unique_scaled_periods(periods, '5min', '5min')
        assert scaled == periods  # No change for same TF


class TestCreatePeriodConfig:
    """Tests for create_period_config function."""

    def test_base_tf_returns_unchanged(self):
        """Test that base timeframe returns unchanged periods."""
        config = create_period_config('5min', '5min')

        assert config['rsi'] == [14]
        assert config['sma'] == [10, 20, 50, 100, 200]
        assert config['ema'] == [9, 12, 21, 26, 50]
        assert config['macd_fast'] == [12]
        assert config['macd_slow'] == [26]
        assert config['macd_signal'] == [9]

    def test_higher_tf_scales_down(self):
        """Test scaling to higher timeframe reduces periods."""
        config = create_period_config('15min', '5min')

        # RSI-14 @ 5min -> RSI-5 @ 15min (70min / 15min = 4.67 -> 5)
        assert config['rsi'] == [5]

        # SMA periods should be reduced
        assert all(p <= orig for p, orig in zip(config['sma'], [10, 20, 50, 100, 200]))

    def test_lower_tf_scales_up(self):
        """Test scaling to lower timeframe increases periods."""
        config = create_period_config('1min', '5min')

        # RSI-14 @ 5min -> RSI-70 @ 1min
        assert config['rsi'] == [70]

        # SMA periods should increase 5x
        expected_sma = [50, 100, 250, 500, 1000]
        assert config['sma'] == expected_sma

    def test_all_indicator_types_present(self):
        """Test that all expected indicator types are in config."""
        config = create_period_config('5min', '5min')

        expected_keys = [
            'sma', 'ema', 'rsi', 'stochastic_k', 'stochastic_d',
            'macd_fast', 'macd_slow', 'macd_signal', 'williams_r',
            'roc', 'cci', 'mfi', 'atr', 'bollinger', 'keltner',
            'hvol', 'parkinson', 'garman_klass', 'adx',
            'supertrend_period', 'volume_sma', 'obv_sma'
        ]

        for key in expected_keys:
            assert key in config, f"Missing key: {key}"

    def test_custom_base_periods(self):
        """Test that custom base periods are used when provided."""
        custom_periods = {
            'rsi': [7, 14, 21],
            'sma': [5, 10],
        }

        config = create_period_config('5min', '5min', base_periods=custom_periods)

        assert config['rsi'] == [7, 14, 21]
        assert config['sma'] == [5, 10]


class TestPeriodScaler:
    """Tests for PeriodScaler class."""

    def test_initialization(self):
        """Test scaler initializes correctly."""
        scaler = PeriodScaler('15min', '5min')
        assert scaler.target_tf == '15min'
        assert scaler.base_tf == '5min'

    def test_get_periods(self):
        """Test get_periods returns scaled periods."""
        scaler = PeriodScaler('15min', '5min')

        # RSI should be scaled
        rsi_periods = scaler.get_periods('rsi')
        assert rsi_periods == [5]

    def test_get_period(self):
        """Test get_period returns single period."""
        scaler = PeriodScaler('5min', '5min')

        # Get first RSI period
        assert scaler.get_period('rsi') == 14
        assert scaler.get_period('rsi', 0) == 14

    def test_get_period_index_out_of_range(self):
        """Test get_period with invalid index raises."""
        scaler = PeriodScaler('5min', '5min')

        with pytest.raises(IndexError):
            scaler.get_period('rsi', 10)

    def test_get_periods_unknown_indicator(self):
        """Test get_periods with unknown indicator raises."""
        scaler = PeriodScaler('5min', '5min')

        with pytest.raises(KeyError, match="Unknown indicator"):
            scaler.get_periods('unknown')

    def test_scale_custom_period(self):
        """Test scaling a custom period."""
        scaler = PeriodScaler('15min', '5min')

        # 30 @ 5min = 150min -> 10 @ 15min
        assert scaler.scale_custom_period(30) == 10

    def test_config_property(self):
        """Test config property returns full config."""
        scaler = PeriodScaler('5min', '5min')

        config = scaler.config
        assert 'rsi' in config
        assert 'sma' in config
        assert 'atr' in config

    def test_repr(self):
        """Test string representation."""
        scaler = PeriodScaler('15min', '5min')
        assert "15min" in repr(scaler)
        assert "5min" in repr(scaler)


class TestScalingMaintainsLookback:
    """
    Integration tests verifying that scaling maintains consistent lookback.

    These tests verify the core requirement: that indicator periods
    are scaled to maintain the same real-time lookback duration.
    """

    def test_rsi_lookback_maintained(self):
        """Test RSI lookback is approximately maintained across timeframes."""
        base_period = 14
        base_tf = '5min'
        base_lookback = 14 * 5  # 70 minutes

        # Test with timeframes that don't hit the minimum period constraint
        for target_tf in ['1min', '15min', '30min']:
            scaled = scale_period(base_period, base_tf, target_tf)
            target_minutes = get_timeframe_minutes(target_tf)
            scaled_lookback = scaled * target_minutes

            # Allow 20% tolerance for rounding
            assert abs(scaled_lookback - base_lookback) <= base_lookback * 0.2, (
                f"Lookback mismatch for {target_tf}: "
                f"expected ~{base_lookback}min, got {scaled_lookback}min"
            )

    def test_minimum_period_limits_accuracy(self):
        """Test that minimum period constraint affects accuracy for high timeframes."""
        # When scaling to 60min, the minimum period of 2 kicks in
        # 14 @ 5min = 70min -> round(70/60) = 1 -> min(2) -> 120min
        scaled = scale_period(14, '5min', '60min')
        assert scaled == 2  # Minimum enforced

        # The lookback is now 120min instead of 70min
        # This is expected behavior - minimum period is required for valid calculations

    def test_sma_lookback_maintained(self):
        """Test SMA lookback is approximately maintained."""
        base_period = 20
        base_tf = '5min'
        base_lookback = 20 * 5  # 100 minutes

        for target_tf in ['1min', '10min', '15min']:
            scaled = scale_period(base_period, base_tf, target_tf)
            target_minutes = get_timeframe_minutes(target_tf)
            scaled_lookback = scaled * target_minutes

            # Allow 20% tolerance
            assert abs(scaled_lookback - base_lookback) <= base_lookback * 0.2

    def test_consistent_scaling_roundtrip(self):
        """Test that scaling up then down returns similar periods."""
        original = 14

        # Scale from 5min -> 15min -> 5min
        scaled_up = scale_period(original, '5min', '15min')  # 5
        scaled_back = scale_period(scaled_up, '15min', '5min')  # 15

        # Should be close to original (within 1 bar tolerance)
        assert abs(scaled_back - original) <= 2


class TestDefaultBasePeriods:
    """Tests for DEFAULT_BASE_PERIODS constant."""

    def test_all_periods_positive(self):
        """Test that all default periods are positive integers."""
        for indicator, periods in DEFAULT_BASE_PERIODS.items():
            for period in periods:
                assert period > 0, f"Non-positive period for {indicator}"
                assert isinstance(period, int), f"Non-integer period for {indicator}"

    def test_standard_indicator_periods(self):
        """Test that standard periods match common TA conventions."""
        # These are the commonly used periods in technical analysis
        assert DEFAULT_BASE_PERIODS['rsi'] == [14]
        assert DEFAULT_BASE_PERIODS['adx'] == [14]
        assert DEFAULT_BASE_PERIODS['bollinger'] == [20]
        assert DEFAULT_BASE_PERIODS['macd_fast'] == [12]
        assert DEFAULT_BASE_PERIODS['macd_slow'] == [26]
        assert DEFAULT_BASE_PERIODS['macd_signal'] == [9]


class TestTimeframeMinutesMapping:
    """Tests for TIMEFRAME_MINUTES constant."""

    def test_all_values_positive(self):
        """Test all timeframe values are positive."""
        for tf, minutes in TIMEFRAME_MINUTES.items():
            assert minutes > 0, f"Non-positive minutes for {tf}"

    def test_ascending_order_consistency(self):
        """Test that larger timeframe names have larger minute values."""
        assert TIMEFRAME_MINUTES['1min'] < TIMEFRAME_MINUTES['5min']
        assert TIMEFRAME_MINUTES['5min'] < TIMEFRAME_MINUTES['15min']
        assert TIMEFRAME_MINUTES['15min'] < TIMEFRAME_MINUTES['60min']
        assert TIMEFRAME_MINUTES['60min'] < TIMEFRAME_MINUTES['1d']

    def test_hourly_aliases(self):
        """Test that hourly timeframes have consistent values."""
        assert TIMEFRAME_MINUTES['60min'] == TIMEFRAME_MINUTES['1h']
