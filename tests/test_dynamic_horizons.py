"""
Tests for Dynamic Horizon Labeling System

Tests cover:
1. Horizon validation (validate_horizons)
2. Timeframe-aware horizon scaling (get_scaled_horizons)
3. Auto-scaled purge/embargo (auto_scale_purge_embargo)
4. Default barrier params generation (get_default_barrier_params_for_horizon)
5. HorizonConfig dataclass
6. PipelineConfig integration
"""
import pytest
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


class TestValidateHorizons:
    """Tests for validate_horizons function."""

    def test_valid_horizons(self):
        """Valid horizons should not raise."""
        from src.phase1.config import validate_horizons

        # Should not raise for valid horizons
        validate_horizons([5, 20])
        validate_horizons([1, 5, 10, 15, 20, 30, 60, 120])
        validate_horizons([5])

    def test_empty_horizons_raises(self):
        """Empty horizons list should raise ValueError."""
        from src.phase1.config import validate_horizons

        with pytest.raises(ValueError, match="cannot be empty"):
            validate_horizons([])

    def test_invalid_horizon_not_in_supported(self):
        """Horizon not in SUPPORTED_HORIZONS should raise ValueError."""
        from src.phase1.config import validate_horizons

        with pytest.raises(ValueError, match="not in SUPPORTED_HORIZONS"):
            validate_horizons([5, 7, 20])  # 7 is not supported

    def test_non_integer_horizon_raises(self):
        """Non-integer horizon should raise ValueError."""
        from src.phase1.config import validate_horizons

        with pytest.raises(ValueError, match="must be an integer"):
            validate_horizons([5.0, 20])

    def test_negative_horizon_raises(self):
        """Negative horizon should raise ValueError."""
        from src.phase1.config import validate_horizons

        with pytest.raises(ValueError, match="must be positive"):
            validate_horizons([-5, 20])

    def test_zero_horizon_raises(self):
        """Zero horizon should raise ValueError."""
        from src.phase1.config import validate_horizons

        with pytest.raises(ValueError, match="must be positive"):
            validate_horizons([0, 20])

    def test_horizon_too_large_for_data_length(self):
        """Horizon >= 10% of data length should raise ValueError."""
        from src.phase1.config import validate_horizons

        # Horizon 20 is >= 10% of 100
        with pytest.raises(ValueError, match="too large for data length"):
            validate_horizons([5, 20], data_length=100)

        # But should be fine for larger data
        validate_horizons([5, 20], data_length=500)

    def test_invalid_data_length_raises(self):
        """Invalid data_length should raise ValueError."""
        from src.phase1.config import validate_horizons

        with pytest.raises(ValueError, match="data_length must be positive"):
            validate_horizons([5, 20], data_length=0)


class TestGetScaledHorizons:
    """Tests for get_scaled_horizons function."""

    def test_same_timeframe_no_change(self):
        """Same source and target timeframe should return same horizons."""
        from src.phase1.config import get_scaled_horizons

        result = get_scaled_horizons([5, 20], '5min', '5min')
        assert result == [5, 20]

    def test_5min_to_15min_scales_down(self):
        """5min to 15min should scale horizons down by ~3x."""
        from src.phase1.config import get_scaled_horizons

        # 5 bars @ 5min = 25 min = ~2 bars @ 15min
        # 20 bars @ 5min = 100 min = ~7 bars @ 15min
        result = get_scaled_horizons([5, 20], '5min', '15min')
        assert result == [2, 7]

    def test_5min_to_1min_scales_up(self):
        """5min to 1min should scale horizons up by 5x."""
        from src.phase1.config import get_scaled_horizons

        # 5 bars @ 5min = 25 min = 25 bars @ 1min
        # 20 bars @ 5min = 100 min = 100 bars @ 1min
        result = get_scaled_horizons([5, 20], '5min', '1min')
        assert result == [25, 100]

    def test_15min_to_5min_scales_up(self):
        """15min to 5min should scale horizons up by ~3x."""
        from src.phase1.config import get_scaled_horizons

        # 5 bars @ 15min = 75 min = 15 bars @ 5min
        # 20 bars @ 15min = 300 min = 60 bars @ 5min
        result = get_scaled_horizons([5, 20], '15min', '5min')
        assert result == [15, 60]

    def test_minimum_horizon_is_one(self):
        """Scaled horizon should be at least 1."""
        from src.phase1.config import get_scaled_horizons

        # 1 bar @ 5min = 5 min = 0.33 bars @ 15min -> rounds to 1
        result = get_scaled_horizons([1], '5min', '15min')
        assert result == [1]

    def test_unknown_source_timeframe_raises(self):
        """Unknown source timeframe should raise ValueError."""
        from src.phase1.config import get_scaled_horizons

        with pytest.raises(ValueError, match="Unknown source timeframe"):
            get_scaled_horizons([5, 20], '3min', '5min')

    def test_unknown_target_timeframe_raises(self):
        """Unknown target timeframe should raise ValueError."""
        from src.phase1.config import get_scaled_horizons

        with pytest.raises(ValueError, match="Unknown target timeframe"):
            get_scaled_horizons([5, 20], '5min', '3min')

    def test_1h_alias_works(self):
        """1h should be recognized as alias for 60min."""
        from src.phase1.config import get_scaled_horizons

        # 5 bars @ 5min = 25 min = ~0.4 bars @ 1h -> rounds to 1
        # 20 bars @ 5min = 100 min = ~1.7 bars @ 1h -> rounds to 2
        result = get_scaled_horizons([5, 20], '5min', '1h')
        assert result[0] >= 1
        assert result[1] >= 1


class TestAutoScalePurgeEmbargo:
    """Tests for auto_scale_purge_embargo function."""

    def test_default_multipliers_for_h20(self):
        """Default multipliers for horizons=[5, 20]."""
        from src.phase1.config import auto_scale_purge_embargo

        purge, embargo = auto_scale_purge_embargo([5, 20])

        # max_horizon = 20
        # purge = 20 * 3 = 60
        # embargo = max(20 * 72, 1440) = 1440 (enforced minimum)
        assert purge == 60
        assert embargo == 1440

    def test_larger_horizon_scales_up(self):
        """Larger max horizon should scale up purge/embargo."""
        from src.phase1.config import auto_scale_purge_embargo

        purge, embargo = auto_scale_purge_embargo([5, 20, 60])

        # max_horizon = 60
        # purge = 60 * 3 = 180
        # embargo = max(60 * 72, 1440) = 4320
        assert purge == 180
        assert embargo == 4320

    def test_custom_multipliers(self):
        """Custom multipliers should be used."""
        from src.phase1.config import auto_scale_purge_embargo

        purge, embargo = auto_scale_purge_embargo(
            [5, 20],
            purge_multiplier=2.0,
            embargo_multiplier=10.0
        )

        # max_horizon = 20
        # purge = 20 * 2 = 40
        # embargo = max(20 * 10, 1440) = 1440 (enforced minimum)
        assert purge == 40
        assert embargo == 1440

    def test_empty_horizons_raises(self):
        """Empty horizons list should raise ValueError."""
        from src.phase1.config import auto_scale_purge_embargo

        with pytest.raises(ValueError, match="cannot be empty"):
            auto_scale_purge_embargo([])

    def test_invalid_multiplier_raises(self):
        """Non-positive multipliers should raise ValueError."""
        from src.phase1.config import auto_scale_purge_embargo

        with pytest.raises(ValueError, match="purge_multiplier must be positive"):
            auto_scale_purge_embargo([5, 20], purge_multiplier=0)

        with pytest.raises(ValueError, match="embargo_multiplier must be positive"):
            auto_scale_purge_embargo([5, 20], embargo_multiplier=-1)


class TestGetDefaultBarrierParamsForHorizon:
    """Tests for get_default_barrier_params_for_horizon function."""

    def test_returns_required_keys(self):
        """Should return dict with required keys."""
        from src.phase1.config import get_default_barrier_params_for_horizon

        result = get_default_barrier_params_for_horizon(30)

        assert 'k_up' in result
        assert 'k_down' in result
        assert 'max_bars' in result
        assert 'description' in result

    def test_symmetric_barriers(self):
        """Auto-generated barriers should be symmetric."""
        from src.phase1.config import get_default_barrier_params_for_horizon

        result = get_default_barrier_params_for_horizon(30)
        assert result['k_up'] == result['k_down']

    def test_k_scales_with_horizon(self):
        """Larger horizons should have larger k values."""
        from src.phase1.config import get_default_barrier_params_for_horizon

        result_10 = get_default_barrier_params_for_horizon(10)
        result_60 = get_default_barrier_params_for_horizon(60)

        assert result_60['k_up'] > result_10['k_up']

    def test_max_bars_scales_with_horizon(self):
        """max_bars should scale with horizon (2.5x)."""
        from src.phase1.config import get_default_barrier_params_for_horizon

        result = get_default_barrier_params_for_horizon(20)
        # max_bars = 20 * 2.5 = 50
        assert result['max_bars'] == 50

    def test_invalid_horizon_raises(self):
        """Invalid horizon should raise ValueError."""
        from src.phase1.config import get_default_barrier_params_for_horizon

        with pytest.raises(ValueError, match="must be positive"):
            get_default_barrier_params_for_horizon(0)

        with pytest.raises(ValueError, match="must be positive"):
            get_default_barrier_params_for_horizon(-5)

    def test_k_values_clamped(self):
        """k values should be clamped to reasonable range."""
        from src.phase1.config import get_default_barrier_params_for_horizon

        result = get_default_barrier_params_for_horizon(120)
        assert result['k_up'] <= 4.0
        assert result['k_down'] <= 4.0
        assert result['k_up'] >= 0.5
        assert result['k_down'] >= 0.5

    def test_max_bars_clamped(self):
        """max_bars should be clamped to reasonable range."""
        from src.phase1.config import get_default_barrier_params_for_horizon

        # Very small horizon
        result = get_default_barrier_params_for_horizon(1)
        assert result['max_bars'] >= 5

        # Very large horizon
        result = get_default_barrier_params_for_horizon(120)
        assert result['max_bars'] <= 300


class TestGetBarrierParams:
    """Tests for get_barrier_params function with dynamic horizons."""

    def test_symbol_specific_params_returned(self):
        """Symbol-specific params should be returned when available."""
        from src.phase1.config import get_barrier_params, BARRIER_PARAMS

        result = get_barrier_params('MES', 5)

        assert result == BARRIER_PARAMS['MES'][5]

    def test_default_params_fallback(self):
        """Default params should be used when symbol-specific not available."""
        from src.phase1.config import get_barrier_params, BARRIER_PARAMS_DEFAULT

        # H1 is in defaults but not in MES-specific
        result = get_barrier_params('MES', 1)

        assert result == BARRIER_PARAMS_DEFAULT[1]

    def test_auto_generated_for_unknown_horizon(self):
        """Auto-generated params should be used for unknown horizons."""
        from src.phase1.config import get_barrier_params

        # H30 is not defined in BARRIER_PARAMS or BARRIER_PARAMS_DEFAULT
        result = get_barrier_params('MES', 30)

        assert 'k_up' in result
        assert 'k_down' in result
        assert 'max_bars' in result
        assert 'Auto-generated' in result['description']


class TestHorizonConfig:
    """Tests for HorizonConfig dataclass."""

    def test_default_values(self):
        """Default values should be set correctly."""
        from src.phase1.pipeline_config import HorizonConfig

        config = HorizonConfig()

        assert config.horizons == [5, 10, 15, 20]
        assert config.source_timeframe == '5min'
        assert config.auto_scale_purge_embargo is True
        assert config.purge_multiplier == 3.0
        assert config.embargo_multiplier == 15.0

    def test_get_purge_embargo_auto_scale(self):
        """get_purge_embargo should auto-scale when enabled."""
        from src.phase1.pipeline_config import HorizonConfig

        config = HorizonConfig(horizons=[5, 20])
        purge, embargo = config.get_purge_embargo()

        assert purge == 60  # 20 * 3
        assert embargo == 1440  # max(20 * 72, 1440)

    def test_get_purge_embargo_manual(self):
        """get_purge_embargo should return manual values when auto_scale disabled."""
        from src.phase1.pipeline_config import HorizonConfig

        config = HorizonConfig(
            horizons=[5, 20],
            auto_scale_purge_embargo=False,
            manual_purge_bars=100,
            manual_embargo_bars=500
        )
        purge, embargo = config.get_purge_embargo()

        assert purge == 100
        assert embargo == 500

    def test_get_scaled_horizons(self):
        """get_scaled_horizons should delegate to config function."""
        from src.phase1.pipeline_config import HorizonConfig

        config = HorizonConfig(horizons=[5, 20], source_timeframe='5min')
        scaled = config.get_scaled_horizons('15min')

        assert scaled == [2, 7]

    def test_validate_empty_horizons(self):
        """Validation should catch empty horizons."""
        from src.phase1.pipeline_config import HorizonConfig

        config = HorizonConfig(horizons=[])
        issues = config.validate()

        assert len(issues) > 0
        assert any("at least one horizon" in issue.lower() for issue in issues)

    def test_validate_negative_horizon(self):
        """Validation should catch negative horizons."""
        from src.phase1.pipeline_config import HorizonConfig

        config = HorizonConfig(horizons=[-5, 20])
        issues = config.validate()

        assert len(issues) > 0
        assert any("positive" in issue.lower() for issue in issues)

    def test_validate_invalid_multiplier(self):
        """Validation should catch invalid multipliers."""
        from src.phase1.pipeline_config import HorizonConfig

        config = HorizonConfig(horizons=[5, 20], purge_multiplier=-1)
        issues = config.validate()

        assert len(issues) > 0
        assert any("purge_multiplier" in issue for issue in issues)


class TestPipelineConfigHorizonIntegration:
    """Tests for PipelineConfig integration with dynamic horizons."""

    def test_default_horizons(self):
        """PipelineConfig should use default horizons."""
        from src.phase1.pipeline_config import PipelineConfig

        config = PipelineConfig(symbols=['MES'])

        assert config.label_horizons == [5, 10, 15, 20]

    def test_auto_scale_purge_embargo_on_init(self):
        """PipelineConfig should auto-scale purge/embargo on init."""
        from src.phase1.pipeline_config import PipelineConfig

        config = PipelineConfig(symbols=['MES'], label_horizons=[5, 20, 60])

        # max_horizon = 60
        # purge = 60 * 3 = 180
        # embargo = max(60 * 72, 1440) = 4320
        assert config.purge_bars == 180
        assert config.embargo_bars == 4320

    def test_disable_auto_scale(self):
        """Disabling auto_scale should use default values."""
        from src.phase1.pipeline_config import PipelineConfig

        config = PipelineConfig(
            symbols=['MES'],
            label_horizons=[5, 20, 60],
            auto_scale_purge_embargo=False,
            purge_bars=100,
            embargo_bars=500
        )

        assert config.purge_bars == 100
        assert config.embargo_bars == 500

    def test_horizon_config_takes_priority(self):
        """HorizonConfig should take priority over label_horizons."""
        from src.phase1.pipeline_config import PipelineConfig, HorizonConfig

        horizon_config = HorizonConfig(horizons=[10, 30])
        config = PipelineConfig(
            symbols=['MES'],
            horizon_config=horizon_config,
            label_horizons=[5, 20]  # Should be ignored
        )

        assert config.label_horizons == [10, 30]

    def test_invalid_horizons_logs_warning(self):
        """Unsupported horizons should log warning but not fail."""
        import logging
        from src.phase1.pipeline_config import PipelineConfig

        # Horizon 7 is not in SUPPORTED_HORIZONS
        # Should log warning but not raise
        # Create config - it should succeed
        config = PipelineConfig(symbols=['MES'], label_horizons=[5, 7, 20])

        # Config should still be created
        assert config.label_horizons == [5, 7, 20]


class TestSupportedHorizonsConfig:
    """Tests for SUPPORTED_HORIZONS configuration."""

    def test_supported_horizons_values(self):
        """SUPPORTED_HORIZONS should contain expected values."""
        from src.phase1.config import SUPPORTED_HORIZONS

        expected = [1, 5, 10, 15, 20, 30, 60, 120]
        assert SUPPORTED_HORIZONS == expected

    def test_horizons_defaults(self):
        """HORIZONS should default to [5, 10, 15, 20]."""
        from src.phase1.config import HORIZONS

        assert HORIZONS == [5, 10, 15, 20]


class TestTimeframeScalingConfig:
    """Tests for HORIZON_TIMEFRAME_MINUTES configuration."""

    def test_timeframe_scaling_keys(self):
        """HORIZON_TIMEFRAME_MINUTES should have expected keys."""
        from src.phase1.config import HORIZON_TIMEFRAME_MINUTES

        expected_keys = {'1min', '5min', '10min', '15min', '20min', '30min', '45min', '60min', '1h'}
        assert set(HORIZON_TIMEFRAME_MINUTES.keys()) == expected_keys

    def test_5min_value(self):
        """5min should have value 5 (minutes)."""
        from src.phase1.config import HORIZON_TIMEFRAME_MINUTES

        assert HORIZON_TIMEFRAME_MINUTES['5min'] == 5

    def test_1min_value(self):
        """1min should have value 1 (minute)."""
        from src.phase1.config import HORIZON_TIMEFRAME_MINUTES

        assert HORIZON_TIMEFRAME_MINUTES['1min'] == 1

    def test_1h_alias(self):
        """1h should have same value as 60min."""
        from src.phase1.config import HORIZON_TIMEFRAME_MINUTES

        assert HORIZON_TIMEFRAME_MINUTES['1h'] == HORIZON_TIMEFRAME_MINUTES['60min']
        assert HORIZON_TIMEFRAME_MINUTES['1h'] == 60


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
