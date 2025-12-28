"""
Tests for the trading presets module.

This module tests:
- TradingPreset enum functionality
- PRESET_CONFIGS validation
- Preset utility functions (get_preset, validate_preset, etc.)
- Preset application to PipelineConfig
- Barrier parameter adjustments

Tests follow the project conventions:
- Deterministic and isolated
- Fast execution
- Clear assertions with specific error messages
"""

import sys
from pathlib import Path

import pytest

# Add src to path for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from src.phase1.presets import (
    TradingPreset,
    PRESET_CONFIGS,
    validate_preset,
    list_available_presets,
    get_preset,
    apply_preset_to_config,
    get_preset_summary,
    get_adjusted_barrier_params,
)


# =============================================================================
# TRADINGPRESET ENUM TESTS
# =============================================================================


class TestTradingPresetEnum:
    """Tests for the TradingPreset enumeration."""

    def test_enum_has_expected_members(self):
        """Verify all expected preset types exist."""
        assert hasattr(TradingPreset, 'SCALPING')
        assert hasattr(TradingPreset, 'DAY_TRADING')
        assert hasattr(TradingPreset, 'SWING')

    def test_enum_values_are_strings(self):
        """Verify enum values are string identifiers."""
        assert TradingPreset.SCALPING.value == 'scalping'
        assert TradingPreset.DAY_TRADING.value == 'day_trading'
        assert TradingPreset.SWING.value == 'swing'

    def test_enum_member_count(self):
        """Verify expected number of preset types."""
        assert len(TradingPreset) == 3

    def test_enum_can_be_created_from_value(self):
        """Verify enum can be instantiated from string value."""
        assert TradingPreset('scalping') == TradingPreset.SCALPING
        assert TradingPreset('day_trading') == TradingPreset.DAY_TRADING
        assert TradingPreset('swing') == TradingPreset.SWING

    def test_invalid_value_raises_error(self):
        """Verify invalid enum value raises ValueError."""
        with pytest.raises(ValueError):
            TradingPreset('invalid_preset')


# =============================================================================
# PRESET_CONFIGS VALIDATION TESTS
# =============================================================================


class TestPresetConfigs:
    """Tests for the PRESET_CONFIGS dictionary structure."""

    def test_all_presets_have_config(self):
        """Verify every TradingPreset has a corresponding config."""
        for preset in TradingPreset:
            assert preset in PRESET_CONFIGS, f"Missing config for {preset}"

    def test_scalping_config_structure(self):
        """Verify scalping preset has required keys with correct values."""
        config = PRESET_CONFIGS[TradingPreset.SCALPING]

        assert config['target_timeframe'] == '1min'
        assert config['horizons'] == [1, 5]
        assert config['sessions'] == ['new_york']
        assert config['labeling_strategy'] == 'threshold'
        assert config['barrier_multiplier'] == 0.7

    def test_day_trading_config_structure(self):
        """Verify day trading preset has required keys with correct values."""
        config = PRESET_CONFIGS[TradingPreset.DAY_TRADING]

        assert config['target_timeframe'] == '5min'
        assert config['horizons'] == [5, 20]
        assert config['sessions'] == ['new_york', 'london']
        assert config['labeling_strategy'] == 'triple_barrier'
        assert config['barrier_multiplier'] == 1.0

    def test_swing_config_structure(self):
        """Verify swing preset has required keys with correct values."""
        config = PRESET_CONFIGS[TradingPreset.SWING]

        assert config['target_timeframe'] == '15min'
        assert config['horizons'] == [20, 60, 120]
        assert config['sessions'] == ['new_york', 'london', 'asia']
        assert config['labeling_strategy'] == 'triple_barrier'
        assert config['barrier_multiplier'] == 1.3

    def test_all_configs_have_required_keys(self):
        """Verify all preset configs have required keys."""
        required_keys = [
            'name',
            'description',
            'target_timeframe',
            'horizons',
            'sessions',
            'labeling_strategy',
            'barrier_multiplier',
        ]

        for preset, config in PRESET_CONFIGS.items():
            for key in required_keys:
                assert key in config, f"Missing '{key}' in {preset.value} config"

    def test_barrier_multiplier_is_positive(self):
        """Verify barrier multipliers are positive numbers."""
        for preset, config in PRESET_CONFIGS.items():
            mult = config['barrier_multiplier']
            assert mult > 0, f"barrier_multiplier must be > 0 for {preset.value}"

    def test_horizons_are_positive_integers(self):
        """Verify all horizons are positive integers."""
        for preset, config in PRESET_CONFIGS.items():
            horizons = config['horizons']
            assert len(horizons) > 0, f"horizons must not be empty for {preset.value}"
            for h in horizons:
                assert isinstance(h, int), f"horizon must be int for {preset.value}"
                assert h > 0, f"horizon must be > 0 for {preset.value}"

    def test_sessions_are_valid(self):
        """Verify all sessions are from the valid list."""
        valid_sessions = ['new_york', 'london', 'asia']

        for preset, config in PRESET_CONFIGS.items():
            sessions = config['sessions']
            for session in sessions:
                assert session in valid_sessions, (
                    f"Invalid session '{session}' in {preset.value}"
                )

    def test_feature_config_structure(self):
        """Verify feature configs have expected structure."""
        for preset, config in PRESET_CONFIGS.items():
            if 'feature_config' in config:
                fc = config['feature_config']
                assert 'sma_periods' in fc
                assert 'ema_periods' in fc
                assert 'atr_periods' in fc
                assert 'rsi_period' in fc


# =============================================================================
# VALIDATE_PRESET FUNCTION TESTS
# =============================================================================


class TestValidatePreset:
    """Tests for the validate_preset function."""

    def test_valid_preset_returns_true(self):
        """Verify valid preset names return True."""
        assert validate_preset('scalping') is True
        assert validate_preset('day_trading') is True
        assert validate_preset('swing') is True

    def test_case_insensitive_validation(self):
        """Verify validation is case-insensitive."""
        assert validate_preset('SCALPING') is True
        assert validate_preset('Scalping') is True
        assert validate_preset('ScAlPiNg') is True

    def test_whitespace_is_stripped(self):
        """Verify leading/trailing whitespace is stripped."""
        assert validate_preset('  scalping  ') is True
        assert validate_preset('\tday_trading\n') is True

    def test_invalid_preset_raises_valueerror(self):
        """Verify invalid preset names raise ValueError."""
        with pytest.raises(ValueError) as exc_info:
            validate_preset('invalid_preset')

        assert 'Unknown preset' in str(exc_info.value)
        assert 'scalping' in str(exc_info.value)  # Should list valid presets


# =============================================================================
# LIST_AVAILABLE_PRESETS FUNCTION TESTS
# =============================================================================


class TestListAvailablePresets:
    """Tests for the list_available_presets function."""

    def test_returns_list(self):
        """Verify function returns a list."""
        result = list_available_presets()
        assert isinstance(result, list)

    def test_returns_all_presets(self):
        """Verify all presets are included."""
        result = list_available_presets()
        assert 'scalping' in result
        assert 'day_trading' in result
        assert 'swing' in result

    def test_returns_correct_count(self):
        """Verify correct number of presets returned."""
        result = list_available_presets()
        assert len(result) == 3


# =============================================================================
# GET_PRESET FUNCTION TESTS
# =============================================================================


class TestGetPreset:
    """Tests for the get_preset function."""

    def test_get_scalping_preset(self):
        """Verify scalping preset can be retrieved."""
        config = get_preset('scalping')

        assert config['target_timeframe'] == '1min'
        assert config['horizons'] == [1, 5]
        assert config['barrier_multiplier'] == 0.7

    def test_get_day_trading_preset(self):
        """Verify day trading preset can be retrieved."""
        config = get_preset('day_trading')

        assert config['target_timeframe'] == '5min'
        assert config['horizons'] == [5, 20]
        assert config['barrier_multiplier'] == 1.0

    def test_get_swing_preset(self):
        """Verify swing preset can be retrieved."""
        config = get_preset('swing')

        assert config['target_timeframe'] == '15min'
        assert config['horizons'] == [20, 60, 120]
        assert config['barrier_multiplier'] == 1.3

    def test_case_insensitive(self):
        """Verify get_preset is case-insensitive."""
        config1 = get_preset('SCALPING')
        config2 = get_preset('scalping')
        assert config1 == config2

    def test_returns_copy(self):
        """Verify get_preset returns a copy, not the original."""
        config1 = get_preset('scalping')
        config1['target_timeframe'] = 'modified'

        config2 = get_preset('scalping')
        assert config2['target_timeframe'] == '1min'

    def test_invalid_preset_raises_valueerror(self):
        """Verify invalid preset raises ValueError."""
        with pytest.raises(ValueError):
            get_preset('nonexistent_preset')


# =============================================================================
# APPLY_PRESET_TO_CONFIG FUNCTION TESTS
# =============================================================================


class TestApplyPresetToConfig:
    """Tests for the apply_preset_to_config function."""

    @pytest.fixture
    def base_config(self):
        """Create a base configuration dictionary for testing."""
        return {
            'target_timeframe': '5min',
            'label_horizons': [5, 20],
            'max_bars_ahead': 50,
            'sma_periods': [10, 20, 50, 100, 200],
            'ema_periods': [9, 21, 50],
            'atr_periods': [7, 14, 21],
            'rsi_period': 14,
            'symbols': ['MES', 'MGC'],
        }

    def test_apply_scalping_preset(self, base_config):
        """Verify scalping preset is correctly applied."""
        result = apply_preset_to_config(TradingPreset.SCALPING, base_config)

        assert result['target_timeframe'] == '1min'
        assert result['label_horizons'] == [1, 5]
        assert result['preset_barrier_multiplier'] == 0.7
        assert result['preset_labeling_strategy'] == 'threshold'

    def test_apply_day_trading_preset(self, base_config):
        """Verify day trading preset is correctly applied."""
        result = apply_preset_to_config(TradingPreset.DAY_TRADING, base_config)

        assert result['target_timeframe'] == '5min'
        assert result['label_horizons'] == [5, 20]
        assert result['preset_barrier_multiplier'] == 1.0
        assert result['preset_labeling_strategy'] == 'triple_barrier'

    def test_apply_swing_preset(self, base_config):
        """Verify swing preset is correctly applied."""
        result = apply_preset_to_config(TradingPreset.SWING, base_config)

        assert result['target_timeframe'] == '15min'
        assert result['label_horizons'] == [20, 60, 120]
        assert result['preset_barrier_multiplier'] == 1.3
        assert result['preset_labeling_strategy'] == 'triple_barrier'

    def test_accepts_string_preset(self, base_config):
        """Verify function accepts string preset name."""
        result = apply_preset_to_config('scalping', base_config)
        assert result['target_timeframe'] == '1min'

    def test_preserves_unrelated_config_values(self, base_config):
        """Verify unrelated config values are preserved."""
        result = apply_preset_to_config(TradingPreset.SCALPING, base_config)
        assert result['symbols'] == ['MES', 'MGC']

    def test_syncs_bar_resolution(self, base_config):
        """Verify bar_resolution is synced with target_timeframe."""
        result = apply_preset_to_config(TradingPreset.SCALPING, base_config)
        assert result['bar_resolution'] == result['target_timeframe']

    def test_override_conflicts_true(self, base_config):
        """Verify preset values override base config when override_conflicts=True."""
        result = apply_preset_to_config(
            TradingPreset.SCALPING,
            base_config,
            override_conflicts=True
        )
        assert result['target_timeframe'] == '1min'

    def test_override_conflicts_false(self, base_config):
        """Verify base config preserved when override_conflicts=False."""
        result = apply_preset_to_config(
            TradingPreset.SCALPING,
            base_config,
            override_conflicts=False
        )
        # target_timeframe was in base_config, should be preserved
        assert result['target_timeframe'] == '5min'

    def test_adds_preset_metadata(self, base_config):
        """Verify preset metadata is added to result."""
        result = apply_preset_to_config(TradingPreset.SCALPING, base_config)

        assert 'preset_name' in result
        assert 'preset_description' in result
        assert 'preset_sessions' in result
        assert 'preset_labeling_strategy' in result
        assert 'preset_barrier_multiplier' in result
        assert 'preset_risk_config' in result

    def test_applies_feature_config(self, base_config):
        """Verify feature config values are applied."""
        result = apply_preset_to_config(TradingPreset.SCALPING, base_config)

        # Scalping has shorter periods
        assert result['sma_periods'] == [5, 10, 20]
        assert result['ema_periods'] == [5, 9, 12]
        assert result['atr_periods'] == [5, 10]
        assert result['rsi_period'] == 7

    def test_does_not_modify_base_config(self, base_config):
        """Verify base config is not modified."""
        original_timeframe = base_config['target_timeframe']
        apply_preset_to_config(TradingPreset.SCALPING, base_config)
        assert base_config['target_timeframe'] == original_timeframe


# =============================================================================
# GET_PRESET_SUMMARY FUNCTION TESTS
# =============================================================================


class TestGetPresetSummary:
    """Tests for the get_preset_summary function."""

    def test_returns_string(self):
        """Verify function returns a string."""
        result = get_preset_summary('scalping')
        assert isinstance(result, str)

    def test_contains_preset_name(self):
        """Verify summary contains preset name."""
        result = get_preset_summary('scalping')
        assert 'Scalping' in result

    def test_contains_timeframe(self):
        """Verify summary contains timeframe."""
        result = get_preset_summary('scalping')
        assert '1min' in result

    def test_contains_horizons(self):
        """Verify summary contains horizons."""
        result = get_preset_summary('day_trading')
        assert '[5, 20]' in result

    def test_contains_sessions(self):
        """Verify summary contains sessions."""
        result = get_preset_summary('swing')
        assert 'new_york' in result
        assert 'london' in result
        assert 'asia' in result

    def test_invalid_preset_raises_error(self):
        """Verify invalid preset raises ValueError."""
        with pytest.raises(ValueError):
            get_preset_summary('invalid')


# =============================================================================
# GET_ADJUSTED_BARRIER_PARAMS FUNCTION TESTS
# =============================================================================


class TestGetAdjustedBarrierParams:
    """Tests for the get_adjusted_barrier_params function."""

    def test_scalping_reduces_barriers(self):
        """Verify scalping preset reduces barrier values (0.7x)."""
        params = get_adjusted_barrier_params('scalping', 'MES', 5)

        # Import to get base params for comparison
        from src.phase1.config import get_barrier_params
        base_params = get_barrier_params('MES', 5)

        # Scalping multiplier is 0.7
        expected_k_up = round(base_params['k_up'] * 0.7, 2)
        expected_k_down = round(base_params['k_down'] * 0.7, 2)

        assert params['k_up'] == expected_k_up
        assert params['k_down'] == expected_k_down
        assert params['barrier_multiplier'] == 0.7

    def test_day_trading_preserves_barriers(self):
        """Verify day trading preset preserves barrier values (1.0x)."""
        params = get_adjusted_barrier_params('day_trading', 'MES', 20)

        from src.phase1.config import get_barrier_params
        base_params = get_barrier_params('MES', 20)

        assert params['k_up'] == base_params['k_up']
        assert params['k_down'] == base_params['k_down']
        assert params['barrier_multiplier'] == 1.0

    def test_swing_increases_barriers(self):
        """Verify swing preset increases barrier values (1.3x)."""
        params = get_adjusted_barrier_params('swing', 'MGC', 20)

        from src.phase1.config import get_barrier_params
        base_params = get_barrier_params('MGC', 20)

        expected_k_up = round(base_params['k_up'] * 1.3, 2)
        expected_k_down = round(base_params['k_down'] * 1.3, 2)

        assert params['k_up'] == expected_k_up
        assert params['k_down'] == expected_k_down
        assert params['barrier_multiplier'] == 1.3

    def test_max_bars_unchanged(self):
        """Verify max_bars is not affected by multiplier."""
        params = get_adjusted_barrier_params('scalping', 'MES', 5)

        from src.phase1.config import get_barrier_params
        base_params = get_barrier_params('MES', 5)

        assert params['max_bars'] == base_params['max_bars']

    def test_accepts_enum(self):
        """Verify function accepts TradingPreset enum."""
        params = get_adjusted_barrier_params(TradingPreset.SCALPING, 'MES', 5)
        assert params['barrier_multiplier'] == 0.7

    def test_includes_base_params(self):
        """Verify result includes original base params."""
        params = get_adjusted_barrier_params('scalping', 'MES', 5)
        assert 'base_params' in params
        assert 'k_up' in params['base_params']

    def test_description_includes_preset_name(self):
        """Verify description includes preset name."""
        params = get_adjusted_barrier_params('swing', 'MES', 20)
        assert 'Swing' in params['description']


# =============================================================================
# INTEGRATION TESTS
# =============================================================================


class TestPresetIntegration:
    """Integration tests for presets with PipelineConfig."""

    def test_preset_with_pipeline_config(self, temp_project_dir):
        """Verify preset can be applied to real PipelineConfig."""
        from src.phase1.pipeline_config import PipelineConfig

        # Create base config
        config = PipelineConfig(symbols=['MES'], project_root=temp_project_dir)
        config_dict = config.to_dict()

        # Apply preset
        updated = apply_preset_to_config(TradingPreset.DAY_TRADING, config_dict)

        # Verify key fields
        assert updated['target_timeframe'] == '5min'
        assert updated['label_horizons'] == [5, 20]
        assert updated['preset_sessions'] == ['new_york', 'london']

    def test_all_presets_produce_valid_configs(self, temp_project_dir):
        """Verify all presets produce valid configurations."""
        from src.phase1.pipeline_config import PipelineConfig

        for preset in TradingPreset:
            config = PipelineConfig(symbols=['MES'], project_root=temp_project_dir)
            config_dict = config.to_dict()
            updated = apply_preset_to_config(preset, config_dict)

            # Verify essential keys are present and valid
            assert 'target_timeframe' in updated
            assert 'label_horizons' in updated
            assert len(updated['label_horizons']) > 0
            assert updated['preset_barrier_multiplier'] > 0

    def test_preset_barrier_params_for_all_symbols(self):
        """Verify barrier params work for all supported symbols."""
        symbols = ['MES', 'MGC']
        horizons = [5, 20]

        for preset in TradingPreset:
            for symbol in symbols:
                for horizon in horizons:
                    params = get_adjusted_barrier_params(preset, symbol, horizon)
                    assert params['k_up'] > 0
                    assert params['k_down'] > 0
                    assert params['max_bars'] > 0


# Fixture from conftest.py needs to be available
@pytest.fixture
def temp_project_dir():
    """Create a temporary project directory for testing."""
    import tempfile
    import shutil

    tmpdir = tempfile.mkdtemp()
    project_root = Path(tmpdir)

    # Create necessary subdirectories
    (project_root / 'data' / 'raw').mkdir(parents=True)
    (project_root / 'data' / 'clean').mkdir(parents=True)
    (project_root / 'data' / 'features').mkdir(parents=True)
    (project_root / 'data' / 'final').mkdir(parents=True)
    (project_root / 'data' / 'splits').mkdir(parents=True)
    (project_root / 'runs').mkdir(parents=True)

    yield project_root

    shutil.rmtree(tmpdir, ignore_errors=True)
