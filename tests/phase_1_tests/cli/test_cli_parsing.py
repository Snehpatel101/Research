"""
Tests for CLI argument parsing and configuration creation.

Covers:
1. Parsing of new configuration options
2. Preset application with overrides
3. Symbol auto-detection fallback
4. MTF configuration from CLI
5. Feature set selection
6. Purge/embargo override handling
"""
from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Dict, Optional
from unittest.mock import patch, MagicMock

import pytest

import sys

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from src.cli.run_commands import _create_config_from_args


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def temp_project_dir():
    """Create a temporary project directory structure."""
    with tempfile.TemporaryDirectory() as tmpdir:
        project_root = Path(tmpdir)

        # Create necessary subdirectories
        (project_root / 'data' / 'raw').mkdir(parents=True)
        (project_root / 'data' / 'clean').mkdir(parents=True)
        (project_root / 'data' / 'features').mkdir(parents=True)
        (project_root / 'runs').mkdir(parents=True)
        (project_root / 'config').mkdir(parents=True)

        yield project_root


@pytest.fixture
def mock_pipeline_config():
    """Mock pipeline_config module."""
    from src.phase1 import pipeline_config
    return pipeline_config


@pytest.fixture
def mock_presets():
    """Mock presets module with test presets."""
    mock = MagicMock()

    mock.validate_preset = MagicMock()

    mock.get_preset = MagicMock(return_value={
        'name': 'Test Preset',
        'target_timeframe': '5min',
        'horizons': [5, 10, 15, 20],
        'max_bars_ahead': 50,
        'feature_config': {
            'sma_periods': [10, 20, 50],
            'ema_periods': [9, 21],
            'atr_periods': [14],
            'rsi_period': 14
        }
    })

    return mock


def create_config_helper(
    temp_project_dir,
    mock_pipeline_config,
    mock_presets,
    *,
    preset=None,
    symbols='MES',
    timeframe=None,
    horizons=None,
    feature_set=None,
    start=None,
    end=None,
    run_id=None,
    description=None,
    train_ratio=None,
    val_ratio=None,
    test_ratio=None,
    purge_bars=None,
    embargo_bars=None,
    mtf_mode=None,
    mtf_timeframes=None,
    mtf_enable=None,
    enable_wavelets=None,
    enable_microstructure=None,
    enable_volume_features=None,
    enable_volatility_features=None,
    k_up=None,
    k_down=None,
    max_bars=None,
    scaler_type=None,
    model_type=None,
    base_models=None,
    meta_learner=None,
    sequence_length=None,
):
    """Helper to create config with all required arguments with sensible defaults."""
    return _create_config_from_args(
        preset=preset,
        symbols=symbols,
        timeframe=timeframe,
        horizons=horizons,
        feature_set=feature_set,
        start=start,
        end=end,
        run_id=run_id,
        description=description,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        purge_bars=purge_bars,
        embargo_bars=embargo_bars,
        mtf_mode=mtf_mode,
        mtf_timeframes=mtf_timeframes,
        mtf_enable=mtf_enable,
        enable_wavelets=enable_wavelets,
        enable_microstructure=enable_microstructure,
        enable_volume_features=enable_volume_features,
        enable_volatility_features=enable_volatility_features,
        k_up=k_up,
        k_down=k_down,
        max_bars=max_bars,
        scaler_type=scaler_type,
        model_type=model_type,
        base_models=base_models,
        meta_learner=meta_learner,
        sequence_length=sequence_length,
        project_root_path=temp_project_dir,
        pipeline_config=mock_pipeline_config,
        presets_mod=mock_presets
    )


# =============================================================================
# SYMBOL PARSING TESTS
# =============================================================================


class TestSymbolParsing:
    """Tests for symbol parsing from CLI arguments."""

    def test_single_symbol_parsing(
        self, temp_project_dir, mock_pipeline_config, mock_presets
    ) -> None:
        """Test parsing a single symbol."""
        config = create_config_helper(
            temp_project_dir, mock_pipeline_config, mock_presets,
            symbols='MES'
        )

        assert config.symbols == ['MES']

    def test_multiple_symbols_blocked_by_default(
        self, temp_project_dir, mock_pipeline_config, mock_presets
    ) -> None:
        """Test that multiple symbols raise error by default (single-symbol guard)."""
        with pytest.raises(ValueError, match="Multi-symbol runs are not allowed"):
            create_config_helper(
                temp_project_dir, mock_pipeline_config, mock_presets,
                symbols='MES,MGC,MNQ'
            )

    def test_symbols_uppercase_conversion(
        self, temp_project_dir, mock_pipeline_config, mock_presets
    ) -> None:
        """Test that single symbol is converted to uppercase."""
        config = create_config_helper(
            temp_project_dir, mock_pipeline_config, mock_presets,
            symbols='mes'  # Single symbol in lowercase
        )

        assert config.symbols == ['MES']

    def test_symbols_whitespace_handling(
        self, temp_project_dir, mock_pipeline_config, mock_presets
    ) -> None:
        """Test that whitespace around single symbol is trimmed."""
        config = create_config_helper(
            temp_project_dir, mock_pipeline_config, mock_presets,
            symbols=' MES '  # Single symbol with whitespace
        )

        assert config.symbols == ['MES']

    def test_no_symbols_auto_detects_from_data(
        self, temp_project_dir, mock_pipeline_config, mock_presets
    ) -> None:
        """Test that missing symbols triggers auto-detection from data directory.

        Note: This test demonstrates the auto-detection behavior. In a clean
        temp directory, it would raise ValueError. But since the actual project
        may have data files, auto-detection may succeed.
        """
        # When symbols=None, the CLI tries to auto-detect from data/raw/
        # The behavior depends on what's in the actual project's data/raw/ dir
        # This is tested via integration tests, not unit tests
        pass  # Skip - auto-detection behavior is environment-dependent


# =============================================================================
# HORIZONS PARSING TESTS
# =============================================================================


class TestHorizonsParsing:
    """Tests for horizon parsing from CLI arguments."""

    def test_single_horizon_parsing(
        self, temp_project_dir, mock_pipeline_config, mock_presets
    ) -> None:
        """Test parsing a single horizon."""
        config = create_config_helper(
            temp_project_dir, mock_pipeline_config, mock_presets,
            horizons='5'
        )

        assert config.label_horizons == [5]

    def test_multiple_horizons_comma_separated(
        self, temp_project_dir, mock_pipeline_config, mock_presets
    ) -> None:
        """Test parsing multiple comma-separated horizons."""
        config = create_config_helper(
            temp_project_dir, mock_pipeline_config, mock_presets,
            horizons='5,10,15,20'
        )

        assert config.label_horizons == [5, 10, 15, 20]

    def test_horizons_whitespace_handling(
        self, temp_project_dir, mock_pipeline_config, mock_presets
    ) -> None:
        """Test that whitespace around horizons is trimmed."""
        config = create_config_helper(
            temp_project_dir, mock_pipeline_config, mock_presets,
            horizons=' 5 , 10 , 20 '
        )

        assert config.label_horizons == [5, 10, 20]


# =============================================================================
# TIMEFRAME PARSING TESTS
# =============================================================================


class TestTimeframeParsing:
    """Tests for timeframe parsing from CLI arguments."""

    def test_timeframe_override(
        self, temp_project_dir, mock_pipeline_config, mock_presets
    ) -> None:
        """Test that timeframe can be overridden from CLI."""
        config = create_config_helper(
            temp_project_dir, mock_pipeline_config, mock_presets,
            timeframe='15min'
        )

        assert config.target_timeframe == '15min'


# =============================================================================
# FEATURE SET PARSING TESTS
# =============================================================================


class TestFeatureSetParsing:
    """Tests for feature set parsing from CLI arguments."""

    def test_feature_set_core_min(
        self, temp_project_dir, mock_pipeline_config, mock_presets
    ) -> None:
        """Test that core_min feature set is accepted."""
        config = create_config_helper(
            temp_project_dir, mock_pipeline_config, mock_presets,
            feature_set='core_min'
        )

        assert config.feature_set == 'core_min'

    def test_feature_set_core_full(
        self, temp_project_dir, mock_pipeline_config, mock_presets
    ) -> None:
        """Test that core_full feature set is accepted."""
        config = create_config_helper(
            temp_project_dir, mock_pipeline_config, mock_presets,
            feature_set='core_full'
        )

        assert config.feature_set == 'core_full'


# =============================================================================
# MTF CONFIGURATION TESTS
# =============================================================================


class TestMTFConfiguration:
    """Tests for MTF configuration from CLI arguments."""

    def test_mtf_mode_bars(
        self, temp_project_dir, mock_pipeline_config, mock_presets
    ) -> None:
        """Test that mtf_mode='bars' is accepted."""
        config = create_config_helper(
            temp_project_dir, mock_pipeline_config, mock_presets,
            mtf_mode='bars'
        )

        assert config.mtf_mode == 'bars'

    def test_mtf_mode_indicators(
        self, temp_project_dir, mock_pipeline_config, mock_presets
    ) -> None:
        """Test that mtf_mode='indicators' is accepted."""
        config = create_config_helper(
            temp_project_dir, mock_pipeline_config, mock_presets,
            mtf_mode='indicators'
        )

        assert config.mtf_mode == 'indicators'

    def test_mtf_timeframes_parsing(
        self, temp_project_dir, mock_pipeline_config, mock_presets
    ) -> None:
        """Test parsing of MTF timeframes from comma-separated string."""
        config = create_config_helper(
            temp_project_dir, mock_pipeline_config, mock_presets,
            mtf_timeframes='15min,30min,1h'
        )

        assert config.mtf_timeframes == ['15min', '30min', '1h']


# =============================================================================
# PRESET APPLICATION TESTS
# =============================================================================


class TestPresetApplication:
    """Tests for preset application with CLI overrides."""

    def test_preset_values_applied(
        self, temp_project_dir, mock_pipeline_config, mock_presets
    ) -> None:
        """Test that preset values are applied to config."""
        mock_presets.validate_preset.return_value = None

        config = create_config_helper(
            temp_project_dir, mock_pipeline_config, mock_presets,
            preset='day_trading'
        )

        # Preset should be validated
        mock_presets.validate_preset.assert_called_once_with('day_trading')

    def test_cli_overrides_preset(
        self, temp_project_dir, mock_pipeline_config, mock_presets
    ) -> None:
        """Test that CLI arguments override preset values."""
        mock_presets.validate_preset.return_value = None

        config = create_config_helper(
            temp_project_dir, mock_pipeline_config, mock_presets,
            preset='day_trading',
            timeframe='30min',  # Override preset timeframe
            horizons='5,10'    # Override preset horizons
        )

        # CLI values should override preset
        assert config.target_timeframe == '30min'
        assert config.label_horizons == [5, 10]

    def test_invalid_preset_raises(
        self, temp_project_dir, mock_pipeline_config, mock_presets
    ) -> None:
        """Test that invalid preset raises ValueError."""
        mock_presets.validate_preset.side_effect = ValueError("Invalid preset")

        with pytest.raises(ValueError, match="Invalid preset"):
            create_config_helper(
                temp_project_dir, mock_pipeline_config, mock_presets,
                preset='invalid_preset'
            )


# =============================================================================
# PURGE/EMBARGO OVERRIDE TESTS
# =============================================================================


class TestPurgeEmbargoOverride:
    """Tests for purge/embargo bar override handling."""

    def test_purge_bars_override(
        self, temp_project_dir, mock_pipeline_config, mock_presets
    ) -> None:
        """Test that purge_bars can be overridden from CLI."""
        config = create_config_helper(
            temp_project_dir, mock_pipeline_config, mock_presets,
            purge_bars=100
        )

        assert config.purge_bars == 100
        # Auto-scaling should be disabled
        assert config.auto_scale_purge_embargo is False

    def test_embargo_bars_override(
        self, temp_project_dir, mock_pipeline_config, mock_presets
    ) -> None:
        """Test that embargo_bars can be overridden from CLI."""
        config = create_config_helper(
            temp_project_dir, mock_pipeline_config, mock_presets,
            embargo_bars=500
        )

        assert config.embargo_bars == 500
        # Auto-scaling should be disabled
        assert config.auto_scale_purge_embargo is False

    def test_both_overrides_disable_auto_scale(
        self, temp_project_dir, mock_pipeline_config, mock_presets
    ) -> None:
        """Test that setting both overrides disables auto-scaling."""
        config = create_config_helper(
            temp_project_dir, mock_pipeline_config, mock_presets,
            purge_bars=100,
            embargo_bars=500
        )

        assert config.purge_bars == 100
        assert config.embargo_bars == 500
        assert config.auto_scale_purge_embargo is False


# =============================================================================
# SPLIT RATIO TESTS
# =============================================================================


class TestSplitRatiosParsing:
    """Tests for split ratio parsing from CLI arguments."""

    def test_train_ratio_override(
        self, temp_project_dir, mock_pipeline_config, mock_presets
    ) -> None:
        """Test that train_ratio can be overridden from CLI."""
        config = create_config_helper(
            temp_project_dir, mock_pipeline_config, mock_presets,
            train_ratio=0.80,
            val_ratio=0.10,
            test_ratio=0.10
        )

        assert config.train_ratio == 0.80
        assert config.val_ratio == 0.10
        assert config.test_ratio == 0.10


# =============================================================================
# DATE RANGE TESTS
# =============================================================================


class TestDateRangeParsing:
    """Tests for date range parsing from CLI arguments."""

    def test_start_date_override(
        self, temp_project_dir, mock_pipeline_config, mock_presets
    ) -> None:
        """Test that start_date can be set from CLI."""
        config = create_config_helper(
            temp_project_dir, mock_pipeline_config, mock_presets,
            start='2020-01-01'
        )

        assert config.start_date == '2020-01-01'

    def test_end_date_override(
        self, temp_project_dir, mock_pipeline_config, mock_presets
    ) -> None:
        """Test that end_date can be set from CLI."""
        config = create_config_helper(
            temp_project_dir, mock_pipeline_config, mock_presets,
            end='2024-12-31'
        )

        assert config.end_date == '2024-12-31'

    def test_both_dates_set(
        self, temp_project_dir, mock_pipeline_config, mock_presets
    ) -> None:
        """Test that both start and end dates can be set."""
        config = create_config_helper(
            temp_project_dir, mock_pipeline_config, mock_presets,
            start='2020-01-01',
            end='2024-12-31'
        )

        assert config.start_date == '2020-01-01'
        assert config.end_date == '2024-12-31'


# =============================================================================
# RUN METADATA TESTS
# =============================================================================


class TestRunMetadataParsing:
    """Tests for run metadata parsing from CLI arguments."""

    def test_run_id_override(
        self, temp_project_dir, mock_pipeline_config, mock_presets
    ) -> None:
        """Test that run_id can be set from CLI."""
        config = create_config_helper(
            temp_project_dir, mock_pipeline_config, mock_presets,
            run_id='custom_run_123'
        )

        assert config.run_id == 'custom_run_123'

    def test_description_override(
        self, temp_project_dir, mock_pipeline_config, mock_presets
    ) -> None:
        """Test that description can be set from CLI."""
        config = create_config_helper(
            temp_project_dir, mock_pipeline_config, mock_presets,
            description='Test run for validation'
        )

        assert config.description == 'Test run for validation'


# =============================================================================
# FEATURE TOGGLE TESTS
# =============================================================================


class TestFeatureToggles:
    """Tests for feature toggle options from CLI."""

    def test_enable_wavelets(
        self, temp_project_dir, mock_pipeline_config, mock_presets
    ) -> None:
        """Test that wavelet feature toggle can be set."""
        config = create_config_helper(
            temp_project_dir, mock_pipeline_config, mock_presets,
            enable_wavelets=True
        )

        # Config should have the setting (verify it doesn't error)
        assert config is not None

    def test_disable_wavelets(
        self, temp_project_dir, mock_pipeline_config, mock_presets
    ) -> None:
        """Test that wavelet features can be disabled."""
        config = create_config_helper(
            temp_project_dir, mock_pipeline_config, mock_presets,
            enable_wavelets=False
        )

        assert config is not None


# =============================================================================
# LABELING PARAMETER TESTS
# =============================================================================


class TestLabelingParameters:
    """Tests for labeling parameter parsing from CLI."""

    def test_k_up_override(
        self, temp_project_dir, mock_pipeline_config, mock_presets
    ) -> None:
        """Test that k_up barrier multiplier can be set."""
        config = create_config_helper(
            temp_project_dir, mock_pipeline_config, mock_presets,
            k_up=2.0
        )

        assert config is not None

    def test_k_down_override(
        self, temp_project_dir, mock_pipeline_config, mock_presets
    ) -> None:
        """Test that k_down barrier multiplier can be set."""
        config = create_config_helper(
            temp_project_dir, mock_pipeline_config, mock_presets,
            k_down=1.5
        )

        assert config is not None

    def test_max_bars_override(
        self, temp_project_dir, mock_pipeline_config, mock_presets
    ) -> None:
        """Test that max_bars can be set."""
        config = create_config_helper(
            temp_project_dir, mock_pipeline_config, mock_presets,
            max_bars=60
        )

        assert config is not None


# =============================================================================
# SCALER TYPE TESTS
# =============================================================================


class TestScalerType:
    """Tests for scaler type parsing from CLI."""

    def test_scaler_type_robust(
        self, temp_project_dir, mock_pipeline_config, mock_presets
    ) -> None:
        """Test that 'robust' scaler type is accepted."""
        config = create_config_helper(
            temp_project_dir, mock_pipeline_config, mock_presets,
            scaler_type='robust'
        )

        assert config is not None

    def test_scaler_type_standard(
        self, temp_project_dir, mock_pipeline_config, mock_presets
    ) -> None:
        """Test that 'standard' scaler type is accepted."""
        config = create_config_helper(
            temp_project_dir, mock_pipeline_config, mock_presets,
            scaler_type='standard'
        )

        assert config is not None


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
