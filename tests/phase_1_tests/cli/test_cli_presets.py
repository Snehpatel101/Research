"""
Tests for CLI preset integration.

Validates that:
1. Presets can be applied via CLI --preset option
2. CLI arguments can override preset values
3. Invalid presets are rejected with clear error messages
4. Default behavior works when no preset is specified
"""
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.cli.run_commands import _create_config_from_args
from src.phase1.presets import TradingPreset, get_preset, list_available_presets


@pytest.fixture
def mock_pipeline_config():
    """Mock pipeline_config module."""
    from src.phase1 import pipeline_config
    return pipeline_config


@pytest.fixture
def mock_presets_module():
    """Mock presets module."""
    from src.phase1 import presets
    return presets


@pytest.fixture
def project_root(tmp_path):
    """Create a temporary project root directory."""
    # Create necessary subdirectories
    (tmp_path / "data" / "raw").mkdir(parents=True)
    (tmp_path / "data" / "clean").mkdir(parents=True)
    (tmp_path / "runs").mkdir(parents=True)
    return tmp_path


class TestPresetApplication:
    """Tests for applying presets via CLI."""

    def test_no_preset_uses_defaults(self, project_root, mock_pipeline_config, mock_presets_module):
        """When no preset is specified, default values are used."""
        config = _create_config_from_args(
            preset=None,
            symbols=None,  # Should use default MES,MGC
            timeframe=None,
            horizons=None,
            start=None,
            end=None,
            run_id=None,
            description=None,
            train_ratio=None,
            val_ratio=None,
            test_ratio=None,
            purge_bars=None,
            embargo_bars=None,
            synthetic=False,
            project_root_path=project_root,
            pipeline_config=mock_pipeline_config,
            presets_mod=mock_presets_module
        )

        # Check defaults are applied
        assert config.target_timeframe == '5min'
        assert config.symbols == ['MES', 'MGC']
        assert 5 in config.label_horizons
        assert 20 in config.label_horizons

    def test_scalping_preset_applied(self, project_root, mock_pipeline_config, mock_presets_module):
        """Scalping preset applies 1min timeframe and short horizons."""
        config = _create_config_from_args(
            preset='scalping',
            symbols=None,
            timeframe=None,
            horizons=None,
            start=None,
            end=None,
            run_id=None,
            description=None,
            train_ratio=None,
            val_ratio=None,
            test_ratio=None,
            purge_bars=None,
            embargo_bars=None,
            synthetic=False,
            project_root_path=project_root,
            pipeline_config=mock_pipeline_config,
            presets_mod=mock_presets_module
        )

        assert config.target_timeframe == '1min'
        assert 1 in config.label_horizons
        assert 5 in config.label_horizons
        # Scalping uses shorter indicator periods
        assert config.rsi_period == 7

    def test_day_trading_preset_applied(self, project_root, mock_pipeline_config, mock_presets_module):
        """Day trading preset applies 5min timeframe and medium horizons."""
        config = _create_config_from_args(
            preset='day_trading',
            symbols=None,
            timeframe=None,
            horizons=None,
            start=None,
            end=None,
            run_id=None,
            description=None,
            train_ratio=None,
            val_ratio=None,
            test_ratio=None,
            purge_bars=None,
            embargo_bars=None,
            synthetic=False,
            project_root_path=project_root,
            pipeline_config=mock_pipeline_config,
            presets_mod=mock_presets_module
        )

        assert config.target_timeframe == '5min'
        assert 5 in config.label_horizons
        assert 20 in config.label_horizons

    def test_swing_preset_applied(self, project_root, mock_pipeline_config, mock_presets_module):
        """Swing preset applies 15min timeframe and long horizons."""
        config = _create_config_from_args(
            preset='swing',
            symbols=None,
            timeframe=None,
            horizons=None,
            start=None,
            end=None,
            run_id=None,
            description=None,
            train_ratio=None,
            val_ratio=None,
            test_ratio=None,
            purge_bars=None,
            embargo_bars=None,
            synthetic=False,
            project_root_path=project_root,
            pipeline_config=mock_pipeline_config,
            presets_mod=mock_presets_module
        )

        assert config.target_timeframe == '15min'
        # Swing uses longer horizons
        assert 60 in config.label_horizons or 120 in config.label_horizons
        assert 20 in config.label_horizons


class TestPresetOverrides:
    """Tests for CLI argument overriding preset values."""

    def test_preset_with_symbol_override(self, project_root, mock_pipeline_config, mock_presets_module):
        """CLI --symbols overrides preset defaults."""
        config = _create_config_from_args(
            preset='day_trading',
            symbols='MES',  # Override to single symbol
            timeframe=None,
            horizons=None,
            start=None,
            end=None,
            run_id=None,
            description=None,
            train_ratio=None,
            val_ratio=None,
            test_ratio=None,
            purge_bars=None,
            embargo_bars=None,
            synthetic=False,
            project_root_path=project_root,
            pipeline_config=mock_pipeline_config,
            presets_mod=mock_presets_module
        )

        # Should use day_trading timeframe but override symbols
        assert config.target_timeframe == '5min'
        assert config.symbols == ['MES']

    def test_preset_with_horizons_override(self, project_root, mock_pipeline_config, mock_presets_module):
        """CLI --horizons overrides preset horizons."""
        config = _create_config_from_args(
            preset='scalping',
            symbols=None,
            timeframe=None,
            horizons='3,10,15',  # Override horizons
            start=None,
            end=None,
            run_id=None,
            description=None,
            train_ratio=None,
            val_ratio=None,
            test_ratio=None,
            purge_bars=None,
            embargo_bars=None,
            synthetic=False,
            project_root_path=project_root,
            pipeline_config=mock_pipeline_config,
            presets_mod=mock_presets_module
        )

        # Should use scalping timeframe but override horizons
        assert config.target_timeframe == '1min'
        assert config.label_horizons == [3, 10, 15]

    def test_preset_with_timeframe_override(self, project_root, mock_pipeline_config, mock_presets_module):
        """CLI --timeframe overrides preset timeframe."""
        config = _create_config_from_args(
            preset='day_trading',
            symbols=None,
            timeframe='10min',  # Override timeframe
            horizons=None,
            start=None,
            end=None,
            run_id=None,
            description=None,
            train_ratio=None,
            val_ratio=None,
            test_ratio=None,
            purge_bars=None,
            embargo_bars=None,
            synthetic=False,
            project_root_path=project_root,
            pipeline_config=mock_pipeline_config,
            presets_mod=mock_presets_module
        )

        # Should override timeframe but keep other day_trading settings
        assert config.target_timeframe == '10min'
        assert 5 in config.label_horizons
        assert 20 in config.label_horizons

    def test_preset_with_multiple_overrides(self, project_root, mock_pipeline_config, mock_presets_module):
        """Multiple CLI arguments can override preset values."""
        config = _create_config_from_args(
            preset='swing',
            symbols='MNQ,MES',
            timeframe='30min',
            horizons='10,40,80',
            start='2023-01-01',
            end='2024-12-31',
            run_id='custom_run',
            description='Custom swing run',
            train_ratio=0.80,
            val_ratio=0.10,
            test_ratio=0.10,
            purge_bars=None,
            embargo_bars=None,
            synthetic=False,
            project_root_path=project_root,
            pipeline_config=mock_pipeline_config,
            presets_mod=mock_presets_module
        )

        assert config.symbols == ['MNQ', 'MES']
        assert config.target_timeframe == '30min'
        assert config.label_horizons == [10, 40, 80]
        assert config.start_date == '2023-01-01'
        assert config.end_date == '2024-12-31'
        assert config.run_id == 'custom_run'
        assert config.description == 'Custom swing run'
        assert config.train_ratio == 0.80


class TestPresetValidation:
    """Tests for preset validation and error handling."""

    def test_invalid_preset_raises_error(self, project_root, mock_pipeline_config, mock_presets_module):
        """Invalid preset name raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            _create_config_from_args(
                preset='invalid_preset',
                symbols=None,
                timeframe=None,
                horizons=None,
                start=None,
                end=None,
                run_id=None,
                description=None,
                train_ratio=None,
                val_ratio=None,
                test_ratio=None,
                purge_bars=None,
                embargo_bars=None,
                synthetic=False,
                project_root_path=project_root,
                pipeline_config=mock_pipeline_config,
                presets_mod=mock_presets_module
            )

        assert 'invalid_preset' in str(exc_info.value).lower()

    def test_preset_case_insensitive(self, project_root, mock_pipeline_config, mock_presets_module):
        """Preset names should be case-insensitive."""
        # Test uppercase
        config_upper = _create_config_from_args(
            preset='SCALPING',
            symbols=None,
            timeframe=None,
            horizons=None,
            start=None,
            end=None,
            run_id=None,
            description=None,
            train_ratio=None,
            val_ratio=None,
            test_ratio=None,
            purge_bars=None,
            embargo_bars=None,
            synthetic=False,
            project_root_path=project_root,
            pipeline_config=mock_pipeline_config,
            presets_mod=mock_presets_module
        )

        # Test mixed case
        config_mixed = _create_config_from_args(
            preset='ScAlPiNg',
            symbols=None,
            timeframe=None,
            horizons=None,
            start=None,
            end=None,
            run_id=None,
            description=None,
            train_ratio=None,
            val_ratio=None,
            test_ratio=None,
            purge_bars=None,
            embargo_bars=None,
            synthetic=False,
            project_root_path=project_root,
            pipeline_config=mock_pipeline_config,
            presets_mod=mock_presets_module
        )

        assert config_upper.target_timeframe == '1min'
        assert config_mixed.target_timeframe == '1min'


class TestPresetDescriptions:
    """Tests for preset auto-descriptions."""

    def test_preset_sets_default_description(self, project_root, mock_pipeline_config, mock_presets_module):
        """When no description is provided, preset name is used."""
        config = _create_config_from_args(
            preset='day_trading',
            symbols=None,
            timeframe=None,
            horizons=None,
            start=None,
            end=None,
            run_id=None,
            description=None,  # No description
            train_ratio=None,
            val_ratio=None,
            test_ratio=None,
            purge_bars=None,
            embargo_bars=None,
            synthetic=False,
            project_root_path=project_root,
            pipeline_config=mock_pipeline_config,
            presets_mod=mock_presets_module
        )

        assert 'day' in config.description.lower() or 'trading' in config.description.lower()

    def test_explicit_description_overrides_preset(self, project_root, mock_pipeline_config, mock_presets_module):
        """Explicit description overrides preset default."""
        config = _create_config_from_args(
            preset='scalping',
            symbols=None,
            timeframe=None,
            horizons=None,
            start=None,
            end=None,
            run_id=None,
            description='My custom description',
            train_ratio=None,
            val_ratio=None,
            test_ratio=None,
            purge_bars=None,
            embargo_bars=None,
            synthetic=False,
            project_root_path=project_root,
            pipeline_config=mock_pipeline_config,
            presets_mod=mock_presets_module
        )

        assert config.description == 'My custom description'


class TestPresetFeatureConfigs:
    """Tests for preset feature configurations."""

    def test_scalping_uses_short_periods(self, project_root, mock_pipeline_config, mock_presets_module):
        """Scalping preset should use shorter indicator periods."""
        config = _create_config_from_args(
            preset='scalping',
            symbols=None,
            timeframe=None,
            horizons=None,
            start=None,
            end=None,
            run_id=None,
            description=None,
            train_ratio=None,
            val_ratio=None,
            test_ratio=None,
            purge_bars=None,
            embargo_bars=None,
            synthetic=False,
            project_root_path=project_root,
            pipeline_config=mock_pipeline_config,
            presets_mod=mock_presets_module
        )

        # Scalping should have shorter RSI period
        assert config.rsi_period < 14

    def test_swing_uses_longer_periods(self, project_root, mock_pipeline_config, mock_presets_module):
        """Swing preset should use longer indicator periods."""
        config = _create_config_from_args(
            preset='swing',
            symbols=None,
            timeframe=None,
            horizons=None,
            start=None,
            end=None,
            run_id=None,
            description=None,
            train_ratio=None,
            val_ratio=None,
            test_ratio=None,
            purge_bars=None,
            embargo_bars=None,
            synthetic=False,
            project_root_path=project_root,
            pipeline_config=mock_pipeline_config,
            presets_mod=mock_presets_module
        )

        # Swing should include longer moving average periods
        assert 200 in config.sma_periods or max(config.sma_periods) >= 100


class TestPresetIntegration:
    """Integration tests for preset functionality."""

    def test_all_presets_create_valid_configs(self, project_root, mock_pipeline_config, mock_presets_module):
        """All available presets should create valid configurations."""
        available = list_available_presets()

        for preset_name in available:
            config = _create_config_from_args(
                preset=preset_name,
                symbols=None,
                timeframe=None,
                horizons=None,
                start=None,
                end=None,
                run_id=None,
                description=None,
                train_ratio=None,
                val_ratio=None,
                test_ratio=None,
                purge_bars=None,
                embargo_bars=None,
                synthetic=False,
                project_root_path=project_root,
                pipeline_config=mock_pipeline_config,
                presets_mod=mock_presets_module
            )

            # Validate the config
            issues = config.validate()
            assert not issues, f"Preset '{preset_name}' produced invalid config: {issues}"

    def test_preset_config_matches_source(self, project_root, mock_pipeline_config, mock_presets_module):
        """Config from preset should match the source preset definition."""
        preset_def = get_preset('day_trading')

        config = _create_config_from_args(
            preset='day_trading',
            symbols=None,
            timeframe=None,
            horizons=None,
            start=None,
            end=None,
            run_id=None,
            description=None,
            train_ratio=None,
            val_ratio=None,
            test_ratio=None,
            purge_bars=None,
            embargo_bars=None,
            synthetic=False,
            project_root_path=project_root,
            pipeline_config=mock_pipeline_config,
            presets_mod=mock_presets_module
        )

        assert config.target_timeframe == preset_def['target_timeframe']
        assert config.label_horizons == preset_def['horizons']
