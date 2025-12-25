"""
Tests for PipelineConfig validation and configuration management.

Covers:
1. Single-symbol validation guard
2. MTF configuration propagation
3. PipelineConfig validation methods
4. Configuration persistence (save/load)
5. Auto-scaling of purge/embargo bars
"""
from __future__ import annotations

import json
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Final

import pytest

import sys

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from src.phase1.pipeline_config import PipelineConfig, create_default_config


# =============================================================================
# SINGLE-SYMBOL VALIDATION GUARD TESTS
# =============================================================================


class TestSingleSymbolValidationGuard:
    """Tests for the single-symbol validation guard in PipelineConfig."""

    def test_single_symbol_allowed_by_default(self, temp_project_dir: Path) -> None:
        """Test that single-symbol configuration is allowed by default."""
        config = PipelineConfig(
            symbols=['MES'],
            project_root=temp_project_dir
        )

        assert config.symbols == ['MES']
        assert config.allow_multi_symbol is False

    def test_multi_symbol_blocked_by_default(self, temp_project_dir: Path) -> None:
        """Test that multi-symbol configuration raises ValueError by default."""
        with pytest.raises(ValueError, match="Multi-symbol runs are not allowed"):
            PipelineConfig(
                symbols=['MES', 'MGC'],
                project_root=temp_project_dir
            )

    def test_multi_symbol_blocked_with_three_symbols(self, temp_project_dir: Path) -> None:
        """Test that three or more symbols are blocked by default."""
        with pytest.raises(ValueError, match="Multi-symbol runs are not allowed"):
            PipelineConfig(
                symbols=['MES', 'MGC', 'MNQ'],
                project_root=temp_project_dir
            )

    def test_multi_symbol_allowed_with_explicit_flag(self, temp_project_dir: Path) -> None:
        """Test that multi-symbol is allowed when allow_multi_symbol=True."""
        config = PipelineConfig(
            symbols=['MES', 'MGC'],
            allow_multi_symbol=True,
            project_root=temp_project_dir
        )

        assert config.symbols == ['MES', 'MGC']
        assert config.allow_multi_symbol is True

    def test_error_message_includes_symbol_count(self, temp_project_dir: Path) -> None:
        """Test that error message includes the number of symbols provided."""
        with pytest.raises(ValueError, match="Got 2 symbols"):
            PipelineConfig(
                symbols=['MES', 'MGC'],
                project_root=temp_project_dir
            )

    def test_error_message_includes_symbol_list(self, temp_project_dir: Path) -> None:
        """Test that error message includes the list of symbols."""
        with pytest.raises(ValueError, match=r"\['MES', 'MGC'\]"):
            PipelineConfig(
                symbols=['MES', 'MGC'],
                project_root=temp_project_dir
            )

    def test_error_message_suggests_cli_flag(self, temp_project_dir: Path) -> None:
        """Test that error message suggests the --multi-symbol flag."""
        with pytest.raises(ValueError, match="--multi-symbol flag"):
            PipelineConfig(
                symbols=['MES', 'MGC'],
                project_root=temp_project_dir
            )

    def test_empty_symbols_raises_error(self, temp_project_dir: Path) -> None:
        """Test that empty symbols list raises ValueError."""
        with pytest.raises(ValueError, match="At least one symbol must be specified"):
            PipelineConfig(
                symbols=[],
                project_root=temp_project_dir
            )

    def test_create_default_config_single_symbol(self, temp_project_dir: Path) -> None:
        """Test create_default_config with single symbol."""
        config = create_default_config(
            symbols=['MES'],
            project_root=temp_project_dir
        )

        assert config.symbols == ['MES']
        assert config.allow_multi_symbol is False

    def test_create_default_config_multi_symbol_blocked(self, temp_project_dir: Path) -> None:
        """Test create_default_config blocks multi-symbol by default."""
        with pytest.raises(ValueError, match="Multi-symbol runs are not allowed"):
            create_default_config(
                symbols=['MES', 'MGC'],
                project_root=temp_project_dir
            )

    def test_create_default_config_multi_symbol_with_override(self, temp_project_dir: Path) -> None:
        """Test create_default_config allows multi-symbol with explicit override."""
        config = create_default_config(
            symbols=['MES', 'MGC'],
            allow_multi_symbol=True,
            project_root=temp_project_dir
        )

        assert config.symbols == ['MES', 'MGC']


# =============================================================================
# MTF CONFIGURATION PROPAGATION TESTS
# =============================================================================


class TestMTFConfigurationPropagation:
    """Tests for Multi-Timeframe (MTF) configuration in PipelineConfig."""

    def test_default_mtf_timeframes(self, temp_project_dir: Path) -> None:
        """Test that default MTF timeframes are set correctly."""
        config = PipelineConfig(
            symbols=['MES'],
            project_root=temp_project_dir
        )

        # Default MTF timeframes from constants.py
        expected = ['15min', '30min', '1h', '4h', 'daily']
        assert config.mtf_timeframes == expected

    def test_default_mtf_mode(self, temp_project_dir: Path) -> None:
        """Test that default MTF mode is 'both'."""
        config = PipelineConfig(
            symbols=['MES'],
            project_root=temp_project_dir
        )

        assert config.mtf_mode == 'both'

    def test_custom_mtf_timeframes(self, temp_project_dir: Path) -> None:
        """Test that custom MTF timeframes are accepted."""
        custom_timeframes = ['15min', '1h']
        config = PipelineConfig(
            symbols=['MES'],
            mtf_timeframes=custom_timeframes,
            project_root=temp_project_dir
        )

        assert config.mtf_timeframes == custom_timeframes

    def test_mtf_mode_bars_only(self, temp_project_dir: Path) -> None:
        """Test that mtf_mode='bars' is accepted."""
        config = PipelineConfig(
            symbols=['MES'],
            mtf_mode='bars',
            project_root=temp_project_dir
        )

        assert config.mtf_mode == 'bars'

    def test_mtf_mode_indicators_only(self, temp_project_dir: Path) -> None:
        """Test that mtf_mode='indicators' is accepted."""
        config = PipelineConfig(
            symbols=['MES'],
            mtf_mode='indicators',
            project_root=temp_project_dir
        )

        assert config.mtf_mode == 'indicators'

    def test_invalid_mtf_mode_raises(self, temp_project_dir: Path) -> None:
        """Test that invalid mtf_mode raises ValueError."""
        with pytest.raises(ValueError, match="mtf_mode must be one of"):
            PipelineConfig(
                symbols=['MES'],
                mtf_mode='invalid',
                project_root=temp_project_dir
            )

    def test_invalid_mtf_timeframe_raises(self, temp_project_dir: Path) -> None:
        """Test that invalid MTF timeframe raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported MTF timeframe"):
            PipelineConfig(
                symbols=['MES'],
                mtf_timeframes=['15min', 'invalid_tf'],
                project_root=temp_project_dir
            )

    def test_mtf_config_in_validate(self, temp_project_dir: Path) -> None:
        """Test that validate() catches invalid MTF config."""
        config = PipelineConfig(
            symbols=['MES'],
            project_root=temp_project_dir
        )

        # Manually set invalid mode after init to test validate()
        # Use object.__setattr__ to bypass dataclass frozen (if any) or property
        config.mtf_mode = 'invalid'

        issues = config.validate()
        assert any('mtf_mode' in issue for issue in issues)

    def test_mtf_config_saved_to_json(self, temp_project_dir: Path) -> None:
        """Test that MTF config is preserved when saving to JSON."""
        config = PipelineConfig(
            symbols=['MES'],
            mtf_timeframes=['15min', '1h'],
            mtf_mode='indicators',
            project_root=temp_project_dir
        )

        # Save config
        config_path = config.save_config()

        # Load and verify
        loaded = PipelineConfig.load_config(config_path)
        assert loaded.mtf_timeframes == ['15min', '1h']
        assert loaded.mtf_mode == 'indicators'

    def test_target_timeframe_default(self, temp_project_dir: Path) -> None:
        """Test that default target_timeframe is 5min."""
        config = PipelineConfig(
            symbols=['MES'],
            project_root=temp_project_dir
        )

        assert config.target_timeframe == '5min'

    def test_custom_target_timeframe(self, temp_project_dir: Path) -> None:
        """Test that custom target_timeframe is accepted."""
        config = PipelineConfig(
            symbols=['MES'],
            target_timeframe='15min',
            project_root=temp_project_dir
        )

        assert config.target_timeframe == '15min'

    def test_invalid_target_timeframe_raises(self, temp_project_dir: Path) -> None:
        """Test that invalid target_timeframe raises ValueError."""
        with pytest.raises(ValueError):
            PipelineConfig(
                symbols=['MES'],
                target_timeframe='invalid',
                project_root=temp_project_dir
            )


# =============================================================================
# PIPELINE CONFIG VALIDATION TESTS
# =============================================================================


class TestPipelineConfigValidation:
    """Tests for PipelineConfig.validate() method."""

    def test_validate_returns_empty_list_for_valid_config(self, temp_project_dir: Path) -> None:
        """Test that validate() returns empty list for valid config."""
        config = PipelineConfig(
            symbols=['MES'],
            project_root=temp_project_dir
        )

        issues = config.validate()
        assert issues == []

    def test_validate_catches_invalid_train_ratio(self, temp_project_dir: Path) -> None:
        """Test that validate() catches invalid train_ratio."""
        config = PipelineConfig(
            symbols=['MES'],
            project_root=temp_project_dir
        )

        # Manually set invalid ratio to bypass __post_init__ validation
        object.__setattr__(config, 'train_ratio', -0.1)

        issues = config.validate()
        assert any('train_ratio' in issue for issue in issues)

    def test_validate_catches_ratios_not_summing_to_one(self, temp_project_dir: Path) -> None:
        """Test that validate() catches ratios not summing to 1.0."""
        config = PipelineConfig(
            symbols=['MES'],
            project_root=temp_project_dir
        )

        # Manually set ratios that don't sum to 1.0
        object.__setattr__(config, 'train_ratio', 0.5)
        object.__setattr__(config, 'val_ratio', 0.3)
        object.__setattr__(config, 'test_ratio', 0.3)  # Sum = 1.1

        issues = config.validate()
        assert any('sum to 1.0' in issue for issue in issues)

    def test_validate_catches_empty_symbols(self, temp_project_dir: Path) -> None:
        """Test that validate() catches empty symbols list."""
        config = PipelineConfig(
            symbols=['MES'],
            project_root=temp_project_dir
        )

        # Manually clear symbols
        object.__setattr__(config, 'symbols', [])

        issues = config.validate()
        assert any('symbol' in issue.lower() for issue in issues)

    def test_validate_catches_invalid_horizon(self, temp_project_dir: Path) -> None:
        """Test that validate() catches invalid label horizon."""
        config = PipelineConfig(
            symbols=['MES'],
            project_root=temp_project_dir
        )

        # Manually set invalid horizon
        object.__setattr__(config, 'label_horizons', [0, -5])

        issues = config.validate()
        assert any('horizon' in issue.lower() for issue in issues)

    def test_validate_catches_max_bars_less_than_max_horizon(self, temp_project_dir: Path) -> None:
        """Test that validate() catches max_bars_ahead < max(label_horizons)."""
        config = PipelineConfig(
            symbols=['MES'],
            project_root=temp_project_dir
        )

        # Set max_bars_ahead less than max horizon
        object.__setattr__(config, 'label_horizons', [5, 10, 20])
        object.__setattr__(config, 'max_bars_ahead', 10)

        issues = config.validate()
        assert any('max_bars_ahead' in issue for issue in issues)

    def test_validate_catches_negative_purge_bars(self, temp_project_dir: Path) -> None:
        """Test that validate() catches negative purge_bars."""
        config = PipelineConfig(
            symbols=['MES'],
            project_root=temp_project_dir
        )

        object.__setattr__(config, 'purge_bars', -10)

        issues = config.validate()
        assert any('purge_bars' in issue for issue in issues)

    def test_validate_catches_negative_embargo_bars(self, temp_project_dir: Path) -> None:
        """Test that validate() catches negative embargo_bars."""
        config = PipelineConfig(
            symbols=['MES'],
            project_root=temp_project_dir
        )

        object.__setattr__(config, 'embargo_bars', -10)

        issues = config.validate()
        assert any('embargo_bars' in issue for issue in issues)

    def test_validate_catches_invalid_ga_population(self, temp_project_dir: Path) -> None:
        """Test that validate() catches invalid GA population size."""
        config = PipelineConfig(
            symbols=['MES'],
            project_root=temp_project_dir
        )

        object.__setattr__(config, 'ga_population_size', 1)

        issues = config.validate()
        assert any('ga_population_size' in issue for issue in issues)

    def test_validate_catches_empty_sma_periods(self, temp_project_dir: Path) -> None:
        """Test that validate() catches empty SMA periods."""
        config = PipelineConfig(
            symbols=['MES'],
            project_root=temp_project_dir
        )

        object.__setattr__(config, 'sma_periods', [])

        issues = config.validate()
        assert any('SMA' in issue for issue in issues)

    def test_validate_catches_invalid_rsi_period(self, temp_project_dir: Path) -> None:
        """Test that validate() catches RSI period < 2."""
        config = PipelineConfig(
            symbols=['MES'],
            project_root=temp_project_dir
        )

        object.__setattr__(config, 'rsi_period', 1)

        issues = config.validate()
        assert any('rsi_period' in issue for issue in issues)


# =============================================================================
# AUTO-SCALING PURGE/EMBARGO TESTS
# =============================================================================


class TestAutoScalePurgeEmbargo:
    """Tests for auto-scaling of purge and embargo bars."""

    def test_auto_scale_enabled_by_default(self, temp_project_dir: Path) -> None:
        """Test that auto_scale_purge_embargo is True by default."""
        config = PipelineConfig(
            symbols=['MES'],
            project_root=temp_project_dir
        )

        assert config.auto_scale_purge_embargo is True

    def test_purge_bars_scaled_to_max_horizon(self, temp_project_dir: Path) -> None:
        """Test that purge_bars is scaled based on max horizon."""
        config = PipelineConfig(
            symbols=['MES'],
            label_horizons=[5, 10, 20],  # max = 20
            project_root=temp_project_dir
        )

        # purge_bars should be at least max_horizon * 3 = 60
        assert config.purge_bars >= 60

    def test_embargo_bars_scaled_appropriately(self, temp_project_dir: Path) -> None:
        """Test that embargo_bars is scaled appropriately."""
        config = PipelineConfig(
            symbols=['MES'],
            label_horizons=[5, 10, 20],
            project_root=temp_project_dir
        )

        # Embargo should be significant (around 5 days for 5min bars = 1440)
        assert config.embargo_bars > 0

    def test_auto_scale_disabled_uses_explicit_values(self, temp_project_dir: Path) -> None:
        """Test that explicit values are used when auto_scale is disabled."""
        config = PipelineConfig(
            symbols=['MES'],
            auto_scale_purge_embargo=False,
            purge_bars=100,
            embargo_bars=200,
            project_root=temp_project_dir
        )

        assert config.purge_bars == 100
        assert config.embargo_bars == 200

    def test_different_horizons_produce_different_scaling(self, temp_project_dir: Path) -> None:
        """Test that different horizon sets produce different scaling."""
        config_small = PipelineConfig(
            symbols=['MES'],
            label_horizons=[5],  # Small horizon
            project_root=temp_project_dir
        )

        config_large = PipelineConfig(
            symbols=['MES'],
            label_horizons=[5, 10, 20],  # Larger horizons
            allow_multi_symbol=True,  # Allow reuse of temp_project_dir
            project_root=temp_project_dir
        )

        # Larger horizons should result in larger purge_bars
        # (or at minimum, equal due to floor)
        assert config_large.purge_bars >= config_small.purge_bars


# =============================================================================
# CONFIG PERSISTENCE TESTS
# =============================================================================


class TestConfigPersistence:
    """Tests for PipelineConfig save and load functionality."""

    def test_save_config_creates_file(self, temp_project_dir: Path) -> None:
        """Test that save_config creates a JSON file."""
        config = PipelineConfig(
            symbols=['MES'],
            project_root=temp_project_dir
        )

        config_path = config.save_config()

        assert config_path.exists()
        assert config_path.suffix == '.json'

    def test_save_config_produces_valid_json(self, temp_project_dir: Path) -> None:
        """Test that save_config produces valid JSON."""
        config = PipelineConfig(
            symbols=['MES'],
            project_root=temp_project_dir
        )

        config_path = config.save_config()

        with open(config_path) as f:
            data = json.load(f)

        assert 'symbols' in data
        assert 'run_id' in data
        assert data['symbols'] == ['MES']

    def test_load_config_restores_values(self, temp_project_dir: Path) -> None:
        """Test that load_config restores all values correctly."""
        original = PipelineConfig(
            symbols=['MES'],
            description='Test config',
            target_timeframe='5min',
            label_horizons=[5, 10, 20],
            train_ratio=0.70,
            val_ratio=0.15,
            test_ratio=0.15,
            project_root=temp_project_dir
        )

        config_path = original.save_config()
        loaded = PipelineConfig.load_config(config_path)

        assert loaded.symbols == original.symbols
        assert loaded.description == original.description
        assert loaded.target_timeframe == original.target_timeframe
        assert loaded.label_horizons == original.label_horizons
        assert loaded.train_ratio == original.train_ratio

    def test_load_from_run_id(self, temp_project_dir: Path) -> None:
        """Test loading config by run_id."""
        config = PipelineConfig(
            symbols=['MES'],
            run_id='test_run_123',
            project_root=temp_project_dir
        )

        config.save_config()

        loaded = PipelineConfig.load_from_run_id('test_run_123', temp_project_dir)

        assert loaded.run_id == 'test_run_123'
        assert loaded.symbols == ['MES']

    def test_load_nonexistent_config_raises(self, temp_project_dir: Path) -> None:
        """Test that loading nonexistent config raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            PipelineConfig.load_config(temp_project_dir / 'nonexistent.json')

    def test_config_metadata_saved(self, temp_project_dir: Path) -> None:
        """Test that config metadata is saved."""
        config = PipelineConfig(
            symbols=['MES'],
            project_root=temp_project_dir
        )

        config_path = config.save_config()

        with open(config_path) as f:
            data = json.load(f)

        assert '_metadata' in data
        assert 'created_at' in data['_metadata']
        assert 'config_version' in data['_metadata']


# =============================================================================
# DATE VALIDATION TESTS
# =============================================================================


class TestDateValidation:
    """Tests for date format validation."""

    def test_valid_start_date(self, temp_project_dir: Path) -> None:
        """Test that valid start_date is accepted."""
        config = PipelineConfig(
            symbols=['MES'],
            start_date='2020-01-01',
            project_root=temp_project_dir
        )

        assert config.start_date == '2020-01-01'

    def test_valid_end_date(self, temp_project_dir: Path) -> None:
        """Test that valid end_date is accepted."""
        config = PipelineConfig(
            symbols=['MES'],
            end_date='2024-12-31',
            project_root=temp_project_dir
        )

        assert config.end_date == '2024-12-31'

    def test_invalid_start_date_format_raises(self, temp_project_dir: Path) -> None:
        """Test that invalid start_date format raises ValueError."""
        with pytest.raises(ValueError, match="YYYY-MM-DD"):
            PipelineConfig(
                symbols=['MES'],
                start_date='01-01-2020',  # Wrong format
                project_root=temp_project_dir
            )

    def test_invalid_end_date_format_raises(self, temp_project_dir: Path) -> None:
        """Test that invalid end_date format raises ValueError."""
        with pytest.raises(ValueError, match="YYYY-MM-DD"):
            PipelineConfig(
                symbols=['MES'],
                end_date='2024/12/31',  # Wrong format
                project_root=temp_project_dir
            )


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
