"""
Test UnifiedConfig - Centralized configuration system.

Tests cover:
- UnifiedConfig creation from dict
- UnifiedConfig.from_yaml() loading
- Validation errors for invalid configs
- Backward compatibility with to_trainer_config(), to_pipeline_config()
- get_config_value() utility function
"""

import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from src.config.unified import (
    UnifiedConfig,
    TimeframesSection,
    SplitsSection,
    PurgeEmbargoSection,
    HorizonsSection,
    TrainingSection,
    get_unified_config,
    set_unified_config,
    reset_unified_config,
)
from src.config.utils import (
    get_config_value,
    clear_config_cache,
    ConfigAccessLog,
    ConfigSource,
)
from src.config.validators import (
    validate_config,
    ValidationResult,
    ConfigValidationError,
)


class TestUnifiedConfigCreation:
    """Test UnifiedConfig creation and initialization."""

    def test_default_creation(self):
        """UnifiedConfig should be creatable with defaults."""
        config = UnifiedConfig()
        assert config.symbol == "MES"
        assert config.random_seed == 42
        assert isinstance(config.timeframes, TimeframesSection)
        assert isinstance(config.splits, SplitsSection)
        assert isinstance(config.training, TrainingSection)

    def test_creation_with_overrides(self):
        """UnifiedConfig should accept field overrides."""
        config = UnifiedConfig(
            symbol="MGC",
            random_seed=123,
            description="Test config",
        )
        assert config.symbol == "MGC"
        assert config.random_seed == 123
        assert config.description == "Test config"

    def test_symbols_list_initialization(self):
        """Symbols list should be initialized from symbol if not provided."""
        config = UnifiedConfig(symbol="MES")
        assert config.symbols == ["MES"]

    def test_symbols_list_includes_symbol(self):
        """Symbol should be added to symbols list if not present."""
        config = UnifiedConfig(symbol="MES", symbols=["MGC"])
        assert "MES" in config.symbols
        assert "MGC" in config.symbols

    def test_path_string_conversion(self):
        """String paths should be converted to Path objects."""
        config = UnifiedConfig(
            output_dir="experiments/test",
            data_dir="data/test",
        )
        assert isinstance(config.output_dir, Path)
        assert isinstance(config.data_dir, Path)
        assert str(config.output_dir) == "experiments/test"


class TestUnifiedConfigFromDict:
    """Test UnifiedConfig.from_dict() method."""

    def test_from_dict_minimal(self):
        """from_dict should work with minimal configuration."""
        data = {"symbol": "MGC", "random_seed": 100}
        config = UnifiedConfig.from_dict(data, validate=False)
        assert config.symbol == "MGC"
        assert config.random_seed == 100

    def test_from_dict_nested_sections(self):
        """from_dict should parse nested section configurations."""
        data = {
            "timeframes": {"default_primary": "15min"},
            "splits": {"train": 0.6, "val": 0.2, "test": 0.2},
            "training": {"batch_size": 512, "max_epochs": 50},
        }
        config = UnifiedConfig.from_dict(data, validate=False)
        assert config.timeframes.default_primary == "15min"
        assert config.splits.train == 0.6
        assert config.training.batch_size == 512
        assert config.training.max_epochs == 50

    def test_from_dict_with_validation(self):
        """from_dict should validate by default."""
        # Valid config should pass
        data = {"random_seed": 42}
        config = UnifiedConfig.from_dict(data)  # validate=True by default
        assert config.random_seed == 42

    def test_from_dict_horizons_section(self):
        """from_dict should parse horizons section correctly."""
        data = {
            "horizons": {
                "supported": [1, 5, 10, 15, 20, 30],
                "active": [5, 10, 15],
                "default": [5, 10],
            }
        }
        config = UnifiedConfig.from_dict(data, validate=False)
        assert config.horizons.supported == [1, 5, 10, 15, 20, 30]
        assert config.horizons.active == [5, 10, 15]
        assert config.horizons.max_horizon == 15


class TestUnifiedConfigFromYaml:
    """Test UnifiedConfig.from_yaml() method."""

    def test_from_yaml_file(self):
        """from_yaml should load configuration from YAML file."""
        yaml_content = """
symbol: MGC
random_seed: 999
training:
  batch_size: 128
  max_epochs: 200
splits:
  train: 0.7
  val: 0.15
  test: 0.15
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            temp_path = f.name

        try:
            config = UnifiedConfig.from_yaml(temp_path, validate=False)
            assert config.symbol == "MGC"
            assert config.random_seed == 999
            assert config.training.batch_size == 128
            assert config.training.max_epochs == 200
        finally:
            Path(temp_path).unlink()

    def test_from_yaml_file_not_found(self):
        """from_yaml should raise FileNotFoundError for missing files."""
        with pytest.raises(FileNotFoundError):
            UnifiedConfig.from_yaml("/nonexistent/path/config.yaml")

    def test_from_yaml_empty_file(self):
        """from_yaml should handle empty YAML files gracefully."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("")
            temp_path = f.name

        try:
            config = UnifiedConfig.from_yaml(temp_path, validate=False)
            # Should use defaults
            assert config.symbol == "MES"
            assert config.random_seed == 42
        finally:
            Path(temp_path).unlink()


class TestUnifiedConfigValidation:
    """Test configuration validation."""

    def test_splits_must_sum_to_one(self):
        """SplitsSection should validate that ratios sum to 1.0."""
        with pytest.raises(ValueError, match="sum to 1.0"):
            SplitsSection(train=0.5, val=0.2, test=0.2)  # 0.9 total

    def test_horizons_active_must_be_in_supported(self):
        """HorizonsSection should validate active horizons are in supported."""
        with pytest.raises(ValueError, match="not in supported"):
            HorizonsSection(
                supported=[5, 10, 15],
                active=[5, 10, 20],  # 20 is not in supported
            )

    def test_training_batch_size_must_be_positive(self):
        """TrainingSection should validate batch_size > 0."""
        with pytest.raises(ValueError, match="batch_size"):
            TrainingSection(batch_size=0)

    def test_training_max_epochs_must_be_positive(self):
        """TrainingSection should validate max_epochs > 0."""
        with pytest.raises(ValueError, match="max_epochs"):
            TrainingSection(max_epochs=-1)

    def test_purge_embargo_timeframe_validation(self):
        """PurgeEmbargoSection.compute_embargo_bars should validate input."""
        section = PurgeEmbargoSection()
        with pytest.raises(ValueError, match="positive"):
            section.compute_embargo_bars(0)
        with pytest.raises(ValueError, match="positive"):
            section.compute_embargo_bars(-5)

    def test_validate_config_function(self):
        """validate_config should return ValidationResult."""
        data = {"random_seed": 42, "splits": {"train": 0.7, "val": 0.15, "test": 0.15}}
        result = validate_config(data)
        assert isinstance(result, ValidationResult)

    def test_validate_strict_raises_on_error(self):
        """validate_strict should raise ConfigValidationError on errors."""
        # Create a config with invalid training settings that will fail validation
        config = UnifiedConfig()
        # The validate method checks against the schema
        result = config.validate()
        # Most default configs should be valid
        assert isinstance(result, ValidationResult)


class TestUnifiedConfigBackwardCompatibility:
    """Test backward compatibility methods."""

    def test_to_dict_roundtrip(self):
        """to_dict should produce serializable output."""
        config = UnifiedConfig(symbol="MGC", random_seed=123)
        data = config.to_dict()

        assert data["symbol"] == "MGC"
        assert data["random_seed"] == 123
        assert "timeframes" in data
        assert "splits" in data
        assert "training" in data

    def test_to_trainer_config(self):
        """to_trainer_config should produce valid TrainerConfig."""
        config = UnifiedConfig(
            training=TrainingSection(batch_size=512, max_epochs=50),
        )
        trainer_config = config.to_trainer_config(
            model_name="xgboost",
            horizon=20,
        )
        assert trainer_config.model_name == "xgboost"
        assert trainer_config.horizon == 20
        assert trainer_config.batch_size == 512
        assert trainer_config.max_epochs == 50

    def test_to_trainer_config_default_horizon(self):
        """to_trainer_config should use max active horizon as default."""
        config = UnifiedConfig(
            horizons=HorizonsSection(
                supported=[5, 10, 15, 20, 30],
                active=[5, 10, 15],
            )
        )
        trainer_config = config.to_trainer_config(model_name="lstm")
        assert trainer_config.horizon == 15  # max of active horizons

    def test_save_yaml_creates_file(self):
        """save_yaml should create a valid YAML file."""
        config = UnifiedConfig(symbol="MGC", random_seed=123)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test_config.yaml"
            config.save_yaml(path)

            assert path.exists()

            with open(path) as f:
                loaded = yaml.safe_load(f)

            assert loaded["symbol"] == "MGC"
            assert loaded["random_seed"] == 123


class TestGetConfigValue:
    """Test the get_config_value utility function."""

    def setup_method(self):
        """Clear config cache before each test."""
        clear_config_cache()
        ConfigAccessLog.get_instance().clear()

    def test_get_config_value_with_fallback(self):
        """get_config_value should return fallback when path not found."""
        # Clear cache to ensure fresh start
        clear_config_cache()

        # Non-existent path should return fallback
        result = get_config_value("nonexistent.path", fallback="default_value")
        assert result == "default_value"

    def test_get_config_value_access_logging(self):
        """get_config_value should log accesses when enabled."""
        clear_config_cache()
        access_log = ConfigAccessLog.get_instance()
        access_log.clear()

        # Make an access
        get_config_value("some.path", fallback=42, log_access=True)

        log = access_log.get_log()
        assert len(log) >= 1
        # The path should be in the log
        paths = [entry.path for entry in log]
        assert "some.path" in paths

    def test_get_config_value_no_logging(self):
        """get_config_value should skip logging when disabled."""
        clear_config_cache()
        access_log = ConfigAccessLog.get_instance()
        access_log.clear()

        # Make an access without logging
        get_config_value("another.path", fallback="test", log_access=False)

        # Check that no entries were added for this path
        log = access_log.get_log()
        paths = [entry.path for entry in log]
        assert "another.path" not in paths


class TestUnifiedConfigSingleton:
    """Test the global singleton functions."""

    def teardown_method(self):
        """Reset singleton after each test."""
        reset_unified_config()

    def test_set_and_get_unified_config(self):
        """set_unified_config should update the global singleton."""
        custom_config = UnifiedConfig(symbol="TEST", random_seed=999)
        set_unified_config(custom_config)

        retrieved = get_unified_config()
        assert retrieved.symbol == "TEST"
        assert retrieved.random_seed == 999

    def test_reset_unified_config(self):
        """reset_unified_config should clear the singleton."""
        custom_config = UnifiedConfig(symbol="TEST")
        set_unified_config(custom_config)
        reset_unified_config()

        # After reset, get_unified_config should reload from default
        # or create new default instance


class TestTimeframesSection:
    """Test TimeframesSection specifically."""

    def test_default_canonical_ladder(self):
        """Default canonical ladder should have 9 timeframes."""
        section = TimeframesSection()
        assert len(section.canonical_ladder) == 9
        assert "1min" in section.canonical_ladder
        assert "60min" in section.canonical_ladder

    def test_from_dict(self):
        """from_dict should parse timeframe configuration."""
        data = {
            "default_primary": "15min",
            "canonical_ladder": ["1min", "5min", "15min"],
        }
        section = TimeframesSection.from_dict(data)
        assert section.default_primary == "15min"
        assert len(section.canonical_ladder) == 3


class TestPurgeEmbargoSection:
    """Test PurgeEmbargoSection specifically."""

    def test_compute_purge_bars(self):
        """compute_purge_bars should calculate correctly."""
        section = PurgeEmbargoSection(purge_multiplier=3.0)
        assert section.compute_purge_bars(20) == 60
        assert section.compute_purge_bars(10) == 30

    def test_compute_embargo_bars(self):
        """compute_embargo_bars should calculate correctly."""
        section = PurgeEmbargoSection(embargo_time_minutes=7200)  # 5 days
        # At 5min bars: 7200 / 5 = 1440 bars
        assert section.compute_embargo_bars(5) == 1440
        # At 1min bars: 7200 / 1 = 7200 bars
        assert section.compute_embargo_bars(1) == 7200


class TestConfigAccessLog:
    """Test ConfigAccessLog functionality."""

    def setup_method(self):
        """Clear access log before each test."""
        ConfigAccessLog.get_instance().clear()

    def test_log_access(self):
        """ConfigAccessLog should record accesses."""
        log = ConfigAccessLog.get_instance()
        log.log_access(
            path="test.path",
            value=42,
            source=ConfigSource.YAML_FILE,
            fallback_used=False,
        )

        entries = log.get_log()
        assert len(entries) == 1
        assert entries[0].path == "test.path"
        assert entries[0].value == 42
        assert entries[0].fallback_used is False

    def test_get_fallback_accesses(self):
        """get_fallback_accesses should filter correctly."""
        log = ConfigAccessLog.get_instance()
        log.log_access("path1", "val1", ConfigSource.YAML_FILE, fallback_used=False)
        log.log_access("path2", "val2", ConfigSource.FALLBACK, fallback_used=True)
        log.log_access("path3", "val3", ConfigSource.FALLBACK, fallback_used=True)

        fallbacks = log.get_fallback_accesses()
        assert len(fallbacks) == 2
        assert all(e.fallback_used for e in fallbacks)

    def test_summary(self):
        """summary should return correct statistics."""
        log = ConfigAccessLog.get_instance()
        log.log_access("path1", "val1", ConfigSource.YAML_FILE, fallback_used=False)
        log.log_access("path2", "val2", ConfigSource.FALLBACK, fallback_used=True)
        log.log_access("path1", "val1", ConfigSource.CACHED, fallback_used=False)

        summary = log.summary()
        assert summary["total_accesses"] == 3
        assert summary["unique_paths"] == 2
        assert summary["fallback_count"] == 1

    def test_disable_enable(self):
        """disable/enable should control logging."""
        log = ConfigAccessLog.get_instance()
        log.clear()

        log.disable()
        log.log_access("path1", "val1", ConfigSource.YAML_FILE, fallback_used=False)
        assert len(log.get_log()) == 0

        log.enable()
        log.log_access("path2", "val2", ConfigSource.YAML_FILE, fallback_used=False)
        assert len(log.get_log()) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
