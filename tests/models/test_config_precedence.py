"""
Tests for configuration precedence (TEST-001).

Tests the config precedence hierarchy:
1. CLI Arguments (Highest Priority)
2. Explicit Config File (if provided via --config)
3. Environment-Specific Overrides
4. Model-Specific YAML (auto-discovered)
5. Provided Defaults (Lowest Priority)

Also tests get_applied_overrides() tracking functionality.
"""

import pytest
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import patch, MagicMock

from src.models.config.merging import (
    build_config,
    get_applied_overrides,
    merge_configs,
    AppliedOverrides,
    create_trainer_config,
)
from src.models.config.environment import Environment


# =============================================================================
# MERGE_CONFIGS TESTS
# =============================================================================


class TestMergeConfigs:
    """Test basic config merging functionality."""

    def test_override_takes_precedence(self) -> None:
        """Override dict values should take precedence over base."""
        base = {"batch_size": 64, "learning_rate": 0.001}
        override = {"batch_size": 128}

        result = merge_configs(base, override)

        assert result["batch_size"] == 128
        assert result["learning_rate"] == 0.001

    def test_new_keys_added(self) -> None:
        """Override can add new keys not in base."""
        base = {"a": 1}
        override = {"b": 2}

        result = merge_configs(base, override)

        assert result["a"] == 1
        assert result["b"] == 2

    def test_deep_merge_nested_dicts(self) -> None:
        """Deep merge should recursively merge nested dictionaries."""
        base = {"outer": {"a": 1, "b": 2}}
        override = {"outer": {"b": 3, "c": 4}}

        result = merge_configs(base, override, deep=True)

        assert result["outer"]["a"] == 1
        assert result["outer"]["b"] == 3
        assert result["outer"]["c"] == 4

    def test_shallow_merge_replaces_nested_dicts(self) -> None:
        """Shallow merge should replace entire nested dict."""
        base = {"outer": {"a": 1, "b": 2}}
        override = {"outer": {"c": 3}}

        result = merge_configs(base, override, deep=False)

        assert result["outer"] == {"c": 3}
        assert "a" not in result["outer"]

    def test_none_values_preserved(self) -> None:
        """None values in override should be preserved (not filtered)."""
        base = {"a": 1, "b": 2}
        override = {"a": None}

        result = merge_configs(base, override)

        assert result["a"] is None
        assert result["b"] == 2


# =============================================================================
# CLI > YAML PRECEDENCE TESTS
# =============================================================================


class TestCLIPrecedence:
    """Test that CLI arguments override YAML values."""

    def test_cli_overrides_yaml_values(self) -> None:
        """CLI args should override model YAML defaults."""
        cli_args = {"n_estimators": 1000, "learning_rate": 0.05}

        config = build_config("xgboost", cli_args=cli_args)

        # CLI values should override YAML
        assert config["n_estimators"] == 1000
        assert config["learning_rate"] == 0.05

    def test_cli_none_values_ignored(self) -> None:
        """CLI args with None values should not override."""
        cli_args = {"n_estimators": None, "learning_rate": 0.05}

        config = build_config("xgboost", cli_args=cli_args)

        # n_estimators should come from YAML, not be None
        assert config["n_estimators"] != None  # noqa: E711
        assert config["learning_rate"] == 0.05

    def test_cli_adds_new_keys(self) -> None:
        """CLI args can add keys not in YAML."""
        cli_args = {"custom_param": "custom_value"}

        config = build_config("xgboost", cli_args=cli_args)

        assert config.get("custom_param") == "custom_value"

    def test_empty_cli_args_uses_yaml(self) -> None:
        """Empty CLI args should use YAML values."""
        config = build_config("xgboost", cli_args={})

        # Should have model name from YAML
        assert config.get("model_name") == "xgboost"


# =============================================================================
# EXPLICIT CONFIG FILE TESTS
# =============================================================================


class TestExplicitConfigFile:
    """Test that explicit config files override YAML defaults."""

    def test_explicit_config_overrides_model_yaml(self) -> None:
        """Explicit config file should override model YAML."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            f.write("n_estimators: 999\nlearning_rate: 0.123\n")
            f.flush()
            config_path = Path(f.name)

        try:
            config = build_config("xgboost", config_file=config_path)

            assert config["n_estimators"] == 999
            assert config["learning_rate"] == 0.123
        finally:
            config_path.unlink()

    def test_cli_overrides_explicit_config(self) -> None:
        """CLI args should override explicit config file."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            f.write("n_estimators: 999\nlearning_rate: 0.123\n")
            f.flush()
            config_path = Path(f.name)

        try:
            cli_args = {"n_estimators": 2000}
            config = build_config("xgboost", cli_args=cli_args, config_file=config_path)

            # CLI overrides explicit config
            assert config["n_estimators"] == 2000
            # Explicit config value preserved where CLI doesn't override
            assert config["learning_rate"] == 0.123
        finally:
            config_path.unlink()

    def test_invalid_explicit_config_raises_error(self) -> None:
        """Invalid explicit config file should raise ConfigError."""
        from src.models.config.exceptions import ConfigError

        with pytest.raises(ConfigError):
            build_config("xgboost", config_file="/nonexistent/path.yaml")


# =============================================================================
# ENVIRONMENT OVERRIDE TESTS
# =============================================================================


class TestEnvironmentOverrides:
    """Test environment-specific configuration overrides."""

    @patch("src.models.config.merging.get_environment_overrides")
    @patch("src.models.config.merging.detect_environment")
    def test_env_overrides_applied(
        self,
        mock_detect_env: MagicMock,
        mock_get_env_overrides: MagicMock,
    ) -> None:
        """Environment overrides should be applied to config."""
        mock_detect_env.return_value = Environment.LOCAL_GPU
        mock_get_env_overrides.return_value = {
            "training": {"batch_size": 512, "mixed_precision": True}
        }

        config = build_config("xgboost", apply_environment_overrides=True)

        # Environment overrides applied
        assert config.get("batch_size") == 512
        assert config.get("mixed_precision") is True

    @patch("src.models.config.merging.get_environment_overrides")
    @patch("src.models.config.merging.detect_environment")
    def test_cli_overrides_env_overrides(
        self,
        mock_detect_env: MagicMock,
        mock_get_env_overrides: MagicMock,
    ) -> None:
        """CLI args should override environment overrides."""
        mock_detect_env.return_value = Environment.LOCAL_GPU
        mock_get_env_overrides.return_value = {
            "training": {"batch_size": 512}
        }

        cli_args = {"batch_size": 128}
        config = build_config("xgboost", cli_args=cli_args)

        # CLI takes precedence over environment
        assert config["batch_size"] == 128

    @patch("src.models.config.merging.get_environment_overrides")
    def test_env_overrides_disabled(
        self,
        mock_get_env_overrides: MagicMock,
    ) -> None:
        """Environment overrides can be disabled."""
        mock_get_env_overrides.return_value = {
            "training": {"batch_size": 512}
        }

        config = build_config("xgboost", apply_environment_overrides=False)

        # Environment overrides NOT applied when disabled
        # batch_size should come from model YAML or defaults
        mock_get_env_overrides.assert_not_called()


# =============================================================================
# DEFAULTS TESTS
# =============================================================================


class TestDefaults:
    """Test that provided defaults are lowest priority."""

    def test_defaults_used_when_no_other_source(self) -> None:
        """Defaults should be used when no other source provides value."""
        defaults = {"custom_default": "default_value"}

        config = build_config("xgboost", defaults=defaults)

        assert config.get("custom_default") == "default_value"

    def test_yaml_overrides_defaults(self) -> None:
        """Model YAML should override defaults."""
        # xgboost.yaml defines model_family = "boosting"
        defaults = {"model_family": "should_be_overridden"}

        config = build_config("xgboost", defaults=defaults)

        assert config["model_family"] == "boosting"

    def test_cli_overrides_defaults(self) -> None:
        """CLI args should override defaults."""
        defaults = {"custom_param": "default_value"}
        cli_args = {"custom_param": "cli_value"}

        config = build_config("xgboost", cli_args=cli_args, defaults=defaults)

        assert config["custom_param"] == "cli_value"


# =============================================================================
# FULL PRECEDENCE CHAIN TESTS
# =============================================================================


class TestFullPrecedenceChain:
    """Test the complete precedence hierarchy."""

    @patch("src.models.config.merging.get_environment_overrides")
    @patch("src.models.config.merging.detect_environment")
    def test_complete_precedence_hierarchy(
        self,
        mock_detect_env: MagicMock,
        mock_get_env_overrides: MagicMock,
    ) -> None:
        """
        Test complete precedence: CLI > explicit > env > YAML > defaults.

        Each layer should only override lower-priority values.
        """
        mock_detect_env.return_value = Environment.LOCAL_CPU
        mock_get_env_overrides.return_value = {
            "training": {"param_env": "from_env", "param_yaml_env": "from_env"}
        }

        defaults = {
            "param_default": "from_default",
            "param_yaml_default": "from_default",
            "param_env_default": "from_default",
            "param_cli_default": "from_default",
        }

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            f.write("param_explicit: from_explicit\nparam_cli_explicit: from_explicit\n")
            f.flush()
            config_path = Path(f.name)

        try:
            cli_args = {
                "param_cli": "from_cli",
                "param_cli_default": "from_cli",
                "param_cli_explicit": "from_cli",
            }

            config = build_config(
                "xgboost",
                cli_args=cli_args,
                config_file=config_path,
                defaults=defaults,
                apply_environment_overrides=True,
            )

            # CLI takes precedence
            assert config["param_cli"] == "from_cli"
            assert config["param_cli_default"] == "from_cli"
            assert config["param_cli_explicit"] == "from_cli"

            # Explicit config preserved where CLI doesn't override
            assert config["param_explicit"] == "from_explicit"

            # Environment preserved where explicit/CLI don't override
            assert config["param_env"] == "from_env"

            # Defaults preserved where nothing else overrides
            assert config["param_default"] == "from_default"

        finally:
            config_path.unlink()


# =============================================================================
# APPLIED OVERRIDES TRACKING TESTS
# =============================================================================


class TestAppliedOverrides:
    """Test get_applied_overrides() tracking functionality."""

    def test_applied_overrides_dataclass_to_dict(self) -> None:
        """AppliedOverrides should serialize to dict correctly."""
        applied = AppliedOverrides(
            environment="local_gpu",
            environment_overrides={"batch_size": 512},
            model_yaml_path="/path/to/model.yaml",
            model_yaml_keys=["n_estimators", "learning_rate"],
            explicit_config_path="/path/to/config.yaml",
            explicit_config_keys=["custom_param"],
            cli_overrides={"batch_size": 128},
        )

        result = applied.to_dict()

        assert result["environment"] == "local_gpu"
        assert result["environment_overrides"] == {"batch_size": 512}
        assert result["model_yaml_path"] == "/path/to/model.yaml"
        assert result["model_yaml_keys"] == ["n_estimators", "learning_rate"]
        assert result["explicit_config_path"] == "/path/to/config.yaml"
        assert result["explicit_config_keys"] == ["custom_param"]
        assert result["cli_overrides"] == {"batch_size": 128}

    def test_get_applied_overrides_tracks_cli(self) -> None:
        """get_applied_overrides() should track CLI overrides."""
        cli_args = {"n_estimators": 1000}
        build_config("xgboost", cli_args=cli_args)

        overrides = get_applied_overrides()

        assert "n_estimators" in overrides["cli_overrides"]
        assert overrides["cli_overrides"]["n_estimators"] == 1000

    def test_get_applied_overrides_tracks_environment(self) -> None:
        """get_applied_overrides() should track detected environment."""
        build_config("xgboost")

        overrides = get_applied_overrides()

        assert "environment" in overrides
        assert overrides["environment"] in ["colab", "local_gpu", "local_cpu"]

    def test_get_applied_overrides_tracks_model_yaml(self) -> None:
        """get_applied_overrides() should track model YAML source."""
        build_config("xgboost")

        overrides = get_applied_overrides()

        # xgboost.yaml should be loaded
        assert overrides.get("model_yaml_path") is not None
        assert "xgboost" in overrides["model_yaml_path"]
        assert len(overrides.get("model_yaml_keys", [])) > 0

    def test_get_applied_overrides_tracks_explicit_config(self) -> None:
        """get_applied_overrides() should track explicit config file."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            f.write("custom_param: custom_value\n")
            f.flush()
            config_path = Path(f.name)

        try:
            build_config("xgboost", config_file=config_path)

            overrides = get_applied_overrides()

            assert overrides.get("explicit_config_path") is not None
            assert "custom_param" in overrides.get("explicit_config_keys", [])
        finally:
            config_path.unlink()

    def test_get_applied_overrides_empty_before_build(self) -> None:
        """get_applied_overrides() should return empty dict before build_config()."""
        # Reset module state by importing fresh
        from src.models.config import merging
        merging._last_applied_overrides = None

        result = get_applied_overrides()
        assert result == {}


# =============================================================================
# CREATE_TRAINER_CONFIG INTEGRATION TESTS
# =============================================================================


class TestCreateTrainerConfigPrecedence:
    """Test precedence in create_trainer_config()."""

    def test_trainer_config_cli_override(self) -> None:
        """TrainerConfig should respect CLI overrides."""
        cli_args = {"batch_size": 256, "max_epochs": 50}

        trainer = create_trainer_config("xgboost", horizon=10, cli_args=cli_args)

        assert trainer.batch_size == 256
        assert trainer.max_epochs == 50

    def test_trainer_config_with_explicit_file(self) -> None:
        """TrainerConfig should respect explicit config file."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            f.write("batch_size: 512\nmax_epochs: 100\n")
            f.flush()
            config_path = Path(f.name)

        try:
            trainer = create_trainer_config(
                "xgboost", horizon=10, config_file=config_path
            )

            assert trainer.batch_size == 512
            assert trainer.max_epochs == 100
        finally:
            config_path.unlink()

    def test_trainer_config_cli_overrides_file(self) -> None:
        """CLI should override explicit config file in TrainerConfig."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            f.write("batch_size: 512\nmax_epochs: 100\n")
            f.flush()
            config_path = Path(f.name)

        try:
            cli_args = {"batch_size": 128}
            trainer = create_trainer_config(
                "xgboost",
                horizon=10,
                cli_args=cli_args,
                config_file=config_path,
            )

            # CLI overrides file
            assert trainer.batch_size == 128
            # File value preserved where CLI doesn't override
            assert trainer.max_epochs == 100
        finally:
            config_path.unlink()
