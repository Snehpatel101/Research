"""
Tests for pipeline utility functions.

Tests:
- capture_config_snapshot() for safe config serialization
- Path sanitization and sensitive key exclusion
- StageResult and StageStatus classes
"""

import pytest
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from src.pipeline.utils import (
    capture_config_snapshot,
    StageResult,
    StageStatus,
    create_stage_result,
    create_failed_result,
    CONFIG_SNAPSHOT_EXCLUDE_KEYS,
    CONFIG_SNAPSHOT_PATH_KEYS,
    _serialize_config_value,
)


# =============================================================================
# TEST FIXTURES
# =============================================================================


@dataclass
class MockConfig:
    """Mock config object for testing."""

    model_name: str = "xgboost"
    batch_size: int = 64
    learning_rate: float = 0.01
    random_seed: int = 42
    # Path fields
    project_root: str = "/Users/test/research"
    data_dir: str = "/Users/test/research/data"
    output_dir: str = "/Users/test/research/output"
    # Private attributes (should be excluded)
    _internal_state: str = "private"


@dataclass
class MockConfigWithSecrets:
    """Mock config with sensitive fields."""

    model_name: str = "test"
    api_key: str = "sk-secret-key-123"
    password: str = "supersecret"
    database_secret: str = "db-secret"
    auth_token: str = "auth-123"
    normal_field: str = "normal"


@pytest.fixture
def basic_config() -> MockConfig:
    """Basic config without secrets."""
    return MockConfig()


@pytest.fixture
def config_with_secrets() -> MockConfigWithSecrets:
    """Config with sensitive fields."""
    return MockConfigWithSecrets()


@pytest.fixture
def dict_config() -> dict:
    """Dictionary-based config."""
    return {
        "model_name": "lstm",
        "batch_size": 128,
        "learning_rate": 0.001,
        "project_root": "/absolute/path/to/project",
        "model_path": "/path/to/model.pt",
        "password": "secret123",
    }


# =============================================================================
# CAPTURE_CONFIG_SNAPSHOT TESTS
# =============================================================================


class TestCaptureConfigSnapshot:
    """Test capture_config_snapshot() function."""

    def test_basic_fields_captured(self, basic_config: MockConfig) -> None:
        """Basic config fields should be captured."""
        snapshot = capture_config_snapshot(basic_config)

        assert snapshot["model_name"] == "xgboost"
        assert snapshot["batch_size"] == 64
        assert snapshot["learning_rate"] == 0.01
        assert snapshot["random_seed"] == 42

    def test_private_attributes_excluded(self, basic_config: MockConfig) -> None:
        """Private attributes (starting with _) should be excluded."""
        snapshot = capture_config_snapshot(basic_config)

        assert "_internal_state" not in snapshot

    def test_path_keys_excluded_by_default(self, basic_config: MockConfig) -> None:
        """Path keys should be excluded when include_paths=False (default)."""
        snapshot = capture_config_snapshot(basic_config, include_paths=False)

        assert "project_root" not in snapshot
        assert "data_dir" not in snapshot
        assert "output_dir" not in snapshot

    def test_path_keys_sanitized_when_included(
        self, basic_config: MockConfig
    ) -> None:
        """Path keys should be basename only when include_paths=True."""
        snapshot = capture_config_snapshot(basic_config, include_paths=True)

        # Should contain basename only, not full path
        assert snapshot.get("project_root") == "research"
        assert snapshot.get("data_dir") == "data"
        assert snapshot.get("output_dir") == "output"

    def test_secrets_excluded(self, config_with_secrets: MockConfigWithSecrets) -> None:
        """Sensitive keys should be excluded."""
        snapshot = capture_config_snapshot(config_with_secrets)

        # Secrets should be excluded
        assert "api_key" not in snapshot
        assert "password" not in snapshot
        assert "database_secret" not in snapshot
        assert "auth_token" not in snapshot

        # Normal field should be included
        assert snapshot["model_name"] == "test"
        assert snapshot["normal_field"] == "normal"

    def test_dict_config_supported(self, dict_config: dict) -> None:
        """Dictionary configs should be supported."""
        snapshot = capture_config_snapshot(dict_config)

        assert snapshot["model_name"] == "lstm"
        assert snapshot["batch_size"] == 128
        # password excluded
        assert "password" not in snapshot
        # path keys excluded by default
        assert "project_root" not in snapshot

    def test_nested_dicts_serialized(self) -> None:
        """Nested dictionaries should be serialized recursively."""
        config = {
            "model_name": "test",
            "training": {
                "epochs": 100,
                "batch_size": 64,
            },
            "optimizer": {
                "name": "adam",
                "params": {
                    "lr": 0.001,
                    "weight_decay": 0.0001,
                },
            },
        }

        snapshot = capture_config_snapshot(config)

        assert snapshot["model_name"] == "test"
        assert snapshot["training"]["epochs"] == 100
        assert snapshot["optimizer"]["params"]["lr"] == 0.001

    def test_lists_serialized(self) -> None:
        """Lists should be serialized correctly."""
        config = {
            "model_name": "test",
            "horizons": [5, 10, 15, 20],
            "base_models": ["xgboost", "lightgbm", "catboost"],
        }

        snapshot = capture_config_snapshot(config)

        assert snapshot["horizons"] == [5, 10, 15, 20]
        assert snapshot["base_models"] == ["xgboost", "lightgbm", "catboost"]

    def test_path_objects_serialized(self) -> None:
        """Path objects should be serialized to basename."""
        config = {
            "model_name": "test",
            "model_file": Path("/path/to/model.pt"),
            "config_file": Path("/some/config.yaml"),
        }

        snapshot = capture_config_snapshot(config)

        # Paths serialized to basename
        assert snapshot["model_file"] == "model.pt"
        assert snapshot["config_file"] == "config.yaml"

    def test_object_with_to_dict(self) -> None:
        """Objects with to_dict() method should be serialized."""

        class ConfigWithToDict:
            def __init__(self):
                self.field1 = "value1"
                self.field2 = 123

            def to_dict(self):
                return {"field1": self.field1, "field2": self.field2}

        config = ConfigWithToDict()
        snapshot = capture_config_snapshot(config)

        assert snapshot["field1"] == "value1"
        assert snapshot["field2"] == 123

    def test_unknown_type_returns_empty(self) -> None:
        """Unknown config types should return empty dict with warning."""
        snapshot = capture_config_snapshot(12345)  # Not a config-like object
        assert snapshot == {}

    def test_error_handling(self) -> None:
        """Errors during serialization should be caught."""

        class BadConfig:
            @property
            def field(self):
                raise RuntimeError("Cannot access")

        # Should not crash, should return error info
        snapshot = capture_config_snapshot(BadConfig())
        # Either empty or contains error
        assert "error" in snapshot or snapshot == {}


# =============================================================================
# SENSITIVE KEY EXCLUSION TESTS
# =============================================================================


class TestSensitiveKeyExclusion:
    """Test that sensitive keys are properly excluded."""

    def test_exclude_keys_constant_complete(self) -> None:
        """CONFIG_SNAPSHOT_EXCLUDE_KEYS should contain expected keys."""
        assert "password" in CONFIG_SNAPSHOT_EXCLUDE_KEYS
        assert "secret" in CONFIG_SNAPSHOT_EXCLUDE_KEYS
        assert "token" in CONFIG_SNAPSHOT_EXCLUDE_KEYS
        assert "api_key" in CONFIG_SNAPSHOT_EXCLUDE_KEYS
        assert "credential" in CONFIG_SNAPSHOT_EXCLUDE_KEYS
        assert "auth" in CONFIG_SNAPSHOT_EXCLUDE_KEYS

    def test_partial_match_excluded(self) -> None:
        """Keys containing sensitive substrings should be excluded."""
        config = {
            "model_name": "test",
            "database_password": "secret",
            "aws_secret_key": "AKIA...",
            "auth_header": "Bearer ...",
            "api_key_v2": "key...",
            "credential_path": "/path/to/creds",
        }

        snapshot = capture_config_snapshot(config)

        assert "database_password" not in snapshot
        assert "aws_secret_key" not in snapshot
        assert "auth_header" not in snapshot
        assert "api_key_v2" not in snapshot
        assert "credential_path" not in snapshot
        assert "model_name" in snapshot

    def test_case_insensitive_exclusion(self) -> None:
        """Sensitive key detection should be case-insensitive."""
        config = {
            "model_name": "test",
            "PASSWORD": "secret",
            "API_KEY": "key",
            "Secret": "value",
        }

        snapshot = capture_config_snapshot(config)

        assert "PASSWORD" not in snapshot
        assert "API_KEY" not in snapshot
        assert "Secret" not in snapshot
        assert "model_name" in snapshot


# =============================================================================
# PATH SANITIZATION TESTS
# =============================================================================


class TestPathSanitization:
    """Test path key handling and sanitization."""

    def test_path_keys_constant_complete(self) -> None:
        """CONFIG_SNAPSHOT_PATH_KEYS should contain expected path keys."""
        expected_keys = {
            "project_root",
            "data_dir",
            "raw_data_dir",
            "clean_data_dir",
            "features_dir",
            "splits_dir",
            "output_dir",
            "model_path",
            "checkpoint_dir",
        }
        assert CONFIG_SNAPSHOT_PATH_KEYS == expected_keys

    def test_path_suffix_excluded(self) -> None:
        """Keys ending with _path should be excluded by default."""
        config = {
            "model_name": "test",
            "config_path": "/path/to/config.yaml",
            "checkpoint_path": "/path/to/checkpoint",
            "log_path": "/path/to/logs",
        }

        snapshot = capture_config_snapshot(config, include_paths=False)

        assert "config_path" not in snapshot
        assert "checkpoint_path" not in snapshot
        assert "log_path" not in snapshot

    def test_path_none_handled(self) -> None:
        """None path values should be handled gracefully."""
        config = {
            "model_name": "test",
            "project_root": None,
            "output_dir": None,
        }

        snapshot = capture_config_snapshot(config, include_paths=True)
        # Should not crash, None values handled


# =============================================================================
# SERIALIZE_CONFIG_VALUE TESTS
# =============================================================================


class TestSerializeConfigValue:
    """Test _serialize_config_value() helper function."""

    def test_primitives_unchanged(self) -> None:
        """Primitive types should pass through unchanged."""
        assert _serialize_config_value(None) is None
        assert _serialize_config_value("string") == "string"
        assert _serialize_config_value(42) == 42
        assert _serialize_config_value(3.14) == 3.14
        assert _serialize_config_value(True) is True
        assert _serialize_config_value(False) is False

    def test_list_recursive(self) -> None:
        """Lists should be serialized recursively."""
        result = _serialize_config_value([1, "two", Path("/three")])
        assert result == [1, "two", "three"]

    def test_tuple_to_list(self) -> None:
        """Tuples should be converted to lists."""
        result = _serialize_config_value((1, 2, 3))
        assert result == [1, 2, 3]

    def test_dict_recursive(self) -> None:
        """Dicts should be serialized recursively."""
        result = _serialize_config_value({
            "a": 1,
            "b": Path("/path/to/file"),
            "c": {"nested": True},
        })
        assert result == {
            "a": 1,
            "b": "file",
            "c": {"nested": True},
        }

    def test_path_to_basename(self) -> None:
        """Path objects should serialize to basename."""
        result = _serialize_config_value(Path("/absolute/path/to/file.txt"))
        assert result == "file.txt"

    def test_object_to_dict(self) -> None:
        """Objects with __dict__ should serialize to dict."""

        class SimpleObject:
            def __init__(self):
                self.field1 = "value1"
                self.field2 = 42
                self._private = "hidden"

        result = _serialize_config_value(SimpleObject())
        assert result["field1"] == "value1"
        assert result["field2"] == 42
        assert "_private" not in result

    def test_fallback_to_string(self) -> None:
        """Unknown types should fall back to string."""
        import datetime

        dt = datetime.datetime(2024, 1, 1, 12, 0, 0)
        result = _serialize_config_value(dt)
        assert isinstance(result, str)


# =============================================================================
# STAGE_RESULT TESTS
# =============================================================================


class TestStageResult:
    """Test StageResult dataclass."""

    def test_create_stage_result(self) -> None:
        """Should create successful StageResult."""
        start = datetime.now()
        result = create_stage_result(
            stage_name="test_stage",
            start_time=start,
            artifacts=[Path("/path/to/artifact.parquet")],
            metadata={"rows": 1000},
        )

        assert result.stage_name == "test_stage"
        assert result.status == StageStatus.COMPLETED
        assert result.error is None
        assert len(result.artifacts) == 1
        assert result.metadata["rows"] == 1000

    def test_create_failed_result(self) -> None:
        """Should create failed StageResult."""
        start = datetime.now()
        result = create_failed_result(
            stage_name="test_stage",
            start_time=start,
            error="Something went wrong",
        )

        assert result.stage_name == "test_stage"
        assert result.status == StageStatus.FAILED
        assert result.error == "Something went wrong"

    def test_to_dict_serialization(self) -> None:
        """StageResult should serialize to dict."""
        start = datetime.now()
        result = create_stage_result(
            stage_name="test",
            start_time=start,
            metadata={"key": "value"},
        )

        data = result.to_dict()

        assert data["stage_name"] == "test"
        assert data["status"] == "completed"
        assert "start_time" in data
        assert "end_time" in data
        assert data["metadata"]["key"] == "value"

    def test_from_dict_deserialization(self) -> None:
        """StageResult should deserialize from dict."""
        data = {
            "stage_name": "test",
            "status": "completed",
            "start_time": "2024-01-01T12:00:00",
            "end_time": "2024-01-01T12:01:00",
            "duration_seconds": 60.0,
            "artifacts": ["/path/to/file.parquet"],
            "error": None,
            "metadata": {"key": "value"},
        }

        result = StageResult.from_dict(data)

        assert result.stage_name == "test"
        assert result.status == StageStatus.COMPLETED
        assert result.duration_seconds == 60.0
        assert len(result.artifacts) == 1
        assert result.metadata["key"] == "value"


# =============================================================================
# STAGE_STATUS TESTS
# =============================================================================


class TestStageStatus:
    """Test StageStatus enum."""

    def test_all_statuses_defined(self) -> None:
        """All expected statuses should be defined."""
        assert StageStatus.PENDING.value == "pending"
        assert StageStatus.IN_PROGRESS.value == "in_progress"
        assert StageStatus.COMPLETED.value == "completed"
        assert StageStatus.FAILED.value == "failed"
        assert StageStatus.SKIPPED.value == "skipped"

    def test_status_from_string(self) -> None:
        """Should create status from string value."""
        assert StageStatus("completed") == StageStatus.COMPLETED
        assert StageStatus("failed") == StageStatus.FAILED
