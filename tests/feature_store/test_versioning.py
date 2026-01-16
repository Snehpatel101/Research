"""
Tests for feature store versioning module.

Tests cover:
- SemanticVersion parsing and comparison
- VersionManager registration and queries
- Version suggestion based on changes
- Hash computation utilities
"""

import pytest
from datetime import datetime

from src.feature_store.versioning import (
    SemanticVersion,
    VersionInfo,
    VersionManager,
    compute_config_hash,
    compute_schema_hash,
)


# =============================================================================
# SEMANTIC VERSION TESTS
# =============================================================================


class TestSemanticVersion:
    """Tests for SemanticVersion dataclass."""

    def test_create_version(self):
        """Test creating a semantic version."""
        v = SemanticVersion(1, 2, 3)
        assert v.major == 1
        assert v.minor == 2
        assert v.patch == 3

    def test_str_representation(self):
        """Test string representation."""
        v = SemanticVersion(1, 2, 3)
        assert str(v) == "1.2.3"

    def test_parse_valid_version(self):
        """Test parsing a valid version string."""
        v = SemanticVersion.parse("1.2.3")
        assert v.major == 1
        assert v.minor == 2
        assert v.patch == 3

    def test_parse_with_whitespace(self):
        """Test parsing version with whitespace."""
        v = SemanticVersion.parse("  2.0.0  ")
        assert v.major == 2
        assert v.minor == 0
        assert v.patch == 0

    def test_parse_invalid_format(self):
        """Test parsing invalid version format raises ValueError."""
        with pytest.raises(ValueError, match="Invalid version string"):
            SemanticVersion.parse("1.2")

        with pytest.raises(ValueError, match="Invalid version string"):
            SemanticVersion.parse("v1.2.3")

        with pytest.raises(ValueError, match="Invalid version string"):
            SemanticVersion.parse("1.2.3.4")

    def test_negative_components_raise(self):
        """Test that negative version components raise ValueError."""
        with pytest.raises(ValueError, match="non-negative"):
            SemanticVersion(-1, 0, 0)

    def test_version_comparison_equal(self):
        """Test version equality."""
        v1 = SemanticVersion(1, 2, 3)
        v2 = SemanticVersion(1, 2, 3)
        assert v1 == v2

    def test_version_comparison_not_equal(self):
        """Test version inequality."""
        v1 = SemanticVersion(1, 2, 3)
        v2 = SemanticVersion(1, 2, 4)
        assert v1 != v2

    def test_version_ordering(self):
        """Test version ordering (less than)."""
        v1 = SemanticVersion(1, 0, 0)
        v2 = SemanticVersion(2, 0, 0)
        v3 = SemanticVersion(1, 1, 0)
        v4 = SemanticVersion(1, 0, 1)

        assert v1 < v2
        assert v1 < v3
        assert v1 < v4
        assert v4 < v3

    def test_version_sorting(self):
        """Test that versions can be sorted."""
        versions = [
            SemanticVersion(2, 0, 0),
            SemanticVersion(1, 0, 1),
            SemanticVersion(1, 0, 0),
            SemanticVersion(1, 1, 0),
        ]
        sorted_versions = sorted(versions)

        assert sorted_versions[0] == SemanticVersion(1, 0, 0)
        assert sorted_versions[1] == SemanticVersion(1, 0, 1)
        assert sorted_versions[2] == SemanticVersion(1, 1, 0)
        assert sorted_versions[3] == SemanticVersion(2, 0, 0)

    def test_version_hashable(self):
        """Test that versions are hashable (can be used in sets/dicts)."""
        v1 = SemanticVersion(1, 0, 0)
        v2 = SemanticVersion(1, 0, 0)
        v3 = SemanticVersion(2, 0, 0)

        version_set = {v1, v2, v3}
        assert len(version_set) == 2  # v1 and v2 are equal

    def test_bump_major(self):
        """Test major version bump."""
        v = SemanticVersion(1, 2, 3)
        bumped = v.bump_major()

        assert bumped == SemanticVersion(2, 0, 0)

    def test_bump_minor(self):
        """Test minor version bump."""
        v = SemanticVersion(1, 2, 3)
        bumped = v.bump_minor()

        assert bumped == SemanticVersion(1, 3, 0)

    def test_bump_patch(self):
        """Test patch version bump."""
        v = SemanticVersion(1, 2, 3)
        bumped = v.bump_patch()

        assert bumped == SemanticVersion(1, 2, 4)


# =============================================================================
# VERSION INFO TESTS
# =============================================================================


class TestVersionInfo:
    """Tests for VersionInfo dataclass."""

    def test_create_version_info(self):
        """Test creating version info."""
        info = VersionInfo(
            version=SemanticVersion(1, 0, 0),
            created_at=datetime.now(),
            description="Initial release",
            schema_hash="abc123",
            config_hash="def456",
        )

        assert info.version == SemanticVersion(1, 0, 0)
        assert info.description == "Initial release"
        assert info.schema_hash == "abc123"

    def test_to_dict(self):
        """Test serialization to dict."""
        now = datetime.now()
        info = VersionInfo(
            version=SemanticVersion(1, 0, 0),
            created_at=now,
            description="Test",
            schema_hash="abc123",
            config_hash="def456",
            metadata={"key": "value"},
        )

        d = info.to_dict()

        assert d["version"] == "1.0.0"
        assert d["description"] == "Test"
        assert d["schema_hash"] == "abc123"
        assert d["metadata"] == {"key": "value"}

    def test_from_dict(self):
        """Test deserialization from dict."""
        data = {
            "version": "2.1.0",
            "created_at": "2024-01-01T12:00:00",
            "description": "Test version",
            "schema_hash": "abc123",
            "config_hash": "def456",
            "metadata": {"foo": "bar"},
        }

        info = VersionInfo.from_dict(data)

        assert info.version == SemanticVersion(2, 1, 0)
        assert info.description == "Test version"
        assert info.metadata == {"foo": "bar"}


# =============================================================================
# VERSION MANAGER TESTS
# =============================================================================


class TestVersionManager:
    """Tests for VersionManager class."""

    @pytest.fixture
    def manager(self):
        """Create a test version manager."""
        return VersionManager("test_features")

    def test_create_manager(self):
        """Test creating a version manager."""
        manager = VersionManager("core_features")
        assert manager.feature_set == "core_features"
        assert len(manager) == 0

    def test_empty_feature_set_raises(self):
        """Test that empty feature set name raises ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            VersionManager("")

    def test_register_version(self, manager):
        """Test registering a new version."""
        v = SemanticVersion(1, 0, 0)
        info = manager.register_version(
            version=v,
            description="Initial",
            schema_hash="abc",
            config_hash="def",
        )

        assert info.version == v
        assert len(manager) == 1
        assert v in manager

    def test_register_duplicate_raises(self, manager):
        """Test that registering duplicate version raises ValueError."""
        v = SemanticVersion(1, 0, 0)
        manager.register_version(version=v)

        with pytest.raises(ValueError, match="already exists"):
            manager.register_version(version=v)

    def test_get_version(self, manager):
        """Test getting version by number."""
        v = SemanticVersion(1, 0, 0)
        manager.register_version(version=v, description="Test")

        info = manager.get_version(v)
        assert info is not None
        assert info.description == "Test"

    def test_get_version_by_string(self, manager):
        """Test getting version by string."""
        v = SemanticVersion(1, 0, 0)
        manager.register_version(version=v, description="Test")

        info = manager.get_version("1.0.0")
        assert info is not None
        assert info.version == v

    def test_get_nonexistent_version(self, manager):
        """Test getting nonexistent version returns None."""
        info = manager.get_version(SemanticVersion(1, 0, 0))
        assert info is None

    def test_latest_version_empty(self, manager):
        """Test latest_version on empty manager returns None."""
        assert manager.latest_version is None

    def test_latest_version(self, manager):
        """Test latest_version returns highest version."""
        manager.register_version(version=SemanticVersion(1, 0, 0))
        manager.register_version(version=SemanticVersion(2, 0, 0))
        manager.register_version(version=SemanticVersion(1, 5, 0))

        assert manager.latest_version == SemanticVersion(2, 0, 0)

    def test_all_versions_sorted(self, manager):
        """Test all_versions returns sorted list."""
        manager.register_version(version=SemanticVersion(2, 0, 0))
        manager.register_version(version=SemanticVersion(1, 0, 0))
        manager.register_version(version=SemanticVersion(1, 1, 0))

        versions = manager.all_versions

        assert versions[0] == SemanticVersion(1, 0, 0)
        assert versions[1] == SemanticVersion(1, 1, 0)
        assert versions[2] == SemanticVersion(2, 0, 0)

    def test_suggest_version_first(self, manager):
        """Test version suggestion for first version."""
        suggested = manager.suggest_version(
            current_schema_hash="abc",
            current_config_hash="def",
        )

        assert suggested == SemanticVersion(1, 0, 0)

    def test_suggest_version_schema_change(self, manager):
        """Test version suggestion when schema changes."""
        manager.register_version(
            version=SemanticVersion(1, 0, 0),
            schema_hash="old_schema",
            config_hash="config",
        )

        suggested = manager.suggest_version(
            current_schema_hash="new_schema",
            current_config_hash="config",
        )

        # Schema change = major bump
        assert suggested == SemanticVersion(2, 0, 0)

    def test_suggest_version_config_change(self, manager):
        """Test version suggestion when only config changes."""
        manager.register_version(
            version=SemanticVersion(1, 0, 0),
            schema_hash="schema",
            config_hash="old_config",
        )

        suggested = manager.suggest_version(
            current_schema_hash="schema",
            current_config_hash="new_config",
        )

        # Config change only = minor bump
        assert suggested == SemanticVersion(1, 1, 0)

    def test_suggest_version_no_change(self, manager):
        """Test version suggestion when nothing changes."""
        manager.register_version(
            version=SemanticVersion(1, 0, 0),
            schema_hash="schema",
            config_hash="config",
        )

        suggested = manager.suggest_version(
            current_schema_hash="schema",
            current_config_hash="config",
        )

        # No change = patch bump
        assert suggested == SemanticVersion(1, 0, 1)

    def test_contains_by_string(self, manager):
        """Test __contains__ with string version."""
        manager.register_version(version=SemanticVersion(1, 0, 0))

        assert "1.0.0" in manager
        assert "2.0.0" not in manager

    def test_iteration(self, manager):
        """Test iteration over version infos."""
        manager.register_version(version=SemanticVersion(1, 0, 0))
        manager.register_version(version=SemanticVersion(2, 0, 0))

        versions = list(manager)

        assert len(versions) == 2
        assert all(isinstance(v, VersionInfo) for v in versions)

    def test_to_dict_from_dict_roundtrip(self, manager):
        """Test serialization roundtrip."""
        manager.register_version(
            version=SemanticVersion(1, 0, 0),
            description="V1",
            schema_hash="abc",
        )
        manager.register_version(
            version=SemanticVersion(1, 1, 0),
            description="V1.1",
            schema_hash="abc",
        )

        data = manager.to_dict()
        restored = VersionManager.from_dict(data)

        assert restored.feature_set == manager.feature_set
        assert len(restored) == len(manager)
        assert restored.latest_version == manager.latest_version


# =============================================================================
# HASH COMPUTATION TESTS
# =============================================================================


class TestHashComputation:
    """Tests for hash computation utilities."""

    def test_compute_schema_hash(self):
        """Test schema hash computation."""
        columns = ["a", "b", "c"]
        dtypes = {"a": "float64", "b": "int64", "c": "object"}

        hash1 = compute_schema_hash(columns, dtypes)

        assert isinstance(hash1, str)
        assert len(hash1) == 16  # Truncated to 16 chars

    def test_schema_hash_order_independent(self):
        """Test that schema hash is order-independent."""
        hash1 = compute_schema_hash(["a", "b", "c"])
        hash2 = compute_schema_hash(["c", "a", "b"])

        assert hash1 == hash2

    def test_schema_hash_changes_with_columns(self):
        """Test that schema hash changes when columns change."""
        hash1 = compute_schema_hash(["a", "b"])
        hash2 = compute_schema_hash(["a", "b", "c"])

        assert hash1 != hash2

    def test_compute_config_hash(self):
        """Test config hash computation."""
        config = {"param1": 10, "param2": "value"}
        hash1 = compute_config_hash(config)

        assert isinstance(hash1, str)
        assert len(hash1) == 16

    def test_config_hash_order_independent(self):
        """Test that config hash is order-independent."""
        hash1 = compute_config_hash({"a": 1, "b": 2})
        hash2 = compute_config_hash({"b": 2, "a": 1})

        assert hash1 == hash2

    def test_config_hash_changes_with_values(self):
        """Test that config hash changes when values change."""
        hash1 = compute_config_hash({"a": 1})
        hash2 = compute_config_hash({"a": 2})

        assert hash1 != hash2
