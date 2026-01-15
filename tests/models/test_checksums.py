"""
Tests for model artifact checksums.

Tests the ArtifactIntegrityManager class that provides SHA256 checksums
for model artifacts to ensure integrity during storage and loading.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from src.models.training.checksums import (
    ArtifactChecksum,
    ArtifactIntegrityManager,
    compute_file_checksum,
)


class TestArtifactIntegrityManager:
    """Tests for ArtifactIntegrityManager class."""

    @pytest.fixture
    def artifacts_dir(self) -> Path:
        """Create temporary artifacts directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def sample_artifacts(self, artifacts_dir: Path) -> dict[str, Path]:
        """Create sample artifact files."""
        artifacts = {}

        # Create checkpoints directory
        checkpoints_dir = artifacts_dir / "checkpoints"
        checkpoints_dir.mkdir(parents=True)

        # Create model file
        model_path = checkpoints_dir / "model.pt"
        model_path.write_bytes(b"fake model data " * 100)
        artifacts["model"] = model_path

        # Create config directory
        config_dir = artifacts_dir / "config"
        config_dir.mkdir(parents=True)

        # Create config file
        config_path = config_dir / "config.json"
        config_path.write_text('{"param": "value"}')
        artifacts["config"] = config_path

        # Create calibrator file
        calibrator_path = checkpoints_dir / "calibrator.pkl"
        calibrator_path.write_bytes(b"fake calibrator data")
        artifacts["calibrator"] = calibrator_path

        return artifacts

    def test_compute_checksum_deterministic(self, sample_artifacts: dict) -> None:
        """Checksum should be deterministic for same file."""
        manager = ArtifactIntegrityManager(sample_artifacts["model"].parent.parent)

        checksum1 = manager.compute_checksum(sample_artifacts["model"])
        checksum2 = manager.compute_checksum(sample_artifacts["model"])

        assert checksum1 == checksum2
        assert len(checksum1) == 64  # SHA256 hex digest

    def test_compute_checksum_different_files(self, sample_artifacts: dict) -> None:
        """Different files should have different checksums."""
        manager = ArtifactIntegrityManager(sample_artifacts["model"].parent.parent)

        checksum1 = manager.compute_checksum(sample_artifacts["model"])
        checksum2 = manager.compute_checksum(sample_artifacts["config"])

        assert checksum1 != checksum2

    def test_register_artifact(
        self, artifacts_dir: Path, sample_artifacts: dict
    ) -> None:
        """Should register artifact and compute checksum."""
        manager = ArtifactIntegrityManager(artifacts_dir)

        checksum = manager.register_artifact(sample_artifacts["model"])

        assert isinstance(checksum, ArtifactChecksum)
        assert checksum.sha256
        assert checksum.size_bytes > 0
        assert checksum.created_at

    def test_register_artifact_not_found(self, artifacts_dir: Path) -> None:
        """Should raise FileNotFoundError for missing file."""
        manager = ArtifactIntegrityManager(artifacts_dir)

        with pytest.raises(FileNotFoundError):
            manager.register_artifact(artifacts_dir / "nonexistent.pt")

    def test_register_all(self, artifacts_dir: Path, sample_artifacts: dict) -> None:
        """Should register all matching artifacts."""
        manager = ArtifactIntegrityManager(artifacts_dir)

        checksums = manager.register_all()

        # Should have registered all sample artifacts
        assert len(checksums) >= 3  # model.pt, config.json, calibrator.pkl

    def test_verify_artifact_valid(
        self, artifacts_dir: Path, sample_artifacts: dict
    ) -> None:
        """Should verify valid artifact."""
        manager = ArtifactIntegrityManager(artifacts_dir)

        manager.register_artifact(sample_artifacts["model"])

        assert manager.verify_artifact(sample_artifacts["model"]) is True

    def test_verify_artifact_modified(
        self, artifacts_dir: Path, sample_artifacts: dict
    ) -> None:
        """Should detect modified artifact."""
        manager = ArtifactIntegrityManager(artifacts_dir)

        manager.register_artifact(sample_artifacts["model"])

        # Modify the file
        sample_artifacts["model"].write_bytes(b"modified data")

        assert manager.verify_artifact(sample_artifacts["model"]) is False

    def test_verify_artifact_missing(
        self, artifacts_dir: Path, sample_artifacts: dict
    ) -> None:
        """Should detect missing artifact."""
        manager = ArtifactIntegrityManager(artifacts_dir)

        manager.register_artifact(sample_artifacts["model"])

        # Delete the file
        sample_artifacts["model"].unlink()

        assert manager.verify_artifact(sample_artifacts["model"]) is False

    def test_verify_artifact_not_registered(
        self, artifacts_dir: Path, sample_artifacts: dict
    ) -> None:
        """Should return False for unregistered artifact."""
        manager = ArtifactIntegrityManager(artifacts_dir)

        # Don't register the artifact
        assert manager.verify_artifact(sample_artifacts["model"]) is False

    def test_verify_all(self, artifacts_dir: Path, sample_artifacts: dict) -> None:
        """Should verify all registered artifacts."""
        manager = ArtifactIntegrityManager(artifacts_dir)

        manager.register_all()

        results = manager.verify_all()

        assert all(results.values())

    def test_verify_all_with_modified(
        self, artifacts_dir: Path, sample_artifacts: dict
    ) -> None:
        """Should detect modified artifacts in verify_all."""
        manager = ArtifactIntegrityManager(artifacts_dir)

        manager.register_all()

        # Modify one file
        sample_artifacts["model"].write_bytes(b"modified data")

        results = manager.verify_all()

        # At least one should fail
        assert not all(results.values())

    def test_save_and_load_checksums(
        self, artifacts_dir: Path, sample_artifacts: dict
    ) -> None:
        """Checksums should persist across manager instances."""
        # First instance - register and save
        manager1 = ArtifactIntegrityManager(artifacts_dir)
        manager1.register_all()
        manager1.save_checksums()

        # Second instance - should load checksums
        manager2 = ArtifactIntegrityManager(artifacts_dir)

        # Verify all should work with loaded checksums
        results = manager2.verify_all()
        assert all(results.values())

    def test_get_artifact_info(
        self, artifacts_dir: Path, sample_artifacts: dict
    ) -> None:
        """Should retrieve artifact info."""
        manager = ArtifactIntegrityManager(artifacts_dir)

        checksum = manager.register_artifact(sample_artifacts["model"])

        info = manager.get_artifact_info(sample_artifacts["model"])

        assert info is not None
        assert info.sha256 == checksum.sha256

    def test_get_artifact_info_not_registered(
        self, artifacts_dir: Path, sample_artifacts: dict
    ) -> None:
        """Should return None for unregistered artifact."""
        manager = ArtifactIntegrityManager(artifacts_dir)

        info = manager.get_artifact_info(sample_artifacts["model"])

        assert info is None

    def test_list_artifacts(
        self, artifacts_dir: Path, sample_artifacts: dict
    ) -> None:
        """Should list all registered artifacts."""
        manager = ArtifactIntegrityManager(artifacts_dir)

        manager.register_all()

        artifacts = manager.list_artifacts()

        assert len(artifacts) >= 3

    def test_clear(self, artifacts_dir: Path, sample_artifacts: dict) -> None:
        """Should clear all checksums."""
        manager = ArtifactIntegrityManager(artifacts_dir)

        manager.register_all()
        manager.save_checksums()

        manager.clear()

        assert len(manager.list_artifacts()) == 0
        assert not manager.checksums_file.exists()

    def test_quick_verify_valid(
        self, artifacts_dir: Path, sample_artifacts: dict
    ) -> None:
        """Quick verify should work without manager instance."""
        # Compute expected checksum
        manager = ArtifactIntegrityManager(artifacts_dir)
        expected = manager.compute_checksum(sample_artifacts["model"])

        # Quick verify
        assert ArtifactIntegrityManager.quick_verify(
            sample_artifacts["model"], expected
        )

    def test_quick_verify_invalid(
        self, artifacts_dir: Path, sample_artifacts: dict
    ) -> None:
        """Quick verify should detect invalid checksum."""
        assert not ArtifactIntegrityManager.quick_verify(
            sample_artifacts["model"], "invalid_checksum"
        )

    def test_quick_verify_missing_file(self, artifacts_dir: Path) -> None:
        """Quick verify should return False for missing file."""
        assert not ArtifactIntegrityManager.quick_verify(
            artifacts_dir / "nonexistent.pt", "any_checksum"
        )


class TestArtifactChecksum:
    """Tests for ArtifactChecksum dataclass."""

    def test_to_dict(self) -> None:
        """Should convert to dictionary."""
        checksum = ArtifactChecksum(
            file_path="checkpoints/model.pt",
            sha256="abc123" * 10 + "abc1",
            size_bytes=1000,
            created_at="2024-01-01T00:00:00",
        )

        d = checksum.to_dict()

        assert d["file_path"] == "checkpoints/model.pt"
        assert d["sha256"] == "abc123" * 10 + "abc1"
        assert d["size_bytes"] == 1000

    def test_from_dict(self) -> None:
        """Should create from dictionary."""
        d = {
            "file_path": "checkpoints/model.pt",
            "sha256": "abc123" * 10 + "abc1",
            "size_bytes": 1000,
            "created_at": "2024-01-01T00:00:00",
        }

        checksum = ArtifactChecksum.from_dict(d)

        assert checksum.file_path == "checkpoints/model.pt"
        assert checksum.size_bytes == 1000

    def test_roundtrip(self) -> None:
        """Should survive roundtrip through dict."""
        checksum = ArtifactChecksum(
            file_path="checkpoints/model.pt",
            sha256="abc123" * 10 + "abc1",
            size_bytes=1000,
            created_at="2024-01-01T00:00:00",
        )

        d = checksum.to_dict()
        checksum2 = ArtifactChecksum.from_dict(d)

        assert checksum == checksum2


class TestComputeFileChecksum:
    """Tests for compute_file_checksum function."""

    @pytest.fixture
    def temp_file(self) -> Path:
        """Create temporary file."""
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(b"test data for checksum")
            path = Path(f.name)
        yield path
        if path.exists():
            path.unlink()

    def test_computes_sha256(self, temp_file: Path) -> None:
        """Should compute SHA256 hash."""
        checksum = compute_file_checksum(temp_file)

        assert len(checksum) == 64  # SHA256 hex digest
        assert all(c in "0123456789abcdef" for c in checksum)

    def test_deterministic(self, temp_file: Path) -> None:
        """Should be deterministic for same file."""
        checksum1 = compute_file_checksum(temp_file)
        checksum2 = compute_file_checksum(temp_file)

        assert checksum1 == checksum2

    def test_different_for_different_content(self) -> None:
        """Different content should have different checksums."""
        with tempfile.NamedTemporaryFile(delete=False) as f1:
            f1.write(b"content 1")
            path1 = Path(f1.name)

        with tempfile.NamedTemporaryFile(delete=False) as f2:
            f2.write(b"content 2")
            path2 = Path(f2.name)

        try:
            checksum1 = compute_file_checksum(path1)
            checksum2 = compute_file_checksum(path2)

            assert checksum1 != checksum2
        finally:
            path1.unlink()
            path2.unlink()
