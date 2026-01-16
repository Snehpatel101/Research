"""
Tests for ArtifactManifest.

Phase 0 SNwH implementation tests.
"""

import json
import tempfile
from pathlib import Path

import pytest
import numpy as np

from src.contracts import (
    DataRank,
    FeatureMode,
    DataContract,
    ModelContract,
    ArtifactManifest,
    get_model_contract,
)


class TestArtifactManifest:
    """Tests for ArtifactManifest dataclass."""

    def test_basic_creation(self):
        """Test basic ArtifactManifest creation."""
        manifest = ArtifactManifest(
            artifact_type="model",
            artifact_name="xgboost_h20",
            artifact_path="/path/to/model.pkl",
        )
        assert manifest.artifact_type == "model"
        assert manifest.artifact_name == "xgboost_h20"
        assert manifest.artifact_path == "/path/to/model.pkl"

    def test_auto_python_version(self):
        """Test automatic Python version capture."""
        manifest = ArtifactManifest(
            artifact_type="model",
            artifact_name="test",
            artifact_path="/path/to/model.pkl",
        )
        assert manifest.python_version is not None
        assert "." in manifest.python_version  # e.g., "3.11.5"

    def test_auto_code_version(self):
        """Test automatic code version capture."""
        manifest = ArtifactManifest(
            artifact_type="model",
            artifact_name="test",
            artifact_path="/path/to/model.pkl",
        )
        # code_version should be set (might be "unknown" in non-git env)
        assert manifest.code_version is not None

    def test_auto_package_versions(self):
        """Test automatic package version capture."""
        manifest = ArtifactManifest(
            artifact_type="model",
            artifact_name="test",
            artifact_path="/path/to/model.pkl",
        )
        # Should have captured numpy at minimum
        assert "numpy" in manifest.package_versions

    def test_to_dict_from_dict_roundtrip(self):
        """Test serialization roundtrip."""
        manifest = ArtifactManifest(
            artifact_type="model",
            artifact_name="xgboost_h20",
            artifact_path="/path/to/model.pkl",
            config_hash="abc123",
            data_hash="def456",
            model_name="xgboost",
            model_family="boosting",
            horizon=20,
            timeframe="15min",
            symbol="MES",
            pipeline_run_id="run-123",
            training_run_id="train-456",
            training_metrics={"val_accuracy": 0.85},
        )

        # Roundtrip
        data = manifest.to_dict()
        manifest2 = ArtifactManifest.from_dict(data)

        assert manifest2.artifact_type == manifest.artifact_type
        assert manifest2.artifact_name == manifest.artifact_name
        assert manifest2.config_hash == manifest.config_hash
        assert manifest2.data_hash == manifest.data_hash
        assert manifest2.model_name == manifest.model_name
        assert manifest2.horizon == manifest.horizon
        assert manifest2.training_metrics == manifest.training_metrics

    def test_to_dict_json_serializable(self):
        """Test to_dict produces JSON-serializable output."""
        manifest = ArtifactManifest(
            artifact_type="model",
            artifact_name="test",
            artifact_path="/path/to/model.pkl",
            training_metrics={"accuracy": 0.95, "f1": 0.92},
        )
        data = manifest.to_dict()
        json_str = json.dumps(data)
        assert isinstance(json_str, str)

    def test_repr(self):
        """Test string representation."""
        manifest = ArtifactManifest(
            artifact_type="model",
            artifact_name="xgboost_h20",
            artifact_path="/path/to/model.pkl",
            model_name="xgboost",
            artifact_hash="abcd1234",
        )
        repr_str = repr(manifest)
        assert "model" in repr_str
        assert "xgboost_h20" in repr_str
        assert "xgboost" in repr_str


class TestArtifactManifestFileOperations:
    """Tests for ArtifactManifest file operations."""

    def test_save_and_load(self, tmp_path):
        """Test save and load manifest."""
        manifest = ArtifactManifest(
            artifact_type="model",
            artifact_name="test_model",
            artifact_path="/path/to/model.pkl",
            model_name="xgboost",
            horizon=20,
            training_metrics={"val_accuracy": 0.85},
        )

        # Save
        manifest_path = tmp_path / "manifest.json"
        manifest.save(manifest_path)

        assert manifest_path.exists()

        # Load
        loaded = ArtifactManifest.load(manifest_path)
        assert loaded.artifact_name == manifest.artifact_name
        assert loaded.model_name == manifest.model_name
        assert loaded.training_metrics == manifest.training_metrics

    def test_compute_artifact_hash(self, tmp_path):
        """Test artifact hash computation."""
        # Create a temp file with known content
        test_file = tmp_path / "test_artifact.bin"
        test_file.write_bytes(b"test content for hashing")

        manifest = ArtifactManifest(
            artifact_type="model",
            artifact_name="test",
            artifact_path=str(test_file),
        )

        hash1 = manifest.compute_artifact_hash(test_file)
        assert hash1 is not None
        assert len(hash1) == 16  # SHA256 truncated to 16 chars

        # Same content should give same hash
        hash2 = manifest.compute_artifact_hash(test_file)
        assert hash1 == hash2

        # Different content should give different hash
        test_file2 = tmp_path / "test_artifact2.bin"
        test_file2.write_bytes(b"different content")
        hash3 = manifest.compute_artifact_hash(test_file2)
        assert hash1 != hash3

    def test_compute_artifact_hash_nonexistent(self, tmp_path):
        """Test artifact hash for nonexistent file."""
        manifest = ArtifactManifest(
            artifact_type="model",
            artifact_name="test",
            artifact_path="/nonexistent/path.pkl",
        )
        hash_val = manifest.compute_artifact_hash("/nonexistent/path.pkl")
        assert hash_val == ""

    def test_validate_artifact_exists(self, tmp_path):
        """Test artifact validation - file exists."""
        test_file = tmp_path / "model.pkl"
        test_file.write_bytes(b"model content")

        manifest = ArtifactManifest(
            artifact_type="model",
            artifact_name="test",
            artifact_path=str(test_file),
        )
        manifest.artifact_hash = manifest.compute_artifact_hash(test_file)

        is_valid, issues = manifest.validate_artifact(test_file)
        assert is_valid
        assert len(issues) == 0

    def test_validate_artifact_not_exists(self, tmp_path):
        """Test artifact validation - file doesn't exist."""
        manifest = ArtifactManifest(
            artifact_type="model",
            artifact_name="test",
            artifact_path="/nonexistent/path.pkl",
        )

        is_valid, issues = manifest.validate_artifact("/nonexistent/path.pkl")
        assert not is_valid
        assert any("not found" in issue.lower() for issue in issues)

    def test_validate_artifact_hash_mismatch(self, tmp_path):
        """Test artifact validation - hash mismatch."""
        test_file = tmp_path / "model.pkl"
        test_file.write_bytes(b"original content")

        manifest = ArtifactManifest(
            artifact_type="model",
            artifact_name="test",
            artifact_path=str(test_file),
        )
        manifest.artifact_hash = manifest.compute_artifact_hash(test_file)

        # Modify file
        test_file.write_bytes(b"modified content")

        is_valid, issues = manifest.validate_artifact(test_file)
        assert not is_valid
        assert any("hash mismatch" in issue.lower() for issue in issues)


class TestArtifactManifestEnvironment:
    """Tests for environment validation."""

    def test_validate_environment_compatible(self):
        """Test environment validation - compatible."""
        import sys

        manifest = ArtifactManifest(
            artifact_type="model",
            artifact_name="test",
            artifact_path="/path/to/model.pkl",
        )
        # Environment should match itself
        is_valid, warnings = manifest.validate_environment()
        assert is_valid
        assert len(warnings) == 0

    def test_validate_environment_python_mismatch(self):
        """Test environment validation - Python version mismatch."""
        manifest = ArtifactManifest(
            artifact_type="model",
            artifact_name="test",
            artifact_path="/path/to/model.pkl",
            python_version="2.7.18",  # Old version
        )

        is_valid, warnings = manifest.validate_environment()
        assert not is_valid  # Major version mismatch
        assert any("python version mismatch" in w.lower() for w in warnings)


class TestArtifactManifestFactoryMethods:
    """Tests for factory methods."""

    def test_create_for_model(self, tmp_path):
        """Test create_for_model factory method."""
        # Create a temp model file
        model_file = tmp_path / "model.pkl"
        model_file.write_bytes(b"model binary content")

        # Create contracts
        data_contract = DataContract(
            symbol="MES",
            timeframe="15min",
            horizon=20,
            split="train",
            n_samples=10000,
            n_features=100,
            data_rank=DataRank.TABULAR_2D,
        )
        model_contract = get_model_contract("xgboost")

        # Create manifest
        manifest = ArtifactManifest.create_for_model(
            model_path=model_file,
            model_name="xgboost",
            data_contract=data_contract,
            model_contract=model_contract,
            config_hash="config123",
            pipeline_run_id="pipeline-abc",
            training_run_id="train-xyz",
            training_metrics={"val_accuracy": 0.85, "val_f1": 0.83},
        )

        assert manifest.artifact_type == "model"
        assert manifest.artifact_name == "xgboost"
        assert manifest.model_name == "xgboost"
        assert manifest.model_family == "boosting"
        assert manifest.horizon == 20
        assert manifest.timeframe == "15min"
        assert manifest.symbol == "MES"
        assert manifest.config_hash == "config123"
        assert manifest.data_hash == data_contract.schema_hash
        assert manifest.artifact_hash != ""  # Should be computed
        assert manifest.training_metrics["val_accuracy"] == 0.85

    def test_create_for_predictions(self, tmp_path):
        """Test create_for_predictions factory method."""
        # Create a temp predictions file
        pred_file = tmp_path / "predictions.npz"
        pred_file.write_bytes(b"predictions content")

        data_contract = DataContract(
            symbol="MES",
            timeframe="15min",
            horizon=20,
            split="test",
            n_samples=2000,
            n_features=100,
            data_rank=DataRank.TABULAR_2D,
        )

        manifest = ArtifactManifest.create_for_predictions(
            predictions_path=pred_file,
            model_name="xgboost",
            data_contract=data_contract,
            split="test",
        )

        assert manifest.artifact_type == "predictions"
        assert "xgboost" in manifest.artifact_name
        assert "test" in manifest.artifact_name
        assert manifest.model_name == "xgboost"
        assert manifest.horizon == 20
        assert manifest.artifact_hash != ""

    def test_create_for_oof(self, tmp_path):
        """Test create_for_oof factory method."""
        # Create a temp OOF file
        oof_file = tmp_path / "oof.npz"
        oof_file.write_bytes(b"oof content")

        data_contract = DataContract(
            symbol="MES",
            timeframe="15min",
            horizon=20,
            split="train",
            n_samples=10000,
            n_features=100,
            data_rank=DataRank.TABULAR_2D,
        )

        manifest = ArtifactManifest.create_for_oof(
            oof_path=oof_file,
            model_name="lstm",
            data_contract=data_contract,
            n_folds=5,
            coverage=0.95,
        )

        assert manifest.artifact_type == "oof"
        assert "lstm" in manifest.artifact_name
        assert manifest.model_name == "lstm"
        assert manifest.training_metrics["n_folds"] == 5.0
        assert manifest.training_metrics["coverage"] == 0.95
        assert manifest.artifact_hash != ""


class TestArtifactManifestIntegration:
    """Integration tests for ArtifactManifest."""

    def test_full_workflow(self, tmp_path):
        """Test complete manifest workflow."""
        # 1. Create contracts
        data_contract = DataContract(
            symbol="MES",
            timeframe="15min",
            horizon=20,
            split="train",
            n_samples=10000,
            n_features=100,
            data_rank=DataRank.TABULAR_2D,
            feature_columns=[f"feature_{i}" for i in range(100)],
        )
        model_contract = get_model_contract("xgboost")

        # 2. Create model artifact
        model_file = tmp_path / "xgboost_h20.pkl"
        model_file.write_bytes(b"xgboost model binary")

        # 3. Create manifest
        manifest = ArtifactManifest.create_for_model(
            model_path=model_file,
            model_name="xgboost",
            data_contract=data_contract,
            model_contract=model_contract,
            training_metrics={"val_accuracy": 0.85},
        )

        # 4. Save manifest
        manifest_file = tmp_path / "manifest.json"
        manifest.save(manifest_file)

        # 5. Load manifest in "deployment"
        loaded_manifest = ArtifactManifest.load(manifest_file)

        # 6. Validate artifact
        is_valid, issues = loaded_manifest.validate_artifact(model_file)
        assert is_valid

        # 7. Validate environment
        is_compatible, warnings = loaded_manifest.validate_environment()
        assert is_compatible

        # 8. Access contract information
        assert loaded_manifest.data_contract["symbol"] == "MES"
        assert loaded_manifest.model_contract["model_name"] == "xgboost"
