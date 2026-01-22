"""
Artifact Manifest - Safety and reproducibility metadata for saved artifacts.

Every saved artifact (model, predictions, metrics) includes a manifest
that enables verification and reproducibility.

Phase 0 of the SNwH (Unified Multi-Timeframe Model Factory) implementation.
"""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .data_contract import DataContract
    from .model_contract import ModelContract


def _get_git_commit() -> str:
    """Get current git commit hash."""
    try:
        import subprocess

        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            cwd=Path(__file__).parent.parent.parent,  # Project root
        )
        if result.returncode == 0:
            return result.stdout.strip()[:8]
    except Exception:
        pass
    return "unknown"


def _get_package_versions() -> dict[str, str]:
    """Capture versions of key ML packages."""
    packages = [
        "torch",
        "numpy",
        "pandas",
        "scikit-learn",
        "xgboost",
        "lightgbm",
        "catboost",
    ]
    versions = {}
    for pkg in packages:
        try:
            # Handle sklearn special case
            import_name = "sklearn" if pkg == "scikit-learn" else pkg
            mod = __import__(import_name)
            versions[pkg] = getattr(mod, "__version__", "unknown")
        except ImportError:
            pass
    return versions


@dataclass
class ArtifactManifest:
    """
    Manifest for saved artifacts (models, predictions, etc.).

    Enables:
    - Verification that loaded artifacts match training environment
    - Reproducibility through config and code version tracking
    - Safe loading with hash validation
    """

    # Identity
    artifact_type: str  # "model", "predictions", "metrics", "oof", "checkpoint"
    artifact_name: str
    artifact_path: str

    # Hashes for verification
    config_hash: str = ""
    data_hash: str = ""
    code_version: str = ""
    artifact_hash: str = ""  # Hash of the artifact file itself

    # Contracts (serialized)
    data_contract: dict[str, Any] = field(default_factory=dict)
    model_contract: dict[str, Any] = field(default_factory=dict)

    # Training context
    model_name: str = ""
    model_family: str = ""
    horizon: int = 0
    timeframe: str = ""
    symbol: str = ""

    # Runtime info
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    python_version: str = ""
    package_versions: dict[str, str] = field(default_factory=dict)

    # Pipeline lineage
    pipeline_run_id: str = ""
    training_run_id: str = ""
    parent_artifact_id: str = ""  # For derived artifacts

    # Training metrics (optional summary)
    training_metrics: dict[str, float] = field(default_factory=dict)

    def __post_init__(self):
        """Capture runtime info if not provided."""
        if not self.python_version:
            self.python_version = (
                f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
            )
        if not self.code_version:
            self.code_version = _get_git_commit()
        if not self.package_versions:
            self.package_versions = _get_package_versions()

    def compute_artifact_hash(self, artifact_path: Path | str) -> str:
        """Compute SHA256 hash of artifact file."""
        path = Path(artifact_path)
        if not path.exists():
            return ""

        sha256 = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                sha256.update(chunk)
        return sha256.hexdigest()[:16]

    def validate_artifact(self, artifact_path: Path | str) -> tuple[bool, list[str]]:
        """
        Validate artifact against manifest.

        Args:
            artifact_path: Path to artifact file

        Returns:
            (is_valid, list_of_issues)
        """
        issues = []
        path = Path(artifact_path)

        # Check file exists
        if not path.exists():
            issues.append(f"Artifact file not found: {path}")
            return False, issues

        # Check artifact hash if stored
        if self.artifact_hash:
            current_hash = self.compute_artifact_hash(path)
            if current_hash != self.artifact_hash:
                issues.append(
                    f"Artifact hash mismatch: expected {self.artifact_hash}, " f"got {current_hash}"
                )

        return len(issues) == 0, issues

    def validate_environment(self) -> tuple[bool, list[str]]:
        """
        Validate current environment matches manifest.

        Returns:
            (is_valid, list_of_warnings)
        """
        warnings = []

        # Check Python version (major.minor)
        current_py = f"{sys.version_info.major}.{sys.version_info.minor}"
        manifest_py = ".".join(self.python_version.split(".")[:2]) if self.python_version else ""
        if manifest_py and current_py != manifest_py:
            warnings.append(
                f"Python version mismatch: trained with {manifest_py}, "
                f"running with {current_py}"
            )

        # Check key packages
        key_packages = ["torch", "numpy", "pandas", "scikit-learn"]
        for pkg in key_packages:
            if pkg in self.package_versions:
                try:
                    import_name = "sklearn" if pkg == "scikit-learn" else pkg
                    mod = __import__(import_name)
                    current_version = getattr(mod, "__version__", "unknown")
                    saved_version = self.package_versions[pkg]
                    # Only compare major.minor
                    current_major_minor = ".".join(current_version.split(".")[:2])
                    saved_major_minor = ".".join(saved_version.split(".")[:2])
                    if current_major_minor != saved_major_minor:
                        warnings.append(
                            f"{pkg} version mismatch: trained with {saved_version}, "
                            f"running with {current_version}"
                        )
                except ImportError:
                    warnings.append(f"Package {pkg} not installed")

        return len(warnings) == 0, warnings

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "artifact_type": self.artifact_type,
            "artifact_name": self.artifact_name,
            "artifact_path": self.artifact_path,
            "config_hash": self.config_hash,
            "data_hash": self.data_hash,
            "code_version": self.code_version,
            "artifact_hash": self.artifact_hash,
            "data_contract": self.data_contract,
            "model_contract": self.model_contract,
            "model_name": self.model_name,
            "model_family": self.model_family,
            "horizon": self.horizon,
            "timeframe": self.timeframe,
            "symbol": self.symbol,
            "created_at": self.created_at,
            "python_version": self.python_version,
            "package_versions": self.package_versions,
            "pipeline_run_id": self.pipeline_run_id,
            "training_run_id": self.training_run_id,
            "parent_artifact_id": self.parent_artifact_id,
            "training_metrics": self.training_metrics,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ArtifactManifest:
        """Deserialize from dictionary."""
        return cls(
            artifact_type=data.get("artifact_type", "unknown"),
            artifact_name=data.get("artifact_name", ""),
            artifact_path=data.get("artifact_path", ""),
            config_hash=data.get("config_hash", ""),
            data_hash=data.get("data_hash", ""),
            code_version=data.get("code_version", ""),
            artifact_hash=data.get("artifact_hash", ""),
            data_contract=data.get("data_contract", {}),
            model_contract=data.get("model_contract", {}),
            model_name=data.get("model_name", ""),
            model_family=data.get("model_family", ""),
            horizon=data.get("horizon", 0),
            timeframe=data.get("timeframe", ""),
            symbol=data.get("symbol", ""),
            created_at=data.get("created_at", ""),
            python_version=data.get("python_version", ""),
            package_versions=data.get("package_versions", {}),
            pipeline_run_id=data.get("pipeline_run_id", ""),
            training_run_id=data.get("training_run_id", ""),
            parent_artifact_id=data.get("parent_artifact_id", ""),
            training_metrics=data.get("training_metrics", {}),
        )

    def save(self, path: Path | str) -> None:
        """Save manifest to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: Path | str) -> ArtifactManifest:
        """Load manifest from JSON file."""
        with open(path) as f:
            data = json.load(f)
        return cls.from_dict(data)

    @classmethod
    def create_for_model(
        cls,
        model_path: Path | str,
        model_name: str,
        data_contract: DataContract,
        model_contract: ModelContract,
        config_hash: str = "",
        pipeline_run_id: str = "",
        training_run_id: str = "",
        training_metrics: dict[str, float] | None = None,
    ) -> ArtifactManifest:
        """
        Create manifest for a saved model.

        Args:
            model_path: Path to saved model file
            model_name: Name of the model
            data_contract: DataContract used for training
            model_contract: ModelContract for the model
            config_hash: Hash of training configuration
            pipeline_run_id: Pipeline run ID for lineage
            training_run_id: Training run ID
            training_metrics: Optional training metrics summary

        Returns:
            ArtifactManifest for the model
        """
        path = Path(model_path)

        manifest = cls(
            artifact_type="model",
            artifact_name=model_name,
            artifact_path=str(path),
            config_hash=config_hash,
            data_hash=data_contract.schema_hash,
            data_contract=data_contract.to_dict(),
            model_contract=model_contract.to_dict(),
            model_name=model_name,
            model_family=model_contract.model_family,
            horizon=data_contract.horizon,
            timeframe=data_contract.timeframe,
            symbol=data_contract.symbol,
            pipeline_run_id=pipeline_run_id,
            training_run_id=training_run_id,
            training_metrics=training_metrics or {},
        )

        # Compute artifact hash if file exists
        if path.exists():
            manifest.artifact_hash = manifest.compute_artifact_hash(path)

        return manifest

    @classmethod
    def create_for_predictions(
        cls,
        predictions_path: Path | str,
        model_name: str,
        data_contract: DataContract,
        parent_model_manifest: ArtifactManifest | None = None,
        split: str = "test",
    ) -> ArtifactManifest:
        """
        Create manifest for saved predictions.

        Args:
            predictions_path: Path to saved predictions file
            model_name: Name of the model that generated predictions
            data_contract: DataContract for the prediction data
            parent_model_manifest: Optional manifest of the model used
            split: Data split (train, val, test)

        Returns:
            ArtifactManifest for the predictions
        """
        path = Path(predictions_path)

        manifest = cls(
            artifact_type="predictions",
            artifact_name=f"{model_name}_{split}_predictions",
            artifact_path=str(path),
            data_hash=data_contract.schema_hash,
            data_contract=data_contract.to_dict(),
            model_name=model_name,
            horizon=data_contract.horizon,
            timeframe=data_contract.timeframe,
            symbol=data_contract.symbol,
            parent_artifact_id=(
                parent_model_manifest.artifact_hash if parent_model_manifest else ""
            ),
        )

        if path.exists():
            manifest.artifact_hash = manifest.compute_artifact_hash(path)

        return manifest

    @classmethod
    def create_for_oof(
        cls,
        oof_path: Path | str,
        model_name: str,
        data_contract: DataContract,
        n_folds: int = 5,
        coverage: float = 1.0,
    ) -> ArtifactManifest:
        """
        Create manifest for OOF predictions.

        Args:
            oof_path: Path to saved OOF predictions file
            model_name: Name of the model
            data_contract: DataContract for the OOF data
            n_folds: Number of CV folds
            coverage: OOF coverage ratio

        Returns:
            ArtifactManifest for OOF predictions
        """
        path = Path(oof_path)

        manifest = cls(
            artifact_type="oof",
            artifact_name=f"{model_name}_oof",
            artifact_path=str(path),
            data_hash=data_contract.schema_hash,
            data_contract=data_contract.to_dict(),
            model_name=model_name,
            horizon=data_contract.horizon,
            timeframe=data_contract.timeframe,
            symbol=data_contract.symbol,
            training_metrics={
                "n_folds": float(n_folds),
                "coverage": coverage,
            },
        )

        if path.exists():
            manifest.artifact_hash = manifest.compute_artifact_hash(path)

        return manifest

    def __repr__(self) -> str:
        """Concise string representation."""
        return (
            f"ArtifactManifest({self.artifact_type}: {self.artifact_name}, "
            f"model={self.model_name}, hash={self.artifact_hash[:8] if self.artifact_hash else 'none'})"
        )


__all__ = [
    "ArtifactManifest",
]
