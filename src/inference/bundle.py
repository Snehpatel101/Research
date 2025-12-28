"""
ModelBundle - Serializable container for trained model artifacts.

Bundles all components needed for inference:
- Trained model(s)
- Feature scaler
- Feature columns
- Probability calibrator (optional)
- Ensemble configuration (if ensemble)
- Metadata (horizon, training date, etc.)

Usage:
    # Save a bundle
    bundle = ModelBundle.from_training(
        model=trained_model,
        scaler=scaler,
        feature_columns=feature_cols,
        horizon=20,
    )
    bundle.save("/path/to/bundle")

    # Load and use
    bundle = ModelBundle.load("/path/to/bundle")
    predictions = bundle.predict(X_new)
"""
from __future__ import annotations

import hashlib
import json
import logging
import pickle
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler, StandardScaler

from src.models.base import BaseModel, PredictionOutput
from src.models.registry import ModelRegistry

logger = logging.getLogger(__name__)


# =============================================================================
# VERSION AND CONSTANTS
# =============================================================================

BUNDLE_VERSION = "1.0.0"
BUNDLE_MANIFEST_FILE = "manifest.json"
BUNDLE_MODEL_DIR = "model"
BUNDLE_SCALER_FILE = "scaler.pkl"
BUNDLE_CALIBRATOR_FILE = "calibrator.pkl"
BUNDLE_FEATURES_FILE = "features.json"
BUNDLE_METADATA_FILE = "metadata.json"


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class BundleMetadata:
    """Metadata for a model bundle."""
    version: str
    created_at: str
    model_name: str
    model_family: str
    horizon: int
    n_features: int
    feature_hash: str
    requires_sequences: bool = False
    sequence_length: int = 0
    has_calibrator: bool = False
    training_metrics: Dict[str, Any] = field(default_factory=dict)
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "created_at": self.created_at,
            "model_name": self.model_name,
            "model_family": self.model_family,
            "horizon": self.horizon,
            "n_features": self.n_features,
            "feature_hash": self.feature_hash,
            "requires_sequences": self.requires_sequences,
            "sequence_length": self.sequence_length,
            "has_calibrator": self.has_calibrator,
            "training_metrics": self.training_metrics,
            "extra": self.extra,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BundleMetadata":
        return cls(
            version=data["version"],
            created_at=data["created_at"],
            model_name=data["model_name"],
            model_family=data.get("model_family", "unknown"),
            horizon=data["horizon"],
            n_features=data["n_features"],
            feature_hash=data["feature_hash"],
            requires_sequences=data.get("requires_sequences", False),
            sequence_length=data.get("sequence_length", 0),
            has_calibrator=data.get("has_calibrator", False),
            training_metrics=data.get("training_metrics", {}),
            extra=data.get("extra", {}),
        )


@dataclass
class BundleManifest:
    """Manifest listing all files in a bundle."""
    version: str
    files: List[str]
    checksums: Dict[str, str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "files": self.files,
            "checksums": self.checksums,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BundleManifest":
        return cls(
            version=data["version"],
            files=data["files"],
            checksums=data.get("checksums", {}),
        )


# =============================================================================
# MODEL BUNDLE
# =============================================================================

class ModelBundle:
    """
    Serializable container for all inference artifacts.

    A bundle contains everything needed to make predictions:
    - The trained model
    - Feature scaler (for normalizing inputs)
    - Feature column names and order
    - Probability calibrator (optional)
    - Metadata about training

    Bundles are saved as directories with a standardized structure:
        bundle_dir/
            manifest.json       # File listing and checksums
            metadata.json       # Model metadata
            features.json       # Feature column names
            scaler.pkl          # Fitted scaler
            calibrator.pkl      # Fitted calibrator (optional)
            model/              # Model artifacts (via model.save())

    Example:
        >>> # Create bundle from trained components
        >>> bundle = ModelBundle.from_training(
        ...     model=trained_xgb,
        ...     scaler=fitted_scaler,
        ...     feature_columns=X.columns.tolist(),
        ...     horizon=20,
        ... )
        >>> bundle.save("./bundles/xgb_h20")

        >>> # Load and predict
        >>> bundle = ModelBundle.load("./bundles/xgb_h20")
        >>> predictions = bundle.predict(X_test)
    """

    def __init__(
        self,
        model: BaseModel,
        scaler: Optional[Union[RobustScaler, StandardScaler]],
        feature_columns: List[str],
        metadata: BundleMetadata,
        calibrator: Optional[Any] = None,
    ) -> None:
        """
        Initialize ModelBundle.

        Args:
            model: Trained model instance
            scaler: Fitted scaler (None for models that don't need scaling)
            feature_columns: Ordered list of feature column names
            metadata: Bundle metadata
            calibrator: Optional fitted probability calibrator
        """
        self.model = model
        self.scaler = scaler
        self.feature_columns = feature_columns
        self.metadata = metadata
        self.calibrator = calibrator

    @classmethod
    def from_training(
        cls,
        model: BaseModel,
        scaler: Optional[Union[RobustScaler, StandardScaler]],
        feature_columns: List[str],
        horizon: int,
        calibrator: Optional[Any] = None,
        training_metrics: Optional[Dict[str, Any]] = None,
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> "ModelBundle":
        """
        Create a bundle from trained components.

        Args:
            model: Trained model
            scaler: Fitted scaler
            feature_columns: Feature column names
            horizon: Prediction horizon
            calibrator: Optional fitted calibrator
            training_metrics: Optional training metrics to store
            extra_metadata: Additional metadata

        Returns:
            ModelBundle ready for saving
        """
        # Get model info
        model_name = getattr(model, "_get_model_type", lambda: "unknown")()
        model_family = getattr(model, "model_family", "unknown")
        requires_sequences = getattr(model, "requires_sequences", False)
        sequence_length = 0
        if requires_sequences:
            sequence_length = getattr(model, "_config", {}).get("sequence_length", 60)

        # Compute feature hash for validation
        feature_hash = hashlib.md5(
            ",".join(feature_columns).encode()
        ).hexdigest()[:12]

        metadata = BundleMetadata(
            version=BUNDLE_VERSION,
            created_at=datetime.now().isoformat(),
            model_name=model_name,
            model_family=model_family,
            horizon=horizon,
            n_features=len(feature_columns),
            feature_hash=feature_hash,
            requires_sequences=requires_sequences,
            sequence_length=sequence_length,
            has_calibrator=calibrator is not None,
            training_metrics=training_metrics or {},
            extra=extra_metadata or {},
        )

        return cls(
            model=model,
            scaler=scaler,
            feature_columns=feature_columns,
            metadata=metadata,
            calibrator=calibrator,
        )

    def save(self, path: Union[str, Path], overwrite: bool = False) -> Path:
        """
        Save bundle to disk.

        Args:
            path: Directory path for the bundle
            overwrite: If True, overwrite existing bundle

        Returns:
            Path to saved bundle

        Raises:
            FileExistsError: If path exists and overwrite=False
        """
        path = Path(path)

        if path.exists():
            if overwrite:
                shutil.rmtree(path)
            else:
                raise FileExistsError(
                    f"Bundle already exists at {path}. Use overwrite=True to replace."
                )

        path.mkdir(parents=True, exist_ok=True)

        files = []
        checksums = {}

        # Save metadata
        metadata_path = path / BUNDLE_METADATA_FILE
        with open(metadata_path, "w") as f:
            json.dump(self.metadata.to_dict(), f, indent=2)
        files.append(BUNDLE_METADATA_FILE)
        checksums[BUNDLE_METADATA_FILE] = self._file_checksum(metadata_path)

        # Save feature columns
        features_path = path / BUNDLE_FEATURES_FILE
        with open(features_path, "w") as f:
            json.dump({"columns": self.feature_columns}, f, indent=2)
        files.append(BUNDLE_FEATURES_FILE)
        checksums[BUNDLE_FEATURES_FILE] = self._file_checksum(features_path)

        # Save scaler
        if self.scaler is not None:
            scaler_path = path / BUNDLE_SCALER_FILE
            with open(scaler_path, "wb") as f:
                pickle.dump(self.scaler, f)
            files.append(BUNDLE_SCALER_FILE)
            checksums[BUNDLE_SCALER_FILE] = self._file_checksum(scaler_path)

        # Save calibrator
        if self.calibrator is not None:
            calibrator_path = path / BUNDLE_CALIBRATOR_FILE
            with open(calibrator_path, "wb") as f:
                pickle.dump(self.calibrator, f)
            files.append(BUNDLE_CALIBRATOR_FILE)
            checksums[BUNDLE_CALIBRATOR_FILE] = self._file_checksum(calibrator_path)

        # Save model
        model_dir = path / BUNDLE_MODEL_DIR
        self.model.save(model_dir)
        files.append(BUNDLE_MODEL_DIR)

        # Save manifest
        manifest = BundleManifest(
            version=BUNDLE_VERSION,
            files=files,
            checksums=checksums,
        )
        manifest_path = path / BUNDLE_MANIFEST_FILE
        with open(manifest_path, "w") as f:
            json.dump(manifest.to_dict(), f, indent=2)

        logger.info(
            f"Saved bundle: {self.metadata.model_name} (H{self.metadata.horizon}) "
            f"with {self.metadata.n_features} features to {path}"
        )

        return path

    @classmethod
    def load(cls, path: Union[str, Path]) -> "ModelBundle":
        """
        Load bundle from disk.

        Args:
            path: Path to bundle directory

        Returns:
            Loaded ModelBundle

        Raises:
            FileNotFoundError: If bundle doesn't exist
            ValueError: If bundle is corrupted or incompatible
        """
        path = Path(path)

        if not path.is_dir():
            raise FileNotFoundError(f"Bundle not found at {path}")

        # Load manifest
        manifest_path = path / BUNDLE_MANIFEST_FILE
        if not manifest_path.exists():
            raise ValueError(f"Invalid bundle: missing {BUNDLE_MANIFEST_FILE}")

        with open(manifest_path) as f:
            manifest = BundleManifest.from_dict(json.load(f))

        # Load metadata
        metadata_path = path / BUNDLE_METADATA_FILE
        with open(metadata_path) as f:
            metadata = BundleMetadata.from_dict(json.load(f))

        # Load feature columns
        features_path = path / BUNDLE_FEATURES_FILE
        with open(features_path) as f:
            feature_columns = json.load(f)["columns"]

        # Load scaler
        scaler = None
        scaler_path = path / BUNDLE_SCALER_FILE
        if scaler_path.exists():
            with open(scaler_path, "rb") as f:
                scaler = pickle.load(f)

        # Load calibrator
        calibrator = None
        calibrator_path = path / BUNDLE_CALIBRATOR_FILE
        if calibrator_path.exists():
            with open(calibrator_path, "rb") as f:
                calibrator = pickle.load(f)

        # Load model
        model_dir = path / BUNDLE_MODEL_DIR
        model = ModelRegistry.create(metadata.model_name)
        model.load(model_dir)

        logger.info(
            f"Loaded bundle: {metadata.model_name} (H{metadata.horizon}) "
            f"from {path}"
        )

        return cls(
            model=model,
            scaler=scaler,
            feature_columns=feature_columns,
            metadata=metadata,
            calibrator=calibrator,
        )

    def predict(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        calibrate: bool = True,
    ) -> PredictionOutput:
        """
        Make predictions using the bundled model.

        Args:
            X: Input features (DataFrame or array)
            calibrate: Whether to apply calibration (if calibrator exists)

        Returns:
            PredictionOutput with predictions and probabilities
        """
        # Convert to array and validate features
        X_array = self._prepare_input(X)

        # Apply scaling
        if self.scaler is not None:
            if self.metadata.requires_sequences:
                # For 3D sequences, reshape, scale, reshape back
                orig_shape = X_array.shape
                X_flat = X_array.reshape(-1, orig_shape[-1])
                X_scaled = self.scaler.transform(X_flat)
                X_array = X_scaled.reshape(orig_shape)
            else:
                X_array = self.scaler.transform(X_array)

        # Make predictions
        output = self.model.predict(X_array)

        # Apply calibration
        if calibrate and self.calibrator is not None:
            output = self._apply_calibration(output)

        return output

    def _prepare_input(
        self,
        X: Union[pd.DataFrame, np.ndarray],
    ) -> np.ndarray:
        """Prepare and validate input data."""
        if isinstance(X, pd.DataFrame):
            # Validate and reorder columns
            missing = set(self.feature_columns) - set(X.columns)
            if missing:
                raise ValueError(f"Missing features: {list(missing)[:10]}")

            X = X[self.feature_columns].values

        X = np.asarray(X, dtype=np.float32)

        # Validate shape
        if self.metadata.requires_sequences:
            if X.ndim != 3:
                raise ValueError(
                    f"Model requires 3D sequences, got shape {X.shape}"
                )
            if X.shape[2] != self.metadata.n_features:
                raise ValueError(
                    f"Expected {self.metadata.n_features} features, "
                    f"got {X.shape[2]}"
                )
        else:
            if X.ndim != 2:
                raise ValueError(f"Expected 2D array, got shape {X.shape}")
            if X.shape[1] != self.metadata.n_features:
                raise ValueError(
                    f"Expected {self.metadata.n_features} features, "
                    f"got {X.shape[1]}"
                )

        return X

    def _apply_calibration(self, output: PredictionOutput) -> PredictionOutput:
        """Apply probability calibration to predictions."""
        calibrated_probs = self.calibrator.calibrate(output.class_probabilities)

        return PredictionOutput(
            class_predictions=output.class_predictions,
            class_probabilities=calibrated_probs,
            confidence=np.max(calibrated_probs, axis=1),
            metadata={**output.metadata, "calibrated": True},
        )

    def validate(self) -> Dict[str, Any]:
        """
        Validate bundle integrity.

        Returns:
            Dict with validation results
        """
        issues = []

        # Check model is fitted
        if not getattr(self.model, "_is_fitted", False):
            issues.append("Model is not fitted")

        # Check scaler consistency
        if self.scaler is not None:
            scaler_features = getattr(self.scaler, "n_features_in_", None)
            if scaler_features and scaler_features != self.metadata.n_features:
                issues.append(
                    f"Scaler features ({scaler_features}) != "
                    f"metadata features ({self.metadata.n_features})"
                )

        # Check calibrator
        if self.metadata.has_calibrator and self.calibrator is None:
            issues.append("Metadata indicates calibrator but none found")

        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "metadata": self.metadata.to_dict(),
        }

    @staticmethod
    def _file_checksum(path: Path) -> str:
        """Compute MD5 checksum of a file."""
        hash_md5 = hashlib.md5()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

    def __repr__(self) -> str:
        return (
            f"ModelBundle(model={self.metadata.model_name}, "
            f"horizon={self.metadata.horizon}, "
            f"features={self.metadata.n_features}, "
            f"calibrated={self.metadata.has_calibrator})"
        )


__all__ = [
    "ModelBundle",
    "BundleMetadata",
    "BundleManifest",
    "BUNDLE_VERSION",
]
