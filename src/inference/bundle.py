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
import tarfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler, StandardScaler

from src.models.base import BaseModel, PredictionResult
from src.models.registry import ModelRegistry

logger = logging.getLogger(__name__)


# =============================================================================
# VERSION AND CONSTANTS
# =============================================================================

BUNDLE_VERSION = "1.3.0"  # Updated for ScalingSource + extended metadata fields
BUNDLE_MANIFEST_FILE = "manifest.json"
BUNDLE_MODEL_DIR = "model"
BUNDLE_SCALER_FILE = "scaler.pkl"
BUNDLE_CALIBRATOR_FILE = "calibrator.pkl"
BUNDLE_FEATURES_FILE = "features.json"
BUNDLE_METADATA_FILE = "metadata.json"
BUNDLE_PREPROCESSING_GRAPH_FILE = "preprocessing_graph.json"
BUNDLE_FEATURE_SPEC_FILE = "feature_spec.json"


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
    requires_4d: bool = False
    sequence_length: int = 0
    n_timeframes: int = 0
    has_calibrator: bool = False
    has_preprocessing_graph: bool = False
    preprocessing_graph_hash: str = ""
    has_feature_spec: bool = False
    feature_spec_hash: str = ""
    symbol: str = ""
    training_metrics: dict[str, Any] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)
    scaling_source: str = "bundle"
    scaler_type: str = ""
    feature_names: list[str] = field(default_factory=list)
    training_run_id: str = ""
    arch_version: str = ""
    label_mapping: dict[int, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "created_at": self.created_at,
            "model_name": self.model_name,
            "model_family": self.model_family,
            "horizon": self.horizon,
            "n_features": self.n_features,
            "feature_hash": self.feature_hash,
            "requires_sequences": self.requires_sequences,
            "requires_4d": self.requires_4d,
            "sequence_length": self.sequence_length,
            "n_timeframes": self.n_timeframes,
            "has_calibrator": self.has_calibrator,
            "has_preprocessing_graph": self.has_preprocessing_graph,
            "preprocessing_graph_hash": self.preprocessing_graph_hash,
            "has_feature_spec": self.has_feature_spec,
            "feature_spec_hash": self.feature_spec_hash,
            "symbol": self.symbol,
            "training_metrics": self.training_metrics,
            "extra": self.extra,
            "scaling_source": self.scaling_source,
            "scaler_type": self.scaler_type,
            "feature_names": self.feature_names,
            "training_run_id": self.training_run_id,
            "arch_version": self.arch_version,
            "label_mapping": {str(k): v for k, v in self.label_mapping.items()},
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BundleMetadata:
        return cls(
            version=data["version"],
            created_at=data["created_at"],
            model_name=data["model_name"],
            model_family=data.get("model_family", "unknown"),
            horizon=data["horizon"],
            n_features=data["n_features"],
            feature_hash=data["feature_hash"],
            requires_sequences=data.get("requires_sequences", False),
            requires_4d=data.get("requires_4d", False),
            sequence_length=data.get("sequence_length", 0),
            n_timeframes=data.get("n_timeframes", 0),
            has_calibrator=data.get("has_calibrator", False),
            has_preprocessing_graph=data.get("has_preprocessing_graph", False),
            preprocessing_graph_hash=data.get("preprocessing_graph_hash", ""),
            has_feature_spec=data.get("has_feature_spec", False),
            feature_spec_hash=data.get("feature_spec_hash", ""),
            symbol=data.get("symbol", ""),
            training_metrics=data.get("training_metrics", {}),
            extra=data.get("extra", {}),
            scaling_source=data.get("scaling_source", "bundle"),
            scaler_type=data.get("scaler_type", ""),
            feature_names=data.get("feature_names", []),
            training_run_id=data.get("training_run_id", ""),
            arch_version=data.get("arch_version", ""),
            label_mapping={int(k): v for k, v in data.get("label_mapping", {}).items()},
        )


@dataclass
class BundleManifest:
    """Manifest listing all files in a bundle."""

    version: str
    files: list[str]
    checksums: dict[str, str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "files": self.files,
            "checksums": self.checksums,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BundleManifest:
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
    - Preprocessing graph (optional, for train/serve parity)
    - Metadata about training

    Bundles are saved as directories with a standardized structure:
        bundle_dir/
            manifest.json               # File listing and checksums
            metadata.json               # Model metadata
            features.json               # Feature column names
            scaler.pkl                  # Fitted scaler
            calibrator.pkl              # Fitted calibrator (optional)
            preprocessing_graph.json    # Preprocessing config (optional)
            model/                      # Model artifacts (via model.save())

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

        >>> # With preprocessing graph for raw OHLCV inference
        >>> from src.inference import PreprocessingGraph
        >>> graph = PreprocessingGraph.from_pipeline_config(config)
        >>> bundle.set_preprocessing_graph(graph)
        >>> bundle.save("./bundles/xgb_h20_with_graph")
        >>>
        >>> # At inference time
        >>> bundle = ModelBundle.load("./bundles/xgb_h20_with_graph")
        >>> features = bundle.preprocess(raw_ohlcv_df)
        >>> predictions = bundle.predict(features)
    """

    def __init__(
        self,
        model: BaseModel,
        scaler: RobustScaler | StandardScaler | None,
        feature_columns: list[str],
        metadata: BundleMetadata,
        calibrator: Any | None = None,
        preprocessing_graph: Any | None = None,
        feature_spec: Any | None = None,
    ) -> None:
        """
        Initialize ModelBundle.

        Args:
            model: Trained model instance
            scaler: Fitted scaler (None for models that don't need scaling)
            feature_columns: Ordered list of feature column names
            metadata: Bundle metadata
            calibrator: Optional fitted probability calibrator
            preprocessing_graph: Optional PreprocessingGraph for raw data inference
            feature_spec: Optional FeatureSpec for 5-dimension optimization parity
        """
        self.model = model
        self.scaler = scaler
        self.feature_columns = feature_columns
        self.metadata = metadata
        self.calibrator = calibrator
        self.preprocessing_graph = preprocessing_graph
        self.feature_spec = feature_spec

    @classmethod
    def from_training(
        cls,
        model: BaseModel,
        scaler: RobustScaler | StandardScaler | None,
        feature_columns: list[str],
        horizon: int,
        calibrator: Any | None = None,
        preprocessing_graph: Any | None = None,
        feature_spec: Any | None = None,
        symbol: str = "",
        training_metrics: dict[str, Any] | None = None,
        extra_metadata: dict[str, Any] | None = None,
        model_name: str = "",
    ) -> ModelBundle:
        """
        Create a bundle from trained components.

        Args:
            model: Trained model
            scaler: Fitted scaler
            feature_columns: Feature column names
            horizon: Prediction horizon
            calibrator: Optional fitted calibrator
            preprocessing_graph: Optional PreprocessingGraph for train/serve parity
            feature_spec: Optional FeatureSpec for 5-dimension optimization parity.
                         Contains all optimization dimensions (triple barrier params,
                         selected features, feature params, timeframes, hyperparameters)
                         to ensure inference uses exact same configuration as training.
            symbol: Trading symbol (e.g., "MES", "MGC")
            training_metrics: Optional training metrics to store
            extra_metadata: Additional metadata
            model_name: Explicit model name (preferred over introspection)

        Returns:
            ModelBundle ready for saving
        """
        # Get model info — prefer explicit name, then introspection, then "unknown"
        if not model_name:
            model_name = getattr(model, "_get_model_type", lambda: "unknown")()
        model_family = getattr(model, "model_family", "unknown")
        requires_sequences = getattr(model, "requires_sequences", False)
        requires_4d = getattr(model, "requires_4d", False)
        sequence_length = 0
        if requires_sequences or requires_4d:
            sequence_length = getattr(model, "_config", {}).get("sequence_length", 60)

        # 4D models need extra shape metadata for validation
        n_timeframes = 0
        n_features = len(feature_columns)
        if requires_4d:
            try:
                from src.core.contracts import FeatureMode, get_model_contract

                contract = get_model_contract(model_name)
                if contract.input_rank.value == 4:
                    n_timeframes = 1 + len(contract.mtf_timeframes)
                    if contract.feature_mode == FeatureMode.RAW:
                        # RAW multi-stream models use OHLCV per timeframe
                        n_features = 5
                else:
                    # Dynamic 4D (e.g. ensembles) - skip strict feature validation
                    n_features = 0
            except Exception:
                # Conservative fallback - skip strict feature validation
                n_features = 0

        # Compute feature hash for validation
        feature_hash = hashlib.md5(",".join(feature_columns).encode()).hexdigest()[:12]

        # Get preprocessing graph hash if available
        preprocessing_graph_hash = ""
        if preprocessing_graph is not None:
            preprocessing_graph_hash = getattr(
                getattr(preprocessing_graph, "config", None),
                "config_hash",
                "",
            )

        # Get feature spec hash if available
        feature_spec_hash = ""
        if feature_spec is not None:
            feature_spec_hash = getattr(feature_spec, "schema_hash", "")

        metadata = BundleMetadata(
            version=BUNDLE_VERSION,
            created_at=datetime.now().isoformat(),
            model_name=model_name,
            model_family=model_family,
            horizon=horizon,
            n_features=n_features,
            feature_hash=feature_hash,
            requires_sequences=requires_sequences,
            requires_4d=requires_4d,
            sequence_length=sequence_length,
            n_timeframes=n_timeframes,
            has_calibrator=calibrator is not None,
            has_preprocessing_graph=preprocessing_graph is not None,
            preprocessing_graph_hash=preprocessing_graph_hash,
            has_feature_spec=feature_spec is not None,
            feature_spec_hash=feature_spec_hash,
            symbol=symbol,
            training_metrics=training_metrics or {},
            extra=extra_metadata or {},
            feature_names=list(feature_columns),
        )

        return cls(
            model=model,
            scaler=scaler,
            feature_columns=feature_columns,
            metadata=metadata,
            calibrator=calibrator,
            preprocessing_graph=preprocessing_graph,
            feature_spec=feature_spec,
        )

    def save(self, path: str | Path, overwrite: bool = False) -> Path:
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
                pickle.dump(self.scaler, f, protocol=pickle.HIGHEST_PROTOCOL)
            files.append(BUNDLE_SCALER_FILE)
            checksums[BUNDLE_SCALER_FILE] = self._file_checksum(scaler_path)

        # Save calibrator
        if self.calibrator is not None:
            calibrator_path = path / BUNDLE_CALIBRATOR_FILE
            with open(calibrator_path, "wb") as f:
                pickle.dump(self.calibrator, f, protocol=pickle.HIGHEST_PROTOCOL)
            files.append(BUNDLE_CALIBRATOR_FILE)
            checksums[BUNDLE_CALIBRATOR_FILE] = self._file_checksum(calibrator_path)

        # Save preprocessing graph
        if self.preprocessing_graph is not None:
            graph_path = path / BUNDLE_PREPROCESSING_GRAPH_FILE
            self.preprocessing_graph.save(graph_path)
            files.append(BUNDLE_PREPROCESSING_GRAPH_FILE)
            checksums[BUNDLE_PREPROCESSING_GRAPH_FILE] = self._file_checksum(graph_path)
            logger.info(f"Saved preprocessing graph to {graph_path}")

        # Save feature spec (5-dimension optimization)
        if self.feature_spec is not None:
            spec_path = path / BUNDLE_FEATURE_SPEC_FILE
            self.feature_spec.save(spec_path)
            files.append(BUNDLE_FEATURE_SPEC_FILE)
            checksums[BUNDLE_FEATURE_SPEC_FILE] = self._file_checksum(spec_path)
            logger.info(f"Saved feature spec to {spec_path}")

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

    def package_bundle(
        self,
        bundle_dir: str | Path,
        output_path: str | Path | None = None,
        compression: str = "gz",
    ) -> Path:
        """
        Package bundle directory into a single tarball for deployment.

        This creates a compressed archive (tar.gz by default) containing all
        bundle artifacts. Useful for deploying trained models to production
        environments or sharing between systems.

        Args:
            bundle_dir: Path to the saved bundle directory (from save())
            output_path: Optional path for output tarball. If not provided,
                        creates {bundle_dir}.tar.{compression} in parent directory
            compression: Compression format: 'gz' (default), 'bz2', 'xz', or '' (no compression)

        Returns:
            Path to created tarball

        Raises:
            FileNotFoundError: If bundle_dir doesn't exist
            ValueError: If bundle directory is invalid

        Example:
            >>> bundle = ModelBundle.from_training(...)
            >>> bundle_dir = bundle.save("./bundles/xgb_h20")
            >>> tarball = bundle.package_bundle(bundle_dir)
            >>> # Creates ./bundles/xgb_h20.tar.gz
            >>>
            >>> # Deploy to production
            >>> # scp xgb_h20.tar.gz production:/models/
            >>> # On production:
            >>> # tar -xzf xgb_h20.tar.gz && python -c "from src.inference import ModelBundle; bundle = ModelBundle.load('xgb_h20')"
        """
        bundle_dir = Path(bundle_dir)

        if not bundle_dir.is_dir():
            raise FileNotFoundError(f"Bundle directory not found: {bundle_dir}")

        # Verify bundle has required files
        manifest_path = bundle_dir / BUNDLE_MANIFEST_FILE
        if not manifest_path.exists():
            raise ValueError(
                f"Invalid bundle directory: missing {BUNDLE_MANIFEST_FILE}. "
                "Did you call save() first?"
            )

        # Determine output path
        if output_path is None:
            compression_ext = f".{compression}" if compression else ""
            output_path = bundle_dir.parent / f"{bundle_dir.name}.tar{compression_ext}"
        else:
            output_path = Path(output_path)

        # Create tarball
        mode = f"w:{compression}" if compression else "w"

        logger.info(f"Packaging bundle {bundle_dir} -> {output_path}")

        with tarfile.open(output_path, mode) as tar:
            # Add all files in bundle directory
            # Use arcname to preserve directory structure
            tar.add(bundle_dir, arcname=bundle_dir.name)

        tarball_size_mb = output_path.stat().st_size / (1024 * 1024)
        logger.info(f"Created deployment bundle: {output_path} ({tarball_size_mb:.2f} MB)")

        return output_path

    @classmethod
    def extract_bundle(
        cls,
        tarball_path: str | Path,
        extract_dir: str | Path | None = None,
    ) -> Path:
        """
        Extract a packaged bundle tarball.

        Complementary to package_bundle(). Extracts the tarball and returns
        the path to the extracted bundle directory, ready for load().

        Args:
            tarball_path: Path to the tarball created by package_bundle()
            extract_dir: Directory to extract to. If None, extracts to same
                        directory as tarball

        Returns:
            Path to extracted bundle directory

        Raises:
            FileNotFoundError: If tarball doesn't exist
            tarfile.TarError: If tarball is corrupted

        Example:
            >>> # On production server
            >>> from src.inference import ModelBundle
            >>> bundle_dir = ModelBundle.extract_bundle("xgb_h20.tar.gz")
            >>> bundle = ModelBundle.load(bundle_dir)
            >>> predictions = bundle.predict(X_new)
        """
        tarball_path = Path(tarball_path)

        if not tarball_path.exists():
            raise FileNotFoundError(f"Tarball not found: {tarball_path}")

        # Determine extraction directory
        if extract_dir is None:
            extract_dir = tarball_path.parent
        else:
            extract_dir = Path(extract_dir)
            extract_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Extracting bundle {tarball_path} -> {extract_dir}")

        # Extract tarball
        with tarfile.open(tarball_path, "r:*") as tar:
            # Security check: prevent path traversal attacks
            for member in tar.getmembers():
                if member.name.startswith("/") or ".." in member.name:
                    raise ValueError(
                        f"Unsafe path in tarball: {member.name}. "
                        "Bundle may be corrupted or malicious."
                    )
            # filter="data" rejects absolute paths, traversal, and special
            # files (Python 3.12+; becomes the default in 3.14)
            tar.extractall(extract_dir, filter="data")

        # Find the extracted bundle directory
        # Should be the first top-level directory in the tarball
        with tarfile.open(tarball_path, "r:*") as tar:
            first_member = tar.getmembers()[0]
            bundle_name = first_member.name.split("/")[0]

        bundle_dir = extract_dir / bundle_name

        logger.info(f"Extracted bundle to {bundle_dir}")

        return bundle_dir

    @classmethod
    def load(cls, path: str | Path) -> ModelBundle:
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
            BundleManifest.from_dict(json.load(f))

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
        from src.core.utils.safe_pickle import safe_pickle_load

        scaler_path = path / BUNDLE_SCALER_FILE
        if scaler_path.exists():
            scaler = safe_pickle_load(scaler_path)

        # Load calibrator
        calibrator = None
        calibrator_path = path / BUNDLE_CALIBRATOR_FILE
        if calibrator_path.exists():
            calibrator = safe_pickle_load(calibrator_path)

        # Load preprocessing graph
        preprocessing_graph = None
        graph_path = path / BUNDLE_PREPROCESSING_GRAPH_FILE
        if graph_path.exists():
            try:
                from src.inference.preprocessing_graph import PreprocessingGraph

                preprocessing_graph = PreprocessingGraph.load(graph_path)
                # Set the scaler on the preprocessing graph
                if preprocessing_graph is not None and scaler is not None:
                    preprocessing_graph.set_scaler(scaler)
                logger.info(f"Loaded preprocessing graph from {graph_path}")
            except ImportError:
                logger.warning("PreprocessingGraph module not available, skipping graph loading")

        # Load feature spec (5-dimension optimization)
        feature_spec = None
        spec_path = path / BUNDLE_FEATURE_SPEC_FILE
        if spec_path.exists():
            try:
                from src.core.contracts.feature_spec import FeatureSpec

                feature_spec = FeatureSpec.load(spec_path)
                logger.info(f"Loaded feature spec from {spec_path}")
            except ImportError:
                logger.warning("FeatureSpec module not available, skipping spec loading")
            except Exception as e:
                logger.warning(f"Failed to load feature spec: {e}")

        # Load model
        model_dir = path / BUNDLE_MODEL_DIR
        model = ModelRegistry.create(metadata.model_name)
        model.load(model_dir)

        logger.info(f"Loaded bundle: {metadata.model_name} (H{metadata.horizon}) from {path}")

        return cls(
            model=model,
            scaler=scaler,
            feature_columns=feature_columns,
            metadata=metadata,
            calibrator=calibrator,
            preprocessing_graph=preprocessing_graph,
            feature_spec=feature_spec,
        )

    def predict(
        self,
        X: pd.DataFrame | np.ndarray,
        calibrate: bool = True,
    ) -> PredictionResult:
        """
        Make predictions using the bundled model.

        Args:
            X: Input features (DataFrame or array)
            calibrate: Whether to apply calibration (if calibrator exists)

        Returns:
            PredictionResult with predictions and probabilities
        """
        # Convert to array and validate features
        X_array = self._prepare_input(X)

        # Apply scaling
        if self.scaler is not None:
            if self.metadata.requires_4d:
                # For 4D sequences, reshape, scale, reshape back
                orig_shape = X_array.shape
                X_flat = X_array.reshape(-1, orig_shape[-1])
                scaler_features = getattr(self.scaler, "n_features_in_", None)
                if scaler_features and scaler_features != orig_shape[-1]:
                    raise ValueError(
                        f"Scaler expects {scaler_features} features but 4D input has "
                        f"{orig_shape[-1]} features per timeframe."
                    )
                X_scaled = self.scaler.transform(X_flat)
                X_array = X_scaled.reshape(orig_shape)
            elif self.metadata.requires_sequences:
                # For 3D sequences, reshape, scale, reshape back
                orig_shape = X_array.shape
                X_flat = X_array.reshape(-1, orig_shape[-1])
                scaler_features = getattr(self.scaler, "n_features_in_", None)
                if scaler_features and scaler_features != orig_shape[-1]:
                    raise ValueError(
                        f"Scaler expects {scaler_features} features but sequence input has "
                        f"{orig_shape[-1]} features."
                    )
                X_scaled = self.scaler.transform(X_flat)
                X_array = X_scaled.reshape(orig_shape)
            else:
                scaler_features = getattr(self.scaler, "n_features_in_", None)
                if scaler_features and scaler_features != X_array.shape[1]:
                    raise ValueError(
                        f"Scaler expects {scaler_features} features but tabular input has "
                        f"{X_array.shape[1]} features."
                    )
                X_array = self.scaler.transform(X_array)

        # Make predictions
        output = self.model.predict(X_array)

        # Apply calibration
        if calibrate and self.calibrator is not None:
            output = self._apply_calibration(output)

        return output

    def _prepare_input(
        self,
        X: pd.DataFrame | np.ndarray,
    ) -> np.ndarray:
        """Prepare and validate input data."""
        if isinstance(X, pd.DataFrame):
            if self.metadata.requires_4d:
                raise ValueError(
                    "4D models require ndarray input of shape "
                    "(n_samples, n_timeframes, seq_len, n_features). "
                    "Pass a 4D tensor produced by container.get_multi_resolution_4d() "
                    "or container.get_multi_stream_4d()."
                )

            # Validate and reorder columns
            missing = set(self.feature_columns) - set(X.columns)
            if missing:
                raise ValueError(f"Missing features: {list(missing)[:10]}")

            X = X[self.feature_columns].values

        X = np.asarray(X, dtype=np.float32)

        # Validate shape
        if self.metadata.requires_4d:
            if X.ndim != 4:
                raise ValueError(f"Model requires 4D input, got shape {X.shape}")
            if self.metadata.n_timeframes and X.shape[1] != self.metadata.n_timeframes:
                raise ValueError(
                    f"Expected {self.metadata.n_timeframes} timeframes, got {X.shape[1]}"
                )
            if self.metadata.sequence_length and X.shape[2] != self.metadata.sequence_length:
                raise ValueError(
                    f"Expected sequence_length={self.metadata.sequence_length}, got {X.shape[2]}"
                )
            if self.metadata.n_features and X.shape[3] != self.metadata.n_features:
                raise ValueError(
                    f"Expected {self.metadata.n_features} features per timeframe, got {X.shape[3]}"
                )
        elif self.metadata.requires_sequences:
            if X.ndim != 3:
                raise ValueError(f"Model requires 3D sequences, got shape {X.shape}")
            if X.shape[2] != self.metadata.n_features:
                raise ValueError(
                    f"Expected {self.metadata.n_features} features, " f"got {X.shape[2]}"
                )
        else:
            if X.ndim != 2:
                raise ValueError(f"Expected 2D array, got shape {X.shape}")
            if X.shape[1] != self.metadata.n_features:
                raise ValueError(
                    f"Expected {self.metadata.n_features} features, " f"got {X.shape[1]}"
                )

        return X

    def _apply_calibration(self, output: PredictionResult) -> PredictionResult:
        """Apply probability calibration to predictions."""
        if self.calibrator is None:
            return output
        calibrated_probs = self.calibrator.calibrate(output.class_probabilities)

        return PredictionResult(
            class_predictions=output.class_predictions,
            class_probabilities=calibrated_probs,
            confidence=np.max(calibrated_probs, axis=1),
            metadata={**output.metadata, "calibrated": True},
        )

    def validate(self) -> dict[str, Any]:
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
            if (
                self.metadata.n_features
                and scaler_features
                and scaler_features != self.metadata.n_features
            ):
                issues.append(
                    f"Scaler features ({scaler_features}) != "
                    f"metadata features ({self.metadata.n_features})"
                )

        # Check calibrator
        if self.metadata.has_calibrator and self.calibrator is None:
            issues.append("Metadata indicates calibrator but none found")

        # Check feature spec
        if self.metadata.has_feature_spec and self.feature_spec is None:
            issues.append("Metadata indicates feature_spec but none found")

        # Validate feature spec if present
        if self.feature_spec is not None:
            is_valid, spec_issues = self.feature_spec.validate()
            if not is_valid:
                issues.extend([f"feature_spec: {i}" for i in spec_issues])

        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "metadata": self.metadata.to_dict(),
        }

    def set_feature_spec(self, feature_spec: Any) -> None:
        """
        Set or update the feature spec.

        The FeatureSpec captures all 5 optimization dimensions used during training,
        ensuring inference parity with the exact same configuration.

        Args:
            feature_spec: FeatureSpec instance from src.core.contracts
        """
        self.feature_spec = feature_spec

        # Update metadata
        self.metadata.has_feature_spec = True
        self.metadata.feature_spec_hash = getattr(feature_spec, "schema_hash", "")

        logger.info(f"Set feature spec (hash: {self.metadata.feature_spec_hash})")

    def set_preprocessing_graph(self, graph: Any) -> None:
        """
        Set or update the preprocessing graph.

        Args:
            graph: PreprocessingGraph instance
        """
        self.preprocessing_graph = graph

        # Update metadata
        self.metadata.has_preprocessing_graph = True
        self.metadata.preprocessing_graph_hash = getattr(
            getattr(graph, "config", None),
            "config_hash",
            "",
        )

        # Set the scaler on the graph
        if self.scaler is not None:
            graph.set_scaler(self.scaler)

        logger.info(f"Set preprocessing graph (hash: {self.metadata.preprocessing_graph_hash})")

    def preprocess(
        self,
        raw_df: pd.DataFrame,
        skip_cleaning: bool = False,
    ) -> pd.DataFrame:
        """
        Apply preprocessing to raw OHLCV data.

        Uses the bundled preprocessing graph to transform raw data into
        features suitable for model prediction. This ensures train/serve
        parity by applying the exact same preprocessing as during training.

        Args:
            raw_df: DataFrame with raw OHLCV data. Must have columns:
                   [datetime, open, high, low, close, volume]
            skip_cleaning: If True, skip resampling (data already at target timeframe)

        Returns:
            DataFrame with features ready for model prediction

        Raises:
            RuntimeError: If no preprocessing graph is available
        """
        if self.preprocessing_graph is None:
            raise RuntimeError(
                "No preprocessing graph available. Either load a bundle with "
                "a preprocessing graph or call set_preprocessing_graph() first."
            )

        # Apply preprocessing (skip_scaling=True because predict() applies
        # the bundle's own scaler — avoids double-scaling)
        features = self.preprocessing_graph.transform(
            raw_df,
            skip_cleaning=skip_cleaning,
            skip_scaling=True,
        )

        # Ensure feature columns match
        available_cols = [c for c in self.feature_columns if c in features.columns]
        if len(available_cols) != len(self.feature_columns):
            missing = set(self.feature_columns) - set(available_cols)
            logger.warning(
                f"Preprocessing generated {len(features.columns)} columns, "
                f"but model expects {len(self.feature_columns)}. "
                f"Missing {len(missing)} columns: {list(missing)[:5]}..."
            )

        return features[available_cols]

    def predict_from_raw(
        self,
        raw_df: pd.DataFrame,
        calibrate: bool = True,
        skip_cleaning: bool = False,
        additional_dfs: dict[str, pd.DataFrame] | None = None,
    ) -> PredictionResult:
        """
        End-to-end prediction from raw OHLCV data.

        Routes data through the correct adapter based on the model's
        input rank:
        - 2D (tabular): preprocess → predict
        - 3D (sequence): preprocess → _build_3d_input → predict
        - 4D (multi-stream): _build_4d_input (raw multi-TF) → predict

        Scaling invariant: preprocess() returns unscaled features
        (skip_scaling=True), then predict() applies the bundle scaler
        exactly once. This guarantees single-scaling regardless of path.

        Args:
            raw_df: DataFrame with raw OHLCV data
            calibrate: Whether to apply probability calibration
            skip_cleaning: If True, skip resampling step
            additional_dfs: Extra timeframe DataFrames for 4D models.
                Maps timeframe string (e.g. "5min") to DataFrame.

        Returns:
            PredictionResult with predictions and probabilities
        """
        if self.metadata.requires_4d:
            # Auto-generate MTF DataFrames when not provided
            if additional_dfs is None:
                additional_dfs = self._generate_mtf_dataframes(raw_df)
            X = self._apply_adapter(
                features_2d=pd.DataFrame(),
                raw_df=raw_df,
                additional_dfs=additional_dfs,
                skip_cleaning=skip_cleaning,
            )
        else:
            features = self.preprocess(raw_df, skip_cleaning=skip_cleaning)
            X = self._apply_adapter(features_2d=features)

        return self.predict(X, calibrate=calibrate)

    # -----------------------------------------------------------------
    # Adapter routing helpers
    # -----------------------------------------------------------------

    def _apply_adapter(
        self,
        features_2d: pd.DataFrame,
        raw_df: pd.DataFrame | None = None,
        additional_dfs: dict[str, pd.DataFrame] | None = None,
        skip_cleaning: bool = False,
    ) -> np.ndarray | pd.DataFrame:
        """Route features through the correct adapter based on model input rank.

        Centralises the 2D → 3D → 4D routing that ``predict_from_raw``
        needs so callers don't have to repeat the branching logic.

        Args:
            features_2d: 2D feature DataFrame from preprocess().
                Ignored for 4D models (which build input from raw data).
            raw_df: Raw OHLCV DataFrame, required for 4D models.
            additional_dfs: Extra timeframe DataFrames for 4D models.
            skip_cleaning: Whether to skip resampling in 4D path.

        Returns:
            Adapted input ready for predict():
            - np.ndarray of shape (N, TF, T, F) for 4D models
            - np.ndarray of shape (N, T, F) for 3D sequence models
            - pd.DataFrame for 2D tabular models
        """
        if self.metadata.requires_4d:
            if raw_df is None:
                from src.inference.errors import AdapterRoutingError

                raise AdapterRoutingError(
                    model_name=self.metadata.model_name,
                    required_rank=4,
                    reason="raw_df is required for 4D adapter routing.",
                )
            return self._build_4d_input(raw_df, additional_dfs, skip_cleaning)

        if self.metadata.requires_sequences:
            return self._build_3d_input(features_2d)

        # Tabular 2D — pass through unchanged
        return features_2d

    def _generate_mtf_dataframes(
        self,
        raw_1min_df: pd.DataFrame,
    ) -> dict[str, pd.DataFrame]:
        """Resample 1-minute OHLCV data to multiple timeframes.

        Generates the ``additional_dfs`` dict that ``_build_4d_input``
        requires, using standard OHLCV aggregation rules.

        Args:
            raw_1min_df: Raw 1-minute OHLCV DataFrame.  Must have a
                DatetimeIndex (or a ``datetime`` column that can be
                converted) and columns: open, high, low, close, volume.

        Returns:
            Dict mapping timeframe strings (e.g. ``"5min"``) to
            resampled DataFrames.

        Raises:
            ValueError: If ``mtf_timeframes`` is not set in bundle
                metadata or if required OHLCV columns are missing.
        """
        mtf_timeframes: list[str] = self.metadata.extra.get("mtf_timeframes", [])
        if not mtf_timeframes:
            raise ValueError(
                f"Cannot auto-generate MTF DataFrames for model "
                f"'{self.metadata.model_name}': metadata.extra['mtf_timeframes'] "
                f"is not set. Either pass additional_dfs explicitly or save "
                f"mtf_timeframes in the bundle's extra metadata."
            )

        # Ensure we have a DatetimeIndex for resample()
        df = raw_1min_df.copy()
        if not isinstance(df.index, pd.DatetimeIndex):
            if "datetime" in df.columns:
                df = df.set_index("datetime")
            else:
                raise ValueError("raw_1min_df must have a DatetimeIndex or a 'datetime' column.")

        required_cols = {"open", "high", "low", "close", "volume"}
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"raw_1min_df is missing OHLCV columns: {sorted(missing)}")

        ohlcv_agg = {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }

        result: dict[str, pd.DataFrame] = {}
        for tf in mtf_timeframes:
            resampled = df.resample(tf, closed="left", label="left").agg(ohlcv_agg).dropna()
            # Anti-lookahead: shift(1) ensures bar N only sees the COMPLETED
            # higher-TF bar (bar N-1), matching factory.py training path.
            resampled = resampled.shift(1).dropna()
            result[tf] = resampled
            logger.debug(
                f"_generate_mtf_dataframes: resampled 1min → {tf} " f"({len(resampled)} bars)"
            )

        return result

    def _build_3d_input(self, features_2d: pd.DataFrame) -> np.ndarray:
        """Build 3D sequence input from 2D feature DataFrame.

        Creates sliding windows of shape (n_samples, seq_len, n_features)
        using the bundle's stored sequence_length.

        Args:
            features_2d: 2D feature DataFrame from preprocess().

        Returns:
            3D numpy array ready for predict().

        Raises:
            ShapeMismatchError: If insufficient rows for sequence_length.
        """
        from src.inference.errors import ShapeMismatchError

        seq_len = self.metadata.sequence_length
        if seq_len <= 0:
            seq_len = 60  # Sensible default

        values = features_2d.values.astype(np.float32)
        n_rows, n_features = values.shape

        if n_rows < seq_len:
            raise ShapeMismatchError(
                expected_rank=3,
                actual_shape=values.shape,
                model_name=self.metadata.model_name,
                context=(
                    f"Need at least {seq_len} rows for sequence_length={seq_len}, " f"got {n_rows}."
                ),
            )

        # Sliding window via stride_tricks (same logic as SequenceAdapter)
        windows = np.lib.stride_tricks.sliding_window_view(values, seq_len, axis=0)
        # windows shape: (n_rows - seq_len + 1, n_features, seq_len)
        # Transpose to (n_sequences, seq_len, n_features)
        X_3d = windows.transpose(0, 2, 1).copy()

        logger.debug(
            f"_build_3d_input: {n_rows} rows → {X_3d.shape[0]} sequences "
            f"(seq_len={seq_len}, features={n_features})"
        )
        return X_3d

    def _build_4d_input(
        self,
        raw_df: pd.DataFrame,
        additional_dfs: dict[str, pd.DataFrame] | None,
        skip_cleaning: bool,
    ) -> np.ndarray:
        """Build 4D multi-stream input for multi-timeframe models.

        Uses the MultiStreamAdapter to convert raw OHLCV DataFrames
        from multiple timeframes into (N, n_tf, seq_len, n_features).

        Args:
            raw_df: Anchor timeframe DataFrame (smallest TF).
            additional_dfs: Other timeframe DataFrames keyed by TF string.
            skip_cleaning: Whether to skip resampling.

        Returns:
            4D numpy array ready for predict().

        Raises:
            AdapterRoutingError: If multi-timeframe data is missing.
        """
        from src.inference.errors import AdapterRoutingError

        if additional_dfs is None:
            raise AdapterRoutingError(
                model_name=self.metadata.model_name,
                required_rank=4,
                reason=(
                    "4D models require additional_dfs mapping timeframe "
                    "strings to DataFrames (e.g. {'5min': df_5, '15min': df_15})."
                ),
            )

        from src.data.adapters import MultiStreamAdapter

        seq_len = self.metadata.sequence_length or 60
        n_timeframes = self.metadata.n_timeframes or (1 + len(additional_dfs))

        # Determine timeframes from additional_dfs keys
        # Anchor is inferred as the smallest timeframe (not in additional_dfs)
        timeframes = list(additional_dfs.keys())
        # Prepend an anchor placeholder — the adapter expects the anchor first
        # We assume anchor = first timeframe the user omitted (i.e. "1min")
        anchor_tf = "1min"
        if timeframes and anchor_tf not in timeframes:
            timeframes = [anchor_tf] + timeframes

        adapter = MultiStreamAdapter(
            sequence_length=seq_len,
            timeframes=timeframes,
            label_column="label_h" + str(self.metadata.horizon),
        )

        try:
            result = adapter.transform(raw_df, additional_dfs=additional_dfs)
        except Exception as e:
            raise AdapterRoutingError(
                model_name=self.metadata.model_name,
                required_rank=4,
                reason=f"MultiStreamAdapter failed: {e}",
            ) from e

        logger.debug(
            f"_build_4d_input: {result.X.shape} " f"({n_timeframes} timeframes, seq_len={seq_len})"
        )
        return result.X

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
            f"calibrated={self.metadata.has_calibrator}, "
            f"has_feature_spec={self.metadata.has_feature_spec}, "
            f"has_preprocessing_graph={self.metadata.has_preprocessing_graph})"
        )


__all__ = [
    "ModelBundle",
    "BundleMetadata",
    "BundleManifest",
    "BUNDLE_VERSION",
    "BUNDLE_PREPROCESSING_GRAPH_FILE",
    "BUNDLE_FEATURE_SPEC_FILE",
]
