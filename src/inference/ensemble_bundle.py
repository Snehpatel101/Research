"""
EnsembleBundle - Serializable container for stacking ensemble with meta-learner.

PHASE_5: Inference bundle for ensemble models.

Bundles:
- Meta-learner model
- Base model references (or embedded bundles)
- OOF alignment configuration
- Feature stacking configuration

Integrates with:
- src/models/ensemble/orchestrator.py - EnsembleResult (PHASE_4)
- src/inference/bundle.py - ModelBundle structure
- src/adapters/alignment.py - OOFAligner for heterogeneous models (PHASE_2)

Usage:
    # Create from EnsembleResult (after training)
    from src.models.ensemble import EnsembleOrchestrator, EnsembleResult
    from src.inference import EnsembleBundle

    orchestrator = EnsembleOrchestrator(config)
    ensemble_result = orchestrator.train(oof_predictions, y_train)

    bundle = EnsembleBundle.from_ensemble_result(
        ensemble_result,
        base_bundle_paths=[Path("./bundles/xgb"), Path("./bundles/lstm")],
        config=config,
    )
    bundle.save("./bundles/ensemble_h20")

    # Load and predict
    bundle = EnsembleBundle.load("./bundles/ensemble_h20")
    predictions = bundle.predict(base_predictions)

    # End-to-end prediction with base bundles
    predictions = bundle.predict_from_base_features(X_test)
"""

from __future__ import annotations

import json
import logging
import pickle
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from src.core import PipelineConfig
    from src.core.interfaces import PredictionResult
    from src.models.ensemble.orchestrator import EnsembleResult

logger = logging.getLogger(__name__)


# =============================================================================
# VERSION AND CONSTANTS
# =============================================================================

ENSEMBLE_BUNDLE_VERSION = "1.0.0"
ENSEMBLE_MANIFEST_FILE = "manifest.json"
ENSEMBLE_METADATA_FILE = "metadata.json"
ENSEMBLE_META_LEARNER_DIR = "meta_learner"
ENSEMBLE_STACKING_FEATURES_FILE = "stacking_features.json"
ENSEMBLE_BASE_BUNDLES_FILE = "base_bundles.json"
ENSEMBLE_SCALER_FILE = "scaler.pkl"
ENSEMBLE_ALIGNMENT_CONFIG_FILE = "alignment_config.json"


# =============================================================================
# DATA CLASSES
# =============================================================================


@dataclass
class EnsembleBundleMetadata:
    """Metadata for ensemble bundle."""

    version: str
    created_at: str
    meta_learner_name: str
    base_model_names: list[str]
    horizon: int
    n_base_models: int
    n_stacking_features: int
    symbol: str = ""
    coverage: float = 1.0
    alignment_offset: int = 0
    metrics: dict[str, float] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "version": self.version,
            "created_at": self.created_at,
            "meta_learner_name": self.meta_learner_name,
            "base_model_names": self.base_model_names,
            "horizon": self.horizon,
            "n_base_models": self.n_base_models,
            "n_stacking_features": self.n_stacking_features,
            "symbol": self.symbol,
            "coverage": self.coverage,
            "alignment_offset": self.alignment_offset,
            "metrics": self.metrics,
            "extra": self.extra,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EnsembleBundleMetadata:
        """Create from dictionary."""
        return cls(
            version=data["version"],
            created_at=data["created_at"],
            meta_learner_name=data["meta_learner_name"],
            base_model_names=data["base_model_names"],
            horizon=data["horizon"],
            n_base_models=data["n_base_models"],
            n_stacking_features=data["n_stacking_features"],
            symbol=data.get("symbol", ""),
            coverage=data.get("coverage", 1.0),
            alignment_offset=data.get("alignment_offset", 0),
            metrics=data.get("metrics", {}),
            extra=data.get("extra", {}),
        )


@dataclass
class AlignmentConfig:
    """Configuration for OOF alignment at inference time."""

    strategy: str = "intersection"
    n_classes: int = 3
    model_offsets: dict[str, int] = field(default_factory=dict)
    sequence_lengths: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "strategy": self.strategy,
            "n_classes": self.n_classes,
            "model_offsets": self.model_offsets,
            "sequence_lengths": self.sequence_lengths,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AlignmentConfig:
        """Create from dictionary."""
        return cls(
            strategy=data.get("strategy", "intersection"),
            n_classes=data.get("n_classes", 3),
            model_offsets=data.get("model_offsets", {}),
            sequence_lengths=data.get("sequence_lengths", {}),
        )


@dataclass
class EnsembleBundleManifest:
    """Manifest listing all files in ensemble bundle."""

    version: str
    files: list[str]
    checksums: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "version": self.version,
            "files": self.files,
            "checksums": self.checksums,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EnsembleBundleManifest:
        """Create from dictionary."""
        return cls(
            version=data["version"],
            files=data["files"],
            checksums=data.get("checksums", {}),
        )


# =============================================================================
# ENSEMBLE BUNDLE
# =============================================================================


class EnsembleBundle:
    """
    Serializable container for stacking ensemble with meta-learner.

    Contains:
    - Trained meta-learner
    - Base model bundle paths (or embedded)
    - OOF alignment configuration for heterogeneous models
    - Stacking feature configuration

    Bundles are saved as directories with a standardized structure:
        bundle_dir/
            manifest.json               # File listing
            metadata.json               # Ensemble metadata
            stacking_features.json      # Feature column names
            base_bundles.json           # Paths to base model bundles
            alignment_config.json       # OOF alignment configuration
            scaler.pkl                  # Fitted scaler (optional)
            meta_learner/               # Meta-learner artifacts

    Usage:
        # Create from EnsembleResult
        bundle = EnsembleBundle.from_ensemble_result(ensemble_result)
        bundle.save("./bundles/ensemble_h20")

        # Load and predict
        bundle = EnsembleBundle.load("./bundles/ensemble_h20")
        predictions = bundle.predict(base_predictions)

        # End-to-end prediction
        predictions = bundle.predict_from_base_features(X_test)
    """

    def __init__(
        self,
        meta_learner: Any,
        metadata: EnsembleBundleMetadata,
        base_bundle_paths: list[Path] | None = None,
        stacking_feature_names: list[str] | None = None,
        scaler: Any | None = None,
        alignment_config: AlignmentConfig | None = None,
    ) -> None:
        """
        Initialize EnsembleBundle.

        Args:
            meta_learner: Trained meta-learner model
            metadata: Bundle metadata
            base_bundle_paths: Paths to base model bundles
            stacking_feature_names: Names of stacking features
            scaler: Optional scaler for stacking features
            alignment_config: Configuration for OOF alignment
        """
        self.meta_learner = meta_learner
        self.metadata = metadata
        self.base_bundle_paths = base_bundle_paths or []
        self.stacking_feature_names = stacking_feature_names or []
        self.scaler = scaler
        self.alignment_config = alignment_config or AlignmentConfig()

        # Loaded base bundles (lazy load)
        self._base_bundles: dict[str, Any] = {}

    @classmethod
    def from_ensemble_result(
        cls,
        ensemble_result: EnsembleResult,
        base_bundle_paths: list[Path] | None = None,
        config: PipelineConfig | None = None,
        scaler: Any | None = None,
    ) -> EnsembleBundle:
        """
        Create bundle from PHASE_4 EnsembleResult.

        Args:
            ensemble_result: Result from EnsembleOrchestrator.train()
            base_bundle_paths: Paths to base model bundles
            config: Optional PipelineConfig
            scaler: Optional fitted scaler for stacking features

        Returns:
            EnsembleBundle ready for saving
        """
        # Extract meta-learner from orchestrator
        # The ensemble_result stores the trained meta-learner
        meta_learner = None

        # Try different attribute names for meta-learner
        if hasattr(ensemble_result, "_ensemble"):
            meta_learner = ensemble_result._ensemble
        elif hasattr(ensemble_result, "ensemble"):
            meta_learner = ensemble_result.ensemble

        if meta_learner is None:
            logger.warning(
                "Could not extract meta-learner from EnsembleResult. "
                "Ensure the orchestrator.ensemble property is accessible."
            )

        # Extract stacking feature names from aligned OOF
        stacking_feature_names: list[str] = []
        if ensemble_result.aligned_oof is not None:
            aligned = ensemble_result.aligned_oof
            if hasattr(aligned, "get_feature_names"):
                stacking_feature_names = aligned.get_feature_names()

        # Determine n_stacking_features
        n_stacking_features = len(stacking_feature_names)
        if n_stacking_features == 0 and ensemble_result.stacking_dataset is not None:
            # Get from stacking dataset shape
            stacking_data = ensemble_result.stacking_dataset.data
            # Exclude y_true and datetime columns
            feature_cols = [c for c in stacking_data.columns if c not in ("y_true", "datetime")]
            n_stacking_features = len(feature_cols)
            stacking_feature_names = feature_cols

        # Build alignment config from aligned_oof
        alignment_config = AlignmentConfig()
        if ensemble_result.aligned_oof is not None:
            aligned = ensemble_result.aligned_oof
            alignment_config.n_classes = getattr(aligned, "n_classes", 3)
            # Store coverage info for each model
            if hasattr(aligned, "coverage"):
                alignment_config.model_offsets = dict.fromkeys(ensemble_result.base_model_names, 0)

        # Extract horizon from config or default
        horizon = 20
        if config is not None and config.horizons:
            horizon = config.horizons[0]

        # Extract symbol from config
        symbol = ""
        if config is not None:
            symbol = getattr(config, "symbol", "")

        metadata = EnsembleBundleMetadata(
            version=ENSEMBLE_BUNDLE_VERSION,
            created_at=datetime.now().isoformat(),
            meta_learner_name=ensemble_result.meta_learner_name,
            base_model_names=ensemble_result.base_model_names,
            horizon=horizon,
            n_base_models=ensemble_result.n_base_models,
            n_stacking_features=n_stacking_features,
            symbol=symbol,
            coverage=ensemble_result.coverage,
            alignment_offset=ensemble_result.alignment_offset,
            metrics=ensemble_result.metrics,
        )

        return cls(
            meta_learner=meta_learner,
            metadata=metadata,
            base_bundle_paths=base_bundle_paths or [],
            stacking_feature_names=stacking_feature_names,
            scaler=scaler,
            alignment_config=alignment_config,
        )

    @classmethod
    def from_orchestrator(
        cls,
        orchestrator: Any,
        base_bundle_paths: list[Path] | None = None,
        scaler: Any | None = None,
    ) -> EnsembleBundle:
        """
        Create bundle directly from EnsembleOrchestrator after training.

        This is a convenience method that extracts both the result and
        the trained meta-learner from the orchestrator.

        Args:
            orchestrator: Trained EnsembleOrchestrator instance
            base_bundle_paths: Paths to base model bundles
            scaler: Optional fitted scaler for stacking features

        Returns:
            EnsembleBundle ready for saving

        Raises:
            ValueError: If orchestrator has not been trained
        """
        if not orchestrator.is_trained:
            raise ValueError("Orchestrator has not been trained. Call train() first.")

        result = orchestrator.result
        config = orchestrator.config

        # Create bundle from result
        bundle = cls.from_ensemble_result(
            ensemble_result=result,
            base_bundle_paths=base_bundle_paths,
            config=config,
            scaler=scaler,
        )

        # Override meta_learner with the actual trained model
        bundle.meta_learner = orchestrator.ensemble

        return bundle

    def save(self, path: str | Path, overwrite: bool = False) -> Path:
        """
        Save ensemble bundle to disk.

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

        files: list[str] = []
        checksums: dict[str, str] = {}

        # Save metadata
        metadata_path = path / ENSEMBLE_METADATA_FILE
        with open(metadata_path, "w") as f:
            json.dump(self.metadata.to_dict(), f, indent=2)
        files.append(ENSEMBLE_METADATA_FILE)

        # Save stacking feature names
        stacking_path = path / ENSEMBLE_STACKING_FEATURES_FILE
        with open(stacking_path, "w") as f:
            json.dump(
                {
                    "feature_names": self.stacking_feature_names,
                    "n_features": len(self.stacking_feature_names),
                },
                f,
                indent=2,
            )
        files.append(ENSEMBLE_STACKING_FEATURES_FILE)

        # Save base bundle paths (relative to ensemble bundle dir for portability)
        bundles_path = path / ENSEMBLE_BASE_BUNDLES_FILE
        relative_paths: list[str] = []
        for p in self.base_bundle_paths:
            try:
                relative_paths.append(str(Path(p).relative_to(path.parent)))
            except ValueError:
                # Path not relative to parent — store absolute as fallback
                relative_paths.append(str(p))
        with open(bundles_path, "w") as f:
            json.dump(
                {
                    "paths": relative_paths,
                    "model_names": self.metadata.base_model_names,
                },
                f,
                indent=2,
            )
        files.append(ENSEMBLE_BASE_BUNDLES_FILE)

        # Save alignment config
        alignment_path = path / ENSEMBLE_ALIGNMENT_CONFIG_FILE
        with open(alignment_path, "w") as f:
            json.dump(self.alignment_config.to_dict(), f, indent=2)
        files.append(ENSEMBLE_ALIGNMENT_CONFIG_FILE)

        # Save meta-learner
        if self.meta_learner is not None:
            meta_dir = path / ENSEMBLE_META_LEARNER_DIR
            if hasattr(self.meta_learner, "save"):
                # Use model's native save method
                self.meta_learner.save(meta_dir)
            else:
                # Fallback to pickle
                meta_dir.mkdir(parents=True, exist_ok=True)
                with open(meta_dir / "model.pkl", "wb") as f:
                    pickle.dump(self.meta_learner, f, protocol=pickle.HIGHEST_PROTOCOL)
            files.append(ENSEMBLE_META_LEARNER_DIR)

        # Save scaler if present
        if self.scaler is not None:
            scaler_path = path / ENSEMBLE_SCALER_FILE
            with open(scaler_path, "wb") as f:
                pickle.dump(self.scaler, f, protocol=pickle.HIGHEST_PROTOCOL)
            files.append(ENSEMBLE_SCALER_FILE)

        # Save manifest
        manifest = EnsembleBundleManifest(
            version=ENSEMBLE_BUNDLE_VERSION,
            files=files,
            checksums=checksums,
        )
        manifest_path = path / ENSEMBLE_MANIFEST_FILE
        with open(manifest_path, "w") as f:
            json.dump(manifest.to_dict(), f, indent=2)

        logger.info(
            f"Saved ensemble bundle: {self.metadata.meta_learner_name} "
            f"({self.metadata.n_base_models} base models) to {path}"
        )

        return path

    @classmethod
    def load(cls, path: str | Path) -> EnsembleBundle:
        """
        Load ensemble bundle from disk.

        Args:
            path: Path to bundle directory

        Returns:
            Loaded EnsembleBundle

        Raises:
            FileNotFoundError: If bundle doesn't exist
            ValueError: If bundle is corrupted or incompatible
        """
        path = Path(path)

        if not path.is_dir():
            raise FileNotFoundError(f"Bundle not found at {path}")

        # Load manifest
        manifest_path = path / ENSEMBLE_MANIFEST_FILE
        if not manifest_path.exists():
            raise ValueError(f"Invalid bundle: missing {ENSEMBLE_MANIFEST_FILE}")

        with open(manifest_path) as f:
            EnsembleBundleManifest.from_dict(json.load(f))

        # Load metadata
        metadata_path = path / ENSEMBLE_METADATA_FILE
        with open(metadata_path) as f:
            metadata = EnsembleBundleMetadata.from_dict(json.load(f))

        # Load stacking feature names
        stacking_feature_names: list[str] = []
        stacking_path = path / ENSEMBLE_STACKING_FEATURES_FILE
        if stacking_path.exists():
            with open(stacking_path) as f:
                stacking_feature_names = json.load(f).get("feature_names", [])

        # Load base bundle paths (resolve relative paths against parent dir)
        base_bundle_paths: list[Path] = []
        bundles_path = path / ENSEMBLE_BASE_BUNDLES_FILE
        if bundles_path.exists():
            with open(bundles_path) as f:
                raw_paths = json.load(f).get("paths", [])
            for p_str in raw_paths:
                p = Path(p_str)
                if not p.is_absolute():
                    # Relative path — resolve against ensemble bundle's parent
                    p = (path.parent / p).resolve()
                base_bundle_paths.append(p)

        # Load alignment config
        alignment_config = AlignmentConfig()
        alignment_path = path / ENSEMBLE_ALIGNMENT_CONFIG_FILE
        if alignment_path.exists():
            with open(alignment_path) as f:
                alignment_config = AlignmentConfig.from_dict(json.load(f))

        # Load meta-learner
        meta_learner = None
        meta_dir = path / ENSEMBLE_META_LEARNER_DIR

        if (meta_dir / "model.pkl").exists():
            # Load from pickle
            from src.core.utils.safe_pickle import safe_pickle_load

            meta_learner = safe_pickle_load(meta_dir / "model.pkl")
        elif meta_dir.exists():
            # Try loading via model interface
            try:
                from src.models.ensemble import get_meta_learner

                meta_learner = get_meta_learner(metadata.meta_learner_name)
                if hasattr(meta_learner, "load"):
                    meta_learner.load(meta_dir)
            except ImportError:
                logger.warning("Could not load meta-learner: get_meta_learner not available")

        # Load scaler
        scaler = None
        scaler_path = path / ENSEMBLE_SCALER_FILE
        if scaler_path.exists():
            from src.core.utils.safe_pickle import safe_pickle_load as _safe_load

            scaler = _safe_load(scaler_path)

        logger.info(
            f"Loaded ensemble bundle: {metadata.meta_learner_name} "
            f"({metadata.n_base_models} base models)"
        )

        return cls(
            meta_learner=meta_learner,
            metadata=metadata,
            base_bundle_paths=base_bundle_paths,
            stacking_feature_names=stacking_feature_names,
            scaler=scaler,
            alignment_config=alignment_config,
        )

    def predict(
        self,
        base_predictions: dict[str, np.ndarray],
        calibrate: bool = True,
    ) -> PredictionResult:
        """
        Make ensemble predictions from base model outputs.

        Args:
            base_predictions: Dict mapping model_name -> probability array.
                Each array should be shape (n_samples, n_classes).
            calibrate: Whether to apply calibration (reserved for future use)

        Returns:
            PredictionResult with class predictions and probabilities

        Raises:
            ValueError: If meta-learner not loaded or predictions invalid
        """
        if self.meta_learner is None:
            raise ValueError("Meta-learner not loaded. Load bundle first.")

        # Align and stack base predictions
        stacking_X = self._stack_predictions(base_predictions)

        # Apply scaling if available
        if self.scaler is not None:
            stacking_X = self.scaler.transform(stacking_X)

        # Predict with meta-learner
        output = self.meta_learner.predict(stacking_X)

        return output

    def predict_proba(
        self,
        base_predictions: dict[str, np.ndarray],
    ) -> np.ndarray:
        """
        Get probability predictions from base model outputs.

        Args:
            base_predictions: Dict mapping model_name -> probability array

        Returns:
            Probability array of shape (n_samples, n_classes)
        """
        output = self.predict(base_predictions, calibrate=False)
        result: np.ndarray = output.class_probabilities
        return result

    def predict_classes(
        self,
        base_predictions: dict[str, np.ndarray],
    ) -> np.ndarray:
        """
        Get class predictions from base model outputs.

        Args:
            base_predictions: Dict mapping model_name -> probability array

        Returns:
            Class prediction array of shape (n_samples,)
        """
        output = self.predict(base_predictions, calibrate=False)
        result: np.ndarray = output.class_predictions
        return result

    def predict_from_base_features(
        self,
        X: pd.DataFrame | np.ndarray,
        calibrate: bool = True,
    ) -> PredictionResult:
        """
        End-to-end prediction from base features.

        Loads base bundles, gets predictions, then combines with meta-learner.

        Args:
            X: Input features for base models
            calibrate: Whether to apply calibration

        Returns:
            PredictionResult with class predictions and probabilities

        Raises:
            ValueError: If base bundles not available
        """
        # Load base bundles if needed
        self._ensure_base_bundles_loaded()

        if not self._base_bundles:
            raise ValueError("No base bundles loaded. Ensure base_bundle_paths are valid.")

        # Get predictions from each base model
        base_predictions: dict[str, np.ndarray] = {}

        for model_name, bundle in self._base_bundles.items():
            output = bundle.predict(X, calibrate=False)
            base_predictions[model_name] = output.class_probabilities

        # Combine with meta-learner
        return self.predict(base_predictions, calibrate=calibrate)

    def predict_from_raw(
        self,
        raw_df: pd.DataFrame,
        calibrate: bool = True,
        skip_cleaning: bool = False,
    ) -> PredictionResult:
        """
        End-to-end prediction from raw OHLCV data.

        Loads each base ModelBundle, calls predict_from_raw on each,
        then combines via the meta-learner.

        Args:
            raw_df: DataFrame with raw OHLCV data.
            calibrate: Whether to apply calibration.
            skip_cleaning: If True, skip resampling step.

        Returns:
            PredictionResult with class predictions and probabilities.

        Raises:
            ValueError: If base bundles not available or not loaded.
        """
        self._ensure_base_bundles_loaded()

        if not self._base_bundles:
            raise ValueError(
                "No base bundles loaded. Ensure base_bundle_paths are valid "
                "and each bundle has a preprocessing graph."
            )

        base_predictions: dict[str, np.ndarray] = {}

        for model_name, bundle in self._base_bundles.items():
            if bundle.preprocessing_graph is None:
                logger.warning(
                    f"Base bundle '{model_name}' has no preprocessing graph, "
                    f"skipping predict_from_raw."
                )
                continue
            output = bundle.predict_from_raw(raw_df, calibrate=False, skip_cleaning=skip_cleaning)
            base_predictions[model_name] = output.class_probabilities

        if not base_predictions:
            raise ValueError(
                "No base bundles produced predictions. "
                "Ensure at least one base bundle has a preprocessing graph."
            )

        return self.predict(base_predictions, calibrate=calibrate)

    def _stack_predictions(
        self,
        base_predictions: dict[str, np.ndarray],
    ) -> np.ndarray:
        """
        Stack base predictions into meta-learner input.

        Uses OOFAligner for proper alignment of heterogeneous model predictions.

        Args:
            base_predictions: Dict mapping model_name -> probability array

        Returns:
            Stacked feature array ready for meta-learner
        """
        from src.core import OOFResult
        from src.data.adapters import OOFAligner

        # If predictions are already aligned (same length), simple stack
        lengths = [arr.shape[0] for arr in base_predictions.values()]
        if len(set(lengths)) == 1:
            # All same length - simple concatenation
            return self._simple_stack(base_predictions)

        # Different lengths - need alignment via OOFAligner
        oof_results: list[OOFResult] = []

        for model_name, probs in base_predictions.items():
            n_samples = probs.shape[0]
            indices = np.arange(n_samples)

            # Convert probabilities to class predictions
            predictions = np.argmax(probs, axis=1) - 1  # 0,1,2 -> -1,0,1

            oof_result = OOFResult(
                predictions=predictions,
                probabilities=probs,
                indices=indices,
                fold_ids=np.zeros(n_samples, dtype=int),
                model_name=model_name,
                coverage=1.0,
            )
            oof_results.append(oof_result)

        # Align predictions
        aligner = OOFAligner(n_classes=self.alignment_config.n_classes)
        aligned = aligner.align(
            oof_results,
            strategy=self.alignment_config.strategy,
        )

        return aligned.stacking_features

    def _simple_stack(
        self,
        base_predictions: dict[str, np.ndarray],
    ) -> np.ndarray:
        """
        Simple concatenation of aligned predictions.

        When all models have the same number of predictions,
        we can simply concatenate them.

        Args:
            base_predictions: Dict mapping model_name -> probability array

        Returns:
            Stacked feature array
        """
        # Ensure order matches metadata
        features = []
        for model_name in self.metadata.base_model_names:
            if model_name in base_predictions:
                features.append(base_predictions[model_name])
            else:
                logger.warning(f"Missing predictions for model: {model_name}")

        if not features:
            raise ValueError("No valid base predictions found")

        # Stack probabilities: (n_samples, n_models * n_classes)
        stacked = np.hstack(features)

        # Add derived features (confidence, agreement)
        n_samples = stacked.shape[0]
        n_models = len(features)
        n_classes = self.alignment_config.n_classes

        # Reshape to (n_samples, n_models, n_classes) for derived features
        probs_reshaped = stacked.reshape(n_samples, n_models, n_classes)

        # Mean confidence per sample
        confidences = np.max(probs_reshaped, axis=2)  # (n_samples, n_models)
        mean_confidence = np.mean(confidences, axis=1, keepdims=True)

        # Prediction agreement
        predictions = np.argmax(probs_reshaped, axis=2)  # (n_samples, n_models)
        agreement = np.zeros((n_samples, 1), dtype=np.float32)

        for i in range(n_samples):
            unique, counts = np.unique(predictions[i], return_counts=True)
            agreement[i] = counts.max() / n_models

        # Combine: probabilities + mean_confidence + agreement
        return np.hstack([stacked, mean_confidence, agreement])

    def _ensure_base_bundles_loaded(self) -> None:
        """Load base bundles if not already loaded."""
        if self._base_bundles:
            return

        from src.inference.bundle import ModelBundle

        for _i, bundle_path in enumerate(self.base_bundle_paths):
            if bundle_path.exists():
                try:
                    bundle = ModelBundle.load(bundle_path)
                    model_name = bundle.metadata.model_name

                    # Use model name from bundle metadata
                    self._base_bundles[model_name] = bundle
                    logger.debug(f"Loaded base bundle: {model_name}")
                except Exception as e:
                    logger.warning(f"Failed to load base bundle {bundle_path}: {e}")
            else:
                logger.warning(f"Base bundle not found: {bundle_path}")

    def validate(self) -> dict[str, Any]:
        """
        Validate bundle integrity.

        Returns:
            Dict with validation results
        """
        issues: list[str] = []

        # Check meta-learner
        if self.meta_learner is None:
            issues.append("Meta-learner not loaded")

        # Check base model count
        if self.metadata.n_base_models < 2:
            issues.append(
                f"Ensemble requires at least 2 base models, got {self.metadata.n_base_models}"
            )

        # Check feature names
        if not self.stacking_feature_names:
            issues.append("No stacking feature names defined")

        # Check base bundle paths
        missing_bundles = [str(p) for p in self.base_bundle_paths if not p.exists()]
        if missing_bundles:
            issues.append(f"Missing base bundles: {missing_bundles}")

        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "metadata": self.metadata.to_dict(),
            "n_base_bundles_available": len([p for p in self.base_bundle_paths if p.exists()]),
        }

    def summary(self) -> str:
        """
        Get human-readable summary of the bundle.

        Returns:
            Summary string
        """
        lines = [
            f"EnsembleBundle: {self.metadata.meta_learner_name}",
            f"  Version: {self.metadata.version}",
            f"  Created: {self.metadata.created_at}",
            f"  Horizon: {self.metadata.horizon}",
            f"  Symbol: {self.metadata.symbol or 'N/A'}",
            f"  Base models: {self.metadata.n_base_models}",
        ]

        # List base models
        for name in self.metadata.base_model_names[:5]:
            lines.append(f"    - {name}")
        if len(self.metadata.base_model_names) > 5:
            lines.append(f"    ... and {len(self.metadata.base_model_names) - 5} more")

        lines.extend(
            [
                f"  Stacking features: {self.metadata.n_stacking_features}",
                f"  Coverage: {self.metadata.coverage:.2%}",
                f"  Alignment offset: {self.metadata.alignment_offset}",
            ]
        )

        # Metrics
        if self.metadata.metrics:
            lines.append("  Metrics:")
            for key, value in self.metadata.metrics.items():
                if isinstance(value, float):
                    lines.append(f"    {key}: {value:.4f}")
                else:
                    lines.append(f"    {key}: {value}")

        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"EnsembleBundle(meta_learner={self.metadata.meta_learner_name}, "
            f"base_models={self.metadata.base_model_names}, "
            f"coverage={self.metadata.coverage:.2%})"
        )


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "EnsembleBundle",
    "EnsembleBundleMetadata",
    "EnsembleBundleManifest",
    "AlignmentConfig",
    "ENSEMBLE_BUNDLE_VERSION",
]
