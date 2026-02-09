"""
TrainerArtifactsMixin - Artifact saving functionality.

Contains methods for:
- Saving training configuration
- Saving environment information
- Saving model requirements
- Saving training artifacts (metrics, predictions)
- Saving trained model
- Saving feature selection results
- Saving probability calibrator
- Generating artifact checksums for integrity verification
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np

from ..base import PredictionOutput, TrainingMetrics
from ..config import detect_environment, get_applied_overrides, save_config_json
from .checksums import ArtifactIntegrityManager

# Import MODEL_DATA_REQUIREMENTS for model requirements saving (MOD-008)
# Canonical location: src.models.config.data_requirements
try:
    from src.models.config import MODEL_DATA_REQUIREMENTS
    from src.models.config.data_requirements import ModelDataRequirements
except ImportError:
    MODEL_DATA_REQUIREMENTS: dict[str, Any] | None = None  # type: ignore[no-redef]


class _TrainerProtocol(Protocol):
    """Protocol defining the interface expected by TrainerArtifactsMixin."""

    config: Any  # TrainerConfig
    model: Any  # Model instance
    run_id: str
    output_path: Path
    feature_selector: Any | None
    calibrator: Any | None

    def _is_feature_selection_enabled(self) -> bool: ...


if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class TrainerArtifactsMixin:
    """
    Mixin providing artifact saving capabilities.

    This mixin assumes the following attributes exist on the class:
    - self.config: TrainerConfig with training settings
    - self.model: Instantiated model from registry
    - self.run_id: Unique run identifier
    - self.output_path: Path to output directory
    - self.feature_selector: FeatureSelectionManager instance
    - self.calibrator: ProbabilityCalibrator instance (optional)
    - self._is_feature_selection_enabled(): Method to check if feature selection is enabled
    """

    # Type hints for mixin attributes (provided by the class this is mixed into)
    config: Any  # TrainerConfig
    model: Any  # Model instance
    run_id: str
    output_path: Path
    feature_selector: Any | None
    calibrator: Any | None

    def _is_feature_selection_enabled(self) -> bool:
        """Check if feature selection is enabled. Must be implemented by subclass."""
        raise NotImplementedError

    def _save_config(self) -> None:
        """Save training configuration and environment information."""
        config_path = self.output_path / "config" / "training_config.json"
        save_config_json(self.config.to_dict(), config_path)

        model_config_path = self.output_path / "config" / "model_config.json"
        save_config_json(self.config.model_config, model_config_path)

        # Save environment information for reproducibility (CFG-008)
        self._save_environment_info()

        # Save model requirements for reproducibility (MOD-008)
        self._save_model_requirements()

    def _save_environment_info(self) -> None:
        """
        Save environment and override information for reproducibility.

        Creates environment_info.json containing:
        - Detected environment (gpu/cpu/colab)
        - Applied configuration overrides from each source
        - Timestamp of run

        This enables debugging which configuration sources affected the run
        and helps reproduce experiments across different environments.
        """
        env = detect_environment()
        applied_overrides = get_applied_overrides()

        env_info = {
            "environment": env.value,
            "timestamp": datetime.now().isoformat(),
            "run_id": self.run_id,
            "model_name": self.config.model_name,
            "device_resolved": self.config.get_resolved_device(),
            "applied_overrides": applied_overrides,
        }

        env_info_path = self.output_path / "config" / "environment_info.json"
        with open(env_info_path, "w") as f:
            json.dump(env_info, f, indent=2)

        # Log environment overrides at training start
        if applied_overrides.get("environment_overrides"):
            logger.info(
                f"Environment '{env.value}' overrides applied: "
                f"{list(applied_overrides['environment_overrides'].keys())}"
            )
        else:
            logger.debug(f"Running in '{env.value}' environment (no specific overrides)")

    def _save_model_requirements(self) -> None:
        """
        Save model requirements to artifacts for reproducibility (MOD-008).

        Creates model_requirements.json containing:
        - The MODEL_DATA_REQUIREMENTS entry for this model
        - Which feature_set was recommended vs configured
        - Sequence length required vs configured

        This enables debugging model configuration mismatches and helps
        reproduce experiments with correct data preparation settings.
        """
        if MODEL_DATA_REQUIREMENTS is None:
            logger.debug("MODEL_DATA_REQUIREMENTS not available, skipping save")
            return

        model_name = self.config.model_name.lower()

        # Build requirements info
        requirements_info: dict[str, Any] = {
            "model_name": self.config.model_name,
            "timestamp": datetime.now().isoformat(),
            "run_id": self.run_id,
        }

        # Add MODEL_DATA_REQUIREMENTS entry if available
        if model_name in MODEL_DATA_REQUIREMENTS:
            req = MODEL_DATA_REQUIREMENTS[model_name]
            requirements_info["model_data_requirements"] = {
                "family": req.family.value if hasattr(req.family, "value") else str(req.family),
                "feature_set_recommended": req.feature_set,
                "requires_scaling": req.requires_scaling,
                "scaler_type": (
                    req.scaler_type.value
                    if hasattr(req.scaler_type, "value")
                    else str(req.scaler_type)
                ),
                "requires_sequences": req.requires_sequences,
                "sequence_length_default": req.sequence_length,
                "max_features": req.max_features,
                "supports_categorical": req.supports_categorical,
                "supports_missing": req.supports_missing,
                "description": req.description,
            }

            # Compare recommended vs configured
            requirements_info["configuration_comparison"] = {
                "feature_set": {
                    "recommended": req.feature_set,
                    "configured": self.config.feature_set or "(auto-resolved)",
                    "match": (not self.config.feature_set)
                    or (self.config.feature_set == req.feature_set),
                },
                "sequence_length": {
                    "default": req.sequence_length,
                    "configured": self.config.sequence_length,
                    "match": self.config.sequence_length == req.sequence_length,
                },
            }
        else:
            requirements_info["model_data_requirements"] = None
            requirements_info["note"] = f"Model '{model_name}' not found in MODEL_DATA_REQUIREMENTS"

        # Add runtime model properties
        requirements_info["runtime_properties"] = {
            "model_family": self.model.model_family,
            "requires_sequences": self.model.requires_sequences,
            "requires_4d": getattr(self.model, "requires_4d", False),
            "requires_scaling": self.model.requires_scaling,
        }

        # Save to file
        requirements_path = self.output_path / "config" / "model_requirements.json"
        with open(requirements_path, "w") as f:
            json.dump(requirements_info, f, indent=2)

        logger.debug(f"Saved model requirements to {requirements_path}")

    def _save_artifacts(
        self,
        training_metrics: TrainingMetrics,
        eval_metrics: dict[str, Any],
        predictions: PredictionOutput,
        test_metrics: dict[str, Any] | None = None,
        test_predictions: PredictionOutput | None = None,
        lineage_validated: bool = False,
        lineage_issues: list[str] | None = None,
    ) -> None:
        """Save training artifacts."""
        # Training metrics
        metrics_path = self.output_path / "metrics" / "training_metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(training_metrics.to_dict(), f, indent=2)

        # Add lineage validation results to eval metrics
        if lineage_issues is None:
            lineage_issues = []
        eval_metrics_with_lineage = {
            **eval_metrics,
            "lineage_validated": lineage_validated,
            "lineage_issues": lineage_issues,
        }

        # Evaluation metrics (validation)
        eval_path = self.output_path / "metrics" / "evaluation_metrics.json"
        with open(eval_path, "w") as f:
            json.dump(eval_metrics_with_lineage, f, indent=2)

        # Validation predictions
        pred_path = self.output_path / "predictions" / "val_predictions.npz"
        np.savez(
            pred_path,
            class_predictions=predictions.class_predictions,
            class_probabilities=predictions.class_probabilities,
            confidence=predictions.confidence,
        )

        # Test set metrics and predictions (if evaluated)
        if test_metrics is not None:
            test_metrics_path = self.output_path / "metrics" / "test_metrics.json"
            with open(test_metrics_path, "w") as f:
                json.dump(test_metrics, f, indent=2)
            logger.info(f"Saved test metrics to {test_metrics_path}")

        if test_predictions is not None:
            test_pred_path = self.output_path / "predictions" / "test_predictions.npz"
            np.savez(
                test_pred_path,
                class_predictions=test_predictions.class_predictions,
                class_probabilities=test_predictions.class_probabilities,
                confidence=test_predictions.confidence,
            )
            logger.info(f"Saved test predictions to {test_pred_path}")

        # Feature importance (if available)
        importance = self.model.get_feature_importance()
        if importance:
            importance_path = self.output_path / "metrics" / "feature_importance.json"
            with open(importance_path, "w") as f:
                json.dump(importance, f, indent=2)

        logger.debug(f"Saved artifacts to {self.output_path}")

    def _save_model(self) -> None:
        """Save trained model."""
        model_path = self.output_path / "checkpoints" / "best_model"
        self.model.save(model_path)
        logger.info(f"Saved model to {model_path}")

    def _save_feature_selection(self) -> None:
        """Save feature selection result with model artifacts."""
        if not self._is_feature_selection_enabled():
            return
        if self.feature_selector is None or not self.feature_selector.is_fitted:
            return

        fs_path = self.output_path / "config" / "feature_selection.json"
        self.feature_selector.save(fs_path)
        logger.info(f"Saved feature selection to {fs_path}")

    def _save_calibrator(self) -> None:
        """Save probability calibrator."""
        if self.calibrator is None:
            return
        calibrator_path = self.output_path / "checkpoints" / "calibrator.pkl"
        self.calibrator.save(calibrator_path)
        logger.info(f"Saved calibrator to {calibrator_path}")

    def _save_checksums(self) -> None:
        """
        Generate and save checksums for all model artifacts.

        Creates checksums.json in the output directory with SHA256 hashes
        for all model files (.pt, .pkl, .json, etc.). Enables integrity
        verification when loading models later.
        """
        integrity_manager = ArtifactIntegrityManager(self.output_path)

        # Register all artifacts
        checksums = integrity_manager.register_all()

        if checksums:
            integrity_manager.save_checksums()
            logger.info(f"Generated checksums for {len(checksums)} artifacts")
        else:
            logger.debug("No artifacts found to checksum")
