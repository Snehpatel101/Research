"""
EnsembleOrchestrator - THE single entry point for ensemble training.

Uses PipelineConfig from src/core as the ONLY configuration source.
Integrates with PHASE_2 OOF alignment and PHASE_3 training.

PHASE_4: This module provides the unified ensemble orchestrator that:
1. Uses PipelineConfig as the ONLY configuration source
2. Integrates with OOFAligner from PHASE_2 adapters for heterogeneous ensembles
3. Supports all 4 meta-learners: ridge_meta, mlp_meta, xgboost_meta, calibrated_meta
4. Returns structured EnsembleResult with comprehensive metrics
5. Has convenience method to train from PHASE_3 TrainingRunResult

Example:
    from src.core import PipelineConfig
    from src.models.ensemble import EnsembleOrchestrator

    config = PipelineConfig(
        symbol="MES",
        data_path="./data/mes.parquet",
        output_dir="./experiments/exp_001",
        models=["xgboost", "lightgbm", "lstm"],
        build_ensemble=True,
        meta_learner="ridge_meta",
    )

    orchestrator = EnsembleOrchestrator(config)
    result = orchestrator.train(oof_predictions, y_train)
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from src.core import (
    MODEL_ADAPTER_MAP,
    MODEL_DATA_RANKS,
    OOFResult,
    PipelineConfig,
)
from src.core.interfaces import OOFPredictionProtocol
from src.data.adapters import (
    AlignedOOFResult,
    OOFAligner,
    align_oof_predictions,
)

# TYPE_CHECKING import to break circular dependency:
# cross_validation/__init__ -> cv_feature_selection -> oof_generator -> oof_core
# -> models.base -> models/__init__ -> models.ensemble -> orchestrator
# -> cross_validation (CIRCULAR)
if TYPE_CHECKING:
    from src.validation.cv import OOFPrediction, StackingDataset
    from src.training.unified_orchestrator import TrainingRunResult
else:
    # At runtime, use Protocol for type hints (structural typing)
    OOFPrediction = OOFPredictionProtocol
    # StackingDataset is used as a type hint and also instantiated,
    # so we need a lazy import in the methods that use it

logger = logging.getLogger(__name__)


# =============================================================================
# RESULT DATACLASS
# =============================================================================


@dataclass
class EnsembleResult:
    """
    Result from ensemble training.

    Attributes:
        ensemble_name: Name of the ensemble (e.g., "ridge_meta_ensemble")
        meta_learner_name: Name of the meta-learner used
        base_model_names: List of base model names included in ensemble
        metrics: Validation metrics dict (val_f1, val_accuracy, etc.)
        stacking_dataset: Optional stacking dataset used for training
        aligned_oof: Optional AlignedOOFResult from alignment
        training_time_seconds: Time taken to train ensemble
        n_base_models: Number of base models in ensemble
        coverage: Coverage ratio after alignment (0.0 to 1.0)
        alignment_offset: Index offset from alignment (for sequence models)
    """

    ensemble_name: str
    meta_learner_name: str
    base_model_names: List[str]
    metrics: Dict[str, float]
    stacking_dataset: Optional[StackingDataset] = None
    aligned_oof: Optional[AlignedOOFResult] = None
    training_time_seconds: float = 0.0
    n_base_models: int = 0
    coverage: float = 1.0
    alignment_offset: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "ensemble_name": self.ensemble_name,
            "meta_learner_name": self.meta_learner_name,
            "base_model_names": self.base_model_names,
            "metrics": self.metrics,
            "training_time": self.training_time_seconds,
            "n_base_models": self.n_base_models,
            "coverage": self.coverage,
            "alignment_offset": self.alignment_offset,
        }

    def summary(self) -> str:
        """Get human-readable summary string."""
        lines = [
            f"EnsembleResult: {self.ensemble_name}",
            f"  Meta-learner: {self.meta_learner_name}",
            f"  Base models: {self.n_base_models} ({', '.join(self.base_model_names[:3])}{'...' if len(self.base_model_names) > 3 else ''})",
            f"  Coverage: {self.coverage:.2%}",
            f"  Training time: {self.training_time_seconds:.1f}s",
            "  Metrics:",
        ]
        for key, value in self.metrics.items():
            lines.append(f"    {key}: {value:.4f}")
        return "\n".join(lines)


# =============================================================================
# ENSEMBLE ORCHESTRATOR
# =============================================================================


class EnsembleOrchestrator:
    """
    THE single entry point for ensemble training in the ML Factory.

    Uses PipelineConfig from src/core as the ONLY configuration source.

    Supports:
    - Heterogeneous ensembles (mixing 2D tabular + 3D sequence + 4D transformer models)
    - OOF alignment via PHASE_2 OOFAligner
    - 4 meta-learners: ridge_meta, mlp_meta, xgboost_meta, calibrated_meta
    - Diversity analysis and model selection

    Usage:
        from src.core import PipelineConfig
        from src.models.ensemble import EnsembleOrchestrator

        config = PipelineConfig(
            symbol="MES",
            data_path="./data/mes.parquet",
            output_dir="./experiments/exp_001",
            models=["xgboost", "lightgbm", "lstm"],
            build_ensemble=True,
            meta_learner="ridge_meta",
        )

        orchestrator = EnsembleOrchestrator(config)
        result = orchestrator.train(oof_predictions, y_train)
    """

    # Available meta-learners
    AVAILABLE_META_LEARNERS = [
        "ridge_meta",
        "mlp_meta",
        "xgboost_meta",
        "calibrated_meta",
    ]

    def __init__(self, config: PipelineConfig) -> None:
        """
        Initialize with PipelineConfig.

        Args:
            config: PipelineConfig instance from src/core

        Raises:
            ValueError: If config.meta_learner is not a valid meta-learner
        """
        self.config = config
        self.output_dir = config.output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Validate meta-learner
        if config.meta_learner not in self.AVAILABLE_META_LEARNERS:
            raise ValueError(
                f"Unknown meta_learner: {config.meta_learner}. "
                f"Available: {self.AVAILABLE_META_LEARNERS}"
            )

        # Results storage
        self._ensemble: Any = None
        self._result: Optional[EnsembleResult] = None
        self._aligner = OOFAligner()

        logger.info("EnsembleOrchestrator initialized")
        logger.info(f"  Meta-learner: {config.meta_learner}")
        logger.info(f"  Base models: {config.models}")
        logger.info(f"  Output dir: {self.output_dir}")

    def train(
        self,
        oof_predictions: Dict[str, OOFPrediction],
        y_train: np.ndarray,
        sample_weights: Optional[np.ndarray] = None,
    ) -> EnsembleResult:
        """
        Train ensemble from OOF predictions.

        This is the main entry point for ensemble training. It:
        1. Converts OOFPrediction to OOFResult format
        2. Aligns predictions using PHASE_2 OOFAligner
        3. Builds stacking features from aligned predictions
        4. Trains the configured meta-learner
        5. Returns comprehensive EnsembleResult

        Args:
            oof_predictions: Dict mapping model_name -> OOFPrediction
                from src.cross_validation
            y_train: Training labels (full length, will be aligned)
            sample_weights: Optional sample weights (full length)

        Returns:
            EnsembleResult with trained ensemble and metrics

        Raises:
            ValueError: If fewer than 2 models provided or alignment fails
        """
        start_time = time.time()

        logger.info("=" * 60)
        logger.info(f"ENSEMBLE TRAINING: {self.config.meta_learner}")
        logger.info("=" * 60)

        if len(oof_predictions) < 2:
            raise ValueError(
                f"Need at least 2 models for ensemble, got {len(oof_predictions)}"
            )

        # Step 1: Convert OOFPrediction to OOFResult for OOFAligner
        oof_results = self._convert_to_oof_results(oof_predictions)

        logger.info(f"Converted {len(oof_results)} OOF predictions to alignment format")

        # Step 2: Align OOF predictions using PHASE_2 OOFAligner
        try:
            aligned = self._aligner.align(oof_results, strategy="intersection")
        except ValueError as e:
            raise ValueError(f"Failed to align OOF predictions: {e}") from e

        logger.info(f"Aligned {len(aligned.model_names)} models")
        logger.info(f"  Valid samples: {aligned.n_common}")
        logger.info(f"  Coverage per model: {aligned.coverage}")

        # Compute overall coverage
        min_coverage = min(aligned.coverage.values()) if aligned.coverage else 1.0

        # Step 3: Build stacking dataset
        stacking_X = aligned.stacking_features  # Includes derived features

        # Align y_train to the common indices
        stacking_y = y_train[aligned.common_indices]

        logger.info(f"Stacking dataset: X={stacking_X.shape}, y={stacking_y.shape}")

        # Step 4: Create and train meta-learner
        from src.models.ensemble import (
            CalibratedMetaLearner,
            MLPMetaLearner,
            RidgeMetaLearner,
            XGBoostMeta,
        )

        meta_learner_map = {
            "ridge_meta": RidgeMetaLearner,
            "mlp_meta": MLPMetaLearner,
            "xgboost_meta": XGBoostMeta,
            "calibrated_meta": CalibratedMetaLearner,
        }

        meta_learner_class = meta_learner_map[self.config.meta_learner]
        meta_learner = meta_learner_class()

        # Split for validation (time-series aware: use last portion)
        val_ratio = 0.2
        n_samples = len(stacking_y)
        n_train = int(n_samples * (1 - val_ratio))

        X_train_meta = stacking_X[:n_train]
        y_train_meta = stacking_y[:n_train]
        X_val_meta = stacking_X[n_train:]
        y_val_meta = stacking_y[n_train:]

        # Align sample weights if provided
        weights_train = None
        if sample_weights is not None:
            aligned_weights = sample_weights[aligned.common_indices]
            weights_train = aligned_weights[:n_train]

        logger.info(f"Training meta-learner: {self.config.meta_learner}")
        logger.info(f"  Train samples: {len(y_train_meta)}")
        logger.info(f"  Val samples: {len(y_val_meta)}")

        # Train meta-learner
        training_metrics = meta_learner.fit(
            X_train=X_train_meta,
            y_train=y_train_meta,
            X_val=X_val_meta,
            y_val=y_val_meta,
            sample_weights=weights_train,
        )

        # Step 5: Evaluate on validation set
        output = meta_learner.predict(X_val_meta)

        from sklearn.metrics import accuracy_score, f1_score, log_loss

        # Compute metrics
        val_accuracy = float(accuracy_score(y_val_meta, output.class_predictions))
        val_f1 = float(
            f1_score(y_val_meta, output.class_predictions, average="macro", zero_division=0)
        )

        # Log loss if probabilities available
        try:
            val_logloss = float(log_loss(y_val_meta, output.class_probabilities))
        except Exception:
            val_logloss = float("nan")

        metrics = {
            "val_accuracy": val_accuracy,
            "val_f1": val_f1,
            "val_logloss": val_logloss,
            "train_loss": training_metrics.train_loss,
            "val_loss": training_metrics.val_loss,
            "train_f1": training_metrics.train_f1,
            "val_f1_from_training": training_metrics.val_f1,
        }

        training_time = time.time() - start_time

        logger.info("Ensemble training complete:")
        logger.info(f"  Val F1: {val_f1:.4f}")
        logger.info(f"  Val Accuracy: {val_accuracy:.4f}")
        logger.info(f"  Val LogLoss: {val_logloss:.4f}")
        logger.info(f"  Training time: {training_time:.1f}s")

        # Store ensemble
        self._ensemble = meta_learner

        # Create stacking dataset for output
        # Lazy import to avoid circular dependency
        from src.validation.cv.oof_stacking import StackingDataset

        feature_names = aligned.get_feature_names()
        stacking_df = pd.DataFrame(stacking_X, columns=feature_names)
        stacking_df["y_true"] = stacking_y

        stacking_ds = StackingDataset(
            data=stacking_df,
            model_names=aligned.model_names,
            horizon=self.config.horizons[0] if self.config.horizons else 20,
            metadata={
                "n_common": aligned.n_common,
                "coverage": aligned.coverage,
                "meta_learner": self.config.meta_learner,
            },
        )

        # Compute alignment offset (minimum offset from sequence models)
        alignment_offset = 0
        if aligned.common_indices is not None and len(aligned.common_indices) > 0:
            alignment_offset = int(aligned.common_indices.min())

        self._result = EnsembleResult(
            ensemble_name=f"{self.config.meta_learner}_ensemble",
            meta_learner_name=self.config.meta_learner,
            base_model_names=list(oof_predictions.keys()),
            metrics=metrics,
            stacking_dataset=stacking_ds,
            aligned_oof=aligned,
            training_time_seconds=training_time,
            n_base_models=len(oof_predictions),
            coverage=min_coverage,
            alignment_offset=alignment_offset,
        )

        # Save results
        self._save_results()

        return self._result

    def train_from_training_result(
        self,
        training_result: "TrainingRunResult",
    ) -> EnsembleResult:
        """
        Train ensemble from PHASE_3 TrainingRunResult.

        Convenience method that extracts OOF predictions from training result.
        This enables direct integration with UnifiedTrainingOrchestrator output.

        Args:
            training_result: Result from UnifiedTrainingOrchestrator.train()

        Returns:
            EnsembleResult

        Raises:
            ValueError: If no OOF predictions found in training result
        """
        # Extract OOF predictions from model results
        oof_predictions: Dict[str, OOFPrediction] = {}
        y_train: Optional[np.ndarray] = None

        for key, model_result in training_result.model_results.items():
            if model_result.oof_prediction is not None:
                oof_predictions[model_result.model_name] = model_result.oof_prediction

                # Extract y_train from first OOF prediction's y_true column
                if y_train is None:
                    oof_df = model_result.oof_prediction.predictions
                    if "y_true" in oof_df.columns:
                        y_train = oof_df["y_true"].values

        if not oof_predictions:
            raise ValueError("No OOF predictions found in training result")

        if y_train is None:
            # Try to extract from aligned_oof if available
            if training_result.aligned_oof is not None:
                raise ValueError(
                    "Cannot extract y_train from OOF predictions. "
                    "Please provide y_train explicitly via train() method."
                )
            raise ValueError(
                "Cannot extract y_train from OOF predictions. "
                "Ensure OOF predictions include 'y_true' column."
            )

        logger.info(
            f"Extracted {len(oof_predictions)} OOF predictions from TrainingRunResult"
        )

        return self.train(oof_predictions, y_train)

    def predict(
        self,
        base_predictions: Dict[str, np.ndarray],
    ) -> np.ndarray:
        """
        Generate ensemble predictions from base model outputs.

        Args:
            base_predictions: Dict mapping model_name -> probability array
                Each array should be shape (n_samples, n_classes)

        Returns:
            Ensemble class predictions array of shape (n_samples,)

        Raises:
            ValueError: If ensemble not trained or alignment fails
        """
        if self._ensemble is None:
            raise ValueError("Ensemble not trained. Call train() first.")

        # Convert to OOFResult format for alignment
        oof_results: List[OOFResult] = []

        for name, probs in base_predictions.items():
            n_samples = probs.shape[0]
            indices = np.arange(n_samples)
            predictions = np.argmax(probs, axis=1)

            # Map 0,1,2 back to -1,0,1 if needed (depends on model output)
            if predictions.min() >= 0:
                predictions = predictions - 1  # 0,1,2 -> -1,0,1

            oof_result = OOFResult(
                predictions=predictions,
                probabilities=probs,
                indices=indices,
                fold_ids=np.zeros(n_samples, dtype=int),
                model_name=name,
                coverage=1.0,
            )
            oof_results.append(oof_result)

        # Align predictions
        aligned = self._aligner.align(oof_results, strategy="intersection")
        stacking_X = aligned.stacking_features

        # Generate ensemble predictions
        output = self._ensemble.predict(stacking_X)
        return output.class_predictions

    def predict_proba(
        self,
        base_predictions: Dict[str, np.ndarray],
    ) -> np.ndarray:
        """
        Generate ensemble probability predictions from base model outputs.

        Args:
            base_predictions: Dict mapping model_name -> probability array

        Returns:
            Ensemble probability array of shape (n_samples, n_classes)

        Raises:
            ValueError: If ensemble not trained
        """
        if self._ensemble is None:
            raise ValueError("Ensemble not trained. Call train() first.")

        # Convert to OOFResult format for alignment
        oof_results: List[OOFResult] = []

        for name, probs in base_predictions.items():
            n_samples = probs.shape[0]
            indices = np.arange(n_samples)
            predictions = np.argmax(probs, axis=1) - 1  # 0,1,2 -> -1,0,1

            oof_result = OOFResult(
                predictions=predictions,
                probabilities=probs,
                indices=indices,
                fold_ids=np.zeros(n_samples, dtype=int),
                model_name=name,
                coverage=1.0,
            )
            oof_results.append(oof_result)

        aligned = self._aligner.align(oof_results, strategy="intersection")
        stacking_X = aligned.stacking_features

        output = self._ensemble.predict(stacking_X)
        return output.class_probabilities

    def _convert_to_oof_results(
        self,
        oof_predictions: Dict[str, OOFPrediction],
    ) -> List[OOFResult]:
        """
        Convert OOFPrediction dict to list of OOFResult for OOFAligner.

        OOFAligner expects OOFResult format from src.core.interfaces,
        while OOFGenerator produces OOFPrediction from src.validation.cv.

        Args:
            oof_predictions: Dict mapping model_name -> OOFPrediction

        Returns:
            List of OOFResult for alignment
        """
        oof_results: List[OOFResult] = []

        for model_name, oof_pred in oof_predictions.items():
            # Extract probabilities and predictions
            probs = oof_pred.get_probabilities()
            preds = oof_pred.get_class_predictions()

            n_samples = len(probs)

            # Use original_indices if available, otherwise sequential
            if oof_pred.original_indices is not None:
                indices = oof_pred.original_indices
            else:
                indices = np.arange(n_samples)

            # Create fold_ids from fold_info
            fold_ids = np.zeros(n_samples, dtype=int)
            # Note: More sophisticated fold assignment could be done here
            # based on oof_pred.fold_info

            oof_result = OOFResult(
                predictions=preds.astype(int),
                probabilities=probs,
                indices=indices,
                fold_ids=fold_ids,
                model_name=model_name,
                coverage=oof_pred.coverage,
            )
            oof_results.append(oof_result)

        return oof_results

    def _save_results(self) -> None:
        """Save ensemble results to disk."""
        if self._result is None:
            return

        ensemble_dir = self.output_dir / "ensemble"
        ensemble_dir.mkdir(parents=True, exist_ok=True)

        # Save metrics as JSON
        metrics_path = ensemble_dir / "ensemble_metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(self._result.to_dict(), f, indent=2)

        logger.info(f"Saved ensemble metrics to: {metrics_path}")

        # Save ensemble model
        if self._ensemble is not None:
            model_path = ensemble_dir / "meta_learner"
            try:
                self._ensemble.save(model_path)
                logger.info(f"Saved meta-learner to: {model_path}")
            except Exception as e:
                logger.warning(f"Failed to save meta-learner: {e}")

        # Save stacking dataset if available
        if self._result.stacking_dataset is not None:
            stacking_path = ensemble_dir / "stacking_dataset.parquet"
            try:
                self._result.stacking_dataset.data.to_parquet(stacking_path)
                logger.info(f"Saved stacking dataset to: {stacking_path}")
            except Exception as e:
                logger.warning(f"Failed to save stacking dataset: {e}")

        logger.info(f"Ensemble results saved to: {ensemble_dir}")

    def load(self, path: Union[str, Path]) -> None:
        """
        Load a trained ensemble from disk.

        Args:
            path: Path to ensemble directory (containing meta_learner/)

        Raises:
            FileNotFoundError: If path does not exist
        """
        path = Path(path)
        model_path = path / "meta_learner"

        if not model_path.exists():
            raise FileNotFoundError(f"Meta-learner not found at: {model_path}")

        # Load the appropriate meta-learner class
        from src.models.ensemble import (
            CalibratedMetaLearner,
            MLPMetaLearner,
            RidgeMetaLearner,
            XGBoostMeta,
        )

        meta_learner_map = {
            "ridge_meta": RidgeMetaLearner,
            "mlp_meta": MLPMetaLearner,
            "xgboost_meta": XGBoostMeta,
            "calibrated_meta": CalibratedMetaLearner,
        }

        meta_learner_class = meta_learner_map[self.config.meta_learner]
        self._ensemble = meta_learner_class()
        self._ensemble.load(model_path)

        logger.info(f"Loaded ensemble from: {path}")

    @property
    def ensemble(self) -> Any:
        """Get trained ensemble model."""
        return self._ensemble

    @property
    def result(self) -> Optional[EnsembleResult]:
        """Get training result."""
        return self._result

    @property
    def is_trained(self) -> bool:
        """Check if ensemble has been trained."""
        return self._ensemble is not None


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


def build_ensemble(
    config: PipelineConfig,
    oof_predictions: Dict[str, OOFPrediction],
    y_train: np.ndarray,
    sample_weights: Optional[np.ndarray] = None,
) -> EnsembleResult:
    """
    Convenience function to build ensemble.

    Usage:
        from src.core import PipelineConfig
        from src.models.ensemble import build_ensemble

        config = PipelineConfig(
            symbol="MES",
            data_path="./data/mes.parquet",
            output_dir="./experiments/exp_001",
            models=["xgboost", "lightgbm", "lstm"],
            build_ensemble=True,
            meta_learner="ridge_meta",
        )

        result = build_ensemble(config, oof_predictions, y_train)

    Args:
        config: PipelineConfig from src/core
        oof_predictions: Dict mapping model_name -> OOFPrediction
        y_train: Training labels
        sample_weights: Optional sample weights

    Returns:
        EnsembleResult with trained ensemble and metrics
    """
    orchestrator = EnsembleOrchestrator(config)
    return orchestrator.train(oof_predictions, y_train, sample_weights)


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "EnsembleOrchestrator",
    "EnsembleResult",
    "build_ensemble",
]
