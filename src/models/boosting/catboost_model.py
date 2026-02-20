"""
CatBoost Model - Ordered boosting for 3-class prediction.

Supports GPU training (task_type='GPU'), early stopping, sample weights,
and feature importance extraction. Uses ordered boosting for better
handling of categorical features and reduced overfitting.

Note: This model is only registered if CatBoost is installed.
Install with: pip install catboost
"""

from __future__ import annotations

import logging
import pickle
import time
from pathlib import Path
from typing import Any

import numpy as np

from ..base import BaseModel, PredictionResult, TrainingMetrics
from ..common import map_classes_to_labels, map_labels_to_classes
from ..neural.numerical_stability import validate_training_inputs

logger = logging.getLogger(__name__)

# Check if CatBoost is available
try:
    from catboost import CatBoostClassifier, Pool  # type: ignore[import-not-found]

    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    CatBoostClassifier = None
    Pool = None
    logger.debug(
        "CatBoost not installed. CatBoostModel will not be registered. "
        "Install with: pip install catboost"
    )


def _check_cuda_available() -> bool:
    """Check if CatBoost GPU is available without running training."""
    if not CATBOOST_AVAILABLE:
        return False
    try:
        # Check if CUDA is available via torch (more reliable)
        import torch

        # CatBoost has excellent GPU support when CUDA is available
        return torch.cuda.is_available()
    except ImportError:
        # Fallback: assume GPU available if use_gpu=True in config
        return True
    except Exception as e:
        logger.warning(f"GPU detection failed: {e}. Assuming no GPU available.")
        return False


class CatBoostModel(BaseModel):
    """CatBoost gradient boosting classifier with GPU support."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        if not CATBOOST_AVAILABLE:
            raise ImportError("CatBoost is not installed. Install with: pip install catboost")
        super().__init__(config)
        self._model: CatBoostClassifier | None = None
        self._feature_names: list[str] | None = None
        self._n_classes: int = 3

        # Check if task_type is explicitly set to CPU (force CPU mode)
        task_type = self._config.get("task_type", "").upper()
        if task_type == "CPU":
            self._use_gpu = False
            logger.info("CatBoost CPU mode forced via task_type config")
        else:
            self._use_gpu = self._config.get("use_gpu", self._detect_gpu())
            if self._use_gpu:
                if _check_cuda_available():
                    logger.info("CatBoost GPU acceleration: ENABLED (auto-detected)")
                else:
                    logger.warning("CUDA not available for CatBoost, falling back to CPU")
                    self._use_gpu = False

    @staticmethod
    def _detect_gpu() -> bool:
        """Auto-detect CUDA availability for GPU acceleration."""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False

    @property
    def model_family(self) -> str:
        return "boosting"

    @property
    def requires_scaling(self) -> bool:
        return False

    @property
    def requires_sequences(self) -> bool:
        return False

    def get_default_config(self) -> dict[str, Any]:
        return {
            "iterations": 500,
            "depth": 6,
            "learning_rate": 0.05,
            "l2_leaf_reg": 3.0,
            "random_strength": 1.0,
            "bagging_temperature": 1.0,
            "early_stopping_rounds": 20,
            "use_gpu": True,  # Phase 12A-8: Enable GPU by default
            "devices": "0",
            "random_state": 42,
            "thread_count": -1,
            "verbose": False,
        }

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        sample_weights: np.ndarray | None = None,
        config: dict[str, Any] | None = None,
    ) -> TrainingMetrics:
        """Train CatBoost model with early stopping."""
        self._validate_input_shape(X_train, "X_train")
        self._validate_input_shape(X_val, "X_val")
        validate_training_inputs(X_train, y_train, X_val, y_val, sample_weights)
        start_time = time.time()

        # Merge config
        train_config = self._config.copy()
        if config:
            train_config.update(config)

        # Convert labels: -1,0,1 -> 0,1,2
        y_train_cat = self._convert_labels_to_cat(y_train)
        y_val_cat = self._convert_labels_to_cat(y_val)

        # Apply class weights if enabled (balances short/neutral/long predictions)
        final_weights = sample_weights
        if train_config.get("use_class_weights", True):
            # Compute balanced class weights
            unique_classes, class_counts = np.unique(y_train_cat, return_counts=True)
            n_samples = len(y_train_cat)
            n_classes = len(unique_classes)
            class_weight_values = n_samples / (n_classes * class_counts)

            # Create mapping from class to weight (handles missing classes)
            class_weight_dict = dict(zip(unique_classes, class_weight_values, strict=False))

            # Map weights to samples using the dictionary
            sample_class_weights = np.array([class_weight_dict[int(c)] for c in y_train_cat])

            # Combine with existing sample weights
            if sample_weights is not None:
                final_weights = sample_weights * sample_class_weights
            else:
                final_weights = sample_class_weights

            logger.debug(f"Class weights applied: {class_weight_dict}")

        # Create Pool objects
        train_pool = Pool(data=X_train, label=y_train_cat, weight=final_weights)
        val_pool = Pool(data=X_val, label=y_val_cat)

        # Build model
        self._model = self._build_model(train_config)
        iterations = train_config.get("iterations", 500)
        early_stopping = train_config.get("early_stopping_rounds", 20)

        logger.info(
            f"Training CatBoost: iterations={iterations}, "
            f"depth={train_config.get('depth', 6)}, "
            f"gpu={'on' if self._use_gpu else 'off'}"
        )

        self._model.fit(
            train_pool,
            eval_set=val_pool,
            early_stopping_rounds=early_stopping,
            verbose=50 if train_config.get("verbose", False) else False,
        )

        training_time = time.time() - start_time

        # Extract metrics from training history
        evals_result = self._model.get_evals_result()
        train_losses = evals_result.get("learn", {}).get("MultiClass", [])
        val_losses = evals_result.get("validation", {}).get("MultiClass", [])

        best_iteration = self._model.get_best_iteration() if self._model is not None else 0
        epochs_trained = best_iteration + 1 if best_iteration is not None else len(train_losses)

        # Compute accuracy and F1
        train_metrics = self._compute_metrics(X_train, y_train)
        val_metrics = self._compute_metrics(X_val, y_val)

        self._is_fitted = True

        logger.info(
            f"Training complete: epochs={epochs_trained}, val_f1={val_metrics['f1']:.4f}, "
            f"time={training_time:.1f}s"
        )

        return TrainingMetrics(
            train_loss=train_losses[-1] if train_losses else 0.0,
            val_loss=val_losses[-1] if val_losses else 0.0,
            train_accuracy=train_metrics["accuracy"],
            val_accuracy=val_metrics["accuracy"],
            train_f1=train_metrics["f1"],
            val_f1=val_metrics["f1"],
            epochs_trained=epochs_trained,
            training_time_seconds=training_time,
            early_stopped=epochs_trained < iterations,
            best_epoch=best_iteration,
            history={"train_MultiClass": train_losses, "val_MultiClass": val_losses},
            metadata={
                "best_iteration": best_iteration,
                "n_features": X_train.shape[1],
                "n_train_samples": len(X_train),
                "n_val_samples": len(X_val),
                "use_gpu": self._use_gpu,
            },
        )

    def predict(self, X: np.ndarray) -> PredictionResult:
        """Generate predictions with class probabilities."""
        self._validate_fitted()
        self._validate_input_shape(X, "X")

        if self._model is None:
            raise RuntimeError("Model not fitted")
        probabilities = self._model.predict_proba(X)
        class_predictions_cat = np.argmax(probabilities, axis=1)
        class_predictions = self._convert_labels_from_cat(class_predictions_cat)
        confidence = np.max(probabilities, axis=1)

        return PredictionResult(
            class_predictions=class_predictions,
            class_probabilities=probabilities,
            confidence=confidence,
            metadata={"model": "catboost"},
        )

    def save(self, path: Path) -> None:
        """Save model (cbm) and metadata (pickle) to directory."""
        self._validate_fitted()
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        if self._model is None:
            raise RuntimeError("Model not fitted")
        self._model.save_model(str(path / "model.cbm"))

        metadata = {
            "config": self._config,
            "feature_names": self._feature_names,
            "n_classes": self._n_classes,
            "use_gpu": self._use_gpu,
            "best_iteration": self._model.get_best_iteration() if self._model is not None else 0,
        }
        with open(path / "metadata.pkl", "wb") as f:
            pickle.dump(metadata, f, protocol=pickle.HIGHEST_PROTOCOL)

        logger.info(f"Saved CatBoost model to {path}")

    def load(self, path: Path) -> None:
        """Load model from directory."""
        path = Path(path)
        model_path = path / "model.cbm"
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        self._model = CatBoostClassifier()
        self._model.load_model(str(model_path))

        metadata_path = path / "metadata.pkl"
        if metadata_path.exists():
            from src.core.utils.safe_pickle import safe_pickle_load

            metadata = safe_pickle_load(metadata_path)
            self._config = metadata.get("config", self._config)
            self._feature_names = metadata.get("feature_names")
            self._n_classes = metadata.get("n_classes", 3)
            self._use_gpu = metadata.get("use_gpu", False)

        self._is_fitted = True
        logger.info(f"Loaded CatBoost model from {path}")

    def get_feature_importance(self) -> dict[str, float] | None:
        """Return feature importances."""
        if not self._is_fitted or self._model is None:
            return None

        importance = self._model.get_feature_importance()
        feature_names = self._feature_names or [f"f{i}" for i in range(len(importance))]

        return dict(zip(feature_names, importance.tolist(), strict=False))

    def set_feature_names(self, names: list[str]) -> None:
        """Set feature names for interpretability."""
        self._feature_names = names

    def _build_model(self, config: dict[str, Any]) -> CatBoostClassifier:
        """Build CatBoostClassifier with config and range validation."""
        depth = config.get("depth", 6)
        learning_rate = config.get("learning_rate", 0.05)
        iterations = config.get("iterations", 500)
        if depth <= 0:
            raise ValueError(f"depth must be > 0, got {depth}")
        if learning_rate <= 0:
            raise ValueError(f"learning_rate must be > 0, got {learning_rate}")
        if iterations <= 0:
            raise ValueError(f"iterations must be > 0, got {iterations}")

        params = {
            "iterations": config.get("iterations", 500),
            "depth": config.get("depth", 6),
            "learning_rate": config.get("learning_rate", 0.05),
            "l2_leaf_reg": config.get("l2_leaf_reg", 3.0),
            "random_strength": config.get("random_strength", 1.0),
            "bagging_temperature": config.get("bagging_temperature", 1.0),
            "loss_function": "MultiClass",
            "eval_metric": "MultiClass",
            "use_best_model": True,
            "random_seed": config.get("random_state", 42),
            "verbose": False,
        }

        if self._use_gpu:
            params["task_type"] = "GPU"
            params["devices"] = config.get("devices", "0")
        else:
            params["task_type"] = "CPU"
            thread_count = config.get("thread_count", -1)
            # CatBoost accepts -1 for "use all cores". Do NOT convert to 0,
            # as thread_count=0 crashes CatBoost 1.2.x with an unsigned int cast error.
            params["thread_count"] = thread_count

        return CatBoostClassifier(**params)

    def _convert_labels_to_cat(self, labels: np.ndarray) -> np.ndarray:
        """Convert labels from -1,0,1 to 0,1,2."""
        return map_labels_to_classes(labels)

    def _convert_labels_from_cat(self, labels: np.ndarray) -> np.ndarray:
        """Convert labels from 0,1,2 to -1,0,1."""
        return map_classes_to_labels(labels)

    def _compute_metrics(self, X: np.ndarray, y_true: np.ndarray) -> dict[str, float]:
        """Compute accuracy and F1 for a dataset."""
        from sklearn.metrics import accuracy_score, f1_score

        if self._model is None:
            raise RuntimeError("Model not fitted")
        probabilities = self._model.predict_proba(X)
        y_pred_cat = np.argmax(probabilities, axis=1)
        y_pred = self._convert_labels_from_cat(y_pred_cat)

        return {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        }


# Conditional registration: Only register if CatBoost is available
# This prevents the model from appearing in the registry when it cannot be instantiated
if CATBOOST_AVAILABLE:
    from ..registry import register

    # Apply the decorator manually to register the class
    # Use a temporary variable to avoid mypy type narrowing issues
    _registered_class = register(
        name="catboost",
        family="boosting",
        description="CatBoost gradient boosting with GPU support",
        aliases=["cat"],
    )(CatBoostModel)
    CatBoostModel = _registered_class  # type: ignore[assignment,misc]


__all__ = ["CatBoostModel", "CATBOOST_AVAILABLE"]
