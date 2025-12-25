"""
BaseModel abstract interface for the Model Factory.

All models in the factory must implement this interface to ensure
consistent training, prediction, and evaluation workflows.

This module provides:
- PredictionOutput: Standardized prediction container
- TrainingMetrics: Standardized training metrics container
- BaseModel: Abstract base class for all models
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np


# =============================================================================
# PREDICTION OUTPUT
# =============================================================================

@dataclass
class PredictionOutput:
    """
    Standardized prediction output for all models.

    All models must return predictions in this format to enable
    unified evaluation and ensemble composition.

    Attributes:
        class_predictions: Predicted class labels, shape (n_samples,)
        class_probabilities: Class probabilities, shape (n_samples, n_classes)
        confidence: Prediction confidence (max probability), shape (n_samples,)
        metadata: Model-specific metadata (feature importance, attention, etc.)

    Example:
        >>> output = model.predict(X_test)
        >>> print(output.class_predictions.shape)  # (1000,)
        >>> print(output.class_probabilities.shape)  # (1000, 3)
        >>> print(output.confidence.mean())  # 0.65
    """
    class_predictions: np.ndarray
    class_probabilities: np.ndarray
    confidence: np.ndarray
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate prediction output shapes."""
        n_samples = len(self.class_predictions)

        if len(self.class_probabilities) != n_samples:
            raise ValueError(
                f"class_probabilities length ({len(self.class_probabilities)}) "
                f"!= class_predictions length ({n_samples})"
            )

        if len(self.confidence) != n_samples:
            raise ValueError(
                f"confidence length ({len(self.confidence)}) "
                f"!= class_predictions length ({n_samples})"
            )

    @property
    def n_samples(self) -> int:
        """Number of samples in predictions."""
        return len(self.class_predictions)

    @property
    def n_classes(self) -> int:
        """Number of classes."""
        return self.class_probabilities.shape[1]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "class_predictions": self.class_predictions.tolist(),
            "class_probabilities": self.class_probabilities.tolist(),
            "confidence": self.confidence.tolist(),
            "metadata": self.metadata,
        }


# =============================================================================
# TRAINING METRICS
# =============================================================================

@dataclass
class TrainingMetrics:
    """
    Standardized training metrics for all models.

    Captures training progress and final results in a consistent
    format across all model types.

    Attributes:
        train_loss: Final training loss
        val_loss: Final validation loss
        train_accuracy: Training accuracy
        val_accuracy: Validation accuracy
        train_f1: Training macro F1 score
        val_f1: Validation macro F1 score
        epochs_trained: Number of epochs completed
        training_time_seconds: Total training time
        early_stopped: Whether training was stopped early
        best_epoch: Epoch with best validation performance
        history: Per-epoch metric history
        metadata: Model-specific training metadata

    Example:
        >>> metrics = model.fit(X_train, y_train, X_val, y_val)
        >>> print(f"Best epoch: {metrics.best_epoch}")
        >>> print(f"Val F1: {metrics.val_f1:.3f}")
    """
    train_loss: float
    val_loss: float
    train_accuracy: float
    val_accuracy: float
    train_f1: float
    val_f1: float
    epochs_trained: int
    training_time_seconds: float
    early_stopped: bool
    best_epoch: Optional[int]
    history: Dict[str, List[float]] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate metrics are in valid ranges."""
        if self.epochs_trained < 0:
            raise ValueError(f"epochs_trained must be >= 0, got {self.epochs_trained}")
        if self.training_time_seconds < 0:
            raise ValueError(
                f"training_time_seconds must be >= 0, got {self.training_time_seconds}"
            )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "train_loss": self.train_loss,
            "val_loss": self.val_loss,
            "train_accuracy": self.train_accuracy,
            "val_accuracy": self.val_accuracy,
            "train_f1": self.train_f1,
            "val_f1": self.val_f1,
            "epochs_trained": self.epochs_trained,
            "training_time_seconds": self.training_time_seconds,
            "early_stopped": self.early_stopped,
            "best_epoch": self.best_epoch,
            "history": self.history,
            "metadata": self.metadata,
        }


# =============================================================================
# BASE MODEL INTERFACE
# =============================================================================

class BaseModel(ABC):
    """
    Abstract base class for all models in the factory.

    All model implementations must inherit from this class and implement
    the abstract methods. This ensures:
    1. Consistent training/prediction interface
    2. Proper model serialization
    3. Unified evaluation across model types
    4. Ensemble compatibility

    Subclasses must implement:
        - model_family (property): Model family classification
        - requires_scaling (property): Whether scaling is needed
        - requires_sequences (property): Whether sequential input is needed
        - get_default_config(): Default hyperparameters
        - fit(): Training logic
        - predict(): Prediction logic
        - save(): Model persistence
        - load(): Model loading

    Optional overrides:
        - get_feature_importance(): For interpretable models

    Example:
        >>> @ModelRegistry.register("my_model", family="boosting")
        ... class MyModel(BaseModel):
        ...     @property
        ...     def model_family(self) -> str:
        ...         return "boosting"
        ...
        ...     def fit(self, X_train, y_train, X_val, y_val, ...):
        ...         # Training implementation
        ...         return TrainingMetrics(...)
        ...
        ...     def predict(self, X) -> PredictionOutput:
        ...         # Prediction implementation
        ...         return PredictionOutput(...)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize the model.

        Args:
            config: Model configuration dict. If None, uses defaults
                   from get_default_config().
        """
        self._config = self._merge_config(config)
        self._is_fitted = False

    def _merge_config(self, config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Merge provided config with defaults."""
        defaults = self.get_default_config()
        if config is None:
            return defaults
        merged = defaults.copy()
        merged.update(config)
        return merged

    @property
    def config(self) -> Dict[str, Any]:
        """Current model configuration."""
        return self._config

    @property
    def is_fitted(self) -> bool:
        """Whether the model has been trained."""
        return self._is_fitted

    # =========================================================================
    # ABSTRACT PROPERTIES (must override)
    # =========================================================================

    @property
    @abstractmethod
    def model_family(self) -> str:
        """
        Return model family classification.

        Returns:
            One of: 'boosting', 'neural', 'transformer', 'classical', 'ensemble'
        """
        pass

    @property
    @abstractmethod
    def requires_scaling(self) -> bool:
        """
        Whether this model requires feature scaling.

        Returns:
            True if features should be scaled before training
        """
        pass

    @property
    @abstractmethod
    def requires_sequences(self) -> bool:
        """
        Whether this model requires sequential input.

        Returns:
            True if input should be shaped as (n_samples, seq_len, n_features)
        """
        pass

    # =========================================================================
    # ABSTRACT METHODS (must implement)
    # =========================================================================

    @abstractmethod
    def get_default_config(self) -> Dict[str, Any]:
        """
        Return default hyperparameters for this model.

        Returns:
            Dictionary of default configuration values

        Example:
            >>> model.get_default_config()
            {'n_estimators': 100, 'max_depth': 6, 'learning_rate': 0.1}
        """
        pass

    @abstractmethod
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        sample_weights: Optional[np.ndarray] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> TrainingMetrics:
        """
        Train the model.

        Args:
            X_train: Training features
                - Shape (n_samples, n_features) for non-sequential
                - Shape (n_samples, seq_len, n_features) for sequential
            y_train: Training labels, shape (n_samples,)
            X_val: Validation features (same shape as X_train)
            y_val: Validation labels, shape (n_val_samples,)
            sample_weights: Optional sample weights, shape (n_samples,)
            config: Optional config overrides for this training run

        Returns:
            TrainingMetrics with training results

        Raises:
            ValueError: If input shapes are invalid
        """
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> PredictionOutput:
        """
        Generate predictions.

        Args:
            X: Features to predict
                - Shape (n_samples, n_features) for non-sequential
                - Shape (n_samples, seq_len, n_features) for sequential

        Returns:
            PredictionOutput with predictions and probabilities

        Raises:
            RuntimeError: If model is not fitted
            ValueError: If input shape is invalid
        """
        pass

    @abstractmethod
    def save(self, path: Path) -> None:
        """
        Save model to disk.

        Args:
            path: Directory or file path to save to

        Raises:
            RuntimeError: If model is not fitted
            IOError: If save fails
        """
        pass

    @abstractmethod
    def load(self, path: Path) -> None:
        """
        Load model from disk.

        Args:
            path: Directory or file path to load from

        Raises:
            FileNotFoundError: If path doesn't exist
            IOError: If load fails
        """
        pass

    # =========================================================================
    # OPTIONAL METHODS (can override)
    # =========================================================================

    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """
        Return feature importances if available.

        Tree-based models (XGBoost, LightGBM, RandomForest) should
        override this to return feature importance scores.

        Returns:
            Dict mapping feature names to importance scores,
            or None if not available
        """
        return None

    def get_attention_weights(self) -> Optional[np.ndarray]:
        """
        Return attention weights if available.

        Transformer models should override this to return
        attention patterns for interpretability.

        Returns:
            Attention weights array or None if not available
        """
        return None

    # =========================================================================
    # VALIDATION HELPERS
    # =========================================================================

    def _validate_input_shape(
        self,
        X: np.ndarray,
        context: str = "input"
    ) -> None:
        """
        Validate input array shape.

        Args:
            X: Input array to validate
            context: Context for error messages

        Raises:
            ValueError: If shape is invalid for this model type
        """
        if X.ndim == 1:
            raise ValueError(
                f"{context} must be 2D or 3D, got 1D array with shape {X.shape}"
            )

        if self.requires_sequences:
            if X.ndim != 3:
                raise ValueError(
                    f"{context} must be 3D (n_samples, seq_len, n_features) "
                    f"for sequential models, got shape {X.shape}"
                )
        else:
            if X.ndim != 2:
                raise ValueError(
                    f"{context} must be 2D (n_samples, n_features) "
                    f"for non-sequential models, got shape {X.shape}"
                )

    def _validate_fitted(self) -> None:
        """
        Check if model is fitted.

        Raises:
            RuntimeError: If model is not fitted
        """
        if not self._is_fitted:
            raise RuntimeError(
                f"{self.__class__.__name__} is not fitted. Call fit() first."
            )

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"{self.__class__.__name__}("
            f"family={self.model_family}, "
            f"fitted={self._is_fitted})"
        )


__all__ = [
    "PredictionOutput",
    "TrainingMetrics",
    "BaseModel",
]
