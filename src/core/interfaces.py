"""
Core interfaces - Abstract base classes for the ML Factory.

PHASE_0: This defines the contracts that all models, adapters, and data handlers
must implement. This ensures consistent behavior across the entire system.

Contracts:
- DataContract: Specifies model data requirements
- ModelContract: Standardized model interface (fit/predict/save/load)
- AdapterContract: Data transformation interface

Result Types:
- AdapterResult: Output from adapter transformation
- TrainingResult: Output from model training
- OOFResult: Out-of-fold predictions with alignment info
- PredictionResult: Model prediction output
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, TypeVar, runtime_checkable

import numpy as np
import pandas as pd

# =============================================================================
# RESULT DATACLASSES
# =============================================================================


@dataclass
class AdapterResult:
    """
    Output from any adapter transformation.

    Adapters transform raw DataFrames into model-ready tensors (2D/3D/4D).
    This dataclass standardizes the output format.

    Attributes:
        data: Transformed features (2D, 3D, or 4D numpy array)
        labels: Target labels (1D array)
        feature_names: List of feature column names
        original_indices: Original DataFrame indices for OOF alignment
        weights: Optional sample weights
        metadata: Additional adapter-specific metadata
    """

    data: np.ndarray  # Shape: (n, f) or (n, s, f) or (n, t, s, f)
    labels: np.ndarray  # Shape: (n,)
    feature_names: list[str]
    original_indices: np.ndarray  # For OOF alignment
    weights: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def n_samples(self) -> int:
        """Number of samples in the dataset."""
        return self.data.shape[0]

    @property
    def n_features(self) -> int:
        """Number of features (last dimension)."""
        return self.data.shape[-1]

    @property
    def rank(self) -> int:
        """Data tensor rank (2, 3, or 4)."""
        return self.data.ndim

    @property
    def shape(self) -> tuple:
        """Full data shape."""
        return self.data.shape

    def validate(self) -> None:
        """Validate the adapter result."""
        if self.data.size == 0:
            raise ValueError("AdapterResult data is empty")
        if len(self.labels) != self.n_samples:
            raise ValueError(f"Labels length ({len(self.labels)}) != n_samples ({self.n_samples})")
        if len(self.original_indices) != self.n_samples:
            raise ValueError("original_indices length != n_samples")
        if np.isnan(self.data).any():
            raise ValueError("AdapterResult data contains NaN values")


@dataclass
class PredictionResult:
    """
    Output from model prediction.

    Attributes:
        class_predictions: Predicted class labels (1D int array)
        class_probabilities: Probability per class (2D array: n_samples x n_classes)
        confidence: Prediction confidence (max probability)
        indices: Original indices for alignment
    """

    class_predictions: np.ndarray  # Shape: (n,) - predicted labels
    class_probabilities: np.ndarray  # Shape: (n, n_classes)
    indices: np.ndarray | None = None

    @property
    def n_samples(self) -> int:
        return len(self.class_predictions)

    @property
    def n_classes(self) -> int:
        return self.class_probabilities.shape[1]

    @property
    def confidence(self) -> np.ndarray:
        """Maximum probability per sample."""
        return self.class_probabilities.max(axis=1)


@dataclass
class TrainingResult:
    """
    Output from model training.

    Attributes:
        model: The trained model instance
        metrics: Training/validation metrics dict
        oof_predictions: Out-of-fold predictions (optional)
        feature_importance: Feature importance dict (optional)
        training_time_seconds: Total training time
        metadata: Additional training metadata
    """

    model: Any
    metrics: dict[str, float]
    oof_predictions: np.ndarray | None = None
    feature_importance: dict[str, float] | None = None
    training_time_seconds: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def has_oof(self) -> bool:
        return self.oof_predictions is not None


@dataclass
class OOFResult:
    """
    Out-of-fold predictions with alignment info.

    Used for stacking/blending where we need aligned OOF predictions
    from heterogeneous models (different adapters, different sample coverage).

    Attributes:
        predictions: Predicted class labels
        probabilities: Class probabilities (n_samples x n_classes)
        indices: Original indices for alignment
        fold_ids: Which fold each prediction came from
        model_name: Name of the model that produced these predictions
        coverage: Fraction of total samples covered (may be < 1.0 for sequence models)
    """

    predictions: np.ndarray  # Shape: (n,)
    probabilities: np.ndarray  # Shape: (n, n_classes)
    indices: np.ndarray  # Original DataFrame indices
    fold_ids: np.ndarray  # Which fold each prediction came from
    model_name: str = ""
    coverage: float = 1.0

    @property
    def n_samples(self) -> int:
        return len(self.predictions)

    @property
    def n_classes(self) -> int:
        return self.probabilities.shape[1]

    def align_to(self, target_indices: np.ndarray) -> OOFResult:
        """
        Align this OOF result to a target index set.

        Returns a new OOFResult with predictions aligned to target_indices.
        Missing indices will have NaN probabilities.
        """
        aligned_probs = np.full((len(target_indices), self.n_classes), np.nan)
        aligned_preds = np.full(len(target_indices), -999, dtype=int)
        aligned_folds = np.full(len(target_indices), -1, dtype=int)

        # Find matching indices
        idx_map = {idx: i for i, idx in enumerate(target_indices)}
        for i, idx in enumerate(self.indices):
            if idx in idx_map:
                j = idx_map[idx]
                aligned_probs[j] = self.probabilities[i]
                aligned_preds[j] = self.predictions[i]
                aligned_folds[j] = self.fold_ids[i]

        return OOFResult(
            predictions=aligned_preds,
            probabilities=aligned_probs,
            indices=target_indices,
            fold_ids=aligned_folds,
            model_name=self.model_name,
            coverage=np.sum(~np.isnan(aligned_probs[:, 0])) / len(target_indices),
        )


# =============================================================================
# PROTOCOLS - For structural typing without circular imports
# =============================================================================


@runtime_checkable
class OOFPredictionProtocol(Protocol):
    """
    Protocol for OOFPrediction objects - enables type checking without
    importing the actual class, breaking circular dependencies.

    This matches the interface of src.cross_validation.oof_core.OOFPrediction.
    Used in heterogeneous_stacking.py to avoid:
        cross_validation -> models -> models.ensemble -> heterogeneous_stacking
        -> oof_core (CIRCULAR)
    """

    model_name: str
    predictions: pd.DataFrame
    fold_info: list[dict[str, Any]]
    coverage: float
    original_indices: np.ndarray | None
    sequence_length: int | None
    n_total_samples: int | None

    @property
    def n_valid(self) -> int:
        """Number of samples with valid predictions."""
        ...

    @property
    def alignment_offset(self) -> int:
        """Offset from start of dataset to first valid prediction."""
        ...

    def get_probabilities(self) -> np.ndarray:
        """Get probability matrix (n_samples, 3)."""
        ...

    def get_class_predictions(self) -> np.ndarray:
        """Get predicted classes (-1, 0, 1)."""
        ...


# =============================================================================
# ABSTRACT CONTRACTS
# =============================================================================


class DataContract(ABC):
    """
    Contract for model data requirements.

    Every model must specify its data requirements through this contract.
    The adapter uses this to prepare data in the correct format.

    Example:
        class XGBoostContract(DataContract):
            rank = 2
            required_features = ["momentum", "volatility"]
            feature_bounds = (20, 200)
    """

    @property
    @abstractmethod
    def rank(self) -> int:
        """
        Data dimensionality: 2=tabular, 3=sequence, 4=multi-stream.
        """
        pass

    @property
    @abstractmethod
    def required_features(self) -> list[str]:
        """
        Minimum required feature families.

        Returns list of FeatureFamily values that must be present.
        """
        pass

    @property
    @abstractmethod
    def feature_bounds(self) -> tuple:
        """
        (min_features, max_features) tuple.

        Defines acceptable feature count range for this model.
        """
        pass

    @property
    def sequence_length(self) -> int | None:
        """Sequence length for 3D/4D models. None for 2D models."""
        return None if self.rank == 2 else 60  # Default: 60 bars


class ModelContract(ABC):
    """
    Contract all models must implement.

    This ensures consistent interface across all model types:
    - Boosting (XGBoost, LightGBM, CatBoost)
    - Classical (RandomForest, Logistic, SVM)
    - Neural (LSTM, GRU, TCN, Transformer, etc.)
    - Meta-learners (Ridge, MLP, XGBoost meta)

    Key methods:
    - fit(): Train the model
    - predict(): Get class predictions
    - predict_proba(): Get probability estimates
    - save()/load(): Serialization
    """

    @abstractmethod
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        sample_weights: np.ndarray | None = None,
        **kwargs: Any,
    ) -> TrainingResult:
        """
        Train the model.

        Args:
            X_train: Training features (2D/3D/4D based on model)
            y_train: Training labels (1D)
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            sample_weights: Sample weights (optional)
            **kwargs: Model-specific arguments

        Returns:
            TrainingResult with trained model and metrics
        """
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Get class predictions.

        Args:
            X: Features (same rank as training data)

        Returns:
            1D array of predicted class labels
        """
        pass

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Get probability estimates.

        Args:
            X: Features (same rank as training data)

        Returns:
            2D array of shape (n_samples, n_classes)
        """
        pass

    @abstractmethod
    def save(self, path: Path) -> None:
        """
        Save model to disk.

        Args:
            path: Directory path to save model artifacts
        """
        pass

    @classmethod
    @abstractmethod
    def load(cls, path: Path) -> ModelContract:
        """
        Load model from disk.

        Args:
            path: Directory path containing model artifacts

        Returns:
            Loaded model instance
        """
        pass

    @property
    @abstractmethod
    def data_contract(self) -> DataContract:
        """
        Get the data contract for this model.

        Returns:
            DataContract specifying data requirements
        """
        pass

    @property
    def name(self) -> str:
        """Model name (class name by default)."""
        return self.__class__.__name__

    @property
    def is_fitted(self) -> bool:
        """Whether the model has been trained."""
        return hasattr(self, "_is_fitted") and self._is_fitted


class AdapterContract(ABC):
    """
    Contract all adapters must implement.

    Adapters transform raw DataFrames into model-ready tensors.
    Three adapter types:
    - TabularAdapter: 2D (n_samples, n_features)
    - SequenceAdapter: 3D (n_samples, seq_len, n_features)
    - MultiStreamAdapter: 4D (n_samples, n_timeframes, seq_len, n_features)
    """

    @abstractmethod
    def transform(
        self,
        df: pd.DataFrame,
        contract: DataContract,
        label_column: str = "label",
    ) -> AdapterResult:
        """
        Transform DataFrame to model-ready tensors.

        Args:
            df: Input DataFrame with features and labels
            contract: Data contract specifying requirements
            label_column: Name of the label column

        Returns:
            AdapterResult with transformed data
        """
        pass

    @property
    @abstractmethod
    def output_rank(self) -> int:
        """Output tensor rank (2, 3, or 4)."""
        pass

    @property
    def name(self) -> str:
        """Adapter name (class name by default)."""
        return self.__class__.__name__


# =============================================================================
# TYPE VARIABLES
# =============================================================================

ModelType = TypeVar("ModelType", bound=ModelContract)
AdapterType = TypeVar("AdapterType", bound=AdapterContract)


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Result types
    "AdapterResult",
    "PredictionResult",
    "TrainingResult",
    "OOFResult",
    # Protocols (for structural typing without circular imports)
    "OOFPredictionProtocol",
    # Contracts
    "DataContract",
    "ModelContract",
    "AdapterContract",
    # Type variables
    "ModelType",
    "AdapterType",
]
