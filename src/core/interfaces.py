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
from typing import Any, Protocol, TypeVar, runtime_checkable

import numpy as np
import pandas as pd

# =============================================================================
# RESULT DATACLASSES
# =============================================================================

# NOTE: AdapterResult is defined in TWO locations (DOCUMENTED EXCEPTION):
#   1. src.data.adapters.base (canonical, uses X/y ML conventions)
#   2. Here (legacy, uses data/labels conventions)
#
# This is INTENTIONAL to avoid circular imports between core and data.adapters.
# Both definitions are kept in sync via backward-compatibility properties:
#   - adapters/base.py: primary fields X/y, with .data/.labels as aliases
#   - interfaces.py: primary fields data/labels, with .X/.y as aliases
#
# Import paths:
#   - Adapters: import from src.data.adapters.base
#   - Other code: import from src.core.interfaces (this file)
#
# Phase 27: Verified as documented exception - not a consolidation target


@dataclass
class AdapterResult:
    """
    Output from any adapter transformation.

    Adapters transform raw DataFrames into model-ready tensors (2D/3D/4D).
    This dataclass standardizes the output format.

    NOTE: This is the LEGACY definition. The canonical version is in
    src.data.adapters.base which uses X/y (ML conventions) instead of
    data/labels. Both versions provide backward-compatible properties.

    Attributes:
        data: Transformed features (2D, 3D, or 4D numpy array) [alias: X]
        labels: Target labels (1D array) [alias: y]
        feature_names: List of feature column names [alias: feature_columns]
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
        return int(self.data.shape[0])

    @property
    def n_features(self) -> int:
        """Number of features (last dimension)."""
        return int(self.data.shape[-1])

    @property
    def rank(self) -> int:
        """Data tensor rank (2, 3, or 4)."""
        return self.data.ndim

    @property
    def shape(self) -> tuple[int, ...]:
        """Full data shape."""
        return self.data.shape

    # Backward compatibility with base.py (ML conventions)
    @property
    def X(self) -> np.ndarray:
        """Alias for data (backward compatibility with adapters.base)."""
        return self.data

    @property
    def y(self) -> np.ndarray:
        """Alias for labels (backward compatibility with adapters.base)."""
        return self.labels

    @property
    def feature_columns(self) -> list[str]:
        """Alias for feature_names (backward compatibility with adapters.base)."""
        return self.feature_names

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
    Standardized prediction result for all models.

    All models must return predictions in this format to enable
    unified evaluation and ensemble composition.

    This is the CANONICAL definition - consolidated from models/base.py,
    core/interfaces.py, and inference/orchestrator.py (Phase 27).

    Core Attributes:
        class_predictions: Predicted class labels, shape (n_samples,)
        class_probabilities: Class probabilities, shape (n_samples, n_classes)
        confidence: Prediction confidence (max probability), shape (n_samples,)
        metadata: Model-specific metadata (feature importance, attention, etc.)

    Optional Inference Attributes:
        indices: Original indices for alignment
        model_name: Name of the model used for prediction
        horizon: Prediction horizon in bars
        inference_time_ms: Time taken for inference in milliseconds
        is_ensemble: Whether ensemble was used for prediction
        individual_predictions: Dict of predictions from individual models

    Example:
        >>> result = model.predict(X_test)
        >>> print(result.class_predictions.shape)  # (1000,)
        >>> print(result.class_probabilities.shape)  # (1000, 3)
        >>> print(result.confidence.mean())  # 0.65
    """

    # Core fields (required)
    class_predictions: np.ndarray
    class_probabilities: np.ndarray
    confidence: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)

    # Optional fields for alignment
    indices: np.ndarray | None = None

    # Optional inference fields
    model_name: str | None = None
    horizon: int | None = None
    inference_time_ms: float | None = None
    is_ensemble: bool = False
    individual_predictions: dict[str, np.ndarray] | None = None

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
        return int(self.class_probabilities.shape[1])

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = {
            "class_predictions": self.class_predictions.tolist(),
            "class_probabilities": self.class_probabilities.tolist(),
            "confidence": self.confidence.tolist(),
            "metadata": self.metadata,
        }
        # Include optional fields if set
        if self.indices is not None:
            result["indices"] = self.indices.tolist()
        if self.model_name is not None:
            result["model_name"] = self.model_name
        if self.horizon is not None:
            result["horizon"] = self.horizon
        if self.inference_time_ms is not None:
            result["inference_time_ms"] = self.inference_time_ms
        if self.is_ensemble:
            result["is_ensemble"] = self.is_ensemble
        if self.individual_predictions is not None:
            result["individual_predictions"] = {
                k: v.tolist() for k, v in self.individual_predictions.items()
            }
        return result

    def to_dataframe(self) -> pd.DataFrame:
        """Convert to DataFrame with predictions (for inference)."""
        df = pd.DataFrame(
            {
                "prediction": self.class_predictions,
                "confidence": self.confidence,
            }
        )
        # Add probability columns if 3-class
        if self.n_classes == 3:
            df["prob_short"] = self.class_probabilities[:, 0]
            df["prob_neutral"] = self.class_probabilities[:, 1]
            df["prob_long"] = self.class_probabilities[:, 2]
        return df

    def summary(self) -> str:
        """Get human-readable summary string."""
        lines = [
            f"PredictionResult: {self.model_name or 'unnamed'}",
            f"  Samples: {self.n_samples}",
            f"  Classes: {self.n_classes}",
            f"  Mean confidence: {float(self.confidence.mean()):.3f}",
        ]
        if self.horizon is not None:
            lines.append(f"  Horizon: {self.horizon}")
        if self.inference_time_ms is not None:
            lines.append(f"  Inference time: {self.inference_time_ms:.1f}ms")
        if self.is_ensemble:
            lines.append("  Is ensemble: True")
        return "\n".join(lines)


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
        return int(self.probabilities.shape[1])

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

# NOTE: DataContract ABC was removed in Phase 27 - it was dead code.
# The canonical DataContract is the dataclass in src/core/contracts/data_contract.py
# which is used for lineage tracking and schema validation throughout the pipeline.


# NOTE: ModelContract ABC was removed in Phase 27 - it was dead code.
# The canonical ModelContract is the frozen dataclass in src/core/contracts/model_contract.py
# which describes model data requirements (input_rank, sequence_length, feature_modes).
# Model implementations inherit from BaseModel in src/models/base.py instead.
#
# Re-export ModelContract and DataContract from contracts for backward compatibility
from src.core.contracts import DataContract, ModelContract


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
