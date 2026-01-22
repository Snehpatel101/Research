"""
Fold-Aware Scaling for Cross-Validation.

Ensures each CV fold's scaler is fit only on training indices,
preventing information leakage from validation/test data.

The Problem:
    Global scaling (fitting scaler on entire train set) leaks future
    statistics into each fold's validation data. When we later evaluate
    OOF predictions, metrics are inflated by 5-15%.

The Solution:
    For each fold, fit a fresh scaler on ONLY the fold's training data,
    then transform both train and validation using those statistics.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler, StandardScaler

logger = logging.getLogger(__name__)


@dataclass
class FoldScalingResult:
    """Result from fold-aware scaling."""

    X_train_scaled: np.ndarray
    X_val_scaled: np.ndarray
    scaler: RobustScaler | StandardScaler
    method: str
    n_features: int


class FoldAwareScaler:
    """
    Scaler that fits on fold training data only.

    Prevents CV leakage by ensuring validation data is transformed
    using only statistics from the training portion of each fold.

    Example:
        >>> scaler = FoldAwareScaler(method="robust")
        >>> result = scaler.fit_transform_fold(X_train, X_val)
        >>> # result.X_train_scaled and result.X_val_scaled are ready

    Note:
        Each call to fit_transform_fold creates a fresh scaler instance,
        ensuring no state is carried between folds.
    """

    def __init__(
        self,
        method: str = "robust",
        clip_outliers: bool = True,
        clip_std: float = 5.0,
    ) -> None:
        """
        Initialize FoldAwareScaler.

        Args:
            method: Scaling method ("robust", "standard", "none")
                - robust: RobustScaler (median/IQR, recommended for trading)
                - standard: StandardScaler (mean/std)
                - none: No scaling (for tree-based models)
            clip_outliers: Whether to clip extreme values after scaling
            clip_std: Number of standard deviations for clipping
        """
        if method not in ("robust", "standard", "none"):
            raise ValueError(
                f"Unknown scaling method: {method}. Use 'robust', 'standard', or 'none'"
            )

        self.method = method
        self.clip_outliers = clip_outliers
        self.clip_std = clip_std
        self._current_scaler = None

    def fit_transform_fold(
        self,
        X_train: np.ndarray,
        X_val: np.ndarray,
    ) -> FoldScalingResult:
        """
        Fit on training data and transform both train and validation.

        Creates a fresh scaler instance for this fold - no state is
        carried between folds.

        Args:
            X_train: Training features, shape (n_train, n_features)
            X_val: Validation features, shape (n_val, n_features)

        Returns:
            FoldScalingResult with scaled arrays and scaler reference
        """
        if self.method == "none":
            # No scaling needed (e.g., tree-based models)
            return FoldScalingResult(
                X_train_scaled=X_train,
                X_val_scaled=X_val,
                scaler=None,
                method="none",
                n_features=X_train.shape[1] if X_train.ndim > 1 else 1,
            )

        # Create fresh scaler for this fold
        if self.method == "robust":
            scaler = RobustScaler()
        else:  # standard
            scaler = StandardScaler()

        # Fit ONLY on training data
        X_train_scaled = scaler.fit_transform(X_train)

        # Transform validation using training statistics
        X_val_scaled = scaler.transform(X_val)

        # Optional outlier clipping
        if self.clip_outliers:
            X_train_scaled = np.clip(X_train_scaled, -self.clip_std, self.clip_std)
            X_val_scaled = np.clip(X_val_scaled, -self.clip_std, self.clip_std)

        self._current_scaler = scaler

        return FoldScalingResult(
            X_train_scaled=X_train_scaled,
            X_val_scaled=X_val_scaled,
            scaler=scaler,
            method=self.method,
            n_features=X_train.shape[1] if X_train.ndim > 1 else 1,
        )

    def fit_transform_fold_df(
        self,
        X_train: pd.DataFrame,
        X_val: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        DataFrame-preserving version of fit_transform_fold.

        Args:
            X_train: Training DataFrame
            X_val: Validation DataFrame

        Returns:
            Tuple of (X_train_scaled_df, X_val_scaled_df)
        """
        result = self.fit_transform_fold(X_train.values, X_val.values)

        X_train_scaled_df = pd.DataFrame(
            result.X_train_scaled,
            columns=X_train.columns,
            index=X_train.index,
        )
        X_val_scaled_df = pd.DataFrame(
            result.X_val_scaled,
            columns=X_val.columns,
            index=X_val.index,
        )

        return X_train_scaled_df, X_val_scaled_df

    @property
    def current_scaler(self) -> RobustScaler | StandardScaler | None:
        """Get the most recently fit scaler (for inspection/serialization)."""
        return self._current_scaler


def scale_cv_fold(
    X: pd.DataFrame | np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    method: str = "robust",
    clip_outliers: bool = True,
    clip_std: float = 5.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convenience function to scale a CV fold.

    Args:
        X: Full feature matrix (DataFrame or ndarray)
        train_idx: Training indices
        val_idx: Validation indices
        method: Scaling method ("robust", "standard", "none")
        clip_outliers: Whether to clip outliers
        clip_std: Clipping threshold in standard deviations

    Returns:
        Tuple of (X_train_scaled, X_val_scaled) as numpy arrays
    """
    if isinstance(X, pd.DataFrame):
        X_train = X.iloc[train_idx].values
        X_val = X.iloc[val_idx].values
    else:
        X_train = X[train_idx]
        X_val = X[val_idx]

    scaler = FoldAwareScaler(
        method=method,
        clip_outliers=clip_outliers,
        clip_std=clip_std,
    )
    result = scaler.fit_transform_fold(X_train, X_val)

    return result.X_train_scaled, result.X_val_scaled


def get_scaling_method_for_model(model_name: str) -> str:
    """
    Get appropriate scaling method based on model type.

    Args:
        model_name: Name of the model

    Returns:
        Scaling method: "robust", "standard", or "none"
    """
    from src.models.registry import ModelRegistry

    try:
        model_info = ModelRegistry.get_model_info(model_name)
        requires_scaling = model_info.get("requires_scaling", True)

        if not requires_scaling:
            return "none"

        # Neural networks may prefer standard scaling
        if model_info.get("family") in ("neural", "transformer"):
            return "standard"

        # Default to robust for others (better for outliers in trading data)
        return "robust"

    except ValueError:
        # Unknown model, default to robust
        logger.warning(f"Unknown model '{model_name}', defaulting to robust scaling")
        return "robust"


__all__ = [
    "FoldAwareScaler",
    "FoldScalingResult",
    "scale_cv_fold",
    "get_scaling_method_for_model",
]
