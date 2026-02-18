"""
Data scaling for adapter outputs.

PHASE_2: Scale adapter outputs before model training.
Fits on train, transforms all sets.

This module provides a unified scaling interface that:
- Fits on training data only (prevents data leakage)
- Transforms all splits consistently
- Handles 2D, 3D, and 4D arrays from different adapters
- Supports save/load for inference pipelines
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.preprocessing import MinMaxScaler as SKMinMaxScaler
from sklearn.preprocessing import RobustScaler as SKRobustScaler
from sklearn.preprocessing import StandardScaler as SKStandardScaler

from src.core.utils.safe_pickle import safe_pickle_load

logger = logging.getLogger(__name__)


@dataclass
class ScalerConfig:
    """
    Configuration for data scaling.

    Attributes:
        method: Scaling method - "robust", "standard", "minmax", or "none"
        clip_value: Clip scaled values to [-clip, clip], 0 disables clipping
        with_centering: For RobustScaler, whether to center data (subtract median)
        with_scaling: For RobustScaler, whether to scale data (divide by IQR)
        quantile_range: For RobustScaler, the quantile range used to calculate scale
        feature_range: For MinMaxScaler, the target range for scaled features
    """

    method: str = "robust"  # "robust", "standard", "minmax", "none"
    clip_value: float = 5.0  # Clip scaled values to [-clip, clip]
    with_centering: bool = True
    with_scaling: bool = True
    quantile_range: tuple[float, float] = (25.0, 75.0)
    feature_range: tuple[float, float] = (0.0, 1.0)  # For MinMaxScaler

    def to_dict(self) -> dict[str, Any]:
        """Convert config to dictionary for serialization."""
        return {
            "method": self.method,
            "clip_value": self.clip_value,
            "with_centering": self.with_centering,
            "with_scaling": self.with_scaling,
            "quantile_range": list(self.quantile_range),
            "feature_range": list(self.feature_range),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ScalerConfig:
        """Create config from dictionary."""
        return cls(
            method=d.get("method", "robust"),
            clip_value=d.get("clip_value", 5.0),
            with_centering=d.get("with_centering", True),
            with_scaling=d.get("with_scaling", True),
            quantile_range=tuple(d.get("quantile_range", [25.0, 75.0])),
            feature_range=tuple(d.get("feature_range", [0.0, 1.0])),
        )


class AdapterScaler:
    """
    Scaler for adapter outputs.

    Fits on training data only, transforms all splits.
    Handles 2D, 3D, and 4D arrays from different adapters.

    Design:
    - RobustScaler by default (handles outliers well for financial data)
    - Reshapes N-D arrays to 2D for sklearn, then restores shape
    - Optional clipping to prevent extreme values
    - Serializable for inference pipelines

    Example:
        >>> scaler = AdapterScaler(ScalerConfig(method="robust"))
        >>> X_train_scaled = scaler.fit_transform(X_train)  # (n, seq, feat)
        >>> X_val_scaled = scaler.transform(X_val)
        >>> X_test_scaled = scaler.transform(X_test)
        >>> scaler.save(Path("./scalers/model_a"))
    """

    # Valid scaling methods
    VALID_METHODS = ("robust", "standard", "minmax", "none")

    def __init__(self, config: ScalerConfig | None = None):
        """
        Initialize the scaler.

        Args:
            config: Scaling configuration. Defaults to RobustScaler with
                    standard settings and clipping at +/- 5.0.
        """
        self.config = config or ScalerConfig()
        self._scaler: SKRobustScaler | SKStandardScaler | SKMinMaxScaler | None = None
        self._is_fitted = False
        self._n_features: int = 0
        self._original_shape: tuple | None = None

        # Validate method
        if self.config.method not in self.VALID_METHODS:
            raise ValueError(
                f"Invalid scaling method: {self.config.method}. "
                f"Valid methods: {self.VALID_METHODS}"
            )

    @property
    def is_fitted(self) -> bool:
        """Check if scaler has been fitted."""
        return self._is_fitted

    @property
    def n_features(self) -> int:
        """Number of features the scaler was fitted on."""
        return self._n_features

    def fit(self, X: np.ndarray) -> AdapterScaler:
        """
        Fit scaler on training data.

        Args:
            X: Training data (2D, 3D, or 4D)
               - 2D: (n_samples, n_features) - tabular
               - 3D: (n_samples, seq_len, n_features) - sequence
               - 4D: (n_samples, n_timeframes, seq_len, n_features) - multi-stream

        Returns:
            Self for method chaining.

        Raises:
            ValueError: If array has unsupported number of dimensions.
        """
        if self.config.method == "none":
            self._is_fitted = True
            logger.debug("Scaler method is 'none', skipping fit")
            return self

        self._original_shape = X.shape
        X_2d = self._to_2d(X)
        self._n_features = X_2d.shape[1]

        # Create appropriate scaler based on method
        if self.config.method == "robust":
            self._scaler = SKRobustScaler(
                with_centering=self.config.with_centering,
                with_scaling=self.config.with_scaling,
                quantile_range=self.config.quantile_range,
            )
        elif self.config.method == "standard":
            self._scaler = SKStandardScaler()
        elif self.config.method == "minmax":
            self._scaler = SKMinMaxScaler(feature_range=self.config.feature_range)
        else:
            raise ValueError(f"Unknown scaling method: {self.config.method}")

        self._scaler.fit(X_2d)
        self._is_fitted = True

        logger.debug(
            f"AdapterScaler fitted: method={self.config.method}, "
            f"n_features={self._n_features}, shape={X.shape}"
        )

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform data using fitted scaler.

        Args:
            X: Data to transform (2D, 3D, or 4D)

        Returns:
            Scaled data with same shape as input.

        Raises:
            ValueError: If scaler not fitted or array has wrong dimensions.
        """
        if self.config.method == "none":
            return X

        if not self._is_fitted:
            raise ValueError("Scaler not fitted. Call fit() first.")

        if self._scaler is None:
            raise ValueError("Scaler not initialized. Call fit() first.")

        original_shape = X.shape
        X_2d = self._to_2d(X)

        # Validate feature count
        if X_2d.shape[1] != self._n_features:
            raise ValueError(
                f"Feature count mismatch: scaler fitted on {self._n_features} features, "
                f"but input has {X_2d.shape[1]} features"
            )

        X_scaled = self._scaler.transform(X_2d)

        # Replace NaN/Inf from zero-variance features (IQR=0 or std=0)
        # These arise when a feature is constant — scaling divides by zero.
        # Replace with 0.0 (centered, no signal) instead of propagating NaN.
        nan_mask = ~np.isfinite(X_scaled)
        if nan_mask.any():
            n_bad = int(nan_mask.sum())
            logger.warning(
                f"Scaler produced {n_bad} non-finite values "
                f"(likely zero-variance features). Replacing with 0.0."
            )
            X_scaled[nan_mask] = 0.0

        # Clip values if configured
        if self.config.clip_value > 0:
            X_scaled = np.clip(X_scaled, -self.config.clip_value, self.config.clip_value)

        return self._from_2d(X_scaled, original_shape)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Fit and transform in one step.

        Args:
            X: Training data to fit and transform.

        Returns:
            Scaled training data.
        """
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Inverse transform scaled data back to original scale.

        Args:
            X: Scaled data (2D, 3D, or 4D)

        Returns:
            Data in original scale with same shape as input.

        Raises:
            ValueError: If scaler not fitted.
        """
        if self.config.method == "none":
            return X

        if not self._is_fitted or self._scaler is None:
            raise ValueError("Scaler not fitted. Call fit() first.")

        original_shape = X.shape
        X_2d = self._to_2d(X)

        # Note: inverse_transform does not "un-clip", so values that were
        # clipped during transform will remain at the clip boundaries
        X_unscaled = self._scaler.inverse_transform(X_2d)

        return self._from_2d(X_unscaled, original_shape)

    def _to_2d(self, X: np.ndarray) -> np.ndarray:
        """
        Reshape N-D array to 2D for sklearn scaler.

        Args:
            X: Input array (2D, 3D, or 4D)

        Returns:
            2D array with shape (n_total_samples, n_features)

        Raises:
            ValueError: If array has unsupported number of dimensions.
        """
        if X.ndim == 2:
            # (n_samples, n_features) -> no change needed
            return X
        elif X.ndim == 3:
            # (n_samples, seq_len, n_features) -> (n_samples * seq_len, n_features)
            n, seq, feat = X.shape
            return X.reshape(-1, feat)
        elif X.ndim == 4:
            # (n_samples, n_tf, seq_len, n_features) -> (n_samples * n_tf * seq_len, n_features)
            n, tf, seq, feat = X.shape
            return X.reshape(-1, feat)
        else:
            raise ValueError(f"Unsupported array rank: {X.ndim}. " f"Expected 2D, 3D, or 4D array.")

    def _from_2d(self, X_2d: np.ndarray, original_shape: tuple) -> np.ndarray:
        """
        Reshape 2D array back to original shape.

        Args:
            X_2d: 2D array from scaler
            original_shape: Original shape to restore

        Returns:
            Array with original shape.
        """
        return X_2d.reshape(original_shape)

    def save(self, path: Path) -> None:
        """
        Save scaler to disk.

        Creates a directory with:
        - scaler.pkl: The sklearn scaler object
        - scaler_config.json: Configuration and metadata

        Args:
            path: Directory path to save scaler artifacts.
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save sklearn scaler if it exists
        if self._scaler is not None:
            scaler_path = path / "scaler.pkl"
            joblib.dump(self._scaler, scaler_path)
            logger.debug(f"Saved sklearn scaler to {scaler_path}")

        # Save config and metadata
        config_dict = {
            "config": self.config.to_dict(),
            "n_features": self._n_features,
            "is_fitted": self._is_fitted,
            "original_shape": list(self._original_shape) if self._original_shape else None,
        }

        config_path = path / "scaler_config.json"
        with open(config_path, "w") as f:
            json.dump(config_dict, f, indent=2)

        logger.info(f"Saved AdapterScaler to {path}")

    @classmethod
    def load(cls, path: Path) -> AdapterScaler:
        """
        Load scaler from disk.

        Args:
            path: Directory path containing scaler artifacts.

        Returns:
            Loaded AdapterScaler instance.

        Raises:
            FileNotFoundError: If config file not found.
        """
        path = Path(path)

        # Load config and metadata
        config_path = path / "scaler_config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"Scaler config not found: {config_path}")

        with open(config_path) as f:
            config_dict = json.load(f)

        # Create instance with loaded config
        config = ScalerConfig.from_dict(config_dict["config"])
        instance = cls(config)
        instance._n_features = config_dict["n_features"]
        instance._is_fitted = config_dict["is_fitted"]

        if config_dict.get("original_shape"):
            instance._original_shape = tuple(config_dict["original_shape"])

        # Load sklearn scaler if it exists
        scaler_path = path / "scaler.pkl"
        if scaler_path.exists():
            instance._scaler = safe_pickle_load(scaler_path)
            logger.debug(f"Loaded sklearn scaler from {scaler_path}")

        logger.info(f"Loaded AdapterScaler from {path}")
        return instance

    def get_statistics(self) -> dict[str, Any]:
        """
        Get scaler statistics for debugging and analysis.

        Returns:
            Dictionary with scaler statistics (center, scale for robust;
            mean, std for standard; min, max for minmax).
        """
        if not self._is_fitted or self._scaler is None:
            return {"fitted": False}

        stats: dict[str, Any] = {
            "fitted": True,
            "method": self.config.method,
            "n_features": self._n_features,
        }

        if self.config.method == "robust":
            stats["center"] = (
                self._scaler.center_.tolist() if hasattr(self._scaler, "center_") else None
            )
            stats["scale"] = (
                self._scaler.scale_.tolist() if hasattr(self._scaler, "scale_") else None
            )
        elif self.config.method == "standard":
            stats["mean"] = self._scaler.mean_.tolist() if hasattr(self._scaler, "mean_") else None
            stats["std"] = self._scaler.scale_.tolist() if hasattr(self._scaler, "scale_") else None
        elif self.config.method == "minmax":
            stats["data_min"] = (
                self._scaler.data_min_.tolist() if hasattr(self._scaler, "data_min_") else None
            )
            stats["data_max"] = (
                self._scaler.data_max_.tolist() if hasattr(self._scaler, "data_max_") else None
            )

        return stats


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================


def create_scaler(
    method: str = "robust",
    clip_value: float = 5.0,
    **kwargs: Any,
) -> AdapterScaler:
    """
    Factory function to create a configured AdapterScaler.

    Args:
        method: Scaling method ("robust", "standard", "minmax", "none")
        clip_value: Value to clip scaled outputs to (0 = no clipping)
        **kwargs: Additional arguments passed to ScalerConfig

    Returns:
        Configured AdapterScaler instance.

    Example:
        >>> scaler = create_scaler("robust", clip_value=5.0)
        >>> scaler = create_scaler("standard", clip_value=0)  # No clipping
        >>> scaler = create_scaler("none")  # Pass-through
    """
    config = ScalerConfig(method=method, clip_value=clip_value, **kwargs)
    return AdapterScaler(config)


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "ScalerConfig",
    "AdapterScaler",
    "create_scaler",
]
