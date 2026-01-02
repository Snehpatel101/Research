"""
Main FeatureScaler Class

This module contains the core FeatureScaler class that handles fitting,
transforming, and persisting feature scaling.

Author: ML Pipeline
Created: 2025-12-20
Updated: 2025-12-20 - Extracted from feature_scaler.py
"""

import json
import logging
import pickle
from datetime import datetime
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd

from .core import FeatureScalingConfig, ScalerConfig, ScalerType, ScalingStatistics
from .scalers import (
    categorize_feature,
    compute_statistics,
    create_scaler,
    get_default_scaler_type,
    should_log_transform,
)

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class FeatureScaler:
    """
    Train-only feature scaler for Phase 2 model training.

    This class ensures that:
    1. Scalers are fitted ONLY on training data
    2. Training statistics are used to transform val/test sets
    3. Scaling parameters are persisted for production inference
    4. Different scaling strategies are applied per feature type

    Attributes:
        scaler_type: Default scaler type ('robust', 'standard', 'minmax')
        feature_config: Optional custom configuration per feature
        is_fitted: Whether the scaler has been fitted
        feature_names: List of feature names in order
        scalers: Dict mapping feature names to fitted sklearn scalers
        configs: Dict mapping feature names to FeatureScalingConfig
        statistics: Dict mapping feature names to ScalingStatistics
    """

    def __init__(
        self,
        scaler_type: str = "robust",
        feature_config: dict[str, dict] | None = None,
        apply_log_to_price_volume: bool = True,
        robust_quantile_range: tuple[float, float] = (25.0, 75.0),
        clip_outliers: bool = True,
        clip_range: tuple[float, float] = (-5.0, 5.0),
        config: ScalerConfig | None = None,
    ):
        """
        Initialize the FeatureScaler.

        Args:
            scaler_type: Default scaler type ('robust', 'standard', 'minmax')
            feature_config: Optional dict mapping feature names to custom configs.
                           Each config can have 'scaler_type', 'apply_log_transform'
            apply_log_to_price_volume: Whether to apply log transform to
                                       price level and volume features
            robust_quantile_range: Quantile range for RobustScaler
            clip_outliers: Whether to clip scaled values to clip_range
            clip_range: Range to clip scaled values (e.g., (-5.0, 5.0))
            config: Optional ScalerConfig object (overrides other parameters)
        """
        # If config provided, use it to set parameters
        if config is not None:
            self.default_scaler_type = ScalerType(config.scaler_type)
            self.clip_outliers = config.clip_outliers
            self.clip_range = config.clip_range
        else:
            self.default_scaler_type = ScalerType(scaler_type)
            self.clip_outliers = clip_outliers
            self.clip_range = clip_range

        self.custom_feature_config = feature_config or {}
        self.apply_log_to_price_volume = apply_log_to_price_volume
        self.robust_quantile_range = robust_quantile_range

        # State (populated during fit)
        self.is_fitted: bool = False
        self.feature_names: list[str] = []
        self.scalers: dict[str, Union] = {}
        self.configs: dict[str, FeatureScalingConfig] = {}
        self.statistics: dict[str, ScalingStatistics] = {}
        self.log_shifts: dict[str, float] = {}

        # Metadata
        self.fit_timestamp: str | None = None
        self.n_samples_train: int = 0
        self.warnings: list[str] = []
        self.errors: list[str] = []

    def _create_config_for_feature(self, feature_name: str) -> FeatureScalingConfig:
        """
        Create scaling configuration for a feature.

        Uses custom config if provided, otherwise infers from feature name.
        """
        # Check for custom config
        if feature_name in self.custom_feature_config:
            custom = self.custom_feature_config[feature_name]
            category = categorize_feature(feature_name)
            return FeatureScalingConfig(
                feature_name=feature_name,
                category=category,
                scaler_type=ScalerType(custom.get("scaler_type", self.default_scaler_type.value)),
                apply_log_transform=custom.get("apply_log_transform", False),
                log_shift=custom.get("log_shift", 0.0),
            )

        # Infer from feature name
        category = categorize_feature(feature_name)
        scaler_type = get_default_scaler_type(category)

        # Override with default scaler type if not NONE
        if scaler_type != ScalerType.NONE and self.default_scaler_type != ScalerType.ROBUST:
            scaler_type = self.default_scaler_type

        apply_log = self.apply_log_to_price_volume and should_log_transform(feature_name, category)

        return FeatureScalingConfig(
            feature_name=feature_name,
            category=category,
            scaler_type=scaler_type,
            apply_log_transform=apply_log,
            log_shift=0.0,
        )

    def fit(self, train_df: pd.DataFrame, feature_cols: list[str]) -> "FeatureScaler":
        """
        Fit scalers on training data only.

        IMPORTANT: This method should ONLY be called with training data.
        Never pass validation or test data to this method.

        Args:
            train_df: Training DataFrame
            feature_cols: List of feature column names to scale

        Returns:
            self (for method chaining)

        Raises:
            ValueError: If train_df is empty or feature_cols is invalid
        """
        logger.info("=" * 60)
        logger.info("FITTING FEATURE SCALER (TRAIN DATA ONLY)")
        logger.info("=" * 60)

        if len(train_df) == 0:
            raise ValueError("Training DataFrame is empty")

        if not feature_cols:
            raise ValueError("feature_cols cannot be empty")

        # Validate all columns exist
        missing_cols = set(feature_cols) - set(train_df.columns)
        if missing_cols:
            raise ValueError(f"Columns not found in DataFrame: {missing_cols}")

        self.feature_names = list(feature_cols)
        self.n_samples_train = len(train_df)
        self.fit_timestamp = datetime.now().isoformat()
        self.warnings = []
        self.errors = []

        logger.info(f"Training samples: {self.n_samples_train:,}")
        logger.info(f"Features to scale: {len(self.feature_names)}")

        # Track category counts
        category_counts: dict[str, int] = {}
        scaler_counts: dict[str, int] = {}

        for fname in self.feature_names:
            col_data = train_df[fname].values.astype(np.float64).copy()

            # Create configuration for this feature
            config = self._create_config_for_feature(fname)
            self.configs[fname] = config

            # Update counts
            cat_name = config.category.value
            category_counts[cat_name] = category_counts.get(cat_name, 0) + 1
            scaler_name = config.scaler_type.value
            scaler_counts[scaler_name] = scaler_counts.get(scaler_name, 0) + 1

            # Handle NaN/Inf
            mask = np.isnan(col_data) | np.isinf(col_data)
            if mask.any():
                valid_data = col_data[~mask]
                if len(valid_data) > 0:
                    fill_value = np.median(valid_data)
                else:
                    fill_value = 0.0
                    self.warnings.append(f"{fname}: All values are NaN/Inf, using 0")
                col_data[mask] = fill_value

            # Apply log transform if configured
            if config.apply_log_transform:
                min_val = np.min(col_data)
                shift = abs(min_val) + 1.0 if min_val <= 0 else 0.0
                self.log_shifts[fname] = shift
                col_data = np.log1p(col_data + shift)

            # Fit scaler
            if config.scaler_type == ScalerType.NONE:
                self.scalers[fname] = None
                scaled_data = col_data
            else:
                scaler = create_scaler(config.scaler_type, self.robust_quantile_range)
                scaler.fit(col_data.reshape(-1, 1))
                self.scalers[fname] = scaler
                scaled_data = scaler.transform(col_data.reshape(-1, 1)).ravel()

            # Compute statistics
            original_data = train_df[fname].values.astype(np.float64)
            self.statistics[fname] = compute_statistics(original_data, scaled_data, fname)

            # Check for issues
            if self.statistics[fname].train_std < 1e-10:
                self.warnings.append(
                    f"{fname}: Near-zero variance (std={self.statistics[fname].train_std:.2e})"
                )

        self.is_fitted = True

        # Log summary
        logger.info("\nFeature categories:")
        for cat, count in sorted(category_counts.items()):
            logger.info(f"  {cat}: {count}")

        logger.info("\nScaler types used:")
        for stype, count in sorted(scaler_counts.items()):
            logger.info(f"  {stype}: {count}")

        if self.warnings:
            logger.warning(f"\nWarnings ({len(self.warnings)}):")
            for w in self.warnings[:5]:
                logger.warning(f"  - {w}")
            if len(self.warnings) > 5:
                logger.warning(f"  ... and {len(self.warnings) - 5} more")

        logger.info("\nFeature scaler fitted successfully (using TRAIN data only)")
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform features using fitted parameters from training data.

        This method uses the statistics computed during fit() to transform
        new data. It can be safely used on validation, test, or production data.

        Args:
            df: DataFrame to transform

        Returns:
            DataFrame with scaled features

        Raises:
            ValueError: If scaler not fitted or features don't match
        """
        if not self.is_fitted:
            raise ValueError("Scaler has not been fitted. Call fit() first with training data.")

        # Validate columns
        missing_cols = set(self.feature_names) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing features in DataFrame: {missing_cols}")

        # Create output DataFrame
        result = df.copy()

        for fname in self.feature_names:
            col_data = df[fname].values.astype(np.float64).copy()
            config = self.configs[fname]

            # Handle NaN/Inf (use training median)
            mask = np.isnan(col_data) | np.isinf(col_data)
            if mask.any():
                fill_value = self.statistics[fname].train_median
                if np.isnan(fill_value):
                    fill_value = 0.0
                col_data[mask] = fill_value

            # Apply log transform if configured
            if config.apply_log_transform:
                shift = self.log_shifts.get(fname, 0.0)
                col_data = np.log1p(col_data + shift)

            # Apply scaler
            scaler = self.scalers.get(fname)
            if scaler is not None:
                col_data = scaler.transform(col_data.reshape(-1, 1)).ravel()

            # Apply outlier clipping if configured
            if self.clip_outliers:
                col_data = np.clip(col_data, self.clip_range[0], self.clip_range[1])

            result[fname] = col_data

        return result

    def fit_transform(self, train_df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
        """
        Fit on training data and transform it.

        Convenience method that calls fit() then transform().

        Args:
            train_df: Training DataFrame (used for fitting)
            feature_cols: List of feature column names

        Returns:
            Transformed training DataFrame
        """
        self.fit(train_df, feature_cols)
        return self.transform(train_df)

    def inverse_transform(
        self, df: pd.DataFrame, features: list[str] | None = None
    ) -> pd.DataFrame:
        """
        Reverse the scaling transformation.

        Args:
            df: Scaled DataFrame
            features: Optional list of features to inverse transform.
                     If None, transforms all fitted features.

        Returns:
            DataFrame with original-scale features
        """
        if not self.is_fitted:
            raise ValueError("Scaler has not been fitted. Call fit() first.")

        features = features or self.feature_names
        result = df.copy()

        for fname in features:
            if fname not in self.feature_names:
                continue

            col_data = df[fname].values.astype(np.float64).copy()
            config = self.configs[fname]
            scaler = self.scalers.get(fname)

            # Inverse scaler
            if scaler is not None:
                col_data = scaler.inverse_transform(col_data.reshape(-1, 1)).ravel()

            # Inverse log transform
            if config.apply_log_transform:
                shift = self.log_shifts.get(fname, 0.0)
                col_data = np.expm1(col_data) - shift

            result[fname] = col_data

        return result

    def save(self, path: Path) -> None:
        """
        Persist scaler parameters to disk.

        Saves all fitted parameters so the scaler can be loaded
        for production inference without access to training data.

        Args:
            path: Path to save the scaler (pickle format)

        Raises:
            ValueError: If scaler not fitted
        """
        if not self.is_fitted:
            raise ValueError("Scaler has not been fitted. Call fit() first.")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        state = {
            "version": "2.1",
            "default_scaler_type": self.default_scaler_type.value,
            "robust_quantile_range": self.robust_quantile_range,
            "apply_log_to_price_volume": self.apply_log_to_price_volume,
            "clip_outliers": self.clip_outliers,
            "clip_range": self.clip_range,
            "is_fitted": self.is_fitted,
            "feature_names": self.feature_names,
            "scalers": self.scalers,
            "configs": {k: v.to_dict() for k, v in self.configs.items()},
            "statistics": {k: v.to_dict() for k, v in self.statistics.items()},
            "log_shifts": self.log_shifts,
            "fit_timestamp": self.fit_timestamp,
            "n_samples_train": self.n_samples_train,
            "warnings": self.warnings,
            "errors": self.errors,
        }

        with open(path, "wb") as f:
            pickle.dump(state, f)

        logger.info(f"Scaler saved to: {path}")

        # Also save human-readable JSON report
        json_path = path.with_suffix(".json")
        self._save_json_report(json_path)

    def _save_json_report(self, path: Path) -> None:
        """Save a human-readable JSON report of the scaler configuration."""
        report = {
            "version": "2.0",
            "fit_timestamp": self.fit_timestamp,
            "n_samples_train": self.n_samples_train,
            "n_features": len(self.feature_names),
            "default_scaler_type": self.default_scaler_type.value,
            "features": {
                fname: {
                    "category": self.configs[fname].category.value,
                    "scaler_type": self.configs[fname].scaler_type.value,
                    "apply_log_transform": self.configs[fname].apply_log_transform,
                    "train_stats": {
                        "mean": self.statistics[fname].train_mean,
                        "std": self.statistics[fname].train_std,
                        "min": self.statistics[fname].train_min,
                        "max": self.statistics[fname].train_max,
                    },
                    "scaled_stats": {
                        "mean": self.statistics[fname].scaled_mean,
                        "std": self.statistics[fname].scaled_std,
                        "min": self.statistics[fname].scaled_min,
                        "max": self.statistics[fname].scaled_max,
                    },
                }
                for fname in self.feature_names
            },
            "warnings": self.warnings,
        }

        with open(path, "w") as f:
            json.dump(report, f, indent=2, default=str)

    @classmethod
    def load(cls, path: Path) -> "FeatureScaler":
        """
        Load a persisted scaler from disk.

        Args:
            path: Path to the saved scaler

        Returns:
            Loaded FeatureScaler instance
        """
        path = Path(path)

        with open(path, "rb") as f:
            state = pickle.load(f)

        scaler = cls(
            scaler_type=state["default_scaler_type"],
            apply_log_to_price_volume=state.get("apply_log_to_price_volume", True),
            robust_quantile_range=state.get("robust_quantile_range", (25.0, 75.0)),
            clip_outliers=state.get("clip_outliers", True),
            clip_range=tuple(state.get("clip_range", (-5.0, 5.0))),
        )

        scaler.is_fitted = state["is_fitted"]
        scaler.feature_names = state["feature_names"]
        scaler.scalers = state["scalers"]
        scaler.configs = {k: FeatureScalingConfig.from_dict(v) for k, v in state["configs"].items()}
        scaler.statistics = {
            k: ScalingStatistics.from_dict(v) for k, v in state["statistics"].items()
        }
        scaler.log_shifts = state.get("log_shifts", {})
        scaler.fit_timestamp = state.get("fit_timestamp")
        scaler.n_samples_train = state.get("n_samples_train", 0)
        scaler.warnings = state.get("warnings", [])
        scaler.errors = state.get("errors", [])

        logger.info(f"Scaler loaded from: {path}")
        logger.info(f"  Features: {len(scaler.feature_names)}")
        logger.info(f"  Trained on: {scaler.n_samples_train:,} samples")

        return scaler

    def get_scaling_report(self) -> dict:
        """
        Get a comprehensive report of scaling configuration and statistics.

        Returns:
            Dictionary with detailed scaling information
        """
        if not self.is_fitted:
            return {"is_fitted": False}

        # Group features by category
        features_by_category: dict[str, list[str]] = {}
        for fname, config in self.configs.items():
            cat = config.category.value
            if cat not in features_by_category:
                features_by_category[cat] = []
            features_by_category[cat].append(fname)

        # Group features by scaler type
        features_by_scaler: dict[str, list[str]] = {}
        for fname, config in self.configs.items():
            stype = config.scaler_type.value
            if stype not in features_by_scaler:
                features_by_scaler[stype] = []
            features_by_scaler[stype].append(fname)

        return {
            "is_fitted": True,
            "fit_timestamp": self.fit_timestamp,
            "n_samples_train": self.n_samples_train,
            "n_features": len(self.feature_names),
            "default_scaler_type": self.default_scaler_type.value,
            "features_by_category": features_by_category,
            "features_by_scaler": features_by_scaler,
            "statistics": {k: v.to_dict() for k, v in self.statistics.items()},
            "warnings": self.warnings,
            "errors": self.errors,
        }
