"""
Feature Scaling Module for Phase 2 Model Training

Provides robust scaling for financial time series features:
- RobustScaler for features with outliers (returns, volume)
- StandardScaler for bounded features (RSI, stochastic)
- No scaling for already-normalized features (sin/cos temporal)

This module is designed for neural network inputs (N-HiTS, TFT, PatchTST)
and handles the 38 selected features from Phase 1 feature selection.

Usage:
    from feature_scaling import FeatureScaler

    scaler = FeatureScaler()
    X_train_scaled = scaler.fit_transform(X_train, feature_names)
    X_val_scaled = scaler.transform(X_val)

    # Save for inference
    scaler.save('models/feature_scaler.pkl')

    # Load for inference
    scaler = FeatureScaler.load('models/feature_scaler.pkl')

Author: ML Pipeline
Created: 2025-12-19
"""

import numpy as np
import pandas as pd
import pickle
import logging
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple, Set
from dataclasses import dataclass, field
from sklearn.preprocessing import RobustScaler, StandardScaler
from scipy import stats as scipy_stats

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# FEATURE CATEGORIZATION
# =============================================================================
# These categories determine which scaler to use for each feature type.
# Based on the 38 selected features from feature_selection.py

@dataclass
class FeatureCategories:
    """
    Feature categories for scaling decisions.

    Categories:
    - robust_scale: Use RobustScaler (outlier-resistant, uses IQR)
    - standard_scale: Use StandardScaler (mean=0, std=1)
    - log_transform_then_robust: Apply log1p transform, then RobustScaler
    - no_scale: Already normalized or categorical, skip scaling
    """

    # Returns and price-based features - have fat tails, use RobustScaler
    robust_scale: Set[str] = field(default_factory=lambda: {
        'log_return',
        'high_low_range',
        'close_to_sma_10',
        'close_to_sma_20',
        'close_to_sma_50',
        'close_to_sma_100',
        'close_to_sma_200',
        'close_to_vwap',
        'macd',
        'macd_hist',
        'macd_crossover',
        'atr_7',
        'atr_7_pct',
        'roc_5',
        'roc_10',
        'roc_20',
        'bb_width',
    })

    # Bounded indicators (0-100 range) - use StandardScaler
    standard_scale: Set[str] = field(default_factory=lambda: {
        'rsi',
        'stoch_k',
        'stoch_d',
        'adx',
        'plus_di',
        'minus_di',
        'bb_position',  # 0-1 range
    })

    # Volume features - apply log transform then RobustScaler
    log_transform_then_robust: Set[str] = field(default_factory=lambda: {
        'volume_ratio',
        'volume_sma_20',
        'volume_zscore',
        'obv',
        'sma_10',  # Price level, needs log for scale invariance
    })

    # Already normalized or categorical - NO scaling
    no_scale: Set[str] = field(default_factory=lambda: {
        'hour_sin',
        'hour_cos',
        'dow_sin',
        'dow_cos',
        'is_rth',
        'vol_regime',
        'trend_regime',
        'rsi_oversold',
        'rsi_overbought',
    })


# Default feature categories instance
DEFAULT_CATEGORIES = FeatureCategories()


def get_scaling_category(
    feature_name: str,
    categories: FeatureCategories = DEFAULT_CATEGORIES
) -> str:
    """
    Determine the scaling category for a feature.

    Args:
        feature_name: Name of the feature
        categories: FeatureCategories instance

    Returns:
        Scaling category: 'robust', 'standard', 'log_robust', 'none', or 'unknown'
    """
    if feature_name in categories.robust_scale:
        return 'robust'
    elif feature_name in categories.standard_scale:
        return 'standard'
    elif feature_name in categories.log_transform_then_robust:
        return 'log_robust'
    elif feature_name in categories.no_scale:
        return 'none'
    else:
        # Unknown feature - default to RobustScaler (safer for financial data)
        return 'unknown'


# =============================================================================
# SCALING STATISTICS
# =============================================================================

@dataclass
class ScalingStats:
    """Container for scaling statistics and diagnostics."""
    feature_name: str
    scaling_type: str
    original_mean: float
    original_std: float
    original_min: float
    original_max: float
    original_median: float
    original_iqr: float
    scaled_mean: float
    scaled_std: float
    scaled_min: float
    scaled_max: float
    zero_variance: bool = False
    nan_count: int = 0
    inf_count: int = 0

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'feature_name': self.feature_name,
            'scaling_type': self.scaling_type,
            'original': {
                'mean': self.original_mean,
                'std': self.original_std,
                'min': self.original_min,
                'max': self.original_max,
                'median': self.original_median,
                'iqr': self.original_iqr
            },
            'scaled': {
                'mean': self.scaled_mean,
                'std': self.scaled_std,
                'min': self.scaled_min,
                'max': self.scaled_max
            },
            'warnings': {
                'zero_variance': self.zero_variance,
                'nan_count': self.nan_count,
                'inf_count': self.inf_count
            }
        }


# =============================================================================
# FEATURE SCALER CLASS
# =============================================================================

class FeatureScaler:
    """
    Feature-aware scaler for financial time series data.

    Applies appropriate scaling based on feature characteristics:
    - RobustScaler for features with outliers (returns, MACD, ATR)
    - StandardScaler for bounded features (RSI, Stochastic, ADX)
    - Log transform + RobustScaler for volume features
    - No scaling for temporal (sin/cos) and regime features

    Attributes:
        feature_names: List of feature names in order
        scalers: Dictionary mapping feature names to fitted scalers
        scaling_types: Dictionary mapping feature names to scaling type
        stats: Dictionary mapping feature names to ScalingStats
        is_fitted: Whether the scaler has been fitted
        categories: FeatureCategories configuration
    """

    def __init__(
        self,
        categories: Optional[FeatureCategories] = None,
        robust_quantile_range: Tuple[float, float] = (25.0, 75.0),
        handle_unknown: str = 'robust'
    ):
        """
        Initialize the FeatureScaler.

        Args:
            categories: Optional custom FeatureCategories. Uses default if None.
            robust_quantile_range: Quantile range for RobustScaler (default 25-75)
            handle_unknown: How to handle unknown features ('robust', 'standard', 'warn')
        """
        self.categories = categories or DEFAULT_CATEGORIES
        self.robust_quantile_range = robust_quantile_range
        self.handle_unknown = handle_unknown

        self.feature_names: List[str] = []
        self.scalers: Dict[str, Union[RobustScaler, StandardScaler, None]] = {}
        self.scaling_types: Dict[str, str] = {}
        self.stats: Dict[str, ScalingStats] = {}
        self.is_fitted: bool = False

        # For log-transformed features, store the shift value
        self.log_shifts: Dict[str, float] = {}

    def _validate_input(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        feature_names: Optional[List[str]] = None
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Validate and convert input to numpy array.

        Args:
            X: Input data (numpy array or DataFrame)
            feature_names: List of feature names (required if X is numpy array)

        Returns:
            Tuple of (numpy array, feature names)

        Raises:
            ValueError: If input is invalid
        """
        if isinstance(X, pd.DataFrame):
            if feature_names is None:
                feature_names = list(X.columns)
            X = X.values.astype(np.float64)
        elif isinstance(X, np.ndarray):
            if feature_names is None:
                raise ValueError("feature_names must be provided when X is numpy array")
            X = X.astype(np.float64)
        else:
            raise ValueError(f"X must be numpy array or DataFrame, got {type(X)}")

        if X.shape[1] != len(feature_names):
            raise ValueError(
                f"Number of features ({X.shape[1]}) does not match "
                f"number of feature names ({len(feature_names)})"
            )

        return X, feature_names

    def _compute_stats(
        self,
        data: np.ndarray,
        feature_name: str,
        scaling_type: str,
        scaled_data: np.ndarray
    ) -> ScalingStats:
        """
        Compute statistics for a single feature.

        Args:
            data: Original feature data (1D array)
            feature_name: Name of the feature
            scaling_type: Type of scaling applied
            scaled_data: Scaled feature data (1D array)

        Returns:
            ScalingStats object
        """
        # Count NaN and Inf
        nan_count = int(np.isnan(data).sum())
        inf_count = int(np.isinf(data).sum())

        # Clean data for statistics
        clean_data = data[~np.isnan(data) & ~np.isinf(data)]
        clean_scaled = scaled_data[~np.isnan(scaled_data) & ~np.isinf(scaled_data)]

        if len(clean_data) == 0:
            return ScalingStats(
                feature_name=feature_name,
                scaling_type=scaling_type,
                original_mean=np.nan,
                original_std=np.nan,
                original_min=np.nan,
                original_max=np.nan,
                original_median=np.nan,
                original_iqr=np.nan,
                scaled_mean=np.nan,
                scaled_std=np.nan,
                scaled_min=np.nan,
                scaled_max=np.nan,
                zero_variance=True,
                nan_count=nan_count,
                inf_count=inf_count
            )

        # Compute IQR
        q75, q25 = np.percentile(clean_data, [75, 25])
        iqr = q75 - q25

        # Check for zero variance
        std_val = float(np.std(clean_data))
        zero_variance = std_val < 1e-10

        return ScalingStats(
            feature_name=feature_name,
            scaling_type=scaling_type,
            original_mean=float(np.mean(clean_data)),
            original_std=std_val,
            original_min=float(np.min(clean_data)),
            original_max=float(np.max(clean_data)),
            original_median=float(np.median(clean_data)),
            original_iqr=float(iqr),
            scaled_mean=float(np.mean(clean_scaled)) if len(clean_scaled) > 0 else np.nan,
            scaled_std=float(np.std(clean_scaled)) if len(clean_scaled) > 0 else np.nan,
            scaled_min=float(np.min(clean_scaled)) if len(clean_scaled) > 0 else np.nan,
            scaled_max=float(np.max(clean_scaled)) if len(clean_scaled) > 0 else np.nan,
            zero_variance=zero_variance,
            nan_count=nan_count,
            inf_count=inf_count
        )

    def fit(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        feature_names: Optional[List[str]] = None
    ) -> 'FeatureScaler':
        """
        Fit the scaler to training data.

        Args:
            X: Training data (n_samples, n_features)
            feature_names: List of feature names (required if X is numpy array)

        Returns:
            self

        Raises:
            ValueError: If input is invalid
        """
        logger.info("=" * 60)
        logger.info("FITTING FEATURE SCALER")
        logger.info("=" * 60)

        X, feature_names = self._validate_input(X, feature_names)
        self.feature_names = feature_names

        n_samples, n_features = X.shape
        logger.info(f"Fitting scaler on {n_samples:,} samples, {n_features} features")

        # Track scaling type counts
        type_counts = {'robust': 0, 'standard': 0, 'log_robust': 0, 'none': 0, 'unknown': 0}

        for i, fname in enumerate(feature_names):
            col_data = X[:, i].copy()

            # Determine scaling type
            scaling_type = get_scaling_category(fname, self.categories)
            self.scaling_types[fname] = scaling_type
            type_counts[scaling_type] += 1

            if scaling_type == 'unknown':
                if self.handle_unknown == 'warn':
                    logger.warning(f"  Unknown feature '{fname}', using RobustScaler")
                scaling_type = 'robust'
                self.scaling_types[fname] = 'robust'

            if scaling_type == 'none':
                # No scaling needed
                self.scalers[fname] = None
                self.stats[fname] = self._compute_stats(col_data, fname, 'none', col_data)

            elif scaling_type == 'robust':
                # RobustScaler for outlier-resistant scaling
                scaler = RobustScaler(quantile_range=self.robust_quantile_range)
                # Handle NaN/Inf by replacing temporarily
                mask = np.isnan(col_data) | np.isinf(col_data)
                if mask.any():
                    col_clean = col_data.copy()
                    col_clean[mask] = np.nanmedian(col_data[~mask])
                else:
                    col_clean = col_data
                scaler.fit(col_clean.reshape(-1, 1))
                self.scalers[fname] = scaler
                scaled = scaler.transform(col_clean.reshape(-1, 1)).ravel()
                self.stats[fname] = self._compute_stats(col_data, fname, 'robust', scaled)

            elif scaling_type == 'standard':
                # StandardScaler for bounded features
                scaler = StandardScaler()
                mask = np.isnan(col_data) | np.isinf(col_data)
                if mask.any():
                    col_clean = col_data.copy()
                    col_clean[mask] = np.nanmean(col_data[~mask])
                else:
                    col_clean = col_data
                scaler.fit(col_clean.reshape(-1, 1))
                self.scalers[fname] = scaler
                scaled = scaler.transform(col_clean.reshape(-1, 1)).ravel()
                self.stats[fname] = self._compute_stats(col_data, fname, 'standard', scaled)

            elif scaling_type == 'log_robust':
                # Log transform then RobustScaler for volume/price level features
                # Shift to ensure all values are positive
                mask = np.isnan(col_data) | np.isinf(col_data)
                if mask.any():
                    col_clean = col_data.copy()
                    col_clean[mask] = np.nanmedian(col_data[~mask])
                else:
                    col_clean = col_data

                min_val = np.min(col_clean)
                shift = abs(min_val) + 1.0 if min_val <= 0 else 0.0
                self.log_shifts[fname] = shift

                # Apply log1p transform
                log_data = np.log1p(col_clean + shift)

                # Then apply RobustScaler
                scaler = RobustScaler(quantile_range=self.robust_quantile_range)
                scaler.fit(log_data.reshape(-1, 1))
                self.scalers[fname] = scaler
                scaled = scaler.transform(log_data.reshape(-1, 1)).ravel()
                self.stats[fname] = self._compute_stats(col_data, fname, 'log_robust', scaled)

        self.is_fitted = True

        # Log summary
        logger.info(f"\nScaling type distribution:")
        logger.info(f"  RobustScaler:     {type_counts['robust']} features")
        logger.info(f"  StandardScaler:   {type_counts['standard']} features")
        logger.info(f"  Log+RobustScaler: {type_counts['log_robust']} features")
        logger.info(f"  No scaling:       {type_counts['none']} features")
        if type_counts['unknown'] > 0:
            logger.info(f"  Unknown (->Robust): {type_counts['unknown']} features")

        # Check for warnings
        zero_var_features = [f for f, s in self.stats.items() if s.zero_variance]
        if zero_var_features:
            logger.warning(f"\nFeatures with zero variance: {zero_var_features}")

        nan_features = [f for f, s in self.stats.items() if s.nan_count > 0]
        if nan_features:
            logger.warning(f"Features with NaN values: {nan_features}")

        inf_features = [f for f, s in self.stats.items() if s.inf_count > 0]
        if inf_features:
            logger.warning(f"Features with Inf values: {inf_features}")

        logger.info("\nFeature scaler fitted successfully")
        return self

    def transform(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        validate: bool = True
    ) -> np.ndarray:
        """
        Transform features using fitted scalers.

        Args:
            X: Data to transform (n_samples, n_features)
            validate: Whether to validate output for NaN/Inf (default True)

        Returns:
            Scaled data as numpy array

        Raises:
            ValueError: If scaler not fitted or features don't match
        """
        if not self.is_fitted:
            raise ValueError("Scaler has not been fitted. Call fit() first.")

        if isinstance(X, pd.DataFrame):
            # Verify feature names match
            input_features = list(X.columns)
            if input_features != self.feature_names:
                # Try to reorder columns to match
                missing = set(self.feature_names) - set(input_features)
                extra = set(input_features) - set(self.feature_names)
                if missing:
                    raise ValueError(f"Missing features: {missing}")
                if extra:
                    logger.warning(f"Extra features ignored: {extra}")
                X = X[self.feature_names]
            X = X.values.astype(np.float64)
        else:
            X = X.astype(np.float64)

        if X.shape[1] != len(self.feature_names):
            raise ValueError(
                f"Expected {len(self.feature_names)} features, got {X.shape[1]}"
            )

        # Transform each feature
        X_scaled = np.zeros_like(X)

        for i, fname in enumerate(self.feature_names):
            col_data = X[:, i].copy()
            scaler = self.scalers.get(fname)
            scaling_type = self.scaling_types.get(fname, 'none')

            if scaling_type == 'none' or scaler is None:
                # No transformation
                X_scaled[:, i] = col_data

            elif scaling_type == 'log_robust':
                # Apply log transform then scaler
                shift = self.log_shifts.get(fname, 0.0)
                # Handle NaN/Inf
                mask = np.isnan(col_data) | np.isinf(col_data)
                if mask.any():
                    col_data[mask] = np.nanmedian(col_data[~mask])
                log_data = np.log1p(col_data + shift)
                X_scaled[:, i] = scaler.transform(log_data.reshape(-1, 1)).ravel()

            else:
                # Standard or Robust scaler
                mask = np.isnan(col_data) | np.isinf(col_data)
                if mask.any():
                    # Replace with appropriate fill value
                    if scaling_type == 'standard':
                        col_data[mask] = np.nanmean(col_data[~mask])
                    else:
                        col_data[mask] = np.nanmedian(col_data[~mask])
                X_scaled[:, i] = scaler.transform(col_data.reshape(-1, 1)).ravel()

        # Validate output
        if validate:
            self._validate_output(X_scaled)

        return X_scaled

    def fit_transform(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        feature_names: Optional[List[str]] = None,
        validate: bool = True
    ) -> np.ndarray:
        """
        Fit the scaler and transform data in one step.

        Args:
            X: Training data (n_samples, n_features)
            feature_names: List of feature names (required if X is numpy array)
            validate: Whether to validate output for NaN/Inf

        Returns:
            Scaled data as numpy array
        """
        self.fit(X, feature_names)
        return self.transform(X, validate=validate)

    def inverse_transform(
        self,
        X_scaled: np.ndarray,
        feature_names: Optional[List[str]] = None
    ) -> np.ndarray:
        """
        Reverse the scaling transformation.

        Args:
            X_scaled: Scaled data (n_samples, n_features)
            feature_names: Optional list of feature names to inverse transform.
                          If None, uses all features.

        Returns:
            Original-scale data as numpy array

        Raises:
            ValueError: If scaler not fitted
        """
        if not self.is_fitted:
            raise ValueError("Scaler has not been fitted. Call fit() first.")

        if feature_names is None:
            feature_names = self.feature_names

        if X_scaled.shape[1] != len(feature_names):
            raise ValueError(
                f"Expected {len(feature_names)} features, got {X_scaled.shape[1]}"
            )

        X_original = np.zeros_like(X_scaled)

        for i, fname in enumerate(feature_names):
            col_scaled = X_scaled[:, i].copy()
            scaler = self.scalers.get(fname)
            scaling_type = self.scaling_types.get(fname, 'none')

            if scaling_type == 'none' or scaler is None:
                X_original[:, i] = col_scaled

            elif scaling_type == 'log_robust':
                # Reverse: inverse scaler, then expm1
                shift = self.log_shifts.get(fname, 0.0)
                log_data = scaler.inverse_transform(col_scaled.reshape(-1, 1)).ravel()
                X_original[:, i] = np.expm1(log_data) - shift

            else:
                X_original[:, i] = scaler.inverse_transform(
                    col_scaled.reshape(-1, 1)
                ).ravel()

        return X_original

    def _validate_output(self, X_scaled: np.ndarray) -> None:
        """
        Validate scaled output for NaN and Inf values.

        Args:
            X_scaled: Scaled data to validate

        Raises:
            ValueError: If NaN or Inf found after scaling
        """
        nan_count = np.isnan(X_scaled).sum()
        inf_count = np.isinf(X_scaled).sum()

        if nan_count > 0:
            # Find which features have NaN
            nan_features = []
            for i, fname in enumerate(self.feature_names):
                if np.isnan(X_scaled[:, i]).any():
                    nan_features.append(fname)
            logger.error(f"NaN values found after scaling in features: {nan_features}")
            raise ValueError(f"Scaling produced {nan_count} NaN values in: {nan_features}")

        if inf_count > 0:
            inf_features = []
            for i, fname in enumerate(self.feature_names):
                if np.isinf(X_scaled[:, i]).any():
                    inf_features.append(fname)
            logger.error(f"Inf values found after scaling in features: {inf_features}")
            raise ValueError(f"Scaling produced {inf_count} Inf values in: {inf_features}")

    def save(self, path: Union[str, Path]) -> None:
        """
        Save the fitted scaler to disk.

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
            'feature_names': self.feature_names,
            'scalers': self.scalers,
            'scaling_types': self.scaling_types,
            'log_shifts': self.log_shifts,
            'stats': {k: v.to_dict() for k, v in self.stats.items()},
            'is_fitted': self.is_fitted,
            'robust_quantile_range': self.robust_quantile_range,
            'handle_unknown': self.handle_unknown,
            'categories': {
                'robust_scale': list(self.categories.robust_scale),
                'standard_scale': list(self.categories.standard_scale),
                'log_transform_then_robust': list(self.categories.log_transform_then_robust),
                'no_scale': list(self.categories.no_scale)
            }
        }

        with open(path, 'wb') as f:
            pickle.dump(state, f)

        logger.info(f"Scaler saved to: {path}")

    @classmethod
    def load(cls, path: Union[str, Path]) -> 'FeatureScaler':
        """
        Load a fitted scaler from disk.

        Args:
            path: Path to the saved scaler

        Returns:
            Loaded FeatureScaler instance
        """
        path = Path(path)

        with open(path, 'rb') as f:
            state = pickle.load(f)

        # Reconstruct categories
        categories = FeatureCategories(
            robust_scale=set(state['categories']['robust_scale']),
            standard_scale=set(state['categories']['standard_scale']),
            log_transform_then_robust=set(state['categories']['log_transform_then_robust']),
            no_scale=set(state['categories']['no_scale'])
        )

        scaler = cls(
            categories=categories,
            robust_quantile_range=state['robust_quantile_range'],
            handle_unknown=state['handle_unknown']
        )

        scaler.feature_names = state['feature_names']
        scaler.scalers = state['scalers']
        scaler.scaling_types = state['scaling_types']
        scaler.log_shifts = state['log_shifts']
        scaler.is_fitted = state['is_fitted']

        # Reconstruct stats (stored as dicts, need to convert back)
        scaler.stats = {}
        for fname, stat_dict in state['stats'].items():
            scaler.stats[fname] = ScalingStats(
                feature_name=stat_dict['feature_name'],
                scaling_type=stat_dict['scaling_type'],
                original_mean=stat_dict['original']['mean'],
                original_std=stat_dict['original']['std'],
                original_min=stat_dict['original']['min'],
                original_max=stat_dict['original']['max'],
                original_median=stat_dict['original']['median'],
                original_iqr=stat_dict['original']['iqr'],
                scaled_mean=stat_dict['scaled']['mean'],
                scaled_std=stat_dict['scaled']['std'],
                scaled_min=stat_dict['scaled']['min'],
                scaled_max=stat_dict['scaled']['max'],
                zero_variance=stat_dict['warnings']['zero_variance'],
                nan_count=stat_dict['warnings']['nan_count'],
                inf_count=stat_dict['warnings']['inf_count']
            )

        logger.info(f"Scaler loaded from: {path}")
        logger.info(f"  Features: {len(scaler.feature_names)}")

        return scaler

    def get_scaling_summary(self) -> Dict:
        """
        Get a summary of scaling configuration and statistics.

        Returns:
            Dictionary with scaling summary
        """
        if not self.is_fitted:
            return {'is_fitted': False}

        type_counts = {}
        for stype in self.scaling_types.values():
            type_counts[stype] = type_counts.get(stype, 0) + 1

        return {
            'is_fitted': True,
            'n_features': len(self.feature_names),
            'scaling_type_counts': type_counts,
            'features_by_type': {
                stype: [f for f, t in self.scaling_types.items() if t == stype]
                for stype in set(self.scaling_types.values())
            },
            'warnings': {
                'zero_variance': [f for f, s in self.stats.items() if s.zero_variance],
                'had_nan': [f for f, s in self.stats.items() if s.nan_count > 0],
                'had_inf': [f for f, s in self.stats.items() if s.inf_count > 0]
            }
        }

    def get_feature_stats(self, feature_name: str) -> Optional[ScalingStats]:
        """
        Get scaling statistics for a specific feature.

        Args:
            feature_name: Name of the feature

        Returns:
            ScalingStats object or None if feature not found
        """
        return self.stats.get(feature_name)


# =============================================================================
# VALIDATION FUNCTIONS
# =============================================================================

def validate_scaled_features(
    X_scaled: np.ndarray,
    feature_names: List[str],
    z_threshold: float = 5.0,
    expected_range: Tuple[float, float] = (-10.0, 10.0)
) -> Dict:
    """
    Validate scaled features for anomalies.

    Args:
        X_scaled: Scaled feature array
        feature_names: List of feature names
        z_threshold: Z-score threshold for outlier detection
        expected_range: Expected range for scaled values

    Returns:
        Dictionary with validation results
    """
    results = {
        'is_valid': True,
        'n_samples': X_scaled.shape[0],
        'n_features': X_scaled.shape[1],
        'issues': [],
        'warnings': [],
        'per_feature': {}
    }

    for i, fname in enumerate(feature_names):
        col = X_scaled[:, i]

        # Check for NaN
        nan_count = int(np.isnan(col).sum())
        if nan_count > 0:
            results['issues'].append(f"{fname}: {nan_count} NaN values")
            results['is_valid'] = False

        # Check for Inf
        inf_count = int(np.isinf(col).sum())
        if inf_count > 0:
            results['issues'].append(f"{fname}: {inf_count} Inf values")
            results['is_valid'] = False

        # Clean data for statistics
        clean = col[~np.isnan(col) & ~np.isinf(col)]
        if len(clean) == 0:
            continue

        mean_val = float(np.mean(clean))
        std_val = float(np.std(clean))
        min_val = float(np.min(clean))
        max_val = float(np.max(clean))

        # Check range
        if min_val < expected_range[0] or max_val > expected_range[1]:
            results['warnings'].append(
                f"{fname}: values outside expected range [{expected_range[0]}, {expected_range[1]}]: "
                f"[{min_val:.2f}, {max_val:.2f}]"
            )

        # Check for extreme outliers
        if std_val > 0:
            z_scores = np.abs((clean - mean_val) / std_val)
            extreme_count = int((z_scores > z_threshold).sum())
            if extreme_count > 0:
                pct = extreme_count / len(clean) * 100
                if pct > 1.0:
                    results['warnings'].append(
                        f"{fname}: {extreme_count} extreme outliers ({pct:.2f}% > {z_threshold} std)"
                    )

        results['per_feature'][fname] = {
            'mean': mean_val,
            'std': std_val,
            'min': min_val,
            'max': max_val
        }

    return results


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def scale_train_val_test(
    X_train: Union[np.ndarray, pd.DataFrame],
    X_val: Union[np.ndarray, pd.DataFrame],
    X_test: Union[np.ndarray, pd.DataFrame],
    feature_names: Optional[List[str]] = None,
    scaler_path: Optional[Union[str, Path]] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, FeatureScaler]:
    """
    Scale train/val/test data with a single scaler fitted on training data.

    Args:
        X_train: Training features
        X_val: Validation features
        X_test: Test features
        feature_names: Feature names (extracted from DataFrame if not provided)
        scaler_path: Optional path to save the fitted scaler

    Returns:
        Tuple of (X_train_scaled, X_val_scaled, X_test_scaled, scaler)
    """
    scaler = FeatureScaler()

    # Fit and transform training data
    X_train_scaled = scaler.fit_transform(X_train, feature_names)

    # Transform validation and test data
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # Save if path provided
    if scaler_path:
        scaler.save(scaler_path)

    logger.info(f"\nScaling summary:")
    logger.info(f"  Train: {X_train_scaled.shape}")
    logger.info(f"  Val:   {X_val_scaled.shape}")
    logger.info(f"  Test:  {X_test_scaled.shape}")

    return X_train_scaled, X_val_scaled, X_test_scaled, scaler


def get_selected_feature_scaler() -> FeatureScaler:
    """
    Get a FeatureScaler configured for the 38 selected features.

    Returns:
        FeatureScaler with appropriate categories for selected features
    """
    return FeatureScaler(categories=DEFAULT_CATEGORIES)


# =============================================================================
# MAIN / CLI
# =============================================================================

def main():
    """Demonstrate feature scaling with sample data."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent))

    from config import SPLITS_DIR, MODELS_DIR

    logger.info("=" * 70)
    logger.info("FEATURE SCALING DEMONSTRATION")
    logger.info("=" * 70)

    # Load training data
    train_path = SPLITS_DIR / "train.parquet"

    if not train_path.exists():
        logger.error(f"Training data not found: {train_path}")
        logger.info("Run the pipeline first to generate train/val/test splits")
        return

    logger.info(f"\nLoading training data from {train_path}")
    train_df = pd.read_parquet(train_path)
    logger.info(f"Loaded {len(train_df):,} rows")

    # Get feature columns (exclude metadata and labels)
    excluded_cols = {'datetime', 'symbol', 'open', 'high', 'low', 'close', 'volume'}
    excluded_prefixes = ('label_', 'bars_to_hit_', 'mae_', 'quality_', 'sample_weight_')

    feature_cols = [
        c for c in train_df.columns
        if c not in excluded_cols
        and not any(c.startswith(p) for p in excluded_prefixes)
    ]

    logger.info(f"Found {len(feature_cols)} feature columns")

    # Create feature matrix
    X = train_df[feature_cols].values

    # Fit and transform
    scaler = FeatureScaler()
    X_scaled = scaler.fit_transform(X, feature_cols)

    # Save scaler
    scaler_path = MODELS_DIR / "feature_scaler.pkl"
    scaler.save(scaler_path)

    # Print summary
    summary = scaler.get_scaling_summary()
    logger.info(f"\n{'='*60}")
    logger.info("SCALING SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"Total features: {summary['n_features']}")
    logger.info(f"\nScaling type distribution:")
    for stype, count in summary['scaling_type_counts'].items():
        logger.info(f"  {stype}: {count}")

    if summary['warnings']['zero_variance']:
        logger.warning(f"\nZero variance features: {summary['warnings']['zero_variance']}")

    # Validate scaled output
    validation = validate_scaled_features(X_scaled, feature_cols)
    logger.info(f"\nValidation: {'PASSED' if validation['is_valid'] else 'FAILED'}")
    if validation['warnings']:
        for w in validation['warnings'][:5]:
            logger.warning(f"  {w}")

    logger.info(f"\nScaler saved to: {scaler_path}")


if __name__ == "__main__":
    main()
