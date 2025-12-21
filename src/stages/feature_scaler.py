"""
Feature Scaler Infrastructure for Phase 1/2 Pipeline - Train-Only Scaling

This module provides a train-only scaling infrastructure to prevent data leakage.
All scalers are fitted ONLY on training data, then applied to validation and test sets.

Key Features:
- Fits scalers exclusively on training data to prevent leakage
- Supports multiple scaler types (StandardScaler, RobustScaler, MinMaxScaler)
- Feature-type-aware scaling (different strategies per feature category)
- Outlier clipping to prevent extreme values from dominating
- Persists scaler parameters to disk for production inference
- Validates scaling correctness on val/test sets
- Integrates with stage8_validate.py

Usage:
    from stages.feature_scaler import FeatureScaler, scale_splits

    # Simple usage with scale_splits convenience function
    train_scaled, val_scaled, test_scaled, scaler = scale_splits(
        train_df, val_df, test_df, feature_cols
    )

    # Or use FeatureScaler directly for more control
    scaler = FeatureScaler(scaler_type='robust')
    train_scaled = scaler.fit_transform(train_df, feature_cols)

    # Transform val/test using training statistics
    val_scaled = scaler.transform(val_df)
    test_scaled = scaler.transform(test_df)

    # Save for production
    scaler.save(Path('models/scaler.pkl'))

    # Load in production
    scaler = FeatureScaler.load(Path('models/scaler.pkl'))

Author: ML Pipeline
Created: 2025-12-20
Updated: 2025-12-20 - Added outlier clipping, simplified scale_splits interface
"""

import numpy as np
import pandas as pd
import pickle
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple, Any, Set
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum

from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler
from scipy import stats

# Configure logger with NullHandler to avoid "No handler found" warnings
# when used as a library. Applications should configure their own handlers.
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


# =============================================================================
# ENUMS AND CONFIGURATION
# =============================================================================

class ScalerType(Enum):
    """Supported scaler types."""
    STANDARD = 'standard'
    ROBUST = 'robust'
    MINMAX = 'minmax'
    NONE = 'none'


@dataclass
class ScalerConfig:
    """
    Simple configuration for feature scaling.

    Attributes:
        scaler_type: Type of scaler to use ('robust', 'standard', 'minmax')
        clip_outliers: Whether to clip scaled values to clip_range
        clip_range: Range to clip scaled values (in units of the scaled distribution)
    """
    scaler_type: str = 'robust'
    clip_outliers: bool = True
    clip_range: Tuple[float, float] = (-5.0, 5.0)

    def to_dict(self) -> Dict:
        return {
            'scaler_type': self.scaler_type,
            'clip_outliers': self.clip_outliers,
            'clip_range': self.clip_range
        }

    @classmethod
    def from_dict(cls, d: Dict) -> 'ScalerConfig':
        return cls(
            scaler_type=d.get('scaler_type', 'robust'),
            clip_outliers=d.get('clip_outliers', True),
            clip_range=tuple(d.get('clip_range', (-5.0, 5.0)))
        )


class FeatureCategory(Enum):
    """Feature categories for scaling strategy selection."""
    RETURNS = 'returns'           # Already normalized returns/percentages
    OSCILLATOR = 'oscillator'     # RSI, Stochastic (0-100 bounded)
    PRICE_LEVEL = 'price_level'   # Raw prices, SMAs
    VOLATILITY = 'volatility'     # ATR, std dev features
    VOLUME = 'volume'             # OBV, volume features
    TEMPORAL = 'temporal'         # Sin/cos encoded time features
    BINARY = 'binary'             # 0/1 flags
    UNKNOWN = 'unknown'           # Default category


# Feature categorization rules
FEATURE_PATTERNS: Dict[FeatureCategory, List[str]] = {
    FeatureCategory.RETURNS: [
        'return', 'log_return', 'simple_return', 'pct_change',
        'close_to_sma', 'close_to_ema', 'close_to_vwap', 'price_to_',
        'roc_', 'high_low_range', 'close_open_range', 'range_pct',
        'macd_hist', 'macd', 'macd_signal'
    ],
    FeatureCategory.OSCILLATOR: [
        'rsi', 'stoch_k', 'stoch_d', 'williams_r', 'cci', 'mfi',
        'bb_position', 'adx', 'plus_di', 'minus_di'
    ],
    FeatureCategory.PRICE_LEVEL: [
        'sma_', 'ema_', 'bb_upper', 'bb_lower', 'bb_middle',
        'kc_upper', 'kc_lower', 'kc_middle', 'vwap', 'supertrend'
    ],
    FeatureCategory.VOLATILITY: [
        'atr_', 'hvol_', 'parkinson', 'gk_vol', 'bb_width', 'kc_position'
    ],
    FeatureCategory.VOLUME: [
        'obv', 'volume_', 'obv_sma'
    ],
    FeatureCategory.TEMPORAL: [
        'hour_sin', 'hour_cos', 'minute_sin', 'minute_cos',
        'dayofweek_sin', 'dayofweek_cos', 'dow_sin', 'dow_cos'
    ],
    FeatureCategory.BINARY: [
        'session_', 'is_rth', 'rsi_overbought', 'rsi_oversold',
        'stoch_overbought', 'stoch_oversold', 'adx_strong_trend',
        'macd_cross_up', 'macd_cross_down', 'supertrend_direction',
        'volatility_regime', 'trend_regime', 'vol_regime'
    ]
}

# Default scaling strategy per category
DEFAULT_SCALING_STRATEGY: Dict[FeatureCategory, ScalerType] = {
    FeatureCategory.RETURNS: ScalerType.NONE,          # Already normalized
    FeatureCategory.OSCILLATOR: ScalerType.MINMAX,     # Keep 0-100 range
    FeatureCategory.PRICE_LEVEL: ScalerType.ROBUST,    # Log transform recommended
    FeatureCategory.VOLATILITY: ScalerType.ROBUST,     # May need log transform
    FeatureCategory.VOLUME: ScalerType.ROBUST,         # Often skewed
    FeatureCategory.TEMPORAL: ScalerType.NONE,         # Already normalized
    FeatureCategory.BINARY: ScalerType.NONE,           # Keep as 0/1
    FeatureCategory.UNKNOWN: ScalerType.ROBUST         # Default to robust
}


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class FeatureScalingConfig:
    """Configuration for a single feature's scaling."""
    feature_name: str
    category: FeatureCategory
    scaler_type: ScalerType
    apply_log_transform: bool = False
    log_shift: float = 0.0

    def to_dict(self) -> Dict:
        return {
            'feature_name': self.feature_name,
            'category': self.category.value,
            'scaler_type': self.scaler_type.value,
            'apply_log_transform': self.apply_log_transform,
            'log_shift': self.log_shift
        }

    @classmethod
    def from_dict(cls, d: Dict) -> 'FeatureScalingConfig':
        return cls(
            feature_name=d['feature_name'],
            category=FeatureCategory(d['category']),
            scaler_type=ScalerType(d['scaler_type']),
            apply_log_transform=d.get('apply_log_transform', False),
            log_shift=d.get('log_shift', 0.0)
        )


@dataclass
class ScalingStatistics:
    """Statistics for a scaled feature."""
    feature_name: str
    train_mean: float
    train_std: float
    train_min: float
    train_max: float
    train_median: float
    train_q25: float
    train_q75: float
    scaled_mean: float
    scaled_std: float
    scaled_min: float
    scaled_max: float
    nan_count: int = 0
    inf_count: int = 0

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict) -> 'ScalingStatistics':
        return cls(**d)


@dataclass
class ScalingReport:
    """Complete scaling report."""
    timestamp: str
    n_features: int
    n_samples_train: int
    scaler_type: str
    features_by_category: Dict[str, List[str]]
    features_by_scaler: Dict[str, List[str]]
    statistics: Dict[str, Dict]
    warnings: List[str]
    errors: List[str]

    def to_dict(self) -> Dict:
        return asdict(self)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def categorize_feature(feature_name: str) -> FeatureCategory:
    """
    Determine the category of a feature based on its name.

    Args:
        feature_name: Name of the feature

    Returns:
        FeatureCategory enum value
    """
    feature_lower = feature_name.lower()

    for category, patterns in FEATURE_PATTERNS.items():
        for pattern in patterns:
            if pattern.lower() in feature_lower or feature_lower.startswith(pattern.lower()):
                return category

    return FeatureCategory.UNKNOWN


def get_default_scaler_type(category: FeatureCategory) -> ScalerType:
    """Get the default scaler type for a feature category."""
    return DEFAULT_SCALING_STRATEGY.get(category, ScalerType.ROBUST)


def should_log_transform(feature_name: str, category: FeatureCategory) -> bool:
    """
    Determine if a feature should have log transform applied.

    Log transform is recommended for:
    - Price level features (SMA, EMA, etc.)
    - Volume features (OBV, etc.)
    - Features with high positive skewness
    """
    if category in [FeatureCategory.PRICE_LEVEL, FeatureCategory.VOLUME]:
        # Check if it's a raw price/volume feature (not a ratio)
        if not any(x in feature_name.lower() for x in ['ratio', 'pct', 'zscore', 'to_']):
            return True
    return False


# =============================================================================
# MAIN FEATURE SCALER CLASS
# =============================================================================

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
        scaler_type: str = 'robust',
        feature_config: Optional[Dict[str, Dict]] = None,
        apply_log_to_price_volume: bool = True,
        robust_quantile_range: Tuple[float, float] = (25.0, 75.0),
        clip_outliers: bool = True,
        clip_range: Tuple[float, float] = (-5.0, 5.0),
        config: Optional[ScalerConfig] = None
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
        self.feature_names: List[str] = []
        self.scalers: Dict[str, Union[RobustScaler, StandardScaler, MinMaxScaler, None]] = {}
        self.configs: Dict[str, FeatureScalingConfig] = {}
        self.statistics: Dict[str, ScalingStatistics] = {}
        self.log_shifts: Dict[str, float] = {}

        # Metadata
        self.fit_timestamp: Optional[str] = None
        self.n_samples_train: int = 0
        self.warnings: List[str] = []
        self.errors: List[str] = []

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
                scaler_type=ScalerType(custom.get('scaler_type', self.default_scaler_type.value)),
                apply_log_transform=custom.get('apply_log_transform', False),
                log_shift=custom.get('log_shift', 0.0)
            )

        # Infer from feature name
        category = categorize_feature(feature_name)
        scaler_type = get_default_scaler_type(category)

        # Override with default scaler type if not NONE
        if scaler_type != ScalerType.NONE and self.default_scaler_type != ScalerType.ROBUST:
            scaler_type = self.default_scaler_type

        apply_log = (
            self.apply_log_to_price_volume and
            should_log_transform(feature_name, category)
        )

        return FeatureScalingConfig(
            feature_name=feature_name,
            category=category,
            scaler_type=scaler_type,
            apply_log_transform=apply_log,
            log_shift=0.0
        )

    def _create_scaler(self, scaler_type: ScalerType) -> Optional[Union[RobustScaler, StandardScaler, MinMaxScaler]]:
        """Create a sklearn scaler instance based on type."""
        if scaler_type == ScalerType.ROBUST:
            return RobustScaler(quantile_range=self.robust_quantile_range)
        elif scaler_type == ScalerType.STANDARD:
            return StandardScaler()
        elif scaler_type == ScalerType.MINMAX:
            return MinMaxScaler(feature_range=(0, 1))
        else:
            return None

    def _compute_statistics(
        self,
        data: np.ndarray,
        scaled_data: np.ndarray,
        feature_name: str
    ) -> ScalingStatistics:
        """Compute statistics for a feature before and after scaling."""
        # Handle NaN/Inf
        nan_count = int(np.isnan(data).sum())
        inf_count = int(np.isinf(data).sum())

        clean_data = data[~np.isnan(data) & ~np.isinf(data)]
        clean_scaled = scaled_data[~np.isnan(scaled_data) & ~np.isinf(scaled_data)]

        if len(clean_data) == 0:
            return ScalingStatistics(
                feature_name=feature_name,
                train_mean=np.nan, train_std=np.nan,
                train_min=np.nan, train_max=np.nan,
                train_median=np.nan, train_q25=np.nan, train_q75=np.nan,
                scaled_mean=np.nan, scaled_std=np.nan,
                scaled_min=np.nan, scaled_max=np.nan,
                nan_count=nan_count, inf_count=inf_count
            )

        return ScalingStatistics(
            feature_name=feature_name,
            train_mean=float(np.mean(clean_data)),
            train_std=float(np.std(clean_data)),
            train_min=float(np.min(clean_data)),
            train_max=float(np.max(clean_data)),
            train_median=float(np.median(clean_data)),
            train_q25=float(np.percentile(clean_data, 25)),
            train_q75=float(np.percentile(clean_data, 75)),
            scaled_mean=float(np.mean(clean_scaled)) if len(clean_scaled) > 0 else np.nan,
            scaled_std=float(np.std(clean_scaled)) if len(clean_scaled) > 0 else np.nan,
            scaled_min=float(np.min(clean_scaled)) if len(clean_scaled) > 0 else np.nan,
            scaled_max=float(np.max(clean_scaled)) if len(clean_scaled) > 0 else np.nan,
            nan_count=nan_count,
            inf_count=inf_count
        )

    def fit(
        self,
        train_df: pd.DataFrame,
        feature_cols: List[str]
    ) -> 'FeatureScaler':
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
        category_counts: Dict[str, int] = {}
        scaler_counts: Dict[str, int] = {}

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
                scaler = self._create_scaler(config.scaler_type)
                scaler.fit(col_data.reshape(-1, 1))
                self.scalers[fname] = scaler
                scaled_data = scaler.transform(col_data.reshape(-1, 1)).ravel()

            # Compute statistics
            original_data = train_df[fname].values.astype(np.float64)
            self.statistics[fname] = self._compute_statistics(
                original_data, scaled_data, fname
            )

            # Check for issues
            if self.statistics[fname].train_std < 1e-10:
                self.warnings.append(f"{fname}: Near-zero variance (std={self.statistics[fname].train_std:.2e})")

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

    def fit_transform(
        self,
        train_df: pd.DataFrame,
        feature_cols: List[str]
    ) -> pd.DataFrame:
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
        self,
        df: pd.DataFrame,
        features: Optional[List[str]] = None
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
            'version': '2.1',
            'default_scaler_type': self.default_scaler_type.value,
            'robust_quantile_range': self.robust_quantile_range,
            'apply_log_to_price_volume': self.apply_log_to_price_volume,
            'clip_outliers': self.clip_outliers,
            'clip_range': self.clip_range,
            'is_fitted': self.is_fitted,
            'feature_names': self.feature_names,
            'scalers': self.scalers,
            'configs': {k: v.to_dict() for k, v in self.configs.items()},
            'statistics': {k: v.to_dict() for k, v in self.statistics.items()},
            'log_shifts': self.log_shifts,
            'fit_timestamp': self.fit_timestamp,
            'n_samples_train': self.n_samples_train,
            'warnings': self.warnings,
            'errors': self.errors
        }

        with open(path, 'wb') as f:
            pickle.dump(state, f)

        logger.info(f"Scaler saved to: {path}")

        # Also save human-readable JSON report
        json_path = path.with_suffix('.json')
        self._save_json_report(json_path)

    def _save_json_report(self, path: Path) -> None:
        """Save a human-readable JSON report of the scaler configuration."""
        report = {
            'version': '2.0',
            'fit_timestamp': self.fit_timestamp,
            'n_samples_train': self.n_samples_train,
            'n_features': len(self.feature_names),
            'default_scaler_type': self.default_scaler_type.value,
            'features': {
                fname: {
                    'category': self.configs[fname].category.value,
                    'scaler_type': self.configs[fname].scaler_type.value,
                    'apply_log_transform': self.configs[fname].apply_log_transform,
                    'train_stats': {
                        'mean': self.statistics[fname].train_mean,
                        'std': self.statistics[fname].train_std,
                        'min': self.statistics[fname].train_min,
                        'max': self.statistics[fname].train_max
                    },
                    'scaled_stats': {
                        'mean': self.statistics[fname].scaled_mean,
                        'std': self.statistics[fname].scaled_std,
                        'min': self.statistics[fname].scaled_min,
                        'max': self.statistics[fname].scaled_max
                    }
                }
                for fname in self.feature_names
            },
            'warnings': self.warnings
        }

        with open(path, 'w') as f:
            json.dump(report, f, indent=2, default=str)

    @classmethod
    def load(cls, path: Path) -> 'FeatureScaler':
        """
        Load a persisted scaler from disk.

        Args:
            path: Path to the saved scaler

        Returns:
            Loaded FeatureScaler instance
        """
        path = Path(path)

        with open(path, 'rb') as f:
            state = pickle.load(f)

        scaler = cls(
            scaler_type=state['default_scaler_type'],
            apply_log_to_price_volume=state.get('apply_log_to_price_volume', True),
            robust_quantile_range=state.get('robust_quantile_range', (25.0, 75.0)),
            clip_outliers=state.get('clip_outliers', True),
            clip_range=tuple(state.get('clip_range', (-5.0, 5.0)))
        )

        scaler.is_fitted = state['is_fitted']
        scaler.feature_names = state['feature_names']
        scaler.scalers = state['scalers']
        scaler.configs = {
            k: FeatureScalingConfig.from_dict(v)
            for k, v in state['configs'].items()
        }
        scaler.statistics = {
            k: ScalingStatistics.from_dict(v)
            for k, v in state['statistics'].items()
        }
        scaler.log_shifts = state.get('log_shifts', {})
        scaler.fit_timestamp = state.get('fit_timestamp')
        scaler.n_samples_train = state.get('n_samples_train', 0)
        scaler.warnings = state.get('warnings', [])
        scaler.errors = state.get('errors', [])

        logger.info(f"Scaler loaded from: {path}")
        logger.info(f"  Features: {len(scaler.feature_names)}")
        logger.info(f"  Trained on: {scaler.n_samples_train:,} samples")

        return scaler

    def get_scaling_report(self) -> Dict:
        """
        Get a comprehensive report of scaling configuration and statistics.

        Returns:
            Dictionary with detailed scaling information
        """
        if not self.is_fitted:
            return {'is_fitted': False}

        # Group features by category
        features_by_category: Dict[str, List[str]] = {}
        for fname, config in self.configs.items():
            cat = config.category.value
            if cat not in features_by_category:
                features_by_category[cat] = []
            features_by_category[cat].append(fname)

        # Group features by scaler type
        features_by_scaler: Dict[str, List[str]] = {}
        for fname, config in self.configs.items():
            stype = config.scaler_type.value
            if stype not in features_by_scaler:
                features_by_scaler[stype] = []
            features_by_scaler[stype].append(fname)

        return {
            'is_fitted': True,
            'fit_timestamp': self.fit_timestamp,
            'n_samples_train': self.n_samples_train,
            'n_features': len(self.feature_names),
            'default_scaler_type': self.default_scaler_type.value,
            'features_by_category': features_by_category,
            'features_by_scaler': features_by_scaler,
            'statistics': {k: v.to_dict() for k, v in self.statistics.items()},
            'warnings': self.warnings,
            'errors': self.errors
        }


# =============================================================================
# VALIDATION FUNCTIONS
# =============================================================================

def validate_scaling(
    scaler: FeatureScaler,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: List[str],
    z_threshold: float = 5.0
) -> Dict:
    """
    Validate that scaling was done correctly.

    Checks:
    1. Train statistics match scaler's stored statistics
    2. Val/test statistics are reasonable relative to train
    3. No extreme outliers introduced by scaling
    4. No NaN/Inf values after scaling

    Args:
        scaler: Fitted FeatureScaler
        train_df: Original training DataFrame
        val_df: Validation DataFrame
        test_df: Test DataFrame
        feature_cols: Feature columns to validate
        z_threshold: Z-score threshold for outlier detection

    Returns:
        Validation report dictionary
    """
    report = {
        'is_valid': True,
        'timestamp': datetime.now().isoformat(),
        'issues': [],
        'warnings': [],
        'statistics': {}
    }

    if not scaler.is_fitted:
        report['is_valid'] = False
        report['issues'].append("Scaler is not fitted")
        return report

    # Transform all splits
    train_scaled = scaler.transform(train_df)
    val_scaled = scaler.transform(val_df)
    test_scaled = scaler.transform(test_df)

    for fname in feature_cols:
        train_col = train_scaled[fname].values
        val_col = val_scaled[fname].values
        test_col = test_scaled[fname].values

        # Check for NaN/Inf
        for name, col in [('train', train_col), ('val', val_col), ('test', test_col)]:
            nan_count = int(np.isnan(col).sum())
            inf_count = int(np.isinf(col).sum())
            if nan_count > 0:
                report['issues'].append(f"{fname} {name}: {nan_count} NaN values")
                report['is_valid'] = False
            if inf_count > 0:
                report['issues'].append(f"{fname} {name}: {inf_count} Inf values")
                report['is_valid'] = False

        # Check val/test statistics relative to train
        train_clean = train_col[~np.isnan(train_col) & ~np.isinf(train_col)]
        val_clean = val_col[~np.isnan(val_col) & ~np.isinf(val_col)]
        test_clean = test_col[~np.isnan(test_col) & ~np.isinf(test_col)]

        if len(train_clean) > 0 and len(val_clean) > 0:
            train_mean, train_std = np.mean(train_clean), np.std(train_clean)
            val_mean, val_std = np.mean(val_clean), np.std(val_clean)
            test_mean, test_std = np.mean(test_clean), np.std(test_clean) if len(test_clean) > 0 else (0, 0)

            # Check if val/test means are within z_threshold of train
            if train_std > 0:
                val_z = abs(val_mean - train_mean) / train_std
                if val_z > z_threshold:
                    report['warnings'].append(
                        f"{fname}: val mean differs significantly from train (z={val_z:.2f})"
                    )

                if len(test_clean) > 0:
                    test_z = abs(test_mean - train_mean) / train_std
                    if test_z > z_threshold:
                        report['warnings'].append(
                            f"{fname}: test mean differs significantly from train (z={test_z:.2f})"
                        )

            report['statistics'][fname] = {
                'train': {'mean': float(train_mean), 'std': float(train_std)},
                'val': {'mean': float(val_mean), 'std': float(val_std)},
                'test': {'mean': float(test_mean), 'std': float(test_std)}
            }

    return report


def validate_no_leakage(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    scaler: FeatureScaler
) -> Dict:
    """
    Validate that no data leakage occurred during scaling.

    This checks that the scaler's stored statistics match what would be
    computed from training data alone.

    Args:
        train_df: Training DataFrame
        val_df: Validation DataFrame
        test_df: Test DataFrame
        scaler: Fitted FeatureScaler

    Returns:
        Leakage validation report
    """
    report = {
        'leakage_detected': False,
        'checks': [],
        'issues': []
    }

    for fname in scaler.feature_names:
        train_data = train_df[fname].values.astype(np.float64)
        train_clean = train_data[~np.isnan(train_data) & ~np.isinf(train_data)]

        if len(train_clean) == 0:
            continue

        stored_mean = scaler.statistics[fname].train_mean
        stored_std = scaler.statistics[fname].train_std

        computed_mean = np.mean(train_clean)
        computed_std = np.std(train_clean)

        # Allow small floating point differences
        mean_diff = abs(stored_mean - computed_mean)
        std_diff = abs(stored_std - computed_std)

        check = {
            'feature': fname,
            'stored_mean': stored_mean,
            'computed_mean': computed_mean,
            'mean_diff': mean_diff,
            'stored_std': stored_std,
            'computed_std': computed_std,
            'std_diff': std_diff,
            'passed': mean_diff < 1e-6 and std_diff < 1e-6
        }
        report['checks'].append(check)

        if not check['passed']:
            report['leakage_detected'] = True
            report['issues'].append(
                f"{fname}: Statistics don't match training data "
                f"(mean_diff={mean_diff:.2e}, std_diff={std_diff:.2e})"
            )

    return report


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def scale_splits(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: List[str],
    scaler_path: Optional[Path] = None,
    config: Optional[ScalerConfig] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, FeatureScaler]:
    """
    Scale train/val/test splits with train-only fitting.

    This is the recommended simple interface for scaling data.

    Args:
        train_df: Training DataFrame
        val_df: Validation DataFrame
        test_df: Test DataFrame
        feature_cols: Feature columns to scale
        scaler_path: Optional path to save the fitted scaler
        config: Optional ScalerConfig for customization

    Returns:
        Tuple of (train_scaled, val_scaled, test_scaled, scaler)

    Example:
        >>> train_scaled, val_scaled, test_scaled, scaler = scale_splits(
        ...     train_df, val_df, test_df, feature_cols
        ... )
    """
    scaler = FeatureScaler(config=config) if config else FeatureScaler()

    # Fit ONLY on training data
    train_scaled = scaler.fit_transform(train_df, feature_cols)

    # Transform val and test using training statistics
    val_scaled = scaler.transform(val_df)
    test_scaled = scaler.transform(test_df)

    # Save scaler if path provided
    if scaler_path:
        scaler.save(scaler_path)

    logger.info("\nScaling complete:")
    logger.info(f"  Train: {len(train_scaled):,} samples")
    logger.info(f"  Val:   {len(val_scaled):,} samples")
    logger.info(f"  Test:  {len(test_scaled):,} samples")

    return train_scaled, val_scaled, test_scaled, scaler


def scale_train_val_test(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: List[str],
    scaler_type: str = 'robust',
    save_path: Optional[Path] = None,
    clip_outliers: bool = True,
    clip_range: Tuple[float, float] = (-5.0, 5.0)
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, FeatureScaler]:
    """
    Scale train/val/test data with a scaler fitted on training data only.

    This is an alternative interface with more explicit parameters.

    Args:
        train_df: Training DataFrame
        val_df: Validation DataFrame
        test_df: Test DataFrame
        feature_cols: Feature columns to scale
        scaler_type: Scaler type ('robust', 'standard', 'minmax')
        save_path: Optional path to save the fitted scaler
        clip_outliers: Whether to clip outliers after scaling
        clip_range: Range to clip scaled values to

    Returns:
        Tuple of (train_scaled, val_scaled, test_scaled, scaler)
    """
    scaler = FeatureScaler(
        scaler_type=scaler_type,
        clip_outliers=clip_outliers,
        clip_range=clip_range
    )

    # Fit ONLY on training data
    train_scaled = scaler.fit_transform(train_df, feature_cols)

    # Transform val and test using training statistics
    val_scaled = scaler.transform(val_df)
    test_scaled = scaler.transform(test_df)

    if save_path:
        scaler.save(save_path)

    logger.info("\nScaling complete:")
    logger.info(f"  Train: {len(train_scaled):,} samples")
    logger.info(f"  Val:   {len(val_scaled):,} samples")
    logger.info(f"  Test:  {len(test_scaled):,} samples")

    return train_scaled, val_scaled, test_scaled, scaler


# =============================================================================
# INTEGRATION WITH stage8_validate.py
# =============================================================================

def validate_scaling_for_splits(
    train_path: Path,
    val_path: Path,
    test_path: Path,
    feature_cols: Optional[List[str]] = None,
    scaler_type: str = 'robust',
    output_path: Optional[Path] = None
) -> Dict:
    """
    Validate that scaling works correctly for train/val/test splits.

    This function is designed to be called from stage8_validate.py or
    after running stage7_splits.py.

    Args:
        train_path: Path to training data parquet
        val_path: Path to validation data parquet
        test_path: Path to test data parquet
        feature_cols: Optional list of feature columns. Auto-detected if None.
        scaler_type: Scaler type to use ('robust', 'standard', 'minmax')
        output_path: Optional path to save validation report

    Returns:
        Validation report dictionary
    """
    logger.info("=" * 60)
    logger.info("SCALING VALIDATION FOR TRAIN/VAL/TEST SPLITS")
    logger.info("=" * 60)

    # Load data
    train_df = pd.read_parquet(train_path)
    val_df = pd.read_parquet(val_path)
    test_df = pd.read_parquet(test_path)

    logger.info(f"Train: {len(train_df):,} samples")
    logger.info(f"Val:   {len(val_df):,} samples")
    logger.info(f"Test:  {len(test_df):,} samples")

    # Identify feature columns if not provided
    if feature_cols is None:
        excluded_cols = {'datetime', 'symbol', 'open', 'high', 'low', 'close', 'volume'}
        excluded_prefixes = ('label_', 'bars_to_hit_', 'mae_', 'quality_', 'sample_weight_')
        feature_cols = [
            c for c in train_df.columns
            if c not in excluded_cols
            and not any(c.startswith(p) for p in excluded_prefixes)
        ]
    logger.info(f"Features: {len(feature_cols)}")

    # Create and fit scaler
    scaler = FeatureScaler(scaler_type=scaler_type)
    train_scaled = scaler.fit_transform(train_df, feature_cols)
    val_scaled = scaler.transform(val_df)
    test_scaled = scaler.transform(test_df)

    # Run validation
    scaling_validation = validate_scaling(
        scaler, train_df, val_df, test_df, feature_cols
    )

    leakage_validation = validate_no_leakage(
        train_df, val_df, test_df, scaler
    )

    # Combine reports
    report = {
        'timestamp': datetime.now().isoformat(),
        'train_samples': len(train_df),
        'val_samples': len(val_df),
        'test_samples': len(test_df),
        'n_features': len(feature_cols),
        'scaler_type': scaler_type,
        'scaling_validation': scaling_validation,
        'leakage_validation': leakage_validation,
        'scaler_report': scaler.get_scaling_report(),
        'overall_status': 'PASSED' if (
            scaling_validation['is_valid'] and
            not leakage_validation['leakage_detected']
        ) else 'FAILED'
    }

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        logger.info(f"Validation report saved to: {output_path}")

    # Log summary
    logger.info("\n" + "=" * 60)
    logger.info("SCALING VALIDATION SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Scaling valid: {scaling_validation['is_valid']}")
    logger.info(f"Leakage detected: {leakage_validation['leakage_detected']}")
    logger.info(f"Overall status: {report['overall_status']}")

    if scaling_validation['warnings']:
        logger.warning(f"Warnings ({len(scaling_validation['warnings'])}):")
        for w in scaling_validation['warnings'][:5]:
            logger.warning(f"  - {w}")

    if scaling_validation['issues']:
        logger.error(f"Issues ({len(scaling_validation['issues'])}):")
        for i in scaling_validation['issues']:
            logger.error(f"  - {i}")

    return report


def add_scaling_validation_to_stage8(
    validator: Any,  # DataValidator from stage8_validate
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: List[str],
    scaler_type: str = 'robust'
) -> Dict:
    """
    Add scaling validation results to a Stage 8 DataValidator.

    This function can be called from stage8_validate.py to include
    scaling validation in the overall validation report.

    Args:
        validator: DataValidator instance from stage8_validate
        train_df: Training DataFrame
        val_df: Validation DataFrame
        test_df: Test DataFrame
        feature_cols: List of feature columns
        scaler_type: Scaler type to use

    Returns:
        Scaling validation results
    """
    logger.info("\n" + "=" * 60)
    logger.info("SCALING VALIDATION (Phase 2 Preparation)")
    logger.info("=" * 60)

    # Create and fit scaler
    scaler = FeatureScaler(scaler_type=scaler_type)
    scaler.fit(train_df, feature_cols)

    # Validate
    scaling_validation = validate_scaling(
        scaler, train_df, val_df, test_df, feature_cols
    )

    leakage_validation = validate_no_leakage(
        train_df, val_df, test_df, scaler
    )

    results = {
        'scaling_validation': scaling_validation,
        'leakage_validation': leakage_validation,
        'scaler_summary': scaler.get_scaling_report()
    }

    # Add to validator's results
    validator.validation_results['scaling'] = results

    # Update warnings/issues
    if scaling_validation['warnings']:
        for w in scaling_validation['warnings']:
            validator.warnings_found.append(f"Scaling: {w}")

    if scaling_validation['issues']:
        for i in scaling_validation['issues']:
            validator.issues_found.append(f"Scaling: {i}")

    if leakage_validation['leakage_detected']:
        validator.issues_found.append("Scaling: Data leakage detected!")

    logger.info(f"Scaling validation: {'PASSED' if scaling_validation['is_valid'] else 'FAILED'}")
    logger.info(f"Leakage check: {'CLEAN' if not leakage_validation['leakage_detected'] else 'LEAKAGE DETECTED'}")

    return results


# =============================================================================
# TESTS
# =============================================================================

def test_fit_only_uses_train_data():
    """Test that fit() only uses training data statistics."""
    print("\n" + "="*60)
    print("TEST: fit_only_uses_train_data")
    print("="*60)

    # Create synthetic data with different distributions
    np.random.seed(42)

    train_df = pd.DataFrame({
        'feature1': np.random.normal(0, 1, 1000),
        'feature2': np.random.normal(100, 10, 1000)
    })

    val_df = pd.DataFrame({
        'feature1': np.random.normal(5, 2, 200),  # Different distribution
        'feature2': np.random.normal(200, 20, 200)
    })

    scaler = FeatureScaler(scaler_type='standard')
    scaler.fit(train_df, ['feature1', 'feature2'])

    # Check that stored statistics match training data
    # Note: We use np.mean/np.std for consistency (population std, ddof=0)
    train_mean = float(np.mean(train_df['feature1'].values))
    train_std = float(np.std(train_df['feature1'].values))
    assert abs(scaler.statistics['feature1'].train_mean - train_mean) < 1e-6, \
        f"Mean mismatch: {scaler.statistics['feature1'].train_mean} vs {train_mean}"
    assert abs(scaler.statistics['feature1'].train_std - train_std) < 1e-6, \
        f"Std mismatch: {scaler.statistics['feature1'].train_std} vs {train_std}"

    # Transform and check that train scaled mean is ~0 and std is ~1
    train_scaled = scaler.transform(train_df)
    assert abs(train_scaled['feature1'].mean()) < 0.1
    assert abs(train_scaled['feature1'].std() - 1.0) < 0.1

    # Val should have different mean (not centered at 0)
    val_scaled = scaler.transform(val_df)
    assert abs(val_scaled['feature1'].mean()) > 1.0  # Should be shifted

    print("PASSED: Scaler uses only training statistics")
    return True


def test_transform_uses_train_statistics():
    """Test that transform() uses training statistics, not new data statistics."""
    print("\n" + "="*60)
    print("TEST: transform_uses_train_statistics")
    print("="*60)

    np.random.seed(42)

    train_df = pd.DataFrame({
        'price': np.random.normal(100, 10, 1000),
        'volume': np.random.exponential(1000, 1000)
    })

    test_df = pd.DataFrame({
        'price': np.random.normal(150, 15, 200),  # Higher prices
        'volume': np.random.exponential(2000, 200)  # Higher volume
    })

    scaler = FeatureScaler(scaler_type='robust')
    train_scaled = scaler.fit_transform(train_df, ['price', 'volume'])
    test_scaled = scaler.transform(test_df)

    # Test should be scaled using train's median/IQR
    # So test scaled values should be shifted (not centered at 0)
    train_median = train_scaled['price'].median()
    test_median = test_scaled['price'].median()

    # Test median should be higher because we used train statistics
    assert test_median > train_median + 1.0

    print("PASSED: Transform uses training statistics")
    return True


def test_save_and_load_scaler():
    """Test that scaler can be saved and loaded correctly."""
    print("\n" + "="*60)
    print("TEST: save_and_load_scaler")
    print("="*60)

    import tempfile

    np.random.seed(42)

    train_df = pd.DataFrame({
        'rsi': np.random.uniform(0, 100, 1000),
        'log_return': np.random.normal(0, 0.01, 1000),
        'sma_20': np.random.normal(100, 10, 1000)
    })

    test_df = pd.DataFrame({
        'rsi': np.random.uniform(20, 80, 100),
        'log_return': np.random.normal(0, 0.01, 100),
        'sma_20': np.random.normal(105, 10, 100)
    })

    scaler = FeatureScaler(scaler_type='robust')
    train_scaled = scaler.fit_transform(train_df, ['rsi', 'log_return', 'sma_20'])
    test_scaled_before = scaler.transform(test_df)

    # Save and load
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / 'scaler.pkl'
        scaler.save(save_path)

        loaded_scaler = FeatureScaler.load(save_path)
        test_scaled_after = loaded_scaler.transform(test_df)

    # Results should be identical
    for col in test_df.columns:
        assert np.allclose(
            test_scaled_before[col].values,
            test_scaled_after[col].values,
            equal_nan=True
        ), f"Mismatch in {col}"

    # Check that configuration was preserved
    assert loaded_scaler.feature_names == scaler.feature_names
    assert loaded_scaler.n_samples_train == scaler.n_samples_train

    print("PASSED: Save and load works correctly")
    return True


def test_different_scaler_types():
    """Test that different scaler types work correctly."""
    print("\n" + "="*60)
    print("TEST: different_scaler_types")
    print("="*60)

    np.random.seed(42)

    train_df = pd.DataFrame({
        'feature': np.random.exponential(10, 1000)
    })

    results = {}
    for scaler_type in ['standard', 'robust', 'minmax']:
        scaler = FeatureScaler(scaler_type=scaler_type)
        scaled = scaler.fit_transform(train_df, ['feature'])
        results[scaler_type] = {
            'mean': scaled['feature'].mean(),
            'std': scaled['feature'].std(),
            'min': scaled['feature'].min(),
            'max': scaled['feature'].max()
        }
        print(f"  {scaler_type}: mean={results[scaler_type]['mean']:.3f}, "
              f"std={results[scaler_type]['std']:.3f}, "
              f"min={results[scaler_type]['min']:.3f}, "
              f"max={results[scaler_type]['max']:.3f}")

    # StandardScaler should have mean ~0, std ~1
    assert abs(results['standard']['mean']) < 0.1
    assert abs(results['standard']['std'] - 1.0) < 0.1

    # MinMaxScaler should have min=0, max=1
    assert abs(results['minmax']['min'] - 0.0) < 0.01
    assert abs(results['minmax']['max'] - 1.0) < 0.01

    # RobustScaler should have median ~0
    # (median is different from mean for exponential)

    print("PASSED: All scaler types work correctly")
    return True


def test_feature_categorization():
    """Test that features are correctly categorized."""
    print("\n" + "="*60)
    print("TEST: feature_categorization")
    print("="*60)

    test_cases = [
        ('log_return', FeatureCategory.RETURNS),
        ('rsi', FeatureCategory.OSCILLATOR),
        ('sma_20', FeatureCategory.PRICE_LEVEL),
        ('atr_14', FeatureCategory.VOLATILITY),
        ('obv', FeatureCategory.VOLUME),
        ('hour_sin', FeatureCategory.TEMPORAL),
        ('session_ny', FeatureCategory.BINARY),
        ('unknown_feature', FeatureCategory.UNKNOWN)
    ]

    for feature_name, expected_category in test_cases:
        actual_category = categorize_feature(feature_name)
        assert actual_category == expected_category, \
            f"{feature_name}: expected {expected_category}, got {actual_category}"
        print(f"  {feature_name:20s} -> {actual_category.value}")

    print("PASSED: Feature categorization is correct")
    return True


def test_outlier_clipping():
    """Test that outlier clipping works correctly."""
    print("\n" + "="*60)
    print("TEST: outlier_clipping")
    print("="*60)

    np.random.seed(42)

    # Create data with extreme outliers
    normal_data = np.random.normal(0, 1, 990)
    outliers = np.array([50, -50, 100, -100, 200, -200, 500, -500, 1000, -1000])
    data = np.concatenate([normal_data, outliers])

    train_df = pd.DataFrame({'feature': data})

    # Test with clipping enabled
    scaler_clip = FeatureScaler(
        scaler_type='standard',
        clip_outliers=True,
        clip_range=(-5.0, 5.0)
    )
    scaled_clip = scaler_clip.fit_transform(train_df, ['feature'])

    # Verify all values are within clip range
    assert scaled_clip['feature'].max() <= 5.0, "Max should be clipped to 5.0"
    assert scaled_clip['feature'].min() >= -5.0, "Min should be clipped to -5.0"
    print(f"  With clipping: min={scaled_clip['feature'].min():.2f}, max={scaled_clip['feature'].max():.2f}")

    # Test with clipping disabled
    scaler_no_clip = FeatureScaler(
        scaler_type='standard',
        clip_outliers=False
    )
    scaled_no_clip = scaler_no_clip.fit_transform(train_df, ['feature'])

    # Verify extreme values are NOT clipped
    assert scaled_no_clip['feature'].max() > 5.0, "Without clipping, max should exceed 5.0"
    assert scaled_no_clip['feature'].min() < -5.0, "Without clipping, min should be below -5.0"
    print(f"  Without clipping: min={scaled_no_clip['feature'].min():.2f}, max={scaled_no_clip['feature'].max():.2f}")

    # Test with ScalerConfig
    config = ScalerConfig(scaler_type='robust', clip_outliers=True, clip_range=(-3.0, 3.0))
    scaler_config = FeatureScaler(config=config)
    scaled_config = scaler_config.fit_transform(train_df, ['feature'])

    assert scaled_config['feature'].max() <= 3.0, "Max should be clipped to 3.0 with config"
    assert scaled_config['feature'].min() >= -3.0, "Min should be clipped to -3.0 with config"
    print(f"  With ScalerConfig (-3,3): min={scaled_config['feature'].min():.2f}, max={scaled_config['feature'].max():.2f}")

    print("PASSED: Outlier clipping works correctly")
    return True


def test_scale_splits_function():
    """Test the scale_splits convenience function."""
    print("\n" + "="*60)
    print("TEST: scale_splits_function")
    print("="*60)

    np.random.seed(42)

    # Create synthetic train/val/test data
    train_df = pd.DataFrame({
        'feature1': np.random.normal(0, 1, 1000),
        'feature2': np.random.normal(100, 10, 1000)
    })
    val_df = pd.DataFrame({
        'feature1': np.random.normal(0.5, 1.2, 200),
        'feature2': np.random.normal(105, 12, 200)
    })
    test_df = pd.DataFrame({
        'feature1': np.random.normal(0.3, 1.1, 200),
        'feature2': np.random.normal(102, 11, 200)
    })

    # Test scale_splits
    train_scaled, val_scaled, test_scaled, scaler = scale_splits(
        train_df, val_df, test_df, ['feature1', 'feature2']
    )

    # Verify shapes
    assert len(train_scaled) == len(train_df), "Train size mismatch"
    assert len(val_scaled) == len(val_df), "Val size mismatch"
    assert len(test_scaled) == len(test_df), "Test size mismatch"

    # Verify scaler is fitted
    assert scaler.is_fitted, "Scaler should be fitted"
    assert len(scaler.feature_names) == 2, "Should have 2 features"

    # Verify train is scaled (approximately zero mean for robust scaler)
    # Note: RobustScaler uses median, so we check that median is near 0
    print(f"  Train feature1 median: {train_scaled['feature1'].median():.4f}")
    print(f"  Val feature1 median: {val_scaled['feature1'].median():.4f}")
    print(f"  Test feature1 median: {test_scaled['feature1'].median():.4f}")

    print("PASSED: scale_splits function works correctly")
    return True


def run_all_tests():
    """Run all unit tests."""
    print("\n" + "="*70)
    print("RUNNING FEATURE SCALER TESTS")
    print("="*70)

    tests = [
        test_fit_only_uses_train_data,
        test_transform_uses_train_statistics,
        test_save_and_load_scaler,
        test_different_scaler_types,
        test_feature_categorization,
        test_outlier_clipping,
        test_scale_splits_function
    ]

    passed = 0
    failed = 0

    for test_func in tests:
        try:
            test_func()
            passed += 1
        except AssertionError as e:
            print(f"FAILED: {e}")
            failed += 1
        except Exception as e:
            print(f"ERROR: {e}")
            failed += 1

    print("\n" + "="*70)
    print(f"TEST RESULTS: {passed} passed, {failed} failed")
    print("="*70)

    return failed == 0


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Run tests and demonstrate usage."""
    import sys

    # Run tests
    if not run_all_tests():
        sys.exit(1)

    print("\n" + "="*70)
    print("FEATURE SCALER DEMONSTRATION")
    print("="*70)

    # Create sample data
    np.random.seed(42)
    n_train = 10000
    n_val = 2000
    n_test = 2000

    # Simulate feature data with realistic distributions
    def create_sample_data(n):
        return pd.DataFrame({
            'log_return': np.random.normal(0, 0.001, n),
            'rsi': np.random.uniform(20, 80, n),
            'sma_20': np.abs(np.random.normal(100, 10, n)) + 50,
            'atr_14': np.abs(np.random.exponential(1, n)),
            'obv': np.random.exponential(10000, n),
            'hour_sin': np.sin(np.random.uniform(0, 2*np.pi, n)),
            'session_ny': np.random.choice([0, 1], n)
        })

    train_df = create_sample_data(n_train)
    val_df = create_sample_data(n_val)
    test_df = create_sample_data(n_test)

    # Scale using the convenience function
    feature_cols = list(train_df.columns)
    train_scaled, val_scaled, test_scaled, scaler = scale_train_val_test(
        train_df, val_df, test_df, feature_cols
    )

    # Get and print report
    report = scaler.get_scaling_report()
    print("\nScaling Report:")
    print(f"  Features: {report['n_features']}")
    print(f"  Training samples: {report['n_samples_train']:,}")
    print("\n  Features by category:")
    for cat, features in report['features_by_category'].items():
        print(f"    {cat}: {len(features)}")
    print("\n  Features by scaler:")
    for stype, features in report['features_by_scaler'].items():
        print(f"    {stype}: {len(features)}")

    # Validate scaling
    print("\n" + "="*60)
    print("VALIDATION")
    print("="*60)

    validation = validate_scaling(
        scaler, train_df, val_df, test_df, feature_cols
    )
    print(f"\nValidation passed: {validation['is_valid']}")
    if validation['warnings']:
        print(f"Warnings: {len(validation['warnings'])}")
        for w in validation['warnings'][:3]:
            print(f"  - {w}")

    leakage_check = validate_no_leakage(train_df, val_df, test_df, scaler)
    print(f"\nLeakage detected: {leakage_check['leakage_detected']}")

    print("\nDemonstration complete.")


if __name__ == '__main__':
    main()
