"""
Feature Scaler Core Classes and Configuration

This module contains the core data classes, enums, and configuration structures
for the feature scaling infrastructure.

Author: ML Pipeline
Created: 2025-12-20
Updated: 2025-12-20 - Extracted from feature_scaler.py
Updated: 2025-12-24 - Added model-family scaling documentation

# =============================================================================
# MODEL-FAMILY SCALING RECOMMENDATIONS
# =============================================================================
#
# Different model families have different scaling requirements:
#
# BOOSTING MODELS (XGBoost, LightGBM, CatBoost):
#   - Recommended scaler: NONE
#   - Rationale: Tree-based models split on feature values, not magnitudes.
#     They are invariant to monotonic transformations and handle different
#     scales internally. Scaling adds no benefit and may hurt interpretability.
#   - Exception: Some implementations benefit from scaling for numerical stability
#     with very large/small values (>1e6 or <1e-6).
#
# NEURAL NETWORKS (LSTM, GRU, MLP):
#   - Recommended scaler: ROBUST
#   - Rationale: Neural networks use gradient-based optimization which is
#     sensitive to feature scales. RobustScaler is preferred because:
#     1. Uses median/IQR instead of mean/std - robust to outliers
#     2. Financial data often has fat tails and extreme values
#     3. Prevents gradient explosion from outlier samples
#   - Alternative: StandardScaler works if outliers are pre-clipped
#   - Clip range: (-5.0, 5.0) after scaling to bound extreme values
#
# TRANSFORMER MODELS:
#   - Recommended scaler: STANDARD
#   - Rationale: Transformers use layer normalization internally and
#     benefit from input features with mean=0, std=1. StandardScaler
#     provides this property directly.
#   - Note: Pre-trained transformers may have specific input requirements
#
# LINEAR MODELS (Logistic, Ridge, Lasso):
#   - Recommended scaler: STANDARD
#   - Rationale: Regularization (L1/L2) penalizes coefficient magnitude.
#     Without scaling, features with larger values dominate regularization.
#   - Critical: Always scale for regularized linear models
#
# ENSEMBLE META-LEARNERS:
#   - Recommended scaler: ROBUST or model-specific
#   - Rationale: Base model predictions should be on similar scales for
#     the meta-learner to weight them appropriately.
#
# =============================================================================
"""

import logging
from dataclasses import asdict, dataclass
from enum import Enum

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

    Notes:
        Default 'robust' scaler uses median/IQR which is optimal for:
        - Financial data with fat-tailed distributions
        - Datasets with outliers (common in OHLCV data)
        - Neural network training (prevents gradient explosion)

        See module docstring for model-family specific recommendations.
    """
    scaler_type: str = 'robust'
    clip_outliers: bool = True
    clip_range: tuple[float, float] = (-5.0, 5.0)

    def to_dict(self) -> dict:
        return {
            'scaler_type': self.scaler_type,
            'clip_outliers': self.clip_outliers,
            'clip_range': self.clip_range
        }

    @classmethod
    def from_dict(cls, d: dict) -> 'ScalerConfig':
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
FEATURE_PATTERNS: dict[FeatureCategory, list[str]] = {
    FeatureCategory.RETURNS: [
        'return', 'log_return', 'simple_return', 'pct_change',
        'close_to_sma', 'close_to_ema', 'close_to_vwap', 'price_to_',
        'roc_', 'high_low_range', 'close_open_range', 'range_pct',
        'macd_hist', 'macd', 'macd_signal',
        # Ratio features (normalized returns)
        'hl_ratio', 'co_ratio', 'volume_ratio',
        'close_sma20_ratio', 'close_ema21_ratio',
        # Z-score features (normalized)
        'close_bb_zscore', 'volume_zscore', '_zscore',
        # Deviation features
        'close_kc_atr_dev', '_dev',
    ],
    FeatureCategory.OSCILLATOR: [
        'rsi', 'stoch_k', 'stoch_d', 'williams_r', 'cci', 'mfi',
        'bb_position', 'adx', 'plus_di', 'minus_di',
    ],
    FeatureCategory.PRICE_LEVEL: [
        'sma_', 'ema_', 'bb_upper', 'bb_lower', 'bb_middle',
        'kc_upper', 'kc_lower', 'kc_middle', 'vwap', 'supertrend',
        # MTF price levels
        'open_', 'high_', 'low_', 'close_'
    ],
    FeatureCategory.VOLATILITY: [
        'atr_', 'hvol_', 'parkinson', 'gk_vol', 'rs_vol', 'yz_vol',
        'bb_width', 'kc_position'
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
        'volatility_regime', 'trend_regime', 'vol_regime', 'structure_regime',
        'missing_bar', 'roll_event', 'roll_window', 'filled',
        # Metadata flags
        'timeframe',
    ]
}

# Default scaling strategy per category
DEFAULT_SCALING_STRATEGY: dict[FeatureCategory, ScalerType] = {
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

    def to_dict(self) -> dict:
        return {
            'feature_name': self.feature_name,
            'category': self.category.value,
            'scaler_type': self.scaler_type.value,
            'apply_log_transform': self.apply_log_transform,
            'log_shift': self.log_shift
        }

    @classmethod
    def from_dict(cls, d: dict) -> 'FeatureScalingConfig':
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

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> 'ScalingStatistics':
        return cls(**d)


@dataclass
class ScalingReport:
    """Complete scaling report."""
    timestamp: str
    n_features: int
    n_samples_train: int
    scaler_type: str
    features_by_category: dict[str, list[str]]
    features_by_scaler: dict[str, list[str]]
    statistics: dict[str, dict]
    warnings: list[str]
    errors: list[str]

    def to_dict(self) -> dict:
        return asdict(self)
