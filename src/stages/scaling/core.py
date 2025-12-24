"""
Feature Scaler Core Classes and Configuration

This module contains the core data classes, enums, and configuration structures
for the feature scaling infrastructure.

Author: ML Pipeline
Created: 2025-12-20
Updated: 2025-12-20 - Extracted from feature_scaler.py
"""

import logging
from dataclasses import dataclass, asdict
from enum import Enum
from typing import Dict, List, Tuple

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
        'volatility_regime', 'trend_regime', 'vol_regime',
        'missing_bar', 'roll_event', 'roll_window', 'filled'
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
