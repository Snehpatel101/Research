"""
Core type definitions - Enums and type aliases for the ML Factory.

PHASE_0: Foundation types that define the vocabulary for the entire system.
All other modules should import types from here to ensure consistency.

This is the SINGLE SOURCE OF TRUTH for:
- DataRank (2D/3D/4D tensor ranks)
- ModelFamily (boosting/classical/neural/ensemble/meta_learner)
- FeatureFamily (momentum/volatility/volume/etc.)
- TrainingMode (standard/walk_forward/regime_aware/meta_labeling)
- CVMethod (purged_kfold/cpcv/walk_forward/pbo)
"""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, TypeVar

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from src.core.contracts import ModelContract


# =============================================================================
# DATA RANK - Tensor dimensionality for different model types
# =============================================================================


class DataRank(int, Enum):
    """
    Data tensor dimensionality for model input.

    - TABULAR_2D: (n_samples, n_features) - boosting, classical, meta-learners
    - SEQUENCE_3D: (n_samples, seq_len, n_features) - RNN, CNN, most neural
    - MULTI_TF_4D: (n_samples, n_timeframes, seq_len, n_features) - PatchTST, iTransformer
    """

    TABULAR_2D = 2
    SEQUENCE_3D = 3
    MULTI_TF_4D = 4

    @classmethod
    def from_model(cls, model_name: str) -> DataRank:
        """Get the expected data rank for a model."""
        from src.core.constants import MODEL_DATA_RANKS

        rank = MODEL_DATA_RANKS.get(model_name.lower())
        if rank is None:
            raise ValueError(f"Unknown model: {model_name}")
        return cls(rank)

    @classmethod
    def from_ndim(cls, ndim: int) -> DataRank:
        """Get DataRank from array dimensions."""
        for rank in cls:
            if rank.value == ndim:
                return rank
        raise ValueError(f"Unsupported array dimension: {ndim}. Must be 2, 3, or 4.")


# =============================================================================
# MODEL FAMILY - Groupings of models with similar characteristics
# =============================================================================


class ModelFamily(str, Enum):
    """
    Model family classification.

    Determines:
    - Default feature selection strategy
    - Adapter type (2D/3D/4D)
    - Training hyperparameter defaults
    """

    BOOSTING = "boosting"
    CLASSICAL = "classical"
    NEURAL = "neural"
    ENSEMBLE = "ensemble"
    META_LEARNER = "meta_learner"
    TRANSFORMER = "transformer"

    @classmethod
    def from_model(cls, model_name: str) -> ModelFamily:
        """Get the family for a model."""
        from src.core.constants import MODEL_TO_FAMILY

        family = MODEL_TO_FAMILY.get(model_name.lower())
        if family is None:
            raise ValueError(f"Unknown model: {model_name}")
        return cls(family)


# =============================================================================
# FEATURE FAMILY - 12 feature families (162 total features)
# =============================================================================


class FeatureFamily(str, Enum):
    """
    Feature family classification (12 families, 162 base features).

    Counts (from PHASE_1_UNIFIED_FEATURES.md):
    - RAW: 5 (open, high, low, close, volume)
    - MOMENTUM: 23 (RSI, MACD, Stochastic, etc.)
    - MOVING_AVERAGE: 16 (SMA, EMA, crossovers)
    - VOLATILITY: 25 (ATR, BB, Keltner, GARCH)
    - VOLUME: 15 (OBV, VWAP, TWAP)
    - TREND: 6 (ADX, Supertrend)
    - PRICE: 12 (returns, ratios, autocorr)
    - MICROSTRUCTURE: 15 (Amihud, Roll, Kyle)
    - ENTROPY: 12 (Shannon, LZ, ApEn, Hurst)
    - WAVELETS: 15 (DWT coefficients)
    - TEMPORAL: 9 (hour/day encoding, session)
    - REGIME: 9 (vol/trend regimes)
    - MTF: ~30 per higher timeframe
    """

    RAW = "raw"
    MOMENTUM = "momentum"
    MOVING_AVERAGE = "moving_average"
    VOLATILITY = "volatility"
    VOLUME = "volume"
    TREND = "trend"
    PRICE = "price"
    MICROSTRUCTURE = "microstructure"
    ENTROPY = "entropy"
    WAVELETS = "wavelets"
    TEMPORAL = "temporal"
    REGIME = "regime"
    MTF = "mtf"


# =============================================================================
# TRAINING MODE - How the model is trained
# =============================================================================


class TrainingMode(str, Enum):
    """
    Training mode - determines training procedure.

    - STANDARD: Simple train/val/test split with CV
    - WALK_FORWARD: Rolling window training for production simulation
    - REGIME_AWARE: Separate models per market regime
    - META_LABELING: Two-stage labeling (primary + confidence)
    """

    STANDARD = "standard"
    WALK_FORWARD = "walk_forward"
    REGIME_AWARE = "regime_aware"
    META_LABELING = "meta_labeling"


# =============================================================================
# CV METHOD - Cross-validation approach
# =============================================================================


class CVMethod(str, Enum):
    """
    Cross-validation method with proper purge/embargo handling.

    - PURGED_KFOLD: Standard k-fold with purge gaps
    - CPCV: Combinatorial Purged Cross-Validation
    - WALK_FORWARD: Expanding/sliding window
    - PBO: Probability of Backtest Overfitting
    """

    PURGED_KFOLD = "purged_kfold"
    CPCV = "cpcv"
    WALK_FORWARD = "walk_forward"
    PBO = "pbo"


# =============================================================================
# ADAPTER TYPE - Data transformation adapter
# =============================================================================


class AdapterType(str, Enum):
    """
    Adapter type for model data preparation.

    - TABULAR: 2D arrays for boosting/classical
    - SEQUENCE: 3D sliding window for RNN/CNN
    - MULTI_STREAM: 4D multi-timeframe for advanced transformers
    """

    TABULAR = "tabular"
    SEQUENCE = "sequence"
    MULTI_STREAM = "multi_stream"

    @classmethod
    def from_rank(cls, rank: int) -> AdapterType:
        """Get adapter type from data rank."""
        mapping = {2: cls.TABULAR, 3: cls.SEQUENCE, 4: cls.MULTI_STREAM}
        if rank not in mapping:
            raise ValueError(f"Invalid rank: {rank}. Must be 2, 3, or 4.")
        return mapping[rank]


# =============================================================================
# LABELING METHOD - How labels are generated
# =============================================================================


class LabelingMethod(str, Enum):
    """
    Labeling method for target generation.

    - TRIPLE_BARRIER: ATR-based upper/lower barriers with timeout
    - DIRECTIONAL: Simple direction-based (up/down/neutral)
    - THRESHOLD: Fixed percentage thresholds
    - REGRESSION: Continuous return values
    """

    TRIPLE_BARRIER = "triple_barrier"
    DIRECTIONAL = "directional"
    THRESHOLD = "threshold"
    REGRESSION = "regression"


# =============================================================================
# TYPE ALIASES - For type hints throughout the codebase
# =============================================================================

# Feature data types
Features = np.ndarray | pd.DataFrame
Labels = np.ndarray

# Model type variable (for generic model handling)
ModelType = TypeVar("ModelType", bound="ModelContract")

# Common array types
Array1D = np.ndarray  # Shape: (n,)
Array2D = np.ndarray  # Shape: (n, features)
Array3D = np.ndarray  # Shape: (n, seq_len, features)
Array4D = np.ndarray  # Shape: (n, timeframes, seq_len, features)

# Index types
DatetimeIndex = pd.DatetimeIndex
Index = np.ndarray | pd.Index


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Enums
    "DataRank",
    "ModelFamily",
    "FeatureFamily",
    "TrainingMode",
    "CVMethod",
    "AdapterType",
    "LabelingMethod",
    # Type aliases
    "Features",
    "Labels",
    "ModelType",
    "Array1D",
    "Array2D",
    "Array3D",
    "Array4D",
    "DatetimeIndex",
    "Index",
]
