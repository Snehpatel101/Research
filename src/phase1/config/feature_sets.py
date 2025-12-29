"""
Feature set definitions for modular model training.

Provides named feature sets that can be selected without code edits.
"""
from dataclasses import dataclass, field


@dataclass(frozen=True)
class FeatureSetDefinition:
    """
    Definition of a named feature set.

    Each symbol is processed in complete isolation - there are no cross-symbol
    or cross-asset features. All features are computed from single-symbol data only.

    Attributes:
        name: Unique identifier for this feature set
        description: Human-readable description
        include_prefixes: Feature name prefixes to include
        exclude_prefixes: Feature name prefixes to exclude
        include_columns: Specific columns to include
        exclude_columns: Specific columns to exclude
        include_mtf: Whether to include multi-timeframe features
        supported_model_types: Model types that work with this feature set
        default_sequence_length: Default sequence length for sequential models
        recommended_scaler: Recommended scaler type for this feature set
    """
    name: str
    description: str
    include_prefixes: list[str] = field(default_factory=list)
    exclude_prefixes: list[str] = field(default_factory=list)
    include_columns: list[str] = field(default_factory=list)
    exclude_columns: list[str] = field(default_factory=list)
    include_mtf: bool = False
    supported_model_types: list[str] = field(default_factory=lambda: [
        "tabular", "sequential", "tree"
    ])
    default_sequence_length: int | None = None
    recommended_scaler: str = "robust"


# =============================================================================
# FEATURE COUNT GUIDELINES
# =============================================================================
# Sample-to-feature ratio is critical to prevent overfitting:
#
# MINIMUM RATIOS:
#   - 10:1  - Absolute minimum (50K samples -> max 5000 features)
#   - 20:1  - Preferred for production (50K samples -> 2500 features)
#   - 50:1  - Conservative, for high-stakes models (50K samples -> 1000 features)
#
# RECOMMENDED FEATURE COUNTS BY SAMPLE SIZE:
#   - 50K samples:   50-80 features (optimal), max 100 features
#   - 100K samples:  80-150 features (optimal), max 200 features
#   - 500K samples:  150-300 features (optimal), max 500 features
#
# CURRENT STATE:
#   This pipeline produces 150+ features which may cause overfitting with
#   smaller datasets (<100K samples). The model-family feature sets below
#   provide reduced feature counts optimized for different model types.
#
# MITIGATION STRATEGIES:
#   1. Use model-specific feature sets (boosting_optimal, neural_optimal)
#   2. Apply correlation-based pruning (CORRELATION_THRESHOLD=0.80)
#   3. Use regularization in models (L1/L2, dropout)
#   4. Collect more training data when possible
# =============================================================================

FEATURE_SET_DEFINITIONS: dict[str, FeatureSetDefinition] = {
    "core_min": FeatureSetDefinition(
        name="core_min",
        description="Minimal base-timeframe feature set (no MTF). Single symbol only.",
        include_prefixes=[
            "return_", "log_return_", "roc_", "rsi_", "macd_", "stoch_", "williams_",
            "cci_", "mfi_", "atr_", "bb_", "kc_", "hvol_", "parkinson_", "garman_",
            "volume_", "vwap", "obv", "adx_", "supertrend", "range_", "hl_", "co_",
            "hour_", "minute_", "dayofweek_", "session_", "is_rth", "trend_regime",
            "volatility_regime",
        ],
        include_columns=["price_to_vwap"],
        include_mtf=False,
        supported_model_types=["tabular", "tree", "sequential"],
        default_sequence_length=60,
        recommended_scaler="robust",
    ),
    "core_full": FeatureSetDefinition(
        name="core_full",
        description="All base-timeframe features (no MTF). Single symbol only.",
        include_prefixes=[],
        include_mtf=False,
        supported_model_types=["tabular", "tree", "sequential"],
        default_sequence_length=60,
        recommended_scaler="robust",
    ),
    "mtf_plus": FeatureSetDefinition(
        name="mtf_plus",
        description="All base-timeframe features plus MTF features. Single symbol only.",
        include_prefixes=[],
        include_mtf=True,
        supported_model_types=["tabular", "tree", "sequential"],
        default_sequence_length=120,
        recommended_scaler="robust",
    ),
    # =========================================================================
    # MODEL-FAMILY OPTIMIZED FEATURE SETS
    # =========================================================================
    # These feature sets are designed for specific model families based on
    # their characteristics and requirements. Use these for Phase 2+ training.
    #
    # Selection criteria:
    # - boosting_optimal: For XGBoost, LightGBM, CatBoost (handles raw features)
    # - neural_optimal: For LSTM, GRU (needs normalized, sequential features)
    # - transformer_raw: For foundation models (minimal features, learns patterns)
    # - ensemble_base: For stacking/blending (diverse, uncorrelated features)
    # =========================================================================
    "boosting_optimal": FeatureSetDefinition(
        name="boosting_optimal",
        description=(
            "Optimal feature set for gradient boosting models (XGBoost, LightGBM, CatBoost). "
            "50-100 features, includes all feature types. Tree-based models handle "
            "correlated features and different scales internally, so no scaling required."
        ),
        include_prefixes=[
            # Price action (most important for boosting)
            "return_", "log_return_", "roc_",
            # Momentum oscillators
            "rsi_", "macd_", "stoch_", "williams_", "cci_", "mfi_",
            # Trend indicators
            "adx_", "supertrend",
            # Volatility (critical for risk-adjusted predictions)
            "atr_", "hvol_", "parkinson_", "garman_", "bb_width",
            # Volume features
            "volume_", "obv",
            # Price position features
            "bb_position", "kc_position", "price_to_",
            # Temporal features (important for regime detection)
            "hour_", "dayofweek_", "session_",
        ],
        include_columns=["is_rth", "trend_regime", "volatility_regime"],
        include_mtf=False,
        supported_model_types=["tree"],
        default_sequence_length=None,  # Not applicable for tabular
        recommended_scaler="none",  # Boosting handles raw features
    ),
    "neural_optimal": FeatureSetDefinition(
        name="neural_optimal",
        description=(
            "Optimal feature set for neural networks (LSTM, GRU, MLP). "
            "Normalized features that work well in sequences. Excludes raw prices "
            "and unbounded features. Focus on returns, oscillators, and ratios."
        ),
        include_prefixes=[
            # Returns (already normalized, excellent for sequences)
            "return_", "log_return_",
            # Bounded oscillators (0-100 or -100 to 100)
            "rsi_", "stoch_", "williams_", "cci_", "mfi_",
            # Normalized volatility ratios
            "hvol_", "atr_ratio", "bb_position", "kc_position",
            # Volume ratios (normalized)
            "volume_ratio", "volume_zscore",
            # Cyclical time features (sin/cos encoded)
            "hour_sin", "hour_cos", "dayofweek_sin", "dayofweek_cos",
        ],
        include_columns=[
            "price_to_vwap", "close_bb_zscore",
            "trend_regime", "volatility_regime",
        ],
        exclude_prefixes=[
            # Exclude raw prices and unbounded features
            "sma_", "ema_", "bb_upper", "bb_lower", "vwap",
            "open_", "high_", "low_", "close_",
        ],
        include_mtf=False,
        supported_model_types=["sequential"],
        default_sequence_length=60,
        recommended_scaler="robust",  # RobustScaler handles outliers well for NNs
    ),
    "transformer_raw": FeatureSetDefinition(
        name="transformer_raw",
        description=(
            "Minimal feature set for transformer/foundation models. "
            "Primarily OHLCV + returns. Transformers learn patterns from raw data "
            "and benefit from minimal preprocessing. Let the model learn features."
        ),
        include_prefixes=[
            # Core returns (let transformer learn patterns)
            "return_", "log_return_",
            # Basic volume information
            "volume_ratio",
        ],
        include_columns=[
            # Minimal temporal context
            "hour_sin", "hour_cos",
            "dayofweek_sin", "dayofweek_cos",
            "is_rth",
        ],
        include_mtf=False,
        supported_model_types=["sequential", "transformer"],
        default_sequence_length=128,  # Longer sequences for transformers
        recommended_scaler="standard",  # Standard scaling for transformers
    ),
    "ensemble_base": FeatureSetDefinition(
        name="ensemble_base",
        description=(
            "Diverse feature set for ensemble meta-learners. "
            "Includes features from different categories to ensure base models "
            "have uncorrelated inputs. Used when training multiple diverse models "
            "for stacking or blending."
        ),
        include_prefixes=[
            # Group 1: Price momentum
            "return_", "roc_",
            # Group 2: Mean reversion signals
            "rsi_", "bb_position", "kc_position",
            # Group 3: Trend following
            "adx_", "macd_", "supertrend",
            # Group 4: Volatility regime
            "atr_", "hvol_",
            # Group 5: Volume analysis
            "volume_", "obv",
            # Group 6: Temporal patterns
            "hour_", "dayofweek_",
        ],
        include_columns=[
            "trend_regime", "volatility_regime",
            "is_rth",
        ],
        include_mtf=True,  # MTF adds diversity
        supported_model_types=["tabular", "tree", "sequential"],
        default_sequence_length=60,
        recommended_scaler="robust",
    ),
    "tcn_optimal": FeatureSetDefinition(
        name="tcn_optimal",
        description=(
            "Optimal feature set for Temporal Convolutional Networks (TCN). "
            "Longer sequences than LSTM/GRU to leverage dilated convolutions. "
            "Focus on features that capture local patterns well."
        ),
        include_prefixes=[
            # Returns and momentum (core for local pattern recognition)
            "return_", "log_return_", "roc_",
            # Bounded oscillators (normalized, good for conv layers)
            "rsi_", "stoch_", "williams_", "cci_", "mfi_",
            # Volatility features (important for risk-adjusted learning)
            "hvol_", "atr_pct", "bb_position", "kc_position",
            # Higher moments (skew/kurtosis capture distribution changes)
            "return_skew_", "return_kurt_",
            # Autocorrelation (captures serial dependence patterns)
            "return_autocorr_",
            # Volume patterns (normalized)
            "volume_ratio", "volume_zscore",
            # Cyclical time features
            "hour_sin", "hour_cos", "dayofweek_sin", "dayofweek_cos",
        ],
        include_columns=[
            "price_to_vwap", "close_bb_zscore", "clv",
            "trend_regime", "volatility_regime",
        ],
        exclude_prefixes=[
            # Exclude raw prices
            "sma_", "ema_", "bb_upper", "bb_lower", "vwap",
            "open_", "high_", "low_", "close_",
        ],
        include_mtf=False,
        supported_model_types=["sequential"],
        default_sequence_length=120,  # Longer sequences for TCN
        recommended_scaler="robust",
    ),
    "patchtst_optimal": FeatureSetDefinition(
        name="patchtst_optimal",
        description=(
            "Optimal feature set for PatchTST transformer model. "
            "Minimal preprocessing - let the transformer learn patterns. "
            "Longer sequences with patch-based attention."
        ),
        include_prefixes=[
            # Minimal features - transformers learn patterns from raw data
            "return_", "log_return_",
            "volume_ratio",
        ],
        include_columns=[
            # Temporal context
            "hour_sin", "hour_cos",
            "dayofweek_sin", "dayofweek_cos",
            "is_rth",
        ],
        include_mtf=False,
        supported_model_types=["transformer"],
        default_sequence_length=256,  # Long sequences for patch attention
        recommended_scaler="standard",
    ),
    "volatility_focus": FeatureSetDefinition(
        name="volatility_focus",
        description=(
            "Volatility-focused feature set for vol prediction tasks. "
            "Includes all volatility estimators and related features."
        ),
        include_prefixes=[
            # All volatility estimators
            "hvol_", "atr_", "parkinson_", "gk_vol", "rs_vol", "yz_vol",
            "bb_width", "kc_",
            # Higher moments (related to vol clustering)
            "return_skew_", "return_kurt_",
            # Returns for realized vol context
            "return_", "log_return_",
            # Volume (often correlated with volatility)
            "volume_", "dollar_volume",
        ],
        include_columns=[
            "volatility_regime",
            "range_pct",
        ],
        include_mtf=True,  # MTF volatility useful
        supported_model_types=["tabular", "tree", "sequential"],
        default_sequence_length=60,
        recommended_scaler="robust",
    ),
}


FEATURE_SET_ALIASES = {
    # Original aliases
    "minimal": "core_min",
    "min": "core_min",
    "full": "core_full",
    "mtf": "mtf_plus",
    # Model-family aliases
    "boosting": "boosting_optimal",
    "xgboost": "boosting_optimal",
    "lightgbm": "boosting_optimal",
    "catboost": "boosting_optimal",
    "neural": "neural_optimal",
    "lstm": "neural_optimal",
    "gru": "neural_optimal",
    "mlp": "neural_optimal",
    "transformer": "transformer_raw",
    "foundation": "transformer_raw",
    "ensemble": "ensemble_base",
    "stacking": "ensemble_base",
    "blending": "ensemble_base",
    # New model-specific aliases
    "tcn": "tcn_optimal",
    "temporal_conv": "tcn_optimal",
    "patchtst": "patchtst_optimal",
    "patch_transformer": "patchtst_optimal",
    "informer": "transformer_raw",
    "volatility": "volatility_focus",
    "vol": "volatility_focus",
}


def get_feature_set_definitions() -> dict[str, FeatureSetDefinition]:
    """Return a copy of feature set definitions."""
    return FEATURE_SET_DEFINITIONS.copy()


def resolve_feature_set_name(name: str) -> str:
    """Resolve a feature set name or alias to a canonical name."""
    if not name:
        raise ValueError("feature_set must be a non-empty string")
    normalized = name.strip().lower()
    canonical = FEATURE_SET_ALIASES.get(normalized, normalized)
    if canonical not in FEATURE_SET_DEFINITIONS and canonical != "all":
        valid = sorted(list(FEATURE_SET_DEFINITIONS.keys()) + ["all"])
        raise ValueError(f"Unknown feature_set '{name}'. Valid options: {valid}")
    return canonical


def resolve_feature_set_names(name: str) -> list[str]:
    """
    Resolve a feature set selection into a list of canonical names.

    Supports 'all' or comma-separated values.
    """
    if not name:
        raise ValueError("feature_set must be a non-empty string")
    selections = [part.strip() for part in name.split(",") if part.strip()]
    resolved: list[str] = []
    for selection in selections:
        canonical = resolve_feature_set_name(selection)
        if canonical == "all":
            return sorted(FEATURE_SET_DEFINITIONS.keys())
        resolved.append(canonical)
    # De-duplicate while preserving order
    unique: list[str] = []
    for item in resolved:
        if item not in unique:
            unique.append(item)
    return unique


def validate_feature_set_config(feature_set: str) -> list[str]:
    """Validate feature set selection."""
    errors: list[str] = []
    try:
        resolve_feature_set_names(feature_set)
    except ValueError as exc:
        errors.append(str(exc))
    return errors
