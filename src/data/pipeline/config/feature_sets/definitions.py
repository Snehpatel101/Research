"""
Feature set definitions for modular model training.

Contains FEATURE_SET_DEFINITIONS dict with all named feature sets.
"""

from .core import FeatureSetDefinition

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
            "return_",
            "log_return_",
            "roc_",
            "rsi_",
            "macd_",
            "stoch_",
            "williams_",
            "cci_",
            "mfi_",
            "atr_",
            "bb_",
            "kc_",
            "hvol_",
            "parkinson_",
            "garman_",
            "volume_",
            "vwap",
            "obv",
            "adx_",
            "supertrend",
            "range_",
            "hl_",
            "co_",
            "hour_",
            "minute_",
            "dayofweek_",
            "session_",
            "is_rth",
            "trend_regime",
            "volatility_regime",
        ],
        include_columns=["price_to_vwap"],
        include_mtf=False,
        supported_model_types=["boosting", "classical", "neural"],
        default_sequence_length=60,
        recommended_scaler="robust",
    ),
    "core_full": FeatureSetDefinition(
        name="core_full",
        description="All base-timeframe features (no MTF). Single symbol only.",
        include_prefixes=[],
        include_mtf=False,
        supported_model_types=["boosting", "classical", "neural"],
        default_sequence_length=60,
        recommended_scaler="robust",
    ),
    "mtf_plus": FeatureSetDefinition(
        name="mtf_plus",
        description="All base-timeframe features plus MTF features. Single symbol only.",
        include_prefixes=[],
        include_mtf=True,
        supported_model_types=["boosting", "classical", "neural"],
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
            "return_",
            "log_return_",
            "roc_",
            # Momentum oscillators
            "rsi_",
            "macd_",
            "stoch_",
            "williams_",
            "cci_",
            "mfi_",
            # Trend indicators
            "adx_",
            "supertrend",
            # Volatility (critical for risk-adjusted predictions)
            "atr_",
            "hvol_",
            "parkinson_",
            "garman_",
            "bb_width",
            # Volume features
            "volume_",
            "obv",
            # Price position features
            "bb_position",
            "kc_position",
            "price_to_",
            # Temporal features (important for regime detection)
            "hour_",
            "dayofweek_",
            "session_",
        ],
        include_columns=["is_rth", "trend_regime", "volatility_regime"],
        include_mtf=False,
        supported_model_types=["boosting"],
        default_sequence_length=None,  # Not applicable for tabular
        recommended_scaler="none",  # Boosting handles raw features
    ),
    # =========================================================================
    # NEURAL vs TRANSFORMER FEATURE SET GUIDANCE
    # =========================================================================
    # Choose the right feature set based on your model's architecture:
    #
    # USE neural_optimal WHEN:
    # - Training sequence models that benefit from pre-computed indicators
    #   (LSTM, GRU, TCN, InceptionTime, ResNet1D, TFT)
    # - You want the model to learn from already-meaningful signals
    # - You have limited training data (pre-computed features reduce what
    #   the model needs to learn from scratch)
    # - You want faster convergence and more stable training
    # - Model architecture doesn't have built-in feature learning
    #
    # USE transformer_raw WHEN:
    # - Training foundation/attention models designed to learn patterns
    #   from raw data (Vanilla Transformer, PatchTST, iTransformer)
    # - You have abundant training data (100K+ samples)
    # - You want the model to discover its own representations
    # - The model architecture includes feature extraction layers
    # - You're experimenting with minimal preprocessing approaches
    #
    # KEY DIFFERENCE:
    # - neural_optimal: ~40-60 pre-computed features (oscillators, ratios)
    # - transformer_raw: ~10-15 minimal features (returns, volume, time)
    #
    # See model_config.py for per-model feature_set recommendations.
    # =========================================================================
    "neural_optimal": FeatureSetDefinition(
        name="neural_optimal",
        description=(
            "Optimal feature set for neural sequence models (LSTM, GRU, TCN, "
            "InceptionTime, ResNet1D, TFT). Includes ~40-60 pre-computed indicators "
            "that are already meaningful signals: bounded oscillators (RSI, Stochastic), "
            "normalized ratios (price-to-VWAP, BB position), and cyclical time features. "
            "USE THIS when you want the model to learn from pre-engineered features "
            "rather than discovering patterns from raw data. Recommended for models "
            "without built-in feature extraction capabilities."
        ),
        include_prefixes=[
            # Returns (already normalized, excellent for sequences)
            "return_",
            "log_return_",
            # Bounded oscillators (0-100 or -100 to 100)
            "rsi_",
            "stoch_",
            "williams_",
            "cci_",
            "mfi_",
            # Normalized volatility ratios
            "hvol_",
            "atr_ratio",
            "bb_position",
            "kc_position",
            # Volume ratios (normalized)
            "volume_ratio",
            "volume_zscore",
            # Cyclical time features (sin/cos encoded)
            "hour_sin",
            "hour_cos",
            "dayofweek_sin",
            "dayofweek_cos",
        ],
        include_columns=[
            "price_to_vwap",
            "close_bb_zscore",
            "trend_regime",
            "volatility_regime",
        ],
        exclude_prefixes=[
            # Exclude raw prices and unbounded features
            "sma_",
            "ema_",
            "bb_upper",
            "bb_lower",
            "vwap",
            "open_",
            "high_",
            "low_",
            "close_",
        ],
        include_mtf=False,
        supported_model_types=["neural"],
        default_sequence_length=60,
        recommended_scaler="robust",  # RobustScaler handles outliers well for NNs
    ),
    "transformer_raw": FeatureSetDefinition(
        name="transformer_raw",
        description=(
            "Minimal feature set for transformer/foundation models (Vanilla Transformer, "
            "PatchTST, iTransformer). Includes only ~10-15 minimal features: returns, "
            "volume ratio, and cyclical time encoding. USE THIS for attention-based models "
            "that are designed to learn patterns from raw sequential data. Transformers "
            "have powerful representation learning and benefit from discovering their own "
            "features rather than relying on pre-computed indicators. Requires more "
            "training data (100K+ samples recommended) for effective feature learning."
        ),
        include_prefixes=[
            # Core returns (let transformer learn patterns)
            "return_",
            "log_return_",
            # Basic volume information
            "volume_ratio",
        ],
        include_columns=[
            # Minimal temporal context
            "hour_sin",
            "hour_cos",
            "dayofweek_sin",
            "dayofweek_cos",
            "is_rth",
        ],
        include_mtf=False,
        supported_model_types=["neural"],
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
            "return_",
            "roc_",
            # Group 2: Mean reversion signals
            "rsi_",
            "bb_position",
            "kc_position",
            # Group 3: Trend following
            "adx_",
            "macd_",
            "supertrend",
            # Group 4: Volatility regime
            "atr_",
            "hvol_",
            # Group 5: Volume analysis
            "volume_",
            "obv",
            # Group 6: Temporal patterns
            "hour_",
            "dayofweek_",
        ],
        include_columns=[
            "trend_regime",
            "volatility_regime",
            "is_rth",
        ],
        include_mtf=True,  # MTF adds diversity
        supported_model_types=["boosting", "classical", "neural", "ensemble"],
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
            "return_",
            "log_return_",
            "roc_",
            # Bounded oscillators (normalized, good for conv layers)
            "rsi_",
            "stoch_",
            "williams_",
            "cci_",
            "mfi_",
            # Volatility features (important for risk-adjusted learning)
            "hvol_",
            "atr_pct",
            "bb_position",
            "kc_position",
            # Higher moments (skew/kurtosis capture distribution changes)
            "return_skew_",
            "return_kurt_",
            # Autocorrelation (captures serial dependence patterns)
            "return_autocorr_",
            # Volume patterns (normalized)
            "volume_ratio",
            "volume_zscore",
            # Cyclical time features
            "hour_sin",
            "hour_cos",
            "dayofweek_sin",
            "dayofweek_cos",
        ],
        include_columns=[
            "price_to_vwap",
            "close_bb_zscore",
            "clv",
            "trend_regime",
            "volatility_regime",
        ],
        exclude_prefixes=[
            # Exclude raw prices
            "sma_",
            "ema_",
            "bb_upper",
            "bb_lower",
            "vwap",
            "open_",
            "high_",
            "low_",
            "close_",
        ],
        include_mtf=False,
        supported_model_types=["neural"],
        default_sequence_length=64,  # Matches TCN receptive field (61)
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
            "return_",
            "log_return_",
            "volume_ratio",
        ],
        include_columns=[
            # Temporal context
            "hour_sin",
            "hour_cos",
            "dayofweek_sin",
            "dayofweek_cos",
            "is_rth",
        ],
        include_mtf=False,
        supported_model_types=["neural"],
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
            "hvol_",
            "atr_",
            "parkinson_",
            "gk_vol",
            "rs_vol",
            "yz_vol",
            "bb_width",
            "kc_",
            # Higher moments (related to vol clustering)
            "return_skew_",
            "return_kurt_",
            # Returns for realized vol context
            "return_",
            "log_return_",
            # Volume (often correlated with volatility)
            "volume_",
            "dollar_volume",
        ],
        include_columns=[
            "volatility_regime",
            "range_pct",
        ],
        include_mtf=True,  # MTF volatility useful
        supported_model_types=["boosting", "classical", "neural"],
        default_sequence_length=60,
        recommended_scaler="robust",
    ),
    # =========================================================================
    # ARCHITECTURE-SPECIFIC FEATURE SETS (Research-based)
    # =========================================================================
    # These feature sets are derived from statistical analysis of:
    # - Feature correlation structure
    # - Model architecture constraints
    # - Bias-variance tradeoff per architecture
    # See: docs/FEATURE_SELECTION_BY_ARCHITECTURE.md
    # =========================================================================
    "nbeats_optimal": FeatureSetDefinition(
        name="nbeats_optimal",
        description=(
            "Minimal feature set for N-BEATS. N-BEATS performs automatic "
            "trend-seasonality decomposition and works best with univariate "
            "or near-univariate input. Adding engineered features typically "
            "degrades performance by conflicting with internal decomposition."
        ),
        include_prefixes=[
            # Only returns for stationary version
            "log_return_",
        ],
        include_columns=[
            # Optional volume signal (can be omitted for pure univariate)
            "volume_ratio_20",
        ],
        include_mtf=False,
        supported_model_types=["neural"],
        default_sequence_length=128,  # N-BEATS uses longer lookback
        recommended_scaler="standard",
    ),
    "inceptiontime_optimal": FeatureSetDefinition(
        name="inceptiontime_optimal",
        description=(
            "Optimal feature set for InceptionTime (55 features). "
            "InceptionTime uses multiple filter sizes in parallel, benefiting "
            "from multi-scale features like wavelets. Includes diverse feature "
            "types for inception modules to learn different patterns."
        ),
        include_prefixes=[
            # Returns and momentum
            "log_return_",
            "roc_",
            "rsi_",
            "stoch_",
            "williams_",
            "cci_",
            "mfi_",
            "macd_",
            # Volatility
            "atr_",
            "hvol_",
            "bb_position",
            "bb_width",
            "close_bb_zscore",
            "kc_position",
            "close_kc_atr_dev",
            "return_skew_",
            "return_kurt_",
            # Wavelets (multi-scale critical for inception modules)
            "wavelet_close_",
            "wavelet_volume_",
            # Volume
            "volume_ratio",
            "volume_zscore",
            "obv_zscore",
            # Microstructure
            "micro_amihud_",
            "micro_roll_spread",
            "micro_efficiency_",
            "micro_rel_spread_",
            "micro_cs_spread",
            "micro_volume_imbalance",
            "micro_trade_intensity_",
            # Temporal
            "hour_",
            "dayofweek_",
        ],
        include_columns=[
            "clv",
            "price_to_vwap",
            "yz_vol",
            "trend_regime",
            "volatility_regime",
            "micro_vol_ratio",
        ],
        include_mtf=False,
        supported_model_types=["neural"],
        default_sequence_length=100,  # InceptionTime works well with medium sequences
        recommended_scaler="robust",
    ),
    "resnet1d_optimal": FeatureSetDefinition(
        name="resnet1d_optimal",
        description=(
            "Optimal feature set for ResNet1D (48 features). "
            "Residual connections benefit from multi-scale wavelet features. "
            "Skip connections help with gradient flow, allowing moderate feature count."
        ),
        include_prefixes=[
            # Returns
            "log_return_",
            # Momentum
            "rsi_",
            "stoch_",
            "cci_",
            "mfi_",
            "macd_hist",  # Only histogram (normalized)
            "roc_",
            # Volatility
            "atr_pct_",
            "hvol_",
            "bb_position",
            "close_bb_zscore",
            "kc_position",
            "close_kc_atr_dev",
            "return_skew_",
            "return_kurt_",
            # Wavelets (multi-scale for residual learning)
            "wavelet_close_approx",
            "wavelet_close_d",
            "wavelet_close_energy_ratio",
            "wavelet_close_volatility",
            "wavelet_close_trend_",
            # Volume
            "volume_ratio",
            "volume_zscore",
            "obv_zscore",
            # Microstructure
            "micro_efficiency_",
            "micro_rel_spread_",
            "micro_trade_intensity_",
            "micro_amihud_",
            "micro_roll_spread_pct",
            "micro_cs_spread",
            # Temporal
            "hour_",
            "dayofweek_",
        ],
        include_columns=[
            "yz_vol",
            "price_to_vwap",
            "micro_volume_imbalance",
            "micro_vol_ratio",
            "trend_regime",
            "volatility_regime",
        ],
        include_mtf=False,
        supported_model_types=["neural"],
        default_sequence_length=80,
        recommended_scaler="robust",
    ),
    "itransformer_optimal": FeatureSetDefinition(
        name="itransformer_optimal",
        description=(
            "Optimal feature set for iTransformer (25-40 features). "
            "iTransformer applies attention over FEATURES (not time), learning "
            "cross-feature correlations. Attention complexity is O(n_features^2), "
            "so moderate feature count is optimal. Features organized in groups "
            "(momentum, volatility, volume) for meaningful cross-correlation learning."
        ),
        include_prefixes=[
            # Momentum group (for cross-correlation learning)
            "return_",
            "log_return_",
            "roc_",
            "rsi_",
            # Mean reversion group
            "stoch_",
            "williams_",
            "bb_position",
            "kc_position",
            # Trend group
            "macd_signal",
            "macd_hist",
            "adx_",
            # Volatility group (correlates with momentum)
            "hvol_",
            "atr_pct",
            # Volume group (correlates with volatility)
            "volume_ratio",
            "volume_zscore",
        ],
        include_columns=[
            # Core ratio features (good for attention)
            "price_to_vwap",
            "close_bb_zscore",
            # Regime indicators (categorical-like)
            "trend_regime",
            "volatility_regime",
            # Temporal context (for cross-correlation with patterns)
            "hour_sin",
            "hour_cos",
            "dayofweek_sin",
            "dayofweek_cos",
            "is_rth",
        ],
        exclude_prefixes=[
            # Exclude raw prices and redundant features
            "sma_",
            "ema_",
            "bb_upper",
            "bb_lower",
            "bb_middle",
            "vwap",
            "open_",
            "high_",
            "low_",
            "close_",
            # Exclude noisy features
            "return_skew_",
            "return_kurt_",
            "return_autocorr_",
        ],
        include_mtf=False,
        supported_model_types=["neural"],
        default_sequence_length=60,
        recommended_scaler="standard",  # Standard for transformer stability
    ),
    "tft_optimal": FeatureSetDefinition(
        name="tft_optimal",
        description=(
            "Rich feature set for Temporal Fusion Transformer (60-80 features). "
            "TFT has Variable Selection Networks (VSN) that learn feature importance "
            "automatically, so it can handle many features. Includes diverse feature "
            "groups - the model will select the most relevant ones. This is the "
            "richest feature set for maximum interpretability through VSN attention."
        ),
        include_prefixes=[
            # Full price action (VSN will select)
            "return_",
            "log_return_",
            "roc_",
            # Full momentum oscillators
            "rsi_",
            "stoch_",
            "williams_",
            "cci_",
            "mfi_",
            # Full trend indicators
            "macd_",
            "adx_",
            "supertrend",
            # Full volatility suite
            "hvol_",
            "atr_",
            "parkinson_",
            "garman_",
            "bb_position",
            "bb_width",
            "kc_position",
            # Volume analysis
            "volume_",
            "obv",
            # Higher moments (TFT can learn when these matter)
            "return_skew_",
            "return_kurt_",
            "return_autocorr_",
            # Wavelets (multi-scale patterns)
            "wavelet_close_",
            # Microstructure
            "micro_amihud_",
            "micro_efficiency_",
            "micro_rel_spread_",
            # Temporal features (TFT handles time well)
            "hour_sin",
            "hour_cos",
            "dayofweek_sin",
            "dayofweek_cos",
            "session_",
        ],
        include_columns=[
            "price_to_vwap",
            "close_bb_zscore",
            "clv",
            "range_pct",
            "trend_regime",
            "volatility_regime",
            "is_rth",
            "yz_vol",
        ],
        exclude_prefixes=[
            # Only exclude raw prices
            "sma_",
            "ema_",
            "bb_upper",
            "bb_lower",
            "bb_middle",
            "vwap",
            "open_",
            "high_",
            "low_",
            "close_",
        ],
        include_mtf=False,  # Can enable for even richer set
        supported_model_types=["neural"],
        default_sequence_length=60,
        recommended_scaler="robust",  # Robust for diverse feature types
    ),
}
