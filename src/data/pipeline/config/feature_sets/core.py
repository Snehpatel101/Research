"""
Core types and constants for feature set definitions.

Contains the FeatureSetDefinition dataclass and FEATURE_SET_ALIASES mapping.
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
        include_prefixes: Feature name prefixes to include (matched with startswith)
        exclude_prefixes: Feature name prefixes to exclude (matched with startswith)
        include_columns: Specific columns to include (exact match)
        exclude_columns: Specific columns to exclude (exact match)
        explicit_columns: Exact column names that MUST be present. If specified,
            these columns are required; if any are missing, a warning is logged.
            Use this for critical features that should always be included.
        include_mtf: Whether to include multi-timeframe features
        supported_model_types: Model families that work with this feature set.
            Must match registry family names: "boosting", "classical", "neural", "ensemble"
        default_sequence_length: Default sequence length for neural sequence models
        recommended_scaler: Recommended scaler type for this feature set
    """

    name: str
    description: str
    include_prefixes: list[str] = field(default_factory=list)
    exclude_prefixes: list[str] = field(default_factory=list)
    include_columns: list[str] = field(default_factory=list)
    exclude_columns: list[str] = field(default_factory=list)
    explicit_columns: list[str] = field(default_factory=list)
    include_mtf: bool = False
    supported_model_types: list[str] = field(
        default_factory=lambda: ["boosting", "classical", "neural", "ensemble"]
    )
    default_sequence_length: int | None = None
    recommended_scaler: str = "robust"


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
    # Architecture-specific aliases (added from research)
    "nbeats": "nbeats_optimal",
    "n_beats": "nbeats_optimal",
    "n-beats": "nbeats_optimal",
    "inceptiontime": "inceptiontime_optimal",
    "inception_time": "inceptiontime_optimal",
    "inception": "inceptiontime_optimal",
    "resnet1d": "resnet1d_optimal",
    "resnet": "resnet1d_optimal",
    "resnet_1d": "resnet1d_optimal",
    "itransformer": "itransformer_optimal",
    "i_transformer": "itransformer_optimal",
    "inverted_transformer": "itransformer_optimal",
    "tft": "tft_optimal",
    "temporal_fusion": "tft_optimal",
    "temporal_fusion_transformer": "tft_optimal",
}
