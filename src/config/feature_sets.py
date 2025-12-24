"""
Feature set definitions for modular model training.

Provides named feature sets that can be selected without code edits.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass(frozen=True)
class FeatureSetDefinition:
    """
    Definition of a named feature set.

    Attributes:
        name: Unique identifier for this feature set
        description: Human-readable description
        include_prefixes: Feature name prefixes to include
        exclude_prefixes: Feature name prefixes to exclude
        include_columns: Specific columns to include
        exclude_columns: Specific columns to exclude
        include_mtf: Whether to include multi-timeframe features
        include_cross_asset: Whether to include cross-asset features
        supported_model_types: Model types that work with this feature set
        default_sequence_length: Default sequence length for sequential models
        recommended_scaler: Recommended scaler type for this feature set
    """
    name: str
    description: str
    include_prefixes: List[str] = field(default_factory=list)
    exclude_prefixes: List[str] = field(default_factory=list)
    include_columns: List[str] = field(default_factory=list)
    exclude_columns: List[str] = field(default_factory=list)
    include_mtf: bool = False
    include_cross_asset: bool = False
    supported_model_types: List[str] = field(default_factory=lambda: [
        "tabular", "sequential", "tree"
    ])
    default_sequence_length: Optional[int] = None
    recommended_scaler: str = "robust"


FEATURE_SET_DEFINITIONS: Dict[str, FeatureSetDefinition] = {
    "core_min": FeatureSetDefinition(
        name="core_min",
        description="Minimal base-timeframe feature set (no MTF, no cross-asset).",
        include_prefixes=[
            "return_", "log_return_", "roc_", "rsi_", "macd_", "stoch_", "williams_",
            "cci_", "mfi_", "atr_", "bb_", "kc_", "hvol_", "parkinson_", "garman_",
            "volume_", "vwap", "obv", "adx_", "supertrend", "range_", "hl_", "co_",
            "hour_", "minute_", "dayofweek_", "session_", "is_rth", "trend_regime",
            "volatility_regime",
        ],
        include_columns=["price_to_vwap"],
        include_mtf=False,
        include_cross_asset=False,
        supported_model_types=["tabular", "tree", "sequential"],
        default_sequence_length=60,
        recommended_scaler="robust",
    ),
    "core_full": FeatureSetDefinition(
        name="core_full",
        description="All base-timeframe features (no MTF, no cross-asset).",
        include_prefixes=[],
        include_mtf=False,
        include_cross_asset=False,
        supported_model_types=["tabular", "tree", "sequential"],
        default_sequence_length=60,
        recommended_scaler="robust",
    ),
    "mtf_plus": FeatureSetDefinition(
        name="mtf_plus",
        description="All base-timeframe features plus MTF and cross-asset features.",
        include_prefixes=[],
        include_mtf=True,
        include_cross_asset=True,
        supported_model_types=["tabular", "tree", "sequential"],
        default_sequence_length=120,
        recommended_scaler="robust",
    ),
}


FEATURE_SET_ALIASES = {
    "minimal": "core_min",
    "min": "core_min",
    "full": "core_full",
    "mtf": "mtf_plus",
}


def get_feature_set_definitions() -> Dict[str, FeatureSetDefinition]:
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


def resolve_feature_set_names(name: str) -> List[str]:
    """
    Resolve a feature set selection into a list of canonical names.

    Supports 'all' or comma-separated values.
    """
    if not name:
        raise ValueError("feature_set must be a non-empty string")
    selections = [part.strip() for part in name.split(",") if part.strip()]
    resolved: List[str] = []
    for selection in selections:
        canonical = resolve_feature_set_name(selection)
        if canonical == "all":
            return sorted(FEATURE_SET_DEFINITIONS.keys())
        resolved.append(canonical)
    # De-duplicate while preserving order
    unique: List[str] = []
    for item in resolved:
        if item not in unique:
            unique.append(item)
    return unique


def validate_feature_set_config(feature_set: str) -> List[str]:
    """Validate feature set selection."""
    errors: List[str] = []
    try:
        resolve_feature_set_names(feature_set)
    except ValueError as exc:
        errors.append(str(exc))
    return errors
