"""
Feature set resolution utilities.
"""
from typing import Dict, Iterable, List, Sequence

import pandas as pd

from src.config.feature_sets import FeatureSetDefinition
from src.config.features import CROSS_ASSET_FEATURES
from src.stages.mtf.constants import MTF_TIMEFRAMES

METADATA_COLUMNS = {
    "datetime", "symbol", "open", "high", "low", "close", "volume",
    "timestamp", "date", "time", "timeframe",
    "session_id", "missing_bar", "roll_event", "roll_window", "filled",
}

LABEL_PREFIXES = (
    "label_", "bars_to_hit_", "mae_", "mfe_", "quality_", "sample_weight_",
    "touch_type_", "pain_to_gain_", "time_weighted_dd_", "fwd_return_",
    "fwd_return_log_", "time_to_hit_",
)


def _mtf_suffixes() -> List[str]:
    suffixes = set()
    for tf in MTF_TIMEFRAMES.keys():
        if tf.endswith("min"):
            minutes = tf.replace("min", "")
            suffixes.add(f"_{minutes}m")
        if tf in ("60min", "1h"):
            suffixes.add("_1h")
    return sorted(suffixes)


def _is_label_column(name: str) -> bool:
    return any(name.startswith(prefix) for prefix in LABEL_PREFIXES)


def _is_cross_asset_column(name: str) -> bool:
    features = CROSS_ASSET_FEATURES.get("features", {})
    return name in features


def _is_mtf_column(name: str) -> bool:
    return any(name.endswith(suffix) for suffix in _mtf_suffixes())


def _base_feature_columns(df: pd.DataFrame) -> List[str]:
    return [
        col for col in df.columns
        if col not in METADATA_COLUMNS
        and not _is_label_column(col)
    ]


def resolve_feature_set(
    df: pd.DataFrame,
    definition: FeatureSetDefinition
) -> List[str]:
    """
    Resolve a feature set definition into a concrete column list.
    """
    candidates = _base_feature_columns(df)

    if not definition.include_mtf:
        candidates = [c for c in candidates if not _is_mtf_column(c)]

    if not definition.include_cross_asset:
        candidates = [c for c in candidates if not _is_cross_asset_column(c)]

    if definition.include_prefixes:
        prefix_matches = [
            c for c in candidates
            if any(c.startswith(prefix) for prefix in definition.include_prefixes)
        ]
        explicit = [c for c in definition.include_columns if c in candidates]
        allowed = set(prefix_matches + explicit)
        candidates = [c for c in candidates if c in allowed]

    if definition.exclude_prefixes:
        candidates = [
            c for c in candidates
            if not any(c.startswith(prefix) for prefix in definition.exclude_prefixes)
        ]

    if definition.exclude_columns:
        excluded = set(definition.exclude_columns)
        candidates = [c for c in candidates if c not in excluded]

    if definition.include_columns:
        for col in definition.include_columns:
            if col in df.columns and col not in candidates:
                candidates.append(col)

    return candidates


def build_feature_set_manifest(
    df: pd.DataFrame,
    definitions: Dict[str, FeatureSetDefinition]
) -> Dict[str, Dict]:
    """
    Build a manifest of all feature sets against a reference DataFrame.
    """
    manifest: Dict[str, Dict] = {}
    for name, definition in definitions.items():
        features = resolve_feature_set(df, definition)
        mtf_count = sum(1 for col in features if _is_mtf_column(col))
        cross_asset_count = sum(1 for col in features if _is_cross_asset_column(col))
        manifest[name] = {
            "description": definition.description,
            "feature_count": len(features),
            "features": features,
            "include_mtf": definition.include_mtf,
            "include_cross_asset": definition.include_cross_asset,
            "mtf_feature_count": mtf_count,
            "cross_asset_feature_count": cross_asset_count,
        }
    return manifest


def validate_feature_set_columns(
    df: pd.DataFrame,
    columns: Sequence[str],
    feature_set_name: str
) -> None:
    """
    Validate that all requested feature columns exist.
    """
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(
            f"Feature set '{feature_set_name}' missing columns: {missing[:10]}"
        )
