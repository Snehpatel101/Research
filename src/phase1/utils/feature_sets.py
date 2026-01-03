"""
Feature set resolution utilities.
"""

from collections.abc import Sequence

import pandas as pd

from src.phase1.config.feature_sets import FeatureSetDefinition
from src.phase1.utils.constants import LABEL_PREFIXES, METADATA_COLUMNS

# Import MTF_TIMEFRAMES lazily to avoid circular imports
_MTF_TIMEFRAMES = None


def _get_mtf_timeframes():
    """Lazy loader for MTF_TIMEFRAMES to avoid circular imports."""
    global _MTF_TIMEFRAMES
    if _MTF_TIMEFRAMES is None:
        from src.phase1.stages.mtf.constants import MTF_TIMEFRAMES
        _MTF_TIMEFRAMES = MTF_TIMEFRAMES
    return _MTF_TIMEFRAMES


def _mtf_suffixes() -> list[str]:
    suffixes = set()
    for tf in _get_mtf_timeframes().keys():
        if tf.endswith("min"):
            minutes = tf.replace("min", "")
            suffixes.add(f"_{minutes}m")
        if tf in ("60min", "1h"):
            suffixes.add("_1h")
    return sorted(suffixes)


def _is_label_column(name: str) -> bool:
    return any(name.startswith(prefix) for prefix in LABEL_PREFIXES)


def _is_mtf_column(name: str) -> bool:
    return any(name.endswith(suffix) for suffix in _mtf_suffixes())


def _base_feature_columns(df: pd.DataFrame) -> list[str]:
    return [col for col in df.columns if col not in METADATA_COLUMNS and not _is_label_column(col)]


def resolve_feature_set(df: pd.DataFrame, definition: FeatureSetDefinition) -> list[str]:
    """
    Resolve a feature set definition into a concrete column list.
    """
    candidates = _base_feature_columns(df)

    if not definition.include_mtf:
        candidates = [c for c in candidates if not _is_mtf_column(c)]

    # Note: Cross-asset features removed - each symbol processed independently

    if definition.include_prefixes:
        prefix_matches = [
            c
            for c in candidates
            if any(c.startswith(prefix) for prefix in definition.include_prefixes)
        ]
        explicit = [c for c in definition.include_columns if c in candidates]
        allowed = set(prefix_matches + explicit)
        candidates = [c for c in candidates if c in allowed]

    if definition.exclude_prefixes:
        candidates = [
            c
            for c in candidates
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
    df: pd.DataFrame, definitions: dict[str, FeatureSetDefinition]
) -> dict[str, dict]:
    """
    Build a manifest of all feature sets against a reference DataFrame.
    """
    manifest: dict[str, dict] = {}
    for name, definition in definitions.items():
        features = resolve_feature_set(df, definition)
        mtf_count = sum(1 for col in features if _is_mtf_column(col))
        manifest[name] = {
            "description": definition.description,
            "feature_count": len(features),
            "features": features,
            "include_mtf": definition.include_mtf,
            "mtf_feature_count": mtf_count,
        }
    return manifest


def validate_feature_set_columns(
    df: pd.DataFrame, columns: Sequence[str], feature_set_name: str
) -> None:
    """
    Validate that all requested feature columns exist.
    """
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(f"Feature set '{feature_set_name}' missing columns: {missing[:10]}")
