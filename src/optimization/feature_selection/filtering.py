"""
Feature Filtering Functions.

Provides utility functions for feature selection including:
- Low variance filtering
- Correlation-based filtering
- Feature prioritization

This module consolidates filtering code from:
- src/phase1/utils/feature_selection.py
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from .priority import get_feature_priority
from .result import FeatureSelectionResult

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def identify_feature_columns(
    df: pd.DataFrame,
    metadata_columns: list[str] | None = None,
    label_prefixes: tuple[str, ...] | None = None,
) -> list[str]:
    """
    Identify feature columns in the dataframe.

    Excludes metadata, label, and target columns.
    Each symbol is processed independently (no cross-symbol correlation).

    Args:
        df: Input DataFrame
        metadata_columns: Columns to exclude (default: common metadata columns)
        label_prefixes: Prefixes to exclude (default: label_, target_, etc.)

    Returns:
        List of feature column names
    """
    if metadata_columns is None:
        metadata_columns = [
            "datetime",
            "symbol",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "timeframe",
            "session_id",
            "missing_bar",
            "roll_event",
            "roll_window",
            "filled",
        ]

    if label_prefixes is None:
        label_prefixes = (
            "label_",
            "target_",
            "bars_to_hit_",
            "mae_",
            "mfe_",
            "quality_",
            "sample_weight_",
            "touch_type_",
            "pain_to_gain_",
            "time_weighted_dd_",
            "fwd_return_",
            "fwd_return_log_",
            "time_to_hit_",
        )

    feature_cols = [
        c
        for c in df.columns
        if c not in metadata_columns and not any(c.startswith(p) for p in label_prefixes)
    ]

    return feature_cols


def filter_low_variance(
    df: pd.DataFrame, feature_cols: list[str], variance_threshold: float = 0.01
) -> tuple[list[str], list[str]]:
    """
    Remove features with variance below threshold.

    Near-constant features provide no discriminative power and should be removed.

    Args:
        df: Input DataFrame
        feature_cols: List of feature column names
        variance_threshold: Minimum variance to keep feature (default 0.01)

    Returns:
        Tuple of (features_to_keep, low_variance_features)
    """
    low_variance = []
    to_keep = []

    for col in feature_cols:
        series = df[col].replace([np.inf, -np.inf], np.nan).dropna()

        if len(series) < 100:
            # Not enough data points, keep feature
            to_keep.append(col)
            continue

        variance = series.var()

        # Normalize variance by scale for fair comparison
        # Use coefficient of variation for features with non-zero mean
        mean_val = abs(series.mean())
        normalized_variance = variance / mean_val**2 if mean_val > 1e-10 else variance

        # Check if variance is below threshold
        if variance < variance_threshold and normalized_variance < variance_threshold:
            low_variance.append(col)
        else:
            to_keep.append(col)

    return to_keep, low_variance


def build_correlation_groups(
    df: pd.DataFrame, feature_cols: list[str], correlation_threshold: float = 0.85
) -> list[set[str]]:
    """
    Build groups of highly correlated features using union-find.

    Features with correlation above threshold are grouped together.
    We keep only one feature from each group.

    Args:
        df: Input DataFrame
        feature_cols: List of feature column names
        correlation_threshold: Threshold for considering features correlated (default 0.85)

    Returns:
        List of sets, each containing correlated feature names
    """
    if len(feature_cols) == 0:
        return []

    # Calculate correlation matrix
    feature_df = df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    corr_matrix = feature_df.corr()

    # Union-Find data structure for grouping
    parent = {col: col for col in feature_cols}

    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])  # Path compression
        return parent[x]

    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py

    # Find all highly correlated pairs and union them
    for i, col1 in enumerate(feature_cols):
        for j, col2 in enumerate(feature_cols):
            if i >= j:
                continue
            corr_val = abs(corr_matrix.loc[col1, col2])
            if corr_val >= correlation_threshold:
                union(col1, col2)

    # Build groups from union-find structure
    groups_dict: dict[str, set[str]] = {}
    for col in feature_cols:
        root = find(col)
        if root not in groups_dict:
            groups_dict[root] = set()
        groups_dict[root].add(col)

    # Return only groups with more than one member (correlated groups)
    correlated_groups = [group for group in groups_dict.values() if len(group) > 1]

    return correlated_groups


def select_from_correlated_group(group: set[str]) -> tuple[str | None, list[str]]:
    """
    Select the best feature from a correlated group.

    Uses interpretability/priority ranking to select the most useful feature.

    Args:
        group: Set of correlated feature names

    Returns:
        Tuple of (selected_feature, removed_features)
    """
    if len(group) == 0:
        return None, []

    if len(group) == 1:
        return list(group)[0], []

    # Sort by priority (descending) then by name (for consistency)
    sorted_features = sorted(group, key=lambda f: (-get_feature_priority(f), f))

    selected = sorted_features[0]
    removed = sorted_features[1:]

    return selected, removed


def filter_correlated_features(
    df: pd.DataFrame, feature_cols: list[str], correlation_threshold: float = 0.85
) -> tuple[list[str], dict[str, str], list[list[str]]]:
    """
    Remove highly correlated features, keeping the most interpretable from each group.

    Args:
        df: Input DataFrame
        feature_cols: List of feature column names
        correlation_threshold: Threshold for considering features correlated (default 0.85)

    Returns:
        Tuple of (features_to_keep, removed_features_dict, correlation_groups)
        - removed_features_dict maps feature name to reason for removal
    """
    correlation_groups = build_correlation_groups(df, feature_cols, correlation_threshold)

    # Track which features are removed and why
    removed_features: dict[str, str] = {}
    features_to_remove: set[str] = set()

    # Convert groups to lists for JSON serialization
    groups_as_lists = []

    for group in correlation_groups:
        groups_as_lists.append(sorted(group))
        selected, removed = select_from_correlated_group(group)

        if selected is None:
            continue

        for feat in removed:
            features_to_remove.add(feat)
            removed_features[feat] = (
                f"correlated with {selected} (priority: {get_feature_priority(selected)} > {get_feature_priority(feat)})"
            )

    # Keep features not in any removed set
    features_to_keep = [f for f in feature_cols if f not in features_to_remove]

    return features_to_keep, removed_features, groups_as_lists


def select_features(
    df: pd.DataFrame,
    correlation_threshold: float = 0.85,
    variance_threshold: float = 0.01,
    feature_cols: list[str] | None = None,
) -> FeatureSelectionResult:
    """
    Main feature selection function.

    Removes highly correlated and low-variance features while keeping
    the most interpretable features from correlated groups.

    Args:
        df: Input DataFrame with features
        correlation_threshold: Threshold for feature correlation (default 0.85)
        variance_threshold: Minimum variance to keep feature (default 0.01)
        feature_cols: Optional list of feature columns (auto-detected if None)

    Returns:
        FeatureSelectionResult with selected and removed features
    """
    logger.info("=" * 60)
    logger.info("FEATURE SELECTION")
    logger.info("=" * 60)

    # Identify feature columns if not provided
    if feature_cols is None:
        feature_cols = identify_feature_columns(df)

    original_count = len(feature_cols)
    logger.info(f"Starting with {original_count} features")

    all_removed: dict[str, str] = {}

    # Step 1: Remove low variance features
    logger.info(f"\n1. Low variance filter (threshold={variance_threshold})")
    features_after_variance, low_variance_features = filter_low_variance(
        df, feature_cols, variance_threshold
    )

    for feat in low_variance_features:
        all_removed[feat] = "low variance (near-constant)"

    logger.info(f"   Removed {len(low_variance_features)} low-variance features")
    if low_variance_features:
        for feat in low_variance_features[:5]:
            logger.info(f"     - {feat}")
        if len(low_variance_features) > 5:
            logger.info(f"     ... and {len(low_variance_features) - 5} more")

    # Step 2: Remove highly correlated features
    logger.info(f"\n2. Correlation filter (threshold={correlation_threshold})")
    selected_features, corr_removed, correlation_groups = filter_correlated_features(
        df, features_after_variance, correlation_threshold
    )

    all_removed.update(corr_removed)

    logger.info(f"   Found {len(correlation_groups)} correlation groups")
    logger.info(f"   Removed {len(corr_removed)} correlated features")

    # Log correlation groups
    for i, group in enumerate(correlation_groups[:5]):
        kept = (
            [f for f in group if f in selected_features][0]
            if any(f in selected_features for f in group)
            else "none"
        )
        logger.info(f"   Group {i+1}: {group}")
        logger.info(f"           Kept: {kept}")
    if len(correlation_groups) > 5:
        logger.info(f"   ... and {len(correlation_groups) - 5} more groups")

    # Build result
    result = FeatureSelectionResult(
        selected_features=sorted(selected_features),
        removed_features=all_removed,
        original_count=original_count,
        final_count=len(selected_features),
        correlation_groups=correlation_groups,
        low_variance_features=low_variance_features,
    )

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("FEATURE SELECTION SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Original features:  {result.original_count}")
    logger.info(f"Features removed:   {len(result.removed_features)}")
    logger.info(f"  - Low variance:   {len(low_variance_features)}")
    logger.info(f"  - Correlated:     {len(corr_removed)}")
    logger.info(f"Final features:     {result.final_count}")
    logger.info(f"Reduction:          {result.to_dict()['reduction_pct']}%")

    logger.info(f"\nSelected features ({result.final_count}):")
    for feat in result.selected_features:
        logger.info(f"  - {feat}")

    return result


def save_feature_selection_report(
    result: FeatureSelectionResult, output_path: Path, include_phase2_recommendations: bool = True
) -> None:
    """
    Save feature selection report to JSON file.

    Args:
        result: FeatureSelectionResult object
        output_path: Path to save the JSON report
        include_phase2_recommendations: Include recommendations for Phase 2
    """
    report = result.to_dict()

    if include_phase2_recommendations:
        report["phase2_recommendations"] = {
            "use_features": result.selected_features,
            "feature_count": result.final_count,
            "notes": [
                "These features have been filtered for low correlation (<0.85)",
                "Low variance features have been removed",
                "Features are ranked by interpretability/importance",
                "Use this feature list for model training in Phase 2",
            ],
        }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)

    logger.info(f"\nFeature selection report saved to: {output_path}")


def apply_feature_selection(
    df: pd.DataFrame, selected_features: list[str], keep_metadata: bool = True
) -> pd.DataFrame:
    """
    Apply feature selection to a DataFrame.

    Args:
        df: Input DataFrame
        selected_features: List of feature columns to keep
        keep_metadata: If True, also keep datetime, symbol, and target columns

    Returns:
        DataFrame with only selected features (and metadata if requested)
    """
    if keep_metadata:
        # Keep metadata and target columns
        metadata_cols = [
            "datetime",
            "symbol",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "timeframe",
            "session_id",
            "missing_bar",
            "roll_event",
            "roll_window",
            "filled",
        ]
        target_prefixes = (
            "label_",
            "bars_to_hit_",
            "mae_",
            "mfe_",
            "quality_",
            "sample_weight_",
            "touch_type_",
            "pain_to_gain_",
            "time_weighted_dd_",
            "fwd_return_",
            "fwd_return_log_",
            "time_to_hit_",
        )

        target_cols = [c for c in df.columns if any(c.startswith(p) for p in target_prefixes)]

        cols_to_keep = (
            [c for c in metadata_cols if c in df.columns]
            + [c for c in selected_features if c in df.columns]
            + target_cols
        )
    else:
        cols_to_keep = [c for c in selected_features if c in df.columns]

    return df[cols_to_keep]


__all__ = [
    "apply_feature_selection",
    "build_correlation_groups",
    "filter_correlated_features",
    "filter_low_variance",
    "identify_feature_columns",
    "save_feature_selection_report",
    "select_features",
    "select_from_correlated_group",
]
