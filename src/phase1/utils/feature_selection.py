"""
Feature Selection Module for Ensemble Price Prediction Pipeline

This module provides feature selection utilities to address:
- Highly correlated features (multicollinearity)
- Low variance features (near-constant)
- Redundant feature removal while preserving interpretability

Author: ML Pipeline
Created: 2025-12-19
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, field
from pathlib import Path
import json

from src.phase1.config.features import get_cross_asset_feature_names, CROSS_ASSET_FEATURES

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


@dataclass
class FeatureSelectionResult:
    """Container for feature selection results."""
    selected_features: List[str]
    removed_features: Dict[str, str]  # feature -> reason
    original_count: int
    final_count: int
    correlation_groups: List[List[str]]
    low_variance_features: List[str]

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'selected_features': self.selected_features,
            'removed_features': self.removed_features,
            'original_count': self.original_count,
            'final_count': self.final_count,
            'reduction_pct': round((1 - self.final_count / self.original_count) * 100, 1) if self.original_count > 0 else 0,
            'correlation_groups': self.correlation_groups,
            'low_variance_features': self.low_variance_features
        }


# Feature interpretability ranking - higher is more interpretable/fundamental
# This guides which feature to keep from correlated pairs
FEATURE_PRIORITY = {
    # Price-based returns (most fundamental)
    'log_return': 100,
    'simple_return': 95,
    'high_low_range': 90,
    'close_open_range': 85,

    # RSI and momentum (classic indicators)
    'rsi': 90,
    'rsi_oversold': 85,
    'rsi_overbought': 85,
    'stoch_k': 80,
    'stoch_d': 75,
    'williams_r': 70,  # Essentially same as stoch_k

    # Moving averages - prefer simpler/shorter
    'sma_10': 80,
    'sma_20': 78,
    'sma_50': 75,
    'sma_100': 72,
    'sma_200': 70,
    'ema_9': 75,
    'ema_21': 73,
    'ema_50': 70,

    # Price relative to moving averages (more useful than raw MA values)
    'close_to_sma_10': 85,
    'close_to_sma_20': 83,
    'close_to_sma_50': 80,
    'close_to_sma_100': 78,
    'close_to_sma_200': 76,
    'close_to_ema_9': 82,
    'close_to_ema_21': 80,
    'close_to_ema_50': 78,

    # MACD components
    'macd': 85,
    'macd_signal': 80,
    'macd_hist': 90,  # Most useful - the difference
    'macd_crossover': 75,

    # Bollinger Bands - prefer derived metrics
    'bb_position': 90,  # Most useful - normalized position
    'bb_width': 85,
    'bb_upper': 60,  # Raw values less useful
    'bb_lower': 60,

    # ATR - prefer percentage versions
    'atr_7_pct': 85,
    'atr_14_pct': 83,
    'atr_21_pct': 80,
    'atr_7': 70,
    'atr_14': 68,
    'atr_21': 65,

    # ADX and directional indicators
    'adx': 85,
    'plus_di': 75,
    'minus_di': 75,

    # Volume indicators
    'volume_ratio': 85,
    'volume_zscore': 80,
    'volume_sma_20': 70,
    'obv': 65,
    'obv_sma_20': 60,

    # VWAP
    'close_to_vwap': 85,
    'vwap': 60,  # Raw VWAP less useful

    # Rate of change - prefer shorter periods
    'roc_5': 80,
    'roc_10': 78,
    'roc_20': 75,

    # Time features
    'hour_sin': 70,
    'hour_cos': 70,
    'dow_sin': 70,
    'dow_cos': 70,
    'is_rth': 80,

    # Regime features
    'vol_regime': 85,
    'trend_regime': 85,
}

# Default priority for unknown features
DEFAULT_PRIORITY = 50


def get_feature_priority(feature_name: str) -> int:
    """
    Get the priority/interpretability score for a feature.

    Higher scores indicate more interpretable/fundamental features that
    should be preferred when removing correlated pairs.

    Args:
        feature_name: Name of the feature

    Returns:
        Priority score (0-100)
    """
    return FEATURE_PRIORITY.get(feature_name, DEFAULT_PRIORITY)


def identify_feature_columns(df: pd.DataFrame) -> List[str]:
    """
    Identify feature columns in the dataframe.

    Excludes metadata, label, and target columns.
    Also excludes cross-asset features when disabled in config.

    Args:
        df: Input DataFrame

    Returns:
        List of feature column names
    """
    excluded_cols = {
        'datetime', 'symbol', 'open', 'high', 'low', 'close', 'volume',
        'timeframe', 'session_id', 'missing_bar', 'roll_event', 'roll_window', 'filled'
    }
    excluded_prefixes = (
        'label_', 'bars_to_hit_', 'mae_', 'mfe_', 'quality_', 'sample_weight_',
        'touch_type_', 'pain_to_gain_', 'time_weighted_dd_', 'fwd_return_',
        'fwd_return_log_', 'time_to_hit_'
    )

    # Get cross-asset feature names
    cross_asset_features = set(get_cross_asset_feature_names())
    cross_asset_enabled = CROSS_ASSET_FEATURES.get('enabled', True)

    feature_cols = [
        c for c in df.columns
        if c not in excluded_cols
        and not any(c.startswith(p) for p in excluded_prefixes)
        and (cross_asset_enabled or c not in cross_asset_features)  # Exclude cross-asset if disabled
    ]

    # Log if cross-asset features were excluded
    if not cross_asset_enabled:
        excluded_cross_asset = [c for c in df.columns if c in cross_asset_features]
        if excluded_cross_asset:
            logger.info(f"Excluding {len(excluded_cross_asset)} cross-asset features (disabled in config): {excluded_cross_asset}")

    return feature_cols


def filter_low_variance(
    df: pd.DataFrame,
    feature_cols: List[str],
    variance_threshold: float = 0.01
) -> Tuple[List[str], List[str]]:
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
        if mean_val > 1e-10:
            normalized_variance = variance / (mean_val ** 2)
        else:
            normalized_variance = variance

        # Check if variance is below threshold
        if variance < variance_threshold and normalized_variance < variance_threshold:
            low_variance.append(col)
        else:
            to_keep.append(col)

    return to_keep, low_variance


def build_correlation_groups(
    df: pd.DataFrame,
    feature_cols: List[str],
    correlation_threshold: float = 0.85
) -> List[Set[str]]:
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
    groups_dict: Dict[str, Set[str]] = {}
    for col in feature_cols:
        root = find(col)
        if root not in groups_dict:
            groups_dict[root] = set()
        groups_dict[root].add(col)

    # Return only groups with more than one member (correlated groups)
    correlated_groups = [group for group in groups_dict.values() if len(group) > 1]

    return correlated_groups


def select_from_correlated_group(group: Set[str]) -> Tuple[str, List[str]]:
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
    sorted_features = sorted(
        group,
        key=lambda f: (-get_feature_priority(f), f)
    )

    selected = sorted_features[0]
    removed = sorted_features[1:]

    return selected, removed


def filter_correlated_features(
    df: pd.DataFrame,
    feature_cols: List[str],
    correlation_threshold: float = 0.85
) -> Tuple[List[str], Dict[str, str], List[List[str]]]:
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
    removed_features: Dict[str, str] = {}
    features_to_remove: Set[str] = set()

    # Convert groups to lists for JSON serialization
    groups_as_lists = []

    for group in correlation_groups:
        groups_as_lists.append(sorted(group))
        selected, removed = select_from_correlated_group(group)

        for feat in removed:
            features_to_remove.add(feat)
            removed_features[feat] = f"correlated with {selected} (priority: {get_feature_priority(selected)} > {get_feature_priority(feat)})"

    # Keep features not in any removed set
    features_to_keep = [f for f in feature_cols if f not in features_to_remove]

    return features_to_keep, removed_features, groups_as_lists


def select_features(
    df: pd.DataFrame,
    correlation_threshold: float = 0.85,
    variance_threshold: float = 0.01,
    feature_cols: Optional[List[str]] = None
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
    logger.info("="*60)
    logger.info("FEATURE SELECTION")
    logger.info("="*60)

    # Identify feature columns if not provided
    if feature_cols is None:
        feature_cols = identify_feature_columns(df)

    original_count = len(feature_cols)
    logger.info(f"Starting with {original_count} features")

    all_removed: Dict[str, str] = {}

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
        kept = [f for f in group if f in selected_features][0] if any(f in selected_features for f in group) else "none"
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
        low_variance_features=low_variance_features
    )

    # Summary
    logger.info(f"\n" + "="*60)
    logger.info("FEATURE SELECTION SUMMARY")
    logger.info("="*60)
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
    result: FeatureSelectionResult,
    output_path: Path,
    include_phase2_recommendations: bool = True
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
        report['phase2_recommendations'] = {
            'use_features': result.selected_features,
            'feature_count': result.final_count,
            'notes': [
                "These features have been filtered for low correlation (<0.85)",
                "Low variance features have been removed",
                "Features are ranked by interpretability/importance",
                "Use this feature list for model training in Phase 2"
            ]
        }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)

    logger.info(f"\nFeature selection report saved to: {output_path}")


def apply_feature_selection(
    df: pd.DataFrame,
    selected_features: List[str],
    keep_metadata: bool = True
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
            'datetime', 'symbol', 'open', 'high', 'low', 'close', 'volume',
            'timeframe', 'session_id', 'missing_bar', 'roll_event', 'roll_window', 'filled'
        ]
        target_prefixes = (
            'label_', 'bars_to_hit_', 'mae_', 'mfe_', 'quality_', 'sample_weight_',
            'touch_type_', 'pain_to_gain_', 'time_weighted_dd_', 'fwd_return_',
            'fwd_return_log_', 'time_to_hit_'
        )

        target_cols = [c for c in df.columns if any(c.startswith(p) for p in target_prefixes)]

        cols_to_keep = (
            [c for c in metadata_cols if c in df.columns] +
            [c for c in selected_features if c in df.columns] +
            target_cols
        )
    else:
        cols_to_keep = [c for c in selected_features if c in df.columns]

    return df[cols_to_keep]


def main():
    """Run feature selection on the combined labeled data."""
    from src.phase1.config import FINAL_DATA_DIR, RESULTS_DIR

    data_path = FINAL_DATA_DIR / "combined_final_labeled.parquet"
    output_path = RESULTS_DIR / "feature_selection_report.json"

    if not data_path.exists():
        logger.error(f"Data file not found: {data_path}")
        return

    # Load data
    logger.info(f"Loading data from {data_path}")
    df = pd.read_parquet(data_path)
    logger.info(f"Loaded {len(df):,} rows, {len(df.columns)} columns")

    # Run feature selection
    result = select_features(
        df,
        correlation_threshold=0.85,
        variance_threshold=0.01
    )

    # Save report
    save_feature_selection_report(result, output_path)

    return result


if __name__ == "__main__":
    main()
