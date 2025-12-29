"""
Feature quality validation checks.
"""
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

logger = logging.getLogger(__name__)


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    """
    Identify feature columns in the DataFrame.

    Each symbol is processed independently (no cross-symbol correlation).

    Args:
        df: DataFrame to analyze

    Returns:
        List of feature column names
    """
    excluded_cols = [
        'datetime', 'symbol', 'open', 'high', 'low', 'close', 'volume',
        'timeframe', 'session_id', 'missing_bar', 'roll_event', 'roll_window', 'filled'
    ]
    excluded_prefixes = (
        'label_', 'bars_to_hit_', 'mae_', 'mfe_', 'quality_', 'sample_weight_',
        'touch_type_', 'pain_to_gain_', 'time_weighted_dd_', 'fwd_return_',
        'fwd_return_log_', 'time_to_hit_'
    )
    feature_cols = [
        c for c in df.columns
        if c not in excluded_cols
        and not any(c.startswith(prefix) for prefix in excluded_prefixes)
    ]

    return feature_cols


def check_feature_correlations(
    feature_df: pd.DataFrame,
    feature_cols: List[str],
    warnings_found: List[str],
    threshold: float = 0.85
) -> List[Dict]:
    """
    Check for highly correlated feature pairs.

    DEPRECATED: Use validate_feature_correlation for more detailed analysis.

    Args:
        feature_df: DataFrame with feature columns
        feature_cols: List of feature column names
        warnings_found: List to append warnings to (mutated)
        threshold: Correlation threshold for flagging

    Returns:
        List of highly correlated pairs
    """
    logger.info("\n1. Correlation analysis...")

    corr_matrix = feature_df.corr()
    high_corr_pairs = []

    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            corr_val = abs(corr_matrix.iloc[i, j])
            if corr_val > threshold:
                pair = {
                    'feature1': corr_matrix.columns[i],
                    'feature2': corr_matrix.columns[j],
                    'correlation': float(corr_val)
                }
                high_corr_pairs.append(pair)

    if high_corr_pairs:
        logger.warning(
            f"  Found {len(high_corr_pairs)} highly correlated feature pairs (>{threshold}):"
        )
        for pair in high_corr_pairs[:10]:
            logger.warning(
                f"    {pair['feature1']} <-> {pair['feature2']}: {pair['correlation']:.3f}"
            )
        if len(high_corr_pairs) > 10:
            logger.warning(f"    ... and {len(high_corr_pairs) - 10} more")
        warnings_found.append(f"{len(high_corr_pairs)} highly correlated feature pairs")
    else:
        logger.info(f"  No highly correlated features found (>{threshold})")

    return high_corr_pairs


def validate_feature_correlation(
    feature_df: pd.DataFrame,
    feature_cols: List[str],
    warnings_found: List[str],
    highly_correlated_threshold: float = 0.95,
    moderately_correlated_threshold: float = 0.80,
    artifacts_dir: Optional[Path] = None,
    save_visualizations: bool = False
) -> Dict:
    """
    Enhanced feature correlation analysis with detailed statistics and recommendations.

    Computes Pearson correlation matrix and identifies:
    - Highly correlated pairs (|correlation| > 0.95)
    - Moderately correlated pairs (|correlation| > 0.80)
    - Provides recommendations for feature removal
    - Optionally generates visualizations

    Args:
        feature_df: DataFrame with feature columns
        feature_cols: List of feature column names
        warnings_found: List to append warnings to (mutated)
        highly_correlated_threshold: Threshold for highly correlated pairs (default: 0.95)
        moderately_correlated_threshold: Threshold for moderately correlated pairs (default: 0.80)
        artifacts_dir: Optional directory to save visualizations
        save_visualizations: Whether to save correlation heatmap (default: False)

    Returns:
        Dictionary with correlation statistics:
        - correlation_matrix: Full correlation matrix (as dict)
        - highly_correlated_pairs: List of pairs with |corr| > 0.95
        - moderately_correlated_pairs: List of pairs with 0.80 < |corr| <= 0.95
        - summary_statistics: Overall correlation stats
        - recommendations: Features to consider removing
    """
    logger.info("\n1. Feature Correlation Analysis...")
    logger.info(f"   - Highly correlated threshold: {highly_correlated_threshold}")
    logger.info(f"   - Moderately correlated threshold: {moderately_correlated_threshold}")

    # Compute correlation matrix
    corr_matrix = feature_df.corr()

    # Extract correlation pairs
    highly_correlated_pairs = []
    moderately_correlated_pairs = []
    all_correlations = []

    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            corr_val = corr_matrix.iloc[i, j]
            abs_corr = abs(corr_val)

            # Skip NaN correlations
            if np.isnan(corr_val):
                continue

            all_correlations.append(abs_corr)

            if abs_corr > highly_correlated_threshold:
                pair = {
                    'feature1': corr_matrix.columns[i],
                    'feature2': corr_matrix.columns[j],
                    'correlation': float(corr_val),
                    'abs_correlation': float(abs_corr)
                }
                highly_correlated_pairs.append(pair)
            elif abs_corr > moderately_correlated_threshold:
                pair = {
                    'feature1': corr_matrix.columns[i],
                    'feature2': corr_matrix.columns[j],
                    'correlation': float(corr_val),
                    'abs_correlation': float(abs_corr)
                }
                moderately_correlated_pairs.append(pair)

    # Sort pairs by absolute correlation (descending)
    highly_correlated_pairs.sort(key=lambda x: x['abs_correlation'], reverse=True)
    moderately_correlated_pairs.sort(key=lambda x: x['abs_correlation'], reverse=True)

    # Compute summary statistics
    summary_stats = {
        'total_features': len(feature_cols),
        'total_pairs_analyzed': len(all_correlations),
        'mean_abs_correlation': float(np.mean(all_correlations)) if all_correlations else 0.0,
        'median_abs_correlation': float(np.median(all_correlations)) if all_correlations else 0.0,
        'max_abs_correlation': float(np.max(all_correlations)) if all_correlations else 0.0,
        'highly_correlated_count': len(highly_correlated_pairs),
        'moderately_correlated_count': len(moderately_correlated_pairs)
    }

    # Log summary
    logger.info(f"\n   Correlation Summary:")
    logger.info(f"   - Total features: {summary_stats['total_features']}")
    logger.info(f"   - Total pairs: {summary_stats['total_pairs_analyzed']:,}")
    logger.info(f"   - Mean |correlation|: {summary_stats['mean_abs_correlation']:.3f}")
    logger.info(f"   - Median |correlation|: {summary_stats['median_abs_correlation']:.3f}")
    logger.info(f"   - Max |correlation|: {summary_stats['max_abs_correlation']:.3f}")

    # Log highly correlated pairs
    if highly_correlated_pairs:
        logger.warning(
            f"\n   Found {len(highly_correlated_pairs)} HIGHLY correlated pairs "
            f"(|r| > {highly_correlated_threshold}):"
        )
        for pair in highly_correlated_pairs[:20]:
            logger.warning(
                f"      {pair['feature1']:30s} <-> {pair['feature2']:30s}: "
                f"{pair['correlation']:+.4f}"
            )
        if len(highly_correlated_pairs) > 20:
            logger.warning(f"      ... and {len(highly_correlated_pairs) - 20} more")

        warnings_found.append(
            f"{len(highly_correlated_pairs)} highly correlated feature pairs (|r| > {highly_correlated_threshold})"
        )
    else:
        logger.info(f"\n   No highly correlated pairs found (|r| > {highly_correlated_threshold})")

    # Log moderately correlated pairs
    if moderately_correlated_pairs:
        logger.info(
            f"\n   Found {len(moderately_correlated_pairs)} MODERATELY correlated pairs "
            f"({moderately_correlated_threshold} < |r| <= {highly_correlated_threshold}):"
        )
        for pair in moderately_correlated_pairs[:10]:
            logger.info(
                f"      {pair['feature1']:30s} <-> {pair['feature2']:30s}: "
                f"{pair['correlation']:+.4f}"
            )
        if len(moderately_correlated_pairs) > 10:
            logger.info(f"      ... and {len(moderately_correlated_pairs) - 10} more")
    else:
        logger.info(
            f"\n   No moderately correlated pairs found "
            f"({moderately_correlated_threshold} < |r| <= {highly_correlated_threshold})"
        )

    # Generate recommendations
    recommendations = _generate_correlation_recommendations(
        highly_correlated_pairs,
        moderately_correlated_pairs,
        feature_cols
    )

    if recommendations['features_to_consider_removing']:
        logger.info(f"\n   Recommendations:")
        logger.info(
            f"   - Consider removing {len(recommendations['features_to_consider_removing'])} "
            f"redundant features"
        )
        logger.info(f"   - This would reduce feature count from "
                   f"{summary_stats['total_features']} to "
                   f"{summary_stats['total_features'] - len(recommendations['features_to_consider_removing'])}")

    # Save visualizations if requested
    if save_visualizations and artifacts_dir:
        _save_correlation_visualizations(
            corr_matrix,
            highly_correlated_pairs,
            moderately_correlated_pairs,
            artifacts_dir
        )

    return {
        'summary_statistics': summary_stats,
        'highly_correlated_pairs': highly_correlated_pairs,
        'moderately_correlated_pairs': moderately_correlated_pairs,
        'recommendations': recommendations
    }


def _generate_correlation_recommendations(
    highly_correlated_pairs: List[Dict],
    moderately_correlated_pairs: List[Dict],
    feature_cols: List[str]
) -> Dict:
    """
    Generate recommendations for which features to consider removing.

    Strategy:
    - For highly correlated pairs, recommend removing one feature from each pair
    - Prefer removing features that appear in multiple correlated pairs
    - Keep features with more interpretable names (e.g., prefer 'sma_20' over 'sma_20_copy')

    Args:
        highly_correlated_pairs: List of highly correlated feature pairs
        moderately_correlated_pairs: List of moderately correlated feature pairs
        feature_cols: All feature column names

    Returns:
        Dictionary with recommendations
    """
    from collections import Counter

    # Count how many times each feature appears in highly correlated pairs
    feature_frequency = Counter()
    for pair in highly_correlated_pairs:
        feature_frequency[pair['feature1']] += 1
        feature_frequency[pair['feature2']] += 1

    # Identify features to remove (greedy approach)
    # Remove features that appear most frequently in correlated pairs
    features_to_remove = set()
    covered_pairs = set()

    for pair_idx, pair in enumerate(highly_correlated_pairs):
        feat1, feat2 = pair['feature1'], pair['feature2']
        pair_key = frozenset([feat1, feat2])

        if pair_key in covered_pairs:
            continue

        # Decide which feature to remove based on:
        # 1. Which appears in more correlated pairs
        # 2. Name heuristics (prefer simpler names)
        freq1 = feature_frequency[feat1]
        freq2 = feature_frequency[feat2]

        if freq1 > freq2:
            to_remove = feat1
        elif freq2 > freq1:
            to_remove = feat2
        else:
            # Same frequency - use name heuristics
            # Prefer to remove features with '_copy', '_dup', longer names, etc.
            to_remove = _choose_feature_to_remove(feat1, feat2)

        features_to_remove.add(to_remove)
        covered_pairs.add(pair_key)

    # Generate detailed recommendations
    recommendations = {
        'features_to_consider_removing': sorted(list(features_to_remove)),
        'removal_rationale': {},
        'feature_frequency_in_correlations': dict(feature_frequency.most_common())
    }

    # Add rationale for each recommended removal
    for feat in features_to_remove:
        correlated_with = [
            pair['feature1'] if pair['feature2'] == feat else pair['feature2']
            for pair in highly_correlated_pairs
            if feat in [pair['feature1'], pair['feature2']]
        ]
        recommendations['removal_rationale'][feat] = {
            'appears_in_pairs': feature_frequency[feat],
            'correlated_with': correlated_with[:5]  # Top 5
        }

    return recommendations


def _choose_feature_to_remove(feat1: str, feat2: str) -> str:
    """
    Choose which feature to remove based on name heuristics.

    Prefers to remove:
    - Features with '_copy', '_dup', '_duplicate' suffixes
    - Features with longer names (likely more derived)
    - Features with higher numerical suffixes

    Args:
        feat1: First feature name
        feat2: Second feature name

    Returns:
        Name of feature to remove
    """
    # Check for obvious copies
    for suffix in ['_copy', '_dup', '_duplicate', '_temp']:
        if suffix in feat1.lower() and suffix not in feat2.lower():
            return feat1
        elif suffix in feat2.lower() and suffix not in feat1.lower():
            return feat2

    # Prefer shorter names (less derived)
    if len(feat1) > len(feat2):
        return feat1
    elif len(feat2) > len(feat1):
        return feat2

    # Alphabetical as fallback
    return feat2 if feat1 < feat2 else feat1


def _save_correlation_visualizations(
    corr_matrix: pd.DataFrame,
    highly_correlated_pairs: List[Dict],
    moderately_correlated_pairs: List[Dict],
    artifacts_dir: Path
) -> None:
    """
    Save correlation visualizations to artifacts directory.

    Creates:
    - correlation_heatmap.png: Full correlation heatmap
    - top_correlated_pairs.txt: List of top 50 correlated pairs

    Args:
        corr_matrix: Correlation matrix
        highly_correlated_pairs: Highly correlated pairs
        moderately_correlated_pairs: Moderately correlated pairs
        artifacts_dir: Directory to save visualizations
    """
    try:
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        logger.warning("matplotlib/seaborn not available - skipping visualizations")
        return

    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # 1. Save correlation heatmap (for subset of features if too large)
    max_features_for_heatmap = 50
    if len(corr_matrix) <= max_features_for_heatmap:
        features_to_plot = corr_matrix.columns
        title = "Feature Correlation Heatmap"
    else:
        # Plot only features involved in high correlations
        features_in_pairs = set()
        for pair in highly_correlated_pairs[:25]:
            features_in_pairs.add(pair['feature1'])
            features_in_pairs.add(pair['feature2'])
        features_to_plot = sorted(list(features_in_pairs))[:max_features_for_heatmap]
        title = f"Feature Correlation Heatmap (Top {len(features_to_plot)} Features)"

    if features_to_plot:
        fig, ax = plt.subplots(figsize=(12, 10))
        corr_subset = corr_matrix.loc[features_to_plot, features_to_plot]

        sns.heatmap(
            corr_subset,
            cmap='RdBu_r',
            center=0,
            vmin=-1,
            vmax=1,
            square=True,
            linewidths=0.5,
            cbar_kws={"shrink": 0.8},
            ax=ax
        )

        ax.set_title(title, fontsize=14, fontweight='bold')
        plt.tight_layout()

        heatmap_path = artifacts_dir / "correlation_heatmap.png"
        plt.savefig(heatmap_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        logger.info(f"   - Saved correlation heatmap: {heatmap_path}")

    # 2. Save top correlated pairs as text file
    pairs_path = artifacts_dir / "top_correlated_pairs.txt"
    with open(pairs_path, 'w') as f:
        f.write("TOP CORRELATED FEATURE PAIRS\n")
        f.write("=" * 80 + "\n\n")

        if highly_correlated_pairs:
            f.write(f"HIGHLY CORRELATED PAIRS (|r| > 0.95): {len(highly_correlated_pairs)}\n")
            f.write("-" * 80 + "\n")
            for i, pair in enumerate(highly_correlated_pairs[:50], 1):
                f.write(
                    f"{i:3d}. {pair['feature1']:30s} <-> {pair['feature2']:30s}: "
                    f"{pair['correlation']:+.6f}\n"
                )
            f.write("\n")

        if moderately_correlated_pairs:
            f.write(f"MODERATELY CORRELATED PAIRS (0.80 < |r| <= 0.95): {len(moderately_correlated_pairs)}\n")
            f.write("-" * 80 + "\n")
            for i, pair in enumerate(moderately_correlated_pairs[:50], 1):
                f.write(
                    f"{i:3d}. {pair['feature1']:30s} <-> {pair['feature2']:30s}: "
                    f"{pair['correlation']:+.6f}\n"
                )

    logger.info(f"   - Saved correlated pairs list: {pairs_path}")


def compute_feature_importance(
    df: pd.DataFrame,
    feature_df: pd.DataFrame,
    feature_cols: List[str],
    label_col: str,
    seed: int = 42,
    sample_size: int = 10000
) -> Tuple[List[Dict], bool]:
    """
    Compute feature importance using Random Forest.

    Args:
        df: Full DataFrame
        feature_df: DataFrame with feature columns
        feature_cols: List of feature column names
        label_col: Label column name
        seed: Random seed
        sample_size: Maximum number of samples to use

    Returns:
        Tuple of (top features list, success boolean)
    """
    logger.info("\n2. Feature importance analysis (Random Forest)...")

    if label_col not in df.columns:
        logger.warning(f"  Label column {label_col} not found")
        return [], False

    # Sample data for speed
    np.random.seed(seed)
    actual_sample_size = min(sample_size, len(df))
    sample_idx = np.random.choice(len(df), size=actual_sample_size, replace=False)
    X_sample = feature_df.iloc[sample_idx].values
    y_sample = df[label_col].iloc[sample_idx].values

    # Remove any remaining NaN/inf
    valid_mask = ~(np.isnan(X_sample).any(axis=1) | np.isinf(X_sample).any(axis=1))
    X_sample = X_sample[valid_mask]
    y_sample = y_sample[valid_mask]

    if len(X_sample) <= 100:
        logger.warning("  Not enough valid samples for feature importance")
        return [], False

    try:
        # Quick RF with few trees
        rf = RandomForestClassifier(
            n_estimators=50, max_depth=10, random_state=42, n_jobs=-1
        )
        rf.fit(X_sample, y_sample)

        # Get top features
        importances = rf.feature_importances_
        top_indices = np.argsort(importances)[-20:][::-1]

        top_features = []
        logger.info("  Top 20 most important features:")
        for idx in top_indices:
            feat_info = {
                'feature': feature_cols[idx],
                'importance': float(importances[idx])
            }
            top_features.append(feat_info)
            logger.info(f"    {feature_cols[idx]:30s}: {importances[idx]:.4f}")

        return top_features, True

    except Exception as e:
        logger.warning(f"  Could not compute feature importance: {e}")
        return [], False


def run_stationarity_tests(
    df: pd.DataFrame, feature_cols: List[str]
) -> List[Dict]:
    """
    Run Augmented Dickey-Fuller tests on selected features.

    Args:
        df: DataFrame with features
        feature_cols: List of feature column names

    Returns:
        List of stationarity test results
    """
    logger.info("\n3. Stationarity tests (Augmented Dickey-Fuller)...")

    from src.phase1.config.features import STATIONARITY_TESTS

    if not STATIONARITY_TESTS.get('enabled', False):
        logger.info("  Stationarity tests disabled by config")
        return []

    from importlib.util import find_spec
    if find_spec("statsmodels") is None:
        raise ModuleNotFoundError(
            "statsmodels is required for stationarity tests when enabled"
        )

    from statsmodels.tsa.stattools import adfuller
    stationarity_results = []

    max_features = STATIONARITY_TESTS.get('max_features', 5)
    # Select features likely to be tested for stationarity
    test_features = [
        c for c in feature_cols if 'return' in c.lower() or 'rsi' in c.lower()
    ][:max_features]

    for feat in test_features:
        series = df[feat].dropna()
        if len(series) > 50:
            result = adfuller(series, autolag='AIC')
            p_value = result[1]
            is_stationary = p_value < 0.05

            stat_info = {
                'feature': feat,
                'adf_statistic': float(result[0]),
                'p_value': float(p_value),
                'is_stationary': bool(is_stationary)
            }
            stationarity_results.append(stat_info)

            status = "Stationary" if is_stationary else "Non-stationary"
            logger.info(f"  {feat:30s}: p={p_value:.4f} {status}")

    return stationarity_results


def check_feature_quality(
    df: pd.DataFrame,
    horizons: List[int],
    warnings_found: List[str],
    seed: int = 42,
    max_features: int = 50,
    artifacts_dir: Optional[Path] = None,
    save_visualizations: bool = False
) -> Dict:
    """
    Run all feature quality checks.

    Args:
        df: DataFrame to validate
        horizons: List of horizons (uses first for importance)
        warnings_found: List to append warnings to (mutated)
        seed: Random seed for reproducibility
        max_features: Maximum features to analyze
        artifacts_dir: Optional directory to save visualizations
        save_visualizations: Whether to save correlation visualizations

    Returns:
        Dictionary with all feature quality results
    """
    logger.info("\n" + "=" * 60)
    logger.info("FEATURE QUALITY CHECKS")
    logger.info("=" * 60)

    results = {}

    # Identify feature columns
    feature_cols = get_feature_columns(df)
    logger.info(f"\nFound {len(feature_cols)} feature columns")
    results['total_features'] = len(feature_cols)

    # Limit features for computational efficiency
    if len(feature_cols) > max_features:
        logger.info(f"Limiting analysis to first {max_features} features for performance")
        feature_cols = feature_cols[:max_features]

    # Prepare feature DataFrame
    feature_df = df[feature_cols].fillna(0)

    # Enhanced correlation analysis
    correlation_results = validate_feature_correlation(
        feature_df=feature_df,
        feature_cols=feature_cols,
        warnings_found=warnings_found,
        highly_correlated_threshold=0.95,
        moderately_correlated_threshold=0.80,
        artifacts_dir=artifacts_dir,
        save_visualizations=save_visualizations
    )
    results['correlation_analysis'] = correlation_results

    # Keep old format for backward compatibility
    results['high_correlations'] = correlation_results['highly_correlated_pairs']

    # Feature importance
    label_col = f'label_h{horizons[0]}'
    top_features, computed = compute_feature_importance(
        df, feature_df, feature_cols, label_col, seed
    )
    results['top_features'] = top_features
    results['feature_importance_computed'] = computed

    # Stationarity tests
    results['stationarity_tests'] = run_stationarity_tests(df, feature_cols)

    return results
