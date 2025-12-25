"""
Feature quality validation checks.
"""
import logging
from typing import Dict, List, Tuple

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
    max_features: int = 50
) -> Dict:
    """
    Run all feature quality checks.

    Args:
        df: DataFrame to validate
        horizons: List of horizons (uses first for importance)
        warnings_found: List to append warnings to (mutated)
        seed: Random seed for reproducibility
        max_features: Maximum features to analyze

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

    # Correlation analysis
    results['high_correlations'] = check_feature_correlations(
        feature_df, feature_cols, warnings_found
    )

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
