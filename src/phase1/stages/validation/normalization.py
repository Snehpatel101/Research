"""
Feature normalization validation checks.
"""

import logging

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """
    Identify feature columns in the DataFrame.

    Args:
        df: DataFrame to analyze

    Returns:
        List of feature column names
    """
    excluded_cols = [
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
    excluded_prefixes = (
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
    feature_cols = [
        c
        for c in df.columns
        if c not in excluded_cols and not any(c.startswith(prefix) for prefix in excluded_prefixes)
    ]
    return feature_cols


def compute_feature_statistics(
    df: pd.DataFrame, feature_cols: list[str], warnings_found: list[str]
) -> tuple[list[dict], list[dict], list[dict]]:
    """
    Compute distribution statistics for all features.

    Args:
        df: DataFrame with features
        feature_cols: List of feature column names
        warnings_found: List to append warnings to (mutated)

    Returns:
        Tuple of (feature_stats, unnormalized_features, high_skew_features)
    """
    logger.info("\n1. Feature distribution statistics...")

    feature_stats = []
    unnormalized_features = []
    high_skew_features = []

    for col in feature_cols:
        series = df[col].replace([np.inf, -np.inf], np.nan).dropna()
        if len(series) < 100:
            continue

        mean_val = float(series.mean())
        std_val = float(series.std())
        min_val = float(series.min())
        max_val = float(series.max())

        # Percentiles
        p1 = float(np.percentile(series, 1))
        p5 = float(np.percentile(series, 5))
        p50 = float(np.percentile(series, 50))
        p95 = float(np.percentile(series, 95))
        p99 = float(np.percentile(series, 99))

        # Skewness and Kurtosis - with proper error handling
        try:
            skewness = float(stats.skew(series))
        except (ValueError, RuntimeWarning) as e:
            logger.warning(f"  Skewness calculation failed for {col}: {e}")
            skewness = np.nan

        try:
            kurtosis = float(stats.kurtosis(series))
        except (ValueError, RuntimeWarning) as e:
            logger.warning(f"  Kurtosis calculation failed for {col}: {e}")
            kurtosis = np.nan

        stat_info = {
            "feature": col,
            "mean": mean_val,
            "std": std_val,
            "min": min_val,
            "max": max_val,
            "p1": p1,
            "p5": p5,
            "p50": p50,
            "p95": p95,
            "p99": p99,
            "skewness": skewness,
            "kurtosis": kurtosis,
        }
        feature_stats.append(stat_info)

        # Flag unnormalized features (std far from 1, mean far from 0)
        if std_val > 100 or abs(mean_val) > 100:
            unnormalized_features.append(
                {"feature": col, "mean": mean_val, "std": std_val, "issue": "large_scale"}
            )

        # Flag highly skewed features (only if skewness was computed successfully)
        if not np.isnan(skewness) and abs(skewness) > 2.0:
            high_skew_features.append({"feature": col, "skewness": skewness, "kurtosis": kurtosis})

    # Log unnormalized features
    if unnormalized_features:
        logger.warning(
            f"  Found {len(unnormalized_features)} features with large scale "
            "(may need normalization):"
        )
        for feat in unnormalized_features[:5]:
            logger.warning(
                f"    {feat['feature']:30s}: mean={feat['mean']:.2f}, std={feat['std']:.2f}"
            )
        if len(unnormalized_features) > 5:
            logger.warning(f"    ... and {len(unnormalized_features) - 5} more")
        warnings_found.append(
            f"{len(unnormalized_features)} features need normalization (large scale)"
        )
    else:
        logger.info("  All features have reasonable scale")

    # Log high skew features
    if high_skew_features:
        logger.warning(f"  Found {len(high_skew_features)} highly skewed features (|skew| > 2):")
        for feat in high_skew_features[:5]:
            logger.warning(f"    {feat['feature']:30s}: skew={feat['skewness']:.2f}")
        if len(high_skew_features) > 5:
            logger.warning(f"    ... and {len(high_skew_features) - 5} more")
        warnings_found.append(f"{len(high_skew_features)} features highly skewed")
    else:
        logger.info("  No highly skewed features")

    return feature_stats, unnormalized_features, high_skew_features


def detect_outliers(
    df: pd.DataFrame, feature_cols: list[str], warnings_found: list[str], z_threshold: float = 3.0
) -> tuple[list[dict], list[dict]]:
    """
    Detect outliers using z-score analysis.

    Args:
        df: DataFrame with features
        feature_cols: List of feature column names
        warnings_found: List to append warnings to (mutated)
        z_threshold: Z-score threshold for detection

    Returns:
        Tuple of (outlier_summary, extreme_outlier_features)
    """
    logger.info(f"\n2. Z-score outlier detection (threshold={z_threshold})...")

    outlier_summary = []
    extreme_outlier_features = []

    for col in feature_cols:
        series = df[col].replace([np.inf, -np.inf], np.nan).dropna()
        if len(series) < 100:
            continue

        mean_val = series.mean()
        std_val = series.std()

        if std_val == 0:
            continue

        z_scores = np.abs((series - mean_val) / std_val)

        # Count outliers at different thresholds
        outliers_3std = (z_scores > 3.0).sum()
        outliers_5std = (z_scores > 5.0).sum()
        outliers_10std = (z_scores > 10.0).sum()

        pct_3std = outliers_3std / len(series) * 100
        pct_5std = outliers_5std / len(series) * 100

        outlier_info = {
            "feature": col,
            "outliers_3std": int(outliers_3std),
            "outliers_5std": int(outliers_5std),
            "outliers_10std": int(outliers_10std),
            "pct_beyond_3std": float(pct_3std),
            "pct_beyond_5std": float(pct_5std),
            "max_z_score": float(z_scores.max()),
        }
        outlier_summary.append(outlier_info)

        # Flag features with extreme outliers (>1% beyond 5 std)
        if pct_5std > 1.0:
            extreme_outlier_features.append(outlier_info)

    if extreme_outlier_features:
        logger.warning(
            f"  Found {len(extreme_outlier_features)} features with extreme "
            "outliers (>1% beyond 5 sigma):"
        )
        for feat in extreme_outlier_features[:5]:
            logger.warning(
                f"    {feat['feature']:30s}: {feat['pct_beyond_5std']:.2f}% beyond "
                f"5 sigma, max_z={feat['max_z_score']:.1f}"
            )
        if len(extreme_outlier_features) > 5:
            logger.warning(f"    ... and {len(extreme_outlier_features) - 5} more")
        warnings_found.append(f"{len(extreme_outlier_features)} features have extreme outliers")
    else:
        logger.info("  No features with excessive extreme outliers")

    return outlier_summary, extreme_outlier_features


def analyze_feature_ranges(
    df: pd.DataFrame, feature_cols: list[str], issues_found: list[str], warnings_found: list[str]
) -> list[dict]:
    """
    Analyze feature value ranges for issues.

    Args:
        df: DataFrame with features
        feature_cols: List of feature column names
        issues_found: List to append issues to (mutated)
        warnings_found: List to append warnings to (mutated)

    Returns:
        List of range issues
    """
    logger.info("\n3. Feature range analysis...")

    range_issues = []

    for col in feature_cols:
        series = df[col].replace([np.inf, -np.inf], np.nan).dropna()
        if len(series) < 100:
            continue

        min_val = series.min()
        max_val = series.max()
        range_val = max_val - min_val

        # Check for potential issues
        if range_val == 0:
            range_issues.append(
                {"feature": col, "issue": "constant_value", "value": float(min_val)}
            )
            # Binary indicator features may legitimately be constant
            binary_indicator_patterns = [
                "cross_up",
                "cross_down",
                "overbought",
                "oversold",
                "_direction",
                "regime_",
                "_flag",
            ]
            is_binary_indicator = any(p in col.lower() for p in binary_indicator_patterns)
            if is_binary_indicator:
                warnings_found.append(f"{col}: constant value ({min_val}) - binary indicator")
            else:
                issues_found.append(f"{col}: constant value ({min_val})")
        elif range_val > 1e6:
            range_issues.append(
                {
                    "feature": col,
                    "issue": "extreme_range",
                    "min": float(min_val),
                    "max": float(max_val),
                    "range": float(range_val),
                }
            )

    if range_issues:
        constant_count = sum(1 for r in range_issues if r["issue"] == "constant_value")
        extreme_count = sum(1 for r in range_issues if r["issue"] == "extreme_range")
        if constant_count > 0:
            logger.warning(f"  Found {constant_count} constant features (zero variance)")
        if extreme_count > 0:
            logger.warning(f"  Found {extreme_count} features with extreme range (>1M)")
    else:
        logger.info("  All features have reasonable ranges")

    return range_issues


def generate_recommendations(
    unnormalized_features: list[dict],
    high_skew_features: list[dict],
    extreme_outlier_features: list[dict],
) -> list[dict]:
    """
    Generate normalization recommendations.

    Args:
        unnormalized_features: List of features needing normalization
        high_skew_features: List of highly skewed features
        extreme_outlier_features: List of features with extreme outliers

    Returns:
        List of recommendations
    """
    logger.info("\n4. Normalization recommendations...")

    recommendations = []

    # Features that need StandardScaler
    needs_scaling = [f for f in unnormalized_features if f["std"] > 10]
    if needs_scaling:
        recommendations.append(
            {
                "type": "StandardScaler",
                "features": [f["feature"] for f in needs_scaling],
                "reason": "Features with std > 10 should be standardized",
            }
        )
        logger.info(f"  StandardScaler recommended for {len(needs_scaling)} features")

    # Features that need log transform (high positive skew)
    needs_log = [f for f in high_skew_features if f["skewness"] > 2.0]
    if needs_log:
        recommendations.append(
            {
                "type": "LogTransform",
                "features": [f["feature"] for f in needs_log],
                "reason": "Features with skew > 2 may benefit from log transform",
            }
        )
        logger.info(f"  Log transform may help {len(needs_log)} skewed features")

    # Features that need RobustScaler (many outliers)
    needs_robust = [f for f in extreme_outlier_features if f["pct_beyond_5std"] > 0.5]
    if needs_robust:
        recommendations.append(
            {
                "type": "RobustScaler",
                "features": [f["feature"] for f in needs_robust],
                "reason": "Features with many outliers should use RobustScaler",
            }
        )
        logger.info(f"  RobustScaler recommended for {len(needs_robust)} features with outliers")

    if not recommendations:
        logger.info("  No specific normalization needed")

    return recommendations


def check_feature_normalization(
    df: pd.DataFrame,
    issues_found: list[str],
    warnings_found: list[str],
    z_threshold: float = 3.0,
    extreme_threshold: float = 5.0,
) -> dict:
    """
    Run all feature normalization checks.

    Args:
        df: DataFrame to validate
        issues_found: List to append issues to (mutated)
        warnings_found: List to append warnings to (mutated)
        z_threshold: Z-score threshold for outlier warning
        extreme_threshold: Z-score threshold for extreme outliers

    Returns:
        Dictionary with all normalization validation results
    """
    logger.info("\n" + "=" * 60)
    logger.info("FEATURE NORMALIZATION CHECKS")
    logger.info("=" * 60)

    results = {}

    # Identify feature columns
    feature_cols = get_feature_columns(df)
    logger.info(f"\nAnalyzing normalization for {len(feature_cols)} features")
    results["total_features"] = len(feature_cols)

    # 1. Feature Distribution Statistics
    feature_stats, unnormalized_features, high_skew_features = compute_feature_statistics(
        df, feature_cols, warnings_found
    )
    results["feature_statistics"] = feature_stats
    results["unnormalized_features"] = unnormalized_features
    results["high_skew_features"] = high_skew_features

    # 2. Z-Score Outlier Detection
    outlier_summary, extreme_outlier_features = detect_outliers(
        df, feature_cols, warnings_found, z_threshold
    )
    results["outlier_analysis"] = outlier_summary
    results["extreme_outlier_features"] = extreme_outlier_features

    # 3. Feature Range Analysis
    results["range_issues"] = analyze_feature_ranges(df, feature_cols, issues_found, warnings_found)

    # 4. Normalization Recommendations
    results["recommendations"] = generate_recommendations(
        unnormalized_features, high_skew_features, extreme_outlier_features
    )

    # 5. Summary statistics
    total_warnings = (
        len(unnormalized_features) + len(high_skew_features) + len(extreme_outlier_features)
    )
    total_issues = sum(1 for r in results["range_issues"] if r["issue"] == "constant_value")

    results["summary"] = {
        "features_analyzed": len(feature_cols),
        "unnormalized_count": len(unnormalized_features),
        "high_skew_count": len(high_skew_features),
        "extreme_outlier_count": len(extreme_outlier_features),
        "constant_features": total_issues,
        "needs_attention": total_warnings > 0 or total_issues > 0,
    }

    logger.info(f"\n  Summary: {len(feature_cols)} features analyzed")
    logger.info(f"    - Unnormalized: {len(unnormalized_features)}")
    logger.info(f"    - High skew: {len(high_skew_features)}")
    logger.info(f"    - Extreme outliers: {len(extreme_outlier_features)}")
    logger.info(f"    - Constant: {total_issues}")

    return results
