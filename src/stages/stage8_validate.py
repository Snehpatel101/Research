"""
Stage 8: Comprehensive Data Validation
Performs integrity, label sanity, feature quality checks, and feature selection
"""
import numpy as np
import pandas as pd
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import warnings
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from feature_selection import (
    select_features,
    save_feature_selection_report,
    FeatureSelectionResult
)

warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DataValidator:
    """Comprehensive data validation for Phase 1 pipeline."""

    def __init__(self, df: pd.DataFrame, horizons: List[int] = [1, 5, 20]):
        self.df = df
        self.horizons = horizons
        self.validation_results = {}
        self.issues_found = []
        self.warnings_found = []

    def check_data_integrity(self) -> Dict:
        """Check for data quality issues."""
        logger.info("\n" + "="*60)
        logger.info("DATA INTEGRITY CHECKS")
        logger.info("="*60)

        results = {}

        # Check for duplicate timestamps per symbol
        logger.info("\n1. Checking for duplicate timestamps...")
        if 'symbol' in self.df.columns:
            dup_counts = {}
            for symbol in self.df['symbol'].unique():
                symbol_df = self.df[self.df['symbol'] == symbol]
                dups = symbol_df['datetime'].duplicated().sum()
                dup_counts[symbol] = dups
                if dups > 0:
                    self.issues_found.append(f"{symbol}: {dups} duplicate timestamps")
                    logger.warning(f"  {symbol}: {dups:,} duplicate timestamps")
                else:
                    logger.info(f"  {symbol}: No duplicates ✓")
            results['duplicate_timestamps'] = dup_counts
        else:
            dups = self.df['datetime'].duplicated().sum()
            results['duplicate_timestamps'] = {'total': dups}
            if dups > 0:
                self.issues_found.append(f"Found {dups} duplicate timestamps")
                logger.warning(f"  Found {dups:,} duplicate timestamps")
            else:
                logger.info("  No duplicate timestamps ✓")

        # Check for NaN values
        logger.info("\n2. Checking for NaN values...")
        nan_counts = self.df.isna().sum()
        nan_cols = nan_counts[nan_counts > 0]
        if len(nan_cols) > 0:
            logger.warning(f"  Found NaN values in {len(nan_cols)} columns:")
            for col, count in nan_cols.items():
                pct = count / len(self.df) * 100
                logger.warning(f"    {col}: {count:,} ({pct:.2f}%)")
                self.issues_found.append(f"{col}: {count} NaN values ({pct:.2f}%)")
            results['nan_values'] = {col: int(count) for col, count in nan_cols.items()}
        else:
            logger.info("  No NaN values found ✓")
            results['nan_values'] = {}

        # Check for infinite values
        logger.info("\n3. Checking for infinite values...")
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        inf_counts = {}
        for col in numeric_cols:
            inf_count = np.isinf(self.df[col]).sum()
            if inf_count > 0:
                inf_counts[col] = int(inf_count)
                self.issues_found.append(f"{col}: {inf_count} infinite values")
                logger.warning(f"  {col}: {inf_count:,} infinite values")
        if inf_counts:
            results['infinite_values'] = inf_counts
        else:
            logger.info("  No infinite values found ✓")
            results['infinite_values'] = {}

        # Gap analysis
        logger.info("\n4. Analyzing time gaps...")
        gaps = []
        if 'symbol' in self.df.columns:
            for symbol in self.df['symbol'].unique():
                symbol_df = self.df[self.df['symbol'] == symbol].sort_values('datetime')
                time_diffs = symbol_df['datetime'].diff()
                median_gap = time_diffs.median()
                large_gaps = time_diffs[time_diffs > median_gap * 3]
                if len(large_gaps) > 0:
                    gap_info = {
                        'symbol': symbol,
                        'count': int(len(large_gaps)),
                        'median_gap': str(median_gap),
                        'max_gap': str(time_diffs.max())
                    }
                    gaps.append(gap_info)
                    logger.info(f"  {symbol}: {len(large_gaps)} large gaps (>{median_gap*3})")
        else:
            time_diffs = self.df.sort_values('datetime')['datetime'].diff()
            median_gap = time_diffs.median()
            large_gaps = time_diffs[time_diffs > median_gap * 3]
            if len(large_gaps) > 0:
                gap_info = {
                    'count': int(len(large_gaps)),
                    'median_gap': str(median_gap),
                    'max_gap': str(time_diffs.max())
                }
                gaps.append(gap_info)
                logger.info(f"  Found {len(large_gaps)} large gaps")

        results['gaps'] = gaps

        # Date range verification
        logger.info("\n5. Date range verification...")
        date_range = {
            'start': str(self.df['datetime'].min()),
            'end': str(self.df['datetime'].max()),
            'duration_days': float((self.df['datetime'].max() - self.df['datetime'].min()).days),
            'total_bars': int(len(self.df))
        }
        results['date_range'] = date_range
        logger.info(f"  Start: {date_range['start']}")
        logger.info(f"  End:   {date_range['end']}")
        logger.info(f"  Duration: {date_range['duration_days']:.1f} days")
        logger.info(f"  Total bars: {date_range['total_bars']:,}")

        self.validation_results['data_integrity'] = results
        return results

    def check_label_sanity(self) -> Dict:
        """Check label distributions and quality metrics."""
        logger.info("\n" + "="*60)
        logger.info("LABEL SANITY CHECKS")
        logger.info("="*60)

        results = {}

        for horizon in self.horizons:
            label_col = f'label_h{horizon}'
            quality_col = f'quality_h{horizon}'
            bars_col = f'bars_to_hit_h{horizon}'

            if label_col not in self.df.columns:
                logger.warning(f"  Label column {label_col} not found, skipping")
                continue

            logger.info(f"\nHorizon {horizon}:")

            horizon_results = {}

            # Label distribution
            label_counts = self.df[label_col].value_counts().sort_index()
            label_dist = {}
            for label, count in label_counts.items():
                label_name = {-1: 'short', 0: 'neutral', 1: 'long'}.get(label, str(label))
                pct = count / len(self.df) * 100
                label_dist[label_name] = {'count': int(count), 'percentage': float(pct)}
                logger.info(f"  {label_name:8s}: {count:,} ({pct:.2f}%)")

            # Check balance (warn if any class < 20% or > 60%)
            for label_name, stats in label_dist.items():
                pct = stats['percentage']
                if pct < 20.0:
                    self.warnings_found.append(f"h{horizon} {label_name}: low representation ({pct:.1f}%)")
                if pct > 60.0:
                    self.warnings_found.append(f"h{horizon} {label_name}: high representation ({pct:.1f}%)")

            horizon_results['distribution'] = label_dist

            # Win rate per symbol (if symbol column exists)
            if 'symbol' in self.df.columns:
                symbol_stats = {}
                for symbol in self.df['symbol'].unique():
                    symbol_df = self.df[self.df['symbol'] == symbol]
                    labels = symbol_df[label_col]
                    total = len(labels)
                    longs = (labels == 1).sum()
                    shorts = (labels == -1).sum()
                    neutrals = (labels == 0).sum()

                    symbol_stats[symbol] = {
                        'total': int(total),
                        'long_count': int(longs),
                        'short_count': int(shorts),
                        'neutral_count': int(neutrals),
                        'long_pct': float(longs / total * 100),
                        'short_pct': float(shorts / total * 100),
                        'neutral_pct': float(neutrals / total * 100)
                    }

                horizon_results['per_symbol'] = symbol_stats
                logger.info(f"\n  Per-symbol distribution:")
                for symbol, stats in symbol_stats.items():
                    logger.info(f"    {symbol}: L={stats['long_pct']:.1f}% S={stats['short_pct']:.1f}% N={stats['neutral_pct']:.1f}%")

            # Average bars to hit
            if bars_col in self.df.columns:
                avg_bars = self.df[bars_col].mean()
                median_bars = self.df[bars_col].median()
                # Only for non-neutral labels
                hit_mask = self.df[label_col] != 0
                if hit_mask.sum() > 0:
                    avg_bars_hit = self.df[hit_mask][bars_col].mean()
                    median_bars_hit = self.df[hit_mask][bars_col].median()
                else:
                    avg_bars_hit = avg_bars
                    median_bars_hit = median_bars

                bars_stats = {
                    'mean_all': float(avg_bars),
                    'median_all': float(median_bars),
                    'mean_hit': float(avg_bars_hit),
                    'median_hit': float(median_bars_hit)
                }
                horizon_results['bars_to_hit'] = bars_stats
                logger.info(f"\n  Bars to hit statistics:")
                logger.info(f"    Mean (all): {avg_bars:.2f}")
                logger.info(f"    Mean (hit): {avg_bars_hit:.2f}")
                logger.info(f"    Median (hit): {median_bars_hit:.2f}")

            # Quality score distribution
            if quality_col in self.df.columns:
                quality_stats = {
                    'mean': float(self.df[quality_col].mean()),
                    'median': float(self.df[quality_col].median()),
                    'std': float(self.df[quality_col].std()),
                    'min': float(self.df[quality_col].min()),
                    'max': float(self.df[quality_col].max())
                }
                horizon_results['quality'] = quality_stats
                logger.info(f"\n  Quality score statistics:")
                logger.info(f"    Mean: {quality_stats['mean']:.3f}")
                logger.info(f"    Median: {quality_stats['median']:.3f}")
                logger.info(f"    Std: {quality_stats['std']:.3f}")
                logger.info(f"    Range: [{quality_stats['min']:.3f}, {quality_stats['max']:.3f}]")

            results[f'horizon_{horizon}'] = horizon_results

        self.validation_results['label_sanity'] = results
        return results

    def check_feature_quality(self, max_features: int = 50) -> Dict:
        """Check feature correlations and basic importance."""
        logger.info("\n" + "="*60)
        logger.info("FEATURE QUALITY CHECKS")
        logger.info("="*60)

        results = {}

        # Identify feature columns
        excluded_cols = ['datetime', 'symbol', 'open', 'high', 'low', 'close', 'volume']
        feature_cols = [c for c in self.df.columns
                       if c not in excluded_cols
                       and not c.startswith('label_')
                       and not c.startswith('bars_to_hit_')
                       and not c.startswith('mae_')
                       and not c.startswith('quality_')
                       and not c.startswith('sample_weight_')]

        logger.info(f"\nFound {len(feature_cols)} feature columns")
        results['total_features'] = len(feature_cols)

        # Limit features for computational efficiency
        if len(feature_cols) > max_features:
            logger.info(f"Limiting analysis to first {max_features} features for performance")
            feature_cols = feature_cols[:max_features]

        # Check for highly correlated features
        logger.info("\n1. Correlation analysis...")
        feature_df = self.df[feature_cols].fillna(0)
        corr_matrix = feature_df.corr()

        # Find highly correlated pairs (> 0.95)
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = abs(corr_matrix.iloc[i, j])
                if corr_val > 0.95:
                    pair = {
                        'feature1': corr_matrix.columns[i],
                        'feature2': corr_matrix.columns[j],
                        'correlation': float(corr_val)
                    }
                    high_corr_pairs.append(pair)

        if high_corr_pairs:
            logger.warning(f"  Found {len(high_corr_pairs)} highly correlated feature pairs (>0.95):")
            for pair in high_corr_pairs[:10]:  # Show first 10
                logger.warning(f"    {pair['feature1']} <-> {pair['feature2']}: {pair['correlation']:.3f}")
            if len(high_corr_pairs) > 10:
                logger.warning(f"    ... and {len(high_corr_pairs)-10} more")
            self.warnings_found.append(f"{len(high_corr_pairs)} highly correlated feature pairs")
        else:
            logger.info("  No highly correlated features found (>0.95) ✓")

        results['high_correlations'] = high_corr_pairs

        # Feature importance (quick random forest on first horizon)
        logger.info("\n2. Feature importance analysis (Random Forest)...")
        label_col = f'label_h{self.horizons[0]}'

        if label_col in self.df.columns:
            # Sample data for speed (max 10k samples)
            sample_size = min(10000, len(self.df))
            sample_idx = np.random.choice(len(self.df), size=sample_size, replace=False)
            X_sample = feature_df.iloc[sample_idx].values
            y_sample = self.df[label_col].iloc[sample_idx].values

            # Remove any remaining NaN/inf
            valid_mask = ~(np.isnan(X_sample).any(axis=1) | np.isinf(X_sample).any(axis=1))
            X_sample = X_sample[valid_mask]
            y_sample = y_sample[valid_mask]

            if len(X_sample) > 100:
                try:
                    # Quick RF with few trees
                    rf = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)
                    rf.fit(X_sample, y_sample)

                    # Get top features
                    importances = rf.feature_importances_
                    top_indices = np.argsort(importances)[-20:][::-1]  # Top 20

                    top_features = []
                    logger.info("  Top 20 most important features:")
                    for idx in top_indices:
                        feat_info = {
                            'feature': feature_cols[idx],
                            'importance': float(importances[idx])
                        }
                        top_features.append(feat_info)
                        logger.info(f"    {feature_cols[idx]:30s}: {importances[idx]:.4f}")

                    results['top_features'] = top_features
                    results['feature_importance_computed'] = True

                except Exception as e:
                    logger.warning(f"  Could not compute feature importance: {e}")
                    results['feature_importance_computed'] = False
            else:
                logger.warning("  Not enough valid samples for feature importance")
                results['feature_importance_computed'] = False
        else:
            logger.warning(f"  Label column {label_col} not found")
            results['feature_importance_computed'] = False

        # Stationarity test on a few key features
        logger.info("\n3. Stationarity tests (Augmented Dickey-Fuller)...")
        test_features = [c for c in feature_cols if 'return' in c.lower() or 'rsi' in c.lower()][:5]

        stationarity_results = []
        for feat in test_features:
            try:
                from statsmodels.tsa.stattools import adfuller
                series = self.df[feat].dropna()
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

                    status = "✓ Stationary" if is_stationary else "✗ Non-stationary"
                    logger.info(f"  {feat:30s}: p={p_value:.4f} {status}")
            except Exception as e:
                logger.warning(f"  Could not test {feat}: {e}")

        results['stationarity_tests'] = stationarity_results

        self.validation_results['feature_quality'] = results
        return results

    def check_feature_normalization(self, z_threshold: float = 3.0, extreme_threshold: float = 5.0) -> Dict:
        """
        Check feature normalization, distributions, and outliers.

        Args:
            z_threshold: Z-score threshold for outlier warning (default 3.0)
            extreme_threshold: Z-score threshold for extreme outlier issue (default 5.0)

        Returns:
            Dictionary with normalization validation results
        """
        logger.info("\n" + "="*60)
        logger.info("FEATURE NORMALIZATION CHECKS")
        logger.info("="*60)

        results = {}

        # Identify feature columns (same pattern as check_feature_quality)
        excluded_cols = ['datetime', 'symbol', 'open', 'high', 'low', 'close', 'volume']
        feature_cols = [c for c in self.df.columns
                       if c not in excluded_cols
                       and not c.startswith('label_')
                       and not c.startswith('bars_to_hit_')
                       and not c.startswith('mae_')
                       and not c.startswith('quality_')
                       and not c.startswith('sample_weight_')]

        logger.info(f"\nAnalyzing normalization for {len(feature_cols)} features")
        results['total_features'] = len(feature_cols)

        # 1. Feature Distribution Statistics
        logger.info("\n1. Feature distribution statistics...")
        feature_stats = []
        unnormalized_features = []
        high_skew_features = []

        for col in feature_cols:
            series = self.df[col].replace([np.inf, -np.inf], np.nan).dropna()
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

            # Skewness and Kurtosis
            try:
                skewness = float(stats.skew(series))
                kurtosis = float(stats.kurtosis(series))
            except Exception:
                skewness = 0.0
                kurtosis = 0.0

            stat_info = {
                'feature': col,
                'mean': mean_val,
                'std': std_val,
                'min': min_val,
                'max': max_val,
                'p1': p1,
                'p5': p5,
                'p50': p50,
                'p95': p95,
                'p99': p99,
                'skewness': skewness,
                'kurtosis': kurtosis
            }
            feature_stats.append(stat_info)

            # Flag unnormalized features (std far from 1, mean far from 0)
            if std_val > 100 or abs(mean_val) > 100:
                unnormalized_features.append({
                    'feature': col,
                    'mean': mean_val,
                    'std': std_val,
                    'issue': 'large_scale'
                })

            # Flag highly skewed features
            if abs(skewness) > 2.0:
                high_skew_features.append({
                    'feature': col,
                    'skewness': skewness,
                    'kurtosis': kurtosis
                })

        results['feature_statistics'] = feature_stats

        if unnormalized_features:
            logger.warning(f"  Found {len(unnormalized_features)} features with large scale (may need normalization):")
            for feat in unnormalized_features[:5]:
                logger.warning(f"    {feat['feature']:30s}: mean={feat['mean']:.2f}, std={feat['std']:.2f}")
            if len(unnormalized_features) > 5:
                logger.warning(f"    ... and {len(unnormalized_features)-5} more")
            self.warnings_found.append(f"{len(unnormalized_features)} features need normalization (large scale)")
        else:
            logger.info("  All features have reasonable scale ✓")

        results['unnormalized_features'] = unnormalized_features

        if high_skew_features:
            logger.warning(f"  Found {len(high_skew_features)} highly skewed features (|skew| > 2):")
            for feat in high_skew_features[:5]:
                logger.warning(f"    {feat['feature']:30s}: skew={feat['skewness']:.2f}")
            if len(high_skew_features) > 5:
                logger.warning(f"    ... and {len(high_skew_features)-5} more")
            self.warnings_found.append(f"{len(high_skew_features)} features highly skewed")
        else:
            logger.info("  No highly skewed features ✓")

        results['high_skew_features'] = high_skew_features

        # 2. Z-Score Outlier Detection
        logger.info(f"\n2. Z-score outlier detection (threshold={z_threshold})...")
        outlier_summary = []
        extreme_outlier_features = []

        for col in feature_cols:
            series = self.df[col].replace([np.inf, -np.inf], np.nan).dropna()
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
                'feature': col,
                'outliers_3std': int(outliers_3std),
                'outliers_5std': int(outliers_5std),
                'outliers_10std': int(outliers_10std),
                'pct_beyond_3std': float(pct_3std),
                'pct_beyond_5std': float(pct_5std),
                'max_z_score': float(z_scores.max())
            }
            outlier_summary.append(outlier_info)

            # Flag features with extreme outliers (>1% beyond 5 std)
            if pct_5std > 1.0:
                extreme_outlier_features.append(outlier_info)

        results['outlier_analysis'] = outlier_summary

        if extreme_outlier_features:
            logger.warning(f"  Found {len(extreme_outlier_features)} features with extreme outliers (>1% beyond 5σ):")
            for feat in extreme_outlier_features[:5]:
                logger.warning(f"    {feat['feature']:30s}: {feat['pct_beyond_5std']:.2f}% beyond 5σ, max_z={feat['max_z_score']:.1f}")
            if len(extreme_outlier_features) > 5:
                logger.warning(f"    ... and {len(extreme_outlier_features)-5} more")
            self.warnings_found.append(f"{len(extreme_outlier_features)} features have extreme outliers")
        else:
            logger.info("  No features with excessive extreme outliers ✓")

        results['extreme_outlier_features'] = extreme_outlier_features

        # 3. Feature Range Analysis
        logger.info("\n3. Feature range analysis...")
        range_issues = []

        for col in feature_cols:
            series = self.df[col].replace([np.inf, -np.inf], np.nan).dropna()
            if len(series) < 100:
                continue

            min_val = series.min()
            max_val = series.max()
            range_val = max_val - min_val

            # Check for potential issues
            if range_val == 0:
                range_issues.append({
                    'feature': col,
                    'issue': 'constant_value',
                    'value': float(min_val)
                })
                self.issues_found.append(f"{col}: constant value ({min_val})")
            elif range_val > 1e6:
                range_issues.append({
                    'feature': col,
                    'issue': 'extreme_range',
                    'min': float(min_val),
                    'max': float(max_val),
                    'range': float(range_val)
                })

        results['range_issues'] = range_issues

        if range_issues:
            constant_count = sum(1 for r in range_issues if r['issue'] == 'constant_value')
            extreme_count = sum(1 for r in range_issues if r['issue'] == 'extreme_range')
            if constant_count > 0:
                logger.warning(f"  Found {constant_count} constant features (zero variance)")
            if extreme_count > 0:
                logger.warning(f"  Found {extreme_count} features with extreme range (>1M)")
        else:
            logger.info("  All features have reasonable ranges ✓")

        # 4. Normalization Recommendations
        logger.info("\n4. Normalization recommendations...")
        recommendations = []

        # Features that need StandardScaler
        needs_scaling = [f for f in unnormalized_features if f['std'] > 10]
        if needs_scaling:
            recommendations.append({
                'type': 'StandardScaler',
                'features': [f['feature'] for f in needs_scaling],
                'reason': 'Features with std > 10 should be standardized'
            })
            logger.info(f"  StandardScaler recommended for {len(needs_scaling)} features")

        # Features that need log transform (high positive skew)
        needs_log = [f for f in high_skew_features if f['skewness'] > 2.0]
        if needs_log:
            recommendations.append({
                'type': 'LogTransform',
                'features': [f['feature'] for f in needs_log],
                'reason': 'Features with skew > 2 may benefit from log transform'
            })
            logger.info(f"  Log transform may help {len(needs_log)} skewed features")

        # Features that need RobustScaler (many outliers)
        needs_robust = [f for f in extreme_outlier_features if f['pct_beyond_5std'] > 0.5]
        if needs_robust:
            recommendations.append({
                'type': 'RobustScaler',
                'features': [f['feature'] for f in needs_robust],
                'reason': 'Features with many outliers should use RobustScaler'
            })
            logger.info(f"  RobustScaler recommended for {len(needs_robust)} features with outliers")

        if not recommendations:
            logger.info("  No specific normalization needed ✓")

        results['recommendations'] = recommendations

        # 5. Summary statistics
        total_warnings = len(unnormalized_features) + len(high_skew_features) + len(extreme_outlier_features)
        total_issues = len([r for r in range_issues if r['issue'] == 'constant_value'])

        results['summary'] = {
            'features_analyzed': len(feature_cols),
            'unnormalized_count': len(unnormalized_features),
            'high_skew_count': len(high_skew_features),
            'extreme_outlier_count': len(extreme_outlier_features),
            'constant_features': total_issues,
            'needs_attention': total_warnings > 0 or total_issues > 0
        }

        logger.info(f"\n  Summary: {len(feature_cols)} features analyzed")
        logger.info(f"    - Unnormalized: {len(unnormalized_features)}")
        logger.info(f"    - High skew: {len(high_skew_features)}")
        logger.info(f"    - Extreme outliers: {len(extreme_outlier_features)}")
        logger.info(f"    - Constant: {total_issues}")

        self.validation_results['feature_normalization'] = results
        return results

    def run_feature_selection(
        self,
        correlation_threshold: float = 0.95,
        variance_threshold: float = 0.01
    ) -> FeatureSelectionResult:
        """
        Run feature selection to identify and remove redundant features.

        This addresses multicollinearity by removing highly correlated features
        while keeping the most interpretable feature from each correlated group.

        Args:
            correlation_threshold: Threshold for feature correlation (default 0.95)
            variance_threshold: Minimum variance to keep feature (default 0.01)

        Returns:
            FeatureSelectionResult with selected and removed features
        """
        logger.info("\n" + "="*60)
        logger.info("FEATURE SELECTION")
        logger.info("="*60)

        result = select_features(
            self.df,
            correlation_threshold=correlation_threshold,
            variance_threshold=variance_threshold
        )

        # Store results in validation_results
        self.validation_results['feature_selection'] = result.to_dict()

        # Update warnings if features were removed
        if len(result.removed_features) > 0:
            # Replace the old correlation warning with a more informative one
            self.warnings_found = [
                w for w in self.warnings_found
                if 'correlated feature pairs' not in w
            ]
            self.warnings_found.append(
                f"Feature selection removed {len(result.removed_features)} redundant features "
                f"({result.original_count} -> {result.final_count})"
            )

        return result

    def generate_summary(self) -> Dict:
        """Generate validation summary."""
        logger.info("\n" + "="*60)
        logger.info("VALIDATION SUMMARY")
        logger.info("="*60)

        summary = {
            'timestamp': datetime.now().isoformat(),
            'total_rows': int(len(self.df)),
            'total_columns': int(len(self.df.columns)),
            'issues_count': len(self.issues_found),
            'warnings_count': len(self.warnings_found),
            'issues': self.issues_found,
            'warnings': self.warnings_found,
            'validation_results': self.validation_results
        }

        # Determine overall status
        if len(self.issues_found) == 0:
            summary['status'] = 'PASSED'
            logger.info("\n✓ All validation checks PASSED")
        else:
            summary['status'] = 'FAILED'
            logger.error(f"\n✗ Validation FAILED with {len(self.issues_found)} issues")

        if len(self.warnings_found) > 0:
            logger.warning(f"\n⚠ {len(self.warnings_found)} warnings found:")
            for warning in self.warnings_found:
                logger.warning(f"  - {warning}")

        if len(self.issues_found) > 0:
            logger.error(f"\n✗ {len(self.issues_found)} issues found:")
            for issue in self.issues_found:
                logger.error(f"  - {issue}")

        return summary


def validate_data(
    data_path: Path,
    output_path: Optional[Path] = None,
    horizons: List[int] = [1, 5, 20],
    run_feature_selection: bool = True,
    correlation_threshold: float = 0.95,
    variance_threshold: float = 0.01,
    feature_selection_output_path: Optional[Path] = None
) -> Tuple[Dict, Optional[FeatureSelectionResult]]:
    """
    Main validation function.

    Args:
        data_path: Path to combined labeled data
        output_path: Optional path to save validation report (JSON)
        horizons: List of label horizons to validate
        run_feature_selection: Whether to run feature selection (default True)
        correlation_threshold: Threshold for feature correlation (default 0.95)
        variance_threshold: Minimum variance to keep feature (default 0.01)
        feature_selection_output_path: Optional path to save feature selection report

    Returns:
        Tuple of (validation summary dict, FeatureSelectionResult or None)
    """
    logger.info("="*70)
    logger.info("STAGE 8: DATA VALIDATION")
    logger.info("="*70)

    # Load data
    logger.info(f"\nLoading data from {data_path}")
    df = pd.read_parquet(data_path)
    logger.info(f"Loaded {len(df):,} rows, {len(df.columns)} columns")

    # Create validator
    validator = DataValidator(df, horizons=horizons)

    # Run all checks
    validator.check_data_integrity()
    validator.check_label_sanity()
    validator.check_feature_quality()
    validator.check_feature_normalization()

    # Run feature selection if requested
    feature_selection_result = None
    if run_feature_selection:
        feature_selection_result = validator.run_feature_selection(
            correlation_threshold=correlation_threshold,
            variance_threshold=variance_threshold
        )

        # Save feature selection report if path provided
        if feature_selection_output_path:
            save_feature_selection_report(
                feature_selection_result,
                feature_selection_output_path
            )

    # Generate summary
    summary = validator.generate_summary()

    # Save report if requested
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Custom JSON encoder for numpy types
        def numpy_encoder(obj):
            if isinstance(obj, (np.integer, np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64, np.float32)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.bool_):
                return bool(obj)
            raise TypeError(f'Object of type {type(obj)} is not JSON serializable')

        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=2, default=numpy_encoder)
        logger.info(f"\nValidation report saved to: {output_path}")

    logger.info("\n" + "="*70)
    logger.info("STAGE 8 COMPLETE")
    logger.info("="*70)

    return summary, feature_selection_result


def main():
    """Run validation on default configuration."""
    from config import FINAL_DATA_DIR, RESULTS_DIR

    data_path = FINAL_DATA_DIR / "combined_final_labeled.parquet"
    output_path = RESULTS_DIR / "validation_report.json"
    feature_selection_path = RESULTS_DIR / "feature_selection_report.json"

    if not data_path.exists():
        logger.error(f"Data file not found: {data_path}")
        return

    summary, feature_selection_result = validate_data(
        data_path,
        output_path,
        run_feature_selection=True,
        feature_selection_output_path=feature_selection_path
    )

    logger.info(f"\nValidation status: {summary['status']}")
    logger.info(f"Issues: {summary['issues_count']}")
    logger.info(f"Warnings: {summary['warnings_count']}")

    if feature_selection_result:
        logger.info(f"\nFeature Selection Results:")
        logger.info(f"  Original features: {feature_selection_result.original_count}")
        logger.info(f"  Selected features: {feature_selection_result.final_count}")
        logger.info(f"  Removed features: {len(feature_selection_result.removed_features)}")


if __name__ == "__main__":
    main()
