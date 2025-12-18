"""
Stage 8: Comprehensive Data Validation
Performs integrity, label sanity, and feature quality checks
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
    horizons: List[int] = [1, 5, 20]
) -> Dict:
    """
    Main validation function.

    Args:
        data_path: Path to combined labeled data
        output_path: Optional path to save validation report (JSON)
        horizons: List of label horizons to validate

    Returns:
        Validation summary dictionary
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

    # Generate summary
    summary = validator.generate_summary()

    # Save report if requested
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=2)
        logger.info(f"\nValidation report saved to: {output_path}")

    logger.info("\n" + "="*70)
    logger.info("STAGE 8 COMPLETE")
    logger.info("="*70)

    return summary


def main():
    """Run validation on default configuration."""
    import sys
    sys.path.insert(0, '/home/user/Research/src')

    from config import FINAL_DATA_DIR, RESULTS_DIR

    data_path = FINAL_DATA_DIR / "combined_final_labeled.parquet"
    output_path = RESULTS_DIR / "validation_report.json"

    if not data_path.exists():
        logger.error(f"Data file not found: {data_path}")
        return

    summary = validate_data(data_path, output_path)

    logger.info(f"\nValidation status: {summary['status']}")
    logger.info(f"Issues: {summary['issues_count']}")
    logger.info(f"Warnings: {summary['warnings_count']}")


if __name__ == "__main__":
    main()
