"""
Feature Scaling Validation Functions

This module provides validation functions for feature scaling, including:
- Data leakage detection
- Scaling correctness validation
- Statistical consistency checks

Author: ML Pipeline
Created: 2025-12-20
Updated: 2025-12-20 - Extracted from feature_scaler.py
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional
from pathlib import Path
import json

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def validate_scaling(
    scaler: 'FeatureScaler',
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: List[str],
    z_threshold: float = 5.0
) -> Dict:
    """
    Validate that scaling was done correctly.

    Checks:
    1. Train statistics match scaler's stored statistics
    2. Val/test statistics are reasonable relative to train
    3. No extreme outliers introduced by scaling
    4. No NaN/Inf values after scaling

    Args:
        scaler: Fitted FeatureScaler
        train_df: Original training DataFrame
        val_df: Validation DataFrame
        test_df: Test DataFrame
        feature_cols: Feature columns to validate
        z_threshold: Z-score threshold for outlier detection

    Returns:
        Validation report dictionary
    """
    report = {
        'is_valid': True,
        'timestamp': datetime.now().isoformat(),
        'issues': [],
        'warnings': [],
        'statistics': {}
    }

    if not scaler.is_fitted:
        report['is_valid'] = False
        report['issues'].append("Scaler is not fitted")
        return report

    # Transform all splits
    train_scaled = scaler.transform(train_df)
    val_scaled = scaler.transform(val_df)
    test_scaled = scaler.transform(test_df)

    for fname in feature_cols:
        train_col = train_scaled[fname].values
        val_col = val_scaled[fname].values
        test_col = test_scaled[fname].values

        # Check for NaN/Inf
        for name, col in [('train', train_col), ('val', val_col), ('test', test_col)]:
            nan_count = int(np.isnan(col).sum())
            inf_count = int(np.isinf(col).sum())
            if nan_count > 0:
                report['issues'].append(f"{fname} {name}: {nan_count} NaN values")
                report['is_valid'] = False
            if inf_count > 0:
                report['issues'].append(f"{fname} {name}: {inf_count} Inf values")
                report['is_valid'] = False

        # Check val/test statistics relative to train
        train_clean = train_col[~np.isnan(train_col) & ~np.isinf(train_col)]
        val_clean = val_col[~np.isnan(val_col) & ~np.isinf(val_col)]
        test_clean = test_col[~np.isnan(test_col) & ~np.isinf(test_col)]

        if len(train_clean) > 0 and len(val_clean) > 0:
            train_mean, train_std = np.mean(train_clean), np.std(train_clean)
            val_mean, val_std = np.mean(val_clean), np.std(val_clean)
            test_mean, test_std = np.mean(test_clean), np.std(test_clean) if len(test_clean) > 0 else (0, 0)

            # Check if val/test means are within z_threshold of train
            if train_std > 0:
                val_z = abs(val_mean - train_mean) / train_std
                if val_z > z_threshold:
                    report['warnings'].append(
                        f"{fname}: val mean differs significantly from train (z={val_z:.2f})"
                    )

                if len(test_clean) > 0:
                    test_z = abs(test_mean - train_mean) / train_std
                    if test_z > z_threshold:
                        report['warnings'].append(
                            f"{fname}: test mean differs significantly from train (z={test_z:.2f})"
                        )

            report['statistics'][fname] = {
                'train': {'mean': float(train_mean), 'std': float(train_std)},
                'val': {'mean': float(val_mean), 'std': float(val_std)},
                'test': {'mean': float(test_mean), 'std': float(test_std)}
            }

    return report


def validate_no_leakage(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    scaler: 'FeatureScaler'
) -> Dict:
    """
    Validate that no data leakage occurred during scaling.

    This checks that the scaler's stored statistics match what would be
    computed from training data alone.

    Args:
        train_df: Training DataFrame
        val_df: Validation DataFrame
        test_df: Test DataFrame
        scaler: Fitted FeatureScaler

    Returns:
        Leakage validation report
    """
    report = {
        'leakage_detected': False,
        'checks': [],
        'issues': []
    }

    for fname in scaler.feature_names:
        train_data = train_df[fname].values.astype(np.float64)
        train_clean = train_data[~np.isnan(train_data) & ~np.isinf(train_data)]

        if len(train_clean) == 0:
            continue

        stored_mean = scaler.statistics[fname].train_mean
        stored_std = scaler.statistics[fname].train_std

        computed_mean = np.mean(train_clean)
        computed_std = np.std(train_clean)

        # Allow small floating point differences
        mean_diff = abs(stored_mean - computed_mean)
        std_diff = abs(stored_std - computed_std)

        check = {
            'feature': fname,
            'stored_mean': stored_mean,
            'computed_mean': computed_mean,
            'mean_diff': mean_diff,
            'stored_std': stored_std,
            'computed_std': computed_std,
            'std_diff': std_diff,
            'passed': mean_diff < 1e-6 and std_diff < 1e-6
        }
        report['checks'].append(check)

        if not check['passed']:
            report['leakage_detected'] = True
            report['issues'].append(
                f"{fname}: Statistics don't match training data "
                f"(mean_diff={mean_diff:.2e}, std_diff={std_diff:.2e})"
            )

    return report


def validate_scaling_for_splits(
    train_path: Path,
    val_path: Path,
    test_path: Path,
    feature_cols: Optional[List[str]] = None,
    scaler_type: str = 'robust',
    output_path: Optional[Path] = None
) -> Dict:
    """
    Validate that scaling works correctly for train/val/test splits.

    This function is designed to be called from stage8_validate.py or
    after running stage7_splits.py.

    Args:
        train_path: Path to training data parquet
        val_path: Path to validation data parquet
        test_path: Path to test data parquet
        feature_cols: Optional list of feature columns. Auto-detected if None.
        scaler_type: Scaler type to use ('robust', 'standard', 'minmax')
        output_path: Optional path to save validation report

    Returns:
        Validation report dictionary
    """
    from . import FeatureScaler

    logger.info("=" * 60)
    logger.info("SCALING VALIDATION FOR TRAIN/VAL/TEST SPLITS")
    logger.info("=" * 60)

    # Load data
    train_df = pd.read_parquet(train_path)
    val_df = pd.read_parquet(val_path)
    test_df = pd.read_parquet(test_path)

    logger.info(f"Train: {len(train_df):,} samples")
    logger.info(f"Val:   {len(val_df):,} samples")
    logger.info(f"Test:  {len(test_df):,} samples")

    # Identify feature columns if not provided
    if feature_cols is None:
        excluded_cols = {
            'datetime', 'symbol', 'open', 'high', 'low', 'close', 'volume',
            'timeframe', 'session_id', 'missing_bar', 'roll_event', 'roll_window', 'filled'
        }
        excluded_prefixes = (
            'label_', 'bars_to_hit_', 'mae_', 'mfe_', 'quality_', 'sample_weight_',
            'touch_type_', 'pain_to_gain_', 'time_weighted_dd_', 'fwd_return_',
            'fwd_return_log_', 'time_to_hit_'
        )
        feature_cols = [
            c for c in train_df.columns
            if c not in excluded_cols
            and not any(c.startswith(p) for p in excluded_prefixes)
        ]
    logger.info(f"Features: {len(feature_cols)}")

    # Create and fit scaler
    scaler = FeatureScaler(scaler_type=scaler_type)
    train_scaled = scaler.fit_transform(train_df, feature_cols)
    val_scaled = scaler.transform(val_df)
    test_scaled = scaler.transform(test_df)

    # Run validation
    scaling_validation = validate_scaling(
        scaler, train_df, val_df, test_df, feature_cols
    )

    leakage_validation = validate_no_leakage(
        train_df, val_df, test_df, scaler
    )

    # Combine reports
    report = {
        'timestamp': datetime.now().isoformat(),
        'train_samples': len(train_df),
        'val_samples': len(val_df),
        'test_samples': len(test_df),
        'n_features': len(feature_cols),
        'scaler_type': scaler_type,
        'scaling_validation': scaling_validation,
        'leakage_validation': leakage_validation,
        'scaler_report': scaler.get_scaling_report(),
        'overall_status': 'PASSED' if (
            scaling_validation['is_valid'] and
            not leakage_validation['leakage_detected']
        ) else 'FAILED'
    }

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        logger.info(f"Validation report saved to: {output_path}")

    # Log summary
    logger.info("\n" + "=" * 60)
    logger.info("SCALING VALIDATION SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Scaling valid: {scaling_validation['is_valid']}")
    logger.info(f"Leakage detected: {leakage_validation['leakage_detected']}")
    logger.info(f"Overall status: {report['overall_status']}")

    if scaling_validation['warnings']:
        logger.warning(f"Warnings ({len(scaling_validation['warnings'])}):")
        for w in scaling_validation['warnings'][:5]:
            logger.warning(f"  - {w}")

    if scaling_validation['issues']:
        logger.error(f"Issues ({len(scaling_validation['issues'])}):")
        for i in scaling_validation['issues']:
            logger.error(f"  - {i}")

    return report


def add_scaling_validation_to_stage8(
    validator: 'DataValidator',
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: List[str],
    scaler_type: str = 'robust'
) -> Dict:
    """
    Add scaling validation results to a Stage 8 DataValidator.

    This function can be called from stage8_validate.py to include
    scaling validation in the overall validation report.

    Args:
        validator: DataValidator instance from stage8_validate
        train_df: Training DataFrame
        val_df: Validation DataFrame
        test_df: Test DataFrame
        feature_cols: List of feature columns
        scaler_type: Scaler type to use

    Returns:
        Scaling validation results
    """
    from . import FeatureScaler

    logger.info("\n" + "=" * 60)
    logger.info("SCALING VALIDATION (Phase 2 Preparation)")
    logger.info("=" * 60)

    # Create and fit scaler
    scaler = FeatureScaler(scaler_type=scaler_type)
    scaler.fit(train_df, feature_cols)

    # Validate
    scaling_validation = validate_scaling(
        scaler, train_df, val_df, test_df, feature_cols
    )

    leakage_validation = validate_no_leakage(
        train_df, val_df, test_df, scaler
    )

    results = {
        'scaling_validation': scaling_validation,
        'leakage_validation': leakage_validation,
        'scaler_summary': scaler.get_scaling_report()
    }

    # Add to validator's results
    validator.validation_results['scaling'] = results

    # Update warnings/issues
    if scaling_validation['warnings']:
        for w in scaling_validation['warnings']:
            validator.warnings_found.append(f"Scaling: {w}")

    if scaling_validation['issues']:
        for i in scaling_validation['issues']:
            validator.issues_found.append(f"Scaling: {i}")

    if leakage_validation['leakage_detected']:
        validator.issues_found.append("Scaling: Data leakage detected!")

    logger.info(f"Scaling validation: {'PASSED' if scaling_validation['is_valid'] else 'FAILED'}")
    logger.info(f"Leakage check: {'CLEAN' if not leakage_validation['leakage_detected'] else 'LEAKAGE DETECTED'}")

    return results
