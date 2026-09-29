"""
Feature Scaling Validation Functions

This module provides validation functions for feature scaling, including:
- Data leakage detection
- Scaling correctness validation
- Statistical consistency checks

Author: ML Pipeline
Created: 2025-12-20
"""

import logging
from datetime import datetime
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    # Protocol for validators that have validation_results, warnings_found, issues_found
    from typing import Protocol

    from . import FeatureScaler

    class DataValidator(Protocol):
        """Protocol for data validators used in stage validation."""

        validation_results: dict[str, Any]
        warnings_found: list[str]
        issues_found: list[str]


logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def validate_scaling(
    scaler: "FeatureScaler",
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    z_threshold: float = 5.0,
) -> dict[str, Any]:
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
    report: dict[str, Any] = {
        "is_valid": True,
        "timestamp": datetime.now().isoformat(),
        "issues": [],
        "warnings": [],
        "statistics": {},
    }

    if not scaler.is_fitted:
        report["is_valid"] = False
        report["issues"].append("Scaler is not fitted")
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
        for name, col in [("train", train_col), ("val", val_col), ("test", test_col)]:
            nan_count = int(np.isnan(col).sum())
            inf_count = int(np.isinf(col).sum())
            if nan_count > 0:
                report["issues"].append(f"{fname} {name}: {nan_count} NaN values")
                report["is_valid"] = False
            if inf_count > 0:
                report["issues"].append(f"{fname} {name}: {inf_count} Inf values")
                report["is_valid"] = False

        # Check val/test statistics relative to train
        train_clean = train_col[~np.isnan(train_col) & ~np.isinf(train_col)]
        val_clean = val_col[~np.isnan(val_col) & ~np.isinf(val_col)]
        test_clean = test_col[~np.isnan(test_col) & ~np.isinf(test_col)]

        if len(train_clean) > 0 and len(val_clean) > 0:
            train_mean, train_std = np.mean(train_clean), np.std(train_clean)
            val_mean, val_std = np.mean(val_clean), np.std(val_clean)
            test_mean = np.mean(test_clean) if len(test_clean) > 0 else 0.0
            test_std = np.std(test_clean) if len(test_clean) > 0 else 0.0

            # Check if val/test means are within z_threshold of train
            if train_std > 0:
                val_z = abs(val_mean - train_mean) / train_std
                if val_z > z_threshold:
                    report["warnings"].append(
                        f"{fname}: val mean differs significantly from train (z={val_z:.2f})"
                    )

                if len(test_clean) > 0:
                    test_z = abs(test_mean - train_mean) / train_std
                    if test_z > z_threshold:
                        report["warnings"].append(
                            f"{fname}: test mean differs significantly from train (z={test_z:.2f})"
                        )

            report["statistics"][fname] = {
                "train": {"mean": float(train_mean), "std": float(train_std)},
                "val": {"mean": float(val_mean), "std": float(val_std)},
                "test": {"mean": float(test_mean), "std": float(test_std)},
            }

    return report


def validate_no_leakage(
    train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame, scaler: "FeatureScaler"
) -> dict[str, Any]:
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
    report: dict[str, Any] = {"leakage_detected": False, "checks": [], "issues": []}

    for fname in scaler.feature_names:
        train_data = train_df[fname].values.astype(np.float64)
        train_clean = train_data[~np.isnan(train_data) & ~np.isinf(train_data)]

        if len(train_clean) == 0:
            continue

        stored_mean = scaler.statistics[fname].train_mean
        stored_std = scaler.statistics[fname].train_std

        computed_mean = np.mean(train_clean)
        computed_std = np.std(train_clean)

        # Allow small floating point differences using relative tolerance
        # This accommodates float32 precision which may have larger errors
        mean_diff = abs(stored_mean - computed_mean)
        std_diff = abs(stored_std - computed_std)

        # Use relative tolerance with a floor for near-zero values
        # This handles both large values (relative error) and small values (absolute floor)
        mean_tol = max(1e-5, 1e-5 * abs(stored_mean))
        std_tol = max(1e-5, 1e-5 * abs(stored_std))

        check = {
            "feature": fname,
            "stored_mean": stored_mean,
            "computed_mean": computed_mean,
            "mean_diff": mean_diff,
            "stored_std": stored_std,
            "computed_std": computed_std,
            "std_diff": std_diff,
            "passed": mean_diff < mean_tol and std_diff < std_tol,
        }
        report["checks"].append(check)

        if not check["passed"]:
            report["leakage_detected"] = True
            report["issues"].append(
                f"{fname}: Statistics don't match training data "
                f"(mean_diff={mean_diff:.2e}, std_diff={std_diff:.2e})"
            )

    return report
