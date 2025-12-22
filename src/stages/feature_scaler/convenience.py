"""
Convenience Functions for Feature Scaling

This module provides high-level convenience functions for common scaling tasks.

Author: ML Pipeline
Created: 2025-12-20
Updated: 2025-12-20 - Extracted from feature_scaler.py
"""

import logging
import pandas as pd
from pathlib import Path
from typing import Optional, Tuple, List

from .scaler import FeatureScaler
from .core import ScalerConfig

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def scale_splits(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: List[str],
    scaler_path: Optional[Path] = None,
    config: Optional[ScalerConfig] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, FeatureScaler]:
    """
    Scale train/val/test splits with train-only fitting.

    This is the recommended simple interface for scaling data.

    Args:
        train_df: Training DataFrame
        val_df: Validation DataFrame
        test_df: Test DataFrame
        feature_cols: Feature columns to scale
        scaler_path: Optional path to save the fitted scaler
        config: Optional ScalerConfig for customization

    Returns:
        Tuple of (train_scaled, val_scaled, test_scaled, scaler)

    Example:
        >>> train_scaled, val_scaled, test_scaled, scaler = scale_splits(
        ...     train_df, val_df, test_df, feature_cols
        ... )
    """
    scaler = FeatureScaler(config=config) if config else FeatureScaler()

    # Fit ONLY on training data
    train_scaled = scaler.fit_transform(train_df, feature_cols)

    # Transform val and test using training statistics
    val_scaled = scaler.transform(val_df)
    test_scaled = scaler.transform(test_df)

    # Save scaler if path provided
    if scaler_path:
        scaler.save(scaler_path)

    logger.info("\nScaling complete:")
    logger.info(f"  Train: {len(train_scaled):,} samples")
    logger.info(f"  Val:   {len(val_scaled):,} samples")
    logger.info(f"  Test:  {len(test_scaled):,} samples")

    return train_scaled, val_scaled, test_scaled, scaler


def scale_train_val_test(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: List[str],
    scaler_type: str = 'robust',
    save_path: Optional[Path] = None,
    clip_outliers: bool = True,
    clip_range: Tuple[float, float] = (-5.0, 5.0)
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, FeatureScaler]:
    """
    Scale train/val/test data with a scaler fitted on training data only.

    This is an alternative interface with more explicit parameters.

    Args:
        train_df: Training DataFrame
        val_df: Validation DataFrame
        test_df: Test DataFrame
        feature_cols: Feature columns to scale
        scaler_type: Scaler type ('robust', 'standard', 'minmax')
        save_path: Optional path to save the fitted scaler
        clip_outliers: Whether to clip outliers after scaling
        clip_range: Range to clip scaled values to

    Returns:
        Tuple of (train_scaled, val_scaled, test_scaled, scaler)
    """
    scaler = FeatureScaler(
        scaler_type=scaler_type,
        clip_outliers=clip_outliers,
        clip_range=clip_range
    )

    # Fit ONLY on training data
    train_scaled = scaler.fit_transform(train_df, feature_cols)

    # Transform val and test using training statistics
    val_scaled = scaler.transform(val_df)
    test_scaled = scaler.transform(test_df)

    if save_path:
        scaler.save(save_path)

    logger.info("\nScaling complete:")
    logger.info(f"  Train: {len(train_scaled):,} samples")
    logger.info(f"  Val:   {len(val_scaled):,} samples")
    logger.info(f"  Test:  {len(test_scaled):,} samples")

    return train_scaled, val_scaled, test_scaled, scaler
