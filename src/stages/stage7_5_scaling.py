"""
Stage 7.5: Feature Scaling with Train-Only Fitting

This stage applies feature scaling after train/val/test splits are created.
CRITICAL: Scalers are fitted ONLY on training data to prevent data leakage.

The scaling process:
1. Load split indices from Stage 7
2. Fit scaler on training data only
3. Transform val/test using training statistics
4. Save scaled data and scaler parameters for production

Integration with Pipeline:
- This stage runs after stage7_splits and before stage8_validate
- It reads from: data/splits/{run_id}/ (train/val/test indices)
- It writes to: data/splits/{run_id}/scaled/ (scaled parquet files)

Usage (standalone):
    python -m stages.stage7_5_scaling

Usage (via pipeline):
    The pipeline runner automatically executes this stage between
    create_splits and validate stages.

Author: ML Pipeline
Created: 2025-12-21
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Configure logging with NullHandler for library usage
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def identify_feature_columns(df: pd.DataFrame) -> List[str]:
    """
    Identify feature columns to scale (exclude labels, metadata, OHLCV).

    Args:
        df: DataFrame with all columns

    Returns:
        List of feature column names to scale
    """
    excluded_cols = {
        'datetime', 'symbol', 'open', 'high', 'low', 'close', 'volume',
        'timestamp', 'date', 'time'
    }
    excluded_prefixes = (
        'label_', 'bars_to_hit_', 'mae_', 'quality_', 'sample_weight_'
    )

    feature_cols = []
    for col in df.columns:
        if col.lower() in excluded_cols:
            continue
        if any(col.startswith(prefix) for prefix in excluded_prefixes):
            continue
        # Only include numeric columns
        if df[col].dtype in [np.float64, np.float32, np.int64, np.int32, float, int]:
            feature_cols.append(col)

    return feature_cols


def scale_splits(
    combined_df: pd.DataFrame,
    train_indices: np.ndarray,
    val_indices: np.ndarray,
    test_indices: np.ndarray,
    scaler_type: str = 'robust',
    clip_outliers: bool = True,
    clip_range: Tuple[float, float] = (-5.0, 5.0),
    output_dir: Optional[Path] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, 'FeatureScaler']:
    """
    Scale train/val/test splits with train-only fitting.

    Args:
        combined_df: Full dataset DataFrame
        train_indices: Training set indices
        val_indices: Validation set indices
        test_indices: Test set indices
        scaler_type: Type of scaler ('robust', 'standard', 'minmax')
        clip_outliers: Whether to clip outliers after scaling
        clip_range: Range to clip scaled values to
        output_dir: Optional directory to save scaled data

    Returns:
        Tuple of (train_scaled, val_scaled, test_scaled, scaler)
    """
    # Import the feature scaler
    from src.stages.feature_scaler import FeatureScaler, ScalerConfig

    # Create split DataFrames
    train_df = combined_df.iloc[train_indices].copy()
    val_df = combined_df.iloc[val_indices].copy()
    test_df = combined_df.iloc[test_indices].copy()

    # Identify feature columns
    feature_cols = identify_feature_columns(train_df)
    logger.info(f"Identified {len(feature_cols)} feature columns to scale")

    if len(feature_cols) == 0:
        raise ValueError("No feature columns identified for scaling")

    # Create scaler configuration
    config = ScalerConfig(
        scaler_type=scaler_type,
        clip_outliers=clip_outliers,
        clip_range=clip_range
    )

    scaler = FeatureScaler(config=config)

    # Fit ONLY on training data (CRITICAL for preventing leakage)
    logger.info("Fitting scaler on TRAINING data only...")
    train_scaled = scaler.fit_transform(train_df, feature_cols)

    # Transform validation and test using training statistics
    logger.info("Transforming validation data using training statistics...")
    val_scaled = scaler.transform(val_df)

    logger.info("Transforming test data using training statistics...")
    test_scaled = scaler.transform(test_df)

    # Save if output directory provided
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        train_scaled.to_parquet(output_dir / "train_scaled.parquet", index=False)
        val_scaled.to_parquet(output_dir / "val_scaled.parquet", index=False)
        test_scaled.to_parquet(output_dir / "test_scaled.parquet", index=False)
        scaler.save(output_dir / "feature_scaler.pkl")

        logger.info(f"Saved scaled data to {output_dir}")

    return train_scaled, val_scaled, test_scaled, scaler


def run_scaling_stage(
    data_path: Path,
    splits_dir: Path,
    output_dir: Optional[Path] = None,
    scaler_type: str = 'robust',
    clip_outliers: bool = True,
    clip_range: Tuple[float, float] = (-5.0, 5.0)
) -> Dict:
    """
    Run the complete scaling stage.

    Args:
        data_path: Path to combined labeled data parquet
        splits_dir: Directory containing split indices
        output_dir: Output directory for scaled data (default: splits_dir/scaled)
        scaler_type: Type of scaler to use
        clip_outliers: Whether to clip outliers
        clip_range: Range for outlier clipping

    Returns:
        Dictionary with scaling metadata and statistics
    """
    logger.info("=" * 70)
    logger.info("STAGE 7.5: FEATURE SCALING (TRAIN-ONLY)")
    logger.info("=" * 70)

    # Load data
    logger.info(f"Loading data from {data_path}")
    df = pd.read_parquet(data_path)
    logger.info(f"Loaded {len(df):,} rows")

    # Load split indices
    train_indices = np.load(splits_dir / "train_indices.npy")
    val_indices = np.load(splits_dir / "val_indices.npy")
    test_indices = np.load(splits_dir / "test_indices.npy")

    logger.info(f"Split sizes - Train: {len(train_indices):,}, "
                f"Val: {len(val_indices):,}, Test: {len(test_indices):,}")

    # Set output directory
    if output_dir is None:
        output_dir = splits_dir / "scaled"

    # Run scaling
    train_scaled, val_scaled, test_scaled, scaler = scale_splits(
        combined_df=df,
        train_indices=train_indices,
        val_indices=val_indices,
        test_indices=test_indices,
        scaler_type=scaler_type,
        clip_outliers=clip_outliers,
        clip_range=clip_range,
        output_dir=output_dir
    )

    # Get scaling report
    scaling_report = scaler.get_scaling_report()

    # Create metadata
    metadata = {
        'timestamp': datetime.now().isoformat(),
        'scaler_type': scaler_type,
        'clip_outliers': clip_outliers,
        'clip_range': list(clip_range),
        'n_features_scaled': len(scaler.feature_names),
        'train_samples': len(train_indices),
        'val_samples': len(val_indices),
        'test_samples': len(test_indices),
        'feature_columns': scaler.feature_names,
        'features_by_category': scaling_report.get('features_by_category', {}),
        'features_by_scaler': scaling_report.get('features_by_scaler', {}),
        'warnings': scaling_report.get('warnings', [])
    }

    # Save metadata
    metadata_path = output_dir / "scaling_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2, default=str)

    logger.info("\n" + "-" * 50)
    logger.info("SCALING SUMMARY")
    logger.info("-" * 50)
    logger.info(f"Features scaled: {len(scaler.feature_names)}")
    logger.info(f"Scaler type: {scaler_type}")
    logger.info(f"Outlier clipping: {clip_range if clip_outliers else 'disabled'}")
    logger.info(f"Output directory: {output_dir}")

    logger.info("\n" + "=" * 70)
    logger.info("STAGE 7.5 COMPLETE")
    logger.info("=" * 70)

    return metadata


def main():
    """Run scaling stage with default configuration."""
    from src.config import FINAL_DATA_DIR, SPLITS_DIR

    # Find the most recent splits directory
    splits_dirs = sorted(SPLITS_DIR.iterdir()) if SPLITS_DIR.exists() else []

    if not splits_dirs:
        logger.error("No splits directories found. Run stage7_splits first.")
        return

    # Use the most recent splits directory
    latest_splits = splits_dirs[-1]
    data_path = FINAL_DATA_DIR / "combined_final_labeled.parquet"

    if not data_path.exists():
        logger.error(f"Data file not found: {data_path}")
        return

    logger.info(f"Using splits from: {latest_splits}")
    logger.info(f"Data path: {data_path}")

    metadata = run_scaling_stage(
        data_path=data_path,
        splits_dir=latest_splits
    )

    logger.info(f"\nScaling complete. Metadata saved.")


if __name__ == "__main__":
    # Set up logging for standalone execution
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logging.getLogger().setLevel(logging.INFO)
    logging.getLogger().addHandler(handler)
    main()
