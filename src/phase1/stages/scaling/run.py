"""
Stage 7.5: Feature Scaling.

Pipeline wrapper for train-only feature scaling.
"""
import json
import logging
import traceback
from datetime import datetime
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from src.phase1.stages.scaling import (
    FeatureScaler,
    ScalerConfig,
    validate_no_leakage,
    validate_scaling,
)
from src.pipeline.utils import StageResult, create_failed_result, create_stage_result

if TYPE_CHECKING:
    from manifest import ArtifactManifest
    from pipeline_config import PipelineConfig

logger = logging.getLogger(__name__)


def _identify_feature_columns(df: pd.DataFrame) -> list[str]:
    """
    Identify feature columns to scale (exclude labels, metadata, OHLCV).

    Args:
        df: DataFrame with all columns

    Returns:
        List of feature column names to scale
    """
    excluded_cols = {
        'datetime', 'symbol', 'open', 'high', 'low', 'close', 'volume',
        'timestamp', 'date', 'time', 'timeframe',
        'session_id', 'missing_bar', 'roll_event', 'roll_window', 'filled'
    }
    excluded_prefixes = (
        'label_', 'bars_to_hit_', 'mae_', 'mfe_', 'quality_', 'sample_weight_',
        'touch_type_', 'pain_to_gain_', 'time_weighted_dd_', 'fwd_return_',
        'fwd_return_log_', 'time_to_hit_'
    )

    feature_cols = []
    for col in df.columns:
        if col.lower() in excluded_cols:
            continue
        if any(col.startswith(prefix) for prefix in excluded_prefixes):
            continue
        if df[col].dtype in [np.float64, np.float32, np.int64, np.int32, float, int]:
            feature_cols.append(col)

    return feature_cols


def run_feature_scaling(
    config: 'PipelineConfig',
    manifest: 'ArtifactManifest'
) -> StageResult:
    """
    Stage 7.5: Feature Scaling.

    Applies train-only scaling to prevent data leakage:
    1. Loads train/val/test splits
    2. Fits scaler ONLY on training data
    3. Transforms all splits using training statistics
    4. Saves scaled data and scaler parameters

    Args:
        config: Pipeline configuration
        manifest: Artifact manifest for tracking outputs

    Returns:
        StageResult with status and artifacts
    """
    start_time = datetime.now()
    logger.info("=" * 70)
    logger.info("STAGE 7.5: Feature Scaling (Train-Only)")
    logger.info("=" * 70)

    try:
        combined_path = config.final_data_dir / "combined_final_labeled.parquet"
        if not combined_path.exists():
            raise FileNotFoundError(f"Combined data not found: {combined_path}")

        df = pd.read_parquet(combined_path)
        logger.info(f"Loaded combined data: {len(df):,} rows")

        train_indices = np.load(config.splits_dir / "train_indices.npy")
        val_indices = np.load(config.splits_dir / "val_indices.npy")
        test_indices = np.load(config.splits_dir / "test_indices.npy")

        logger.info(f"Split sizes - Train: {len(train_indices):,}, "
                    f"Val: {len(val_indices):,}, Test: {len(test_indices):,}")

        train_df = df.iloc[train_indices].copy()
        val_df = df.iloc[val_indices].copy()
        test_df = df.iloc[test_indices].copy()

        feature_cols = _identify_feature_columns(train_df)
        logger.info(f"Identified {len(feature_cols)} feature columns to scale")

        if len(feature_cols) == 0:
            raise ValueError("No feature columns identified for scaling")

        logger.info("Sample features to scale:")
        for col in feature_cols[:10]:
            logger.info(f"  - {col}")
        if len(feature_cols) > 10:
            logger.info(f"  ... and {len(feature_cols) - 10} more")

        scaler_config = ScalerConfig(
            scaler_type='robust',
            clip_outliers=True,
            clip_range=(-5.0, 5.0)
        )

        scaler = FeatureScaler(config=scaler_config)

        logger.info("\nFitting scaler on TRAINING data only...")
        train_scaled = scaler.fit_transform(train_df, feature_cols)
        logger.info(f"Scaler fitted on {scaler.n_samples_train:,} training samples")

        logger.info("Transforming validation data using training statistics...")
        val_scaled = scaler.transform(val_df)

        logger.info("Transforming test data using training statistics...")
        test_scaled = scaler.transform(test_df)

        logger.info("\nValidating scaling integrity...")
        scaling_validation = validate_scaling(
            scaler, train_df, val_df, test_df, feature_cols
        )

        if not scaling_validation['is_valid']:
            logger.warning("Scaling validation found issues:")
            for issue in scaling_validation['issues'][:5]:
                logger.warning(f"  - {issue}")

        logger.info("Checking for data leakage...")
        leakage_check = validate_no_leakage(train_df, val_df, test_df, scaler)

        if leakage_check['leakage_detected']:
            raise RuntimeError(
                f"DATA LEAKAGE DETECTED in scaling! Issues: {leakage_check['issues']}"
            )
        logger.info("No data leakage detected in scaling")

        scaled_data_dir = config.splits_dir / "scaled"
        scaled_data_dir.mkdir(parents=True, exist_ok=True)

        train_scaled_path = scaled_data_dir / "train_scaled.parquet"
        val_scaled_path = scaled_data_dir / "val_scaled.parquet"
        test_scaled_path = scaled_data_dir / "test_scaled.parquet"

        train_scaled.to_parquet(train_scaled_path, index=False)
        val_scaled.to_parquet(val_scaled_path, index=False)
        test_scaled.to_parquet(test_scaled_path, index=False)

        logger.info("\nSaved scaled data:")
        logger.info(f"  Train: {train_scaled_path}")
        logger.info(f"  Val:   {val_scaled_path}")
        logger.info(f"  Test:  {test_scaled_path}")

        scaler_path = scaled_data_dir / "feature_scaler.pkl"
        scaler.save(scaler_path)
        logger.info(f"  Scaler: {scaler_path}")

        scaling_report = scaler.get_scaling_report()
        scaling_metadata = {
            'run_id': config.run_id,
            'timestamp': datetime.now().isoformat(),
            'scaler_type': 'robust',
            'clip_outliers': True,
            'clip_range': [-5.0, 5.0],
            'n_features_scaled': len(feature_cols),
            'train_samples': len(train_indices),
            'val_samples': len(val_indices),
            'test_samples': len(test_indices),
            'feature_columns': feature_cols,
            'scaling_validation': {
                'is_valid': scaling_validation['is_valid'],
                'warnings_count': len(scaling_validation.get('warnings', [])),
                'issues_count': len(scaling_validation.get('issues', []))
            },
            'leakage_check': {
                'leakage_detected': leakage_check['leakage_detected'],
                'checks_passed': len([
                    c for c in leakage_check.get('checks', [])
                    if c.get('passed', False)
                ])
            },
            'features_by_category': scaling_report.get('features_by_category', {}),
            'features_by_scaler': scaling_report.get('features_by_scaler', {})
        }

        metadata_path = scaled_data_dir / "scaling_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(scaling_metadata, f, indent=2, default=str)
        logger.info(f"  Metadata: {metadata_path}")

        logger.info("\n" + "-" * 50)
        logger.info("SCALING SUMMARY")
        logger.info("-" * 50)
        logger.info(f"Features scaled: {len(feature_cols)}")
        logger.info("Scaler type: robust")
        logger.info("Outlier clipping: [-5.0, 5.0]")
        logger.info(f"Scaling validation: {'PASSED' if scaling_validation['is_valid'] else 'WARNINGS'}")
        logger.info("Leakage check: PASSED")

        artifacts = [
            train_scaled_path,
            val_scaled_path,
            test_scaled_path,
            scaler_path,
            metadata_path
        ]

        for artifact_path in artifacts:
            manifest.add_artifact(
                name=f"scaling_{artifact_path.name}",
                file_path=artifact_path,
                stage="feature_scaling",
                metadata=scaling_metadata
            )

        logger.info("\n" + "=" * 70)
        logger.info("STAGE 7.5 COMPLETE: Feature Scaling")
        logger.info("=" * 70)

        return create_stage_result(
            stage_name="feature_scaling",
            start_time=start_time,
            artifacts=artifacts,
            metadata=scaling_metadata
        )

    except Exception as e:
        logger.error(f"Feature scaling failed: {e}")
        logger.error(traceback.format_exc())
        return create_failed_result(
            stage_name="feature_scaling",
            start_time=start_time,
            error=str(e)
        )
