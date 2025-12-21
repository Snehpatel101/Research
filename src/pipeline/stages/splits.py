"""
Stage 7: Create Train/Val/Test Splits.

Creates time-aware data splits with purging and embargo to prevent data leakage.
"""
import json
import logging
import traceback
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from ..utils import StageResult, StageStatus, create_stage_result, create_failed_result

if TYPE_CHECKING:
    from pipeline_config import PipelineConfig
    from manifest import ArtifactManifest

logger = logging.getLogger(__name__)


def run_create_splits(
    config: 'PipelineConfig',
    manifest: 'ArtifactManifest'
) -> StageResult:
    """
    Stage 7: Create Train/Val/Test Splits.

    Creates time-series aware splits with:
    - Purging: Removes data at split boundaries to prevent label leakage
    - Embargo: Adds buffer period between splits

    Args:
        config: Pipeline configuration
        manifest: Artifact manifest for tracking outputs

    Returns:
        StageResult with status and artifacts
    """
    start_time = datetime.now()
    logger.info("=" * 70)
    logger.info("STAGE 7: Create Train/Val/Test Splits")
    logger.info("=" * 70)

    try:
        # Load and combine labeled data
        dfs = []
        for symbol in config.symbols:
            fpath = config.final_data_dir / f"{symbol}_labeled.parquet"
            if fpath.exists():
                df = pd.read_parquet(fpath)
                dfs.append(df)
                logger.info(f"Loaded {len(df):,} rows for {symbol}")

        if not dfs:
            raise RuntimeError("No labeled data found!")

        combined_df = pd.concat(dfs, ignore_index=True)
        combined_df = combined_df.sort_values('datetime').reset_index(drop=True)
        logger.info(f"Combined dataset: {len(combined_df):,} rows")

        # Save combined dataset
        combined_path = config.final_data_dir / "combined_final_labeled.parquet"
        combined_df.to_parquet(combined_path, index=False)
        logger.info(f"Saved combined dataset to {combined_path}")

        # Create splits
        n = len(combined_df)
        train_end = int(n * config.train_ratio)
        val_end = int(n * (config.train_ratio + config.val_ratio))

        # Apply purging at train_end
        train_end_purged = train_end - config.purge_bars

        if train_end_purged <= 0:
            raise ValueError(
                f"Train set eliminated by purging. "
                f"train_end={train_end}, purge_bars={config.purge_bars}. "
                f"Need more data or reduce purge_bars."
            )

        val_start = train_end + config.embargo_bars

        # Apply purging at val_end too (prevents label leakage into test set)
        val_end_purged = val_end - config.purge_bars
        test_start = val_end + config.embargo_bars  # Use original val_end for embargo

        # Create indices
        train_indices = np.arange(0, train_end_purged)
        val_indices = np.arange(val_start, val_end_purged)
        test_indices = np.arange(test_start, n)

        logger.info("Split sizes:")
        logger.info(f"  Train: {len(train_indices):,} samples")
        logger.info(f"  Val:   {len(val_indices):,} samples")
        logger.info(f"  Test:  {len(test_indices):,} samples")

        # Get date ranges
        train_dates = combined_df.iloc[train_indices]['datetime']
        val_dates = combined_df.iloc[val_indices]['datetime']
        test_dates = combined_df.iloc[test_indices]['datetime']

        logger.info("Date ranges:")
        logger.info(f"  Train: {train_dates.min()} to {train_dates.max()}")
        logger.info(f"  Val:   {val_dates.min()} to {val_dates.max()}")
        logger.info(f"  Test:  {test_dates.min()} to {test_dates.max()}")

        # Validate per-symbol distribution and label distribution
        from stages.stage7_splits import (
            validate_per_symbol_distribution,
            validate_label_distribution
        )

        validate_per_symbol_distribution(
            combined_df, train_indices, val_indices, test_indices
        )
        validate_label_distribution(
            combined_df, train_indices, val_indices, test_indices,
            horizons=config.label_horizons
        )

        # Save indices
        config.splits_dir.mkdir(parents=True, exist_ok=True)
        np.save(config.splits_dir / "train_indices.npy", train_indices)
        np.save(config.splits_dir / "val_indices.npy", val_indices)
        np.save(config.splits_dir / "test_indices.npy", test_indices)

        # Save metadata
        split_config = {
            "run_id": config.run_id,
            "total_samples": n,
            "train_samples": len(train_indices),
            "val_samples": len(val_indices),
            "test_samples": len(test_indices),
            "purge_bars": config.purge_bars,
            "embargo_bars": config.embargo_bars,
            "train_date_start": str(train_dates.min()),
            "train_date_end": str(train_dates.max()),
            "val_date_start": str(val_dates.min()),
            "val_date_end": str(val_dates.max()),
            "test_date_start": str(test_dates.min()),
            "test_date_end": str(test_dates.max()),
        }

        with open(config.splits_dir / "split_config.json", 'w') as f:
            json.dump(split_config, f, indent=2)

        logger.info(f"Saved splits to {config.splits_dir}")

        # Track artifacts
        artifacts = [
            combined_path,
            config.splits_dir / "train_indices.npy",
            config.splits_dir / "val_indices.npy",
            config.splits_dir / "test_indices.npy",
            config.splits_dir / "split_config.json"
        ]

        for artifact_path in artifacts:
            manifest.add_artifact(
                name=f"splits_{artifact_path.name}",
                file_path=artifact_path,
                stage="create_splits",
                metadata=split_config
            )

        return create_stage_result(
            stage_name="create_splits",
            start_time=start_time,
            artifacts=artifacts,
            metadata=split_config
        )

    except Exception as e:
        logger.error(f"Create splits failed: {e}")
        logger.error(traceback.format_exc())
        return create_failed_result(
            stage_name="create_splits",
            start_time=start_time,
            error=str(e)
        )
