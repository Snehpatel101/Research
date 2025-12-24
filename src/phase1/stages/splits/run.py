"""
Stage 7: Create Train/Val/Test Splits.

Pipeline wrapper for splits creation.
"""
import json
import logging
import traceback
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from src.pipeline.utils import StageResult, create_stage_result, create_failed_result
from .core import (
    create_chronological_splits,
    validate_per_symbol_distribution,
    validate_label_distribution
)

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

        combined_path = config.final_data_dir / "combined_final_labeled.parquet"
        combined_df.to_parquet(combined_path, index=False)
        logger.info(f"Saved combined dataset to {combined_path}")

        train_indices, val_indices, test_indices, metadata = create_chronological_splits(
            df=combined_df,
            train_ratio=config.train_ratio,
            val_ratio=config.val_ratio,
            test_ratio=config.test_ratio,
            purge_bars=config.purge_bars,
            embargo_bars=config.embargo_bars
        )

        validate_per_symbol_distribution(
            combined_df, train_indices, val_indices, test_indices
        )
        validate_label_distribution(
            combined_df, train_indices, val_indices, test_indices,
            horizons=config.label_horizons
        )

        config.splits_dir.mkdir(parents=True, exist_ok=True)
        np.save(config.splits_dir / "train_indices.npy", train_indices)
        np.save(config.splits_dir / "val_indices.npy", val_indices)
        np.save(config.splits_dir / "test_indices.npy", test_indices)

        metadata['run_id'] = config.run_id
        with open(config.splits_dir / "split_config.json", 'w') as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Saved splits to {config.splits_dir}")

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
                metadata=metadata
            )

        return create_stage_result(
            stage_name="create_splits",
            start_time=start_time,
            artifacts=artifacts,
            metadata=metadata
        )

    except Exception as e:
        logger.error(f"Create splits failed: {e}")
        logger.error(traceback.format_exc())
        return create_failed_result(
            stage_name="create_splits",
            start_time=start_time,
            error=str(e)
        )
