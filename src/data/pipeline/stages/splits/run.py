"""
Stage 7: Create Train/Val/Test Splits.

Pipeline wrapper for splits creation with multi-timeframe support.

PIPE-001/PIPE-002 Fix: This stage now iterates over effective_output_timeframes
to match the file naming pattern from Stage 6 ({symbol}_{tf}_labeled.parquet).
"""

import json
import logging
import traceback
from datetime import datetime
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from src.data.pipeline.utils import StageResult, create_failed_result, create_stage_result

from .core import (
    create_chronological_splits,
    validate_label_distribution,
    validate_per_symbol_distribution,
)

if TYPE_CHECKING:
    from src.core.common.manifest import ArtifactManifest
    from src.data.pipeline.data_config import DataConfig as PipelineConfig

logger = logging.getLogger(__name__)


def run_create_splits(config: "PipelineConfig", manifest: "ArtifactManifest") -> StageResult:
    """
    Stage 7: Create Train/Val/Test Splits.

    Creates time-series aware splits with:
    - Purging: Removes data at split boundaries to prevent label leakage
    - Embargo: Adds buffer period between splits
    - Multi-TF support: Creates per-timeframe splits to preserve temporal integrity

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
        # PIPE-008: Enforce single-symbol isolation at splits stage
        # This is a single-contract ML factory - no cross-symbol features or correlation.
        # Each symbol must be trained in complete isolation.
        if len(config.symbols) > 1 and not config.allow_batch_symbols:
            raise ValueError(
                f"PIPE-008: Splits stage requires single-symbol processing. "
                f"Got {len(config.symbols)} symbols: {config.symbols}. "
                f"This is a single-contract factory - train each symbol in a separate run. "
                f"If batch processing is intentional, set allow_batch_symbols=True."
            )

        # Log single-symbol enforcement for audit trail
        if len(config.symbols) == 1:
            logger.info(f"Single-symbol mode: Processing {config.symbols[0]} in isolation")
        else:
            logger.warning(
                f"Batch symbol mode enabled (allow_batch_symbols=True). "
                f"Processing {len(config.symbols)} symbols: {config.symbols}. "
                f"Note: Cross-symbol features are NOT supported in this architecture."
            )

        # Get effective output timeframes (supports multi-TF mode)
        output_timeframes = config.effective_output_timeframes
        is_multi_tf = len(output_timeframes) > 1

        logger.info(f"Output timeframes: {output_timeframes}")
        logger.info(f"Multi-TF mode: {is_multi_tf}")

        artifacts = []
        all_metadata = {}

        # Process each timeframe separately to preserve temporal integrity
        for tf in output_timeframes:
            logger.info(f"\n{'='*50}")
            logger.info(f"Processing splits for timeframe: {tf}")
            logger.info(f"{'='*50}")

            dfs = []
            for symbol in config.symbols:
                # Stage 6 saves as {symbol}_{tf}_labeled.parquet
                fpath = config.final_data_dir / f"{symbol}_{tf}_labeled.parquet"
                if fpath.exists():
                    df = pd.read_parquet(fpath)
                    # Add timeframe column if not present (for downstream tracking)
                    if "timeframe" not in df.columns:
                        df["timeframe"] = tf
                    dfs.append(df)
                    logger.info(f"Loaded {len(df):,} rows for {symbol} @ {tf}")
                else:
                    logger.warning(f"Labeled data not found for {symbol} @ {tf}: {fpath}")

            if not dfs:
                logger.warning(f"No labeled data found for timeframe {tf}, skipping...")
                continue

            combined_df = pd.concat(dfs, ignore_index=True)
            combined_df = combined_df.sort_values("datetime").reset_index(drop=True)
            logger.info(f"Combined dataset for {tf}: {len(combined_df):,} rows")

            # Save combined file with timeframe suffix for multi-TF mode
            if is_multi_tf:
                combined_path = config.final_data_dir / f"combined_{tf}_final_labeled.parquet"
            else:
                # Backward compatibility: single-TF mode uses original naming
                combined_path = config.final_data_dir / "combined_final_labeled.parquet"

            combined_df.to_parquet(combined_path, index=False)
            logger.info(f"Saved combined dataset to {combined_path}")
            artifacts.append(combined_path)

            # Create splits for this timeframe
            train_indices, val_indices, test_indices, metadata = create_chronological_splits(
                df=combined_df,
                train_ratio=config.train_ratio,
                val_ratio=config.val_ratio,
                test_ratio=config.test_ratio,
                purge_bars=config.purge_bars,
                embargo_bars=config.embargo_bars,
            )

            validate_per_symbol_distribution(combined_df, train_indices, val_indices, test_indices)
            validate_label_distribution(
                combined_df,
                train_indices,
                val_indices,
                test_indices,
                horizons=config.label_horizons,
            )

            # Create per-timeframe splits directory for multi-TF mode
            if is_multi_tf:
                tf_splits_dir = config.splits_dir / tf
            else:
                tf_splits_dir = config.splits_dir

            tf_splits_dir.mkdir(parents=True, exist_ok=True)

            np.save(tf_splits_dir / "train_indices.npy", train_indices)
            np.save(tf_splits_dir / "val_indices.npy", val_indices)
            np.save(tf_splits_dir / "test_indices.npy", test_indices)

            metadata["run_id"] = config.run_id
            metadata["timeframe"] = tf
            metadata["multi_tf_mode"] = is_multi_tf

            with open(tf_splits_dir / "split_config.json", "w") as f:
                json.dump(metadata, f, indent=2)

            logger.info(f"Saved splits for {tf} to {tf_splits_dir}")

            tf_artifacts = [
                tf_splits_dir / "train_indices.npy",
                tf_splits_dir / "val_indices.npy",
                tf_splits_dir / "test_indices.npy",
                tf_splits_dir / "split_config.json",
            ]
            artifacts.extend(tf_artifacts)

            for artifact_path in tf_artifacts:
                manifest.add_artifact(
                    name=f"splits_{tf}_{artifact_path.name}",
                    file_path=artifact_path,
                    stage="create_splits",
                    metadata=metadata,
                )

            all_metadata[tf] = metadata

        # For backward compatibility in single-TF mode, also register combined file
        if not is_multi_tf and artifacts:
            manifest.add_artifact(
                name="splits_combined_final_labeled.parquet",
                file_path=artifacts[0],
                stage="create_splits",
                metadata=all_metadata.get(output_timeframes[0], {}),
            )

        # Create summary metadata
        summary_metadata = {
            "run_id": config.run_id,
            "output_timeframes": output_timeframes,
            "multi_tf_mode": is_multi_tf,
            "timeframe_metadata": all_metadata,
        }

        return create_stage_result(
            stage_name="create_splits",
            start_time=start_time,
            artifacts=artifacts,
            metadata=summary_metadata,
        )

    except Exception as e:
        logger.error(f"Create splits failed: {e}")
        logger.error(traceback.format_exc())
        return create_failed_result(stage_name="create_splits", start_time=start_time, error=str(e))
