"""
Stage 4: Initial Triple-Barrier Labeling - Pipeline Wrapper.

This module provides the pipeline integration for initial labeling,
extracting the orchestration logic from pipeline/stages/labeling.py.
"""
import logging
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Any

import pandas as pd

from .triple_barrier import triple_barrier_numba

if TYPE_CHECKING:
    from pipeline_config import PipelineConfig
    from manifest import ArtifactManifest

logger = logging.getLogger(__name__)


def run_initial_labeling(
    config: 'PipelineConfig',
    manifest: 'ArtifactManifest',
    create_stage_result,
    create_failed_result
) -> 'StageResult':
    """
    Stage 4: Initial Triple-Barrier Labeling.

    Applies initial triple-barrier labeling with default parameters for
    use as input to GA optimization.

    Args:
        config: Pipeline configuration
        manifest: Artifact manifest for tracking outputs
        create_stage_result: Factory function for success results
        create_failed_result: Factory function for failure results

    Returns:
        StageResult with status and artifacts
    """
    start_time = datetime.now()
    logger.info("=" * 70)
    logger.info("STAGE 4: Initial Triple-Barrier Labeling")
    logger.info("=" * 70)

    try:
        # Create labels directory for GA input (run-scoped)
        labels_dir = config.run_data_dir / 'labels'
        labels_dir.mkdir(parents=True, exist_ok=True)

        artifacts = []
        label_stats: Dict[str, Dict[str, Any]] = {}

        target_timeframe = config.target_timeframe

        for symbol in config.symbols:
            # Load features data
            features_path = config.features_dir / f"{symbol}_{target_timeframe}_features.parquet"
            if not features_path.exists():
                raise FileNotFoundError(f"Features file not found: {features_path}")

            logger.info(f"\nProcessing {symbol}...")
            df = pd.read_parquet(features_path)
            logger.info(f"  Loaded {len(df):,} rows")

            # Check for ATR column
            atr_col = 'atr_14'
            if atr_col not in df.columns:
                raise ValueError(f"ATR column '{atr_col}' not found in features")

            symbol_stats = {}

            # Apply initial labeling with default parameters for each horizon
            for horizon in config.label_horizons:
                # Default parameters (will be optimized by GA)
                k_up = 2.0
                k_down = 1.0
                max_bars = horizon * 3

                logger.info(
                    f"  Horizon {horizon}: k_up={k_up}, k_down={k_down}, max_bars={max_bars}"
                )

                labels, bars_to_hit, mae, mfe, touch_type = triple_barrier_numba(
                    df['close'].values,
                    df['high'].values,
                    df['low'].values,
                    df['open'].values,
                    df[atr_col].values,
                    k_up, k_down, max_bars
                )

                # Add columns
                df[f'label_h{horizon}'] = labels
                df[f'bars_to_hit_h{horizon}'] = bars_to_hit
                df[f'mae_h{horizon}'] = mae
                df[f'mfe_h{horizon}'] = mfe

                # Calculate distribution
                n_long = (labels == 1).sum()
                n_short = (labels == -1).sum()
                n_neutral = (labels == 0).sum()
                total = len(labels)

                symbol_stats[horizon] = {
                    'long': int(n_long),
                    'short': int(n_short),
                    'neutral': int(n_neutral),
                    'long_pct': n_long / total * 100,
                    'short_pct': n_short / total * 100,
                    'neutral_pct': n_neutral / total * 100
                }

                logger.info(
                    f"    Distribution: L={n_long/total*100:.1f}% "
                    f"S={n_short/total*100:.1f}% N={n_neutral/total*100:.1f}%"
                )

            # Save to labels directory for GA input
            output_path = labels_dir / f"{symbol}_labels_init.parquet"
            df.to_parquet(output_path, index=False)
            artifacts.append(output_path)

            label_stats[symbol] = symbol_stats

            manifest.add_artifact(
                name=f"initial_labels_{symbol}",
                file_path=output_path,
                stage="initial_labeling",
                metadata={'symbol': symbol, 'horizons': config.label_horizons}
            )

            logger.info(f"  Saved initial labels to {output_path}")

        return create_stage_result(
            stage_name="initial_labeling",
            start_time=start_time,
            artifacts=artifacts,
            metadata={
                'horizons': config.label_horizons,
                'label_stats': label_stats
            }
        )

    except Exception as e:
        logger.error(f"Initial labeling failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return create_failed_result(
            stage_name="initial_labeling",
            start_time=start_time,
            error=str(e)
        )
