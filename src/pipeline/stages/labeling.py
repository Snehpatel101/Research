"""
Stage 4 & 6: Labeling Stages.

Contains both initial triple-barrier labeling and final optimized labeling.
"""
import json
import logging
import traceback
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Any

import pandas as pd

from ..utils import StageResult, StageStatus, create_stage_result, create_failed_result

if TYPE_CHECKING:
    from pipeline_config import PipelineConfig
    from manifest import ArtifactManifest

logger = logging.getLogger(__name__)


def run_initial_labeling(
    config: 'PipelineConfig',
    manifest: 'ArtifactManifest'
) -> StageResult:
    """
    Stage 4: Initial Triple-Barrier Labeling.

    Applies initial triple-barrier labeling with default parameters for
    use as input to GA optimization.

    Args:
        config: Pipeline configuration
        manifest: Artifact manifest for tracking outputs

    Returns:
        StageResult with status and artifacts
    """
    start_time = datetime.now()
    logger.info("=" * 70)
    logger.info("STAGE 4: Initial Triple-Barrier Labeling")
    logger.info("=" * 70)

    try:
        from stages.stage4_labeling import triple_barrier_numba

        # Create labels directory for GA input
        labels_dir = config.project_root / 'data' / 'labels'
        labels_dir.mkdir(parents=True, exist_ok=True)

        artifacts = []
        label_stats: Dict[str, Dict[str, Any]] = {}

        for symbol in config.symbols:
            # Load features data
            features_path = config.features_dir / f"{symbol}_5m_features.parquet"
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
        logger.error(traceback.format_exc())
        return create_failed_result(
            stage_name="initial_labeling",
            start_time=start_time,
            error=str(e)
        )


def run_final_labels(
    config: 'PipelineConfig',
    manifest: 'ArtifactManifest'
) -> StageResult:
    """
    Stage 6: Apply optimized labels with quality scores and sample weights.

    Uses GA-optimized parameters to apply final triple-barrier labels.

    Args:
        config: Pipeline configuration
        manifest: Artifact manifest for tracking outputs

    Returns:
        StageResult with status and artifacts
    """
    start_time = datetime.now()
    logger.info("=" * 70)
    logger.info("STAGE 6: Final Labels with Quality Scores")
    logger.info("=" * 70)

    try:
        from stages.stage6_final_labels import apply_optimized_labels, generate_labeling_report

        # GA results directory
        ga_results_dir = config.project_root / 'config' / 'ga_results'

        artifacts = []
        all_dfs = {}

        for symbol in config.symbols:
            logger.info(f"\n{'='*50}")
            logger.info(f"Processing {symbol}")
            logger.info(f"{'='*50}")

            # Load features data (original, without initial labels)
            features_path = config.features_dir / f"{symbol}_5m_features.parquet"
            if not features_path.exists():
                raise FileNotFoundError(f"Features file not found: {features_path}")

            df = pd.read_parquet(features_path)
            logger.info(f"Loaded {len(df):,} rows from features")

            # Apply optimized labels for each horizon
            for horizon in config.label_horizons:
                results_path = ga_results_dir / f"{symbol}_ga_h{horizon}_best.json"

                if results_path.exists():
                    with open(results_path, 'r') as f:
                        results = json.load(f)
                    best_params = {
                        'k_up': results['best_k_up'],
                        'k_down': results['best_k_down'],
                        'max_bars': results['best_max_bars']
                    }
                    logger.info(f"\n  Horizon {horizon}: Using GA-optimized params")
                    logger.info(
                        f"    k_up={best_params['k_up']:.3f}, "
                        f"k_down={best_params['k_down']:.3f}, "
                        f"max_bars={best_params['max_bars']}"
                    )
                else:
                    # Fall back to defaults if no GA results
                    logger.warning(
                        f"  Horizon {horizon}: No GA results found, using defaults"
                    )
                    best_params = {
                        'k_up': 2.0,
                        'k_down': 1.0,
                        'max_bars': horizon * 3
                    }

                # Apply optimized labeling with quality scores
                df = apply_optimized_labels(df, horizon, best_params, atr_column='atr_14')

            # Save final labeled data
            output_path = config.final_data_dir / f"{symbol}_labeled.parquet"
            df.to_parquet(output_path, index=False)
            artifacts.append(output_path)

            manifest.add_artifact(
                name=f"final_labeled_{symbol}",
                file_path=output_path,
                stage="final_labels",
                metadata={'symbol': symbol, 'horizons': config.label_horizons}
            )

            all_dfs[symbol] = df
            logger.info(f"\n  Saved final labels to {output_path}")

        # Generate labeling report
        if all_dfs:
            generate_labeling_report(all_dfs)
            report_path = config.results_dir / 'labeling_report.md'
            if report_path.exists():
                artifacts.append(report_path)

        return create_stage_result(
            stage_name="final_labels",
            start_time=start_time,
            artifacts=artifacts,
            metadata={
                'symbols': config.symbols,
                'horizons': config.label_horizons
            }
        )

    except Exception as e:
        logger.error(f"Final labeling failed: {e}")
        logger.error(traceback.format_exc())
        return create_failed_result(
            stage_name="final_labels",
            start_time=start_time,
            error=str(e)
        )
