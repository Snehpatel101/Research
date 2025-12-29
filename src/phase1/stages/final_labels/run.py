"""
Stage 6: Final Labels - Pipeline Wrapper.

This module provides the pipeline integration for final labeling,
extracting the orchestration logic from pipeline/stages/labeling.py.
"""
import json
import logging
import subprocess
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd

from .core import apply_optimized_labels, generate_labeling_report

if TYPE_CHECKING:
    from manifest import ArtifactManifest
    from pipeline_config import PipelineConfig

logger = logging.getLogger(__name__)


def _get_git_commit_hash(project_root: Path) -> str | None:
    """Get current git commit hash."""
    git_dir = project_root / ".git"
    if not git_dir.exists():
        return None
    try:
        result = subprocess.run(
            ["git", "-C", str(project_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as e:
        logger.debug(f"Git command failed: {e}")
        return None
    except Exception as e:
        logger.debug(f"Could not get git commit hash: {e}")
        return None
    return result.stdout.strip() or None


def run_final_labels(
    config: 'PipelineConfig',
    manifest: 'ArtifactManifest',
    create_stage_result,
    create_failed_result
) -> 'StageResult':
    """
    Stage 6: Apply optimized labels with quality scores and sample weights.

    Uses GA-optimized parameters to apply final triple-barrier labels.

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
    logger.info("STAGE 6: Final Labels with Quality Scores")
    logger.info("=" * 70)

    try:
        from src.phase1.config import TRANSACTION_COSTS
        try:
            from src import __version__ as src_version
        except ImportError:
            logger.debug("src.__version__ not available")
            src_version = None

        # GA results directory (run-scoped for reproducibility)
        ga_results_dir = config.run_artifacts_dir / 'ga_results'

        artifacts = []
        all_dfs = {}
        label_params: dict[str, dict[str, dict[str, Any]]] = {}

        target_timeframe = config.target_timeframe

        for symbol in config.symbols:
            logger.info(f"\n{'='*50}")
            logger.info(f"Processing {symbol}")
            logger.info(f"{'='*50}")

            # Load features data (original, without initial labels)
            features_path = config.features_dir / f"{symbol}_{target_timeframe}_features.parquet"
            if not features_path.exists():
                raise FileNotFoundError(f"Features file not found: {features_path}")

            df = pd.read_parquet(features_path)
            logger.info(f"Loaded {len(df):,} rows from features")

            symbol_params: dict[str, dict[str, Any]] = {}

            # Apply optimized labels for each horizon
            for horizon in config.label_horizons:
                results_path = ga_results_dir / f"{symbol}_ga_h{horizon}_best.json"

                if results_path.exists():
                    with open(results_path) as f:
                        results = json.load(f)
                    best_params = {
                        'k_up': results['best_k_up'],
                        'k_down': results['best_k_down'],
                        'max_bars': results['best_max_bars']
                    }
                    symbol_params[str(horizon)] = {
                        'k_up': best_params['k_up'],
                        'k_down': best_params['k_down'],
                        'max_bars': best_params['max_bars'],
                        'source': 'ga',
                        'ga_results_path': str(results_path),
                        'best_fitness': results.get('best_fitness'),
                        'population_size': results.get('population_size'),
                        'generations': results.get('generations'),
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
                    symbol_params[str(horizon)] = {
                        'k_up': best_params['k_up'],
                        'k_down': best_params['k_down'],
                        'max_bars': best_params['max_bars'],
                        'source': 'default',
                        'ga_results_path': None,
                    }

                # Apply optimized labeling with quality scores
                df = apply_optimized_labels(df, horizon, best_params, symbol=symbol, atr_column='atr_14')

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
            label_params[symbol] = symbol_params
            logger.info(f"\n  Saved final labels to {output_path}")

        # Create label manifest
        label_manifest_path = config.run_artifacts_dir / "label_manifest.json"
        label_manifest_path.parent.mkdir(parents=True, exist_ok=True)
        label_manifest = {
            'run_id': config.run_id,
            'created_at': datetime.now().isoformat(),
            'labeling_stage': 'final_labels',
            'labeling_strategy': 'triple_barrier',
            'target_timeframe': config.target_timeframe,
            'symbols': config.symbols,
            'horizons': config.label_horizons,
            'transaction_costs': {
                symbol: TRANSACTION_COSTS.get(symbol)
                for symbol in config.symbols
            },
            'ga_results_dir': str(ga_results_dir),
            'barrier_params': label_params,
            'label_columns': {
                'label': 'label_h{h}',
                'bars_to_hit': 'bars_to_hit_h{h}',
                'mae': 'mae_h{h}',
                'mfe': 'mfe_h{h}',
                'touch_type': 'touch_type_h{h}',
                'quality': 'quality_h{h}',
                'sample_weight': 'sample_weight_h{h}',
                'pain_to_gain': 'pain_to_gain_h{h}',
                'time_weighted_dd': 'time_weighted_dd_h{h}',
                'fwd_return': 'fwd_return_h{h}',
                'fwd_return_log': 'fwd_return_log_h{h}',
            },
            'forward_returns': {
                'method': 'close[t+h] / close[t] - 1',
                'log_method': 'log(close[t+h] / close[t])',
                'tail_nan_bars': 'horizon',
            },
            'code_version': {
                'src_version': src_version,
                'git_commit': _get_git_commit_hash(config.project_root),
            },
        }

        with open(label_manifest_path, 'w') as f:
            json.dump(label_manifest, f, indent=2)
        artifacts.append(label_manifest_path)
        manifest.add_artifact(
            name="label_manifest",
            file_path=label_manifest_path,
            stage="final_labels",
            metadata={'symbols': config.symbols, 'horizons': config.label_horizons}
        )

        # Generate labeling report (run-scoped)
        if all_dfs:
            report_path = generate_labeling_report(
                all_dfs,
                config.run_artifacts_dir,
                config.label_horizons,
            )
            if report_path.exists():
                artifacts.append(report_path)

        return create_stage_result(
            stage_name="final_labels",
            start_time=start_time,
            artifacts=artifacts,
            metadata={
                'symbols': config.symbols,
                'horizons': config.label_horizons,
                'label_manifest_path': str(label_manifest_path)
            }
        )

    except Exception as e:
        logger.error(f"Final labeling failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return create_failed_result(
            stage_name="final_labels",
            start_time=start_time,
            error=str(e)
        )
