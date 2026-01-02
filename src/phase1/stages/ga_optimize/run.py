"""
Stage 5: Barrier Optimization - Pipeline Wrapper.

This module provides the pipeline integration for barrier optimization
using Optuna TPE (Tree-structured Parzen Estimator).

Note: Function names retain "ga" prefix for backward compatibility,
but internally use Optuna TPE which is 27% more sample-efficient.
"""

import json
import logging
from datetime import datetime
from typing import TYPE_CHECKING, Any

import pandas as pd

from .optimization import run_ga_optimization, run_ga_optimization_safe
from .plotting import plot_convergence

if TYPE_CHECKING:
    from manifest import ArtifactManifest
    from pipeline_config import PipelineConfig

logger = logging.getLogger(__name__)


def run_ga_optimization_stage(
    config: "PipelineConfig",
    manifest: "ArtifactManifest",
    create_stage_result,
    create_failed_result,
) -> "StageResult":
    """
    Stage 5: Barrier Optimization using Optuna TPE.

    Optimizes barrier parameters (k_up, k_down, max_bars) for each symbol
    and horizon using Optuna's Tree-structured Parzen Estimator (TPE).

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
    logger.info("STAGE 5: Barrier Optimization (Optuna TPE)")
    logger.info("=" * 70)

    try:
        # GA results directory (run-scoped for reproducibility)
        ga_results_dir = config.run_artifacts_dir / "ga_results"
        ga_results_dir.mkdir(parents=True, exist_ok=True)

        # Plots directory (run-scoped)
        plots_dir = config.run_artifacts_dir / "ga_plots"
        plots_dir.mkdir(parents=True, exist_ok=True)

        artifacts = []
        all_results: dict[str, dict[int, dict[str, Any]]] = {}

        # GA configuration
        population_size = config.ga_population_size
        generations = config.ga_generations

        logger.info(
            f"Optuna Config: n_trials~{min(population_size * 2, 150)} (mapped from pop={population_size})"
        )
        logger.info(f"Horizons: {config.label_horizons}")

        for symbol in config.symbols:
            logger.info(f"\n{'='*50}")
            logger.info(f"Optimizing {symbol}")
            logger.info(f"{'='*50}")

            # Load labels data (run-scoped)
            labels_dir = config.run_data_dir / "labels"
            labels_path = labels_dir / f"{symbol}_labels_init.parquet"

            if not labels_path.exists():
                raise FileNotFoundError(f"Labels file not found: {labels_path}")

            df = pd.read_parquet(labels_path)
            logger.info(f"Loaded {len(df):,} rows")

            symbol_results: dict[int, dict[str, Any]] = {}

            for horizon in config.label_horizons:
                # Check if already optimized (skip if results exist AND are valid)
                results_path = ga_results_dir / f"{symbol}_ga_h{horizon}_best.json"

                # Minimum fitness threshold - results below this are INVALID
                # The fitness function returns -10000 for hard constraint violations
                # Valid fitness range is approximately -10 to +16, so -100 catches only catastrophic failures
                MIN_VALID_FITNESS = -100.0

                if results_path.exists():
                    with open(results_path) as f:
                        cached_results = json.load(f)

                    cached_fitness = cached_results.get("best_fitness", float("-inf"))

                    # CRITICAL: Reject cached results with negative fitness
                    # Negative fitness indicates constraint violations (e.g., <10% neutral)
                    if cached_fitness >= MIN_VALID_FITNESS:
                        logger.info(
                            f"\n  Horizon {horizon}: Loading existing VALID results from {results_path.name}"
                        )
                        results = cached_results
                        symbol_results[horizon] = results
                        artifacts.append(results_path)
                        logger.info(
                            f"    k_up={results['best_k_up']:.3f}, "
                            f"k_down={results['best_k_down']:.3f}, "
                            f"max_bars={results['best_max_bars']}, "
                            f"fitness={results['best_fitness']:.4f}"
                        )
                        continue
                    else:
                        # Cached results are INVALID - re-run optimization
                        logger.warning(
                            f"\n  Horizon {horizon}: Cached results INVALID (fitness={cached_fitness:.2f} < {MIN_VALID_FITNESS})"
                        )
                        logger.warning(
                            "    Negative fitness indicates constraint violation (e.g., <10% neutral)"
                        )
                        logger.warning("    Re-running optimization with stricter constraints...")
                        # Remove invalid cached file
                        results_path.unlink()

                # Run Optuna TPE optimization
                # Check if safe mode is enabled (prevents test data leakage)
                safe_mode = getattr(config, "ga_safe_mode", True)  # Default to safe

                if safe_mode:
                    logger.info(f"\n  Horizon {horizon}: Running SAFE Optuna TPE optimization...")
                    logger.info(
                        "    (Safe mode: using only training portion to prevent test data leakage)"
                    )
                    results, logbook = run_ga_optimization_safe(
                        df,
                        horizon,
                        symbol=symbol,
                        train_ratio=config.train_ratio,  # Use config's train ratio
                        population_size=population_size,
                        generations=min(generations, 30),  # Cap at 30 for performance
                        subset_fraction=0.3,
                        atr_column="atr_14",
                        seed=config.random_seed,
                    )
                else:
                    logger.warning(
                        f"\n  Horizon {horizon}: Running UNSAFE Optuna TPE optimization..."
                    )
                    logger.warning(
                        "    WARNING: Safe mode disabled - test data may influence optimization!"
                    )
                    logger.warning(
                        "    Only use this for research when you understand the implications."
                    )
                    results, logbook = run_ga_optimization(
                        df,
                        horizon,
                        symbol=symbol,
                        population_size=population_size,
                        generations=min(generations, 30),  # Cap at 30 for performance
                        subset_fraction=0.3,
                        atr_column="atr_14",
                        seed=config.random_seed,
                    )

                # CRITICAL: Validate new results before saving
                new_fitness = results.get("best_fitness", float("-inf"))
                if new_fitness < MIN_VALID_FITNESS:
                    # Optimization FAILED to find valid parameters
                    # Fall back to symbol-specific defaults from barriers_config.py
                    logger.error(
                        f"\n  Horizon {horizon}: Optimization FAILED (fitness={new_fitness:.2f})"
                    )
                    logger.error(
                        "    Could not find parameters satisfying constraints (neutral >= 10%, etc.)"
                    )
                    logger.warning("    Falling back to default barrier parameters...")

                    # Import defaults
                    from src.phase1.config import get_barrier_params

                    default_params = get_barrier_params(symbol, horizon)

                    results = {
                        "horizon": horizon,
                        "best_k_up": default_params["k_up"],
                        "best_k_down": default_params["k_down"],
                        "best_max_bars": default_params["max_bars"],
                        "best_fitness": 0.0,  # Indicate not optimized
                        "optimizer": "default_fallback",
                        "n_trials": 0,
                        "warning": "Optimization failed - using default parameters",
                        "original_fitness": new_fitness,
                    }
                    logger.warning(
                        f"    Using defaults: k_up={results['best_k_up']:.3f}, "
                        f"k_down={results['best_k_down']:.3f}, "
                        f"max_bars={results['best_max_bars']}"
                    )

                symbol_results[horizon] = results

                # Save results
                with open(results_path, "w") as f:
                    json.dump(results, f, indent=2)
                artifacts.append(results_path)

                logger.info(
                    f"    Best: k_up={results['best_k_up']:.3f}, "
                    f"k_down={results['best_k_down']:.3f}, "
                    f"max_bars={results['best_max_bars']}, "
                    f"fitness={results['best_fitness']:.4f}"
                )

                # Check signal rate
                val = results.get("validation", {})
                signal_rate = val.get("signal_rate", 0)
                if signal_rate < 0.40:
                    logger.warning(
                        f"    WARNING: Signal rate {signal_rate*100:.1f}% below 40% threshold!"
                    )

                # Plot convergence
                plot_path = plots_dir / f"{symbol}_ga_h{horizon}_convergence.png"
                try:
                    plot_convergence(results, plot_path)
                    artifacts.append(plot_path)
                except Exception as plot_err:
                    logger.warning(f"Failed to generate convergence plot: {plot_err}")

                manifest.add_artifact(
                    name=f"ga_results_{symbol}_h{horizon}",
                    file_path=results_path,
                    stage="ga_optimize",
                    metadata={
                        "symbol": symbol,
                        "horizon": horizon,
                        "best_fitness": results["best_fitness"],
                        "signal_rate": signal_rate,
                    },
                )

            all_results[symbol] = symbol_results

        # Save combined summary
        summary: dict[str, dict[str, dict[str, Any]]] = {}
        for symbol, symbol_results in all_results.items():
            summary[symbol] = {
                str(h): {
                    "k_up": res["best_k_up"],
                    "k_down": res["best_k_down"],
                    "max_bars": res["best_max_bars"],
                    "fitness": res["best_fitness"],
                    "signal_rate": res.get("validation", {}).get("signal_rate", None),
                }
                for h, res in symbol_results.items()
            }

        summary_path = ga_results_dir / "optimization_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        artifacts.append(summary_path)

        logger.info(f"\nOptimization summary saved to {summary_path}")

        return create_stage_result(
            stage_name="ga_optimize",
            start_time=start_time,
            artifacts=artifacts,
            metadata={"all_results": summary},
        )

    except Exception as e:
        logger.error(f"Barrier optimization failed: {e}")
        import traceback

        logger.error(traceback.format_exc())
        return create_failed_result(stage_name="ga_optimize", start_time=start_time, error=str(e))
