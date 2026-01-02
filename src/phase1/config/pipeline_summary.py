"""Summary generation for PipelineConfig."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.phase1.pipeline_config import PipelineConfig


def generate_pipeline_summary(config: "PipelineConfig") -> str:
    """Generate a human-readable summary of the configuration."""
    return f"""
Pipeline Configuration Summary
==============================
Run ID: {config.run_id}
Description: {config.description}

Data Parameters:
  - Symbols: {', '.join(config.symbols)}
  - Symbol Isolation: Each symbol processed independently (no cross-symbol operations)
  - Batch Processing: {'Enabled' if config.allow_batch_symbols else 'Single symbol only'}
  - Date Range: {config.start_date or 'N/A'} to {config.end_date or 'N/A'}
  - Target Timeframe: {config.target_timeframe}

Features:
  - Feature Set: {config.feature_set}
  - SMA Periods: {config.sma_periods}
  - EMA Periods: {config.ema_periods}
  - ATR Periods: {config.atr_periods}
  - RSI Period: {config.rsi_period}

Multi-Timeframe (MTF):
  - Timeframes: {', '.join(config.mtf_timeframes)}
  - Mode: {config.mtf_mode}

Labeling:
  - Horizons: {config.label_horizons}
  - Barrier Params: config.BARRIER_PARAMS (symbol-specific)
  - Max Bars Ahead: {config.max_bars_ahead}

Splits:
  - Train: {config.train_ratio:.1%}
  - Validation: {config.val_ratio:.1%}
  - Test: {config.test_ratio:.1%}
  - Purge Bars: {config.purge_bars}
  - Embargo Bars: {config.embargo_bars}

GA Settings (Phase 2):
  - Population: {config.ga_population_size}
  - Generations: {config.ga_generations}
  - Crossover Rate: {config.ga_crossover_rate}
  - Mutation Rate: {config.ga_mutation_rate}
  - Elite Size: {config.ga_elite_size}

Scaling:
  - Scaler Type: {config.scaler_type}

Feature Toggles: {config.feature_toggles or 'All enabled (default)'}
Barrier Overrides: {config.barrier_overrides or 'Using symbol-specific defaults'}
Model Config: {config.model_config or 'Not specified'}

Paths:
  - Project Root: {config.project_root}
  - Run Directory: {config.run_dir}
  - Data Directory: {config.data_dir}
"""
