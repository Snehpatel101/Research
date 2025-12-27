"""
Pipeline Package - Phase 1 Data Preparation Pipeline.

This package provides a modular, stage-based pipeline for processing
raw OHLCV data into ML-ready features and labels.

Stage Flow:
1. data_generation    - Generate/validate raw data
2. data_cleaning      - Clean and resample OHLCV data
3. feature_engineering - Generate technical features
4. initial_labeling   - Apply initial triple-barrier labels
5. ga_optimize        - Genetic algorithm optimization of barrier params
6. final_labels       - Apply optimized labels with quality scores
7. create_splits      - Create train/val/test splits
8. validate           - Comprehensive data validation
9. generate_report    - Generate completion report

Usage:
    from src.pipeline.runner import PipelineRunner
    from src.phase1.pipeline_config import create_default_config

    # Single symbol run (recommended - each symbol processed in isolation)
    config = create_default_config(
        symbols=['MES'],  # Specify your target symbol
        start_date='2020-01-01',
        end_date='2024-12-31'
    )

    runner = PipelineRunner(config)
    success = runner.run()
"""
from .utils import StageStatus, StageResult
from .stage_registry import PipelineStage, get_stage_definitions, get_stage_order

__all__ = [
    'PipelineRunner',
    'StageStatus',
    'StageResult',
    'PipelineStage',
    'get_stage_definitions',
    'get_stage_order',
]

__version__ = '1.0.0'


def __getattr__(name: str):
    """Lazy import PipelineRunner to avoid circular dependencies."""
    if name == 'PipelineRunner':
        from .runner import PipelineRunner
        return PipelineRunner
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
