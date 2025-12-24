"""
Ensemble Price Prediction Pipeline

A comprehensive ML pipeline for financial price prediction using ensemble methods
with triple-barrier labeling and genetic algorithm optimization.

Submodules are imported lazily to avoid circular dependencies.
Use explicit imports like:
    from src.phase1.stages import DataIngestor
    from src.pipeline.runner import PipelineRunner
"""

__version__ = "0.1.0"
__author__ = "Research Team"

__all__ = [
    "__version__",
    "__author__",
    "PipelineConfig",
    "create_default_config",
    "PipelineRunner",
    "phase1",
    "common",
]


def __getattr__(name: str):
    """Lazy import to avoid circular dependencies."""
    if name == 'PipelineConfig':
        from src.phase1.pipeline_config import PipelineConfig
        return PipelineConfig
    elif name == 'create_default_config':
        from src.phase1.pipeline_config import create_default_config
        return create_default_config
    elif name == 'PipelineRunner':
        from src.pipeline.runner import PipelineRunner
        return PipelineRunner
    elif name == 'phase1':
        from src import phase1 as _phase1
        return _phase1
    elif name == 'common':
        from src import common as _common
        return _common
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
