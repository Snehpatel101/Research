"""
ML Pipeline - Unified Machine Learning Pipeline.

Usage:
    from src import MLPipeline, PipelineConfig

    config = PipelineConfig(
        symbol="MES",
        models=["xgboost", "lstm"],
        build_ensemble=True,
    )

    result = MLPipeline(config).run()

Presets:
    from src import quick_config, production_config, research_config

    config = quick_config("MES")       # Fast iteration
    config = production_config("MES")  # Full optimization
    config = research_config("MES")    # Many models
"""

__version__ = "1.0.0"
__author__ = "Research Team"

__all__ = [
    "__version__",
    "__author__",
    "MLPipeline",
    "PipelineConfig",
    "PipelineResult",
    "quick_config",
    "production_config",
    "research_config",
]


def __getattr__(name: str):
    """Lazy imports."""
    if name == "MLPipeline":
        # DEPRECATED: Use src.factory.MLFactory instead
        from src.orchestrator import MLPipeline

        return MLPipeline

    if name == "PipelineConfig":
        from src.core.config import PipelineConfig

        return PipelineConfig

    if name == "PipelineResult":
        # DEPRECATED: Use src.factory.ExperimentResult instead
        from src.orchestrator import PipelineResult

        return PipelineResult

    if name == "quick_config":
        from src.core.config import quick_config

        return quick_config

    if name == "production_config":
        from src.core.config import production_config

        return production_config

    if name == "research_config":
        from src.core.config import research_config

        return research_config

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
