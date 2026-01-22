"""
ML Pipeline - Unified Machine Learning Pipeline.

THE SIMPLE API (NEW):
    from src import MLPipeline, PipelineConfig

    config = PipelineConfig(
        symbol="MES",
        models=["xgboost", "lstm"],
        build_ensemble=True,
    )

    result = MLPipeline(config).run()

    print(f"Best model: {result.best_model}")
    print(result.summary())

That's it. ONE config. ONE orchestrator. EVERYTHING runs.

PRESET CONFIGURATIONS:
    from src import quick_config, production_config, research_config

    # Fast iteration (no optimization)
    config = quick_config("MES")

    # Full production (all optimizations)
    config = production_config("MES")

    # Research mode (many models)
    config = research_config("MES")

LEGACY API (deprecated):
    from src import MLFactory  # Use MLPipeline instead
"""

__version__ = "1.0.0"
__author__ = "Research Team"

__all__ = [
    "__version__",
    "__author__",
    # =========================================================================
    # THE SIMPLE API (use these)
    # =========================================================================
    "MLPipeline",           # THE orchestrator
    "PipelineConfig",       # THE config
    "PipelineResult",       # Result from pipeline.run()
    # Preset configs
    "quick_config",
    "production_config",
    "research_config",
    # =========================================================================
    # LEGACY (deprecated, for backward compatibility)
    # =========================================================================
    "MLFactory",            # Deprecated: Use MLPipeline
    "run_pipeline",         # Deprecated: Use MLPipeline(config).run()
    "quick_run",            # Deprecated: Use quick_config + MLPipeline
    "create_default_config",
    "PipelineRunner",
    "phase1",
    "common",
]


def __getattr__(name: str):
    """Lazy import to avoid circular dependencies."""
    import importlib
    import warnings

    # =========================================================================
    # THE SIMPLE API
    # =========================================================================

    if name == "MLPipeline":
        from src.orchestrator import MLPipeline
        return MLPipeline

    elif name == "PipelineConfig":
        from src.pipeline_config import PipelineConfig
        return PipelineConfig

    elif name == "PipelineResult":
        from src.orchestrator import PipelineResult
        return PipelineResult

    elif name == "quick_config":
        from src.pipeline_config import quick_config
        return quick_config

    elif name == "production_config":
        from src.pipeline_config import production_config
        return production_config

    elif name == "research_config":
        from src.pipeline_config import research_config
        return research_config

    # =========================================================================
    # LEGACY (deprecated)
    # =========================================================================

    elif name == "MLFactory":
        warnings.warn(
            "MLFactory is deprecated. Use MLPipeline instead:\n"
            "  from src import MLPipeline, PipelineConfig\n"
            "  result = MLPipeline(PipelineConfig(symbol='MES')).run()",
            DeprecationWarning,
            stacklevel=2,
        )
        from src.factory import MLFactory
        return MLFactory

    elif name == "run_pipeline":
        warnings.warn(
            "run_pipeline is deprecated. Use MLPipeline instead:\n"
            "  result = MLPipeline(config).run()",
            DeprecationWarning,
            stacklevel=2,
        )
        from src.factory import run_pipeline
        return run_pipeline

    elif name == "quick_run":
        warnings.warn(
            "quick_run is deprecated. Use quick_config + MLPipeline instead:\n"
            "  from src import quick_config, MLPipeline\n"
            "  result = MLPipeline(quick_config('MES')).run()",
            DeprecationWarning,
            stacklevel=2,
        )
        from src.factory import quick_run
        return quick_run

    elif name == "create_default_config":
        from src.pipeline.config.pipeline_defaults import create_default_config
        return create_default_config

    elif name == "PipelineRunner":
        from src.pipeline.runner import PipelineRunner
        return PipelineRunner

    elif name == "phase1":
        return importlib.import_module("src.pipeline")

    elif name == "common":
        return importlib.import_module("src.common")

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
