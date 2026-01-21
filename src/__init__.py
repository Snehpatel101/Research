"""
ML Factory - Unified Machine Learning Pipeline.

A comprehensive ML pipeline for financial price prediction using ensemble methods
with triple-barrier labeling, regime-aware training, and meta-labeling.

THE SINGLE ENTRY POINT:
    from src import MLFactory, PipelineConfig

    config = PipelineConfig(
        symbol="MES",
        data_path="./data/mes.parquet",
        output_dir="./experiments",
        models=["xgboost", "lightgbm", "lstm"],
        training_mode="standard",  # or "regime_aware", "meta_labeling"
        build_ensemble=True,
    )

    factory = MLFactory(config)
    result = factory.run(df)

    # Access results
    print(f"Best model: {result.best_model}")
    bundle = result.get_inference_bundle()

Alternative imports are available for specific phases:
    from src.core import PipelineConfig  # Configuration
    from src.training import UnifiedTrainingOrchestrator  # Training only
    from src.inference import InferenceOrchestrator  # Inference only
    from src.features.compute import compute_all_features  # Features only
"""

__version__ = "0.2.0"
__author__ = "Research Team"

__all__ = [
    "__version__",
    "__author__",
    # THE SINGLE ENTRY POINT
    "MLFactory",
    "PipelineConfig",
    "run_pipeline",
    "quick_run",
    # Convenience functions
    "PipelineResult",
    # Legacy exports (for backward compatibility)
    "create_default_config",
    "PipelineRunner",
    "phase1",
    "common",
]


def __getattr__(name: str):
    """Lazy import to avoid circular dependencies."""
    import importlib

    # NEW: MLFactory - THE single entry point
    if name == "MLFactory":
        from src.factory import MLFactory

        return MLFactory
    elif name == "run_pipeline":
        from src.factory import run_pipeline

        return run_pipeline
    elif name == "quick_run":
        from src.factory import quick_run

        return quick_run
    elif name == "PipelineResult":
        from src.factory import PipelineResult

        return PipelineResult
    # NEW: PipelineConfig from src.core
    elif name == "PipelineConfig":
        from src.core import PipelineConfig

        return PipelineConfig
    # Legacy exports
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
