"""
DEPRECATED: Import from 'src.data.pipeline' instead.

This module will be removed in v2.0.0.

Migration:
    # Old (deprecated)
    from src.pipeline import PipelineRunner, DataConfig

    # New (preferred)
    from src.data.pipeline import PipelineRunner, DataConfig
"""

import warnings

_PIPELINE_EXPORTS = [
    "PipelineRunner",
    "DataConfig",
    "StageStatus",
    "StageResult",
    "PipelineStage",
    "get_stage_definitions",
    "get_stage_order",
]


def __getattr__(name: str):
    """Lazy import to avoid circular imports."""
    if name in _PIPELINE_EXPORTS:
        warnings.warn(
            f"Importing '{name}' from 'src.pipeline' is deprecated. "
            f"Use 'from src.data.pipeline import {name}' instead. "
            "This will be removed in v2.0.0.",
            DeprecationWarning,
            stacklevel=2,
        )
        from src.data import pipeline as pipeline_module

        return getattr(pipeline_module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return _PIPELINE_EXPORTS
