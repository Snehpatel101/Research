"""
DEPRECATED: This module is deprecated. Use src.core.config instead.

The canonical PipelineConfig is now in src.core.config.

Usage:
    # New way (preferred)
    from src.core.config import PipelineConfig, quick_config, production_config

    # Or via src package
    from src import PipelineConfig, quick_config, production_config

This module re-exports from src.core.config for backwards compatibility.
It will be removed in a future version.
"""

import warnings

warnings.warn(
    "src.pipeline_config is deprecated. Use src.core.config instead. "
    "This module will be removed in v2.0.0.",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export from canonical location for backwards compatibility
from src.core.config import (  # noqa: E402
    PipelineConfig,
    production_config,
    quick_config,
    regime_aware_config,
    research_config,
)

__all__ = [
    "PipelineConfig",
    "quick_config",
    "production_config",
    "research_config",
    "regime_aware_config",
]
