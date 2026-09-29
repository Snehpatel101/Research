"""
Configuration path constants.

CONFIG-001: This module now re-exports from src.core.paths for backward compatibility.
New code should import directly from src.core.paths.
"""

# Re-export from unified paths module (no deprecation warning to avoid transitive issues)
from src.core.paths import (
    CONFIG_DIR,
    CONFIG_MODELS_DIR,
    CONFIG_ROOT,
)

__all__ = [
    "CONFIG_ROOT",
    "CONFIG_DIR",
    "CONFIG_MODELS_DIR",
]
