"""
DEPRECATED: Import from 'src.core.coordination' instead.

This module will be removed in v2.0.0.

Migration:
    # Old (deprecated)
    from src.coordination import TimeframeCoordinator, align_to_anchor

    # New (preferred)
    from src.core.coordination import TimeframeCoordinator, align_to_anchor
"""

import warnings

_COORDINATION_EXPORTS = [
    # Alignment utilities
    "align_to_anchor",
    "apply_mtf_lag",
    "compute_sequence_offset",
    "validate_timestamp_alignment",
    # TimeframeCoordinator
    "TimeframeCoordinator",
    "TimeframeData",
]


def __getattr__(name: str):
    """Lazy import to avoid circular imports."""
    if name in _COORDINATION_EXPORTS:
        warnings.warn(
            f"Importing '{name}' from 'src.coordination' is deprecated. "
            f"Use 'from src.core.coordination import {name}' instead. "
            "This will be removed in v2.0.0.",
            DeprecationWarning,
            stacklevel=2,
        )
        from src.core import coordination as coordination_module

        return getattr(coordination_module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return _COORDINATION_EXPORTS
