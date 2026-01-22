"""
DEPRECATED: Import from 'src.validation.evaluation' instead.

This module will be removed in v2.0.0.

Migration:
    # Old (deprecated)
    from src.evaluation import CVEvaluator

    # New (preferred)
    from src.validation.evaluation import CVEvaluator
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

# Names that will be lazily loaded from src.validation.evaluation
_EVALUATION_EXPORTS = [
    "CVEvaluator",
    "WalkForwardEvaluator",
    "CPCVPBOEvaluator",
]


def __getattr__(name: str):
    """Lazy import to avoid circular imports."""
    if name in _EVALUATION_EXPORTS:
        # Show deprecation warning
        warnings.warn(
            f"Importing '{name}' from 'src.evaluation' is deprecated. "
            f"Use 'from src.validation.evaluation import {name}' instead. "
            "This will be removed in v2.0.0.",
            DeprecationWarning,
            stacklevel=2,
        )
        # Import from new location
        from src.validation import evaluation as eval_module
        return getattr(eval_module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    """Return list of available names."""
    return _EVALUATION_EXPORTS


__all__ = _EVALUATION_EXPORTS
