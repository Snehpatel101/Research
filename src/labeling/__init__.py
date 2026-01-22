"""
DEPRECATED: Import from 'src.data.labeling' instead.

This module will be removed in v2.0.0.

Migration:
    # Old (deprecated)
    from src.labeling import TripleBarrierLabeler, LabelOptimizer

    # New (preferred)
    from src.data.labeling import TripleBarrierLabeler, LabelOptimizer
"""

import warnings

_LABELING_EXPORTS = [
    # Base classes
    "LabelingResult",
    "LabelingStrategy",
    "LabelingType",
    # Configuration
    "TripleBarrierConfig",
    # Labeler
    "TripleBarrierLabeler",
    # Numba functions
    "triple_barrier_numba",
    "triple_barrier_numba_with_costs",
    # Optimization
    "LabelOptimizationResult",
    "LabelOptimizer",
    "optimize_labels",
]


def __getattr__(name: str):
    """Lazy import to avoid circular imports."""
    if name in _LABELING_EXPORTS:
        warnings.warn(
            f"Importing '{name}' from 'src.labeling' is deprecated. "
            f"Use 'from src.data.labeling import {name}' instead. "
            "This will be removed in v2.0.0.",
            DeprecationWarning,
            stacklevel=2,
        )
        from src.data import labeling as labeling_module

        return getattr(labeling_module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return _LABELING_EXPORTS
