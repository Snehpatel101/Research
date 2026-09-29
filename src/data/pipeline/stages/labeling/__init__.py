"""
Labeling stage (PipelineRunner Stage 4).

Exposes the triple-barrier labeler used by the data pipeline and the numba
kernels shared by the GA optimization and final-labels stages.
"""

from .base import LabelingResult, LabelingStrategy, LabelingType
from .triple_barrier import (
    TripleBarrierLabeler,
    triple_barrier_numba,
    triple_barrier_numba_with_costs,
)

__all__ = [
    "LabelingType",
    "LabelingStrategy",
    "LabelingResult",
    "TripleBarrierLabeler",
    "triple_barrier_numba",
    "triple_barrier_numba_with_costs",
]
