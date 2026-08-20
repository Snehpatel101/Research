"""
Feature-name resolution and event-driven feature utilities.

The live feature ENGINE is src/data/pipeline/stages/features/ — this package
holds only:
- strategies / strategy_manager: per-model feature-set resolution (used by
  the trainer and TrainerConfig)
- cusum_filter: symmetric CUSUM event filter (Phase 99)
- frac_diff: fractional differentiation (Phase 99)
"""

from .strategies import (
    MODEL_FEATURE_STRATEGIES,
    ModelFeatureStrategy,
    get_baseline_features,
    get_strategy_for_model,
)
from .strategy_manager import (
    FeatureStrategyManager,
    ResolvedFeatureSet,
    get_features_for_model,
)

__all__ = [
    # Strategies
    "ModelFeatureStrategy",
    "MODEL_FEATURE_STRATEGIES",
    "get_strategy_for_model",
    "get_baseline_features",
    # Strategy Manager
    "ResolvedFeatureSet",
    "FeatureStrategyManager",
    "get_features_for_model",
]
