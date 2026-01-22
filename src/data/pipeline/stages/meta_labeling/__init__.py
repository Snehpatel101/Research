"""
Meta-Labeling Stage for Bet Sizing.

This module implements meta-labeling as described by Lopez de Prado in "Advances in
Financial Machine Learning" (Chapter 3) and expanded by Hudson & Thames.

Meta-labeling enables a two-model approach for trading systems:
1. Primary Model: High-recall direction classifier (predicts side: -1 or 1)
2. Meta Model: Confidence-based bet sizing (predicts whether to act: 0 or 1)

The separation of "what to trade" from "whether to trade" allows each model
to be optimized independently, improving overall system performance.

Workflow
--------
Triple-Barrier Label: {-1, 0, 1} (direction)
            |
Primary Model (high recall): Predicts side {-1, 1}
            |
Meta-Label: {0, 1} (whether primary prediction was correct)
            |
Meta-Model: Predicts probability of primary being correct
            |
Bet Size: f(meta_probability, volatility, account_risk)

Key Classes
-----------
- PrimaryClassifier: High-recall classifier for direction prediction
- MetaLabelGenerator: Generates meta-labels from primary predictions
- BetSizer: Converts meta-model confidence to position sizes
- MetaLabelingConfig: Configuration for the meta-labeling pipeline

Usage
-----
>>> from src.data.pipeline.stages.meta_labeling import (
...     PrimaryClassifier,
...     MetaLabelGenerator,
...     BetSizer,
...     MetaLabelingConfig,
... )
>>>
>>> # Train primary model with high recall
>>> primary = PrimaryClassifier(recall_threshold=0.7)
>>> primary.fit(X_train, y_train)
>>>
>>> # Generate meta-labels
>>> meta_gen = MetaLabelGenerator()
>>> meta_labels = meta_gen.generate(y_true=y_val, y_primary=primary.predict_side(X_val))
>>>
>>> # Size positions based on meta-model confidence
>>> sizer = BetSizer(max_position=10, min_confidence=0.55)
>>> positions = sizer.get_position_sizes(meta_probabilities, volatility)

Engineering Rules Applied
-------------------------
- Clear contracts with type hints
- Input validation at boundaries
- Fail fast with explicit errors
- Modular design with single responsibility
"""

from .bet_sizer import (
    BetSizer,
    BetSizingMethod,
    KellyCriterion,
    VolatilityScaler,
)
from .meta_labeler import (
    MetaLabelGenerator,
    MetaLabelingConfig,
    MetaLabelResult,
)
from .primary_model import (
    PrimaryClassifier,
    PrimaryModelConfig,
    RecallOptimizer,
)
from .run import add_meta_labels_standalone, run_meta_labeling

__all__ = [
    # Primary model
    "PrimaryClassifier",
    "PrimaryModelConfig",
    "RecallOptimizer",
    # Meta-labeling
    "MetaLabelGenerator",
    "MetaLabelingConfig",
    "MetaLabelResult",
    # Bet sizing
    "BetSizer",
    "BetSizingMethod",
    "KellyCriterion",
    "VolatilityScaler",
    # Pipeline runner
    "run_meta_labeling",
    "add_meta_labels_standalone",
]
