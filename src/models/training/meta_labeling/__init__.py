"""
Meta-labeling components for enhanced trading decisions.

Meta-labeling is a technique where a secondary model learns to predict
whether the primary model's predictions will be correct, enabling:
- Filtering out low-confidence trades
- Variable position sizing based on confidence
- Risk-adjusted returns through selective trading

Components:
- bet_sizing: Enhanced bet sizing strategies beyond binary trade/no-trade
"""

from .bet_sizing import (
    BetSizingConfig,
    BetSizingStrategy,
    compute_bet_sizes,
    get_strategy_description,
    predict_with_sizing,
)

__all__ = [
    "BetSizingStrategy",
    "BetSizingConfig",
    "compute_bet_sizes",
    "predict_with_sizing",
    "get_strategy_description",
]
