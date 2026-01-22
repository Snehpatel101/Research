"""
Enhanced bet sizing strategies for meta-labeling.

Goes beyond binary trade/no-trade to variable position sizing
based on model confidence and risk management principles.
"""

from __future__ import annotations

from enum import Enum
from dataclasses import dataclass
from typing import Tuple
import logging

import numpy as np

logger = logging.getLogger(__name__)


class BetSizingStrategy(Enum):
    """Bet sizing strategies for meta-labeling."""
    BINARY = "binary"              # Current: trade or no-trade
    PROPORTIONAL = "proportional"  # Size proportional to probability
    KELLY = "kelly"                # Kelly Criterion
    HALF_KELLY = "half_kelly"      # Half Kelly (more conservative)
    CONFIDENCE = "confidence"      # Based on prediction confidence


@dataclass
class BetSizingConfig:
    """Configuration for bet sizing."""
    strategy: BetSizingStrategy = BetSizingStrategy.BINARY
    threshold: float = 0.5         # Minimum probability to trade
    max_size: float = 1.0          # Maximum position size (fraction of capital)
    min_size: float = 0.0          # Minimum position size (if trading)
    kelly_fraction: float = 0.5    # Fraction of Kelly to use


def compute_bet_sizes(
    probabilities: np.ndarray,
    strategy: BetSizingStrategy = BetSizingStrategy.BINARY,
    threshold: float = 0.5,
    max_size: float = 1.0,
    min_size: float = 0.0,
    kelly_fraction: float = 0.5,
) -> np.ndarray:
    """
    Compute position sizes based on meta-model probabilities.

    Args:
        probabilities: P(correct) from meta-model, shape (n_samples,), range [0, 1]
        strategy: Bet sizing strategy to use
        threshold: Minimum probability to trade
        max_size: Maximum position size (as fraction of capital)
        min_size: Minimum position size if trading
        kelly_fraction: Fraction of full Kelly to use (for KELLY strategy)

    Returns:
        Position sizes, shape (n_samples,), range [0, max_size]
    """
    probabilities = np.asarray(probabilities)

    if strategy == BetSizingStrategy.BINARY:
        # Current approach: max_size if prob > threshold, else 0.0
        sizes = np.where(probabilities > threshold, max_size, 0.0)

    elif strategy == BetSizingStrategy.PROPORTIONAL:
        # Size proportional to confidence above threshold
        # Map [threshold, 1.0] -> [min_size, max_size]
        above_threshold = probabilities > threshold
        sizes = np.zeros_like(probabilities)
        if np.any(above_threshold):
            normalized = (probabilities[above_threshold] - threshold) / (1.0 - threshold)
            sizes[above_threshold] = min_size + normalized * (max_size - min_size)

    elif strategy == BetSizingStrategy.KELLY:
        # Kelly Criterion: f* = (p * (b + 1) - 1) / b
        # Assuming even odds (b = 1): f* = 2p - 1
        kelly_fractions = 2 * probabilities - 1
        kelly_fractions = np.clip(kelly_fractions, 0, 1)
        sizes = kelly_fractions * max_size * kelly_fraction
        sizes = np.where(probabilities > threshold, sizes, 0.0)

    elif strategy == BetSizingStrategy.HALF_KELLY:
        # Half Kelly is more conservative
        kelly_fractions = 2 * probabilities - 1
        kelly_fractions = np.clip(kelly_fractions, 0, 1)
        sizes = kelly_fractions * max_size * 0.5
        sizes = np.where(probabilities > threshold, sizes, 0.0)

    elif strategy == BetSizingStrategy.CONFIDENCE:
        # Square of excess probability (more aggressive scaling)
        above_threshold = probabilities > threshold
        sizes = np.zeros_like(probabilities)
        if np.any(above_threshold):
            excess = (probabilities[above_threshold] - threshold) / (1.0 - threshold)
            sizes[above_threshold] = min_size + (excess ** 2) * (max_size - min_size)

    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    # Ensure within bounds
    sizes = np.clip(sizes, 0, max_size)

    return sizes


def predict_with_sizing(
    primary_predictions: np.ndarray,
    meta_probabilities: np.ndarray,
    config: BetSizingConfig | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Combine primary model predictions with meta-model bet sizing.

    Args:
        primary_predictions: Direction predictions from primary model (-1, 0, +1)
        meta_probabilities: P(correct) from meta-model [0, 1]
        config: Bet sizing configuration

    Returns:
        Tuple of:
        - directions: Final directions (-1, 0, +1)
        - sizes: Position sizes [0, max_size]
    """
    if config is None:
        config = BetSizingConfig()

    # Compute bet sizes
    sizes = compute_bet_sizes(
        meta_probabilities,
        strategy=config.strategy,
        threshold=config.threshold,
        max_size=config.max_size,
        min_size=config.min_size,
        kelly_fraction=config.kelly_fraction,
    )

    # Apply sizes to directions
    # If size = 0, direction becomes 0 (no trade)
    directions = np.where(sizes > 0, primary_predictions, 0)

    return directions.astype(int), sizes


def get_strategy_description(strategy: BetSizingStrategy) -> str:
    """Get human-readable description of a strategy."""
    descriptions = {
        BetSizingStrategy.BINARY: "Binary: Trade at full size or don't trade",
        BetSizingStrategy.PROPORTIONAL: "Proportional: Size scales linearly with confidence",
        BetSizingStrategy.KELLY: "Kelly: Optimal growth rate (can be aggressive)",
        BetSizingStrategy.HALF_KELLY: "Half Kelly: Conservative Kelly variant",
        BetSizingStrategy.CONFIDENCE: "Confidence: Size scales with squared confidence",
    }
    return descriptions.get(strategy, str(strategy))
