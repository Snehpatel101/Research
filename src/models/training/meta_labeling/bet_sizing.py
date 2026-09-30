"""
Enhanced bet sizing strategies for meta-labeling.

Goes beyond binary trade/no-trade to variable position sizing
based on model confidence and risk management principles.
"""

from __future__ import annotations

import logging
from enum import StrEnum

import numpy as np

logger = logging.getLogger(__name__)


class BetSizingStrategy(StrEnum):
    """Bet sizing strategies for meta-labeling."""

    BINARY = "binary"  # Current: trade or no-trade
    PROPORTIONAL = "proportional"  # Size proportional to probability
    KELLY = "kelly"  # Kelly Criterion
    HALF_KELLY = "half_kelly"  # Half Kelly (more conservative)
    CONFIDENCE = "confidence"  # Based on prediction confidence


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
            sizes[above_threshold] = min_size + (excess**2) * (max_size - min_size)

    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    # Ensure within bounds
    sizes = np.clip(sizes, 0, max_size)

    return sizes
