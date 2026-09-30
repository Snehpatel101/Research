"""
Timeframe competition for MTF features.

Budgets features per higher timeframe: within each timeframe only the top-N
features by importance survive, so e.g. ``sma_20_15m``, ``ema_21_15m`` and
``close_15m`` do not all take slots in a model's feature set. Base-timeframe
features are never touched.

Timeframes are identified by the column suffix the MTF generator writes
(``get_timeframe_suffix``: ``15min`` -> ``_15m``, ``60min`` -> ``_1h``).

Pure function of a ranking computed on train rows, so it adds no leakage of
its own. Enabled by ``data.features.mtf_max_per_timeframe`` (default off).
"""

from __future__ import annotations

import logging

import pandas as pd

from src.core.common.timeframes import get_timeframe_suffix

logger = logging.getLogger(__name__)


def apply_timeframe_budget(
    ranking: pd.Series,
    feature_names: list[str],
    timeframes: list[str],
    max_per_timeframe: int,
) -> list[str]:
    """Keep at most ``max_per_timeframe`` features per MTF timeframe.

    Args:
        ranking: Importance per feature, ordered most important first (as
            ``rank_by_importance`` returns it); the order decides who stays.
        feature_names: Candidate features.
        timeframes: Higher timeframes whose features are budgeted (e.g.
            ``["15min", "60min"]``); a feature belongs to a timeframe when its
            name ends with that timeframe's column suffix.
        max_per_timeframe: Features kept per timeframe (>= 1).

    Returns:
        ``feature_names`` minus the lowest-ranked MTF features of every
        timeframe over budget, in the original order. Features missing from
        ``ranking`` are kept.
    """
    if max_per_timeframe < 1:
        raise ValueError(f"max_per_timeframe must be >= 1, got {max_per_timeframe}")
    suffixes = {get_timeframe_suffix(tf): tf for tf in timeframes}
    if ranking.empty or not suffixes:
        return list(feature_names)

    candidates = set(feature_names)
    dropped: set[str] = set()
    for suffix, tf in sorted(suffixes.items()):
        ranked = [f for f in ranking.index if f in candidates and str(f).endswith(suffix)]
        if len(ranked) <= max_per_timeframe:
            continue
        over = ranked[max_per_timeframe:]
        dropped.update(over)
        logger.debug(
            f"    Timeframe {tf}: kept {max_per_timeframe}/{len(ranked)} (dropped {len(over)})"
        )
    return [f for f in feature_names if f not in dropped]


__all__ = ["apply_timeframe_budget"]
