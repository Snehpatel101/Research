"""
Timeframe competition for MTF features.

Prevents MTF redundancy by budgeting features per timeframe.
Within each timeframe, only the top-N features by importance survive.
This eliminates the substitution effect where e.g. rsi_14_5min, rsi_14_15min,
rsi_14_60min all compete for the same importance budget.

Reference: Phase E4 of Feature Governance roadmap.
"""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)

# MTF timeframe suffixes used in feature naming convention
MTF_SUFFIXES = ("_1min", "_5min", "_15min", "_30min", "_60min")


def apply_timeframe_budget(
    ranking: pd.Series,
    feature_names: list[str],
    max_per_timeframe: int = 8,
) -> list[str]:
    """Apply per-timeframe feature budget to prevent MTF redundancy.

    Groups features by their timeframe suffix. For each timeframe group,
    keeps only the top ``max_per_timeframe`` features by importance ranking.
    Base-timeframe features (no MTF suffix) are kept unconditionally.

    Args:
        ranking: Feature importance scores (pd.Series, index=feature names,
                 higher values = more important). Only features present in
                 this Series AND in feature_names are considered.
        feature_names: Complete list of candidate feature names.
        max_per_timeframe: Maximum features to keep per timeframe group.
                          Default 8 allows diverse feature types within
                          each timeframe while preventing redundancy.

    Returns:
        Filtered list of feature names respecting the timeframe budget.
        Order follows ranking (most important first).

    Example:
        >>> ranking = pd.Series({
        ...     'rsi_14_5min': 0.8, 'macd_line_5min': 0.6,
        ...     'atr_14_5min': 0.4, 'bb_width_5min': 0.2,
        ...     'rsi_14_15min': 0.7, 'macd_line_15min': 0.5,
        ...     'log_return': 0.9, 'adx_14': 0.85,
        ... })
        >>> result = apply_timeframe_budget(ranking, list(ranking.index), max_per_timeframe=2)
        >>> # Keeps top 2 per timeframe + all base features
    """
    if ranking is None or ranking.empty or not feature_names:
        return list(feature_names)

    # Separate base features from MTF features
    base_features = []
    mtf_groups: dict[str, list[str]] = {}

    for feat in feature_names:
        matched_suffix = None
        for suffix in MTF_SUFFIXES:
            if feat.endswith(suffix):
                matched_suffix = suffix
                break

        if matched_suffix is not None:
            if matched_suffix not in mtf_groups:
                mtf_groups[matched_suffix] = []
            mtf_groups[matched_suffix].append(feat)
        else:
            base_features.append(feat)

    if not mtf_groups:
        # No MTF features, nothing to budget
        return list(feature_names)

    # For each timeframe, keep only top-N by ranking
    budgeted_mtf = []
    for suffix, tf_features in sorted(mtf_groups.items()):
        # Get ranking scores for features in this timeframe
        tf_in_ranking = [f for f in tf_features if f in ranking.index]
        tf_not_ranked = [f for f in tf_features if f not in ranking.index]

        if tf_in_ranking:
            tf_scores = ranking[tf_in_ranking].sort_values(ascending=False)
            kept = tf_scores.head(max_per_timeframe).index.tolist()
            removed_count = len(tf_in_ranking) - len(kept)
            if removed_count > 0:
                logger.info(
                    f"    Timeframe {suffix}: kept {len(kept)}/{len(tf_in_ranking)} "
                    f"features (removed {removed_count})"
                )
            budgeted_mtf.extend(kept)

        # Keep unranked MTF features (safety — shouldn't normally happen)
        budgeted_mtf.extend(tf_not_ranked)

    # Combine: base features + budgeted MTF features
    result = base_features + budgeted_mtf

    # Sort by ranking for consistent ordering (most important first)
    def sort_key(f):
        return -ranking.get(f, 0.0) if f in ranking.index else 0.0

    result.sort(key=sort_key)

    total_removed = len(feature_names) - len(result)
    if total_removed > 0:
        logger.info(
            f"  Timeframe budget: {len(feature_names)} -> {len(result)} features "
            f"(removed {total_removed} redundant MTF features, "
            f"max {max_per_timeframe}/timeframe)"
        )

    return result


__all__ = ["MTF_SUFFIXES", "apply_timeframe_budget"]
