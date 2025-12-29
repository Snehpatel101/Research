"""
Label sanity validation checks.
"""
import logging

import pandas as pd

logger = logging.getLogger(__name__)


def check_label_distribution(
    df: pd.DataFrame, label_col: str
) -> dict:
    """
    Calculate label distribution for a horizon.

    Args:
        df: DataFrame with labels
        label_col: Name of the label column

    Returns:
        Dictionary with label distribution
    """
    label_counts = df[label_col].value_counts().sort_index()
    label_dist = {}

    for label, count in label_counts.items():
        label_name = {-1: 'short', 0: 'neutral', 1: 'long'}.get(label, str(label))
        pct = count / len(df) * 100
        label_dist[label_name] = {'count': int(count), 'percentage': float(pct)}
        logger.info(f"  {label_name:8s}: {count:,} ({pct:.2f}%)")

    return label_dist


def check_label_balance(
    label_dist: dict, horizon: int, warnings_found: list[str]
) -> None:
    """
    Check if labels are balanced, add warnings if not.

    Args:
        label_dist: Label distribution dictionary
        horizon: The horizon number
        warnings_found: List to append warnings to (mutated)
    """
    valid_labels = {'short', 'neutral', 'long'}
    for label_name, stats in label_dist.items():
        if label_name not in valid_labels:
            continue
        pct = stats['percentage']
        if pct < 20.0:
            warnings_found.append(
                f"h{horizon} {label_name}: low representation ({pct:.1f}%)"
            )
        if pct > 60.0:
            warnings_found.append(
                f"h{horizon} {label_name}: high representation ({pct:.1f}%)"
            )


def check_per_symbol_distribution(df: pd.DataFrame, label_col: str) -> dict:
    """
    Calculate label distribution per symbol.

    Args:
        df: DataFrame with labels
        label_col: Name of the label column

    Returns:
        Dictionary with per-symbol statistics
    """
    symbol_stats = {}

    for symbol in df['symbol'].unique():
        symbol_df = df[df['symbol'] == symbol]
        labels = symbol_df[label_col]
        total = len(labels)
        longs = (labels == 1).sum()
        shorts = (labels == -1).sum()
        neutrals = (labels == 0).sum()

        symbol_stats[symbol] = {
            'total': int(total),
            'long_count': int(longs),
            'short_count': int(shorts),
            'neutral_count': int(neutrals),
            'long_pct': float(longs / total * 100),
            'short_pct': float(shorts / total * 100),
            'neutral_pct': float(neutrals / total * 100)
        }

    logger.info("\n  Per-symbol distribution:")
    for symbol, stats in symbol_stats.items():
        logger.info(
            f"    {symbol}: L={stats['long_pct']:.1f}% "
            f"S={stats['short_pct']:.1f}% N={stats['neutral_pct']:.1f}%"
        )

    return symbol_stats


def check_bars_to_hit(df: pd.DataFrame, label_col: str, bars_col: str) -> dict:
    """
    Calculate bars-to-hit statistics.

    Args:
        df: DataFrame with labels
        label_col: Name of the label column
        bars_col: Name of the bars-to-hit column

    Returns:
        Dictionary with bars-to-hit statistics
    """
    avg_bars = df[bars_col].mean()
    median_bars = df[bars_col].median()

    # Only for non-neutral labels
    hit_mask = df[label_col] != 0
    if hit_mask.sum() > 0:
        avg_bars_hit = df[hit_mask][bars_col].mean()
        median_bars_hit = df[hit_mask][bars_col].median()
    else:
        avg_bars_hit = avg_bars
        median_bars_hit = median_bars

    bars_stats = {
        'mean_all': float(avg_bars),
        'median_all': float(median_bars),
        'mean_hit': float(avg_bars_hit),
        'median_hit': float(median_bars_hit)
    }

    logger.info("\n  Bars to hit statistics:")
    logger.info(f"    Mean (all): {avg_bars:.2f}")
    logger.info(f"    Mean (hit): {avg_bars_hit:.2f}")
    logger.info(f"    Median (hit): {median_bars_hit:.2f}")

    return bars_stats


def check_quality_scores(df: pd.DataFrame, quality_col: str) -> dict:
    """
    Calculate quality score statistics.

    Args:
        df: DataFrame with quality scores
        quality_col: Name of the quality column

    Returns:
        Dictionary with quality score statistics
    """
    quality_stats = {
        'mean': float(df[quality_col].mean()),
        'median': float(df[quality_col].median()),
        'std': float(df[quality_col].std()),
        'min': float(df[quality_col].min()),
        'max': float(df[quality_col].max())
    }

    logger.info("\n  Quality score statistics:")
    logger.info(f"    Mean: {quality_stats['mean']:.3f}")
    logger.info(f"    Median: {quality_stats['median']:.3f}")
    logger.info(f"    Std: {quality_stats['std']:.3f}")
    logger.info(f"    Range: [{quality_stats['min']:.3f}, {quality_stats['max']:.3f}]")

    return quality_stats


def check_label_sanity(
    df: pd.DataFrame,
    horizons: list[int],
    warnings_found: list[str]
) -> dict:
    """
    Run all label sanity checks.

    Args:
        df: DataFrame to validate
        horizons: List of horizons to check
        warnings_found: List to append warnings to (mutated)

    Returns:
        Dictionary with all label sanity results
    """
    logger.info("\n" + "=" * 60)
    logger.info("LABEL SANITY CHECKS")
    logger.info("=" * 60)

    results = {}

    for horizon in horizons:
        label_col = f'label_h{horizon}'
        quality_col = f'quality_h{horizon}'
        bars_col = f'bars_to_hit_h{horizon}'

        if label_col not in df.columns:
            logger.warning(f"  Label column {label_col} not found, skipping")
            continue

        logger.info(f"\nHorizon {horizon}:")

        horizon_results = {}

        # Label distribution
        label_dist = check_label_distribution(df, label_col)
        horizon_results['distribution'] = label_dist

        # Check balance
        check_label_balance(label_dist, horizon, warnings_found)

        # Per-symbol distribution
        if 'symbol' in df.columns:
            horizon_results['per_symbol'] = check_per_symbol_distribution(
                df, label_col
            )

        # Bars to hit statistics
        if bars_col in df.columns:
            horizon_results['bars_to_hit'] = check_bars_to_hit(
                df, label_col, bars_col
            )

        # Quality score statistics
        if quality_col in df.columns:
            horizon_results['quality'] = check_quality_scores(df, quality_col)

        results[f'horizon_{horizon}'] = horizon_results

    return results
