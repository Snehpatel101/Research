"""
Stage 6: Final Labels - Core Logic.

Applies optimized triple-barrier labels with quality scoring and sample weights.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.phase1.config import TRANSACTION_COSTS
from src.phase1.stages.labeling import triple_barrier_numba

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def compute_quality_scores(
    bars_to_hit: np.ndarray,
    mae: np.ndarray,
    mfe: np.ndarray,
    labels: np.ndarray,
    horizon: int,
    symbol: str = "MES",
) -> tuple:
    """
    Compute quality scores for each sample with risk-adjusted metrics.

    Quality is based on:
    1. Speed score: faster hits (within reason) = higher quality
    2. MAE score: lower adverse excursion = higher quality (DIRECTION-AWARE)
    3. MFE score: higher favorable excursion = higher quality (DIRECTION-AWARE)
    4. Pain-to-gain ratio: risk per unit profit
    5. Time-weighted drawdown: penalize trades that spend time in drawdown

    IMPORTANT: MAE/MFE from triple_barrier_numba are computed from a LONG perspective:
      - MFE = max upside (positive) = favorable for LONG, adverse for SHORT
      - MAE = max downside (negative) = adverse for LONG, favorable for SHORT

    This function correctly interprets these values based on trade direction (label):
      - For LONG (label=1): favorable=MFE, adverse=|MAE|
      - For SHORT (label=-1): favorable=|MAE|, adverse=MFE

    Parameters:
    -----------
    bars_to_hit : array of bars until barrier hit
    mae : Maximum Adverse Excursion from long perspective (always negative or zero)
    mfe : Maximum Favorable Excursion from long perspective (always positive or zero)
    labels : array of labels (-1=short, 0=neutral, 1=long)
    horizon : horizon identifier
    symbol : 'MES' or 'MGC' for transaction cost consideration

    Returns:
    --------
    quality_scores : array of quality scores (0-1 range, higher is better)
    pain_to_gain : array of pain-to-gain ratios (risk per unit profit)
    time_weighted_dd : array of time-weighted drawdown scores
    """
    n = len(bars_to_hit)
    quality_scores = np.zeros(n, dtype=np.float32)
    pain_to_gain = np.zeros(n, dtype=np.float32)
    time_weighted_dd = np.zeros(n, dtype=np.float32)

    # 1. Speed score (normalized)
    speed_scores = np.zeros(n, dtype=np.float32)
    ideal_speed = horizon * 1.5

    for i in range(n):
        bars = bars_to_hit[i]
        if bars <= 0:
            speed_scores[i] = 0.0
        else:
            deviation = abs(bars - ideal_speed) / ideal_speed
            speed_scores[i] = np.exp(-(deviation**2))

    # 2. & 3. DIRECTION-AWARE MAE/MFE scores
    # MAE/MFE from triple_barrier_numba are computed from a LONG perspective:
    #   - MFE = max upside (positive) = favorable for LONG, adverse for SHORT
    #   - MAE = max downside (negative) = adverse for LONG, favorable for SHORT
    #
    # We compute "adverse excursion" and "favorable excursion" per-sample
    # based on the actual trade direction (label).

    # Compute direction-aware adverse and favorable excursions
    adverse_excursion = np.zeros(n, dtype=np.float32)
    favorable_excursion = np.zeros(n, dtype=np.float32)

    for i in range(n):
        if labels[i] == 1:  # LONG trade
            # For longs: favorable = price up (MFE), adverse = price down (MAE)
            favorable_excursion[i] = max(mfe[i], 0.0)
            adverse_excursion[i] = abs(min(mae[i], 0.0))
        elif labels[i] == -1:  # SHORT trade
            # For shorts: favorable = price down (|MAE|), adverse = price up (MFE)
            favorable_excursion[i] = abs(min(mae[i], 0.0))
            adverse_excursion[i] = max(mfe[i], 0.0)
        else:  # Neutral (label == 0)
            # For neutrals, use absolute values (no directional bias)
            favorable_excursion[i] = max(abs(mfe[i]), abs(mae[i]))
            adverse_excursion[i] = min(abs(mfe[i]), abs(mae[i]))

    # MAE score: lower adverse excursion = higher quality
    ae_max = (
        np.percentile(adverse_excursion[adverse_excursion > 0], 95)
        if np.any(adverse_excursion > 0)
        else 1.0
    )
    if ae_max > 0:
        ae_normalized = adverse_excursion / ae_max
        mae_scores = 1.0 - np.clip(ae_normalized, 0, 1)
    else:
        mae_scores = np.ones(n, dtype=np.float32)

    # MFE score: higher favorable excursion = higher quality
    fe_max = (
        np.percentile(favorable_excursion[favorable_excursion > 0], 95)
        if np.any(favorable_excursion > 0)
        else 1.0
    )
    if fe_max > 0:
        mfe_scores = np.clip(favorable_excursion / fe_max, 0, 1)
    else:
        mfe_scores = np.zeros(n, dtype=np.float32)

    # 4. PAIN-TO-GAIN RATIO
    pain_to_gain = np.ones(n, dtype=np.float32) * 0.5  # Default to moderate

    for i in range(n):
        if labels[i] == 1:  # Long trade
            gain = max(mfe[i], 1e-6)
            pain = abs(min(mae[i], 0))
            pain_to_gain[i] = pain / gain
        elif labels[i] == -1:  # Short trade
            gain = max(abs(mae[i]), 1e-6)
            pain = max(mfe[i], 0)
            pain_to_gain[i] = pain / gain
        else:  # Neutral
            pain_to_gain[i] = 1.0

    # Normalize pain-to-gain
    valid_ptg = pain_to_gain[labels != 0]
    if len(valid_ptg) > 0:
        ptg_max = np.percentile(valid_ptg, 95)
        if ptg_max > 0:
            ptg_normalized = np.clip(pain_to_gain / ptg_max, 0, 1)
            ptg_scores = 1.0 - ptg_normalized
        else:
            ptg_scores = np.ones(n, dtype=np.float32)
    else:
        ptg_scores = np.ones(n, dtype=np.float32)

    # 5. TIME-WEIGHTED DRAWDOWN
    max_bars = horizon * 3
    MAX_PAIN_RATIO = 2.0

    for i in range(n):
        if labels[i] != 0 and bars_to_hit[i] > 0:
            time_fraction = bars_to_hit[i] / max_bars

            if labels[i] == 1:  # Long trade
                pain = abs(min(mae[i], 0))
                gain = mfe[i]
            else:  # Short trade
                pain = max(mfe[i], 0)
                gain = abs(mae[i])

            if gain > 1e-6:
                pain_ratio = pain / gain
            else:
                pain_ratio = MAX_PAIN_RATIO

            pain_ratio = min(pain_ratio, MAX_PAIN_RATIO)
            time_weighted_dd[i] = time_fraction * pain_ratio
        else:
            time_weighted_dd[i] = 0.5

    # Normalize time-weighted drawdown
    valid_twdd = time_weighted_dd[labels != 0]
    if len(valid_twdd) > 0:
        twdd_max = np.percentile(valid_twdd, 95)
        if twdd_max > 0:
            twdd_normalized = np.clip(time_weighted_dd / twdd_max, 0, 1)
            twdd_scores = 1.0 - twdd_normalized
        else:
            twdd_scores = np.ones(n, dtype=np.float32)
    else:
        twdd_scores = np.ones(n, dtype=np.float32)

    # COMBINED QUALITY SCORE
    quality_scores = (
        0.20 * speed_scores
        + 0.25 * mae_scores
        + 0.20 * mfe_scores
        + 0.20 * ptg_scores
        + 0.15 * twdd_scores
    )

    return quality_scores, pain_to_gain, time_weighted_dd


def assign_sample_weights(quality_scores: np.ndarray) -> np.ndarray:
    """
    Assign sample weights based on quality score tiers.

    Tier 1 (top 20%): weight 1.5
    Tier 2 (middle 60%): weight 1.0
    Tier 3 (bottom 20%): weight 0.5
    """
    n = len(quality_scores)
    sample_weights = np.ones(n, dtype=np.float32)

    p20 = np.percentile(quality_scores, 20)
    p80 = np.percentile(quality_scores, 80)

    sample_weights[quality_scores >= p80] = 1.5
    sample_weights[(quality_scores >= p20) & (quality_scores < p80)] = 1.0
    sample_weights[quality_scores < p20] = 0.5

    tier1_count = (quality_scores >= p80).sum()
    tier2_count = ((quality_scores >= p20) & (quality_scores < p80)).sum()
    tier3_count = (quality_scores < p20).sum()

    logger.info("  Sample weight distribution:")
    logger.info(f"    Tier 1 (1.5x): {tier1_count:6d} ({tier1_count/n*100:5.1f}%)")
    logger.info(f"    Tier 2 (1.0x): {tier2_count:6d} ({tier2_count/n*100:5.1f}%)")
    logger.info(f"    Tier 3 (0.5x): {tier3_count:6d} ({tier3_count/n*100:5.1f}%)")

    return sample_weights


def add_forward_return_columns(
    df: pd.DataFrame, horizon: int, close_column: str = "close"
) -> pd.DataFrame:
    """Add forward return columns for the given horizon."""
    forward_close = df[close_column].shift(-horizon)
    denom = df[close_column].replace(0, np.nan)
    ratio = (forward_close / denom).where(lambda x: x > 0)

    df[f"fwd_return_h{horizon}"] = (ratio - 1.0).astype(np.float32)
    df[f"fwd_return_log_h{horizon}"] = np.log(ratio).astype(np.float32)
    return df


def apply_optimized_labels(
    df: pd.DataFrame,
    horizon: int,
    best_params: dict,
    symbol: str = "MES",
    atr_column: str = "atr_14",
) -> pd.DataFrame:
    """
    Apply optimized triple barrier labeling and compute quality scores.

    Parameters:
    -----------
    df : DataFrame with OHLCV and ATR
    horizon : horizon identifier
    best_params : dict with 'k_up', 'k_down', 'max_bars'
    symbol : 'MES' or 'MGC' for symbol-specific metrics
    atr_column : ATR column name

    Returns:
    --------
    df : DataFrame with optimized labels, quality scores, and sample weights
    """
    logger.info(f"Applying optimized labels for {symbol} horizon {horizon}")
    logger.info(
        f"  Parameters: k_up={best_params['k_up']:.3f}, "
        f"k_down={best_params['k_down']:.3f}, "
        f"max_bars={best_params['max_bars']}"
    )

    # Log barrier type
    k_ratio = best_params["k_up"] / best_params["k_down"] if best_params["k_down"] > 0 else 1.0
    if abs(k_ratio - 1.0) < 0.15:
        barrier_type = "SYMMETRIC"
    elif k_ratio > 1.0:
        barrier_type = f"ASYMMETRIC (k_up/k_down = {k_ratio:.2f})"
    else:
        barrier_type = f"INVERTED (k_down > k_up, ratio = {1/k_ratio:.2f})"
    logger.info(f"  Barrier type: {barrier_type}")

    # Extract arrays
    close = df["close"].values
    high = df["high"].values
    low = df["low"].values
    open_prices = df["open"].values
    atr = df[atr_column].values

    # Apply labeling
    labels, bars_to_hit, mae, mfe, touch_type = triple_barrier_numba(
        close,
        high,
        low,
        open_prices,
        atr,
        best_params["k_up"],
        best_params["k_down"],
        best_params["max_bars"],
    )

    # Compute quality scores
    quality_scores, pain_to_gain, time_weighted_dd = compute_quality_scores(
        bars_to_hit, mae, mfe, labels, horizon, symbol
    )

    # Assign sample weights
    sample_weights = assign_sample_weights(quality_scores)

    # Add to dataframe
    df[f"label_h{horizon}"] = labels
    df[f"bars_to_hit_h{horizon}"] = bars_to_hit
    df[f"mae_h{horizon}"] = mae
    df[f"mfe_h{horizon}"] = mfe
    df[f"touch_type_h{horizon}"] = touch_type
    df[f"quality_h{horizon}"] = quality_scores
    df[f"sample_weight_h{horizon}"] = sample_weights
    df[f"pain_to_gain_h{horizon}"] = pain_to_gain
    df[f"time_weighted_dd_h{horizon}"] = time_weighted_dd
    df = add_forward_return_columns(df, horizon)

    # Compute label_end_time: datetime when label outcome is known
    # CRITICAL: We must look up the actual datetime of the bar that is bars_to_hit forward,
    # NOT compute entry_time + bars_to_hit * bar_duration. The latter is WRONG when there
    # are gaps (overnight, weekends, session breaks) because it under-estimates the true
    # end time, leading to insufficient purging and label leakage.
    if "datetime" in df.columns:
        datetime_col = pd.to_datetime(df["datetime"])
        datetime_arr = datetime_col.values
        n = len(datetime_col)

        # Vectorized forward-lookup: for each bar i, find datetime at index i + bars_to_hit[i]
        # This correctly handles gaps because we use actual bar indices, not time arithmetic
        forward_indices = np.arange(n) + bars_to_hit
        # Clip to valid range (last bar's datetime for samples near the end)
        forward_indices = np.clip(forward_indices, 0, n - 1)

        # Look up the actual datetime at those forward indices
        label_end_times = pd.Series(datetime_arr[forward_indices], index=datetime_col.index)

        df[f"label_end_time_h{horizon}"] = label_end_times
        logger.info("  Added label_end_time column for purging (gap-aware lookup)")

    # Log statistics
    label_counts = pd.Series(labels).value_counts().sort_index()
    total = len(labels)

    logger.info("  Label distribution:")
    for label_val in [-1, 0, 1]:
        count = label_counts.get(label_val, 0)
        pct = count / total * 100
        label_name = {-1: "Short", 0: "Neutral", 1: "Long"}[label_val]
        logger.info(f"    {label_name:10s}: {count:6d} ({pct:5.1f}%)")

    # Check neutral rate
    neutral_pct = label_counts.get(0, 0) / total * 100
    if 20 <= neutral_pct <= 30:
        neutral_status = "OK (target: 20-30%)"
    elif neutral_pct < 20:
        neutral_status = f"LOW (target: 20-30%, got {neutral_pct:.1f}%)"
    else:
        neutral_status = f"HIGH (target: 20-30%, got {neutral_pct:.1f}%)"
    logger.info(f"  Neutral rate: {neutral_status}")

    # Additional metrics
    avg_bars = bars_to_hit[bars_to_hit > 0].mean()
    avg_quality = quality_scores.mean()
    avg_ptg = pain_to_gain[labels != 0].mean() if (labels != 0).sum() > 0 else 0
    avg_twdd = time_weighted_dd[labels != 0].mean() if (labels != 0).sum() > 0 else 0

    logger.info(f"  Avg bars to hit: {avg_bars:.1f}")
    logger.info(f"  Avg quality score: {avg_quality:.3f}")
    logger.info(f"  Avg pain-to-gain: {avg_ptg:.3f}")
    logger.info(f"  Avg time-weighted DD: {avg_twdd:.3f}")

    return df


def generate_labeling_report(
    all_results: dict[str, pd.DataFrame],
    output_dir: Path,
    horizons: list[int],
) -> Path:
    """Generate a summary report of labeling results with risk-adjusted metrics."""
    report_lines = []
    report_lines.append("# Labeling Report - Production Triple Barrier")
    report_lines.append("")
    report_lines.append("## Configuration")
    report_lines.append("")
    report_lines.append("| Symbol | Barrier Type | Transaction Cost |")
    report_lines.append("|--------|--------------|------------------|")
    for symbol in sorted(all_results.keys()):
        if symbol == "MES":
            barrier_type = "ASYMMETRIC (k_up > k_down)"
        elif symbol == "MGC":
            barrier_type = "SYMMETRIC (k_up = k_down)"
        else:
            barrier_type = "SYMMETRIC"
        report_lines.append(
            f"| {symbol} | {barrier_type} | {TRANSACTION_COSTS.get(symbol, 0)} ticks |"
        )
    report_lines.append("")
    report_lines.append("## Summary by Symbol and Horizon")
    report_lines.append("")

    for symbol, df in all_results.items():
        report_lines.append(f"### {symbol}")
        report_lines.append(f"Total samples: {len(df):,}")
        report_lines.append("")

        for horizon in horizons:
            label_col = f"label_h{horizon}"
            quality_col = f"quality_h{horizon}"
            bars_col = f"bars_to_hit_h{horizon}"
            ptg_col = f"pain_to_gain_h{horizon}"
            twdd_col = f"time_weighted_dd_h{horizon}"

            if label_col not in df.columns:
                continue

            labels = df[label_col].values
            quality = df[quality_col].values
            bars = df[bars_col].values

            label_counts = pd.Series(labels).value_counts().sort_index()
            total = len(labels)

            n_long = label_counts.get(1, 0)
            n_short = label_counts.get(-1, 0)
            n_neutral = label_counts.get(0, 0)

            avg_bars = bars[bars > 0].mean()
            avg_quality = quality.mean()

            if ptg_col in df.columns:
                ptg = df[ptg_col].values
                avg_ptg = ptg[labels != 0].mean() if (labels != 0).sum() > 0 else 0
            else:
                avg_ptg = 0

            if twdd_col in df.columns:
                twdd = df[twdd_col].values
                avg_twdd = twdd[labels != 0].mean() if (labels != 0).sum() > 0 else 0
            else:
                avg_twdd = 0

            neutral_pct = n_neutral / total * 100
            neutral_status = "OK" if 20 <= neutral_pct <= 30 else "WARN"

            report_lines.append(f"#### Horizon {horizon}")
            report_lines.append(f"- Long: {n_long:,} ({n_long/total*100:.1f}%)")
            report_lines.append(f"- Short: {n_short:,} ({n_short/total*100:.1f}%)")
            report_lines.append(f"- Neutral: {n_neutral:,} ({neutral_pct:.1f}%) [{neutral_status}]")
            report_lines.append(f"- Signal rate: {(n_long + n_short)/total*100:.1f}%")
            report_lines.append(f"- Avg bars to hit: {avg_bars:.1f}")
            report_lines.append(f"- Avg quality: {avg_quality:.3f}")
            report_lines.append(f"- Avg pain-to-gain: {avg_ptg:.3f}")
            report_lines.append(f"- Avg time-weighted DD: {avg_twdd:.3f}")
            report_lines.append("")

    report_lines.append("## Quality Metrics Explanation")
    report_lines.append("")
    report_lines.append("- **Pain-to-Gain Ratio**: Risk per unit of profit. Lower is better.")
    report_lines.append(
        "- **Time-Weighted DD**: Penalizes trades that spend time in drawdown. Lower is better."
    )
    report_lines.append("- **Neutral Rate Target**: 20-30% for selective trading signals.")
    report_lines.append("")
    report_lines.append("---")
    report_lines.append("*Generated by Stage 6: Final Labeling (Production Fix)*")

    # Save report
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "labeling_report.md"

    with open(report_path, "w") as f:
        f.write("\n".join(report_lines))

    logger.info(f"Labeling report saved to {report_path}")
    return report_path
