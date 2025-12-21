"""
Stage 6: Apply Optimized Labels with Quality Scoring
Re-labels full dataset using GA-optimized parameters and assigns sample weights

PRODUCTION FIXES:
1. Symbol-specific barrier parameters (MES: asymmetric, MGC: symmetric)
2. Risk-adjusted quality metrics:
   - Pain-to-gain ratio (risk per unit profit)
   - Time-weighted drawdown consideration
3. Uses get_barrier_params() for symbol-specific defaults
"""
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Dict, List
import json

# Import labeling function
import sys
sys.path.insert(0, str(Path(__file__).parent))
from stage4_labeling import triple_barrier_numba

# Import config for symbol-specific parameters
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import get_barrier_params, TRANSACTION_COSTS

# Configure logging - use NullHandler to avoid duplicate logs when imported as module
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def compute_quality_scores(
    bars_to_hit: np.ndarray,
    mae: np.ndarray,
    mfe: np.ndarray,
    labels: np.ndarray,
    horizon: int,
    symbol: str = 'MES'
) -> tuple:
    """
    Compute quality scores for each sample with risk-adjusted metrics.

    Quality is based on:
    1. Speed score: faster hits (within reason) = higher quality
    2. MAE score: lower adverse excursion = higher quality
    3. MFE score: higher favorable excursion = higher quality
    4. Pain-to-gain ratio: risk per unit profit (NEW)
    5. Time-weighted drawdown: penalize trades that spend time in drawdown (NEW)

    Parameters:
    -----------
    bars_to_hit : array of bars until barrier hit
    mae : Maximum Adverse Excursion (always negative or zero)
    mfe : Maximum Favorable Excursion (always positive or zero)
    labels : array of labels (-1, 0, 1)
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
    # Ideal: around horizon to 2*horizon bars
    # Penalize too fast (< horizon/2) or too slow (> horizon*3)
    speed_scores = np.zeros(n, dtype=np.float32)
    ideal_speed = horizon * 1.5

    for i in range(n):
        bars = bars_to_hit[i]
        if bars <= 0:
            speed_scores[i] = 0.0
        else:
            # Gaussian-like scoring around ideal
            deviation = abs(bars - ideal_speed) / ideal_speed
            speed_scores[i] = np.exp(-deviation ** 2)

    # 2. MAE score (lower is better)
    # Normalize and invert
    mae_abs = np.abs(mae)
    mae_max = np.percentile(mae_abs[mae_abs > 0], 95) if np.any(mae_abs > 0) else 1.0
    if mae_max > 0:
        mae_normalized = mae_abs / mae_max
        mae_scores = 1.0 - np.clip(mae_normalized, 0, 1)
    else:
        mae_scores = np.ones(n, dtype=np.float32)

    # 3. MFE score (higher is better)
    mfe_abs = np.abs(mfe)
    mfe_max = np.percentile(mfe_abs[mfe_abs > 0], 95) if np.any(mfe_abs > 0) else 1.0
    if mfe_max > 0:
        mfe_scores = np.clip(mfe_abs / mfe_max, 0, 1)
    else:
        mfe_scores = np.zeros(n, dtype=np.float32)

    # ==========================================================================
    # 4. PAIN-TO-GAIN RATIO (NEW)
    # ==========================================================================
    # Measures risk per unit of profit
    # For LONG trades: pain = |MAE| (drawdown), gain = MFE (upside)
    # For SHORT trades: pain = MFE (upside = loss for short), gain = |MAE| (downside = profit)
    # Lower is better (less pain per unit gain)

    for i in range(n):
        if labels[i] == 1:  # Long
            gain = max(mfe[i], 1e-6)
            pain = abs(mae[i])
            pain_to_gain[i] = pain / gain
        elif labels[i] == -1:  # Short
            gain = max(abs(mae[i]), 1e-6)
            pain = max(mfe[i], 0)
            pain_to_gain[i] = pain / gain
        else:  # Neutral
            pain_to_gain[i] = 1.0  # Default for neutrals

    # Normalize pain-to-gain to 0-1 (clip at 95th percentile)
    valid_ptg = pain_to_gain[labels != 0]
    if len(valid_ptg) > 0:
        ptg_max = np.percentile(valid_ptg, 95)
        if ptg_max > 0:
            ptg_normalized = np.clip(pain_to_gain / ptg_max, 0, 1)
            ptg_scores = 1.0 - ptg_normalized  # Invert: lower pain-to-gain = higher score
        else:
            ptg_scores = np.ones(n, dtype=np.float32)
    else:
        ptg_scores = np.ones(n, dtype=np.float32)

    # ==========================================================================
    # 5. TIME-WEIGHTED DRAWDOWN (NEW)
    # ==========================================================================
    # Penalize trades that spend a lot of time in drawdown relative to total time
    # Approximated by: (bars_to_hit / max_bars) * (|MAE| / |MFE| + epsilon)
    # Lower score = less time in pain = better

    max_bars = horizon * 3  # Approximate max_bars
    for i in range(n):
        if labels[i] != 0 and bars_to_hit[i] > 0:
            time_fraction = bars_to_hit[i] / max_bars
            # Ratio of pain to gain (clamped)
            if mfe[i] > 0:
                pain_ratio = abs(mae[i]) / mfe[i]
            else:
                pain_ratio = 1.0
            pain_ratio = min(pain_ratio, 2.0)  # Clamp at 2x

            # Time-weighted drawdown: longer time * more pain = worse
            time_weighted_dd[i] = time_fraction * pain_ratio
        else:
            time_weighted_dd[i] = 0.5  # Default for neutrals

    # Normalize time-weighted drawdown to 0-1 and invert
    valid_twdd = time_weighted_dd[labels != 0]
    if len(valid_twdd) > 0:
        twdd_max = np.percentile(valid_twdd, 95)
        if twdd_max > 0:
            twdd_normalized = np.clip(time_weighted_dd / twdd_max, 0, 1)
            twdd_scores = 1.0 - twdd_normalized  # Invert: lower = better
        else:
            twdd_scores = np.ones(n, dtype=np.float32)
    else:
        twdd_scores = np.ones(n, dtype=np.float32)

    # ==========================================================================
    # COMBINED QUALITY SCORE
    # ==========================================================================
    # Weights: Speed 20%, MAE 25%, MFE 20%, Pain-to-Gain 20%, Time-Weighted DD 15%
    quality_scores = (
        0.20 * speed_scores +
        0.25 * mae_scores +
        0.20 * mfe_scores +
        0.20 * ptg_scores +
        0.15 * twdd_scores
    )

    return quality_scores, pain_to_gain, time_weighted_dd


def assign_sample_weights(quality_scores: np.ndarray) -> np.ndarray:
    """
    Assign sample weights based on quality score tiers.

    Tier 1 (top 20%): weight 1.5
    Tier 2 (middle 60%): weight 1.0
    Tier 3 (bottom 20%): weight 0.5

    Returns:
    --------
    sample_weights : array of sample weights
    """
    n = len(quality_scores)
    sample_weights = np.ones(n, dtype=np.float32)

    # Compute percentiles
    p20 = np.percentile(quality_scores, 20)
    p80 = np.percentile(quality_scores, 80)

    # Assign weights
    sample_weights[quality_scores >= p80] = 1.5  # Tier 1
    sample_weights[(quality_scores >= p20) & (quality_scores < p80)] = 1.0  # Tier 2
    sample_weights[quality_scores < p20] = 0.5  # Tier 3

    tier1_count = (quality_scores >= p80).sum()
    tier2_count = ((quality_scores >= p20) & (quality_scores < p80)).sum()
    tier3_count = (quality_scores < p20).sum()

    logger.info(f"  Sample weight distribution:")
    logger.info(f"    Tier 1 (1.5x): {tier1_count:6d} ({tier1_count/n*100:5.1f}%)")
    logger.info(f"    Tier 2 (1.0x): {tier2_count:6d} ({tier2_count/n*100:5.1f}%)")
    logger.info(f"    Tier 3 (0.5x): {tier3_count:6d} ({tier3_count/n*100:5.1f}%)")

    return sample_weights


def apply_optimized_labels(
    df: pd.DataFrame,
    horizon: int,
    best_params: Dict,
    symbol: str = 'MES',
    atr_column: str = 'atr_14'
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
    logger.info(f"  Parameters: k_up={best_params['k_up']:.3f}, "
                f"k_down={best_params['k_down']:.3f}, "
                f"max_bars={best_params['max_bars']}")

    # Log barrier type
    k_ratio = best_params['k_up'] / best_params['k_down'] if best_params['k_down'] > 0 else 1.0
    if abs(k_ratio - 1.0) < 0.15:
        barrier_type = "SYMMETRIC"
    elif k_ratio > 1.0:
        barrier_type = f"ASYMMETRIC (k_up/k_down = {k_ratio:.2f})"
    else:
        barrier_type = f"INVERTED (k_down > k_up, ratio = {1/k_ratio:.2f})"
    logger.info(f"  Barrier type: {barrier_type}")

    # Extract arrays
    close = df['close'].values
    high = df['high'].values
    low = df['low'].values
    open_prices = df['open'].values
    atr = df[atr_column].values

    # Apply labeling (includes open_prices for same-bar race condition fix)
    labels, bars_to_hit, mae, mfe, touch_type = triple_barrier_numba(
        close, high, low, open_prices, atr,
        best_params['k_up'],
        best_params['k_down'],
        best_params['max_bars']
    )

    # Compute quality scores with new risk-adjusted metrics
    quality_scores, pain_to_gain, time_weighted_dd = compute_quality_scores(
        bars_to_hit, mae, mfe, labels, horizon, symbol
    )

    # Assign sample weights
    sample_weights = assign_sample_weights(quality_scores)

    # Add to dataframe
    df[f'label_h{horizon}'] = labels
    df[f'bars_to_hit_h{horizon}'] = bars_to_hit
    df[f'mae_h{horizon}'] = mae
    df[f'mfe_h{horizon}'] = mfe
    df[f'touch_type_h{horizon}'] = touch_type
    df[f'quality_h{horizon}'] = quality_scores
    df[f'sample_weight_h{horizon}'] = sample_weights
    df[f'pain_to_gain_h{horizon}'] = pain_to_gain  # NEW
    df[f'time_weighted_dd_h{horizon}'] = time_weighted_dd  # NEW

    # Log statistics
    label_counts = pd.Series(labels).value_counts().sort_index()
    total = len(labels)

    logger.info(f"  Label distribution:")
    for label_val in [-1, 0, 1]:
        count = label_counts.get(label_val, 0)
        pct = count / total * 100
        label_name = {-1: "Short", 0: "Neutral", 1: "Long"}[label_val]
        logger.info(f"    {label_name:10s}: {count:6d} ({pct:5.1f}%)")

    # Check neutral rate against target
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


def process_symbol_final(
    symbol: str,
    horizons: List[int] = [5, 20]  # Exclude H1 - not viable after transaction costs
) -> pd.DataFrame:
    """
    Apply optimized labels for all horizons for a symbol.

    Uses symbol-specific barrier defaults (MES: asymmetric, MGC: symmetric)
    when GA results are not available.

    Parameters:
    -----------
    symbol : 'MES' or 'MGC'
    horizons : list of horizons to process (default excludes H1)

    Returns:
    --------
    df : Final labeled DataFrame
    """
    logger.info("=" * 70)
    logger.info(f"FINAL LABELING: {symbol}")
    logger.info(f"  Barrier mode: {'SYMMETRIC' if symbol == 'MGC' else 'ASYMMETRIC'}")
    logger.info(f"  Transaction cost: {TRANSACTION_COSTS.get(symbol, 0.5)} ticks")
    logger.info("=" * 70)

    # Load features data (original features without labels)
    project_root = Path(__file__).parent.parent.parent.resolve()
    features_dir = project_root / 'data' / 'features'
    input_path = features_dir / f"{symbol}_5m_features.parquet"

    if not input_path.exists():
        raise FileNotFoundError(f"Features file not found: {input_path}")

    df = pd.read_parquet(input_path)
    logger.info(f"Loaded {len(df):,} rows from {input_path}")

    # Load optimized parameters
    ga_results_dir = project_root / 'config' / 'ga_results'

    for horizon in horizons:
        results_path = ga_results_dir / f"{symbol}_ga_h{horizon}_best.json"

        if not results_path.exists():
            # Use symbol-specific defaults from config
            logger.warning(f"No GA results found for {symbol} horizon {horizon}, using symbol-specific defaults")
            default_params = get_barrier_params(symbol, horizon)
            best_params = {
                'k_up': default_params['k_up'],
                'k_down': default_params['k_down'],
                'max_bars': default_params['max_bars']
            }
            logger.info(f"  Using: k_up={best_params['k_up']:.2f}, k_down={best_params['k_down']:.2f}, "
                       f"max_bars={best_params['max_bars']}")
        else:
            with open(results_path, 'r') as f:
                results = json.load(f)
            best_params = {
                'k_up': results['best_k_up'],
                'k_down': results['best_k_down'],
                'max_bars': results['best_max_bars']
            }

        # Apply optimized labeling with symbol parameter
        df = apply_optimized_labels(df, horizon, best_params, symbol=symbol)

    # Save final labeled data
    final_dir = project_root / 'data' / 'final'
    final_dir.mkdir(parents=True, exist_ok=True)
    output_path = final_dir / f"{symbol}_final_labeled.parquet"

    df.to_parquet(output_path, index=False)
    logger.info(f"Saved final labeled data to {output_path}")
    logger.info("")

    return df


def generate_labeling_report(all_results: Dict[str, pd.DataFrame]):
    """Generate a summary report of labeling results with risk-adjusted metrics."""
    report_lines = []
    report_lines.append("# Labeling Report - Production Triple Barrier")
    report_lines.append("")
    report_lines.append("## Configuration")
    report_lines.append("")
    report_lines.append("| Symbol | Barrier Type | Transaction Cost |")
    report_lines.append("|--------|--------------|------------------|")
    report_lines.append(f"| MES | ASYMMETRIC (k_up > k_down) | {TRANSACTION_COSTS.get('MES', 0.5)} ticks |")
    report_lines.append(f"| MGC | SYMMETRIC (k_up = k_down) | {TRANSACTION_COSTS.get('MGC', 0.3)} ticks |")
    report_lines.append("")
    report_lines.append("## Summary by Symbol and Horizon")
    report_lines.append("")

    for symbol, df in all_results.items():
        report_lines.append(f"### {symbol}")
        report_lines.append(f"Total samples: {len(df):,}")
        report_lines.append("")

        for horizon in [5, 20]:  # Exclude H1
            label_col = f'label_h{horizon}'
            quality_col = f'quality_h{horizon}'
            bars_col = f'bars_to_hit_h{horizon}'
            ptg_col = f'pain_to_gain_h{horizon}'
            twdd_col = f'time_weighted_dd_h{horizon}'

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

            # New risk-adjusted metrics
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

            # Check neutral rate
            neutral_pct = n_neutral / total * 100
            if 20 <= neutral_pct <= 30:
                neutral_status = "OK"
            else:
                neutral_status = "WARN"

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
    report_lines.append("- **Time-Weighted DD**: Penalizes trades that spend time in drawdown. Lower is better.")
    report_lines.append("- **Neutral Rate Target**: 20-30% for selective trading signals.")
    report_lines.append("")
    report_lines.append("---")
    report_lines.append("*Generated by Stage 6: Final Labeling (Production Fix)*")

    # Save report
    project_root = Path(__file__).parent.parent.parent.resolve()
    report_path = project_root / 'results' / 'labeling_report.md'
    report_path.parent.mkdir(parents=True, exist_ok=True)

    with open(report_path, 'w') as f:
        f.write('\n'.join(report_lines))

    logger.info(f"Labeling report saved to {report_path}")


def main():
    """Run Stage 6: Final labeling for all symbols."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from config import SYMBOLS, ACTIVE_HORIZONS

    # Use only ACTIVE_HORIZONS (excludes H1 which is not viable after costs)
    horizons = ACTIVE_HORIZONS  # [5, 20]

    logger.info("=" * 70)
    logger.info("STAGE 6: FINAL LABELING WITH RISK-ADJUSTED QUALITY SCORES")
    logger.info("=" * 70)
    logger.info(f"Horizons: {horizons} (H1 excluded - not viable after transaction costs)")
    logger.info("")
    logger.info("PRODUCTION FIXES APPLIED:")
    logger.info("  - Symbol-specific barriers (MES: asymmetric, MGC: symmetric)")
    logger.info("  - Risk-adjusted quality metrics:")
    logger.info("    - Pain-to-gain ratio")
    logger.info("    - Time-weighted drawdown")
    logger.info("  - Target neutral rate: 20-30%")
    logger.info("")

    all_results = {}
    errors = []

    for symbol in SYMBOLS:
        try:
            df = process_symbol_final(symbol, horizons)
            all_results[symbol] = df
        except Exception as e:
            errors.append({
                'symbol': symbol,
                'error': str(e),
                'type': type(e).__name__
            })
            logger.error(f"Error processing {symbol}: {e}", exc_info=True)

    if errors:
        error_summary = f"{len(errors)}/{len(SYMBOLS)} symbols failed final labeling"
        logger.error(f"Final labeling completed with errors: {error_summary}")
        raise RuntimeError(f"{error_summary}. Errors: {errors[:5]}")

    # Generate report
    if all_results:
        generate_labeling_report(all_results)

    logger.info("")
    logger.info("=" * 70)
    logger.info("STAGE 6 COMPLETE")
    logger.info("=" * 70)
    logger.info("Final labeled data ready in data/final/")


if __name__ == "__main__":
    main()
