"""
Stage 6: Apply Optimized Labels with Quality Scoring
Re-labels full dataset using GA-optimized parameters and assigns sample weights
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

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def compute_quality_scores(
    bars_to_hit: np.ndarray,
    mae: np.ndarray,
    mfe: np.ndarray,
    horizon: int
) -> np.ndarray:
    """
    Compute quality scores for each sample.

    Quality is based on:
    1. Speed score: faster hits (within reason) = higher quality
    2. MAE score: lower adverse excursion = higher quality
    3. MFE score: higher favorable excursion = higher quality

    Returns:
    --------
    quality_scores : array of quality scores (0-1 range, higher is better)
    """
    n = len(bars_to_hit)
    quality_scores = np.zeros(n, dtype=np.float32)

    # 1. Speed score (normalized)
    # Ideal: around horizon to 2*horizon bars
    # Penalize too fast (< horizon/2) or too slow (> horizon*3)
    speed_scores = np.zeros(n)
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
    mae_max = np.percentile(mae_abs, 95)  # use 95th percentile to avoid outliers
    if mae_max > 0:
        mae_normalized = mae_abs / mae_max
        mae_scores = 1.0 - np.clip(mae_normalized, 0, 1)
    else:
        mae_scores = np.ones(n)

    # 3. MFE score (higher is better)
    mfe_abs = np.abs(mfe)
    mfe_max = np.percentile(mfe_abs, 95)
    if mfe_max > 0:
        mfe_scores = np.clip(mfe_abs / mfe_max, 0, 1)
    else:
        mfe_scores = np.zeros(n)

    # Combined quality (weighted average)
    # Speed: 30%, MAE: 40%, MFE: 30%
    quality_scores = (
        0.30 * speed_scores +
        0.40 * mae_scores +
        0.30 * mfe_scores
    )

    return quality_scores


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
    atr_column: str = 'atr_14'
) -> pd.DataFrame:
    """
    Apply optimized triple barrier labeling and compute quality scores.

    Parameters:
    -----------
    df : DataFrame with OHLCV and ATR
    horizon : horizon identifier
    best_params : dict with 'k_up', 'k_down', 'max_bars'
    atr_column : ATR column name

    Returns:
    --------
    df : DataFrame with optimized labels, quality scores, and sample weights
    """
    logger.info(f"Applying optimized labels for horizon {horizon}")
    logger.info(f"  Parameters: k_up={best_params['k_up']:.3f}, "
                f"k_down={best_params['k_down']:.3f}, "
                f"max_bars={best_params['max_bars']}")

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

    # Compute quality scores
    quality_scores = compute_quality_scores(bars_to_hit, mae, mfe, horizon)

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

    # Log statistics
    label_counts = pd.Series(labels).value_counts().sort_index()
    total = len(labels)

    logger.info(f"  Label distribution:")
    for label_val in [-1, 0, 1]:
        count = label_counts.get(label_val, 0)
        pct = count / total * 100
        label_name = {-1: "Short/Loss", 0: "Neutral/Timeout", 1: "Long/Win"}[label_val]
        logger.info(f"    {label_name:20s}: {count:6d} ({pct:5.1f}%)")

    avg_bars = bars_to_hit[bars_to_hit > 0].mean()
    avg_quality = quality_scores.mean()
    logger.info(f"  Avg bars to hit: {avg_bars:.1f}")
    logger.info(f"  Avg quality score: {avg_quality:.3f}")

    return df


def process_symbol_final(
    symbol: str,
    horizons: List[int] = [1, 5, 20]
) -> pd.DataFrame:
    """
    Apply optimized labels for all horizons for a symbol.

    Returns:
    --------
    df : Final labeled DataFrame
    """
    logger.info("=" * 70)
    logger.info(f"FINAL LABELING: {symbol}")
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
            logger.warning(f"No GA results found for horizon {horizon}, using defaults")
            best_params = {
                'k_up': 2.0,
                'k_down': 1.0,
                'max_bars': horizon * 3
            }
        else:
            with open(results_path, 'r') as f:
                results = json.load(f)
            best_params = {
                'k_up': results['best_k_up'],
                'k_down': results['best_k_down'],
                'max_bars': results['best_max_bars']
            }

        # Apply optimized labeling
        df = apply_optimized_labels(df, horizon, best_params)

    # Save final labeled data
    final_dir = project_root / 'data' / 'final'
    final_dir.mkdir(parents=True, exist_ok=True)
    output_path = final_dir / f"{symbol}_final_labeled.parquet"

    df.to_parquet(output_path, index=False)
    logger.info(f"Saved final labeled data to {output_path}")
    logger.info("")

    return df


def generate_labeling_report(all_results: Dict[str, pd.DataFrame]):
    """Generate a summary report of labeling results."""
    report_lines = []
    report_lines.append("# Labeling Report - Optimized Triple Barrier")
    report_lines.append("")
    report_lines.append("## Summary by Symbol and Horizon")
    report_lines.append("")

    for symbol, df in all_results.items():
        report_lines.append(f"### {symbol}")
        report_lines.append(f"Total samples: {len(df):,}")
        report_lines.append("")

        for horizon in [1, 5, 20]:
            label_col = f'label_h{horizon}'
            quality_col = f'quality_h{horizon}'
            bars_col = f'bars_to_hit_h{horizon}'

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

            report_lines.append(f"#### Horizon {horizon}")
            report_lines.append(f"- Long/Win: {n_long:,} ({n_long/total*100:.1f}%)")
            report_lines.append(f"- Short/Loss: {n_short:,} ({n_short/total*100:.1f}%)")
            report_lines.append(f"- Neutral: {n_neutral:,} ({n_neutral/total*100:.1f}%)")
            report_lines.append(f"- Avg bars to hit: {avg_bars:.1f}")
            report_lines.append(f"- Avg quality: {avg_quality:.3f}")
            report_lines.append("")

    report_lines.append("---")
    report_lines.append("*Generated by Stage 6: Final Labeling*")

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

    from config import SYMBOLS

    horizons = [1, 5, 20]

    logger.info("=" * 70)
    logger.info("STAGE 6: FINAL LABELING WITH QUALITY SCORES")
    logger.info("=" * 70)
    logger.info(f"Horizons: {horizons}")
    logger.info("")

    all_results = {}

    for symbol in SYMBOLS:
        try:
            df = process_symbol_final(symbol, horizons)
            all_results[symbol] = df
        except Exception as e:
            logger.error(f"Error processing {symbol}: {e}", exc_info=True)

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
