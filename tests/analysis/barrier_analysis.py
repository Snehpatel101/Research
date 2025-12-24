#!/usr/bin/env python3
"""
Statistical Analysis for Optimal Triple-Barrier Parameters
Analyzes price movements relative to ATR to determine optimal k_up/k_down multipliers.
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Paths
FEATURES_PATH = Path("/Users/sneh/research/data/features")
RESULTS_PATH = Path("/Users/sneh/research/results")
RESULTS_PATH.mkdir(exist_ok=True)

def load_data():
    """Load feature data for both symbols."""
    print("=" * 70)
    print("LOADING DATA")
    print("=" * 70)

    data = {}
    for symbol in ['MES', 'MGC']:
        filepath = FEATURES_PATH / f"{symbol}_5m_features.parquet"
        df = pd.read_parquet(filepath)
        data[symbol] = df
        print(f"\n{symbol}: {len(df):,} rows")
        print(f"  Date range: {df.index.min()} to {df.index.max()}")
        print(f"  Columns: {len(df.columns)}")

    return data

def identify_atr_column(df):
    """Find the ATR column in the dataframe."""
    atr_cols = [c for c in df.columns if 'atr' in c.lower()]
    print(f"  ATR columns found: {atr_cols[:10]}")

    # Prefer standard ATR columns
    for preferred in ['atr_14', 'atr', 'ATR_14', 'ATR']:
        if preferred in df.columns:
            return preferred

    # Fall back to first ATR column
    if atr_cols:
        return atr_cols[0]

    return None

def calculate_forward_returns(df, horizons=[1, 5, 20]):
    """Calculate forward returns for different horizons."""
    results = {}
    close = df['close'].values

    for h in horizons:
        # Forward price move (in points)
        forward_close = np.roll(close, -h)
        forward_move = forward_close - close

        # Max/min within horizon
        max_move = np.zeros(len(close))
        min_move = np.zeros(len(close))

        for i in range(len(close) - h):
            future_prices = close[i+1:i+h+1]
            max_move[i] = np.max(future_prices) - close[i]
            min_move[i] = np.min(future_prices) - close[i]

        # Invalidate last h rows
        forward_move[-h:] = np.nan
        max_move[-h:] = np.nan
        min_move[-h:] = np.nan

        results[h] = {
            'forward_move': forward_move,
            'max_move': max_move,
            'min_move': min_move,
            'max_abs_move': np.maximum(np.abs(max_move), np.abs(min_move))
        }

    return results

def analyze_barrier_hits(df, atr_col, horizons=[1, 5, 20]):
    """Analyze what percentage of bars hit various ATR multiples."""
    print("\n" + "=" * 70)
    print("BARRIER HIT ANALYSIS")
    print("=" * 70)

    atr = df[atr_col].values
    forward_returns = calculate_forward_returns(df, horizons)

    multipliers = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.7, 0.75, 0.8, 0.9, 1.0, 1.25, 1.5, 2.0]

    results = {}

    for h in horizons:
        print(f"\n--- Horizon H={h} bars ({h*5} minutes) ---")

        max_abs = forward_returns[h]['max_abs_move']
        max_up = forward_returns[h]['max_move']
        max_down = np.abs(forward_returns[h]['min_move'])

        # Filter valid rows
        valid_mask = ~np.isnan(max_abs) & ~np.isnan(atr) & (atr > 0)
        max_abs_valid = max_abs[valid_mask]
        max_up_valid = max_up[valid_mask]
        max_down_valid = max_down[valid_mask]
        atr_valid = atr[valid_mask]

        n_valid = len(max_abs_valid)

        # Basic statistics
        mean_abs_move = np.mean(max_abs_valid)
        median_abs_move = np.median(max_abs_valid)
        mean_atr = np.mean(atr_valid)
        median_atr = np.median(atr_valid)

        print(f"  Valid samples: {n_valid:,}")
        print(f"  Mean ATR: {mean_atr:.4f} points")
        print(f"  Median ATR: {median_atr:.4f} points")
        print(f"  Mean max absolute move: {mean_abs_move:.4f} points")
        print(f"  Median max absolute move: {median_abs_move:.4f} points")
        print(f"  Move/ATR ratio: {mean_abs_move/mean_atr:.3f}")

        # Calculate hit rates for different multipliers
        hit_rates = {}
        for mult in multipliers:
            barrier = atr_valid * mult

            # Either direction hits
            hits_either = (max_up_valid >= barrier) | (max_down_valid >= barrier)
            hit_rate = np.mean(hits_either) * 100

            # Up hits only
            hits_up = max_up_valid >= barrier
            up_rate = np.mean(hits_up) * 100

            # Down hits only
            hits_down = max_down_valid >= barrier
            down_rate = np.mean(hits_down) * 100

            hit_rates[mult] = {
                'either': hit_rate,
                'up': up_rate,
                'down': down_rate,
                'neutral': 100 - hit_rate
            }

        print(f"\n  Hit Rates by ATR Multiplier:")
        print(f"  {'Mult':>6} | {'Hit%':>7} | {'Up%':>7} | {'Down%':>7} | {'Neutral%':>8}")
        print(f"  {'-'*6}-+-{'-'*7}-+-{'-'*7}-+-{'-'*7}-+-{'-'*8}")
        for mult in multipliers:
            hr = hit_rates[mult]
            print(f"  {mult:>6.2f} | {hr['either']:>7.2f} | {hr['up']:>7.2f} | {hr['down']:>7.2f} | {hr['neutral']:>8.2f}")

        results[h] = {
            'n_samples': n_valid,
            'mean_atr': float(mean_atr),
            'median_atr': float(median_atr),
            'mean_abs_move': float(mean_abs_move),
            'median_abs_move': float(median_abs_move),
            'move_atr_ratio': float(mean_abs_move / mean_atr),
            'hit_rates': {str(k): v for k, v in hit_rates.items()}
        }

    return results

def find_optimal_multiplier(hit_rates, target_hit_rate=67.5):
    """Find multiplier that achieves target hit rate (leaving ~30-35% neutral)."""
    multipliers = sorted([float(k) for k in hit_rates.keys()])

    best_mult = None
    best_diff = float('inf')

    for mult in multipliers:
        hit_rate = hit_rates[str(mult)]['either']
        diff = abs(hit_rate - target_hit_rate)
        if diff < best_diff:
            best_diff = diff
            best_mult = mult

    return best_mult, hit_rates[str(best_mult)]['either']

def generate_recommendations(all_results):
    """Generate optimal parameter recommendations."""
    print("\n" + "=" * 70)
    print("OPTIMAL PARAMETER RECOMMENDATIONS")
    print("=" * 70)

    recommendations = {}

    for symbol, symbol_results in all_results.items():
        print(f"\n{symbol}:")
        recommendations[symbol] = {}

        for h, h_results in symbol_results.items():
            hit_rates = h_results['hit_rates']

            # Find multiplier for ~65-70% hit rate
            opt_mult, opt_rate = find_optimal_multiplier(hit_rates, target_hit_rate=67.5)

            # Also find for 60% and 75% targets
            mult_60, rate_60 = find_optimal_multiplier(hit_rates, target_hit_rate=60)
            mult_75, rate_75 = find_optimal_multiplier(hit_rates, target_hit_rate=75)

            print(f"  H={h} bars:")
            print(f"    For ~60% hit rate: k={mult_60:.2f} (actual: {rate_60:.1f}%)")
            print(f"    For ~67.5% hit rate: k={opt_mult:.2f} (actual: {opt_rate:.1f}%)")
            print(f"    For ~75% hit rate: k={mult_75:.2f} (actual: {rate_75:.1f}%)")

            recommendations[symbol][f"h{h}"] = {
                'optimal_k': float(opt_mult),
                'optimal_hit_rate': float(opt_rate),
                'k_for_60pct': float(mult_60),
                'k_for_75pct': float(mult_75),
                'mean_atr': h_results['mean_atr'],
                'mean_abs_move': h_results['mean_abs_move']
            }

    return recommendations

def create_summary_report(all_results, recommendations):
    """Create final summary report."""
    print("\n" + "=" * 70)
    print("SUMMARY REPORT")
    print("=" * 70)

    report = {
        'timestamp': datetime.now().isoformat(),
        'analysis_type': 'Triple-Barrier Optimal Parameter Analysis',
        'symbols_analyzed': list(all_results.keys()),
        'horizons_analyzed': [1, 5, 20],
        'target_hit_rate': '65-70% (30-35% neutral)',
        'detailed_results': {},
        'recommendations': recommendations,
        'final_recommendations': {}
    }

    # Aggregate recommendations across symbols
    for h in [1, 5, 20]:
        h_key = f"h{h}"
        k_values = []
        for symbol in all_results.keys():
            if h_key in recommendations[symbol]:
                k_values.append(recommendations[symbol][h_key]['optimal_k'])

        if k_values:
            avg_k = np.mean(k_values)
            report['final_recommendations'][h_key] = {
                'recommended_k_up': round(avg_k, 2),
                'recommended_k_down': round(avg_k, 2),
                'horizon_bars': h,
                'horizon_minutes': h * 5,
                'expected_neutral_pct': round(100 - np.mean([
                    recommendations[s][h_key]['optimal_hit_rate']
                    for s in all_results.keys()
                    if h_key in recommendations[s]
                ]), 1)
            }

    print("\nFINAL RECOMMENDED PARAMETERS:")
    print("-" * 40)
    for h_key, rec in report['final_recommendations'].items():
        print(f"  {h_key} ({rec['horizon_minutes']} min):")
        print(f"    k_up = k_down = {rec['recommended_k_up']}")
        print(f"    Expected neutral: ~{rec['expected_neutral_pct']}%")

    # Store detailed results
    for symbol, symbol_results in all_results.items():
        report['detailed_results'][symbol] = {}
        for h, h_results in symbol_results.items():
            report['detailed_results'][symbol][f"h{h}"] = {
                'n_samples': h_results['n_samples'],
                'mean_atr': round(h_results['mean_atr'], 6),
                'median_atr': round(h_results['median_atr'], 6),
                'mean_abs_move': round(h_results['mean_abs_move'], 6),
                'move_atr_ratio': round(h_results['move_atr_ratio'], 4),
                'hit_rates': {k: {kk: round(vv, 2) for kk, vv in v.items()}
                             for k, v in h_results['hit_rates'].items()}
            }

    return report

def main():
    """Main analysis function."""
    print("\n" + "=" * 70)
    print("TRIPLE-BARRIER OPTIMAL PARAMETER ANALYSIS")
    print("=" * 70)
    print(f"Timestamp: {datetime.now().isoformat()}")

    # Load data
    data = load_data()

    # Analyze each symbol
    all_results = {}

    for symbol, df in data.items():
        print(f"\n{'='*70}")
        print(f"ANALYZING {symbol}")
        print("=" * 70)

        # Find ATR column
        atr_col = identify_atr_column(df)
        if atr_col is None:
            print(f"  ERROR: No ATR column found for {symbol}")
            continue

        print(f"  Using ATR column: {atr_col}")

        # Analyze barrier hits
        results = analyze_barrier_hits(df, atr_col, horizons=[1, 5, 20])
        all_results[symbol] = results

    # Generate recommendations
    recommendations = generate_recommendations(all_results)

    # Create summary report
    report = create_summary_report(all_results, recommendations)

    # Save results
    output_path = RESULTS_PATH / "barrier_analysis.json"
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\nResults saved to: {output_path}")

    return report

if __name__ == "__main__":
    report = main()
