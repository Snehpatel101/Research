#!/usr/bin/env python3
"""
Phase 1 Complete Pipeline Runner
Orchestrates the entire data preparation pipeline
"""
import sys
sys.path.insert(0, 'src')

import logging
from pathlib import Path
from datetime import datetime
import json

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("phase1")


def main():
    """Run the complete Phase 1 pipeline."""
    start_time = datetime.now()
    logger.info("="*70)
    logger.info("PHASE 1: DATA PREPARATION PIPELINE")
    logger.info("="*70)

    from config import (
        RAW_DATA_DIR, CLEAN_DATA_DIR, FEATURES_DIR,
        FINAL_DATA_DIR, SPLITS_DIR, SYMBOLS, RESULTS_DIR,
        RANDOM_SEED, set_global_seeds
    )

    # Set random seeds for reproducibility across the entire pipeline
    set_global_seeds(RANDOM_SEED)
    logger.info(f"Random seed set to {RANDOM_SEED} for reproducibility")

    # Step 1: Generate synthetic data (if no real data exists)
    logger.info("\n" + "="*60)
    logger.info("STEP 1: Data Generation / Acquisition")
    logger.info("="*60)

    from generate_synthetic_data import main as generate_data
    raw_files_exist = all(
        (RAW_DATA_DIR / f"{s}_1m.parquet").exists() or
        (RAW_DATA_DIR / f"{s}_1m.csv").exists()
        for s in SYMBOLS
    )

    if not raw_files_exist:
        logger.info("No raw data found. Generating synthetic data...")
        generate_data()
    else:
        logger.info("Raw data files already exist. Skipping generation.")

    # Step 2: Data Cleaning
    logger.info("\n" + "="*60)
    logger.info("STEP 2: Data Cleaning")
    logger.info("="*60)

    from stages.stage2_clean import main as clean_data
    clean_data()

    # Step 3: Feature Engineering
    logger.info("\n" + "="*60)
    logger.info("STEP 3: Feature Engineering")
    logger.info("="*60)

    from feature_engineering import main as generate_features
    generate_features()

    # Step 4: Triple-Barrier Labeling
    logger.info("\n" + "="*60)
    logger.info("STEP 4: Triple-Barrier Labeling")
    logger.info("="*60)

    from stages.stage4_labeling import main as apply_labels
    apply_labels()

    # Step 5: Create Splits
    logger.info("\n" + "="*60)
    logger.info("STEP 5: Create Train/Val/Test Splits")
    logger.info("="*60)

    import pandas as pd
    import numpy as np
    from config import TRAIN_RATIO, VAL_RATIO, PURGE_BARS, EMBARGO_BARS

    # Load and combine labeled data
    dfs = []
    for symbol in SYMBOLS:
        fpath = FINAL_DATA_DIR / f"{symbol}_labeled.parquet"
        if fpath.exists():
            df = pd.read_parquet(fpath)
            dfs.append(df)
            logger.info(f"Loaded {len(df):,} rows for {symbol}")

    if not dfs:
        raise RuntimeError("No labeled data found!")

    combined_df = pd.concat(dfs, ignore_index=True)
    combined_df = combined_df.sort_values('datetime').reset_index(drop=True)
    logger.info(f"Combined dataset: {len(combined_df):,} rows")

    # Save combined dataset
    combined_path = FINAL_DATA_DIR / "combined_final_labeled.parquet"
    combined_df.to_parquet(combined_path, index=False)
    logger.info(f"Saved combined dataset to {combined_path}")

    # Create splits
    n = len(combined_df)
    train_end = int(n * TRAIN_RATIO)
    val_end = int(n * (TRAIN_RATIO + VAL_RATIO))

    # Apply purging
    train_end_purged = train_end - PURGE_BARS
    val_start = train_end + EMBARGO_BARS
    test_start = val_end + EMBARGO_BARS

    # Create indices
    train_indices = np.arange(0, train_end_purged)
    val_indices = np.arange(val_start, val_end)
    test_indices = np.arange(test_start, n)

    logger.info(f"Split sizes:")
    logger.info(f"  Train: {len(train_indices):,} samples")
    logger.info(f"  Val:   {len(val_indices):,} samples")
    logger.info(f"  Test:  {len(test_indices):,} samples")

    # Get date ranges
    train_dates = combined_df.iloc[train_indices]['datetime']
    val_dates = combined_df.iloc[val_indices]['datetime']
    test_dates = combined_df.iloc[test_indices]['datetime']

    logger.info(f"Date ranges:")
    logger.info(f"  Train: {train_dates.min()} to {train_dates.max()}")
    logger.info(f"  Val:   {val_dates.min()} to {val_dates.max()}")
    logger.info(f"  Test:  {test_dates.min()} to {test_dates.max()}")

    # Save indices
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    np.save(SPLITS_DIR / "train_indices.npy", train_indices)
    np.save(SPLITS_DIR / "val_indices.npy", val_indices)
    np.save(SPLITS_DIR / "test_indices.npy", test_indices)

    # Save metadata
    split_config = {
        "total_samples": n,
        "train_samples": len(train_indices),
        "val_samples": len(val_indices),
        "test_samples": len(test_indices),
        "purge_bars": PURGE_BARS,
        "embargo_bars": EMBARGO_BARS,
        "train_date_start": str(train_dates.min()),
        "train_date_end": str(train_dates.max()),
        "val_date_start": str(val_dates.min()),
        "val_date_end": str(val_dates.max()),
        "test_date_start": str(test_dates.min()),
        "test_date_end": str(test_dates.max()),
    }

    with open(SPLITS_DIR / "split_config.json", 'w') as f:
        json.dump(split_config, f, indent=2)

    logger.info(f"Saved splits to {SPLITS_DIR}")

    # Step 6: Generate Report
    logger.info("\n" + "="*60)
    logger.info("STEP 6: Generate Completion Report")
    logger.info("="*60)

    # Count features
    feature_cols = [c for c in combined_df.columns
                    if c not in ['datetime', 'symbol', 'open', 'high', 'low', 'close', 'volume']
                    and not c.startswith('label_') and not c.startswith('bars_to_hit_')
                    and not c.startswith('mae_') and not c.startswith('quality_')
                    and not c.startswith('sample_weight_')]

    # Label statistics
    label_stats = {}
    for horizon in [1, 5, 20]:
        col = f'label_h{horizon}'
        if col in combined_df.columns:
            counts = combined_df[col].value_counts().sort_index()
            label_stats[horizon] = {
                'short': int(counts.get(-1, 0)),
                'neutral': int(counts.get(0, 0)),
                'long': int(counts.get(1, 0))
            }

    # Generate report
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    report = f'''# Phase 1 Completion Report
## Ensemble Price Prediction System

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---

## Executive Summary

Phase 1 successfully processed raw OHLCV data through the complete pipeline:
- Data Cleaning (1-min to 5-min resampling)
- Feature Engineering ({len(feature_cols)} technical features)
- Triple-Barrier Labeling (3 horizons: 1, 5, 20 bars)
- Train/Val/Test Splits (with purging & embargo)

---

## Dataset Overview

| Metric | Value |
|--------|-------|
| **Total Samples** | {len(combined_df):,} |
| **Symbols** | {', '.join(SYMBOLS)} |
| **Date Range** | {combined_df['datetime'].min()} to {combined_df['datetime'].max()} |
| **Resolution** | 5-minute bars |
| **Features** | {len(feature_cols)} |

---

## Label Distribution

### Horizon 1 (1-bar ahead)
| Class | Count | Percentage |
|-------|-------|------------|
| Short (-1) | {label_stats.get(1, {}).get('short', 0):,} | {label_stats.get(1, {}).get('short', 0)/len(combined_df)*100:.1f}% |
| Neutral (0) | {label_stats.get(1, {}).get('neutral', 0):,} | {label_stats.get(1, {}).get('neutral', 0)/len(combined_df)*100:.1f}% |
| Long (+1) | {label_stats.get(1, {}).get('long', 0):,} | {label_stats.get(1, {}).get('long', 0)/len(combined_df)*100:.1f}% |

### Horizon 5 (5-bar ahead)
| Class | Count | Percentage |
|-------|-------|------------|
| Short (-1) | {label_stats.get(5, {}).get('short', 0):,} | {label_stats.get(5, {}).get('short', 0)/len(combined_df)*100:.1f}% |
| Neutral (0) | {label_stats.get(5, {}).get('neutral', 0):,} | {label_stats.get(5, {}).get('neutral', 0)/len(combined_df)*100:.1f}% |
| Long (+1) | {label_stats.get(5, {}).get('long', 0):,} | {label_stats.get(5, {}).get('long', 0)/len(combined_df)*100:.1f}% |

### Horizon 20 (20-bar ahead)
| Class | Count | Percentage |
|-------|-------|------------|
| Short (-1) | {label_stats.get(20, {}).get('short', 0):,} | {label_stats.get(20, {}).get('short', 0)/len(combined_df)*100:.1f}% |
| Neutral (0) | {label_stats.get(20, {}).get('neutral', 0):,} | {label_stats.get(20, {}).get('neutral', 0)/len(combined_df)*100:.1f}% |
| Long (+1) | {label_stats.get(20, {}).get('long', 0):,} | {label_stats.get(20, {}).get('long', 0)/len(combined_df)*100:.1f}% |

---

## Data Splits

| Split | Samples | Percentage | Date Range |
|-------|---------|------------|------------|
| **Train** | {split_config['train_samples']:,} | {split_config['train_samples']/split_config['total_samples']*100:.1f}% | {split_config['train_date_start'][:10]} to {split_config['train_date_end'][:10]} |
| **Validation** | {split_config['val_samples']:,} | {split_config['val_samples']/split_config['total_samples']*100:.1f}% | {split_config['val_date_start'][:10]} to {split_config['val_date_end'][:10]} |
| **Test** | {split_config['test_samples']:,} | {split_config['test_samples']/split_config['total_samples']*100:.1f}% | {split_config['test_date_start'][:10]} to {split_config['test_date_end'][:10]} |

### Leakage Prevention
- **Purge bars:** {split_config['purge_bars']} bars removed at boundaries
- **Embargo period:** {split_config['embargo_bars']} bars (~1 day) buffer

---

## Feature Categories ({len(feature_cols)} total)

| Category | Features |
|----------|----------|
| Price | log_return, simple_return, high_low_range, close_open_range |
| Moving Averages | SMA (10,20,50,100,200), EMA (9,21,50) |
| Momentum | RSI, MACD, Stochastic, ROC, Williams %R |
| Volatility | ATR (7,14,21), Bollinger Bands |
| Volume | OBV, Volume Z-score, VWAP |
| Trend | ADX, +DI/-DI |
| Regime | vol_regime, trend_regime |
| Temporal | hour_sin/cos, dow_sin/cos, is_rth |

---

## Output Files

### Data Files
- `data/clean/MES_5m_clean.parquet`
- `data/clean/MGC_5m_clean.parquet`
- `data/features/MES_5m_features.parquet`
- `data/features/MGC_5m_features.parquet`
- `data/final/MES_labeled.parquet`
- `data/final/MGC_labeled.parquet`
- `data/final/combined_final_labeled.parquet`

### Split Files
- `data/splits/train_indices.npy`
- `data/splits/val_indices.npy`
- `data/splits/test_indices.npy`
- `data/splits/split_config.json`

---

## Next Steps: Phase 2

1. Load training data with splits
2. Train base models (N-HiTS, TFT, PatchTST)
3. Use sample weights for quality-aware training

```python
import numpy as np
import pandas as pd

# Load data and splits
train_idx = np.load('data/splits/train_indices.npy')
df = pd.read_parquet('data/final/combined_final_labeled.parquet')
train_df = df.iloc[train_idx]

# Get features and labels
feature_cols = [c for c in train_df.columns
                if c not in ['datetime', 'symbol', 'open', 'high', 'low', 'close', 'volume']
                and not c.startswith('label_') and not c.startswith('bars_to_hit_')
                and not c.startswith('mae_') and not c.startswith('quality_')
                and not c.startswith('sample_weight_')]

X_train = train_df[feature_cols].values
y_train = train_df['label_h5'].values  # For 5-bar horizon
sample_weights = train_df['sample_weight_h5'].values
```

---

*Phase 1 Complete - Ready for Phase 2*
'''

    report_path = RESULTS_DIR / "PHASE1_COMPLETION_REPORT.md"
    with open(report_path, 'w') as f:
        f.write(report)

    logger.info(f"Report saved to: {report_path}")

    # Final Summary
    elapsed = datetime.now() - start_time
    logger.info("\n" + "="*70)
    logger.info("PHASE 1 COMPLETE!")
    logger.info("="*70)
    logger.info(f"Total time: {elapsed}")
    logger.info(f"Total samples: {len(combined_df):,}")
    logger.info(f"Total features: {len(feature_cols)}")
    logger.info(f"Report: {report_path}")


if __name__ == "__main__":
    main()
