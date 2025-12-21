import sys
sys.path.insert(0, 'src')

import pandas as pd
import numpy as np
from pathlib import Path
from config import FINAL_DATA_DIR, SPLITS_DIR, TRAIN_RATIO, VAL_RATIO, PURGE_BARS, EMBARGO_BARS
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("splits")

# Wait for labeled files
import time
max_wait = 300  # 5 minutes
waited = 0
while waited < max_wait:
    mes_exists = (FINAL_DATA_DIR / "MES_labeled.parquet").exists()
    mgc_exists = (FINAL_DATA_DIR / "MGC_labeled.parquet").exists()
    if mes_exists and mgc_exists:
        break
    print(f"Waiting for labeled files... ({waited}s)")
    time.sleep(10)
    waited += 10

if waited >= max_wait:
    raise RuntimeError("Timeout waiting for labeled files")

# Load and combine labeled data
print("Loading labeled data...")
dfs = []
for symbol in ['MES', 'MGC']:
    df = pd.read_parquet(FINAL_DATA_DIR / f"{symbol}_labeled.parquet")
    dfs.append(df)

combined_df = pd.concat(dfs, ignore_index=True)
combined_df = combined_df.sort_values('datetime').reset_index(drop=True)
print(f"Combined dataset: {len(combined_df):,} rows")

# Save combined dataset
combined_path = FINAL_DATA_DIR / "combined_final_labeled.parquet"
combined_df.to_parquet(combined_path)
print(f"Saved combined dataset to {combined_path}")

# Create time-based splits using config values
n = len(combined_df)
train_end = int(n * TRAIN_RATIO)
val_end = int(n * (TRAIN_RATIO + VAL_RATIO))

# Apply purging (remove bars from train end to prevent label leakage)
purge_bars = PURGE_BARS
train_end_purged = train_end - purge_bars

# Apply embargo (buffer between splits)
embargo_bars = EMBARGO_BARS
val_start = train_end + embargo_bars

# Apply purge at val_end too (prevents label leakage into test set)
val_end_purged = val_end - purge_bars
test_start = val_end + embargo_bars  # Use original val_end for embargo

# Create indices
train_indices = np.arange(0, train_end_purged)
val_indices = np.arange(val_start, val_end_purged)
test_indices = np.arange(test_start, n)

print(f"\nSplit sizes:")
print(f"  Train: {len(train_indices):,} samples")
print(f"  Val:   {len(val_indices):,} samples")
print(f"  Test:  {len(test_indices):,} samples")

# Get date ranges
train_dates = combined_df.iloc[train_indices]['datetime']
val_dates = combined_df.iloc[val_indices]['datetime']
test_dates = combined_df.iloc[test_indices]['datetime']

print(f"\nDate ranges:")
print(f"  Train: {train_dates.min()} to {train_dates.max()}")
print(f"  Val:   {val_dates.min()} to {val_dates.max()}")
print(f"  Test:  {test_dates.min()} to {test_dates.max()}")

# Save indices
SPLITS_DIR.mkdir(parents=True, exist_ok=True)
np.save(SPLITS_DIR / "train_indices.npy", train_indices)
np.save(SPLITS_DIR / "val_indices.npy", val_indices)
np.save(SPLITS_DIR / "test_indices.npy", test_indices)

# Save metadata
metadata = {
    "total_samples": n,
    "train_samples": len(train_indices),
    "val_samples": len(val_indices),
    "test_samples": len(test_indices),
    "purge_bars": purge_bars,
    "embargo_bars": embargo_bars,
    "train_date_start": str(train_dates.min()),
    "train_date_end": str(train_dates.max()),
    "val_date_start": str(val_dates.min()),
    "val_date_end": str(val_dates.max()),
    "test_date_start": str(test_dates.min()),
    "test_date_end": str(test_dates.max()),
}

with open(SPLITS_DIR / "split_config.json", 'w') as f:
    json.dump(metadata, f, indent=2)

print(f"\nSaved splits to {SPLITS_DIR}")
print("Split creation complete!")
