#!/usr/bin/env python3
import pandas as pd

df = pd.read_parquet("/Users/sneh/research/data/splits/final_correct/scaled/train_scaled.parquet")

ohlcv = ["open", "high", "low", "close", "volume"]

print("OHLCV columns present:")
for col in ohlcv:
    print(f"  {col}: {col in df.columns}")

print("\nOHLCV sample stats:")
for col in ohlcv:
    if col in df.columns:
        print(f"  {col}: mean={df[col].mean():.2f}, std={df[col].std():.2f}")
