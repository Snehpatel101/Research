# Phase 1: Data Preparation Pipeline

## What This Does (Plain English)

Phase 1 takes **raw trading data** and transforms it into **clean, labeled datasets** that any machine learning model can use to learn trading patterns.

Think of it like a factory assembly line:

```
Raw 1-minute bars  →  Clean 5-minute bars  →  Add 100+ indicators  →  Label "good" trades  →  Split for training
```

---

## The Data Flow

### Step 1: Ingest Raw Data
**Input:** 1-minute OHLCV bars (Open, High, Low, Close, Volume)
```
2024-01-01 09:30  Open: 4500.25  High: 4501.00  Low: 4499.50  Close: 4500.75  Volume: 1234
2024-01-01 09:31  Open: 4500.75  High: 4502.00  Low: 4500.00  Close: 4501.50  Volume: 987
...
```
**What happens:** Validates the data, fixes timezone issues, removes bad rows.

### Step 2: Resample to 5-Minute Bars
**Why:** 1-minute data is too noisy. 5-minute bars smooth out the noise while keeping enough detail.
```
1-min bars: [09:30, 09:31, 09:32, 09:33, 09:34] → One 5-min bar: 09:30-09:35
```

### Step 3: Add Technical Indicators (Features)
**What:** Calculate 100+ signals that traders use to make decisions:
- **Momentum:** RSI, MACD, Stochastic
- **Trend:** Moving averages (SMA, EMA), Supertrend
- **Volatility:** ATR, Bollinger Bands
- **Volume:** OBV, VWAP

### Step 4: Multi-Timeframe Features
**The key insight:** What happens on the 1-hour chart affects the 5-minute chart.

We resample the data to higher timeframes and bring those signals back:
```
5-minute (base)  →  15-minute  →  30-minute  →  1-hour  →  4-hour  →  Daily
```

Each timeframe adds its own perspective:
- **15min/30min:** Short-term momentum
- **1h/4h:** Medium-term trend
- **Daily:** Big picture direction

### Step 5: Label the Data (Triple-Barrier Method)
**The question:** "If I bought here, would it be a good trade?"

For each bar, we look into the future and ask:
- Did price hit my **profit target** first? → Label: LONG (1)
- Did price hit my **stop loss** first? → Label: SHORT (-1)
- Did it just sit there until timeout? → Label: NEUTRAL (0)

```
          Profit Target (+1.5%)
               ┌────────────────────
               │
    Entry ─────┼──────────────────── Time
               │
               └────────────────────
          Stop Loss (-1.0%)
```

We do this for multiple **horizons** (5, 10, 15, 20 bars ahead) because some models predict short-term, others long-term.

### Step 6: Optimize Labels with Genetic Algorithm
**Problem:** Fixed barrier sizes don't work for all market conditions.
**Solution:** Use a genetic algorithm to find the best barrier sizes for each symbol.

MES (S&P futures) might need different settings than MGC (Gold futures).

### Step 7: Split into Train/Validation/Test
**Critical:** This is time-series data. You can't shuffle it randomly.

```
[========= Training (70%) =========][== Val (15%) ==][== Test (15%) ==]
        Jan 2020 - Dec 2022          Jan-Jun 2023      Jul-Dec 2023
```

**Purge gap:** We leave a buffer between splits so the model can't cheat by seeing "almost future" data.

### Step 8: Scale the Features
**Why:** Neural networks train better when all features are on similar scales.

```
Before: RSI ranges 0-100, Price ranges 4000-5000, Volume ranges 0-1,000,000
After:  All features centered around 0, mostly between -3 and +3
```

**Important:** We fit the scaler ONLY on training data, then apply it to val/test. This prevents data leakage.

---

## What Comes Out

```
data/splits/scaled/
├── train_scaled.parquet   # 70% of data - train your model on this
├── val_scaled.parquet     # 15% of data - tune hyperparameters with this
└── test_scaled.parquet    # 15% of data - final evaluation (touch once!)
```

Each file contains:
- **129 features** (technical indicators + multi-timeframe)
- **Labels** for 4 horizons (H5, H10, H15, H20)
- **Quality weights** (some samples are higher quality than others)
- **Metadata** (symbol, timestamp)

---

## What This Sets Up (Phase 2 and Beyond)

Phase 1 is the **foundation**. The output format works with ANY model type:

### Phase 2: Model Factory
Train any model on the Phase 1 data:
```python
from src.phase1.stages.datasets import TimeSeriesDataContainer

container = TimeSeriesDataContainer.from_parquet_dir('data/splits/scaled', horizon=20)

# For XGBoost/LightGBM (tabular)
X_train, y_train, weights = container.get_sklearn_arrays('train')

# For LSTM/Transformer (sequences)
dataset = container.get_pytorch_sequences('train', seq_len=64)

# For N-HiTS/TFT (neuralforecast)
df = container.get_neuralforecast_df('train')
```

### Phase 3: Cross-Validation
Walk-forward validation, purged k-fold, out-of-sample predictions.

### Phase 4: Ensemble
Combine multiple models into a meta-learner that's smarter than any single model.

### Phase 5: Production
Real-time inference, monitoring, A/B testing live strategies.

### Phase 6: Orchestrator
Central dashboard to manage everything.

---

## Key Design Decisions

| Decision | Why |
|----------|-----|
| 5-minute bars | Balance between noise reduction and signal preservation |
| 60-bar purge | Prevents label leakage (labels look 20 bars ahead × 3 safety margin) |
| Multi-timeframe | Higher timeframes provide context that single-timeframe misses |
| Triple-barrier | More realistic than simple up/down labels - includes stop losses |
| Robust scaling | Handles outliers better than standard scaling |
| Train-only fit | Scaler fitted on train data only to prevent data leakage |

---

## Quick Start

```bash
# Run the full pipeline
./pipeline run --symbols MES,MGC

# Check outputs
ls -lh data/splits/scaled/

# Use in Python
python -c "
from src.phase1.stages.datasets import TimeSeriesDataContainer
container = TimeSeriesDataContainer.from_parquet_dir('data/splits/scaled', horizon=20)
X, y, w = container.get_sklearn_arrays('train')
print(f'Ready to train: {X.shape[0]} samples, {X.shape[1]} features')
"
```

---

## Summary

**Phase 1 turns raw price data into model-ready datasets.**

It handles all the tedious but critical data engineering:
- Cleaning and validation
- Feature engineering (100+ indicators)
- Multi-timeframe analysis (5min to daily)
- Realistic trade labeling
- Proper train/val/test splits
- Feature scaling without leakage

The result is clean, labeled data that any ML model can consume - from simple random forests to complex transformers.
