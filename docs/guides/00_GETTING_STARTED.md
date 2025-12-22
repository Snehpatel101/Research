# Getting Started with Ensemble Price Prediction

## Quick Links
- [What This Project Does](#what-this-project-does)
- [Quick Start (5 Minutes)](#quick-start)
- [Key Concepts](#key-concepts-explained-simply)
- [Architecture Overview](#architecture-overview)
- [What's Next](#whats-next)

---

## What This Project Does

### The Big Picture

Imagine hiring **three expert traders**, each with a different approach:
- **Trader 1 (N-HiTS)**: Fast, practical, looks at multiple timeframes simultaneously
- **Trader 2 (TFT)**: Methodical analyst who explains exactly why they're making each trade
- **Trader 3 (PatchTST)**: Pattern recognition expert who spots long-term trends

Each trader looks at the same price data (OHLCV = Open, High, Low, Close, Volume) and makes predictions:
- **"Go Long" (+1)**: Price will go UP in the next X bars
- **"Go Short" (-1)**: Price will go DOWN in the next X bars
- **"Stay Neutral" (0)**: No clear signal

Then a **"Manager"** (the Meta-Learner) watches all three traders' track records and learns:
- "When Trader 2 is confident about going long, they're usually right"
- "When all three agree, the signal is much stronger"
- "During high volatility, trust Trader 1 less"

This creates an **ENSEMBLE** - a system that's smarter than any single trader alone.

### Why Ensembles Work

**Analogy**: If you ask one friend for restaurant recommendations, you might get one good option. If you ask three friends with different tastes and combine their suggestions, you'll likely find a place that's broadly excellent.

**In ML terms**:
- Each model has **blind spots** (things it gets wrong)
- Different models have **different blind spots**
- By combining them, the ensemble fills in the gaps

---

## Quick Start

### Prerequisites

```bash
# Python 3.10+
python --version

# Install dependencies
pip install -r requirements_labeling.txt
pip install typer rich
```

### Run Your First Pipeline

```bash
# Option 1: Complete pipeline with defaults
./pipeline run

# Option 2: Custom symbols and date range
./pipeline run --symbols MES,MGC --start 2020-01-01 --end 2024-12-31

# Option 3: Test with synthetic data
./pipeline run --synthetic --run-id first_test
```

### Check Progress

```bash
# View status
./pipeline status first_test

# View logs in real-time
tail -f runs/first_test/logs/pipeline.log
```

### Expected Output

After ~10-15 minutes (for synthetic data), you'll have:

```
data/final/
├── MES_final_labeled.parquet     # Ready for model training
└── MGC_final_labeled.parquet

results/
└── PHASE1_COMPLETION_REPORT_*.md # Summary report

runs/first_test/
├── config/config.json             # Your settings
├── logs/pipeline.log              # Execution log
└── artifacts/manifest.json        # Data versioning
```

### Use the Labels

```python
import pandas as pd
import numpy as np

# Load labeled data
df = pd.read_parquet('data/final/MES_final_labeled.parquet')

# Extract features and labels for horizon 5
exclude_cols = ['datetime', 'symbol', 'open', 'high', 'low', 'close', 'volume']
label_cols = [c for c in df.columns if c.startswith(('label_', 'bars_to_hit_',
                                                       'mae_', 'mfe_', 'touch_type_',
                                                       'quality_', 'sample_weight_'))]
feature_cols = [c for c in df.columns if c not in exclude_cols + label_cols]

X = df[feature_cols].values
y = df['label_h5'].values
weights = df['sample_weight_h5'].values

print(f"Features: {X.shape}")
print(f"Labels: Long={sum(y==1)}, Short={sum(y==-1)}, Neutral={sum(y==0)}")
```

---

## Key Concepts Explained Simply

### 1. OHLCV Data (The Raw Material)

```
Each "bar" (candle) contains:
┌─────────────────────────────┐
│  O = Open price at start    │
│  H = Highest price reached  │
│  L = Lowest price reached   │
│  C = Close price at end     │
│  V = Volume (how much traded)│
└─────────────────────────────┘
```

**For this project:**
- **MES** = Micro E-mini S&P 500 (stock market index)
- **MGC** = Micro Gold (commodity)
- **5-minute bars** resampled from 1-minute data

### 2. Triple-Barrier Labeling (Creating "Answer Keys")

**The Problem**: How do you define "price went up"?
- Up by 0.01%? Too noisy.
- Up by 10%? Never happens.

**The Solution**: Use volatility-adjusted barriers (ATR-based)

```
Entry Price: $100
ATR (volatility measure): $2

Upper Barrier (profit target): $100 + (1.5 × $2) = $103
Lower Barrier (stop loss):     $100 - (1.0 × $2) = $98
Time Barrier: 5 bars maximum

What happens in the next 5 bars?
- Price hits $103 first → Label = +1 (good long trade)
- Price hits $98 first  → Label = -1 (bad trade, stop hit)
- Neither hit           → Label = 0 (no decisive move)
```

**Why ATR**: In calm markets, use tighter barriers. In volatile markets, use wider barriers. This prevents mislabeling noise as signals.

### 3. Quality-Based Sample Weighting

Not all labels are equally reliable:

```python
# High quality (weight = 1.5): Hit barrier quickly and decisively
# Medium quality (weight = 1.0): Normal behavior
# Low quality (weight = 0.5): Barely touched barrier, took maximum time
```

When training models, high-quality samples get more influence.

### 4. Genetic Algorithm Optimization

The GA finds optimal barrier parameters by:
1. Trying different combinations (population of 50)
2. Keeping the best performers (fitness based on balance + speed)
3. Evolving over generations (typically 30-40)
4. Outputting symbol-specific and horizon-specific parameters

### 5. Purging and Embargo

**Purging**: Remove training samples whose labels "see into" the validation/test period

```
Training ends: Dec 31, 2017 23:55
Max horizon: 20 bars = 100 minutes

Purge threshold: Dec 31, 2017 22:15 (subtract 100 min)
Actually remove training samples after this time
```

**Embargo**: Additional buffer (288 bars ≈ 1 day) to prevent information leakage

---

## Architecture Overview

### Phase 1: Data Preparation and Labeling (Current)

```
┌─────────────┐
│ Raw 1m Data │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ Clean 5m    │  ← Resample, remove gaps, outliers
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ +50 Features│  ← RSI, ATR, MACD, Bollinger, etc.
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ GA Optimize │  ← Find best barrier parameters
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ Final Labels│  ← Triple-barrier + quality scores
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ Train/Val/  │  ← 70/15/15 split with purge/embargo
│ Test Splits │
└─────────────┘
```

**Status**: ✅ Phase 1 Complete (Production-Ready)

**Output**: ~50+ features, 3 horizons (H1, H5, H20), quality-weighted labels

### Phase 2: Training Base Models (Next)

Train three specialist models independently:

| Model | Type | Strengths | Training Time |
|-------|------|-----------|---------------|
| N-HiTS | MLP + Hierarchical | Fast, multi-scale | 2-4 hours |
| TFT | Transformer + LSTM | Interpretable | 8-12 hours |
| PatchTST | Patch Transformer | Long-range patterns | 6-10 hours |

**Each model outputs**: Probabilities for [short, neutral, long] × 3 horizons

### Phase 3: Cross-Validation (Getting Unbiased Predictions)

```
Validation set divided into 5 folds:
Fold 1: Train on folds 2-5, predict on fold 1
Fold 2: Train on folds 1,3-5, predict on fold 2
... (repeat for all folds)

Result: Every validation sample has predictions
        that were made WITHOUT seeing that sample during training
```

**Why needed**: Prevents the meta-learner from cheating

### Phase 4: Training the Meta-Learner (Ensemble Magic)

```
     [N-HiTS Probs, TFT Probs, PatchTST Probs]
                      │
                      ▼
                ┌───────────┐
                │Meta-Learner│
                │(Logistic   │
                │Regression) │
                └─────┬─────┘
                      │
                      ▼
              Final Prediction
```

**Expected improvement**: Ensemble beats best single model by Sharpe +0.05 to +0.10

### Phase 5: Final Test (Moment of Truth)

Run ensemble on completely unseen test data and measure:
- Does it generalize to new data?
- Is performance similar to validation? (gap < 15%)
- Any evidence of overfitting or data leakage?

---

## Current Status

### ✅ Phase 1 Complete (Score: 7.5/10)

**Strengths:**
- Triple-barrier labeling with symbol-specific asymmetric barriers (MES: 1.5:1.0)
- GA optimization with transaction cost penalties
- Proper purge (60) and embargo (288) for leakage prevention
- Quality-based sample weighting (0.5x-1.5x)

**Expected Performance (Phase 5 Test Set):**

| Horizon | Sharpe | Win Rate | Max DD |
|---------|--------|----------|--------|
| H5 | 0.3-0.8 | 45-50% | 10-25% |
| H20 | 0.5-1.2 | 48-55% | 8-18% |

### 🚧 Phase 2 Next Steps

Need to implement:
- `TimeSeriesDataset` for PyTorch training
- Model training scripts for N-HiTS, TFT, PatchTST
- Hyperparameter tuning with Optuna

---

## What's Next

### For New Users

1. **Run the Quick Start** (above) - Get familiar with the pipeline
2. **Read the CLI Guide** - Learn all available commands ([01_PIPELINE_CLI.md](01_PIPELINE_CLI.md))
3. **Check the Labeling Guide** - Understand how labeling works ([02_LABELING_GUIDE.md](02_LABELING_GUIDE.md))
4. **Run Validation Tests** - Ensure Phase 1 output is correct ([03_VALIDATION_GUIDE.md](03_VALIDATION_GUIDE.md))

### For Developers

1. **Read Phase Specifications** - Detailed specs for each phase ([/docs/phases/](../phases/))
2. **Review Architecture Docs** - Understand the design ([/docs/reference/architecture/](../reference/architecture/))
3. **Check Code Standards** - See [CLAUDE.md](/CLAUDE.md) for engineering rules

### For Researchers

1. **Review Phase 1 Report** - See comprehensive analysis ([/docs/reference/reviews/PHASE1_COMPREHENSIVE_REVIEW.md](../reference/reviews/PHASE1_COMPREHENSIVE_REVIEW.md))
2. **Compare Pipeline Versions** - See improvements made ([/docs/reference/architecture/PIPELINE_COMPARISON.md](../reference/architecture/PIPELINE_COMPARISON.md))
3. **Check Feature Catalog** - See all implemented features ([/docs/reference/technical/FEATURES_CATALOG.md](../reference/technical/FEATURES_CATALOG.md))

---

## Getting Help

**CLI Help:**
```bash
./pipeline --help
./pipeline run --help
```

**Common Issues:**
- Pipeline fails at feature engineering → Check [Troubleshooting](01_PIPELINE_CLI.md#troubleshooting)
- Labels all neutral → Run GA optimization or adjust barriers
- Data quality issues → See [Validation Guide](03_VALIDATION_GUIDE.md)

**Documentation:**
- User Guides: `/docs/guides/`
- Technical Reference: `/docs/reference/`
- Phase Specs: `/docs/phases/`

---

**Ready to dive deeper?** Continue to:
- [Pipeline CLI Guide](01_PIPELINE_CLI.md) - Complete command reference
- [Labeling Guide](02_LABELING_GUIDE.md) - Advanced labeling techniques
- [Phase 1 Spec](../phases/PHASE_1_Data_Preparation_and_Labeling.md) - Full technical specification
