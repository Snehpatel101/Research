# Layman's Guide: AI Ensemble Training for OHLCV Sequential Trading

## Table of Contents
1. [The Big Picture](#the-big-picture)
2. [Key Concepts Explained Simply](#key-concepts-explained-simply)
3. [Phase-by-Phase Overview](#phase-by-phase-overview)
4. [Model Architectures (TFT vs TCN vs Others)](#model-architectures)
5. [Data & Features](#data-and-features)
6. [Infrastructure (Google Colab, Databases)](#infrastructure)
7. [Critical Implementation Warnings](#critical-warnings)
8. [Quick Reference Cheat Sheet](#cheat-sheet)

---

## The Big Picture

### What Are We Building?

Imagine you're hiring **three expert traders**, each with a different approach:
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

For your project:
- **MES** = Micro E-mini S&P 500 (stock market index)
- **MGC** = Micro Gold (commodity)
- **20 years** of 5-minute bars = ~2-5 million data points per instrument

### 2. Sequential Training (Why Order Matters)

**Wrong way (Image Classification)**: Shuffle photos randomly, model learns from any photo in any order.

**Right way (Time Series)**: You MUST preserve chronological order. The model learns patterns like:
- "When price does X, it usually does Y within the next 5 bars"
- The model CANNOT see the future during training (no lookahead)

```
Timeline: ───────────────────────────────────────────►
          [Training Data]──[Gap]──[Validation]──[Gap]──[Test]
                                                        ↑
                                            Never touched until final exam
```

### 3. Lookback Windows (The Model's "Memory")

Models don't just see one bar - they see a **sequence** of recent bars:

```
Current bar: 128
Model sees bars 1-128 (the "lookback window")
Makes prediction about bars 129, 133, 148 (1, 5, 20 bars ahead)

   Lookback Window (128 bars)        Prediction Horizon
   ▼──────────────────────▼          ▼───────────────▼
   [1][2][3]...[126][127][128] → → → [129]...[133]...[148]
                                       ↑       ↑       ↑
                                     1-bar   5-bar   20-bar
```

### 4. Triple-Barrier Labeling (Creating "Answer Keys")

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

### 5. Ensembles (The Core Concept)

**Level 0 - Base Models**: Three independent experts making predictions

```
       ┌─────────────┐
       │  OHLCV      │
       │  Features   │
       └──────┬──────┘
              │
     ┌────────┼────────┐
     ▼        ▼        ▼
┌─────────┐ ┌────┐ ┌──────────┐
│ N-HiTS  │ │TFT │ │ PatchTST │
└────┬────┘ └──┬─┘ └────┬─────┘
     │         │        │
     ▼         ▼        ▼
   Probs_1   Probs_2   Probs_3
   [0.2,     [0.1,     [0.15,
    0.3,      0.4,      0.35,
    0.5]      0.5]      0.50]
```

**Level 1 - Meta-Learner**: Learns optimal weights for combining base models

```
     [Probs_1, Probs_2, Probs_3]
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
        [0.12, 0.38, 0.50] → Class +1 (Long)
```

### 6. Meta-Labeling (Advanced Ensemble Concept)

**Standard Labeling**: Predict direction (up/down/neutral)

**Meta-Labeling** (mentioned in your notes):
1. First model predicts: "I think it's going up"
2. Second model (meta-model) predicts: "Should we trust that prediction?"

This allows:
- Filtering out low-confidence trades
- Adjusting position size based on confidence
- Reducing false positives

### 7. Reinforcement Learning (RL) Layer

**What it does**: Instead of predicting price direction, RL learns a **trading policy**:
- When to enter (and how much)
- When to exit (take profit or stop loss)
- Risk management (position sizing)

**RL in your ensemble** (mentioned as "RL MLP TCN, RISK layer"):
```
Base Model Predictions → RL Agent → Actual Trading Decisions
                              │
                              └── Considers: Portfolio risk,
                                            Transaction costs,
                                            Current positions
```

**Why add RL**: A prediction of "+1" doesn't mean "bet everything". RL learns nuanced decisions:
- "Prediction is +1, but recent trades lost money, reduce size"
- "Prediction is +1, all models agree, increase size"

---

## Phase-by-Phase Overview

### Phase 1: Data Preparation and Labeling
**Analogy**: Creating the exam questions and answer key

**What happens**:
1. Clean 20 years of OHLCV data (remove gaps, outliers, handle contract rolls)
2. Create **features** (RSI, ATR, MACD, VIX proxy, etc.) - ~40-80 indicators
3. Use **Genetic Algorithm** to optimize triple-barrier parameters
4. Create labels (+1, 0, -1) for three horizons (1-bar, 5-bar, 20-bar)
5. Split data: 70% train, 15% validation, 15% test (chronological, with purging)

**Key concept - Purging**: Remove training samples whose labels "see into" the test period

```
Training ends: Dec 31, 2017 23:55
Max horizon: 20 bars = 100 minutes

Purge threshold: Dec 31, 2017 22:15 (subtract 100 min)
Actually remove training samples after this time

This prevents: Training sample at 23:00 with label looking at 23:00+20bars=24:40
(which would be in 2018, potentially test data)
```

### Phase 2: Training Base Models
**Analogy**: Training each expert trader individually

**Three models trained**:

| Model | Type | Strengths | Time to Train |
|-------|------|-----------|---------------|
| N-HiTS | MLP + Hierarchical | Fast, multi-scale patterns | 2-4 hours |
| TFT | Transformer + LSTM | Interpretable, feature importance | 8-12 hours |
| PatchTST | Patch Transformer | Long-range patterns, state-of-art | 6-10 hours |

**Each model outputs**: Probabilities for 3 classes × 3 horizons
- P(short), P(neutral), P(long) for 1-bar
- P(short), P(neutral), P(long) for 5-bar
- P(short), P(neutral), P(long) for 20-bar

### Phase 3: Cross-Validation (Getting Unbiased Predictions)
**Analogy**: Running practice exams under test conditions

**Why needed**: To train the meta-learner, we need predictions that weren't "seen" during training.

**How it works**:
```
Validation set divided into 5 folds:
Fold 1: Train on folds 2-5, predict on fold 1
Fold 2: Train on folds 1,3-5, predict on fold 2
... (repeat for all folds)

Result: Every validation sample has predictions
        that were made WITHOUT seeing that sample during training
```

**This prevents**: The meta-learner cheating by memorizing which model is "right" on samples it was trained on.

### Phase 4: Training the Meta-Learner (Ensemble Magic)
**Analogy**: Teaching the manager to listen to the three traders

**Input**: 9 probability values per sample (3 models × 3 classes)
**Output**: Final prediction (which class)

**Why Logistic Regression works**: It learns simple weights like:
- "For long signals, trust TFT 40%, PatchTST 35%, N-HiTS 25%"
- These weights are interpretable!

**Expected improvement**: Ensemble beats best single model by Sharpe +0.05 to +0.10

### Phase 5: Final Test (Moment of Truth)
**Analogy**: The real exam - unseen questions

**Critical rule**: Test set has NEVER been touched. No peeking, no tuning, no iterations.

**What we measure**:
- Does ensemble generalize to truly new data?
- Is performance similar to validation? (gap < 15%)
- Any evidence of overfitting or data leakage?

---

## Model Architectures

### TFT (Temporal Fusion Transformer)

**What it is**: A transformer designed specifically for time series

**Key components**:
1. **LSTM Encoder**: Processes sequential data, remembers patterns
2. **Variable Selection**: Learns which features are important
3. **Multi-Head Attention**: Finds relationships across time

**Strengths**:
- Excellent on heterogeneous features (mix of indicators)
- Provides **interpretability** (which features mattered)
- Designed for multi-horizon forecasting

**Weaknesses**:
- Slower to train (8-12 hours)
- More complex to tune

**When to use**: When you have many features and want to understand what's driving predictions

### TCN (Temporal Convolutional Network)

**What it is**: Uses convolutions (like image processing) on time series

**Key concept - Dilated Convolutions**:
```
Regular CNN: looks at 3 adjacent bars
                [●][●][●]

Dilated CNN: looks at bars with gaps (larger receptive field)
        dilation=1: [●][●][●]
        dilation=2: [●][ ][●][ ][●]
        dilation=4: [●][ ][ ][ ][●][ ][ ][ ][●]
```

**Strengths**:
- Very fast (parallelizable unlike LSTM)
- No vanishing gradient (handles long sequences)
- Simple architecture

**Weaknesses**:
- Less interpretable than TFT
- Fixed receptive field

**TFT vs TCN Summary**:
| Aspect | TFT | TCN |
|--------|-----|-----|
| Speed | Slower | Faster |
| Interpretability | High | Low |
| Long-range patterns | Good | Very Good |
| Feature selection | Built-in | Manual |
| Best for | Multi-horizon + explanations | Pure speed/performance |

### N-HiTS (Neural Hierarchical Interpolation)

**What it is**: MLP-based model with multi-scale processing

**How it works**:
```
Input → Stack 1 (fine details, fast patterns)
     → Stack 2 (medium patterns)
     → Stack 3 (coarse patterns, slow trends)
     → Combine all scales
```

**Strengths**:
- **Very fast** (50× faster than transformers)
- Multi-scale naturally captures different timeframes
- Good baseline model

### PatchTST (Patch Time Series Transformer)

**What it is**: State-of-the-art transformer using "patches"

**Key innovation - Patching**:
```
Instead of: each bar = 1 token (128 tokens for 128 bars)
Use:        each patch of 16 bars = 1 token (8 tokens for 128 bars)

This makes transformers practical for long sequences
```

**Strengths**:
- ICLR 2023 state-of-the-art
- Excellent long-range dependencies
- Handles long lookbacks efficiently

### LSTM (Long Short-Term Memory)

**What it is**: Recurrent neural network with "memory cells"

```
Classic RNN problem: forgets early inputs (vanishing gradient)

LSTM solution: Gates control information flow
- Forget Gate: What to forget
- Input Gate: What new info to add
- Output Gate: What to output

Like a person who:
- Decides what to remember/forget
- Updates mental notes
- Chooses what to say based on notes
```

**Role in your system**: TFT uses LSTM as its encoder

### RISK Layer (RL-based Position Sizing)

**Concept**: After getting ensemble predictions, use RL to decide:
1. Should we trade at all?
2. How much to bet?
3. Where to set stop loss?

```
Ensemble says: +1 (Long) with 70% confidence

RISK Layer considers:
- Current drawdown: 5%  → reduce risk
- Recent win streak: 3  → slight increase
- Volatility: High      → reduce size
- Correlation: All models agree → increase confidence

Final decision: Go long with 0.6 units (not full 1.0)
```

---

## Data and Features

### What Features Matter for Price Prediction?

**Your base model needs features that capture**:

1. **Momentum** (is price moving in a direction?)
   - RSI (Relative Strength Index)
   - MACD (Moving Average Convergence Divergence)
   - ROC (Rate of Change)

2. **Volatility** (how much is price moving?)
   - ATR (Average True Range) - **CRITICAL for your labeling**
   - Bollinger Band Width
   - Historical Volatility

3. **Trend** (what's the longer-term direction?)
   - Moving Averages (SMA, EMA)
   - ADX (trend strength)
   - Price vs MA distance

4. **Volume** (confirmation of moves)
   - Volume SMA
   - OBV (On-Balance Volume)
   - Volume spikes

5. **Market Regime** (what "mood" is the market in?)
   - VIX (fear index) or proxy
   - Volatility regime (low/medium/high)
   - Trend regime (up/down/sideways)

6. **Temporal** (time patterns)
   - Hour of day (encoded as sin/cos)
   - Day of week
   - Is it regular trading hours?

**Total**: 40-80 features recommended

### Optuna for Hyperparameter Tuning

**What it is**: Automated hyperparameter search

**Example**: Instead of guessing "learning_rate = 0.001":
```python
def objective(trial):
    lr = trial.suggest_loguniform('lr', 1e-5, 1e-2)
    hidden_size = trial.suggest_categorical('hidden', [64, 128, 256])
    dropout = trial.suggest_float('dropout', 0.1, 0.5)

    model = create_model(lr, hidden_size, dropout)
    score = train_and_evaluate(model)
    return score

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)
```

Optuna intelligently explores the search space, focusing on promising regions.

---

## Infrastructure

### Google Colab (Free/Pro)

**What you get**:
- Free: ~12GB GPU RAM, ~12h sessions, disconnections
- Pro ($10/month): Better GPUs, longer sessions, priority
- Pro+ ($50/month): V100/A100 GPUs, 24h sessions

**For your project**:
- Phase 2 training: Pro minimum (need ~16GB GPU RAM for TFT)
- Phase 3 CV: Pro+ ideal (15 training runs)

**Tips**:
- Save checkpoints frequently to Google Drive
- Use smaller batch sizes if OOM
- Keep sessions alive with periodic interactions

### Parquet Files (Best for ML Training)

**Why not CSV?**
- CSV: Slow to read, large files, no type preservation
- Parquet: Fast, compressed, preserves types

```python
# Writing
df.to_parquet('data.parquet', compression='snappy')

# Reading (fast!)
df = pd.read_parquet('data.parquet')

# Selective reading (even faster!)
df = pd.read_parquet('data.parquet', columns=['close', 'volume'])
```

**For 20 years of data**: Parquet can be 10× smaller and 50× faster to load

### Database Choices

**DuckDB (Recommended for analytics)**:
- In-process (no server needed)
- SQL interface
- Excellent for Parquet files
- Great for local development

```python
import duckdb

conn = duckdb.connect('trading.db')
conn.execute("CREATE TABLE bars AS SELECT * FROM read_parquet('data/*.parquet')")
result = conn.execute("SELECT * FROM bars WHERE symbol='MES' AND atr > 5").fetchdf()
```

**TimescaleDB (Recommended for production)**:
- PostgreSQL extension for time-series
- Automatic time-based partitioning
- Compression for old data
- Great for live data streaming

**Comparison**:
| Aspect | DuckDB | TimescaleDB |
|--------|--------|-------------|
| Setup | Zero (embedded) | Requires server |
| Speed (analytics) | Very fast | Fast |
| Live streaming | Not designed | Excellent |
| Production use | Development | Production |

**Your workflow**:
1. Development: Parquet files + DuckDB
2. Production: TimescaleDB for live data → Parquet export for training

---

## Critical Implementation Warnings

### 1. Symbol Mixing in Sequences (CRITICAL)

**The bug mentioned**:
```python
# WRONG: Can mix MES and MGC in same sequence
seq = df.iloc[i-128:i]  # If df has both symbols!

# RIGHT: Ensure single symbol per sequence
seq = df[df['symbol'] == symbol].iloc[i-128:i]
```

**Fix**: Build sequences per-symbol or filter during creation

### 2. reset_index() Hiding Gaps (CRITICAL)

**The bug**:
```python
# WRONG: Gaps become invisible
df = df.iloc[valid_indices].reset_index(drop=True)
# Now df looks continuous but actually has time jumps!

# RIGHT: Keep original index or check timestamp continuity
df['time_diff'] = df['timestamp'].diff()
assert df['time_diff'].max() < pd.Timedelta('6min')  # Expected 5min bars
```

### 3. Lookahead Bias (CRITICAL)

**Types of lookahead to watch for**:
1. **Label lookahead**: Training sample's label uses future data (Triple-barrier looks forward!)
2. **Feature lookahead**: Indicator uses future data (some implementations are buggy)
3. **Scaling lookahead**: Normalizing using full dataset statistics

**Fix for labels**: Purging (remove training samples near test boundary)

### 4. Over-Engineering the Meta-Learner

**Temptation**: "Let's use a deep neural network for the meta-learner!"

**Reality**: Simple logistic regression usually works best. Complex meta-learners overfit because:
- Only 9 input features (model probabilities)
- Base models already learned patterns
- Too much flexibility = fitting noise

---

## Quick Reference Cheat Sheet

### Key Terms

| Term | Simple Definition |
|------|-------------------|
| **Ensemble** | Multiple models working together |
| **Meta-Learner** | Model that combines other models |
| **Stacking** | Using predictions as features for another model |
| **OHLCV** | Open, High, Low, Close, Volume |
| **Lookback** | How many past bars model sees |
| **Horizon** | How many bars ahead we predict |
| **Triple-Barrier** | Labeling method using profit target + stop loss + time limit |
| **Purging** | Removing samples that could leak future info |
| **Embargo** | Additional buffer zone around test data |
| **ATR** | Average True Range (volatility measure) |
| **F1 Score** | Balance of precision and recall |
| **Sharpe Ratio** | Return per unit of risk |

### Expected Performance Targets

| Metric | Phase 2 (Single) | Phase 4 (Ensemble) | Phase 5 (Test) |
|--------|-----------------|-------------------|----------------|
| F1 | 0.38-0.46 | 0.46-0.48 | >0.35 |
| Sharpe | 0.45-0.65 | 0.68-0.72 | >0.40 |
| Max DD | 8-12% | 7-9% | <15% |

### File Structure (What You'll Create)

```
Research/
├── data/
│   ├── clean/          # Cleaned OHLCV
│   ├── features/       # With indicators
│   ├── final/          # With labels
│   └── stacking/       # CV predictions
├── config/
│   ├── ga_results/     # Optimized barriers
│   ├── splits/         # Train/val/test indices
│   └── cv_folds.json
├── models/
│   ├── nhits_final.pth
│   ├── tft_final.pth
│   ├── patchtst_final.pth
│   └── meta_learner_*.pkl
├── predictions/
│   └── *_val_probs_*.npy
└── reports/
    └── phase*_*.html/csv/md
```

### Training Time Estimates

| Phase | Duration | Hardware |
|-------|----------|----------|
| Phase 1 | 1-2 weeks | CPU + 64GB RAM |
| Phase 2 | 1-2 weeks | 1× GPU (16GB) |
| Phase 3 | 1-2 weeks | 1× GPU (15 runs) |
| Phase 4 | 1-2 days | CPU only |
| Phase 5 | 2-3 days | 1× GPU |
| **Total** | **4-7 weeks** | |

---

## Next Steps

1. **Get Data**: Acquire 20 years of MES and MGC 5-minute bars
2. **Set Up Environment**: Google Colab Pro or local GPU
3. **Start Phase 1**: Follow the detailed spec in your repo
4. **Iterate**: Each phase builds on the previous

**Remember**: This is a skeleton framework. The models will improve as you:
- Add more features
- Tune hyperparameters
- Collect more diverse training data
- Add specialized models for different regimes

---

*This guide accompanies the 5-Phase specification documents in the Research repository.*
