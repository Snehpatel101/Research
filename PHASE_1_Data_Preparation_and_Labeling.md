# Phase 1 – Data Preparation and Labeling

## Overview

Phase 1 transforms raw market data into a clean, feature-rich, properly-labeled dataset ready for deep learning model training. This phase establishes the foundation for the entire ensemble forecasting system by:

1. Loading and cleaning historical OHLCV data for MES and MGC futures
2. Engineering comprehensive features (technical indicators, regime flags, temporal context)
3. Implementing triple-barrier labeling with ATR-based dynamic thresholds
4. Optimizing barrier parameters using Genetic Algorithm (GA)
5. Producing final labeled dataset with train/validation/test splits

**Think of Phase 1 as:** Creating the "answer key" before the models take the test.

---

## Objectives

### Primary Goals
- Produce a clean, validated dataset spanning 20 years of intraday data for MES and MGC
- Generate triple-barrier labels (−1, 0, +1) for three horizons: 1-bar, 5-bar, and 20-bar
- Optimize barrier parameters via GA to maximize label quality (profitability + risk metrics)
- Create time-aware train/validation/test splits that prevent lookahead bias
- Document all feature engineering and labeling decisions for reproducibility

### Success Criteria
- Zero missing or corrupt bars in critical trading hours
- Feature matrix with 40–80 engineered features per bar
- Labels that produce positive Sharpe ratio (>0.5) in simple baseline backtest
- GA-optimized barrier parameters that outperform naive fixed thresholds
- Clear temporal splits with proper purging around test boundaries

---

## Prerequisites

### Data Requirements
- Historical intraday OHLCV data for MES (Micro E-mini S&P 500)
- Historical intraday OHLCV data for MGC (Micro Gold)
- Minimum 20 years of data (or maximum available)
- Bar resolution: 1-minute or 5-minute (consistent across dataset)
- Timezone: UTC or exchange time (must be consistent)

### Infrastructure Requirements
- Storage: 100–500 GB for raw + processed data
- RAM: 64–128 GB recommended for in-memory operations on large datasets
- CPU: Multi-core (12+ cores) for parallel feature computation and GA
- Libraries: pandas/polars, numpy, ta-lib (or equivalent), vectorbt, pygad (or DEAP for GA)

### Technical Dependencies
```
pandas >= 2.0
polars >= 0.19 (optional, for faster operations)
numpy >= 1.24
ta-lib >= 0.4.28 (or pandas-ta)
vectorbt >= 0.26
pygad >= 3.0 (or DEAP >= 1.4 for genetic algorithm)
pyarrow >= 14.0 (for Parquet I/O)
```

---

## Detailed Steps

### Step 1: Data Ingestion and Cleaning

#### 1.1 Load Raw OHLCV Data

**Objective:** Load historical bars for MES and MGC into memory-efficient format.

**Actions:**
- Load from source (CSV, database, API, etc.) into pandas/polars DataFrame
- Required columns: `timestamp`, `open`, `high`, `low`, `close`, `volume`
- Ensure timestamp is parsed as datetime64 with proper timezone
- Sort by timestamp ascending
- Add symbol identifier column: `symbol` ∈ {MES, MGC}

**Data Shape Check:**
```
Expected columns: [timestamp, symbol, open, high, low, close, volume]
Expected rows: ~2-5 million per symbol for 20 years of 5-min bars
```

#### 1.2 Handle Missing Bars and Gaps

**Objective:** Identify and resolve data quality issues.

**Actions:**
- Detect missing bars using expected sampling frequency
  - For 5-min bars: gaps > 5 minutes during RTH (Regular Trading Hours)
  - For 1-min bars: gaps > 1 minute during RTH
- Options for handling gaps:
  - **Forward-fill:** Use last valid OHLC for missing bars (volume = 0)
  - **Drop:** Remove entire sessions with >10% missing bars
  - **Interpolate:** Linear interpolation for small gaps (<3 bars)
- Flag holiday sessions and known half-days (adjust expected hours)

**Quality Checks:**
- Max consecutive missing bars < 10 during RTH
- No OHLC values = 0 (except volume during no-trade periods)
- High/Low boundaries respected: `low <= open/close <= high`

#### 1.3 Outlier Detection and Handling

**Objective:** Remove or cap extreme price spikes and data errors.

**Actions:**
- Calculate rolling statistics (50-bar window):
  - Mean return
  - Std dev of returns
  - Z-score for each bar's return
- Flag outliers:
  - Returns with |z-score| > 5.0 (likely data errors)
  - Volume spikes > 10× median (may be legitimate but check)
- Handle outliers:
  - **Soft cap:** Winsorize extreme returns to 99th percentile
  - **Remove:** Drop bars with impossible values (e.g., negative prices)

#### 1.4 Contract Roll Handling

**Objective:** Ensure continuous price series across futures contract rollovers.

**Actions:**
- Identify roll dates for MES and MGC contracts
- Apply adjustment method:
  - **Back-adjustment (preferred):** Adjust historical prices for roll gaps
  - **Forward-adjustment:** Adjust future prices from roll point
  - **Ratio adjustment:** Multiply by roll ratio to maintain relative moves
- Verify no artificial jumps remain at roll boundaries (check daily returns)

**Output:** Clean continuous OHLCV dataset saved as Parquet files
```
data/
  clean/
    MES_clean.parquet
    MGC_clean.parquet
```

---

### Step 2: Feature Engineering

#### 2.1 Basic Price Transforms

**Objective:** Create fundamental price-based features.

**Features to compute per symbol:**

1. **Returns (log and simple)**
   - `log_return = log(close_t / close_{t-1})`
   - `simple_return = (close_t - close_{t-1}) / close_{t-1}`

2. **Range features**
   - `high_low_range = high - low`
   - `close_open_range = close - open`
   - `body_ratio = abs(close - open) / (high - low + epsilon)` (avoid div-by-zero)

3. **True Range (TR)**
   - `TR = max(high - low, abs(high - prev_close), abs(low - prev_close))`
   - **ATR (Average True Range)** = EMA or SMA of TR over N periods
     - Use N=14 as default, also compute ATR(7) and ATR(21) for multi-scale

4. **VWAP components**
   - `typical_price = (high + low + close) / 3`
   - `vwap = cumsum(typical_price * volume) / cumsum(volume)` (session-based reset)
   - `vwap_distance = (close - vwap) / vwap`

#### 2.2 Technical Indicators

**Objective:** Add standard TA indicators that capture momentum, trend, volatility.

**Momentum Indicators:**
- RSI (14, 21 periods)
- Stochastic Oscillator (14, 3, 3)
- Rate of Change (ROC) (10, 20 periods)
- Williams %R (14)

**Trend Indicators:**
- Moving Averages:
  - SMA (10, 20, 50, 100, 200)
  - EMA (9, 21, 50)
- MACD (12, 26, 9)
- ADX (14) – Average Directional Index for trend strength

**Volatility Indicators:**
- Bollinger Bands (20, 2):
  - Upper band, Lower band, %B position
- ATR (7, 14, 21) – computed above
- Historical Volatility (std dev of returns over 20, 50 periods)

**Volume Indicators:**
- Volume SMA (20 periods)
- Volume z-score: `(volume - volume_mean) / volume_std` (20-bar rolling)
- On-Balance Volume (OBV)
- Volume-weighted returns

#### 2.3 Regime Features

**Objective:** Identify market regime (volatility, trend) for adaptive labeling.

**Volatility Regime:**
- Compute rolling ATR percentile (200-bar lookback)
- Classify:
  - Low volatility: ATR < 33rd percentile
  - Medium volatility: 33rd ≤ ATR < 67th percentile
  - High volatility: ATR ≥ 67th percentile
- Store as categorical: `vol_regime` ∈ {0, 1, 2}

**Trend Regime:**
- Method 1: Moving Average Crossover
  - If `SMA(50) > SMA(200)`: uptrend (1)
  - If `SMA(50) < SMA(200)`: downtrend (−1)
  - Else: sideways (0)
- Method 2: ADX + DMI
  - If `ADX > 25` and `+DI > -DI`: strong uptrend (2)
  - If `ADX > 25` and `-DI > +DI`: strong downtrend (−2)
  - If `ADX ≤ 25`: chop / no trend (0)
- Store as: `trend_regime` ∈ {−2, −1, 0, 1, 2}

#### 2.4 Temporal Features

**Objective:** Capture session structure and time-of-day patterns.

**Time-based features:**
- `hour_of_day` (0–23)
- `minute_of_hour` (0–59)
- `day_of_week` (0–6, Monday=0)
- `is_rth` (boolean: Regular Trading Hours 9:30–16:00 ET)
- `session_start` (boolean: first bar of RTH)
- `session_end` (boolean: last bar of RTH)

**Encode cyclically for models:**
- `hour_sin = sin(2π * hour / 24)`
- `hour_cos = cos(2π * hour / 24)`
- `dow_sin = sin(2π * dow / 7)`
- `dow_cos = cos(2π * dow / 7)`

#### 2.5 Rolling Window Context

**Objective:** Prepare sequences for deep learning models.

**Actions:**
- For each training sample at time `t`, create a window of past `L` bars:
  - Window length `L` ∈ {64, 128, 256} (to be tuned)
  - Window includes bars from `t-L+1` to `t`
- Features per bar in window:
  - All computed indicators from steps 2.1–2.4
  - Raw OHLCV (normalized or scaled as needed)

**Storage format:**
- Store features as 2D array per sample: `[L, F]` where F = number of features
- Or store flat DataFrame with sample_id and lookback_index

**Feature Scaling (applied later in modeling):**
- Returns: already normalized (unitless)
- Indicators: normalize per-feature using training set statistics
  - StandardScaler: `(x - μ) / σ`
  - Or MinMaxScaler: `(x - min) / (max - min)`
- ATR and volatility: log-scale if distributions are skewed

**Output:** Feature-rich dataset with ~40–80 features per bar
```
data/
  features/
    MES_features.parquet  # columns: [timestamp, symbol, feature_1, ..., feature_N]
    MGC_features.parquet
```

---

### Step 3: Triple-Barrier Labeling (Initial Pass)

#### 3.1 Understand Triple-Barrier Method

**Concept:**
For each bar `t` (potential entry), define three barriers:
1. **Upper barrier (profit target):** `entry_price + k_up * ATR_t`
2. **Lower barrier (stop loss):** `entry_price - k_down * ATR_t`
3. **Vertical barrier (time limit):** `t + max_holding_bars`

**Walk forward from `t+1` to `t + max_holding_bars`:**
- If price touches **upper barrier first** → Label = **+1** (profitable long)
- If price touches **lower barrier first** → Label = **−1** (unprofitable long / stop hit)
- If **neither touched** before time limit → Label = **0** (neutral / no decisive move)

**Why ATR-based:**
- ATR scales with volatility → barriers widen in high vol, tighten in low vol
- Prevents over-labeling noise as signals in quiet markets
- Prevents under-capturing moves in explosive markets

#### 3.2 Initial Barrier Parameters (Naive Starting Point)

**Before GA optimization, use reasonable defaults:**

| Horizon | max_holding_bars | k_up (ATR multiplier) | k_down (ATR multiplier) |
|---------|------------------|------------------------|--------------------------|
| 1-bar   | 1                | 1.0                    | 1.0                      |
| 5-bar   | 5                | 1.5                    | 1.0                      |
| 20-bar  | 20               | 2.0                    | 1.5                      |

**Rationale:**
- Symmetric or slightly asymmetric (more room for stop)
- Larger horizons allow wider barriers (more price movement expected)

#### 3.3 Label Generation Function

**Pseudocode:**
```
function triple_barrier_label(entry_idx, entry_price, atr, k_up, k_down, max_bars, ohlc_data):
    upper_barrier = entry_price + k_up * atr
    lower_barrier = entry_price - k_down * atr
    
    for i in range(1, max_bars + 1):
        if entry_idx + i >= len(ohlc_data):
            return 0  # Not enough future data
        
        bar = ohlc_data[entry_idx + i]
        
        # Check if upper barrier touched
        if bar.high >= upper_barrier:
            return +1  # Hit profit target first
        
        # Check if lower barrier touched
        if bar.low <= lower_barrier:
            return -1  # Hit stop loss first
    
    return 0  # Neither barrier hit within time limit
```

**Implementation Notes:**
- Use vectorized operations where possible (pandas/numpy) for speed
- For each symbol (MES, MGC), iterate through all bars as potential entries
- Skip bars near dataset end where `entry_idx + max_bars` exceeds data length

#### 3.4 Apply Initial Labels

**Actions:**
- For each bar `t` in dataset:
  - Compute ATR at `t` (use ATR(14) from features)
  - For each horizon (1, 5, 20):
    - Call `triple_barrier_label(t, close_t, ATR_t, k_up, k_down, max_holding)`
    - Store result as `label_1`, `label_5`, `label_20`

**Output:**
- Add three new columns to feature DataFrame:
  - `label_1` ∈ {−1, 0, +1}
  - `label_5` ∈ {−1, 0, +1}
  - `label_20` ∈ {−1, 0, +1}

**Sanity Check:**
- Check label distribution (% of −1, 0, +1 per horizon)
  - Expect: 0 labels to be most common (~50-70%), ±1 labels less frequent
  - If heavily imbalanced (e.g., 95% zeros), barriers too wide
  - If too balanced (e.g., 33% each), barriers too narrow

**Save intermediate result:**
```
data/
  labeled/
    MES_labeled_initial.parquet
    MGC_labeled_initial.parquet
```

---

### Step 4: Genetic Algorithm (GA) for Barrier Optimization

#### 4.1 GA Objective

**Goal:** Find optimal `(k_up, k_down, max_holding_bars)` per horizon that maximize label quality.

**Label quality defined as:**
- High Sharpe ratio when labels used in simple strategy
- Low maximum drawdown
- Reasonable number of trades (not over-trading noise)

**Fitness function:**
```
fitness = Sharpe_ratio - λ * MaxDrawdown - μ * (|NumTrades - TargetTrades| / TargetTrades)
```
Where:
- `λ` = penalty weight for drawdown (e.g., 0.5–1.0)
- `μ` = penalty for deviating from target trade count (optional, e.g., 0.1)
- `TargetTrades` = desired trades per year (e.g., 50–200 for intraday)

#### 4.2 GA Search Space (per horizon)

**Genes per individual (chromosome):**
- `k_up`: float in [0.5, 3.0] (ATR multiplier for upper barrier)
- `k_down`: float in [0.5, 3.0] (ATR multiplier for lower barrier)
- `max_holding_bars`: int in [H/2, 2*H] where H is nominal horizon
  - For 5-bar horizon: search in [3, 10]
  - For 20-bar horizon: search in [10, 40]

**Example encoding:**
```
Individual for 5-bar horizon: [k_up=1.3, k_down=0.8, max_hold=7]
```

#### 4.3 GA Subset Selection (Critical for 20-Year Data)

**Problem:** Running GA on full 20 years × 200 candidates = too slow.

**Solution:** Select representative subset of data for GA evaluation.

**Subset Selection Strategy:**

1. **Time-based sampling:**
   - Choose 3–5 distinct periods (2–3 years each) that cover different regimes:
     - **High volatility:** e.g., 2008–2009 (GFC), 2020 (COVID)
     - **Low volatility:** e.g., 2017, 2019
     - **Bull trend:** e.g., 2013–2014
     - **Bear/chop:** e.g., 2015–2016

2. **Total subset size:**
   - Aim for 5–7 years total
   - ~1–2 million bars (for 5-min data)
   - Enough to get stable backtest statistics

3. **Combine periods:**
   - Concatenate selected periods into one GA evaluation dataset
   - Label each period for later regime-specific analysis if needed

**Store subset indices:**
```
data/
  ga_subset/
    ga_subset_indices.parquet  # List of timestamps in GA subset
```

#### 4.4 GA Evaluation (Fitness Computation)

**For each GA candidate (barrier parameter set):**

1. **Relabel subset:**
   - Apply triple-barrier labeling with candidate's `k_up`, `k_down`, `max_hold`
   - Generate labels for all bars in GA subset

2. **Simulate simple strategy:**
   - Rules:
     - Label = +1 → Enter long at next bar open, exit at label expiry or next signal
     - Label = −1 → Enter short at next bar open, exit at label expiry or next signal
     - Label = 0 → No position
   - Use vectorbt for fast vectorized backtesting:
     ```python
     import vectorbt as vbt
     
     # Create signals
     entries = (labels == +1).shift(1)  # Enter next bar
     exits = (labels == -1).shift(1)
     
     # Run portfolio backtest
     pf = vbt.Portfolio.from_signals(
         close=close_prices,
         entries=entries,
         exits=exits,
         freq='5min'
     )
     ```

3. **Compute metrics:**
   - **Sharpe ratio:** `SR = mean(returns) / std(returns) * sqrt(periods_per_year)`
   - **Max drawdown:** `MDD = max(peak - trough) / peak`
   - **Number of trades:** `N_trades`
   - **Profit factor:** `total_profit / total_loss`

4. **Calculate fitness:**
   ```
   fitness = Sharpe - 0.5 * MDD - 0.1 * (abs(N_trades - 100) / 100)
   ```

**Return:** Single scalar fitness value for this candidate

#### 4.5 GA Algorithm Configuration

**GA Library:** Use `pygad` or `DEAP`

**Parameters:**
- **Population size:** 50–100 individuals
- **Generations:** 30–50
- **Crossover:** Single-point or uniform crossover
- **Crossover rate:** 0.7–0.8
- **Mutation:** Gaussian mutation for floats, uniform for ints
- **Mutation rate:** 0.1–0.2
- **Selection:** Tournament selection (size 3) or roulette wheel
- **Elitism:** Keep top 5% of population each generation

**Pseudocode:**
```
Initialize population randomly within bounds
For each generation:
    For each individual:
        Compute fitness (relabel + backtest on subset)
    
    Select parents (tournament selection)
    Apply crossover to create offspring
    Apply mutation to offspring
    Replace population (keep top 5% elite + new offspring)
    
    Log best fitness and parameters
    
    If converged (fitness plateau for 5 generations):
        Break

Return best individual across all generations
```

#### 4.6 Run GA for Each Horizon

**Process:**
1. Run GA separately for 1-bar, 5-bar, and 20-bar horizons
2. Each run outputs optimal parameters:
   - `best_k_up`, `best_k_down`, `best_max_hold` per horizon
3. Log convergence curves (fitness vs generation)
4. Save top 3–5 candidates per horizon for sanity checking

**Validation:**
- After GA completes on subset, test top candidates on **larger** data slice (e.g., 10+ years)
- Verify that ranking remains similar
- If top candidate performs poorly on full data → suspect overfitting, choose more robust 2nd/3rd candidate

**Output:**
```
config/
  ga_results/
    ga_1bar_best.json
    ga_5bar_best.json
    ga_20bar_best.json
    ga_convergence.png  # Plot of fitness over generations
```

**Example optimal params:**
```json
{
  "horizon": "5bar",
  "k_up": 1.4,
  "k_down": 0.9,
  "max_holding_bars": 6,
  "fitness": 1.23,
  "sharpe": 1.8,
  "max_dd": 0.08,
  "num_trades": 87
}
```

---

### Step 5: Relabel Full Dataset with GA-Optimized Parameters

#### 5.1 Apply Final Barrier Settings

**Objective:** Use GA-optimized parameters to relabel entire 20-year dataset.

**Actions:**
- Load full feature dataset (MES + MGC)
- For each horizon, use best parameters from GA:
  - 1-bar: `k_up_1, k_down_1, max_hold_1`
  - 5-bar: `k_up_5, k_down_5, max_hold_5`
  - 20-bar: `k_up_20, k_down_20, max_hold_20`

**Labeling Process:**
- Iterate through all bars (or use vectorized method if possible)
- For each bar `t`:
  - Compute ATR at `t`
  - For each horizon, apply triple-barrier logic with optimized params
  - Store labels: `label_1`, `label_5`, `label_20`

**Optimization Tips:**
- Process in chunks (e.g., year-by-year) if memory constrained
- Use multiprocessing to parallelize across symbols or date ranges
- Expected time: 10–60 minutes depending on hardware and data size

#### 5.2 Sample Weighting by Quality Tiers

**Objective:** Assign higher weights to high-quality signal events for model training.

**Quality Tiers Definition:**

| Tier | Criteria | Example | Weight |
|------|----------|---------|--------|
| A+   | Hit target quickly (<50% of max_hold), minimal drawdown | Price moved strongly in predicted direction | 2.0 |
| A    | Hit target within max_hold, moderate path | Standard winning trade | 1.5 |
| B+   | Hit target late or small profit margin | Borderline win | 1.0 |
| C    | Label = 0 (no decisive move) or hit target after almost hitting stop | Low-information event | 0.5 |

**Implementation:**
- For each labeled event where label ∈ {−1, +1}:
  - Track how quickly barrier was hit (number of bars)
  - Track maximum adverse excursion (MAE): worst price movement against position
  - Calculate: `signal_quality_score = (1 - MAE/ATR) * (max_hold / bars_to_hit)`
  
- Assign tier based on score:
  - `score > 1.5` → A+
  - `1.0 < score ≤ 1.5` → A
  - `0.5 < score ≤ 1.0` → B+
  - `score ≤ 0.5` → C

- Add column: `sample_weight` to dataset

**For label = 0 (neutral):**
- Default weight = 0.5 (less informative)
- Or completely exclude from training if desired (set weight = 0)

**Alternative:** Pre-filter dataset to remove C-tier samples entirely

#### 5.3 Final Dataset Structure

**Columns:**
```
timestamp            : datetime64
symbol               : str (MES, MGC)
open, high, low, close, volume : float64
feature_1, ..., feature_N : float64 (40-80 features)
atr_14               : float64
vol_regime           : int8 (0=low, 1=med, 2=high)
trend_regime         : int8 (-2 to +2)
label_1              : int8 (-1, 0, +1)
label_5              : int8 (-1, 0, +1)
label_20             : int8 (-1, 0, +1)
sample_weight_1      : float32 (quality weight for 1-bar label)
sample_weight_5      : float32
sample_weight_20     : float32
```

**Save as:**
```
data/
  final/
    MES_final_labeled.parquet
    MGC_final_labeled.parquet
```

---

### Step 6: Train/Validation/Test Split

#### 6.1 Time-Based Splitting

**Objective:** Create splits that respect temporal order and prevent lookahead.

**Split Strategy:**

| Set        | Purpose | % of Data | Approx Years (if 20y total) |
|------------|---------|-----------|------------------------------|
| Train      | Model training | 70%       | ~14 years (oldest)           |
| Validation | Hyperparameter tuning, model selection | 15% | ~3 years (middle) |
| Test       | Final evaluation (untouched) | 15%       | ~3 years (most recent) |

**Critical Rules:**
1. **Chronological order:** Train < Validation < Test in time
2. **No overlap:** Strict boundaries between sets
3. **Purging:** Remove samples whose label lookforward overlaps with next set
4. **Embargo:** Add buffer period (e.g., 20 bars) between sets

#### 6.2 Purging for Label Lookahead

**Problem:** A training sample at time `t` with 20-bar label uses data up to `t+20`. If `t+20` falls into validation set, there's leakage.

**Solution:**
- For each split boundary (e.g., train/val cutoff at time `T`):
  - Remove training samples where: `sample_time + max_horizon_lookahead > T`
  - Example: If cutoff is 2018-01-01 and max horizon is 20 bars (5-min = 100 minutes):
    - Remove training samples after 2017-12-31 22:20

**Implementation:**
```python
train_end = pd.Timestamp('2018-01-01')
max_lookahead_bars = 20
bar_duration = pd.Timedelta('5min')

purge_threshold = train_end - (max_lookahead_bars * bar_duration)
train_set = df[df.timestamp < purge_threshold]
```

#### 6.3 Embargo Period

**Objective:** Add extra buffer to prevent subtle correlation leakage.

**Actions:**
- After purging, add embargo period (e.g., 1 day) between sets
- Skip samples in embargo zone entirely (don't use in any set)

**Example:**
```
Train:      [data_start ... 2017-12-31 00:00]
Purge:      [2017-12-31 00:00 ... 2017-12-31 22:20] (removed)
Embargo:    [2017-12-31 22:20 ... 2018-01-01 22:20] (removed)
Validation: [2018-01-01 22:20 ... 2021-01-01 00:00]
(repeat purge/embargo)
Test:       [2021-01-01 ... data_end]
```

#### 6.4 Save Split Metadata

**Output:**
```
config/
  splits/
    train_indices.npy       # Array of row indices for train set
    val_indices.npy
    test_indices.npy
    split_config.json       # Timestamp ranges, purge/embargo settings
```

**split_config.json example:**
```json
{
  "train": {
    "start": "2004-01-01",
    "end": "2017-12-31",
    "n_samples": 1456789
  },
  "validation": {
    "start": "2018-01-02",
    "end": "2020-12-31",
    "n_samples": 312456
  },
  "test": {
    "start": "2021-01-02",
    "end": "2024-12-31",
    "n_samples": 312123
  },
  "purge_bars": 20,
  "embargo_days": 1
}
```

---

### Step 7: Dataset Validation and Quality Checks

#### 7.1 Label Distribution Analysis

**Check per horizon and per set (train/val/test):**
- Percentage of −1, 0, +1 labels
- Expected: 0 label dominant (~50-70%), ±1 minority but significant (~15-25% each)

**Alert if:**
- Any class < 5% (severe imbalance)
- Train/val/test distributions wildly different (suggests regime shift or sampling issue)

#### 7.2 Feature Correlation Check

**Actions:**
- Compute correlation matrix of features
- Identify highly correlated pairs (|corr| > 0.95)
- Consider dropping redundant features (keep more interpretable one)

**Example:**
- If `RSI_14` and `RSI_21` correlate at 0.97, consider keeping only one

#### 7.3 Temporal Consistency

**Verify:**
- No duplicate timestamps within same symbol
- Monotonically increasing timestamps
- No gaps longer than expected holiday periods

#### 7.4 Baseline Strategy Backtest

**Objective:** Sanity check that labels have predictive value.

**Simple Strategy:**
- On validation set:
  - When `label_5 == +1`: go long next bar
  - When `label_5 == -1`: go short next bar
  - When `label_5 == 0`: no position
- Measure:
  - Sharpe ratio (expect > 0.5 for GA-optimized labels)
  - Max drawdown (expect < 20%)
  - Win rate (expect 40-55%)

**If Sharpe < 0.3:**
- Labels may be too noisy
- Consider: tighter barriers, different GA fitness function, or regime filtering

**Output:**
```
reports/
  phase1_baseline_backtest.html  # Vectorbt tear sheet
  phase1_label_distribution.png
  phase1_feature_correlation.png
```

---

## Outputs and Deliverables

### Primary Outputs

1. **Clean OHLCV Dataset**
   - `data/clean/MES_clean.parquet`
   - `data/clean/MGC_clean.parquet`

2. **Final Feature + Label Dataset**
   - `data/final/MES_final_labeled.parquet`
   - `data/final/MGC_final_labeled.parquet`
   - Shape: ~2-5M rows × 85-100 columns

3. **GA Optimization Results**
   - `config/ga_results/ga_1bar_best.json`
   - `config/ga_results/ga_5bar_best.json`
   - `config/ga_results/ga_20bar_best.json`

4. **Split Configuration**
   - `config/splits/train_indices.npy`
   - `config/splits/val_indices.npy`
   - `config/splits/test_indices.npy`
   - `config/splits/split_config.json`

5. **Documentation**
   - `reports/PHASE1_summary.md` – Summary of data stats, GA results, baseline metrics
   - `reports/phase1_baseline_backtest.html`

### Quality Metrics to Report

| Metric | Target | Actual |
|--------|--------|--------|
| Total bars (MES) | 2-5M | ___ |
| Total bars (MGC) | 2-5M | ___ |
| Missing bar rate | <0.1% | ___ |
| Outlier removal rate | <0.5% | ___ |
| Feature count | 40-80 | ___ |
| Train set size | ~70% | ___ |
| Val set size | ~15% | ___ |
| Test set size | ~15% | ___ |
| Label +1 rate (5-bar) | 15-25% | ___ |
| Label -1 rate (5-bar) | 15-25% | ___ |
| Label 0 rate (5-bar) | 50-70% | ___ |
| Baseline Sharpe (val, 5-bar) | >0.5 | ___ |
| Baseline Max DD (val, 5-bar) | <20% | ___ |

---

## Implementation Checklist

### Pre-Phase Tasks
- [ ] Set up project directory structure
- [ ] Install required libraries (pandas, vectorbt, pygad, etc.)
- [ ] Acquire raw OHLCV data for MES and MGC (20 years)
- [ ] Document data source and licensing

### Step 1: Data Cleaning
- [ ] Load raw data into DataFrames
- [ ] Parse timestamps and sort chronologically
- [ ] Detect and handle missing bars
- [ ] Identify and remove/cap outliers
- [ ] Handle futures contract rolls
- [ ] Save clean data to Parquet

### Step 2: Feature Engineering
- [ ] Compute basic price transforms (returns, ranges, TR)
- [ ] Calculate ATR (7, 14, 21 periods)
- [ ] Add momentum indicators (RSI, Stoch, ROC)
- [ ] Add trend indicators (SMA, EMA, MACD, ADX)
- [ ] Add volatility indicators (Bollinger, ATR, HV)
- [ ] Add volume indicators (Volume SMA, z-score, OBV)
- [ ] Compute regime flags (volatility, trend)
- [ ] Add temporal features (hour, day, session markers)
- [ ] Encode cyclical features (sin/cos transforms)
- [ ] Validate feature distributions and correlations
- [ ] Save feature dataset to Parquet

### Step 3: Initial Labeling
- [ ] Implement triple-barrier labeling function
- [ ] Apply initial labels with naive parameters (1-bar, 5-bar, 20-bar)
- [ ] Check label distributions
- [ ] Save initially labeled dataset

### Step 4: GA Optimization
- [ ] Select representative subset of data for GA (5-7 years)
- [ ] Define GA search space per horizon
- [ ] Implement fitness function (Sharpe - λ*MDD)
- [ ] Set up GA library (pygad or DEAP) with proper parameters
- [ ] Run GA separately for 1-bar, 5-bar, 20-bar horizons
- [ ] Log convergence and save top candidates
- [ ] Validate best candidates on larger data slice
- [ ] Save final optimized parameters to JSON

### Step 5: Final Relabeling
- [ ] Load full dataset and GA-optimized parameters
- [ ] Relabel entire dataset with optimized barrier settings
- [ ] Compute sample quality tiers and weights
- [ ] Add sample weight columns to dataset
- [ ] Save final labeled dataset

### Step 6: Train/Val/Test Split
- [ ] Define split boundaries (70/15/15)
- [ ] Implement purging for label lookahead
- [ ] Add embargo periods between sets
- [ ] Generate and save split indices
- [ ] Document split configuration

### Step 7: Validation
- [ ] Analyze label distributions per set
- [ ] Check feature correlations
- [ ] Verify temporal consistency
- [ ] Run baseline strategy backtest on validation set
- [ ] Generate quality metrics report
- [ ] Create visualizations (label dist, corr matrix, baseline tearsheet)

### Post-Phase Tasks
- [ ] Archive intermediate datasets if storage permits
- [ ] Update project README with Phase 1 results
- [ ] Prepare dataset for Phase 2 (model training)
- [ ] Review and document any issues or edge cases encountered

---

## Notes and Considerations

### Computational Considerations for 20-Year Dataset

**Memory Management:**
- 20 years of 5-min bars ≈ 2-5M rows per symbol
- Full dataset with features ≈ 5-10 GB in memory
- Use chunked processing or polars for large operations
- Save intermediate results frequently

**GA Subset Strategy:**
- Critical: don't run GA on full 20 years
- Use 5-7 representative years for GA (as described in Step 4.3)
- Final relabeling can be done in chunks/parallel on full dataset

### Feature Engineering Tips

**Avoid Lookahead:**
- All features must use only past data up to bar `t`
- When using EMA/SMA, ensure proper initialization (not using future for warm-up)
- Be careful with session-based calculations (e.g., VWAP resets)

**Feature Selection:**
- Start with comprehensive set (60-80 features)
- Can prune later based on importance scores from models
- Keep interpretation in mind: prefer explainable features

### Labeling Nuances

**ATR Calculation:**
- Use consistent ATR period (14 recommended)
- Ensure ATR is calculated from data up to bar `t` only
- Handle first 14 bars (warm-up period) appropriately

**Edge Cases:**
- Bars near end of dataset: cannot compute labels (insufficient future data)
- Mark these with `label = NaN` and exclude from training
- Typically lose last `max_horizon` bars per symbol

**Regime Adaptation:**
- Optionally: adjust barrier multipliers based on `vol_regime`
  - Low vol: use 0.8 × k_up/k_down
  - High vol: use 1.2 × k_up/k_down
- Can be implemented in Step 5 if GA results suggest it

### Common Pitfalls

1. **Lookahead Bias:**
   - Most common error in financial ML
   - Double-check every feature and label generation logic
   - Use strict temporal splits with purging

2. **Data Leakage:**
   - Ensure no normalization/scaling uses test set statistics
   - All feature engineering must be reproducible in live environment

3. **Overfitting GA:**
   - GA on subset prevents this
   - Validate on out-of-sample before committing
   - If GA results look "too good," suspect overfitting

4. **Imbalanced Labels:**
   - Neutral (0) labels dominating is expected and okay
   - Models will learn to be selective (precision over recall)
   - Sample weighting helps focus on high-quality signals

### Extensions and Future Improvements

**Multi-Asset Considerations:**
- If scaling beyond MES/MGC:
  - Normalize features per-asset to account for different price scales
  - Consider asset-specific barrier parameters
  - Add asset ID as categorical feature

**Alternative Labeling Methods:**
- Meta-labeling: first predict direction, then predict bet size
- Trend-following vs mean-reversion labels
- Multi-label classification (e.g., magnitude of move as separate target)

**Advanced Features:**
- Order flow indicators (if available)
- Inter-market correlations (SPY/VIX, gold/dollar)
- Sentiment indicators (VIX term structure, put/call ratio)
- Microstructure features (bid-ask spread, trade intensity)

---

## Success Criteria Summary

Phase 1 is complete and successful when:

1. ✅ Clean, validated OHLCV dataset for MES and MGC (20 years)
2. ✅ Feature matrix with 40-80 indicators per bar
3. ✅ GA-optimized triple-barrier labels for 3 horizons
4. ✅ Baseline strategy Sharpe > 0.5 on validation set
5. ✅ Proper train/val/test splits with purging and embargo
6. ✅ Sample weights assigned based on signal quality
7. ✅ All outputs saved and documented
8. ✅ Quality metrics report generated

**Proceed to Phase 2** (Model Training) only after all criteria are met.

---

**End of Phase 1 Specification**
