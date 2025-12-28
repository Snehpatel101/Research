# Ultra-Deep Quantitative Trading Analysis

## ML Model Factory for OHLCV Time Series

**Analysis Date:** 2025-12-28
**Analyst Role:** Quantitative Trading System Review

---

## 1. Executive Summary

This ML Model Factory implements a sophisticated labeling and signal generation system for futures trading. The core innovation is the **triple-barrier labeling method** with symbol-specific asymmetric barriers, transaction cost penalties, and quality-weighted samples. The system is designed for single-contract trading with complete isolation between symbols.

### Key Quantitative Insights

| Aspect | Implementation | Trading Implication |
|--------|----------------|---------------------|
| Labeling | Triple-barrier with ATR-scaled barriers | Adapts to volatility regimes |
| Asymmetry | MES 1.5:1.0 (k_up:k_down) | Counteracts equity risk premium drift |
| Transaction Costs | 0.5 ticks + regime-adaptive slippage | Realistic P&L estimation |
| Quality Weighting | 0.5x-1.5x sample weights | Prioritizes clean signals |
| Neutral Rate | 20-30% target | Prevents overtrading |

---

## 2. Triple-Barrier Labeling Method

### 2.1 Mathematical Foundation

The triple-barrier method (Lopez de Prado, "Advances in Financial Machine Learning") defines a profitable trade by which barrier is hit first:

```
Entry Price: P_0 = close[i]
ATR at Entry: sigma = ATR_14[i]

Upper Barrier: B_up = P_0 + k_up * sigma
Lower Barrier: B_down = P_0 - k_down * sigma
Time Barrier: t_max = max_bars

Label Assignment:
  +1 (Long Win)   : if high[i+j] >= B_up for some j < t_max (upper hit first)
  -1 (Short Loss) : if low[i+j] <= B_down for some j < t_max (lower hit first)
   0 (Neutral)    : if neither barrier hit within t_max bars (timeout)
```

### 2.2 Implementation Details

From `/Users/sneh/research/src/phase1/stages/labeling/triple_barrier.py`:

```python
@nb.jit(nopython=True, cache=True)
def triple_barrier_numba(close, high, low, open_prices, atr, k_up, k_down, max_bars):
    """
    Numba-optimized triple barrier labeling.

    Key features:
    - Uses bar high/low for barrier hit detection (not close-to-close)
    - Resolves simultaneous barrier hits using distance from bar open
    - Tracks MAE/MFE for quality scoring
    - Excludes last max_bars samples (sentinel value -99)
    """
```

**Critical Implementation Choice:** When both barriers are hit on the same bar, the algorithm uses distance from the bar's open price to determine which barrier was likely hit first. This eliminates systematic bias from always checking upper barrier first.

### 2.3 Why Triple-Barrier Over Simple Direction?

| Approach | Pros | Cons |
|----------|------|------|
| Simple Direction (sign of return) | Easy to implement | Ignores trade viability |
| Fixed Threshold | Simple | Doesn't adapt to volatility |
| **Triple-Barrier** | ATR-adaptive, risk-aware, realistic | More complex, requires tuning |

The triple-barrier method naturally filters out low-quality signals:
- **Timeout (neutral)** = Price didn't move enough = Noise, not tradeable
- **Quick hits** = Strong momentum = High-confidence signal
- **Slow hits** = Grinding price action = Lower confidence

---

## 3. Symbol-Specific Asymmetric Barriers

### 3.1 The Equity Drift Problem

From `/Users/sneh/research/src/phase1/config/barriers_config.py`:

```python
# MATHEMATICAL REASONING (2025-12 correction):
# - MES drifts UP due to equity risk premium (~7% annually)
# - This makes the UPPER barrier naturally EASIER to hit
# - To balance long/short signals, make UPPER barrier HARDER (k_up > k_down)
# - Previous config had k_down > k_up which AMPLIFIED long bias (wrong direction)
```

**The Problem:** S&P 500 futures (MES) have a structural upward drift from the equity risk premium (~7% annually). In a frictionless symmetric labeling scheme, this drift causes:
- Upper barrier hit more frequently than lower
- Resulting in 87-91% long labels
- Models learn to always predict "long"
- Zero alpha, just riding the drift

**The Solution:** Asymmetric barriers where k_up > k_down:

```python
BARRIER_PARAMS = {
    'MES': {
        5:  {'k_up': 1.50, 'k_down': 1.00, 'max_bars': 12},   # 1.50:1.00 ratio
        10: {'k_up': 2.00, 'k_down': 1.40, 'max_bars': 25},   # 1.43:1.00 ratio
        15: {'k_up': 2.50, 'k_down': 1.75, 'max_bars': 38},   # 1.43:1.00 ratio
        20: {'k_up': 3.00, 'k_down': 2.10, 'max_bars': 50},   # 1.43:1.00 ratio
    },
    'MGC': {  # Gold: symmetric because no drift
        5:  {'k_up': 1.20, 'k_down': 1.20, 'max_bars': 12},
        10: {'k_up': 1.60, 'k_down': 1.60, 'max_bars': 25},
        15: {'k_up': 2.00, 'k_down': 2.00, 'max_bars': 38},
        20: {'k_up': 2.50, 'k_down': 2.50, 'max_bars': 50},
    }
}
```

### 3.2 Mathematical Interpretation

For MES H20:
- **Upper barrier:** 3.0 ATR above entry (harder to hit)
- **Lower barrier:** 2.1 ATR below entry (easier to hit)
- **Ratio:** 1.43:1

This means:
- A long signal requires price to move 43% MORE than a short signal
- Counterbalances the ~7% annual drift that makes longs "too easy"
- Target: ~50/50 long/short distribution among signals

For MGC (Gold):
- **Symmetric barriers:** Gold is a store of value with mean-reverting characteristics
- No structural drift to counteract
- k_up = k_down for unbiased labeling

---

## 4. Transaction Cost Modeling

### 4.1 Cost Components

From `/Users/sneh/research/src/phase1/config/barriers_config.py`:

```python
# Commission costs (round-trip)
TRANSACTION_COSTS = {
    'MES': 0.5,  # $0.625 round-trip (0.5 ticks * $1.25/tick)
    'MGC': 0.3,  # $0.30 round-trip (0.3 ticks * $1.00/tick)
}

# Slippage costs (per fill, regime-adaptive)
SLIPPAGE_TICKS = {
    'MES': {'low_vol': 0.5, 'high_vol': 1.0},   # $0.625-$1.25 per fill
    'MGC': {'low_vol': 0.75, 'high_vol': 1.5},  # $0.75-$1.50 per fill
}

# Tick values
TICK_VALUES = {'MES': 1.25, 'MGC': 1.00}  # Dollars per tick
```

### 4.2 Total Trade Cost Calculation

```python
def get_total_trade_cost(symbol, regime='low_vol', include_slippage=True):
    """
    Total round-trip cost = Commission + 2 * Slippage

    For MES in low_vol regime:
      = 0.5 ticks + 2 * 0.5 ticks = 1.5 ticks = $1.875

    For MES in high_vol regime:
      = 0.5 ticks + 2 * 1.0 ticks = 2.5 ticks = $3.125
    """
```

### 4.3 Transaction Cost Penalty in Fitness Function

From `/Users/sneh/research/src/phase1/stages/ga_optimize/fitness.py`:

```python
# Transaction cost penalty in fitness evaluation
cost_ticks = get_total_trade_cost(symbol, regime, include_slippage)
tick_value = TICK_VALUES.get(symbol, 1.0)
cost_in_price_units = cost_ticks * tick_value

if n_trades > 0 and atr_mean > 0:
    avg_profit_per_trade = total_profit / n_trades * atr_mean
    cost_ratio = cost_in_price_units / (avg_profit_per_trade + 1e-6)

    if cost_ratio > 0.20:  # Costs > 20% of profit
        transaction_penalty = max(-10.0, -(cost_ratio - 0.20) * 10.0)
    else:
        transaction_penalty = 0.5  # Small reward for cost-efficient trading
```

**Key Insight:** The optimizer penalizes barrier configurations where transaction costs would consume more than 20% of expected profit. This prevents the system from generating labels that are profitable in theory but unprofitable after costs.

---

## 5. Quality-Based Sample Weighting (0.5x-1.5x)

### 5.1 Quality Score Components

From `/Users/sneh/research/src/phase1/stages/final_labels/core.py`:

```python
def compute_quality_scores(bars_to_hit, mae, mfe, labels, horizon, symbol):
    """
    Quality is based on 5 components:

    1. Speed Score (20%): Faster hits = higher quality
       - Ideal speed = horizon * 1.5 bars
       - Uses exponential decay for deviation

    2. MAE Score (25%): Lower adverse excursion = higher quality
       - Normalized to 95th percentile
       - 1.0 - clip(mae / mae_95, 0, 1)

    3. MFE Score (20%): Higher favorable excursion = higher quality
       - Normalized to 95th percentile
       - clip(mfe / mfe_95, 0, 1)

    4. Pain-to-Gain Ratio (20%): Risk per unit profit
       - For longs: |MAE| / MFE
       - For shorts: MFE / |MAE|
       - Lower is better

    5. Time-Weighted Drawdown (15%): Penalize time in drawdown
       - (bars_to_hit / max_bars) * pain_ratio
       - Lower is better
    """

    # Combined quality score
    quality_scores = (
        0.20 * speed_scores +
        0.25 * mae_scores +
        0.20 * mfe_scores +
        0.20 * ptg_scores +
        0.15 * twdd_scores
    )
```

### 5.2 Sample Weight Assignment

```python
def assign_sample_weights(quality_scores):
    """
    Tier-based weight assignment:

    Tier 1 (top 20%):    weight = 1.5x
    Tier 2 (middle 60%): weight = 1.0x
    Tier 3 (bottom 20%): weight = 0.5x
    """
```

### 5.3 Trading Interpretation

| Quality Indicator | Low Quality (0.5x) | High Quality (1.5x) |
|-------------------|-------------------|---------------------|
| Hit Speed | Very slow or very fast | Near ideal speed |
| MAE (drawdown) | Large adverse move before win | Minimal drawdown |
| MFE (profit) | Barely hit barrier | Exceeded barrier significantly |
| Pain-to-Gain | High risk per unit profit | Low risk per unit profit |
| Time in DD | Spent most time underwater | Quick to profit |

**Why This Matters for Training:**
- High-quality samples represent "clean" trades with clear signals
- Low-quality samples are noisy, ambiguous, or barely viable
- Models should learn from clean examples more than noisy ones
- Reduces overfitting to market noise

---

## 6. Expected Performance Metrics

### 6.1 Phase 1 Analysis Summary

From `/Users/sneh/research/CLAUDE.md`:

```
Overall Score: 8.5/10 (Production-Ready)

Expected Performance by Horizon:
| Horizon | Sharpe | Win Rate | Max DD |
|---------|--------|----------|--------|
| H5      | 0.3-0.8| 45-50%   | 10-25% |
| H10     | 0.4-0.9| 46-52%   | 9-20%  |
| H15     | 0.4-1.0| 47-53%   | 8-18%  |
| H20     | 0.5-1.2| 48-55%   | 8-18%  |
```

### 6.2 Interpretation

**Sharpe Ratio 0.3-1.2:**
- These are annualized Sharpe ratios for the labeling strategy
- Not the model's expected Sharpe (which depends on model accuracy)
- Represents the "ceiling" if labels were perfectly predicted

**Win Rate 45-55%:**
- Below 50% is acceptable if winners are larger than losers
- Asymmetric barriers create this asymmetry intentionally
- Average win size > Average loss size due to k_up > k_down

**Max Drawdown 8-25%:**
- Shorter horizons have higher variance (more trades, more noise)
- Longer horizons smooth out but have larger individual losses
- H20 has lowest expected drawdown due to longer time to resolve

### 6.3 Why Longer Horizons Perform Better

| Factor | H5 (25 min) | H20 (100 min) |
|--------|-------------|---------------|
| Signal-to-Noise | Lower | Higher |
| Transaction Cost Impact | Higher | Lower |
| Feature Predictability | Harder | Easier |
| Barrier Distance | Smaller (closer to noise) | Larger (requires real moves) |

---

## 7. Prediction Horizons (H5, H10, H15, H20)

### 7.1 Horizon Definitions

```python
ACTIVE_HORIZONS = [5, 10, 15, 20]  # Bars ahead for prediction
# At 5-minute bars:
# H5  = 25 minutes
# H10 = 50 minutes
# H15 = 75 minutes
# H20 = 100 minutes (~1.67 hours)
```

### 7.2 Barrier Scaling with Horizon

Barriers scale with horizon to maintain consistent hit rates:

| Horizon | k_up (MES) | k_down (MES) | max_bars | ATR Window |
|---------|------------|--------------|----------|------------|
| H5      | 1.50       | 1.00         | 12       | ~1 hour    |
| H10     | 2.00       | 1.40         | 25       | ~2 hours   |
| H15     | 2.50       | 1.75         | 38       | ~3 hours   |
| H20     | 3.00       | 2.10         | 50       | ~4 hours   |

**Pattern:** k values increase with sqrt(horizon) approximately, following volatility scaling.

### 7.3 Purge and Embargo Scaling

```python
PURGE_MULTIPLIER = 3.0     # purge_bars = max_horizon * 3 = 60 bars
EMBARGO_MULTIPLIER = 72.0  # embargo_bars = max_horizon * 72 = 1440 bars
MIN_EMBARGO_BARS = 1440    # Minimum 5 days for 5-min data
```

**Purpose:**
- **Purge (60 bars):** Removes samples at split boundaries where labels might leak
- **Embargo (1440 bars = 5 days):** Ensures feature decorrelation between train/val/test

---

## 8. Single-Contract Architecture

### 8.1 Design Philosophy

From `/Users/sneh/research/CLAUDE.md`:

```
This is a single-contract ML factory. Each contract is trained in complete
isolation. No cross-symbol correlation or feature engineering.

Key Principles:
1. One contract at a time
2. Complete isolation
3. Symbol configurability
```

### 8.2 Why No Cross-Symbol Features?

| Approach | Pros | Cons |
|----------|------|------|
| **Single-Contract** | Simple, no spurious correlations, robust | Misses macro signals |
| Cross-Contract | Can capture macro moves | Correlation changes, more complexity |
| Multi-Asset | Diversification benefits | Requires portfolio-level risk mgmt |

**The Decision:** Single-contract isolation prevents:
- **Spurious correlations:** GC and ES might be correlated during some periods, not others
- **Overfitting:** More features from other symbols = more ways to overfit
- **Deployment complexity:** Each model is standalone, easy to deploy/retire

### 8.3 When Cross-Contract Might Help

Cross-contract signals could add value for:
- Macro regime detection (risk-on/risk-off from ES leading GC)
- Intermarket spread trading (ES vs. NQ)
- Currency hedge ratios (DX impact on commodities)

**Current architecture supports adding these as future phases.**

---

## 9. Label Balance Constraints

### 9.1 GA Optimization Constraints

From `/Users/sneh/research/src/phase1/config/labeling_config.py`:

```python
LABEL_BALANCE_CONSTRAINTS = {
    'min_long_pct': 0.05,            # Minimum long share
    'min_short_pct': 0.05,           # Minimum short share
    'min_neutral_pct': 0.10,         # MINIMUM neutral (HARD constraint)
    'target_neutral_low': 0.20,      # Target neutral range lower
    'target_neutral_high': 0.30,     # Target neutral range upper
    'max_neutral_pct': 0.40,         # Maximum neutral (too few signals)
    'min_short_signal_ratio': 0.10,  # Minimum short among signals
    'max_short_signal_ratio': 0.90,  # Maximum short among signals
    'min_any_class_pct': 0.10,       # Minimum for ANY class
}
```

### 9.2 Why 20-30% Neutral Rate?

| Neutral Rate | Trading Implication |
|--------------|---------------------|
| < 10% | Overtrading, high costs, probably noise |
| 10-20% | Aggressive, might work in trending markets |
| **20-30%** | Balanced, selective, cost-effective |
| 30-40% | Conservative, might miss moves |
| > 40% | Too few signals, not enough trades |

**Target 20-30% because:**
- Filters out ~25% of bars as "no trade" = noise reduction
- Keeps ~75% as actionable signals = sufficient trade frequency
- Transaction costs amortized over meaningful moves

### 9.3 Hard Constraints vs. Soft Penalties

```python
# From fitness.py

# HARD CONSTRAINT: Neutral below 10% = fitness -10000
if neutral_pct < min_neutral_pct:
    return -10000.0 + (neutral_pct * 10.0)

# SOFT PENALTY: Neutral below 20% but above 10%
if neutral_pct < target_neutral_low:
    deviation = target_neutral_low - neutral_pct
    neutral_score = 10.0 - (deviation * 40.0)
    if neutral_pct < 0.15:
        neutral_score -= (0.15 - neutral_pct) * 50.0
```

---

## 10. Data Flow: From Market Data to Tradeable Signals

### 10.1 Complete Pipeline

```
Raw OHLCV (1-min bars)
        |
        v
[1] Resample to 5-min
        |
        v
[2] Feature Engineering (150+ indicators)
    - Momentum: RSI, MACD, ROC, Stoch
    - Volatility: ATR, Bollinger, Keltner
    - Volume: OBV, VWAP, Volume Profile
    - Wavelets: Multi-scale decomposition
    - Microstructure: Spread, order flow
        |
        v
[3] Initial Triple-Barrier Labeling
    - Apply default barriers
    - Generate initial labels
        |
        v
[4] GA/Optuna Optimization
    - Optimize k_up, k_down, max_bars
    - Subject to balance constraints
    - With transaction cost penalties
        |
        v
[5] Final Labels + Quality Scores
    - Apply optimized barriers
    - Compute quality (speed, MAE, MFE, PTG, TWDD)
    - Assign sample weights (0.5x-1.5x)
        |
        v
[6] Train/Val/Test Splits
    - 70/15/15 chronological
    - 60-bar purge
    - 1440-bar embargo (~5 days)
        |
        v
[7] Model Training (Phase 2)
    - 12 model families available
    - Sample weights applied
    - Cross-validation with purge/embargo
        |
        v
[8] Predictions -> Trading Signals
    - Model predicts {-1, 0, +1}
    - 0 = hold/flat
    - +1 = go long
    - -1 = go short
```

### 10.2 Signal Interpretation

| Model Output | Position Action | Hold Period |
|--------------|-----------------|-------------|
| +1 (Long) | Buy/Hold Long | Until barrier hit or timeout |
| -1 (Short) | Sell/Hold Short | Until barrier hit or timeout |
| 0 (Neutral) | Flat/No Position | Until next non-zero signal |

### 10.3 Trade Lifecycle

```
Time t: Model predicts +1 (Long) for H20

Entry:
  - Buy at t+1 open (or close, depending on execution model)
  - Entry price = P_0 = close[t]
  - Set upper barrier = P_0 + 3.0 * ATR_14[t]
  - Set lower barrier = P_0 - 2.1 * ATR_14[t]

Exit (one of three):
  1. Upper barrier hit -> Exit with profit
  2. Lower barrier hit -> Exit with loss
  3. 50 bars elapsed   -> Exit at market (timeout)

P&L:
  - Gross P&L = Exit price - Entry price
  - Transaction costs = 1.5 ticks * $1.25/tick = $1.875 (low vol)
  - Net P&L = Gross P&L - Transaction costs
```

---

## 11. Key Files Reference

| File | Purpose |
|------|---------|
| `src/phase1/stages/labeling/triple_barrier.py` | Core labeling algorithm |
| `src/phase1/config/barriers_config.py` | Symbol-specific barrier params, costs |
| `src/phase1/config/labeling_config.py` | Label balance constraints |
| `src/phase1/stages/ga_optimize/fitness.py` | Fitness function with cost penalties |
| `src/phase1/stages/ga_optimize/optuna_optimizer.py` | TPE-based parameter optimization |
| `src/phase1/stages/final_labels/core.py` | Quality scoring, sample weights |
| `src/phase1/stages/splits/core.py` | Purge/embargo split logic |
| `src/common/horizon_config.py` | Horizon scaling, purge/embargo config |
| `src/phase1/config/regime_config.py` | Regime-adaptive barrier adjustments |

---

## 12. Conclusion

This ML factory implements a quantitatively rigorous approach to generating trading signals:

1. **Triple-barrier labeling** translates price action into actionable labels with built-in risk management
2. **Asymmetric barriers** counteract structural market biases (equity drift for MES, none for MGC)
3. **Transaction cost modeling** ensures generated signals are profitable after realistic costs
4. **Quality weighting** prioritizes clean, high-confidence signals during training
5. **Proper purge/embargo** prevents lookahead bias in cross-validation

The expected Sharpe ratios of 0.3-1.2 represent the theoretical ceiling. Actual model performance depends on:
- Model accuracy in predicting labels
- Execution quality (slippage realization)
- Market regime stability
- Risk management overlay

**The system is production-ready for Phase 2 model training and backtesting.**
