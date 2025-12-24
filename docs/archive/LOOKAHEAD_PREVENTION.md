# Lookahead Bias Prevention Guide

**Quick Reference for Feature Engineering**

---

## The Golden Rule

> **Features at bar[t] must ONLY use data up to and including bar[t-1]**

If your feature at bar[t] includes ANY information from bar[t], you have lookahead bias.

---

## Why This Matters

In live trading:
- We make decisions at the END of bar[t-1]
- We enter positions at the OPEN of bar[t]
- We DON'T know bar[t]'s OHLC until bar[t] closes

If features use bar[t] data, backtest performance will be artificially inflated and won't match live trading.

---

## Common Patterns That Cause Lookahead

### 1. Rolling Windows Without Shift
```python
# ❌ WRONG - includes current bar
df['sma_20'] = df['close'].rolling(20).mean()
# sma_20[t] = mean(close[t-19:t+1]) includes close[t]

# ✅ CORRECT - excludes current bar
sma = df['close'].rolling(20).mean()
df['sma_20'] = sma.shift(1)
# sma_20[t] = mean(close[t-20:t]) excludes close[t]
```

### 2. Binary Flags Based on Current Bar
```python
# ❌ WRONG - flag based on current RSI
df['rsi_overbought'] = (df['rsi_14'] > 70).astype(int)
# rsi_overbought[t] uses rsi_14[t] which includes close[t]

# ✅ CORRECT - flag based on prior RSI
df['rsi_overbought'] = (df['rsi_14'] > 70).astype(int).shift(1)
# rsi_overbought[t] uses rsi_14[t-1]
```

### 3. Crossover Signals
```python
# ❌ WRONG - crossover detected at current bar
df['cross_up'] = ((df['fast'] > df['slow']) &
                  (df['fast'].shift(1) <= df['slow'].shift(1))).astype(int)
# Detects crossover AT bar[t], but we can't act until bar[t] closes

# ✅ CORRECT - crossover signal available before entry
df['cross_up'] = ((df['fast'] > df['slow']) &
                  (df['fast'].shift(1) <= df['slow'].shift(1))).astype(int).shift(1)
# cross_up[t] signals crossover between bar[t-1] and bar[t-2]
```

### 4. Cumulative Returns
```python
# ❌ WRONG - includes current bar's return
returns = df['close'].pct_change()
df['cum_ret_20'] = returns.rolling(20).sum()
# cum_ret_20[t] includes returns[t]

# ✅ CORRECT - excludes current bar's return
returns = df['close'].pct_change()
returns_shifted = returns.shift(1)
df['cum_ret_20'] = returns_shifted.rolling(20).sum()
# cum_ret_20[t] uses returns up to bar[t-1]
```

---

## How to Fix Lookahead Bias

### Step 1: Identify the Issue
Ask yourself: "Does this feature at bar[t] use ANY data from bar[t]?"

Common culprits:
- Rolling statistics (mean, std, sum, min, max)
- Binary conditions (>, <, ==)
- Crossovers
- Percentage changes
- Ratios using current bar

### Step 2: Apply shift(1)
```python
# General pattern:
feature_raw = calculate_feature(df)
df['feature'] = feature_raw.shift(1)
```

### Step 3: Add Comment
```python
# ANTI-LOOKAHEAD: Shift 1 bar forward so feature[t] uses data up to bar[t-1]
df['feature'] = feature_raw.shift(1)
```

### Step 4: Test
```python
# Feature at bar[t] should NOT change when bar[t] data changes
df1 = calculate_features(data)
df2_modified = data.copy()
df2_modified.loc[t, 'close'] *= 1.5  # Dramatic change at bar t
df2 = calculate_features(df2_modified)

assert df1.loc[t, 'feature'] == df2.loc[t, 'feature'], "Lookahead detected!"
```

---

## Exception: Point-in-Time Features

Some features are naturally point-in-time and don't need shifting:

### 1. Simple Price Transformations
```python
# ✅ OK - uses prior bar's close
df['log_price'] = np.log(df['close'].shift(1))

# ✅ OK - ratio uses prior bars
df['hl_ratio'] = (df['high'].shift(1) / df['low'].shift(1))
```

### 2. Lagged Features
```python
# ✅ OK - explicitly using prior bars
df['close_lag1'] = df['close'].shift(1)
df['close_lag5'] = df['close'].shift(5)
```

### 3. Returns (if properly calculated)
```python
# ✅ OK - return from bar[t-2] to bar[t-1]
df['returns'] = df['close'].pct_change().shift(1)
```

---

## Testing Checklist

Before merging any feature code:

- [ ] Does feature[t] use ONLY data up to bar[t-1]?
- [ ] Are all rolling windows shifted by 1?
- [ ] Are all binary flags shifted by 1?
- [ ] Are all crossover signals shifted by 1?
- [ ] Is there an ANTI-LOOKAHEAD comment?
- [ ] Does changing bar[t] leave feature[t] unchanged?
- [ ] Are there tests proving no lookahead?

---

## Examples from Codebase

### Cross-Asset Correlation (FIXED)
```python
# Calculate correlation
correlation = calculate_rolling_correlation_numba(
    mes_returns.astype(np.float64),
    mgc_returns.astype(np.float64),
    period=20
)

# ANTI-LOOKAHEAD: Shift 1 bar forward
df['mes_mgc_correlation_20'] = pd.Series(correlation).shift(1).values
```

### MACD Crossover (FIXED)
```python
# ANTI-LOOKAHEAD: Shift signals forward 1 bar
df['macd_cross_up'] = ((df['macd_line'] > df['macd_signal']) &
                       (df['macd_line'].shift(1) <= df['macd_signal'].shift(1))).astype(int).shift(1)
```

### RSI Flags (FIXED)
```python
df['rsi_14'] = calculate_rsi_numba(df['close'].values, 14)

# ANTI-LOOKAHEAD: Shift flags forward 1 bar
df['rsi_overbought'] = (df['rsi_14'] > 70).astype(int).shift(1)
df['rsi_oversold'] = (df['rsi_14'] < 30).astype(int).shift(1)
```

---

## Impact on NaN Values

Each shift(1) adds one leading NaN:
- Position 0: NaN (from shift)
- Position 1-19: NaN (from rolling window size 20)
- Position 20: First valid value (uses data from positions 0-19)

This is expected and correct. The pipeline already handles NaN removal.

---

## Performance Impact

After fixing lookahead bias, expect:
- **Sharpe Ratio:** ↓ 10-20%
- **Win Rate:** ↓ 2-5%
- **Max Drawdown:** ↑ 5-15%

This is GOOD - you're removing artificial edge that wouldn't exist in live trading.

---

## References

- **Test Suite:** `tests/phase_1_tests/stages/test_lookahead_bias_prevention.py`
- **Summary Doc:** `docs/LOOKAHEAD_BIAS_FIX_SUMMARY.md`
- **Fixed Files:**
  - `src/stages/features/cross_asset.py`
  - `src/stages/features/momentum.py`

---

## Questions?

**Q: Why shift(1) instead of using close.shift(1) in calculations?**

A: Both work, but shift(1) on the final feature is clearer and ensures consistency across all feature types.

**Q: What about features that use open price?**

A: Open[t] is known at the start of bar[t], so it's OK to use for signals that trigger at the open. But most features should still use prior bar data.

**Q: How do I test for lookahead?**

A: See `test_lookahead_bias_prevention.py` for comprehensive examples. The master test is `test_feature_at_t_independent_of_bar_t_data`.

**Q: Can I skip shifting for some features?**

A: NO. All features must follow the golden rule. If you think you have an exception, discuss with the team first.

---

**Remember:** Lookahead bias is the #1 killer of quantitative trading strategies. When in doubt, shift it out!
