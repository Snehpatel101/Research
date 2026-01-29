### CONFIG
- We DON'T have a config option for what type of MTF mode we want to do MTF INDICATROS.
- Optuna optimization for?
- feature optimization selection
- compatability matrix visually for the user , or reccomeneded settings.

### Features, also not included
-Inference pipeline
-Serving bundle.
- meta learner + sizing learner? For 5 model esembles.



Got it — I’ll translate those warnings into concrete refactors: how to stop DataFrame fragmentation, speed up feature generation, reduce memory churn, and handle the sqrt/NaN and fillna deprecation cleanly (with drop-in code patterns).


You’re hitting two *real* issues here:

1. **Pandas “highly fragmented”** = you’re adding hundreds/thousands of columns one-at-a-time (`df[col] = ...`, `insert`, loops). That turns the underlying BlockManager into confetti and everything slows to a crawl.

2. A couple **math / API warnings** (`sqrt` invalid value; `fillna(method=...)` deprecation) that you should fix so you’re not quietly generating garbage features.

Below are the highest-leverage improvements (with drop-in patterns).

---

## 1) Stop adding columns one-by-one (main fix for fragmentation)

### ✅ Best pattern: build a dict of new columns, concat once

Instead of:

```py
df["hour"] = ...
df["hour_sin"] = ...
df["hour_cos"] = ...
...
```

Do:

```py
new_cols = {
    "hour": df["datetime"].dt.hour.astype("int16"),
    "hour_sin": np.sin(2 * np.pi * df["datetime"].dt.hour / 24),
    "hour_cos": np.cos(2 * np.pi * df["datetime"].dt.hour / 24),
}
df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
```

### ✅ If you’re in loops: collect per-loop outputs, concat once at the end

Instead of:

```py
for period in periods:
    df[f"acf_{period}"] = ...
```

Do:

```py
out = {}
for period in periods:
    out[f"acf_{period}"] = ...
df = pd.concat([df, pd.DataFrame(out, index=df.index)], axis=1)
```

### ✅ If you must assign, at least do it in batches

Even `df = df.assign(**out)` is better than repeated `df[col] = ...`, but `concat` tends to be most predictable for big batches.

---

## 2) Defragment strategically (cheap “band-aid” when needed)

If you have a stage that *must* do incremental additions (maybe legacy code), then **defragment at stage boundaries**:

```py
# end of stage
df = df.copy()
```

This is exactly what the warning suggests. It’s not as good as batching, but it can prevent runaway slowdown.

---

## 3) Make a “FeatureFrameBuilder” utility (so every stage follows the same rule)

A simple pattern that prevents this problem across your pipeline:

```py
class FeatureBuilder:
    def __init__(self, index):
        self.index = index
        self.cols = {}

    def add(self, name, series):
        self.cols[name] = series

    def add_many(self, mapping: dict):
        self.cols.update(mapping)

    def to_frame(self):
        return pd.DataFrame(self.cols, index=self.index)

# usage
fb = FeatureBuilder(df.index)
fb.add("supertrend", pd.Series(supertrend, index=df.index).shift(1))
fb.add("supertrend_direction", pd.Series(direction, index=df.index).shift(1))
df = pd.concat([df, fb.to_frame()], axis=1)
```

Now every stage can return `features_df`, and your pipeline can do **one concat per stage**.

---

## 4) Fix the `sqrt` invalid value warning (you’re taking sqrt of negatives)

This line:

```
RuntimeWarning: invalid value encountered in sqrt
```

Means you’ve got something like:

```py
np.sqrt(x)
```

where `x` sometimes < 0 (common when a variance estimate goes slightly negative due to numeric noise, or a formula assumes positivity but you feed it noisy data).

### ✅ Safe patterns (choose based on what “negative” means for that feature)

**A) Clamp to 0 (common for variance-like quantities):**

```py
x = np.maximum(x, 0.0)
y = np.sqrt(x)
```

**B) Preserve sign (if negative is meaningful):**

```py
y = np.sign(x) * np.sqrt(np.abs(x))
```

**C) Mask invalids (if negative should become NaN):**

```py
y = np.sqrt(np.where(x >= 0, x, np.nan))
```

You should locate the exact feature doing this (search for `sqrt(` in your feature stages) and apply the right option.

---

## 5) Replace deprecated `fillna(method="bfill")`

You have:

```
FutureWarning: DataFrame.fillna with 'method' is deprecated
```

Replace:

```py
features = features.fillna(method="bfill").fillna(0)
```

With:

```py
features = features.bfill().fillna(0)
```

(Or `ffill()` depending on intent.)

---

## 6) Don’t create temporary Series repeatedly inside tight loops

You’ve got patterns like:

```py
df["supertrend"] = pd.Series(supertrend).shift(1).values
```

Better:

* Avoid `pd.Series(...)` without index
* Avoid `.values` unless you *need* ndarray
* Keep alignment explicit

```py
supertrend_s = pd.Series(supertrend, index=df.index).shift(1)
direction_s  = pd.Series(direction, index=df.index).shift(1)

df = pd.concat([df, pd.DataFrame({
    "supertrend": supertrend_s,
    "supertrend_direction": direction_s,
}, index=df.index)], axis=1)
```

---

## 7) Use smaller dtypes for categorical/time/session features (big memory win)

For temporal/session one-hots and regimes:

* `session_*` can be `uint8`
* `hour/minute/dayofweek` can be `uint8/int16`
* `trend_regime` can be `int8` (-1/0/1)
* `volatility_regime` can be `uint8`

Example:

```py
hour = df["datetime"].dt.hour.astype("uint8")
dow  = df["datetime"].dt.dayofweek.astype("uint8")
trend_regime = trend_regime.astype("int8")
vol_regime   = vol_regime.astype("uint8")
```

Less memory → fewer cache misses → faster everything.

---

## 8) Reduce rolling overhead (common hotspot in your logs)

You’re doing lots of:

```py
returns.rolling(period).skew().shift(1)
returns.rolling(period).kurt().shift(1)
...
```

Two improvements:

### A) Reuse the rolling object

```py
r = returns.rolling(period)
out[f"skew_{period}"] = r.skew().shift(1)
out[f"kurt_{period}"] = r.kurt().shift(1)
```

### B) Compute multiple windows in a single pass where possible

If you have many periods, consider computing in fewer calls (depends on feature). Pandas doesn’t batch all rolling stats, but you can at least structure your code so each period does a block of stats and you concat once.

---

## 9) Stage outputs: return `features_df`, not mutated `df`

A clean architecture that also prevents fragmentation:

* each feature stage takes `df` (read-only)
* stage returns `features_df` (only new cols)
* pipeline does:

```py
df = pd.concat([df, f_trend, f_temporal, f_micro, f_entropy, ...], axis=1)
df = df.copy()  # optional defrag once at end
```

This also makes it easy to cache, test, and profile stages.

---

## 10) Make shifting consistent and cheap

You’re doing `.shift(1)` everywhere (good for leakage control). Standardize it so you don’t forget:

* Compute raw feature columns
* Shift all *predictor* columns once at the end of the stage

Example:

```py
features = pd.DataFrame(out, index=df.index)
features = features.shift(1)
df = pd.concat([df, features], axis=1)
```

This often cuts repeated shift calls substantially.

---

### If you want the fastest “first pass” fix

Do these three things and most warnings + slowdown go away:

1. In every stage, **replace all `df[col] = ...` with building `out = {}` and one `pd.concat`**.
2. Add a **single `df = df.copy()` at the end of each stage** while refactoring (temporary safety).
3. Fix the `sqrt` + `bfill` warnings (`np.maximum(x,0)` and `.bfill()`).

If you paste one of your stage files (e.g. `temporal.py` or the chunk of `microstructure.py` that adds `new_features`), I’ll rewrite that stage into the “batch + concat once” style so it’s directly usable.


ExperimentConfig assembled:
  Name:            my_experiment
  Symbol:          MES
  Data path:       /content/Research/data/raw/mes-1m_data_2020.parquet
  Input TF:        1min
  Target TF:       5min
  MTF enabled:     True
  MTF mode:        indicators
  MTF timeframes:  ['10min', '15min', '30min']
  Models:          ['lightgbm', 'tcn', 'patchtst']
  Horizons:        [5, 15]
  Ensemble:        True
  Meta-learner:    ridge_meta
  Optimize for:    sharpe_ratio
  Optuna trials:   50
  Labeling:        triple_barrier
  Features:        ['price', 'momentum', 'volatility', 'volume', 'trend', 'regime', 'wavelet']
  Backtest:        True
  Pos. sizing:     fixed
  Random seed:     42
  Output dir:      experiments/runs/20260128_225816

Starting ML Factory pipeline...
This may take a while depending on model selection and Optuna trials.

============================================================
Starting ML Factory Experiment: my_experiment
============================================================

[Phase 1/4] Data Pipeline
Running data pipeline...
  Loaded: 350464 rows from /content/Research/data/raw/mes-1m_data_2020.parquet
  Generating features...
/content/Research/src/data/pipeline/stages/features/trend.py:167: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  df["supertrend"] = pd.Series(supertrend).shift(1).values
/content/Research/src/data/pipeline/stages/features/trend.py:168: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  df["supertrend_direction"] = pd.Series(direction).shift(1).values
/content/Research/src/data/pipeline/stages/features/temporal.py:38: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  df["hour"] = df["datetime"].dt.hour
/content/Research/src/data/pipeline/stages/features/temporal.py:39: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
/content/Research/src/data/pipeline/stages/features/temporal.py:40: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
/content/Research/src/data/pipeline/stages/features/temporal.py:43: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  df["minute"] = df["datetime"].dt.minute
/content/Research/src/data/pipeline/stages/features/temporal.py:44: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  df["minute_sin"] = np.sin(2 * np.pi * df["minute"] / 60)
/content/Research/src/data/pipeline/stages/features/temporal.py:45: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  df["minute_cos"] = np.cos(2 * np.pi * df["minute"] / 60)
/content/Research/src/data/pipeline/stages/features/temporal.py:48: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  df["dayofweek"] = df["datetime"].dt.dayofweek
/content/Research/src/data/pipeline/stages/features/temporal.py:49: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  df["dayofweek_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7)
/content/Research/src/data/pipeline/stages/features/temporal.py:50: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  df["dayofweek_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7)
/content/Research/src/data/pipeline/stages/features/temporal.py:64: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  df["session"] = df["hour"].apply(get_session)
/content/Research/src/data/pipeline/stages/features/temporal.py:68: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  df[f"session_{session}"] = (df["session"] == session).astype(int)
/content/Research/src/data/pipeline/stages/features/temporal.py:68: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  df[f"session_{session}"] = (df["session"] == session).astype(int)
/content/Research/src/data/pipeline/stages/features/temporal.py:68: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  df[f"session_{session}"] = (df["session"] == session).astype(int)
/content/Research/src/data/pipeline/stages/features/regime.py:96: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  df["volatility_regime"] = (df["hvol_20"] > hvol_median).astype(int)
/content/Research/src/data/pipeline/stages/features/regime.py:113: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  df["trend_regime"] = np.where(uptrend, 1, np.where(downtrend, -1, 0))
/content/Research/src/data/pipeline/stages/features/price_features.py:150: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  df[col] = autocorr
/content/Research/src/data/pipeline/stages/features/price_features.py:150: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  df[col] = autocorr
/content/Research/src/data/pipeline/stages/features/price_features.py:150: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  df[col] = autocorr
/content/Research/src/data/pipeline/stages/features/price_features.py:150: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  df[col] = autocorr
/content/Research/src/data/pipeline/stages/features/price_features.py:150: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  df[col] = autocorr
/content/Research/src/data/pipeline/stages/features/price_features.py:187: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  df["clv"] = clv_raw.shift(1)
/content/Research/src/data/pipeline/stages/features/volatility.py:348: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  df[skew_col] = returns.rolling(period).skew().shift(1)
/content/Research/src/data/pipeline/stages/features/volatility.py:353: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  df[kurt_col] = returns.rolling(period).kurt().shift(1)
/content/Research/src/data/pipeline/stages/features/volatility.py:348: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  df[skew_col] = returns.rolling(period).skew().shift(1)
/content/Research/src/data/pipeline/stages/features/volatility.py:353: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  df[kurt_col] = returns.rolling(period).kurt().shift(1)
/content/Research/src/data/pipeline/stages/features/microstructure.py:76: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  df["micro_amihud"] = amihud_raw.shift(1)
/content/Research/src/data/pipeline/stages/features/microstructure.py:83: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  df[col_name] = amihud_raw.rolling(window=period).mean().shift(1)
/content/Research/src/data/pipeline/stages/features/microstructure.py:83: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  df[col_name] = amihud_raw.rolling(window=period).mean().shift(1)
/content/Research/src/data/pipeline/stages/features/microstructure.py:130: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  df["micro_roll_spread"] = roll_spread_raw.shift(1)
/content/Research/src/data/pipeline/stages/features/microstructure.py:135: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  df["micro_roll_spread_pct"] = roll_spread_pct.shift(1)
/content/Research/src/data/pipeline/stages/features/microstructure.py:181: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  df["micro_kyle_lambda"] = kyle_lambda_raw.shift(1)
/content/Research/src/data/pipeline/stages/features/microstructure.py:250: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  df["micro_cs_spread"] = pd.Series(cs_spread_raw, index=df.index).shift(1)
/content/Research/src/data/pipeline/stages/features/microstructure.py:290: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  df["micro_rel_spread"] = rel_spread_raw.shift(1)
/content/Research/src/data/pipeline/stages/features/microstructure.py:297: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  df[col_name] = rel_spread_raw.rolling(window=period).mean().shift(1)
/content/Research/src/data/pipeline/stages/features/microstructure.py:297: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  df[col_name] = rel_spread_raw.rolling(window=period).mean().shift(1)
/content/Research/src/data/pipeline/stages/features/microstructure.py:336: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  df["micro_volume_imbalance"] = imbalance_raw.shift(1)
/content/Research/src/data/pipeline/stages/features/microstructure.py:339: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  df["micro_cum_imbalance_20"] = imbalance_raw.rolling(window=20).sum().shift(1)
/content/Research/src/data/pipeline/stages/features/microstructure.py:390: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  df[col_name] = intensity_raw.shift(1)
/content/Research/src/data/pipeline/stages/features/microstructure.py:390: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  df[col_name] = intensity_raw.shift(1)
/content/Research/src/data/pipeline/stages/features/microstructure.py:442: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  df[col_name] = efficiency_raw.shift(1)
/content/Research/src/data/pipeline/stages/features/microstructure.py:442: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  df[col_name] = efficiency_raw.shift(1)
/content/Research/src/data/pipeline/stages/features/microstructure.py:485: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  df["micro_vol_ratio"] = vol_ratio_raw.shift(1)
/usr/local/lib/python3.12/dist-packages/pandas/core/arraylike.py:399: RuntimeWarning: invalid value encountered in sqrt
  result = getattr(ufunc, method)(*inputs, **kwargs)
/content/Research/src/data/pipeline/stages/features/microstructure_proxies.py:504: FutureWarning: DataFrame.fillna with 'method' is deprecated and will raise in a future version. Use obj.ffill() or obj.bfill() instead.
  features = features.fillna(method="bfill").fillna(0)
/content/Research/src/data/pipeline/stages/features/microstructure.py:590: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  df[col] = new_features[col]
/content/Research/src/data/pipeline/stages/features/microstructure.py:590: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  df[col] = new_features[col]
/content/Research/src/data/pipeline/stages/features/microstructure.py:590: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  df[col] = new_features[col]
/content/Research/src/data/pipeline/stages/features/microstructure.py:590: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  df[col] = new_features[col]
/content/Research/src/data/pipeline/stages/features/microstructure.py:590: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  df[col] = new_features[col]
/content/Research/src/data/pipeline/stages/features/microstructure.py:590: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  df[col] = new_features[col]
/content/Research/src/data/pipeline/stages/features/microstructure.py:590: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  df[col] = new_features[col]
/content/Research/src/data/pipeline/stages/features/microstructure.py:590: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  df[col] = new_features[col]
/content/Research/src/data/pipeline/stages/features/microstructure.py:590: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  df[col] = new_features[col]
/content/Research/src/data/pipeline/stages/features/microstructure.py:590: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  df[col] = new_features[col]
/content/Research/src/data/pipeline/stages/features/microstructure.py:590: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  df[col] = new_features[col]
/content/Research/src/data/pipeline/stages/features/microstructure.py:590: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  df[col] = new_features[col]
/content/Research/src/data/pipeline/stages/features/microstructure.py:590: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  df[col] = new_features[col]
/content/Research/src/data/pipeline/stages/features/microstructure.py:590: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  df[col] = new_features[col]
/content/Research/src/data/pipeline/stages/features/microstructure.py:590: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  df[col] = new_features[col]
/content/Research/src/data/pipeline/stages/features/microstructure.py:590: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  df[col] = new_features[col]
/content/Research/src/data/pipeline/stages/features/microstructure.py:590: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  df[col] = new_features[col]
/content/Research/src/data/pipeline/stages/features/microstructure.py:590: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  df[col] = new_features[col]
/content/Research/src/data/pipeline/stages/features/entropy.py:234: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  df[col_name] = pd.Series(entropy, index=df.index).shift(1)
/content/Research/src/data/pipeline/stages/features/entropy.py:239: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  df[norm_col_name] = df[col_name] / max_entropy
/content/Research/src/data/pipeline/stages/features/entropy.py:234: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  df[col_name] = pd.Series(entropy, index=df.index).shift(1)
/content/Research/src/data/pipeline/stages/features/entropy.py:239: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  df[norm_col_name] = df[col_name] / max_entropy
/content/Research/src/data/pipeline/stages/features/entropy.py:234: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  df[col_name] = pd.Series(entropy, index=df.index).shift(1)
/content/Research/src/data/pipeline/stages/features/entropy.py:239: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  df[norm_col_name] = df[col_name] / max_entropy
/content/Research/src/data/pipeline/stages/features/entropy.py:420: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  df[col_name] = pd.Series(lz_values, index=df.index).shift(1)
/content/Research/src/data/pipeline/stages/features/entropy.py:420: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  df[col_name] = pd.Series(lz_values, index=df.index).shift(1)
/content/Research/src/data/pipeline/stages/features/entropy.py:420: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  df[col_name] = pd.Series(lz_values, index=df.index).shift(1)
/content/Research/src/data/pipeline/stages/features/entropy.py:621: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  df[col_name] = pd.Series(apen, index=df.index).shift(1)
/content/Research/src/data/pipeline/stages/features/entropy.py:621: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  df[col_name] = pd.Series(apen, index=df.index).shift(1)
/content/Research/src/data/pipeline/stages/features/entropy.py:1063: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  df[col_name] = pd.Series(sampen, index=df.index).shift(1)
/content/Research/src/data/pipeline/stages/features/entropy.py:1063: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  df[col_name] = pd.Series(sampen, index=df.index).shift(1)
/content/Research/src/data/pipeline/stages/features/entropy.py:837: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  df[col_name] = pd.Series(hurst, index=df.index).shift(1)
/content/Research/src/data/pipeline/stages/features/entropy.py:837: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  df[col_name] = pd.Series(hurst, index=df.index).shift(1)
/content/Research/src/data/pipeline/stages/features/entropy.py:837: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  df[col_name] = pd.Series(hurst, index=df.index).shift(1)
/content/Research/src/data/pipeline/stages/features/entropy.py:854: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  df["hurst_regime"] = np.select(conditions, choices, default=np.nan)
/content/Research/src/data/pipeline/stages/features/wavelets.py:154: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  df[approx_col] = pd.Series(_normalize_coefficients(approx)).shift(1).values
/content/Research/src/data/pipeline/stages/features/wavelets.py:164: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  df[detail_col] = pd.Series(_normalize_coefficients(details[lev])).shift(1).values
/content/Research/src/data/pipeline/stages/features/wavelets.py:164: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  df[detail_col] = pd.Series(_normalize_coefficients(details[lev])).shift(1).values
/content/Research/src/data/pipeline/stages/features/wavelets.py:164: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  df[detail_col] = pd.Series(_normalize_coefficients(details[lev])).shift(1).values
/content/Research/src/data/pipeline/stages/features/wavelets.py:154: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  df[approx_col] = pd.Series(_normalize_coefficients(approx)).shift(1).values
/content/Research/src/data/pipeline/stages/features/wavelets.py:164: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  df[detail_col] = pd.Series(_normalize_coefficients(details[lev])).shift(1).values
/content/Research/src/data/pipeline/stages/features/wavelets.py:164: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  df[detail_col] = pd.Series(_normalize_coefficients(details[lev])).shift(1).values
/content/Research/src/data/pipeline/stages/features/wavelets.py:164: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  df[detail_col] = pd.Series(_normalize_coefficients(details[lev])).shift(1).values
/content/Research/src/data/pipeline/stages/features/wavelets.py:200: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  df[approx_energy_col] = pd.Series(np.log1p(approx_energy)).shift(1).values
/content/Research/src/data/pipeline/stages/features/wavelets.py:207: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  df[energy_col] = pd.Series(np.log1p(detail_energies[lev])).shift(1).values
/content/Research/src/data/pipeline/stages/features/wavelets.py:207: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  df[energy_col] = pd.Series(np.log1p(detail_energies[lev])).shift(1).values
/content/Research/src/data/pipeline/stages/features/wavelets.py:207: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  df[energy_col] = pd.Series(np.log1p(detail_energies[lev])).shift(1).values
/content/Research/src/data/pipeline/stages/features/wavelets.py:212: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  df[ratio_col] = pd.Series(energy_ratio).shift(1).values
/content/Research/src/data/pipeline/stages/features/wavelets.py:200: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  df[approx_energy_col] = pd.Series(np.log1p(approx_energy)).shift(1).values
/content/Research/src/data/pipeline/stages/features/wavelets.py:207: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  df[energy_col] = pd.Series(np.log1p(detail_energies[lev])).shift(1).values
/content/Research/src/data/pipeline/stages/features/wavelets.py:207: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  df[energy_col] = pd.Series(np.log1p(detail_energies[lev])).shift(1).values
/content/Research/src/data/pipeline/stages/features/wavelets.py:207: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  df[energy_col] = pd.Series(np.log1p(detail_energies[lev])).shift(1).values
/content/Research/src/data/pipeline/stages/features/wavelets.py:212: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  df[ratio_col] = pd.Series(energy_ratio).shift(1).values
/content/Research/src/data/pipeline/stages/features/wavelets.py:253: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  df[vol_col] = pd.Series(wavelet_vol).shift(1).values
/content/Research/src/data/pipeline/stages/features/wavelets.py:300: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  df[strength_col] = pd.Series(trend_strength).shift(1).values
/content/Research/src/data/pipeline/stages/features/wavelets.py:301: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  df[direction_col] = pd.Series(trend_direction).shift(1).values
WARNING:src.data.pipeline.stages.mtf.generator:include_ohlcv and include_indicators are deprecated. Use mode='bars', 'indicators', or 'both' instead.
WARNING:src.data.pipeline.stages.features.nan_handling:Columns with 100% NaN (1): ['return_autocorr_lag20']
WARNING:src.data.pipeline.stages.features.nan_handling:Dropping 1 columns exceeding NaN threshold: ['return_autocorr_lag20']
  Features: 226 columns
  Generating labels...
    label_h5: {0: 188526, -1: 49592, 1: 13760, -99: 63}
    label_h15: {-1: 108831, 0: 92000, 1: 50990, -99: 120}
  Pipeline complete: 251941 rows, 229 columns
  Label distribution: {0: 188526, -1: 49592, 1: 13760, -99: 63}

[Phase 2/4] Model Training
Training 3 models...
  Models: ['lightgbm', 'tcn', 'patchtst']
  Mode: standard
ERROR:src.factory:MLFactory experiment failed
Traceback (most recent call last):
  File "/content/Research/src/factory.py", line 275, in run
    training_result = self._run_training(df)
                      ^^^^^^^^^^^^^^^^^^^^^^
  File "/content/Research/src/factory.py", line 588, in _run_training
    result = orchestrator.train(df)
             ^^^^^^^^^^^^^^^^^^^^^^
  File "/content/Research/src/models/training/unified_orchestrator.py", line 501, in train
    self._pre_training_validation(df)
  File "/content/Research/src/models/training/unified_orchestrator.py", line 458, in _pre_training_validation
    raise PreTrainingValidationError(
src.core.exceptions.PreTrainingValidationError: Pre-training validation failed:
  - Contract violation for lightgbm: Too many features: model max is 200, data has 218
  - Contract violation for tcn: Data rank mismatch: model expects 3D, data is 2D; Too many features: model max is 120, data has 218
  - Contract violation for patchtst: Data rank mismatch: model expects 4D, data is 2D; Too many features: model max is 10, data has 218

To bypass validation, set in PipelineConfig:
  - strict_validation=False (disable contract validation)
  - check_leakage=False (disable leakage detection)
  - check_lookahead=False (disable lookahead audit)

============================================================
ML Factory Experiment: FAILED
============================================================
Run ID: 20260128_225816
Duration: 11108.8s
Models Trained: 0
Output: experiments/runs/20260128_225816

Error: Pre-training validation failed:
  - Contract violation for lightgbm: Too many features: model max is 200, data has 218
  - Contract violation for tcn: Data rank mismatch: model expects 3D, data is 2D; Too many features: model max is 120, data has 218
  - Contract violation for patchtst: Data rank mismatch: model expects 4D, data is 2D; Too many features: model max is 10, data has 218

To bypass validation, set in PipelineConfig:
  - strict_validation=False (disable contract validation)
  - check_leakage=False (disable leakage detection)
  - check_lookahead=False (disable lookahead audit)
============================================================

ERROR: Factory run failed: Pre-training validation failed:
  - Contract violation for lightgbm: Too many features: model max is 200, data has 218
  - Contract violation for tcn: Data rank mismatch: model expects 3D, data is 2D; Too many features: model max is 120, data has 218
  - Contract violation for patchtst: Data rank mismatch: model expects 4D, data is 2D; Too many features: model max is 10, data has 218

To bypass validation, set in PipelineConfig:
  - strict_validation=False (disable contract validation)
  - check_leakage=False (disable leakage detection)
  - check_lookahead=False (disable lookahead audit)

To resume from the last checkpoint, run:
  result = factory.resume_from_checkpoint()

Or fix the issue above and re-run this cell.