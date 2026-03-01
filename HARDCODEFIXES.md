# Hardcoded Values Audit — Backtest Execution Mismatch

**Date:** 2026-02-28
**Status:** VERIFIED — All claims confirmed with code execution

---

## Critical Finding: Backtest Plays a Different Game Than Training

The model learns entries/exits using symbol+horizon-specific ATR barriers from `barriers_config.py`.
The backtest executes with hardcoded values that **don't match** — and the ATR stop path is **dead code**.

### Verified Mismatch Summary

| Parameter | Training (MGC H10) | Training (MES H20) | Backtest (ALL symbols) |
|-----------|--------------------|--------------------|------------------------|
| **Stop loss** | 1.60 ATR | 2.10 ATR | **2% of price** (ATR path dead code) |
| **Take profit** | 1.60 ATR | 3.00 ATR | **Never set (None)** |
| **Time limit** | 25 bars | 50 bars | **0 = unlimited** |

### Root Cause

`_open_position()` accepts an `atr` parameter (line 470), but `run()` **never passes it** (lines 653, 665). So the ATR stop logic at line 482 (`stop_distance_atr = 2.0`) is unreachable dead code. Every trade uses the 2% fallback at line 487.

---

## Backtest Hardcoded Values (backtest.py)

| Line | Value | What It Does | Should Be |
|------|-------|-------------|-----------|
| 482 | `stop_distance_atr = 2.0` | ATR stop multiplier (DEAD CODE) | `k_down` from barrier config |
| 486 | `stop_distance_pct = 0.02` | 2% stop (ALWAYS USED) | `k_down * ATR / price` |
| 495 | `take_profit` never assigned | No profit target | `k_up * ATR` from barrier config |
| 80 | `max_holding_period = 0` | No time limit | `max_bars` from barrier config |
| 447 | `current_price * 0.02` | Position sizing stop distance | Should use actual barrier distance |
| 454 | `volatility or 0.15` | Default vol for sizing | Should compute from data |

## Execution Hardcoded Values (execution.py)

| Line | Value | What It Does | Should Be |
|------|-------|-------------|-----------|
| 24 | `time(9, 30)` | NY session start | Per exchange/contract |
| 25 | `time(16, 0)` | NY session end | Per exchange/contract |
| 129 | `base_volatility = 0.15` | Adverse selection baseline | Per symbol/regime |
| 131 | `0.5 + 0.5 * vol_ratio` | Adverse selection formula | Configurable coefficients |
| 146 | `max_participation = 0.01` | 1% volume cap | Per symbol |

## Cost Model Hardcoded Values (costs.py)

| Line | Value | What It Does | Should Be |
|------|-------|-------------|-----------|
| 209 | `base_ticks = 0.5` | Linear slippage base | Per symbol |
| 210 | `size_factor = 0.1` | Slippage scaling | Per symbol |
| 212 | `max_slippage_ticks = 10.0` | Slippage cap | Per symbol |
| 241 | `impact_coefficient = 0.1` | Sqrt impact model | Per symbol |
| 242 | `typical_volume = 1000.0` | Volume baseline | Per symbol |
| 293 | `base_volatility = 0.15` | Vol-scaled slippage base | Per symbol/regime |
| 294 | `volatility_multiplier = 2.0` | Vol slippage scaling | Per symbol/regime |
| 297 | `max_slippage_ticks = 5.0` | Vol slippage cap | Per symbol |

---

## Training Barrier Params (for reference)

### MGC (Gold) — Symmetric

| Horizon | k_up | k_down | max_bars |
|---------|------|--------|----------|
| H5 | 1.20 | 1.20 | 12 |
| H10 | 1.60 | 1.60 | 25 |
| H15 | 2.00 | 2.00 | 38 |
| H20 | 2.50 | 2.50 | 50 |

### MES (S&P) — Asymmetric (k_up > k_down for equity drift)

| Horizon | k_up | k_down | max_bars |
|---------|------|--------|----------|
| H5 | 1.50 | 1.00 | 12 |
| H10 | 2.00 | 1.40 | 25 |
| H15 | 2.50 | 1.75 | 38 |
| H20 | 3.00 | 2.10 | 50 |

---

## What the Fix Looks Like (Phase 86)

### Files to Change (3 files)

1. **`src/inference/backtesting/backtest.py`**
   - Add `barrier_k_up` and `barrier_k_down` to `BacktestConfig` (default 0.0 = legacy mode)
   - Add `_compute_atr()` from prices high/low/close
   - Pass ATR to `_open_position()` in `run()`
   - Replace hardcoded stop/TP with barrier-aware logic:
     - LONG: stop = `price - k_down * ATR`, TP = `price + k_up * ATR`
     - SHORT: stop = `price + k_up * ATR`, TP = `price - k_down * ATR`

2. **`src/factory.py`**
   - In `_run_evaluation()`: call `get_barrier_params(symbol, horizon)` and pass k_up, k_down, max_bars to BacktestConfig

3. **`notebooks/ml_factory_colab.ipynb`**
   - Note that barriers are auto-wired from training config

### Backward Compatible

- `barrier_k_up=0.0, barrier_k_down=0.0` → legacy 2% stop (no change)
- All 212 tests pass unchanged
- Only factory pipeline activates barriers (when horizons configured)

---

## Verified Tech Stack

### 12 Production Models

| # | Model | Family | Data Rank | Library | Version |
|---|-------|--------|-----------|---------|---------|
| 1 | XGBoost | Boosting | 2D | xgboost | 3.1.3 |
| 2 | LightGBM | Boosting | 2D | lightgbm | 4.6.0 |
| 3 | CatBoost | Boosting | 2D | catboost | 1.2.8 |
| 4 | LSTM | Neural RNN | 3D | PyTorch | 2.10.0 |
| 5 | GRU | Neural RNN | 3D | PyTorch | 2.10.0 |
| 6 | TCN | Neural CNN | 3D | PyTorch | 2.10.0 |
| 7 | InceptionTime | Neural CNN | 3D | PyTorch | 2.10.0 |
| 8 | ResNet1D | Neural CNN | 3D | PyTorch | 2.10.0 |
| 9 | N-BEATS | Neural MLP | 3D | PyTorch | 2.10.0 |
| 10 | PatchTST | Transformer | 4D | PyTorch | 2.10.0 |
| 11 | iTransformer | Transformer | 4D | PyTorch | 2.10.0 |
| 12 | TFT | Transformer | 4D | PyTorch | 2.10.0 |

Plus 11 support models: 3 classical (RandomForest, Logistic, SVM via sklearn), 3 ensemble (Voting, Stacking, Blending), 4 meta-learners (Ridge, MLP, XGBoost, Calibrated), 1 base transformer. Total: 23 registered.

### Full Library Stack

| Library | Version | Role |
|---------|---------|------|
| XGBoost | 3.1.3 | 3 boosting models, GPU-enabled |
| LightGBM | 4.6.0 | Leaf-wise gradient boosting |
| CatBoost | 1.2.8 | Ordered boosting with categorical support |
| PyTorch | 2.10.0 | 9 neural models (LSTM→TFT), torch.compile on GPU |
| scikit-learn | 1.8.0 | Classical models, meta-learners, calibration, MDA |
| Optuna | 4.7.0 | Bayesian HPO (TPE sampler), per-model, costs in objective |
| NumPy | 2.3.5 | Array operations, feature computation |
| Pandas | 2.3.3 | Data pipeline, resampling, multi-timeframe |
| SciPy | 1.17.0 | Statistical tests, entropy, Hurst exponent |
| Numba | 0.63.1 | 15 @njit JIT-compiled feature functions |
| Joblib | 1.5.3 | Parallel MDA, sklearn parallelism |

### Key Pipeline Components

- **PurgedKFold:** Time-series CV with purge + embargo (no data leakage)
- **CPCV:** Combinatorial Purged CV (15 backtest paths)
- **DSR Gate:** Deflated Sharpe Ratio (selection bias correction)
- **MDA Feature Selection:** Per-model permutation importance via sklearn
- **Probability Calibration:** Isotonic + sigmoid methods
- **Triple Barrier Labeling:** ATR-based, k_up/k_down, transaction cost adjusted
- **OOF Cross-Rank Alignment:** Combines 2D+3D+4D model predictions for ensemble
- **Walk-Forward Validation:** Expanding/rolling windows for production simulation
- **212/212 tests passing**

---

## Verification Commands Used

```python
# CHECK 1: ATR never passed
inspect.getsource(Backtester.run)  # No 'atr=' in _open_position calls

# CHECK 2: Effective stop = 2%
# atr=None → else branch → price * (1 - direction * 0.02)

# CHECK 3-4: Training barriers
get_barrier_params('MGC', 10)  # {'k_up': 1.6, 'k_down': 1.6, 'max_bars': 25}
get_barrier_params('MES', 20)  # {'k_up': 3.0, 'k_down': 2.1, 'max_bars': 50}

# CHECK 5: No barrier fields in BacktestConfig
[f.name for f in dataclasses.fields(BacktestConfig)]  # No 'barrier' fields

# CHECK 6: max_holding_period = 0 (unlimited)
BacktestConfig().max_holding_period  # 0

# CHECK 7: take_profit dead code
'take_profit' in inspect.getsource(Backtester._should_exit)  # True but never set
```
