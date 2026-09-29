# Multi-Symbol Robustness Audit Report

**Date:** 2026-03-02
**Scope:** All 12 models, label pipeline, TP/SL mechanics, feature engineering, feature selection
**Symbols:** MES (S&P 500 micro), MGC (Gold micro), MNQ (Nasdaq micro)
**Method:** 6 parallel agents (label, TP/SL, model, feature, online research, verifier)
**Status:** Research only — no code changes made

---

## Executive Summary

Audited the full ML Factory pipeline for multi-symbol robustness. Found **3 CRITICAL**, **6 SIGNIFICANT**, and **3 LOW** issues. The two most impactful findings: (1) the backtester computes ATR differently than the training pipeline, causing barrier misalignment between labels and execution, and (2) pipeline-stage labels don't include transaction costs, meaning models train on trades that may be unprofitable after costs.

---

## CRITICAL Findings

### C1: ATR Computation Mismatch Between Training and Backtest

Three distinct ATR implementations exist in the codebase:

| Location | Method | Shift | File |
|----------|--------|-------|------|
| Training pipeline | Wilder's smoothing: `(atr[i-1]*(period-1) + tr[i]) / period` | shift(1) applied | `src/data/pipeline/stages/features/numba_functions.py:195` |
| Backtester | Simple rolling mean: `tr.rolling(window=14, min_periods=1).mean()` | No shift | `src/inference/backtesting/backtest.py:494` |
| TripleBarrierLabeler fallback | EMA: `alpha = 2/(period+1)` | No shift | `src/data/labeling/triple_barrier.py:551` |

**Impact:** The same `k_up * ATR` barrier produces materially different dollar widths between training and backtest. Wilder's smoothing is more responsive to recent volatility than SMA. The backtest also uses current-bar ATR (no shift), introducing mild lookahead bias.

**Fix:** Backtest `_compute_atr()` should use Wilder's smoothing with shift(1), or pass the pre-computed `atr_14` column from the feature pipeline.

### C2: Pipeline Labels Don't Apply Transaction Costs

All three pipeline labeling stages call `triple_barrier_numba()` (the cost-free version):

| Location | Function Called | File:Line |
|----------|---------------|-----------|
| Stage 4 initial labeling | `triple_barrier_numba()` | `src/data/pipeline/stages/labeling/run.py:317` |
| Stage 7 final labeling | `triple_barrier_numba()` | `src/data/pipeline/stages/final_labels/core.py:289` |
| Optuna optimizer labeling | `triple_barrier_numba()` | `src/data/pipeline/stages/ga_optimize/optuna_optimizer.py:222` |

The `triple_barrier_numba_with_costs()` function exists (`triple_barrier.py:360`) and is used by `TripleBarrierLabeler.compute_labels()` (via factory.py), but the pipeline stages never call it.

**Impact:** A label of +1 (long win) may not be profitable after real transaction costs. Models overfit to signals that lose money in live trading.

**Fix:** Wire `triple_barrier_numba_with_costs` into all three pipeline labeling sites, computing `cost_in_atr` from per-symbol transaction costs.

### C3: sqrt(252) Hardcoded in 7 Volatility Features

Seven features in the unified compute module use daily annualization for intraday data:

| Line | Function | Code |
|------|----------|------|
| 273 | `compute_hvol_10` | `* np.sqrt(252)` |
| 279 | `compute_hvol_20` | `* np.sqrt(252)` |
| 285 | `compute_hvol_60` | `* np.sqrt(252)` |
| 305 | `compute_parkinson_vol` | `* np.sqrt(252)` |
| 323 | `compute_gk_vol` | `* np.sqrt(252)` |
| 338 | `compute_rs_vol` | `* np.sqrt(252)` |
| 366 | `compute_yz_vol` | `* np.sqrt(252)` |

**File:** `src/data/features/compute/volatility.py`

For 5-min bars (78 bars/day), the correct factor is `sqrt(252 * 78) = ~140.2`, not `sqrt(252) = ~15.87`. Features are **~8.8x too small**.

The correct `get_annualization_factor(timeframe)` function already exists at `src/data/pipeline/stages/features/constants.py:101` and IS used by the pipeline-stages volatility module — but NOT by the unified compute module.

**Fix:** Import and use `get_annualization_factor()` in all 7 functions, accepting `timeframe` as a parameter.

---

## SIGNIFICANT Findings

### S1: MNQ Missing from Barrier Configuration

MNQ is absent from all pipeline barrier config dictionaries:

| Config | MES | MGC | MNQ | File:Line |
|--------|-----|-----|-----|-----------|
| `BARRIER_PARAMS` | Yes (asymmetric) | Yes (symmetric) | **Missing** (falls to default) | `barriers_config.py:110-201` |
| `TRANSACTION_COSTS` | 0.5 ticks | 0.3 ticks | **0.5 default** | `barriers_config.py:39-42` |
| `SLIPPAGE_TICKS` | 0.5/1.0 | 0.75/1.5 | **0.5/1.0 default** | `barriers_config.py:60-69` |
| `TICK_VALUES` | $1.25 | $1.00 | **$1.00 default (WRONG: should be $0.50)** | `barriers_config.py:72-77` |

MNQ IS correctly configured in the backtest layer (`costs.py:135-144` has `TransactionCosts.for_mnq()` with `tick_value=0.50`).

**Impact:** MNQ pipeline uses wrong tick value ($1.00 vs $0.50), doubling cost calculations. MNQ falls to default barriers with weaker asymmetry than MES (1.23x vs 1.50x at H5), despite Nasdaq having STRONGER upward drift (~12% annual vs ~7% for S&P).

**Fix:** Add MNQ entries to all four config dictionaries with correct tick_value=$0.50 and stronger asymmetry than MES.

### S2: ~15 Raw Volume/Order Flow Features Not Normalized

Features that output raw absolute values (scale varies 10-100x across symbols):

| Module | Raw Features | Normalized Alternatives (already exist) |
|--------|-------------|----------------------------------------|
| `volume.py` | obv, obv_sma_20, volume_sma_20, vwap, twap, twap_10, twap_20, dollar_volume, dollar_volume_sma_10/20 | volume_ratio, volume_zscore, price_to_vwap, dollar_volume_ratio |
| `order_flow.py` | buy_volume, sell_volume, cum_order_flow | pressure_ratio, order_imbalance, net_order_flow_*, cum_order_flow_normalized |
| `microstructure.py` | micro_amihud, micro_kyle_lambda, micro_volume_imbalance, micro_cum_imbalance_20 | micro_roll_spread_pct, micro_rel_spread, micro_efficiency_10, micro_vol_ratio |

**Impact:** MES volume ~1M/day vs MGC ~50K/day. Raw values dominate tree model splits and skew neural model scaling. Boosting models handle this somewhat (split on thresholds), but neural models are more affected. FoldAwareScaler normalizes per-fold which helps, but the raw range differences remain problematic.

**Fix:** Either remove raw features from the feature map (normalized versions exist) or convert to z-score/ratio versions.

### S3: Adaptive Barriers Ignore Transaction Costs

The `AdaptiveTripleBarrierLabeler` (`src/data/pipeline/stages/labeling/adaptive_barriers.py`, 472 lines) contains zero references to "cost", "transaction", or `cost_in_atr`. Barriers are computed as:

```python
upper_barrier = entry_price + adj_k_up * entry_atr  # line 345
lower_barrier = entry_price - adj_k_down * entry_atr  # line 346
```

No transaction cost adjustment is applied, even when the standard labeler supports it.

**Fix:** Add `cost_in_atr` term to barrier computation, matching the `triple_barrier_numba_with_costs` pattern.

### S4: Feature Selection tick_value Hardcoded Wrong

**File:** `src/models/training/feature_selection.py:557`

```python
tick_value=1.25 if self.config.symbol == "MES" else 0.10,
```

MNQ gets $0.10 (that's MGC's tick_value). MNQ should be $0.50. Financial report P&L is 5x too low for MNQ.

**Fix:** Use `SymbolConfig.from_symbol_or_default(self.config.symbol).tick_value` instead.

### S5: Session Features Are Generic, Not Per-Symbol

**File:** `src/data/features/compute/temporal.py:112-151`

```python
# Asia session: 00:00 - 09:00 UTC (generic)
# London session: 08:00 - 17:00 UTC (generic)
# NY session: 13:00 - 22:00 UTC (generic)
```

Meanwhile, per-contract session times exist in `execution.py:26-33`:
- MES: NYSE 9:30-16:00 ET
- MGC: COMEX 8:20-13:30 ET
- MNQ: CME 9:30-16:00 ET

The temporal features serve as "global macro context" but don't capture each contract's own primary session.

**Fix:** Add per-symbol session features using execution.py's `CONTRACT_SESSION_TIMES`, or make session hours configurable.

### S6: Annualization Hardcoded in 5 Scoring/Evaluation Sites

| File:Line | Code | Issue |
|-----------|------|-------|
| `src/optimization/scoring.py:91` | `np.sqrt(252 * 78)` | Assumes 5-min equity (78 bars/day). MGC has ~62 bars/day. |
| `src/cli/commands/evaluate.py:579` | `np.sqrt(252)` | Assumes daily returns for intraday data |
| `src/data/pipeline/stages/evaluation/run.py:193` | `np.sqrt(252)` | Same |
| `src/inference/regime_detector.py:131` | `np.sqrt(252)` | Intraday regime detection |
| `src/inference/regime_detector.py:155` | `np.sqrt(252)` | Same |

Only `five_dimension_objective.py:95-139` correctly infers the annualization factor from `DatetimeIndex` median timedelta.

**Fix:** Adopt the `_infer_annualization_factor()` pattern from `five_dimension_objective.py` everywhere, or accept a `timeframe` parameter.

---

## LOW Findings (Tuning Opportunities)

### L1: Hurst Thresholds Hardcoded (H < 0.45 / H > 0.55)

**File:** `src/data/features/compute/entropy.py:478-479`

These are standard academic thresholds for Hurst exponent regime classification. Different assets may have different baseline Hurst values (gold tends lower, NQ tends higher), but the thresholds themselves are well-established.

**Potential improvement:** Add per-symbol Hurst thresholds to `SymbolConfig`, similar to the ADX threshold pattern from Phase 93.

### L2: Uniform seq_len=60 for All Neural Models

**File:** `src/config/data.py:311` — `SequenceConfig.seq_len: int = 60`

All neural models (LSTM, GRU, TCN, InceptionTime, ResNet1D, PatchTST, iTransformer, TFT, N-BEATS) use seq_len=60 (300 minutes at 5-min bars, ~5 hours). This is reasonable for intraday trading, covering approximately one trading session.

Gold (slower mean-reversion) might benefit from seq_len=90-120. Nasdaq (fast momentum) might prefer seq_len=30-45. This is a tuning parameter, not a bug.

### L3: RSI/Stoch/MFI Binary Flags at Standard Thresholds

**File:** `src/data/features/compute/momentum.py:186-412`

- RSI overbought > 70, oversold < 30
- Stochastic overbought > 80, oversold < 20
- MFI overbought > 80, oversold < 20

Industry standard thresholds. The continuous RSI/Stoch/MFI values are also features, so binary flags are supplementary. Feature selection will naturally discard them if unhelpful for a given symbol.

---

## Verified OK (Working Correctly)

| Item | Evidence |
|------|----------|
| k_up/k_down/max_bars correctly wired training to backtest | `factory.py:899-907` reads `get_barrier_params()`, backtest applies at `backtest.py:508-535` |
| MES asymmetric barriers (k_up > k_down) correct direction | `barriers_config.py:119-156` — counteracts equity drift |
| MGC symmetric barriers (k_up = k_down) | `barriers_config.py:164-201` — correct for mean-reverting gold |
| Barriers are static (set once at entry) in both training and backtest | Training: `triple_barrier.py:169-170`. Backtest: `backtest.py:541-548`. Consistent. |
| FoldAwareScaler per-fold normalization | `fold_scaling.py:84-163` — RobustScaler per fold, handles different price scales |
| Boosting models scale-invariant | XGBoost/LightGBM/CatBoost all set `requires_scaling=False` |
| MDA/RF feature selection symbol-agnostic | `purged_selector.py` — permutation importance is scale-independent |
| Low-variance filter normalizes by coefficient of variation | `filtering.py` — dimensionless |
| Correlation filter dimensionless | Pearson r, no scale dependence |
| All ratio/z-score/bounded features (~40+) | volume_ratio, volume_zscore, RSI, Stoch, Williams %R, ROC, etc. |
| Per-symbol ADX thresholds (Phase 93) | `symbol.py:106,118,130` — MES=20.0, MGC=23.0, MNQ=25.0 |
| MNQ correct in backtest costs | `costs.py:135-144` — `TransactionCosts.for_mnq()` with tick_value=0.50 |
| Per-contract session times in execution | `execution.py:26-33` — MES=NYSE, MGC=COMEX, MNQ=CME |

---

## Missing from Codebase (from Online Research)

| Item | Source | Description |
|------|--------|-------------|
| Sample uniqueness weighting | AFML Ch.4 (Lopez de Prado) | Triple-barrier labels overlap in time, violating IID. Weight each sample by `1/concurrency` to reduce impact of overlapping labels. |
| Sequential bootstrap | AFML Ch.4 (Lopez de Prado) | Standard random sampling doesn't account for label temporal overlap. Sequential bootstrap adds samples that are sufficiently unique. |
| Label distribution logging | Best practice | No warnings when class imbalance exceeds thresholds (e.g., any class < 20%). Should log distribution after labeling. |

---

## 12-Model Architecture Summary

| Model | Family | Seq Len | Batch | Patience | Symbol Issue | Severity |
|-------|--------|---------|-------|----------|-------------|----------|
| XGBoost | Boosting (2D) | N/A | N/A | 10 | Scale-invariant, no issue | OK |
| LightGBM | Boosting (2D) | N/A | N/A | 20 | Scale-invariant, no issue | OK |
| CatBoost | Boosting (2D) | N/A | N/A | 20 | Scale-invariant, no issue | OK |
| LSTM | RNN (3D) | 60 | 512 | 7 | Uniform seq_len | LOW |
| GRU | RNN (3D) | 60 | 512 | 7 | Uniform seq_len | LOW |
| TCN | CNN (3D) | 64 | 512 | 7 | Matches receptive field 61 | OK |
| InceptionTime | CNN (3D) | 60 | 64 | 15 | Kernel sizes fixed (10,20,40) | LOW |
| ResNet1D | CNN (3D) | 60 | 64 | 15 | No symbol issue | OK |
| PatchTST | Transformer (4D) | 60 | 128 | 10 | patch_len=16 fixed, only 6 patches | LOW |
| iTransformer | Transformer (4D) | 60 | 256 | 10 | seq_len baked into architecture | LOW |
| TFT | Transformer (4D) | 60 | 128 | 10 | VSN may adapt per symbol (good) | OK |
| N-BEATS | MLP (3D) | 60 | 128 | 15 | n_harmonics=4 fixed | LOW |

All models use the same hyperparameters regardless of symbol. Scaling is handled correctly via per-fold FoldAwareScaler. The primary improvement opportunity is per-symbol sequence length tuning.

---

## Recommended Fix Order

| Priority | Fix | Effort | Files |
|----------|-----|--------|-------|
| 1 | **C1: Unify ATR computation** | ~30 min | `backtest.py` |
| 2 | **C2: Wire transaction costs into pipeline labels** | ~30 min | `run.py`, `core.py`, `optuna_optimizer.py` |
| 3 | **C3: Fix volatility annualization** | ~20 min | `compute/volatility.py` |
| 4 | **S1: Add MNQ to barriers_config** | ~20 min | `barriers_config.py` |
| 5 | **S4: Fix tick_value in feature_selection** | 5 min | `feature_selection.py` |
| 6 | **S6: Unify annualization in scoring/eval** | ~30 min | `scoring.py`, `evaluate.py`, `run.py`, `regime_detector.py` |
| 7 | **S2: Normalize raw volume features** | ~45 min | `volume.py`, `order_flow.py`, `microstructure.py` |
| 8 | **S3: Add costs to adaptive barriers** | ~20 min | `adaptive_barriers.py` |
| 9 | **S5: Per-symbol session features** | ~30 min | `temporal.py` |

---

## External References

- Lopez de Prado, *Advances in Financial Machine Learning* (2018), Ch.3 (Labeling), Ch.4 (Sample Weights)
- [Quantreo: Triple Barrier Labeling](https://www.newsletter.quantreo.com/p/the-triple-barrier-labeling-of-marco)
- [Hudson & Thames: Meta Labeling + Triple Barrier](https://hudsonthames.org/does-meta-labeling-add-to-signal-efficacy-triple-barrier-method/)
- [MQL5: Label Concurrency Deep Dive](https://www.mql5.com/en/articles/19850)
- [CFA Institute: ML in Commodity Futures](https://rpc.cfainstitute.org/research/foundation/2025/chapter-8-machine-learning-commodity-futures)
- [Amihud (2002): Illiquidity and Stock Returns](https://www.cis.upenn.edu/~mkearns/finread/amihud.pdf)
- [QuantStart: Annualised Sharpe](https://www.quantstart.com/articles/annualised-rolling-sharpe-ratio-in-qstrader/)
- [LuxAlgo: ATR Dynamic Stop-Loss](https://www.luxalgo.com/blog/average-true-range-dynamic-stop-loss-levels/)
- [MDPI: GA-Driven Triple Barrier](https://www.mdpi.com/2227-7390/12/5/780)
