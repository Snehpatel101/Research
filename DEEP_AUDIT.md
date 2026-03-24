# ML FACTORY — FULL-SYSTEM DEEP AUDIT

**Date:** 2026-03-24
**Scope:** Adversarial institutional-grade review across 22 dimensions
**Auditors:** 5 specialized agents (math, leakage, model, backtest, architecture)
**Codebase:** 467 source files, 717 classes, 4,142 functions, 196 features, 12 models

---

## A. EXECUTIVE SUMMARY

### Top 14 Findings (by Impact)

| # | Severity | Finding | Subsystem | File(s) |
|---|----------|---------|-----------|---------|
| 1 | **CRITICAL** | 50-feature truncation bias in Optuna — features at index >50 NEVER evaluated | Optuna 5D | `five_dimension_objective.py:891` |
| 2 | **CRITICAL** | Feature selection (correlation/variance filters) runs on FULL dataset before train/test split | Feature Selection | `feature_selection.py:245,259` |
| 3 | **CRITICAL** | ATR computation divergence: EMA (labels) vs SMA (backtest) vs Wilder's EMA (regime) | Label-Backtest Parity | `triple_barrier.py:551`, `backtest.py:494`, `regime.py:63` |
| 4 | **CRITICAL** | Stop/TP exit uses close price instead of barrier price — systematically biased P&L | Backtest | `backtest.py:721-727` |
| 5 | **CRITICAL** | Cumulative features (VWAP, TWAP, OBV, cum_order_flow) leak dataset position | Features | `volume.py:144-192`, `order_flow.py:268-282` |
| 6 | **CRITICAL** | ExperimentConfig to_dict/from_dict double-nests output_dir on round-trip | Config | `experiment.py:219,266,356-361` |
| 7 | **CRITICAL** | Binary mode (n_classes=2) crashes OOF generation and stacking ensemble | OOF/Ensemble | `oof_core.py:269`, `stacking.py:639` |
| 8 | **CRITICAL** | Adverse selection configured but NEVER applied — dead code | Backtest | `backtest.py:280-284`, `execution.py:134-170` |
| 9 | **CRITICAL** | Hardcoded sqrt(252) annualization wrong for intraday data (~20x under-annualization) | Features + Backtest | `volatility.py:273+`, `metrics.py:27` |
| 10 | **CRITICAL** | Yang-Zhang volatility uses hardcoded k for window=20 only | Features | `volatility.py:363` |
| 11 | **CRITICAL** | Multiple testing burden: ~940K implicit tests, DSR only corrects per-study not across 48 studies | False Alpha | `five_dimension_objective.py:1098` |
| 12 | **HIGH** | 3x cost discrepancy: labels=$1.875, backtest=$5.54 for MES | Cost Parity | `barriers_config.py`, `costs.py` |
| 13 | **HIGH** | LightGBM proxy for neural/transformer Optuna — Dim 5 hyperparams completely ignored | Optuna 5D | `five_dimension_objective.py:917-928` |
| 14 | **HIGH** | 4D OOF generation skips fold-aware scaling (2D and 3D paths apply it) | OOF | `oof_generation.py:276-279` |

### Top 5 Systemic Risks

1. **Label-Backtest Parity Violation:** The system trains on one game (EMA ATR barriers, $1.875 costs, barrier-widened labels) and evaluates on a different game (SMA ATR, $5.54 costs, close-price exits). This invalidates the entire train→evaluate pipeline.

2. **Feature Selection Leakage:** Correlation and variance filters see test data. Combined with the 50-feature truncation bias (only first 50 columns evaluated in Optuna), the feature selection pipeline is both leaking AND biased toward early-computed features.

3. **Massive Multiple Testing:** ~940K implicit tests across 196 features × 12 models × 4 horizons × 100+ trials. DSR corrects within ONE study but not across the 48 studies. Expected false discoveries at p=0.05: ~47,040.

4. **Positional Information Leakage:** Cumulative features (VWAP, TWAP, OBV, cum_order_flow) monotonically encode dataset position. Models can trivially distinguish early vs late bars, creating spurious fits in cross-validation.

5. **Binary Mode Completely Broken:** All 3 OOF generators and the stacking ensemble hardcode 3-class columns. Any n_classes=2 configuration crashes at OOF generation.

### Top 5 Strengths

1. **Triple-barrier labeling is mathematically correct** — entry at close[i], barrier scan from j=1..max_bars, symmetric cost adjustment, sentinel marking, Numba/Python parity verified.

2. **PurgedKFold CV is properly implemented** — purge before test, embargo after test, label-aware purging for overlapping labels, mask-based fold construction.

3. **MTF anti-lookahead is solid** — shift(1) on resampled index BEFORE forward-fill, gap detection with NaN-out, `label='left', closed='left'` resampling. No MTF lookahead detected.

4. **Walk-forward CV is clean** — expanding/rolling window, gap between train/test, cumulative embargo across test periods, label-aware purging.

5. **DSR formula is correct** — verified against Bailey & Lopez de Prado 2014, gamma correction for non-normality, scipy excess kurtosis handled properly.

---

## B. DEEP AUDIT BY SUBSYSTEM

### B1. Feature Engineering (Math Correctness)

#### CRITICAL

**C-FEAT-1: Hardcoded sqrt(252) annualization factor**
- **Files:** `src/data/features/compute/volatility.py:273,279,285,305,323,338,366`
- **Problem:** All 7 volatility features (`hvol_10`, `hvol_20`, `hvol_60`, `parkinson_vol`, `gk_vol`, `rs_vol`, `yz_vol`) multiply by `np.sqrt(252)`. For 1-minute bars, annualization should be `sqrt(252 * bars_per_day)`. Using `sqrt(252)` under-annualizes by ~20x for 1-min data.
- **Impact:** Absolute values wrong; relative comparisons within same timeframe OK. Cross-timeframe or scale-sensitive models distorted.
- **Fix:** Accept `bars_per_day` parameter (or derive from data frequency), compute `sqrt(252 * bars_per_day)`.

**C-FEAT-2: Yang-Zhang volatility hardcoded k constant**
- **File:** `src/data/features/compute/volatility.py:363`
- **Code:** `k = 0.34 / (1.34 + (21) / (20 - 1))`
- **Problem:** `21` and `20-1` should be `(window+1)` and `(window-1)`. Correct only for window=20.
- **Fix:** `k = 0.34 / (1.34 + (window + 1) / (window - 1))`

**C-FEAT-3: RSI Numba vs pandas path parity issue**
- **File:** `src/data/features/compute/momentum.py:74-86`
- **Problem:** First few RSI values differ between Numba (SMA-seeded, no EMA smoothing at index `period`) and pandas (`ewm` with different warmup). Not numerically identical.
- **Fix:** Align the Numba implementation to match pandas `ewm(alpha=1/period, min_periods=period, adjust=False)` warmup.

**C-FEAT-4: Cumulative features leak positional information**
- **Files:** `src/data/features/compute/volume.py:144-164,185-192` (VWAP, TWAP), `volume.py:75-93` (OBV), `order_flow.py:268-282` (cum_order_flow)
- **Problem:** `.cumsum()` and `.expanding()` from dataset start create unbounded features that encode bar position. Models can trivially distinguish early from late bars → spurious CV fits.
- **Fix:** Replace with session-reset cumulative (reset each trading day) or rolling-window variants (e.g., 20-bar rolling VWAP).

#### HIGH

**H-FEAT-1: Entropy features missing shift(1) in compute path**
- **File:** `src/data/features/compute/entropy.py:357-376,420-432`
- **Problem:** Pipeline stage versions shift(1), but `FEATURE_COMPUTE_MAP` versions don't. If `compute_all_features()` is used directly, entropy has 1-bar lookahead.
- **Fix:** Add `.shift(1)` to all `compute_entropy_*` functions in the compute map.

**H-FEAT-2: Dual Hurst exponent (raw prices vs log prices)**
- **Files:** `entropy.py:451-456` (raw prices) vs `mean_reversion.py:389-415` (log prices)
- **Problem:** Two features measuring the same thing with different inputs → different estimates.
- **Fix:** Standardize on log prices (the standard approach in finance).

**H-FEAT-3: OU half-life fragile Numba/non-Numba dispatch**
- **File:** `src/data/features/compute/mean_reversion.py:182-205`
- **Problem:** Numba path expects raw prices (logs internally); non-Numba path receives pre-logged prices. Dispatch is fragile.
- **Fix:** Make both paths explicitly handle raw prices → internal log conversion.

**H-FEAT-4: Corwin-Schultz spread formula deviation**
- **File:** `src/data/features/compute/microstructure.py:214-247`
- **Problem:** Intermediate quantities (beta, gamma) are averaged over 20-period window before computing alpha. Paper averages final spread estimates. Variable naming suggests `alpha_sq` but math computes `alpha`.
- **Fix:** Clarify naming; document the estimation approach vs paper.

**H-FEAT-5: id(df) cache can return stale data**
- **Files:** `volatility.py`, `trend.py`, `microstructure.py`, `regime.py`, `order_flow.py`, `volume.py`
- **Problem:** `id()` returns memory address. If DataFrame is GC'd and new one allocated at same address, cache returns wrong results silently.
- **Fix:** Use `(id(df), len(df), df.index[0], df.index[-1])` as cache key, or use weakref.

**H-FEAT-6: Feature duplication**
- **Files:** `volume.py:112-121` vs `microstructure.py:344-356`
- **Problem:** `volume_ratio` and `micro_trade_intensity_20` are the exact same computation: `volume / sma(volume, 20)`. Also `volume_ratio_liq` is nearly identical.
- **Fix:** Remove duplicates, keep one canonical version.

#### MEDIUM

| ID | Issue | File | Fix |
|----|-------|------|-----|
| M-FEAT-1 | Bollinger Bands ddof=1 vs canonical ddof=0 | `volatility.py:152` | Document or parameterize |
| M-FEAT-2 | GARCH stubs are permanent NaN (2 of 196 slots) | `volatility.py:391-409` | Remove or implement |
| M-FEAT-3 | Wavelet slope threshold 0.01 not scale-invariant | `wavelets.py:386` | Normalize by price |
| M-FEAT-4 | `compute_single_timeframe` mutates config | `mtf.py:530` | Copy config before mutation |
| M-FEAT-5 | Kyle's Lambda uses sign(price_diff)*volume proxy | `microstructure.py:186-206` | Document approximation limits |
| M-FEAT-6 | Autocorrelation Numba strips NaNs, changes effective lag | `price.py:38-81` | Handle NaN in-place |
| M-FEAT-7 | Shannon entropy: equal-width bins (Numba) vs quantile (Python) | `entropy.py:192` vs `entropy.py:128` | Standardize binning |
| M-FEAT-8 | Temporal session features assume UTC timezone | `temporal.py:122-151` | Accept timezone parameter |

---

### B2. Multi-Timeframe (MTF) System

**Status: SAFE**

- shift(1) correctly placed on resampled index BEFORE forward-fill (`mtf.py:491,548`)
- Gap handling correct: gaps >4h NaN-out forward-filled values (`mtf.py:299-315`)
- Resampling uses `label='left', closed='left'` (standard)
- **One issue:** `compute_single_timeframe` mutates `self.config.timeframes` (M-FEAT-4 above)

---

### B3. Labeling System

**Status: MOSTLY SAFE (2 issues)**

- Triple-barrier math is correct: entry at close[i], barrier scan from j=1..max_bars
- Symmetric cost adjustment verified
- Numba and Python implementations verified identical
- Sentinel marking for last max_bars samples correct

**Issues:**
- **MEDIUM:** Transaction cost ATR uses median of FULL dataset (`triple_barrier.py:596-602`) — leaks future volatility into early labels
- **LOW:** Both-barriers tie-break favors long (systematic long bias for volatile bars)

---

### B4. Cross-Validation System

**Status: SAFE (minor config risks)**

| Component | Status | Verification |
|-----------|--------|-------------|
| PurgedKFold | **SAFE** | Purge before test, embargo after, label-aware purging |
| Walk-Forward | **SAFE** | Past-only training, gap, cumulative embargo |
| CPCV | **SAFE** | Purge/embargo per test group, label-aware |
| Fold-Aware Scaling | **SAFE** | Fresh scaler per fold, fit on train only |
| OOF Tabular | **SAFE** | New model per fold, predictions at val_idx |
| OOF 4D | **SAFE** | Split by index, post-training leakage verification |
| Sequence Adapter | **SAFE** | Label at last timestep, backward-looking windows |

**Config risks:**
- PurgedKFold not label-aware by default (needs explicit `label_end_times`)
- CPCV uses percentage-based purge vs bar-count (inconsistent)

---

### B5. Feature Selection Pipeline

#### CRITICAL

**C-FSEL-1: Correlation and variance filters run on full dataset**
- **File:** `src/models/training/feature_selection.py:245,259`
- **Call site:** `src/models/training/unified_orchestrator.py:399`
- **Mechanism:** `_run_feature_selection_pipeline()` receives complete DataFrame including test data. MDA uses internal purged CV (good), but correlation filtering (`filter_correlated_features`) and variance filtering (`filter_low_variance`) operate on full `df` — test statistics influence which features survive.
- **Fix:** Move correlation and variance filtering INSIDE the CV loop, or compute only on training data after initial train/test split.

**C-FSEL-2: 50-feature truncation bias in Optuna validation**
- **File:** `src/optimization/five_dimension_objective.py:891-893`
- **Code:** `n_features = min(X_train_valid.shape[1], 50)` then `X_train_sub = X_train_valid[:, :n_features]`
- **Problem:** Takes FIRST 50 columns by index, not the most important 50. Features at index >50 (microstructure, entropy, regime, wavelets) NEVER evaluated during optimization.
- **Fix:** Sort by MDA importance before truncation, or use all features.

---

### B6. Optuna 5D Optimization

#### CRITICAL

- **C-FSEL-2** (above): 50-feature truncation
- **C-OPT-1: Insufficient labels returns 0.0 instead of -inf** (`five_dimension_objective.py:766`). Degenerate trials score better than crashing trials, polluting TPE sampler. Fix: return `float("-inf")`.

#### HIGH

- **H-OPT-1: LightGBM proxy for neural/transformer models** (`five_dimension_objective.py:917-928`). Dim 5 hyperparams (hidden_size, num_layers, d_model, nhead) completely ignored. Neural model hyperparameter tuning has zero signal.
- **H-OPT-2: Label cache stores labels.copy() twice** (`five_dimension_objective.py:411`). Second copy never used. 2x cache memory waste.
- **H-OPT-3: DSR gate is study-level, not experiment-level** (`five_dimension_objective.py:1098-1165`). Running 48 studies (12 models × 4 horizons) without cross-study correction inflates reported DSR.

#### MEDIUM

- Timeout inconsistency: OptunaConfig defaults 43200s (12h), OptunaSection defaults 3600s (1h)
- Reproducibility: `n_jobs=-1` + TPE seed = non-deterministic (parallel trial completion order varies)
- Feature selection can select fewer than min_features after column filtering

---

### B7. All 12 Models

**All 12 model families audited for: GPU auto-detect, float32, cleanup, DataLoader, early stopping, OOM recovery.**

| Model | Status | Notes |
|-------|--------|-------|
| XGBoost | OK | QuantileDMatrix for memory efficiency |
| LightGBM | OK | GPU auto-detect verified |
| CatBoost | OK | gpu_ram_part=0.95 cap |
| LSTM | OK | torch.from_numpy, float32 |
| GRU | OK | Same as LSTM |
| TCN | OK | CausalConv1d padding verified correct, RF=121 > seq_len=64 |
| InceptionTime | OK | — |
| ResNet1D | OK | — |
| PatchTST | OK | Causal mask, non-overlapping patches by design |
| iTransformer | OK | Better Transformer fast path |
| TFT | OK | SDPA in training, manual attention in inference |
| N-BEATS | OK | — |

**Issue:** TransformerModel (`transformer_model.py`) has correct causal mask but `is_production_safe=False` and alarming docstring saying it's "inherently non-causal" — **documentation is wrong, code is correct** (H-MODEL-1).

---

### B8. Adapters (2D/3D/4D)

| Adapter | Status | Notes |
|---------|--------|-------|
| TabularAdapter (2D) | SAFE | Standard feature matrix |
| SequenceAdapter (3D) | SAFE | Label at last timestep, backward-looking windows |
| MultiStreamAdapter (4D) | SAFE | Correct higher-TF alignment, but O(n) Python loop is slow (M-MODEL-2) |

**Issues:**
- AdapterScaler.load() doesn't restore float32 manual scaling stats → always uses float64 path (M-MODEL-3)

---

### B9. OOF/Ensemble System

#### CRITICAL

**C-OOF-1: Binary mode crashes OOF generation**
- **Files:** `oof_core.py:269-271`, `oof_sequence.py:319-321`, `oof_generation.py:353-355`
- All 3 OOF generators hardcode column names `_prob_short`, `_prob_neutral`, `_prob_long` and index `oof_probs[:, 2]`. When n_classes=2, `oof_probs[:, 2]` raises IndexError.
- **Fix:** Dynamic column generation based on n_classes.

**C-OOF-2: Stacking ensemble hardcoded to n_classes=3**
- **File:** `src/models/ensemble/stacking.py:639,913,1275`
- OOF feature matrix, base predictions, and diversity analysis all assume 3 classes.
- **Fix:** Thread n_classes through entire stacking pipeline.

#### HIGH

- **H-OOF-1:** 4D OOF generation skips fold-aware scaling (`oof_generation.py:276-279`). 2D and 3D paths apply it, 4D does not.
- **H-OOF-2:** Shared `OOFGenerator` instance across models (`oof_generation.py:70-77`). Potential state contamination.
- **H-OOF-3:** Hard vote tie-breaking biased toward lower class index (`voting.py:539`).

#### MEDIUM

- OOF calibration fits and evaluates on same data (in-sample metrics)
- `oof_stacking.py:298` also hardcodes 3 probability columns (secondary to C-OOF-1)
- Ensemble meta-learner trained on ALL OOF data without its own CV → ensemble weight overfitting

---

### B10. Backtest System

#### CRITICAL

**C-BT-1: ATR computation divergence**
- **Labeling:** EMA with `alpha = 2/(period+1)` — `triple_barrier.py:551`
- **Backtest:** SMA with `rolling(period).mean()` — `backtest.py:494`
- **Regime:** Wilder's EMA with `ewm(alpha=1/period)` — `regime.py:63`
- Same k_up/k_down multipliers × different ATR values = different barrier distances.
- **Fix:** Standardize on ONE ATR computation (Wilder's EMA is the industry standard for ATR).

**C-BT-2: Stop/TP exit uses close price instead of barrier price**
- **File:** `src/inference/backtesting/backtest.py:721-727`
- When high >= take_profit or low <= stop_loss, exit at `close_price` not barrier level.
- Stop exits OPTIMISTIC (close recovers from low), TP exits PESSIMISTIC (close falls from high).
- **Fix:** Exit at min(close, barrier_level) for stops, max(close, barrier_level) for TPs.

**C-BT-3: Adverse selection is dead code**
- **Files:** `backtest.py:280-284` creates filter; `execution.py:134-170` has method; backtest loop NEVER calls it.
- **Fix:** Wire `apply_adverse_selection()` into `_get_execution_price_fast()`.

#### HIGH

| ID | Issue | File | Impact |
|----|-------|------|--------|
| H-BT-1 | Sharpe annualized with 252 (not intraday-adjusted) | `metrics.py:27`, `equity_curve.py:216` | Sharpe inflated 5-10x for 5-min data |
| H-BT-2 | Labels: $1.875 costs; Backtest: $5.54 costs (3x gap) | `barriers_config.py`, `costs.py` | Systematic negative performance gap |
| H-BT-3 | Same-bar entry+exit possible (churn) | `backtest.py:620-652` | Phantom trades |
| H-BT-4 | MNQ tick_value wrong in execution.py ($0.25 vs $0.50) | `execution.py:201` | Wrong adverse selection for MNQ |

#### MEDIUM

- VWAP execution approximated as `(high + low) / 2` (M-BT-1)
- Daily loss circuit breaker delayed trigger (checks at day boundaries only) (M-BT-2)
- VolatilityScaledSlippage never receives actual volatility, falls back to 15% constant (M-BT-3)

---

### B11. Transaction Cost System

**Three different cost models exist:**

| System | MES Round-Trip Cost | How Applied |
|--------|-------------------|-------------|
| Labels (`barriers_config.py`) | $1.875 (0.5 + 2×0.5 ticks) | Barrier widening |
| Backtest defaults (`costs.py`) | $5.54 ($2.50 + $0.52 + $0.02 + 2×$1.25) | P&L deduction |
| Vol-scaled slippage | Variable (higher) | P&L deduction |

**Impact:** Labels say "trade is profitable after $1.875 costs" but backtest deducts $5.54. The model trains on an easier game than it's evaluated on. This creates systematic pessimism in backtest results relative to training expectations.

**Fix:** Unify cost assumptions. Either (a) increase label costs to match backtest, or (b) reduce backtest costs to match labels. The backtest defaults are more realistic.

---

### B12. Regime System

**Status: CLEAN (no lookahead)**

All regime computations use `.rolling()`, `.pct_change()`, `.shift()`, `.ewm()` — backward-looking only.

**Minor issues:**
- Composite regime uses ordinal encoding (0-5) — misleading for neural models
- Structure regime has persistence bias (rolling median creates regime stickiness)
- ADX thresholds (MES=20, MGC=23, MNQ=25) not empirically validated

---

### B13. DSR Gate

**Status: CORRECT (formula verified)**

- Bailey & Lopez de Prado 2014 formula correctly implemented
- Gamma correction for non-normality correct
- scipy excess kurtosis handled properly
- `is_sharpe_like_metric()` guard properly wired

**Issue:** DSR is study-level, not experiment-level. Selecting the best across 48 studies (12 models × 4 horizons) without Bonferroni/FDR correction inflates reported significance. (H-OPT-3)

---

### B14. Configuration System

#### CRITICAL

**C-CFG-1: ExperimentConfig round-trip double-nests output_dir**
- `__post_init__()` appends `run_id` to `output_dir`
- `to_dict()` serializes the already-modified path
- `from_dict()` → `__post_init__()` appends `run_id` AGAIN
- Result: `experiments/runs/20260323_120000/20260323_120000`
- **Fix:** Don't modify output_dir in `__post_init__`, or strip run_id in `to_dict()`.

#### HIGH

- Three duplicate ExperimentConfig classes (only one has importers)
- Dual config system: ExperimentConfig vs UnifiedConfig vs GlobalConfig — three sources of truth
- No ExperimentConfig → PipelineConfig parity verification
- Timeout inconsistency: 43200s vs 3600s depending on config path

---

### B15. Training/Inference Parity

| Aspect | Status | Notes |
|--------|--------|-------|
| Feature column order | **SAFE** | Bundle saves and restores order |
| Scaling | **SAFE** | Bundle saves scaler, inference applies |
| Feature drift detection | **MISSING** | No hash of feature engineering code in bundle |
| MTF resampling params | **MISSING** | `closed/label` not persisted in bundle |

---

## C. MATH VERIFICATION FINDINGS

### Verified Correct
- Triple-barrier math (entry, barrier scan, cost adjustment, sentinel)
- PurgedKFold purge/embargo logic
- Walk-forward expanding/rolling window with gap
- DSR formula (Bailey & Lopez de Prado 2014)
- MTF shift(1) placement and gap handling
- TCN CausalConv1d padding removal
- Sequence adapter label alignment (last timestep)

### Verified Incorrect
- Yang-Zhang k constant (hardcoded for window=20 only)
- RSI Numba vs pandas path (numerically different)
- Shannon entropy binning (equal-width Numba vs quantile Python)
- OU half-life dispatch (raw prices vs log prices depending on Numba)
- Hurst exponent (raw prices in entropy.py vs log prices in mean_reversion.py)

### Verified Approximation
- Corwin-Schultz spread (intermediate averaging, not per-pair as in paper)
- Kyle's Lambda (sign(price_diff)*volume proxy)
- Bollinger Bands (ddof=1 vs canonical ddof=0)
- Sortino (partial lower moment — conservative variant)

---

## D. LEAKAGE AND ROBUSTNESS FINDINGS

### Confirmed Leakage Paths

| Path | Severity | Mechanism | File |
|------|----------|-----------|------|
| Feature selection on full dataset | **CRITICAL** | Correlation/variance filters see test data | `feature_selection.py:245,259` |
| 50-feature truncation by index | **CRITICAL** | Late feature families never evaluated | `five_dimension_objective.py:891` |
| Transaction cost ATR global median | **MEDIUM** | Future volatility leaks into early labels | `triple_barrier.py:600` |
| Cumulative features (VWAP/OBV/etc) | **CRITICAL** | Positional information encoded | `volume.py`, `order_flow.py` |

### Verified Safe

| Component | Verification |
|-----------|-------------|
| Triple-barrier labeling | Entry at close[i], forward scan j=1..max_bars, no future data |
| PurgedKFold | Purge before test, embargo after, label-aware option |
| Walk-Forward CV | Past-only training, gap, cumulative embargo |
| CPCV | Per-group purge/embargo, label-aware |
| Fold-Aware Scaling | Fresh scaler per fold, fit on train only |
| OOF Tabular | New model per fold, val_idx predictions |
| OOF 4D | Index-based split, post-training leakage verification |
| Sequence windows | Last-timestep labels, backward-looking only |
| MTF system | shift(1) before forward-fill, gap NaN-out |
| Regime features | All backward-looking (.rolling, .ewm, .shift) |

---

## E. ENSEMBLE AND EVALUATION FINDINGS

### Ensemble Issues
1. **Binary mode crashes** — OOF + stacking hardcode 3 classes (CRITICAL)
2. **Shared OOF generator state** — potential cross-model contamination (HIGH)
3. **Meta-learner overfitting** — trained on all OOF data without its own CV (MEDIUM)
4. **Hard vote bias** — argmax favors lower class index on ties (MEDIUM)

### Evaluation Issues
1. **Sharpe annualization wrong for intraday** — 252 instead of 252×bars_per_day (HIGH)
2. **Stop/TP exit at close not barrier** — biased P&L (CRITICAL)
3. **3x cost gap** between training and evaluation (HIGH)
4. **Adverse selection dead code** — fills unrealistically clean (CRITICAL)

---

## F. MISSING TESTS AND MISSING INVARIANTS

### Missing Tests (12 specific tests that should exist)

| # | Test | Would Catch |
|---|------|-------------|
| 1 | `ExperimentConfig.from_dict(config.to_dict()) == config` | C-CFG-1 (double-nested output_dir) |
| 2 | `ExperimentConfig.from_yaml(path)` after `config.save_yaml(path)` | Same |
| 3 | Feature order parity: bundle columns == training columns | Feature drift |
| 4 | Optuna: features at index >50 contribute to objective signal | C-FSEL-2 |
| 5 | Degenerate labels (all same class) score -inf, not 0.0 | C-OPT-1 |
| 6 | `config.to_pipeline_config()` parity verification | H-CFG-4 |
| 7 | Inference parity: same data → training vs bundle.predict → same result | P3 |
| 8 | Feature engineering determinism: same input → same features | Nondeterminism |
| 9 | Scaling parity: FoldAwareScaler vs bundle scaler → identical | Scaling drift |
| 10 | Config hash stability: same config → same hash across runs | Checkpoint |
| 11 | Property: barrier labels ∈ {-1, 0, 1, -99}, never NaN | Label corruption |
| 12 | Property: adapter output shapes match contract specs | Shape mismatch |

### Missing Invariants

1. **ATR consistency:** No invariant that labeling ATR == backtest ATR == regime ATR
2. **Cost consistency:** No invariant that label costs == backtest costs
3. **Feature purity:** No invariant that features are position-independent
4. **Config single-source:** No invariant that only one config class provides defaults
5. **Scale independence:** No invariant that features work across price levels

### Test Coverage

- **168 test functions** covering **467 source files** (717 classes, 4,142 functions)
- **~4% test density** — most tests verify non-crashing (import, smoke) not mathematical correctness
- **Zero property-based tests** (QuickCheck/Hypothesis style)
- **Zero golden/snapshot tests** for feature engineering determinism

---

## G. RANKED ACTION PLAN

### Tier 1: IMMEDIATE MUST-FIX (blocks production validity)

| # | Action | Files | Effort | Impact |
|---|--------|-------|--------|--------|
| 1 | **Unify ATR computation** — use ONE method (Wilder's EMA) in labeling, backtest, and regime | `triple_barrier.py`, `backtest.py`, `regime.py` | 2h | Fixes label-backtest parity violation |
| 2 | **Fix stop/TP exit price** — exit at barrier level, not close | `backtest.py:721-727` | 1h | Eliminates systematic P&L bias |
| 3 | **Fix 50-feature truncation** — sort by importance before truncation, or remove cap | `five_dimension_objective.py:891-893` | 30m | Unlocks late feature families for optimization |
| 4 | **Move correlation/variance filters inside CV** — compute only on training data | `feature_selection.py:245,259` | 2h | Eliminates feature selection leakage |
| 5 | **Unify cost assumptions** — label costs should match backtest costs | `barriers_config.py`, `costs.py` | 1h | Eliminates 3x cost gap |
| 6 | **Fix cumulative features** — session-reset or rolling variants | `volume.py`, `order_flow.py` | 2h | Eliminates positional leakage |
| 7 | **Fix ExperimentConfig round-trip** — don't double-nest output_dir | `experiment.py:219,266,356-361` | 30m | Fixes checkpoint resume |
| 8 | **Return -inf for degenerate Optuna trials** — not 0.0 | `five_dimension_objective.py:766` | 15m | Stops TPE pollution |

### Tier 2: HIGH-VALUE SHORT-TERM (significant quality improvement)

| # | Action | Files | Effort | Impact |
|---|--------|-------|--------|--------|
| 9 | **Fix binary mode** — dynamic n_classes in OOF + stacking | `oof_core.py`, `oof_sequence.py`, `oof_generation.py`, `stacking.py` | 4h | Unblocks binary classification |
| 10 | **Wire adverse selection** — call it in backtest loop | `backtest.py` | 1h | More realistic fills |
| 11 | **Fix Sharpe annualization** — derive periods_per_year from data frequency | `metrics.py`, `equity_curve.py` | 1h | Correct Sharpe/Sortino/Calmar |
| 12 | **Add fold-aware scaling to 4D OOF** | `oof_generation.py:276` | 1h | Consistent OOF quality |
| 13 | **Fix Yang-Zhang k** — parameterize by window | `volatility.py:363` | 15m | Correct YZ volatility |
| 14 | **Align RSI Numba/pandas** — match warmup behavior | `momentum.py:74-86` | 1h | Deterministic RSI |
| 15 | **Fix TransformerModel metadata** — `is_production_safe=True` | `transformer_model.py` | 15m | Correct model selection |
| 16 | **Fix MNQ tick value** — $0.50 not $0.25 | `execution.py:201` | 5m | Correct MNQ costs |

### Tier 3: MEDIUM-TERM (robustness hardening)

| # | Action | Files | Effort | Impact |
|---|--------|-------|--------|--------|
| 17 | **Experiment-level DSR** — Bonferroni/FDR across all studies | `five_dimension_objective.py` | 4h | Reduces false alpha |
| 18 | **Replace id(df) caching** — use composite key or weakref | 6 files | 2h | Prevents stale cache |
| 19 | **Remove GARCH NaN stubs** | `volatility.py:391-409` | 15m | Cleaner feature set |
| 20 | **Remove duplicate features** — volume_ratio/trade_intensity | `volume.py`, `microstructure.py` | 30m | Less redundancy |
| 21 | **Add entropy shift(1) to compute path** | `entropy.py` | 30m | Prevents potential lookahead |
| 22 | **Standardize Hurst exponent** — one implementation on log prices | `entropy.py`, `mean_reversion.py` | 1h | Consistent features |
| 23 | **Unify config system** — single source of truth | `experiment.py`, `unified.py`, `training.py` | 8h | Eliminates config conflicts |
| 24 | **Delete dead ExperimentConfig classes** | `training.py:401`, `models/training/config.py:21` | 15m | Less confusion |
| 25 | **Add property-based tests** — barriers, adapters, feature purity | New test files | 8h | Mathematical correctness |
| 26 | **Add config round-trip test** | New test file | 1h | Catches serialization bugs |
| 27 | **Fix OU half-life dispatch** — consistent raw→log handling | `mean_reversion.py:182-205` | 1h | Correct half-life |
| 28 | **Parameterize annualization** — accept bars_per_day | `volatility.py` (7 sites) | 1h | Correct absolute values |

### Tier 4: NICE-TO-HAVE (polish)

| # | Action | Effort | Impact |
|---|--------|--------|--------|
| 29 | Document Corwin-Schultz approximation vs paper | 30m | Transparency |
| 30 | Parameterize Bollinger ddof | 15m | Canonical compliance |
| 31 | Make wavelet slope threshold scale-invariant | 30m | Cross-symbol robustness |
| 32 | Fix temporal session timezone assumption | 30m | Correct sessions |
| 33 | Fix autocorrelation NaN handling in Numba | 1h | Edge case correctness |
| 34 | Add feature engineering determinism test | 2h | Reproducibility |
| 35 | Optimize MultiStreamAdapter O(n) loop | 4h | Performance |
| 36 | Fix AdapterScaler float32 restoration on load | 1h | Memory efficiency |

### Tier 5: SKIP (not worth fixing)

| Item | Reason |
|------|--------|
| Sortino partial lower moment vs zero-fill | Both valid, current is conservative |
| Supertrend initial direction assumption | Standard practice |
| MFI unchanged-bar handling | Correct per standard definition |
| TCN RF > seq_len | By design, causal padding handles it |
| PatchTST non-overlapping patches | Per original paper |

---

## H. DEAD CODE TO DELETE

| File | What | Lines |
|------|------|-------|
| `src/config/training.py` | Dead `ExperimentConfig(BaseConfig)` class | 401-448 |
| `src/models/training/config.py` | Dead `ExperimentConfig` + `ModelConfig` | 21+ |
| `src/data/features/compute/volatility.py` | GARCH stubs (permanent NaN) | 391-409 |
| `src/inference/backtesting/execution.py` | `apply_adverse_selection()` (if not wired) | 134-170 |

---

## I. NUMERICAL INCONSISTENCIES MATRIX

| Quantity | Location 1 | Location 2 | Location 3 | Status |
|----------|-----------|-----------|-----------|--------|
| ATR smoothing | EMA (labeling) | SMA (backtest) | Wilder's EMA (regime) | **BROKEN** |
| Annualization | sqrt(252) features | 252 metrics | 252×78 Optuna fallback | **BROKEN** |
| Transaction costs | $1.875 labels | $5.54 backtest | Vol-scaled (variable) | **BROKEN** |
| Hurst input | Raw prices (entropy) | Log prices (mean_reversion) | — | **BROKEN** |
| Shannon binning | Equal-width (Numba) | Quantile (Python) | — | **BROKEN** |
| OU half-life input | Raw (Numba) | Pre-logged (Python) | — | **BROKEN** |
| RSI warmup | SMA-seeded (Numba) | EWM-seeded (pandas) | — | **BROKEN** |
| Tick value MNQ | $0.50 (costs.py) | $0.25 (execution.py) | — | **BROKEN** |
| YZ k constant | Hardcoded window=20 | — | — | **FRAGILE** |
| Bollinger ddof | ddof=1 (pandas default) | — | — | **MINOR** |

---

## J. AGGREGATE STATISTICS

| Severity | Count | Subsystems |
|----------|-------|------------|
| CRITICAL | 14 | Features (5), Feature Selection (2), Backtest (3), OOF (2), Config (1), Optuna (1) |
| HIGH | 14 | Features (6), Backtest (4), OOF (2), Optuna (2) |
| MEDIUM | 23 | Features (8), Backtest (5), OOF (3), Config (4), Parity (3) |
| LOW | 13 | Various |
| VERIFIED SAFE | 12 | CV system, MTF, DSR, triple-barrier, adapters |

**Total findings: 64+ (14 critical, 14 high, 23 medium, 13 low)**
**Verified safe components: 12**

---

*This audit was conducted adversarially — actively looking for ways the system could produce false alpha, leak information, or misrepresent performance. The findings above represent real risks, not theoretical concerns. The most urgent action items are Tier 1 (items 1-8), which collectively address the label-backtest parity violation, feature selection leakage, and positional information leakage that together could explain the majority of any observed alpha.*
