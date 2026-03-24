# ML Factory Feature Governance Framework

**Author:** Sneh (with AI architectural design)
**Date:** 2026-03-23
**Status:** Design Document — Ready for Implementation Planning
**Scope:** Regime-aware, leakage-safe, ticker-aware feature selection and governance system

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Core Philosophy](#2-core-philosophy)
3. [Proposed Full Architecture](#3-proposed-full-architecture)
4. [Robustness Scoring Framework](#4-robustness-scoring-framework)
5. [Regime-Aware Feature Selection](#5-regime-aware-feature-selection)
6. [Timeframe-Aware Feature Selection](#6-timeframe-aware-feature-selection)
7. [Ticker-Aware Feature Selection](#7-ticker-aware-feature-selection)
8. [Leakage-Safe Pipeline Design](#8-leakage-safe-pipeline-design)
9. [Feature Redundancy and Clustering](#9-feature-redundancy-and-clustering)
10. [Economic Value vs Pure Statistical Importance](#10-economic-value-vs-pure-statistical-importance)
11. [Feature Promotion/Demotion Lifecycle](#11-feature-promotiondemotion-lifecycle)
12. [Model-Family Interaction](#12-model-family-interaction)
13. [Implementation Blueprint for ML Factory](#13-implementation-blueprint-for-ml-factory)
14. [Pseudocode / Algorithmic Flow](#14-pseudocode--algorithmic-flow)
15. [Blunt Critique of Current Risks](#15-blunt-critique-of-current-risks)
16. [Ranked Recommendations](#16-ranked-recommendations)

---

## 1. Executive Summary

ML Factory's current feature selection has a **critical architectural flaw**: MDA ranking runs on the full dataset before train/test split, which is a form of data leakage. Beyond that, the system has ~6 orphaned feature selection modules that were never wired into the production pipeline, no regime-conditional feature selection, no ticker-aware selection, no timeframe competition, no feature lifecycle management, and no economic value assessment of individual features.

This document designs a complete replacement: a **7-layer feature governance framework** that treats feature selection as a first-class production system with:

- **Leakage-safe selection** inside each CV fold (never on test data)
- **Robustness scoring** across 8 dimensions with concrete formulas
- **Regime-conditional feature sets** (core + regime-local) with overfitting guards
- **Timeframe competition** to eliminate redundant MTF variants
- **Ticker-aware selection** with portable vs local feature classification
- **Economic value scoring** based on cost-adjusted Sharpe contribution
- **Feature lifecycle governance** with promotion/demotion rules and evidence thresholds

The framework is designed to be **implementation-ready** for ML Factory's existing architecture, plugging into the existing `UnifiedTrainingOrchestrator`, adapter system, and Optuna pipeline.

### What's Broken Today (Top 5 Critical Issues)

| # | Issue | Severity | Location |
|---|-------|----------|----------|
| 1 | MDA ranking sees full dataset including test data | **CRITICAL** | `feature_selection.py:_run_feature_selection_pipeline()` |
| 2 | 6 feature selection modules are orphaned/dead code | HIGH | `walk_forward.py`, `purged_selector.py`, `ohlcv_selector.py`, `manager.py`, `optimization.py`, `strategies.py` |
| 3 | Feature stability scores computed but discarded | HIGH | `WalkForwardFeatureSelector` stability never used |
| 4 | No regime-conditional feature selection | HIGH | Regime features exist but don't influence selection |
| 5 | `fillna(0)` in correlation creates spurious groupings | MEDIUM | `filtering.py:build_correlation_groups()` |

---

## 2. Core Philosophy

### 2.1 Why Financial ML Feature Selection Differs from Standard ML

In standard ML (image classification, NLP, tabular prediction), features are relatively stationary. A pixel pattern that predicts "cat" today will still predict "cat" next year. Feature importance from a single train/test split is usually reliable.

**Financial markets are fundamentally different:**

1. **Non-stationarity.** The joint distribution P(features, labels) changes over time. A momentum signal that worked in 2020's trending market is noise in 2022's choppy market. Feature selection must be conditional on regime, not averaged across regimes.

2. **Adversarial dynamics.** Unlike natural phenomena, markets adapt. When enough participants trade on the same signal, the signal decays. Feature selection must distinguish between durable structural features (e.g., volatility clustering) and transient alpha signals (e.g., a specific RSI threshold).

3. **Low signal-to-noise ratio.** Financial returns have SNR of roughly 0.01-0.1, compared to 1-100 in typical ML tasks. This means feature importance estimates have enormous variance. A feature that appears "top-5" in one backtest may be "bottom-50" in another. You need stability, not point estimates.

4. **Multiple testing bias.** With 196+ features, 4 timeframes, 12 models, and 100+ Optuna trials, you are running thousands of implicit hypothesis tests. The probability of finding features that appear significant by chance alone is near 1.0. Without correction (DSR, FWER, Bonferroni), your "best features" are likely false discoveries.

5. **Costs destroy edge.** A feature that improves classification accuracy by 0.5% but increases trade frequency by 200% will lose money after costs. Feature value must be measured in cost-adjusted risk-adjusted returns, not accuracy.

### 2.2 Why "Top Feature Importance" Is Not Enough

Standard MDA (permutation importance) answers: *"If I randomly shuffle feature X, how much does model accuracy drop?"*

This is insufficient for trading because:

| Problem | Why It Matters |
|---------|---------------|
| **Substitution effect** | Feature A and Feature B are correlated. MDA assigns all importance to whichever the RF tree splits on first. The other gets near-zero. Drop one, the other becomes important. |
| **Instability** | MDA with n_estimators=20 and n_repeats=3 has high variance. Different random seeds produce different rankings. ML Factory currently uses 20 estimators — far too few for stable ranking. |
| **Regime blindness** | MDA averages importance across all market conditions. A feature critical in trending markets but useless in ranging markets gets mediocre average importance. |
| **No economic filter** | MDA treats all correct predictions equally. A feature that helps predict 0.01% moves (untradable after costs) gets the same credit as one that predicts 2% moves. |
| **Point-in-time bias** | MDA on one dataset is a single sample from the distribution of possible importances. Without rolling-window stability analysis, you're overfitting to one realization. |

### 2.3 Three Classes of Features

Not all predictive features are tradable. The framework must distinguish:

**Predictive features** — statistically associated with future returns in-sample.
- Example: RSI < 30 predicts +0.1% next-bar return (p < 0.05)
- Problem: May not survive costs, may not be stable, may be data-mined

**Robust features** — predictive across multiple time periods, regimes, and data perturbations.
- Example: ATR(14) is predictive in 80% of rolling windows, across trending and ranging regimes, with robustness score > 0.7
- Property: Stability > point-estimate importance

**Tradable features** — robust features whose predictions generate positive cost-adjusted risk-adjusted returns.
- Example: ATR(14)-based signal generates Sharpe > 1.0 after costs, with DSR > 0.5
- This is what we actually want. Everything else is noise.

**The hierarchy:** Tradable C Robust C Predictive. Most predictive features are not robust. Most robust features are not tradable. The feature governance system must promote features through this hierarchy with evidence.

### 2.4 How Regime Change, Concept Drift, and Non-Stationarity Affect Feature Selection

| Phenomenon | Impact on Features | Defense |
|------------|-------------------|---------|
| **Regime change** | A feature's predictive power can flip sign across regimes. Momentum features predict well in trends, revert in ranges. | Regime-conditional feature sets with separate importance per regime. |
| **Concept drift** | The relationship between features and labels slowly changes. A feature that worked in 2023 may decay by 2025. | Rolling-window stability analysis. Features must maintain importance across recent windows, not just historically. |
| **Non-stationarity** | Feature distributions shift (mean, variance, higher moments). Scaling assumptions break. | Fold-aware scaling. Robustness to parameter perturbation. Prefer ratio/rank features over level features. |
| **Alpha decay** | Profitable signals attract capital and decay over time. | Feature freshness tracking. Demotion of features whose economic value is declining. |
| **Structural breaks** | Market microstructure changes (e.g., tick size changes, new regulations). | Breakpoint detection in feature importance time series. Auto-demotion when importance drops discontinuously. |

---

## 3. Proposed Full Architecture

### 3.1 The 7-Layer Feature Governance Pipeline

```
Layer 0: UNIVERSE DEFINITION
   Input: 196 base features + MTF variants (~286 total)
   Output: Feature universe with metadata

Layer 1: PRE-FILTERING (stateless, fast)
   - Low-variance filter (CV^2 < 0.01)
   - NaN density filter (>30% NaN = drop)
   - Constant-in-regime filter (zero variance within any regime)
   Output: ~150-200 surviving features

Layer 2: REDUNDANCY REDUCTION (correlation + clustering)
   - Hierarchical clustering on |correlation| distance
   - ONC-style cluster count selection (eigen-gap)
   - Within-cluster champion selection by MDA
   - Cross-family redundancy check
   Output: ~60-100 non-redundant features

Layer 3: STABILITY FILTERING (rolling window)
   - Walk-forward importance across 8+ time windows
   - Min selection frequency >= 0.6
   - Importance variance penalty
   - Breakpoint detection for structural drops
   Output: ~40-70 stable features

Layer 4: REGIME-CONDITIONAL SELECTION
   - Core features: stable across ALL regimes (>= 0.5 importance in each)
   - Regime-local features: high importance in specific regime(s)
   - Regime interaction features: predictive of regime transitions
   Output: Core set (~25-40) + Regime-local sets (~5-15 per regime)

Layer 5: TIMEFRAME + TICKER COMPETITION
   - Timeframe tournament: same feature across TFs, keep best 1-2
   - Ticker portability test: features that work on 2+ symbols = portable
   - Ticker-local features: high importance on single symbol only
   Output: Final feature sets per (model_family, ticker, regime)

Layer 6: ECONOMIC VALUE GATE
   - Cost-adjusted Sharpe contribution per feature
   - Marginal improvement in tradability (trade rate, win rate, expectancy)
   - DSR-corrected significance
   Output: Economically valuable features only

Layer 7: GOVERNANCE & LIFECYCLE
   - Promotion/demotion state machine
   - Feature registry with audit trail
   - Periodic re-evaluation triggers
   Output: Production-approved feature manifest
```

### 3.2 Data Flow Diagram

```
Raw OHLCV
    |
    v
Feature Engineering (196 base + 90 MTF = ~286)
    |
    v
+-------------------------------------------+
| FEATURE GOVERNANCE PIPELINE               |
|                                           |
| For each CV fold (INSIDE purged k-fold):  |
|   L1: Pre-filter (variance, NaN)          |
|   L2: Cluster & deduplicate              |
|   L3: Walk-forward stability (sub-folds)  |
|   L4: Regime-conditional scoring          |
|   L5: Timeframe/ticker competition        |
|   L6: Economic value gate                 |
|                                           |
| Aggregate across folds:                   |
|   - Selection frequency per feature       |
|   - Robustness score per feature          |
|   - Regime-conditional importance maps    |
|                                           |
| L7: Governance lifecycle update           |
+-------------------------------------------+
    |
    v
Per-Model Feature Subsets
    |
    v
Training (boosting/neural/transformer)
```

### 3.3 Key Design Decision: Selection INSIDE CV Folds

**Current ML Factory:** Feature selection runs ONCE on the full dataset, BEFORE any train/test split. This is leakage.

**New design:** Feature selection runs INSIDE each CV fold, using only training data. The final feature set is the intersection (or frequency-weighted union) of features selected across folds.

This is more expensive but correct. The cost is mitigated by:
- Subsampling large datasets (50K cap)
- Caching per-fold results
- Lightweight RF (n_estimators=50, max_depth=5) for MDA
- Parallelizing across folds

---

## 4. Robustness Scoring Framework

### 4.1 The 8 Robustness Dimensions

Every feature gets scored on 8 dimensions. Each dimension produces a score in [0, 1]. The final robustness score is a weighted combination.

| Dimension | Symbol | Weight | What It Measures |
|-----------|--------|--------|-----------------|
| Fold Stability | S_fold | 0.20 | Selection frequency across CV folds |
| Window Stability | S_window | 0.20 | Selection frequency across rolling time windows |
| Regime Breadth | S_regime | 0.15 | Importance across different market regimes |
| Cost Sensitivity | S_cost | 0.10 | Stability when transaction costs are perturbed |
| Label Sensitivity | S_label | 0.10 | Stability when barrier params are perturbed |
| Data Perturbation | S_data | 0.10 | Stability under bootstrap/noise injection |
| Parameter Sensitivity | S_param | 0.05 | Stability when feature params change (e.g., RSI period) |
| Economic Contribution | S_econ | 0.10 | Cost-adjusted Sharpe contribution |

### 4.2 Scoring Formulas

**Fold Stability (S_fold):**
```
S_fold = (# folds where feature is in top-N) / (total # folds)
```
Where top-N = model contract's max_features. A feature selected in 4/5 folds gets S_fold = 0.8.

**Window Stability (S_window):**
```
windows = [W1, W2, ..., W_k] (rolling 6-month windows, 3-month stride)
importance_i = MDA importance in window i
S_window = 1 - CV(importance)  where CV = std(importance) / mean(importance)
```
Clipped to [0, 1]. Low coefficient of variation = high stability.

**Regime Breadth (S_regime):**
```
regimes = [trending, ranging, high_vol, low_vol, transitioning]
importance_r = MDA importance in regime r (using only data from that regime)
S_regime = (# regimes where importance_r > threshold) / (total # regimes)
```
Threshold = median importance of all features in that regime. A feature important in 4/5 regimes gets S_regime = 0.8.

**Cost Sensitivity (S_cost):**
```
costs = [0.5x, 0.75x, 1.0x, 1.5x, 2.0x] * base_transaction_cost
sharpe_c = model Sharpe using this feature under cost level c
S_cost = 1 - CV(sharpe across cost levels)
```
Features whose value collapses when costs increase slightly are fragile.

**Label Sensitivity (S_label):**
```
perturbations = [(k_up * 0.9, k_down * 0.9), (k_up * 1.1, k_down * 1.1), ...]
importance_p = MDA importance under label perturbation p
S_label = 1 - CV(importance across perturbations)
```
Features that are only important under one specific barrier setting are overfit to that setting.

**Data Perturbation (S_data):**
```
bootstraps = [B1, B2, ..., B_20] (20 bootstrap samples with replacement)
importance_b = MDA importance on bootstrap b
S_data = 1 - CV(importance across bootstraps)
```
Features whose importance varies wildly across bootstrap samples are unreliable.

**Parameter Sensitivity (S_param):**
```
For feature RSI(14):
  variants = [RSI(10), RSI(12), RSI(14), RSI(16), RSI(18)]
  importance_v = MDA importance for each variant
  S_param = 1 - CV(importance across variants)
```
If RSI(14) is important but RSI(12) and RSI(16) are not, the importance is likely spurious.

**Economic Contribution (S_econ):**
```
sharpe_with = Sharpe ratio of model WITH this feature (cost-adjusted)
sharpe_without = Sharpe ratio of model WITHOUT this feature (cost-adjusted)
marginal_sharpe = sharpe_with - sharpe_without
S_econ = sigmoid(marginal_sharpe / sigma_sharpe)  # normalized to [0,1]
```
Where sigma_sharpe = std of marginal Sharpe across CV folds. Features must improve cost-adjusted Sharpe to score well.

### 4.3 Final Robustness Score

```
R(feature) = 0.20 * S_fold
           + 0.20 * S_window
           + 0.15 * S_regime
           + 0.10 * S_cost
           + 0.10 * S_label
           + 0.10 * S_data
           + 0.05 * S_param
           + 0.10 * S_econ
```

**Thresholds:**
- R >= 0.70: **Core feature** (always included)
- 0.50 <= R < 0.70: **Approved feature** (included if model needs more features)
- 0.30 <= R < 0.50: **Probationary feature** (under evaluation, not in production)
- R < 0.30: **Rejected** (insufficient evidence of robustness)

### 4.4 Practical Simplification

Computing all 8 dimensions for every feature is expensive. In practice:

**Fast path (always computed):** S_fold, S_window, S_regime (these use data already available from walk-forward CV)

**Slow path (computed periodically or on-demand):** S_cost, S_label, S_data, S_param, S_econ (these require re-running models with perturbations)

The fast path alone is sufficient for day-to-day feature selection. The slow path should run monthly or when the feature universe changes.

---

## 5. Regime-Aware Feature Selection

### 5.1 Regime Definition

**Use a 3-regime model, not 6.** The current 6-state composite regime (vol * 3 + trend) is too granular. With limited data per symbol, 6 regimes means ~16% of data per regime on average. After purge/embargo and 3-fold CV, each regime-fold may have only ~5% of data — too little for reliable feature selection.

**Recommended 3-regime model:**

| Regime | Definition | Detection |
|--------|-----------|-----------|
| **Trending** | ADX > threshold AND abs(SMA20 - SMA50)/SMA50 > 0.005 | Rule-based (existing ADX + trend features) |
| **Mean-Reverting** | ADX < threshold AND Hurst < 0.45 AND vol_regime = low | Rule-based (existing features) |
| **Volatile/Transitioning** | Neither trending nor mean-reverting | Catch-all |

**Why rule-based, not learned:** Learned regimes (HMM, Bayesian switching) add a model-within-a-model that can overfit. Rule-based regimes using well-understood indicators (ADX, Hurst, volatility) are interpretable, stable, and don't introduce another layer of optimization. The regime labels should be **lagged by 1 bar** (use regime computed on bars 0..t-1 to classify bar t) to prevent lookahead.

**Per-symbol ADX thresholds** (already implemented): MES=20, MGC=23, MNQ=25.

### 5.2 Regime-Conditional Feature Scoring

For each CV fold, feature importance is computed SEPARATELY per regime:

```python
for fold in cv_folds:
    X_train, y_train = fold.train_data
    for regime in [TRENDING, MEAN_REVERTING, VOLATILE]:
        mask = regime_labels[X_train.index] == regime
        if mask.sum() < MIN_REGIME_SAMPLES:  # e.g., 200
            continue
        X_regime = X_train[mask]
        y_regime = y_train[mask]
        importance[regime][fold] = compute_mda(X_regime, y_regime)
```

### 5.3 Core + Regime-Local Feature Sets

**Core features:** Features with importance above median in ALL 3 regimes across ALL folds. These are always included regardless of current regime.

**Regime-local features:** Features with importance above median in at least 1 regime but below median in others. These are included only when the current regime matches.

**Architecture:**
```
FeatureSet = CoreFeatures + RegimeLocalFeatures[current_regime]
```

At inference time, the current regime is detected from the most recent data, and the appropriate regime-local features are appended to the core set.

### 5.4 Overfitting Guards for Regime Layer

The biggest risk of regime-conditional selection is **regime overfitting** — the regime labels themselves may be optimized to make certain features look good.

**Guards:**

1. **Regime labels are fixed and rule-based.** No optimization of regime boundaries. ADX thresholds are per-symbol constants, not tuned.

2. **Minimum data per regime.** At least 200 samples per regime per fold. If a regime has fewer samples, skip regime-conditional selection for that fold and use the full-data importance instead.

3. **Regime feature sets are small.** Max 10 regime-local features per regime. This limits the "degrees of freedom" available for overfitting.

4. **Cross-validation of regime benefit.** After selecting regime-conditional features, compare CV performance of (core + regime-local) vs (core only). If regime-local features don't improve OOS performance by at least 0.5% Sharpe, discard them.

5. **Regime stability check.** If regime labels change for more than 30% of samples when ADX threshold is perturbed by +/-2, the regime definition is unstable. Fall back to regime-agnostic selection.

---

## 6. Timeframe-Aware Feature Selection

### 6.1 The Problem

ML Factory computes ~30 features at each of 3 MTF timeframes (5min, 15min, 60min), producing 90 MTF features. Many are redundant: RSI(14) at 5min, 15min, and 60min measure the same economic concept at different granularities. Including all three wastes capacity and introduces substitution effects.

### 6.2 Timeframe Tournament

For each base feature, run a tournament across timeframes:

```python
feature_families = group_by_base_name(mtf_features)
# e.g., {"rsi_14": ["rsi_14_5min", "rsi_14_15min", "rsi_14_60min"]}

for family, variants in feature_families.items():
    scores = {}
    for variant in variants:
        scores[variant] = robustness_score(variant)  # R(feature) from Section 4

    # Sort by robustness score
    ranked = sorted(scores.items(), key=lambda x: -x[1])

    # Keep the winner
    winners.add(ranked[0])

    # Keep runner-up only if:
    #   1. Its correlation with winner < 0.70
    #   2. Its robustness score > 0.50
    #   3. It provides regime-complementary signal
    if len(ranked) > 1:
        runner_up = ranked[1]
        corr = abs(correlation(ranked[0], ranked[1]))
        if corr < 0.70 and runner_up[1] > 0.50:
            winners.add(runner_up)
```

### 6.3 Rules

1. **Default:** Keep 1 timeframe per feature family. The one with the highest robustness score wins.
2. **Exception:** Keep 2 timeframes if they are decorrelated (|r| < 0.70) and both score above 0.50. This happens when a feature captures different dynamics at different scales (e.g., RSI at 5min captures intraday mean-reversion while RSI at 60min captures trend).
3. **Never keep 3+ timeframes** of the same feature family. If you need 3, your feature is not capturing a clear signal — it's capturing noise at multiple scales.
4. **Base timeframe (1min) features do not compete with MTF.** They're a separate category because they don't have shift(1) lookahead adjustment.

### 6.4 Timeframe Preferences by Feature Family

Based on financial theory and the nature of each feature:

| Feature Family | Expected Best TF | Rationale |
|---------------|-----------------|-----------|
| Momentum (RSI, MACD) | 15min or 60min | Intraday momentum needs smoothing to be signal |
| Volatility (ATR, BBands) | 5min or 15min | Volatility clusters at short horizons |
| Volume (VWAP, OBV) | 5min | Volume patterns are most informative at trade level |
| Trend (ADX, SMA cross) | 60min | Trend is a low-frequency phenomenon |
| Microstructure | 1min (base) | Microstructure is a high-frequency phenomenon |
| Mean Reversion | 15min | Mean-reversion works at intermediate horizons |

These are priors, not rules. The tournament decides.

---

## 7. Ticker-Aware Feature Selection

### 7.1 The 3-Tier Feature Architecture

```
Tier 1: UNIVERSAL FEATURES (work on all tickers)
  - Features portable across MES, MGC, MNQ
  - Must pass robustness threshold on ALL symbols
  - Examples: ATR-normalized returns, volatility regime, time features

Tier 2: ASSET-CLASS FEATURES (work within asset class)
  - Equity index features (MES, MNQ share dynamics)
  - Commodity features (MGC has different microstructure)
  - Must pass on 2+ symbols of same class

Tier 3: TICKER-LOCAL FEATURES (work on one symbol only)
  - Symbol-specific patterns (MGC session effects, MNQ volatility clustering)
  - Higher bar for inclusion (robustness score > 0.60)
  - Max 10 ticker-local features per symbol
```

### 7.2 Portability Testing

```python
def test_feature_portability(feature, symbols):
    """Test if a feature is portable across symbols."""
    importance_by_symbol = {}
    for symbol in symbols:
        data = load_data(symbol)
        importance_by_symbol[symbol] = walk_forward_importance(feature, data)

    # Feature is portable if important on 2+ symbols
    threshold = 0.5  # median importance
    important_on = sum(1 for imp in importance_by_symbol.values()
                       if imp > threshold)

    if important_on == len(symbols):
        return FeatureTier.UNIVERSAL
    elif important_on >= 2:
        return FeatureTier.ASSET_CLASS
    elif important_on == 1:
        return FeatureTier.TICKER_LOCAL
    else:
        return FeatureTier.REJECTED
```

### 7.3 Transfer Validation

To test whether a feature is genuinely portable (not just coincidentally important on multiple symbols):

1. Train model on Symbol A using feature set S
2. Evaluate on Symbol B (out-of-sample symbol, not just out-of-sample time)
3. Features whose importance drops by > 50% when tested cross-symbol are NOT portable — they're capturing symbol-specific noise

This is expensive but provides the strongest evidence of feature universality. Run it quarterly, not on every training cycle.

### 7.4 Per-Symbol Feature Budget

| Symbol | Core | Asset-Class | Ticker-Local | Total Budget |
|--------|------|------------|-------------|-------------|
| MES | 30 | 10 | 10 | 50 |
| MGC | 30 | 5 | 15 | 50 |
| MNQ | 30 | 10 | 10 | 50 |

MGC gets more ticker-local features because it trades on COMEX (different session, different microstructure) and has fewer asset-class peers.

---

## 8. Leakage-Safe Pipeline Design

### 8.1 The Cardinal Rule

> **Feature selection MUST happen inside each CV fold, using ONLY the training portion of that fold.**

This means:
- No feature statistics computed on test data
- No correlation matrices computed on test data
- No MDA importance computed on test data
- No variance filtering computed on test data
- No feature scaling computed on test data

### 8.2 Current Leakage in ML Factory

**Location:** `src/models/training/feature_selection.py:_run_feature_selection_pipeline()`

**Problem:** This function is called in `_pre_training_validation()` which runs BEFORE any train/test split. The MDA ranking, correlation filter, and variance filter all see the full DataFrame including future test data.

**Impact:** Features are selected partly based on their predictive power on test data. This inflates apparent model performance and creates unrealistically optimistic backtests.

### 8.3 Correct Pipeline Architecture

```
For each outer CV fold (PurgedKFold):
    X_train_outer, X_test_outer = split(data, fold)

    # ALL feature selection happens on X_train_outer ONLY
    selected_features = feature_governance_pipeline(X_train_outer, y_train_outer)

    # Train model on X_train_outer with selected_features
    model.fit(X_train_outer[selected_features], y_train_outer)

    # Evaluate on X_test_outer (never seen during selection)
    score = model.score(X_test_outer[selected_features], y_test_outer)
```

### 8.4 Where Selection Must Sit Relative to Splits

```
Full Dataset
    |
    +-- Outer Split (PurgedKFold, 5 folds)
    |       |
    |       +-- Train (80% of data)
    |       |       |
    |       |       +-- Feature Selection Pipeline (Layers 1-6)
    |       |       |       |
    |       |       |       +-- Inner Split (for MDA, 3 sub-folds)
    |       |       |               |
    |       |       |               +-- Sub-train: fit RF, compute MDA
    |       |       |               +-- Sub-test: score MDA on holdout
    |       |       |
    |       |       +-- Train Model (on selected features)
    |       |
    |       +-- Test (20% of data) -- NEVER used for selection
    |
    +-- Final Hold-Out (optional, for deployment decision only)
```

### 8.5 MTF Feature Selection Without Lookahead

MTF features have an inherent lookahead risk: higher-timeframe candles aggregate future data within their period.

**Current protection:** `shift(1)` on all MTF features before reindexing. This is correct but must be verified:

1. **Feature selection must use the shifted MTF features**, not raw higher-TF data
2. **Correlation computation between base-TF and MTF features** must use aligned timestamps (both shifted)
3. **Walk-forward windows for stability analysis** must respect the MTF shift — a window ending at time T uses MTF features computed from data up to T-1 bar

### 8.6 Leakage in Feature Scaling

**Current protection:** `FoldAwareScaler` fits on fold training data only. This is correct.

**Risk:** If feature selection is done BEFORE scaling, selected features are chosen on unscaled data. If done AFTER scaling, the scaling itself might introduce information (e.g., the scaler's median is computed on data that includes future samples).

**Correct order:**
```
1. Split data into fold train/test
2. Select features on fold train (unscaled) -- scaling doesn't affect MDA for tree models
3. Fit scaler on fold train
4. Scale fold train and fold test
5. Train model on scaled fold train
6. Evaluate on scaled fold test
```

### 8.7 Leakage in Sequence Creation

For 3D/4D models, sequence windows overlap in time. A window at time T includes data from T-seq_len to T.

**Risk:** If feature selection is done on windowed data, adjacent windows share most of their data. This creates pseudo-replication that inflates importance estimates.

**Correct approach:** Feature selection on 2D (tabular) data ONLY, even for sequence models. Select features on the flat DataFrame, then create sequences from the selected features. This is already how ML Factory's adapter system works — selection happens before `DataPreparer.prepare()` creates windows.

---

## 9. Feature Redundancy and Clustering

### 9.1 Why Correlation Filtering Is Not Enough

The current approach (Union-Find on pairs with |r| > 0.85) has three problems:

1. **Transitive chaining:** Features A-B correlated at 0.86, B-C at 0.86, but A-C only at 0.50. Union-Find merges all three into one group, potentially dropping a feature (C) that's actually distinct from A.

2. **fillna(0) creates phantom correlations:** Two features with non-overlapping NaN patterns both become 0 where the other is valid. Their correlation is computed on artificial zeros, not actual values.

3. **Static threshold:** 0.85 is arbitrary. In a low-SNR domain, features with 0.80 correlation may still carry distinct information. In a high-redundancy domain, even 0.70 correlation may be too permissive.

### 9.2 Better Approach: Hierarchical Clustering with Eigen-Gap

**Step 1:** Compute distance matrix using `1 - |Spearman correlation|` (not Pearson — Spearman handles non-linear relationships).

**Step 2:** Drop features with > 30% NaN BEFORE computing correlation. For remaining NaN, use pairwise complete observations (not fillna(0)).

**Step 3:** Hierarchical clustering with Ward linkage.

**Step 4:** Determine optimal cluster count using the **eigen-gap** method (ONC - Optimal Number of Clusters from Lopez de Prado):

```python
def optimal_cluster_count(corr_matrix, max_clusters=None):
    """ONC via eigen-gap on correlation matrix."""
    eigenvalues = np.linalg.eigvalsh(corr_matrix)
    eigenvalues = np.sort(eigenvalues)[::-1]  # descending

    # Marchenko-Pastur bound for noise eigenvalues
    n, p = corr_matrix.shape
    q = n / p
    lambda_plus = (1 + 1/np.sqrt(q))**2

    # Count eigenvalues above noise threshold
    n_signal = np.sum(eigenvalues > lambda_plus)

    if max_clusters:
        n_signal = min(n_signal, max_clusters)

    return max(n_signal, 2)  # at least 2 clusters
```

**Step 5:** Within each cluster, select the champion by:
1. MDA importance (primary)
2. Robustness score (secondary)
3. Interpretability priority (tertiary, tie-breaker)

### 9.3 When to Keep Multiple Features from Same Cluster

Sometimes a cluster contains features that are correlated but capture different economic information. Examples:

- RSI(14) and Stochastic K both measure momentum but with different normalization
- ATR(14) and Bollinger Width both measure volatility but with different baselines

**Keep both if:**
1. They appear in different top-20 lists across CV folds (not always substituting)
2. Their partial correlation (controlling for cluster mean) is significant
3. Their regime-conditional importance profiles differ (one matters in trending, other in ranging)
4. The model's performance improves by > 0.3% Sharpe with both vs just the champion

### 9.4 Family-Level Redundancy

Beyond individual features, entire FAMILIES may be redundant:

| Potentially Redundant Pair | Test |
|---------------------------|------|
| Momentum + Mean Reversion | These are mathematical inverses. If both families survive, one is likely capturing the negative signal of the other. Keep the one with higher robustness. |
| Volatility + Entropy | Both measure "how much the market is moving." Entropy may add non-linear information beyond volatility. Test with forward selection. |
| Microstructure + Volume | Significant overlap in information content. Microstructure may be redundant when volume features are included. |
| Wavelets + Raw Price | Wavelets are derived from price. If wavelet features don't add information beyond momentum + volatility, they're redundant. |

**Test:** Compare model performance with Family A only, Family B only, and both. If both together isn't better than the best individual, drop the weaker family.

---

## 10. Economic Value vs Pure Statistical Importance

### 10.1 The Economic Value Score

A feature's value is not its MDA importance. It's its **marginal contribution to tradable edge**.

```
EV(feature) = marginal_sharpe * (1 - turnover_penalty) * regime_breadth * DSR_correction
```

Where:

**marginal_sharpe:**
```
S_with = cost-adjusted Sharpe of model WITH feature
S_without = cost-adjusted Sharpe of model WITHOUT feature
marginal_sharpe = max(S_with - S_without, 0)
```

**turnover_penalty:**
```
trades_with = trade count of model WITH feature
trades_without = trade count of model WITHOUT feature
extra_turnover = (trades_with - trades_without) / trades_without
turnover_penalty = min(extra_turnover * cost_per_trade / avg_profit_per_trade, 1.0)
```
Features that increase trading frequency without proportionally increasing profit are penalized.

**regime_breadth:**
```
= (# regimes where marginal_sharpe > 0) / (total # regimes)
```
Features that improve Sharpe in all regimes are more valuable.

**DSR_correction:**
```
= max(DSR(sharpe_with) - DSR(sharpe_without), 0) / marginal_sharpe
```
Adjusts for the probability that the marginal Sharpe improvement is due to chance (multiple testing).

### 10.2 Cost-Adjusted Feature Ranking

Instead of ranking features by MDA importance, rank by:

```
adjusted_rank(feature) = alpha * MDA_importance
                       + beta  * robustness_score
                       + gamma * economic_value

where alpha=0.3, beta=0.4, gamma=0.3
```

**Robustness gets the highest weight.** A feature with mediocre importance but excellent stability is more valuable than a flashy feature with high importance in one backtest.

### 10.3 Trade Filtering Quality

Some features don't improve predictions but improve WHICH trades are taken. A feature that helps the model avoid losing trades (even if it doesn't predict winners better) is valuable.

**Metric:**
```
filter_value(feature) = win_rate_with - win_rate_without
                      + (avg_win/avg_loss)_with - (avg_win/avg_loss)_without
```

Features with positive filter_value improve trade quality even if they don't improve raw accuracy.

---

## 11. Feature Promotion/Demotion Lifecycle

### 11.1 Feature States

```
                    evidence_threshold_met
   CANDIDATE -----> PROBATIONARY -----> APPROVED
       |                  |                 |
       |                  | fails           | sustained
       |                  v                 v
       |              REJECTED           CORE
       |                  |                 |
       |                  |        decline  |  symbol-specific
       |                  v        detected |  only
       |              RETIRED        |      v
       |                             v    LOCAL
       +<-- re-evaluate ---------- DEPRECATED
```

### 11.2 State Definitions and Transition Rules

**CANDIDATE:**
- New feature, never tested in production
- Entry: feature is added to the universe (new code, new parameter)
- Exit: after 1 full training cycle with robustness scoring
- Promoted to PROBATIONARY if: R(feature) >= 0.30

**PROBATIONARY:**
- Feature under evaluation
- Duration: minimum 2 training cycles, maximum 5
- Promoted to APPROVED if: R(feature) >= 0.50 for 2 consecutive cycles
- Demoted to REJECTED if: R(feature) < 0.30 for 2 consecutive cycles

**APPROVED:**
- Feature validated for production use
- Included in model training
- Promoted to CORE if: R(feature) >= 0.70 for 3 consecutive cycles AND important in 3+ regimes
- Demoted to DEPRECATED if: R(feature) drops below 0.40 for 2 consecutive cycles

**CORE:**
- Always included in every model (within contract bounds)
- Highest confidence features
- Demoted to APPROVED if: R(feature) drops below 0.60 for 2 consecutive cycles

**LOCAL:**
- Feature approved only for specific ticker(s) or regime(s)
- Annotated with allowed_symbols and allowed_regimes
- Demoted to DEPRECATED if: robustness drops below 0.40 on its target scope

**DEPRECATED:**
- Feature marked for removal
- Still included in training for monitoring (to detect recovery)
- Duration: 2 training cycles
- Promoted back to APPROVED if: R(feature) recovers above 0.50
- Demoted to RETIRED if: no recovery after 2 cycles

**RETIRED:**
- Feature permanently removed from the active universe
- Retained in registry for audit trail
- Can be re-evaluated as CANDIDATE if manually triggered

**REJECTED:**
- Feature that never passed probation
- Not included in training
- Can re-enter as CANDIDATE if implementation changes

### 11.3 Evidence Thresholds

| Transition | Required Evidence |
|-----------|------------------|
| CANDIDATE -> PROBATIONARY | R >= 0.30, at least 1 cycle |
| PROBATIONARY -> APPROVED | R >= 0.50 for 2 consecutive cycles |
| APPROVED -> CORE | R >= 0.70 for 3 cycles, regime breadth >= 0.60 |
| APPROVED -> LOCAL | R >= 0.50 on target scope, < 0.30 on other scopes |
| Any -> DEPRECATED | R drops below 0.40 for 2 cycles |
| DEPRECATED -> RETIRED | No recovery in 2 cycles |
| DEPRECATED -> APPROVED | R recovers above 0.50 |

### 11.4 Training Cycle Definition

A "training cycle" = one complete pipeline run with walk-forward cross-validation. For daily retraining, a cycle is 1 day. For weekly, 1 week. The lifecycle system is agnostic to calendar time — it counts pipeline executions.

---

## 12. Model-Family Interaction

### 12.1 Strong Opinion: Feature Selection Must Be Model-Family-Specific

**Universal feature selection across model families is wrong.** Here's why:

1. **Tree models (XGBoost, LightGBM, CatBoost)** are invariant to monotonic transformations. They don't care if a feature is scaled, log-transformed, or raw. They handle high cardinality and interactions natively.

2. **Neural models (LSTM, GRU, TCN)** are sensitive to scale, require smooth features, and struggle with high cardinality. Features that work well for trees may be useless for neural models.

3. **Transformer models (PatchTST, iTransformer, TFT)** attend across time and variables. They benefit from features with temporal structure and cross-variable relationships. Static features (e.g., day-of-week encoding) add noise.

### 12.2 Three-Tier Selection Architecture

```
Tier A: SHARED SELECTION (computed once, shared across families)
  - Layer 1: Pre-filtering (variance, NaN) -- family-agnostic
  - Layer 2: Redundancy reduction -- family-agnostic
  - Layer 9 clustering: Feature clustering -- family-agnostic

Tier B: FAMILY-SPECIFIC SELECTION (per model family)
  - Layer 3: Stability filtering -- different thresholds per family
  - Layer 4: Regime-conditional -- same regime system, different feature sets
  - Layer 6: Economic value -- measured with family's actual model

Tier C: CONTRACT-AWARE TRUNCATION (per model)
  - Final feature count bounded by ModelContract.max_features
  - Top-N by family-specific ranking
```

### 12.3 Family-Specific Configurations

| Config | Boosting | Neural (RNN/CNN) | Transformer |
|--------|----------|-----------------|-------------|
| MDA model | RandomForest | RandomForest* | RandomForest* |
| n_features target | 50-80 | 40-60 | 30-50 |
| Stability threshold | 0.50 | 0.60 | 0.65 |
| Prefer ratio features | No (handles raw) | Yes (scale-sensitive) | Yes |
| Include temporal features | Optional | Yes (sequential) | Yes (attention) |
| Include microstructure | Yes | Conditional | Conditional |
| Regime-local budget | 15 | 10 | 8 |

*MDA uses RandomForest even for neural/transformer feature selection because it's fast and model-agnostic. The actual model's performance is tested in Layer 6 (economic value).

### 12.4 Contract-Aware Data Rank Routing

The adapter system already handles 2D/3D/4D routing. Feature selection should respect data rank:

- **2D models:** All features are valid. No sequence constraints.
- **3D models:** Features must have temporal continuity (no large gaps). Prefer features with autocorrelation > 0.1 (they carry sequential information).
- **4D models:** MTF features are natural inputs. Prefer features available across multiple timeframes (the model can attend across TF channels).

### 12.5 Ensemble Feature Coordination

When building cross-family ensembles (boosting + transformer), the base models should ideally use **different but complementary** features. If both models use identical features, the ensemble gains little diversity.

**Recommendation:** After selecting features for each family, compute the overlap. If overlap > 70%, force the second model to include more features from under-represented families:

```python
overlap = len(set(boosting_features) & set(transformer_features))
overlap_pct = overlap / min(len(boosting_features), len(transformer_features))
if overlap_pct > 0.70:
    # Force transformer to explore different feature space
    transformer_features = diversify_features(
        base=transformer_features,
        avoid=boosting_features,
        candidates=approved_features
    )
```

---

## 13. Implementation Blueprint for ML Factory

### 13.1 New Module Structure

```
src/
├── governance/                          # NEW: Feature governance system
│   ├── __init__.py
│   ├── registry.py                     # FeatureRegistry: central feature store
│   ├── scorer.py                       # RobustnessScorer: 8-dimension scoring
│   ├── lifecycle.py                    # FeatureLifecycle: state machine
│   ├── selector.py                     # GovernanceSelector: main pipeline
│   ├── regime_selector.py              # RegimeAwareSelector: regime-conditional
│   ├── timeframe_selector.py           # TimeframeCompetitor: TF tournament
│   ├── ticker_selector.py             # TickerAwareSelector: portability testing
│   ├── economic_scorer.py              # EconomicValueScorer: cost-adjusted value
│   ├── clustering.py                   # FeatureClusterer: ONC + hierarchical
│   ├── manifest.py                     # GovernanceManifest: audit trail
│   └── config.py                       # GovernanceConfig: all settings
```

### 13.2 Core Classes

**FeatureRegistry** — Central source of truth for all features:
```python
@dataclass
class FeatureEntry:
    name: str
    family: FeatureFamily
    base_timeframe: str
    mtf_timeframe: Optional[str]  # None for base TF features
    state: FeatureState  # CANDIDATE, PROBATIONARY, APPROVED, CORE, LOCAL, DEPRECATED, RETIRED
    robustness_score: float
    economic_value: float
    regime_importance: Dict[str, float]  # {regime: importance}
    ticker_tier: FeatureTier  # UNIVERSAL, ASSET_CLASS, TICKER_LOCAL
    allowed_symbols: List[str]  # empty = all symbols
    allowed_regimes: List[str]  # empty = all regimes
    selection_history: List[SelectionRecord]  # audit trail
    created_at: datetime
    last_evaluated: datetime
    promotion_count: int
    demotion_count: int

class FeatureRegistry:
    def __init__(self, registry_path: Path):
        self.entries: Dict[str, FeatureEntry] = {}
        self.registry_path = registry_path

    def get_active_features(self, symbol: str, regime: str,
                          model_family: ModelFamily) -> List[str]: ...
    def update_scores(self, scores: Dict[str, RobustnessScore]): ...
    def apply_lifecycle_transitions(self): ...
    def save(self): ...
    def load(self): ...
```

**RobustnessScorer** — Computes all 8 robustness dimensions:
```python
@dataclass
class RobustnessScore:
    fold_stability: float      # S_fold
    window_stability: float    # S_window
    regime_breadth: float      # S_regime
    cost_sensitivity: float    # S_cost
    label_sensitivity: float   # S_label
    data_perturbation: float   # S_data
    param_sensitivity: float   # S_param
    economic_contribution: float  # S_econ
    composite: float           # weighted combination

class RobustnessScorer:
    def __init__(self, config: GovernanceConfig):
        self.weights = config.robustness_weights
        self.min_fold_frequency = config.min_fold_frequency

    def score_fast(self, feature: str, fold_importances: Dict,
                   window_importances: Dict, regime_importances: Dict
                   ) -> RobustnessScore: ...

    def score_full(self, feature: str, ...,
                   cost_perturbations: List[float],
                   label_perturbations: List[Tuple],
                   bootstrap_importances: Dict
                   ) -> RobustnessScore: ...
```

**GovernanceSelector** — Main pipeline orchestrator:
```python
class GovernanceSelector:
    """Replaces _run_feature_selection_pipeline in feature_selection.py."""

    def __init__(self, config: GovernanceConfig, registry: FeatureRegistry):
        self.config = config
        self.registry = registry
        self.clusterer = FeatureClusterer(config)
        self.regime_selector = RegimeAwareSelector(config)
        self.tf_competitor = TimeframeCompetitor(config)
        self.ticker_selector = TickerAwareSelector(config)
        self.economic_scorer = EconomicValueScorer(config)
        self.robustness_scorer = RobustnessScorer(config)

    def select_features(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        regime_labels: pd.Series,
        symbol: str,
        model_family: ModelFamily,
        cv_splitter: PurgedKFold,
    ) -> FeatureSelectionResult:
        """Full governance pipeline. Runs INSIDE each outer CV fold."""
        ...
```

### 13.3 Integration Points

**Into UnifiedTrainingOrchestrator:**
```python
# BEFORE (current, leaky):
def _pre_training_validation(self, df):
    self._run_feature_selection_pipeline(df)  # sees all data!

# AFTER (governance, leakage-safe):
def _train_standard(self, df, additional_dfs):
    governance = GovernanceSelector(self.config.governance, self.registry)
    for fold_idx, (train_idx, test_idx) in enumerate(cv_splits):
        X_train = df.iloc[train_idx]
        y_train = labels.iloc[train_idx]
        regime_labels = compute_regime(X_train)
        selected = governance.select_features(
            X_train, y_train, regime_labels,
            symbol=self.config.symbol,
            model_family=model.family,
            cv_splitter=inner_cv_splitter
        )
        # Train model on selected features only
        model.fit(X_train[selected.features], y_train)
```

**Into Optuna:**
```python
# The 5D objective should use the governance selector's approved features
# as the searchable universe, not all features
searchable = registry.get_features_by_state(
    states=[FeatureState.APPROVED, FeatureState.CORE, FeatureState.LOCAL],
    symbol=symbol,
    model_family=model_family
)
# Optuna Dimension 2 searches within this approved set only
```

**Into Regime Models:**
```python
# Regime-aware training mode uses governance selector's regime-conditional features
core_features = registry.get_core_features()
regime_features = registry.get_regime_local_features(regime=current_regime)
features = core_features + regime_features
```

**Into Live Inference:**
```python
# At inference time, load the governance manifest
manifest = GovernanceManifest.load(deploy_path / "governance_manifest.json")
current_regime = detect_regime(latest_data)
features = manifest.get_inference_features(
    symbol=symbol,
    regime=current_regime,
    model_family=model_family
)
# Use only these features for prediction
prediction = model.predict(latest_data[features])
```

### 13.4 GovernanceConfig

```python
@dataclass
class GovernanceConfig:
    # Layer 1: Pre-filtering
    min_variance_cv: float = 0.01
    max_nan_fraction: float = 0.30

    # Layer 2: Clustering
    correlation_method: str = "spearman"  # not pearson
    clustering_linkage: str = "ward"
    max_clusters: Optional[int] = None  # None = ONC auto
    use_onc: bool = True

    # Layer 3: Stability
    min_fold_frequency: float = 0.60
    n_stability_windows: int = 8
    stability_window_months: int = 6
    stability_stride_months: int = 3

    # Layer 4: Regime
    n_regimes: int = 3
    min_regime_samples: int = 200
    max_regime_local_features: int = 10
    regime_benefit_threshold: float = 0.005  # min Sharpe improvement

    # Layer 5: Timeframe
    max_tf_per_feature_family: int = 2
    tf_decorrelation_threshold: float = 0.70
    min_runner_up_robustness: float = 0.50

    # Layer 6: Economic value
    cost_perturbation_factors: List[float] = field(
        default_factory=lambda: [0.5, 0.75, 1.0, 1.5, 2.0]
    )
    min_marginal_sharpe: float = 0.0

    # Layer 7: Lifecycle
    probation_cycles: int = 2
    core_promotion_cycles: int = 3
    deprecation_cycles: int = 2
    core_robustness_threshold: float = 0.70
    approved_robustness_threshold: float = 0.50
    probationary_robustness_threshold: float = 0.30
    max_ticker_local_features: int = 10

    # Robustness weights
    robustness_weights: Dict[str, float] = field(default_factory=lambda: {
        "fold_stability": 0.20,
        "window_stability": 0.20,
        "regime_breadth": 0.15,
        "cost_sensitivity": 0.10,
        "label_sensitivity": 0.10,
        "data_perturbation": 0.10,
        "param_sensitivity": 0.05,
        "economic_contribution": 0.10,
    })

    # MDA settings
    mda_n_estimators: int = 50  # up from current 20
    mda_n_repeats: int = 5     # up from current 3
    mda_max_depth: int = 5
    mda_subsample_train: int = 50_000
    mda_subsample_test: int = 20_000
```

---

## 14. Pseudocode / Algorithmic Flow

### 14.1 Full Feature Governance Pipeline

```python
def run_feature_governance(
    df: pd.DataFrame,
    labels: pd.Series,
    symbol: str,
    model_families: List[ModelFamily],
    config: GovernanceConfig,
    registry: FeatureRegistry,
) -> Dict[ModelFamily, FeatureSelectionResult]:
    """
    Complete feature governance pipeline.
    Called ONCE per training run.
    All selection happens INSIDE CV folds (no leakage).
    """

    # =========================================================
    # LAYER 0: Universe Definition
    # =========================================================
    all_features = identify_feature_columns(df)
    mtf_features = [f for f in all_features if is_mtf_feature(f)]
    base_features = [f for f in all_features if not is_mtf_feature(f)]

    # =========================================================
    # LAYER 1: Pre-Filtering (stateless, on full data OK)
    # These are purely statistical filters that don't use labels
    # =========================================================
    # Remove near-constant features
    variance_mask = compute_normalized_variance(df[all_features]) >= config.min_variance_cv
    surviving = [f for f, keep in zip(all_features, variance_mask) if keep]

    # Remove high-NaN features
    nan_mask = df[surviving].isna().mean() <= config.max_nan_fraction
    surviving = [f for f, keep in zip(surviving, nan_mask) if keep]

    logger.info(f"Layer 1: {len(all_features)} -> {len(surviving)} features")

    # =========================================================
    # LAYER 2: Redundancy Reduction (clustering)
    # Uses feature values only, no labels -- OK on full data
    # =========================================================
    # Compute Spearman correlation (pairwise complete, NOT fillna(0))
    corr_matrix = df[surviving].corr(method="spearman", min_periods=100)

    # ONC clustering
    distance_matrix = 1 - corr_matrix.abs()
    n_clusters = optimal_cluster_count_onc(corr_matrix) if config.use_onc \
                 else config.max_clusters
    cluster_labels = hierarchical_cluster(distance_matrix, n_clusters, config.clustering_linkage)

    # Map features to clusters
    feature_clusters: Dict[int, List[str]] = defaultdict(list)
    for feat, cluster_id in zip(surviving, cluster_labels):
        feature_clusters[cluster_id].append(feat)

    logger.info(f"Layer 2: {len(surviving)} features -> {n_clusters} clusters")

    # =========================================================
    # LAYERS 3-6: Per-Family, Inside CV Folds
    # THIS IS THE LEAKAGE-SAFE CORE
    # =========================================================
    outer_cv = PurgedKFold(
        n_splits=5,
        purge_bars=config.purge_bars,
        embargo_bars=config.embargo_bars
    )

    results_by_family: Dict[ModelFamily, FeatureSelectionResult] = {}

    for model_family in model_families:
        contract = get_model_contract(model_family)
        fold_selections: List[List[str]] = []
        fold_importances: List[Dict[str, float]] = []
        fold_robustness: List[Dict[str, RobustnessScore]] = []

        for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(df)):
            X_train = df.iloc[train_idx][surviving]
            y_train = labels.iloc[train_idx]
            regime_train = compute_regime_labels(X_train, symbol)

            # -------------------------------------------------
            # LAYER 3: Stability Filtering (inside fold)
            # Use inner CV for importance stability
            # -------------------------------------------------
            inner_cv = PurgedKFold(n_splits=3, ...)
            inner_importances: List[Dict[str, float]] = []

            for inner_train, inner_test in inner_cv.split(X_train):
                Xi_train = X_train.iloc[inner_train]
                yi_train = y_train.iloc[inner_train]
                Xi_test = X_train.iloc[inner_test]
                yi_test = y_train.iloc[inner_test]

                # Subsample for speed
                if len(Xi_train) > config.mda_subsample_train:
                    Xi_train, yi_train = stratified_subsample(
                        Xi_train, yi_train, config.mda_subsample_train
                    )
                if len(Xi_test) > config.mda_subsample_test:
                    Xi_test, yi_test = stratified_subsample(
                        Xi_test, yi_test, config.mda_subsample_test
                    )

                # MDA importance (clustered)
                rf = RandomForestClassifier(
                    n_estimators=config.mda_n_estimators,
                    max_depth=config.mda_max_depth,
                    random_state=42 + fold_idx * 10 + inner_fold_idx
                )
                rf.fit(Xi_train, yi_train)
                perm_imp = permutation_importance(
                    rf, Xi_test, yi_test,
                    n_repeats=config.mda_n_repeats,
                    n_jobs=-1
                )
                inner_importances.append(dict(zip(Xi_train.columns, perm_imp.importances_mean)))

            # Feature stability: fraction of inner folds where feature is in top-N
            target_n = min(contract.max_features, len(surviving))
            stability_scores = compute_selection_frequency(inner_importances, top_n=target_n)
            stable_features = [f for f, freq in stability_scores.items()
                             if freq >= config.min_fold_frequency]

            # -------------------------------------------------
            # LAYER 3b: Cluster Champion Selection
            # From each cluster, keep the most stable+important member
            # -------------------------------------------------
            mean_importance = average_importance(inner_importances)
            champions = []
            for cluster_id, cluster_members in feature_clusters.items():
                in_stable = [f for f in cluster_members if f in stable_features]
                if not in_stable:
                    # Keep the single best from the cluster
                    best = max(cluster_members, key=lambda f: mean_importance.get(f, 0))
                    if mean_importance.get(best, 0) > 0:
                        champions.append(best)
                else:
                    # Keep all stable members (up to 2 per cluster)
                    ranked = sorted(in_stable, key=lambda f: -mean_importance.get(f, 0))
                    champions.extend(ranked[:2])

            # -------------------------------------------------
            # LAYER 4: Regime-Conditional Selection (inside fold)
            # -------------------------------------------------
            core_features = []
            regime_local_features: Dict[str, List[str]] = defaultdict(list)

            for regime in REGIME_NAMES:
                regime_mask = regime_train == regime
                if regime_mask.sum() < config.min_regime_samples:
                    continue

                X_regime = X_train[regime_mask][champions]
                y_regime = y_train[regime_mask]

                rf_regime = RandomForestClassifier(n_estimators=config.mda_n_estimators, ...)
                rf_regime.fit(X_regime, y_regime)
                regime_imp = permutation_importance(rf_regime, X_regime, y_regime, ...)
                regime_importance[regime] = dict(zip(champions, regime_imp.importances_mean))

            # Core = important in ALL regimes
            for feat in champions:
                important_in_regimes = sum(
                    1 for r in regime_importance
                    if regime_importance[r].get(feat, 0) > np.median(
                        list(regime_importance[r].values())
                    )
                )
                if important_in_regimes == len(regime_importance):
                    core_features.append(feat)
                elif important_in_regimes >= 1:
                    for r in regime_importance:
                        if regime_importance[r].get(feat, 0) > np.median(
                            list(regime_importance[r].values())
                        ):
                            regime_local_features[r].append(feat)

            # Cap regime-local features
            for r in regime_local_features:
                regime_local_features[r] = sorted(
                    regime_local_features[r],
                    key=lambda f: -mean_importance.get(f, 0)
                )[:config.max_regime_local_features]

            # -------------------------------------------------
            # LAYER 5: Timeframe Competition (inside fold)
            # -------------------------------------------------
            # Group MTF features by base name
            tf_families = group_mtf_by_base_name(core_features + flatten(regime_local_features))
            tf_winners = []
            for base_name, variants in tf_families.items():
                if len(variants) <= 1:
                    tf_winners.extend(variants)
                    continue
                # Rank by mean importance
                ranked = sorted(variants, key=lambda f: -mean_importance.get(f, 0))
                tf_winners.append(ranked[0])  # Always keep winner
                if len(ranked) > 1:
                    # Keep runner-up if decorrelated
                    corr = abs(X_train[ranked[0]].corr(X_train[ranked[1]], method="spearman"))
                    if corr < config.tf_decorrelation_threshold:
                        tf_winners.append(ranked[1])

            # Replace in core and regime-local
            core_features = [f for f in core_features if f in tf_winners or f not in flatten(tf_families.values())]
            for r in regime_local_features:
                regime_local_features[r] = [f for f in regime_local_features[r]
                                          if f in tf_winners or f not in flatten(tf_families.values())]

            # -------------------------------------------------
            # Record fold results
            # -------------------------------------------------
            fold_selections.append(core_features + flatten(regime_local_features.values()))
            fold_importances.append(mean_importance)

        # =========================================================
        # AGGREGATE ACROSS FOLDS
        # =========================================================
        # Final feature set = features selected in >= min_fold_frequency of outer folds
        feature_counts = Counter(f for sel in fold_selections for f in sel)
        n_folds = len(fold_selections)
        final_features = [
            f for f, count in feature_counts.items()
            if count / n_folds >= config.min_fold_frequency
        ]

        # Truncate to contract max_features
        avg_importance = {}
        for f in final_features:
            imps = [fold_imp.get(f, 0) for fold_imp in fold_importances]
            avg_importance[f] = np.mean(imps)

        if len(final_features) > contract.max_features:
            final_features = sorted(final_features, key=lambda f: -avg_importance[f])
            final_features = final_features[:contract.max_features]

        if len(final_features) < contract.min_features:
            # Add more features from the fold_importances to reach minimum
            remaining = [f for f in surviving if f not in final_features]
            remaining_ranked = sorted(remaining, key=lambda f: -np.mean(
                [fold_imp.get(f, 0) for fold_imp in fold_importances]
            ))
            while len(final_features) < contract.min_features and remaining_ranked:
                final_features.append(remaining_ranked.pop(0))

        # =========================================================
        # LAYER 6: Economic Value Gate (post-aggregation)
        # =========================================================
        # Compute marginal Sharpe for top features
        # (expensive -- only run on final candidates, not all 200+)
        if config.min_marginal_sharpe > 0:
            final_features = economic_value_filter(
                final_features, df, labels, symbol,
                min_marginal_sharpe=config.min_marginal_sharpe,
                model_family=model_family
            )

        results_by_family[model_family] = FeatureSelectionResult(
            features=final_features,
            importances=avg_importance,
            stability_scores=feature_counts,
            regime_features=aggregate_regime_features(fold_selections),
            n_folds=n_folds,
        )

    # =========================================================
    # LAYER 7: Governance Lifecycle Update
    # =========================================================
    for family, result in results_by_family.items():
        for feature in all_features:
            new_score = compute_robustness_score(feature, result, config)
            registry.update_score(feature, new_score, family)
        registry.apply_lifecycle_transitions()
    registry.save()

    return results_by_family
```

### 14.2 Robustness Score Computation

```python
def compute_robustness_score(
    feature: str,
    result: FeatureSelectionResult,
    config: GovernanceConfig,
) -> RobustnessScore:
    """Compute fast-path robustness score (3 dimensions)."""

    # S_fold: selection frequency across outer folds
    s_fold = result.stability_scores.get(feature, 0) / result.n_folds

    # S_window: computed from rolling window importances (if available)
    if feature in result.window_importances:
        imps = result.window_importances[feature]
        mean_imp = np.mean(imps)
        if mean_imp > 0:
            s_window = max(0, 1 - np.std(imps) / mean_imp)
        else:
            s_window = 0.0
    else:
        s_window = s_fold  # fallback to fold stability

    # S_regime: fraction of regimes where feature is important
    if feature in result.regime_features:
        n_regimes_important = sum(
            1 for r, feats in result.regime_features.items()
            if feature in feats
        )
        s_regime = n_regimes_important / max(len(result.regime_features), 1)
    else:
        s_regime = 0.0

    # Composite (fast path uses 3 dimensions, renormalized)
    w = config.robustness_weights
    fast_weights = {
        "fold_stability": w["fold_stability"],
        "window_stability": w["window_stability"],
        "regime_breadth": w["regime_breadth"],
    }
    total_weight = sum(fast_weights.values())

    composite = (
        fast_weights["fold_stability"] * s_fold +
        fast_weights["window_stability"] * s_window +
        fast_weights["regime_breadth"] * s_regime
    ) / total_weight

    return RobustnessScore(
        fold_stability=s_fold,
        window_stability=s_window,
        regime_breadth=s_regime,
        cost_sensitivity=0.0,  # slow path
        label_sensitivity=0.0,  # slow path
        data_perturbation=0.0,  # slow path
        param_sensitivity=0.0,  # slow path
        economic_contribution=0.0,  # slow path
        composite=composite,
    )
```

### 14.3 Regime Label Computation

```python
REGIME_TRENDING = "trending"
REGIME_MEAN_REVERTING = "mean_reverting"
REGIME_VOLATILE = "volatile"
REGIME_NAMES = [REGIME_TRENDING, REGIME_MEAN_REVERTING, REGIME_VOLATILE]

def compute_regime_labels(df: pd.DataFrame, symbol: str) -> pd.Series:
    """
    Compute 3-regime labels. Rule-based, lagged by 1 bar.
    Uses existing regime features but classifies into 3 states.
    """
    symbol_config = SymbolConfig.from_symbol_or_default(symbol)
    adx_threshold = symbol_config.adx_trending_threshold

    # Compute indicators (these are already available as features)
    adx = compute_adx(df, period=14)
    sma_fast = df["close"].rolling(20).mean()
    sma_slow = df["close"].rolling(50).mean()
    trend_strength = (sma_fast - sma_slow).abs() / sma_slow
    hurst = compute_rolling_hurst(df["close"], window=100)
    vol_short = df["close"].pct_change().rolling(20).std()
    vol_long = df["close"].pct_change().rolling(60).std()
    high_vol = vol_short > vol_long

    # Classify
    regime = pd.Series(REGIME_VOLATILE, index=df.index)  # default

    trending_mask = (adx > adx_threshold) & (trend_strength > 0.005)
    mean_reverting_mask = (adx < adx_threshold) & (hurst < 0.45) & (~high_vol)

    regime[trending_mask] = REGIME_TRENDING
    regime[mean_reverting_mask] = REGIME_MEAN_REVERTING
    # Everything else stays VOLATILE

    # LAG BY 1 BAR to prevent lookahead
    regime = regime.shift(1)
    regime.iloc[0] = REGIME_VOLATILE  # fill first bar

    return regime
```

---

## 15. Blunt Critique of Current Risks

### 15.1 CRITICAL: Feature Selection on Full Dataset Is Leakage

**The single biggest problem in ML Factory today.**

`_run_feature_selection_pipeline()` is called in `_pre_training_validation()`, which runs before any train/test split. This means MDA importance is computed on data that includes the test set. Features are being selected partly based on their correlation with future labels that the model will later be evaluated on.

**Impact:** Every backtest result is optimistically biased. Features that are good "on average" (including the test period) are selected, giving the model an unfair advantage. In production, the model won't have seen the future — so it will underperform its backtest.

**Fix:** Move feature selection INSIDE each CV fold. This is the single most important change in this document.

### 15.2 HIGH: Too Many Features, Not Enough Data

196 base features + 90 MTF = ~286 features. For MES at 5-min bars, a year of data is ~100K rows. After purge/embargo and CV splitting, each training fold might have 60-70K rows.

The rule of thumb for tabular ML is 10-50 observations per feature per class. With 3 classes and 286 features, you need at minimum 286 * 30 = ~8,500 rows. You have enough rows numerically, but the effective degrees of freedom are much lower because:
- Many features are highly correlated (reducing effective dimensionality)
- Financial data has low SNR (each observation carries less information)
- Serial correlation means adjacent rows are not independent

**Recommendation:** Target 40-60 features for boosting, 30-50 for neural/transformer. The current system allows up to 200 features for XGBoost — this is almost certainly overfitting.

### 15.3 HIGH: Too Many Models

12 models is too many for most use cases. Each model introduces:
- Feature selection complexity (per-model subsets)
- Training time
- OOF generation complexity
- Ensemble weight estimation (more base models = more weights to estimate = more overfitting in the meta-learner)

**Recommendation:** Use 3-4 models in production: 1 boosting (XGBoost or LightGBM, not both), 1 sequence model (LSTM or TCN), and optionally 1 transformer (PatchTST). CatBoost, GRU, InceptionTime, ResNet1D, iTransformer, and N-BEATS add marginal value at significant complexity cost.

### 15.4 HIGH: Regime Overfitting Risk

The current 6-state composite regime (vol * 3 + trend_encoded) with per-symbol ADX thresholds creates a combinatorial explosion of configurations. With 3 symbols x 6 regimes = 18 regime-symbol combinations, each with potentially different optimal features, the system has enormous degrees of freedom for overfitting.

**Recommendation:** Use 3 regimes (trending, mean-reverting, volatile). Keep regime definitions simple and rule-based. Don't optimize regime boundaries.

### 15.5 HIGH: Feature Instability

MDA with n_estimators=20 is far too unstable for reliable feature ranking. Small random forests have high variance in feature importance estimates. A feature that appears "top-5" with seed=42 might be "top-50" with seed=43.

**Recommendation:** n_estimators=50 minimum (ideally 100), n_repeats=5. Use stability scores to filter out features whose importance fluctuates across folds.

### 15.6 MEDIUM: Substitution Effects Not Fully Addressed

The ONC clustering (clustered MDA) partially addresses substitution effects, but:
- It uses maxclust=20, which may be too few clusters for 200+ features
- Cluster importance is distributed equally among members (lossy)
- Correlation filter runs AFTER MDA, not before, so MDA rankings are already corrupted by substitution

**Recommendation:** Cluster FIRST, then compute per-cluster importance, then select champion from each cluster. Current order is: MDA first, cluster/filter second — which is backwards.

### 15.7 MEDIUM: False Discovery Risk

With 196 features, 100+ Optuna trials, multiple timeframes, and multiple label horizons, you are running thousands of implicit statistical tests. Even with DSR correction, the probability of finding features that appear significant by chance is high.

**Quantifying the risk:** With 200 features and alpha=0.05, you expect ~10 false positives. With 4 timeframes, that's ~40 false positives across the MTF feature space. Many of your "selected" features may be noise.

**Recommendation:** Require features to pass stability filtering (selection frequency >= 0.60 across 5+ folds) AND regime breadth filtering (important in 2+ regimes). This dramatically reduces false discovery rate because random noise features are unlikely to be consistently selected across multiple independent tests.

### 15.8 MEDIUM: Optimization Overreach

The 5-dimension Optuna objective simultaneously optimizes barrier params, feature selection, feature params, timeframes, and model hyperparams. This is a ~500-dimensional search space. With 100 trials and TPE sampling, you are barely scratching the surface of this space. Most "optimal" configurations found are local optima that won't generalize.

**Recommendation:** Fix features and barrier params FIRST (using the governance pipeline). Then use Optuna for model hyperparams only (a ~10-20 dimensional space that 100 trials can actually explore). Don't try to optimize everything at once.

### 15.9 LOW: Research-to-Live Mismatch

ML Factory has excellent backtest infrastructure but no mention of:
- Online prediction serving
- Feature computation latency at inference time
- Data staleness detection
- Model monitoring and alerting
- Automatic model retraining triggers

Features selected in research may not be computable in real-time. For example, entropy features require 50-100 bar lookback windows; if the latest bars are delayed or missing, the feature is stale.

**Recommendation:** Track feature computation latency. Flag features that require > 1 second to compute at inference time. Prefer features that degrade gracefully when data is stale.

### 15.10 LOW: Orphaned Code Is Technical Debt

6 feature selection modules exist but are never called from the production pipeline: `WalkForwardFeatureSelector`, `PurgedFeatureSelector`, `OHLCVFeatureSelector`, `FeatureSelectionManager`, `FeatureOptimizer`, `strategies.py`. This is confusing and creates maintenance burden.

**Recommendation:** Either integrate them into the governance pipeline or delete them. Dead code attracts bugs.

---

## 16. Ranked Recommendations

### 16.1 The Best Practical Design

**What to implement if you want the best balance of quality and effort:**

1. Fix the leakage: move feature selection inside CV folds
2. Increase MDA stability: n_estimators=50, n_repeats=5
3. Add 3 robustness dimensions: S_fold, S_window, S_regime (fast path)
4. Add timeframe competition (single winner per feature family)
5. Reduce active models to 4 (XGBoost, LSTM, PatchTST, meta-learner)
6. Delete orphaned feature selection code

**Effort:** ~2-3 weeks. **Impact:** Eliminates leakage, reduces overfitting, improves stability.

### 16.2 The Safest Design

**What to implement if you prioritize correctness above all:**

1. Everything in "Best Practical" above
2. Full 8-dimension robustness scoring (including slow path: cost/label sensitivity)
3. Feature lifecycle governance with registry
4. Regime-conditional feature sets (core + regime-local)
5. Ticker portability testing
6. Fix Optuna to only optimize hyperparams (not features)

**Effort:** ~5-6 weeks. **Impact:** Production-grade feature governance with audit trail.

### 16.3 The Most Robust Design

**What to implement if you want maximum resistance to overfitting:**

1. Everything in "Safest Design" above
2. Bootstrap stability (20 bootstrap samples per feature)
3. Label perturbation testing (5 barrier parameter variants)
4. Parameter sensitivity testing (feature param variants)
5. Cross-symbol transfer validation (quarterly)
6. CPCV with PBO computation wired into production path
7. Permutation-based significance test for non-Sharpe metrics

**Effort:** ~8-10 weeks. **Impact:** Institutional-grade, publication-quality validation.

### 16.4 The Most Computationally Expensive but Highest-Quality Design

**What to implement if compute is unlimited:**

1. Everything in "Most Robust" above
2. Full economic value scoring with marginal Sharpe per feature per fold
3. Per-model feature selection with actual model (not RF proxy)
4. Real-time feature importance monitoring in production
5. Automatic feature demotion from live performance data
6. Adversarial feature testing (synthetic regime injection)
7. Ensemble diversity-aware feature selection (force decorrelation across model families)

**Effort:** ~3-4 months. **Impact:** State-of-the-art, exceeds most hedge fund internal systems.

### 16.5 What I Would Implement First

If this were my system and I had to ship in 1 week, I would do exactly these 5 things in this order:

**Day 1-2: Fix the leakage (CRITICAL)**
- Move `_run_feature_selection_pipeline()` inside the CV fold loop in `_train_standard()`
- Pass only `X_train_fold` to MDA, never test data
- This is the single highest-impact change

**Day 3: Stabilize MDA**
- n_estimators: 20 -> 50
- n_repeats: 3 -> 5
- Add selection frequency tracking: only keep features selected in >= 60% of folds
- Use the existing `WalkForwardFeatureSelector` (it's already written and orphaned!)

**Day 4: Add timeframe competition**
- Group MTF features by base name
- Keep top-1 per family by average MDA importance
- Keep runner-up only if correlation with winner < 0.70

**Day 5: Cluster before MDA, not after**
- Compute correlation clusters FIRST (ONC or hierarchical)
- Run MDA on cluster representatives (mean of cluster features)
- Select champion from each cluster by MDA importance
- This eliminates substitution effects

**Day 6-7: Reduce feature/model count**
- Hard cap at 60 features for boosting, 50 for neural/transformer
- Default to 3 models (XGBoost, LSTM or TCN, PatchTST)
- Delete or archive orphaned feature selection code

These 5 changes would transform ML Factory's feature selection from "likely overfit" to "defensibly robust" in one week.

---

## Appendix A: Glossary

| Term | Definition |
|------|-----------|
| **MDA** | Mean Decrease in Accuracy — permutation importance on holdout data |
| **ONC** | Optimal Number of Clusters — eigen-gap method for determining cluster count |
| **DSR** | Deflated Sharpe Ratio — corrects for selection bias in multiple testing |
| **CPCV** | Combinatorial Purged Cross-Validation — generates multiple train/test paths |
| **PBO** | Probability of Backtest Overfitting — fraction of CPCV paths that underperform |
| **FWER** | Family-Wise Error Rate — probability of at least one false positive |
| **SNR** | Signal-to-Noise Ratio — ratio of signal power to noise power |
| **Substitution effect** | When correlated features steal each other's MDA importance |
| **Alpha decay** | Gradual loss of a trading signal's profitability over time |
| **Concept drift** | Change in the joint distribution P(X, Y) over time |

## Appendix B: References

1. Lopez de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley. — Chapters 6-8 (purged CV, ONC, feature importance)
2. Bailey, D. & Lopez de Prado, M. (2014). "The Deflated Sharpe Ratio." — Multiple testing correction
3. Lopez de Prado, M. (2020). "Clustering (HCBM) and ONC." — Optimal number of clusters
4. Coqueret, G. & Guida, T. (2020). *Machine Learning for Factor Investing*. — Feature engineering for finance
5. Gu, S., Kelly, B., & Xiu, D. (2020). "Empirical Asset Pricing via Machine Learning." — Feature selection in financial ML

---

*End of Feature Governance Framework Design Document*
*Next step: Prioritize and implement per Section 16.5*
