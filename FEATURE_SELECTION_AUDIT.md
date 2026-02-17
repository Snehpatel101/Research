# Feature Selection Audit — ML Factory

**Date:** 2026-02-16
**Scope:** Full feature selection pipeline audit with 4 specialized subagents + online research

---

## Table of Contents

1. [How It Actually Works Right Now](#1-how-it-actually-works-right-now)
2. [Why Variance-Based Selection Is A Problem](#2-why-variance-based-selection-is-a-problem)
3. [What Lopez de Prado Recommends](#3-what-lopez-de-prado-recommends)
4. [The Transformer Feature Limit Problem](#4-the-transformer-feature-limit-problem)
5. [Ensemble Feature Flow](#5-ensemble-feature-flow)
6. [Per-Model Feature Subsets Not Saved to Bundle](#6-per-model-feature-subsets-not-saved-to-bundle)
7. [Notebook Configuration Gaps](#7-notebook-configuration-gaps)
8. [Orphaned Code — Built But Never Called](#8-orphaned-code--built-but-never-called)
9. [Gap Summary](#9-gap-summary)
10. [Recommended Fixes (Priority Order)](#10-recommended-fixes-priority-order)
11. [State of the Art — Financial ML Feature Selection (2024-2026)](#11-state-of-the-art--financial-ml-feature-selection-2024-2026)
12. [Sources](#12-sources)

---

## 1. How It Actually Works Right Now

The system has **5 feature selection gates**, but only **2 are active by default**:

```
227 engineered features come out of the pipeline
     │
     ▼
┌─────────────────────────────────────────────┐
│  GATE 1: Variance Ranking (ALWAYS ON)       │  <-- The problem
│  Ranks ALL features by .var()               │
│  Each model gets top N by contract:         │
│    XGBoost/LightGBM/CatBoost → top 200     │
│    LSTM/GRU                  → top 150     │
│    TCN                       → top 120     │
│    N-BEATS                   → top 20      │
│    PatchTST/iTransformer     → top 10      │
└──────────────────┬──────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────┐
│  GATE 2: Contract Bounds (ALWAYS ON)        │  <-- Just validates
│  Checks min_features <= N <= max_features   │      bounds
└──────────────────┬──────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────┐
│  GATE 3: Correlation Filtering              │  <-- EXISTS but
│  Groups correlated features (threshold 0.85)│      NEVER RUNS
│  Keeps most interpretable per group         │      (orphaned)
└─────────────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────┐
│  GATE 4: Optuna Feature Selection           │  <-- Opt-in,
│  Binary include/exclude per feature         │      boosting
│  Only if FEATURE_SELECTION_ENABLED=True     │      models ONLY
│  Only for tabular models (not neural/xfmr)  │
└─────────────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────┐
│  GATE 5: Importance-Based Pruning           │  <-- EXISTS but
│  MDI, MDA, permutation importance           │      NEVER RUNS
│  Null importance testing                    │      (orphaned)
│  Walk-forward MDA with stability scoring    │
└─────────────────────────────────────────────┘
```

**In plain English:** The ONLY thing deciding which features each model sees is "which features wiggle the most" (variance). That's it.

### Gate Details

| Gate | Location | Method | Status |
|------|----------|--------|--------|
| **1. Variance** | `unified_orchestrator.py:404-428` | `.var().sort_values()` top-N | **Active** |
| **2. Contracts** | `model_contract.py:76-78` | min/max bounds check | **Active** |
| **3. Correlation** | `feature_selection/filtering.py` | Union-find grouping, threshold 0.85 | **Orphaned** |
| **4. Optuna** | `trainer.py:617-635` | Binary optimization per feature | **Opt-in, tabular only** |
| **5. Importance** | `feature_selection/`, `features/pruning.py` | MDI, MDA, permutation, null test | **Orphaned** |

### Per-Model Feature Limits (from ModelContract)

| Model | min_features | max_features | Selection Method |
|-------|-------------|-------------|-----------------|
| XGBoost/LightGBM/CatBoost | 40 | 200 | Variance + Optuna (optional) |
| Random Forest | 30 | 150 | Variance + Optuna (optional) |
| LSTM/GRU | 50 | 150 | Variance only |
| TCN | 50 | 120 | Variance only |
| InceptionTime/ResNet1D | 30 | 100 | Variance only |
| TFT | 20 | 80 | Variance only |
| PatchTST/iTransformer | 4 | 10 | Variance only |
| N-BEATS | 2 | 20 | Variance only |

**Key observation:** Sequence and transformer models SKIP Gate 4 (Optuna) entirely — they only get variance filtering.

---

## 2. Why Variance-Based Selection Is A Problem

Variance measures how much a feature moves around. It says **nothing** about whether those movements predict your labels.

### Failure Modes

| Failure | Example | Result |
|---------|---------|--------|
| **Noise wins** | A random-walk feature has huge variance but zero predictive power | Selected over useful features |
| **Scale bias** | Dollar volume variance >> normalized volume ratio variance | Volume ratio dropped even if more predictive |
| **Subtle signals killed** | Momentum z-score oscillating tightly near zero — low variance but may be the strongest regime predictor | Dropped for N-BEATS/transformers with tight limits |
| **Outlier sensitivity** | Single volatility spike inflates variance of otherwise low-info feature | Retained while stable predictors dropped |
| **Interactions missed** | Two moderate-variance features form a highly predictive pair | Neither selected individually |

### What Variance Filtering IS Good For

- Removing strictly constant or near-constant features (a legitimate pre-filter)
- Computational cheapness as a first-pass to reduce from thousands to hundreds

### What It Should NOT Be

The primary feature selection mechanism. It should be a pre-filter only, followed by a target-aware method.

---

## 3. What Lopez de Prado Recommends

From *Advances in Financial Machine Learning* (2018) and *Machine Learning for Asset Managers* (2020):

### The Three Core Methods

**Mean Decrease Impurity (MDI)**
- Uses sklearn's `feature_importances_` from tree-based models
- Measures how much each feature reduces impurity across all splits
- In-sample only; tree-based classifiers only
- Biased toward high-cardinality features; suffers from substitution effect

**Mean Decrease Accuracy (MDA)**
- Permute each feature's values, measure the drop in out-of-sample accuracy
- Works with any classifier; out-of-sample
- Still suffers from substitution effect — permuting one of two correlated features has little impact because the other compensates

**Single Feature Importance (SFI)**
- Train a separate model using only one feature at a time
- Eliminates substitution effect entirely
- Cannot capture interaction effects; very expensive (N models for N features)

### The Substitution Effect (Critical Concept)

When two features share information (are correlated), importance methods cannot reliably distinguish which one matters. One gets high importance; the other gets near-zero. This is pervasive in financial ML where many features are derived from the same OHLCV data (RSI and Stochastic, various MA crossovers, etc.).

### Lopez de Prado's Solution: Clustered Feature Importance

```
Step 1: Cluster correlated features (ONC algorithm)
         ┌──────────────────────────────────────────────┐
         │  RSI_14, Stochastic_K, Williams_R  → Cluster A│
         │  SMA_20, EMA_20, DEMA_20           → Cluster B│
         │  ATR_14, Bollinger_Width            → Cluster C│
         └──────────────────────────────────────────────┘

Step 2: Apply MDA at CLUSTER level
         - Permute ALL features in Cluster A simultaneously
         - Measure OOS accuracy drop
         - Big drop → cluster matters
         - No drop → cluster is noise

Step 3: Select representative features from important clusters
```

This handles the substitution effect because clusters are constructed to be internally correlated but mutually dissimilar. Permuting an entire cluster removes all redundant information, giving a clean importance signal.

**This code already exists in `src/optimization/feature_selection/`.** It's just not wired into the training pipeline.

---

## 4. The Transformer Feature Limit Problem

### Current Limits vs. Architecture Evidence

| Architecture | Current Limit | Evidence-Based Range | Paper Benchmark |
|---|---|---|---|
| PatchTST | 10 | 20-100+ | 7-862 variates (ICLR 2023) |
| iTransformer | 10 | 20-100+ | 21-862 variates (ICLR 2024) |
| TFT | 10 | 30-100+ | Built-in variable selection |

### Why These Limits Are Too Low

**PatchTST** (ICLR 2023) uses channel-independence — each feature is processed as its own univariate series through a shared Transformer encoder. The architecture explicitly supports arbitrary numbers of input channels. Adding more features does NOT cause cross-channel interference. Benchmarked on Traffic (862 variates), Electricity (321 variates), Weather (21 variates).

**iTransformer** (ICLR 2024 Spotlight) inverts the standard approach: **each feature becomes a token**, and attention captures cross-variate correlations. It is designed so that more features = more tokens for cross-variate attention to discover relationships. Limiting to 10 features cripples its core mechanism.

**TFT** has a built-in Variable Selection Network that dynamically weights features at each timestep. It can self-select relevant features. Limiting to 10 features means we're doing TFT's job for it, worse than it would do itself.

**With only 10 features** (likely 5 OHLCV + 5 engineered), the attention mechanism has almost nothing to attend over. The primary advantage of using transformers is lost.

### Recommended New Limits

| Model | Current | Recommended | Rationale |
|-------|---------|-------------|-----------|
| PatchTST | 4-10 | 20-80 | Channel-independent; more features = more channels, no interference |
| iTransformer | 4-10 | 20-80 | Variates-as-tokens; benefits from cross-variate attention |
| TFT | 20-80 | 30-100 | Built-in variable selection handles feature relevance internally |

---

## 5. Ensemble Feature Flow

### How Features Flow Through Ensemble Training

```
Raw Features (227 available)
    │
    ▼
Per-Model Feature Selection (unified_orchestrator._pre_training_validation)
    ├─ XGBoost: top 200 by variance
    ├─ LSTM: top 60 by variance
    └─ PatchTST: top 5 by variance
    │
    ▼
Data Preparation (prepare_with_cache filters to per-model subset)
    ├─ XGBoost: 2D with 200 features
    ├─ LSTM: 3D with 60 features
    └─ PatchTST: 4D with 5 features
    │
    ▼
Model Training with Per-Model Features
    ├─ XGBoost trains on 200 features
    ├─ LSTM trains on 60 features
    └─ PatchTST trains on 5 features
    │
    ▼
OOF Generation (uses prepared data as-is)
    ├─ XGBoost OOF: 10000 samples
    ├─ LSTM OOF: 9941 samples (missing 59 from seq_len windowing)
    └─ PatchTST OOF: 9941 samples
    │
    ▼
OOF Alignment (intersection of sample indices)
    ├─ Common range: [59...9999] = 9941 samples
    ├─ All OOF predictions aligned to common_indices
    └─ Creates AlignedOOFResult
    │
    ▼
Meta-Learner Training
    ├─ Input: OOF probabilities (9 features for 3 models x 3 classes)
    ├─ + 2 derived features (mean_confidence, prediction_agreement)
    ├─ = 11 stacking features total
    └─ Trains on 9941 aligned samples
```

### Key Finding: No Feature Leakage in Ensemble Stage

The ensemble stage is properly protected:
1. OOF generation uses **default configs** (prevents tuned hyperparameters from leaking)
2. OOF uses **PurgedKFold** with time-based splits (purge + embargo)
3. Meta-learner trains on **80/20 time-based split** of stacking features (separate from OOF)
4. Meta-learner sees **OOF predictions only**, never original features (unless passthrough=True)

---

## 6. Per-Model Feature Subsets Not Saved to Bundle

### The Bug

Per-model feature subsets are computed during training (`unified_orchestrator._per_model_features`) but **never serialized to the bundle**.

```
TRAINING:                          INFERENCE FROM BUNDLE:
──────────                         ──────────────────────
XGBoost → 200 features  ✓         Bundle loads → 200 features ✓
LSTM    → 60 features   ✓         Bundle loads → ??? features ✗
PatchTST→ 5 features    ✓         Bundle loads → ??? features ✗
```

### Where It Should Be Fixed

- `ModelBundle.from_training()` should accept and store the per-model feature list
- `BundleMetadata` should include `feature_names: list[str]` (not just `n_features: int`)
- `UniversalInferencePipeline` should filter features per model before prediction

### Impact

Inference bundles could feed wrong features to non-boosting models. This doesn't affect training or backtesting (which use the orchestrator directly), but would affect production inference via `load_deploy_artifact()`.

---

## 7. Notebook Configuration Gaps

### What Users CAN Control

| Setting | Location | Default |
|---------|----------|---------|
| `FEATURE_SELECTION_ENABLED` | Notebook Cell 2 | `False` |
| `FEATURE_SELECTION_METHOD` | Notebook Cell 2 | `"mda"` |

### What Users CANNOT Control (Hidden Defaults)

| Setting | Default | Location |
|---------|---------|----------|
| `selection_n_features` | 50 | `src/config/data.py:80` |
| `selection_cv_splits` | 5 | `src/config/data.py:82` |
| `min_feature_frequency` | 0.6 | `feature_selection/config.py:160` |
| `n_estimators` (for importance) | 100 | `feature_selection/config.py:163` |
| `use_clustered_importance` | False | `feature_selection/config.py:166` |
| `max_clusters` | 20 | `feature_selection/config.py:168` |
| Per-model feature limits | Hardcoded in contracts | `model_contract.py` |

### Method Mismatch Bug

The notebook exposes `"shap"` and `"mutual_info"` as valid methods, but the implementation (`FeatureSelectionConfig`) only supports `"mda"`, `"mdi"`, and `"hybrid"`. Selecting `"shap"` or `"mutual_info"` would silently fall back to default behavior.

---

## 8. Orphaned Code — Built But Never Called

The following feature selection systems are **complete implementations** that are **not wired into the main training pipeline**:

| System | Location | What It Does |
|--------|----------|-------------|
| Low-variance filter | `feature_selection/filtering.py` | Removes features below variance threshold |
| Correlation filter | `feature_selection/filtering.py` | Groups correlated features, keeps representative |
| Walk-forward MDA | `feature_selection/walk_forward.py` | Per-fold feature importance with stability scoring |
| Clustered MDA/MDI | `feature_selection/` | Lopez de Prado's clustered importance |
| Feature pruning | `data/features/pruning.py` | Iterative importance-based removal |
| Null importance test | `data/features/pruning.py` | Statistical significance testing against permuted features |
| Feature clustering | `feature_selection/` | ONC-based clustering for substitution effect control |

**These represent significant engineering effort that is currently unused.** The infrastructure for proper feature selection exists — it just needs to be connected.

---

## 9. Gap Summary

### Critical

1. **Primary selection is TARGET-BLIND** — variance != signal
2. **No handling of SUBSTITUTION EFFECT** — correlated features not grouped
3. **Transformer limits TOO LOW** — 10 vs 30-100 evidence-based
4. **Per-model features NOT SAVED to bundle** — inference bug

### Moderate

5. **Orphaned code** — Correlation filter, MDA, MDI, null importance, walk-forward selection all built but never called
6. **Optuna selection only works for boosting** — neural/transformer models get zero target-aware selection
7. **Notebook method mismatch** — "shap" and "mutual_info" advertised but not implemented

### Minor

8. **Notebook config incomplete** — selection_n_features, min_feature_frequency, per-model overrides not exposed
9. **No post-training importance validation** — no feedback loop to verify selection quality

---

## 10. Recommended Fixes (Priority Order)

| # | Change | Effort | Impact | Details |
|---|--------|--------|--------|---------|
| **1** | Replace variance ranking with Clustered MDA | Medium | **HIGH** | Code exists in `feature_selection/`. Wire it into `unified_orchestrator._pre_training_validation()` as primary selector |
| **2** | Increase transformer feature limits | Low | **HIGH** | Change `max_features` in ModelContract: PatchTST/iTransformer 10→80, TFT 80→100 |
| **3** | Save per-model feature subsets to bundle | Low | **HIGH** | Add `feature_names: list[str]` to BundleMetadata, populate in `BundleBuilder` |
| **4** | Wire correlation filtering before selection | Low | **MEDIUM** | Call `filter_correlated_features()` from `filtering.py` before variance/importance ranking |
| **5** | Add post-training importance validation | Low | **MEDIUM** | After training, compute permutation importance on trained model, log comparison with selected features |
| **6** | Fix notebook method mismatch | Low | **LOW** | Either implement SHAP/MI methods or remove from notebook options |
| **7** | Expose hidden config in notebook | Low | **LOW** | Add `selection_n_features`, `min_feature_frequency`, `use_clustered_importance` to Cell 2 |

### The Single Highest-Impact Change

**Swap variance ranking for Clustered MDA.** This would move feature selection from "which features wiggle most" to "which features actually predict your labels, accounting for correlations." The code already exists — it's a wiring task, not a build task.

---

## 11. State of the Art — Financial ML Feature Selection (2024-2026)

### Tier 1: Gold Standard Methods

| Method | Approach | Strengths | Weaknesses |
|--------|----------|-----------|------------|
| **Clustered Feature Importance** (Lopez de Prado) | Cluster correlated features via ONC, apply MDA/MDI at cluster level | Handles substitution effect; interpretable | Requires initial model training |
| **SHAP / Boruta-SHAP** | Shapley values from cooperative game theory; Boruta adds statistical shadow-feature testing | Captures interactions; statistically rigorous | Expensive (TreeSHAP helps for boosting) |
| **Boruta** | Creates shadow features (permuted copies), iteratively tests if real features outperform | Controls false positives | Requires many iterations |

### Tier 2: Strong Practical Methods

| Method | Approach | Strengths | Weaknesses |
|--------|----------|-----------|------------|
| **Permutation Importance** (with purged CV) | Permute feature values, measure OOS accuracy drop | Target-aware; model-agnostic; OOS | Substitution effect; expensive |
| **MRMR** (Max Relevance Min Redundancy) | Maximize mutual info with target while minimizing feature redundancy | Handles redundancy explicitly; fast | May miss complex interactions |
| **Embedded** (L1/LASSO, tree importance) | Model-internal importance during training | Fast; no extra computation | Substitution effect without clustering |

### Tier 3: Emerging Methods (2024-2025)

| Method | Approach | Status |
|--------|----------|--------|
| **LLM-Guided Selection** | Use LLMs for semantic feature subset identification | Early stage; promising for domain knowledge |
| **Feature Selection with Annealing** | Simulated annealing on feature subset combinatorial space | Escapes local optima; expensive |
| **Neural Architecture-Integrated** | TFT variable selection, Key Factor Selection Transformer | Production-ready for specific architectures |

### Critical Requirement: Temporal Integrity

All feature selection in financial ML **must** respect temporal ordering:
- Importance scores computed using purged/embargoed cross-validation
- Walk-forward re-selection at each step is most rigorous
- Features selected on full dataset = lookahead bias

---

## 12. Sources

### Academic Papers
- Lopez de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley.
- Lopez de Prado, M. (2020). *Machine Learning for Asset Managers*. Cambridge University Press.
- Nie, Y. et al. (2023). *A Time Series is Worth 64 Words: Long-term Forecasting with Transformers* (PatchTST). ICLR 2023. [arXiv:2211.14730](https://arxiv.org/abs/2211.14730)
- Liu, Y. et al. (2024). *iTransformer: Inverted Transformers Are Effective for Time Series Forecasting*. ICLR 2024 Spotlight. [arXiv:2310.06625](https://arxiv.org/abs/2310.06625)
- [Clustered Feature Importance (SSRN)](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3517595)
- [Feature Selection with Annealing for Financial Time Series](https://arxiv.org/abs/2303.02223)

### Documentation & Implementations
- [mlfinlab: MDI, MDA, SFI](https://random-docs.readthedocs.io/en/latest/implementations/feature_importance.html)
- [mlfinlab: Clustered MDA/MDI](https://www.mlfinlab.com/en/latest/feature_importance/clustered.html)
- [scikit-learn Feature Selection](https://scikit-learn.org/stable/modules/feature_selection.html)
- [Boruta-SHAP Feature Selection](https://towardsdatascience.com/boruta-shap-an-amazing-tool-for-feature-selection-every-data-scientist-should-know-33a5f01285c0/)
- [shap-select: Lightweight Feature Selection Using SHAP Values](https://arxiv.org/html/2410.06815v1)

### Codebase Locations Referenced
- `src/models/training/unified_orchestrator.py:404-428` — Variance-based filtering
- `src/core/contracts/model_contract.py:76-78` — Per-model feature limits
- `src/optimization/feature_selection/` — Orphaned feature selection infrastructure
- `src/optimization/feature_selection/filtering.py` — Correlation filter (orphaned)
- `src/optimization/feature_selection/walk_forward.py` — Walk-forward MDA (orphaned)
- `src/optimization/feature_selection/config.py` — FeatureSelectionConfig + ModelFamilyDefaults
- `src/optimization/feature_selection/manager.py` — FeatureSelectionManager
- `src/models/training/trainer.py:617-635` — Optuna feature selection (tabular only)
- `src/data/features/pruning.py` — Importance-based pruning (orphaned)
- `src/inference/bundle.py` — ModelBundle (missing per-model feature names)
- `src/models/training/services/oof_generation.py` — OOF generation (2D/3D/4D paths)
- `src/models/ensemble/stacking.py` — Stacking meta-learner
- `src/data/adapters/alignment.py` — OOF alignment
- `src/config/data.py:68-128` — FeatureConfig
- `notebooks/ml_factory_colab.ipynb` — User-facing configuration

---

*Generated: 2026-02-16 by 4 specialized subagents + online research*
*Audit scope: Feature selection pipeline, ensemble feature flow, notebook config, academic best practices*
