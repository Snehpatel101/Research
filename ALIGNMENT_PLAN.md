# Codebase Alignment Plan: Target System vs Current State

**Generated:** 2025-12-29
**Status:** Actionable roadmap to production-grade ML factory
**Estimated Effort:** 10-14 weeks (1 engineer) | 6-8 weeks (2 engineers)

---

## Executive Summary

The Research codebase is **54% aligned** with the target vision of a production-grade, model-agnostic ML factory for OHLCV time-series research and deployment. The current implementation has:

- ✅ **Strong foundation:** 13 models, 15-stage data pipeline, OOF-based stacking
- ⚠️ **Critical blockers:** 3 data pipeline bugs creating leakage
- 🔴 **Missing components:** 9+ advanced models, automated bundling, production infrastructure

**Path to 100% alignment:** Fix 3 critical bugs (Week 1-2), add 9 missing model architectures (Week 3-8), enhance meta-learning (Week 9-11), build production infra (Week 12-14).

---

## Part 1: Target System (What You're Building)

### Core Vision

A **production-grade, model-agnostic ML factory** where:

1. **Single data source → many model families**
   - Ingest raw OHLCV (one contract, one timeframe)
   - Produce standardized datasets for tabular, sequence, and foundation models
   - Adapt representations to model architectures (feature matrices vs windowed tensors vs patches)

2. **Fair evaluation under identical controls**
   - All models trained on same leakage-safe data (purge/embargo)
   - Identical transaction costs, slippage, and risk constraints
   - Standardized metrics (Sharpe, win rate, max drawdown, regime performance)

3. **Ensemble via meta-learning**
   - OOF predictions from diverse base learners
   - Stacking meta-learners that learn *when to trust which model*
   - Regime-aware weighting (volatility, trend, structure)

4. **Inference as first-class workflow**
   - Train/serve parity (same feature pipelines)
   - One-command train→bundle→deploy
   - Predictable outputs (class probabilities, expected return, confidence)

### Strategic Differentiators

| Feature | Implementation | Value Proposition |
|---------|----------------|-------------------|
| **Representation Adapters** | Tabular models get 2D feature matrices; sequence models get 3D windowed tensors | Each model family receives data in its optimal format |
| **Leakage Paranoia** | Purge (60 bars), embargo (1440 bars), train-only scaling | Prevents overfitting to future data |
| **Regime-Aware Meta-Learning** | Meta-learners learn model performance by volatility/trend regime | Adaptive ensemble performance across market conditions |
| **Foundation Model Integration** | Chronos, TimesFM as zero-shot baselines | Fast baselines without custom training |
| **Probabilistic Forecasts** | Quantile RNN, DeepAR for distribution outputs | Risk-aware position sizing and stop-loss placement |

---

## Part 2: Current State vs Target

### Model Coverage

| Family | Current (13 models) | Target (22+ models) | Gap |
|--------|---------------------|---------------------|-----|
| **Boosting (GBDT)** | XGBoost, LightGBM, CatBoost | ✅ Complete | 0 |
| **Neural (RNN/CNN)** | LSTM, GRU, TCN, Transformer | Add: 1D ResNet, InceptionTime | 2 missing |
| **Classical ML** | Random Forest, Logistic, SVM | ✅ Complete | 0 |
| **Transformers (Advanced)** | Basic Transformer | Add: PatchTST, iTransformer, TimesNet, Autoformer | **4 missing** |
| **State-Space Models** | None | Add: S-Mamba, N-BEATS, N-HiTS | **3 missing** |
| **Foundation Models** | None | Add: Chronos-Bolt, TimesFM 2.5 | **2 missing** |
| **Ensemble** | Voting, Stacking, Blending | ✅ Complete (need regime-aware meta-learner) | 0 |

**Total: 13/22 models (59% coverage)**

### Infrastructure Completeness

| Component | Current | Target | Status |
|-----------|---------|--------|--------|
| **Data Pipeline** | 15-stage pipeline with 150+ features | Bug-free with validated leakage prevention | ⚠️ **3 critical bugs** |
| **Training** | `train_model.py` supports all 13 models | Auto-generate inference bundles | 🟡 Missing auto-bundling |
| **Inference** | ModelBundle + InferencePipeline exist | Automated train→bundle→deploy | 🟡 Manual bundling step |
| **Meta-Learning** | OOF-based stacking | Regime-aware meta-learners | 🟡 40% complete |
| **Production** | REST API exists | Docker + CI/CD + drift monitoring | 🔴 0% automated |

---

## Part 3: Critical Alignment Gaps

### 🔴 Gap 1: Data Pipeline Critical Bugs (BLOCKING PRODUCTION)

**Problem:** Three leakage bugs invalidate model performance:

1. **HMM Lookahead Bias** (`src/phase1/stages/regime/hmm.py:329-354`)
   - HMM trains on full dataset including future data
   - Fix: Set `expanding=False` or implement incremental training
   - Impact: Regime features leak future information

2. **GA Test Data Leakage** (`src/phase1/stages/ga_optimize/optuna_optimizer.py`)
   - Optuna optimization uses full dataset before train/val/test splits
   - Fix: Restrict optimization to train portion (70%)
   - Impact: Barrier parameters optimized on test data

3. **No Transaction Costs in Labels** (`src/phase1/stages/labeling/triple_barrier.py`)
   - Triple-barrier ignores round-trip costs
   - Fix: Adjust barriers by `cost_in_atr = (cost_ticks * tick_value) / atr`
   - Impact: Labels assume zero slippage (unrealistic)

**Estimated Effort:** 3-5 days
**Priority:** **CRITICAL** (blocks all production use)

---

### 🟡 Gap 2: Missing Advanced Transformer Models

**Problem:** Only vanilla Transformer implemented. SOTA models (PatchTST, iTransformer) show 21% MSE improvement but are absent.

**Missing Models:**
- **PatchTST:** Patch-based attention (21% MSE reduction vs vanilla)
- **iTransformer:** Inverted attention for multivariate correlations
- **TimesNet:** 2D convolutions on periodicities
- **Autoformer:** Decomposition + auto-correlation (best Sharpe in financial tests)

**Estimated Effort:** 2-3 weeks (3-4 days per model)
**Priority:** **HIGH** (competitive advantage)

---

### 🟡 Gap 3: Missing Foundation Models

**Problem:** No zero-shot baseline models. Chronos/TimesFM provide 51%+ directional accuracy without training.

**Missing Models:**
- **Chronos-Bolt:** Amazon, 250x faster than original
- **TimesFM 2.5:** Google, 200M params, probabilistic forecasts

**Value:** Fast baselines to compare against custom-trained models.

**Estimated Effort:** 1 week (wrappers are lightweight)
**Priority:** **MEDIUM** (quick baseline comparisons)

---

### 🟡 Gap 4: Inference Pipeline Not Integrated

**Problem:** Training outputs loose artifacts (model, scaler, config in separate files). No automatic bundle creation.

**Current Workflow:**
```bash
# Manual bundling required
python scripts/train_model.py --model xgb --horizon 20
# Outputs: experiments/runs/<run_id>/checkpoints/best_model
# Then manually: python notebooks/create_bundle.ipynb
```

**Target Workflow:**
```bash
# Auto-bundling
python scripts/train_model.py --model xgb --horizon 20 --create-bundle
# Outputs: experiments/bundles/xgb_h20/ (ready to deploy)
```

**Estimated Effort:** 1 week
**Priority:** **HIGH** (blocks production deployment)

---

### 🟡 Gap 5: No Regime-Aware Meta-Learning

**Problem:** Stacking ensembles use simple meta-learners (Logistic, Ridge). Cannot adapt to market conditions.

**Current:** Fixed ensemble weights across all regimes
**Target:** Meta-learners learn *when to trust which model* by regime (volatility, trend)

**Example:**
```python
# Current: simple stacking
meta_pred = logistic(base_model_preds)

# Target: regime-aware stacking
meta_pred = f(base_model_preds, volatility_regime, trend_regime)
# High volatility → favor boosting models
# Trending → favor sequence models
```

**Estimated Effort:** 2 weeks
**Priority:** **HIGH** (leverages existing regime features)

---

## Part 4: Implementation Roadmap

### Phase 1: Foundation (Week 1-2)

**Goal:** Fix critical bugs, make system production-safe.

**Tasks:**
1. Fix HMM lookahead bias (2 days)
2. Fix GA test data leakage (2 days)
3. Fix transaction costs in labels (2 days)
4. Regression test suite (2 days)

**Success Criteria:**
- ✅ HMM uses only past data
- ✅ GA optimization confined to train set
- ✅ Labels include transaction costs
- ✅ All 2,060 tests pass

**Files to Modify:**
- `src/phase1/stages/regime/hmm.py` (line 329)
- `src/phase1/stages/ga_optimize/optuna_optimizer.py` (full refactor)
- `src/phase1/stages/labeling/triple_barrier.py` (barrier calculation)

---

### Phase 2: Model Expansion (Week 3-8)

**Goal:** Expand from 13 to 22+ models.

#### Week 3-4: Foundation Models + Simple Baselines
5. Chronos-Bolt wrapper (3 days)
6. TimesFM 2.5 wrapper (3 days)
7. DLinear, TiDE baselines (2 days)

#### Week 5-6: Advanced Transformers
8. PatchTST implementation (4 days)
9. iTransformer implementation (3 days)
10. TimesNet + Autoformer (4 days)

#### Week 7-8: State-Space Models
11. S-Mamba implementation (5 days)
12. N-BEATS / N-HiTS (3 days)

**Success Criteria:**
- ✅ 9 new models registered (total: 22)
- ✅ All trainable via `train_model.py --model <name>`
- ✅ Performance benchmarks documented

**New Files:**
- `src/models/foundation/{chronos,timesfm}.py`
- `src/models/transformer/{patchtst,itransformer,timesnet,autoformer}.py`
- `src/models/ssm/{mamba,nbeats,nhits}.py`

---

### Phase 3: Meta-Learning Enhancement (Week 9-11)

**Goal:** Regime-aware stacking and confidence calibration.

13. Regime-aware meta-learner (5 days)
14. Confidence calibration integration (3 days)
15. OOF stacking with regime features (4 days)
16. Ensemble performance analysis (2 days)

**Success Criteria:**
- ✅ Meta-learner learns regime-conditional weights
- ✅ Calibrated probabilities match true frequencies
- ✅ Ensemble Sharpe > best single model

**Files to Modify:**
- `src/models/ensemble/stacking.py` (extend meta-learner)
- `src/cross_validation/oof_stacking.py` (add regime features)
- Create `src/models/ensemble/regime_meta_learner.py`

---

### Phase 4: Production Infrastructure (Week 12-14)

**Goal:** Automated train→bundle→deploy pipeline.

17. Auto-bundle generation (3 days)
18. Test set evaluation script (2 days)
19. Drift detection daemon (4 days)
20. CI/CD pipeline (3 days)

**Success Criteria:**
- ✅ `train_model.py --create-bundle` works end-to-end
- ✅ Test set evaluation is one command
- ✅ Drift detection running in production
- ✅ CI/CD pipeline green

**Files to Create:**
- `scripts/evaluate_test.py` (new)
- `src/monitoring/drift_daemon.py` (new)
- `.github/workflows/model_pipeline.yml` (new)

---

## Part 5: Quick Wins (Immediate Value)

**High-impact changes with minimal effort:**

1. **Add DLinear Baseline (4 hours)**
   - Simplest model: trend decomposition + linear layer
   - Often beats complex models
   - File: `src/models/neural/dlinear.py` (150 lines)

2. **Fix HMM Expanding Mode Default (1 hour)**
   - Change line 329: `expanding=False`
   - Immediate leakage fix

3. **Add Model Count Sanity Check (2 hours)**
   - Assert `len(ModelRegistry.list_all()) >= 13`
   - Prevents regression in model registration

4. **Document Current Model Performance (4 hours)**
   - Create `docs/MODEL_BENCHMARKS.md`
   - Baseline for future comparisons

5. **Bundle Creation Helper (3 hours)**
   - Add `create_bundle_from_run(run_id)` to `bundle.py`
   - Simplifies manual bundling

---

## Part 6: Recommended Execution Order

**Prioritized task list with dependencies:**

### Week 1-2: Critical Path (Blocking Production)
1. ✅ Fix HMM lookahead bias
2. ✅ Fix GA test data leakage
3. ✅ Fix transaction costs in labels
4. ✅ Regression test suite

### Week 1-2: Quick Wins (Parallel)
5. ✅ Add DLinear baseline
6. ✅ Model count sanity check
7. ✅ Document current performance
8. ✅ Bundle creation helper

### Week 3-4: Foundation Models (Fast ROI)
9. ✅ Chronos-Bolt wrapper
10. ✅ TimesFM 2.5 wrapper
11. ✅ TiDE implementation

### Week 5-6: Advanced Transformers (Research Edge)
12. ✅ PatchTST implementation
13. ✅ iTransformer implementation
14. ✅ TimesNet implementation
15. ✅ Autoformer implementation

### Week 7-8: State-Space Models (Efficiency)
16. ✅ S-Mamba implementation
17. ✅ N-BEATS / N-HiTS implementation

### Week 9-11: Meta-Learning (Ensemble Intelligence)
18. ✅ Regime-aware meta-learner
19. ✅ Confidence calibration integration
20. ✅ OOF stacking with regime features

### Week 12-14: Production Infrastructure (Deployment)
21. ✅ Auto-bundle generation
22. ✅ Test set evaluation script
23. ✅ Drift detection daemon
24. ✅ CI/CD pipeline

---

## Recommended Team Allocation

**Option 1: Single Engineer (10-14 weeks)**
- Focus on critical path → foundation models → transformers → meta-learning → production

**Option 2: Two Engineers (6-8 weeks)**
- **Engineer 1:** Critical bugs (Week 1-2) → Foundation models (Week 3-4) → Meta-learning (Week 9-11)
- **Engineer 2:** Transformers (Week 3-6) → State-space models (Week 7-8) → Production infra (Week 9-11)

**Parallelization opportunities:**
- Phase 2 (model expansion) and Phase 3 (meta-learning) can overlap
- Quick wins can be interleaved with critical path
- Production infra can start after Phase 1 completes

---

## Success Metrics

**Completion Criteria:**

| Phase | Metric | Target |
|-------|--------|--------|
| Phase 1 | Zero leakage bugs | All 3 bugs fixed + regression tests pass |
| Phase 2 | Model count | 22+ models registered |
| Phase 3 | Ensemble Sharpe | Regime-aware stacking > simple average |
| Phase 4 | Deployment time | Train→bundle→deploy in one command (<5 min) |

**Final System Capabilities:**

- ✅ Train any of 22+ model architectures with single command
- ✅ Fair evaluation under identical leakage-safe conditions
- ✅ Regime-aware ensemble meta-learning
- ✅ One-command train→bundle→deploy workflow
- ✅ Automated drift detection and CI/CD
- ✅ Probabilistic forecasts for risk management

---

## Next Steps

1. **Review this plan** with stakeholders
2. **Prioritize** based on business needs (research vs production focus)
3. **Start with Quick Wins** (Week 1, parallel with critical bugs)
4. **Execute Phase 1** (critical bugs) before any model expansion
5. **Iterate** based on model performance benchmarks

**Contact:** Update this plan as requirements evolve. Track progress in GitHub Issues/Projects.
