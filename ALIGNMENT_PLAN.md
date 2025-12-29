# Codebase Alignment Plan: Dynamic ML Factory for OHLCV Trading

**Generated:** 2025-12-29 (Updated with dynamic registry architecture)
**Status:** Actionable roadmap to production-grade ML factory
**Estimated Effort:** 12-16 weeks (1 engineer) | 7-10 weeks (2 engineers)
**Architecture:** **Dynamic factory with plug-and-play registries** (not hardcoded model selection)

---

## Executive Summary

The Research codebase is evolving into a **dynamic, registry-based ML factory** where users can mix-and-match any combination of models, ensembles, calibration, and inference strategies via configuration—without code changes.

### Core Philosophy: Configuration > Code

**Not this:** "Train these 3 specific models in this hardcoded ensemble"
**Instead:** "Select any models from registry + any ensemble method + any calibration + optional RL policy"

### Current State Assessment

- ✅ **Strong foundation:** 15-stage data pipeline, 13 models, OOF stacking infrastructure
- ⚠️ **Critical blockers:** 3 data pipeline bugs creating leakage
- 🔴 **Missing architecture:** No dynamic registries, hardcoded model selection
- 🔴 **Missing models:** 15+ SOTA architectures (PatchTST, N-BEATS, DeepAR, etc.)
- 🔴 **Missing inference layer:** No calibration, conformal prediction, or gating

### Transformation Strategy

**Phase 1 (Week 1-2):** Fix 3 critical data pipeline bugs → Zero-leakage pipeline

**Phase 2 (Week 3-4):** Build dynamic registry system → Config-driven model selection

**Phase 3 (Week 5-10):** Add 15+ models across 5 families → 28+ total models

**Phase 4 (Week 11-12):** Add meta-learners, calibration, conformal → Inference layer

**Phase 5 (Week 13-14):** Add optional RL policy layer → Adaptive trading

**Phase 6 (Week 15-16):** Production infrastructure → One-command deploy

---

## Table of Contents

1. [Target System Vision](#part-1-target-system-vision)
2. [Dynamic Registry Architecture](#part-2-dynamic-registry-architecture)
3. [Full Model Menu (28+ Models)](#part-3-full-model-menu)
4. [Contracts & Artifacts](#part-4-contracts--artifacts)
5. [Critical Data Pipeline Bugs](#part-5-critical-data-pipeline-bugs)
6. [Implementation Roadmap](#part-6-implementation-roadmap)
7. [Quick Wins](#part-7-quick-wins)
8. [Risk Analysis](#part-8-risk-analysis)
9. [Testing Strategy](#part-9-testing-strategy)

---

## Part 1: Target System Vision

### What You're Building

A **dynamic, registry-based ML factory** for systematic futures trading where model selection, ensemble strategy, calibration, and RL policies are **configuration-driven**—not hardcoded.

#### 1. Single Data Source → Dynamic Model Selection

- **Input:** Raw OHLCV (one contract: MES, MGC, etc.)
- **Output:** Any combination of models from 5 families (28+ options)
- **Selection:** Via YAML config, not code changes

```yaml
# Example: Mix tabular + sequence + foundation models
base_models:
  - {name: lightgbm, view: feature_matrix, outputs: [p_up]}
  - {name: tcn, view: window_tensor, outputs: [E_r]}
  - {name: patchtst, view: window_tensor, outputs: [E_r, q05, q50, q95]}
  - {name: chronos, view: window_tensor, outputs: [p_up], zero_shot: true}
```

#### 2. Dynamic Ensembling (Not Hardcoded Recipes)

- **Ensemble methods:** Voting, Stacking (Ridge, Elastic Net, LightGBM meta, CatBoost meta, MLP meta), Blending
- **Gating:** Softmax gate, HMM regime gate, Markov switching, Contextual bandit
- **Selection:** Via config

```yaml
ensemble:
  method: stacking
  meta_learner: ridge  # or elastic_net, lightgbm_meta, catboost_meta, mlp_meta
  oof: true
  gating:
    enabled: true
    method: hmm_regime  # or softmax, markov_switching, contextual_bandit
```

#### 3. Inference Layer (Calibration + Uncertainty)

- **Calibration:** Temperature scaling, Isotonic, Platt, Beta
- **Conformal prediction:** CQR, Split Conformal, CV+/Jackknife+
- **Selection:** Via config

```yaml
inference:
  calibrate: [temperature_scaling, isotonic]
  uncertainty:
    method: cqr  # or split_conformal, cv_plus, jackknife_plus
    coverage: 0.90
```

#### 4. Optional RL Policy Layer

- **RL as consumer** of inference outputs (not replacement for stacking)
- **Inputs:** `p_up`, `E[r]`, `q05/q50/q95`, `uncertainty`, `regime_score`, costs
- **Outputs:** Position size, entry/exit thresholds, risk throttles
- **Algorithms:** SAC, TD3, PPO, DQN, Contextual Bandits

```yaml
policy:
  enabled: true
  algorithm: sac  # or td3, ppo, dqn, contextual_bandit
  inputs: [p_up, E_r, q05, q50, q95, uncertainty, regime_score]
  outputs: [position_size, entry_threshold, exit_threshold]
```

#### 5. Plug-and-Play via Contracts

Every model/ensemble/inference component adheres to strict contracts:

- **Input View Contract:** `feature_matrix`, `window_tensor`, `mtf_bundle`
- **Output Contract:** `p_up`, `E[r]`, `q05/q50/q95`, `regime_score`
- **Artifact Contract:** `preproc + schema + model + metrics + inference_signature`

### Strategic Differentiators

| Feature | Implementation | Value Proposition |
|---------|----------------|-------------------|
| **Dynamic Selection** | All models/ensembles/inference via config, not code | Experiment without rewriting pipelines |
| **Strict Contracts** | Every component emits standardized outputs | Mix tabular + sequence + foundation models freely |
| **Leakage Paranoia** | Purge (60 bars), embargo (1440 bars), train-only scaling, causal models only | Prevents overfitting to future data |
| **Inference Layer** | Calibration + conformal prediction + gating | Production-grade uncertainty quantification |
| **Optional RL** | RL as policy consumer (not model replacement) | Adaptive position sizing + risk management |

---

## Part 2: Dynamic Registry Architecture

### Philosophy: Registries Replace Hardcoded Logic

Instead of:
```python
# BAD: Hardcoded
if model_name == "xgboost":
    model = XGBoost()
elif model_name == "lstm":
    model = LSTM()
```

Use:
```python
# GOOD: Registry-based
model = ModelRegistry.create(config["model_name"], config["model_params"])
```

### Four Registry Types

#### A) Model Registry (Base Learners)

**Purpose:** All trainable models (tabular, sequence, foundation)

**Entry Schema (`ModelSpec`):**
```python
@dataclass
class ModelSpec:
    name: str                    # "lightgbm", "patchtst", "chronos"
    family: str                  # "boosting", "transformer", "foundation"
    input_view: str              # "feature_matrix" or "window_tensor"
    outputs: List[str]           # ["p_up", "E_r", "q05", "q50", "q95", "regime_score"]
    needs: Dict[str, Any]        # {normalization, lookback, covariates, categorical_support}
    artifacts: Dict[str, str]    # {preproc, schema, weights, metrics, inference_signature}
    is_causal: bool              # Production safety (no future leakage)
    zero_shot: bool              # Foundation models (no training)
```

**Example Registration:**
```python
@register_model(
    name="patchtst",
    family="transformer",
    input_view="window_tensor",
    outputs=["p_up", "E_r", "q05", "q50", "q95"],
    is_causal=True,
    zero_shot=False
)
class PatchTSTModel(BaseModel):
    ...
```

---

#### B) Ensemble Registry (How Base Models Combine)

**Purpose:** Stacking, voting, blending, gating strategies

**Entry Schema (`EnsembleSpec`):**
```python
@dataclass
class EnsembleSpec:
    name: str                    # "ridge_stacking", "lightgbm_meta", "hmm_gating"
    method: str                  # "stacking", "voting", "blending", "gating"
    meta_learner: Optional[str]  # For stacking: "ridge", "elastic_net", "lightgbm_meta"
    gating_method: Optional[str] # For gating: "softmax", "hmm_regime", "markov_switching"
    requires_oof: bool           # True for stacking/blending
    supports_heterogeneous: bool # Can mix tabular + sequence models?
```

**Example Registration:**
```python
@register_ensemble(
    name="ridge_stacking",
    method="stacking",
    meta_learner="ridge",
    requires_oof=True,
    supports_heterogeneous=True  # Can mix any model families
)
class RidgeStackingEnsemble(BaseEnsemble):
    ...
```

---

#### C) Inference Registry (Calibration + Uncertainty)

**Purpose:** Post-processors for probabilities and prediction intervals

**Entry Schema (`InferenceSpec`):**
```python
@dataclass
class InferenceSpec:
    name: str                    # "temperature_scaling", "cqr", "split_conformal"
    category: str                # "calibration", "conformal", "gating"
    inputs: List[str]            # ["p_up", "y_true"] for calibration
    outputs: List[str]           # ["p_up_calibrated", "confidence_interval"]
    requires_holdout: bool       # True for conformal methods
```

**Example Registration:**
```python
@register_inference(
    name="cqr",
    category="conformal",
    inputs=["q05", "q50", "q95", "y_true"],
    outputs=["q05_conf", "q95_conf", "coverage"],
    requires_holdout=True
)
class CQRConformal(BaseInference):
    ...
```

---

#### D) Policy Registry (Optional RL Layer)

**Purpose:** Adaptive decision-making consuming inference outputs

**Entry Schema (`PolicySpec`):**
```python
@dataclass
class PolicySpec:
    name: str                    # "sac", "ppo", "dqn", "contextual_bandit"
    algorithm: str               # "sac", "td3", "ppo", "dqn"
    inputs: List[str]            # ["p_up", "E_r", "q05", "q50", "q95", "uncertainty", "regime_score"]
    outputs: List[str]           # ["position_size", "entry_threshold", "exit_threshold"]
    action_space: str            # "continuous", "discrete"
```

**Example Registration:**
```python
@register_policy(
    name="sac",
    algorithm="sac",
    inputs=["p_up", "E_r", "q05", "q50", "q95", "uncertainty"],
    outputs=["position_size"],
    action_space="continuous"
)
class SACPolicy(BasePolicy):
    ...
```

---

### Registry Implementation Files

```
src/registry/
├── model_registry.py       # ModelRegistry, @register_model decorator
├── ensemble_registry.py    # EnsembleRegistry, @register_ensemble
├── inference_registry.py   # InferenceRegistry, @register_inference
├── policy_registry.py      # PolicyRegistry, @register_policy
└── contracts.py            # ModelSpec, EnsembleSpec, InferenceSpec, PolicySpec
```

---

## Part 3: Full Model Menu (28+ Models)

### Current Models (Keep All 13)

**Boosting (3):**
1. XGBoost - Stable benchmark, SHAP importance
2. LightGBM - Fastest training, leaf-wise growth
3. CatBoost - Robust default, categorical handling

**Neural (4):**
4. LSTM - Long-term dependencies, classic RNN
5. GRU - Faster RNN, good "cheap" baseline
6. TCN - Causal dilations, parallelizable
7. Transformer - Basic self-attention (non-causal, research only)

**Classical (3):**
8. Random Forest - Diversity baseline, bagging
9. Logistic Regression - Fast baseline, meta-learner
10. SVM - Nonlinear kernel (slow, niche use)

**Ensemble (3):**
11. Voting - Simple averaging
12. Stacking - OOF-based meta-learning
13. Blending - Holdout-based meta-learning

---

### Models to Add (15+ New)

#### 1. Tabular / Feature-Matrix (Add 1)

**14. ExtraTrees**
- **Why:** Diversity from Random Forest (extra randomization)
- **Use case:** Ensemble diversity, less overfitting than RF
- **Effort:** 4 hours (similar to RandomForest)

---

#### 2. CNN / Local-Pattern Sequence (Add 3)

**15. InceptionTime**
- **Why:** Multi-kernel ensemble bias, strong classification
- **Architecture:** Inception modules (parallel 3x1, 5x1, 7x1 convs)
- **Effort:** 3 days

**16. 1D ResNet**
- **Why:** Strong baseline backbone, residual learning
- **Architecture:** Residual blocks with 1D convolutions
- **Effort:** 2 days

**17. WaveNet**
- **Why:** Long receptive field, dilated causal convs
- **Architecture:** Stacked dilated conv blocks (similar to TCN but more layers)
- **Effort:** 2 days

---

#### 3. Transformer / Long-Context (Add 3)

**18. PatchTST**
- **Why:** SOTA long-term forecasting, 21% MSE reduction
- **Architecture:** Patch-based attention (16-token patches)
- **Causal:** Yes (production-safe)
- **Effort:** 4 days

**19. iTransformer**
- **Why:** Inverted attention for multivariate (features as tokens)
- **Architecture:** Attention over features instead of time
- **Effort:** 3 days

**20. TFT (Temporal Fusion Transformer)**
- **Why:** Rich covariates + multi-horizon + interpretability
- **Architecture:** Variable selection + gating + multi-head attention
- **Effort:** 5 days

**21. Informer**
- **Why:** Efficient long-seq transformer (ProbSparse attention)
- **Architecture:** Sparse attention for O(L log L) complexity
- **Effort:** 4 days

---

#### 4. Probabilistic Sequence (Add 2)

**22. DeepAR**
- **Why:** Distribution forecasting, calibrated uncertainty
- **Architecture:** Auto-regressive RNN with probabilistic outputs
- **Outputs:** `q05`, `q50`, `q95` (quantiles)
- **Effort:** 4 days

**23. Quantile RNN**
- **Why:** Direct q05/q50/q95 for risk bands
- **Architecture:** LSTM/GRU with quantile loss
- **Effort:** 2 days

---

#### 5. MLP / Linear Baselines (Add 4)

**24. N-BEATS**
- **Why:** Interpretable decomposition (trend + seasonal), M4 winner
- **Architecture:** Stacked blocks with basis expansion
- **Effort:** 1 day

**25. N-HiTS**
- **Why:** Hierarchical N-BEATS, 2x faster
- **Architecture:** Multi-rate inputs (short/medium/long-term)
- **Effort:** 1 day

**26. TSMixer**
- **Why:** MLP mixing over time/features, good diversity
- **Architecture:** Time-mixing + feature-mixing MLPs
- **Effort:** 2 days

**27. DLinear**
- **Why:** Ultra-fast sanity gate, trend/seasonality baseline
- **Architecture:** Decomposition + two linear layers
- **Effort:** 4 hours

---

#### 6. Foundation Models (Add 2)

**28. Chronos-Bolt**
- **Why:** 51%+ directional accuracy (zero-shot)
- **Architecture:** Pre-trained transformer (Amazon, 200M params)
- **Zero-shot:** Yes (no training required)
- **Effort:** 3 days (API wrapper)

**29. TimesFM 2.5**
- **Why:** Probabilistic forecasts (quantiles), Google foundation model
- **Architecture:** Decoder-only transformer (200M params)
- **Outputs:** `q05`, `q50`, `q95`
- **Effort:** 3 days (API wrapper)

---

### Final Model Suite: 29 Models Across 6 Families

| Family | Models | Count | Use Case |
|--------|--------|-------|----------|
| **Boosting** | XGBoost, LightGBM, CatBoost | 3 | Fast tabular baselines |
| **Neural (RNN)** | LSTM, GRU | 2 | Temporal dependencies, probabilistic |
| **Neural (CNN)** | TCN, InceptionTime, 1D ResNet, WaveNet | 4 | Local patterns, causal convolutions |
| **Classical** | Random Forest, ExtraTrees, Logistic, SVM | 4 | Diversity baselines, meta-learners |
| **Transformers** | Transformer, PatchTST, iTransformer, TFT, Informer | 5 | Long-term, multivariate, interpretable |
| **Probabilistic** | DeepAR, Quantile RNN | 2 | Distribution forecasting, quantiles |
| **MLP/Linear** | N-BEATS, N-HiTS, TSMixer, DLinear | 4 | Fast baselines, decomposition |
| **Foundation** | Chronos, TimesFM | 2 | Zero-shot baselines |
| **Ensemble** | Voting, Stacking, Blending | 3 | Model combination |

**Total: 29 models** (13 current + 16 new)

---

## Part 4: Contracts & Artifacts

### Input View Contract

Every model declares its input format:

| View | Shape | Models |
|------|-------|--------|
| `feature_matrix` | `(n_samples, n_features)` | XGBoost, LightGBM, CatBoost, RF, ExtraTrees, Logistic, SVM |
| `window_tensor` | `(n_samples, seq_len, n_features)` | LSTM, GRU, TCN, InceptionTime, ResNet, WaveNet, PatchTST, iTransformer, TFT, Informer, DeepAR, Quantile RNN, N-BEATS, N-HiTS, TSMixer, DLinear, Chronos, TimesFM |
| `mtf_bundle` | Multi-timeframe aligned set | (future extension) |

### Output Contract

Every model maps to standardized outputs:

| Output | Type | Description | Example Models |
|--------|------|-------------|----------------|
| `p_up` | `float [0,1]` | Probability of up move | All classification models |
| `E[r]` | `float` | Expected return | Regression models |
| `q05` | `float` | 5th percentile (quantile) | DeepAR, Quantile RNN, TimesFM |
| `q50` | `float` | Median (50th percentile) | DeepAR, Quantile RNN, TimesFM |
| `q95` | `float` | 95th percentile (quantile) | DeepAR, Quantile RNN, TimesFM |
| `regime_score` | `float [0,1]` | Regime confidence | (future: regime-aware models) |

### Artifact Contract

Every trained model emits:

```python
{
    "preproc": "path/to/scaler.pkl",         # Preprocessing pipeline
    "schema": "path/to/feature_schema.json", # Feature names + types
    "weights": "path/to/model.pth",          # Model weights
    "metrics": "path/to/metrics.json",       # Train/val/test performance
    "inference_signature": {                 # API contract
        "inputs": ["feature_matrix"],
        "outputs": ["p_up", "E_r"],
        "version": "v1.2.3"
    }
}
```

For ensembles, add:
```python
{
    "oof_preds": "path/to/oof_predictions.parquet",  # Out-of-fold base model predictions
    "meta_model": "path/to/meta_learner.pkl",        # Ridge/Elastic Net/LightGBM meta
    "calibrator": "path/to/calibrator.pkl",          # Temperature scaling/Isotonic
    "gating_model": "path/to/gate.pkl"               # Optional: HMM/Softmax gate
}
```

---

## Part 5: Critical Data Pipeline Bugs

*(Same as before - 3 bugs to fix)*

### 🔴 Bug 1: HMM Lookahead Bias

**File:** `src/phase1/stages/regime/hmm.py:329-354`

**Problem:** HMM trains on entire dataset including future data when `expanding=True`.

**Fix:**
```python
expanding = kwargs.get("expanding", False)  # Change default to False
```

**Estimated Effort:** 2 days

---

### 🔴 Bug 2: GA Test Data Leakage

**File:** `src/phase1/stages/ga_optimize/optuna_optimizer.py`

**Problem:** Optuna optimization uses full dataset before train/val/test splits.

**Fix:**
```python
train_end_idx = int(0.7 * len(df))
train_df = df.iloc[:train_end_idx]
study.optimize(objective, train_df)  # Only train portion
```

**Estimated Effort:** 2 days

---

### 🔴 Bug 3: No Transaction Costs in Labels

**File:** `src/phase1/stages/labeling/triple_barrier.py`

**Problem:** Triple-barrier labels assume zero transaction costs.

**Fix:**
```python
cost_in_atr = (cost_ticks * tick_value) / atr
upper_barrier = entry_price + (k_up - cost_in_atr) * atr
lower_barrier = entry_price - (k_down + cost_in_atr) * atr
```

**Estimated Effort:** 2 days

---

## Part 6: Implementation Roadmap

### Phase 1: Fix Data Pipeline Bugs (Week 1-2)

**Goal:** Zero-leakage pipeline

**Tasks:**
1. Fix HMM lookahead bias (2 days)
2. Fix GA test data leakage (2 days)
3. Fix transaction costs in labels (2 days)
4. Regression test suite (2 days)

**Success Criteria:**
- ✅ All 3 bugs fixed
- ✅ All 2,060 tests pass

**Estimated Effort:** 1-2 weeks
**Priority:** **CRITICAL**

---

### Phase 2: Build Dynamic Registry System (Week 3-4)

**Goal:** Config-driven model selection

**Tasks:**
1. Implement ModelRegistry + ModelSpec (3 days)
2. Implement EnsembleRegistry + EnsembleSpec (2 days)
3. Implement InferenceRegistry + InferenceSpec (2 days)
4. Implement PolicyRegistry + PolicySpec (2 days)
5. Update all 13 existing models to use @register_model (1 day)
6. Add YAML config loading (1 day)

**New Files:**
```
src/registry/
├── model_registry.py
├── ensemble_registry.py
├── inference_registry.py
├── policy_registry.py
└── contracts.py
```

**Success Criteria:**
- ✅ All 13 models registered
- ✅ Config-driven training: `python scripts/train_model.py --config config.yaml`
- ✅ Registry tests pass

**Estimated Effort:** 2 weeks

---

### Phase 3: Add 16 New Models (Week 5-10)

**Goal:** 29 total models across 6 families

#### Week 5-6: Foundation + MLP Baselines (Quick Wins)
1. Chronos-Bolt (3 days)
2. TimesFM (3 days)
3. N-BEATS (1 day)
4. N-HiTS (1 day)
5. DLinear (4 hours)
6. TSMixer (2 days)

**Output:** 6 new models (total: 19)

#### Week 7-8: Advanced Transformers
7. PatchTST (4 days)
8. iTransformer (3 days)
9. TFT (5 days)
10. Informer (4 days)

**Output:** 4 new models (total: 23)

#### Week 9-10: CNN + Probabilistic
11. InceptionTime (3 days)
12. 1D ResNet (2 days)
13. WaveNet (2 days)
14. DeepAR (4 days)
15. Quantile RNN (2 days)
16. ExtraTrees (4 hours)

**Output:** 6 new models (total: 29)

**Success Criteria:**
- ✅ 16 new models registered
- ✅ All trainable via config
- ✅ Benchmarks documented

**Estimated Effort:** 6 weeks (3 weeks with 2 engineers)

---

### Phase 4: Add Inference Layer (Week 11-12)

**Goal:** Calibration + conformal prediction + gating

#### Calibration Methods (4)
1. Temperature Scaling (1 day)
2. Isotonic Regression (1 day)
3. Platt Scaling (1 day)
4. Beta Calibration (1 day)

#### Conformal Prediction Methods (3)
5. CQR (Conformalized Quantile Regression) (2 days)
6. Split Conformal Prediction (2 days)
7. CV+/Jackknife+ Conformal (2 days)

#### Gating Methods (4)
8. Softmax Gating Network (2 days)
9. HMM Regime Gating (3 days)
10. Markov Switching (3 days)
11. Contextual Bandit (2 days)

#### Meta-Learners (3 new, beyond Ridge/Logistic)
12. Elastic Net meta-learner (1 day)
13. LightGBM meta-learner (1 day)
14. CatBoost meta-learner (1 day)
15. Small MLP meta-learner (2 days)

**Success Criteria:**
- ✅ 15 inference components registered
- ✅ Config-driven: `inference: {calibrate: [temperature_scaling], uncertainty: cqr, gating: hmm_regime}`

**Estimated Effort:** 2 weeks

---

### Phase 5: Add Optional RL Policy Layer (Week 13-14)

**Goal:** Adaptive decision-making

**RL Algorithms (4):**
1. SAC (Soft Actor-Critic) (4 days)
2. TD3 (Twin Delayed DDPG) (3 days)
3. PPO (Proximal Policy Optimization) (4 days)
4. DQN (Deep Q-Network) (3 days)
5. Contextual Bandits (2 days)

**Integration:**
- Consume inference outputs: `p_up`, `E[r]`, `q05/q50/q95`, `uncertainty`, `regime_score`
- Output: `position_size`, `entry_threshold`, `exit_threshold`
- Config-driven: `policy: {enabled: true, algorithm: sac}`

**Success Criteria:**
- ✅ 5 RL policies registered
- ✅ Optional via config (default: `enabled: false`)

**Estimated Effort:** 2 weeks

---

### Phase 6: Production Infrastructure (Week 15-16)

**Goal:** Automated train→bundle→deploy

**Tasks:**
1. Auto-bundle generation (3 days)
2. Test set evaluation script (2 days)
3. Drift detection daemon (4 days)
4. CI/CD pipeline (3 days)

**Success Criteria:**
- ✅ One-command: `python scripts/train_model.py --config config.yaml --create-bundle`
- ✅ Drift monitoring running

**Estimated Effort:** 2 weeks

---

## Part 7: Quick Wins

**High-impact changes with minimal effort:**

1. **Fix HMM expanding mode (1 hour)** - Immediate leakage fix
2. **Add DLinear (4 hours)** - Fastest SOTA baseline
3. **Add N-BEATS (1 day)** - Interpretable decomposition
4. **Add Temperature Scaling (1 day)** - Calibrated probabilities
5. **Implement ModelRegistry (3 days)** - Unlocks config-driven selection

---

## Part 8: Risk Analysis

### High-Risk Items

1. **Registry complexity** - 4 registries may introduce coupling
   - **Mitigation:** Strict interfaces, unit tests for each registry

2. **Config explosion** - Too many options may confuse users
   - **Mitigation:** Provide preset configs (e.g., `config/presets/fast_baseline.yaml`)

3. **Heterogeneous ensemble bugs** - Mixing tabular + sequence models
   - **Mitigation:** Compatibility validation in EnsembleRegistry

4. **RL instability** - RL policies may diverge during training
   - **Mitigation:** Make RL optional (default: `enabled: false`), provide stable baselines

---

## Part 9: Testing Strategy

### Phase 1: Bug Fix Tests
- [ ] HMM: `expanding=False` uses only past data
- [ ] GA: Optimization confined to train set
- [ ] Transaction costs: Barrier adjustment math validated

### Phase 2: Registry Tests
- [ ] ModelRegistry: Create models from config
- [ ] EnsembleRegistry: Validate compatibility checks
- [ ] InferenceRegistry: Calibration correctness
- [ ] PolicyRegistry: RL policy creation

### Phase 3: New Model Tests
- [ ] Each new model: `test_fit()`, `test_predict()`, `test_save_load()`
- [ ] PatchTST: Causal masking verified
- [ ] Foundation models: Zero-shot prediction (no training)

### Phase 4: Inference Tests
- [ ] Calibration: Reliability diagrams match
- [ ] Conformal: Coverage matches target (e.g., 90%)
- [ ] Gating: Regime weights sum to 1.0

### Phase 5: RL Tests
- [ ] SAC/TD3/PPO: Policy converges on simple envs
- [ ] Integration: RL consumes inference outputs correctly

---

## Success Metrics

| Phase | Metric | Target |
|-------|--------|--------|
| Phase 1 | Zero leakage bugs | All 3 bugs fixed |
| Phase 2 | Registry coverage | All 13 models registered |
| Phase 3 | Model count | 29 models |
| Phase 4 | Inference components | 15 calibration/conformal/gating methods |
| Phase 5 | RL policies | 5 algorithms (optional) |
| Phase 6 | Deployment time | <5 min train→bundle→deploy |

**Final System Capabilities:**

- ✅ 29+ models across 6 families
- ✅ Config-driven selection (no code changes)
- ✅ 15+ inference components (calibration, conformal, gating)
- ✅ Optional RL policy layer (5 algorithms)
- ✅ One-command deploy
- ✅ Automated drift detection

---

## Appendix A: Example Configs

### Config 1: Fast Baseline (Boosting Only)

```yaml
base_models:
  - {name: lightgbm, view: feature_matrix, outputs: [p_up]}

ensemble:
  enabled: false

inference:
  calibrate: [temperature_scaling]

policy:
  enabled: false
```

### Config 2: Heterogeneous Ensemble (Tabular + Sequence + Foundation)

```yaml
base_models:
  - {name: lightgbm, view: feature_matrix, outputs: [p_up]}
  - {name: tcn, view: window_tensor, outputs: [E_r]}
  - {name: patchtst, view: window_tensor, outputs: [E_r, q05, q50, q95]}
  - {name: chronos, view: window_tensor, outputs: [p_up], zero_shot: true}

ensemble:
  method: stacking
  meta_learner: ridge
  oof: true
  gating:
    enabled: true
    method: hmm_regime

inference:
  calibrate: [temperature_scaling, isotonic]
  uncertainty:
    method: cqr
    coverage: 0.90

policy:
  enabled: false
```

### Config 3: Full Stack (Ensemble + Calibration + RL)

```yaml
base_models:
  - {name: lightgbm, view: feature_matrix, outputs: [p_up]}
  - {name: catboost, view: feature_matrix, outputs: [p_up]}
  - {name: patchtst, view: window_tensor, outputs: [E_r, q05, q50, q95]}
  - {name: deepar, view: window_tensor, outputs: [q05, q50, q95]}

ensemble:
  method: stacking
  meta_learner: lightgbm_meta
  oof: true
  gating:
    enabled: true
    method: markov_switching

inference:
  calibrate: [beta_calibration]
  uncertainty:
    method: jackknife_plus
    coverage: 0.95

policy:
  enabled: true
  algorithm: sac
  inputs: [p_up, E_r, q05, q50, q95, uncertainty, regime_score]
  outputs: [position_size, entry_threshold, exit_threshold]
```

---

## Appendix B: Model Implementation Checklist

For each new model:

- [ ] Implement `BaseModel` interface (`fit`, `predict`, `save`, `load`)
- [ ] Register with `@register_model(name=..., family=..., input_view=..., outputs=...)`
- [ ] Add config file: `config/models/<model_name>.yaml`
- [ ] Write tests: `tests/models/test_<model_name>.py`
- [ ] Verify `is_causal` property (production safety)
- [ ] Add to model count assertion
- [ ] Document in `CLAUDE.md`
- [ ] Run benchmark vs baselines
- [ ] Add example usage to `docs/`

**Target:** 16 new models × 9 checklist items = 144 tasks
