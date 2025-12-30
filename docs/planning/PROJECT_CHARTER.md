# ML Trading Model Factory - Project Charter

**Version:** 2.0 (Accurate Implementation Status)
**Last Updated:** 2025-12-30
**Purpose:** Production ML system for futures trading signal generation
**Status:** **PRODUCTION-READY** (13 models deployed, MTF in progress)

---

## Vision

Build a **production-grade, model-agnostic ML factory** for systematic futures trading that trains, evaluates, and deploys models with:

- ✅ **Zero leakage** (purge/embargo enforced)
- ✅ **Deterministic outputs** (same data + seed = same results)
- ✅ **Fair model comparison** (identical experimental controls)
- ✅ **Research → production parity** (same pipeline for train and serve)

**This is NOT research - this is a production deployment system for live trading with real capital.**

---

## Current Implementation Status

### Phase 1: Data Pipeline
- ✅ **COMPLETE** - 14 stages fully implemented
- ✅ 150+ technical indicators
- ✅ Triple-barrier labeling with GA optimization
- ✅ Purge/embargo for leakage prevention
- ✅ Train/val/test splits (70/15/15)
- ⚠️ **MTF partially implemented** (missing 20min/25min timeframes, Strategy 3)

### Phase 2: Models
- ✅ **13 MODELS IMPLEMENTED** (not 19 - see below)
- ✅ Boosting: XGBoost, LightGBM, CatBoost
- ✅ Neural: LSTM, GRU, TCN, Transformer
- ✅ Classical: Random Forest, Logistic, SVM
- ✅ Ensemble: Voting, Stacking, Blending

### Phase 3: Cross-Validation
- ✅ **COMPLETE** - Time-series aware CV
- ✅ PurgedKFold with purge/embargo
- ✅ OOF prediction generation
- ✅ Walk-forward validation
- ✅ CPCV/PBO for overfitting detection

### Phase 4: Ensemble
- ✅ **COMPLETE** - OOF-based stacking
- ✅ Voting ensembles
- ✅ Compatibility validation (same-family only)

### Phase 5: Inference
- ✅ **COMPLETE** - Production serving
- ✅ Feature pipeline for inference
- ✅ FastAPI server
- ✅ Batch inference support

---

## Architecture Principles

### 1. Single-Contract Isolation

**One contract at a time.** No cross-symbol correlation or features.

- Each symbol (MES, MGC, ES, GC) trains separately
- Complete isolation prevents cross-contamination
- Easy to switch symbols via config

```bash
./pipeline run --symbols MES   # Train on MES
./pipeline run --symbols MGC   # Train on MGC (separate model)
```

### 2. Factory Pattern

**One data source → many model backends** with unified evaluation.

```
Raw 1min OHLCV
    ↓
[ Phase 1: 14-Stage Pipeline ]
    ├── Clean & resample
    ├── Features (150+ indicators)
    ├── MTF (multi-timeframe)
    ├── Labeling (triple-barrier)
    ├── Optimize (GA)
    ├── Splits (purge/embargo)
    ├── Scaling (train-only)
    └── Datasets (TimeSeriesDataContainer)
    ↓
[ Model Registry ]
    ├── get_tabular_data() → XGBoost, LightGBM, CatBoost, RF, Logistic, SVM
    ├── get_sequence_data() → LSTM, GRU, TCN, Transformer
    └── (future) get_multi_resolution() → Advanced transformers
    ↓
[ 13 Models Train ]
    ├── Identical data
    ├── Identical splits
    ├── Identical metrics
    └── Fair comparison
    ↓
[ Unified Evaluation ]
    ├── Sharpe ratio
    ├── Win rate
    ├── Max drawdown
    └── Regime-aware performance
```

### 3. Inference-First Design

Training and serving share **identical pipelines:**

- Same feature engineering
- Same resampling logic
- Same scaling (train-fitted)
- Same data transformations

**Output contract:**
```python
{
    "signal": +1,              # -1 (short), 0 (neutral), +1 (long)
    "probabilities": [0.1, 0.2, 0.7],  # [p_short, p_neutral, p_long]
    "confidence": 0.70,        # max(probabilities)
    "expected_return": 0.0043  # E[r]
}
```

### 4. Leakage Paranoia

**Every step prevents lookahead bias:**

- ✅ Chronological splits (no shuffling)
- ✅ Purge bars (60) between splits
- ✅ Embargo bars (1440 = ~5 days)
- ✅ Train-only scaling (RobustScaler fit on train only)
- ✅ Forward-fill for MTF alignment (shift + ffill)
- ✅ PurgedKFold for CV

**If we can't prove it's leakage-free, it doesn't ship.**

---

## Model Inventory (13 Implemented)

### Boosting (3 models)

1. **XGBoost**
   - Use case: Stable benchmark, SHAP interpretability
   - Input: 2D (n_samples, 150 features)
   - Training: 2-5 min (CPU)
   - Inference: <1ms
   - Status: ✅ Production-ready

2. **LightGBM**
   - Use case: Fastest training, lowest memory
   - Input: 2D (n_samples, 150 features)
   - Training: 1-3 min (CPU)
   - Inference: <1ms
   - Status: ✅ Production-ready

3. **CatBoost**
   - Use case: Handles categorical features, robust to overfitting
   - Input: 2D (n_samples, 150 features)
   - Training: 3-7 min (CPU)
   - Inference: <1ms
   - Status: ✅ Production-ready

### Neural Sequence (4 models)

4. **LSTM**
   - Use case: Long-term dependencies, recurrent baseline
   - Input: 3D (n_samples, 60 timesteps, 25 features)
   - Training: 20-40 min (GPU)
   - Inference: 5-10ms (GPU)
   - Status: ✅ Production-ready

5. **GRU**
   - Use case: Faster than LSTM, simpler gating
   - Input: 3D (n_samples, 60 timesteps, 25 features)
   - Training: 15-30 min (GPU)
   - Inference: 5-10ms (GPU)
   - Status: ✅ Production-ready

6. **TCN (Temporal Convolutional Network)**
   - Use case: Causal dilations, parallelizable
   - Input: 3D (n_samples, 60 timesteps, 25 features)
   - Training: 25-45 min (GPU)
   - Inference: 5-10ms (GPU)
   - Status: ✅ Production-ready

7. **Transformer (basic)**
   - Use case: Self-attention for temporal patterns
   - Input: 3D (n_samples, 60 timesteps, 25 features)
   - Training: 30-60 min (GPU)
   - Inference: 10-20ms (GPU)
   - Status: ✅ Production-ready
   - Note: Causal masks prevent lookahead

### Classical (3 models)

8. **Random Forest**
   - Use case: Robust baseline, feature importance
   - Input: 2D (n_samples, 150 features)
   - Training: 2-5 min (CPU)
   - Inference: <1ms
   - Status: ✅ Production-ready

9. **Logistic Regression**
   - Use case: Fast baseline, meta-learner for stacking
   - Input: 2D (n_samples, 150 features)
   - Training: 10-30s (CPU)
   - Inference: <0.5ms
   - Status: ✅ Production-ready

10. **SVM (Support Vector Machine)**
    - Use case: Non-linear decision boundaries (RBF kernel)
    - Input: 2D (n_samples, 150 features)
    - Training: 5-15 min (CPU)
    - Inference: <1ms
    - Status: ✅ Production-ready

### Ensemble (3 models)

11. **Voting Ensemble**
    - Use case: Simple weighted averaging, fast
    - Input: Base model predictions
    - Training: Sum of base models
    - Inference: Sum of base latencies + <1ms
    - Status: ✅ Production-ready

12. **Stacking Ensemble**
    - Use case: OOF-based meta-learning, best performance
    - Architecture: PurgedKFold OOF + Logistic/LightGBM meta-learner
    - Input: OOF predictions + optional regime features
    - Training: Sum of base models + 5 min (meta)
    - Status: ✅ Production-ready

13. **Blending Ensemble**
    - Use case: Holdout-based meta-learning
    - Input: Holdout predictions
    - Training: Sum of base models + 3 min (meta)
    - Status: ✅ Production-ready

---

## Planned Future Models (Not Yet Implemented)

### Advanced Transformers (3 models)

- **PatchTST:** SOTA long-term forecasting (patch-based attention)
- **iTransformer:** Multivariate correlations (features as tokens)
- **TFT (Temporal Fusion Transformer):** Interpretable + variable selection

### CNN Models (2 models)

- **InceptionTime:** Multi-scale kernels (3x1, 5x1, 7x1)
- **1D ResNet:** Residual learning for deep networks

### MLP Baselines (3 models)

- **N-BEATS:** Interpretable decomposition (trend + seasonal)
- **N-HiTS:** Hierarchical N-BEATS (2x faster)
- **DLinear:** Ultra-fast linear baseline

### Foundation Models (2 models)

- **Chronos-Bolt:** Zero-shot pre-trained transformer (Amazon, 200M params)
- **TimesFM 2.5:** Zero-shot probabilistic forecasts (Google, 200M params)

### Probabilistic (2 models)

- **DeepAR:** Distribution forecasting
- **Quantile RNN:** Direct quantile predictions (q05, q50, q95)

**Status:** Not prioritized. Current 13 models are sufficient for production.

---

## Multi-Timeframe (MTF) Architecture

### Current Implementation

**Ingestion:** Always 1-minute OHLCV bars
**Training Timeframe:** Configurable (1m, 5m, 10m, 15m, 30m, 45m, 1h)
**MTF Timeframes:** 1m, 5m, 10m, 15m, 30m, 45m, 1h, 4h, daily

**Gaps:**
- ⚠️ Missing 20min and 25min (needed for 9-TF ladder)
- ⚠️ 4h and daily should be deprecated (not in 9-TF ladder)
- ⚠️ Strategy 3 (multi-resolution ingestion) not implemented

### Three MTF Strategies (Design)

**Strategy 1: Single-Timeframe**
- Train on one timeframe (e.g., 15m)
- NO MTF features
- Use case: Baselines, simple models
- Status: ⚠️ Partially implemented (needs mtf_strategy config)

**Strategy 2: MTF Indicators**
- Train on one timeframe (e.g., 15m)
- Add features from other timeframes (1m, 5m, 30m, 1h)
- Use case: Tabular models (XGBoost, LightGBM)
- Status: ✅ Implemented (default behavior)

**Strategy 3: MTF Ingestion**
- Train on one timeframe (e.g., 15m)
- Feed multiple timeframe tensors together (1m + 5m + 15m + 1h OHLCV)
- Use case: Advanced transformers (PatchTST, TFT)
- Status: ❌ Not implemented (needs multi_resolution.py)

---

## Data Pipeline (Phase 1)

### 14 Stages

1. **Ingest** - Load and validate raw OHLCV
2. **Clean** - Resample 1m→training_timeframe, handle gaps
3. **Sessions** - Filter trading sessions (RTH/ETH)
4. **Features** - Compute 150+ technical indicators
5. **Regime** - Detect market regimes (HMM, volatility, trend)
6. **MTF** - Multi-timeframe feature generation
7. **Labeling** - Triple-barrier method (initial labels)
8. **GA Optimize** - Optuna-based barrier optimization
9. **Final Labels** - Apply optimized parameters
10. **Splits** - Train/val/test with purge/embargo
11. **Scaling** - RobustScaler (train-only fit)
12. **Datasets** - Build TimeSeriesDataContainer
13. **Validation** - Feature correlation, drift detection
14. **Reporting** - Generate pipeline report

**Output:** `TimeSeriesDataContainer` with train/val/test splits

---

## Key Parameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| **Symbol** | MES, MGC, ES, GC | One per run |
| **Ingest Timeframe** | 1 minute | Always 1min raw data |
| **Training Timeframe** | 5 min (default) | Configurable: 1m, 5m, 10m, 15m, 30m, 45m, 1h |
| **Label Horizons** | 5, 10, 15, 20 bars | Forward-looking prediction windows |
| **Train / Val / Test** | 70% / 15% / 15% | Chronological splits |
| **Purge Bars** | 60 (= max_horizon × 3) | Prevents label overlap |
| **Embargo Bars** | 1440 (~5 days at 5min) | Prevents serial correlation |
| **Sequence Length** | 60 bars | For LSTM/GRU/TCN/Transformer |
| **Features** | 150+ | MTF indicators + wavelets + microstructure |
| **Classes** | 3 | -1 (SHORT), 0 (HOLD), +1 (LONG) |
| **Sample Weights** | 0.5x - 1.5x | Quality-based weighting |

---

## Usage Examples

### Run Complete Pipeline

```bash
# Train on MES (Micro E-mini S&P 500)
./pipeline run --symbols MES

# Train on MGC (Micro Gold)
./pipeline run --symbols MGC
```

### Train Individual Model

```bash
# Boosting model (2D input)
python scripts/train_model.py --model xgboost --horizon 20

# Sequence model (3D input)
python scripts/train_model.py --model lstm --horizon 20 --seq-len 60
```

### Train Ensemble

```bash
# Voting ensemble (same-family models only!)
python scripts/train_model.py --model voting \
    --base-models xgboost,lightgbm,catboost \
    --horizon 20

# Stacking ensemble with OOF
python scripts/train_model.py --model stacking \
    --base-models xgboost,lightgbm,random_forest \
    --horizon 20
```

### Cross-Validation

```bash
# 5-fold PurgedKFold CV
python scripts/run_cv.py --models xgboost --horizons 20 --n-splits 5

# Hyperparameter tuning with Optuna
python scripts/run_cv.py --models xgboost --tune --n-trials 100
```

### Walk-Forward Validation

```bash
python scripts/run_walk_forward.py --model xgboost --horizon 20 --n-windows 10
```

### Check for Overfitting (PBO)

```bash
python scripts/run_cpcv_pbo.py --models xgboost,lightgbm --horizon 20
```

---

## Anti-Patterns Prevented

| Anti-Pattern | How We Prevent It |
|--------------|-------------------|
| **Lookahead Bias** | Purge/embargo in CV, chronological splits, train-only scaling |
| **Data Leakage** | Strict split boundaries, no shuffling, label_end_times purging |
| **Survivorship Bias** | All outcomes labeled (-1, 0, +1), no filtering by outcome |
| **Overfitting to Backtest** | Walk-forward validation, PBO calculation, embargo periods |
| **Regime Blindness** | Regime features, regime-aware evaluation, walk-forward windows |

---

## Performance Expectations

**Do NOT treat any performance targets as built-in or guaranteed.**

- Sharpe ratios are empirical and symbol/period dependent
- Win rates vary by regime and market conditions
- Transaction costs significantly impact net returns
- Out-of-sample performance may differ from validation

**Always validate with:**
- Cross-validation (`run_cv.py`)
- Walk-forward validation (`run_walk_forward.py`)
- CPCV/PBO analysis (`run_cpcv_pbo.py`)

---

## Production Deployment

### Inference Pipeline

1. **Load model bundle** (model + scaler + config)
2. **Receive new OHLCV bar**
3. **Feature engineering** (same as training)
4. **Scale features** (using train-fitted scaler)
5. **Model predict** (probabilities + confidence)
6. **Output signal** (-1, 0, +1)

**Latency:**
- Boosting: <1ms (CPU)
- Neural: 5-10ms (GPU)
- Ensemble: Sum of base latencies + <1ms

### Deployment Options

```python
# Option 1: FastAPI server (included)
python src/inference/server.py

# Option 2: Batch inference
python scripts/batch_inference.py --model-path models/xgboost/

# Option 3: Direct integration
from src.inference.pipeline import InferencePipeline
pipeline = InferencePipeline.load("models/xgboost/")
signal = pipeline.predict(new_bar)
```

---

## Engineering Principles

1. **Modularity:** Small files (<800 lines), clear boundaries
2. **Fail Fast:** Validate inputs at boundaries, explicit error messages
3. **Less Code is Better:** Simple solutions win, avoid premature abstraction
4. **Delete Legacy Code:** If unused, remove it (git history is the archive)
5. **No Exception Swallowing:** Explicit validation, let failures propagate
6. **Clear Tests:** Unit + integration + regression tests
7. **Definition of Done:** Implementation + tests + docs

---

## Next Steps (Roadmap)

### Immediate (Week 1)

1. Add 20min/25min MTF timeframes
2. Add `mtf_strategy` config parameter
3. Implement MTF Strategy 1 (single-timeframe)
4. Clean up root documentation

### Short-Term (Weeks 2-4)

5. Implement MTF Strategy 3 (multi-resolution)
6. Make `training_timeframe` configurable
7. Test coverage audit
8. CI/CD setup (pre-commit, GitHub Actions)

### Medium-Term (Months 2-3)

9. Decide on advanced models (add or defer)
10. Production monitoring dashboard
11. A/B testing framework
12. Model registry service

---

## Documentation

### Quick Reference

- **README.md** - Project overview and quickstart
- **THIS FILE** - Project charter and vision
- **PIPELINE_FLOW.md** - Visual pipeline flow
- **REPO_ORGANIZATION_ANALYSIS.md** - Discrepancy analysis and reorganization plan

### Implementation Guides

- **MODEL_INTEGRATION_GUIDE.md** - How to add new models
- **FEATURE_ENGINEERING_GUIDE.md** - Feature strategies per model family
- **HYPERPARAMETER_OPTIMIZATION_GUIDE.md** - GA and Optuna tuning
- **MODEL_INFRASTRUCTURE_REQUIREMENTS.md** - Hardware/GPU requirements

### Detailed Docs

- **docs/phases/PHASE_1.md** - Data pipeline details
- **docs/phases/PHASE_2.md** - Model training details
- **docs/phases/PHASE_3.md** - Cross-validation details
- **docs/reference/ARCHITECTURE.md** - System architecture
- **docs/reference/FEATURES.md** - Feature catalog

---

## Contact & Contributing

For questions, issues, or contributions:

1. Check existing documentation
2. Review `REPO_ORGANIZATION_ANALYSIS.md` for current status
3. Open GitHub issue for bugs/feature requests
4. Follow contribution guidelines (TBD: CONTRIBUTING.md)

---

**Version History:**
- **v1.0** (2025-12-29): Initial charter (19 models claimed, MTF planned)
- **v2.0** (2025-12-30): Accurate status (13 models implemented, MTF in progress)
