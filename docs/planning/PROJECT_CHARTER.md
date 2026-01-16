# ML Trading Model Factory - Project Charter

**Version:** 2.0 (Accurate Implementation Status)
**Last Updated:** 2026-01-15
**Purpose:** Production ML system for futures trading signal generation
**Status:** **PRODUCTION-READY** (23 models deployed, MTF complete, 9 TFs)

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
- ✅ **MTF complete** (9 intraday timeframes: 1m, 5m, 10m, 15m, 20m, 25m, 30m, 45m, 1h)

### Phase 2: Models
- ✅ **23 MODELS IMPLEMENTED** (see inventory below)
- ✅ Boosting: XGBoost, LightGBM, CatBoost
- ✅ Neural: LSTM, GRU, TCN, Transformer, PatchTST, iTransformer, TFT, N-BEATS, InceptionTime, ResNet1D
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
- ✅ Heterogeneous stacking support (mixed tabular + sequence)
- ✅ 4 meta-learners: Ridge, MLP, Calibrated, XGBoost

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
[ 23 Models Train ]
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

## Model Inventory (23 Implemented)

### Boosting (3 models) - Tabular 2D

| # | Model | Use Case | Training | Status |
|---|-------|----------|----------|--------|
| 1 | **XGBoost** | Stable benchmark, SHAP interpretability | 2-5 min (CPU/GPU) | ✅ |
| 2 | **LightGBM** | Fastest training, lowest memory | 1-3 min (CPU/GPU) | ✅ |
| 3 | **CatBoost** | Categorical features, ordered boosting | 3-7 min (CPU/GPU) | ✅ (optional) |

### Classical (3 models) - Tabular 2D

| # | Model | Use Case | Training | Status |
|---|-------|----------|----------|--------|
| 4 | **Random Forest** | Robust baseline, feature importance | 2-5 min (CPU) | ✅ |
| 5 | **Logistic Regression** | Fast baseline, meta-learner | 10-30s (CPU) | ✅ |
| 6 | **SVM** | Non-linear boundaries (RBF kernel) | 5-15 min (CPU) | ✅ |

### Neural Sequence (10 models) - 3D/4D

| # | Model | Use Case | Training | Status |
|---|-------|----------|----------|--------|
| 7 | **LSTM** | Long-term dependencies | 20-40 min (GPU) | ✅ |
| 8 | **GRU** | Faster than LSTM, simpler | 15-30 min (GPU) | ✅ |
| 9 | **TCN** | Causal dilations, parallelizable | 25-45 min (GPU) | ✅ |
| 10 | **Transformer** | Self-attention for patterns | 30-60 min (GPU) | ✅ |
| 11 | **PatchTST** | SOTA long-term forecasting | 30-45 min (GPU) | ✅ |
| 12 | **iTransformer** | Multivariate correlations | 30-45 min (GPU) | ✅ |
| 13 | **TFT** | Interpretable, variable selection | 40-60 min (GPU) | ✅ |
| 14 | **N-BEATS** | Trend + seasonal decomposition | 25-40 min (GPU) | ✅ |
| 15 | **InceptionTime** | Multi-scale kernels | 30-50 min (GPU) | ✅ |
| 16 | **ResNet1D** | Deep residual learning | 25-40 min (GPU) | ✅ |

### Ensemble (3 models)

| # | Model | Use Case | Training | Status |
|---|-------|----------|----------|--------|
| 17 | **Voting** | Simple weighted averaging | Sum of bases | ✅ |
| 18 | **Stacking** | OOF-based meta-learning | Sum + 5 min | ✅ |
| 19 | **Blending** | Holdout-based meta-learning | Sum + 3 min | ✅ |

### Meta-Learners (4 models)

| # | Model | Use Case | Training | Status |
|---|-------|----------|----------|--------|
| 20 | **Ridge Meta** | L2-regularized linear stacking | <1 min | ✅ |
| 21 | **MLP Meta** | Non-linear learned blending | 1-2 min | ✅ |
| 22 | **Calibrated Meta** | Probability calibration | <1 min | ✅ |
| 23 | **XGBoost Meta** | Non-linear feature interactions | 1-2 min | ✅ |

**Note:** CatBoost has conditional registration. If unavailable, total = 22 models.

---

## ✅ Implemented Advanced Models (Now Part of 23 Total)

### Advanced Transformers (3 models) - ✅ COMPLETE
- **PatchTST:** SOTA long-term forecasting (patch-based attention)
- **iTransformer:** Multivariate correlations (features as tokens)
- **TFT (Temporal Fusion Transformer):** Interpretable + variable selection

### CNN Models (2 models) - ✅ COMPLETE
- **InceptionTime:** Multi-scale kernels (10, 20, 40)
- **ResNet1D:** Residual learning for deep networks

### N-BEATS (1 model) - ✅ COMPLETE
- **N-BEATS:** Interpretable decomposition (trend + seasonal)

## Potential Future Models (Not Prioritized)

### Foundation Models (Zero-Shot)
- **Chronos-Bolt:** Zero-shot pre-trained transformer (Amazon)
- **TimesFM 2.5:** Zero-shot probabilistic forecasts (Google)

### Other Candidates
- **N-HiTS:** Hierarchical N-BEATS (2x faster)
- **DLinear:** Ultra-fast linear baseline
- **DeepAR:** Distribution forecasting
- **TimesNet:** Multi-periodic pattern extraction

**Status:** Current 23 models are sufficient for production. Foundation models may be added for zero-shot baseline comparisons.

---

## Multi-Timeframe (MTF) Architecture

### Current Implementation

**Ingestion:** Always 1-minute OHLCV bars
**Training Timeframe:** Configurable (1m, 5m, 10m, 15m, 30m, 45m, 1h)
**MTF Timeframes:** 1m, 5m, 10m, 15m, 30m, 45m, 1h, 4h, daily

**Status:**
- ✅ 9 intraday timeframes implemented (1m, 5m, 10m, 15m, 20m, 25m, 30m, 45m, 1h)
- ⚠️ Strategy 3 (multi-resolution ingestion) planned for future

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
- **PIPELINE_STAGES.md** - Visual pipeline flow
- **REPO_ORGANIZATION_ANALYSIS.md** - Discrepancy analysis and reorganization plan

### Implementation Guides

- **MODEL_INTEGRATION.md** - How to add new models
- **FEATURE_ENGINEERING.md** - Feature strategies per model family
- **HYPERPARAMETER_TUNING.md** - GA and Optuna tuning
- **reference/INFRASTRUCTURE.md** - Hardware/GPU requirements

### Detailed Docs

- **docs/implementation/PHASE_1_INGESTION.md** - Data pipeline details
- **docs/implementation/PHASE_6_TRAINING.md** - Model training details
- **docs/implementation/PHASE_3_FEATURES.md** - Feature engineering details
- **docs/ARCHITECTURE.md** - System architecture
- **docs/guides/FEATURE_ENGINEERING.md** - Feature catalog

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
- **v3.0** (2026-01-13): Updated to 23 models, MTF complete (9 intraday timeframes)
- **v3.1** (2026-01-15): Comprehensive update - 23 models verified, all advanced neural models implemented, heterogeneous stacking complete
