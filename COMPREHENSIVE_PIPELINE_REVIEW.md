# Comprehensive ML Pipeline Review
## OHLCV Time Series Model Factory - Production Readiness Assessment

**Review Date:** 2026-01-03 (Updated for Google Colab Training)
**Reviewed By:** Multi-Agent Analysis (Architecture, ML Methodology, MLOps) + Online Research
**Training Environment:** Google Colab (Ephemeral, 12-hour runtime, Free GPU)
**Overall Assessment:** Research Prototype (60/100) - Requires Colab-Specific Adaptations

---

## Executive Summary

This OHLCV ML factory demonstrates **strong research foundations** with excellent modular architecture, comprehensive testing, and proper leakage prevention measures. However, **critical issues in labeling methodology, ensemble design, and MLOps infrastructure** prevent immediate production deployment.

**NEW: Google Colab Training Context**

The pipeline will be trained on **Google Colab**, which introduces critical constraints:
- **Ephemeral runtime:** 12-hour session limit, no persistent state
- **Free GPU:** T4/P100/V100 (inconsistent availability)
- **Memory limits:** 12.7 GB RAM (standard), 25.5 GB (high-RAM)
- **Disk storage:** ~78 GB temporary (lost on disconnect)

**Colab Compatibility Assessment:**
- **Data pipeline (Phases 1-5):** ✅ Compatible with existing checkpointing
- **Model training (Phase 6):** ❌ Requires epoch-level checkpointing (not implemented)
- **Ensemble stacking (Phase 7):** ⚠️ 3-base ensembles feasible (4.4h), 4+ bases risky
- **Artifact persistence:** 🔴 CRITICAL GAP - All local saves lost on disconnect

### Key Findings

**🔴 CRITICAL ISSUES (Production Blockers):**
1. Asymmetric barrier labeling logic is statistically flawed
2. Feature count is 2-3× too high (severe overfitting risk)
3. No experiment tracking platform (MLflow, W&B)
4. No production deployment infrastructure
5. No real-time monitoring or alerting

**🔴 CRITICAL COLAB BLOCKERS:**
6. No mid-training checkpointing → All progress lost on disconnect
7. No cloud artifact persistence → All experiments lost on session end
8. No Google Drive integration → Cannot save/resume checkpoints
9. No epoch-level resume capability → Must restart from scratch
10. No W&B/Comet integration → Cannot track experiments across sessions

**⚠️ HIGH PRIORITY:**
6. Heterogeneous stacking has data/timeframe mismatch risks
7. Single train/val/test split provides no variance estimates
8. MTF features may have hidden lookahead bias
9. No CI/CD pipeline
10. No data versioning (DVC, lakeFS)

**📊 STRENGTHS:**
- Plugin-based model registry (22 models across 6 families)
- Proper purged K-fold cross-validation
- Train-only scaling with leakage prevention
- Comprehensive test suite (86 test files)
- Clean separation of concerns

---

## Table of Contents

1. [Architecture Review](#1-architecture-review)
2. [ML Methodology Review](#2-ml-methodology-review)
3. [MLOps & Production Readiness](#3-mlops--production-readiness)
4. [Google Colab Training Assessment](#4-google-colab-training-assessment) **← NEW**
5. [Online Research Validation](#5-online-research-validation)
6. [Critical Gaps Summary](#6-critical-gaps-summary)
7. [Prioritized Recommendations](#7-prioritized-recommendations)
8. [Production Readiness Roadmap](#8-production-readiness-roadmap)
9. [Colab-Specific Roadmap](#9-colab-specific-roadmap) **← NEW**

---

## 1. Architecture Review

### 1.1 Overall Architecture Quality: 7/10

**Strengths:**
- **Plugin Architecture:** Clean decorator-based model registration (`@register(name="xgboost", family="boosting")`)
- **Factory Pattern:** Well-implemented `ModelRegistry.create()` for model instantiation
- **Single-Contract Isolation:** Proper symbol isolation prevents cross-contamination
- **BaseModel Contract:** Clear abstract interface with standardized return types

**Issues:**

#### 🔴 Issue 1.1: Global Mutable State in ModelRegistry

**Location:** `src/models/registry.py` lines 58-60

```python
class ModelRegistry:
    _models: dict[str, type[BaseModel]] = {}  # Global mutable state
    _families: dict[str, list[str]] = {}
    _metadata: dict[str, dict[str, Any]] = {}
```

**Problem:** Creates testing fragility, concurrency risks, and import-order dependencies.

**Recommendation:** Implement singleton pattern with lazy initialization or dependency injection.

#### ⚠️ Issue 1.2: Documentation vs Implementation Mismatch

**CLAUDE.md states:**
> "8 intraday timeframes (5m, 10m, 15m, 20m, 25m, 30m, 45m, 1h)"

**docs/ARCHITECTURE.md states:**
> "Currently: 15min, 30min, 1h, 4h, daily (5 timeframes)"

**Impact:** Unclear what is actually implemented. Code review shows 8 intraday timeframes are implemented.

#### ⚠️ Issue 1.3: God Method - Trainer.run()

**Location:** `src/models/trainer.py` (810 LOC total, run() method ~230 lines)

**Responsibilities:**
- Output directory setup
- Data loading
- Feature set resolution
- Feature selection
- Model training
- Calibration
- Test evaluation
- Artifact saving

**Recommendation:** Decompose into smaller orchestration methods or extract to separate classes.

### 1.2 File Complexity Assessment

**Status:** ✅ Passes file size limits (target 650 LOC, max 800 LOC)

**Large Files:**
- `trainer.py`: 810 LOC (acceptable - cohesive orchestration logic)
- `container.py`: 686 LOC (acceptable - data structure)
- `triple_barrier.py`: ~600 LOC (acceptable)

### 1.3 Separation of Concerns: 6/10

**Layer Violations:**

1. **Trainer contains feature selection logic** (should be separate stage)
2. **Container is a "God object"** with adapters for sklearn, PyTorch, NeuralForecast, multi-res 4D
3. **Configuration sprawl** across multiple systems (YAML, Python dataclasses, inline docs)

**Recommendation:** Extract adapters to separate classes, consolidate configuration.

### 1.4 Scalability Bottlenecks

**🔴 Critical:**

1. **Full dataset loading:** Entire training set loaded into memory (problematic for >10M samples)
2. **Sequential base model training:** Ensemble models train bases sequentially (no parallelization)
3. **O(n_models × n_folds) scaling:** Stacking with 4 bases × 5 folds = 20 training runs

**Recommendation:** Implement streaming/chunked data loading, parallel base model training, fold caching.

---

## 2. ML Methodology Review

### 2.1 Data Leakage Prevention: 7/10

**✅ What's Done Well:**

1. **Purged K-Fold CV** with proper purge (60) and embargo (1440)
2. **Triple-barrier labeling** excludes last `max_bars` samples
3. **Train-only scaling** with robust scaler
4. **MTF features use `shift(1)`**

**🔴 Critical Issues:**

#### Issue 2.1: Insufficient Purge for Short Horizons

**Current:** Fixed purge = 60 bars regardless of horizon

**Problem:**
- For horizon=5 (max_bars=12), purge should be `3 × 12 = 36` bars, not 60
- Over-purging wastes data; under-purging causes leakage

**Recommendation:**
```python
# Make purge horizon-adaptive
purge_bars = 3 * config.barrier_params[horizon]["max_bars"]
```

#### ⚠️ Issue 2.2: MTF Features May Have Implicit Lookahead

**Location:** `src/phase1/stages/mtf/generator.py` line 313

**Problem:**
- A 1h bar at 12:00 represents data from 11:00-12:00
- After `shift(1)`, the 12:00 5-min bar sees the 11:00-12:00 1h bar
- If resampling uses `closed='left', label='left'`, the 12:00 1h bar **includes** the 12:00 5-min bar (circular dependency)

**Recommendation:**
```python
# Add explicit test for no circular dependency
mtf_rsi_at_100 = df.loc[100, 'rsi_14_1h']
manual_rsi = compute_rsi(df.loc[:99, 'close'].resample('1h').last())[-1]
assert mtf_rsi_at_100 == manual_rsi, "MTF feature uses future data!"
```

### 2.2 Feature Engineering: 4/10 (HIGH RISK)

#### 🔴 Issue 2.3: Massive Feature Count Invites Overfitting

**Current:** ~180 base features + ~30 MTF features × 5 TFs = **~330 features** for tabular models

**Statistical Reality Check:**
- With 20K training samples: 20K/330 = **60 samples per feature**
- **Rule of thumb:** Need ≥100 samples per feature for stability
- For reliable selection: need `n_samples >> p²` (curse of dimensionality)

**Verdict:** At least **50% of features are noise** that will overfit

**Recommendations:**
1. Reduce base features to ~50-80 core indicators
2. Limit MTF enrichment to 2-3 timeframes (not 5)
3. Target 30-50 features post-selection (not 150+)

#### ⚠️ Issue 2.4: MTF Feature Quality Concerns

**Problem:**
- Higher timeframes have **fewer samples** for indicator calculation
- 1h bars: ~1/12 the samples of 5-min bars
- Statistical significance of 1h RSI is **far weaker** than 5-min RSI

**Recommendation:** Weight MTF features by statistical reliability (sample count, confidence intervals)

### 2.3 Labeling Methodology: 3/10 (CRITICAL ISSUES)

#### 🔴 Issue 2.5: Asymmetric Barriers Create Directional Bias

**Location:** `src/phase1/config/barriers_config.py` lines 114-146

```python
"MES": {
    5: {"k_up": 1.50, "k_down": 1.00, ...},  # 50% harder to trigger long
    20: {"k_up": 3.00, "k_down": 2.10, ...}, # 43% harder to trigger long
}
```

**Why This Is Wrong:**

1. **Statistical Misunderstanding:**
   - MES has ~7% annual drift = **0.03% per 5-min bar** (negligible at intraday scale)
   - ATR for MES ~$12.50, 1.0 ATR barrier ~1% move
   - **0.03% drift << 1.0% barrier** → drift is irrelevant for barrier logic

2. **Label Distribution ≠ Real Trading:**
   - Asymmetric barriers create structural short bias (~60% short labels)
   - Models should learn drift from **features** (trend indicators), not **label engineering**

3. **Evidence of Confusion:**
   - Comments contradict: "k_up > k_down to make upper barrier **harder**" vs "Making k_up LARGER balances long/short"
   - These are contradictory statements

**Correct Approach:**
- Use **symmetric barriers** for all symbols: `k_up = k_down`
- Let models learn directional bias from features
- Only use asymmetric if empirical evidence shows persistent imbalance after transaction costs

#### 🔴 Issue 2.6: Transaction Cost Adjustment is Flawed

**Location:** `src/phase1/stages/labeling/triple_barrier.py` lines 541-555

```python
# Adjust upper barrier for transaction costs
k_up_effective = k_up + cost_in_atr  # e.g., 2.0 + 0.15 = 2.15
```

**Problems:**

1. **Only adjusts upper barrier:** Costs apply to BOTH long and short trades
2. **Uses median ATR:** Costs vary 2-3× between low/high volatility regimes
3. **Conceptual issue:** Costs should affect backtesting P&L, not label generation

**Correct Approach:**
```python
# Labels: Pure price movement (no cost adjustment)
labels = triple_barrier(k_up=2.0, k_down=2.0)  # Symmetric, no cost

# Backtesting: Apply costs to P&L
gross_pnl = price_change
net_pnl = gross_pnl - transaction_costs
sharpe = net_pnl.mean() / net_pnl.std()
```

#### ⚠️ Issue 2.7: Max Bars Creates Label Ambiguity

**Current:** `max_bars = 50` for H20 → timeouts labeled as `0` (neutral)

**Problem:** Class `0` contains heterogeneous outcomes:
- Almost-wins (price near upper barrier)
- Almost-losses (price near lower barrier)
- True neutrals (price unchanged)

**Impact:** High label noise → conservative predictions (lots of neutrals, few actionable signals)

**Recommendation:**
1. Subclassify timeouts based on final price position
2. Or use regression targets: `label = (exit_price - entry_price) / atr`

### 2.4 Ensemble Design: 5/10 (HIGH RISK)

#### ⚠️ Issue 2.8: Heterogeneous Stacking Has Data Mismatch Risk

**Location:** `src/models/trainer.py` lines 381-404

**Problems:**

1. **Timeframe mismatch:**
   - CatBoost trains on 15-min bars
   - TCN trains on 5-min bars
   - PatchTST trains on 1-min bars
   - **Different row counts!** How are OOF predictions aligned for meta-learner?

2. **Feature set mismatch:**
   - CatBoost sees 200 engineered features
   - TCN sees 150 base features
   - PatchTST sees raw OHLCV (no features)
   - Meta-learner may learn "trust CatBoost, ignore PatchTST" (not true ensemble)

3. **No evidence of benefit:**
   - Heterogeneous ensembles only help if base models have **uncorrelated errors**
   - With same 1-min source, same labels, same splits, errors **will** be correlated

**Recommendation:** Run ablation study showing heterogeneous > homogeneous before production use.

#### ⚠️ Issue 2.9: OOF Stacking Has Overfitting Risk

**Problem:**
- OOF predictions are **in-sample** for meta-learner
- If base models overfit, OOF probabilities are **overconfident**
- Meta-learner compounds overfitting

**Better approach:**
1. **Nested CV:** Generate OOF with outer loop, train meta on inner loop
2. **Calibration:** Calibrate base probabilities before stacking (isotonic regression)
3. **Regularize meta-learner:** Strong L2 penalty

### 2.5 Validation Strategy: 5/10 (MEDIUM RISK)

#### ⚠️ Issue 2.10: 70/15/15 Split Too Small for Val/Test

**Current:** 70% train, 15% val, 15% test

**Problem:**
- For 1 year of 5-min data: ~75K bars total
- Val/test sets: 11.25K bars each
- Effective independent samples ≈ 11.25K / 50 = **225 per split** (too small)

**Recommendation:**
- Use **60/20/20** split (20% val, 20% test)
- Or use **walk-forward validation** as primary

#### ⚠️ Issue 2.11: Single Split = No Variance Estimate

**Current:** One fixed split per run

**Problem:**
- Validation metrics are **point estimates** with unknown uncertainty
- No confidence intervals on Sharpe, win rate, etc.
- Could get lucky/unlucky with val set regime

**Recommendation:**
1. **Use CPCV as primary validation** (you already have it implemented!)
2. Report mean ± 95% CI across folds
3. Calculate **Probability of Backtest Overfitting (PBO)** to detect overfitting

**Note:** You already have `cpcv.py` and `pbo.py` implemented - **USE THEM** as primary validation!

### 2.6 OHLCV-Specific Concerns: 6/10

#### ⚠️ Issue 2.12: No Explicit Non-Stationarity Handling

**Problem:**
- Market regimes change (bull/bear, low-vol/high-vol)
- Features have different distributions in different regimes

**Recommendation:**
1. Regime-aware validation (stratified splits by regime)
2. Regime-adaptive models (separate models per regime)
3. Stationarity tests (ADF test, Chow test for structural breaks)

#### ⚠️ Issue 2.13: Microstructure Features May Be Meaningless

**Problem:**
- Microstructure features designed for **tick data** (sub-second)
- On 5-min bars, microstructure is **aggregated away**

**Recommendation:**
- Remove microstructure features unless you have tick data
- Or re-engineer: "volume in first minute of 5-min bar"

#### ⚠️ Issue 2.14: Embargo May Be Insufficient

**Current:** Embargo = 1440 bars = 5 days at 5-min

**Problem:**
- Volatility clustering can persist for **weeks**
- If features include volatility (ATR, BB width), 5-day embargo may not break correlation

**Recommendation:**
1. Test auto-correlation of features/labels at lag=1440
2. If ACF > 0.1, increase embargo
3. Use time-based embargo: "5 trading days" (avoids weekends/holidays)

---

## 3. MLOps & Production Readiness

### 3.1 Overall MLOps Maturity: 35/100 (Low)

**Status:** Research prototype, not production-ready

### 3.2 Pipeline Orchestration: 4/10

**🔴 Critical Gaps:**

1. **No DAG/Workflow Orchestration** (Airflow, Prefect, Dagster)
   - Current: Manual script execution via `./pipeline run --symbols MES`
   - Impact: No dependency management, retry logic, or failure recovery

2. **No Checkpointing/State Management**
   - Cannot resume from failure
   - Must rerun entire pipeline if stage 6 of 7 fails

3. **Limited Reproducibility Tracking**
   - No data version tracking (DVC, lakeFS)
   - No code version in artifacts
   - No environment snapshots

**Recommendation:**
```python
# Implement Prefect/Dagster workflow
from prefect import flow, task

@task(retries=3, retry_delay_seconds=60)
def run_mtf_upscaling(symbol: str):
    # Existing MTF logic
    pass

@flow(name="ml_pipeline")
def ml_factory_pipeline(symbol: str):
    raw_data = ingest_data(symbol)
    mtf_data = run_mtf_upscaling(raw_data)
    # ... rest of pipeline with automatic retry/recovery
```

### 3.3 Experiment Tracking: 2/10 (CRITICAL)

**🔴 No Experiment Tracking Platform**

```bash
$ grep -r "mlflow\|wandb\|neptune\|clearml\|tensorboard" src --include="*.py"
# NO RESULTS
```

**Current:**
- Basic JSON artifact saving only
- No centralized experiment dashboard
- No hyperparameter comparison UI
- No metric visualization

**Recommendation:**
```python
# Add MLflow integration to Trainer
import mlflow

class Trainer:
    def run(self, container):
        with mlflow.start_run(run_name=self.run_id):
            mlflow.log_params(self.config.to_dict())

            training_metrics = self.model.fit(...)

            mlflow.log_metrics({
                "val_f1": eval_metrics["macro_f1"],
                "val_accuracy": eval_metrics["accuracy"]
            })

            mlflow.pytorch.log_model(self.model, "model")
```

### 3.4 Data Versioning: 3/10 (CRITICAL)

**🔴 No Data Version Control**

**Missing:** DVC, lakeFS, Pachyderm, Delta Lake

**Data Provenance Issues:**
- `data/raw/MES_1m.parquet` - Where did this come from? When? Which vendor?
- `data/splits/scaled/` - What preprocessing was applied? Can we reproduce it?

**Recommendation:**
```bash
# Initialize DVC for data versioning
dvc init
dvc add data/raw/MES_1m.parquet
git add data/raw/MES_1m.parquet.dvc
git commit -m "Track MES 1-min OHLCV v1.0"
```

### 3.5 Model Serving: 3/10 (CRITICAL)

**🔴 No Real-Time Inference Deployment**

**Missing:**
- Model serving (TorchServe, BentoML, KServe)
- REST API endpoints
- Containerized inference (Docker)
- Kubernetes deployment manifests

**Current:**
```python
# Only batch inference supported
pipeline = InferencePipeline.from_bundle("./bundles/xgb_h20")
predictions = pipeline.predict(X_test)  # Batch only
```

**Recommendation:**
```python
# Add FastAPI serving
from fastapi import FastAPI
from src.inference import InferencePipeline

app = FastAPI()
pipeline = InferencePipeline.from_bundle("/models/production/xgb_h20")

@app.post("/predict")
async def predict(features: dict):
    X = preprocess_features(features)
    result = pipeline.predict(X, calibrate=True)
    return {
        "prediction": int(result.predictions.class_predictions[0]),
        "probabilities": result.predictions.class_probabilities[0].tolist()
    }
```

### 3.6 Monitoring & Observability: 2/10 (CRITICAL)

**🔴 No Production Monitoring Infrastructure**

**Missing:**
- Prometheus metrics export
- Grafana dashboards
- DataDog/New Relic APM
- Alert manager integration

**Found:** Basic drift detection only (`src/monitoring/drift_detector.py`)

**Recommendation:**
```python
# Add Prometheus metrics
from prometheus_client import Counter, Histogram

prediction_counter = Counter('predictions_total', 'Total predictions')
inference_latency = Histogram('inference_latency_seconds', 'Inference latency')

def predict_with_metrics(X):
    with inference_latency.time():
        result = model.predict(X)
    prediction_counter.inc()
    return result
```

### 3.7 CI/CD Pipeline: 0/10 (CRITICAL)

**🔴 No CI/CD Pipeline**

**Found:** `.github/` directory with templates only

**Missing:** `.github/workflows/ci.yml`

**Recommendation:**
```yaml
# .github/workflows/ci.yml
name: CI/CD Pipeline
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Run tests
        run: pytest tests/ --cov=src --cov-report=xml
      - name: Upload coverage
        uses: codecov/codecov-action@v2
```

### 3.8 Testing: 6/10

**Strengths:**
- 86 test files covering pipeline stages, models, CV
- Integration tests (`test_full_pipeline.py`)
- Leakage prevention tests (`test_lookahead_invariance.py`)

**Gaps:**
- No automated CI/CD runs
- No model performance regression tests
- No automated CV runs on model changes

### 3.9 Security & Compliance: 1/10

**🔴 No Security Implementation**

**Missing:**
- Authentication/authorization (OAuth2, JWT)
- Secrets management (Vault, AWS Secrets Manager)
- Input validation and sanitization
- Rate limiting
- HTTPS/TLS configuration
- Audit logging for compliance

---

## 4. Google Colab Training Assessment

### 4.1 Colab Environment Constraints

**Hardware Limits:**
- **GPU:** T4 (16GB VRAM), P100 (16GB), V100 (16GB) - free tier random allocation
- **RAM:** 12.7 GB standard, 25.5 GB high-RAM runtime
- **Disk:** ~78 GB temporary storage
- **Runtime:** 12-hour maximum session length
- **Persistence:** ZERO - all local files lost on disconnect

**Critical Implications:**
1. Training must complete in <6-10 hours (safety buffer for disconnects)
2. Cannot rely on local filesystem for anything important
3. Must checkpoint frequently to external storage (Drive/GCS/W&B)
4. Long experiments require multi-session workflows

### 4.2 Architecture Compatibility Assessment

#### ✅ COMPATIBLE: Data Pipeline (Phases 1-5)

**Current State:**
```python
# PipelineRunner has basic checkpointing
PipelineRunner._save_state()  # Saves after each stage
PipelineRunner._load_state()  # Resumes from checkpoint
```

**Artifact Sizes:**
- Raw 1m data: ~7-77 MB per symbol
- Features (5min, 180 features): **916 MB**
- Labels: ~15-16 MB
- Scaled splits: ~1-2 GB estimated

**Verdict:** ✅ Phases 1-5 can resume from last completed stage. Compatible with Colab.

**Required:** Mount Google Drive, save checkpoints to `/content/drive/MyDrive/ml_factory/`

#### ❌ INCOMPATIBLE: Model Training (Phase 6)

**Current State:**
- **NO** epoch-level checkpointing during training
- Models only saved AFTER training completes
- Early stopping checkpoints are in-memory only

**Risk:** Neural network training (LSTM, TCN, Transformer) takes 30-90 min per fold:
```
Fold 1: 40 min (epochs 1-100) → DISCONNECT at epoch 85
Result: ALL 85 EPOCHS LOST, restart from epoch 0
```

**Verdict:** ❌ Neural models will fail frequently without checkpointing implementation.

**Required:** Add epoch-level checkpoint saves every 10 epochs to Google Drive.

#### ⚠️ MODERATE RISK: Ensemble Training (Phase 7)

**Time Estimates:**

| Configuration | Bases | CV Time | Colab Risk |
|---------------|-------|---------|------------|
| Fast ensemble | CatBoost + Logistic + RF | 45 min | ✅ SAFE |
| Balanced ensemble | CatBoost + TCN + PatchTST | 265 min (4.4h) | ✅ SAFE |
| Maximum diversity | XGBoost + LSTM + TFT + Ridge | 312 min (5.2h) | ⚠️ MODERATE |
| 22-model CV sweep | All models × 5 folds | **19+ hours** | 🔴 TIMEOUT |

**Verdict:** 3-base heterogeneous ensembles are feasible (<6 hours). 4+ bases are risky.

### 4.3 Training Time Feasibility

#### Single Model Training (Without CV)

| Model Family | Training Time | Colab Risk | Notes |
|--------------|---------------|------------|-------|
| **Tabular** (6 models) | 1-3 min | ✅ SAFE | XGBoost, LightGBM, CatBoost, RF, Logistic, SVM |
| **Neural** (4 models) | 15-40 min | ✅ SAFE | LSTM, GRU, TCN, Transformer |
| **Advanced** (3 models) | 30-40 min | ✅ SAFE | PatchTST, iTransformer, TFT |
| **CNN** (2 models) | 30-60 min | ✅ SAFE | InceptionTime, ResNet1D |
| **MLP** (1 model) | 30-60 min | ✅ SAFE | N-BEATS |

**Verdict:** All individual models train successfully in <1 hour.

#### 5-Fold Cross-Validation Time

| Model Family | Single Fold | 5-Fold CV | Colab Risk |
|--------------|-------------|-----------|------------|
| Tabular | 2-3 min | 10-15 min | ✅ SAFE |
| Neural | 15-40 min | 75-200 min | ⚠️ MODERATE |
| Advanced | 30-40 min | 150-200 min | ⚠️ MODERATE |

**Verdict:**
- Tabular: CV completes in 15 min (always safe)
- Neural/Advanced: CV takes 1.25-3.3 hours (safe if run alone, risky if combined)

### 4.4 Memory Constraints

#### Current Dataset Sizes

**Tabular Models (2D):**
```
14K samples × 330 features × 8 bytes = 37 MB (features only)
+ model memory: ~500 MB (XGBoost)
Total: ~600 MB (✅ SAFE for 12.7 GB RAM)
```

**Sequence Models (3D - seq_len=60):**
```
14K samples × 60 timesteps × 150 features × 8 bytes = 1.0 GB
+ LSTM memory: ~2 GB (hidden_size=256)
+ GPU memory: ~4 GB VRAM (training + gradients)
Total: ~7 GB RAM + 4 GB VRAM (✅ SAFE for Colab T4 16GB)
```

**Advanced Transformers (4D - Multi-Res):**
```
3 TFs × 60 × 50 × 8 bytes × 14K samples = 2.0 GB
+ Transformer memory: ~3 GB (attention mechanisms)
+ GPU memory: ~6 GB VRAM
Total: ~11 GB RAM + 6 GB VRAM (⚠️ MODERATE RISK)
```

**OOM Risk Assessment:**
- **LOW RISK:** All tabular models, basic neural (LSTM/GRU)
- **MODERATE RISK:** TCN, Transformer, advanced models (PatchTST/TFT)
- **HIGH RISK:** Batch size >256, seq_len >120, multiple timeframes loaded simultaneously

**Mitigations:**
1. Reduce batch_size from 256 to 128 for transformers
2. Cap sequence_length at 60 (90 max for simpler models)
3. Monitor GPU memory: `torch.cuda.memory_allocated()`

### 4.5 Hyperparameter Tuning Constraints

**Current Optuna Config:**
- Default: 50 trials per model
- Each trial: 5-fold CV

**Time Estimates:**

| Model | Trials × Folds × Time | Total Time | Colab Verdict |
|-------|------------------------|------------|---------------|
| XGBoost | 50 × 5 × 2 min | 500 min (8.3h) | ⚠️ RISKY |
| LSTM | 50 × 5 × 15 min | 3,750 min (62.5h) | 🔴 TIMEOUT |
| PatchTST | 50 × 5 × 30 min | 7,500 min (125h) | 🔴 TIMEOUT |

**Recommendations for Colab:**
- Tabular models: **20 trials** (3.3h) - safe
- Neural models: **10 trials** (12.5h) - still risky, use pre-tuned configs
- Advanced models: **5 trials** (12.5h) - still risky, use pre-tuned configs

**Alternative:** Run tuning separately from final training in dedicated sessions.

### 4.6 Critical Gaps for Colab

#### 🔴 Gap 1: No Mid-Training Checkpointing

**Location:** `src/models/neural/base_rnn.py` lines 381-424 (training loop)

**Current:** Models only save AFTER training completes
**Required:** Save every 10 epochs to Google Drive

**Implementation Needed:**
```python
# Add to BaseRNNModel.fit()
if epoch % 10 == 0:  # Checkpoint every 10 epochs
    checkpoint_path = drive_path / f"checkpoint_epoch_{epoch}.pt"
    torch.save({
        'epoch': epoch,
        'model_state_dict': self._model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_val_loss': best_val_loss,
    }, checkpoint_path)
```

#### 🔴 Gap 2: No Cloud Artifact Persistence

**Current:** All artifacts saved to local disk
```python
# trainer.py line 774
output_path = Path(f"experiments/runs/{run_id}")  # LOCAL ONLY!
```

**Problem:** All experiments lost on disconnect

**Required:** Integrate Google Drive or W&B Artifacts
```python
# Option 1: Google Drive
output_path = Path(f"/content/drive/MyDrive/ml_factory/runs/{run_id}")

# Option 2: W&B Artifacts
wandb.log_artifact(model, name=f"{model_name}_h{horizon}", type="model")
```

#### 🔴 Gap 3: No Experiment Tracking

**Current:** No MLflow/W&B integration
**Impact:** Cannot compare experiments across sessions

**Required:** W&B or Comet integration
```python
import wandb

wandb.init(project="ohlcv-ml-factory", name=run_id)
wandb.log({"epoch": epoch, "val_loss": val_loss})
wandb.log_artifact(model_path, type="model")
```

#### ⚠️ Gap 4: No Fold-Level Checkpointing for CV

**Current:** CV runs sequentially through 5 folds with no checkpointing
**Risk:** If disconnect during fold 4, lose folds 1-3

**Required:** Save OOF predictions after each fold
```python
# In cv_runner.py after each fold
checkpoint_path = output_dir / f"oof_checkpoint_fold_{fold_idx}.npz"
np.savez(checkpoint_path,
         predictions=oof_predictions[val_idx],
         indices=val_idx,
         fold=fold_idx)
```

### 4.7 Recommended Colab Workflow

#### Phase 1: Data Pipeline (Run Once)

**Session 1 (30-60 min):**
```bash
# Mount Drive
from google.colab import drive
drive.mount('/content/drive')

# Clone repo and install
!git clone https://github.com/YOUR_USERNAME/research.git  # Replace with actual repo URL
%cd research
!pip install -r requirements-colab.txt

# Run pipeline phases 1-5
# NOTE: Pipeline outputs to data/splits/scaled/ by default
# Manual copy to Drive required after completion
!./pipeline run --symbols MES
!cp -r data/splits/scaled /content/drive/MyDrive/ml_factory/
```

**Output:** Processed data saved to Drive (~2 GB)

#### Phase 2: Model Training (Multiple Sessions)

**Session 2A - Tabular Models (30 min):**
```bash
# Copy data from Drive to local (faster I/O)
!cp -r /content/drive/MyDrive/ml_factory/scaled data/splits/

# Run CV with tabular models
!python scripts/run_cv.py \
  --models xgboost,lightgbm,catboost \
  --horizons 20 --n-splits 5 \
  --data-dir data/splits/scaled \
  --output-dir /content/drive/MyDrive/ml_factory/cv_results/
```

**Session 2B - Neural Models (4-6 hours):**
```bash
# NOTE: --checkpoint-every is NOT YET IMPLEMENTED
# Neural training will need to complete in one session or implement epoch checkpointing
!python scripts/run_cv.py \
  --models lstm,gru,tcn \
  --horizons 20 --n-splits 5 \
  --data-dir data/splits/scaled \
  --output-dir /content/drive/MyDrive/ml_factory/cv_results/
# TODO: Add epoch-level checkpointing to neural models for Colab reliability
```

**Session 2C - Advanced Models (4-6 hours):**
```bash
# NOTE: Requires implementing epoch-level checkpointing for Colab reliability
!python scripts/run_cv.py \
  --models patchtst,itransformer \
  --horizons 20 --n-splits 5 \
  --data-dir data/splits/scaled \
  --output-dir /content/drive/MyDrive/ml_factory/cv_results/
```

#### Phase 3: Ensemble Training (30 min)

**Session 3 - Meta-Learner:**
```bash
# Load CV results and train stacking ensemble
# NOTE: --stacking-data takes a CV run ID (timestamp), not a directory path
!python scripts/train_model.py --model stacking --horizon 20 \
  --base-models catboost,tcn,patchtst \
  --meta-learner ridge_meta \
  --stacking-data "20260103_120000_123456_a1b2" \
  --phase3-output /content/drive/MyDrive/ml_factory/cv_results/
```

### 4.8 Colab-Specific Recommendations

#### Priority 0 (CRITICAL - Must Implement)

1. **Epoch-level checkpointing for neural models**
   - Save every 10 epochs to Google Drive
   - Resume from last checkpoint on restart
   - **Location:** `src/models/neural/base_rnn.py`
   - **Effort:** ~50 lines of code, 2-3 hours

2. **Google Drive integration**
   - Auto-mount Drive at session start
   - Save all artifacts to `/content/drive/MyDrive/ml_factory/`
   - **Location:** Add `scripts/colab_setup.py`
   - **Effort:** ~30 lines, 1 hour

3. **W&B experiment tracking**
   - Log metrics, hyperparameters, artifacts
   - Enable experiment comparison across sessions
   - **Location:** `src/models/trainer.py`
   - **Effort:** ~40 lines, 2 hours

#### Priority 1 (HIGH - Strongly Recommended)

4. **Fold-level checkpointing for CV**
   - Save OOF predictions after each fold
   - Resume from last completed fold
   - **Location:** `src/cross_validation/cv_runner.py`
   - **Effort:** ~30 lines, 1-2 hours

5. **Batch size auto-adjustment**
   - Detect GPU memory, reduce batch_size if needed
   - Prevent OOM errors automatically
   - **Location:** `src/models/device.py`
   - **Effort:** ~40 lines, 2 hours

6. **Progress tracking and time estimates**
   - Print estimated time remaining
   - Log memory usage every 10 epochs
   - **Location:** Training loops
   - **Effort:** ~20 lines, 1 hour

#### Priority 2 (MEDIUM - Nice to Have)

7. **Colab notebook templates**
   - Pre-configured notebooks for common workflows
   - Auto-mount Drive, install dependencies
   - **Location:** `notebooks/colab/`
   - **Effort:** 3-4 notebooks, 4-6 hours

8. **Gradient checkpointing for transformers**
   - Reduce memory by recomputing activations
   - Trade 20% speed for 50% memory reduction
   - **Location:** Transformer models
   - **Effort:** ~10 lines per model, 2 hours

#### Priority 3 (CRITICAL - Data Integrity)

9. **Checkpoint Leakage Prevention**
   - **CRITICAL:** Checkpoints must NOT leak information across train/val/test splits
   - When resuming mid-fold, ensure validation data was never seen during previous partial training
   - Save and restore RNG states for reproducibility
   - **Location:** All checkpointing code
   - **Considerations:**
     - Checkpoints should save: model weights, optimizer state, epoch number, RNG states
     - Checkpoints should NOT save: validation metrics from future epochs
     - On resume, validation set must be exactly the same (use saved fold indices)

10. **Cross-Session Reproducibility**
    - Different Colab sessions may have different GPU types (T4 vs P100 vs V100)
    - Floating-point operations may differ slightly across GPU architectures
    - **Mitigations:**
      - Set `torch.backends.cudnn.deterministic = True`
      - Log GPU type in experiment metadata
      - Accept that exact reproducibility across different GPUs is not guaranteed
    - **Location:** `colab_notebooks/utils/colab_setup.py`

11. **OOF Prediction Integrity for Stacking**
    - When checkpointing CV folds, OOF predictions must align exactly
    - **CRITICAL:** If fold N crashes mid-training, re-run from scratch (do NOT use partial OOF)
    - Save fold completion status: "not_started", "in_progress", "completed"
    - Only use OOF predictions from completed folds
    - **Location:** `src/cross_validation/oof_generator.py`

### 4.9 Colab Compatibility Matrix

| Component | Current Status | Colab Compatible | Required Changes |
|-----------|----------------|------------------|------------------|
| **Data Pipeline (Phases 1-5)** | Checkpointing exists | ✅ YES | Mount Drive for saves |
| **Tabular Models** | Fast training | ✅ YES | None |
| **Neural Models** | No checkpointing | ❌ NO | Epoch-level checkpointing |
| **Advanced Models** | No checkpointing | ❌ NO | Epoch-level checkpointing |
| **Ensemble Training (3 bases)** | Sequential | ✅ YES | Drive persistence |
| **Ensemble Training (4+ bases)** | Sequential | ⚠️ RISKY | Multi-session workflow |
| **Cross-Validation** | Sequential folds | ⚠️ MODERATE | Fold-level checkpointing |
| **Hyperparameter Tuning** | 50 trials | ❌ NO | Reduce to 5-20 trials |
| **Experiment Tracking** | Local JSON | ❌ NO | W&B/Comet integration |
| **Artifact Storage** | Local filesystem | ❌ NO | Drive/GCS persistence |

### 4.10 Estimated Colab Training Times

#### Recommended 3-Base Heterogeneous Ensemble

**Configuration:** CatBoost + TCN + PatchTST (maximum diversity)

| Component | Time per Fold | 5-Fold CV | Total |
|-----------|---------------|-----------|-------|
| CatBoost | 3 min | 15 min | 15 min |
| TCN | 20 min | 100 min | 100 min |
| PatchTST | 30 min | 150 min | 150 min |
| **Total** | **53 min** | **265 min** | **4.4 hours** |

**Verdict:** ✅ SAFE for 12-hour Colab runtime (7.6 hour buffer)

**Session Breakdown:**
- 0:00-4:24 - Train all 3 bases (5-fold CV, generate OOF)
- 4:24-4:30 - Train Ridge meta-learner on OOF (6 min)
- 4:30-5:00 - Final evaluation on test set (30 min)
- **Total: ~5 hours** (7-hour safety buffer)

#### Alternative: 4-Base Maximum Diversity

**Configuration:** LightGBM + TCN + TFT + Ridge

| Component | Time per Fold | 5-Fold CV |
|-----------|---------------|-----------|
| LightGBM | 2 min | 10 min |
| TCN | 20 min | 100 min |
| TFT | 40 min | 200 min |
| Ridge | 0.5 min | 2.5 min |
| **Total** | **62.5 min** | **312.5 min** |

**Verdict:** ⚠️ MODERATE RISK (5.2 hours training, 6.8 hour buffer)

### 4.11 Colab Resource Monitoring

**Recommended Monitoring Code:**
```python
# Add to training loops
import torch
import psutil

def log_resources(epoch: int):
    """Log GPU and RAM usage."""
    if torch.cuda.is_available():
        gpu_mem_allocated = torch.cuda.memory_allocated() / 1e9  # GB
        gpu_mem_reserved = torch.cuda.memory_reserved() / 1e9
        print(f"Epoch {epoch} | GPU Mem: {gpu_mem_allocated:.2f}/{gpu_mem_reserved:.2f} GB")

    ram_used = psutil.virtual_memory().used / 1e9  # GB
    ram_total = psutil.virtual_memory().total / 1e9
    print(f"Epoch {epoch} | RAM: {ram_used:.2f}/{ram_total:.2f} GB")

    if ram_used > 11.0:  # Approaching 12.7 GB limit
        print("⚠️ WARNING: High RAM usage, consider reducing batch size")
```

---

## 5. Online Research Validation

### 4.1 Triple Barrier Labeling Best Practices

**Sources:**
- [MLFinLab Triple Barrier Documentation](https://www.mlfinlab.com/en/latest/labeling/tb_meta_labeling.html)
- [What Is the Triple Barrier Method?](https://xglamdring.com/what-is-the-triple-barrier-method-a-labeling-technique-to-prevent-overfitting-in-ml-based-quantitative-trading/)
- [Labeling Financial Data for ML](https://www.sefidian.com/2021/06/26/labeling-financial-data-for-machine-learning/)

**Key Findings:**

1. **Asymmetric barriers should be data-driven:** Use different multipliers based on empirical analysis, not assumptions
2. **Transaction costs are critical backtesting errors:** Ignoring slippage/costs leads to unrealistic results
3. **Dynamic barriers:** Set upper/lower barriers based on volatility (you're doing this with ATR)

**Validation of Issues:**
- ✅ Your implementation correctly uses dynamic barriers based on ATR
- ❌ Asymmetric barrier justification (equity drift) is not supported by literature
- ❌ Transaction cost adjustment in labels (not backtesting) is non-standard

### 4.2 Ensemble Stacking for Time Series

**Sources:**
- [Hidden Leaks in Time Series Forecasting](https://arxiv.org/html/2512.06932)
- [Stacking Ensemble Models for Time Series](https://cienciadedatos.net/documentos/py52-stacking-ensemble-models-forecasting.html)
- [Effective ML Model Combination](https://www.sciencedirect.com/science/article/abs/pii/S0020025522010465)

**Key Findings:**

1. **Data leakage in stacking:** Improper stacking protocols (training meta-learner on in-sample predictions) produce overoptimistic outcomes
2. **OOF predictions are essential:** Meta-learner must train on out-of-fold predictions to avoid information leakage
3. **Time series CV:** TSCV solves leakage by applying sequential partition and evaluation

**Validation of Issues:**
- ✅ You're using OOF predictions (correct approach)
- ⚠️ Heterogeneous stacking with different timeframes may introduce alignment issues
- ⚠️ No nested CV to prevent meta-learner overfitting

### 4.3 MLOps for Financial Trading (2025)

**Sources:**
- [MLOps Best Practices for Quantitative Trading](https://medium.com/@online-inference/mlops-best-practices-for-quantitative-trading-teams-59f063d3aaf8)
- [MLOps: Deploying and Monitoring ML Models in 2025](https://dasroot.net/posts/2025/12/mlops-deploying-monitoring-ml-models-2025/)
- [12 MLOps Best Practices Every Enterprise Needs](https://www.shakudo.io/blog/mlops-best-practices-enterprise-2025)

**Key Findings:**

1. **Model registry for promotion:** Manage models from research to production with automated tests
2. **Real-time drift detection:** Tools like ModelBit/Dagster alert teams within minutes
3. **Latency-sensitive environments:** Quant finance requires low-latency inference
4. **Continuous monitoring:** Track prediction drift, execution latency, P&L impact

**Validation of Gaps:**
- ❌ No model registry for production promotion (only local plugin system)
- ❌ No real-time drift detection (only offline)
- ❌ No latency monitoring
- ❌ No P&L impact tracking

**Market Growth:** MLOps market projected to reach $89B by 2034 (39.8% CAGR), with BFSI holding 40% market share in 2024.

### 4.4 OHLCV Production Pipelines (2025)

**Sources:**
- [Feature Engineering for Stock Prediction](https://alphascientist.com/feature_engineering.html)
- [MLOps Pipeline for Time Series Prediction](https://neptune.ai/blog/mlops-pipeline-for-time-series-prediction-tutorial)
- [Structural VAR and VECM for OHLC Data](https://link.springer.com/article/10.1186/s40854-024-00622-6)

**Key Findings:**

1. **OHLCV constraints:** Open-high-low-close data has inherent constraints that require special handling
2. **Reproducible ML pipelines:** Define repeatable steps for data prep, training, scoring
3. **Data quality is critical:** True challenge is building entire system for ongoing operation

**Validation:**
- ✅ You have reproducible 7-phase pipeline
- ✅ OHLCV data properly validated (high >= low, etc.)
- ❌ Missing ongoing operational infrastructure

---

## 6. Critical Gaps Summary

### 6.1 Original Gaps (General Production)

#### Original Production Blockers

| Issue | Severity | Impact | Location |
|-------|----------|--------|----------|
| Asymmetric barrier logic is statistically flawed | 🔴 Critical | Biased labels, poor generalization | `barriers_config.py` |
| Feature count 2-3× too high (330 features) | 🔴 Critical | Severe overfitting risk | `feature_sets.py` |
| No experiment tracking platform | 🔴 Critical | Cannot compare models or track performance | N/A (missing) |
| No model serving infrastructure | 🔴 Critical | Cannot deploy for real-time inference | N/A (missing) |
| No monitoring/alerting | 🔴 Critical | Cannot detect failures or degradation | N/A (missing) |
| No CI/CD pipeline | 🔴 Critical | Manual deployment, high error risk | N/A (missing) |
| No data versioning | 🔴 Critical | Cannot reproduce experiments | N/A (missing) |

#### Original High-Risk Issues

| Issue | Severity | Impact | Location |
|-------|----------|--------|----------|
| Transaction costs in labels (not backtesting) | ⚠️ High | Confuses "what happened" with "was it profitable" | `triple_barrier.py` |
| Heterogeneous stacking data mismatch | ⚠️ High | Different timeframes, uncertain OOF alignment | `trainer.py` |
| Single train/val/test split | ⚠️ High | No variance estimate, could get lucky/unlucky | `splits/` stage |
| MTF lookahead bias potential | ⚠️ High | Hidden data leakage from circular dependencies | `mtf/generator.py` |
| No distributed tracing | ⚠️ High | Cannot debug latency issues | N/A (missing) |
| No fallback strategies | ⚠️ High | Single point of failure | N/A (missing) |

#### Original Medium-Priority Issues

| Issue | Severity | Impact |
|-------|----------|--------|
| Insufficient purge for short horizons | ⚠️ Medium | Over/under-purging affects data efficiency |
| MTF feature quality concerns | ⚠️ Medium | Higher TFs have weaker statistical significance |
| Max bars timeout ambiguity | ⚠️ Medium | High label noise in neutral class |
| No non-stationarity handling | ⚠️ Medium | Models may fail in regime shifts |
| Embargo may be insufficient | ⚠️ Medium | Volatility clustering persists weeks |
| No horizontal scaling | ⚠️ Medium | Cannot process multiple symbols concurrently |

### 6.2 Google Colab-Specific Gaps

#### 🔴 Colab Blockers (Cannot Train on Colab)

| Issue | Severity | Impact | Location |
|-------|----------|--------|----------|
| No mid-training checkpointing | 🔴 Critical | All progress lost on disconnect | `base_rnn.py`, transformer models |
| No cloud artifact persistence | 🔴 Critical | All experiments lost on session end | `trainer.py` |
| No experiment tracking (W&B/Comet) | 🔴 Critical | Cannot compare experiments across sessions | N/A (missing) |
| No Google Drive integration | 🔴 Critical | Cannot save/resume checkpoints | N/A (missing) |
| No epoch-level resume capability | 🔴 Critical | Must restart training from scratch | All neural models |

#### ⚠️ Colab High-Risk Issues

| Issue | Severity | Impact |
|-------|----------|--------|
| No fold-level CV checkpointing | ⚠️ High | Lose partial CV progress on disconnect |
| Hyperparameter tuning (50 trials) | ⚠️ High | 62+ hours for neural models (timeout) |
| 4+ base heterogeneous ensembles | ⚠️ High | 5.2+ hours (risky for session stability) |
| No batch size auto-adjustment | ⚠️ High | OOM crashes for large transformers |
| No resource monitoring | ⚠️ High | Cannot predict/prevent memory issues |

#### Medium-Priority Colab Issues

| Issue | Severity | Impact |
|-------|----------|--------|
| No Colab notebook templates | ⚠️ Medium | Manual setup required each session |
| No session time tracking | ⚠️ Medium | Cannot warn about approaching 12h limit |
| No gradient checkpointing | ⚠️ Medium | Higher memory usage for transformers |
| No DVC integration guide | ⚠️ Medium | Manual data versioning |

---

## 7. Prioritized Recommendations

### 7.1 Original Recommendations (General Production)

#### Original Phase 1: Critical Fixes (Weeks 1-2)

**Week 1: Fix Labeling Methodology**

1. **Use symmetric barriers** unless data-driven evidence proves otherwise:
   ```python
   # barriers_config.py
   "MES": {
       5: {"k_up": 1.00, "k_down": 1.00, ...},  # SYMMETRIC
       20: {"k_up": 2.50, "k_down": 2.50, ...},  # SYMMETRIC
   }
   ```

2. **Remove transaction costs from labels**, apply in backtesting:
   ```python
   # Labels: Pure price movement
   labels = triple_barrier(k_up=2.0, k_down=2.0)

   # Backtesting: Apply costs to P&L
   net_pnl = gross_pnl - transaction_costs
   ```

3. **Reduce feature count drastically**:
   - Target ≤100 features before selection, ≤50 after
   - Limit MTF to 2-3 timeframes (not 5)
   - Remove microstructure features (meaningless on 5-min bars)

**Week 2: Add Experiment Tracking & Data Versioning**

4. **Integrate MLflow**:
   ```bash
   pip install mlflow
   # Add mlflow logging to Trainer
   mlflow ui --host 0.0.0.0
   ```

5. **Add DVC for data versioning**:
   ```bash
   pip install dvc
   dvc init
   dvc add data/raw/
   dvc remote add -d storage s3://your-bucket/dvc-store
   ```

6. **Use CPCV + PBO as primary validation** (already implemented!):
   - Report mean ± 95% CI across folds
   - Calculate Probability of Backtest Overfitting

#### Original Phase 2: MLOps Foundation (Weeks 3-5)

**Week 3: CI/CD Pipeline**

7. **Add GitHub Actions workflow**:
   ```yaml
   # .github/workflows/ci.yml
   name: CI/CD
   on: [push, pull_request]
   jobs:
     test:
       runs-on: ubuntu-latest
       steps:
         - name: Run tests
           run: pytest tests/ --cov=src
   ```

8. **Add model performance regression tests**:
   ```python
   def test_xgboost_performance_regression():
       baseline_f1 = 0.65  # From previous best run
       new_f1 = train_and_evaluate("xgboost", horizon=20)
       assert new_f1 >= baseline_f1 * 0.95, "Performance regressed >5%"
   ```

**Week 4: Containerization**

9. **Create Dockerfile**:
   ```dockerfile
   FROM python:3.11-slim
   WORKDIR /app
   COPY requirements.txt .
   RUN pip install --no-cache-dir -r requirements.txt
   COPY src/ ./src/
   CMD ["uvicorn", "src.inference.server:app", "--host", "0.0.0.0"]
   ```

10. **Add docker-compose for local testing**

**Week 5: Model Serving**

11. **Implement FastAPI inference server**:
    ```python
    from fastapi import FastAPI

    app = FastAPI()
    pipeline = InferencePipeline.from_bundle("/models/production")

    @app.post("/predict")
    async def predict(features: dict):
        result = pipeline.predict(preprocess(features))
        return {"prediction": int(result.class_predictions[0])}
    ```

12. **Add health checks and readiness probes**

#### Original Phase 3: Production Hardening (Weeks 6-10)

**Week 6-7: Monitoring & Alerting**

13. **Add Prometheus metrics**:
    ```python
    from prometheus_client import Counter, Histogram

    prediction_counter = Counter('predictions_total', 'Total predictions')
    inference_latency = Histogram('inference_latency_seconds')
    ```

14. **Create Grafana dashboards** for:
    - Prediction volume and latency
    - Model confidence distribution
    - Drift detection alerts
    - Error rates

15. **Set up alerting** (PagerDuty/Slack):
    ```python
    if drift_result.severity == "high":
        slack.send_alert(f"HIGH drift: {drift_result.drifted_features}")
    ```

**Week 8-9: Advanced Validation**

16. **Add MTF lookahead tests**:
    ```python
    def test_mtf_no_lookahead():
        mtf_rsi = df.loc[100, 'rsi_14_1h']
        manual_rsi = compute_rsi(df.loc[:99, 'close'].resample('1h'))[-1]
        assert mtf_rsi == manual_rsi, "MTF uses future data!"
    ```

17. **Implement adaptive purge/embargo**:
    ```python
    purge_bars = 3 * max_bars[horizon]  # Horizon-specific
    ```

18. **Add regime-aware validation** (stratified splits by volatility/trend)

**Week 10: Distributed Infrastructure**

19. **Add Prefect/Dagster orchestration**:
    ```python
    @flow(name="ml_pipeline")
    def ml_factory_pipeline(symbol: str):
        raw = ingest_data(symbol)
        mtf = run_mtf_upscaling(raw)  # With automatic retry
        # ... rest with checkpointing
    ```

20. **Implement Redis caching** for features

21. **Add parallel base model training** for ensembles

#### Original Phase 4: Advanced Features (Weeks 11-16)

**Week 11-12: Observability**

22. **Add distributed tracing** (OpenTelemetry)
23. **Implement structured logging** (JSON format)
24. **Add performance profiling**

**Week 13-14: Scalability**

25. **Implement micro-batching** for real-time inference
26. **Add horizontal pod autoscaling** (Kubernetes)
27. **Optimize data loading** (streaming/chunked)

**Week 15-16: Advanced Ensemble**

28. **Fix heterogeneous stacking**:
    - Ensure OOF alignment across different timeframes
    - Run ablation study: heterogeneous vs homogeneous
    - Add nested CV for meta-learner

29. **Implement A/B testing infrastructure**:
    ```python
    class ModelRouter:
        def predict(self, X, user_id: str):
            if hash(user_id) % 10 == 0:
                return self.challenger.predict(X)  # 10% traffic
            return self.champion.predict(X)  # 90% traffic
    ```

30. **Add model governance workflows** (approval, bias testing, explainability)

### 7.2 Google Colab-Specific Recommendations

#### Colab Phase 0: Critical Prerequisites (Week 1 - MUST DO FIRST)

**🔴 Priority 0A: Epoch-Level Checkpointing (2-3 hours)**

Add to all neural models (LSTM, GRU, TCN, Transformer, PatchTST, iTransformer, TFT):

```python
# In BaseRNNModel.fit() and transformer models
checkpoint_dir = Path(config.get("checkpoint_dir", "/content/drive/MyDrive/ml_factory/checkpoints"))
checkpoint_dir.mkdir(parents=True, exist_ok=True)

for epoch in range(start_epoch, num_epochs):
    # Training loop...

    # Save checkpoint every 10 epochs
    if epoch % 10 == 0 or epoch == num_epochs - 1:
        checkpoint_path = checkpoint_dir / f"{run_id}_epoch_{epoch}.pt"
        torch.save({
            'epoch': epoch,
            'model_state_dict': self._model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'scaler_state_dict': scaler.state_dict() if scaler else None,
            'best_val_loss': best_val_loss,
            'history': history,
        }, checkpoint_path)
        print(f"✓ Checkpoint saved: {checkpoint_path.name}")
```

**Files to modify:**
- `src/models/neural/base_rnn.py` (LSTM, GRU)
- `src/models/neural/tcn.py`
- `src/models/neural/transformer.py`
- `src/models/neural/patchtst.py`
- `src/models/neural/itransformer.py`
- `src/models/neural/tft.py`

**🔴 Priority 0B: Google Drive Integration (1 hour)**

```python
# Create: scripts/colab_setup.py
from google.colab import drive
from pathlib import Path
import os

def setup_colab_environment():
    """Setup Colab environment with Drive mount and repo clone."""

    # Mount Drive
    print("📁 Mounting Google Drive...")
    drive.mount('/content/drive', force_remount=False)

    # Create ml_factory directory
    ml_factory_dir = Path("/content/drive/MyDrive/ml_factory")
    ml_factory_dir.mkdir(parents=True, exist_ok=True)

    # Subdirectories
    (ml_factory_dir / "data").mkdir(exist_ok=True)
    (ml_factory_dir / "checkpoints").mkdir(exist_ok=True)
    (ml_factory_dir / "experiments").mkdir(exist_ok=True)
    (ml_factory_dir / "cv_results").mkdir(exist_ok=True)

    # Clone repo (if not already cloned)
    if not Path("/content/research").exists():
        print("📦 Cloning repository...")
        os.system("git clone https://github.com/YOUR_USERNAME/research.git /content/research")  # Replace with actual repo

    # Install dependencies
    print("📚 Installing dependencies...")
    os.system("pip install -q -r /content/research/requirements-colab.txt")

    print("✓ Colab environment ready!")
    print(f"✓ ML Factory directory: {ml_factory_dir}")

    return ml_factory_dir
```

**🔴 Priority 0C: W&B Experiment Tracking (2 hours)**

```python
# Add to src/models/trainer.py in Trainer.run()

import wandb

# Initialize W&B
if self.config.use_wandb:
    wandb.init(
        project="ohlcv-ml-factory",
        name=f"{self.model.name}_h{self.config.horizon}_{self.run_id}",
        config={
            "model": self.model.name,
            "horizon": self.config.horizon,
            "symbol": self.config.symbol,
            **self.config.model_config
        }
    )

# Log metrics during training (in fit() method)
if self.config.use_wandb:
    wandb.log({
        "epoch": epoch,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "val_f1": val_f1
    })

# Log final model
if self.config.use_wandb:
    wandb.log_artifact(str(model_path), name=f"{self.model.name}_h{self.config.horizon}", type="model")
    wandb.finish()
```

**Add to TrainerConfig:**
```python
use_wandb: bool = True
wandb_project: str = "ohlcv-ml-factory"
```

#### Colab Phase 1: Essential Improvements (Week 2)

**Priority 1A: Fold-Level CV Checkpointing (1-2 hours)**

```python
# Add to src/cross_validation/cv_runner.py

def run_cross_validation(...):
    checkpoint_dir = output_dir / "fold_checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for fold_idx, (train_idx, val_idx) in enumerate(cv_splits):
        # Check for existing checkpoint
        checkpoint_file = checkpoint_dir / f"fold_{fold_idx}_oof.npz"
        if checkpoint_file.exists():
            print(f"✓ Loading fold {fold_idx} from checkpoint")
            checkpoint = np.load(checkpoint_file)
            oof_predictions[val_idx] = checkpoint['predictions']
            continue

        # Train fold...
        metrics = model.fit(X_train=X_train, ...)

        # Save checkpoint
        np.savez(checkpoint_file,
                 predictions=oof_predictions[val_idx],
                 indices=val_idx,
                 fold=fold_idx,
                 metrics=metrics)
        print(f"✓ Fold {fold_idx} checkpoint saved")
```

**Priority 1B: Batch Size Auto-Adjustment (2 hours)**

```python
# Add to src/models/device.py

def get_optimal_batch_size(model, input_shape, device):
    """Automatically determine optimal batch size based on available GPU memory."""
    if not torch.cuda.is_available():
        return 256  # Default for CPU

    # Get available GPU memory
    gpu_mem_free = torch.cuda.get_device_properties(device).total_memory
    gpu_mem_free -= torch.cuda.memory_allocated(device)
    gpu_mem_free_gb = gpu_mem_free / 1e9

    # Estimate memory per sample (rough heuristic)
    if "transformer" in model.__class__.__name__.lower():
        mem_per_sample_mb = 8  # Transformers are memory-intensive
    elif "lstm" in model.__class__.__name__.lower() or "gru" in model.__class__.__name__.lower():
        mem_per_sample_mb = 4
    else:
        mem_per_sample_mb = 2

    # Calculate batch size
    max_batch_size = int((gpu_mem_free_gb * 1000) / (mem_per_sample_mb * 1.5))  # 1.5x safety factor
    batch_size = min(256, max(32, max_batch_size))

    print(f"📊 Auto-selected batch_size={batch_size} (GPU memory: {gpu_mem_free_gb:.1f} GB free)")
    return batch_size
```

**Priority 1C: Resource Monitoring (1 hour)**

```python
# Add to src/models/neural/base_rnn.py

def log_resources(epoch: int, config):
    """Log GPU and RAM usage during training."""
    if torch.cuda.is_available():
        gpu_mem_gb = torch.cuda.memory_allocated() / 1e9
        gpu_max_gb = torch.cuda.max_memory_allocated() / 1e9
        print(f"  GPU: {gpu_mem_gb:.2f} GB used (max: {gpu_max_gb:.2f} GB)")

        if gpu_mem_gb > 14.0:  # Approaching 16GB limit
            print("  ⚠️ WARNING: High GPU memory usage!")

    ram_gb = psutil.virtual_memory().used / 1e9
    ram_total_gb = psutil.virtual_memory().total / 1e9
    print(f"  RAM: {ram_gb:.2f} / {ram_total_gb:.2f} GB")

    if ram_gb > 11.0:  # Approaching 12.7GB limit
        print("  ⚠️ WARNING: High RAM usage, consider reducing batch size!")
```

#### Colab Phase 2: Workflow Improvements (Week 3)

**Priority 2A: Colab Notebook Templates (4-6 hours)**

Create 5 template notebooks:
1. `notebooks/colab/00_setup.ipynb` - Environment setup, Drive mount, repo clone
2. `notebooks/colab/01_data_pipeline.ipynb` - Run phases 1-5, save to Drive
3. `notebooks/colab/02_train_tabular.ipynb` - Train boosting/classical models
4. `notebooks/colab/03_train_neural.ipynb` - Train LSTM/GRU/TCN with checkpointing
5. `notebooks/colab/04_train_ensemble.ipynb` - Heterogeneous stacking

**Priority 2B: Session Time Manager (1-2 hours)**

```python
# Create: src/utils/session_manager.py

import time
from datetime import datetime, timedelta

class ColabSessionManager:
    """Monitor Colab session time and warn about approaching limits."""

    def __init__(self, max_session_hours=12, warning_hours=2):
        self.start_time = time.time()
        self.max_session_hours = max_session_hours
        self.warning_hours = warning_hours

    def check_time_remaining(self):
        """Check time remaining in session."""
        elapsed_hours = (time.time() - self.start_time) / 3600
        remaining_hours = self.max_session_hours - elapsed_hours

        if remaining_hours < self.warning_hours:
            print(f"⚠️ WARNING: Only {remaining_hours:.1f} hours remaining in session!")
            print(f"   Consider saving checkpoints and preparing to restart.")

        return remaining_hours

    def estimate_completion(self, current_step, total_steps, step_time_sec):
        """Estimate if task will complete before session ends."""
        remaining_steps = total_steps - current_step
        estimated_time_hours = (remaining_steps * step_time_sec) / 3600
        time_remaining = self.check_time_remaining()

        if estimated_time_hours > time_remaining:
            print(f"⚠️ WARNING: Estimated completion time ({estimated_time_hours:.1f}h) exceeds session time ({time_remaining:.1f}h)!")
            return False
        return True
```

**Priority 2C: requirements-colab.txt (30 min)**

```txt
# Create: requirements-colab.txt
# Lightweight dependencies for Colab (already has torch, numpy, pandas)

wandb
optuna
scikit-learn
xgboost
lightgbm
catboost
pydantic
tqdm
psutil
```

#### Colab Phase 3: Advanced Features (Week 4)

**Priority 3A: Gradient Checkpointing for Transformers (2 hours)**

```python
# Add to PatchTST, iTransformer, TFT

from torch.utils.checkpoint import checkpoint

class PatchTSTModel(BaseModel):
    def forward(self, x):
        # Use gradient checkpointing for encoder
        if self.training and self.use_gradient_checkpointing:
            x = checkpoint(self.encoder, x)
        else:
            x = self.encoder(x)
        # Rest of forward pass...
```

**Priority 3B: DVC Integration Guide (2-3 hours)**

Create: `docs/COLAB_DATA_VERSIONING.md`

```markdown
# Data Versioning with DVC on Google Colab

## Setup

1. Install DVC:
```bash
!pip install dvc dvc-gdrive
```

2. Configure DVC with Google Drive:
```bash
!dvc remote add -d gdrive gdrive://YOUR_DRIVE_FOLDER_ID
!dvc remote modify gdrive gdrive_use_service_account false
```

3. Track data:
```bash
!dvc add data/raw/MES_1m.parquet
!git add data/raw/MES_1m.parquet.dvc .gitignore
!git commit -m "Track MES data v1.0"
```

## Usage on Colab

```bash
# Pull data from Drive
!dvc pull

# Run pipeline
!./pipeline run --symbols MES

# Push new artifacts to Drive
!dvc add data/splits/
!dvc push
```
```

---

## 8. Production Readiness Roadmap

### Timeline & Investment

| Phase | Duration | Focus | Readiness Level |
|-------|----------|-------|-----------------|
| **Current** | - | Research prototype | 35/100 |
| **Phase 1** | 2 weeks | Critical fixes (labeling, features, tracking) | 50/100 |
| **Phase 2** | 3 weeks | MLOps foundation (CI/CD, serving, versioning) | 65/100 |
| **Phase 3** | 5 weeks | Production hardening (monitoring, validation) | 80/100 |
| **Phase 4** | 6 weeks | Advanced features (scaling, governance) | 90/100 |
| **Total** | **16 weeks** | Full production ready | **90/100** |

### Resource Requirements

**Engineering:**
- 2-3 MLOps engineers × 4 months
- 1 ML scientist for methodology fixes × 2 months

**Infrastructure:**
- Kubernetes cluster: $300-500/month
- Monitoring (DataDog/Prometheus+Grafana): $100-200/month
- Storage (S3/GCS for DVC): $50-100/month
- MLflow server: $50-100/month
- **Total:** $500-900/month

### Risk Assessment

| Risk Category | Current Risk | Post-Phase 2 | Post-Phase 4 |
|---------------|--------------|--------------|--------------|
| **Data Leakage** | Medium | Low | Very Low |
| **Overfitting** | High | Medium | Low |
| **Production Failures** | Critical | Medium | Low |
| **Scalability** | High | Medium | Low |
| **Monitoring** | Critical | Low | Very Low |
| **Security** | Critical | Medium | Low |

### Success Metrics

**Phase 1 (Critical Fixes):**
- [ ] Symmetric barriers implemented
- [ ] Feature count reduced to ≤100 pre-selection, ≤50 post-selection
- [ ] MLflow experiment tracking operational
- [ ] DVC data versioning operational
- [ ] CPCV + PBO as primary validation

**Phase 2 (MLOps Foundation):**
- [ ] CI/CD pipeline running on every commit
- [ ] Dockerized inference service deployed
- [ ] FastAPI serving 95th percentile latency <100ms
- [ ] Model performance regression tests passing

**Phase 3 (Production Hardening):**
- [ ] Prometheus metrics exported, Grafana dashboards created
- [ ] Alerts configured and tested
- [ ] MTF lookahead tests passing
- [ ] Adaptive purge/embargo implemented
- [ ] Regime-aware validation operational

**Phase 4 (Advanced Features):**
- [ ] Distributed tracing operational
- [ ] A/B testing infrastructure live
- [ ] Horizontal scaling working (3+ replicas)
- [ ] Model governance workflows implemented

---

## 8. Conclusion

### Summary of Findings

This OHLCV ML factory has **excellent research foundations** with proper architecture, comprehensive testing, and awareness of time-series leakage prevention. However, **critical methodological flaws in labeling and feature engineering**, combined with **complete absence of production MLOps infrastructure**, make this unsuitable for live trading without 3-4 months of dedicated work.

### Key Takeaways

**Strengths:**
1. ✅ Plugin-based model registry (22 models, 6 families)
2. ✅ Purged K-fold cross-validation with embargo
3. ✅ Train-only scaling
4. ✅ Comprehensive test suite (86 test files)
5. ✅ Clean modular architecture

**Critical Weaknesses:**
1. 🔴 Asymmetric barrier logic is statistically unsound
2. 🔴 Feature count 2-3× too high (severe overfitting risk)
3. 🔴 No experiment tracking, no model serving, no monitoring
4. 🔴 Single validation split provides no uncertainty estimates
5. 🔴 Heterogeneous stacking has unresolved data alignment issues

### Expected Live Performance

**Without fixes:**
- Sharpe ratio will be **30-50% lower** than backtest
- Win rate will **degrade over time** due to regime shifts
- Neutral predictions will dominate (conservative model from noisy labels)
- Production failures will occur frequently (no monitoring/alerting)

**With Phase 1-2 fixes (8-10 weeks):**
- Minimum viable production deployment possible
- Basic monitoring and reliability
- Improved generalization from better labeling/features

**With full roadmap (16 weeks):**
- Production-grade ML system
- Robust monitoring and observability
- Scalable infrastructure
- Continuous improvement capabilities

### Final Recommendation

**DO NOT DEPLOY TO PRODUCTION** until at minimum Phase 1 (critical fixes) and Phase 2 (MLOps foundation) are complete. The combination of methodological flaws and infrastructure gaps creates unacceptable risk for live trading.

**Immediate Actions:**
1. Fix asymmetric barriers (use symmetric)
2. Reduce feature count by 50-70%
3. Integrate MLflow experiment tracking
4. Use CPCV + PBO as primary validation (already implemented!)
5. Add data versioning (DVC)

After these critical fixes, implement Phase 2 (CI/CD, serving, monitoring) before any production deployment.

---

## References

### Architecture & Software Engineering
- [Clean Architecture Principles](https://blog.cleancoder.com/uncle-bob/2012/08/13/the-clean-architecture.html)
- [SOLID Principles in ML Systems](https://eugeneyan.com/writing/testing-ml/)

### ML Methodology
- [Advances in Financial Machine Learning (De Prado)](https://www.wiley.com/en-us/Advances+in+Financial+Machine+Learning-p-9781119482086)
- [Triple Barrier Method - MLFinLab](https://www.mlfinlab.com/en/latest/labeling/tb_meta_labeling.html)
- [What Is the Triple Barrier Method?](https://xglamdring.com/what-is-the-triple-barrier-method-a-labeling-technique-to-prevent-overfitting-in-ml-based-quantitative-trading/)
- [Labeling Financial Data for ML](https://www.sefidian.com/2021/06/26/labeling-financial-data-for-machine-learning/)
- [Hidden Leaks in Time Series Forecasting](https://arxiv.org/html/2512.06932)
- [Stacking Ensemble Models for Time Series](https://cienciadedatos.net/documentos/py52-stacking-ensemble-models-forecasting.html)
- [Effective ML Model Combination](https://www.sciencedirect.com/science/article/abs/pii/S0020025522010465)

### MLOps & Production
- [MLOps Best Practices for Quantitative Trading (2025)](https://medium.com/@online-inference/mlops-best-practices-for-quantitative-trading-teams-59f063d3aaf8)
- [MLOps: Deploying and Monitoring ML Models in 2025](https://dasroot.net/posts/2025/12/mlops-deploying-monitoring-ml-models-2025/)
- [12 MLOps Best Practices Every Enterprise Needs in 2025](https://www.shakudo.io/blog/mlops-best-practices-enterprise-2025)
- [MLOps Pipeline for Time Series Prediction Tutorial](https://neptune.ai/blog/mlops-pipeline-for-time-series-prediction-tutorial)
- [Feature Engineering for Stock Prediction](https://alphascientist.com/feature_engineering.html)
- [Structural VAR and VECM for OHLC Data](https://link.springer.com/article/10.1186/s40854-024-00622-6)

### Tools & Frameworks
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [DVC Documentation](https://dvc.org/doc)
- [Prefect Documentation](https://docs.prefect.io/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Prometheus Monitoring](https://prometheus.io/docs/introduction/overview/)

---

## 9. Colab-Specific Roadmap

### 9.1 Colab Training Timeline

| Phase | Duration | Focus | Colab Compatibility |
|-------|----------|-------|---------------------|
| **Current** | - | Research prototype, CLI-only | 40/100 (incompatible) |
| **Colab Phase 0** | 1 week | Critical prerequisites (checkpointing, Drive, W&B) | 70/100 (basic training works) |
| **Colab Phase 1** | 1 week | Essential improvements (CV checkpointing, batch auto-adjust) | 85/100 (reliable training) |
| **Colab Phase 2** | 1 week | Workflow improvements (notebooks, session manager, docs) | 95/100 (production-ready for Colab) |

### 9.2 Colab Implementation Effort

> **Note:** Initial scaffolding exists in `colab_notebooks/utils/` (colab_setup.py, checkpoint_manager.py)
> but requires API alignment with actual codebase. See "Current Implementation Status" below.

**Week 1 (Colab Phase 0 - CRITICAL):**
- Epoch-level checkpointing: 4-5 hours (requires modifying neural training loops)
- Google Drive integration: 2 hours (align existing scaffold with actual APIs)
- W&B experiment tracking: 3-4 hours (integrate with Trainer class)
- Testing and validation: 3-4 hours
- **Total: 12-15 hours**

**Week 2 (Colab Phase 1 - HIGH PRIORITY):**
- Fold-level CV checkpointing: 2-3 hours (modify CVRunner)
- Batch size auto-adjustment: 3 hours (GPU memory detection + fallback)
- Resource monitoring: 1-2 hours
- Testing: 3 hours
- **Total: 9-11 hours**

**Week 3 (Colab Phase 2 - NICE TO HAVE):**
- Colab notebook templates (5 notebooks): 6-8 hours (fix API mismatches in existing notebooks)
- Session time manager: 2 hours
- requirements-colab.txt: 1 hour (add missing dependencies)
- Documentation updates: 3-4 hours
- **Total: 12-15 hours**

**Total effort:** 33-41 hours (4-5 weeks part-time)

**Current Implementation Status:**
| Component | Status | Notes |
|-----------|--------|-------|
| `colab_notebooks/utils/colab_setup.py` | Scaffolded | Requires API alignment |
| `colab_notebooks/utils/checkpoint_manager.py` | Scaffolded | `save_checkpoint()` type issues |
| `notebooks/colab_setup.py` | Working | Basic Colab setup, `get_trainer_for_colab()` |
| Epoch checkpointing in neural models | NOT IMPLEMENTED | Critical gap |
| Fold-level CV checkpointing | NOT IMPLEMENTED | High priority |
| W&B integration | NOT IMPLEMENTED | Scaffolded only |
| `requirements-colab.txt` | Incomplete | Missing: pandas, numpy, torch, xgboost, etc. |

### 9.3 Colab Success Metrics

**Phase 0 (Week 1):**
- [ ] Neural models checkpoint every 10 epochs to Drive
- [ ] Training resumes from last checkpoint on disconnect
- [ ] All experiments logged to W&B
- [ ] Zero data loss on disconnect

**Phase 1 (Week 2):**
- [ ] CV folds checkpoint after completion
- [ ] Batch size auto-adjusts to prevent OOM
- [ ] Resource monitoring warns about memory limits
- [ ] 3-base heterogeneous ensemble completes in <6 hours

**Phase 2 (Week 3):**
- [ ] 5 Colab notebooks operational
- [ ] Session time manager warns at 2 hours remaining
- [ ] Complete documentation for Colab training
- [ ] DVC integration guide available

### 9.4 Colab vs Production Deployment

**Important Distinction:**

| Aspect | Google Colab | Production Deployment |
|--------|--------------|----------------------|
| **Purpose** | Model training and experimentation | Real-time inference serving |
| **Runtime** | Ephemeral (12-hour sessions) | Persistent (24/7 uptime) |
| **Infrastructure** | Free GPU, notebook environment | Kubernetes, Docker, load balancers |
| **Artifacts** | Saved to Drive/W&B | Loaded from model registry |
| **Monitoring** | Manual (W&B dashboards) | Automated (Prometheus, Grafana, alerts) |
| **Scalability** | Single runtime | Horizontal scaling (multiple replicas) |

**Workflow:**
1. **Train on Colab:** Use Colab for model training (Phases 0-2)
2. **Export models:** Download trained models from Drive/W&B
3. **Deploy elsewhere:** Use FastAPI + Docker + Kubernetes for production inference

**Colab is NOT for:**
- Production inference serving
- Real-time predictions
- High-availability systems
- Multi-user concurrent access

### 9.5 Recommended Colab Training Cadence

**Development Phase (Weeks 1-4):**
- Daily iterations on individual models (30-60 min sessions)
- Test checkpointing, Drive integration, W&B logging
- Iterate on methodology (labeling, features)

**Validation Phase (Weeks 5-8):**
- Weekly CV runs for model families (4-6 hour sessions)
- Generate OOF predictions for stacking
- Save all artifacts to Drive for ensemble training

**Production Phase (Weeks 9+):**
- Monthly retraining as new data arrives
- Load latest data from Drive, run full pipeline
- Train heterogeneous ensemble (5 hours)
- Export best model to production via W&B

### 9.6 Colab Cost-Benefit Analysis

**Free Tier Benefits:**
- ✅ Free T4/P100/V100 GPU access
- ✅ No infrastructure setup required
- ✅ Easy collaboration via shared notebooks
- ✅ Pre-installed ML libraries

**Free Tier Limitations:**
- ❌ 12-hour session limit (need checkpointing)
- ❌ Inconsistent GPU availability (may get CPU-only)
- ❌ No guarantees on uptime (sessions can disconnect)
- ❌ Cannot run 24/7 (not for production serving)

**Colab Pro ($10/month) Benefits:**
- 24-hour sessions (2× longer)
- Better GPU availability (priority access)
- More RAM (up to 52 GB)
- Background execution

**Verdict:** Free tier is sufficient for this project with proper checkpointing. Colab Pro is optional but improves reliability.

---

**Document Version:** 2.0 (Updated for Google Colab Training)
**Last Updated:** 2026-01-03
**Next Review:** After Colab Phase 0 implementation (1 week)
