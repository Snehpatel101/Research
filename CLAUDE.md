# ML Model Factory for OHLCV Time Series

## Goal

Keep the codebase modular, readable, and easy to extend as we build a **model factory** that can train, evaluate, and compare ANY model type on OHLCV bar data.

This is not a single pipeline — it's a **factory** with a plugin architecture.

---

## Pipeline Architecture

The ML factory implements a **single unified pipeline** that ingests canonical OHLCV data and deterministically derives model-specific representations:

**Data Flow:**
```
Raw 1-min OHLCV (canonical - single source of truth)
  ↓
[MTF Upscaling] → ⚠️ 5 of 9 timeframes implemented (partial - see Phase 2 status)
  Currently: 15m, 30m, 1h, 4h, daily
  Planned: 1m, 5m, 10m, 15m, 20m, 25m, 30m, 45m, 1h
  ↓
EACH base model independently chooses (configurable per-model):
  • Primary training TF (e.g., CatBoost→15min, TCN→5min, PatchTST→1min)
  • MTF strategy (single-TF / MTF indicators / MTF ingestion)
  • Which other TFs to use for enrichment/multi-stream
  ↓
All models derive features from same 1-min canonical OHLCV source
  ↓
Feature Engineering (~180 indicators + wavelets + microstructure)
  ↓
Triple-Barrier Labeling (Optuna-optimized)
  ↓
Model-Family Adapters
  ├─ Tabular (2D): XGBoost, LightGBM, CatBoost, RF, Logistic, SVM
  ├─ Sequence (3D): LSTM, GRU, TCN, Transformer
  └─ Multi-Res (4D): Planned for advanced models
  ↓
Training (single models + heterogeneous ensembles via meta-learner stacking)
  ↓
Standardized Artifacts (models, predictions, metrics)
```

**Key Architectural Points:**
- **ONE Canonical Source:** Single 1-min OHLCV dataset → ⚠️ 5 of 9 timeframes implemented (partial)
- **Per-Model Timeframe Configuration:** EACH base model independently chooses its primary training timeframe:
  - CatBoost trains on 15min (derived from 1-min canonical)
  - TCN trains on 5min (derived from 1-min canonical)
  - PatchTST trains on 1min (uses 1-min canonical directly)
  - All configurable per-model, all from same source
- **SAME Underlying Data:** All models see same timestamps, same target labels, same train/val/test splits
- **DIFFERENT Feature Sets (Per-Model Feature Selection):** Each base model gets features tailored to its inductive biases:
  - **Tabular (CatBoost):** 15min primary TF + MTF indicators from 1m/5m/1h → ~200 engineered features (indicators, wavelets, MTF indicators)
  - **Sequence (TCN):** 5min primary TF, single-TF (no MTF) → ~150 base features in 3D windows (indicators, wavelets, raw price features)
  - **Transformer (PatchTST):** Multi-stream MTF ingestion (1m+5m+15m raw OHLCV) → 3 streams × 4 OHLC (no engineered features, model learns from raw data)
- **MTF Mix-and-Match:** Each model chooses its MTF strategy independently (single-TF / MTF indicators / MTF ingestion)
- **All Derived from 1-min:** Every timeframe (5m, 10m, 15m, 1h) is resampled from the canonical 1-min OHLCV
- **Heterogeneous Ensembles:** 3-4 base families (different TFs, different features) → 1 meta-learner
- **Direct Stacking:** Meta-learner trained on OOF predictions from heterogeneous bases

**Implementation Status:**
- Phases 1-6: Complete (13 base models + 4 meta-learners across 5 families)
- Phase 7: ⚠️ PLANNED (heterogeneous ensemble training script not yet implemented)
- MTF: ⚠️ PARTIAL (5 of 9 timeframes implemented - 15m, 30m, 1h, 4h, daily)

**Documentation:** See `docs/ARCHITECTURE.md` and `docs/phases/` for comprehensive guides.

---

## Single-Contract Architecture

**This is a single-contract ML factory. Each contract is trained in complete isolation. No cross-symbol correlation or feature engineering.**

### Key Principles

1. **One contract at a time** - The pipeline processes and trains models for exactly one futures contract per run
2. **Complete isolation** - No features, labels, or data from other contracts influence the model
3. **Symbol configurability** - Easy to switch between MES, MGC, or other contracts via configuration

### Symbol Configuration

**Specify the contract to train:**

```python
# In config or CLI
symbol = "MES"  # or "MGC", "ES", "GC", etc.
```

**Data path resolution:**
- Raw data: `data/raw/{symbol}_1m.parquet` or `data/raw/{symbol}_1m.csv`
- Processed: `data/splits/scaled/` (contains single-symbol data after pipeline)
- Models: `experiments/runs/{run_id}/` (trained on single symbol)

**Switching contracts:**
```bash
# Train on MES
./pipeline run --symbols MES

# Train on MGC (separate run, separate model)
./pipeline run --symbols MGC
```

**Multi-symbol processing is blocked by default** (`allow_batch_symbols=False` in PipelineConfig). Each symbol requires its own pipeline run and produces its own trained model.

---

## OHLCV ML Modeling: Factory Pattern

We are building an **ML Model Factory** for OHLCV time series. The factory can train any model family (boosting, neural, transformers, classical ML, ensembles) using:

1. **Unified Data Pipeline** - One 1-min canonical OHLCV → Per-model feature selection (different models get different features tailored to their inductive biases)
2. **Plugin-Based Model Registry** - Add new model types without rewriting pipelines
3. **Unified Evaluation Framework** - Compare models using identical metrics
4. **Ensemble Support Built-In** - Combine multiple models into meta-learners

### Factory Architecture Principles

```
Raw OHLCV → [ Data Pipeline ] → Standardized Datasets
                                       ↓
                            [ Model Registry Plugin System ]
                            ├── XGBoost Trainer
                            ├── LSTM Trainer
                            ├── Transformer Trainer
                            ├── Random Forest Trainer
                            └── Ensemble Meta-Learner
                                       ↓
                            [ Unified Evaluation Engine ]
                                       ↓
                          Trained Models + Performance Reports
```

### Data Pipeline Details

**Configurable Primary Timeframe:**
- ⚠️ PLANNED: Primary training timeframe configurable per experiment
- Current: Hardcoded to 5-min base timeframe
- Intended: 9-TF ladder (1m, 5m, 10m, 15m, 20m, 25m, 30m, 45m, 1h)
- Status: Only 5 TFs implemented (15m, 30m, 1h, 4h, daily)

**MTF Enrichment (Optional):**
- **Strategy 1: Single-TF** - Train on primary timeframe only, no MTF features
- **Strategy 2: MTF Indicators** - Add indicator features from multiple timeframes to primary TF
- **Strategy 3: MTF Ingestion** - Raw OHLCV bars from multiple timeframes for sequence models

**Unified Pipeline (7 Phases):**

| Phase | Name | Description | Status |
|:-----:|------|-------------|:------:|
| 1 | Ingestion | Load and validate raw OHLCV | ✅ Complete |
| 2 | MTF Upscaling | Multi-timeframe resampling (optional) | ⚠️ Partial (5/9 TFs) |
| 3 | Features | 180+ indicator features | ✅ Complete |
| 4 | Labeling | Triple-barrier + Optuna | ✅ Complete |
| 5 | Adapters | Model-family data preparation | ✅ Complete |
| 6 | Training | 17 models, 5 families | ✅ Complete |
| 7 | Stacking | Heterogeneous ensemble training | ⚠️ PLANNED |

**Data Shapes by Model Family:**
- **Tabular models** (Boosting + Classical): 2D arrays `(n_samples, ~180)`
- **Sequence models** (Neural): 3D windows `(n_samples, seq_len, ~180)`
- **Advanced models** (Planned): 4D multi-resolution `(n_samples, n_timeframes, seq_len, 4)`

**See:** `docs/ARCHITECTURE.md` for comprehensive architecture documentation

### Core Contracts

**Data Pipeline (Phases 1-5):**
- **Phase 1:** Ingest raw 1-min OHLCV (canonical - single source of truth)
- **Phase 2:** Multi-timeframe upscaling
  - ⚠️ PARTIAL: Only 5 of 9 timeframes implemented (15m, 30m, 1h, 4h, daily)
  - Planned: Full 9-TF ladder (1m, 5m, 10m, 15m, 20m, 25m, 30m, 45m, 1h)
  - All timeframes resampled from same 1-min source
- **Phase 3:** Feature engineering (~150 base indicators + wavelets + microstructure)
- **Phase 4:** Triple-barrier labeling with Optuna optimization (same labels for all models)
- **Phase 5:** Model-family adapters (2D tabular, 3D sequence, 4D multi-res planned)

**Then EACH base model independently configures:**
- **Primary training TF:** Which timeframe to train on (configurable per-model)
  - Example: CatBoost→15min, TCN→5min, PatchTST→1min
  - All derived from same 1-min canonical OHLCV
- **MTF strategy:** How to use other timeframes (configurable per-model)
  - single-TF (no MTF), MTF indicators (add features), or MTF ingestion (multi-stream)
- **Which TFs for enrichment:** Which other timeframes to include (flexible per-model)

**Example - Heterogeneous Ensemble (same 1-min source, different configurations):**
- **CatBoost:** Primary=15min (from 1-min) + MTF indicators from 1m/5m/1h → ~200 features
- **TCN:** Primary=5min (from 1-min), single-TF, no MTF → ~150 features
- **PatchTST:** Primary=1min (canonical) + multi-stream ingestion 1m+5m+15m (all from 1-min source) → 3 streams × 4 OHLC

**Leakage Prevention:**
- Proper purging (60) + embargo (1440)
- MTF features use shift(1)
- Train-only scaling
- Time-series aware splits (70/15/15)

**Model Contract (Phase 2 - Complete):**
```python
class BaseModel(ABC):
    @abstractmethod
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        sample_weights: Optional[np.ndarray] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> TrainingMetrics:
        """Train the model on provided data."""
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> PredictionOutput:
        """Generate predictions with probabilities and confidence."""
        pass

    @abstractmethod
    def save(self, path: Path) -> None:
        """Persist trained model."""
        pass
```

**Evaluation Contract:**
- Same backtest assumptions for all models
- Identical metrics: Sharpe, win rate, max drawdown, transaction costs
- Regime-aware performance breakdown
- Quality-weighted evaluation

### Plugin Registration

Adding a new model type should be **trivial**:

```python
from src.models import BaseModel, register

@register(name="my_model", family="boosting")
class MyModel(BaseModel):
    # Implement BaseModel: fit/predict/save/load (+ config properties)
    ...
```

Then use it:
```bash
# Phase 1 (data)
./pipeline run --symbols MES

# Phase 2 (training)
python scripts/train_model.py --model my_model --horizon 20
```

### Ensemble Support

The factory supports both **single models** and **ensembles**:

```bash
# Train individual models (Phase 2)
python scripts/train_model.py --model xgboost --horizon 20
python scripts/train_model.py --model lstm --horizon 20 --seq-len 60

# Train an ensemble from scratch (Phase 2)
# Note: All base models must be from the same family (tabular or sequence)
python scripts/train_model.py --model voting --horizon 20 --base-models xgboost,lightgbm,catboost
python scripts/train_model.py --model voting --horizon 20 --base-models lstm,gru,tcn
```

### Model Families (17 Implemented + 6 Planned = 23 Total)

| Family | Models | Data Format | Strengths | Status |
|--------|--------|-------------|-----------|--------|
| **Boosting** (3) | XGBoost, LightGBM, CatBoost | 2D tabular | Fast, interpretable, feature interactions | ✅ Complete |
| **Neural** (4) | LSTM, GRU, TCN, Transformer | 3D sequences | Temporal dependencies, sequential patterns | ✅ Complete |
| **Classical** (3) | Random Forest, Logistic, SVM | 2D tabular | Robust baselines, interpretable | ✅ Complete |
| **Ensemble** (3) | Voting, Stacking, Blending | OOF predictions | Same-family ensemble methods | ✅ Complete |
| **Inference/Meta** (4) | Ridge Meta, MLP Meta, Calibrated Meta, XGBoost Meta | OOF predictions | Meta-learner stacking from heterogeneous bases | ✅ Complete |
| **CNN** (2) | InceptionTime, 1D ResNet | 3D sequences | Multi-scale pattern detection, deep residual learning | ⚠️ Planned |
| **Advanced Transformers** (3) | PatchTST, iTransformer, TFT | 3D sequences | SOTA long-term forecasting, interpretable attention | ⚠️ Planned |
| **MLP** (1) | N-BEATS | 3D sequences | Interpretable decomposition, M4 winner | ⚠️ Planned |

**Family Classification:**
- **Tabular models** (Boosting + Classical): 2D arrays `(n_samples, n_features)` - 6 models
- **Sequence models** (Neural + CNN + Advanced + MLP): 3D windows `(n_samples, seq_len, n_features)` - 4 implemented + 6 planned
- **Ensemble models**: Same-family ensemble methods - 3 models (Voting, Stacking, Blending)
- **Inference/Meta models**: OOF predictions from heterogeneous bases - 4 meta-learners

**Inference/Meta Family Details:**
| Meta-Learner | Method | Use Case |
|--------------|--------|----------|
| **Ridge Meta** | L2-regularized Ridge classifier | Fast linear stacking |
| **MLP Meta** | Multi-layer perceptron | Learned non-linear blending |
| **Calibrated Meta** | Isotonic/Platt calibration | Calibrated probability scaling |
| **XGBoost Meta** | Gradient boosted meta-learner | Non-linear feature interactions |

**See:** `docs/roadmaps/ADVANCED_MODELS_ROADMAP.md` for 6 planned models (14-18 days implementation)

**Registry:** Models register via the `@register(...)` decorator for automatic discovery.

### Heterogeneous Ensemble Architecture

The factory supports **heterogeneous ensembles** where base models from different families train on **different timeframes and features**, all derived from the **same 1-min canonical OHLCV source**, then feed a single meta-learner via OOF stacking.

**Architecture:**
```
1-min OHLCV Canonical Source (single source of truth)
       ↓
Derive 5 of 9 timeframes (currently: 15m, 30m, 1h, 4h, daily)
       ↓
Base Model Selection (1 per family, EACH chooses its own TF + features)
  |-- Tabular: CatBoost → 15min + MTF indicators (from 1-min source)
  |-- CNN/TCN: TCN → 5min, single-TF (from 1-min source)
  |-- Transformer: PatchTST → 1min + multi-stream 1m+5m+15m (all from 1-min source)
  |-- Optional: Ridge → 1h, single-TF (from 1-min source)
       ↓
OOF Generation (PurgedKFold with purge/embargo, same splits for all)
       |
       v
Meta-Learner Training (Logistic/Ridge/MLP on OOF predictions)
       |
       v
Full Retrain (base models on full train set)
       |
       v
Test Evaluation (meta-learner combines base predictions)
```

**Why Heterogeneous > Homogeneous:**
- **Diversity of Inductive Biases:** Different model families capture different patterns
- **Reduced Correlation:** Errors from diverse models are less correlated
- **Robustness:** No single family's weakness dominates

**Recommended Base Model Configurations:**

| Configuration | Base Models | Meta-Learner | Use Case |
|---------------|-------------|--------------|----------|
| **3 Bases (Standard)** | CatBoost + TCN + PatchTST | Logistic | Balanced diversity |
| **4 Bases (Maximum)** | LightGBM + TCN + TFT + Ridge | Ridge | Maximum diversity |
| **2 Bases (Minimal)** | XGBoost + LSTM | Logistic | Fast prototyping |

**Training Protocol:**
1. **Generate OOF predictions:** Run PurgedKFold on each base model
2. **Stack OOF predictions:** Concatenate OOF probabilities as meta-features
3. **Train meta-learner:** Fit Logistic/Ridge/MLP on stacked OOF
4. **Full retrain:** Retrain all base models on complete training set
5. **Test evaluation:** Base models predict test set, meta-learner combines

**CLI Usage:**
```bash
# ⚠️ TODO: Heterogeneous ensemble training script not yet implemented
# Planned: scripts/train_ensemble.py for automated heterogeneous stacking

# Current workaround: Train base models individually, then manually stack
python scripts/train_model.py --model catboost --horizon 20
python scripts/train_model.py --model tcn --horizon 20 --seq-len 60
# ... then use stacking model for manual meta-learner training
```

See `docs/implementation/PHASE_7_META_LEARNER_STACKING.md` for planned implementation.

---

## Engineering Rules (Non-Negotiables)

### Architecture and Modularity
We do not build monoliths. Responsibilities must be split into small, composable modules with clear contracts and minimal coupling. Each module should do one thing well and expose a narrow, well-documented interface. Prefer dependency injection and explicit wiring over hidden globals or implicit side effects.

### File and Complexity Limits
Files should target **650 lines** as the ideal maximum. Files up to **800 lines** are acceptable if the logic is cohesive and cannot be reasonably split without introducing artificial abstractions. Beyond 800 lines is a signal that boundaries are wrong and responsibilities need to be refactored. Keep functions short, keep layers separated, and keep the cognitive load low.

### Fail Fast, Fail Hard
We would rather crash early than silently continue in an invalid state. Inputs are validated at the boundary. Assumptions are enforced with explicit checks. If something is wrong, we stop and surface a clear error message that points to the cause.

### Less Code is Better
Simpler implementations win. Prefer straightforward, boring solutions over clever abstractions. Avoid premature generalization. If a feature can be expressed with fewer moving parts, do that. Complexity must earn its place.

### Delete Legacy Code (If Unused, Remove It)
Legacy code is debt. If a file, function, or feature is not used, not referenced by any active code path, and not needed for the next planned milestone, delete it.

- Prefer deletion over commenting-out or leaving "dead" branches around.
- Remove unused imports, stale utilities, and orphaned tests/docs along with the code.
- If you're unsure whether something is needed, prove it's used (ripgrep call sites, run the feature, confirm tests). If you can't prove it, delete it.
- Git history is the archive — do not keep code "just in case."

### Concise Output and Minimal Documentation
Default to concise answers unless explicitly asked to expand.

Not every agent action needs a document. Write documentation only when it is needed for:
- An end-of-pass summary (what changed, what remains, and next steps)
- A decision/contract other work will depend on (schemas, interfaces, invariants)
- Investigation artifacts others must reuse (repro steps, evidence, links, key findings)

Otherwise, keep notes brief and inline (PR description, issue comment, short checklist).

### No Exception Swallowing
Do not paper over failures with try/except. We do not swallow errors or "recover" by guessing. Use explicit validation, explicit return types, and explicit preconditions. If a dependency can fail, make that failure visible in the function contract and test it. Exceptions are allowed to propagate naturally so failures are obvious and diagnosable.

### Clear Validation
Every boundary validates what it receives: configuration, CLI inputs, dataset schemas, feature matrices, labels, and model parameters. Validation errors must be actionable, specific, and consistent. Prefer schema-based validation and typed structures over ad hoc checks.

### Clear Tests
Every module ships with tests that prove the contract. Unit tests cover pure logic. Integration tests cover pipeline wiring and data flow. Regression tests lock down previously fixed issues. Tests should be deterministic, fast, and easy to run locally and in CI.

### Definition of Done
A change is complete only when:
- Implementation is modular
- Stays within file limits (target 650 lines, max 800 lines)
- Validates inputs at boundaries
- Backed by tests that clearly demonstrate correctness

---

## Auto-Activated Agents

These agents trigger **automatically** based on context - no manual invocation needed:

| When You're Doing... | Agent Auto-Activates | What It Does |
|---------------------|---------------------|--------------|
| Building pipeline stages | `ml-engineer` | Creates DAG-based ML workflows |
| Writing feature code | `data-engineer` | Spark optimization, data pipelines |
| Designing labeling logic | `quant-analyst` | Trading strategies, risk metrics |
| Creating data classes | `python-pro` | Modern Python patterns, Pydantic |
| Adding tests | `tdd-orchestrator` | Red-green-refactor cycles |
| Debugging issues | `debugger` | Error investigation |
| Optimizing performance | `performance-engineer` | Profiling, caching |

---

## Build Commands

```bash
# Build new pipeline stage
/ml-pipeline-workflow "create stage for [description]"

# Build feature engineering
/data-engineering:spark-optimization "optimize feature calculation for [task]"

# Build labeling system
/quantitative-trading:quant-analyst "implement triple-barrier labeling"

# Build validation
/tdd-workflows:tdd-cycle "add validation for [component]"
```

---

## Sequential Build Flow

When building new functionality, agents chain automatically:

```
You say: "Build a new resampling stage"
         ↓
    ml-engineer activates (pipeline design)
         ↓
    data-engineer activates (implementation)
         ↓
    python-pro activates (code patterns)
         ↓
    tdd-orchestrator activates (tests)
```

---

## Context Auto-Save

Context saves automatically at:
- Stage completion → `runs/{run_id}/artifacts/pipeline_state.json`
- Checkpoint → `_save_state()` in PipelineRunner

Restore: `/context-restore --project research --mode full`

---

## Factory Data Pipeline (Phase 1 - Complete)

The data pipeline produces standardized datasets for all model types:

```
src/phase1/stages/
├── ingest/             → Load and validate raw data
├── clean/              → Resample 1min→5min, gap handling
├── sessions/           → Session filtering and normalization
├── features/           → 150+ indicators (momentum, wavelets, microstructure)
├── regime/             → Regime detection (volatility, trend, composite)
├── mtf/                → Multi-timeframe indicator features (~30 MTF features from 5 timeframes; intended: 9 timeframes)
├── labeling/           → Triple-barrier initial labels
├── ga_optimize/        → Optuna parameter optimization
├── final_labels/       → Apply optimized parameters
├── splits/             → Train/val/test with purge/embargo
├── scaling/            → Train-only robust scaling
├── datasets/           → Build TimeSeriesDataContainer
├── scaled_validation/  → Validate scaled data quality
├── validation/         → Feature correlation and quality checks
└── reporting/          → Generate completion reports
```

**Output:** Model-specific features based on per-model feature selection:
- Tabular models: ~200 engineered features (base indicators + MTF indicators)
- Sequence models: ~150 base features (indicators + wavelets, single-TF)
- Advanced models (planned): Raw multi-stream OHLCV bars (no pre-engineering)

---

## Model Factory (Phase 6 - Complete)

Plugin-based model training system with **17 models across 5 families**:

```
src/models/
├── registry.py         → ModelRegistry plugin system (17 models registered)
├── base.py             → BaseModel interface, TrainingMetrics, PredictionOutput
├── config/             → Configuration package (modular)
│   ├── trainer_config.py  → TrainerConfig dataclass
│   ├── loaders.py         → YAML config loading
│   ├── paths.py           → Config file paths
│   └── environment.py     → Environment detection
├── trainer.py          → Unified training orchestration
├── metrics.py          → Metric calculation utilities
├── data_preparation.py → Dataset preparation utilities
├── device.py           → GPU detection, memory estimation
├── boosting/           → XGBoost, LightGBM, CatBoost (3 models)
├── neural/             → LSTM, GRU, TCN, Transformer (4 models)
├── classical/          → Random Forest, Logistic, SVM (3 models)
└── ensemble/           → Voting, Stacking, Blending (3 models)
```

**Output:** Trained models + unified performance reports

**17 models available:**
- **Base (10):** `xgboost`, `lightgbm`, `catboost`, `lstm`, `gru`, `tcn`, `transformer`, `random_forest`, `logistic`, `svm`
- **Ensemble (3):** `voting`, `stacking`, `blending`
- **Meta-learners (4):** `ridge_meta`, `mlp_meta`, `calibrated_meta`, `xgboost_meta`

---

## Cross-Validation (Phase 3 - Complete)

Time-series aware cross-validation with purge/embargo:

```
src/cross_validation/
├── purged_kfold.py     → PurgedKFold with configurable purge/embargo
├── feature_selector.py → Walk-forward MDA/MDI feature selection
├── oof_generator.py    → Unified OOF generator interface
├── oof_core.py         → Core tabular OOF generation
├── oof_sequence.py     → Sequence model OOF generation
├── oof_stacking.py     → Stacking dataset builder
├── oof_validation.py   → Coverage and correlation validation
├── oof_io.py           → Save/load OOF datasets
├── cv_runner.py        → CrossValidationRunner, Optuna tuning
└── param_spaces.py     → Hyperparameter search spaces
```

**Output:** CV results, OOF predictions, stacking datasets

---

## Quick Commands

```bash
# Run data pipeline (Phase 1)
./pipeline run --symbols MGC

# Train specific model (Phase 6)
python scripts/train_model.py --model xgboost --horizon 20
python scripts/train_model.py --model lstm --horizon 20 --seq-len 30
python scripts/train_model.py --model random_forest --horizon 20

# Train ensemble (Phase 6 - same-family only)
python scripts/train_model.py --model voting --horizon 20 --base-models xgboost,lightgbm,catboost
python scripts/train_model.py --model stacking --horizon 20 --base-models lstm,gru,tcn --seq-len 30

# ⚠️ TODO: Heterogeneous ensemble training (Phase 7 - not yet implemented)
# Planned: scripts/train_ensemble.py --base-models catboost,tcn,patchtst --meta-learner ridge_meta --horizon 20

# Run cross-validation (Phase 3)
python scripts/run_cv.py --models xgboost --horizons 20 --n-splits 5
python scripts/run_cv.py --models all --horizons 5,10,15,20 --tune

# List available models (should print 17)
python scripts/train_model.py --list-models
python -c "from src.models import ModelRegistry; print(len(ModelRegistry.list_all()))"
```

---

## Key Params

```python
# Single contract per run (no cross-symbol features or correlation)
SYMBOL = 'MES'  # or 'MGC' - one symbol per pipeline run
LABEL_HORIZONS = [5, 10, 15, 20]  # All supported horizons
TRAIN/VAL/TEST = 70/15/15
# Purge/embargo are auto-scaled from max horizon:
# PURGE_BARS = max_horizon * 3 = 60 bars (prevents label leakage)
# EMBARGO_BARS = 1440 bars (~5 days at 5-min for serial correlation)
```

---

## Implementation Summary

**Data Pipeline (Phases 1-5):**
- Triple-barrier labeling with symbol-specific asymmetric barriers
- Optuna-based parameter optimization with transaction cost penalties
- Proper purge (60) and embargo (1440) for leakage prevention
- Quality-based sample weighting (0.5x-1.5x)
- **Per-Model Feature Selection:** Different models get different features based on inductive biases
  - Tabular models: ~200 engineered features (base indicators + MTF indicators from 5 TFs)
  - Sequence models: ~150 base features (indicators + wavelets, single primary TF)
  - Advanced models (planned): Raw multi-stream OHLCV bars from multiple TFs
- TimeSeriesDataContainer for unified model training interface (2D for tabular, 3D for sequence)

**Models (Phase 6):**
- 17 models implemented across 5 families (Boosting, Neural, Classical, Ensemble, Meta-learners)
- Plugin-based model registry with `@register` decorator
- 3 ensemble methods: Voting, Stacking, Blending (same-family)
- 4 meta-learners: Ridge Meta, MLP Meta, Calibrated Meta, XGBoost Meta

**Roadmap:**
- Phase 2 extension: 9-timeframe ladder (4 additional timeframes to complete)
- Phase 7: Heterogeneous ensemble training (scripts/train_ensemble.py implementation)
- Phase 8: Advanced meta-learners (regime-aware, adaptive)
- 6 advanced models: InceptionTime, 1D ResNet, PatchTST, iTransformer, TFT, N-BEATS

**Performance expectations:** Do not treat any Sharpe/win-rate targets as "built-in". Measure performance empirically via `scripts/run_cv.py`, `scripts/run_walk_forward.py`, and `scripts/run_cpcv_pbo.py` on your own data/cost assumptions.
