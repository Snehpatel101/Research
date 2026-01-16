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
[MTF Upscaling] → ✅ 9 intraday timeframes (1m, 5m, 10m, 15m, 20m, 25m, 30m, 45m, 1h)
  Full 9-TF ladder available: 1m, 5m, 10m, 15m, 20m, 25m, 30m, 45m, 1h
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
  └─ Neural (3D/4D): LSTM, GRU, TCN, Transformer, PatchTST, iTransformer, TFT, N-BEATS, InceptionTime, ResNet1D
  ↓
Training (single models + heterogeneous ensembles via meta-learner stacking)
  ├─ Walk-forward validation (expanding/sliding windows)
  ├─ Regime-aware training (volatility/trend/composite regimes)
  └─ Meta-labeling (Lopez de Prado methodology)
  ↓
Standardized Artifacts (models, predictions, metrics)
```

**Key Architectural Points:**
- **UnifiedConfig:** Single source of truth for all configuration (consolidates GlobalConfig, MLConfig, TrainerConfig, PipelineConfig)
- **Phase Registry:** Automatic dependency tracking between pipeline phases with state management
- **ONE Canonical Source:** Single 1-min OHLCV dataset → ✅ 9 intraday timeframes implemented
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
- **Memory Management:** Automatic caching, cleanup, and OOM recovery for large datasets

**Implementation Status:**
- Phases 1-6: Complete (19 base models + 4 meta-learners = 23 models across 6 families)
- Phase 7: ✅ Complete (heterogeneous stacking in trainer.py implemented)
- MTF Stage 2: ✅ Complete (9 intraday timeframes: 1m, 5m, 10m, 15m, 20m, 25m, 30m, 45m, 1h)
- MTF Stages 3-6: ✅ Complete (all stages iterate over effective_output_timeframes; multi-TF enabled via --process-all-timeframes)
- Training Modes: ✅ Complete (walk-forward, regime-aware, meta-labeling)
- State Management: ✅ Complete (robust phase tracking with versioning and rollback)

**Documentation:** See `docs/ARCHITECTURE.md` and `docs/implementation/` for comprehensive guides.

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
- ✅ Complete: Primary training timeframe configurable per experiment
- Default: 9 intraday timeframes (1m, 5m, 10m, 15m, 20m, 25m, 30m, 45m, 1h)
- Full 9-TF ladder available: 1m, 5m, 10m, 15m, 20m, 25m, 30m, 45m, 1h
- Status: All timeframes implemented and configurable

**MTF Enrichment (Optional):**
- **Strategy 1: Single-TF** - Train on primary timeframe only, no MTF features
- **Strategy 2: MTF Indicators** - Add indicator features from multiple timeframes to primary TF
- **Strategy 3: MTF Ingestion** - Raw OHLCV bars from multiple timeframes for sequence models

**Unified Pipeline (7 Phases):**

| Phase | Name | Description | Status |
|:-----:|------|-------------|:------:|
| 1 | Ingestion | Load and validate raw OHLCV | ✅ Complete |
| 2 | MTF Upscaling | Multi-timeframe resampling (9 intraday TFs) | ✅ Complete |
| 3 | Features | 180+ indicator features | ✅ Complete |
| 4 | Labeling | Triple-barrier + Optuna | ✅ Complete |
| 5 | Adapters | Model-family data preparation (2D, 3D, 4D) | ✅ Complete |
| 6 | Training | 23 models, 6 families | ✅ Complete |
| 7 | Stacking | Heterogeneous ensemble training | ✅ Complete |

**Data Shapes by Model Family:**
- **Tabular models** (Boosting + Classical): 2D arrays `(n_samples, ~180)`
- **Sequence models** (Neural): 3D windows `(n_samples, seq_len, ~180)`
- **Advanced models** (Multi-Res): 4D tensors `(n_samples, n_timeframes, seq_len, n_features)`

**See:** `docs/ARCHITECTURE.md` for comprehensive architecture documentation

### Core Contracts

**Data Pipeline (Phases 1-5):**
- **Phase 1:** Ingest raw 1-min OHLCV (canonical - single source of truth)
- **Phase 2:** Multi-timeframe upscaling
  - ✅ Complete: 9 intraday timeframes (1m, 5m, 10m, 15m, 20m, 25m, 30m, 45m, 1h)
  - Full 9-TF ladder available: 1m, 5m, 10m, 15m, 20m, 25m, 30m, 45m, 1h
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

### Model Families (23 Models Implemented)

| Family | Models | Data Format | Strengths | Status |
|--------|--------|-------------|-----------|--------|
| **Boosting** (3) | XGBoost, LightGBM, CatBoost | 2D tabular | Fast, interpretable, feature interactions | ✅ Complete |
| **Classical** (3) | Random Forest, Logistic, SVM | 2D tabular | Robust baselines, interpretable | ✅ Complete |
| **Neural** (10) | LSTM, GRU, TCN, Transformer, PatchTST, iTransformer, TFT, N-BEATS, InceptionTime, ResNet1D | 3D/4D sequences | Temporal dependencies, sequential patterns, multi-scale detection | ✅ Complete |
| **Ensemble** (3) | Voting, Stacking, Blending | OOF predictions | Same-family or heterogeneous ensemble methods | ✅ Complete |
| **Meta-Learners** (4) | Ridge Meta, MLP Meta, Calibrated Meta, XGBoost Meta | OOF predictions | Meta-learner stacking from heterogeneous bases | ✅ Complete |

**Family Classification:**
- **Tabular models** (Boosting + Classical): 2D arrays `(n_samples, n_features)` - 6 models
- **Neural models**: 3D/4D sequences - 10 models (LSTM, GRU, TCN, Transformer, PatchTST, iTransformer, TFT, N-BEATS, InceptionTime, ResNet1D)
- **Ensemble models**: Same-family or heterogeneous ensembles - 3 models (Voting, Stacking, Blending)
- **Meta-Learners**: OOF predictions from heterogeneous bases - 4 meta-learners

**Inference/Meta Family Details:**
| Meta-Learner | Method | Use Case |
|--------------|--------|----------|
| **Ridge Meta** | L2-regularized Ridge classifier | Fast linear stacking |
| **MLP Meta** | Multi-layer perceptron | Learned non-linear blending |
| **Calibrated Meta** | Isotonic/Platt calibration | Calibrated probability scaling |
| **XGBoost Meta** | Gradient boosted meta-learner | Non-linear feature interactions |

**See:** `docs/roadmaps/ADVANCED_MODELS_ROADMAP.md` for implementation history of the 6 advanced neural models

**Registry:** Models register via the `@register(...)` decorator for automatic discovery.

**Note on CatBoost:** CatBoost has **conditional registration** - it only registers if `catboost` library is installed. If CatBoost is unavailable, the model count will be 22 instead of 23. Install with `pip install catboost` if needed.

### Heterogeneous Ensemble Architecture

The factory supports **heterogeneous ensembles** where base models from different families train on **different timeframes and features**, all derived from the **same 1-min canonical OHLCV source**, then feed a single meta-learner via OOF stacking.

**Architecture:**
```
1-min OHLCV Canonical Source (single source of truth)
       ↓
Derive 9 intraday timeframes (1m, 5m, 10m, 15m, 20m, 25m, 30m, 45m, 1h)
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
# Heterogeneous stacking ensemble (trainer.py now supports dual data loading)
python scripts/train_model.py --model stacking --horizon 20 \
  --base-models xgboost,lstm,tcn --meta-learner ridge_meta

# Base models from different families now work together
python scripts/train_model.py --model stacking --horizon 20 \
  --base-models catboost,gru,patchtst --meta-learner xgboost_meta
```

See `docs/implementation/PHASE_7_META_LEARNER_STACKING.md` for planned implementation.

---

## Engineering Rules (Non-Negotiables)

### Architecture and Modularity
We do not build monoliths. Responsibilities must be split into small, composable modules with clear contracts and minimal coupling. Each module should do one thing well and expose a narrow, well-documented interface. Prefer dependency injection and explicit wiring over hidden globals or implicit side effects.

### File and Complexity Limits
Files should target **650 lines** as the ideal maximum. Files up to **1300 lines** are acceptable if the logic is cohesive and cannot be reasonably split without introducing artificial abstractions. Beyond 1250 lines is a signal that boundaries are wrong and responsibilities need to be refactored. Keep functions short, keep layers separated, and keep the cognitive load low.

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
- Stays within file limits (target 800 lines, max 1250 lines)
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

Plugin-based model training system with **23 models across 4 families** (22 if CatBoost unavailable):

```
src/models/
├── registry.py         → ModelRegistry plugin system (23 models registered)
├── base.py             → BaseModel interface, TrainingMetrics, PredictionOutput
├── trainer.py          → Backward-compat re-export (uses training/ package)
├── training/           → Training orchestration package (split from trainer.py)
│   ├── trainer.py      → Core Trainer class
│   ├── features.py     → TrainerFeaturesMixin (feature set resolution)
│   ├── evaluation.py   → TrainerEvaluationMixin (test set evaluation)
│   └── artifacts.py    → TrainerArtifactsMixin (save methods)
├── config/             → Configuration package (modular)
│   ├── trainer_config.py    → TrainerConfig dataclass
│   ├── data_requirements.py → MODEL_DATA_REQUIREMENTS, ModelFamily (moved from phase1)
│   ├── loaders.py           → YAML config loading
│   ├── merging.py           → Config merging utilities
│   ├── paths.py             → Config file paths
│   └── environment.py       → Environment detection
├── metrics.py          → Metric calculation utilities
├── data_preparation.py → Dataset preparation utilities
├── device.py           → GPU detection, memory estimation
├── boosting/           → XGBoost, LightGBM, CatBoost (3 models)
├── neural/             → LSTM, GRU, TCN, Transformer, PatchTST, iTransformer, TFT, N-BEATS, InceptionTime, ResNet1D (10 models)
│   ├── *_model.py      → Each model in separate file (consistent naming)
│   ├── cnn_base.py     → Shared CNN utilities
│   └── base_rnn.py     → Shared RNN base class
├── classical/          → Random Forest, Logistic, SVM (3 models)
└── ensemble/           → Voting, Stacking, Blending + Meta-learners (7 models)
```

**Output:** Trained models + unified performance reports

**23 models available** (22 if CatBoost unavailable):
- **Tabular (6):** `xgboost`, `lightgbm`, `catboost`, `random_forest`, `logistic`, `svm`
- **Neural (10):** `lstm`, `gru`, `tcn`, `transformer`, `patchtst`, `itransformer`, `tft`, `nbeats`, `inceptiontime`, `resnet1d`
- **Ensemble (3):** `voting`, `stacking`, `blending`
- **Meta-learners (4):** `ridge_meta`, `mlp_meta`, `calibrated_meta`, `xgboost_meta`

---

## Cross-Validation (Phase 3 - Complete)

Time-series aware cross-validation with purge/embargo:

```
src/cross_validation/
├── purged_kfold.py         → PurgedKFold with configurable purge/embargo
├── feature_selector.py     → Backward-compat re-export (uses src/feature_selection)
├── cv_runner.py            → CrossValidationRunner (core orchestration)
├── cv_dataclasses.py       → FoldMetrics, CVResult dataclasses
├── cv_tuner.py             → TimeSeriesOptunaTuner (Optuna integration)
├── cv_feature_selection.py → Per-fold feature selection helpers
├── cv_stacking.py          → Stacking dataset building utilities
├── oof_generator.py        → Unified OOF generator interface
├── oof_core.py             → Core tabular OOF generation
├── oof_sequence.py         → Sequence model OOF generation
├── oof_stacking.py         → Stacking dataset builder
├── oof_validation.py       → Coverage and correlation validation
├── oof_io.py               → Save/load OOF datasets
└── param_spaces.py         → Hyperparameter search spaces
```

**Output:** CV results, OOF predictions, stacking datasets

---

## Feature Selection (Consolidated)

All feature selection code consolidated into a single canonical package:

```
src/feature_selection/
├── __init__.py         → Unified exports with lazy loading
├── result.py           → FeatureSelectionResult, PersistedFeatureSelection
├── config.py           → FeatureSelectionConfig, ModelFamilyDefaults
├── walk_forward.py     → WalkForwardFeatureSelector (from cross_validation)
├── manager.py          → FeatureSelectionManager (from models)
├── filtering.py        → filter_low_variance, filter_correlated (from phase1)
├── priority.py         → FEATURE_PRIORITY dict
├── ohlcv_selector.py   → OHLCVFeatureSelector
└── purged_selector.py  → PurgedFeatureSelector
```

**Usage:**
```python
# Canonical imports (recommended)
from src.feature_selection import (
    FeatureSelectionResult,
    WalkForwardFeatureSelector,
    FeatureSelectionManager,
    FeatureSelectionConfig,
)

# Old imports still work with deprecation warnings
from src.cross_validation.feature_selector import WalkForwardFeatureSelector  # warns
from src.models.feature_selection import FeatureSelectionManager  # warns
```

---

## Centralized Configuration

Unified config access via `src/config/` package with UnifiedConfig as single source of truth:

```
src/config/
├── unified.py          → UnifiedConfig (single source of truth for all configuration)
├── __init__.py         → Top-level exports (TrainerConfig, CANONICAL_TIMEFRAMES, etc.)
├── constants/          → Re-exports from src/common/
│   └── __init__.py     → CANONICAL_TIMEFRAMES, DEFAULT_SPLIT_RATIOS, etc.
├── models/             → Re-exports from src/models/config/
│   └── __init__.py     → TrainerConfig, detect_environment, etc.
└── pipeline/           → Re-exports from src/phase1/config/
    └── __init__.py     → MODEL_DATA_REQUIREMENTS, ModelFamily, etc.
```

**UnifiedConfig (New - Single Source of Truth):**
```python
from src.config.unified import UnifiedConfig

# Load from YAML
config = UnifiedConfig.from_yaml("config/global.yaml")

# Or create with overrides
config = UnifiedConfig(
    symbol="MES",
    horizons=HorizonsSection(active=[5, 10, 15, 20]),
    training=TrainingSection(batch_size=512, max_epochs=100),
)

# Access nested configuration
batch_size = config.training.batch_size
active_horizons = config.horizons.active
primary_tf = config.timeframes.default_primary

# Convert to legacy configs (backward compatibility)
trainer_config = config.to_trainer_config(model_name="xgboost")
pipeline_config = config.to_pipeline_config()
ml_config = config.to_ml_config(models=["xgboost", "lstm"])

# Save configuration
config.save_yaml("experiments/runs/{run_id}/config.yaml")

# Validate configuration
errors = config.validate()
if not errors.is_valid:
    print(f"Config errors: {errors.errors}")
```

**UnifiedConfig Sections:**
- `timeframes` - Default primary TF, canonical ladder (9 TFs), extended TFs
- `splits` - Train/val/test ratios (70/15/15)
- `purge_embargo` - Purge multiplier, embargo time, min embargo bars
- `horizons` - Supported, active, default horizons
- `features` - Feature engineering params (SMA/EMA periods, RSI, MACD, Bollinger)
- `mtf` - Multi-timeframe mode and timeframes
- `training` - Batch size, epochs, early stopping, device, mixed precision
- `calibration` - Calibration enabled/method
- `optimization` - GA and Optuna hyperparameter tuning
- `cross_validation` - CV splits, purge/embargo
- `processing` - N jobs, batch symbols allowed
- `scaler` - Default scaler type (robust, standard, minmax)
- `tracking` - Experiment tracking (local, mlflow, wandb)
- `oom_recovery` - Out-of-memory recovery settings

**Legacy Config Access (Backward Compatible):**
```python
# Facade imports (recommended for new code)
from src.config import TrainerConfig, CANONICAL_TIMEFRAMES, MODEL_DATA_REQUIREMENTS
from src.config.constants import DEFAULT_SPLIT_RATIOS
from src.config.models import detect_environment
from src.config.pipeline import ModelFamily

# Original imports still work (100% backward compatible)
from src.common.timeframes import CANONICAL_TIMEFRAMES
from src.models.config import TrainerConfig
from src.phase1.config import MODEL_DATA_REQUIREMENTS
```

**Benefits:**
- **Single source of truth** - All configuration in one place
- **Schema validation** - Type checking and validation
- **Backward compatibility** - Legacy configs still work via delegation
- **Version control** - Save/load config with git-friendly YAML
- **Config drift detection** - Hash tracking alerts changes

---

## State Management

The pipeline implements robust state tracking with versioning, validation, and rollback capabilities:

```
src/ml_pipeline/state.py
├── PipelineState       → Core state management class
├── PhaseState          → Enum: NOT_STARTED, IN_PROGRESS, COMPLETED, FAILED, SKIPPED
├── PhaseResult         → Detailed phase execution results with metrics
├── StateVersion        → Schema versioning (V1 legacy, V2 current)
└── compute_config_hash → Config drift detection
```

**Features:**
- **Thread-safe state updates** - RLock ensures concurrent safety
- **Schema versioning** - Forward/backward compatibility (V1 legacy, V2 current)
- **Config hash tracking** - Detects configuration drift across runs
- **Detailed phase tracking** - Status, timing, metrics, artifacts, checkpoints
- **Rollback capability** - Restore to previous phase state (20 snapshots kept)
- **State comparison** - Diff two states for debugging with StateDiff
- **Validation** - Comprehensive checks for state consistency

**Phase States:**
```python
class PhaseState(Enum):
    NOT_STARTED = "not_started"  # Phase not yet executed
    IN_PROGRESS = "in_progress"  # Currently executing
    COMPLETED = "completed"      # Successfully finished
    FAILED = "failed"            # Execution failed with error
    SKIPPED = "skipped"          # Intentionally skipped
```

**Usage:**
```python
from src.ml_pipeline.state import PipelineState, PhaseState

# Create new state
state = PipelineState(run_id="20260116_120000_abc123", config_hash="abc123def456")

# Start a phase
state.start_phase("data_generation")

# Complete with metrics
state.complete_phase("data_generation", metrics={"rows": 10000, "features": 180})

# Check dependencies before starting
if state.can_start_phase("feature_engineering", dependencies=["data_generation"]):
    state.start_phase("feature_engineering")

# Handle failure
try:
    run_phase()
except Exception as e:
    state.fail_phase("feature_engineering", error=str(e))

# Skip a phase
state.skip_phase("optional_validation", reason="Disabled in config")

# Save and load
state.save()  # Saves to experiments/runs/{run_id}/pipeline_state.json
loaded = PipelineState.load("20260116_120000_abc123")

# Validate state consistency
errors = state.validate()
if errors:
    print(f"State validation failed: {errors}")

# Rollback to previous phase
state.rollback_to_phase("feature_engineering")

# Compare states for debugging
diff = state.diff(other_state)
print(diff.to_summary())
```

**State File Format (V2):**
```json
{
  "__version__": "v2",
  "run_id": "20260116_120000_abc123",
  "created_at": "2026-01-16T12:00:00",
  "updated_at": "2026-01-16T12:15:30",
  "config_hash": "abc123def456",
  "current_phase": "training",
  "phases": {
    "data_generation": {
      "__type__": "PhaseResult",
      "status": "completed",
      "started_at": "2026-01-16T12:00:00",
      "completed_at": "2026-01-16T12:05:00",
      "duration_seconds": 300.0,
      "metrics": {"rows": 10000},
      "artifacts": ["features_5min.parquet"]
    },
    "training": {
      "__type__": "PhaseResult",
      "status": "in_progress",
      "started_at": "2026-01-16T12:10:00"
    }
  },
  "metadata": {"symbol": "MES", "horizon": 20}
}
```

**Benefits:**
- **Resumable pipelines** - Restart from last completed phase
- **Debugging** - Full audit trail with timing and metrics
- **Config drift detection** - Hash comparison alerts configuration changes
- **Robustness** - Validation prevents invalid state transitions
- **Compatibility** - V1 legacy state automatically upgraded to V2

---

## Training Modes

The factory supports three specialized training strategies beyond standard single-shot training:

### Walk-Forward Validation

Time-series aware validation with expanding or sliding windows:

```python
from src.training.modes import WalkForwardTrainer, WalkForwardTrainerConfig

config = WalkForwardTrainerConfig(
    n_windows=5,                    # Number of train/test windows
    window_type="expanding",        # "expanding" or "sliding"
    test_size_bars=1000,            # Size of each test window
    min_train_size_bars=5000,       # Minimum training data required
    purge_bars=60,                  # Purge between train/test
    embargo_bars=1440,              # Embargo after each test window
)

wf_trainer = WalkForwardTrainer(experiment_config, config)
results = wf_trainer.run(container)
```

**Features:**
- **Expanding windows** - Training set grows, test set slides forward (more data over time)
- **Sliding windows** - Fixed-size training window slides forward (stationary assumption)
- **Proper purge/embargo** - Prevents leakage between windows
- **Aggregate metrics** - Combined performance across all windows

**Use Cases:**
- Simulate live trading deployment (realistic out-of-sample testing)
- Detect model degradation over time
- Validate strategy robustness across market regimes

### Regime-Aware Training

Train models specific to market regimes (volatility, trend, or composite):

```python
from src.training.modes import RegimeAwareTrainer, RegimeAwareConfig

config = RegimeAwareConfig(
    regime_type="composite",         # "volatility", "trend", or "composite"
    train_separate_models=True,      # Train one model per regime vs. single model with regime features
    regime_features_mode="indicators", # "indicators" or "labels_only"
)

regime_trainer = RegimeAwareTrainer(experiment_config, config)
results = regime_trainer.run(container)
```

**Regime Types:**
- **Volatility regimes:** Low, Medium, High (based on rolling volatility percentiles)
- **Trend regimes:** Downtrend, Sideways, Uptrend (based on trend strength)
- **Composite regimes:** 9 combinations (e.g., "low_vol_uptrend", "high_vol_downtrend")

**Training Strategies:**
- `train_separate_models=True` - Train N separate models (one per regime), select at inference based on current regime
- `train_separate_models=False` - Single model with regime indicator features

**Use Cases:**
- Capture regime-specific patterns (e.g., mean reversion in low volatility, momentum in high volatility)
- Improve robustness by specializing models
- Analyze per-regime performance

### Meta-Labeling

Lopez de Prado's meta-labeling methodology - use ML to size bets, not just predict direction:

```python
from src.training.modes import MetaLabelingTrainer, MetaLabelingConfig

config = MetaLabelingConfig(
    primary_model="xgboost",         # Base model for directional predictions
    meta_model="logistic",           # Meta-model for bet sizing
    meta_label_type="bet_size",      # "bet_size" or "directional"
    confidence_threshold=0.55,       # Only bet when primary model confident
)

meta_trainer = MetaLabelingTrainer(experiment_config, config)
results = meta_trainer.run(container)
```

**Workflow:**
1. **Primary model** predicts direction (long/short/neutral)
2. **Meta-model** predicts bet size (0 to 1) based on:
   - Primary model's predicted probability
   - Feature quality metrics
   - Historical accuracy in similar conditions
3. **Final position** = direction × bet_size

**Meta-Label Types:**
- **bet_size:** Continuous [0, 1] - how much to bet given primary prediction
- **directional:** Binary - whether to take the primary model's bet or not

**Use Cases:**
- Reduce false positives (meta-model filters low-confidence predictions)
- Dynamic position sizing based on conviction
- Separate "what to predict" from "how much to bet"

**CLI Usage:**
```bash
# Walk-forward validation
python scripts/train_model.py --model xgboost --horizon 20 \
  --training-mode walk_forward --n-windows 5 --window-type expanding

# Regime-aware training
python scripts/train_model.py --model lstm --horizon 20 \
  --training-mode regime_aware --regime-type composite --train-separate-models

# Meta-labeling
python scripts/train_model.py --horizon 20 \
  --training-mode meta_labeling --primary-model xgboost --meta-model logistic
```

---

## Memory Management

The pipeline includes comprehensive memory management to handle large datasets and prevent OOM errors:

```
src/utils/
├── memory.py    → MemoryManager, MemoryInfo, memory tracking utilities
└── cache.py     → DataCache, intelligent caching with automatic invalidation
```

**Features:**

### MemoryManager
```python
from src.utils.memory import MemoryManager, get_memory_info, check_available_memory

# Get current memory status
info = get_memory_info()
print(f"Available: {info.available_gb:.2f} GB / {info.total_gb:.2f} GB")
print(f"Used: {info.percent_used:.1f}%")

# Check if operation is safe
if check_available_memory(required_gb=4.0):
    # Proceed with operation requiring 4GB
    pass
else:
    # Fall back to smaller batch size
    pass

# Estimate array memory usage
from src.utils.memory import estimate_array_size
size_bytes = estimate_array_size(large_array)
print(f"Array size: {size_bytes / 1024**3:.2f} GB")
```

### DataCache
```python
from src.utils.cache import DataCache

cache = DataCache()

# Cache features with source file tracking
features = cache.get_or_compute(
    key="features_5min_MES",
    compute_fn=lambda: expensive_feature_computation(),
    source_files=["data/raw/MES_1m.parquet"],
)

# Cache automatically invalidates if source files change
# Cache falls back to disk for large items (>500MB)

# Clear cache manually
cache.clear()
```

**Cache Features:**
- **Automatic invalidation** - Based on source file modification times
- **Disk fallback** - Large datasets (>500MB) automatically cached to disk
- **Memory limits** - Configurable max memory (default 2GB)
- **Thread-safe** - Safe for concurrent access
- **TTL support** - Optional time-to-live for cache entries

### OOM Recovery
```python
from src.models.config import TrainerConfig

config = TrainerConfig(
    model_name="lstm",
    batch_size=512,
    oom_recovery_enabled=True,      # Enable automatic OOM recovery
    oom_max_retries=3,              # Retry up to 3 times
    oom_batch_reduction_factor=0.5, # Halve batch size on OOM
    oom_min_batch_size=8,           # Don't go below 8
)

# If OOM occurs during training:
# 1. Catches torch.cuda.OutOfMemoryError
# 2. Reduces batch_size = 512 * 0.5 = 256
# 3. Clears CUDA cache and retries
# 4. If OOM again: batch_size = 256 * 0.5 = 128
# 5. Repeats until success or min_batch_size reached
```

**Ensemble Memory Cleanup:**
```python
from src.models.ensemble.stacking import StackingEnsemble

ensemble = StackingEnsemble(...)
ensemble.fit(X_train, y_train, X_val, y_val)

# Automatic memory cleanup after training
# - Deletes base model OOF predictions
# - Clears intermediate feature matrices
# - Triggers garbage collection
```

**Memory Utilities:**
- `estimate_array_size()` - Estimate NumPy array memory usage
- `estimate_object_size()` - Estimate arbitrary Python object size
- `clear_cache()` - Manual cache clearing
- `@memory_tracked` - Decorator to log memory usage of functions
- `check_available_memory(required_gb)` - Safety check before large allocations

**Benefits:**
- **Prevents OOM crashes** - Automatic batch size reduction and retry
- **Faster iteration** - Intelligent caching avoids recomputation
- **Disk overflow** - Large datasets automatically spill to disk
- **Debugging** - Memory tracking helps identify bottlenecks

---

## Unified Training System (2026-01-16)

**NEW: Notebook-first "build-a-bear" interface for ML experimentation**

### Overview

The unified training system provides a simple, powerful interface for training models with:
- **Per-model feature strategies** - Each model gets baseline features tailored to its inductive biases
- **Per-model feature optimization** - Optuna-based pruning from baseline to optimal subset
- **Per-model timeframe selection** - Different models train on different timeframes (all from same 1-min source)
- **Heterogeneous ensembles** - Combine models with different features and timeframes

### Quick Start (Notebook)

```python
from src.training import TrainingOrchestrator, ExperimentConfig, ModelConfig

config = ExperimentConfig(
    symbol="MES",
    horizons=[20],
    models=[
        ModelConfig(name="xgboost", timeframe="15min", optimize_features=True, feature_opt_trials=30),
        ModelConfig(name="lstm", timeframe="5min", optimize_features=True, sequence_length=60),
        ModelConfig(name="patchtst", timeframe="1min"),  # Raw OHLCV only
    ],
    build_ensemble=True,
    meta_learner="ridge_meta",
)

orchestrator = TrainingOrchestrator(config)
results = orchestrator.run()
orchestrator.display_results()
```

**Output:**
- XGBoost trains on optimized 15min features (~60 from ~100 baseline)
- LSTM trains on optimized 5min features (~50 from ~80 baseline)
- PatchTST uses raw 1min OHLCV (5 features, no optimization)
- Meta-learner stacks all 3 predictions

**All from same 1-min canonical OHLCV source!**

### Feature Strategy System

Each of 23 models has a tailored baseline feature strategy:

**Location:** `src/features/strategies.py`

```python
from src.features.strategies import MODEL_FEATURE_STRATEGIES

# XGBoost strategy
strategy = MODEL_FEATURE_STRATEGIES["xgboost"]
# {
#   "baseline_features": ["momentum", "volatility", "volume", "microstructure", "mtf"],  # ~100 features
#   "preferred_families": ["boosting"],
#   "mtf_mode": "indicators",
#   "min_features": 20,
#   "max_features": 120,
# }

# LSTM strategy
strategy = MODEL_FEATURE_STRATEGIES["lstm"]
# {
#   "baseline_features": ["momentum", "volatility", "wavelets", "mtf"],  # ~80 features
#   "preferred_families": ["neural"],
#   "mtf_mode": "indicators",
#   "min_features": 30,
#   "max_features": 100,
# }

# PatchTST strategy
strategy = MODEL_FEATURE_STRATEGIES["patchtst"]
# {
#   "baseline_features": ["raw_ohlcv"],  # 5 features only (Open, High, Low, Close, Volume)
#   "preferred_families": ["transformer"],
#   "mtf_mode": "multi_stream",  # Uses multi-TF ingestion
#   "min_features": 4,
#   "max_features": 5,
# }
```

### Feature Optimization Flow

```
1. Pipeline generates ~180 total features
   ↓
2. Model-specific baseline strategy selects subset
   - XGBoost: ~100 features (momentum + volatility + volume + microstructure + MTF)
   - LSTM: ~80 features (momentum + volatility + wavelets + MTF)
   - PatchTST: 5 features (raw OHLCV only)
   ↓
3. IF optimize_features=True, run Optuna
   - Prune from baseline to optimal subset
   - Example: XGBoost 100 → 60 features
   ↓
4. Train model on optimized (or baseline) features
```

### Architecture Files

**Configuration:**
- `src/training/config.py` - ExperimentConfig, ModelConfig dataclasses
- `src/features/strategies.py` - MODEL_FEATURE_STRATEGIES for 23 models
- `src/features/optimization.py` - Optuna-based feature pruning

**Orchestration:**
- `src/training/orchestrator.py` - TrainingOrchestrator (main controller)
- `src/training/__init__.py` - Exports ExperimentConfig, ModelConfig

**Documentation:**
- `docs/implementation/UNIFIED_TRAINING_SYSTEM.md` - Complete architecture guide
- `docs/implementation/ORCHESTRATOR_COMPLETION_PLAN.md` - Implementation details
- `notebooks/unified_training_colab.ipynb` - Working notebook example

### Key Design Decisions

1. **Keep ALL ~180 features** - All features ARE important, just intelligently allocated
2. **Per-model strategies** - Different models get different baseline features
3. **Optuna optimization** - Prune from baseline to optimal subset per model
4. **Per-model timeframes** - Different models train on different TFs from same 1-min source
5. **Backward compatible** - Old dict-based config still works

### Feature Family Names

When specifying baseline features in strategies, use these family names:

```python
FEATURE_FAMILIES = {
    "momentum": ["rsi", "macd", "stoch", "cci", ...],           # ~40 features
    "volatility": ["atr", "bbands", "keltner", ...],            # ~25 features
    "volume": ["obv", "vwap", "adl", "mfi", ...],               # ~20 features
    "microstructure": ["spread", "imbalance", "vpin", ...],     # ~30 features
    "wavelets": ["cwt_*", "dwt_*", ...],                        # ~25 features
    "mtf": ["*_1m", "*_5m", "*_15m", "*_1h", ...],             # ~30 MTF indicators
    "regime": ["regime_*", "volatility_regime", ...],           # ~10 features
    "temporal": ["hour", "day_of_week", ...],                   # ~5 features
    "raw_ohlcv": ["open", "high", "low", "close", "volume"],   # 5 features
}
```

---

## Quick Commands

```bash
# Run data pipeline (Phase 1)
./pipeline run --symbols MGC

# Run pipeline with all 9 timeframes (MTF-P1-002: enables heterogeneous ensembles)
./pipeline run --symbols MGC --process-all-timeframes
# Produces: features_{tf}.parquet, labels_{tf}.parquet, scaled/{tf}/ for each TF

# Run pipeline with specific output timeframes
./pipeline run --symbols MGC --output-timeframes 5min,15min,60min
./pipeline run --symbols MGC --output-timeframes 9tf  # Same as --process-all-timeframes

# Train specific model (Phase 6)
python scripts/train_model.py --model xgboost --horizon 20
python scripts/train_model.py --model lstm --horizon 20 --seq-len 30
python scripts/train_model.py --model random_forest --horizon 20

# Train ensemble (Phase 6 - same-family)
python scripts/train_model.py --model voting --horizon 20 --base-models xgboost,lightgbm,catboost
python scripts/train_model.py --model stacking --horizon 20 --base-models lstm,gru,tcn --seq-len 30

# Heterogeneous ensemble training (Phase 7 - now supported!)
python scripts/train_model.py --model stacking --horizon 20 \
  --base-models xgboost,lstm,patchtst --meta-learner ridge_meta

# Walk-forward validation training
python scripts/train_model.py --model xgboost --horizon 20 \
  --training-mode walk_forward --n-windows 5 --window-type expanding

# Regime-aware training
python scripts/train_model.py --model lstm --horizon 20 \
  --training-mode regime_aware --regime-type composite --train-separate-models

# Meta-labeling training
python scripts/train_model.py --horizon 20 \
  --training-mode meta_labeling --primary-model xgboost --meta-model logistic

# Run cross-validation (Phase 3)
python scripts/run_cv.py --models xgboost --horizons 20 --n-splits 5
python scripts/run_cv.py --models all --horizons 5,10,15,20 --tune

# List available models (should print 23, or 22 if CatBoost unavailable)
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
- 23 models implemented across 4 families (Tabular, Neural, Ensemble, Meta-learners) - 22 if CatBoost unavailable
- Plugin-based model registry with `@register` decorator
- 3 ensemble methods: Voting, Stacking, Blending (same-family or heterogeneous)
- 4 meta-learners: Ridge Meta, MLP Meta, Calibrated Meta, XGBoost Meta
- 10 neural models: LSTM, GRU, TCN, Transformer, PatchTST, iTransformer, TFT, N-BEATS, InceptionTime, ResNet1D

**Roadmap:**
- Phase 8: Advanced meta-learners (regime-aware, adaptive)
- Phase 9: Real-time inference pipeline with streaming predictions

**Performance expectations:** Do not treat any Sharpe/win-rate targets as "built-in". Measure performance empirically via `scripts/run_cv.py`, `scripts/run_walk_forward.py`, and `scripts/run_cpcv_pbo.py` on your own data/cost assumptions.
