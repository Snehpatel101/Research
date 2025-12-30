# ML Model Factory for OHLCV Time Series

## Goal

Keep the codebase modular, readable, and easy to extend as we build a **model factory** that can train, evaluate, and compare ANY model type on OHLCV bar data.

This is not a single pipeline — it's a **factory** with a plugin architecture.

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

1. **Unified Data Pipeline** - One 1-min dataset → model-specific feature/data extraction (⚠️ currently all models receive same indicator features; model-specific strategies in roadmap)
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

### Data Pipeline Architecture: Current vs. Intended

#### Current Implementation (Phase 1 Complete)

**Universal Indicator Pipeline:**
- All models receive ~180 **indicator-derived** features (RSI, MACD, wavelets, microstructure, etc.)
- MTF indicators from **5 timeframes** (15min, 30min, 1h, 4h, daily) - **intended: 9 timeframes** (1min, 5min, 10min, 15min, 20min, 25min, 30min, 45min, 1h)
- Data served in model-appropriate **shapes**:
  - **Tabular models** (Boosting + Classical): 2D arrays `(n_samples, 180)`
  - **Sequence models** (Neural + CNN + Advanced): 3D windows `(n_samples, seq_len, 180)`

**Limitations:**
1. Only 5 of 9 intended timeframes implemented
2. All features are pre-computed indicators (sequence/CNN models should receive raw multi-resolution OHLCV bars)

#### Intended Architecture (Roadmap)

**Model-Specific MTF Strategies (from `docs/roadmaps/MTF_IMPLEMENTATION_ROADMAP.md`):**

| Strategy | Data Type | Model Families | Status |
|----------|-----------|----------------|--------|
| **Strategy 1: Single-TF** | One timeframe, no MTF | All models (baselines) | ❌ Not implemented |
| **Strategy 2: MTF Indicators** | Indicator features from 9 timeframes | Tabular: Boosting (XGBoost, LightGBM, CatBoost) + Classical (RF, Logistic, SVM) | ⚠️ Partial (5 TFs, all models get this) |
| **Strategy 3: MTF Ingestion** | Raw OHLCV bars from 9 timeframes as multi-resolution tensors | Sequence: Neural (LSTM, GRU, TCN, Transformer) + CNN (InceptionTime, 1D ResNet) + Advanced (PatchTST, iTransformer, TFT, N-BEATS) | ❌ Not implemented |

**When Strategy 3 is implemented:**
- **Tabular models** (Boosting + Classical) → Keep indicator-derived features from 9 timeframes
- **Sequence models** (Neural + CNN + Advanced) → Receive raw MTF OHLCV bars from 9 timeframes: `{'5min': (T,60,4), '15min': (T,20,4), '30min': (T,10,4), '1h': (T,5,4)}`
- **Enables:** Multi-resolution temporal learning for InceptionTime, PatchTST, iTransformer, TFT, N-BEATS

**See:** `docs/CURRENT_VS_INTENDED_ARCHITECTURE.md` for detailed analysis

### Core Contracts

**Data Pipeline (Phase 1 - Complete):**
- Raw 1-min OHLCV → 5-min base → ~150 indicator features → MTF upscaling (**5 of 9 timeframes**) → ~30 MTF indicators → ~180 total features
- Triple-barrier labeling with Optuna optimization
- No lookahead bias (proper purging + embargo, MTF uses shift(1))
- Time-series aware train/val/test splits (70/15/15)
- Quality-weighted samples
- **Limitations:** Only 5 timeframes (intended: 9), all features are indicator-derived (raw multi-resolution bars for Strategy 3 not yet implemented)

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

### Model Families (13 Implemented + 6 Planned = 19 Total)

| Family | Models | Data Format | Strengths | Status |
|--------|--------|-------------|-----------|--------|
| **Boosting** (3) | XGBoost, LightGBM, CatBoost | 2D tabular | Fast, interpretable, feature interactions | ✅ Complete |
| **Neural** (4) | LSTM, GRU, TCN, Transformer | 3D sequences | Temporal dependencies, sequential patterns | ✅ Complete |
| **Classical** (3) | Random Forest, Logistic, SVM | 2D tabular | Robust baselines, interpretable | ✅ Complete |
| **Ensemble** (3) | Voting, Stacking, Blending | Mixed (all-tabular OR all-sequence) | Combines diverse model strengths | ✅ Complete |
| **CNN** (2) | InceptionTime, 1D ResNet | 3D sequences | Multi-scale pattern detection, deep residual learning | 📋 Planned |
| **Advanced Transformers** (3) | PatchTST, iTransformer, TFT | 3D sequences | SOTA long-term forecasting, interpretable attention | 📋 Planned |
| **MLP** (1) | N-BEATS | 3D sequences | Interpretable decomposition, M4 winner | 📋 Planned |

**Input Format Summary:**
- **Tabular models** (Boosting + Classical): 2D arrays `(n_samples, n_features)` - 6 models ✅
- **Sequence models** (Neural + CNN + Advanced + MLP): 3D windows `(n_samples, seq_len, n_features)` - 7 implemented + 6 planned = 13 models

**See:** `docs/roadmaps/ADVANCED_MODELS_ROADMAP.md` for 6 planned models (14-18 days implementation)

**Registry:** Models register via the `@register(...)` decorator for automatic discovery.

### Ensemble Model Compatibility

**CRITICAL:** Ensembles require all base models to have the same input shape:
- **Tabular models (2D input):** `xgboost`, `lightgbm`, `catboost`, `random_forest`, `logistic`, `svm`
- **Sequence models (3D input):** `lstm`, `gru`, `tcn`, `transformer`
- **Mixed ensembles are NOT supported** and will raise `EnsembleCompatibilityError`

### Recommended Ensemble Configurations

**Tabular-Only Ensembles:**

| Ensemble Type | Models | Method | Use Case |
|---------------|--------|--------|----------|
| Boosting Trio | `xgboost` + `lightgbm` + `catboost` | Voting | Fast baseline ensemble |
| Boosting + Forest | `xgboost` + `lightgbm` + `random_forest` | Voting/Blending | Balanced accuracy/speed |
| All Tabular | All tabular models | Stacking | Maximum diversity (higher overfit risk) |

**Sequence-Only Ensembles:**

| Ensemble Type | Models | Method | Use Case |
|---------------|--------|--------|----------|
| RNN Variants | `lstm` + `gru` | Voting | Temporal pattern diversity |
| Temporal Stack | `lstm` + `gru` + `tcn` | Stacking | Sequential pattern learning |
| All Neural | `lstm` + `gru` + `tcn` + `transformer` | Stacking | Maximum temporal diversity |

**INVALID Configurations (Will Fail):**
- ❌ `xgboost` + `lstm` (mixing tabular + sequence)
- ❌ `lightgbm` + `gru` + `tcn` (mixing tabular + sequence)
- ❌ `random_forest` + `transformer` (mixing tabular + sequence)

**Implemented Ensemble Methods:**
- **Voting:** Combine predictions via weighted/unweighted averaging
- **Stacking:** Train meta-learner on base model out-of-fold predictions
- **Blending:** Train meta-learner on holdout predictions

See `docs/phases/PHASE_4.md` for detailed ensemble architecture, validation utilities, and compatibility matrix.

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

**Output:** ~180 indicator-derived features consumed by all model trainers (tabular: 2D arrays, sequence: 3D windows)

---

## Model Factory (Phase 2 - Complete)

Plugin-based model training system with **13 models across 4 families**:

```
src/models/
├── registry.py         → ModelRegistry plugin system (13 models registered)
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

**All 13 models available:** `xgboost`, `lightgbm`, `catboost`, `lstm`, `gru`, `tcn`, `transformer`, `random_forest`, `logistic`, `svm`, `voting`, `stacking`, `blending`

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

# Train specific model (Phase 2)
python scripts/train_model.py --model xgboost --horizon 20
python scripts/train_model.py --model lstm --horizon 20 --seq-len 30
python scripts/train_model.py --model random_forest --horizon 20

# Train ensemble (Phase 4) - all base models must be same family (tabular or sequence)
python scripts/train_model.py --model voting --base-models xgboost,lightgbm,catboost --horizon 20
python scripts/train_model.py --model stacking --base-models xgboost,lightgbm,random_forest --horizon 20
python scripts/train_model.py --model voting --base-models lstm,gru,tcn --horizon 20

# Run cross-validation (Phase 3)
python scripts/run_cv.py --models xgboost --horizons 20 --n-splits 5
python scripts/run_cv.py --models all --horizons 5,10,15,20 --tune

# List available models (should print 12)
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

## Phase 1 Analysis Summary (2025-12-24)

**Strengths:**
- Triple-barrier labeling with symbol-specific asymmetric barriers (MES: 1.5:1.0)
- Optuna-based parameter optimization with transaction cost penalties
- Proper purge (60) and embargo (1440) for leakage prevention
- Quality-based sample weighting (0.5x-1.5x)
- 150+ base features including wavelets and microstructure
- Multi-timeframe indicator features from 5 timeframes (15min, 30min, 1h, 4h, daily) - **intended: 9-timeframe ladder** (1min, 5min, 10min, 15min, 20min, 25min, 30min, 45min, 1h)
- Strategy 2 MTF indicators partially implemented (5 of 9 timeframes)
- TimeSeriesDataContainer for unified model training interface (2D for tabular, 3D for sequence)

**Recent Improvements:**
- Added a synthetic OHLCV helper for smoke tests (`src/utils/notebook.py`), but real training expects real data in `data/raw/`
- Added wavelet decomposition features
- Added microstructure features (bid-ask spread, order flow)
- Improved embargo to 1440 bars (5 days) for better serial correlation handling
- DataIngestor validates OHLCV data at pipeline entry

**Performance expectations:** do not treat any Sharpe/win-rate targets as “built-in”. Measure performance empirically via `scripts/run_cv.py`, `scripts/run_walk_forward.py`, and `scripts/run_cpcv_pbo.py` on your own data/cost assumptions.
