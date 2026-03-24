# CLAUDE.md - ML Factory

**What this is:** Context file for AI assistants working on ML Factory.

---

## How We Work

1. **Read root docs first** - DIRECTION.md for vision, CLEANUP_PLAN.md for current phase
2. **Get user approval before changes** - Propose plans, wait for confirmation
3. **Update docs as you go** - Every change updates CLEANUP_TASKS and eventually COMPLETION
4. **Delete, don't adapt** - Remove duplicates rather than maintaining compatibility layers
5. **Clean code always** - Run linters, format code, no shortcuts

---

## The Four Root Documents

| Document | Purpose | Update Trigger |
|----------|---------|----------------|
| **DIRECTION.md** | Architecture vision, what we're building, blockers, trajectory | Major architectural decisions |
| **CLEANUP_PLAN.md** | Phase roadmap with architecture diagrams and rationale | Phase completes or priorities change |
| **CLEANUP_TASKS.md** | Same phases as PLAN but with specific file:line tasks | Starting/completing any task |
| **COMPLETION.md** | Running archive of all completed work | After each phase completes |

### CLEANUP_PLAN vs CLEANUP_TASKS

These two documents **mirror each other** - same phases, different detail levels:

| CLEANUP_PLAN.md | CLEANUP_TASKS.md |
|-----------------|------------------|
| Phase overview | Same phases |
| Architecture diagrams | Specific file locations |
| "What and why" | "Where and how" |
| Execution order | Task checklists |
| Validation criteria | Verification commands |

**Always update both together.** Plan changes → Tasks change.

### Update Flow

```
User Request
     ↓
Read DIRECTION.md + CLEANUP_PLAN.md (understand context)
     ↓
Propose approach → Get user approval
     ↓
Execute changes → Update CLEANUP_TASKS.md
     ↓
Phase complete → Move summary to COMPLETION.md
     ↓
If architecture changed → Update DIRECTION.md
```

### Rules

- **CLEANUP_PLAN.md and CLEANUP_TASKS.md update together** - They're mirrors
- **Check COMPLETION.md before investigating** - Many issues already resolved/disproven
- **DIRECTION.md changes require user approval** - It's the architectural source of truth

---

## What is ML Factory?

**ML Factory** = Config-driven system for building production ML ensembles for financial time-series prediction.

### The Goal

Put data in, get optimized trading model out. No data leakage, reproducible results, realistic financial metrics.

### Core Flow

```
Raw OHLCV → Pipeline (12 stages) → Features + Labels → Adapters → Models → Ensemble
```

### Key Guarantees

| Guarantee | How |
|-----------|-----|
| No data leakage | Purge/embargo in all CV splits |
| No lookahead | All MTF operations use `shift(1)` |
| Reproducible | Same config = same output |
| Realistic metrics | Transaction costs, slippage included |

### Model Support

All 12 models are production-ready:

| Category | Models |
|----------|--------|
| **Boosting** | XGBoost, LightGBM, CatBoost |
| **Neural RNN** | LSTM, GRU |
| **Neural CNN** | TCN, InceptionTime, 1D ResNet |
| **Transformer** | PatchTST, iTransformer, TFT |
| **MLP** | N-BEATS |

---

## Code Quality

**Clean code is always better.** Run linters before committing.

### Linting & Formatting

```bash
# Linting (required - must pass)
ruff check src/
ruff check src/ --fix  # Auto-fix what's possible

# Formatting (required)
black src/
black --check src/  # Check without modifying

# Type checking (informational - many false positives from stubs)
mypy src/ --ignore-missing-imports
```

### Standards

| Tool | Purpose | Config |
|------|---------|--------|
| **ruff** | Linting + import sorting | `pyproject.toml` |
| **black** | Code formatting | Default settings |
| **mypy** | Type checking | Ignore missing imports |

### Before Every Commit

1. `ruff check src/ --fix` - Fix linting issues
2. `black src/` - Format code
3. Import verification (see below)
4. No new pyright errors (existing stub issues OK)

### Clean Code Principles

- **Delete dead code** - Don't comment it out, delete it
- **One definition per concept** - No duplicates
- **Imports from canonical locations** - Re-export for compatibility
- **No magic numbers without context** - Use constants or document inline
- **Functions do one thing** - If it needs "and" to describe, split it

---

## Project Structure

```
src/
├── core/           # Types, contracts, base interfaces
├── data/           # Adapters, features, pipeline, labeling
├── models/         # All model implementations + training
├── optimization/   # Optuna, feature selection
├── validation/     # Leakage detection, lookahead audit, CV
├── inference/      # Backtesting, prediction
├── config/         # Configuration classes
└── cli/            # Command-line interface
```

### Canonical Locations

| Thing | Location |
|-------|----------|
| All enums/types | `src/core/types.py` |
| Model contracts | `src/core/contracts/` |
| Adapters | `src/data/adapters/` |
| Feature selection | `src/optimization/feature_selection/` |
| Validation | `src/validation/` |

---

## Current Status

**Phases 0-6: COMPLETE**
- Phase 0: Removed ~5,336 lines of duplicate code
- Phase 1: Contract enforcement with blocking validation
- Phase 2: 4D data infrastructure for transformers
- Phase 3: Enhanced adapter error handling
- Phase 4: Feature manifest with lineage tracking
- Phase 5: Performance optimizations
- Phase 6: Deprecation cleanup and orchestrator consolidation

**Phases 24-50: COMPLETE**
- See CLEANUP_PLAN.md and COMPLETION.md for full details

**Phases 51-52 (Phase 3 Master Plan): COMPLETE — 26/26 tasks**
- Phase 51: Deploy artifact system, single-call production inference, TrainerProtocol, adapter routing
- Phase 52: UniversalInferencePipeline, special mode bundles (WalkForward, Regime, MetaLabeling), safe_pickle_load migration (16 sites), neural architecture versioning

**Phase 53: COMPLETE — Security Hardening & SymbolConfig**
- Complete safe_pickle_load migration (0 joblib.load remaining, 36 safe sites)
- SymbolConfig standalone class (src/config/symbol.py) with MES/MGC/MNQ presets
- Explicit resample anti-lookahead params on all inference sites
- 12-model training smoke test: ALL PASS

**Phase 54: COMPLETE — E2E Pipeline Bug Fixes (5 bugs)**
- Trainer.save() added for model persistence
- Per-model feature selection (replaces global truncation that caused conflicts)
- 4D multi-stream data wired through MLFactory for PatchTST/iTransformer
- Timeframe key normalization (1h → 60min)
- Empty test split guard + date-range filtering for additional_dfs
- Optuna flags propagated when n_trials=0
- Full E2E verified: 10/12 models PASS (2 queued behind CPU time)

**Phase 55: COMPLETE — Deploy Manifest Fix**
- Bundle metadata `model_name` now correctly set (was "unknown" for boosting models)
- Deploy manifest `primary_model` selects best model by macro_f1
- Verified: all 6 bundles + manifest have correct model names

**Phase 56: COMPLETE — Backtest Pipeline Fix**
- Fixed `_extract_predictions()` — majority vote from AlignedOOFResult, proper OOFPrediction API
- Fixed timestamp column mismatch (datetime→timestamp rename for Backtester merge)
- Full verification: standard pipeline (7/7), backtest (5 trades), MTF (28 columns), Optuna (3 trials)

**Phase 57: COMPLETE — 4D OOF Generation (Cross-Family Ensembles)**
- Added `_generate_4d_oof()` to OOFGenerationService for transformer models (PatchTST, iTransformer, TFT)
- 4D data split by sample index using PurgedKFold (no re-windowing — samples already windowed)
- Enables cross-family ensembles: boosting (2D) + transformer (4D) working together
- Verified: xgboost+patchtst ensemble PASS, boosting-only regression PASS

**Phase 58: COMPLETE — Feature Selection Pipeline Overhaul**
- Wired low-variance and correlation pre-filters into orchestrator
- Per-model feature selection (respects model contract max_features)
- Features saved per-model to bundles

**Phase 59: COMPLETE — MDA Feature Ranking + Test Split Fix**
- Replaced variance ranking with MDA (permutation importance) in orchestrator
- MDA is target-aware: ranks features by predictive power, not just spread
- Fallback to variance if MDA fails (no labels, too few rows, CV error)
- Fixed test split crash: embargo_bars > remaining data caused KeyError
- Guard in trainer.py skips test eval gracefully when no test split exists

**Phase 60: COMPLETE — DatetimeIndex Pipeline Fix & Cross-Family Ensembles**
- Fixed broken `calculate_atr_numba` import in clean module (moved to canonical location)
- Fixed `.cv` attribute access error in factory.py (config restructured)
- Fixed impossible data sufficiency validation formula
- Fixed DatetimeIndex loss after feature engineering (root cause of 4D failures)
- Fixed backtest timestamp extraction for DatetimeIndex DataFrames
- Fixed OOF datetime extraction from index instead of column
- **Result: All 8 ensemble combinations now PASS (was 4/8)**
  - 2D+2D+2D, 2D+3D, 2D+4D, 3D+3D, 3D+4D, 4D+4D, 2D+3D+4D all working
  - Walk-forward mode verified for boosting models

**Phase 66: COMPLETE — Financial Rigor Improvements (4 enhancements)**
- ONC clustered feature selection enabled (prevents substitution effect in MDA)
- Transaction costs in Optuna (labels include real $3.75 MES costs)
- DSR gate enforcement (rejects selection-bias-inflated Sharpe ratios)
- CPCV support in hyperparameter tuning (15 backtest paths via cv_method="cpcv")
- 6 files modified, 212/212 tests still passing

**Phase 67: COMPLETE — Consistency Hardening & 3D OOF Scalability (15 fixes)**
- Fixed all 14 inconsistencies from codebase audit (2 critical, 6 high, 6 medium)
- Critical: OOF artifacts no longer dropped before save, multi-horizon OOFs filtered before ensemble
- High: model config propagated to OOF, walk-forward per-model data prep, per-model features in all modes
- 3D OOF chunked processing: scale-before-window reduces peak memory from ~800 GiB to ~152 MiB/chunk
- Enables 1.7M+ row datasets on Colab (230GB RAM) without OOM
- 16 files modified, 212/212 tests still passing

**Phase 68: COMPLETE — Performance Optimizations (9 items, ~4.4x overall speedup on H100)**
- GPU auto-enable for boosting: XGBoost/LightGBM/CatBoost auto-detect CUDA (4-10x per model)
- torch.compile max-autotune on CUDA, disabled on CPU (1.2-1.5x neural training)
- Batch size 256→512 default (1.3-1.8x H100 throughput)
- n_jobs=-1 default (3-6x on MDA/sklearn parallelism)
- MDA subsampling: caps at 50K rows for datasets >50K (34x on 1.7M rows)
- Checkpoint interval 10→50 (80% less I/O overhead)
- Numba JIT for rolling.apply(): liquidity, mean_reversion, price (10-25x on those features)
- Feature caching: hash-based parquet disk cache (5-30x on cache hits)
- Fixed walk-forward max_epochs bug (was 50 fallback, now uses DEFAULT_MAX_EPOCHS=100)
- 12 files modified, 212/212 tests still passing

**Phase 69: COMPLETE — Calibrator Single-Class Fix**
- Fixed crash in `src/models/calibration/calibrator.py` when OOF predictions are all one class
- sklearn's LogisticRegression requires 2+ classes; now skips with pass-through when only 1 class present
- Smoke test (MES 1-week, 1 epoch): 11/12 models PASS, 3/3 ensembles PASS, 3/3 walk-forward PASS
- TFT OOM on 16GB RAM (hardware limit, not code bug)

**Phase 70: COMPLETE — Lint Fixes (14 ruff + 15 black)**
- Fixed 14 ruff linting errors across src/ (unused imports, type annotations)
- Fixed 15 black formatting issues (22 files total modified)
- Zero lint errors remaining: `ruff check src/` clean, `black --check src/` clean

**Phase 71: COMPLETE — Comprehensive Notebook Overhaul (12 fixes, 25 cells)**
- Added data format instructions and bring-your-own-data guide (was missing — notebook unusable)
- Moved EDA cell to correct position after EDA markdown header
- Added calibration & conformal prediction cells (ProbabilityCalibrator, ConformalPredictor)
- Added leakage detection cell (comprehensive_leakage_check with fallback)
- Added Sortino ratio, Calmar ratio, expectancy to backtest stats display
- Surfaced transaction costs config (COMMISSION_PER_TRADE, SLIPPAGE_TICKS, TICK_VALUE)
- Added conformal prediction config (CONFORMAL_ENABLED, CONFORMAL_ALPHA, CONFORMAL_METHOD)
- Final notebook: 25 cells (9 markdown, 16 code), all syntax verified, 212/212 tests passing

**Phase 72: COMPLETE — Memory Cleanup (OOM Prevention for Large Datasets)**
- Fixed OOM crash on 230GB H100 with 1.6M row MGC dataset (peak 225+ GB → ~80-100 GB)
- Walk-forward: del model + arrays + gc.collect + empty_cache between windows
- Sequential models: evict PreparedData from _prepared_cache after each model
- OOF sequence folds: del model + 3D train sequences between folds
- OOF tabular folds: del model + scaled arrays between folds
- 4 files modified, 212/212 tests still passing

**Phase 73: COMPLETE — Scaler Serialization Fix + Notebook Warnings**
- Fixed AdapterScaler save/load format mismatch (joblib vs pickle)
- Suppressed notebook Jupyter warnings

**Phase 74: COMPLETE — Memory Optimization + Training Bug Fixes + Notebook Visualizations**
- Optuna stratified subsampling (50K cap) for large datasets
- float32 conversion in Optuna, walk-forward, and training ops
- Optuna best_params now written to model_config (were silently discarded)
- Financial reports generated BEFORE oof_predictions.clear() (were always empty)
- Added 11 notebook visualizations (trading analytics, model diagnostics, multi-model, walk-forward)

**Phase 75: COMPLETE — OOM Root Cause Fix + Pipeline Bug Fixes (11 items)**
- CRITICAL: Fixed cache eviction bug in training_ops.py — eviction never fired (string match on tuple keys)
- float32 downcast in PreparedData creation (halves memory: TCN 55→27 GB)
- model.cpu() + torch._dynamo.reset() between sequential neural models
- CSV support in factory.py (auto-detect from file extension)
- "date" column handling in factory.py (normalize to datetime index)
- Default training_mode fix: "single_horizon" → "standard" (experiment.py)
- deploy_artifact parsing added to from_dict() (experiment.py)
- Full config round-trip serialization in to_dict() (nested sub-configs: optuna, calibration, checkpoint, walk_forward, features, labeling, scaler, sequence, mtf, splits)
- Deploy path resolution fallback (deploy.py, 2 sites) — deploy_dir.parent for bundle paths
- 5 files modified, 212/212 tests still passing

**Phase 76: COMPLETE — Walk-Forward Feature Selection Fix + Float32 Scaler (2 critical bugs)**
- CRITICAL: Walk-forward feature selection was dead code — `hasattr(prepared, "with_features")` always False (PreparedData has no such method). Fixed 3 sites to filter df columns BEFORE prepare(), matching _prepare_with_cache pattern. Without this fix, walk-forward trained on 227 features instead of 60 (~3.8x more memory for 3D models).
- CRITICAL: sklearn RobustScaler upcasts float32→float64 internally, doubling memory. Manual numpy-based float32 scaling in fold_scaling.py (median/IQR for robust, mean/std for standard). Still fits sklearn scaler for inference compatibility.
- Slice indexing optimization correctly skipped — embargo creates non-contiguous train indices
- 2 files modified, 212/212 tests still passing

**Phase 77: COMPLETE — Pipeline Audit Fixes (6 items across 6 files)**
- AdapterScaler float32 manual scaling (same pattern as Phase 76 FoldAwareScaler fix) — eliminates ~37 GB temporary spike for TCN
- Meta-labeling `del prepared` after flattening + training — frees ~28 GB of 3D arrays
- OOF generation `del X_train_2d, prepared` after DataFrame creation — prevents 2x peak memory
- OOM retry batch_size propagation fixed (was dead code — reduced batch never reached model)
- XGBoost `use_label_encoder=False` removed from all 3 locations (deprecated in XGBoost 2.0)
- Walk-forward timestamp misalignment fixed (RangeIndex after filter_invalid_labels)
- 6 files modified, 212/212 tests still passing

**Phase 78: COMPLETE — Deep Memory Fixes (4 items, ~19 GB saved per neural model)**
- CRITICAL: `torch.tensor()` → `torch.from_numpy()` in _create_dataloader (base_rnn.py) — eliminates full copy of all training data for every neural model (~19 GB for TCN)
- Float32 downcast added to walk-forward, regime-aware, meta-labeling modes (were bypassing _prepare_with_cache)
- model.cpu() + torch._dynamo.reset() added to walk-forward, regime, meta-labeling between models
- Boosting cache eviction after parallel training (247 MB leaked until end of pipeline)
- 3 files modified, 212/212 tests still passing

**Phase 79: COMPLETE — In-Place Scaling + Factory Float32 (4 fixes, saves ~27 GB peak)**
- Root cause of remaining Colab OOM: FoldAwareScaler created duplicate 24.6 GB arrays during scaling
- In-place scaling (`X_train -= median; X_train /= iqr`) eliminates the duplication
- CRITICAL: oof_sequence.py passed `seq_builder._X` directly as X_val — in-place scaling corrupted data for folds 2-5. Fixed with `.copy()`
- Factory df + additional_dfs downcast to float32 at source
- Peak drops from ~86 GB to ~59 GB for TCN walk-forward window 5
- 4 files modified, 212/212 tests still passing

**Phase 80: COMPLETE — Audit-Driven Fixes (4 critical, 20 files)**
- Label balance: symmetric transaction costs in triple_barrier.py (root cause of 2.6% long rate at H5)
- predict() memory: torch.from_numpy across 10 neural model files (16 edit sites)
- XGBoost early stopping: patience 20→10, config drift fix, walk_forward.py key mismatch
- TCN seq_len: 120→64 across 7 files (matches receptive field of 61)
- 20 files modified, 212/212 tests still passing

**Phase 81: COMPLETE — Fix 5 Dead Notebook Cells**
- Calibration, leakage, feature importance, equity underwater, agreement matrix
- All cells now use correct ExperimentResult API paths

**Phase 82: COMPLETE — Checkpoint Resume 4D Persistence**
- PatchTST/iTransformer/TFT crashed on resume_from_checkpoint() — additional_dfs was None
- Checkpoint now saves/loads MTF data as mtf_*.parquet files
- Backward compat: regenerates from raw source file if no cached MTF exists
- 1 file modified, 212/212 tests still passing

**Phase 83: COMPLETE — Audit Cleanup (min_frequency wiring + missed fixes)**
- Wired FEATURE_SELECTION_MIN_FREQUENCY end-to-end: FeatureConfig → TrainerConfig → FeatureSelectionConfig
- Fixed XGBoost fallback default (20→10), mtf_plus seq_len (120→64), weight_norm deprecation
- Item #13 (XGBoost in global.yaml) skipped — would be dead config
- 6 files + notebook modified, 212/212 tests still passing

**Phase 84: COMPLETE — Signal Quality (Logloss Metrics + Binary Classification)**
- Added `logloss_unweighted` and `logloss_weighted` to compute_classification_metrics() (flows to ExperimentResult.metrics)
- Binary classification mode: `LabelingConfig(binary_mode=True)` remaps {-1,0,+1} → {0,1} (no move vs significant move)
- Dynamic label mapping (n_classes=2/3), n_classes threaded ExperimentConfig → PipelineConfig
- All 17 audit items from AUDIT_2026-02-26.md now fully addressed
- 7 files + notebook modified, 212/212 tests still passing

**Phase 85: COMPLETE — Full Audit Fixes (8-agent audit + 7 fixes)**
- PatchTST seq_len alignment: hardcoded 128 → 60 (matches contract and SeqConfig default)
- batch_size alignment: 256 → 512 unified across 4 config paths (defaults.py, experiment.py, unified.py, trainer_config.py)
- OOF -99 sentinel filtering: PreparedData.filter_invalid_labels() before OOF generation
- y_true column added to all 3 OOF DataFrames (oof_core.py, oof_sequence.py, oof_generation.py)
- Binary mode experimental warning in factory.py
- Notebook Cell 23 confusion matrix fix + dynamic class labels; Cell 2 dead variable docs
- Deferred: n_classes threading (30+ files) → future phase
- 11 files modified, 212/212 tests still passing

**Phase 86: COMPLETE — Wire Triple-Barrier Params + Hardcoded Fixes (4 files, 75/75 checks)**
- CRITICAL: Backtest was playing a different game than training — ATR never passed to `_open_position()`, stop was always 2% fallback, take_profit always None, max_holding_period always 0
- Added `barrier_k_up` and `barrier_k_down` fields to BacktestConfig (default 0.0 = legacy mode)
- Added `_compute_atr()` method to Backtester (ATR(14) from high/low/close)
- Barrier-aligned stop/TP: LONG stop = price - k_down*ATR, TP = price + k_up*ATR; SHORT reversed
- Barrier-aware position sizing: uses actual barrier distance instead of hardcoded 2%
- Factory auto-wires from `get_barrier_params(symbol, horizon)` — k_up, k_down, max_bars
- execution.py: Per-contract session times (MGC=COMEX 8:20-13:30, MES=NYSE 9:30-16:00), configurable adverse selection params
- costs.py: Per-symbol slippage defaults (SYMBOL_SLIPPAGE_DEFAULTS) for MES/MGC/MNQ
- Fixed logger shadowing bug in 3 circuit breaker blocks (caused UnboundLocalError with barriers active)
- Backward compatible: barrier_k_up=0.0 → legacy path unchanged
- 4 files modified (backtest.py, factory.py, execution.py, costs.py), 212/212 tests passing, 75/75 verification checks pass

**Phase 87: COMPLETE — DSR Metric Gate + Phase 86 Tests + Notebook Docs (3 items, 7 files)**
- DSR gate fix: `is_sharpe_like_metric()` utility guards DSR computation — only applies to Sharpe-like metrics (sharpe_ratio, sortino_ratio, calmar_ratio), skips for F1/accuracy/precision/recall
- Gated in `cv_tuner.py` (skips DSR when metric is not Sharpe-like) and `five_dimension_objective.py` (skips when custom non-Sharpe metric provided)
- 12 new Phase 86 tests: ATR computation, barrier stop/TP (long/short/asymmetric/legacy), per-contract sessions, per-symbol slippage, factory barrier wiring (total 60 tests, all pass)
- Notebook: barrier alignment documentation cell added before Backtest Results section
- 7 files modified, 60/60 tests passing, ruff + black clean

**Phase 88: COMPLETE — Safety Guards (5 items, 6 files, 8/8 verification checks)**
- Thread-safe label cache: `threading.Lock()` around all `_label_cache` access in `five_dimension_objective.py` (fixes race condition in parallel Optuna with n_jobs > 1)
- Execution model guard: `_get_execution_price()` else clause raises `ValueError` instead of silently defaulting to close
- DEBUG_PATH removal: removed debug logging line from `trainer.py`
- n_classes parameterized: `oof_sequence.py`, `oof_core.py`, `oof_generation.py` — `n_classes=3` default for backward compat, enables binary mode pipeline
- Configurable alignment threshold: `BacktestConfig.alignment_loss_warn_pct` replaces hardcoded 5% magic number
- 6 files modified, 223/223 tests passing, ruff + black clean

**Phase 89: COMPLETE — Pipeline Speed Optimizations (~3-5x for 1.6M rows, 6 items, 9 files)**
- MDA subsampling in purged_selector: 50K train + 20K test cap per fold (was RF on 1.28M rows — ~25x speedup for feature selection)
- Numba liquidity: `_rolling_percentile_rank_numba` @njit replaces pandas rolling.apply for 3 liquidity regime features (~10-25x per call)
- Backtest numpy: pre-extract columns as numpy arrays, eliminate all data.iloc[i] calls in hot loop (~5-10x backtest). Fixed O(n^2) drawdown circuit breaker.
- Feature selection: n_repeats 5→3 (1.67x), n_estimators 100→50 (2x), MTF .copy() eliminated
- 9 files modified, 223/223 tests passing, ruff + black clean

**Phase 90: COMPLETE — CUDA Memory Guards + Model-Specific Optimizations (4 items, 10 files)**
- All 12 model families audited: GPU auto-detect, float32, proper cleanup, DataLoader settings, early stopping, OOM recovery — all verified
- CUDA allocator config (`expandable_segments:True,max_split_size_mb:256`) in device.py — reduces GPU memory fragmentation
- `torch.cuda.synchronize()` before `empty_cache()` at 10 sites — prevents async CUDA race conditions
- CatBoost `gpu_ram_part=0.95` cap — prevents GPU OOM interference with other models
- XGBoost `QuantileDMatrix` — ~4x memory reduction for data matrix (pre-bins to quantile buckets)
- 10 files modified, 223/223 tests passing, ruff + black clean

**Phase 91: COMPLETE — Gradient Checkpointing + TFT SDPA (3 files)**
- Gradient checkpointing for PatchTST, iTransformer, TFT (opt-in: `{"gradient_checkpointing": true}`)
  Trades ~20-30% speed for 30-50% activation memory savings on transformer encoder/attention
- TFT InterpretableMultiHeadAttention uses SDPA during training (Flash Attention fused kernel)
  O(n²)→O(n) attention memory, ~2x faster. Manual attention preserved for inference interpretability
- Confirmed: PatchTST + iTransformer already get Better Transformer fast path (nn.TransformerEncoder)
- 3 files modified, 223/223 tests passing, ruff + black clean

**Phase 92: COMPLETE — Optuna Robustness + Hardcoded Values + Sequential Ensemble (6 files)**
- CRITICAL: Annualization factor derived from data frequency (was hardcoded 252*78 — broke crypto/other timeframes)
- CRITICAL: Failed Optuna trials return `-inf` (was `0.0`, polluting best_value selection)
- CRITICAL: Default 12h Optuna timeout (was unlimited — could run forever)
- Ensemble sequential safety: boosting parallel training now respects `parallel_training` config flag (default False = one model at a time)
- Optuna config: n_startup_trials 5→10, variance penalty/max_samples/n_startup_trials now in OptunaConfig
- Hardcoded magic numbers moved to constants.py (trade rate thresholds, max_features_to_search)
- 6 files modified, 223/223 tests passing, ruff + black clean

**Phase 93: COMPLETE — Per-Symbol ADX Regime Thresholds (7 files)**
- ADX trending threshold now per-symbol: MES=20.0, MGC=23.0, MNQ=25.0 (was hardcoded 25.0)
- `SymbolConfig.adx_trending_threshold` field with per-symbol presets and positive validation
- `get_regime_config(symbol)` returns REGIME_CONFIG copy with per-symbol adx_threshold
- `PipelineConfig.__post_init__()` auto-wires regime_adx_threshold from SymbolConfig
- `add_adx()` and `compute_adx_strong_trend()` accept threshold parameter (backward compatible)
- 7 files modified, 223/223 tests passing, ruff + black clean

**Phase 94: COMPLETE — Deep Audit Phase A: 8 Critical Fixes (19 files, 304 lines)**
- CRITICAL: Unified ATR to Wilder's EMA (alpha=1/period) in labeling + backtest (was EMA vs SMA — labels and backtest now play same game)
- CRITICAL: Stop/TP exits now at barrier price, not close (ExitReason enum, _get_exit_price dispatch) — eliminates systematic P&L bias
- CRITICAL: Unified label/backtest costs (MES 2.43 ticks, MGC 3.04, MNQ 6.08) — eliminates 2.95x cost gap
- CRITICAL: Feature selection moved to train-only data (`_run_feature_selection_on_train_data()`) — eliminates textbook data leakage
- HIGH: 50-feature truncation replaced with importance-based selection (quick LightGBM ranking) — all features now get evaluated
- HIGH: Cumulative features (OBV/VWAP/TWAP/cum_order_flow) reset at session boundaries via `session_cumsum()` — eliminates positional leakage
- MEDIUM: ExperimentConfig round-trip fixed (idempotent output_dir check) — config persistence works
- MEDIUM: Degenerate Optuna trials return -inf (was 0.0) — stops TPE pollution
- 19 files modified, 223/223 tests passing, ruff + black clean

**Phase 95: COMPLETE — Deep Audit Phase B: 9 High-Value Fixes (8 files)**
- Binary mode: n_classes parameterized in stacking.py (3 sites, default=3 for backward compat)
- Entropy lookahead: shift(1) added to all 12 entropy compute functions (prevents using current bar's close)
- 4D OOF scaling: per-fold median/IQR scaling in _generate_4d_oof (matches 2D tabular OOF pattern)
- Yang-Zhang k: formula parameterized by window (`(window+1)/(window-1)` instead of hardcoded 21/19)
- Adverse selection: wired into backtest entry prices (long + short) with NaN-safe ATR proxy
- Sharpe annualization: derived from data frequency via _derive_periods_per_year() (was hardcoded 252)
- Hard vote tie-breaking: random selection with deterministic seed (was np.argmax bias toward class -1)
- OOF generator: verified stateless — all config comes from OOFRequest per model call
- Volatility annualization: _ANNUAL_FACTOR constant replaces 7x hardcoded np.sqrt(252)
- CAGR overflow guard for negative total returns + NaN safety in metrics
- 8 files modified, 223/223 tests passing, ruff + black clean

**See CLEANUP_PLAN.md for full phase details.**

---

## Workflow Patterns

### Making Changes

1. **Spawn specialized agents** for analysis (3-7 depending on scope)
2. **Verify findings** before acting (batch verification for large changes)
3. **Execute with handoffs** (sequential agents, each passes context to next)
4. **Verify after each step** (spawn verification subagent)
5. **Update documentation** (CLEANUP_TASKS during, COMPLETION after)

### When to Ask User

- Architectural decisions
- Deleting more than one file
- Changes to core interfaces
- Anything not in current phase scope
- When unsure about intent

### Verification Commands

```bash
# Import checks (should all succeed)
python -c "from src.core.types import DataRank, ModelFamily; print('OK')"
python -c "from src.core.contracts import get_model_contract; print('OK')"
python -c "from src.data.adapters import get_adapter; print('OK')"

# Single definition checks (should each return 1)
grep -r "class DataRank" src/ --include="*.py" | wc -l
grep -r "class ModelFamily" src/ --include="*.py" | wc -l

# Dead import checks (should return 0)
grep -r "from src\.coordination" src/ --include="*.py" | wc -l
grep -r "from src\.feature_selection" src/ --include="*.py" | wc -l
```

---

## Don'ts

| Don't | Do Instead |
|-------|------------|
| Make changes without reading docs | Read DIRECTION + CLEANUP_PLAN first |
| Execute without user approval | Propose approach, wait for confirmation |
| Skip linting | Run `ruff check` and `black` before commit |
| Create duplicate definitions | Import from canonical location |
| Comment out dead code | Delete it completely |
| Ignore validation failures | Fix or document as exception |

---

## Documented Exceptions

| Exception | Reason | Status |
|-----------|--------|--------|
| Dual AdapterResult | Circular import prevention | Bidirectional properties added |
| Pyright pandas errors | Type stub limitations | Not blocking, document when seen |

---

## Commands

**See COMMANDS.md** for the full command system reference including:
- Visual command matrix (tiered by scope)
- Subagent architecture and reference
- Standard workflows for phases and quick fixes
- Anti-patterns to avoid

Quick reference:
| Category | Light | Medium | Heavy |
|----------|-------|--------|-------|
| Analyze | `/analysis-targeted(1c)` | `/analysis-optimization(1b)` | `/analysis-full(1a)` |
| Verify | `/verify-claim(2a)` | `/verify-batch(2b)` | `/verify-contracts(2c)` |
| Execute | `/execute-surgical(4c)` | `/execute-standard(4a)` | `/execute-large(4b)` |
| Docs | `/docs-tasks(3a)` | `/docs-full(3b)` | `/docs-final(6a)` |
| Check | `/check-standard(5a)` | `/check-deep(5b)` | `/check-behavior(5c)` |

---

## Templates

Templates for DIRECTION, CLEANUP_PLAN, CLEANUP_TASKS, and COMPLETION are in:
`X ( IN PROGRESS DOCS) X/TEMPLATES/`

Use these when starting fresh or resetting documentation.

---

*Last updated: 2026-02-28*
*See CLEANUP_PLAN.md for current phase*
*See COMMANDS.md for command reference*
