# Migration Roadmap: Universal Pipeline → Model-Specific Strategies

**Status:** 🛣️ Implementation Plan
**Purpose:** Detailed plan to migrate from current universal pipeline to intended model-specific architecture
**Effort:** 6-8 weeks (1 engineer) | 4-5 weeks (2 engineers)
**Last Updated:** 2026-01-01

---

## Executive Summary

**Goal:** Migrate from universal pipeline (all models get same indicators) to model-specific strategies (tabular → indicators, sequence → raw bars)

**Approach:** Incremental enhancement, not rewrite
- Keep current pipeline as "Strategy 2 (partial)"
- Add Strategy 1 (single-TF baselines)
- Add Strategy 3 (multi-resolution ingestion)
- Enhance Strategy 2 (complete 9-TF ladder)
- Update documentation to clarify current vs intended

**Timeline:** 6 phases over 6-8 weeks

---

## Phase 0: Prerequisites (Before Starting)

### Checkpoint: Understand Current State

**Tasks:**
1. Read `docs/CURRENT_LIMITATIONS.md` - Understand what's wrong
2. Read `docs/INTENDED_ARCHITECTURE.md` - Understand the goal
3. Read `docs/CURRENT_VS_INTENDED_ARCHITECTURE.md` - Understand the gap
4. Read `docs/roadmaps/MTF_IMPLEMENTATION_ROADMAP.md` - Detailed implementation tasks

**Validation:**
- [ ] Can explain why sequence models shouldn't get indicators
- [ ] Can explain 3 MTF strategies
- [ ] Can explain 9-timeframe ladder
- [ ] Understand model-strategy mapping

---

## Phase 1: MTF Infrastructure (1-2 weeks)

**Goal:** Add missing timeframes, make training timeframe configurable

### Task 1.1: Add 20min and 25min Timeframes
**Duration:** 1 day
**Files:** `src/phase1/stages/mtf/constants.py`, `src/phase1/config/features.py`

```python
# Before
MTF_TIMEFRAMES = ['15min', '30min', '1h', '4h', 'daily']

# After
MTF_TIMEFRAMES = ['1min', '5min', '10min', '15min', '20min', '25min', '30min', '45min', '1h']
```

**Tests:**
- `pytest tests/phase_1_tests/stages/test_mtf_features.py::test_20min_resampling`
- `pytest tests/phase_1_tests/stages/test_mtf_features.py::test_25min_resampling`

### Task 1.2: Add `training_timeframe` to PipelineConfig
**Duration:** 1 day
**Files:** `src/phase1/pipeline_config.py`

```python
# Add new fields
@dataclass
class PipelineConfig:
    ingest_timeframe: str = '1min'        # Always 1min (raw data)
    training_timeframe: str = '5min'      # Configurable training TF
    mtf_strategy: str = 'single_tf'       # NEW: Strategy selection
```

**Tests:**
- `pytest tests/phase_1_tests/test_pipeline_config.py::test_training_timeframe`

### Task 1.3: Update Clean Stage
**Duration:** 2 days
**Files:** `src/phase1/stages/clean/pipeline.py`

```python
# Before (hardcoded)
df_5min = resample_ohlcv(df, freq='5min')

# After (configurable)
df_training = resample_ohlcv(df, freq=config.training_timeframe)
```

**Tests:**
- `pytest tests/phase_1_tests/stages/test_clean.py::test_resample_all_timeframes`

### Task 1.4: Update MTF Stage
**Duration:** 2 days
**Files:** `src/phase1/stages/mtf/generator.py`

```python
# Use training_timeframe as base
mtf_generator = MTFFeatureGenerator(
    base_timeframe=config.training_timeframe,  # Changed from '5min'
    mtf_timeframes=get_higher_timeframes(config.training_timeframe),
    mode=config.mtf_mode
)
```

**Tests:**
- `pytest tests/phase_1_tests/stages/test_mtf_features.py::test_base_timeframe_config`

### Task 1.5: Auto-Scale Purge/Embargo
**Duration:** 1 day
**Files:** `src/phase1/config/features.py`

```python
def auto_scale_purge_embargo(horizons, timeframe='5min'):
    """Scale purge/embargo based on timeframe."""
    tf_minutes = parse_timeframe_to_minutes(timeframe)
    bars_per_day = (6.5 * 60) / tf_minutes
    embargo_bars = int(bars_per_day * 5)  # 5 trading days
    return purge_bars, embargo_bars
```

**Tests:**
- `pytest tests/phase_1_tests/test_purge_embargo_scaling.py`

**Phase 1 Deliverable:**
- ✅ 9-timeframe ladder complete
- ✅ Configurable training timeframe
- ✅ Auto-scaled purge/embargo
- ✅ All tests passing

---

## Phase 2: Strategy 1 - Single-Timeframe (1 week)

**Goal:** Support training on one timeframe without MTF features

### Task 2.1: Add `mtf_strategy` Config
**Duration:** 1 day
**Files:** `src/phase1/pipeline_config.py`

```python
@dataclass
class PipelineConfig:
    mtf_strategy: str = 'single_tf'  # Options: single_tf, mtf_indicators, mtf_ingestion
    mtf_source_timeframes: list[str] | None = None
    mtf_input_timeframes: list[str] | None = None
```

**Tests:**
- `pytest tests/phase_1_tests/test_pipeline_config.py::test_mtf_strategy_validation`

### Task 2.2: Implement Strategy 1 Logic
**Duration:** 2 days
**Files:** `src/phase1/stages/features/run.py`

```python
if config.mtf_strategy == 'single_tf':
    logger.info("Strategy 1: Skipping MTF features")
    # Compute only base timeframe features (~40 features)
    df = compute_base_features(df, config)
elif config.mtf_strategy == 'mtf_indicators':
    logger.info("Strategy 2: Adding MTF indicators")
    df = compute_base_features(df, config)
    df = add_mtf_features(df, config)
```

**Tests:**
- `pytest tests/phase_1_tests/stages/test_features.py::test_strategy_1_single_tf`

### Task 2.3: Update Model Trainers
**Duration:** 1 day
**Files:** `scripts/train_model.py`

```bash
python scripts/train_model.py \
    --model xgboost \
    --training-timeframe 15min \
    --mtf-strategy single_tf \
    --horizon 20
```

**Tests:**
- `pytest tests/phase_2_tests/test_training_strategy_1.py`

### Task 2.4: Integration Tests
**Duration:** 1 day
**Files:** `tests/integration/test_mtf_strategy_1.py`

**Test matrix:**
- Train XGBoost on 1min, 5min, 15min, 1h (single-TF each)
- Verify feature counts (~40 per timeframe)
- Compare performance across timeframes

**Phase 2 Deliverable:**
- ✅ Strategy 1 working for all models
- ✅ Baseline performance benchmarks
- ✅ CLI supports --mtf-strategy flag

---

## Phase 3: Strategy 2 Enhancement (1 week)

**Goal:** Enhance MTF indicators with full 9-TF ladder, tabular-only restriction

### Task 3.1: Add `mtf_source_timeframes` Config
**Duration:** 1 day
**Files:** `src/phase1/stages/mtf/generator.py`

```python
if config.mtf_strategy == 'mtf_indicators':
    if config.mtf_source_timeframes:
        mtf_timeframes = config.mtf_source_timeframes
    else:
        # Default: all higher timeframes
        mtf_timeframes = get_higher_timeframes(config.training_timeframe)
```

**Tests:**
- `pytest tests/phase_1_tests/stages/test_mtf_source_timeframes.py`

### Task 3.2: Model-Specific Recommendations
**Duration:** 2 days
**Files:** `src/models/mtf_config.py` (new)

```python
MTF_RECOMMENDATIONS = {
    'xgboost': {
        'strategy': 'mtf_indicators',
        'training_timeframe': '15min',
        'source_timeframes': ['1min', '5min', '30min', '1h'],
    },
    'lstm': {
        'strategy': 'single_tf',  # Until Strategy 3 implemented
        'training_timeframe': '15min',
    },
}
```

**Tests:**
- `pytest tests/phase_2_tests/test_mtf_recommendations.py`

### Task 3.3: Feature Count Validation
**Duration:** 1 day
**Files:** `src/phase1/stages/validation/integrity.py`

```python
EXPECTED_FEATURE_COUNTS = {
    'single_tf': {'5min': 40, '15min': 40, '1h': 40},
    'mtf_indicators': {'5min': (100, 200), '15min': (80, 150)},
}
```

**Tests:**
- `pytest tests/phase_1_tests/stages/test_feature_count_validation.py`

### Task 3.4: Documentation Update
**Duration:** 1 day
**Files:** `docs/phases/PHASE_1.md`, `CLAUDE.md`

**Phase 3 Deliverable:**
- ✅ Strategy 2 enhanced with 9-TF ladder
- ✅ Model-specific MTF recommendations
- ✅ Validation detects unexpected feature counts

---

## Phase 4: Strategy 3 - Multi-Resolution Ingestion (2 weeks)

**Goal:** Support feeding raw multi-resolution OHLCV to sequence models

### Task 4.1: Multi-Resolution Dataset Builder
**Duration:** 3 days
**Files:** `src/phase1/stages/datasets/multi_resolution.py` (new)

```python
class MultiResolutionDatasetBuilder:
    def build_multi_resolution_tensors(
        self, df_1min, labels_df
    ) -> dict[str, np.ndarray]:
        """
        Returns:
            {
                '1min': (n_samples, 60, 5),
                '5min': (n_samples, 12, 5),
                '15min': (n_samples, 4, 5),
                '1h': (n_samples, 1, 5),
            }
        """
```

**Tests:**
- `pytest tests/phase_1_tests/stages/test_multi_resolution_builder.py`

### Task 4.2: Update TimeSeriesDataContainer
**Duration:** 2 days
**Files:** `src/phase1/stages/datasets/container.py`

```python
class TimeSeriesDataContainer:
    def get_multi_resolution_tensors(
        self, split, input_timeframes, lookback_minutes=60
    ) -> tuple[dict, np.ndarray, np.ndarray]:
        """Get multi-resolution tensors for sequence models."""
```

**Tests:**
- `pytest tests/phase_1_tests/stages/test_container_multi_resolution.py`

### Task 4.3: Tensor Utilities
**Duration:** 1 day
**Files:** `src/phase1/stages/datasets/tensor_utils.py` (new)

```python
def concatenate_multi_resolution(tensors, timeframe_order):
    """Concatenate along sequence dimension."""

def stack_multi_resolution(tensors, timeframe_order, pad_to_max_len=True):
    """Stack into 4D tensor."""
```

**Tests:**
- `pytest tests/phase_1_tests/stages/test_tensor_utils.py`

### Task 4.4: Update Model Trainers
**Duration:** 3 days
**Files:** `src/models/neural/base_rnn.py`, `scripts/train_model.py`

```python
if args.mtf_strategy == 'mtf_ingestion':
    X_multi_train, y_train, weights = container.get_multi_resolution_tensors(
        split='train',
        input_timeframes=args.mtf_input_timeframes
    )

    if args.model in ['lstm', 'tcn']:
        X_train = concatenate_multi_resolution(X_multi_train)
    elif args.model in ['patchtst']:
        X_train = stack_multi_resolution(X_multi_train)
```

**Tests:**
- `pytest tests/phase_2_tests/test_training_strategy_3.py`

### Task 4.5: Integration Tests
**Duration:** 2 days
**Files:** `tests/integration/test_mtf_strategy_3.py`

**Test matrix:**
- Build multi-resolution for LSTM with [1min, 5min, 15min]
- Build multi-resolution for TCN with [1min, 5min, 15min, 1h]
- Compare performance: single-TF vs multi-resolution

**Phase 4 Deliverable:**
- ✅ Strategy 3 working for sequence models
- ✅ Multi-resolution tensor builder
- ✅ LSTM/GRU/TCN/Transformer support raw bars

---

## Phase 5: Model Integration (1 week)

**Goal:** Update all models and CLI for 3 strategies

### Task 5.1: Model Input Validation
**Duration:** 3 days
**Files:** All 19 model files

```python
def fit(self, X_train, y_train, ...):
    if self.family == 'boosting':
        assert X_train.ndim == 2
    elif self.family == 'neural':
        assert X_train.ndim in [3, 4]
```

**Tests:**
- `pytest tests/phase_2_tests/test_model_input_validation.py`

### Task 5.2: CLI Updates
**Duration:** 2 days
**Files:** `scripts/train_model.py`, `scripts/run_cv.py`, `pipeline`

```bash
python scripts/train_model.py \
    --model lstm \
    --training-timeframe 15min \
    --mtf-strategy mtf_ingestion \
    --mtf-input-timeframes 1min 5min 15min 1h \
    --horizon 20
```

**Tests:**
- `pytest tests/cli/test_mtf_cli_flags.py`

### Task 5.3: Config YAML Examples
**Duration:** 1 day
**Files:** `config/mtf_strategy_{1,2,3}.yaml` (new)

**Phase 5 Deliverable:**
- ✅ All 19 models support all 3 strategies
- ✅ CLI supports all MTF flags
- ✅ Example configs for each strategy

---

## Phase 6: Production & Documentation (1 week)

**Goal:** Production-ready with comprehensive docs and tests

### Task 6.1: Inference Pipeline
**Duration:** 3 days
**Files:** `src/inference/pipeline.py`, `scripts/serve_model.py`

```python
class InferencePipeline:
    def predict(self, df_1min):
        # 1. Resample to training_timeframe
        # 2. Apply MTF strategy
        # 3. Scale features
        # 4. Model prediction
```

**Tests:**
- `pytest tests/inference/test_mtf_inference.py`

### Task 6.2: Config Validation
**Duration:** 1 day
**Files:** `src/phase1/config/mtf_validation.py` (new)

**Validations:**
1. mtf_strategy valid
2. training_timeframe valid
3. Strategy 2: source TFs >= training TF
4. Strategy 3: input TFs specified
5. Model supports chosen strategy

**Tests:**
- 20+ validation test cases

### Task 6.3: Comprehensive Testing
**Duration:** 2 days
**Files:** `tests/integration/test_mtf_end_to_end.py`

**Test matrix:** 3 strategies × 4 model families = 12 integration tests

**Phase 6 Deliverable:**
- ✅ Production-ready inference
- ✅ Comprehensive validation
- ✅ Full test coverage

### Task 6.4: Documentation Update
**Duration:** 1 day
**Files:** Update all major docs

**Files to update:**
- `CLAUDE.md` - Update architecture description
- `docs/phases/PHASE_1.md` - Add MTF strategy section
- `docs/QUICK_REFERENCE.md` - Add MTF commands
- `docs/INDEX.md` - Update with new docs

**Files to create:**
- `docs/MTF_GUIDE.md` - User guide for MTF strategies

**Phase 6 Deliverable:**
- ✅ All documentation updated
- ✅ Migration guide complete
- ✅ Troubleshooting guide

---

## Dependency Graph

```
Phase 1: Infrastructure (1-2 weeks)
    ↓
Phase 2: Strategy 1 ────┐
    ↓                   │
Phase 3: Strategy 2 ────┤ Can be parallel
    ↓                   │
Phase 4: Strategy 3 ────┘
    ↓
Phase 5: Model Integration (1 week)
    ↓
Phase 6: Production & Docs (1 week)
```

**Parallelization:**
- Phases 2, 3, 4 can be partially parallelized with 2 engineers
- Phase 2 → 1 engineer (simpler)
- Phase 3 + 4 → 1 engineer (can do sequentially)

---

## Critical Files to Modify

### Top 10 Most Critical Files

1. **`src/phase1/pipeline_config.py`** - Core config changes
2. **`src/phase1/stages/mtf/generator.py`** - MTF generation logic
3. **`src/phase1/stages/datasets/multi_resolution.py`** - NEW: Strategy 3 builder
4. **`src/phase1/stages/features/run.py`** - Feature orchestration
5. **`scripts/train_model.py`** - Model training CLI
6. **`src/models/registry.py`** - MTF metadata
7. **`src/phase1/stages/datasets/container.py`** - Multi-resolution support
8. **`src/inference/pipeline.py`** - Inference with MTF
9. **`tests/integration/test_mtf_end_to_end.py`** - Integration tests
10. **`docs/phases/PHASE_1.md`** - Documentation

---

## Success Metrics

| Phase | Success Metric | Target |
|-------|---------------|--------|
| Phase 1 | 9-timeframe ladder | All 9 TFs working |
| Phase 2 | Strategy 1 working | All models train on single-TF |
| Phase 3 | Strategy 2 enhanced | 9-TF ladder for tabular models |
| Phase 4 | Strategy 3 working | Raw bars for sequence models |
| Phase 5 | Model integration | All 19 models support all strategies |
| Phase 6 | Production ready | Full docs, tests, inference |

---

## Quick Start Commands (After Migration)

### Strategy 1: Single-Timeframe Baseline
```bash
python scripts/train_model.py \
    --model xgboost \
    --training-timeframe 15min \
    --mtf-strategy single_tf \
    --horizon 20
```

### Strategy 2: MTF Indicators (Tabular)
```bash
python scripts/train_model.py \
    --model xgboost \
    --training-timeframe 15min \
    --mtf-strategy mtf_indicators \
    --mtf-source-timeframes 1min 5min 30min 1h \
    --horizon 20
```

### Strategy 3: MTF Ingestion (Sequence)
```bash
python scripts/train_model.py \
    --model lstm \
    --training-timeframe 15min \
    --mtf-strategy mtf_ingestion \
    --mtf-input-timeframes 1min 5min 15min 1h \
    --horizon 20
```

---

## Risk Mitigation

### Risk 1: Breaking Existing Pipelines
**Mitigation:** Default to current behavior (Strategy 2 partial)
```python
# Default config preserves current behavior
training_timeframe = '5min'  # Current default
mtf_strategy = 'mtf_indicators'  # Current behavior (but warns incomplete)
```

### Risk 2: Performance Regression
**Mitigation:** Comprehensive benchmarking before/after
- Run existing pipelines before migration
- Run identical pipelines after migration
- Compare outputs, performance metrics
- Require < 1% difference in metrics

### Risk 3: Incomplete Testing
**Mitigation:** Test matrix coverage
- 3 strategies × 4 families = 12 integration tests minimum
- All 19 models tested individually
- Inference pipeline tested for all strategies

---

## Rollback Plan

If migration fails, can rollback to current state:

1. **Keep current code in git branch** - Never force push
2. **Feature flags** - Disable new strategies via config
3. **Backward compatibility** - Old configs still work
4. **Gradual rollout** - Test on subset of models first

---

## Post-Migration Validation

After completing all 6 phases:

### Functional Validation
- [ ] All 19 models train successfully with Strategy 1
- [ ] 6 tabular models train with Strategy 2 (9 TFs)
- [ ] 13 sequence models train with Strategy 3 (raw bars)
- [ ] Inference works for all strategies
- [ ] All tests passing (unit + integration)

### Performance Validation
- [ ] Strategy 3 sequence models outperform Strategy 2 (if not, investigate)
- [ ] Strategy 2 tabular models match or exceed current performance
- [ ] Strategy 1 baselines provide meaningful comparisons

### Documentation Validation
- [ ] All docs distinguish current vs intended clearly
- [ ] Migration guide helps users transition
- [ ] Troubleshooting guide covers common issues
- [ ] API docs updated

---

## Timeline Summary

| Phase | Duration | Can Parallelize? |
|-------|----------|------------------|
| Phase 1: Infrastructure | 1-2 weeks | No (foundation) |
| Phase 2: Strategy 1 | 1 week | Yes (with Phase 3) |
| Phase 3: Strategy 2 | 1 week | Yes (with Phase 2) |
| Phase 4: Strategy 3 | 2 weeks | Yes (with Phase 2/3) |
| Phase 5: Integration | 1 week | No (depends on 2-4) |
| Phase 6: Production | 1 week | No (final) |
| **Total (1 engineer)** | **6-8 weeks** | Sequential |
| **Total (2 engineers)** | **4-5 weeks** | Parallelized |

---

## References

- **Intended Architecture:** `docs/INTENDED_ARCHITECTURE.md`
- **Current Limitations:** `docs/CURRENT_LIMITATIONS.md`
- **Current vs Intended:** `docs/CURRENT_VS_INTENDED_ARCHITECTURE.md`
- **Detailed MTF Tasks:** `docs/roadmaps/MTF_IMPLEMENTATION_ROADMAP.md`
- **Model Requirements:** `docs/research/FEATURE_REQUIREMENTS_BY_MODEL.md`

---

**Next Step:** Start with Phase 1 (MTF Infrastructure). Do not proceed to Phase 2 until Phase 1 is complete and tested.
