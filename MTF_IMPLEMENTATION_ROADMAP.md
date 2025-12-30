# Multi-Timeframe (MTF) Implementation Roadmap

**Generated:** 2025-12-29
**Purpose:** Step-by-step guide to implement 3 MTF strategies for production ML pipeline
**Total Effort:** 6-8 weeks (1 engineer) | 4-5 weeks (2 engineers)

---

## Table of Contents

1. [Overview](#overview)
2. [Current State Analysis](#current-state-analysis)
3. [Phase 1: MTF Resampling Infrastructure (1-2 weeks)](#phase-1-mtf-resampling-infrastructure-1-2-weeks)
4. [Phase 2: Strategy 1 - Single-Timeframe (1 week)](#phase-2-strategy-1---single-timeframe-training-1-week)
5. [Phase 3: Strategy 2 - MTF Indicators (1 week)](#phase-3-strategy-2---mtf-indicators-enhancement-1-week)
6. [Phase 4: Strategy 3 - MTF Ingestion (2 weeks)](#phase-4-strategy-3---multi-resolution-ingestion-2-weeks)
7. [Phase 5: Model Integration (1 week)](#phase-5-model-integration-and-cli-updates-1-week)
8. [Phase 6: Production Deployment (1 week)](#phase-6-production-deployment-and-testing-1-week)
9. [Dependency Graph](#dependency-graph)
10. [Critical Files](#critical-files-for-implementation)

---

## Overview

This roadmap implements **configurable multi-timeframe (MTF) training** with **3 distinct strategies**:

| Strategy | Description | Use Case | Models |
|----------|-------------|----------|--------|
| **Strategy 1: Single-TF** | Train on one timeframe, no MTF | Baselines, simple models | All models |
| **Strategy 2: MTF Indicators** | Train on one TF, add features from other TFs | Dense feature engineering | XGBoost, LightGBM |
| **Strategy 3: MTF Ingestion** | Feed multiple TF tensors together | Cross-scale learning | LSTM, TCN, PatchTST, TFT |

**Key architectural change:** `1min ingest` ≠ `training_timeframe` (configurable: 1m, 5m, 10m, 15m, 20m, 25m, 30m, 45m, 1h)

---

## Current State Analysis

### What Exists
✅ MTF module (`src/phase1/stages/mtf/`) with resampling and feature alignment
✅ Anti-lookahead protection (shift(1) before forward-fill)
✅ Timeframes: 1m, 5m, 10m, 15m, 30m, 45m, 1h, 4h, daily

### What's Missing
❌ **20m, 25m timeframes** (needed for 9-timeframe ladder)
❌ **Configurable training timeframe** (currently hardcoded to 5min)
❌ **Strategy 1** (single-timeframe, no MTF)
❌ **Strategy 3** (multi-resolution tensor ingestion)
❌ **Model-specific MTF configurations**

---

## Phase 1: MTF Resampling Infrastructure (1-2 weeks)

**Goal:** Add 9-timeframe ladder support and make training timeframe configurable

### Task 1.1: Add 20m and 25m to Timeframe Ladder
**Duration:** 1 day (4 hours)

**Files to modify:**
- `src/phase1/stages/mtf/constants.py`
- `src/phase1/config/features.py`

**Changes:**
```python
# src/phase1/stages/mtf/constants.py
MTF_TIMEFRAMES = {
    '1min': 1,
    '5min': 5,
    '10min': 10,
    '15min': 15,
    '20min': 20,   # NEW
    '25min': 25,   # NEW
    '30min': 30,
    '45min': 45,
    '60min': 60,
    '1h': 60,
}

PANDAS_FREQ_MAP = {
    # ... existing ...
    '20min': '20min',  # NEW
    '25min': '25min',  # NEW
}
```

**Success criteria:**
- [ ] 20min and 25min added to MTF_TIMEFRAMES
- [ ] 20min and 25min added to PANDAS_FREQ_MAP
- [ ] Tests pass: `pytest tests/phase_1_tests/stages/test_mtf_features.py::test_20min_resampling`
- [ ] Tests pass: `pytest tests/phase_1_tests/stages/test_mtf_features.py::test_25min_resampling`

---

### Task 1.2: Add training_timeframe to PipelineConfig
**Duration:** 1 day (6 hours)

**Files to modify:**
- `src/phase1/pipeline_config.py`
- `src/phase1/config/features.py`

**Changes:**
```python
# src/phase1/pipeline_config.py
@dataclass
class PipelineConfig:
    # ... existing fields ...

    # NEW: Separate ingest from training timeframe
    ingest_timeframe: str = '1min'  # Always 1min (raw data ingestion)
    training_timeframe: str = '5min'  # Configurable training TF

    # DEPRECATED: target_timeframe (use training_timeframe instead)
    target_timeframe: str = field(default=None)

    def __post_init__(self):
        # Backward compatibility
        if self.target_timeframe is not None:
            self.training_timeframe = self.target_timeframe

        # Validation: training_timeframe must be in supported list
        valid_tfs = ['1min', '5min', '10min', '15min', '20min', '25min', '30min', '45min', '1h']
        if self.training_timeframe not in valid_tfs:
            raise ValueError(f"training_timeframe must be one of {valid_tfs}")
```

**Success criteria:**
- [ ] `training_timeframe` field added to PipelineConfig
- [ ] `ingest_timeframe` defaults to '1min'
- [ ] Backward compatibility with `target_timeframe`
- [ ] Validation: training_timeframe must be in [1m, 5m, 10m, 15m, 20m, 25m, 30m, 45m, 1h]
- [ ] Tests pass: `pytest tests/phase_1_tests/test_pipeline_config.py::test_training_timeframe_config`

---

### Task 1.3: Update Clean Stage to Support Configurable Training Timeframe
**Duration:** 2 days (12 hours)

**Files to modify:**
- `src/phase1/stages/clean/pipeline.py`
- `src/phase1/stages/clean/run.py`

**Changes:**
```python
# src/phase1/stages/clean/pipeline.py
def run_clean_pipeline(
    df: pd.DataFrame,
    config: PipelineConfig,
    # ...
) -> pd.DataFrame:
    # Use config.training_timeframe instead of hardcoded '5min'
    df_resampled = resample_ohlcv(
        df,
        freq=config.training_timeframe,  # Changed from '5min'
        # ...
    )
    return df_resampled
```

**Success criteria:**
- [ ] Clean stage uses `config.training_timeframe` for resampling
- [ ] Tests pass for all 9 timeframes: `pytest tests/phase_1_tests/stages/test_clean.py::test_resample_all_timeframes`
- [ ] Verify deterministic resampling: same input + seed → same output
- [ ] Smoke test: Run pipeline with training_timeframe=15min

---

### Task 1.4: Update MTF Stage to Use training_timeframe as Base
**Duration:** 2 days (14 hours)

**Files to modify:**
- `src/phase1/stages/mtf/generator.py`
- `src/phase1/stages/features/run.py`

**Changes:**
```python
# src/phase1/stages/features/run.py
def run_features_stage(
    df: pd.DataFrame,
    config: PipelineConfig,
    # ...
) -> pd.DataFrame:
    # ... compute base features on training_timeframe data ...

    # MTF features: only from timeframes HIGHER than training_timeframe
    mtf_timeframes = get_higher_timeframes(
        base_tf=config.training_timeframe,
        all_tfs=['5min', '10min', '15min', '20min', '25min', '30min', '45min', '1h']
    )

    mtf_generator = MTFFeatureGenerator(
        base_timeframe=config.training_timeframe,  # Changed from '5min'
        mtf_timeframes=mtf_timeframes,
        mode=config.mtf_mode
    )
    # ...
```

**Success criteria:**
- [ ] MTF base_timeframe uses `config.training_timeframe`
- [ ] Only generate MTF features from HIGHER timeframes
- [ ] Tests pass: `pytest tests/phase_1_tests/stages/test_mtf_features.py::test_base_timeframe_config`
- [ ] Smoke test: Train on 15min with MTF from 30min and 1h

---

### Task 1.5: Update Purge/Embargo Auto-Scaling for Timeframe
**Duration:** 1 day (6 hours)

**Files to modify:**
- `src/phase1/config/features.py`

**Changes:**
```python
# src/phase1/config/features.py
def auto_scale_purge_embargo(
    horizons: list,
    timeframe: str = '5min',  # NEW parameter
    purge_multiplier: float | None = None,
    embargo_multiplier: float | None = None,
) -> tuple:
    """
    Auto-scale purge and embargo bars based on horizons AND timeframe.

    Example:
    - 5min timeframe: 1440 bars = 5 days
    - 15min timeframe: 480 bars = 5 days (same time, fewer bars)
    """
    max_horizon = max(horizons)
    purge_bars = int(max_horizon * (purge_multiplier or 3))

    # Embargo: 5 trading days scaled by timeframe
    tf_minutes = parse_timeframe_to_minutes(timeframe)
    minutes_per_day = 6.5 * 60  # 6.5 hour trading day
    bars_per_day = minutes_per_day / tf_minutes
    embargo_days = 5
    embargo_bars = int(bars_per_day * embargo_days)

    return purge_bars, embargo_bars
```

**Success criteria:**
- [ ] Embargo scales with timeframe (5min: 1440 bars, 15min: 480 bars, both = 5 days)
- [ ] Purge scales with max_horizon (same as before)
- [ ] Tests pass: `pytest tests/phase_1_tests/test_purge_embargo_scaling.py`
- [ ] Verify: 5min embargo ≈ 15min embargo in real time

---

## Phase 2: Strategy 1 - Single-Timeframe Training (1 week)

**Goal:** Support training on a single timeframe without MTF features

### Task 2.1: Add mtf_strategy Config Parameter
**Duration:** 1 day (6 hours)

**Files to modify:**
- `src/phase1/pipeline_config.py`

**Changes:**
```python
# src/phase1/pipeline_config.py
@dataclass
class PipelineConfig:
    # ... existing ...

    # MTF Strategy configuration
    mtf_strategy: str = 'single_tf'  # Options: 'single_tf', 'mtf_indicators', 'mtf_ingestion'
    mtf_source_timeframes: list[str] | None = None  # For Strategy 2
    mtf_input_timeframes: list[str] | None = None   # For Strategy 3

    def __post_init__(self):
        # ... existing validation ...

        # Validate mtf_strategy
        valid_strategies = ['single_tf', 'mtf_indicators', 'mtf_ingestion']
        if self.mtf_strategy not in valid_strategies:
            raise ValueError(
                f"mtf_strategy must be one of {valid_strategies}, got '{self.mtf_strategy}'"
            )
```

**Success criteria:**
- [ ] `mtf_strategy` field added with validation
- [ ] Default strategy is 'single_tf'
- [ ] Auto-population of mtf_source_timeframes for Strategy 2
- [ ] Tests pass: `pytest tests/phase_1_tests/test_pipeline_config.py::test_mtf_strategy_validation`

---

### Task 2.2: Implement Strategy 1 in Feature Stage
**Duration:** 2 days (12 hours)

**Files to modify:**
- `src/phase1/stages/features/run.py`
- `src/phase1/stages/features/engineer.py`

**Changes:**
```python
# src/phase1/stages/features/run.py
def run_features_stage(
    df: pd.DataFrame,
    config: PipelineConfig,
    # ...
) -> pd.DataFrame:
    # Compute base features on training_timeframe
    df = compute_base_features(df, config)

    # Conditional MTF generation based on strategy
    if config.mtf_strategy == 'single_tf':
        logger.info("Strategy 1 (single-timeframe): Skipping MTF features")
        # No MTF features added

    elif config.mtf_strategy == 'mtf_indicators':
        logger.info("Strategy 2 (MTF indicators): Generating MTF features")
        df = add_mtf_features(df, config)

    elif config.mtf_strategy == 'mtf_ingestion':
        logger.info("Strategy 3 (MTF ingestion): Preserving multi-resolution data")
        # Don't add MTF features here - handled in datasets stage
        pass

    return df
```

**Success criteria:**
- [ ] Strategy 1 skips MTF feature generation
- [ ] Feature count for Strategy 1 is ~40 (only base timeframe features)
- [ ] Tests pass: `pytest tests/phase_1_tests/stages/test_features.py::test_strategy_1_single_tf`
- [ ] Smoke test: Run pipeline with mtf_strategy='single_tf', verify no MTF columns

---

### Task 2.3: Update Model Trainers for Strategy 1
**Duration:** 1 day (6 hours)

**Files to modify:**
- `scripts/train_model.py`
- `src/models/trainer.py`

**Changes:**
```python
# scripts/train_model.py
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--training-timeframe', type=str, default='5min')
    parser.add_argument('--mtf-strategy', type=str, default='single_tf',
                        choices=['single_tf', 'mtf_indicators', 'mtf_ingestion'])
    # ...
```

**Success criteria:**
- [ ] CLI supports --training-timeframe and --mtf-strategy flags
- [ ] Model training works with Strategy 1 datasets
- [ ] Tests pass: `pytest tests/phase_2_tests/test_training_strategy_1.py`
- [ ] Smoke test: Train XGBoost on 15min single-timeframe data

---

### Task 2.4: Create Strategy 1 Integration Tests
**Duration:** 1 day (8 hours)

**Files to create:**
- `tests/integration/test_mtf_strategy_1.py`

**Test cases:**
1. Pipeline run with training_timeframe=1min, mtf_strategy='single_tf'
2. Pipeline run with training_timeframe=15min, mtf_strategy='single_tf'
3. Pipeline run with training_timeframe=1h, mtf_strategy='single_tf'
4. Verify feature counts match expectations (~40 features per timeframe)
5. Train XGBoost on each timeframe, verify predictions
6. Compare performance: 1min vs 5min vs 15min vs 1h

**Success criteria:**
- [ ] All 6 test cases pass
- [ ] Integration tests run in < 5 minutes
- [ ] CI pipeline includes Strategy 1 tests

---

## Phase 3: Strategy 2 - MTF Indicators Enhancement (1 week)

**Goal:** Enhance existing MTF indicators support with configurable source timeframes

### Task 3.1: Add mtf_source_timeframes Configuration
**Duration:** 1 day (6 hours)

**Files to modify:**
- `src/phase1/stages/mtf/generator.py`
- `src/phase1/stages/features/run.py`

**Changes:**
```python
# src/phase1/stages/features/run.py
if config.mtf_strategy == 'mtf_indicators':
    # Use explicit source timeframes if provided
    if config.mtf_source_timeframes:
        mtf_timeframes = config.mtf_source_timeframes
    else:
        # Default: all higher timeframes
        mtf_timeframes = get_higher_timeframes(config.training_timeframe)

    mtf_generator = MTFFeatureGenerator(
        base_timeframe=config.training_timeframe,
        mtf_timeframes=mtf_timeframes,
        mode='indicators'
    )
```

**Success criteria:**
- [ ] Config supports explicit mtf_source_timeframes list
- [ ] Auto-populates with higher timeframes if not provided
- [ ] Validation: all source TFs must be >= training_timeframe
- [ ] Tests pass: `pytest tests/phase_1_tests/stages/test_mtf_source_timeframes.py`

---

### Task 3.2: Add Model-Specific MTF Recommendations
**Duration:** 2 days (12 hours)

**Files to create:**
- `src/models/mtf_config.py`

**Content:**
```python
# src/models/mtf_config.py
MTF_RECOMMENDATIONS = {
    'xgboost': {
        'strategy': 'mtf_indicators',
        'training_timeframe': '15min',
        'source_timeframes': ['1min', '5min', '30min', '1h'],
        'rationale': 'Tabular models benefit from dense cross-TF features'
    },
    'lstm': {
        'strategy': 'single_tf',
        'training_timeframe': '15min',
        'source_timeframes': None,
        'rationale': 'RNNs work best with single-TF + wavelets'
    },
    # ... all 19 models ...
}
```

**Success criteria:**
- [ ] Recommendations defined for all 19 models
- [ ] CLI supports --use-recommended-mtf flag
- [ ] Tests pass: `pytest tests/phase_2_tests/test_mtf_recommendations.py`

---

### Task 3.3: Feature Count Validation by Strategy
**Duration:** 1 day (6 hours)

**Files to modify:**
- `src/phase1/stages/validation/integrity.py`

**Changes:**
```python
# src/phase1/stages/validation/integrity.py
EXPECTED_FEATURE_COUNTS = {
    'single_tf': {
        '1min': 50,
        '5min': 40,
        '15min': 40,
        '1h': 40,
    },
    'mtf_indicators': {
        '5min': (100, 200),
        '15min': (80, 150),
        '1h': (50, 80),
    },
}
```

**Success criteria:**
- [ ] Validation detects unexpected feature counts
- [ ] Warning if Strategy 2 has too few/many MTF features
- [ ] Tests pass: `pytest tests/phase_1_tests/stages/test_feature_count_validation.py`

---

### Task 3.4: Update Documentation for Strategy 2
**Duration:** 1 day (6 hours)

**Files to modify:**
- `docs/phases/PHASE_1.md`
- `ALIGNMENT_PLAN.md`

**Success criteria:**
- [ ] Documentation includes all 3 MTF strategies
- [ ] Examples for each model family
- [ ] Table of recommended configs
- [ ] Updated ALIGNMENT_PLAN.md with implementation status

---

## Phase 4: Strategy 3 - Multi-Resolution Ingestion (2 weeks)

**Goal:** Support feeding multiple timeframe tensors to models (for transformers, CNNs)

### Task 4.1: Create Multi-Resolution Dataset Builder
**Duration:** 3 days (20 hours)

**Files to create:**
- `src/phase1/stages/datasets/multi_resolution.py`

**Key class:**
```python
class MultiResolutionDatasetBuilder:
    """
    Build synchronized multi-resolution tensors for MTF ingestion (Strategy 3).

    For each training timeframe sample, create synchronized lookback windows
    from multiple timeframes.
    """

    def build_multi_resolution_tensors(
        self,
        df_1min: pd.DataFrame,
        labels_df: pd.DataFrame,
    ) -> dict[str, np.ndarray]:
        """
        Returns:
            {
                '1min': (n_samples, seq_len_1min, 5),
                '5min': (n_samples, seq_len_5min, 5),
                '15min': (n_samples, seq_len_15min, 5),
                '1h': (n_samples, seq_len_1h, 5),
            }
        """
        # Implementation...
```

**Success criteria:**
- [ ] Builds synchronized multi-resolution tensors
- [ ] Handles variable sequence lengths per timeframe
- [ ] Handles insufficient history (padding)
- [ ] Tests pass: `pytest tests/phase_1_tests/stages/test_multi_resolution_builder.py`

---

### Task 4.2: Add Multi-Resolution Support to TimeSeriesDataContainer
**Duration:** 2 days (14 hours)

**Files to modify:**
- `src/phase1/stages/datasets/container.py`

**Changes:**
```python
# src/phase1/stages/datasets/container.py
class TimeSeriesDataContainer:
    def get_multi_resolution_tensors(
        self,
        split: str,
        input_timeframes: list[str],
        lookback_minutes: int = 60,
    ) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
        """Get multi-resolution tensors for a split."""
        # Implementation...
```

**Success criteria:**
- [ ] Container supports multi-resolution tensor retrieval
- [ ] Returns dict of tensors (one per input timeframe)
- [ ] Tests pass: `pytest tests/phase_1_tests/stages/test_container_multi_resolution.py`

---

### Task 4.3: Add Concatenation and Stacking Utilities
**Duration:** 1 day (6 hours)

**Files to create:**
- `src/phase1/stages/datasets/tensor_utils.py`

**Functions:**
```python
def concatenate_multi_resolution(
    tensors: dict[str, np.ndarray],
    timeframe_order: list[str]
) -> np.ndarray:
    """Concatenate along sequence dimension. Returns (n, total_seq_len, 5)"""

def stack_multi_resolution(
    tensors: dict[str, np.ndarray],
    timeframe_order: list[str],
    pad_to_max_len: bool = True
) -> np.ndarray:
    """Stack into 4D tensor. Returns (n, n_timeframes, seq_len, 5)"""
```

**Success criteria:**
- [ ] Concatenation works for variable seq_len tensors
- [ ] Stacking supports padding to max_len
- [ ] Tests pass: `pytest tests/phase_1_tests/stages/test_tensor_utils.py`

---

### Task 4.4: Update Model Trainers for Strategy 3
**Duration:** 3 days (20 hours)

**Files to modify:**
- `src/models/neural/base_rnn.py`
- `scripts/train_model.py`

**Changes:**
```python
# scripts/train_model.py
if args.mtf_strategy == 'mtf_ingestion':
    X_multi_train, y_train, weights_train = container.get_multi_resolution_tensors(
        split='train',
        input_timeframes=args.mtf_input_timeframes
    )

    # Concatenate or stack based on model
    if args.model in ['lstm', 'tcn']:
        X_train = concatenate_multi_resolution(X_multi_train, args.mtf_input_timeframes)
    elif args.model in ['patchtst', 'itransformer']:
        X_train = stack_multi_resolution(X_multi_train, args.mtf_input_timeframes)
```

**Success criteria:**
- [ ] LSTM/TCN support concatenated multi-resolution inputs
- [ ] Transformer models support stacked multi-resolution inputs
- [ ] Tests pass: `pytest tests/phase_2_tests/test_training_strategy_3.py`

---

### Task 4.5: Create Strategy 3 Integration Tests
**Duration:** 2 days (14 hours)

**Files to create:**
- `tests/integration/test_mtf_strategy_3.py`

**Test matrix:**
- Build multi-resolution dataset for 15min training with [1min, 5min, 15min]
- Build multi-resolution dataset for 15min training with [1min, 5min, 15min, 1h]
- Train LSTM with concatenated multi-resolution inputs
- Train TCN with concatenated multi-resolution inputs
- Verify prediction shapes
- Compare performance: single-TF vs multi-resolution

**Success criteria:**
- [ ] All 6 test cases pass
- [ ] Integration tests run in < 10 minutes
- [ ] CI includes Strategy 3 tests

---

## Phase 5: Model Integration and CLI Updates (1 week)

**Goal:** Update all model trainers and CLI to support all 3 MTF strategies

### Task 5.1: Add MTF Strategy Support to All Model Families
**Duration:** 3 days (20 hours)

**Files to modify:**
- All 19 model files

**Changes:**
```python
# Each model's fit() method validates input shape
def fit(self, X_train, y_train, ...):
    if self.family == 'boosting':
        assert X_train.ndim == 2, f"Boosting models expect 2D input"
    elif self.family == 'neural':
        assert X_train.ndim in [3, 4], f"Neural models expect 3D or 4D input"
```

**Success criteria:**
- [ ] All 19 models validate input shapes
- [ ] Clear error messages for shape mismatches
- [ ] Tests pass: `pytest tests/phase_2_tests/test_model_input_validation.py`

---

### Task 5.2: Update CLI with MTF Strategy Flags
**Duration:** 2 days (12 hours)

**Files to modify:**
- `scripts/train_model.py`
- `scripts/run_cv.py`
- `pipeline` (CLI wrapper)

**New flags:**
```bash
python scripts/train_model.py \
    --model xgboost \
    --horizon 20 \
    --training-timeframe 15min \
    --mtf-strategy mtf_indicators \
    --mtf-source-timeframes 1min 5min 30min 1h \
    --use-recommended-mtf
```

**Success criteria:**
- [ ] CLI supports all MTF flags
- [ ] --use-recommended-mtf applies model-specific config
- [ ] --list-mtf-strategies shows all 3 strategies
- [ ] Tests pass: `pytest tests/cli/test_mtf_cli_flags.py`

---

### Task 5.3: Add MTF Strategy to Config YAML
**Duration:** 1 day (6 hours)

**Files to create:**
- `config/mtf_strategy_1.yaml`
- `config/mtf_strategy_2.yaml`
- `config/mtf_strategy_3.yaml`

**Example:**
```yaml
# config/mtf_strategy_2.yaml
training_timeframe: 15min
mtf_strategy: mtf_indicators
mtf_source_timeframes: [1min, 5min, 30min, 1h]
model: xgboost
horizon: 20
```

**Success criteria:**
- [ ] 3 example configs created
- [ ] Configs pass validation
- [ ] Smoke test: Run pipeline with each config

---

### Task 5.4: Update Model Registry with MTF Metadata
**Duration:** 1 day (6 hours)

**Files to modify:**
- `src/models/registry.py`

**Changes:**
```python
@dataclass
class ModelSpec:
    # ... existing ...
    supports_single_tf: bool = True
    supports_mtf_indicators: bool = True
    supports_mtf_ingestion: bool = False
    recommended_mtf_strategy: str = 'single_tf'
    recommended_training_tf: str = '15min'
```

**Success criteria:**
- [ ] All 19 models have MTF metadata
- [ ] Registry.list_all() shows MTF capabilities
- [ ] Tests pass: `pytest tests/phase_2_tests/test_model_registry_mtf.py`

---

## Phase 6: Production Deployment and Testing (1 week)

**Goal:** Production-ready MTF pipeline with comprehensive testing

### Task 6.1: Add MTF Support to Inference Pipeline
**Duration:** 3 days (20 hours)

**Files to modify:**
- `src/inference/pipeline.py`
- `scripts/serve_model.py`

**Key method:**
```python
class InferencePipeline:
    def predict(self, df_1min: pd.DataFrame) -> np.ndarray:
        # 1. Resample 1min → training_timeframe
        # 2. Apply MTF strategy (1/2/3)
        # 3. Scale features
        # 4. Model prediction
```

**Success criteria:**
- [ ] Inference supports all 3 MTF strategies
- [ ] Deterministic: same input → same output as training
- [ ] Tests pass: `pytest tests/inference/test_mtf_inference.py`

---

### Task 6.2: Add MTF Config Validation
**Duration:** 1 day (8 hours)

**Files to create:**
- `src/phase1/config/mtf_validation.py`

**Validations:**
1. mtf_strategy is valid
2. training_timeframe in supported list
3. Strategy 2: source TFs >= training TF
4. Strategy 3: input TFs specified
5. Model supports chosen strategy

**Success criteria:**
- [ ] Validation catches all invalid configs
- [ ] Clear error messages
- [ ] 20+ test cases

---

### Task 6.3: Create Comprehensive MTF Integration Tests
**Duration:** 2 days (14 hours)

**Files to create:**
- `tests/integration/test_mtf_end_to_end.py`

**Test matrix:** All 3 strategies × all model families

**Success criteria:**
- [ ] 6+ integration tests pass
- [ ] Tests run in < 15 minutes
- [ ] CI includes full MTF test suite

---

### Task 6.4: Performance Benchmarking
**Duration:** 2 days (12 hours)

**Files to create:**
- `scripts/benchmark_mtf_strategies.py`

**Metrics:**
- Pipeline runtime
- Feature count
- Training time
- Inference latency
- Sharpe ratio
- Win rate
- Max drawdown

**Success criteria:**
- [ ] Benchmark script runs
- [ ] Results exported to CSV/JSON
- [ ] Visualization charts
- [ ] Results in `docs/MTF_BENCHMARKS.md`

---

### Task 6.5: Update All Documentation
**Duration:** 1 day (8 hours)

**Files to modify:**
- `CLAUDE.md`
- `ALIGNMENT_PLAN.md`
- `docs/phases/PHASE_1.md`
- `docs/QUICK_REFERENCE.md`

**Files to create:**
- `docs/MTF_GUIDE.md`

**Success criteria:**
- [ ] All 5 doc files updated
- [ ] MTF_GUIDE.md created
- [ ] Migration guide included
- [ ] Troubleshooting section

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
Phase 6: Production Deployment (1 week)
```

**Parallelization:**
- Phases 2, 3, 4 can be partially parallelized
- With 2 engineers: 4-5 weeks total

---

## Critical Files for Implementation

### Top 5 Most Critical Files

1. **`src/phase1/pipeline_config.py`**
   Core configuration changes (training_timeframe, mtf_strategy, all configs)

2. **`src/phase1/stages/mtf/generator.py`**
   MTF feature generation (add 20m/25m, configurable base_timeframe)

3. **`src/phase1/stages/datasets/multi_resolution.py`** (NEW)
   Strategy 3: multi-resolution tensor builder

4. **`src/phase1/stages/features/run.py`**
   Feature orchestration (conditional MTF by strategy)

5. **`scripts/train_model.py`**
   Model training script (CLI flags, data loading by strategy)

### Additional Important Files

- `src/models/registry.py` - MTF capability metadata
- `src/inference/pipeline.py` - Inference with MTF support
- `tests/integration/test_mtf_end_to_end.py` - Integration tests
- `src/phase1/stages/datasets/container.py` - Multi-resolution container
- `src/phase1/stages/datasets/tensor_utils.py` - Concatenation/stacking

---

## Quick Start Commands

### Strategy 1: Single-Timeframe
```bash
python scripts/train_model.py \
    --model xgboost \
    --training-timeframe 15min \
    --mtf-strategy single_tf \
    --horizon 20
```

### Strategy 2: MTF Indicators
```bash
python scripts/train_model.py \
    --model xgboost \
    --training-timeframe 15min \
    --mtf-strategy mtf_indicators \
    --mtf-source-timeframes 1min 5min 30min 1h \
    --horizon 20
```

### Strategy 3: MTF Ingestion
```bash
python scripts/train_model.py \
    --model lstm \
    --training-timeframe 15min \
    --mtf-strategy mtf_ingestion \
    --mtf-input-timeframes 1min 5min 15min \
    --horizon 20
```

---

## Success Metrics

| Phase | Metric | Target |
|-------|--------|--------|
| Phase 1 | 9-timeframe ladder | All 9 TFs supported |
| Phase 2 | Strategy 1 working | All models train on single-TF |
| Phase 3 | Strategy 2 enhanced | Configurable source TFs |
| Phase 4 | Strategy 3 working | Multi-res tensors for sequence models |
| Phase 5 | Model integration | All 19 models support all strategies |
| Phase 6 | Production ready | Full test coverage, docs complete |

**Final deliverable:** Production-grade MTF pipeline supporting 3 strategies, 9 timeframes, 19 models, fully tested and documented.
