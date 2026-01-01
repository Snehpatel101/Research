# ML Factory Implementation Roadmap

## Executive Summary

**Total Work Remaining:** 36-40 engineer-days (~7-8 weeks single engineer, ~3 weeks with 3 engineers)
**Critical Blocker:** HMM Lookahead Bug (0.5 days) - MUST FIX FIRST

---

## Current State vs Target State

### What's IMPLEMENTED (Working)
- ✅ 13 base models (Boosting, Neural, Classical, Ensemble)
- ✅ Tabular 2D adapter
- ✅ Sequence 3D adapter
- ✅ OOF generation (supports heterogeneous)
- ✅ PurgedKFold cross-validation
- ✅ 5 MTF timeframes (15m, 30m, 1h, 4h, daily)
- ✅ ~180 features (indicators, wavelets, microstructure)
- ✅ Triple-barrier labeling with transaction costs
- ✅ Purge/embargo leakage prevention

### What's MISSING (Gaps)
- ❌ HMM lookahead bug (CRITICAL)
- ❌ 4 MTF timeframes (20min, 25min missing; 1min, 5min, 10min, 45min not default)
- ❌ Per-model feature selection at runtime
- ❌ Multi-resolution 4D adapter
- ❌ Heterogeneous stacking (same-family constraint exists)
- ❌ 4 meta-learners (Ridge, MLP, Calibrated Blender)
- ❌ 6 advanced models (CNN, Transformers, N-BEATS)

---

## Prioritized Implementation Order

| Priority | Task | Effort | Sprint |
|----------|------|--------|--------|
| **P0** | HMM Lookahead Bug Fix | 0.5 days | Sprint 1 |
| **P1** | MTF 9-Timeframe Ladder | 0.5 days | Sprint 1 |
| **P1** | N-BEATS Model | 1 day | Sprint 1 |
| **P1** | Meta-Learners (4) | 3 days | Sprint 1 |
| **P1** | Heterogeneous Stacking | 1.5 days | Sprint 2 |
| **P1** | Per-Model Feature Selection | 2-3 days | Sprint 1-2 |
| **P2** | CNN Models (2) | 5 days | Sprint 2 |
| **P2** | Multi-Resolution 4D Adapter | 5-7 days | Sprint 2 |
| **P3** | Advanced Transformers (3) | 12 days | Sprint 3-4 |

---

## Sprint Plan (3 Engineers)

### Sprint 1 (Week 1) - Foundation
| Engineer A | Engineer B | Engineer C |
|------------|------------|------------|
| P0: HMM Fix | P1: MTF Ladder | P1: N-BEATS |
| P1: Meta-Learners | P1: Per-Model Features | P2: CNN Models (start) |

**Sprint 1 Deliverables:**
- Production-safe HMM regime features
- 9-timeframe MTF ladder
- N-BEATS model working
- 4 meta-learners implemented
- Per-model feature selection integrated

### Sprint 2 (Week 2) - Core Infrastructure
| Engineer A | Engineer B | Engineer C |
|------------|------------|------------|
| P1: Heterogeneous Stacking | P2: 4D Adapter | P2: CNN Models (finish) |
| Support/Review | Continue 4D | Testing |

**Sprint 2 Deliverables:**
- Heterogeneous ensemble stacking working
- Multi-resolution 4D adapter complete
- InceptionTime + 1D ResNet working

### Sprint 3-4 (Weeks 3-4) - Advanced Models
| Engineer A | Engineer B | Engineer C |
|------------|------------|------------|
| P3: PatchTST | P3: iTransformer | P3: TFT |

**Sprint 3-4 Deliverables:**
- All 3 advanced transformers working
- Full model suite (23 models) complete

---

## Detailed Implementation Plans by Phase

### PHASE 0: Critical Bug Fix (BLOCKER)

**Task:** Fix HMM Lookahead Bias
**File:** `src/phase1/stages/regime/hmm.py`
**Effort:** 0.5 days

**Code Changes:**
```python
# Line 358 - Change:
window_obs = observations[:i + 1]
# To:
window_obs = observations[:i]

# Line 390 - Change:
predict_window = observations[max(0, i - self.config.lookback):i + 1]
# To:
predict_window = observations[max(0, i - self.config.lookback):i]

# Line 416 - Change:
window = observations[i - self.config.lookback:i + 1]
# To:
window = observations[i - self.config.lookback:i]
```

**Validation:**
```bash
pytest tests/phase_1_tests/stages/test_regime.py -v
pytest tests/phase_1_tests/stages/test_lookahead_invariance.py -v
```

---

### PHASE 1: MTF 9-Timeframe Ladder

**Task:** Add missing timeframes (20min, 25min) and update defaults
**Files:** 
- `src/phase1/stages/mtf/constants.py`
- `src/phase1/config/features.py`
**Effort:** 0.5 days

**Code Changes:**

File: `src/phase1/stages/mtf/constants.py`
```python
# Add to MTF_TIMEFRAMES dict (after line 26):
'20min': 20,
'25min': 25,

# Update DEFAULT_MTF_TIMEFRAMES (line 46):
DEFAULT_MTF_TIMEFRAMES = ['1min', '5min', '10min', '15min', '20min', '25min', '30min', '45min', '1h']

# Add to PANDAS_FREQ_MAP (after line 65):
'20min': '20min',
'25min': '25min',
```

---

### PHASE 2: Per-Model Feature Selection

**Task:** Integrate feature set configs at runtime
**Files:**
- `src/phase1/stages/datasets/container.py`
- `src/models/data_preparation.py`
- `src/phase1/utils/feature_sets.py`
**Effort:** 2-3 days

**Implementation Steps:**
1. Add `feature_set` parameter to `TimeSeriesDataContainer.from_parquet_dir()`
2. Add `_resolve_feature_set()` method to filter columns
3. Update `ModelDataPreparer` to pass model-specific feature set
4. Create mapping: model_family → feature_set_name

**Feature Set Mapping:**
```python
FAMILY_FEATURE_SETS = {
    'boosting': 'boosting_optimal',  # ~200 features with MTF
    'neural': 'neural_optimal',       # ~150 features, no MTF
    'classical': 'boosting_optimal',  # Same as boosting
    'cnn': 'neural_optimal',          # Same as neural
    'transformer': 'transformer_raw', # Minimal features
}
```

---

### PHASE 3: Meta-Learners

**Task:** Implement 4 meta-learner models
**Files to Create:**
- `src/models/inference/ridge_meta.py`
- `src/models/inference/mlp_meta.py`
- `src/models/inference/calibrated_blender.py`
- `src/models/inference/__init__.py`
**Effort:** 3 days

**Ridge Meta-Learner:**
```python
@register(name="meta_ridge", family="inference")
class RidgeMetaLearner(BaseModel):
    def __init__(self, config: dict = None):
        self.model = RidgeClassifier(alpha=1.0)
    
    def fit(self, X_oof, y, **kwargs):
        self.model.fit(X_oof, y)
        return TrainingMetrics(...)
    
    def predict(self, X_oof):
        probs = self.model.decision_function(X_oof)
        return PredictionOutput(probabilities=probs)
```

**MLP Meta-Learner:**
```python
@register(name="meta_mlp", family="inference")
class MLPMetaLearner(BaseModel):
    def __init__(self, config: dict = None):
        self.model = MLPClassifier(
            hidden_layer_sizes=(64, 32),
            early_stopping=True
        )
```

**Calibrated Blender:**
```python
@register(name="meta_calibrated", family="inference")
class CalibratedBlender(BaseModel):
    def __init__(self, config: dict = None):
        self.calibrator = IsotonicRegression()
        self.weights = None  # Learned during fit
```

---

### PHASE 4: Heterogeneous Stacking

**Task:** Remove same-family constraint, add representation adapter
**Files:**
- `src/models/ensemble/validator.py`
- `src/models/ensemble/stacking.py`
- `src/models/ensemble/representation_adapter.py` (NEW)
**Effort:** 1.5 days

**Validator Change:**
```python
def validate_stacking_config(base_model_names: list[str]) -> tuple[bool, str]:
    # For stacking: Allow mixed families because OOF predictions are always 2D
    # Only validate that base models exist and are registered
    for name in base_model_names:
        if not ModelRegistry.is_registered(name):
            return False, f"Model {name} not registered"
    return True, ""
```

**Representation Adapter:**
```python
class RepresentationAdapter:
    def prepare_for_model(self, X: np.ndarray, model_name: str) -> np.ndarray:
        model_info = ModelRegistry.get_model_info(model_name)
        if model_info['requires_sequences']:
            return self._to_sequences(X, seq_len=60)
        return X
```

---

### PHASE 5: Multi-Resolution 4D Adapter

**Task:** Create 4D tensor builder for advanced transformers
**Files to Create:**
- `src/phase1/stages/datasets/multi_resolution.py`
**Effort:** 5-7 days

**MultiResolutionDataset:**
```python
class MultiResolutionDataset(Dataset):
    """
    Produces 4D tensors: (batch, timeframes, seq_len, channels)
    
    Example: (1000, 5, 60, 4)
    - 1000 samples
    - 5 timeframes (5min, 15min, 30min, 1h, 4h)
    - 60 bars per timeframe
    - 4 channels (O, H, L, C)
    """
    
    def __init__(
        self,
        df: pd.DataFrame,
        timeframes: list[str],
        seq_len: int,
        channels: list[str] = ['open', 'high', 'low', 'close'],
    ):
        self.timeframes = timeframes
        self.seq_len = seq_len
        self.channels = channels
        self._build_multi_res_data(df)
    
    def _build_multi_res_data(self, df):
        # Resample df to each timeframe
        # Create aligned sequences
        # Stack into 4D tensor
        pass
```

---

### PHASE 6: CNN Models

**Task:** Implement InceptionTime and 1D ResNet
**Files to Create:**
- `src/models/cnn/__init__.py`
- `src/models/cnn/inceptiontime.py`
- `src/models/cnn/resnet1d.py`
**Effort:** 5 days

**InceptionTime Architecture:**
- Multi-scale inception blocks (kernels: 10, 20, 40)
- Residual connections
- Global average pooling
- Classification head

**1D ResNet Architecture:**
- Residual blocks with 1D convolutions
- Batch normalization
- Skip connections
- Classification head

---

### PHASE 7: Advanced Transformers

**Task:** Implement PatchTST, iTransformer, TFT
**Files to Create:**
- `src/models/transformers/__init__.py`
- `src/models/transformers/patchtst.py`
- `src/models/transformers/itransformer.py`
- `src/models/transformers/tft.py`
**Effort:** 12 days (4 days each)

**PatchTST:**
- Patch embedding for time series
- Transformer encoder
- Classification head

**iTransformer:**
- Feature tokens (each feature as token)
- Cross-feature attention
- Temporal aggregation

**TFT:**
- Variable selection networks
- Gated residual networks
- Multi-head attention
- Quantile outputs

---

## Testing Strategy

Each implementation phase includes tests:

1. **Unit Tests:** Per-function correctness
2. **Integration Tests:** End-to-end pipeline
3. **Regression Tests:** Ensure no performance degradation
4. **Lookahead Tests:** Verify no future data leakage

**Test Commands:**
```bash
# Run all tests
pytest tests/ -v

# Run specific phase tests
pytest tests/phase_1_tests/ -v
pytest tests/models/ -v

# Run lookahead validation
pytest tests/phase_1_tests/stages/test_lookahead_invariance.py -v
```

---

## Success Criteria

### Phase Completion Checklist

- [ ] HMM bug fixed, tests passing
- [ ] 9-timeframe MTF ladder complete
- [ ] Per-model feature selection working
- [ ] 4 meta-learners registered
- [ ] Heterogeneous stacking working
- [ ] 4D adapter complete
- [ ] CNN models working
- [ ] Advanced transformers working
- [ ] All 23 models registered
- [ ] Full test suite passing

### Performance Baselines

After all implementations:
- Run CV on all models
- Establish Sharpe ratio baselines
- Compare heterogeneous vs homogeneous ensembles
- Document model performance rankings

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| HMM fix breaks regime features | Run full pipeline after fix, compare distributions |
| 4D adapter memory issues | Implement lazy loading, batch processing |
| Transformer models overfit | Early stopping, dropout, weight decay |
| Feature selection reduces performance | A/B test with/without |

---

Last Updated: 2026-01-01
