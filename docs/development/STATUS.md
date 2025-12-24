# TODO Before Phase 2 - Detailed Checklist

## Overview

This is the **complete checklist** of work needed to finish Phase 1 before moving to Phase 2 (ML Model Factory).

**Estimated Time:** 1.5-2 days
**Priority:** All items are required before Phase 2

---

## Critical Context

### What We're Fixing
Phase 1 (Sneh's work) is **8.5/10 complete** as a modular data preparation pipeline. However:
- Cross-asset features cause validation failures when running single-symbol
- Minor bugs and edge cases need fixing
- Documentation needs updates for Phase 2 integration

### What We're NOT Building
- ❌ PyTorch/TensorFlow integration (Phase 2's job)
- ❌ Model training code (Phase 2's job)
- ❌ Ensemble framework (Phase 2's job)

Phase 1 = Data prep → Parquet files
Phase 2 = Load Parquets → Train models

---

## PART 1: Cross-Asset Feature Removal (PRIORITY 1)

### Issue
- Cross-asset features (e.g., `mes_mgc_correlation_20`, `relative_strength`) require BOTH MES+MGC
- When running single symbol → 4 features = 100% NaN → validation fails
- We trade **one market at a time**, so cross-asset features are unnecessary

### Files to Modify

#### 1.1. Disable Cross-Asset Features in Config
**File:** `src/config/features.py`

**Action:** Add feature flag
```python
# Add at top of file
CROSS_ASSET_ENABLED = False  # Set to False to disable cross-asset features

# In FEATURE_CONFIG dict, add:
FEATURE_CONFIG = {
    # ... existing config ...
    "cross_asset": {
        "enabled": CROSS_ASSET_ENABLED,
        "min_symbols": 2,  # Require at least 2 symbols for cross-asset
    }
}
```

**Estimated Time:** 5 minutes

---

#### 1.2. Update Feature Engineer to Skip Cross-Asset
**File:** `src/stages/features/engineer.py`

**Action:** Add conditional check before cross-asset feature generation

Find the section that generates cross-asset features (around line 200-300) and wrap it:

```python
# Before (generates cross-asset features unconditionally):
if self.symbols and len(self.symbols) > 1:
    df = self._add_cross_asset_features(df)

# After (check config flag):
from src.config.features import CROSS_ASSET_ENABLED

if CROSS_ASSET_ENABLED and self.symbols and len(self.symbols) >= 2:
    df = self._add_cross_asset_features(df)
else:
    logger.info("Skipping cross-asset features (disabled or insufficient symbols)")
```

**Estimated Time:** 10 minutes

---

#### 1.3. Update Cross-Asset Feature Module
**File:** `src/stages/features/cross_asset.py`

**Action:** Add early return if disabled

```python
# At top of main function (e.g., add_cross_asset_features):
from src.config.features import CROSS_ASSET_ENABLED

def add_cross_asset_features(df, symbols):
    """Generate cross-asset correlation features."""

    # Early return if disabled
    if not CROSS_ASSET_ENABLED:
        logger.info("Cross-asset features disabled in config")
        return df

    # Early return if insufficient symbols
    if len(symbols) < 2:
        logger.warning(f"Cross-asset features require >=2 symbols, got {len(symbols)}. Skipping.")
        return df

    # ... existing implementation ...
```

**Estimated Time:** 10 minutes

---

#### 1.4. Update Validation to Allow Missing Cross-Asset Features
**File:** `src/stages/stage8_validate.py`

**Action:** Don't fail validation if cross-asset features are missing (expected behavior)

Find the feature validation section and add:

```python
from src.config.features import CROSS_ASSET_ENABLED

# When checking for missing/NaN features:
CROSS_ASSET_FEATURES = [
    'mes_mgc_correlation_20',
    'mes_mgc_beta',
    'mes_mgc_spread_zscore',
    'relative_strength'
]

# Filter out cross-asset features from required checks if disabled
if not CROSS_ASSET_ENABLED:
    required_features = [f for f in required_features if f not in CROSS_ASSET_FEATURES]
```

**Estimated Time:** 15 minutes

---

#### 1.5. Update Tests
**Files:** `tests/test_cross_asset_features.py` (if exists)

**Action:** Skip cross-asset tests if disabled

```python
import pytest
from src.config.features import CROSS_ASSET_ENABLED

@pytest.mark.skipif(not CROSS_ASSET_ENABLED, reason="Cross-asset features disabled")
def test_cross_asset_correlation():
    # ... existing test ...
```

**Estimated Time:** 10 minutes

---

### 1.6. Test Cross-Asset Removal

**Action:** Run pipeline with single symbol and verify no failures

```bash
# Clean previous runs
rm -rf runs/* data/splits/* data/clean/* data/features/* data/final/*

# Run with single symbol (MES only)
./pipeline run --symbols MES --stages all

# Check for validation failures
cat runs/*/logs/pipeline.log | grep -i "error\|critical\|failed"

# Verify no NaN features
cat runs/*/artifacts/validation_report.json | grep -A 5 "critical_issues"

# Expected: 0 critical issues
```

**Estimated Time:** 30 minutes (includes pipeline runtime)

---

**Total Time for Part 1:** 1.5-2 hours

---

## PART 2: Bug Fixes & Edge Cases (PRIORITY 2)

### 2.1. Fix Import Paths (Already Fixed by Debugger Agent)
**Status:** ✅ DONE (debugger agent fixed 9 files)

**Verification:**
```bash
# Check no remaining import issues
grep -r "from stages\." src/ | grep -v "from src.stages"
# Should return nothing
```

---

### 2.2. Handle Empty/Missing Data Gracefully
**File:** `src/stages/ingest/__init__.py`

**Issue:** Pipeline crashes if raw data file doesn't exist

**Action:** Add better error messages

```python
def ingest_file(self, file_path, symbol):
    """Ingest and validate a single OHLCV file."""

    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"Raw data file not found: {file_path}\n"
            f"Expected location: data/raw/{symbol}_1m.parquet or .csv\n"
            f"Please ensure raw data is available before running pipeline."
        )

    # ... rest of implementation ...
```

**Estimated Time:** 10 minutes

---

### 2.3. Validate Sufficient Data After Each Stage
**File:** `src/pipeline/runner.py`

**Issue:** Pipeline should fail fast if insufficient data at any stage

**Action:** Add row count validation after each stage

```python
def _validate_stage_output(self, stage_name, output_path):
    """Validate stage output has sufficient data."""

    MIN_ROWS = {
        'data_generation': 1000,      # At least 1K raw bars
        'data_cleaning': 500,          # At least 500 after resampling
        'feature_engineering': 200,    # At least 200 after feature calc
        'create_splits': 100,          # At least 100 for training
    }

    if stage_name in MIN_ROWS:
        # Load output and check row count
        df = pd.read_parquet(output_path)
        min_required = MIN_ROWS[stage_name]

        if len(df) < min_required:
            raise ValueError(
                f"Stage {stage_name} output has insufficient data: "
                f"{len(df)} rows (minimum {min_required} required)"
            )
```

**Estimated Time:** 20 minutes

---

### 2.4. Fix High NaN Drop Rate (57.6%)
**File:** `src/stages/features/engineer.py`

**Issue:** 690K bars → 141K after features → 60K after dropping NaN (57.6% loss)

**Investigation needed:**
- Which features cause the most NaN values?
- Is warmup period too long (200+ bars for SMA_200)?
- Can we use forward-fill for some features?

**Action:** Add NaN diagnostics

```python
def _diagnose_nan_sources(self, df):
    """Identify which features cause the most NaN values."""

    nan_counts = df.isnull().sum()
    nan_features = nan_counts[nan_counts > 0].sort_values(ascending=False)

    logger.info(f"Features with NaN values:")
    for feature, count in nan_features.head(10).items():
        pct = 100 * count / len(df)
        logger.info(f"  {feature}: {count} ({pct:.1f}%)")

    return nan_features
```

**Estimated Time:** 30 minutes (investigation + potential fixes)

---

**Total Time for Part 2:** 1-1.5 hours

---

## PART 3: Documentation Updates (PRIORITY 3)

### 3.1. Create Phase 2 Integration Guide
**File:** `docs/PHASE2_INTEGRATION_GUIDE.md` (new file)

**Content:**

```markdown
# Phase 2 Integration Guide

## How to Load Phase 1 Outputs

### File Locations
Phase 1 outputs are in: `data/splits/{run_id}/scaled/`

Files:
- `train_scaled.parquet` - Training data
- `val_scaled.parquet` - Validation data
- `test_scaled.parquet` - Test data
- `feature_scaler.pkl` - Scaler for production inference
- `split_config.json` - Metadata

### Schema
Each Parquet file contains ~120 columns:

**Metadata (not scaled):**
- `datetime`: pd.datetime64[ns]
- `symbol`: str

**OHLCV (not scaled):**
- `open`, `high`, `low`, `close`, `volume`: float64

**Features (scaled, ~80-100 columns):**
- Price features: `close_return`, `log_return`, ...
- Moving averages: `sma_10`, `ema_21`, ...
- Momentum: `rsi_14`, `macd`, ...
- Volatility: `atr_14`, `bb_upper`, ...
- Volume: `obv`, `vwap`, ...
- (See feature list in split_config.json)

**Labels (not scaled):**
- `label_h5`, `label_h10`, `label_h15`, `label_h20`: int8 (-1, 0, 1, -99)
- `quality_h5`, `quality_h10`, ...: float32 (0-1)
- `sample_weight_h5`, `sample_weight_h10`, ...: float32 (0.5-1.5)

### Example: Load in PyTorch

\`\`\`python
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader

# Load data
train = pd.read_parquet('data/splits/20251223_181236/scaled/train_scaled.parquet')

# Filter invalid labels
train = train[train['label_h5'] != -99]

# Separate features and labels
feature_cols = [c for c in train.columns if c not in [
    'datetime', 'symbol', 'open', 'high', 'low', 'close', 'volume',
    'label_h5', 'label_h10', 'quality_h5', 'sample_weight_h5', ...
]]

X = train[feature_cols].values
y = train['label_h5'].values
weights = train['sample_weight_h5'].values

# Create PyTorch dataset
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.long)
weights_tensor = torch.tensor(weights, dtype=torch.float32)

dataset = TensorDataset(X_tensor, y_tensor, weights_tensor)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Training loop
for X_batch, y_batch, weights_batch in loader:
    # ... train model ...
    pass
\`\`\`

### Example: Load in XGBoost

\`\`\`python
import pandas as pd
import xgboost as xgb

# Load data
train = pd.read_parquet('data/splits/.../train_scaled.parquet')
train = train[train['label_h5'] != -99]

# Prepare features
feature_cols = [...]  # Same as PyTorch example
X_train = train[feature_cols]
y_train = train['label_h5']
weights = train['sample_weight_h5']

# Train XGBoost
dtrain = xgb.DMatrix(X_train, label=y_train, weight=weights)
params = {
    'objective': 'multi:softmax',
    'num_class': 3,
    'max_depth': 6,
}
model = xgb.train(params, dtrain, num_boost_round=100)
\`\`\`

### Important Notes

1. **Filter invalid labels:** Always filter `label != -99` before training
2. **Use sample weights:** Phase 1 provides quality-based weights (0.5-1.5x)
3. **Multi-horizon:** Train separate models for H5, H10, H15, H20
4. **Class imbalance:** Consider balanced sampling or focal loss

### Production Inference

\`\`\`python
import pickle
import pandas as pd

# Load scaler from Phase 1
with open('data/splits/.../scaled/feature_scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# New data (unscaled)
new_data = pd.read_parquet('new_bars.parquet')

# Extract features (same columns as training)
X_new = new_data[feature_cols]

# Scale using Phase 1 scaler
X_scaled = scaler.transform(X_new)

# Predict
predictions = model.predict(X_scaled)
\`\`\`
```

**Estimated Time:** 45 minutes

---

### 3.2. Update Main README
**File:** `README.md` (root)

**Action:** Add section linking to Phase 1 completion docs

```markdown
## Phase 1: Data Preparation (COMPLETE)

Phase 1 is a modular OHLCV data preparation pipeline that outputs clean, labeled datasets.

**Status:** 8.5/10 - Nearly complete
**Remaining work:** See `docs/phase1/completion/TODO_BEFORE_PHASE2.md`

**Quick Start:**
```bash
./pipeline run --symbols MES --stages all
```

**Outputs:** `data/splits/{run_id}/scaled/*.parquet`

**Next:** Phase 2 (ML Model Factory) - train/ensemble/evaluate models

**Documentation:** See `docs/phase1/completion/` for completion checklist and audit reports.
```

**Estimated Time:** 10 minutes

---

### 3.3. Document Known Limitations
**File:** `docs/phase1/KNOWN_LIMITATIONS.md` (new file)

**Content:**

```markdown
# Phase 1 Known Limitations

## Current Limitations

### 1. Single Symbol Only
- **Limitation:** Cross-asset features disabled
- **Impact:** Cannot process MES+MGC together
- **Workaround:** Run pipeline separately for each symbol
- **Future:** Could re-enable with multi-symbol support (not needed now)

### 2. Batch Processing Only
- **Limitation:** No real-time/streaming support
- **Impact:** Must reprocess entire dataset for updates
- **Workaround:** Use incremental reprocessing (manual)
- **Future:** Add streaming ingestion in Phase 3

### 3. No Model Training
- **Limitation:** Phase 1 only does data prep, not model training
- **Impact:** Need Phase 2 for actual ML models
- **Workaround:** Load Parquet files in custom training scripts
- **Future:** Phase 2 will provide full model training pipeline

### 4. High NaN Drop Rate (57.6%)
- **Limitation:** Feature engineering drops 57.6% of bars due to NaN
- **Impact:** Reduced dataset size (690K → 60K final samples)
- **Cause:** Long warmup periods (SMA_200 = 200 bars)
- **Workaround:** Use shorter feature periods or forward-fill
- **Investigation:** Needs profiling to identify main NaN sources

### 5. Hardcoded Feature Periods
- **Limitation:** SMA/EMA periods hardcoded in FeatureEngineer
- **Impact:** Must edit code to change feature config
- **Workaround:** Edit `src/stages/features/engineer.py`
- **Future:** Move to `PipelineConfig` for full configurability

## Design Decisions (Not Bugs)

### 1. Framework-Agnostic
- **Decision:** No PyTorch/TensorFlow dependencies in Phase 1
- **Rationale:** Keep data prep separate from model training
- **Benefit:** Any framework can consume Phase 1 outputs

### 2. Parquet Output Format
- **Decision:** Output as Parquet (not TFRecord, PyTorch tensors, etc.)
- **Rationale:** Universal format, efficient, schema-preserving
- **Benefit:** Works with pandas, PyTorch, TensorFlow, XGBoost, etc.

### 3. Train-Only Scaler Fit
- **Decision:** Scaler fitted ONLY on training data
- **Rationale:** Prevent data leakage
- **Benefit:** Proper ML hygiene, prevents overfitting

## Non-Issues (Intentional Behavior)

### 1. Invalid Labels (-99)
- **Behavior:** Last `max_bars` samples have `label = -99`
- **Reason:** Insufficient future data for barrier calculation
- **Action Required:** Filter `label != -99` before training (documented)

### 2. Sample Weights 0.5x-1.5x
- **Behavior:** Sample weights vary from 0.5x to 1.5x (not 0-1)
- **Reason:** Quality-based weighting (high quality = higher weight)
- **Action Required:** Pass weights to model training (documented)

### 3. Neutral Labels Dominate (~40-50%)
- **Behavior:** Neutral (0) labels are ~40-50% of dataset
- **Reason:** Most price movements timeout without hitting barriers
- **Action Required:** Use balanced sampling or focal loss in Phase 2
```

**Estimated Time:** 20 minutes

---

**Total Time for Part 3:** 1.5-2 hours

---

## PART 4: Testing & Validation (PRIORITY 4)

### 4.1. End-to-End Pipeline Test
**Action:** Run full pipeline and verify all outputs

```bash
# Clean slate
rm -rf runs/* data/splits/* data/clean/* data/features/* data/final/*

# Run full pipeline with MES only
./pipeline run --symbols MES --stages all

# Verify outputs exist
ls -lh data/splits/*/scaled/
# Expected files:
# - train_scaled.parquet
# - val_scaled.parquet
# - test_scaled.parquet
# - feature_scaler.pkl
# - scaling_metadata.json

# Check for errors
cat runs/*/logs/pipeline.log | grep -i "error\|critical"
# Expected: No critical errors

# Check validation report
cat runs/*/artifacts/validation_report.json | python -m json.tool
# Expected: 0 critical_issues
```

**Estimated Time:** 45 minutes

---

### 4.2. Multi-Symbol Test (Verify Graceful Handling)
**Action:** Test that pipeline still works with multiple symbols (cross-asset disabled)

```bash
# Clean slate
rm -rf runs/* data/splits/* data/clean/* data/features/* data/final/*

# Run with both symbols (cross-asset features should be skipped)
./pipeline run --symbols MES,MGC --stages all

# Verify no cross-asset features in output
head -n 1 data/final/combined_final_labeled.parquet | grep -i "correlation\|beta\|spread"
# Expected: No cross-asset feature columns

# Check logs confirm cross-asset skip
cat runs/*/logs/pipeline.log | grep -i "cross.asset"
# Expected: "Skipping cross-asset features (disabled)"
```

**Estimated Time:** 30 minutes

---

### 4.3. Data Quality Checks
**Action:** Verify final dataset quality

```python
import pandas as pd

# Load final dataset
train = pd.read_parquet('data/splits/.../scaled/train_scaled.parquet')

# Check 1: No NaN in features (after dropping invalid labels)
train_valid = train[train['label_h5'] != -99]
feature_cols = [c for c in train_valid.columns if c not in [...]]
nan_counts = train_valid[feature_cols].isnull().sum()
assert nan_counts.sum() == 0, f"Found NaN values: {nan_counts[nan_counts > 0]}"

# Check 2: Label distribution reasonable
label_dist = train_valid['label_h5'].value_counts(normalize=True)
print("Label distribution:", label_dist)
# Expected: Short ~20-30%, Neutral ~40-50%, Long ~20-30%

# Check 3: Sample weights in valid range
weights = train_valid['sample_weight_h5']
assert weights.min() >= 0.4, f"Weights too low: {weights.min()}"
assert weights.max() <= 1.6, f"Weights too high: {weights.max()}"

# Check 4: Feature scaling applied
feature_means = train_valid[feature_cols].mean()
feature_stds = train_valid[feature_cols].std()
print(f"Feature means (should be ~0): {feature_means.mean():.4f}")
print(f"Feature stds (should be ~1): {feature_stds.mean():.4f}")

# Check 5: OHLCV not scaled
assert train_valid['close'].mean() > 1, "OHLCV should not be scaled"
```

**Estimated Time:** 30 minutes

---

### 4.4. Performance Benchmarking
**Action:** Measure pipeline performance

```bash
# Time full pipeline
time ./pipeline run --symbols MES --stages all

# Expected runtime (690K bars → 60K samples):
# - Total: 30-45 seconds
# - Stage 3 (features): 20-25 seconds (largest bottleneck)
# - All other stages: <5 seconds

# Profile Stage 3 if too slow
python -m cProfile -o profile.stats src/pipeline_cli.py run --symbols MES --stages feature_engineering
python -c "import pstats; p = pstats.Stats('profile.stats'); p.sort_stats('cumulative').print_stats(20)"
```

**Estimated Time:** 30 minutes

---

**Total Time for Part 4:** 2-2.5 hours

---

## PART 5: Final Checklist (PRIORITY 5)

### Before Declaring Phase 1 Complete:

- [ ] **Cross-asset features removed/disabled**
  - [ ] Config flag added
  - [ ] Feature engineer updated
  - [ ] Validation updated
  - [ ] Tests updated
  - [ ] Tested with single symbol (MES only)

- [ ] **Bug fixes applied**
  - [ ] Import paths verified
  - [ ] Empty data handling improved
  - [ ] Row count validation added
  - [ ] NaN drop rate investigated

- [ ] **Documentation complete**
  - [ ] Phase 2 integration guide created
  - [ ] Main README updated
  - [ ] Known limitations documented
  - [ ] This TODO list completed

- [ ] **Testing complete**
  - [ ] End-to-end pipeline test passed
  - [ ] Multi-symbol test passed (cross-asset disabled)
  - [ ] Data quality checks passed
  - [ ] Performance benchmarked

- [ ] **Outputs verified**
  - [ ] train/val/test Parquet files exist
  - [ ] feature_scaler.pkl saved
  - [ ] split_config.json contains metadata
  - [ ] No validation failures (0 critical issues)

- [ ] **Ready for Phase 2**
  - [ ] Example code for loading Parquet in PyTorch
  - [ ] Example code for loading Parquet in XGBoost
  - [ ] Clear contract defined (what columns, what format)
  - [ ] No known blockers

---

## Estimated Total Time

| Part | Task | Time |
|------|------|------|
| 1 | Cross-asset removal | 1.5-2 hours |
| 2 | Bug fixes | 1-1.5 hours |
| 3 | Documentation | 1.5-2 hours |
| 4 | Testing | 2-2.5 hours |
| 5 | Final checklist | 0.5 hours |
| **TOTAL** | **All tasks** | **6.5-8.5 hours (1-2 days)** |

---

## Success Criteria

**Phase 1 is COMPLETE when:**
1. Pipeline runs successfully with single symbol (no errors)
2. Validation passes (0 critical issues)
3. Cross-asset features removed
4. Documentation updated for Phase 2
5. No known bugs or blockers

**Then you can start Phase 2: ML Model Factory! 🚀**

---

## Questions?

**What if I find new bugs during testing?**
- Add them to this TODO list
- Prioritize by severity (P0 = blocker, P1 = important, P2 = nice-to-have)
- Fix P0 bugs before declaring Phase 1 complete

**What if cross-asset features are needed later?**
- Re-enable flag in `src/config/features.py`
- Run pipeline with both symbols (MES,MGC)
- Verify validation passes

**What if NaN drop rate is too high?**
- Profile features to find main NaN sources
- Consider shorter warmup periods (e.g., SMA_100 instead of SMA_200)
- Consider forward-fill for some features
- Document decision in KNOWN_LIMITATIONS.md

**When should I start Phase 2?**
- After ALL items in this TODO are complete
- After Phase 2 architecture is designed
- After clear handoff contract is defined

---

**Ready to get started? Begin with Part 1: Cross-Asset Feature Removal! 💪**
