# Phase 1 Configuration Fixes

**Date:** 2024-12-24
**Status:** Complete
**Goal:** Improve modularity and Phase 2 readiness

---

## Overview

Fixed three critical configuration issues to make Phase 1 more modular and ready for Phase 2 model training:

1. **Feature categorization incomplete** - Many features fell into UNKNOWN category
2. **Label templates hardcoded** - Duplicated across files instead of centralized
3. **Feature sets missing model hints** - No guidance on model compatibility

---

## Changes Made

### 1. Feature Categorization (100% Coverage)

**File:** `/home/jake/Desktop/Research/src/stages/scaling/core.py`

**Before:** 67 features (50%) categorized as UNKNOWN
**After:** 0 features (0%) categorized as UNKNOWN

**Added patterns:**
- Ratio features: `hl_ratio`, `co_ratio`, `volume_ratio`, `close_sma20_ratio_*`
- Z-score features: `close_bb_zscore`, `volume_zscore`, `*_zscore`
- Deviation features: `close_kc_atr_dev`, `*_dev`
- Cross-asset features: `*_spread`, `*_beta`, `*_correlation`
- MTF price levels: `open_*`, `high_*`, `low_*`, `close_*`
- Additional binary: `structure_regime`, `timeframe`
- Additional oscillator: `relative_strength`

**Final distribution:**
```
RETURNS     : 43 features (32%)
PRICE_LEVEL : 37 features (28%)
OSCILLATOR  : 18 features (13%)
VOLATILITY  : 15 features (11%)
BINARY      : 12 features (9%)
TEMPORAL    :  6 features (4%)
VOLUME      :  3 features (2%)
UNKNOWN     :  0 features (0%) ✓
```

---

### 2. Label Templates Centralized

**New file:** `/home/jake/Desktop/Research/src/config/labels.py` (176 lines)

**What it provides:**

```python
# Template definitions
REQUIRED_LABEL_TEMPLATES = ['label_h{h}', 'sample_weight_h{h}']
OPTIONAL_LABEL_TEMPLATES = [
    'quality_h{h}', 'bars_to_hit_h{h}', 'mae_h{h}', 'mfe_h{h}',
    'touch_type_h{h}', 'pain_to_gain_h{h}', 'time_weighted_dd_h{h}',
    'fwd_return_h{h}', 'fwd_return_log_h{h}', 'time_to_hit_h{h}',
]

# Helper functions
get_required_label_columns(horizon: int) -> List[str]
get_optional_label_columns(horizon: int) -> List[str]
get_all_label_columns(horizon: int) -> List[str]
is_label_column(column_name: str) -> bool
get_label_metadata(column_template: str, horizon: int) -> Dict
```

**Label metadata includes:**
- Data types (int8, float32, etc.)
- Value ranges
- Meanings (e.g., -1 = short, 0 = neutral, 1 = long)
- Descriptions

**Updated files:**
- `/home/jake/Desktop/Research/src/stages/datasets/run.py` - Now imports from config
- `/home/jake/Desktop/Research/src/config/__init__.py` - Exports labels module

---

### 3. Feature Sets Enhanced with Model Hints

**File:** `/home/jake/Desktop/Research/src/config/feature_sets.py`

**Added fields to `FeatureSetDefinition`:**

```python
@dataclass(frozen=True)
class FeatureSetDefinition:
    # ... existing fields ...
    supported_model_types: List[str]      # NEW
    default_sequence_length: Optional[int] # NEW
    recommended_scaler: str                # NEW
```

**Feature set configurations:**

| Feature Set | MTF | Cross-Asset | Seq Length | Models | Scaler |
|-------------|-----|-------------|------------|--------|--------|
| `core_min`  | No  | No          | 60         | tabular, tree, sequential | robust |
| `core_full` | No  | No          | 60         | tabular, tree, sequential | robust |
| `mtf_plus`  | Yes | Yes         | 120        | tabular, tree, sequential | robust |

**Model type hints:**
- `tabular`: LightGBM, XGBoost, CatBoost (feed-forward)
- `tree`: Tree-based models
- `sequential`: LSTM, GRU, Transformer (requires sequences)

**Sequence length guidance:**
- Base features: 60 bars (~1 hour at 1-min resolution)
- MTF features: 120 bars (~2 hours) - longer context for MTF alignment

---

## Testing Results

All integration tests passed:

```
✓ Labels configuration: PASS
✓ Feature sets with model hints: PASS
✓ Feature categorization (100%): PASS
✓ Dataset stage integration: PASS
✓ Backward compatibility: PASS
```

**File size compliance:**
- `src/stages/scaling/core.py`: 211 lines (< 650 limit)
- `src/config/labels.py`: 176 lines (< 650 limit)
- `src/config/feature_sets.py`: 139 lines (< 650 limit)
- `src/stages/datasets/run.py`: 196 lines (< 650 limit)
- `src/config/__init__.py`: 355 lines (< 650 limit)

---

## Impact on Phase 2

### Model Training Benefits

1. **Feature selection is now guided by category:**
   ```python
   # Example: Select only returns + oscillators for tree models
   categories = [FeatureCategory.RETURNS, FeatureCategory.OSCILLATOR]
   features = [f for f in all_features if categorize(f) in categories]
   ```

2. **Scaler selection is automatic:**
   ```python
   # ScalerType determined by feature category
   scaler = DEFAULT_SCALING_STRATEGY[category]
   ```

3. **Model hints prevent incompatibilities:**
   ```python
   feature_set = FEATURE_SET_DEFINITIONS['mtf_plus']
   assert 'sequential' in feature_set.supported_model_types  # OK for LSTM
   seq_length = feature_set.default_sequence_length  # 120 bars
   ```

### Label Access is Now Consistent

```python
# Old way (hardcoded)
label_cols = ['label_h5', 'sample_weight_h5', 'quality_h5', ...]

# New way (config-driven)
from src.config.labels import get_all_label_columns
label_cols = get_all_label_columns(horizon=5)
```

---

## Breaking Changes

**None.** All changes are backward compatible:

- Old imports still work via `src.config.__init__`
- Datasets stage transparently uses new label config
- Feature categorization is internal to scaling module

---

## Next Steps for Phase 2

With these fixes, Phase 2 model training can now:

1. **Select features by category** for different model types
2. **Use model hints** to validate feature set compatibility
3. **Access label metadata** programmatically for validation
4. **Create sequences** with correct lengths per feature set

Example Phase 2 usage:

```python
from src.config import FEATURE_SET_DEFINITIONS, get_required_label_columns

# Get feature set
fset = FEATURE_SET_DEFINITIONS['mtf_plus']

# Validate model compatibility
assert 'sequential' in fset.supported_model_types

# Create sequences
seq_length = fset.default_sequence_length  # 120
X_sequences = create_sequences(X, seq_length)

# Get labels
label_cols = get_required_label_columns(horizon=5)
y = df[label_cols[0]]  # 'label_h5'
```

---

## Verification Commands

```bash
# Test label config
python -c "from src.config.labels import get_required_label_columns; print(get_required_label_columns(5))"

# Test feature sets
python -c "from src.config import FEATURE_SET_DEFINITIONS; print(FEATURE_SET_DEFINITIONS['mtf_plus'].supported_model_types)"

# Test categorization
python -c "from src.stages.scaling.core import FeatureCategory, FEATURE_PATTERNS; print(len(FEATURE_PATTERNS[FeatureCategory.RETURNS]))"
```

---

## Files Modified

1. `/home/jake/Desktop/Research/src/stages/scaling/core.py` - Feature patterns
2. `/home/jake/Desktop/Research/src/config/labels.py` - **NEW** Label templates
3. `/home/jake/Desktop/Research/src/config/feature_sets.py` - Model hints
4. `/home/jake/Desktop/Research/src/stages/datasets/run.py` - Import from config
5. `/home/jake/Desktop/Research/src/config/__init__.py` - Export labels

---

## Summary

All Phase 1 configuration issues resolved:

- ✅ Feature categorization: 0% unknown (was 50%)
- ✅ Label templates: Centralized in `src.config.labels`
- ✅ Feature sets: Model compatibility hints added
- ✅ All tests passing
- ✅ Backward compatible
- ✅ Under 650-line limit

**Phase 1 is now fully modular and ready for Phase 2 model training.**
