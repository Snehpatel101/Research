# Modular Architecture Guide

Quick reference for the refactored modular architecture of the pipeline.

---

## Feature Scaler Package

**Location:** `/src/stages/feature_scaler/`

### Module Breakdown

```
feature_scaler/
├── __init__.py .............. Package exports (all public APIs)
├── core.py .................. Enums, data classes, constants (195 lines)
│   ├── ScalerType enum
│   ├── ScalerConfig dataclass
│   ├── FeatureCategory enum
│   ├── FeatureScalingConfig dataclass
│   ├── ScalingStatistics dataclass
│   ├── ScalingReport dataclass
│   ├── FEATURE_PATTERNS dict
│   └── DEFAULT_SCALING_STRATEGY dict
│
├── scalers.py ............... Utility functions (125 lines)
│   ├── categorize_feature()
│   ├── get_default_scaler_type()
│   ├── should_log_transform()
│   ├── create_scaler()
│   └── compute_statistics()
│
├── scaler.py ................ Main FeatureScaler class (546 lines)
│   └── FeatureScaler
│       ├── fit() - Fit on training data only
│       ├── transform() - Transform using training statistics
│       ├── fit_transform()
│       ├── inverse_transform()
│       ├── save() - Pickle to disk
│       ├── load() - Load from pickle
│       └── get_scaling_report()
│
├── validators.py ............ Validation functions (366 lines)
│   ├── validate_scaling()
│   ├── validate_no_leakage()
│   ├── validate_scaling_for_splits()
│   └── add_scaling_validation_to_stage8()
│
└── convenience.py ........... High-level functions (122 lines)
    ├── scale_splits()
    └── scale_train_val_test()
```

### Usage Examples

**Simple API:**
```python
from stages.feature_scaler import scale_splits

train_scaled, val_scaled, test_scaled, scaler = scale_splits(
    train_df, val_df, test_df, feature_cols
)
```

**Advanced API:**
```python
from stages.feature_scaler import FeatureScaler, ScalerConfig

config = ScalerConfig(
    scaler_type='robust',
    clip_outliers=True,
    clip_range=(-5.0, 5.0)
)

scaler = FeatureScaler(config=config)
train_scaled = scaler.fit_transform(train_df, feature_cols)
val_scaled = scaler.transform(val_df)
test_scaled = scaler.transform(test_df)
```

**Validation:**
```python
from stages.feature_scaler import validate_scaling, validate_no_leakage

scaling_report = validate_scaling(scaler, train_df, val_df, test_df, feature_cols)
leakage_report = validate_no_leakage(train_df, val_df, test_df, scaler)
```

---

## Stage 2 Clean Package

**Location:** `/src/stages/stage2_clean/`

### Module Breakdown

```
stage2_clean/
├── __init__.py .............. Package exports (all public APIs)
│
├── utils.py ................. Core utilities (199 lines)
│   ├── calculate_atr_numba() [Numba-optimized ATR]
│   ├── validate_ohlc()
│   ├── detect_gaps_simple()
│   ├── fill_gaps_simple()
│   └── resample_to_5min()
│
├── cleaner.py ............... Main DataCleaner class (589 lines)
│   └── DataCleaner
│       ├── detect_gaps() - Detailed gap analysis
│       ├── fill_gaps() - Multiple gap fill methods
│       ├── detect_duplicates()
│       ├── detect_outliers_zscore()
│       ├── detect_outliers_iqr()
│       ├── detect_spikes_atr()
│       ├── clean_outliers()
│       ├── handle_contract_rolls()
│       ├── clean_file() - Complete pipeline for one file
│       ├── save_results()
│       └── clean_directory() - Batch processing
│
└── pipeline.py .............. Simple pipeline function (96 lines)
    └── clean_symbol_data() - Basic single-file cleaning
```

### Usage Examples

**Simple API (single file):**
```python
from stages.stage2_clean import clean_symbol_data

cleaned_df = clean_symbol_data(
    Path('data/raw/MES.parquet'),
    Path('data/clean/MES.parquet'),
    'MES'
)
```

**Advanced API (batch processing):**
```python
from stages.stage2_clean import DataCleaner

cleaner = DataCleaner(
    input_dir='data/raw',
    output_dir='data/clean',
    timeframe='1min',
    gap_fill_method='forward',
    max_gap_fill_minutes=5,
    outlier_method='atr',
    atr_threshold=5.0
)

results = cleaner.clean_directory(pattern='*.parquet')
```

**Per-file processing with custom logic:**
```python
from stages.stage2_clean import DataCleaner

cleaner = DataCleaner(input_dir='data/raw', output_dir='data/clean')
df, report = cleaner.clean_file(Path('data/raw/MES.parquet'))

# Inspect cleaning report
print(f"Gaps found: {report['gaps']['total_gaps']}")
print(f"Outliers removed: {report['outliers']['total_outliers']}")
print(f"Retention: {report['retention_pct']:.2f}%")
```

---

## Design Principles

### 1. Single Responsibility
Each module has one clear purpose:
- `core.py`: Define types and constants
- `utils.py`: Provide reusable utilities
- `scalers.py`: Scaler-specific logic
- `scaler.py`: Main class implementation
- `validators.py`: Validation logic
- `convenience.py`: High-level APIs

### 2. Dependency Direction
```
convenience.py  ┐
                ├─> scaler.py ┐
validators.py ──┘              ├─> core.py, scalers.py
                                └─> utils.py (for stage2)
```

### 3. Module Size Limits
**All modules < 650 lines** (enforced per CLAUDE.md)
- Longest: `scaler.py` (546 lines)
- Others: 69-366 lines

### 4. Import Strategy
**Package `__init__.py` exports public API:**
```python
# Users can import from package
from stages.feature_scaler import FeatureScaler

# Or import specific modules
from stages.feature_scaler.scaler import FeatureScaler
from stages.feature_scaler.core import FeatureCategory
```

---

## Testing

### feature_scaler tests
```
✓ tests/test_feature_scaler.py (24 tests)
✓ tests/phase_1_tests/utilities/test_feature_scaler.py (24 tests)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Total: 48/48 PASSED
```

### stage2_clean tests
```
✓ tests/phase_1_tests/stages/test_stage2_data_cleaning.py (14/16 passed)
  - 2 failures due to pre-existing fixture issues
```

---

## Migration from Old Structure

### Before
```python
from stages.feature_scaler import FeatureScaler  # 1730-line monolith
from stages.stage2_clean import DataCleaner      # 967-line monolith
```

### After
```python
# Same imports still work!
from stages.feature_scaler import FeatureScaler  # Now from scaler.py
from stages.stage2_clean import DataCleaner      # Now from cleaner.py

# New specific imports available
from stages.feature_scaler.core import FeatureCategory
from stages.feature_scaler.validators import validate_scaling
from stages.stage2_clean.utils import validate_ohlc
```

---

## File Locations

### Refactored Packages
- `/src/stages/feature_scaler/` - Modular scaler package
- `/src/stages/stage2_clean/` - Modular cleaner package

### Archived Old Files
- `/src/stages/feature_scaler_old.py` - Original 1730-line file
- `/src/stages/stage2_clean_old.py` - Original 967-line file

---

## Development Guidelines

### Adding New Features

**To add scaler type:**
1. Add enum to `core.py`
2. Add handler to `create_scaler()` in `scalers.py`
3. Update tests

**To add validation check:**
1. Add function to `validators.py`
2. Call from appropriate location in `cleaner.py`
3. Update test suite

**To add cleaning method:**
1. Add method to `DataCleaner` in `cleaner.py`
2. Call from `clean_file()` pipeline
3. Document in docstring
4. Test with actual data

### File Size Guidelines
- Max 650 lines per file (CLAUDE.md)
- Aim for 200-400 lines for good modularity
- Split if adding > 100 lines of code

---

## Performance Characteristics

### feature_scaler
- Memory: ~10-50 MB for fitted scaler (depends on n_features)
- Speed: O(n_features * n_samples) for fit/transform
- Features: Train-only fitting prevents leakage

### stage2_clean
- Gap filling: O(n_rows)
- Outlier detection: O(n_rows)
- Batch processing: O(n_files) with parallel potential

---

## Troubleshooting

### Import Issues
```python
# If getting ImportError, ensure you're using
from stages.feature_scaler import FeatureScaler  # Works
from stages.stage2_clean import DataCleaner      # Works

# NOT
from src.stages.feature_scaler import FeatureScaler  # May fail outside project root
```

### Circular Imports
None! Design ensures one-directional dependencies.

### Test Failures
See REFACTORING_SUMMARY_PHASE1.md for known issues.

---

## Summary

- **Modular:** 6 focused modules for feature_scaler, 4 for stage2_clean
- **Testable:** 48/48 feature tests passing, 14/16 clean tests passing
- **Maintainable:** Clear boundaries, single responsibility
- **Compatible:** All old imports still work
- **Extensible:** Easy to add new functionality
