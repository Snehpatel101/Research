# Workflow Integration and Run ID Collision Fixes

## Summary of Changes

This document describes the fixes implemented to resolve workflow integration issues between Phase 3 (cross-validation) and Phase 4 (ensemble training), as well as run ID collision prevention for parallel jobs.

---

## 1. Run ID Collision Prevention

### Problem
Run IDs used second-granularity timestamps (`%Y%m%d_%H%M%S`), causing parallel jobs to overwrite each other when launched within the same second.

### Solution
Added milliseconds (%f) and 4-character random suffix to all run ID generation:

**Format:** `{timestamp_with_ms}_{random_suffix}`
**Example:** `20251228_143025_789456_a3f9`

### Files Modified

#### `/home/user/Research/src/models/trainer.py` (Line 153-167)
```python
def _generate_run_id(self) -> str:
    """
    Generate unique run identifier with collision prevention.

    Format: {model}_{horizon}_{timestamp_with_ms}_{random_suffix}
    Example: xgboost_h20_20251228_143025_789456_a3f9

    Milliseconds + random suffix ensure uniqueness even for parallel runs.
    """
    import secrets
    # Include milliseconds (%f) for sub-second precision
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    # Add 4-character random suffix for collision prevention
    random_suffix = secrets.token_hex(2)  # 2 bytes = 4 hex chars
    return f"{self.config.model_name}_h{self.config.horizon}_{timestamp}_{random_suffix}"
```

#### `/home/user/Research/src/phase1/pipeline_config.py` (Line 31-38)
```python
# Run identification
# Format: {timestamp_with_ms}_{random_suffix} for collision prevention
# Example: 20251228_143025_789456_a3f9
run_id: str = field(
    default_factory=lambda: (
        lambda: f"{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}_{__import__('secrets').token_hex(2)}"
    )()
)
```

### Impact
- **Parallel training jobs** no longer overwrite each other's outputs
- **Parallel CV runs** create separate output directories
- **Pipeline runs** get unique IDs even when launched in rapid succession

---

## 2. CV Output Directory Isolation

### Problem
`scripts/run_cv.py` saved all CV results to a single shared `data/stacking/` directory, causing parallel CV runs to overwrite each other.

### Solution
Each CV run now creates a unique subdirectory with millisecond + random suffix.

**Format:** `data/stacking/{run_id}/`
**Example:** `data/stacking/20251228_143025_789456_a3f9/`

### Files Modified

#### `/home/user/Research/scripts/run_cv.py` (Line 61-74)
```python
def generate_cv_run_id() -> str:
    """
    Generate unique run ID for CV output directory.

    Format: {timestamp_with_ms}_{random_suffix}
    Example: 20251228_143025_789456_a3f9

    Prevents collision between parallel CV runs.
    """
    import secrets
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    random_suffix = secrets.token_hex(2)  # 2 bytes = 4 hex chars
    return f"{timestamp}_{random_suffix}"
```

### New CLI Arguments
- `--output-name`: Custom subdirectory name (default: auto-generated timestamp)

### Example Usage
```bash
# Auto-generated unique run ID
python scripts/run_cv.py --models xgboost,lightgbm --horizons 20

# Custom run name
python scripts/run_cv.py --models xgboost --horizons 20 --output-name "xgb_tuned_v1"
```

### Output Structure
```
data/stacking/
├── 20251228_143025_789456_a3f9/    # Auto-generated run ID
│   ├── cv_results.json
│   ├── stacking/
│   │   ├── stacking_dataset_h5.parquet
│   │   ├── stacking_dataset_h10.parquet
│   │   └── stacking_dataset_h20.parquet
│   └── tuned_params/
│       └── xgboost_h20.json
└── xgb_tuned_v1/                   # Custom run name
    └── ...
```

---

## 3. Phase 3→4 Workflow Integration

### Problem
Phase 3 (`run_cv.py`) generates leakage-safe OOF predictions for stacking, but Phase 4 (`train_model.py --model stacking`) didn't load them. Instead, it regenerated OOF predictions from scratch, wasting computation and potentially introducing leakage.

### Solution
Added `--stacking-data` argument to `train_model.py` to load Phase 3 stacking datasets directly.

### Files Modified

#### `/home/user/Research/scripts/train_model.py`

**New CLI Arguments:**
- `--stacking-data`: CV run ID for loading Phase 3 stacking data
- `--phase3-output`: Base directory for Phase 3 outputs (default: `data/stacking`)

**New Function: `load_phase3_stacking_data()` (Line 54-125)**
```python
def load_phase3_stacking_data(
    cv_run_id: str,
    horizon: int,
    phase3_base_dir: Path,
) -> Optional[Dict[str, Any]]:
    """
    Load Phase 3 stacking dataset for ensemble training.

    Args:
        cv_run_id: CV run ID (e.g., '20251228_143025_789456_a3f9')
        horizon: Label horizon
        phase3_base_dir: Base directory for Phase 3 outputs

    Returns:
        Dict with 'data' (DataFrame) and 'metadata' (dict), or None if not found

    Raises:
        FileNotFoundError: If stacking data does not exist
        ValueError: If data format is invalid
    """
    # ... implementation ...
```

**Training Workflow Update (Line 560-643)**
- Detects `--stacking-data` flag
- Loads Phase 3 OOF predictions
- Trains meta-learner directly on loaded data
- Skips redundant OOF generation

### Workflow Examples

#### Standard Phase 3→4 Workflow (Recommended)

**Step 1: Run Phase 3 Cross-Validation**
```bash
# Generate OOF predictions for xgboost and lightgbm on H20
python scripts/run_cv.py \
  --models xgboost,lightgbm \
  --horizons 20 \
  --n-splits 5

# Output:
# CV Run ID: 20251228_143025_789456_a3f9
# Results saved to: /home/user/Research/data/stacking/20251228_143025_789456_a3f9
# Stacking datasets saved to: /home/user/Research/data/stacking/20251228_143025_789456_a3f9/stacking
#
# To use in Phase 4:
#   python scripts/train_model.py --model stacking --horizon 20 --stacking-data 20251228_143025_789456_a3f9
```

**Step 2: Train Stacking Ensemble Using Phase 3 Data**
```bash
# Train meta-learner on Phase 3 OOF predictions
python scripts/train_model.py \
  --model stacking \
  --horizon 20 \
  --stacking-data 20251228_143025_789456_a3f9

# This workflow:
# ✓ Loads leakage-safe OOF predictions from Phase 3
# ✓ Skips redundant OOF generation
# ✓ Trains meta-learner quickly (no CV needed)
# ✓ Saves trained ensemble model
```

#### Alternative: Legacy Workflow (Generate OOF On-the-Fly)
```bash
# Train stacking ensemble from scratch (slower, generates OOF internally)
python scripts/train_model.py \
  --model stacking \
  --horizon 20 \
  --base-models xgboost,lightgbm,catboost

# This workflow:
# - Generates OOF predictions during training
# - Slower (runs internal CV)
# - Use only if you don't have Phase 3 CV results
```

---

## 4. Validation and Error Handling

### Phase 3 Stacking Data Validation
The `load_phase3_stacking_data()` function validates:

1. **Directory exists**: `data/stacking/{cv_run_id}/stacking/`
2. **Parquet file exists**: `stacking_dataset_h{horizon}.parquet`
3. **Data structure is valid**: Contains `y_true` column
4. **Metadata is present**: Loads model names and configuration

### Error Messages
```bash
# Missing CV run
FileNotFoundError: Phase 3 CV run not found: 20251228_143025_789456_a3f9
Expected directory: /home/user/Research/data/stacking/20251228_143025_789456_a3f9/stacking
Available runs: [...]

# Missing horizon
FileNotFoundError: Stacking dataset not found for H20
Expected: /home/user/Research/data/stacking/20251228_143025_789456_a3f9/stacking/stacking_dataset_h20.parquet
Available: [stacking_dataset_h5.parquet, stacking_dataset_h10.parquet]

# Invalid data structure
ValueError: Invalid stacking dataset: missing 'y_true' column
Columns: [xgboost_pred, xgboost_prob_short, ...]
```

---

## 5. Backward Compatibility

All changes are **backward compatible**:

✓ Existing scripts without `--stacking-data` work unchanged
✓ Existing config files continue to work
✓ Existing run ID formats are still supported (just longer now)
✓ Old CV results in `data/stacking/` are not affected

---

## 6. Testing Run ID Uniqueness

```python
# Test collision prevention
import secrets
from datetime import datetime

# Generate 10 IDs rapidly
ids = []
for i in range(10):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    random_suffix = secrets.token_hex(2)
    run_id = f"{timestamp}_{random_suffix}"
    ids.append(run_id)

print(f"Generated: {len(ids)}")
print(f"Unique: {len(set(ids))}")  # Should be 10
```

Expected output:
```
Generated: 10
Unique: 10
Status: PASS
```

---

## 7. Complete Workflow Example

### Scenario: Train XGBoost + LightGBM Stacking Ensemble for H20

```bash
# Step 1: Run Phase 1 (data pipeline) - if not already done
./pipeline run --symbols MES

# Step 2: Run Phase 3 (cross-validation with OOF generation)
python scripts/run_cv.py \
  --models xgboost,lightgbm \
  --horizons 20 \
  --n-splits 5 \
  --tune \
  --n-trials 100

# Output shows:
# CV Run ID: 20251228_143025_789456_a3f9
# To use in Phase 4:
#   python scripts/train_model.py --model stacking --horizon 20 --stacking-data 20251228_143025_789456_a3f9

# Step 3: Train stacking ensemble using Phase 3 OOF predictions
python scripts/train_model.py \
  --model stacking \
  --horizon 20 \
  --stacking-data 20251228_143025_789456_a3f9 \
  --meta-learner logistic

# Step 4: Evaluate final ensemble on test set (Phase 5)
python scripts/evaluate_model.py \
  --run-id stacking_h20_20251228_143030_123456_b8c2 \
  --split test
```

---

## 8. Benefits

### Performance
- **Faster ensemble training**: No redundant OOF generation
- **Parallelization-safe**: No run ID collisions

### Correctness
- **No leakage**: Uses truly out-of-sample predictions from Phase 3
- **Reproducible**: Same CV run → same stacking dataset

### Usability
- **Clear workflow**: Phase 3 → Phase 4 with explicit run IDs
- **Good error messages**: Tells you exactly what's missing
- **Backward compatible**: Old workflows still work

---

## 9. File Summary

| File | Lines Modified | Change Type |
|------|----------------|-------------|
| `src/models/trainer.py` | 153-167 | Run ID generation with ms + random |
| `src/phase1/pipeline_config.py` | 31-38 | Run ID generation with ms + random |
| `scripts/run_cv.py` | 61-74, 175-179, 244-250, 363-367 | CV output isolation + helper |
| `scripts/train_model.py` | 54-125, 109-120, 465-487, 489-517, 560-643 | Phase 3→4 integration |

---

## 10. Next Steps

1. **Test the new workflow** on real data
2. **Update CI/CD** to use new run ID format
3. **Document in CLAUDE.md** for future reference
4. **Consider adding** `--list-cv-runs` command to show available Phase 3 runs

---

## Questions?

Check the help text:
```bash
python scripts/run_cv.py --help
python scripts/train_model.py --help
```

Or examine the code:
- Phase 3 CV: `src/cross_validation/cv_runner.py`
- Phase 4 Ensemble: `src/models/ensemble/`
- Stacking data: `src/cross_validation/oof_generator.py`
