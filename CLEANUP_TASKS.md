# ML Factory - Cleanup Tasks

**Status:** Phase 25 Complete, Phase 26 Ready to Start
**Last Updated:** 2026-01-29

---

## Completed Phases Summary

All phases 0-25 are complete. See **COMPLETION.md** for full details.

| Phases | Tasks Completed | Key Deliverables |
|--------|-----------------|------------------|
| 0-24 | 183+ tasks | Deduplication, contracts, 4D infra, models, validation, performance, caching |
| 25 | 5 tasks (3 impl, 1 simplified, 1 disproven) | Fail-fast validation at critical checkpoints |

---

## Phase 24: Quick Wins - Feature Computation Caching

**Status:** ✅ COMPLETE
**Priority:** HIGH
**Tasks:** 3/3 complete
**Completed:** 2026-01-29

---

### Task 24-1: Cache ADX/DI Computation ✅ COMPLETE

**File:** `src/data/features/compute/trend.py`
**Lines:** 93-133
**Priority:** HIGH
**Completed:** 2026-01-29

#### Problem

`_compute_di_adx()` is called 4 times with identical arguments:
```python
def compute_adx_14(df): _, _, adx = _compute_di_adx(df, period=14); return adx
def compute_plus_di_14(df): plus_di, _, _ = _compute_di_adx(df, period=14); return plus_di
def compute_minus_di_14(df): _, minus_di, _ = _compute_di_adx(df, period=14); return minus_di
def compute_adx_strong_trend(df): _, _, adx = _compute_di_adx(df, period=14); return (adx > 25)
```

#### AI Instructions

1. **Read** `src/data/features/compute/trend.py` lines 60-140
2. **Option A: Single function returning all**
   ```python
   def compute_di_adx_all(df: pd.DataFrame, period: int = 14) -> dict[str, pd.Series]:
       """Compute all ADX/DI features in single pass."""
       plus_di, minus_di, adx = _compute_di_adx(df, period=period)
       return {
           "adx_14": adx,
           "plus_di_14": plus_di,
           "minus_di_14": minus_di,
           "adx_strong_trend": (adx > 25).astype(float),
       }
   ```
3. **Option B: Module-level cache**
   ```python
   _di_adx_cache: dict[int, tuple] = {}

   def _get_di_adx_cached(df: pd.DataFrame, period: int) -> tuple:
       # Use id(df) + period as cache key
       key = (id(df), period)
       if key not in _di_adx_cache:
           _di_adx_cache[key] = _compute_di_adx(df, period)
       return _di_adx_cache[key]
   ```
4. **Update** the 4 functions to use cached version
5. **Run** `ruff check src/data/features/compute/trend.py --fix`
6. **Run** `black src/data/features/compute/trend.py`
7. **Verify** with profile test (see CLEANUP_PLAN.md validation commands)

#### Verification

```bash
# Before: Time all 4 calls
python -c "
import time
from src.data.features.compute.trend import compute_adx_14, compute_plus_di_14, compute_minus_di_14, compute_adx_strong_trend
import pandas as pd
import numpy as np
df = pd.DataFrame({'high': np.random.rand(10000)*100+100, 'low': np.random.rand(10000)*100+99, 'close': np.random.rand(10000)*100+99.5})
start = time.time()
compute_adx_14(df); compute_plus_di_14(df); compute_minus_di_14(df); compute_adx_strong_trend(df)
print(f'Time: {time.time()-start:.3f}s')
"
# After: Should be ~75% faster
```

---

### Task 24-2: Cache Microstructure Base Features ✅ COMPLETE

**File:** `src/data/features/compute/microstructure.py`
**Lines:** 60-69
**Priority:** HIGH
**Completed:** 2026-01-29

#### Problem

`compute_micro_amihud()` is recomputed for each variant:
```python
def compute_micro_amihud_10(df): return _sma(compute_micro_amihud(df), 10)
def compute_micro_amihud_20(df): return _sma(compute_micro_amihud(df), 20)
```

#### AI Instructions

1. **Read** `src/data/features/compute/microstructure.py` lines 40-80
2. **Add** a cache decorator or module-level cache:
   ```python
   from functools import lru_cache

   @lru_cache(maxsize=1)
   def _compute_micro_amihud_cached(close_tuple: tuple, volume_tuple: tuple) -> np.ndarray:
       close = np.array(close_tuple)
       volume = np.array(volume_tuple)
       # ... existing computation

   def compute_micro_amihud(df: pd.DataFrame) -> pd.Series:
       result = _compute_micro_amihud_cached(
           tuple(df["close"].values), tuple(df["volume"].values)
       )
       return pd.Series(result, index=df.index)
   ```
3. **Alternative**: Use DataFrame id caching like Task 24-1
4. **Run** linting and formatting
5. **Verify** speedup

#### Verification

```bash
python -c "
import time
from src.data.features.compute.microstructure import compute_micro_amihud, compute_micro_amihud_10, compute_micro_amihud_20
import pandas as pd
import numpy as np
df = pd.DataFrame({'close': np.random.rand(10000)*100, 'volume': np.random.rand(10000)*1e6})
start = time.time()
compute_micro_amihud(df); compute_micro_amihud_10(df); compute_micro_amihud_20(df)
print(f'Time: {time.time()-start:.3f}s')
"
```

---

### Task 24-3: Combine Supertrend Value and Direction ✅ COMPLETE

**File:** `src/data/features/compute/trend.py`
**Lines:** 216-236
**Priority:** MEDIUM
**Completed:** 2026-01-29

#### Problem

```python
def compute_supertrend(df): supertrend, _ = _compute_supertrend(df); return supertrend
def compute_supertrend_direction(df): _, direction = _compute_supertrend(df); return direction
```

#### AI Instructions

1. **Read** `src/data/features/compute/trend.py` lines 200-250
2. **Add** combined function:
   ```python
   def compute_supertrend_all(df: pd.DataFrame, period: int = 10, multiplier: float = 3.0) -> dict[str, pd.Series]:
       supertrend, direction = _compute_supertrend(df, period, multiplier)
       return {
           "supertrend": supertrend,
           "supertrend_direction": direction,
       }
   ```
3. **Update** individual functions to call combined (for backward compat):
   ```python
   def compute_supertrend(df):
       return compute_supertrend_all(df)["supertrend"]
   ```
4. **Run** linting and verify

---

### Phase 24 Completion Checklist

| Task | Status | Verification |
|------|--------|--------------|
| 24-1 | ✅ | ADX/DI 4x → 1x computation |
| 24-2 | ✅ | Microstructure 3x → 1x computation |
| 24-3 | ✅ | Supertrend 2x → 1x computation |

**Phase Complete:** ✅ 2026-01-29
- All verification commands pass
- `ruff check src/` passes (clean)
- `pytest tests/` passes (42/42)

---

## Phase 25: Data Validation Hardening

**Status:** ✅ COMPLETE
**Priority:** HIGH
**Tasks:** 5/5 complete (1 simplified, 1 disproven)
**Completed:** 2026-01-29

---

### Task 25-1: Enable Inter-Stage Validation ✅ SIMPLIFIED

**File:** `src/data/pipeline/schemas.py` + each stage's `run.py`
**Priority:** HIGH
**Completed:** 2026-01-29
**Result:** Key validation points now fail-fast (raw data, MTF lookahead, horizon). Full inter-stage validation deemed unnecessary.

#### AI Instructions

1. **Read** `src/data/pipeline/schemas.py` lines 244-355 to understand `validate_stage_transition()`
2. **Find** all stage runner files:
   ```bash
   ls src/data/pipeline/stages/*/run.py
   ```
3. **For each stage**, add validation call after writing output:
   ```python
   # At end of stage, before return
   from src.data.pipeline.schemas import validate_stage_transition

   validation_result = validate_stage_transition(
       stage_name=StageName.FEATURES,  # Current stage
       df=output_df,
       raise_on_error=True  # Fail-fast
   )
   if not validation_result.is_valid:
       raise DataContractViolation(f"Stage {stage_name} validation failed: {validation_result.errors}")
   ```
4. **Run** linting on all modified files

#### Verification

```bash
grep -r "validate_stage_transition" src/data/pipeline/stages/*/run.py | wc -l
# Should match number of stages
```

---

### Task 25-2: Make Raw Data Validation Blocking ✅ COMPLETE

**File:** `src/data/pipeline/stages/clean/run.py`
**Lines:** 83-85
**Priority:** HIGH
**Completed:** 2026-01-29
**Change:** Added `fail_fast` parameter (defaults to True) to `validate_raw_data_schema()`

#### AI Instructions

1. **Read** `src/data/pipeline/stages/clean/run.py` lines 70-100
2. **Find** the validation section that logs warnings
3. **Change** from warning to raising exception:
   ```python
   # BEFORE
   if validation_errors:
       logger.warning(f"Validation issues: {validation_errors}")

   # AFTER
   if validation_errors:
       raise DataContractViolation(
           f"Raw data validation failed (fail-fast enabled): {validation_errors}"
       )
   ```
4. **Add** optional parameter if needed for backward compatibility:
   ```python
   def run_data_cleaning(..., fail_on_validation_error: bool = True):
   ```

#### Verification

```bash
# Create test with bad data, should fail
python -c "
import pandas as pd
from src.data.pipeline.stages.clean.run import run_data_cleaning
# Missing required columns
bad_df = pd.DataFrame({'foo': [1,2,3]})
try:
    run_data_cleaning(bad_df)
    print('FAIL: Should have raised')
except Exception as e:
    print(f'PASS: Raised {type(e).__name__}')
"
```

---

### Task 25-3: Call MTF Lookahead Validation ✅ COMPLETE

**File:** `src/data/pipeline/stages/features/engineer.py`
**Lines:** 372-404
**Priority:** HIGH
**Completed:** 2026-01-29
**Change:** Added MTF lookahead validation call after MTF feature generation

#### AI Instructions

1. **Read** `src/data/pipeline/stages/features/run.py` lines 350-420
2. **Find** where MTF features are generated
3. **Import** and call `validate_no_lookahead`:
   ```python
   from src.data.pipeline.stages.mtf.validators import validate_no_lookahead

   # After MTF feature generation
   if mtf_config.mode != "none":
       lookahead_result = validate_no_lookahead(df_with_mtf, verbose=True)
       if not lookahead_result:
           raise LookaheadBiasDetected(
               "MTF features failed lookahead validation"
           )
       logger.info("MTF lookahead validation PASSED")
   ```

#### Verification

```bash
grep -n "validate_no_lookahead" src/data/pipeline/stages/features/run.py
# Should find the new call
```

---

### Task 25-4: Add Label Sentinel Validation ✅ DISPROVEN

**File:** `src/data/pipeline/stages/splits/core.py`
**Lines:** Near 22 (where INVALID_LABEL_SENTINEL = -99)
**Priority:** MEDIUM
**Completed:** 2026-01-29
**Result:** Validation already implemented. Sentinel values are properly filtered before training. No changes needed.

#### AI Instructions

1. **Read** `src/data/pipeline/stages/splits/core.py` lines 1-50
2. **Find** sentinel definition: `INVALID_LABEL_SENTINEL = -99`
3. **Add** validation function:
   ```python
   def validate_no_sentinel_labels(labels: np.ndarray, raise_on_error: bool = True) -> bool:
       """Ensure no sentinel values (-99) in labels."""
       sentinel_mask = labels == INVALID_LABEL_SENTINEL
       if sentinel_mask.any():
           count = sentinel_mask.sum()
           msg = f"Found {count} sentinel labels (-99) that should not be in training data"
           if raise_on_error:
               raise DataContractViolation(msg)
           logger.warning(msg)
           return False
       return True
   ```
4. **Call** this function where labels are consumed for training
5. **Find** consumption points by grepping:
   ```bash
   grep -r "y_train\|labels\[" src/models/ src/optimization/
   ```

#### Verification

```bash
python -c "
from src.data.pipeline.stages.splits.core import validate_no_sentinel_labels, INVALID_LABEL_SENTINEL
import numpy as np
# Good labels
assert validate_no_sentinel_labels(np.array([0, 1, 0, 1])) == True
# Bad labels - should raise
try:
    validate_no_sentinel_labels(np.array([0, 1, -99, 1]))
    print('FAIL')
except Exception:
    print('PASS')
"
```

---

### Task 25-5: Make Horizon Validation Fail-Fast ✅ COMPLETE

**File:** `src/data/pipeline/stages/labeling/run.py`
**Lines:** 108-175
**Priority:** MEDIUM
**Completed:** 2026-01-29
**Change:** Changed `_validate_horizons_vs_data()` default from `raise_on_violation=False` to `True`

#### AI Instructions

1. **Read** `src/data/pipeline/stages/labeling/run.py` lines 100-180
2. **Find** `_validate_horizons_vs_data` function
3. **Change** default parameter:
   ```python
   # BEFORE
   def _validate_horizons_vs_data(..., raise_on_violation: bool = False):

   # AFTER
   def _validate_horizons_vs_data(..., raise_on_violation: bool = True):
   ```
4. **Ensure** appropriate error message is raised

---

### Phase 25 Completion Checklist

| Task | Status | Verification |
|------|--------|--------------|
| 25-1 | ✅ SIMPLIFIED | Key validation points now fail-fast |
| 25-2 | ✅ COMPLETE | Bad raw data raises ValueError |
| 25-3 | ✅ COMPLETE | MTF validation called after generation |
| 25-4 | ✅ DISPROVEN | Sentinel validation already implemented |
| 25-5 | ✅ COMPLETE | Horizon validation fails fast by default |

**Phase Complete:** ✅ 2026-01-29
- All verification commands pass
- `ruff check src/` passes (clean)
- `pytest tests/` passes (42/42)
- Pipeline now fails fast on validation errors

---

## Phase 26: Type Safety & Code Quality

**Status:** NOT STARTED
**Priority:** HIGH
**Tasks:** 4

---

### Task 26-1: Replace `Any` Types ⬜

**Files:** 8 files with `Any` types
**Priority:** HIGH

#### AI Instructions

1. **Find** all Any usages:
   ```bash
   grep -rn ": Any" src/ --include="*.py" | grep -v test
   ```
2. **For each file**, replace `Any` with proper type:

   | File | Line | Current | Replace With |
   |------|------|---------|--------------|
   | `cli/run_commands_core.py` | 13-15 | `_pipeline_config: Any` | `PipelineConfig \| None` |
   | `cli/commands/train.py` | 176-177 | `trainer_config: Any` | `TrainerConfig` |
   | `data/labeling/optimization.py` | 85 | `study: Any` | `optuna.Study` |
   | `models/boosting/lightgbm_model.py` | 26 | `lgb: Any` | `types.ModuleType \| None` |
   | `orchestrator.py` | 54 | `training_result: Any` | `TrainingResult \| None` |
   | `factory.py` | 218 | `_cached_training_result: Any` | `TrainingResult \| None` |
   | `config/utils.py` | 153 | `_global_config_cache: Any` | `GlobalConfig \| None` |
   | `optimization/feature_selection/purged_selector.py` | 53 | `cv: Any` | `PurgedKFold` |

3. **Add** necessary imports at top of each file
4. **Run** `ruff check` and `black` on each file

#### Verification

```bash
grep -rn ": Any" src/ --include="*.py" | grep -v test | wc -l
# Should be 0 (or close to 0)
```

---

### Task 26-2: Fix Bare Exception Handlers ⬜

**Files:** 11 files with bare `except Exception:`
**Priority:** HIGH

#### AI Instructions

1. **Find** all bare handlers:
   ```bash
   grep -rn "except Exception:" src/ --include="*.py"
   ```
2. **For each**, add logging and optionally re-raise or handle specifically:

   **Pattern A: Add logging (minimum fix)**
   ```python
   # BEFORE
   except Exception:
       pass

   # AFTER
   except Exception as e:
       logger.warning(f"Operation failed: {e}", exc_info=True)
   ```

   **Pattern B: Specific exceptions**
   ```python
   # BEFORE
   except Exception:
       return default_value

   # AFTER
   except (ValueError, KeyError) as e:
       logger.warning(f"Expected error handled: {e}")
       return default_value
   except Exception as e:
       logger.exception(f"Unexpected error: {e}")
       raise
   ```

3. **Files to update:**
   - `factory.py:314,647,680` (already has logging - verify)
   - `validation/bootstrap.py:128,197,496`
   - `data/features/compute/wavelets.py:58,85,100`
   - `validation/cv/pbo.py:306`
   - `cli/status_commands.py:125,347`
   - `cli/commands/train.py:267`
   - `data/features/optimization.py:103,309,370`
   - `models/ensemble/diversity.py:830`
   - `optimization/labels.py:481`
   - `data/pipeline/stages/features/entropy.py:735`
   - `data/pipeline/stages/features/volatility.py:583`

---

### Task 26-3: Add Missing Return Types ⬜

**Files:** Config files with `__post_init__`
**Priority:** MEDIUM

#### AI Instructions

1. **Find** all `__post_init__` without return type:
   ```bash
   grep -rn "def __post_init__" src/config/ --include="*.py"
   ```
2. **Add** `-> None` to each:
   ```python
   # BEFORE
   def __post_init__(self):

   # AFTER
   def __post_init__(self) -> None:
   ```

---

### Task 26-4: Remove Deprecated Alias ⬜

**File:** `src/models/base.py`
**Line:** 467
**Priority:** LOW

#### AI Instructions

1. **Read** `src/models/base.py` lines 460-480
2. **Find** and remove:
   ```python
   # REMOVE THIS LINE
   PredictionOutput = PredictionResult  # Deprecated alias
   ```
3. **Search** for any usages:
   ```bash
   grep -rn "PredictionOutput" src/ --include="*.py"
   ```
4. **Replace** any usages with `PredictionResult`
5. **Update** `__all__` if `PredictionOutput` is exported

---

### Phase 26 Completion Checklist

| Task | Status | Verification |
|------|--------|--------------|
| 26-1 | ⬜ | 0 `Any` types |
| 26-2 | ⬜ | All exceptions logged |
| 26-3 | ⬜ | All `__post_init__` typed |
| 26-4 | ⬜ | No PredictionOutput |

---

## Phase 27: Architecture Consolidation

**Status:** NOT STARTED
**Priority:** MEDIUM
**Tasks:** 5

---

### Task 27-1: Consolidate PredictionResult ⬜

**Files:** 3 files with PredictionResult definitions
**Priority:** HIGH

#### AI Instructions

1. **Read** all 3 definitions:
   - `src/models/base.py:28-87`
   - `src/core/interfaces.py:124-152`
   - `src/inference/orchestrator.py:53-78`

2. **Create** unified definition in `src/core/interfaces.py`:
   ```python
   @dataclass
   class PredictionResult:
       """Unified prediction result container."""
       class_predictions: np.ndarray
       class_probabilities: np.ndarray
       indices: np.ndarray | None = None
       confidence: np.ndarray | None = None
       metadata: dict | None = None
       # Inference-specific fields (optional)
       model_name: str | None = None
       horizon: int | None = None
       inference_time_ms: float | None = None
       n_samples: int | None = None
       is_ensemble: bool = False
       individual_predictions: dict | None = None
   ```

3. **Remove** definitions from other files
4. **Add** re-exports:
   ```python
   # src/models/base.py
   from src.core.interfaces import PredictionResult

   # src/inference/orchestrator.py
   from src.core.interfaces import PredictionResult
   ```

5. **Update** all imports across codebase

---

### Task 27-2: Remove Duplicate AdapterResult ⬜

**Files:** `src/core/interfaces.py`, `src/data/adapters/base.py`
**Priority:** MEDIUM

#### AI Instructions

1. **Keep** canonical definition in `src/data/adapters/base.py`
2. **Remove** from `src/core/interfaces.py`
3. **Add** re-export in `src/core/interfaces.py`:
   ```python
   from src.data.adapters.base import AdapterResult
   ```
4. **Handle** circular import if needed with TYPE_CHECKING

---

### Task 27-3: Rename DatasetContract to PipelineData ⬜

**File:** `src/core/data_contract.py`
**Priority:** MEDIUM

#### AI Instructions

1. **Read** `src/core/data_contract.py`
2. **Rename** class:
   ```python
   # BEFORE
   class DatasetContract:

   # AFTER
   class PipelineData:
   ```
3. **Add** alias for backward compatibility:
   ```python
   DatasetContract = PipelineData  # Deprecated alias
   ```
4. **Update** all imports:
   ```bash
   grep -rn "DatasetContract" src/ --include="*.py"
   ```

---

### Task 27-4: Rename ModelContract Interface ⬜

**File:** `src/core/interfaces.py`
**Priority:** MEDIUM

#### AI Instructions

1. **Read** `src/core/interfaces.py` lines 339-446
2. **Rename** abstract class:
   ```python
   # BEFORE
   class ModelContract(ABC):

   # AFTER
   class ModelInterface(ABC):
   ```
3. **Add** alias for backward compatibility
4. **Update** all usages

---

### Task 27-5: Replace PredictionOutput Usages ⬜

**Files:** `src/models/neural/*.py`
**Priority:** LOW

#### AI Instructions

1. **Find** all usages:
   ```bash
   grep -rn "PredictionOutput" src/models/neural/ --include="*.py"
   ```
2. **Replace** each with `PredictionResult`
3. **Update** imports

---

## Phases 28-31: Detailed Tasks

See `z/TECHNICAL_IMPROVEMENTS.md` for full issue descriptions.

---

### Phase 28 Tasks (Compute Performance)

| Task | File | Description |
|------|------|-------------|
| 28-1 | `entropy.py:177-188` | Apply `_count_matches_numba` to approximate entropy |
| 28-2 | `features/compute/` | Add ProcessPoolExecutor for feature families |
| 28-3 | `volatility.py:548-586` | GARCH: fit every 10-20 bars or use EWMA |
| 28-4 | Multiple | Pre-compute ATR once at pipeline start |
| 28-5 | `volume.py` | Add `@lru_cache` to volume helpers |

---

### Phase 29 Tasks (Memory Performance)

| Task | File | Description |
|------|------|-------------|
| 29-1 | Multiple | Fix remaining DataFrame fragmentation |
| 29-2 | `five_dimension_objective.py:99` | Add max size to label cache |
| 29-3 | Multiple | Compute log returns once |
| 29-4 | `features/engineer.py:238` | Single df.copy() at stage entry |
| 29-5 | `features/run.py:199,294` | Column pruning on parquet reads |

---

### Phase 30 Tasks (Advanced Architecture)

| Task | File | Description |
|------|------|-------------|
| 30-1 | `core/types.py` | Standardize transformer family naming |
| 30-2 | `core/constants.py` | Derive from MODEL_CONTRACTS |
| 30-3 | `inference/orchestrator.py` | Move types to core |
| 30-4 | `core/interfaces.py` | Fix circular imports |
| 30-5 | `volatility.py` | Cache computation context |

---

### Phase 31 Tasks (Polish)

| Task | File | Description |
|------|------|-------------|
| 31-1 | `monitor.py:264-265` | Address TODOs |
| 31-2 | Multiple | Extract magic numbers |
| 31-3 | `config/unified.py` | Consolidate defaults |
| 31-4 | `adapters/base.py` | Complete exclusion list |
| 31-5 | `multi_stream.py` | Fix temporal alignment |
| 31-6 | `features/engineer.py` | Define feature DAG |
| 31-7 | `adapters/*.py` | Move common methods |

---

## Verification Commands

### Core Imports
```bash
python -c "from src.core.types import DataRank, ModelFamily; print('OK')"
python -c "from src.core.contracts import get_model_contract; print('OK')"
python -c "from src.data.adapters import get_adapter; print('OK')"
```

### Linting
```bash
ruff check src/
black --check src/
```

### Tests
```bash
pytest tests/ -v
```

---

## Summary Checklist

### Phase 24: Feature Caching
- [x] 24-1: Cache ADX/DI
- [x] 24-2: Cache Microstructure
- [x] 24-3: Combine Supertrend

### Phase 25: Validation
- [x] 25-1: Inter-stage validation (SIMPLIFIED)
- [x] 25-2: Raw data fail-fast
- [x] 25-3: MTF lookahead check
- [x] 25-4: Sentinel validation (DISPROVEN - already done)
- [x] 25-5: Horizon fail-fast

### Phase 26: Type Safety
- [ ] 26-1: Replace Any types
- [ ] 26-2: Fix bare exceptions
- [ ] 26-3: Add return types
- [ ] 26-4: Remove deprecated alias

### Phase 27: Architecture
- [ ] 27-1: Consolidate PredictionResult
- [ ] 27-2: Remove duplicate AdapterResult
- [ ] 27-3: Rename DatasetContract
- [ ] 27-4: Rename ModelContract
- [ ] 27-5: Replace PredictionOutput

### Phase 28-31
See task tables above.

---

**Total Tasks:** 38 across 8 phases

---

*See COMPLETION.md for implementation details after phase completion*
