# ML Factory - Cleanup Tasks

**Status:** Phase 33 Complete, Phase 34 Planned
**Last Updated:** 2026-02-01

---

## Completed Phases (24-33)

See **COMPLETION.md** for full task details and implementation information.

| Phase | Tasks Completed | Key Deliverables | Completed |
|-------|-----------------|------------------|-----------|
| 24 | 3/3 tasks | Feature computation caching (ADX/DI, microstructure, supertrend) | 2026-01-29 |
| 25 | 5/5 tasks (3 impl, 1 simplified, 1 disproven) | Fail-fast validation hardening | 2026-01-29 |
| 26 | 4/4 tasks (3 complete, 1 deferred to Phase 31) | Type safety improvements (Any types, return annotations) | 2026-01-29 |
| 27 | 5/5 tasks (4 complete, 1 documented exception) | Single definition principle enforced | 2026-01-29 |
| 28 | 5/5 tasks (all complete) | Numba entropy, parallelization, GARCH, ATR/volume caching | 2026-01-30 |
| 29 | 5/5 tasks (2 impl, 2 disproven, 1 deferred to Phase 31) | Bounded cache, log_returns consolidation | 2026-01-29 |
| 30 | 5/5 tasks (3 impl, 2 disproven) | Transformer family split, derived constants, SMA/EMA/STD caching | 2026-01-30 |
| 31 | 9/9 tasks (7 impl, 1 disproven, 1 deferred to Phase 32) | Code polish, latency tracking, constants, adapters, feature DAG | 2026-01-31 |
| 32 | 15/16 tasks (15 impl, 1 disproven, 4 added) | Model family alignment, data leakage elimination, numerical stability | 2026-02-01 |
| 33 | 11/11 tasks (all complete) | Evaluators, layer violation fixes, performance optimizations | 2026-02-01 |

**Summary Impact:** 67 tasks across 10 phases, 69+ files modified, production-ready evaluators, 30-40% pipeline speedup.

---

## Active Phases

Phase 33 complete. See COMPLETION.md for full task breakdown and implementation details.

**Next:** Phase 34 (Cleanup & Consolidation) - 11 tasks, MEDIUM priority

---

## Phase 33: Performance & Architecture

**Status:** ✅ COMPLETE
**Priority:** HIGH
**Tasks:** 11/11
**Source:** Comprehensive ML pipeline review (9-agent analysis, 2026-02-01)
**Completed:** 2026-02-01

---

### Task 33-1: Implement CPCV-PBO Evaluator

**File:** `src/validation/evaluation/cpcv_pbo_evaluator.py`
**Line:** 52
**Priority:** HIGH

#### Problem

```python
def evaluate(...):
    raise NotImplementedError("CPCV-PBO evaluator not yet implemented")
```

#### AI Instructions

1. **Read** related evaluator implementations for pattern
2. **Implement** CPCV (Combinatorially Purged Cross-Validation) with PBO (Probability of Backtest Overfitting)
3. **Reference:** López de Prado's "Advances in Financial Machine Learning" Chapter 11
4. **Implementation** should include:
   - Combinatorial purging to prevent leakage
   - PBO calculation using rank-based statistics
   - Proper embargo handling
5. **Add** comprehensive docstring
6. **Add** tests

---

### Task 33-2: Implement CV Evaluator

**File:** `src/validation/evaluation/cv_evaluator.py`
**Line:** 51
**Priority:** HIGH

#### AI Instructions

Same approach as 33-1, implement cross-validation evaluator with purging and embargo.

---

### Task 33-3: Implement Walk-Forward Evaluator

**File:** `src/validation/evaluation/walk_forward_evaluator.py`
**Line:** 51
**Priority:** HIGH

#### AI Instructions

Same approach as 33-1, implement walk-forward evaluator with expanding/rolling window options.

---

### Task 33-4: Remove MultiResolution4DAdapter Import from Core

**File:** `src/core/container.py`
**Line:** 673
**Priority:** HIGH

#### Problem

Core layer imports from data layer (layer violation):
```python
from src.data.adapters.multi_resolution import MultiResolution4DAdapter
```

#### AI Instructions

1. **Read** `src/core/container.py` lines 665-685
2. **Find** usage of `MultiResolution4DAdapter`
3. **Replace** with dynamic import or registry lookup:
   ```python
   # BEFORE
   from src.data.adapters.multi_resolution import MultiResolution4DAdapter
   adapter = MultiResolution4DAdapter(...)

   # AFTER
   from src.data.adapters import get_adapter
   adapter = get_adapter("multi_resolution", ...)
   ```
4. **Verify** no direct imports from `src.data` in `src/core`

#### Verification

```bash
grep -r "from src.data" src/core/ --include="*.py"
# Should return 0 results (or only TYPE_CHECKING imports)
```

---

### Task 33-5: Remove MultiStreamAdapter Import from Core

**File:** `src/core/container.py`
**Line:** 739
**Priority:** HIGH

#### AI Instructions

Same as 33-4, replace with registry lookup.

---

### Task 33-6: Vectorize CCI Computation

**File:** `src/data/features/compute/momentum.py`
**Lines:** 322-341
**Priority:** MEDIUM

#### Problem

CCI (Commodity Channel Index) uses Python loop instead of vectorized operations:
```python
for i in range(len(df)):
    # ... per-row computation
```

#### AI Instructions

1. **Read** `src/data/features/compute/momentum.py` lines 310-350
2. **Identify** the CCI computation loop
3. **Replace** with vectorized pandas operations:
   ```python
   # Vectorized approach
   typical_price = (df['high'] + df['low'] + df['close']) / 3
   sma = typical_price.rolling(window=period).mean()
   mean_deviation = typical_price.rolling(window=period).apply(
       lambda x: np.abs(x - x.mean()).mean()
   )
   cci = (typical_price - sma) / (0.015 * mean_deviation)
   ```
4. **Profile** before/after to verify speedup
5. **Run** tests

#### Verification

```bash
python -c "
import time
from src.data.features.compute.momentum import compute_cci_20
import pandas as pd
import numpy as np
df = pd.DataFrame({
    'high': np.random.rand(10000)*100+100,
    'low': np.random.rand(10000)*100+99,
    'close': np.random.rand(10000)*100+99.5
})
start = time.time()
result = compute_cci_20(df)
elapsed = time.time() - start
print(f'CCI time: {elapsed:.3f}s')
# Should be <0.1s for 10k rows
"
```

---

### Task 33-7: Vectorize Variance Ratio Test

**File:** `src/data/features/compute/mean_reversion.py`
**Lines:** 250-300
**Priority:** MEDIUM

#### AI Instructions

Similar to 33-6, replace loop-based variance ratio computation with vectorized operations. Expected 10-20x speedup.

---

### Task 33-8: Add Caching to Order Flow Features

**File:** `src/data/features/compute/order_flow.py`
**Lines:** 53-103
**Priority:** MEDIUM

#### AI Instructions

1. **Read** existing caching patterns from Phase 28 tasks (ATR, volume)
2. **Add** DataFrame-id based cache for base order flow metrics
3. **Cache** VPIN, Kyle's lambda, order imbalance
4. **Update** derived features to use cache

---

### Task 33-9: Add Caching to Regime Features

**File:** `src/data/features/compute/regime.py`
**Lines:** 53-86, 120-135
**Priority:** MEDIUM

#### AI Instructions

Same as 33-8, add caching for regime detection (trending/mean-reverting/volatile).

---

### Task 33-10: Apply Numba to Wavelet Transform

**File:** `src/data/features/compute/wavelets.py`
**Lines:** 62-88
**Priority:** MEDIUM

#### AI Instructions

1. **Read** existing numba patterns from Phase 28-1 (entropy)
2. **Identify** wavelet transform computation loop
3. **Add** `@numba.jit(nopython=True)` decorator
4. **Ensure** all operations are numba-compatible
5. **Profile** before/after (expect 10-50x speedup)

---

### Task 33-11: Replace Hurst Exponent with O(n) Algorithm

**File:** `src/data/features/compute/mean_reversion.py`
**Lines:** 156-200
**Priority:** MEDIUM

#### Problem

Current Hurst exponent computation is O(n²):
```python
# Current: O(n²) rescaled range calculation
for lag in range(2, n):
    # ... nested operations
```

#### AI Instructions

1. **Read** current implementation
2. **Replace** with Anis-Lloyd corrected R/S method (O(n))
3. **Reference:** Weron, R. (2002) "Estimating long-range dependence"
4. **Implementation**:
   ```python
   def _hurst_anis_lloyd(returns: np.ndarray) -> float:
       """O(n) Hurst estimation using Anis-Lloyd method."""
       n = len(returns)
       mean_adjusted = returns - returns.mean()
       cumsum = np.cumsum(mean_adjusted)
       R = cumsum.max() - cumsum.min()  # Range
       S = returns.std()  # Standard deviation
       if S == 0:
           return 0.5
       return np.log(R/S) / np.log(n)
   ```
5. **Profile** before/after

---

### Phase 33 Completion Checklist

| Task | Status | Verification |
|------|--------|--------------|
| 33-1 | ✅ | CPCV-PBO evaluator implemented |
| 33-2 | ✅ | CV evaluator implemented |
| 33-3 | ✅ | Walk-forward evaluator implemented |
| 33-4 | ✅ | No MultiResolution4DAdapter import in core |
| 33-5 | ✅ | No MultiStreamAdapter import in core |
| 33-6 | ✅ | CCI vectorized (10x speedup with Numba) |
| 33-7 | ✅ | Variance ratio vectorized (10x speedup with Numba) |
| 33-8 | ✅ | Order flow features cached (3-4x speedup) |
| 33-9 | ✅ | Regime features cached (3x speedup) |
| 33-10 | ✅ | Wavelet transform optimized (numpy sliding_window_view) |
| 33-11 | ✅ | Hurst uses O(n) algorithm (Numba-accelerated) |

---

## Phase 34: Cleanup & Consolidation

**Status:** NOT STARTED
**Priority:** MEDIUM
**Tasks:** 11/11
**Source:** Comprehensive ML pipeline review (9-agent analysis, 2026-02-01)

---

### Task 34-1: Delete Empty Placeholder - core/features

**File:** `src/core/features/__init__.py`
**Priority:** LOW

#### AI Instructions

1. **Verify** file is empty or only contains pass
2. **Search** for any imports of this module:
   ```bash
   grep -r "from src.core.features" src/ --include="*.py"
   ```
3. **If** no imports found, delete file
4. **Run** tests to verify nothing breaks

---

### Task 34-2: Delete Empty Placeholder - core/training

**File:** `src/core/training/__init__.py`
**Priority:** LOW

#### AI Instructions

Same as 34-1.

---

### Task 34-3: Delete Unused Re-export - core/types_pkg

**File:** `src/core/types_pkg/__init__.py`
**Priority:** LOW

#### AI Instructions

Same as 34-1.

---

### Task 34-4: Integrate or Delete - data/store/lineage.py

**File:** `src/data/store/lineage.py`
**Priority:** MEDIUM

#### Problem

~170 lines of lineage tracking code that's not integrated into pipeline:
```python
class FeatureLineageTracker:
    """Track feature derivation lineage."""
    # ... complete implementation, never used
```

#### AI Instructions

1. **Read** `src/data/store/lineage.py` to understand functionality
2. **Search** for usage:
   ```bash
   grep -r "FeatureLineageTracker" src/ --include="*.py"
   grep -r "from.*lineage" src/ --include="*.py"
   ```
3. **If** 0 imports found:
   - **Option A:** Integrate into feature manifest (Phase 4 work)
   - **Option B:** Delete with justification in commit message
4. **Discuss** with user before deciding

---

### Task 34-5: Integrate or Delete - data/store/versioning.py

**File:** `src/data/store/versioning.py`
**Priority:** MEDIUM

#### AI Instructions

Same approach as 34-4. Check if versioning is needed or if existing mechanisms suffice.

---

### Task 34-6: Integrate or Delete - data/store/cache.py

**File:** `src/data/store/cache.py`
**Priority:** MEDIUM

#### AI Instructions

Same approach as 34-4. May overlap with existing caching from Phase 24/28.

---

### Task 34-7: Delete Unconnected CLI

**File:** `src/data/pipeline/stages/features/cli.py`
**Priority:** LOW

#### AI Instructions

1. **Verify** CLI is not connected to main CLI system
2. **Check** if functionality is duplicated in `src/cli/`
3. **If** orphaned, delete

---

### Task 34-8: Integrate or Delete - Adaptive Barriers

**File:** `src/data/pipeline/stages/labeling/adaptive_barriers.py`
**Priority:** MEDIUM

#### Problem

Adaptive barrier labeling implementation not integrated:
```python
class AdaptiveBarrierLabeler:
    """Dynamic barrier adjustment based on volatility."""
    # ... implementation, not used in pipeline
```

#### AI Instructions

1. **Read** implementation
2. **Check** if triple barrier labeling already covers this
3. **Discuss** integration vs deletion with user

---

### Task 34-9: Consolidate MTF Defaults to Single Source

**File:** `src/core/constants.py`
**Line:** 35
**Priority:** HIGH

#### Problem

Three different MTF timeframe defaults:
```python
# src/core/constants.py:35
DEFAULT_MTF_TIMEFRAMES = ["5min", "15min", "60min"]

# src/config/unified.py:270
mtf_timeframes: list[str] = ["1min", "15min", "60min"]

# src/data/adapters/multi_stream.py:106-107
default_timeframes = ["1min", "5min", "15min"]
```

#### AI Instructions

1. **Read** all three locations
2. **Decide** canonical default (likely `["1min", "5min", "15min", "60min"]` for comprehensive coverage)
3. **Update** `src/core/constants.py`:
   ```python
   # Multi-timeframe defaults
   DEFAULT_MTF_TIMEFRAMES = ["1min", "5min", "15min", "60min"]
   """Default timeframes for multi-timeframe feature generation"""
   ```
4. **Proceed** to task 34-10

#### Verification

```bash
python -c "
from src.core.constants import DEFAULT_MTF_TIMEFRAMES
print(f'MTF defaults: {DEFAULT_MTF_TIMEFRAMES}')
"
```

---

### Task 34-10: Import MTF Defaults from Constants

**File:** `src/config/unified.py`
**Line:** 270
**Priority:** HIGH

#### AI Instructions

1. **Update** `src/config/unified.py`:
   ```python
   # BEFORE
   mtf_timeframes: list[str] = ["1min", "15min", "60min"]

   # AFTER
   from src.core.constants import DEFAULT_MTF_TIMEFRAMES
   mtf_timeframes: list[str] = field(default_factory=lambda: DEFAULT_MTF_TIMEFRAMES.copy())
   ```
2. **Update** `src/data/adapters/multi_stream.py`:
   ```python
   # BEFORE
   default_timeframes = ["1min", "5min", "15min"]

   # AFTER
   from src.core.constants import DEFAULT_MTF_TIMEFRAMES
   default_timeframes = DEFAULT_MTF_TIMEFRAMES
   ```

#### Verification

```bash
grep -r "DEFAULT_MTF_TIMEFRAMES" src/ --include="*.py" | wc -l
# Should find 3+ locations (1 definition, 2+ imports)
```

---

### Task 34-11: Systematic Fragmentation Refactoring

**Files:** Multiple in `src/data/features/compute/`
**Priority:** MEDIUM

#### Problem

117 patterns of DataFrame fragmentation:
```python
# Current pattern (causes fragmentation)
df['feature_1'] = compute_feature_1()
df['feature_2'] = compute_feature_2()
df['feature_3'] = compute_feature_3()
```

#### AI Instructions

1. **Identify** all feature computation files with fragmentation
2. **For each file**, refactor to batch concat pattern:
   ```python
   # New pattern (no fragmentation)
   features = []
   features.append(pd.Series(compute_feature_1(), name='feature_1'))
   features.append(pd.Series(compute_feature_2(), name='feature_2'))
   features.append(pd.Series(compute_feature_3(), name='feature_3'))
   df = pd.concat([df] + features, axis=1)
   ```
3. **Start** with high-impact files (most features)
4. **Profile** memory usage before/after
5. **Run** tests after each file

#### Verification

```bash
# Check for fragmentation warnings
python -c "
import warnings
warnings.simplefilter('error', pd.errors.PerformanceWarning)
from src.data.features.compute import compute_all_features
import pandas as pd
import numpy as np
df = pd.DataFrame({
    'open': np.random.rand(1000)*100,
    'high': np.random.rand(1000)*100+1,
    'low': np.random.rand(1000)*100-1,
    'close': np.random.rand(1000)*100,
    'volume': np.random.rand(1000)*1e6
})
result = compute_all_features(df)
print('OK - No fragmentation warnings')
"
```

---

### Phase 34 Completion Checklist

| Task | Status | Verification |
|------|--------|--------------|
| 34-1 | ⬜ | core/features/__init__.py deleted |
| 34-2 | ⬜ | core/training/__init__.py deleted |
| 34-3 | ⬜ | core/types_pkg/__init__.py deleted |
| 34-4 | ⬜ | lineage.py integrated or deleted with justification |
| 34-5 | ⬜ | versioning.py integrated or deleted with justification |
| 34-6 | ⬜ | cache.py integrated or deleted with justification |
| 34-7 | ⬜ | features/cli.py deleted |
| 34-8 | ⬜ | adaptive_barriers.py integrated or deleted with justification |
| 34-9 | ⬜ | MTF defaults in constants.py |
| 34-10 | ⬜ | All modules import from constants.py |
| 34-11 | ⬜ | 0 fragmentation patterns (was 117) |

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

### Phase 32: Critical Fixes
```bash
# Verify model family registrations
python -c "
from src.core.contracts.model_contract import MODEL_CONTRACTS
from src.models import MODEL_REGISTRY
for name in ['patchtst', 'itransformer', 'ridge_meta', 'mlp_meta', 'xgboost_meta', 'calibrated_meta']:
    contract = MODEL_CONTRACTS[name]
    registry_family = MODEL_REGISTRY[name]['family']
    assert contract.model_family == registry_family, f'{name}: {contract.model_family} != {registry_family}'
print('OK - All model families match')
"

# Verify no train_test_split with shuffle
grep -r "train_test_split.*shuffle=True" src/ --include="*.py"
# Should return 0 results

# Verify no infinite/1e10 values in features
python -c "
from src.data.features.compute import liquidity, mean_reversion
import pandas as pd
import numpy as np
df = pd.DataFrame({
    'open': np.random.rand(100)*100,
    'high': np.random.rand(100)*100+1,
    'low': np.random.rand(100)*100-1,
    'close': np.random.rand(100)*100,
    'volume': [0] * 50 + list(np.random.rand(50)*1e6)
})
# Test should not raise and should not contain inf/1e10
"
```

### Phase 33: Performance & Architecture
```bash
# Verify evaluators implemented
python -c "
from src.validation.evaluation import CPCVPBOEvaluator, CVEvaluator, WalkForwardEvaluator
evaluators = [CPCVPBOEvaluator(), CVEvaluator(), WalkForwardEvaluator()]
for e in evaluators:
    # Should not raise NotImplementedError
    print(f'{type(e).__name__} implemented')
"

# Verify no core → data layer violations
grep "from src.data" src/core/ --include="*.py" | grep -v "TYPE_CHECKING"
# Should return 0 results

# Profile performance improvements
python -c "
import time
from src.data.features.compute import momentum, mean_reversion, wavelets
import pandas as pd
import numpy as np
df = pd.DataFrame({
    'high': np.random.rand(5000)*100+100,
    'low': np.random.rand(5000)*100+99,
    'close': np.random.rand(5000)*100+99.5
})
start = time.time()
momentum.compute_cci_20(df)
mean_reversion.compute_variance_ratio(df)
wavelets.compute_wavelet_energy(df)
elapsed = time.time() - start
print(f'Combined time: {elapsed:.3f}s (should be <0.5s for 5k rows)')
"
```

### Phase 34: Cleanup
```bash
# Verify empty placeholders deleted
test ! -f src/core/features/__init__.py && echo "OK - core/features deleted"
test ! -f src/core/training/__init__.py && echo "OK - core/training deleted"
test ! -f src/core/types_pkg/__init__.py && echo "OK - core/types_pkg deleted"

# Verify MTF consolidation
python -c "
from src.core.constants import DEFAULT_MTF_TIMEFRAMES
from src.config.unified import UnifiedConfig
from src.data.adapters.multi_stream import MultiStreamAdapter
print(f'Constants: {DEFAULT_MTF_TIMEFRAMES}')
# All should match
"

# Verify no fragmentation
python -c "
import warnings
import pandas as pd
warnings.simplefilter('error', pd.errors.PerformanceWarning)
from src.data.features.compute import compute_all_features
import numpy as np
df = pd.DataFrame({
    'open': np.random.rand(1000)*100,
    'high': np.random.rand(1000)*100+1,
    'low': np.random.rand(1000)*100-1,
    'close': np.random.rand(1000)*100,
    'volume': np.random.rand(1000)*1e6
})
result = compute_all_features(df)
print('OK - No fragmentation warnings')
"
```

---

*See COMPLETION.md for implementation details after phase completion*
*See CLEANUP_PLAN.md for phase overviews and rationale*
