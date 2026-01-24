# ML Factory Cleanup Tasks - Active Work

**Last Updated:** 2026-01-24
**Status:** Phase 0 Complete | Phase 1 Complete | Phase 2 Complete | Phase 3 Ready
**Phase 0 Impact:** -5,336 lines removed | 7 naming conflicts resolved
**Phase 1 Impact:** +616 lines added | 7 new exceptions | Commit 7f71b52
**Phase 2 Impact:** +958 lines added | 9 files modified | 4D models enabled

---

## Table of Contents
- [Phase 3: 5-Dimension Optuna](#phase-3-5-dimension-optuna) ← NEXT
- [Completed: Phase 2](#completed-phase-2-4d-infrastructure)
- [Completed: Phase 1](#completed-phase-1-contract-enforcement)
- [Completed: Phase 0](#completed-phase-0-deduplication)
- [False Positives](#false-positives)
- [Deferred Items](#deferred-items)
- [Verification Commands](#verification-commands)

---

## Phase 3: 5-Dimension Optuna

**Source:** CLEANUP_PLAN.md Phase 3
**Status:** Ready to Execute
**Target:** Optuna optimizes all 5 dimensions (barriers, features, params, timeframes, hyperparams)
**Estimated Effort:** 5-7 days
**Blocked By:** Phase 2 (COMPLETE)

See `CLEANUP_PLAN.md` for detailed Phase 3 tasks (3A-3F).

---

## Completed: Phase 2 4D Infrastructure

**Status:** ✅ COMPLETE (2026-01-24)
**Commit:** [pending]
**Impact:** +958 lines, 9 files modified, 1 new file

### Tasks

| ID | Task | Status |
|----|------|--------|
| 2A | Create `raw_mtf_store.py` (445 lines) | ✅ |
| 2B | MTF generator saves to store | ✅ |
| 2C | PatchTST/iTransformer → MULTI_TF_4D | ✅ |
| 2D | MultiStreamAdapter + `from_store()` | ✅ |
| 2E | Verify adapter registration | ✅ |
| 2F | Wire UnifiedDataPreparation | ✅ |
| 2G | Add `get_multi_stream_4d()` | ✅ |

### Files Modified

| File | Change |
|------|--------|
| `src/data/store/raw_mtf_store.py` | NEW (+445 lines) |
| `src/data/store/__init__.py` | Exports |
| `src/data/pipeline/stages/mtf/generator.py` | Save to store |
| `src/data/pipeline/stages/mtf/convenience.py` | Config params |
| `src/core/contracts/model_contract.py` | DataRank.MULTI_TF_4D |
| `src/data/adapters/multi_stream.py` | `from_store()` factory |
| `src/data/adapters/__init__.py` | Exports |
| `src/data/adapters/preparation.py` | multi_stream routing |
| `src/core/container.py` | `get_multi_stream_4d()` |

---

## Completed: Phase 1 Contract Enforcement

**Status:** ✅ COMPLETE (2026-01-23)
**Commit:** 7f71b52
**Impact:** +616 lines, 7 new exceptions

### 1A: Make DataContract.validate_dataframe() Raise on Failure - ✅ DONE

**Priority:** CRITICAL
**Effort:** LOW (2 hours)
**Location:** `src/core/contracts/data_contract.py:195-224`
**Blocked By:** None

**Problem:**
`validate_dataframe()` returns `(bool, list[str])` but callers ignore the return value and proceed with invalid data. This allows corrupted data to enter training silently.

**Current Code:**
```python
# src/core/contracts/data_contract.py:195-224
def validate_dataframe(self, df: pd.DataFrame) -> tuple[bool, list[str]]:
    """Validate DataFrame against contract."""
    issues = []

    # Check required columns
    missing = set(self.required_columns) - set(df.columns)
    if missing:
        issues.append(f"Missing columns: {missing}")

    # Check data types
    for col, expected_type in self.column_types.items():
        if col in df.columns and not df[col].dtype == expected_type:
            issues.append(f"Column {col} has wrong type")

    return len(issues) == 0, issues
```

**Fix:**
```python
# src/core/contracts/data_contract.py - Add new exception class at top
class DataContractViolation(Exception):
    """Raised when data violates contract."""
    def __init__(self, issues: list[str]):
        self.issues = issues
        super().__init__(f"Contract violations: {issues}")

# src/core/contracts/data_contract.py - Add new method after validate_dataframe()
def validate_dataframe_strict(self, df: pd.DataFrame) -> None:
    """Validate DataFrame, raising DataContractViolation on failure.

    Use this method when validation failure should block execution.
    """
    is_valid, issues = self.validate_dataframe(df)
    if not is_valid:
        raise DataContractViolation(issues)
```

**Verification:**
- [ ] `DataContractViolation` exception class exists
- [ ] `validate_dataframe_strict()` method exists
- [ ] Method raises on invalid data
- [ ] Method passes silently on valid data
- [ ] Unit test added: `tests/core/contracts/test_data_contract.py`

```bash
# Verification commands
python -c "from src.core.contracts.data_contract import DataContractViolation; print('Exception OK')"
python -c "from src.core.contracts.data_contract import DataContract; print(hasattr(DataContract, 'validate_dataframe_strict'))"
pytest tests/core/contracts/test_data_contract.py -v -k "strict"
```

**Commit:** 7f71b52

---

### 1B: Call ModelContract.validate_data_contract() in Adapter Load Path - ✅ DONE

**Priority:** CRITICAL
**Effort:** LOW (2 hours)
**Location:** `src/data/adapters/base.py` (load method)
**Blocked By:** 1A

**Problem:**
`ModelContract.validate_data_contract()` method exists but is NEVER called. Adapters load data without verifying it matches the model's requirements.

**Current Code:**
```python
# src/data/adapters/base.py - BaseAdapter.load() method
def load(self, path: str, **kwargs) -> AdapterResult:
    """Load data from path."""
    df = pd.read_parquet(path)
    # NO VALIDATION - data goes straight to model
    return self._adapt(df, **kwargs)
```

**Fix:**
```python
# src/data/adapters/base.py - Updated load() method
def load(self, path: str, model_contract: ModelContract = None, **kwargs) -> AdapterResult:
    """Load data from path with optional contract validation.

    Args:
        path: Path to parquet file
        model_contract: If provided, validates data against model requirements
        **kwargs: Additional adapter arguments

    Raises:
        DataContractViolation: If data doesn't match model contract
    """
    df = pd.read_parquet(path)

    # Validate against model contract if provided
    if model_contract is not None:
        data_contract = model_contract.get_data_contract()
        data_contract.validate_dataframe_strict(df)

    return self._adapt(df, **kwargs)
```

**Files to Update:**
| File | Change |
|------|--------|
| `src/data/adapters/base.py` | Add model_contract param to load() |
| `src/data/adapters/tabular.py` | Update load() signature |
| `src/data/adapters/sequence.py` | Update load() signature |
| `src/data/adapters/multi_resolution.py` | Update load() signature |

**Verification:**
- [ ] `load()` accepts `model_contract` parameter
- [ ] Validation called when contract provided
- [ ] No validation when contract is None (backward compat)
- [ ] `DataContractViolation` raised on mismatch
- [ ] All adapter subclasses updated

```bash
# Verification commands
python -c "from src.data.adapters.base import BaseAdapter; import inspect; print('model_contract' in inspect.signature(BaseAdapter.load).parameters)"
pytest tests/data/adapters/test_base.py -v -k "contract"
```

**Commit:** 7f71b52

---

### 1C: Add Pre-Training Validation Hook - ✅ DONE

**Priority:** HIGH
**Effort:** MEDIUM (3 hours)
**Location:** `src/training/orchestrator.py`
**Blocked By:** 1A, 1B

**Problem:**
Training proceeds without validating data integrity. Leakage detection and lookahead audit exist but are never called before training starts.

**Current Code:**
```python
# src/training/orchestrator.py - train() method (approximate)
def train(self, data: TimeSeriesDataContainer, config: TrainerConfig) -> TrainingResult:
    """Train models on data."""
    # NO PRE-TRAINING VALIDATION
    for model_name in config.models:
        model = self._create_model(model_name)
        model.fit(data.X_train, data.y_train)
    ...
```

**Fix:**
```python
# src/training/orchestrator.py - Add validation hook
from src.validation.leakage_detection import LeakageDetector
from src.validation.lookahead_audit import LookaheadAudit
from src.core.contracts.data_contract import DataContractViolation

class PreTrainingValidationError(Exception):
    """Raised when pre-training validation fails."""
    pass

def _pre_training_validation(self, data: TimeSeriesDataContainer, config: TrainerConfig) -> None:
    """Validate data before training. Raises on failure.

    Validates:
    1. Data contract compliance
    2. No data leakage between splits
    3. No lookahead bias in features

    Raises:
        PreTrainingValidationError: If any validation fails
    """
    errors = []

    # 1. Contract validation (if strict mode enabled)
    if config.strict_validation:
        try:
            for model_name in config.models:
                contract = get_model_contract(model_name)
                contract.get_data_contract().validate_dataframe_strict(data.df)
        except DataContractViolation as e:
            errors.append(f"Contract violation: {e}")

    # 2. Leakage detection
    if config.check_leakage:
        detector = LeakageDetector()
        leakage_report = detector.detect(
            train_df=data.train_df,
            val_df=data.val_df,
            test_df=data.test_df
        )
        if leakage_report.has_leakage:
            errors.append(f"Data leakage detected: {leakage_report.summary}")

    # 3. Lookahead audit
    if config.check_lookahead:
        auditor = LookaheadAudit()
        lookahead_report = auditor.audit(data.df)
        if lookahead_report.has_lookahead:
            errors.append(f"Lookahead bias detected: {lookahead_report.summary}")

    if errors:
        raise PreTrainingValidationError("\n".join(errors))

def train(self, data: TimeSeriesDataContainer, config: TrainerConfig) -> TrainingResult:
    """Train models on data."""
    # PRE-TRAINING VALIDATION
    self._pre_training_validation(data, config)

    for model_name in config.models:
        model = self._create_model(model_name)
        model.fit(data.X_train, data.y_train)
    ...
```

**Config Additions:**
```python
# src/config/training.py - Add to TrainerConfig
@dataclass
class TrainerConfig:
    # ... existing fields ...

    # Validation settings
    strict_validation: bool = True  # Enforce data contracts
    check_leakage: bool = True      # Run leakage detection
    check_lookahead: bool = True    # Run lookahead audit
```

**Verification:**
- [ ] `_pre_training_validation()` method exists
- [ ] Called at start of `train()`
- [ ] Raises `PreTrainingValidationError` on failure
- [ ] Config flags control which checks run
- [ ] Training blocked when validation fails

```bash
# Verification commands
python -c "from src.training.orchestrator import UnifiedTrainingOrchestrator; print(hasattr(UnifiedTrainingOrchestrator, '_pre_training_validation'))"
pytest tests/training/test_orchestrator.py -v -k "validation"
```

**Commit:** 7f71b52

---

### 1D: Wire Leakage Detection to Block Training - ✅ DONE

**Priority:** CRITICAL
**Effort:** LOW (2 hours)
**Location:** `src/validation/leakage_detection.py`
**Blocked By:** 1C

**Problem:**
Leakage detection returns a report but callers just log warnings. Detected leakage should raise an exception to block training.

**Current Code:**
```python
# src/validation/leakage_detection.py
class LeakageDetector:
    def detect(self, train_df, val_df, test_df) -> LeakageReport:
        """Detect data leakage between splits."""
        # ... detection logic ...
        return LeakageReport(
            has_leakage=bool(issues),
            issues=issues,
            summary=self._format_summary(issues)
        )
```

**Fix:**
```python
# src/validation/leakage_detection.py - Add exception and blocking mode

class LeakageDetectedError(Exception):
    """Raised when data leakage is detected and blocking mode is enabled."""
    def __init__(self, report: LeakageReport):
        self.report = report
        super().__init__(f"Data leakage detected: {report.summary}")

class LeakageDetector:
    def detect(
        self,
        train_df,
        val_df,
        test_df,
        raise_on_leakage: bool = False
    ) -> LeakageReport:
        """Detect data leakage between splits.

        Args:
            train_df: Training data
            val_df: Validation data
            test_df: Test data
            raise_on_leakage: If True, raise LeakageDetectedError when leakage found

        Returns:
            LeakageReport with detection results

        Raises:
            LeakageDetectedError: If leakage detected and raise_on_leakage=True
        """
        # ... existing detection logic ...
        report = LeakageReport(
            has_leakage=bool(issues),
            issues=issues,
            summary=self._format_summary(issues)
        )

        if report.has_leakage and raise_on_leakage:
            raise LeakageDetectedError(report)

        return report
```

**Verification:**
- [ ] `LeakageDetectedError` exception class exists
- [ ] `detect()` accepts `raise_on_leakage` parameter
- [ ] Exception raised when `raise_on_leakage=True` and leakage found
- [ ] Backward compatible (default `raise_on_leakage=False`)
- [ ] Exception includes full report

```bash
# Verification commands
python -c "from src.validation.leakage_detection import LeakageDetectedError; print('Exception OK')"
python -c "from src.validation.leakage_detection import LeakageDetector; import inspect; print('raise_on_leakage' in inspect.signature(LeakageDetector.detect).parameters)"
pytest tests/validation/test_leakage_detection.py -v -k "blocking"
```

**Commit:** 7f71b52

---

### 1E: Wire Lookahead Audit to Block Training - ✅ DONE

**Priority:** CRITICAL
**Effort:** LOW (2 hours)
**Location:** `src/validation/lookahead_audit.py`
**Blocked By:** 1C

**Problem:**
Lookahead audit returns a report but callers just log warnings. Detected lookahead bias should raise an exception to block training.

**Current Code:**
```python
# src/validation/lookahead_audit.py
class LookaheadAudit:
    def audit(self, df: pd.DataFrame) -> LookaheadReport:
        """Audit DataFrame for lookahead bias."""
        # ... audit logic ...
        return LookaheadReport(
            has_lookahead=bool(issues),
            issues=issues,
            summary=self._format_summary(issues)
        )
```

**Fix:**
```python
# src/validation/lookahead_audit.py - Add exception and blocking mode

class LookaheadBiasError(Exception):
    """Raised when lookahead bias is detected and blocking mode is enabled."""
    def __init__(self, report: LookaheadReport):
        self.report = report
        super().__init__(f"Lookahead bias detected: {report.summary}")

class LookaheadAudit:
    def audit(
        self,
        df: pd.DataFrame,
        raise_on_lookahead: bool = False
    ) -> LookaheadReport:
        """Audit DataFrame for lookahead bias.

        Args:
            df: DataFrame to audit
            raise_on_lookahead: If True, raise LookaheadBiasError when bias found

        Returns:
            LookaheadReport with audit results

        Raises:
            LookaheadBiasError: If lookahead detected and raise_on_lookahead=True
        """
        # ... existing audit logic ...
        report = LookaheadReport(
            has_lookahead=bool(issues),
            issues=issues,
            summary=self._format_summary(issues)
        )

        if report.has_lookahead and raise_on_lookahead:
            raise LookaheadBiasError(report)

        return report
```

**Verification:**
- [ ] `LookaheadBiasError` exception class exists
- [ ] `audit()` accepts `raise_on_lookahead` parameter
- [ ] Exception raised when `raise_on_lookahead=True` and bias found
- [ ] Backward compatible (default `raise_on_lookahead=False`)
- [ ] Exception includes full report

```bash
# Verification commands
python -c "from src.validation.lookahead_audit import LookaheadBiasError; print('Exception OK')"
python -c "from src.validation.lookahead_audit import LookaheadAudit; import inspect; print('raise_on_lookahead' in inspect.signature(LookaheadAudit.audit).parameters)"
pytest tests/validation/test_lookahead_audit.py -v -k "blocking"
```

**Commit:** 7f71b52

---

### 1F: Add Scaler Fit Verification - ✅ DONE

**Priority:** HIGH
**Effort:** LOW (1 hour)
**Location:** `src/data/pipeline/stages/scaling/core.py`
**Blocked By:** None

**Problem:**
Scalers must only be fit on training data to prevent data leakage. Currently no verification that `fit()` is called only on train split.

**Current Code:**
```python
# src/data/pipeline/stages/scaling/core.py
class ScalerStage:
    def fit(self, df: pd.DataFrame) -> None:
        """Fit scaler on data."""
        self.scaler.fit(df[self.feature_columns])

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform data using fitted scaler."""
        df[self.feature_columns] = self.scaler.transform(df[self.feature_columns])
        return df
```

**Fix:**
```python
# src/data/pipeline/stages/scaling/core.py - Add split verification

class ScalerFitError(Exception):
    """Raised when scaler fit is called on non-training data."""
    pass

class ScalerStage:
    def __init__(self, ...):
        # ... existing init ...
        self._fitted_on_split: str | None = None

    def fit(self, df: pd.DataFrame, split: str = "train") -> None:
        """Fit scaler on data.

        Args:
            df: DataFrame to fit on
            split: Name of split (must be "train" to prevent leakage)

        Raises:
            ScalerFitError: If split is not "train"
        """
        if split != "train":
            raise ScalerFitError(
                f"Scaler must only be fit on 'train' split, got '{split}'. "
                "Fitting on val/test data causes data leakage."
            )

        self.scaler.fit(df[self.feature_columns])
        self._fitted_on_split = split

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform data using fitted scaler."""
        if self._fitted_on_split is None:
            raise RuntimeError("Scaler must be fit before transform")

        df[self.feature_columns] = self.scaler.transform(df[self.feature_columns])
        return df

    @property
    def fitted_on_split(self) -> str | None:
        """Return the split this scaler was fitted on."""
        return self._fitted_on_split
```

**Verification:**
- [ ] `ScalerFitError` exception class exists
- [ ] `fit()` requires `split` parameter
- [ ] Exception raised if `split != "train"`
- [ ] `_fitted_on_split` tracks what split was used
- [ ] `transform()` fails if not fitted

```bash
# Verification commands
python -c "from src.data.pipeline.stages.scaling.core import ScalerFitError; print('Exception OK')"
pytest tests/data/pipeline/stages/test_scaling.py -v -k "fit"
```

**Commit:** 7f71b52

---

### 1G: Fix Chronological Splits Validation - ✅ DONE

**Priority:** MEDIUM
**Effort:** LOW (1 hour)
**Location:** `src/data/pipeline/stages/splits/core.py`
**Blocked By:** None

**Problem:**
Split computation assumes data is chronologically sorted but doesn't verify this. If data is unsorted, splits will be invalid (mixing future and past data).

**Current Code:**
```python
# src/data/pipeline/stages/splits/core.py
class ChronologicalSplitter:
    def split(self, df: pd.DataFrame, train_ratio: float, val_ratio: float) -> SplitResult:
        """Split data chronologically."""
        n = len(df)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))

        # ASSUMES df is sorted by time - no verification!
        return SplitResult(
            train=df.iloc[:train_end],
            val=df.iloc[train_end:val_end],
            test=df.iloc[val_end:]
        )
```

**Fix:**
```python
# src/data/pipeline/stages/splits/core.py - Add sort verification

class ChronologicalSortError(Exception):
    """Raised when data is not chronologically sorted."""
    pass

class ChronologicalSplitter:
    def __init__(self, timestamp_column: str = "timestamp"):
        self.timestamp_column = timestamp_column

    def _verify_chronological_order(self, df: pd.DataFrame) -> None:
        """Verify DataFrame is sorted chronologically.

        Raises:
            ChronologicalSortError: If data is not sorted by timestamp
        """
        if self.timestamp_column not in df.columns:
            raise ChronologicalSortError(
                f"Timestamp column '{self.timestamp_column}' not found in DataFrame"
            )

        timestamps = df[self.timestamp_column]
        if not timestamps.is_monotonic_increasing:
            # Find first out-of-order index for helpful error message
            for i in range(1, len(timestamps)):
                if timestamps.iloc[i] < timestamps.iloc[i-1]:
                    raise ChronologicalSortError(
                        f"Data not chronologically sorted. "
                        f"Row {i} ({timestamps.iloc[i]}) < Row {i-1} ({timestamps.iloc[i-1]}). "
                        f"Sort by '{self.timestamp_column}' before splitting."
                    )

    def split(self, df: pd.DataFrame, train_ratio: float, val_ratio: float) -> SplitResult:
        """Split data chronologically.

        Args:
            df: DataFrame to split (must be sorted by timestamp)
            train_ratio: Fraction for training
            val_ratio: Fraction for validation

        Returns:
            SplitResult with train, val, test DataFrames

        Raises:
            ChronologicalSortError: If data is not sorted
        """
        # VERIFY SORT ORDER BEFORE SPLITTING
        self._verify_chronological_order(df)

        n = len(df)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))

        return SplitResult(
            train=df.iloc[:train_end],
            val=df.iloc[train_end:val_end],
            test=df.iloc[val_end:]
        )
```

**Verification:**
- [ ] `ChronologicalSortError` exception class exists
- [ ] `_verify_chronological_order()` method exists
- [ ] Verification called before computing indices
- [ ] Helpful error message with first violation
- [ ] Works with different timestamp column names

```bash
# Verification commands
python -c "from src.data.pipeline.stages.splits.core import ChronologicalSortError; print('Exception OK')"
pytest tests/data/pipeline/stages/test_splits.py -v -k "chronological"
```

**Commit:** 7f71b52

---

## Completed: Phase 0 Deduplication

**Status:** ✅ COMPLETE (2026-01-23)
**Lines Removed:** ~5,336
**Full Details:** `X ( IN PROGRESS DOCS) X/PHASE_0_COMPLETION.md`

| Task | Description | Lines | Status | Commit |
|------|-------------|-------|--------|--------|
| 0A | DataRank → `src/core/types.py` | -15 | ✅ Done | 3262996 |
| 0B | ModelFamily + TRANSFORMER → `src/core/types.py` | -30 | ✅ Done | 3262996 |
| 0C | Delete `src/coordination/` | -1,166 | ✅ Done | 3262996 |
| 0D | Delete `src/feature_selection/` | -3,508 | ✅ Done | 3262996 |
| 0E | MultiResolution4DAdapter → `src/data/adapters/` | -617 | ✅ Done | 3262996 |
| 0F | AdapterResult compatibility properties | ±0 | ✅ Done | 3262996 |
| 0G | Rename validation DataContract → OHLCVValidationSchema | ±0 | ✅ Done | 3262996 |

**Verification Results:**
- 3 parallel review agents + Task Agent 7 all APPROVED
- All imports resolve correctly
- Single source of truth for all enums

---

## False Positives

Items investigated but determined to NOT need fixing:

| Item | Location | Reason OK |
|------|----------|-----------|
| ModelFamilyDefaults | `feature_selection/config.py` | Different class (dataclass, not enum) |
| Multiple FeatureSelector | `data/features/`, `models/training/` | Different contexts, both valid |
| DatasetContract vs DataContract | `core/data_contract.py` vs `core/contracts/` | Intentionally different purposes |
| Dual AdapterResult | `adapters/base.py`, `adapters/result.py` | Circular import prevention - bidirectional properties added |

---

## Deferred Items

| Item | Original Phase | Reason Deferred | Target Phase |
|------|----------------|-----------------|--------------|
| Scaler Consolidation | 0 | Requires pipeline integration understanding | Phase 2+ |
| Validator Pattern Standardization | 0 | Multiple valid patterns for different contexts | Phase 3+ |
| Pipeline Stage Refactoring | 0 | Large architectural change | Phase 4+ |

---

## Verification Commands

### Phase 1 Pre-Flight Check

```bash
# Verify Phase 0 still works
python -c "from src.core.types import DataRank, ModelFamily; print('Types OK')"
python -c "from src.core.coordination import TimeframeCoordinator; print('Coordination OK')"
python -c "from src.optimization.feature_selection import FeatureSelectionResult; print('Feature Selection OK')"
python -c "from src.data.adapters import MultiResolution4DAdapter; print('Adapters OK')"
```

### Phase 1 Completion Check

```bash
# All new exceptions exist
python -c "from src.core.contracts.data_contract import DataContractViolation; print('1A OK')"
python -c "from src.validation.leakage_detection import LeakageDetectedError; print('1D OK')"
python -c "from src.validation.lookahead_audit import LookaheadBiasError; print('1E OK')"
python -c "from src.data.pipeline.stages.scaling.core import ScalerFitError; print('1F OK')"
python -c "from src.data.pipeline.stages.splits.core import ChronologicalSortError; print('1G OK')"

# Run Phase 1 tests
pytest tests/core/contracts/ -v
pytest tests/validation/ -v
pytest tests/data/pipeline/stages/ -v
pytest tests/training/ -v -k "validation"
```

### Full Suite

```bash
ruff check src/
black --check src/
pytest tests/ -x --tb=short
```

---

## Change Log

| Date | Phase | Task | Impact | Commit |
|------|-------|------|--------|--------|
| 2026-01-24 | 2 | All (2A-2G) | +958 lines, 4D enabled | [pending] |
| 2026-01-23 | 1 | All (1A-1G) | +616 lines, 7 exceptions | 7f71b52 |
| 2026-01-23 | 0 | All (0A-0G) | -5,336 lines | 3262996 |

---

*Next: Execute Phase 3 tasks (5-Dimension Optuna)*
*Full cleanup plan: `CLEANUP_PLAN.md`*
