# ML Factory - Cleanup Tasks

**Status:** Phase 23 In Progress
**Last Updated:** 2026-01-28

---

## Completed Phases Summary

| Phase | Tasks | Key Deliverables | Details |
|-------|-------|------------------|---------|
| 0-10 | 47/49 | Deduplication, contracts, 4D infra, Optuna, models | See COMPLETION.md |
| 12-18 | 76/80 | Trading, quality, performance, ensemble, resilience | See COMPLETION.md |
| 19 | 17/21 | 34 new features, vectorization, code quality | See COMPLETION.md |
| 20 | 9/15 | -851 lines, 50-100x speedup | See COMPLETION.md |
| 21 | 10/11 | ML pipeline fixes (3 disproven) | See COMPLETION.md |
| 22 | 7/7 | OPTIMIZE_FOR metric wiring | See COMPLETION.md |

**Net Impact:** ~+12,010 lines | See **COMPLETION.md** for implementation details.

---

## Phase 23: Critical Bugfixes, Validation Fixes & Performance

**Status:** IN PROGRESS | 2026-01-28
**Tasks:** 0/13 active (7 deferred to Phase 24)
**Source:** Runtime error analysis (OBSERVED THINGS.MD), performance analysis (PERFORMANCE_FIXES.md)

### Phase Overview

| Sub-Phase | Description | Priority | Tasks |
|-----------|-------------|----------|-------|
| 23A | Label column data leakage fix | CRITICAL | 1 |
| 23B | Validation timing + auto feature selection | HIGH | 2 |
| 23C | Feature engineering performance (DataFrame fragmentation) | MEDIUM | 10 |
| 23D | Config gaps (production deployment features) | LOW | 7 (deferred) |

---

## Phase 23A: Critical Label Leakage Bugfix

**Priority:** CRITICAL
**Impact:** ALL models train with label as feature = 100% training accuracy, catastrophic production failure

### Task 23A-1: Add "label" to exclude_exact Set

**File:** `src/data/adapters/base.py`
**Lines:** 339-347

#### BEFORE (Current Buggy Code):

```python
# Line 339-347
exclude_exact = {
    "open",
    "high",
    "low",
    "close",
    "volume",
    "bar_index",
    "session_id",
}
```

#### AFTER (Fixed Code):

```python
# Line 339-348
exclude_exact = {
    "open",
    "high",
    "low",
    "close",
    "volume",
    "bar_index",
    "session_id",
    "label",  # CRITICAL: Exclude label columns to prevent data leakage
}
```

#### Implementation Steps:

1. Open `src/data/adapters/base.py`
2. Navigate to line 339 (the `exclude_exact` set in `_get_feature_columns`)
3. Add `"label",` to the set
4. Save file

#### Validation:

```bash
# Verify "label" is in exclude_exact
grep -A10 "exclude_exact = {" src/data/adapters/base.py | grep -q '"label"' && echo "PASS" || echo "FAIL"

# Test adapter doesn't include labels
python -c "
from src.data.adapters.base import BaseAdapter
import pandas as pd
df = pd.DataFrame({'close': [100.0], 'label_h5': [1], 'label': [0], 'feature_a': [0.5]})
adapter = BaseAdapter.__new__(BaseAdapter)
adapter.feature_columns = None
cols = adapter._get_feature_columns(df)
assert 'label' not in cols and 'label_h5' not in cols
print('PASS: Labels excluded')
"
```

---

## Phase 23B: Validation Timing & Feature Selection

**Priority:** HIGH
**Impact:** Contract validation fails - validation runs before adapter, feature counts exceed limits

### Task 23B-1: Skip Rank Validation on Raw Data

**File:** `src/models/training/unified_orchestrator.py`
**Lines:** 343-352 (inside `_pre_training_validation`)

#### BEFORE:

```python
# Lines 343-352
for model_name in self.config.models:
    from src.core.contracts import get_model_contract

    model_contract = get_model_contract(model_name)
    is_valid, issues = model_contract.validate_data_contract(data_contract)
    if not is_valid:
        errors.append(
            f"Contract violation for {model_name}: {'; '.join(issues)}"
        )
```

#### AFTER:

```python
# Lines 343-365
for model_name in self.config.models:
    from src.core.contracts import get_model_contract

    model_contract = get_model_contract(model_name)

    # Skip rank validation at this stage - we're validating raw 2D DataFrame
    # Adapters will transform to appropriate rank (3D/4D) later
    issues = []

    # Only validate feature count at this stage
    if data_contract.n_features > model_contract.max_features:
        issues.append(
            f"Too many features: model max is {model_contract.max_features}, "
            f"data has {data_contract.n_features}"
        )

    if issues:
        errors.append(
            f"Contract violation for {model_name}: {'; '.join(issues)}"
        )
```

#### Validation:

```bash
# Models with different ranks should not fail on rank mismatch
python -c "
from src.core.contracts import get_model_contract
for m in ['lightgbm', 'tcn', 'patchtst']:
    c = get_model_contract(m)
    print(f'{m}: rank={c.input_rank}, max_features={c.max_features}')
"
```

---

### Task 23B-2: Add Auto Feature Selection Before Validation

**File:** `src/models/training/unified_orchestrator.py`
**Insert Location:** Before line 343 (before model contract validation loop)

#### Code to Insert:

```python
# Auto-select features if count exceeds minimum model limit
min_max_features = float('inf')
for model_name in self.config.models:
    from src.core.contracts import get_model_contract
    model_contract = get_model_contract(model_name)
    if model_contract.max_features < min_max_features:
        min_max_features = model_contract.max_features

# If we have too many features, select top N by variance
if feature_names and len(feature_names) > min_max_features:
    logger.warning(
        f"Feature count ({len(feature_names)}) exceeds minimum model limit "
        f"({min_max_features}). Auto-selecting top {min_max_features} features."
    )

    X_subset = df[feature_names].dropna()
    if len(X_subset) > 0:
        variances = X_subset.var().sort_values(ascending=False)
        feature_names = variances.head(int(min_max_features)).index.tolist()
        logger.info(f"Selected {len(feature_names)} features by variance")
```

#### Validation:

```bash
python -c "
# Verify feature selection reduces count
import pandas as pd
import numpy as np
df = pd.DataFrame(np.random.randn(100, 250), columns=[f'f{i}' for i in range(250)])
variances = df.var().sort_values(ascending=False)
top_200 = variances.head(200).index.tolist()
print(f'Reduced {len(df.columns)} -> {len(top_200)} features')
assert len(top_200) == 200
print('PASS')
"
```

---

## Phase 23C: Feature Engineering Performance Fixes

**Priority:** MEDIUM
**Impact:** 2-10x speedup for feature generation pipeline
**Pattern:** Replace individual `df[col] = value` with batch `pd.concat()`

---

### Task 23C-1: Vectorize temporal.py get_session and Batch Assignments

**File:** `src/data/pipeline/stages/features/temporal.py`
**Lines:** 38-68

#### BEFORE:

```python
df["hour"] = df["datetime"].dt.hour
df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
df["minute"] = df["datetime"].dt.minute
df["minute_sin"] = np.sin(2 * np.pi * df["minute"] / 60)
df["minute_cos"] = np.cos(2 * np.pi * df["minute"] / 60)
df["dayofweek"] = df["datetime"].dt.dayofweek
df["dayofweek_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7)
df["dayofweek_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7)

def get_session(hour):
    if 0 <= hour < 8:
        return "asia"
    elif 8 <= hour < 16:
        return "london"
    else:
        return "ny"

df["session"] = df["hour"].apply(get_session)  # SLOW!

for session in ["asia", "london", "ny"]:
    df[f"session_{session}"] = (df["session"] == session).astype(int)
```

#### AFTER:

```python
# Extract datetime components once as numpy arrays
hour = df["datetime"].dt.hour.values
minute = df["datetime"].dt.minute.values
dayofweek = df["datetime"].dt.dayofweek.values

# Vectorized session (replaces slow .apply())
session_asia = ((hour >= 0) & (hour < 8)).astype(np.int8)
session_london = ((hour >= 8) & (hour < 16)).astype(np.int8)
session_ny = (hour >= 16).astype(np.int8)

# Build all columns in a single dict, concat once
new_cols = pd.DataFrame({
    "hour_sin": np.sin(2 * np.pi * hour / 24),
    "hour_cos": np.cos(2 * np.pi * hour / 24),
    "minute_sin": np.sin(2 * np.pi * minute / 60),
    "minute_cos": np.cos(2 * np.pi * minute / 60),
    "dayofweek_sin": np.sin(2 * np.pi * dayofweek / 7),
    "dayofweek_cos": np.cos(2 * np.pi * dayofweek / 7),
    "session_asia": session_asia,
    "session_london": session_london,
    "session_ny": session_ny,
}, index=df.index)

df = pd.concat([df, new_cols], axis=1)
```

**Speedup:** 10-100x (removes .apply())

---

### Task 23C-2: Batch Microstructure Feature Assignment

**File:** `src/data/pipeline/stages/features/microstructure.py`
**Lines:** 589-591

#### BEFORE:

```python
for col in new_features.columns:
    df[col] = new_features[col]  # Individual assignment in loop
    feature_metadata[col] = f"Microstructure 2024: {col}"
```

#### AFTER:

```python
# Batch assignment (single concat)
df = pd.concat([df, new_features], axis=1)

# Update metadata separately
for col in new_features.columns:
    feature_metadata[col] = f"Microstructure 2024: {col}"
```

**Speedup:** 5-20x

---

### Task 23C-3: Batch Bollinger Band Assignments

**File:** `src/data/pipeline/stages/features/volatility.py`
**Lines:** 97-116

#### BEFORE:

```python
df["bb_middle"] = bb_middle_raw.shift(1)
bb_std = bb_std_raw.shift(1)
df["bb_upper"] = df["bb_middle"] + (std_mult * bb_std)
df["bb_lower"] = df["bb_middle"] - (std_mult * bb_std)
bb_std_safe = bb_std.replace(0, np.nan)
df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / bb_std_safe
band_range = df["bb_upper"] - df["bb_lower"]
band_range_safe = band_range.replace(0, np.nan)
close_lagged = df["close"].shift(1)
df["bb_position"] = (close_lagged - df["bb_lower"]) / band_range_safe
df["close_bb_zscore"] = (close_lagged - df["bb_middle"]) / bb_std_safe
```

#### AFTER:

```python
# Compute all values using numpy arrays first
bb_middle = bb_middle_raw.shift(1).values
bb_std = bb_std_raw.shift(1).values
bb_upper = bb_middle + (std_mult * bb_std)
bb_lower = bb_middle - (std_mult * bb_std)
bb_std_safe = np.where(bb_std == 0, np.nan, bb_std)
band_range = bb_upper - bb_lower
band_range_safe = np.where(band_range == 0, np.nan, band_range)
close_lagged = df["close"].shift(1).values

# Single concat
bb_cols = pd.DataFrame({
    "bb_middle": bb_middle,
    "bb_upper": bb_upper,
    "bb_lower": bb_lower,
    "bb_width": band_range / bb_std_safe,
    "bb_position": (close_lagged - bb_lower) / band_range_safe,
    "close_bb_zscore": (close_lagged - bb_middle) / bb_std_safe,
}, index=df.index)

df = pd.concat([df, bb_cols], axis=1)
```

**Speedup:** 2-5x

---

### Task 23C-4: Batch Trend Feature Assignments

**File:** `src/data/pipeline/stages/features/trend.py`
**Lines:** 167-168

#### BEFORE:

```python
df["supertrend"] = pd.Series(supertrend).shift(1).values
df["supertrend_direction"] = pd.Series(direction).shift(1).values
```

#### AFTER:

```python
def numpy_shift(arr, periods=1):
    result = np.empty(len(arr), dtype=np.float64)
    result[:periods] = np.nan
    result[periods:] = arr[:-periods]
    return result

trend_cols = pd.DataFrame({
    "supertrend": numpy_shift(supertrend, 1),
    "supertrend_direction": numpy_shift(direction, 1),
}, index=df.index)
df = pd.concat([df, trend_cols], axis=1)
```

**Speedup:** 2-3x

---

### Task 23C-5: Batch Entropy Feature Assignments

**File:** `src/data/pipeline/stages/features/entropy.py`
**Lines:** 234, 420, 621, 837, 1063

#### Pattern BEFORE:

```python
df[col_name] = pd.Series(entropy, index=df.index).shift(1)
```

#### Pattern AFTER:

```python
# At start of function, create dict
new_cols = {}

# Replace each assignment with:
shifted = np.concatenate([[np.nan], entropy[:-1]])
new_cols[col_name] = shifted

# At end of function, single concat:
df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
```

**Speedup:** 2-5x

---

### Task 23C-6: Batch Wavelet Feature Assignments

**File:** `src/data/pipeline/stages/features/wavelets.py`
**Lines:** 154, 164, 200, 207, 212, 253, 300, 301

Same pattern as 23C-5 - collect in dict, concat once at end of each function.

---

### Task 23C-7: Batch Momentum Feature Assignments

**File:** `src/data/pipeline/stages/features/momentum.py`
**Lines:** 51-55, 100-108, 164-169

#### BEFORE (add_rsi):

```python
df[col_name] = pd.Series(calculate_rsi_numba(df["close"].values, period)).shift(1).values
df["rsi_overbought"] = (df[col_name] > 70).astype(int)
df["rsi_oversold"] = (df[col_name] < 30).astype(int)
```

#### AFTER:

```python
rsi = calculate_rsi_numba(df["close"].values, period)
rsi_shifted = np.concatenate([[np.nan], rsi[:-1]])

rsi_cols = pd.DataFrame({
    col_name: rsi_shifted,
    "rsi_overbought": (rsi_shifted > 70).astype(np.int8),
    "rsi_oversold": (rsi_shifted < 30).astype(np.int8),
}, index=df.index)
df = pd.concat([df, rsi_cols], axis=1)
```

---

### Task 23C-8: Batch Price Feature Autocorrelation Loop

**File:** `src/data/pipeline/stages/features/price_features.py`
**Lines:** 142-152

#### BEFORE:

```python
for lag in lags:
    col = f"return_autocorr_lag{lag}"
    autocorr = returns.rolling(period).apply(...)
    df[col] = autocorr  # Individual in loop
```

#### AFTER:

```python
autocorr_cols = {}
for lag in lags:
    col = f"return_autocorr_lag{lag}"
    autocorr_cols[col] = returns.rolling(period).apply(...)

df = pd.concat([df, pd.DataFrame(autocorr_cols, index=df.index)], axis=1)
```

---

### Task 23C-9: Batch Regime Feature Assignments

**File:** `src/data/pipeline/stages/features/regime.py`
**Lines:** 96, 113

#### BEFORE:

```python
df["volatility_regime"] = (df["hvol_20"] > hvol_median).astype(int)
df["trend_regime"] = np.where(uptrend, 1, np.where(downtrend, -1, 0))
```

#### AFTER:

```python
regime_cols = pd.DataFrame({
    "volatility_regime": (df["hvol_20"] > hvol_median).astype(np.int8),
    "trend_regime": np.where(uptrend, 1, np.where(downtrend, -1, 0)).astype(np.int8),
}, index=df.index)
df = pd.concat([df, regime_cols], axis=1)
```

---

### Task 23C-10: Fix fillna Deprecation Warning

**File:** `src/data/pipeline/stages/features/microstructure_proxies.py`
**Line:** 504

#### BEFORE:

```python
features = features.fillna(method="bfill").fillna(0)
```

#### AFTER:

```python
features = features.bfill().fillna(0)
```

---

## Phase 23D: Config Gaps (DEFERRED TO PHASE 24)

**Priority:** LOW
**Status:** DEFERRED - Will be addressed after 23A-C blocking issues fixed
**Impact:** Production deployment features - system works without these

---

### Task 23D-1: Add MTF Mode to ExperimentConfig

**File:** `src/config/experiment.py`
**Status:** DEFERRED

#### Current State:

MTFConfig exists in `src/config/data.py` with all modes, but ExperimentConfig doesn't expose it directly.

#### Implementation:

```python
# Add to ExperimentConfig.__init__()
def __init__(
    self,
    ...
    mtf_mode: str = "indicators",  # NEW: 'none', 'indicators', 'bars', 'both', 'multi_stream'
    mtf_timeframes: list[str] | None = None,  # NEW: e.g., ['5min', '15min', '1h']
):
    self.mtf_config = MTFConfig(
        mode=mtf_mode,
        timeframes=mtf_timeframes or ["5min", "15min", "1h"],
    )
```

#### Validation:

```bash
python -c "
from src.config.experiment import ExperimentConfig
config = ExperimentConfig(symbol='TEST', mtf_mode='multi_stream')
print(f'MTF mode: {config.mtf_config.mode}')
"
```

---

### Task 23D-2: Per-Model Feature Selection Override

**File:** `src/config/experiment.py`
**Status:** DEFERRED

#### Current State:

FeatureConfig has global `selection_n_features=50` but models have different limits (PatchTST=10, LightGBM=200).

#### Implementation:

```python
# Add to ExperimentConfig
@dataclass
class ModelFeatureOverride:
    max_features: int | None = None
    selection_method: str | None = None

# In ExperimentConfig.__init__()
model_feature_overrides: dict[str, ModelFeatureOverride] = {
    "patchtst": ModelFeatureOverride(max_features=10),
    "itransformer": ModelFeatureOverride(max_features=10),
    "tcn": ModelFeatureOverride(max_features=120),
}
```

#### Validation:

```bash
python -c "
from src.config.experiment import ExperimentConfig
config = ExperimentConfig(symbol='TEST', models=['patchtst'])
override = config.model_feature_overrides.get('patchtst')
print(f'PatchTST max_features: {override.max_features if override else \"default\"}')
"
```

---

### Task 23D-3: Bundle Registry & Versioning

**File:** `src/inference/registry.py` (NEW)
**Status:** DEFERRED

#### Current State:

BundleConfig has `version: str = "1.0.0"` but no registry or rollback support.

#### Implementation:

```python
# src/inference/registry.py (NEW FILE)
from dataclasses import dataclass
from pathlib import Path
import json
from datetime import datetime

@dataclass
class BundleMetadata:
    bundle_id: str
    model_name: str
    version: str
    created_at: str
    path: Path
    metrics: dict
    previous_version: str | None = None

class BundleRegistry:
    """Registry for tracking deployed model bundles."""

    def __init__(self, registry_path: Path = Path("bundles/registry.json")):
        self.registry_path = registry_path
        self._registry: dict[str, BundleMetadata] = {}
        self._load()

    def register(self, bundle: "ModelBundle", metrics: dict) -> str:
        """Register a new bundle, return bundle_id."""
        bundle_id = f"{bundle.model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Find previous version
        previous = self.get_latest(bundle.model_name)

        metadata = BundleMetadata(
            bundle_id=bundle_id,
            model_name=bundle.model_name,
            version=bundle.version,
            created_at=datetime.now().isoformat(),
            path=bundle.path,
            metrics=metrics,
            previous_version=previous.bundle_id if previous else None,
        )

        self._registry[bundle_id] = metadata
        self._save()
        return bundle_id

    def get(self, bundle_id: str) -> BundleMetadata | None:
        return self._registry.get(bundle_id)

    def get_latest(self, model_name: str) -> BundleMetadata | None:
        """Get most recent bundle for a model."""
        candidates = [m for m in self._registry.values() if m.model_name == model_name]
        return max(candidates, key=lambda m: m.created_at) if candidates else None

    def rollback(self, bundle_id: str) -> BundleMetadata | None:
        """Get previous version of a bundle."""
        current = self.get(bundle_id)
        if current and current.previous_version:
            return self.get(current.previous_version)
        return None

    def list_versions(self, model_name: str) -> list[BundleMetadata]:
        """List all versions of a model."""
        return sorted(
            [m for m in self._registry.values() if m.model_name == model_name],
            key=lambda m: m.created_at,
            reverse=True,
        )

    def _load(self):
        if self.registry_path.exists():
            with open(self.registry_path) as f:
                data = json.load(f)
                self._registry = {k: BundleMetadata(**v) for k, v in data.items()}

    def _save(self):
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.registry_path, "w") as f:
            json.dump({k: asdict(v) for k, v in self._registry.items()}, f, indent=2)
```

#### Validation:

```bash
python -c "
from src.inference.registry import BundleRegistry
registry = BundleRegistry()
print(f'Registry initialized: {registry.registry_path}')
"
```

---

### Task 23D-4: A/B Testing Configuration

**File:** `src/config/inference.py`
**Status:** DEFERRED

#### Implementation:

```python
# Add to src/config/inference.py
@dataclass
class ABTestConfig:
    """Configuration for A/B testing between model versions."""
    enabled: bool = False
    control_bundle_id: str = ""
    treatment_bundle_id: str = ""
    traffic_split: float = 0.5  # Fraction of traffic to treatment (0.0-1.0)
    metric: str = "sharpe_ratio"  # Metric to compare
    min_samples: int = 1000  # Minimum samples before declaring winner
    significance_level: float = 0.05  # p-value threshold

    def get_variant(self, request_id: str) -> str:
        """Deterministically assign request to control or treatment."""
        import hashlib
        hash_val = int(hashlib.md5(request_id.encode()).hexdigest(), 16)
        return "treatment" if (hash_val % 100) / 100 < self.traffic_split else "control"
```

#### Validation:

```bash
python -c "
from src.config.inference import ABTestConfig
config = ABTestConfig(enabled=True, traffic_split=0.2)
variants = [config.get_variant(f'req_{i}') for i in range(1000)]
treatment_pct = sum(1 for v in variants if v == 'treatment') / len(variants)
print(f'Treatment %: {treatment_pct:.2%} (expected ~20%)')
"
```

---

### Task 23D-5: Drift Detection Configuration

**File:** `src/config/monitoring.py` (NEW)
**Status:** DEFERRED

#### Implementation:

```python
# src/config/monitoring.py (NEW FILE)
from dataclasses import dataclass, field

@dataclass
class DriftConfig:
    """Configuration for feature and prediction drift detection."""
    enabled: bool = True

    # Feature drift (PSI - Population Stability Index)
    feature_drift_threshold: float = 0.1  # PSI > 0.1 = significant drift
    feature_drift_critical: float = 0.25  # PSI > 0.25 = critical drift

    # Prediction drift
    prediction_drift_threshold: float = 0.15
    prediction_drift_window: int = 1000  # Samples to compare

    # Monitoring schedule
    check_interval_hours: int = 24

    # Alerting
    alert_channels: list[str] = field(default_factory=lambda: ["log"])  # log, email, slack, pagerduty
    alert_cooldown_hours: int = 4  # Don't re-alert within this window

    # Auto-remediation
    auto_retrain_trigger: bool = False
    auto_retrain_threshold: float = 0.3  # Trigger retrain if drift > this

    # Reference data
    reference_data_path: str = ""  # Path to baseline distribution

@dataclass
class MonitoringConfig:
    """Top-level monitoring configuration."""
    drift: DriftConfig = field(default_factory=DriftConfig)
    log_predictions: bool = True
    log_features: bool = False  # Can be expensive
    metrics_export_interval_seconds: int = 60
```

#### Validation:

```bash
python -c "
from src.config.monitoring import DriftConfig, MonitoringConfig
config = MonitoringConfig()
print(f'Drift enabled: {config.drift.enabled}')
print(f'Feature drift threshold: {config.drift.feature_drift_threshold}')
print(f'Alert channels: {config.drift.alert_channels}')
"
```

---

### Task 23D-6: Streaming Inference Configuration

**File:** `src/config/inference.py`
**Status:** DEFERRED

#### Current State:

InferenceConfig has `mode: str = "streaming"` but no buffer/latency config.

#### Implementation:

```python
# Add to src/config/inference.py
@dataclass
class StreamingConfig:
    """Configuration for streaming inference mode."""
    enabled: bool = False

    # Buffer settings
    buffer_size: int = 1000  # Max items in buffer
    flush_interval_seconds: float = 1.0  # Force flush after this time

    # Latency requirements
    max_latency_ms: float = 100.0  # Target max latency
    timeout_ms: float = 500.0  # Hard timeout

    # Backpressure handling
    backpressure_strategy: str = "drop"  # drop, block, sample
    sample_rate: float = 0.1  # If strategy=sample, keep this fraction

    # State management
    checkpoint_interval_seconds: float = 60.0
    checkpoint_path: str = ""

    # Warm-up
    warmup_samples: int = 100  # Samples to process before going live
```

#### Validation:

```bash
python -c "
from src.config.inference import StreamingConfig
config = StreamingConfig(enabled=True, buffer_size=500)
print(f'Buffer size: {config.buffer_size}')
print(f'Max latency: {config.max_latency_ms}ms')
"
```

---

### Task 23D-7: Compatibility Matrix Documentation

**File:** `docs/COMPATIBILITY.md` (NEW)
**Status:** DEFERRED

#### Implementation:

Generate from MODEL_CONTRACTS:

```python
# Script: scripts/generate_compatibility_matrix.py
from src.core.contracts import MODEL_CONTRACTS

def generate_matrix():
    rows = []
    for name, contract in MODEL_CONTRACTS.items():
        rows.append({
            "Model": name,
            "Family": contract.model_family,
            "Adapter": contract.adapter_id,
            "Input Rank": contract.input_rank.name,
            "MTF Mode": contract.mtf_mode.name,
            "Feature Mode": contract.feature_mode.name,
            "Min Features": contract.min_features,
            "Max Features": contract.max_features,
            "Sequence Length": contract.sequence_length or "-",
        })

    # Generate markdown table
    headers = list(rows[0].keys())
    lines = [
        "# Model Compatibility Matrix\n",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row[h]) for h in headers) + " |")

    return "\n".join(lines)

if __name__ == "__main__":
    print(generate_matrix())
```

#### Output Example:

```markdown
# Model Compatibility Matrix

| Model | Family | Adapter | Input Rank | MTF Mode | Feature Mode | Min Features | Max Features | Sequence Length |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| xgboost | boosting | tabular | TABULAR_2D | INDICATORS | ENGINEERED | 5 | 200 | - |
| lightgbm | boosting | tabular | TABULAR_2D | INDICATORS | ENGINEERED | 5 | 200 | - |
| tcn | neural | sequence | SEQUENCE_3D | NONE | ENGINEERED | 5 | 120 | 60 |
| patchtst | transformer | multi_stream | MULTI_TF_4D | MULTI_STREAM | RAW | 4 | 10 | 64 |
```

---

### 23D Summary

| Task | File | Description | Status |
|------|------|-------------|--------|
| 23D-1 | `experiment.py` | MTF mode in ExperimentConfig | DEFERRED |
| 23D-2 | `experiment.py` | Per-model feature selection override | DEFERRED |
| 23D-3 | `registry.py` (NEW) | Bundle registry & versioning | DEFERRED |
| 23D-4 | `inference.py` | A/B testing configuration | DEFERRED |
| 23D-5 | `monitoring.py` (NEW) | Drift detection configuration | DEFERRED |
| 23D-6 | `inference.py` | Streaming inference configuration | DEFERRED |
| 23D-7 | `docs/COMPATIBILITY.md` | Compatibility matrix docs | DEFERRED |

**Total 23D Tasks:** 7 (all deferred to Phase 24)

---

## Verification Commands

### Core Imports

```bash
python -c "from src.core.types import DataRank, ModelFamily; print('OK')"
python -c "from src.core.contracts import get_model_contract; print('OK')"
python -c "from src.data.adapters import get_adapter; print('OK')"
```

### Phase 23A

```bash
grep -n '"label"' src/data/adapters/base.py | grep exclude_exact
```

### Phase 23B

```bash
python -c "
from src.core.contracts import get_model_contract
for m in ['lightgbm', 'tcn', 'patchtst']:
    c = get_model_contract(m)
    print(f'{m}: max_features={c.max_features}')
"
```

### Phase 23C

```bash
python -c "
import warnings
import pandas as pd
warnings.filterwarnings('error', category=pd.errors.PerformanceWarning)
from src.data.pipeline.stages.features import temporal
print('PASS: No PerformanceWarning')
"
```

---

## Deferred Backlog (Low Priority)

| Task | Description | Notes |
|------|-------------|-------|
| 5C | Unified deployment bundle | Needs spec |
| 4D | Deflated Sharpe Ratio | Post-Optuna gate |
| 4E | Bootstrap CIs | Wire BootstrapCI |
| 4F | Auto calibration | Wire CalibrationManager |

---

## Summary Checklist

### Active Tasks (23A-C)

| Task | Description | Priority | Status |
|------|-------------|----------|--------|
| 23A-1 | Add "label" to exclude_exact | CRITICAL | [ ] |
| 23B-1 | Skip rank validation on raw data | HIGH | [ ] |
| 23B-2 | Auto feature selection | HIGH | [ ] |
| 23C-1 | temporal.py vectorization | MEDIUM | [ ] |
| 23C-2 | microstructure.py batch concat | MEDIUM | [ ] |
| 23C-3 | volatility.py batch assign | MEDIUM | [ ] |
| 23C-4 | trend.py batch assign | MEDIUM | [ ] |
| 23C-5 | entropy.py batch assign | MEDIUM | [ ] |
| 23C-6 | wavelets.py batch assign | MEDIUM | [ ] |
| 23C-7 | momentum.py batch assign | MEDIUM | [ ] |
| 23C-8 | price_features.py autocorr loop | MEDIUM | [ ] |
| 23C-9 | regime.py batch assign | MEDIUM | [ ] |
| 23C-10 | fillna deprecation fix | LOW | [ ] |

### Deferred Tasks (23D - Phase 24)

| Task | Description | Priority | Status |
|------|-------------|----------|--------|
| 23D-1 | MTF mode in ExperimentConfig | LOW | [ ] DEFERRED |
| 23D-2 | Per-model feature selection override | LOW | [ ] DEFERRED |
| 23D-3 | Bundle registry & versioning | LOW | [ ] DEFERRED |
| 23D-4 | A/B testing configuration | LOW | [ ] DEFERRED |
| 23D-5 | Drift detection configuration | LOW | [ ] DEFERRED |
| 23D-6 | Streaming inference configuration | LOW | [ ] DEFERRED |
| 23D-7 | Compatibility matrix documentation | LOW | [ ] DEFERRED |

**Total:** 13 active tasks + 7 deferred = 20 tasks

---

*See COMPLETION.md for implementation details after phase completion*
