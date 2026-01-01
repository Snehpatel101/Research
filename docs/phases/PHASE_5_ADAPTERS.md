# Phase 5: Model-Family Adapters

**Status:** ✅ Complete
**Effort:** 2 days (completed)
**Dependencies:** Phase 4 (labeled, scaled data)

---

## Goal

Transform the canonical labeled dataset into model-family-specific formats (2D tabular, 3D sequences, 4D multi-resolution tensors) via deterministic adapters, enabling a single pipeline to serve all model types.

**Output:** Model-specific datasets with appropriate shapes and lookback windows, ready for training.

---

## Current Status

### Implemented
- ✅ **Tabular adapter** (2D): For boosting and classical models
- ✅ **Sequence adapter** (3D): For neural network models with lookback windows
- ✅ **TimeSeriesDataContainer**: Unified interface for all model families
- ✅ **Automatic adapter selection**: Based on model family registration

### Not Yet Implemented
- ❌ **Multi-resolution adapter** (4D): For PatchTST, iTransformer, TFT, N-BEATS (requires Phase 2 Strategy 3)

---

## Architecture: One Pipeline, Multiple Adapters

```
Canonical Dataset (Phase 4 output)
         ↓
┌────────────────────────────────────────┐
│   TimeSeriesDataContainer              │
│   (Unified interface)                  │
└────────────────────────────────────────┘
         ↓
  [Model Family Router]
         ↓
    ┌────┴────┬──────────┬───────────┐
    ↓         ↓          ↓           ↓
Tabular   Sequence   MultiRes    (Future)
2D Array  3D Windows 4D Tensors
(N, F)    (N, T, F)  (N, TF, T, 4)
    ↓         ↓          ↓
 Boosting   Neural    Advanced
Classical   TCN       PatchTST
           Trans.     iTransformer
```

**Key Principle:** One canonical dataset → Deterministic adapters → Model-specific formats

---

## Data Contracts

### Input Specification

**Files:**
- `data/splits/scaled/{symbol}_train.parquet`
- `data/splits/scaled/{symbol}_val.parquet`
- `data/splits/scaled/{symbol}_test.parquet`

**Schema:**
```python
{
    "timestamp": datetime64[ns],
    "label": int64,
    "sample_weight": float64,
    # ~180 scaled feature columns (mean=0, std=1)
}
```

### Output Specification: Tabular Models

**Shape:** 2D arrays `(n_samples, n_features)`

**Data Structure:**
```python
{
    "X_train": np.ndarray,  # (N_train, 180)
    "y_train": np.ndarray,  # (N_train,)
    "w_train": np.ndarray,  # (N_train,)
    "X_val": np.ndarray,    # (N_val, 180)
    "y_val": np.ndarray,    # (N_val,)
    "w_val": np.ndarray,    # (N_val,)
    "X_test": np.ndarray,   # (N_test, 180)
    "y_test": np.ndarray,   # (N_test,)
    "w_test": np.ndarray,   # (N_test,)
}
```

**Models:** XGBoost, LightGBM, CatBoost, Random Forest, Logistic, SVM

### Output Specification: Sequence Models

**Shape:** 3D arrays `(n_samples, seq_len, n_features)`

**Data Structure:**
```python
{
    "X_train": np.ndarray,  # (N_train, seq_len, 180)
    "y_train": np.ndarray,  # (N_train,)
    "w_train": np.ndarray,  # (N_train,)
    "X_val": np.ndarray,    # (N_val, seq_len, 180)
    "y_val": np.ndarray,    # (N_val,)
    "w_val": np.ndarray,    # (N_val,)
    "X_test": np.ndarray,   # (N_test, seq_len, 180)
    "y_test": np.ndarray,   # (N_test,)
    "w_test": np.ndarray,   # (N_test,)
}
```

**Sequence Length:** Configurable (default: 30 for LSTM/GRU, 60 for TCN/Transformer)

**Models:** LSTM, GRU, TCN, Transformer

### Output Specification: Multi-Resolution Models (Future)

**Shape:** 4D arrays `(n_samples, n_timeframes, lookback, 4)`

**Data Structure:**
```python
{
    "X_train": np.ndarray,  # (N_train, 9, max_lookback, 4)
    # 9 timeframes: 1m, 5m, 10m, 15m, 20m, 25m, 30m, 45m, 1h
    # max_lookback: longest window (e.g., 60 bars)
    # 4: OHLC features
}
```

**Models:** PatchTST, iTransformer, TFT, N-BEATS

**Status:** ❌ Not implemented (requires Phase 2 Strategy 3)

---

## Implementation Tasks

### Task 5.1: TimeSeriesDataContainer
**File:** `src/phase1/stages/datasets/time_series_container.py`

**Status:** ✅ Complete

**Implementation:**
```python
@dataclass
class TimeSeriesDataContainer:
    """Unified container for time series datasets."""

    # Data arrays
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray

    # Sample weights
    w_train: Optional[np.ndarray] = None
    w_val: Optional[np.ndarray] = None
    w_test: Optional[np.ndarray] = None

    # Metadata
    feature_names: List[str] = field(default_factory=list)
    symbol: str = ""
    horizon: int = 0
    seq_len: Optional[int] = None  # For sequence models

    # Shape validation
    def __post_init__(self):
        """Validate shapes are consistent."""
        # Check X and y have matching sample counts
        assert len(self.X_train) == len(self.y_train)
        assert len(self.X_val) == len(self.y_val)
        assert len(self.X_test) == len(self.y_test)

        # Check weights if provided
        if self.w_train is not None:
            assert len(self.w_train) == len(self.y_train)

    @property
    def is_sequence(self) -> bool:
        """Check if this is sequence data (3D)."""
        return len(self.X_train.shape) == 3

    @property
    def n_features(self) -> int:
        """Get feature count."""
        if self.is_sequence:
            return self.X_train.shape[2]
        return self.X_train.shape[1]
```

### Task 5.2: Tabular Adapter
**File:** `src/phase1/stages/datasets/dataset_builder.py`

**Status:** ✅ Complete

**Implementation:**
```python
class DatasetBuilder:
    def build_tabular_dataset(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        feature_cols: List[str],
        label_col: str = "label",
        weight_col: str = "sample_weight"
    ) -> TimeSeriesDataContainer:
        """Build 2D tabular dataset."""

        # Extract arrays
        X_train = train_df[feature_cols].values  # Shape: (N_train, F)
        y_train = train_df[label_col].values
        w_train = train_df[weight_col].values

        X_val = val_df[feature_cols].values
        y_val = val_df[label_col].values
        w_val = val_df[weight_col].values

        X_test = test_df[feature_cols].values
        y_test = test_df[label_col].values
        w_test = test_df[weight_col].values

        # Return container
        return TimeSeriesDataContainer(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            X_test=X_test,
            y_test=y_test,
            w_train=w_train,
            w_val=w_val,
            w_test=w_test,
            feature_names=feature_cols,
            symbol=self.symbol,
            horizon=self.horizon
        )
```

### Task 5.3: Sequence Adapter
**File:** `src/phase1/stages/datasets/dataset_builder.py`

**Status:** ✅ Complete

**Implementation:**
```python
class DatasetBuilder:
    def build_sequence_dataset(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        feature_cols: List[str],
        label_col: str = "label",
        weight_col: str = "sample_weight",
        seq_len: int = 30
    ) -> TimeSeriesDataContainer:
        """Build 3D sequence dataset with lookback windows."""

        # Create windowed views
        X_train, y_train, w_train = self._create_windows(
            train_df, feature_cols, label_col, weight_col, seq_len
        )
        X_val, y_val, w_val = self._create_windows(
            val_df, feature_cols, label_col, weight_col, seq_len
        )
        X_test, y_test, w_test = self._create_windows(
            test_df, feature_cols, label_col, weight_col, seq_len
        )

        return TimeSeriesDataContainer(
            X_train=X_train,  # Shape: (N_train, seq_len, F)
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            X_test=X_test,
            y_test=y_test,
            w_train=w_train,
            w_val=w_val,
            w_test=w_test,
            feature_names=feature_cols,
            symbol=self.symbol,
            horizon=self.horizon,
            seq_len=seq_len
        )

    def _create_windows(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        label_col: str,
        weight_col: str,
        seq_len: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Create sliding windows from DataFrame."""

        features = df[feature_cols].values
        labels = df[label_col].values
        weights = df[weight_col].values

        n_samples = len(features) - seq_len + 1
        X = np.zeros((n_samples, seq_len, len(feature_cols)))
        y = np.zeros(n_samples)
        w = np.zeros(n_samples)

        for i in range(n_samples):
            X[i] = features[i:i+seq_len]
            y[i] = labels[i+seq_len-1]  # Label at end of window
            w[i] = weights[i+seq_len-1]

        return X, y, w
```

**Windowing Logic:**
- For each prediction point at time `t`
- Look back `seq_len` bars: `[t - seq_len + 1, ..., t]`
- Label corresponds to time `t` (end of window)

### Task 5.4: Model Family Router
**File:** `src/models/trainer.py`

**Status:** ✅ Complete

**Implementation:**
```python
class ModelTrainer:
    def prepare_data(
        self,
        model_family: str,
        container: TimeSeriesDataContainer,
        seq_len: Optional[int] = None
    ) -> TimeSeriesDataContainer:
        """Route to appropriate adapter based on model family."""

        if model_family in ["boosting", "classical"]:
            # Tabular models: use 2D data directly
            if container.is_sequence:
                raise ValueError(f"Tabular model {model_family} requires 2D data")
            return container

        elif model_family == "neural":
            # Sequence models: ensure 3D data
            if not container.is_sequence:
                # Convert 2D to 3D with seq_len
                if seq_len is None:
                    raise ValueError("seq_len required for neural models")
                # Rebuild as sequence dataset
                return self.dataset_builder.build_sequence_dataset(
                    train_df, val_df, test_df, feature_cols, seq_len=seq_len
                )
            return container

        elif model_family == "ensemble":
            # Ensembles: all base models must be same family
            # Use base model family's adapter
            base_family = self.get_base_model_family()
            return self.prepare_data(base_family, container, seq_len)

        else:
            raise ValueError(f"Unknown model family: {model_family}")
```

### Task 5.5: Multi-Resolution Adapter (TODO)
**File:** `src/phase1/stages/datasets/multires_builder.py`

**Status:** ❌ Not implemented

**Implementation:**
```python
class MultiResolutionBuilder:
    def build_multires_dataset(
        self,
        mtf_dfs: Dict[str, pd.DataFrame],
        lookback_config: Dict[str, int],
        label_col: str = "label"
    ) -> TimeSeriesDataContainer:
        """Build 4D multi-resolution dataset.

        Args:
            mtf_dfs: Dict of {timeframe: DataFrame} from Phase 2
            lookback_config: Dict of {timeframe: num_bars}
                Example: {'5min': 60, '15min': 20, '30min': 10, '1h': 5}

        Returns:
            Container with X shape: (N, n_timeframes, max_lookback, 4)
        """
        # 1. For each prediction point:
        #    a. For each timeframe:
        #       - Extract lookback window
        #       - Extract OHLC (4 features)
        #       - Pad if necessary
        #    b. Stack into 4D array
        # 2. Return TimeSeriesDataContainer with 4D X arrays
```

**Effort:** 2-3 days (depends on Phase 2 Strategy 3 completion)

---

## Testing Requirements

### Unit Tests
**File:** `tests/phase1/test_adapters.py`

```python
def test_tabular_adapter():
    """Test 2D tabular adapter."""
    # 1. Create synthetic DataFrame (100 samples, 10 features)
    # 2. Build tabular dataset
    # 3. Assert X shape is (N, 10)
    # 4. Assert y shape is (N,)

def test_sequence_adapter():
    """Test 3D sequence adapter."""
    # 1. Create synthetic DataFrame (100 samples, 10 features)
    # 2. Build sequence dataset (seq_len=30)
    # 3. Assert X shape is (70, 30, 10)  # 100 - 30 + 1 = 71, but label at end
    # 4. Assert y shape is (70,)

def test_windowing_labels():
    """Test sequence windows have correct labels."""
    # 1. Create DataFrame with known labels
    # 2. Build windows (seq_len=5)
    # 3. Assert each window's label matches end-of-window label

def test_model_family_router():
    """Test router selects correct adapter."""
    # 1. Create 2D container
    # 2. Route to boosting family
    # 3. Assert returns 2D container unchanged
    # 4. Route to neural family with seq_len
    # 5. Assert returns 3D container
```

### Integration Tests
**File:** `tests/phase1/test_dataset_pipeline.py`

```python
def test_end_to_end_dataset_building():
    """Test full dataset building pipeline."""
    # 1. Load splits from Phase 4
    # 2. Build tabular dataset
    # 3. Build sequence dataset
    # 4. Assert both containers valid
    # 5. Assert shapes correct
```

---

## Artifacts

### No Persistent Files (In-Memory)

Adapters produce **in-memory** `TimeSeriesDataContainer` objects consumed directly by model trainers.

**Rationale:** Avoid redundant storage; canonical data in `data/splits/scaled/` is single source of truth.

### Optional Caching (For Large Datasets)

If datasets are very large and building is expensive:

```python
# Optional: Save adapted datasets
container.save("data/adapted/{symbol}_{model_family}_dataset.pkl")

# Load instead of rebuilding
container = TimeSeriesDataContainer.load("data/adapted/{symbol}_{model_family}_dataset.pkl")
```

**Current Implementation:** No caching (datasets small enough to rebuild on-the-fly)

---

## Configuration

**File:** `config/models.yaml`

```yaml
adapters:
  tabular:
    families: ["boosting", "classical"]
    expected_shape: "2D"  # (N, F)

  sequence:
    families: ["neural"]
    expected_shape: "3D"  # (N, T, F)
    default_seq_lens:
      lstm: 30
      gru: 30
      tcn: 60
      transformer: 60

  multires:  # Not yet implemented
    families: ["advanced"]
    expected_shape: "4D"  # (N, TF, T, 4)
    timeframes: 9
```

---

## Model Family Compatibility

| Model Family | Input Shape | Adapter | Seq Len | Status |
|-------------|-------------|---------|---------|--------|
| **Boosting** | 2D `(N, F)` | Tabular | N/A | ✅ Complete |
| **Classical** | 2D `(N, F)` | Tabular | N/A | ✅ Complete |
| **Neural** | 3D `(N, T, F)` | Sequence | 30-60 | ✅ Complete |
| **Ensemble** | Same as base models | Base adapter | Varies | ✅ Complete |
| **Advanced** | 4D `(N, TF, T, 4)` | MultiRes | Varies | ❌ Not implemented |

**CRITICAL:** Ensemble models require all base models from same family (same input shape).

---

## Dependencies

**Internal:**
- Phase 4 (labeled, scaled splits)

**External:**
- `numpy >= 1.24.0` - Array operations
- `pandas >= 2.0.0` - DataFrame operations

---

## Next Steps

**After Phase 5 completion:**
1. ✅ Adapters ready to serve all 13 implemented models
2. ➡️ Proceed to **Phase 6: Training Pipeline** to train models
3. ➡️ Multi-resolution adapter enables Phase 2 Strategy 3 models (PatchTST, etc.)

**Validation Checklist:**
- [ ] TimeSeriesDataContainer validates shapes
- [ ] Tabular adapter produces 2D arrays
- [ ] Sequence adapter produces 3D arrays
- [ ] Windowing logic correct (label at end of window)
- [ ] Model family router selects correct adapter
- [ ] Ensemble compatibility validated

---

## Performance

**Benchmarks (MES 1-year data, ~73K train samples):**
- Tabular adapter: ~0.2 seconds (array conversion)
- Sequence adapter (seq_len=30): ~1.5 seconds (windowing)
- **Total Phase 5 runtime: <2 seconds**

**Memory:**
- Tabular (2D): ~50 MB
- Sequence (3D, seq_len=30): ~150 MB (3x memory due to windows)

---

## References

**Code Files:**
- `src/phase1/stages/datasets/time_series_container.py` - Container dataclass
- `src/phase1/stages/datasets/dataset_builder.py` - Tabular and sequence adapters
- `src/models/trainer.py` - Model family router

**Config Files:**
- `config/models.yaml` - Adapter configuration

**Documentation:**
- `docs/phases/PHASE_2_MTF_UPSCALING.md` - Strategy 3 for multi-resolution data

**Tests:**
- `tests/phase1/test_adapters.py` - Unit tests
- `tests/phase1/test_dataset_pipeline.py` - Integration tests
