# Phase 0: Canonical Contracts - Implementation Complete

**Implemented:** 2026-01-16
**Status:** ✅ Complete (81 tests passing)

---

## Overview

Phase 0 establishes the foundational contracts that all subsequent SNwH phases depend on. These contracts define:

1. **DataContract** - Schema for canonical data flowing through the pipeline
2. **ModelContract** - Input requirements each model declares at registration
3. **ArtifactManifest** - Safety and reproducibility metadata for saved artifacts

---

## Files Created

| File | Lines | Purpose |
|------|-------|---------|
| `src/contracts/__init__.py` | 65 | Package exports |
| `src/contracts/data_contract.py` | 380 | DataContract, DataRank, FeatureMode, MTFMode |
| `src/contracts/model_contract.py` | 520 | ModelContract, MODEL_CONTRACTS (23 models) |
| `src/contracts/artifact_manifest.py` | 340 | ArtifactManifest for reproducibility |
| `tests/contracts/__init__.py` | 1 | Test package |
| `tests/contracts/test_data_contract.py` | 340 | 30 tests |
| `tests/contracts/test_model_contract.py` | 380 | 32 tests |
| `tests/contracts/test_artifact_manifest.py` | 280 | 19 tests |

**Total:** ~2,300 lines, 81 tests

---

## Key Classes

### DataRank Enum
```python
class DataRank(int, Enum):
    TABULAR_2D = 2   # (n_samples, n_features)
    SEQUENCE_3D = 3  # (n_samples, seq_len, n_features)
    MULTI_TF_4D = 4  # (n_samples, n_timeframes, seq_len, n_features)
```

### FeatureMode Enum
```python
class FeatureMode(str, Enum):
    ENGINEERED = "engineered"  # ~180 pre-computed indicators
    RAW = "raw"                # Raw OHLCV only (4-5 features)
    HYBRID = "hybrid"          # Mix of raw + selected indicators
    OOF_PROBS = "oof_probs"    # OOF predictions for meta-learners
```

### MTFMode Enum
```python
class MTFMode(str, Enum):
    NONE = "none"              # Single timeframe, no MTF
    INDICATORS = "indicators"  # MTF indicator features added
    MULTI_STREAM = "multi_stream"  # Multiple TF streams (4D)
```

### DataContract
- Captures data shape, lineage, feature/label metadata
- Validates DataFrames and numpy arrays
- Computes deterministic schema hash
- Factory methods: `from_array()`, `from_dataframe()`, `from_dict()`

### ModelContract
- Declares input requirements (rank, features, scaling)
- Validates compatibility with DataContract
- Properties: `requires_sequences`, `requires_multi_timeframe`, `adapter_id`

### ArtifactManifest
- Tracks artifacts with hashes for verification
- Captures Python/package versions for reproducibility
- Factory methods: `create_for_model()`, `create_for_predictions()`, `create_for_oof()`

---

## MODEL_CONTRACTS Registry (23 Models)

| Model | Family | Rank | TF | Scaling | MTF Mode |
|-------|--------|------|-----|---------|----------|
| xgboost | boosting | 2D | 15min | No | indicators |
| lightgbm | boosting | 2D | 15min | No | indicators |
| catboost | boosting | 2D | 15min | No | indicators |
| random_forest | classical | 2D | 15min | No | indicators |
| logistic | classical | 2D | 15min | Yes (std) | none |
| svm | classical | 2D | 15min | Yes (std) | none |
| lstm | neural | 3D | 5min | Yes (robust) | indicators |
| gru | neural | 3D | 5min | Yes (robust) | indicators |
| tcn | neural | 3D | 5min | Yes (robust) | none |
| transformer | neural | 3D | 5min | Yes (std) | indicators |
| patchtst | neural | 3D | 1min | Yes (std) | multi_stream |
| itransformer | neural | 3D | 1min | Yes (robust) | multi_stream |
| tft | neural | 3D | 5min | Yes (robust) | indicators |
| nbeats | neural | 3D | 5min | Yes (robust) | none |
| inceptiontime | neural | 3D | 5min | Yes (robust) | none |
| resnet1d | neural | 3D | 5min | Yes (robust) | none |
| voting | ensemble | 2D | 5min | No | none |
| stacking | ensemble | 2D | 5min | No | none |
| blending | ensemble | 2D | 5min | No | none |
| ridge_meta | meta_learner | 2D | 5min | No | none |
| mlp_meta | meta_learner | 2D | 5min | Yes (std) | none |
| calibrated_meta | meta_learner | 2D | 5min | No | none |
| xgboost_meta | meta_learner | 2D | 5min | No | none |

---

## Usage Examples

### Get model contract
```python
from src.contracts import get_model_contract, DataRank

contract = get_model_contract("xgboost")
print(contract.input_rank)       # DataRank.TABULAR_2D
print(contract.requires_scaling) # False
print(contract.adapter_id)       # "tabular"
```

### Create data contract from array
```python
from src.contracts import DataContract
import numpy as np

X = np.random.randn(1000, 100)
data_contract = DataContract.from_array(
    X, symbol="MES", timeframe="15min", horizon=20, split="train"
)
print(data_contract.data_rank)   # DataRank.TABULAR_2D
print(data_contract.schema_hash) # "a1b2c3d4e5f6..."
```

### Validate compatibility
```python
model_contract = get_model_contract("lstm")
data_contract = DataContract.from_array(
    X_3d, symbol="MES", timeframe="5min", horizon=20, split="train"
)

is_valid, issues = model_contract.validate_data_contract(data_contract)
if not is_valid:
    print("Issues:", issues)
```

### Create artifact manifest
```python
from src.contracts import ArtifactManifest

manifest = ArtifactManifest.create_for_model(
    model_path=Path("model.pkl"),
    model_name="xgboost",
    data_contract=data_contract,
    model_contract=model_contract,
    training_metrics={"val_accuracy": 0.85},
)
manifest.save(Path("manifest.json"))
```

---

## Helper Functions

```python
# List contracts by family
from src.contracts import list_model_contracts
boosting = list_model_contracts("boosting")  # 3 models

# Get models by rank
from src.contracts import get_models_by_rank, DataRank
tabular_models = get_models_by_rank(DataRank.TABULAR_2D)  # 13 models

# Get models requiring scaling
from src.contracts import get_models_requiring_scaling
scaling_models = get_models_requiring_scaling()  # 12 models

# Get models by MTF mode
from src.contracts import get_models_by_mtf_mode, MTFMode
multi_stream = get_models_by_mtf_mode(MTFMode.MULTI_STREAM)  # patchtst, itransformer
```

---

## Integration Points

Phase 0 contracts will be used by:

1. **Phase 1 (Config Layer):** TrainerConfig.from_model_contract()
2. **Phase 2 (Adapters):** AdapterRegistry uses contract.adapter_id
3. **Phase 3 (Timeframe Coord):** TimeframeCoordinator uses contract.primary_timeframe
4. **Phase 4 (OOF):** OOFAlignmentValidator uses DataContract for coverage
5. **Phase 5 (Features):** FeatureStrategyManager uses contract.feature_mode

---

## Dependencies

- No external dependencies beyond numpy, pandas (already in project)
- No breaking changes to existing code
- Pure additive implementation

---

## Next Steps

Proceed to **Phase 1 (Configuration Layer)** which will:
1. Extend TrainerConfig with fields from ModelContract
2. Add per-model configuration to UnifiedConfig
3. Wire contracts into the existing configuration system

---

**Last Updated:** 2026-01-16
