# Code Naming Conventions

This document establishes standardized naming conventions for the research codebase to ensure consistency, readability, and maintainability.

## Table of Contents
- [Configuration Classes](#configuration-classes)
- [Result/Output Classes](#resultoutput-classes)
- [Function Naming](#function-naming)
- [Import Paths](#import-paths)
- [Module Naming](#module-naming)
- [Variable Naming](#variable-naming)
- [Constants](#constants)

---

## Configuration Classes

**Standard**: Use the `*Config` suffix for all configuration classes.

### Correct Examples
```python
# Good - use *Config suffix
class ServerConfig:
    host: str = "0.0.0.0"
    port: int = 8080

class PipelineConfig:
    batch_size: int = 32
    epochs: int = 100

class FeatureSelectionConfig:
    n_features: int = 50
    method: str = "mrmr"
```

### Avoid
```python
# Avoid - inconsistent suffixes
class ServerSettings:  # Use ServerConfig instead
    ...

class PipelineOptions:  # Use PipelineConfig instead
    ...

class FeatureParams:  # Use FeatureConfig instead
    ...
```

### Rationale
- Single consistent suffix reduces cognitive load
- `Config` is widely recognized across the Python ecosystem
- Easier to search/grep for configuration classes

---

## Result/Output Classes

**Standard**: Use the `*Result` suffix for classes that represent operation outputs/results.

### Correct Examples
```python
# Good - use *Result suffix
@dataclass
class PredictionResult:
    """Standardized prediction output for all models."""
    class_predictions: np.ndarray
    class_probabilities: np.ndarray
    confidence: np.ndarray
    metadata: dict[str, Any]

@dataclass
class TrainingResult:
    """Standardized training output."""
    metrics: TrainingMetrics
    model_path: Path
    artifacts: dict[str, Any]

@dataclass
class ValidationResult:
    """Cross-validation fold results."""
    fold_metrics: list[dict]
    oof_predictions: np.ndarray
```

### Exception: API Response Models
For HTTP API endpoints (FastAPI/Pydantic models), use `*Response` suffix as it follows REST conventions:

```python
# Exception - API response models keep *Response suffix
class PredictionResponse(BaseModel):
    """HTTP API response for predictions endpoint."""
    predictions: list[int]
    probabilities: list[list[float]]

class HealthResponse(BaseModel):
    """Health check endpoint response."""
    status: str
```

### Avoid for Domain Classes
```python
# Avoid for domain/business logic classes
class PredictionOutput:  # Use PredictionResult instead
    ...

class TrainingOutput:  # Use TrainingResult instead
    ...
```

### Rationale
- `*Result` clearly indicates a completed operation's output
- `*Response` is reserved for HTTP/API layer to maintain clear separation
- Consistent naming helps distinguish domain logic from API contracts

---

## Function Naming

Use specific prefixes to indicate the function's purpose and behavior:

### Builders: `create_*()`
Creates new instances, performs initialization.

```python
def create_model(config: ModelConfig) -> BaseModel:
    """Create and initialize a new model instance."""
    ...

def create_pipeline(stages: list[Stage]) -> Pipeline:
    """Create a new pipeline with the given stages."""
    ...

def create_data_loader(path: Path, batch_size: int) -> DataLoader:
    """Create a new data loader for the given path."""
    ...
```

### Loaders: `load_*()`
Loads data from disk, database, or external sources.

```python
def load_model(path: Path) -> BaseModel:
    """Load a trained model from disk."""
    ...

def load_config(path: Path) -> PipelineConfig:
    """Load configuration from a YAML/JSON file."""
    ...

def load_features(symbol: str, start_date: str) -> pd.DataFrame:
    """Load feature data from the feature store."""
    ...
```

### Getters: `get_*()`
Retrieves existing, cached, or computed values without side effects.

```python
def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Get list of feature column names from dataframe."""
    ...

def get_default_config() -> dict[str, Any]:
    """Get default configuration values."""
    ...

def get_model_info(model: BaseModel) -> dict[str, Any]:
    """Get model metadata and information."""
    ...
```

### Validators: `validate_*()`
Validates input, returns validation results or raises exceptions.

```python
def validate_config(config: PipelineConfig) -> list[str]:
    """Validate configuration, return list of errors."""
    ...

def validate_features(df: pd.DataFrame) -> bool:
    """Validate feature dataframe structure."""
    ...

def validate_model_input(X: np.ndarray, expected_shape: tuple) -> None:
    """Validate model input, raise ValueError if invalid."""
    ...
```

### Transformers: `transform_*()`
Transforms data from one format/structure to another.

```python
def transform_to_sequences(df: pd.DataFrame, seq_len: int) -> np.ndarray:
    """Transform flat data to sequences for RNN models."""
    ...
```

### Processors: `process_*()`
Performs processing operations, may have side effects.

```python
def process_batch(batch: list[dict]) -> list[dict]:
    """Process a batch of records."""
    ...
```

---

## Import Paths

**Standard**: Always use full canonical paths from the `src` root.

### Correct Examples
```python
# Good - full canonical paths
from src.models.base import BaseModel, PredictionResult
from src.data.features.compute import momentum
from src.pipeline.stages.scaling import ScalerConfig
from src.config.unified import UnifiedConfig
```

### Avoid
```python
# Avoid - relative or abbreviated paths
from ..base import BaseModel  # Use absolute path
from src.features.compute import momentum  # Wrong path structure
```

### Rationale
- Explicit paths are easier to trace and refactor
- Reduces confusion about module locations
- Better IDE support for navigation and refactoring

---

## Module Naming

**Standard**: Use lowercase with underscores (snake_case) for all module names.

### Correct Examples
```
src/
  models/
    base.py
    boosting/
      xgboost_model.py
      lightgbm_model.py
  feature_selection/
    mrmr_selector.py
    importance_filter.py
  cross_validation/
    purged_kfold.py
    walk_forward.py
```

### Avoid
```
src/
  featureSelection/        # Use feature_selection/
  CrossValidation/         # Use cross_validation/
  models/
    XGBoostModel.py        # Use xgboost_model.py
```

### Rationale
- Follows PEP 8 naming conventions
- Consistent with Python standard library
- Better cross-platform compatibility

---

## Variable Naming

### DataFrames
```python
df_features = ...      # Feature dataframe
df_labels = ...        # Labels dataframe
df_raw = ...           # Raw/unprocessed dataframe
df_train = ...         # Training split
df_val = ...           # Validation split
df_test = ...          # Test split
```

### Arrays
```python
X_train = ...          # Training features (uppercase X for features)
y_train = ...          # Training labels (lowercase y for labels)
X_val = ...            # Validation features
y_val = ...            # Validation labels
```

### Configurations
```python
config = ...           # Generic config object
model_config = ...     # Model-specific config
pipeline_config = ...  # Pipeline config
```

### Models
```python
model = ...            # Generic model instance
base_model = ...       # Base/primary model
meta_model = ...       # Meta-learner model
ensemble = ...         # Ensemble model
```

---

## Constants

**Standard**: Use SCREAMING_SNAKE_CASE for constants.

```python
# Module-level constants
DEFAULT_BATCH_SIZE = 32
MAX_SEQUENCE_LENGTH = 100
SUPPORTED_TIMEFRAMES = ["1m", "5m", "15m", "1h", "4h", "1d"]

# Configuration defaults
DEFAULT_CONFIG = {
    "batch_size": 32,
    "learning_rate": 0.001,
}
```

---

## Standardization Summary

The following changes were made to standardize naming conventions:

### Renamed Classes

| Original Name | New Name | Files Updated |
|--------------|----------|---------------|
| `PredictionOutput` | `PredictionResult` | Core definition and key usages |

**Files updated:**
- `src/models/base.py` - Core class definition renamed
- `src/models/__init__.py` - Export updated, backward compat alias added
- `src/models/boosting/xgboost_model.py` - Updated to use PredictionResult
- `src/models/boosting/lightgbm_model.py` - Updated to use PredictionResult
- `src/inference/pipeline.py` - Updated type hints and instantiations

**Backward Compatibility:**
A `PredictionOutput` alias is maintained in `src/models/base.py` for backward compatibility:
```python
# Backward compatibility alias (deprecated, will be removed in future version)
PredictionOutput = PredictionResult
```

### API Response Classes (Kept as-is)
The following classes in `src/inference/server.py` retain their `*Response` suffix as they are HTTP API models:
- `PredictionResponse`
- `HealthResponse`
- `ModelInfoResponse`
- `MetricsResponse`
- `ErrorResponse`

### Classes Already Following Convention
The codebase already follows the `*Config` convention with 100+ configuration classes properly named.

### Migration Guide for Remaining Files
Other files still using `PredictionOutput` can migrate gradually:

1. Update import: `from src.models.base import PredictionResult`
2. Update type hints: `-> PredictionResult`
3. Update instantiations: `return PredictionResult(...)`

The backward compatibility alias ensures existing code continues to work.

---

## Enforcement

### Pre-commit Hooks
Consider adding custom linting rules to enforce these conventions:

```yaml
# .pre-commit-config.yaml
repos:
  - repo: local
    hooks:
      - id: naming-conventions
        name: Check naming conventions
        entry: python scripts/check_naming.py
        language: python
        files: \.py$
```

### Code Review Checklist
- [ ] Configuration classes use `*Config` suffix
- [ ] Result classes use `*Result` suffix (except API responses)
- [ ] Functions use appropriate prefixes (create_, load_, get_, validate_)
- [ ] Import paths are full canonical paths
- [ ] Module names are snake_case

---

## References
- [PEP 8 - Style Guide for Python Code](https://peps.python.org/pep-0008/)
- [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)
