# Code Style & Conventions

## Formatting Tools

| Tool | Config | Purpose |
|------|--------|---------|
| Black | `line-length = 100` | Code formatting |
| Ruff | `pyproject.toml` | Linting (pycodestyle, pyflakes, isort, bugbear) |
| mypy | `check_untyped_defs = true` | Type checking |

## Python Version

- **Minimum**: Python 3.11
- **Type hints**: Use modern syntax (`dict[str, Any]` not `Dict[str, Any]`)

## Naming Conventions

| Type | Convention | Example |
|------|------------|---------|
| Classes | PascalCase | `BaseModel`, `PredictionOutput` |
| Functions/Methods | snake_case | `get_default_config`, `_validate_input` |
| Constants | UPPER_SNAKE_CASE | `LABEL_HORIZONS`, `MAX_DEPTH` |
| Private | Single underscore | `_config`, `_is_fitted` |
| Module files | snake_case | `base.py`, `oof_generator.py` |

## Docstrings (Google Style)

```python
def fit(
    self,
    X_train: np.ndarray,
    y_train: np.ndarray,
    sample_weights: np.ndarray | None = None,
) -> TrainingMetrics:
    """
    Train the model.

    Args:
        X_train: Training features, shape (n_samples, n_features)
        y_train: Training labels, shape (n_samples,)
        sample_weights: Optional sample weights

    Returns:
        TrainingMetrics with training results

    Raises:
        ValueError: If input shapes are invalid
    """
```

## Type Hints

```python
# Modern union syntax (Python 3.10+)
def method(config: dict[str, Any] | None = None) -> str:
    ...

# Use Path for file paths
from pathlib import Path
def save(self, path: Path) -> None:
    ...

# Optional with default
def predict(self, X: np.ndarray, threshold: float = 0.5) -> PredictionOutput:
    ...
```

## Class Structure

```python
class MyModel(BaseModel):
    """
    Short description of class purpose.

    Longer description with usage examples if needed.

    Example:
        >>> model = MyModel(config={'param': 1})
        >>> model.fit(X, y)
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        """Initialize with config."""
        super().__init__(config)
        self._model = None

    # Properties first
    @property
    def model_family(self) -> str:
        """Return model family."""
        return "boosting"

    # =========================================================================
    # ABSTRACT METHODS
    # =========================================================================

    def fit(self, X: np.ndarray, y: np.ndarray) -> TrainingMetrics:
        """Train the model."""
        ...

    # =========================================================================
    # PRIVATE METHODS
    # =========================================================================

    def _validate_config(self) -> None:
        """Validate configuration."""
        ...
```

## Import Order

```python
# 1. Standard library
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

# 2. Third-party
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

# 3. Local imports
from src.models.base import BaseModel, PredictionOutput
from src.utils.logging import get_logger
```

## File Organization

1. Module docstring
2. Imports (stdlib → third-party → local)
3. Constants
4. Type definitions / TypedDicts / Dataclasses
5. Main classes
6. Helper functions
7. `__all__` exports (at bottom)

## File Size Limits

- **Target**: 650 lines
- **Maximum**: 800 lines
- **Action**: If exceeding, split into logical submodules

## Error Handling

```python
# GOOD: Explicit validation with clear message
def process(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        raise ValueError("DataFrame cannot be empty")
    if 'close' not in df.columns:
        raise ValueError(f"Missing required column 'close'. Got: {df.columns.tolist()}")
    return df

# BAD: Swallowing exceptions
def process(df: pd.DataFrame) -> pd.DataFrame:
    try:
        return df['close'] * 2
    except:
        return df  # Silent failure!
```

## Comments

```python
# GOOD: Explain WHY, not WHAT
# Shift by 1 to prevent lookahead bias - feature at bar[i] must not
# use information from bar[i] or later
df['feature'] = calculate_feature(df).shift(1)

# BAD: Explaining obvious code
# Add 1 to x
x = x + 1
```
