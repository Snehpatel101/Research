# Editing Guidelines for Serena

## When to Use Symbolic Editing

**PREFER symbolic editing** when modifying entire functions, methods, or classes:

```python
# Use replace_symbol_body for full method replacement
# Use insert_after_symbol for adding new methods to a class
# Use insert_before_symbol for adding imports or new functions
```

## When to Use Line-Based Editing

**Use line-based editing** when:
- Changing a few lines within a large function
- Adding/removing specific statements
- Modifying configurations or YAML files

## Code Style Requirements (MUST FOLLOW)

### Formatting
- **Line length**: 100 characters
- **Formatter**: Black (run before committing)
- **Linter**: Ruff

### Type Hints
```python
# Use modern syntax (Python 3.11+)
def method(config: dict[str, Any] | None = None) -> str:
    ...
```

### Docstrings (Google-style)
```python
def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> TrainingMetrics:
    """
    Train the model.

    Args:
        X_train: Training features, shape (n_samples, n_features)
        y_train: Training labels, shape (n_samples,)

    Returns:
        TrainingMetrics with training results

    Raises:
        ValueError: If input shapes are invalid
    """
```

### Class Structure
1. Class docstring
2. `__init__`
3. Properties (decorated with `@property`)
4. Abstract methods (if ABC)
5. Public methods
6. Private methods (prefixed with `_`)

## Editing Checklist

Before saving any edit:

- [ ] Type hints added for all public functions
- [ ] Docstrings for classes and public methods
- [ ] No unused imports
- [ ] File stays under 800 lines (target 650)
- [ ] Run `black` and `ruff check`

## Adding New Models

When adding a new model to the registry:

1. Create file in appropriate family directory (`boosting/`, `neural/`, `classical/`)
2. Inherit from `BaseModel`
3. Implement ALL abstract methods:
   - `model_family` (property)
   - `requires_scaling` (property)
   - `requires_sequences` (property)
   - `get_default_config()`
   - `fit()`
   - `predict()`
   - `save()`
   - `load()`
4. Add `@ModelRegistry.register("name", family="family")` decorator
5. Add tests in `tests/models/`

## Adding New Pipeline Stages

1. Create stage in `src/phase1/stages/{category}/`
2. Implement required interface (check existing stages for pattern)
3. Register in pipeline DAG
4. Add tests in `tests/phase_1_tests/stages/`

## Leakage Prevention (CRITICAL)

When editing data processing code:

- **ALWAYS** check for `shift(1)` on any computed features
- **NEVER** use future data in calculations
- **VALIDATE** with lookahead invariance tests
- **PURGE/EMBARGO** must be applied to all CV splits

```python
# CORRECT: shift(1) prevents lookahead
df['feature'] = calculate_something(df).shift(1)

# WRONG: No shift means bar[i] uses bar[i] data (lookahead!)
df['feature'] = calculate_something(df)
```

## Validation After Edits

```bash
# Always run after editing production code
pytest tests/ -v --tb=short
black src tests scripts
ruff check src tests scripts
```
