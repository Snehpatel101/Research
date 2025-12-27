# Config Loading Error Handling - Implementation Summary

## Overview

Fixed silent config loading failures in `/home/user/Research/src/models/config.py` and updated callers to fail hard when user explicitly provides config files.

## Changes Made

### 1. Added `ConfigError` Exception Class

**Location:** `/home/user/Research/src/models/config.py:408-410`

```python
class ConfigError(Exception):
    """Raised when configuration loading or parsing fails."""
    pass
```

**Purpose:** Distinguish between validation errors (`ConfigValidationError`) and loading/parsing errors (`ConfigError`).

**Exported:** Added to `__all__` list for public API usage.

---

### 2. Updated `load_yaml_config()` - Explicit vs Implicit Loading

**Location:** `/home/user/Research/src/models/config.py:148-196`

**New Signature:**
```python
def load_yaml_config(
    path: Union[str, Path],
    explicit: bool = False,  # NEW PARAMETER
) -> Dict[str, Any]:
```

**Behavior:**

| Scenario | `explicit=False` (Auto-Discovery) | `explicit=True` (User-Requested) |
|----------|-----------------------------------|----------------------------------|
| File not found | Raises `FileNotFoundError` | Raises `ConfigError` with actionable message |
| Invalid YAML | Raises `yaml.YAMLError` | Raises `ConfigError` with parsing details |
| Empty file | Warns, returns `{}` | Warns, returns `{}` |

**Example Error Messages:**

**File Not Found (explicit=True):**
```
ConfigError: Configuration file not found: /path/to/config.yaml
Suggestion: Check that the file exists and the path is correct.
```

**Invalid YAML (explicit=True):**
```
ConfigError: Failed to parse YAML configuration from /path/to/config.yaml
Error: mapping values are not allowed here
Suggestion: Check that the file contains valid YAML syntax.
```

---

### 3. Updated `load_model_config()` - Model-Specific Config Loading

**Location:** `/home/user/Research/src/models/config.py:199-252`

**New Signature:**
```python
def load_model_config(
    model_name: str,
    config_dir: Optional[Path] = None,
    flatten: bool = True,
    explicit: bool = False,  # NEW PARAMETER
) -> Dict[str, Any]:
```

**Behavior:**
- `explicit=False`: Missing model configs log warnings (auto-discovery mode)
- `explicit=True`: Missing model configs raise `ConfigError` (user-requested mode)

**Example Error Message (explicit=True):**
```
ConfigError: Model configuration not found for 'xgboost'
Expected location: /home/user/Research/config/models/xgboost.yaml
Suggestion: Check that the model name is correct and the config file exists.
```

---

### 4. Updated `build_config()` - Fail Hard for Explicit Config Files

**Location:** `/home/user/Research/src/models/config.py:368-456`

**Updated Logic:**

```python
# Auto-discovery (model YAML): WARN on failure
model_yaml = load_model_config(model_name, flatten=True, explicit=False)

# User-provided config file: FAIL HARD on errors
if config_file:
    file_config = load_yaml_config(config_file, explicit=True)  # FAIL HARD
```

**Behavior:**
- **Model config auto-discovery:** Warns and continues with defaults
- **Explicit config_file argument:** Raises `ConfigError` immediately

**Example Error Message:**
```
ConfigError: Failed to load configuration from /tmp/invalid_config.yaml
Error: file not found
Suggestion: Check that the file exists and has valid YAML syntax.
```

---

### 5. Updated `get_model_info()` - Optional Fail-Hard Mode

**Location:** `/home/user/Research/src/models/config.py:655-702`

**New Signature:**
```python
def get_model_info(
    model_name: str,
    explicit: bool = False,  # NEW PARAMETER
) -> Dict[str, Any]:
```

**Behavior:**
- `explicit=False`: Returns fallback dict with "unknown" values (default)
- `explicit=True`: Raises `ConfigError` if model config not found/invalid

---

### 6. Updated `create_trainer_config()` - Propagates ConfigError

**Location:** `/home/user/Research/src/models/config.py:459-526`

**Updated Docstring:**
```python
def create_trainer_config(...) -> TrainerConfig:
    """
    ...
    Raises:
        ConfigError: If config_file is provided and loading fails
    """
```

**Behavior:** Calls `build_config()` which will raise `ConfigError` if user-provided `config_file` is invalid.

---

### 7. Updated `train_model.py` Script - Use Config File Argument

**Location:** `/home/user/Research/scripts/train_model.py:367-391`

**Previous Behavior:** `--config` argument was parsed but **never used**.

**New Behavior:**
```python
# Pass config_file to create_trainer_config (will FAIL HARD if invalid)
config_file = PROJECT_ROOT / args.config if args.config else None

try:
    trainer_config = create_trainer_config(
        model_name=args.model,
        horizon=args.horizon,
        cli_args=config_overrides,
        config_file=config_file,  # NOW ACTUALLY USED
    )
except Exception as e:
    logger.error(f"Configuration error: {e}")
    return 1
```

**Result:** Invalid config files now cause immediate script failure with clear error message.

---

## Testing

### Manual Testing Script

Created `/home/user/Research/test_config_errors.py` to demonstrate behavior:

**Test Cases:**
1. Explicit config file not found → `ConfigError`
2. Implicit config file not found → `FileNotFoundError`
3. `build_config()` with explicit invalid file → `ConfigError`
4. `build_config()` without explicit file (auto-discovery) → Warns, continues
5. Invalid YAML syntax with `explicit=True` → `ConfigError`

**Run tests:**
```bash
python test_config_errors.py
```

---

## Backward Compatibility

**Maintained:**
- Default behavior (`explicit=False`) unchanged for auto-discovery scenarios
- Existing code calling `load_yaml_config(path)` continues to work
- Model config auto-discovery still warns (doesn't fail) when configs missing

**Breaking Changes:**
- None (new parameter defaults to `False`)

**New Requirement:**
- Code explicitly passing `config_file` parameter must handle `ConfigError` exceptions

---

## File Statistics

| File | Lines | Functions Modified |
|------|-------|-------------------|
| `/home/user/Research/src/models/config.py` | 729 | 5 |
| `/home/user/Research/scripts/train_model.py` | 420 | 1 (main) |

---

## Example Usage

### CLI with Invalid Config File (Now Fails Hard)

**Before:**
```bash
$ python scripts/train_model.py --model xgboost --config /tmp/missing.yaml
# Silently ignored, used defaults
```

**After:**
```bash
$ python scripts/train_model.py --model xgboost --config /tmp/missing.yaml
ERROR | Configuration error: Configuration file not found: /tmp/missing.yaml
Suggestion: Check that the file exists and the path is correct.
# Script exits with code 1
```

### Programmatic Usage

```python
from src.models.config import build_config, ConfigError

# Auto-discovery (warns on failure)
config = build_config(
    model_name="xgboost",
    # No config_file provided - auto-discovers, warns if missing
)

# Explicit config file (fails hard on error)
try:
    config = build_config(
        model_name="xgboost",
        config_file="/path/to/custom_config.yaml",  # User-provided
    )
except ConfigError as e:
    print(f"Failed to load config: {e}")
    # Handle error appropriately
```

---

## Error Message Quality

All `ConfigError` messages follow this format:

1. **Problem description:** What failed (file not found, invalid YAML, etc.)
2. **Context:** Full absolute path to problematic file
3. **Suggestion:** Actionable fix (check file exists, check YAML syntax, etc.)

**Example:**
```
ConfigError: Failed to parse YAML configuration from /home/user/Research/config/models/xgboost.yaml
Error: mapping values are not allowed here
  in "<unicode string>", line 5, column 10
Suggestion: Check that the file contains valid YAML syntax.
```

---

## Key Design Decisions

1. **Explicit vs Implicit:** New `explicit` parameter distinguishes user-requested configs (fail hard) from auto-discovery (warn and continue)

2. **No Breaking Changes:** Default `explicit=False` maintains backward compatibility

3. **Clear Error Messages:** All `ConfigError` messages include:
   - Absolute path (not relative)
   - Root cause from underlying exception
   - Actionable suggestion for fix

4. **Exception Hierarchy:**
   - `ConfigError`: Loading/parsing failures (file not found, invalid YAML)
   - `ConfigValidationError`: Schema/validation failures (invalid values, missing required fields)

5. **train_model.py Integration:** Now actually uses `--config` argument and handles `ConfigError` gracefully

---

## Validation

**Syntax Check:**
```bash
python -m py_compile src/models/config.py
python -m py_compile scripts/train_model.py
# ✓ Both files have valid Python syntax
```

**Import Check:**
```python
from src.models.config import ConfigError  # ✓ Available
from src.models import config
assert "ConfigError" in config.__all__  # ✓ Exported
```

---

## Future Improvements

1. **Type Hints:** Add `TypedDict` for config structure validation
2. **Config Schema:** JSON Schema or Pydantic models for strict validation
3. **Better YAML Error Reporting:** Parse YAML errors to highlight specific line/column
4. **Config File Validation:** Pre-validate config files before attempting load

---

## Summary

**Problem:** Config loading failures were silently caught and logged as warnings, making debugging difficult when users explicitly provided invalid config files.

**Solution:** Added `explicit` parameter to distinguish user-requested configs (fail hard with clear error) from auto-discovery (warn and continue). Updated CLI to use this mechanism.

**Result:**
- User-provided config files (`--config` argument) now fail immediately with actionable error messages
- Auto-discovery of model configs still warns gracefully
- No breaking changes to existing code
- Clear, actionable error messages guide users to fix

**Files Modified:**
- `/home/user/Research/src/models/config.py` (729 lines)
- `/home/user/Research/scripts/train_model.py` (420 lines)
