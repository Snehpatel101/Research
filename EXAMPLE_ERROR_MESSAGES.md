# Config Error Messages - Before & After

## Scenario 1: User Provides Invalid Config File Path

### Before (Silent Warning)
```bash
$ python scripts/train_model.py --model xgboost --horizon 20 --config /tmp/missing.yaml
2025-12-27 10:30:15 | WARNING  | src.models.config | Failed to load /tmp/missing.yaml: [Errno 2] No such file or directory: '/tmp/missing.yaml'
2025-12-27 10:30:16 | INFO     | __main__ | Starting training: model=xgboost, horizon=20
# Training continues with defaults (user may not notice warning)
```

### After (Fail Hard)
```bash
$ python scripts/train_model.py --model xgboost --horizon 20 --config /tmp/missing.yaml
2025-12-27 10:30:15 | ERROR    | __main__ | Configuration error: Configuration file not found: /tmp/missing.yaml
Suggestion: Check that the file exists and the path is correct.
# Script exits with code 1
```

---

## Scenario 2: Invalid YAML Syntax in User Config

### Before (Silent Warning)
```bash
$ python scripts/train_model.py --model xgboost --config /tmp/bad_syntax.yaml
2025-12-27 10:30:15 | WARNING  | src.models.config | Failed to load /tmp/bad_syntax.yaml: mapping values are not allowed here
2025-12-27 10:30:16 | INFO     | __main__ | Starting training: model=xgboost, horizon=20
# Training continues, user config silently ignored
```

### After (Fail Hard with Context)
```bash
$ python scripts/train_model.py --model xgboost --config /tmp/bad_syntax.yaml
2025-12-27 10:30:15 | ERROR    | __main__ | Configuration error: Failed to parse YAML configuration from /tmp/bad_syntax.yaml
Error: mapping values are not allowed here
  in "<unicode string>", line 3, column 15
Suggestion: Check that the file contains valid YAML syntax.
# Script exits with code 1
```

---

## Scenario 3: Missing Model Config (Auto-Discovery)

### Before
```bash
$ python scripts/train_model.py --model nonexistent_model --horizon 20
2025-12-27 10:30:15 | WARNING  | src.models.config | Failed to load config/models/nonexistent_model.yaml: [Errno 2] No such file or directory
2025-12-27 10:30:16 | INFO     | __main__ | Starting training: model=nonexistent_model, horizon=20
# Training continues with defaults
```

### After (Same Behavior - Warns and Continues)
```bash
$ python scripts/train_model.py --model nonexistent_model --horizon 20
2025-12-27 10:30:15 | WARNING  | src.models.config | Failed to load default config from config/models/nonexistent_model.yaml: [Errno 2] No such file or directory: 'config/models/nonexistent_model.yaml'. Using built-in defaults.
2025-12-27 10:30:16 | INFO     | __main__ | Starting training: model=nonexistent_model, horizon=20
# Training continues with defaults (auto-discovery mode)
```

**Note:** Auto-discovery warnings are now more descriptive ("Using built-in defaults").

---

## Scenario 4: Programmatic Usage - Explicit Config Loading

### Before
```python
from src.models.config import build_config

# User provides bad config path
config = build_config(
    model_name="xgboost",
    config_file="/tmp/missing.yaml"
)
# Silently logged warning, returned defaults
# User may not realize config wasn't applied
```

### After (Clear Exception)
```python
from src.models.config import build_config, ConfigError

# User provides bad config path
try:
    config = build_config(
        model_name="xgboost",
        config_file="/tmp/missing.yaml"
    )
except ConfigError as e:
    print(f"Config error: {e}")
    # ConfigError: Configuration file not found: /tmp/missing.yaml
    # Suggestion: Check that the file exists and the path is correct.
    raise
```

---

## Scenario 5: Valid Config File (Success Case)

### Before & After (Identical Success Behavior)
```bash
$ python scripts/train_model.py --model xgboost --horizon 20 --config config/custom_xgb.yaml
2025-12-27 10:30:15 | INFO     | src.models.config | Loaded config from config/custom_xgb.yaml: 15 keys
2025-12-27 10:30:15 | DEBUG    | src.models.config | Merged config from config/custom_xgb.yaml
2025-12-27 10:30:16 | INFO     | __main__ | Starting training: model=xgboost, horizon=20
# Training continues with custom config
```

---

## Error Message Anatomy

All `ConfigError` messages follow this structure:

```
ConfigError: <PROBLEM DESCRIPTION>
<CONTEXT (file path, error details)>
Suggestion: <ACTIONABLE FIX>
```

### Example 1: File Not Found
```
ConfigError: Configuration file not found: /home/user/Research/config/missing.yaml
Suggestion: Check that the file exists and the path is correct.
```

### Example 2: Invalid YAML
```
ConfigError: Failed to parse YAML configuration from /home/user/Research/config/bad.yaml
Error: mapping values are not allowed here
  in "<unicode string>", line 5, column 10
Suggestion: Check that the file contains valid YAML syntax.
```

### Example 3: Model Config Not Found
```
ConfigError: Model configuration not found for 'my_custom_model'
Expected location: /home/user/Research/config/models/my_custom_model.yaml
Suggestion: Check that the model name is correct and the config file exists.
```

---

## Key Improvements

1. **Absolute Paths:** All error messages show full absolute paths (not relative)
2. **Root Cause:** Original exception message preserved and displayed
3. **Actionable Suggestions:** Every error tells user how to fix it
4. **Fail Fast:** Explicit config errors stop execution immediately
5. **Backward Compatible:** Auto-discovery still warns gracefully

---

## Testing Commands

```bash
# Test explicit config file not found (should FAIL)
python scripts/train_model.py --model xgboost --config /tmp/nonexistent.yaml

# Test auto-discovery (should WARN and continue)
python scripts/train_model.py --model nonexistent_model_xyz --horizon 20

# Test valid config (should SUCCESS)
echo "learning_rate: 0.05\nn_estimators: 1000" > /tmp/test_config.yaml
python scripts/train_model.py --model xgboost --config /tmp/test_config.yaml --skip-save

# Test invalid YAML syntax (should FAIL)
echo "invalid: yaml: syntax: here:" > /tmp/bad_syntax.yaml
python scripts/train_model.py --model xgboost --config /tmp/bad_syntax.yaml
```

---

## Summary

| Scenario | Before | After |
|----------|--------|-------|
| User-provided config missing | Warn, continue | **Fail hard, exit 1** |
| User-provided config invalid YAML | Warn, continue | **Fail hard, exit 1** |
| Auto-discovered model config missing | Warn, continue | Warn, continue (same) |
| Valid config file | Success | Success (same) |

**Key Change:** User-provided configs (`--config` argument) now fail immediately with clear, actionable error messages instead of silently falling back to defaults.
