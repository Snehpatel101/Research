# config/models/ — per-model YAML overrides

This directory is scanned by `create_trainer_config()`
(`src/models/config/merging.py` → `find_model_config()` in
`src/models/config/loaders.py`). Files here override built-in model defaults.

Precedence (highest wins):
1. CLI arguments
2. Explicit `--config <file>`
3. `config/models/<model_name>.yaml` (this directory)
4. `config/global.yaml` `training.*` section
5. Built-in defaults (`src/models/config/defaults.py`)

Example `config/models/xgboost.yaml`:

```yaml
n_estimators: 500
max_depth: 8
learning_rate: 0.03
```

If no file exists for a model, built-in defaults are used silently — this
directory being empty is a valid state, not an error.
