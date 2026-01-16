# Simplified Configuration System Design

## Executive Summary

The current configuration system has grown organically into a tangled mess with 40+ YAML files, 6+ overlapping Python config classes, and the same values (like `batch_size`) defined in 7+ places. This document proposes a radical simplification: **ONE Python class with sensible defaults, no YAML required**.

---

## Problem Analysis

### Current State: Configuration Chaos

**YAML Files (40+ in `config/` root):**
```
config/
├── global.yaml              # 156 lines - "single source of truth" (it isn't)
├── models/                  # 20 YAML files - per-model defaults
│   ├── xgboost.yaml        # Duplicates defaults from Python
│   ├── lstm.yaml           # Same
│   └── ... (18 more)
├── ensembles/              # 5 YAML files
├── experiments/            # 2 YAML files
├── features/               # 3 YAML files
├── pipeline/               # 2 YAML files
└── training/               # 3+ YAML files
```

**Python Config Classes (6+ overlapping):**
```
src/config/unified.py       → UnifiedConfig (983 lines, 15 nested sections)
src/ml_pipeline/config.py   → MLConfig (329 lines, 84 fields)
src/training/config.py      → ExperimentConfig + ModelConfig (53 lines)
src/models/config/trainer_config.py → TrainerConfig (168 lines, 40+ fields)
src/phase1/pipeline_config.py → PipelineConfig (483 lines, 50+ fields)
src/config/global_config.py → GlobalConfig (YAML loader)
```

**Where `batch_size` is defined:**
1. `config/global.yaml` → `training.batch_size: 256`
2. `src/config/unified.py` → `TrainingSection.batch_size = 256`
3. `src/ml_pipeline/config.py` → `MLConfig.batch_size` (via `_get_global_or_default`)
4. `src/models/config/trainer_config.py` → `TrainerConfig.batch_size` (via `_get_global_or_default`)
5. Individual model YAML files (implicit via training config)
6. Test files with hardcoded values
7. CLI argument defaults

**The Core Problem:** Too many layers of indirection, YAML that duplicates Python defaults, and no single obvious place to change a value.

---

## Proposed Solution: One Config, All Python

### Design Philosophy

1. **Python over YAML** - Defaults in code, not files. IDE autocomplete works. Type checking works.
2. **Flat over Nested** - Most users touch 5-10 params. Don't hide them in 15 nested sections.
3. **Convention over Configuration** - Sensible defaults for 95% of runs.
4. **Runtime Mutability** - Everything changeable without restart.

### The New Config Class

```python
# src/config.py - THE ONLY CONFIG FILE

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

@dataclass
class Config:
    """
    Single configuration class for the ML Model Factory.

    All fields have sensible defaults. Override only what you need.

    Usage:
        # Minimal - just train xgboost
        train("xgboost")

        # With overrides
        train("xgboost", batch_size=512, horizons=[20])

        # Full control
        config = Config(symbol="MGC", models=["xgboost", "lstm"])
        results = train(config)
    """

    # =========================================================================
    # ESSENTIAL (the 10 params users actually touch)
    # =========================================================================

    symbol: str = "MES"
    """Which contract to train. One symbol per run."""

    models: list[str] = field(default_factory=lambda: ["xgboost"])
    """Models to train. Can be single model or list."""

    horizons: list[int] = field(default_factory=lambda: [20])
    """Prediction horizons in bars. Default: 20 bars (~100min at 5min)."""

    timeframe: str = "5min"
    """Primary training timeframe."""

    batch_size: int = 256
    """Training batch size. Reduced automatically on OOM."""

    epochs: int = 100
    """Maximum training epochs. Early stopping usually triggers first."""

    cv_splits: int = 5
    """Cross-validation splits for evaluation."""

    device: str = "auto"
    """Device: 'auto', 'cuda', 'cpu', 'mps'."""

    seed: int = 42
    """Random seed for reproducibility."""

    output_dir: Path = field(default_factory=lambda: Path("experiments/runs"))
    """Where to save results."""

    # =========================================================================
    # SEQUENCE MODELS (only needed for LSTM, TCN, Transformer, etc.)
    # =========================================================================

    sequence_length: int = 60
    """Lookback window for sequence models."""

    # =========================================================================
    # DATA SPLITS (rarely changed from defaults)
    # =========================================================================

    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15

    purge_bars: int = 60
    """Gap between train/val/test to prevent leakage."""

    embargo_bars: int = 1440
    """Additional buffer after test set."""

    # =========================================================================
    # ENSEMBLE CONFIG (only for ensemble training)
    # =========================================================================

    ensemble: bool = False
    """Whether to build ensemble from models."""

    meta_learner: str = "ridge"
    """Meta-learner for stacking: 'ridge', 'logistic', 'mlp', 'xgboost'."""

    # =========================================================================
    # OPTIMIZATION (for hyperparameter tuning runs)
    # =========================================================================

    optimize: bool = False
    """Run Optuna hyperparameter optimization."""

    optuna_trials: int = 100
    """Number of Optuna trials if optimize=True."""

    # =========================================================================
    # ADVANCED (99% of users never touch these)
    # =========================================================================

    early_stopping: int = 15
    """Patience for early stopping."""

    mixed_precision: bool = True
    """Use mixed precision training (faster on GPU)."""

    num_workers: int = 4
    """DataLoader workers."""

    scaler: str = "robust"
    """Feature scaler: 'robust', 'standard', 'minmax', 'none'."""

    calibrate: bool = True
    """Calibrate model probabilities."""

    # =========================================================================
    # INTERNAL (set automatically, users don't touch)
    # =========================================================================

    run_id: str = field(default_factory=lambda: _generate_run_id())
    data_dir: Path = field(default_factory=lambda: Path("data"))

    def __post_init__(self):
        """Validate and normalize."""
        if isinstance(self.output_dir, str):
            self.output_dir = Path(self.output_dir)
        if isinstance(self.data_dir, str):
            self.data_dir = Path(self.data_dir)

        # Validate splits sum to 1.0
        total = self.train_ratio + self.val_ratio + self.test_ratio
        if abs(total - 1.0) > 0.01:
            raise ValueError(f"Split ratios must sum to 1.0, got {total}")

        # Validate models exist
        if isinstance(self.models, str):
            self.models = [self.models]

    def with_overrides(self, **kwargs) -> "Config":
        """Return new config with overrides applied."""
        from dataclasses import replace
        return replace(self, **kwargs)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        from dataclasses import asdict
        d = asdict(self)
        d["output_dir"] = str(self.output_dir)
        d["data_dir"] = str(self.data_dir)
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Config":
        """Create from dictionary."""
        return cls(**d)

    def save(self, path: Path | str | None = None) -> Path:
        """Save config to JSON (for reproducibility)."""
        import json
        path = path or (self.output_dir / self.run_id / "config.json")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        return path

    @classmethod
    def load(cls, path: Path | str) -> "Config":
        """Load config from JSON."""
        import json
        with open(path) as f:
            return cls.from_dict(json.load(f))


def _generate_run_id() -> str:
    """Generate unique run ID."""
    from datetime import datetime
    import secrets
    return f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{secrets.token_hex(2)}"


# =============================================================================
# MODEL-SPECIFIC DEFAULTS (hardcoded, not YAML)
# =============================================================================

MODEL_DEFAULTS: dict[str, dict[str, Any]] = {
    # Boosting models
    "xgboost": {
        "n_estimators": 500,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
    },
    "lightgbm": {
        "n_estimators": 500,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
    },
    "catboost": {
        "iterations": 500,
        "depth": 6,
        "learning_rate": 0.05,
        "l2_leaf_reg": 3.0,
    },

    # Neural models
    "lstm": {
        "hidden_size": 128,
        "num_layers": 2,
        "dropout": 0.2,
        "bidirectional": False,
    },
    "gru": {
        "hidden_size": 128,
        "num_layers": 2,
        "dropout": 0.2,
        "bidirectional": False,
    },
    "tcn": {
        "num_channels": [64, 64, 64],
        "kernel_size": 3,
        "dropout": 0.2,
    },
    "transformer": {
        "d_model": 64,
        "nhead": 4,
        "num_layers": 2,
        "dropout": 0.1,
    },

    # Classical
    "random_forest": {
        "n_estimators": 200,
        "max_depth": 10,
        "min_samples_split": 10,
    },
    "logistic": {
        "C": 1.0,
        "max_iter": 1000,
    },
    "svm": {
        "C": 1.0,
        "kernel": "rbf",
    },
}


def get_model_defaults(model_name: str) -> dict[str, Any]:
    """Get default hyperparameters for a model."""
    return MODEL_DEFAULTS.get(model_name, {}).copy()
```

### Usage Examples

**1. Minimal (80% of use cases):**
```python
from src.config import Config

# Train with all defaults
results = train("xgboost")

# Equivalent to:
results = train(Config(models=["xgboost"]))
```

**2. Common Overrides:**
```python
# Change batch size and horizon
results = train("xgboost", batch_size=512, horizons=[5, 10, 20])

# Different symbol
results = train("xgboost", symbol="MGC")

# Multiple models
results = train(["xgboost", "lstm"], sequence_length=60)
```

**3. Full Control:**
```python
config = Config(
    symbol="MES",
    models=["xgboost", "lstm", "tcn"],
    horizons=[20],
    batch_size=512,
    epochs=200,
    ensemble=True,
    meta_learner="ridge",
)
results = train(config)
```

**4. Runtime Modification:**
```python
config = Config(symbol="MES")

# Create variant for different timeframe
config_15min = config.with_overrides(timeframe="15min")

# Modify in place (for notebooks)
config.batch_size = 1024
```

---

## What to Delete

### YAML Files to Delete (ALL of them)

```bash
# Delete entire config/ directory
rm -rf config/

# That's 40+ YAML files gone:
# - config/global.yaml
# - config/models/*.yaml (20 files)
# - config/ensembles/*.yaml (5 files)
# - config/experiments/*.yaml (2 files)
# - config/features/*.yaml (3 files)
# - config/pipeline/*.yaml (2 files)
# - config/training/*.yaml (3+ files)
# - config/INDEX.md, config/README.md
```

### Python Files to Delete or Consolidate

**DELETE entirely:**
```
src/config/global_config.py         # YAML loader, no longer needed
src/config/validators.py            # Over-engineered validation
src/config/utils.py                 # get_config_value machinery
src/config/constants/               # Re-exports, unnecessary
src/config/models/                  # Re-exports, unnecessary
src/config/pipeline/                # Re-exports, unnecessary
```

**CONSOLIDATE into src/config.py:**
```
src/config/unified.py               # 983 lines -> ~200 lines in new Config
src/ml_pipeline/config.py           # MLConfig -> DELETE
src/training/config.py              # ExperimentConfig -> DELETE
src/models/config/trainer_config.py # TrainerConfig -> merge into Config
src/phase1/pipeline_config.py       # PipelineConfig -> merge into Config
```

**KEEP but simplify:**
```
src/config/__init__.py              # Just exports Config class
src/models/config/                  # Keep model registry stuff, remove config cruft
```

### Lines Removed vs Added

| Component | Current Lines | New Lines | Reduction |
|-----------|---------------|-----------|-----------|
| YAML files | ~1500 lines | 0 | -1500 |
| UnifiedConfig + sections | 983 | 0 | -983 |
| MLConfig | 329 | 0 | -329 |
| TrainerConfig | 168 | 0 | -168 |
| PipelineConfig | 483 | 0 | -483 |
| ExperimentConfig | 53 | 0 | -53 |
| GlobalConfig | ~100 | 0 | -100 |
| validators.py | ~300 | 0 | -300 |
| utils.py | ~200 | 0 | -200 |
| **New Config** | 0 | ~200 | +200 |
| **TOTAL** | ~4116 | ~200 | **-3916 lines** |

---

## Migration Path

### Phase 1: Create New Config (Non-breaking)

1. Add `src/config.py` with new `Config` class
2. Keep all old configs working
3. Add adapter methods to convert old -> new

```python
# Backward compatibility adapters
def config_to_trainer_config(cfg: Config, model_name: str) -> TrainerConfig:
    """Convert new Config to legacy TrainerConfig."""
    return TrainerConfig(
        model_name=model_name,
        horizon=cfg.horizons[0],
        batch_size=cfg.batch_size,
        max_epochs=cfg.epochs,
        early_stopping_patience=cfg.early_stopping,
        random_seed=cfg.seed,
        device=cfg.device,
        # ... etc
    )

def config_to_pipeline_config(cfg: Config) -> PipelineConfig:
    """Convert new Config to legacy PipelineConfig."""
    return PipelineConfig(
        symbols=[cfg.symbol],
        target_timeframe=cfg.timeframe,
        train_ratio=cfg.train_ratio,
        # ... etc
    )
```

### Phase 2: Update Entry Points

Update `train()`, `pipeline run`, and notebooks to accept new Config:

```python
def train(
    model_or_config: str | list[str] | Config,
    **overrides
) -> TrainingResult:
    """Train model(s) with unified config."""

    # Normalize to Config
    if isinstance(model_or_config, str):
        config = Config(models=[model_or_config], **overrides)
    elif isinstance(model_or_config, list):
        config = Config(models=model_or_config, **overrides)
    else:
        config = model_or_config.with_overrides(**overrides)

    # Convert to legacy configs for internal use (temporary)
    trainer_config = config_to_trainer_config(config, config.models[0])

    # ... run training
```

### Phase 3: Delete Legacy (Breaking)

1. Remove all YAML files
2. Remove old config classes
3. Update all internal code to use new Config directly
4. Update tests

**Migration script for users:**
```python
# scripts/migrate_config.py
"""Migrate old YAML config to new format."""

def migrate_yaml_to_config(yaml_path: str) -> Config:
    """Convert old YAML config to new Config object."""
    import yaml
    with open(yaml_path) as f:
        old = yaml.safe_load(f)

    return Config(
        symbol=old.get("symbol", "MES"),
        models=old.get("models", ["xgboost"]),
        horizons=old.get("horizons", {}).get("active", [20]),
        timeframe=old.get("timeframes", {}).get("default_primary", "5min"),
        batch_size=old.get("training", {}).get("batch_size", 256),
        # ... map other fields
    )
```

---

## FAQ

### Q: What about per-model hyperparameters?

**A:** Use `MODEL_DEFAULTS` dict in code. Override at runtime:

```python
from src.config import Config, get_model_defaults

# Get defaults
xgb_params = get_model_defaults("xgboost")
# {'n_estimators': 500, 'max_depth': 6, ...}

# Override
xgb_params["n_estimators"] = 1000

# Pass to training
results = train("xgboost", model_params=xgb_params)
```

### Q: What about complex experiment configs?

**A:** Python is the config format. Use a script:

```python
# experiments/benchmark_2026.py
from src.config import Config

BENCHMARK_CONFIG = Config(
    symbol="MES",
    models=["xgboost", "lightgbm", "catboost", "lstm", "tcn"],
    horizons=[5, 10, 15, 20],
    cv_splits=10,
    ensemble=True,
)

if __name__ == "__main__":
    results = train(BENCHMARK_CONFIG)
```

### Q: What about environment-specific config (dev vs prod)?

**A:** Use environment variables or simple conditionals:

```python
import os

config = Config(
    batch_size=int(os.getenv("BATCH_SIZE", 256)),
    device="cpu" if os.getenv("CI") else "auto",
)
```

### Q: What about YAML for non-programmers?

**A:** We're building an ML factory for ML engineers. They can write Python. If YAML is really needed later, we can add `Config.from_yaml()` that reads a simple flat YAML - but defaults remain in Python.

### Q: How do I know what params are available?

**A:** IDE autocomplete. `Config.` shows all fields with docstrings. No more hunting through 40 YAML files.

---

## Summary

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| Config files | 40+ YAML + 6 Python | 1 Python | 97% fewer files |
| Lines of config code | ~4100 | ~200 | 95% reduction |
| Places batch_size defined | 7+ | 1 | Single source of truth |
| Time to find a setting | Minutes | Seconds | IDE autocomplete |
| Nested config depth | 3-4 levels | 1 level | Flat is better |
| YAML parsing at startup | Yes | No | Faster startup |

**The key insight:** Configuration should be boring. One class, sensible defaults, override what you need. Delete everything else.
