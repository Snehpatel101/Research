# Phase 2 Design Decisions - Q&A

**Date:** 2025-12-21
**Project:** Ensemble Trading Pipeline - Phase 2 Architecture

---

## Your Questions Answered

### 1. Model Registry Pattern: How to register and instantiate different model families?

**Answer: Decorator-based Plugin Architecture**

```python
# Models self-register using decorator
@ModelRegistry.register(
    name="xgboost",
    family="boosting",
    description="XGBoost gradient boosting classifier",
    requires_gpu=False
)
class XGBoostModel(BaseModel):
    ...

# Auto-discovery on import
from src.models.boosting import xgboost  # Triggers registration

# Factory instantiation
model = ModelRegistry.create(
    model_name="xgboost",  # or "boosting:xgboost"
    config={'n_estimators': 100, 'max_depth': 6},
    horizon=5,
    feature_columns=['feat1', 'feat2', ...]
)
```

**Why This Approach:**
- ✅ Models are self-contained (no central registry file to maintain)
- ✅ Fail-fast validation (checks BaseModel inheritance at registration time)
- ✅ Easy to add new models (just create file, import it)
- ✅ Metadata tracking (GPU requirements, multivariate support)
- ✅ Auto-discovery (scans `models/{family}/` directories)

**Key Design Decisions:**
1. **Full names:** `family:name` prevents collisions (e.g., `boosting:xgboost` vs `neural:xgboost_nn`)
2. **Short name resolution:** `create('xgboost')` auto-resolves if unambiguous
3. **Validation at registration:** Checks required methods exist (fit, predict, save, load)
4. **Class-level storage:** Registry is static, shared across application

---

### 2. Base Model Interface: What common interface should all models implement?

**Answer: Abstract Base Class with Enforced Contract**

```python
class BaseModel(ABC):
    """All models must implement these methods."""

    @abstractmethod
    def fit(self, X_train, y_train, X_val, y_val, metadata_train, metadata_val) -> Dict:
        """Train the model. Returns training history."""
        pass

    @abstractmethod
    def predict(self, X, metadata) -> PredictionOutput:
        """Generate predictions. Returns standardized output."""
        pass

    @abstractmethod
    def save(self, path: Path):
        """Save model to disk (weights + config + history)."""
        pass

    @abstractmethod
    def load(self, path: Path):
        """Load model from disk. Sets is_fitted=True."""
        pass

    @abstractmethod
    def _build_config(self, config: dict, horizon: int) -> ModelConfig:
        """Build model-specific config dataclass."""
        pass

    @abstractmethod
    def _build_model(self):
        """Initialize underlying model architecture."""
        pass

    # Concrete methods (shared across all models)
    def validate_inputs(self, X, y):
        """Fail-fast validation of input shapes and values."""
        ...

    def get_feature_importance(self) -> Optional[pd.DataFrame]:
        """Optional: return feature importance (tree models)."""
        return None
```

**Standardized Output Format:**

```python
@dataclass
class PredictionOutput:
    predictions: np.ndarray      # Shape: (n,), values: {-1, 0, 1}
    probabilities: np.ndarray    # Shape: (n, 3), probs for each class
    timestamps: pd.DatetimeIndex # Shape: (n,)
    symbols: np.ndarray          # Shape: (n,)
    horizons: np.ndarray         # Shape: (n,), all same value (5 or 20)
    uncertainty: Optional[np.ndarray] = None  # Model-specific

    def to_dataframe(self) -> pd.DataFrame:
        """Convert to DataFrame for analysis."""
        ...
```

**Why This Approach:**
- ✅ **Type safety:** Abstract methods enforce implementation
- ✅ **Fail-fast:** Missing methods detected at class definition time
- ✅ **Consistent outputs:** All models return PredictionOutput dataclass
- ✅ **Metadata propagation:** Timestamps/symbols flow through predictions
- ✅ **Extensibility:** Models can add custom methods (e.g., `get_attention_weights()`)

**Key Design Decisions:**
1. **Abstract _build_config:** Forces each model to define its config dataclass
2. **Metadata in fit/predict:** Preserves temporal info for evaluation
3. **PredictionOutput dataclass:** Standardizes output format across all models
4. **validate_inputs() in base:** Shared validation logic (DRY principle)
5. **Optional feature_importance:** Returns None by default, override for tree models

---

### 3. Data Loading: TimeSeriesDataset design for feeding models

**Answer: Temporal Windowing with Leakage Prevention**

```python
class TimeSeriesDataset:
    """
    Load Phase 1 splits and create windowed sequences.

    Key features:
    - Symbol isolation (no cross-symbol windows)
    - Temporal ordering (past features only)
    - Flexible windowing (configurable sequence length)
    """

    def __init__(self, config: DatasetConfig):
        # Load Phase 1 outputs
        self.train_df = pd.read_parquet(config.train_path)
        self.val_df = pd.read_parquet(config.val_path)
        self.test_df = pd.read_parquet(config.test_path)

        # Create sequences
        self.train_sequences = self._create_sequences(self.train_df, 'train')
        self.val_sequences = self._create_sequences(self.val_df, 'val')
        self.test_sequences = self._create_sequences(self.test_df, 'test')

    def _create_sequences(self, df, split_name):
        """
        Create windowed sequences with zero-leakage guarantee.

        For each symbol:
        1. Sort by datetime (ascending)
        2. For i in [seq_len, len(data)]:
            - Features: df[i-seq_len : i]  (past only)
            - Label: df[i]  (future relative to window)

        Returns:
            X: (n_sequences, seq_len, n_features)
            y: (n_sequences,)
            metadata: DataFrame(datetime, symbol, index)
        """
        sequences_X, sequences_y, sequences_meta = [], [], []

        for symbol in df['symbol'].unique():
            symbol_df = df[df['symbol'] == symbol].sort_values('datetime')

            X_full = symbol_df[self.feature_columns].values
            y_full = symbol_df[self.label_column].values
            times = symbol_df['datetime'].values

            for i in range(self.config.sequence_length, len(X_full)):
                # Window of past features
                X_window = X_full[i - self.config.sequence_length : i]
                y_label = y_full[i]

                # Skip NaN or neutrals (if configured)
                if np.isnan(y_label):
                    continue
                if self.config.exclude_neutrals and y_label == 0:
                    continue

                sequences_X.append(X_window)
                sequences_y.append(y_label)
                sequences_meta.append({'datetime': times[i], 'symbol': symbol, 'index': i})

        return np.array(sequences_X), np.array(sequences_y), pd.DataFrame(sequences_meta)

    def get_split(self, split: str):
        """Get train/val/test sequences."""
        if split == 'train':
            return self.train_sequences
        elif split == 'val':
            return self.val_sequences
        elif split == 'test':
            return self.test_sequences
```

**Why This Approach:**
- ✅ **Zero leakage:** Windows never look forward in time
- ✅ **Symbol isolation:** No cross-symbol contamination
- ✅ **Flexible windowing:** seq_len configurable (1 for boosting, 60 for LSTM)
- ✅ **Metadata preservation:** Timestamps/symbols tracked for predictions
- ✅ **Memory efficient:** Can use memmap for large datasets

**Key Design Decisions:**
1. **Separate sequences per split:** Prevents train/val/test leakage
2. **Symbol-level processing:** Ensures temporal continuity within symbols
3. **Configurable filtering:** exclude_neutrals, include_symbols options
4. **Auto-detect columns:** Feature/label columns inferred from Phase 1 schema
5. **Return (X, y, metadata):** Consistent interface for all models

**Integration with Models:**
```python
# Boosting models (no windowing needed)
dataset_config = DatasetConfig(
    train_path=...,
    sequence_length=1,  # Single bar
    horizon=5
)

# Time series models (windowing for LSTM/Transformer)
dataset_config = DatasetConfig(
    train_path=...,
    sequence_length=60,  # 60 bars = 5 hours of history
    horizon=5
)
```

---

### 4. Training Loop: Reusable training infrastructure vs model-specific?

**Answer: Hybrid Approach - Reusable Orchestration + Model-Specific Loops**

**Reusable Orchestration (Trainer):**

```python
class Trainer:
    """
    Orchestrates training across all model families.

    Responsibilities:
    - Data loading (via TimeSeriesDataset)
    - Model instantiation (via ModelRegistry)
    - MLflow tracking (params, metrics, artifacts)
    - Checkpoint management
    - Evaluation and reporting
    """

    def run_full_pipeline(self):
        # 1. Prepare data
        self.prepare_data()  # Load Phase 1 splits, create sequences

        # 2. Build model
        self.build_model()   # Instantiate via ModelRegistry

        # 3. Train (delegates to model.fit())
        training_results = self.train()

        # 4. Evaluate (delegates to model.predict())
        eval_results = self.evaluate()

        return {'training': training_results, 'evaluation': eval_results}
```

**Model-Specific Training Loop:**

Each model implements its own `fit()` method:

```python
# XGBoost (uses built-in training loop)
def fit(self, X_train, y_train, X_val, y_val, ...):
    self.model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        early_stopping_rounds=self.config.early_stopping_rounds
    )
    return self.model.evals_result()

# LSTM (custom PyTorch training loop)
def fit(self, X_train, y_train, X_val, y_val, ...):
    for epoch in range(self.config.n_epochs):
        # Training step
        train_loss = self._train_epoch(X_train, y_train)

        # Validation step
        val_loss = self._validate_epoch(X_val, y_val)

        # Early stopping check
        if early_stopping.should_stop(val_loss):
            break

    return {'train_loss': train_losses, 'val_loss': val_losses}
```

**Why This Approach:**
- ✅ **Reusable orchestration:** Common workflow (load data, train, evaluate)
- ✅ **Model flexibility:** Each model controls its training loop
- ✅ **Easy to add models:** New models just implement fit()
- ✅ **Consistent interface:** All models return same dict structure

**Key Design Decisions:**
1. **Trainer doesn't know about training loops:** Delegates to model.fit()
2. **Models return standardized dicts:** {'train_loss': [...], 'val_loss': [...]}
3. **Callbacks optional:** Models can use them or implement custom logic
4. **MLflow in Trainer:** Tracking is centralized, not per-model
5. **Checkpoint management:** Trainer saves models, not models themselves

---

### 5. Artifact Management: How to organize model checkpoints, predictions, metrics?

**Answer: Structured Run Directory + MLflow Tracking**

**Directory Structure:**

```
experiments/
├── runs/                              # Training run outputs
│   └── xgboost_h5_20251221_143022/    # run_id (model + timestamp)
│       ├── checkpoints/
│       │   └── model/                 # Model weights + config
│       │       ├── xgboost_model.json
│       │       └── metadata.pkl
│       ├── predictions/
│       │   ├── val_predictions.parquet
│       │   └── test_predictions.parquet
│       ├── metrics/
│       │   └── metrics.json           # All metrics in one file
│       └── plots/
│           ├── confusion_matrix.png
│           └── feature_importance.png
│
├── mlruns/                            # MLflow artifact store
│   ├── 0/                             # Experiment ID
│   │   └── {run_uuid}/
│   │       ├── params/                # Hyperparameters
│   │       ├── metrics/               # Metrics over time
│   │       └── artifacts/             # Linked to runs/
│   └── models/                        # Registered models
│
└── registry/                          # Production models
    ├── xgboost_h5_v1/
    │   ├── model/
    │   └── metadata.json
    └── nhits_h20_v2/
        ├── model/
        └── metadata.json
```

**Artifact Management Flow:**

```python
class Trainer:
    def train(self):
        # 1. Setup run directory
        self.run_dir = output_dir / f"{model_name}_{timestamp}"
        self.run_dir.mkdir(parents=True, exist_ok=True)

        # 2. MLflow tracking
        if self.use_mlflow:
            mlflow.start_run(run_name=self.run_id)
            mlflow.log_params(self.model_config)
            mlflow.log_params({'horizon': self.horizon, 'n_features': self.n_features})

        # 3. Train model
        training_results = self.model.fit(...)

        # 4. Save model checkpoint
        model_path = self.run_dir / "checkpoints" / "model"
        self.model.save(model_path)

        # 5. Log to MLflow
        if self.use_mlflow:
            mlflow.log_artifact(str(model_path))
            for metric_name, values in training_results.items():
                for epoch, value in enumerate(values):
                    mlflow.log_metric(metric_name, value, step=epoch)

    def evaluate(self):
        # 6. Generate predictions
        val_preds = self.model.predict(X_val, meta_val)
        test_preds = self.model.predict(X_test, meta_test)

        # 7. Save predictions
        pred_dir = self.run_dir / "predictions"
        pred_dir.mkdir(exist_ok=True)
        val_preds.to_dataframe().to_parquet(pred_dir / "val_predictions.parquet")
        test_preds.to_dataframe().to_parquet(pred_dir / "test_predictions.parquet")

        # 8. Compute metrics
        val_metrics = evaluator.evaluate(val_preds, y_val)
        test_metrics = evaluator.evaluate(test_preds, y_test)

        # 9. Save metrics
        metrics_path = self.run_dir / "metrics" / "metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump({'val': val_metrics, 'test': test_metrics}, f, indent=2)

        # 10. Log to MLflow
        if self.use_mlflow:
            mlflow.log_metrics({f'val_{k}': v for k, v in val_metrics.items()})
            mlflow.log_metrics({f'test_{k}': v for k, v in test_metrics.items()})
            mlflow.log_artifact(str(pred_dir))
            mlflow.end_run()
```

**Why This Approach:**
- ✅ **Run isolation:** Each training run in separate directory
- ✅ **MLflow integration:** Centralized tracking + UI for comparison
- ✅ **Easy recovery:** All artifacts in run_dir (checkpoints, predictions, metrics)
- ✅ **Production registry:** Separate directory for deployed models
- ✅ **Timestamped runs:** Easy to track experiments over time

**Key Design Decisions:**
1. **run_id = model + timestamp:** Unique, human-readable identifiers
2. **Parquet for predictions:** Efficient storage, easy to load for analysis
3. **JSON for metrics:** Human-readable, version control friendly
4. **MLflow artifacts link to run_dir:** Avoid duplication
5. **Production registry separate:** Clear separation of experiments vs deployed models

---

### 6. Configuration: Extend config.py or separate model configs?

**Answer: Hybrid - Global config.py + Model-Specific YAML Files**

**Global Config (`src/config.py`):**

```python
# Existing Phase 1 config
SYMBOLS = ['MES', 'MGC']
ACTIVE_HORIZONS = [5, 20]
PURGE_BARS = 60
EMBARGO_BARS = 1440

# Phase 2 additions (minimal)
MODELS_DIR = PROJECT_ROOT / "models"
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"

# Model training defaults
DEFAULT_SEQUENCE_LENGTH = 60  # For time series models
DEFAULT_RANDOM_SEED = 42
```

**Model-Specific Configs (`config/models/*.yaml`):**

```yaml
# config/models/xgboost.yaml
model:
  name: "xgboost"
  family: "boosting"

hyperparameters:
  n_estimators: 500
  max_depth: 8
  learning_rate: 0.05
  subsample: 0.8
  colsample_bytree: 0.8
  min_child_weight: 3
  gamma: 0.1
  reg_alpha: 0.1
  reg_lambda: 1.0

training:
  early_stopping_rounds: 20
  random_seed: 42
  verbose: true
  n_jobs: -1

dataset:
  sequence_length: 1  # No windowing for XGBoost
  exclude_neutrals: false
```

```yaml
# config/models/nhits.yaml
model:
  name: "nhits"
  family: "timeseries"

hyperparameters:
  input_size: 60          # Lookback window
  h: 5                    # Horizon (overridden by CLI)
  n_freq_downsample: [2, 1, 1]
  n_pool_kernel_size: [2, 2, 1]
  n_blocks: [1, 1, 1]
  mlp_units: [[512, 512], [512, 512], [512, 512]]
  dropout_prob_theta: 0.5
  activation: 'ReLU'

training:
  max_epochs: 100
  learning_rate: 0.001
  batch_size: 256
  early_stopping_patience: 10
  random_seed: 42

dataset:
  sequence_length: 60
  exclude_neutrals: false
```

**Experiment Configs (`config/experiments/*.yaml`):**

```yaml
# config/experiments/baseline.yaml
experiment:
  name: "baseline_comparison"
  description: "Compare all model families on H5 and H20"

horizons:
  - 5
  - 20

models:
  - name: "xgboost"
    config_path: "config/models/xgboost.yaml"

  - name: "lightgbm"
    config_path: "config/models/lightgbm.yaml"

  - name: "timeseries:nhits"
    config_path: "config/models/nhits.yaml"

data:
  train_path: "data/splits/scaled/train_scaled.parquet"
  val_path: "data/splits/scaled/val_scaled.parquet"
  test_path: "data/splits/scaled/test_scaled.parquet"

mlflow:
  tracking_uri: "experiments/mlruns"
  experiment_name: "baseline"

output:
  base_dir: "experiments/runs"
```

**Config Loading in Code:**

```python
# CLI script
import yaml
from pathlib import Path
from src.config import SPLITS_DIR, set_global_seeds

# Load model config
with open(Path("config/models/xgboost.yaml")) as f:
    config = yaml.safe_load(f)

model_config = config['hyperparameters']
dataset_params = config['dataset']

# Merge with global config
dataset_config = DatasetConfig(
    train_path=SPLITS_DIR / 'scaled' / 'train_scaled.parquet',
    val_path=SPLITS_DIR / 'scaled' / 'val_scaled.parquet',
    test_path=SPLITS_DIR / 'scaled' / 'test_scaled.parquet',
    horizon=args.horizon,  # From CLI
    sequence_length=dataset_params.get('sequence_length', 60),
    exclude_neutrals=dataset_params.get('exclude_neutrals', False)
)

# Set global seeds
set_global_seeds(model_config.get('random_seed', 42))
```

**Why This Approach:**
- ✅ **Separation of concerns:** Global config for project-wide settings, YAML for model-specific
- ✅ **Easy to modify:** Non-developers can tune hyperparameters in YAML
- ✅ **Version control friendly:** YAML files track hyperparameter changes
- ✅ **Experiment reproducibility:** Experiment configs capture full setup
- ✅ **No code changes:** New models just need new YAML files

**Key Design Decisions:**
1. **Don't bloat config.py:** Keep it focused on Phase 1 params
2. **YAML for hyperparameters:** Easier to version and share
3. **Experiment YAML:** Define multi-model experiments declaratively
4. **CLI overrides:** Command-line args override YAML (e.g., --horizon)
5. **Config validation:** DatasetConfig.validate() catches errors early

---

## Summary of Key Architectural Patterns

### 1. Plugin Architecture
- Models self-register via decorators
- Auto-discovery scans model directories
- Factory pattern for instantiation

### 2. Abstract Base Class
- Enforces consistent interface (fit/predict/save/load)
- Shared validation logic in base class
- Standardized output format (PredictionOutput)

### 3. Temporal Dataset
- Symbol-isolated windowing
- Zero-leakage guarantee (past features only)
- Flexible sequence lengths (1 for boosting, 60+ for time series)

### 4. Orchestrated Training
- Trainer handles workflow (data → model → train → evaluate)
- Models implement custom training loops
- MLflow tracks everything automatically

### 5. Structured Artifacts
- Run-specific directories (model + timestamp)
- Consistent structure (checkpoints, predictions, metrics, plots)
- MLflow integration for comparison

### 6. Hybrid Configuration
- Global config.py for project-wide settings
- YAML files for model-specific hyperparameters
- Experiment YAML for multi-model experiments

---

## Trade-offs and Alternatives Considered

### Model Registry: Decorator vs Manual Registration

**Chosen: Decorator-based**
- ✅ Pro: Self-contained, no central file to maintain
- ✅ Pro: Fail-fast validation at import time
- ❌ Con: Slightly more boilerplate (decorator on each model)

**Rejected: Manual registration in registry.py**
- ✅ Pro: All models visible in one place
- ❌ Con: Tight coupling, must edit registry for each new model
- ❌ Con: Easy to forget to register

### Dataset: Pre-compute Sequences vs On-the-fly

**Chosen: Pre-compute in __init__**
- ✅ Pro: Faster training (no repeated windowing)
- ✅ Pro: Memory overhead acceptable for 100k samples
- ❌ Con: Higher initial load time

**Rejected: On-the-fly windowing**
- ✅ Pro: Lower memory usage
- ❌ Con: Slower training (repeated windowing)
- ❌ Con: Harder to implement correctly (leakage risks)

### Training: Reusable Loop vs Model-Specific

**Chosen: Model-specific fit()**
- ✅ Pro: Flexibility for different model families
- ✅ Pro: Models use native training loops (XGBoost, PyTorch)
- ❌ Con: Some code duplication (early stopping logic)

**Rejected: Shared training loop in Trainer**
- ✅ Pro: No duplication
- ❌ Con: Hard to support diverse models (XGBoost vs PyTorch vs TensorFlow)
- ❌ Con: Becomes monolithic and complex

### Configuration: Python Dicts vs YAML

**Chosen: YAML files**
- ✅ Pro: Non-developers can modify
- ✅ Pro: Version control friendly
- ✅ Pro: Experiment reproducibility
- ❌ Con: Requires parsing, less type-safe

**Rejected: Python config dicts**
- ✅ Pro: Type-safe, can use dataclasses
- ❌ Con: Harder for non-developers
- ❌ Con: Config changes require code changes

---

## Next Steps

1. **Implement core infrastructure** (Week 1)
   - BaseModel, ModelRegistry, TimeSeriesDataset
   - Focus on correctness, validation, fail-fast

2. **Implement first model family** (Week 2)
   - Start with XGBoost (simpler than time series)
   - Test end-to-end with real Phase 1 data
   - Validate artifact management, MLflow tracking

3. **Build training infrastructure** (Week 3)
   - Trainer, Evaluator, Callbacks
   - CLI scripts for training
   - Integration tests

4. **Add time series models** (Week 4)
   - N-HiTS, TFT
   - Validate 3D input handling
   - Compare against boosting baselines

5. **Run experiments and tune** (Week 5)
   - Baseline experiments (all models, both horizons)
   - Hyperparameter tuning with Optuna
   - Generate comparison report
   - Lock in production configs

---

**End of Design Decisions Document**
