# Phase 2 Architecture Diagrams

## 1. System Architecture Overview

```mermaid
graph TB
    subgraph "Phase 1 Outputs"
        P1[data/splits/scaled/<br/>train/val/test.parquet]
    end

    subgraph "Phase 2: Data Layer"
        DS[TimeSeriesDataset<br/>- Windowing<br/>- Temporal ordering<br/>- Symbol isolation]
        DL[DataLoaders<br/>- Batch iteration<br/>- Shuffling]
    end

    subgraph "Phase 2: Model Layer"
        REG[ModelRegistry<br/>- Auto-discovery<br/>- Factory pattern<br/>- Validation]

        subgraph "Model Families"
            BOOST[Boosting Models<br/>XGBoost/LightGBM/CatBoost]
            TS[Time Series Models<br/>N-HiTS/TFT/PatchTST]
            NN[Neural Models<br/>LSTM/GRU/Transformer]
        end

        BASE[BaseModel<br/>Abstract Interface<br/>- fit/predict/save/load]
    end

    subgraph "Phase 2: Training Layer"
        TRAIN[Trainer<br/>- Orchestration<br/>- Callbacks<br/>- Checkpointing]
        EVAL[Evaluator<br/>- Metrics<br/>- Predictions<br/>- Reports]
        TUNE[Optuna Tuner<br/>- Search spaces<br/>- Cross-validation]
    end

    subgraph "Phase 2: Experiment Tracking"
        MLF[MLflow<br/>- Runs<br/>- Artifacts<br/>- Registry]
        CONF[YAML Configs<br/>- Models<br/>- Experiments]
    end

    subgraph "Phase 2: Outputs"
        OUT1[experiments/runs/<br/>- Checkpoints<br/>- Predictions<br/>- Metrics]
        OUT2[experiments/registry/<br/>Production models]
    end

    P1 --> DS
    DS --> DL
    DL --> TRAIN

    CONF --> TRAIN
    CONF --> REG

    REG --> BOOST
    REG --> TS
    REG --> NN

    BOOST --> BASE
    TS --> BASE
    NN --> BASE

    BASE --> TRAIN
    TRAIN --> EVAL
    TRAIN --> TUNE

    TRAIN --> MLF
    EVAL --> MLF
    TUNE --> MLF

    MLF --> OUT1
    EVAL --> OUT1
    TRAIN --> OUT2

    style P1 fill:#e1f5ff
    style BASE fill:#fff4e1
    style REG fill:#fff4e1
    style TRAIN fill:#e8f5e9
    style MLF fill:#f3e5f5
    style OUT1 fill:#fce4ec
```

## 2. Model Registration & Instantiation Flow

```mermaid
sequenceDiagram
    participant User
    participant Registry as ModelRegistry
    participant Module as models/boosting/xgboost.py
    participant Base as BaseModel
    participant Model as XGBoostModel

    Note over Module: At import time
    Module->>Registry: @register("xgboost", family="boosting")
    Registry->>Registry: Validate BaseModel inheritance
    Registry->>Registry: Store in _registry["boosting:xgboost"]

    Note over User: At runtime
    User->>Registry: create("xgboost", config, horizon, features)
    Registry->>Registry: Resolve name -> "boosting:xgboost"
    Registry->>Registry: Validate config dict
    Registry->>Model: __init__(config, horizon, features)
    Model->>Base: super().__init__(...)
    Base->>Base: validate_inputs()
    Base->>Model: _build_config(config, horizon)
    Model->>Model: Create XGBoostConfig dataclass
    Model->>Base: Return XGBoostConfig
    Base->>Base: config.validate()
    Base->>Model: _build_model()
    Model->>Model: Initialize XGBClassifier
    Model->>User: Return XGBoostModel instance
```

## 3. Training Pipeline Flow

```mermaid
sequenceDiagram
    participant CLI as scripts/train_model.py
    participant Trainer
    participant Dataset as TimeSeriesDataset
    participant Model as BaseModel
    participant MLflow
    participant Evaluator

    CLI->>Trainer: __init__(model_name, config, dataset_config)
    Trainer->>Trainer: Create run_dir

    CLI->>Trainer: prepare_data()
    Trainer->>Dataset: __init__(dataset_config)
    Dataset->>Dataset: Load train/val/test parquet
    Dataset->>Dataset: Create windowed sequences
    Dataset->>Trainer: Return dataset

    CLI->>Trainer: build_model()
    Trainer->>Model: ModelRegistry.create(...)
    Model->>Trainer: Return model instance

    CLI->>Trainer: train()
    Trainer->>MLflow: start_run()
    Trainer->>MLflow: log_params(config)
    Trainer->>Dataset: get_split('train')
    Trainer->>Dataset: get_split('val')
    Trainer->>Model: fit(X_train, y_train, X_val, y_val)
    Model->>Model: validate_inputs()
    Model->>Model: Training loop with early stopping
    Model->>Trainer: Return training_results
    Trainer->>MLflow: log_metrics(training_results)
    Trainer->>Model: save(run_dir/model)

    CLI->>Trainer: evaluate()
    Trainer->>Evaluator: __init__(model)
    Trainer->>Model: predict(X_val)
    Trainer->>Evaluator: evaluate(predictions, y_val)
    Evaluator->>Trainer: Return val_metrics
    Trainer->>Model: predict(X_test)
    Trainer->>Evaluator: evaluate(predictions, y_test)
    Evaluator->>Trainer: Return test_metrics
    Trainer->>MLflow: log_metrics(val/test)
    Trainer->>MLflow: log_artifacts(predictions)
    Trainer->>MLflow: end_run()
    Trainer->>CLI: Return results
```

## 4. Data Flow: Phase 1 → Phase 2

```mermaid
graph LR
    subgraph "Phase 1 Output"
        P1A[train_scaled.parquet<br/>87,094 rows × 126 cols]
        P1B[val_scaled.parquet<br/>18,591 rows × 126 cols]
        P1C[test_scaled.parquet<br/>18,592 rows × 126 cols]
    end

    subgraph "TimeSeriesDataset Processing"
        LOAD[Load Parquet<br/>- Parse datetime<br/>- Filter symbols]
        WIN[Create Windows<br/>- seq_len=60<br/>- Symbol isolation<br/>- Temporal ordering]
        SEQ[Sequences<br/>X: (n, 60, 107)<br/>y: (n,)<br/>meta: DataFrame]
    end

    subgraph "Model Input"
        M1[Boosting Models<br/>Flatten to (n, 60×107)]
        M2[Time Series Models<br/>Use (n, 60, 107) directly]
    end

    P1A --> LOAD
    P1B --> LOAD
    P1C --> LOAD

    LOAD --> WIN
    WIN --> SEQ

    SEQ --> M1
    SEQ --> M2

    style P1A fill:#e1f5ff
    style P1B fill:#e1f5ff
    style P1C fill:#e1f5ff
    style SEQ fill:#e8f5e9
    style M1 fill:#fff4e1
    style M2 fill:#fff4e1
```

## 5. Model Family Architecture

```mermaid
classDiagram
    class BaseModel {
        <<abstract>>
        +config: ModelConfig
        +horizon: int
        +feature_columns: List[str]
        +is_fitted: bool
        +training_history: Dict

        +fit(X_train, y_train, X_val, y_val)*
        +predict(X, metadata)*
        +save(path)*
        +load(path)*
        +validate_inputs(X, y)
        +get_feature_importance()
    }

    class ModelConfig {
        +model_name: str
        +model_family: str
        +horizon: int
        +random_seed: int
        +early_stopping: bool
        +patience: int
        +validate()
    }

    class PredictionOutput {
        +predictions: ndarray
        +probabilities: ndarray
        +timestamps: DatetimeIndex
        +symbols: ndarray
        +horizons: ndarray
        +uncertainty: ndarray
        +to_dataframe()
    }

    class XGBoostModel {
        +model: XGBClassifier
        +_build_config()
        +_build_model()
        +fit()
        +predict()
        +save()
        +load()
        +get_feature_importance()
    }

    class NHiTSModel {
        +model: NHiTS
        +_build_config()
        +_build_model()
        +fit()
        +predict()
        +save()
        +load()
    }

    class LSTMModel {
        +model: LSTM
        +_build_config()
        +_build_model()
        +fit()
        +predict()
        +save()
        +load()
    }

    class ModelRegistry {
        -_registry: Dict
        -_metadata: Dict
        +register(name, family, desc)$
        +create(model_name, config, horizon)$
        +list_models(family)$
        +get_metadata(name)$
    }

    BaseModel <|-- XGBoostModel
    BaseModel <|-- NHiTSModel
    BaseModel <|-- LSTMModel
    BaseModel --> ModelConfig
    BaseModel --> PredictionOutput
    ModelRegistry --> BaseModel
```

## 6. Configuration Hierarchy

```mermaid
graph TB
    subgraph "Global Config"
        CONF[src/config.py<br/>- SYMBOLS<br/>- HORIZONS<br/>- PURGE/EMBARGO<br/>- Paths]
    end

    subgraph "Model Configs"
        XGB[config/models/xgboost.yaml<br/>- n_estimators<br/>- max_depth<br/>- learning_rate]
        NHITS[config/models/nhits.yaml<br/>- input_size<br/>- hidden_layers<br/>- dropout]
        LSTM[config/models/lstm.yaml<br/>- hidden_size<br/>- num_layers<br/>- dropout]
    end

    subgraph "Experiment Configs"
        BASE[config/experiments/baseline.yaml<br/>- Models to run<br/>- Horizons<br/>- Data paths]
        PROD[config/experiments/production.yaml<br/>- Best hyperparameters<br/>- Ensemble weights]
    end

    subgraph "Runtime"
        TRAIN[Trainer Instance<br/>- Merge configs<br/>- Validate<br/>- Execute]
    end

    CONF --> XGB
    CONF --> NHITS
    CONF --> LSTM

    XGB --> BASE
    NHITS --> BASE
    LSTM --> BASE

    BASE --> TRAIN
    PROD --> TRAIN
    CONF --> TRAIN

    style CONF fill:#e1f5ff
    style BASE fill:#e8f5e9
    style TRAIN fill:#fff4e1
```

## 7. Experiment Tracking & Artifacts

```mermaid
graph TB
    subgraph "Training Run"
        RUN[Trainer.run_full_pipeline]
    end

    subgraph "MLflow Tracking"
        PARAMS[Parameters<br/>- model_name<br/>- horizon<br/>- hyperparameters]
        METRICS[Metrics<br/>- train_loss<br/>- val_loss<br/>- accuracy/F1]
        ARTIFACTS[Artifacts<br/>- model checkpoint<br/>- predictions.parquet<br/>- plots]
    end

    subgraph "File System"
        RUNDIR[experiments/runs/xgboost_h5_20251221_143022/<br/>├── checkpoints/<br/>│   └── model/<br/>├── predictions/<br/>│   ├── val_predictions.parquet<br/>│   └── test_predictions.parquet<br/>└── metrics/<br/>    └── metrics.json]
    end

    subgraph "MLflow UI"
        UI[Web Interface<br/>- Compare runs<br/>- View metrics<br/>- Download artifacts]
    end

    RUN --> PARAMS
    RUN --> METRICS
    RUN --> ARTIFACTS

    PARAMS --> RUNDIR
    METRICS --> RUNDIR
    ARTIFACTS --> RUNDIR

    RUNDIR --> UI

    style RUN fill:#e8f5e9
    style RUNDIR fill:#fce4ec
    style UI fill:#f3e5f5
```

## 8. Hyperparameter Tuning Flow

```mermaid
sequenceDiagram
    participant User
    participant Tuner as OptunaModelTuner
    participant Optuna
    participant Trainer
    participant Model

    User->>Tuner: __init__(model_name, search_space_fn, n_trials)
    User->>Tuner: tune()

    Tuner->>Optuna: create_study(direction='maximize')

    loop n_trials times
        Optuna->>Tuner: objective(trial)
        Tuner->>Tuner: search_space_fn(trial) -> config
        Tuner->>Trainer: __init__(model_name, config)
        Tuner->>Trainer: prepare_data()
        Tuner->>Trainer: build_model()
        Tuner->>Trainer: train()
        Trainer->>Model: fit()
        Trainer->>Tuner: Return training results
        Tuner->>Trainer: evaluate()
        Trainer->>Tuner: Return val_metrics
        Tuner->>Tuner: Extract metric_value (e.g., val_f1)
        Tuner->>Optuna: Return metric_value
        Optuna->>Optuna: Update best_params
    end

    Optuna->>Tuner: Return study
    Tuner->>User: Return best_params, best_value

    style Tuner fill:#f3e5f5
    style Optuna fill:#fff4e1
```

## 9. Phase 2 Milestone Timeline

```mermaid
gantt
    title Phase 2 Implementation Timeline
    dateFormat  YYYY-MM-DD
    section Infrastructure
    BaseModel + Registry           :2025-12-22, 3d
    TimeSeriesDataset             :2025-12-25, 2d
    Trainer + Evaluator           :2025-12-27, 3d

    section Boosting Models
    XGBoost Implementation        :2025-12-30, 2d
    LightGBM Implementation       :2026-01-01, 2d
    CatBoost Implementation       :2026-01-03, 2d

    section Time Series Models
    N-HiTS Implementation         :2026-01-05, 3d
    TFT Implementation            :2026-01-08, 3d
    PatchTST Implementation       :2026-01-11, 3d

    section Experiments
    Baseline Experiments          :2026-01-14, 4d
    Hyperparameter Tuning         :2026-01-18, 5d
    Production Model Selection    :2026-01-23, 3d
```

---

## Key Design Principles Visualized

### Modularity (650-line limit)
```
├── base.py              (~250 lines)  ✓
├── registry.py          (~180 lines)  ✓
├── dataset.py           (~200 lines)  ✓
├── trainer.py           (~200 lines)  ✓
└── models/
    ├── xgboost.py       (~180 lines)  ✓
    ├── nhits.py         (~220 lines)  ✓
    └── lstm.py          (~190 lines)  ✓
```

### Fail-Fast Validation Points
```
1. Config validation  → ModelConfig.validate()
2. Input validation   → BaseModel.validate_inputs()
3. Registry check     → ModelRegistry.create()
4. Data loading       → DatasetConfig.validate()
5. Training           → Trainer.train()
```

### Zero-Leakage Guarantees
```
Phase 1: Purge (60) + Embargo (1440) at split boundaries
         ↓
TimeSeriesDataset: Symbol isolation + Temporal windows
         ↓
Phase 2 Models: Only past features in sequences
```

