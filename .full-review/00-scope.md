# Review Scope

## Target

Cell 5 (labeled "CELL 5: RUN ML FACTORY") in notebooks/ml_factory_colab.ipynb — notebook cell index 7.
This is the main pipeline execution cell that constructs ExperimentConfig and calls MLFactory.run().

The user reports this cell consistently fails. Focus is on **wiring issues** — what's being connected incorrectly between the notebook config and the source code, NOT security or testing.

## Key Config Variables from Cell 2

- MODELS: ["xgboost", "lightgbm", "tcn", "patchtst"]
- TRAINING_MODE: "walk_forward"
- HORIZONS: [10, 20]
- MTF_ENABLED: True, MTF_TIMEFRAMES: ["15min", "30min", "1h"]
- FEATURE_SELECTION_ENABLED: True, method="mda", n_features=60
- OPTUNA_ENABLED: True, n_trials=50
- CONFORMAL_ENABLED: True (but not wired into ExperimentConfig!)
- TICK_VALUE: 0.10 (not wired into ExperimentConfig!)
- Walk-forward: 5 windows, expanding, min_train=0.4, test=0.1

## Files to Investigate

### Notebook
- notebooks/ml_factory_colab.ipynb (Cell 7 = "CELL 5: RUN ML FACTORY")

### Config Classes (what Cell 5 constructs)
- src/config/experiment.py — ExperimentConfig, DataSection, TrainingSection, EvaluationSection, BundlingSection
- src/config/training.py — OptunaConfig
- src/config/data.py — FeatureConfig, LabelingConfig, MTFConfig
- src/config/cv.py — WalkForwardConfig

### Pipeline Entry (what Cell 5 calls)
- src/factory.py — MLFactory class, run() method

### Supporting
- src/config/symbol.py — SymbolConfig (tick_value, commission presets)
- src/models/training/ — training_ops.py, trainer.py

## Flags

- Security Focus: no
- Performance Critical: no
- Strict Mode: no
- Framework: Python ML pipeline
