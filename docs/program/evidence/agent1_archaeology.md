# Agent 1 — Codebase Archaeology & Architecture
**Repo:** /home/user/Research · **Date:** 2026-08-24 · **Method:** static analysis (AST + grep). No code changed.

**Environment caveat (OBSERVED):** the checked-in venv `/home/user/Research/.venv` has **zero third-party
packages installed** (`.venv/bin/python -c "import pandas"` → `ModuleNotFoundError`). Nothing in this repo
could be executed. **Every finding below is static.** Anything requiring runtime is tagged UNKNOWN.

**Doc-volume correction (OBSERVED):** the brief said "~380k lines of markdown". Actual:
`find . -name "*.md" | xargs wc -l` → **42,881 lines** across all markdown. `COMPLETION.md` is
378,901 **bytes** / ~11k lines. Still enormous relative to 156,923 LOC in `src/` (435 files).

---

## 0. TL;DR — What actually blocks arbitrary model composition

| # | Blocker | Severity |
|---|---------|----------|
| B1 | Registering a model via `@register` is **not sufficient**. `get_model_contract()` hard-raises on unknown names and is on the data-prep critical path. A new model needs edits in ≥5 hardcoded tables across 4 packages. | **CRITICAL** |
| B2 | Model capability is encoded in **5 parallel sources of truth** that can and do drift (registry properties, `MODEL_CONTRACTS`, `MODEL_DATA_REQUIREMENTS`, `FEATURE_SET_ALIASES`, `PARAM_SPACES`). | **CRITICAL** |
| B3 | Ensembling is restricted to a **hardcoded 4-entry `meta_learner_map`** in the only live path. The registered `VotingEnsemble`/`StackingEnsemble`/`BlendingEnsemble` classes are unreachable from the pipeline. `MetaLearnerFactory` (the actual plugin system for meta-learners) has zero consumers. | **CRITICAL** |
| B4 | `sequence_length` is read from **5 different places** with different values; standard-mode training uses the global config value while OOF alignment and walk-forward use the per-model contract value. Cross-rank ensembles are therefore aligned against a window length the model was never trained with. | **CRITICAL** |
| B5 | Rank is a 3-way `if/elif` on `data_rank`/`requires_sequences`/`requires_4d`, not a capability negotiation. `requires_sequences` and `requires_4d` are both True for PatchTST/iTransformer — ambiguous encoding. | HIGH |
| B6 | 29-module logical import cycle spanning `src.core` ↔ `src.models` ↔ `src.validation.cv`, held together by **278 function-scoped `from src.*` imports**. Package extraction is impossible without refactor. | HIGH |
| B7 | `PipelineConfig` is a **79-field flat god object**; every new mode/model/strategy adds fields to it. | HIGH |
| B8 | Feature engineering is a **hardcoded imperative sequence of `add_*()` calls**; no feature registry/plugin. | HIGH |
| B9 | Per-model persistence formats are ad-hoc and heterogeneous (`model.pt` vs `model.json`+`metadata.pkl`); no uniform artifact contract. | MEDIUM |

---

## 1. Package boundaries and the real dependency graph

### 1.1 Package-level edges — OBSERVED
Counted with `grep -rhoE "from src\.[a-z_]+" src/<pkg>`:

```
core       → core 81 | models 9  | data 6  | config 5
data       → data 153| core 96   | config 5| models 4 | validation 3 | optimization 2
models     → models 101| core 98 | validation 27 | data 19 | optimization 9 | inference 4 | (src.training 1 ← BROKEN)
optimization→ optimization 18 | core 7 | validation 4 | data 1
validation → validation 32 | models 14 | core 7 | data 5 | optimization 3 | inference 1
inference  → inference 73 | core 28 | data 16 | models 15 | config 3 | validation 1
config     → config 63 | core 17 | data 12 | models 5 | factory 1
cli        → models 18 | validation 11 | cli 7 | core 6 | data 4 | config 2 | factory 1
```

**No layer is respected.** `core` (supposed to be the bottom) imports from `models`, `data`, and `config`.

Concrete inversions (OBSERVED):
- `src/core/utils/config_validator.py:231,315` — `from src.models import ModelRegistry`
- `src/core/utils/config_validator.py:411` — `from src.models.ensemble.validator import ...`
- `src/core/container.py:714,777` — `from src.data.adapters import ...`
- `src/core/config.py:453` — `from src.config.symbol import SymbolConfig`
- `src/core/config.py:546` — `from src.data.pipeline.config_adapter import to_data_config`
- `src/core/protocols.py:16` — `from src.models.base import PredictionResult`
- `src/config/experiment.py:22` — `from src.factory import MLFactory` (config → top-level orchestrator)
- `src/validation/__init__.py:40` — `from src.inference.backtesting import (...)`
- `src/models/evaluation/financial_report.py:20` — `from src.inference.backtesting.equity_curve import ...`

### 1.2 Import cycles — OBSERVED (Tarjan SCC over AST-resolved imports)

**Top-level-only imports:** exactly **1 cycle**, size 3:
`src.core.contracts.data_contract ↔ src.core.contracts.model_contract ↔ src.core.exceptions`

**Including deferred (function-scoped) imports:** **7 cycles**, the largest of size **29**:
```
src.core, src.core.utils, src.core.utils.colab_setup, src.core.utils.config_validator,
src.models, src.models.ensemble, src.models.ensemble.{heterogeneous_stacking, meta_factory,
meta_selection, orchestrator, second_level}, src.models.trainer, src.models.training,
src.models.training.{feature_selection, regime_detector, regime_trainer, training_ops,
unified_orchestrator}, src.models.training.services.{artifact_persistence, data_preparer,
ensemble_service, hyperparameter_tuning, model_training, oof_generation, parallel_training},
src.models.training_utils, src.validation.cv, src.validation.cv.cv_orchestrator
```
Other SCCs: `src.data.pipeline.config.*` (4), `src.models.tracking.*` (3),
`src.core.config ↔ src.data.pipeline.config_adapter` (2),
`src.optimization.feature_selection.{ohlcv_selector, result}` (2).

**INFERRED:** the codebase imports cleanly only because **278 imports were pushed inside function
bodies** to break cycles. Distribution (top offenders): `cli/commands/evaluate.py` 18,
`cli/commands/train.py` 14, `inference/preprocessing_graph.py` 13, `models/training/trainer.py` 12,
`factory.py` 11, `models/training/feature_selection.py` 10, `models/training/training_ops.py` 9.
This is hidden coupling: `src/core`, `src/models`, and `src/validation` are one logical unit.

### 1.3 Broken import — OBSERVED
`src/models/training/regime_detector.py:19` documents
`from src.training.regime_detector import RegimeDetector`. **`src/training/` does not exist**
(`ls src/` → cli, config, core, data, factory.py, inference, models, optimization, pipeline_cli.py, validation).
It is inside a docstring, so it does not crash — but it is copy-paste example code that will fail for any user.

---

## 2. The model abstraction(s)

### 2.1 How many "a model" abstractions exist — OBSERVED: **five**

| # | Source of truth | Location | Populated by | Entries |
|---|---|---|---|---|
| 1 | `BaseModel` ABC + runtime properties | `src/models/base.py:116` | subclassing | 21 classes |
| 2 | `ModelRegistry` | `src/models/registry.py:31` | `@register` decorator | 23 |
| 3 | `MODEL_CONTRACTS` | `src/core/contracts/model_contract.py:229` | **hand-edited dict literal** | 23 |
| 4 | `MODEL_DATA_REQUIREMENTS` | `src/models/config/data_requirements.py:113` | **hand-edited dict literal** | 24 |
| 5a | `FEATURE_SET_ALIASES` | `src/data/pipeline/config/feature_sets/core.py:50` | hand-edited dict | ~20 keys |
| 5b | `PARAM_SPACES` (Optuna) | `src/validation/cv/param_spaces.py:77` | hand-edited dict | 23 |

`MODEL_ADAPTER_MAP` / `MODEL_DATA_RANKS` (`src/core/constants.py:119-165`) are **derived** from
`MODEL_CONTRACTS` via `__getattr__` lazy init — that part is fine, not a 6th source.

### 2.2 The registry is real and well-formed — OBSERVED
`src/models/registry.py` implements `register(name, family, description, aliases)`, `create()`,
`get()`, `list_models()`, `list_family()`, `get_model_info()`, `is_available()`.
`src/models/__init__.py:47-52` auto-imports `boosting, classical, ensemble, neural` to trigger
registration. `get_model_info()` (registry.py:~290) exposes `requires_scaling`,
`requires_sequences`, `requires_4d` — a genuine capability query. This is the good part of the codebase.

### 2.3 Registered models — OBSERVED (23)
| Family | Names | Class : file:line |
|---|---|---|
| boosting | xgboost, lightgbm, catboost | `XGBoostModel` boosting/xgboost_model.py:74; `LightGBMModel` :106; `CatBoostModel` boosting/catboost_model.py:61 |
| classical | logistic, random_forest, svm | classical/logistic.py:41, random_forest.py:35, svm.py:35 |
| neural (all subclass `BaseRNNModel` @ neural/base_rnn.py:174) | lstm, gru, tcn, transformer, patchtst, itransformer, tft, nbeats, inceptiontime, resnet1d | lstm_model.py:88, gru_model.py:93, tcn_model.py:162, transformer_model.py:272, patchtst_model.py:261, itransformer_model.py:279, tft_model.py:518, nbeats_model.py:496, inceptiontime_model.py:317, resnet1d_model.py:340 |
| ensemble | voting, stacking, blending | ensemble/voting.py:47, stacking.py:44, blending.py:34 |
| meta_learner | ridge_meta, mlp_meta, xgboost_meta, calibrated_meta | ensemble/ridge_meta.py:34, mlp_meta.py:33, xgboost_meta.py:28, calibrated_meta.py:35 |

Lookup is **by lowercase string** (`registry.py` `name.lower().strip()`), instantiated by
`ModelRegistry.create(name, config=...)`. Model classes are **never referenced by name outside their
own module** (verified by symbol-reference scan) — dispatch is 100% registry-driven. Good.

### 2.4 Table 3 vs Table 4 drift — OBSERVED (AST diff of both dict literals)
23 shared keys. `requires_scaling`, `input_rank`, `sequence_length`, `mtf_mode` agree on all 23.
**Divergences:**
- `MODEL_DATA_REQUIREMENTS` has an extra key **`"mlp"`** that has no `ModelContract` and **no registered
  model** (`ModelRegistry` has `mlp_meta`, not `mlp`). Calling it → `get_model_requirements("mlp")` succeeds,
  `get_model_contract("mlp")` raises, `ModelRegistry.create("mlp")` raises. Dead/trap entry.
- `max_features`: contracts say `blending/calibrated_meta/mlp_meta/ridge_meta/stacking/xgboost_meta`=60,
  `voting`=200; requirements say `None` for all seven. The contract value is what
  `_run_feature_selection` bounds against — silent inconsistency.

### 2.5 Rank encoding is ambiguous — OBSERVED
`BaseRNNModel.requires_sequences → True` (base_rnn.py:174 block). `PatchTSTModel` and
`iTransformerModel` additionally override `requires_4d → True`
(patchtst_model.py:261, itransformer_model.py:279). So for those two, **both flags are True**, and every
consumer has to know that 4d wins. `TFTModel` does **not** override `requires_4d` (contract says
`input_rank=3`), yet `src/models/training/services/oof_generation.py:229` docstring claims
"4D models (PatchTST, iTransformer, **TFT**)". Doc/code mismatch.

Ensembles compute their own rank via a **duplicated private helper `_rank_from_info`** defined
identically in `ensemble/blending.py` and `ensemble/stacking.py` (AST-extracted, byte-identical bodies),
and a *third, different* implementation inline in `ensemble/voting.py` `requires_4d`
(`ranks == {4}` — requires **all** base models be 4D, whereas the other two return 4 if **any** is).
**Three semantics for "what rank is this ensemble".**

### 2.6 THE BLOCKER — adding a model — OBSERVED
`get_model_contract()` (`model_contract.py:563-580`):
```python
if name_lower not in MODEL_CONTRACTS:
    raise ValueError(f"No contract for model '{model_name}'. Available: {available}")
```
It is called on the **critical path** in: `src/data/adapters/preparation.py:488` (data prep),
`src/data/adapters/registry.py:~105` (`AdapterRegistry.get_for_model`),
`src/validation/cv/oof_alignment.py:432`, `src/models/training/modes/walk_forward.py:411,452,644`,
`src/models/training/trainer.py`, `src/models/training/feature_selection.py`,
`src/models/config/trainer_config.py`, `src/models/config/per_model_config.py`,
`src/factory.py`, `src/inference/bundle.py`, `src/cli/commands/train.py`.

`get_model_requirements()` (`data_requirements.py:663-687`) raises identically.

**To ship a third-party model today you must edit source in 4 packages:**
1. `@register(...)` in your own module (fine) **and** get it imported by `src/models/__init__.py`
2. add a `ModelContract` to `src/core/contracts/model_contract.py:229` (hardcoded dict in `core`)
3. add a `ModelDataRequirements` to `src/models/config/data_requirements.py:113`
4. add a `FEATURE_SET_ALIASES` entry in `src/data/pipeline/config/feature_sets/core.py:50` or silently get "all features"
5. add a `PARAM_SPACES` entry in `src/validation/cv/param_spaces.py:77` or silently get `{}` (no tuning; `get_param_space` uses `.get(name, {})`)

That is the single biggest structural obstacle to a marketplace.

---

## 3. Runtime entry points

| Entry point | Declared where | What it actually invokes | Status |
|---|---|---|---|
| `ensemble-pipeline` console script | `pyproject.toml:108` → `src.pipeline_cli:main` | `src/pipeline_cli.py` (12 lines) → `src.cli.main` → `unified_cli.app` | OBSERVED, alive |
| `python -m src.cli` | `src/cli/__main__.py` | same Typer app | OBSERVED |
| `ml run` / `ml data` / `ml status` / `ml resume` | `src/cli/unified_cli.py:42-47` | `pipeline_app.registered_commands[N].callback` — **index-based** wiring | OBSERVED, brittle |
| `ml train model` / `ml train ensemble` | `unified_cli.py:56` `add_typer(train_app)` | `cli/commands/train.py:303,728` | OBSERVED |
| `ml cv` / `ml walk-forward` / `ml cpcv-pbo` | `unified_cli.py:63-67`, again `registered_commands[0..2]` | `cli/commands/evaluate.py:46,400,706` | OBSERVED |
| `MLFactory(config).run()` | `src/factory.py:159,246` | in-memory pipeline (§4) | OBSERVED — the **primary** path |
| `notebooks/ml_factory_colab.ipynb` (34 cells) | — | imports only `MLFactory`, `ExperimentConfig`, `SymbolConfig`, `FeatureConfig/LabelingConfig/MTFConfig`, `OptunaConfig`, `WalkForwardConfig`, `load_deploy_artifact`, `ConformalPredictor`, `comprehensive_leakage_check` | OBSERVED |
| 31 files in `scripts/` | — | ad-hoc; `serve_model.py` and `batch_inference.py` are the **only** consumers of `ModelServer` / `BatchPredictor` | OBSERVED |

**Brittleness (OBSERVED):** `src/cli/unified_cli.py:42-67` wires 6 top-level commands by
positional index into `pipeline_app.registered_commands[...]` / `evaluate_app.registered_commands[...]`.
Reordering `@pipeline_app.command` decorators in `cli/commands/pipeline.py` silently remaps
`ml run` → `ml data`. Verified the current order is run(94), data(194), status(256), resume(325) and
cv(46), walk-forward(400), cpcv-pbo(706).

---

## 4. Training paths — **two disjoint pipelines**, four modes

### 4.1 Two data pipelines that do not meet — OBSERVED

**Pipeline A — `PipelineRunner` (12 file-based stages).** `src/data/pipeline/runner.py:120`, stage map at
`runner.py:193-205`: data_generation, data_cleaning, feature_engineering, initial_labeling, ga_optimize,
final_labels, create_splits, feature_scaling, build_datasets, validate_scaled, validate, generate_report.
Reachable **only** from `cli/commands/pipeline.py:213 (ml data)` and `:345 (ml resume)`.

**Pipeline B — `MLFactory._run_data_pipeline()` (`src/factory.py:697-851`).** In-memory, **two steps**:
`FeatureEngineer.engineer_features()` (factory.py:746-758) then `TripleBarrierLabeler.create_labels()`
(factory.py:767-790). Plus index normalisation, binary remap, float32 downcast. **No cleaning, no
ingest, no GA, no splits stage, no scaling stage, no validation stage, no reporting.**

`src/factory.py:5` and `:170` docstrings claim "Data Pipeline (via PipelineRunner)". **`grep -n
"PipelineRunner" src/factory.py` returns only those two docstring lines — factory never imports it.**
The CLAUDE.md claim `Raw OHLCV → Pipeline (12 stages)` is **DISPROVEN for the primary path.**

**Dead stage packages (OBSERVED, external-importer count = 0):**
| stage | LOC | external importers |
|---|---:|---:|
| ga_optimize | 1,930 | 0 |
| scaling | 1,982 | 0 |
| reporting | 1,686 | 0 |
| ingest | 1,047 | 0 |
| datasets | 877 | 0 |
| final_labels | 816 | 0 |
| splits | 579 | 0 |
| scaled_validation | 243 | 0 |
| evaluation | 234 | 0 |
| **total** | **~9,394** | |
Their `run_*` entry functions (`run_data_cleaning`, `run_build_datasets`, `run_evaluation`,
`run_feature_engineering`, `run_final_labels`, `run_ga_optimization`, `run_data_generation`,
`run_initial_labeling`, `run_generate_report`, `run_scaled_validation`, `run_feature_scaling`,
`run_create_splits`) each have **0 references outside `src/data/pipeline/`** — they are alive
*only* through `PipelineRunner`, i.e. only through `ml data`. ⚠️ NEEDS REVIEW: they are not
strictly dead (Pipeline A uses them), but they are unreachable from the primary product path.

### 4.2 Four training modes, one god-mixin — OBSERVED
`UnifiedTrainingOrchestrator.train()` (`unified_orchestrator.py:402-441`) dispatches:
```python
mode = TrainingMode(self.config.training_mode)
if   mode == TrainingMode.STANDARD:      self._train_standard(df, additional_dfs)
elif mode == TrainingMode.WALK_FORWARD:  self._train_walk_forward(df, additional_dfs)
elif mode == TrainingMode.REGIME_AWARE:  self._train_regime_aware(df, additional_dfs)
elif mode == TrainingMode.META_LABELING: self._train_meta_labeling(df, additional_dfs)
```
All four bodies live in **one 1,135-line mixin**, `src/models/training/training_ops.py`
(`_train_standard:38`, `_train_walk_forward:412`, `_train_regime_aware:684`, `_train_meta_labeling:784`).

`src/models/training/modes/` contains **only** `walk_forward.py`; its `__init__.py` explicitly says
regime and meta-labeling "are implemented in the canonical training chain … and inline in
training_ops.py, not here." So the modes package is a half-finished extraction.

**Copy-paste duplication (OBSERVED).** The per-model feature-filter + float32-downcast block is
byte-similar in at least three places:
- `training_ops.py:449-467` (walk-forward)
- `training_ops.py:706-720` (regime-aware)
- `unified_orchestrator.py:334+` `_prepare_with_cache` (standard)
Each independently does `drop_cols = [c for c in df.columns if c in all_features and c not in
model_features]` then `df.astype(dict.fromkeys(float64_cols, np.float32))`. A change to feature
routing must be made 3–4×.

**`WalkForwardTrainer` re-implements the trainer.** `modes/walk_forward.py` has its own
`compute_classification_metrics()` (`:137`), its own `_create_sequences()` (`:683`), its own scaler
call (`:400`), its own `ModelRegistry.create` + `model.fit` (`:487-488`), and its own
`_select_features_for_window()` (`:617`). None of this goes through `Trainer`.

### 4.3 Trainer layering
- `src/models/trainer.py` (62 lines) — pure re-export shim → OK.
- `src/models/training/trainer.py:59` — `class Trainer(TrainerFeaturesMixin, TrainerEvaluationMixin,
  TrainerArtifactsMixin)`. Mixin-based, 1,122 lines + 344 (artifacts) + 346 (evaluation) + 554 (features).
- Two different `ModelTrainingResult` dataclasses (§9.2).

---

## 5. Inference / prediction paths and persistence

### 5.1 Five parallel inference façades — OBSERVED

| Class | LOC | External consumers |
|---|---:|---|
| `BundleBuilder` / `ModelBundle` (`builder.py:160`, `bundle.py:189`) | 976 + 1,286 | `src/factory.py:1007-1010`, `tests/test_bundle_roundtrip.py` — **ALIVE** |
| `load_deploy_artifact` (`deploy.py:250`) | 303 | notebook + `scripts/test_notebook.py` — **ALIVE** |
| `UniversalInferencePipeline` (`universal_pipeline.py:108`) | 649 | only `inference/server.py:37` |
| `InferencePipeline` (`pipeline.py:126`) | 458 | only `inference/server.py:39` (fallback) |
| `InferenceOrchestrator` (`orchestrator.py:56`) | 791 | **ZERO** (verified across src/, scripts/, tests/, notebooks/) |
| `BatchPredictor`/`BatchInference` (`batch.py:103,414`) | 689 | only `scripts/batch_inference.py` |
| `ModelServer` (`server.py`) | 584 | only `scripts/serve_model.py` |
| `EnsembleBundle` (`ensemble_bundle.py`) | 990 | referenced from `ensemble_service.py:540-554` docstring/bridge only |
| `WalkForwardBundle`, `RegimeBundle`, `MetaLabelingBundle` | 264+279+386 | **ZERO producers, ZERO consumers** — only `src/inference/__init__.py` re-exports (lines 135/167/185) |

✅ VERIFIED: `DECISIONS.md` items 1 and 2 are accurate. The special-mode bundles have no producer:
`BundleBuilder` never constructs them.

### 5.2 Persistence — heterogeneous, no artifact contract — OBSERVED
- Neural (`neural/base_rnn.py:686-704`): `torch.save({model_state_dict, config, n_features, n_classes,
  seq_len, arch_version}, path/"model.pt")`; load uses `weights_only=False`.
- XGBoost (`boosting/xgboost_model.py:271-289`): `model.save_model(path/"model.json")` +
  `pickle.dump(metadata, path/"metadata.pkl")`; load via `safe_pickle_load`.
- LightGBM / CatBoost / classical: own `save()` at `lightgbm_model.py:326`,
  `catboost_model.py:253`, `logistic.py:223`, `random_forest.py:206`, `svm.py:236`.
- Repo-wide token counts under `src/models` + `src/inference`: `safe_pickle` 69, `joblib` 69,
  `pickle` 45, `json.dump` 36, `.pt` 13, `torch.save` 3, `to_parquet` 3.

There is **no single manifest schema** a third-party model must satisfy — `BaseModel.save/load` take a
`Path` and are free-form. `ModelBundle` re-derives metadata by reflection:
`bundle.py:308-312`
```python
requires_sequences = getattr(model, "requires_sequences", False)
requires_4d       = getattr(model, "requires_4d", False)
sequence_length   = getattr(model, "_config", {}).get("sequence_length", 60)
```
— reaching into a **private** attribute with a magic default. A model that stores seq_len elsewhere
serialises `60` and mis-windows at inference.

---

## 6. Configuration system(s)

### 6.1 Scale — OBSERVED
**145 classes matching `^class .*Config`** across `src/`. `src/config/` alone is 6,846 LOC / 16 files.

### 6.2 Duplicate config class names — OBSERVED
| Name | Definitions |
|---|---|
| `BacktestConfig` | `config/inference.py:236` **and** `inference/backtesting/backtest.py:58` |
| `MTFConfig` | `config/data.py:352`, `config/global_config.py:64`, `inference/preprocessing_graph.py:66` |
| `ScalerConfig` | `config/data.py:220`, `config/global_config.py:125` |
| `OptunaConfig` | `config/training.py:39`, `config/global_config.py:99` |
| `CalibrationConfig` | `config/training.py:142`, `config/global_config.py:83` |
| `GAConfig` | `config/training.py:458`, `config/global_config.py:89` |
| `PurgeEmbargoConfig` | `config/cv.py:84`, `config/global_config.py:25` |
| `SplitConfig` | `config/data.py:500`, `config/global_config.py:18` |
| `FeatureSelectionConfig` | `optimization/feature_selection/config.py:127`, `config/global_config.py:39` |
| `CPCVConfig` | `config/cv.py:203` **and** `validation/cv/cpcv.py:42` |
| `PBOConfig` | `config/cv.py:356` **and** `validation/cv/pbo.py:42` |
| `PositionSizerConfig` | `config/inference.py:313`, `inference/backtesting/position_sizing.py:593` |
| `PreprocessingGraphConfig` | `config/inference.py:370`, `inference/preprocessing_graph.py:206` |
| `EnsembleConfig` | `config/ensemble.py:61`, `models/config/data_requirements.py:605` |
| `MetaLearnerConfig` | `config/ensemble.py:124`, `models/ensemble/meta_factory.py` |
| `ExperimentTrackingConfig` vs `ExperimentConfig` | `config/training.py:401` vs `config/experiment.py:168` |

### 6.3 Which config modules are actually consumed — OBSERVED
External-importer counts (excluding `src/config/` internals):
```
config/base.py        30   config/experiment.py 22   config/training.py 15
config/data.py        10   config/utils.py       9   config/symbol.py     8
config/validators.py   5   config/cv.py          2   config/global_config.py 1
config/ensemble.py     0   config/inference.py   0   config/model_configs.py 0
```
Symbol-level reference counts across `src/`, `scripts/`, `tests/`, `notebooks/` excluding their own
definition files and `src/config/__init__.py`:
```
XGBoostConfig 0  LightGBMConfig 0  CatBoostConfig 0  LSTMConfig 0  GRUConfig 0
TCNConfig 0      TransformerConfig 0  PatchTSTConfig 0
StackingConfig 0 VotingConfig 0    BlendingConfig 0  OOFAlignmentConfig 0
InferenceConfig 0
```
✅ VERIFIED DEAD: `src/config/model_configs.py` (605 LOC), `src/config/ensemble.py` (434 LOC),
and the model/ensemble half of `src/config/inference.py` (482 LOC). ~1,500 LOC of a dead
parallel config layer. `CPCVConfig`/`PBOConfig`/`DSRConfig` appear "used" (25/23/19 hits) but every
hit resolves to the `src/validation/cv/` definitions, not `src/config/cv.py`.

### 6.4 The live config chain — OBSERVED
`ExperimentConfig` (`config/experiment.py:168`, sections `DataSection:67`, `TrainingSection:95`,
`EvaluationSection:133`, `BundlingSection:150`)
→ `.to_pipeline_config()` (`:435`)
→ `PipelineConfig` (`core/config.py:66`) — **79 flat fields** (AST-extracted), covering symbol, models,
CV, walk-forward, regime, meta-labeling, labeling, optuna, calibration, bet-sizing, validation, etc.
→ `TrainerConfig` (`models/config/trainer_config.py`).

`ExperimentConfig.to_trainer_config()` (`:502`), `.to_backtest_config()` (`:534`),
`.to_bundle_config()` (`:552`) — **only reference to `to_trainer_config` in the whole repo is
its own docstring at `config/experiment.py:28`.** ⚠️ NEEDS REVIEW (may be intended public API).

`PipelineConfig.to_phase1_config()` (`core/config.py:551`) is a self-documented deprecated alias
calling `to_data_config()`.

**Blocker:** a new model or ensemble strategy that needs its own knobs must add fields to the
79-field `PipelineConfig` god object. There is no `model_config: dict[str, Any]` escape hatch at
the pipeline level (per-model dicts exist only inside `training_ops`/`OOFRequest`).

---

## 7. Ensemble logic

### 7.1 What is actually reachable — OBSERVED
Live path: `unified_orchestrator._build_ensemble()` (`:494-542`) → `EnsembleService.build_ensemble()`
(`services/ensemble_service.py:75`) → `_train_meta_learner()` (`:346`).

`ensemble_service.py:400-411`:
```python
meta_learner_map: dict[str, type] = {
    "ridge_meta": RidgeMetaLearner, "mlp_meta": MLPMetaLearner,
    "xgboost_meta": XGBoostMeta,   "calibrated_meta": CalibratedMetaLearner,
}
if meta_learner_name not in meta_learner_map:
    raise ValueError(...)
```
**That is the entire ensemble strategy space of the product: 4 stacking meta-learners.**

- `VotingEnsemble` / `StackingEnsemble` / `BlendingEnsemble` (registered, 672 + 1,398 + 516 LOC) are
  **never instantiated by the pipeline**. Only non-doc instantiation anywhere:
  `scripts/benchmark_ensemble.py:135`.
- `src/models/ensemble/orchestrator.py:136 EnsembleOrchestrator` (730 LOC) is described in
  `ensemble/__init__.py:138` as "**THE** single entry point for ensemble training" and in
  `ensemble_service.py:5` as what the service "delegates to". **`ensemble_service.py` imports it only at
  line 554 for the `EnsembleResult` type.** ❌ The delegation claim is DISPROVEN.
- **Four separate hardcoded `meta_learner_map` dicts** exist:
  `ensemble_service.py:400`, `models/ensemble/orchestrator.py:275`, `models/ensemble/orchestrator.py:649`,
  `inference/orchestrator.py:600`.
- `MetaLearnerFactory` + `register_meta_learner` + `create_meta_learner_from_config` +
  `list_meta_learners` (`ensemble/meta_factory.py`, 575 LOC) — the **actual plugin system for
  meta-learners** — have **0 external references** (only `get_meta_learner` is used, at
  `inference/ensemble_bundle.py:578`).
- `SecondLevelStacker` / `build_second_level_stacking` (715 LOC), `MetaLearnerSelector` /
  `select_meta_learner_with_optuna` (517 LOC), `StackingFeatures` / `build_stacking_features` —
  **0 external references**.

### 7.2 Silent failure — OBSERVED
`ensemble_service.py:_train_meta_learner` wraps everything in `except Exception as e: logger.error(...);
return None, {"error": str(e)}`. Any ensemble failure degrades to "no ensemble" with a log line.

### 7.3 How heterogeneous predictions are combined — OBSERVED
1. Each model emits an `OOFPrediction` (`validation/cv/oof_core.py:48`).
2. `OOFAlignmentValidator` (`validation/cv/oof_alignment.py:112`) computes per-model coverage.
   For sequence models start index = `sequence_length - 1` (`:167-169`), and
   `sequence_length` comes from **`get_model_contract(model_name).sequence_length`** (`:432-433`).
3. `align_oof_predictions()` (`:276-342`) intersects to a common range and slices
   `oof_array[offset:end_idx]`.
4. `EnsembleService` builds a `StackingDataset` DataFrame, drops NaN rows (`:376-386`), 80/20
   time split (`:394-395`), fits the meta-learner.

**B4 CRITICAL (OBSERVED):** step 2 uses the **contract** seq_len, but standard-mode windows are cut with
the **global config** seq_len — `src/data/adapters/preparation.py:553,563,578,849` all pass
`sequence_length=self.config.sequence_length` (default 60, `core/config.py:172`), and
`get_model_contract` is used there only for `primary_timeframe` (`preparation.py:488-489`).
Contracts say `tcn=64` and `transformer=128`. Therefore for TCN the alignment offset is computed as 63
when the model actually lost 59 samples, and for `transformer` as 127 vs 59. Meanwhile
`modes/walk_forward.py:411-413,452` **does** use `contract.sequence_length`, and
`validation/cv/oof_sequence.py:27` defines yet another `DEFAULT_SEQUENCE_LENGTH = 60` used at
`oof_generator.py:262` (`config.get("sequence_length", DEFAULT_SEQUENCE_LENGTH)`), and
`inference/bundle.py:312` uses `model._config.get("sequence_length", 60)`.
**Five seq_len sources; three disagree by design.** This independently confirms `DECISIONS.md` item 4
and extends it: it is not merely "results differ by mode", it is a **cross-model OOF misalignment**.

### 7.4 3D OOF re-windows from flattened data — OBSERVED
`services/oof_generation.py:189` flattens `prepared.X_train` 3D→2D (`_flatten_to_2d`, `:90-103`), then
`oof_generator.py:255-265` **re-windows** it via `SequenceOOFGenerator.generate_sequence_oof(seq_len=...)`.
So the carefully-prepared 3D tensor is discarded and rebuilt by a second, independent windowing
implementation. INFERRED: any divergence between `SequenceAdapter` windowing and `oof_sequence`
windowing silently produces OOF predictions that do not correspond to the trained model's inputs.

---

## 8. Feature pipeline

### 8.1 `FeatureEngineer` is a hardcoded call sequence — OBSERVED
`src/data/pipeline/stages/features/engineer.py:310 engineer_features()` runs a literal sequence of
~40 `add_*()` calls (`:380-520`): `add_returns`, `add_price_ratios`, `add_sma`, `add_ema`, `add_rsi`,
`add_macd`, `add_stochastic`, `add_williams_r`, `add_roc`, `add_cci`, `add_mfi`, then
`if self.enable_volatility_features:` (atr, bollinger, keltner, hvol, parkinson, garman_klass,
rogers_satchell, yang_zhang, garch), `if self.enable_volume_features:`, `add_temporal_features`,
`add_regime_features`, `add_microstructure_features`, `add_entropy_features`, `add_wavelet_features`,
`add_mtf_features`. Feature modules: entropy(1,563), volatility(766), volume, momentum, trend,
moving_averages, microstructure, microstructure_proxies, price_features, regime, temporal, wavelets,
numba_functions — total 8,435 LOC under `stages/features/`.

**There is no feature registry.** Toggles are coarse booleans (`enable_volatility_features`,
`enable_volume_features`). Adding a feature = editing `engineer.py`. ❌ blocks "mix-and-match features".

### 8.2 Feature *sets* are name→pattern maps — OBSERVED
`FEATURE_SET_ALIASES` (`data/pipeline/config/feature_sets/core.py:50`) +
`FEATURE_SET_DEFINITIONS` (`.../definitions.py:36`). Resolved at
`src/models/training/features.py:280-345`, which uses **`importlib.import_module(...)` "to avoid
circular imports through `__init__.py` chain"** (`:295-301`) — an explicit acknowledgement of B6.
Aliases cover xgboost/lightgbm/catboost/lstm/gru/mlp/transformer/tcn/patchtst/nbeats/informer +
family names. **Missing: itransformer, tft, resnet1d, inceptiontime, svm, logistic, random_forest** →
those silently fall through to `return None` ("use all features") at `features.py:311-313`.

### 8.3 Feature selection
`FeatureSelectionMixin._run_feature_selection_pipeline` (`models/training/feature_selection.py`, 775 LOC)
+ `optimization/feature_selection/` (ohlcv_selector 776 LOC, plus bootstrap_stability, label_perturbation,
param_sensitivity, lifecycle, registry, economic_value, robustness_scoring, timeframe_budget,
regime_selection). ⚠️ NEEDS REVIEW — I did not trace which governance modules are wired; `DECISIONS.md`
item 3 claims only timeframe_budget / regime_selection / robustness_scoring are live and the rest are
test-only. My orphan scan corroborates for `validation/ticker_portability.py` (0 in-src importers).

---

## 9. Dead code, duplicated abstractions, incomplete implementations

### 9.1 Truly orphaned modules (0 importers anywhere in `src/`) — OBSERVED
| File | LOC |
|---|---:|
| `src/data/pipeline/presets.py` | 520 |
| `src/validation/ticker_portability.py` | 243 |
| `src/data/features/cusum_filter.py` | 141 |
| `src/data/features/frac_diff.py` | 126 |
| `src/models/meta_learner.py` | **150 — verified 0 references repo-wide** |
| `src/models/neural/cnn.py` | 48 |

`src/models/meta_learner.py:11 class MetaLearner` is a **third** meta-labeling implementation
alongside `training_ops._train_meta_labeling` (`:784-1135`) and
`src/data/pipeline/stages/meta_labeling/` (2,586 LOC) / `stages/labeling/meta.py`.

### 9.2 Duplicated dataclasses with drifted schemas — OBSERVED (AST field extraction)
| Class | A | B | Drift |
|---|---|---|---|
| `AdapterResult` | `data/adapters/base.py:70` — 13 fields `[X, y, weights, n_samples, n_features, data_rank, sequence_length, original_indices, n_timeframes, timeframe_names, feature_columns, data_contract, adapter_name]` | `core/interfaces.py:49` — 6 fields `[data, labels, feature_names, original_indices, weights, metadata]` | **Different field names for the same concept.** Only A is produced (`tabular.py`, `sequence.py`, `multi_stream.py`). B is re-exported from `core/__init__.py:248` and used only by the dead `AdapterContract`. |
| `ModelTrainingResult` | `training/services/model_training.py:46` — 9 fields (has `calibration_metrics`) | `training/unified_orchestrator.py:75` — 10 fields (has `oof_prediction`, `training_degraded`) | different payloads, same name, same package |
| `HeterogeneousStackingBuilder` | `models/ensemble/heterogeneous_stacking.py:81` | `validation/cv/oof_stacking.py:337` | **Same class name, two packages.** Only the `validation/cv` one has external consumers. |
| `LabelingStrategy` (ABC) | `data/labeling/base.py:56` | `data/pipeline/stages/labeling/base.py:60` | two labeling frameworks |
| `EnsembleConfig` | `config/ensemble.py:61` | `models/config/data_requirements.py:605` | |
| `_rank_from_info` | `ensemble/blending.py` | `ensemble/stacking.py` | byte-identical bodies |

`core/interfaces.py:32` literally contains the comment "NOTE: AdapterResult is defined in TWO
locations (DOCUMENTED EXCEPTION)" — the duplication is known and accepted, and it has now drifted.

### 9.3 Dead abstract interface — OBSERVED
`core/interfaces.py:405 class AdapterContract(ABC)` — **zero subclasses**. All three real adapters
(`TabularAdapter` tabular.py:33, `SequenceAdapter` sequence.py:31, `MultiStreamAdapter`
multi_stream.py:55) subclass `data/adapters/base.py:234 BaseAdapter`. `AdapterContract` is only
re-exported (`core/__init__.py:246,430`).

### 9.4 Incomplete implementations — OBSERVED
Only **7** `NotImplementedError` in all of `src/`, and **1** TODO/FIXME/HACK comment:
- `models/training/evaluation.py:92,96,102,108` and `models/training/artifacts.py:97` — Protocol stubs
  (`_HasTrainerAttributes`, `_TrainerProtocol`), legitimate.
- `validation/monitoring/drift_detectors.py:279,394` — intentional "use compare() instead".
- `validation/evaluation/cpcv_pbo_evaluator.py:199` — `# TODO: Pass actual group indices from CPCV splitter`.
This codebase is not full of stubs; it is full of **complete-but-unreachable** code.

### 9.5 Modules whose public symbols are referenced only via `__init__` re-export — OBSERVED
44 modules, 16,803 LOC (scan: symbol-level references across `src/`+`scripts/`+`tests/`, excluding
own file and any `__init__.py`). The model implementation files legitimately appear here (registry
dispatch by string). The genuinely suspicious ones: `inference/orchestrator.py` (791),
`ensemble/second_level.py` (715), `ensemble/meta_selection.py` (517),
`inference/meta_labeling_bundle.py` (386), `inference/walk_forward_bundle.py` (264),
`inference/production/monitor.py` (319), `models/neural/lr_finder.py` (510),
`data/pipeline/stages/meta_labeling/run.py` (747), `data/pipeline/stages/reporting/formatters.py` (325).

### 9.6 Test suite reality — OBSERVED
38 test files, 10,043 LOC, **511 `def test_*` functions**. **5 files contain source-grep assertions**
(`read_text()` / `inspect.getsource`, 20 occurrences): `test_config_seams.py`,
`test_d3_feature_index.py`, `test_phase100_lifecycle_registry_drawdown.py`, `test_phases_1_3.py`,
`test_phases_4_11.py`. Those tests assert that *text exists in a file*, not that behaviour holds —
they will pass on dead code. ⚠️ INFERRED: a meaningful share of the "473/473 passing" claims rest on
grep assertions.

---

## 10. Brittle abstractions & model-specific hacks

### 10.1 Hardcoded model-name literals outside model implementations — OBSERVED
`grep -rniE '"(patchtst|itransformer|tft|xgboost|...)"' src/ -l`, excluding
`src/models/{neural,boosting,classical}/` → **68 files**. Highest-value offenders:

| Site | Hack |
|---|---|
| `models/training/training_ops.py:1077-1131` | `_create_meta_model()` — **`if model_name == "logistic" / elif "random_forest" / elif "xgboost" / elif "lightgbm" / elif "catboost" / else logistic`**, each branch building an sklearn/xgb/lgb/cb estimator inline with hardcoded `n_estimators=100, max_depth=3, learning_rate=0.1`. Completely bypasses `ModelRegistry`. |
| `models/training/services/ensemble_service.py:400` | 4-entry `meta_learner_map` (see §7.1) |
| `inference/orchestrator.py:600-607` | another `meta_learner_map`, `.get(meta_type, RidgeMetaLearner)` silent default |
| `models/ensemble/orchestrator.py:275, 649` | two more `meta_learner_map` |
| `data/adapters/factory.py:418` | `"multi_stream": ["patchtst", "itransformer"]` hardcoded in a docstring/example, mirroring derived data |
| `data/adapters/factory.py:446` | `4: ["patchtst", "itransformer"]` |
| `validation/cv/param_spaces.py:77` | 23-model hardcoded HP space table; `get_param_space` returns `{}` for unknowns → new models silently untunable |
| `data/pipeline/config/feature_sets/core.py:50` | 20-key model→feature-set alias table; 7 registered models missing |
| `models/training/features.py:416` | comment `- patchtst -> patchtst_optimal (23 features)` |
| `models/ensemble/{blending,stacking}.py` | duplicated `_rank_from_info` |
| `models/ensemble/voting.py` `requires_4d` | third, semantically different rank rule |

### 10.2 Reflection/private-attribute reach-ins — OBSERVED
- `inference/bundle.py:312` `getattr(model, "_config", {}).get("sequence_length", 60)`
- `models/training/features.py:295-301` `importlib.import_module("src.data.pipeline.config.feature_sets")`
  with explicit "avoid circular imports" comment
- `training_ops.py:425-431` `getattr(self.config, "wf_n_windows", self.config.n_splits)` ×4 — the fields
  DO exist on `PipelineConfig`, so the defensive `getattr` only serves to hide typos silently
- `cli/unified_cli.py:42-67` positional `registered_commands[N].callback`

### 10.3 Silent-degradation patterns — OBSERVED
- `registry.py get_model_info/is_available` swallow `ImportError/TypeError/ValueError/RuntimeError/
  AttributeError` and return **`requires_scaling=True, requires_sequences=False, requires_4d=False`
  "safe defaults"** — a model whose constructor fails is reported as a 2D scaled model.
- `ensemble_service._train_meta_learner` → `except Exception: return None`
- `get_param_space` → `{}` for unknown model
- `_resolve_feature_set_columns` → `None` (all features) for unknown model
- `_create_meta_model` → `else: logistic`

---

## 11. Prioritised blocker list for "mix-and-match marketplace"

1. **Collapse the 5 model-metadata tables into the registry.** Make `@register` carry rank, scaling,
   seq_len, mtf mode, min/max features, HP space, feature-set. Delete `MODEL_CONTRACTS`,
   `MODEL_DATA_REQUIREMENTS`, `FEATURE_SET_ALIASES` model keys, `PARAM_SPACES`. Until then, no
   third-party model can be added without editing `src/core`.
2. **Fix seq_len**: single accessor, used by data prep, OOF alignment, walk-forward and bundle
   metadata. Today five sources disagree (§7.3) and cross-rank ensembles are misaligned.
3. **Make ensembling pluggable**: replace the four hardcoded `meta_learner_map` dicts with the
   already-written-but-dead `MetaLearnerFactory`; decide whether
   `Voting/Stacking/Blending` BaseModel classes are the API or delete them.
4. **Break the core↔models↔validation SCC** (29 modules / 278 deferred imports) before attempting any
   package split or plugin entry-point mechanism.
5. **Give `PreparedData` / `AdapterResult` one definition** and one rank field; delete
   `core/interfaces.AdapterResult` and `AdapterContract`.
6. **Extract the four training modes** out of the 1,135-line `training_ops.py` mixin behind one
   `TrainingMode` protocol; the per-model feature-filter block is duplicated 3–4×.
7. **Decide on Pipeline A vs Pipeline B.** ~9,400 LOC of stage code is reachable only from `ml data`,
   while `MLFactory` (the documented primary path) reimplements 2 of the 12 stages inline and its own
   docstring falsely claims to use `PipelineRunner`.
8. **Feature registry** to replace the hardcoded `add_*()` sequence in `engineer.py`.
9. **Uniform artifact contract** (manifest schema) so any model's `save()/load()` is bundle-compatible.
10. **Delete or wire** the ~7,000 LOC of complete-but-unreachable inference code
    (`InferenceOrchestrator` 791, special-mode bundles 929, `server.py` 584, `batch.py` 689,
    `second_level.py` 715, `meta_selection.py` 517, `meta_factory.py` 575, `config/model_configs.py` 605,
    `config/ensemble.py` 434) — it is what makes the repo look 2× its effective size.

---

## 12. Claims checked against source

| Claim (source) | Verdict | Evidence |
|---|---|---|
| "Raw OHLCV → Pipeline (12 stages)" (CLAUDE.md) | ❌ DISPROVEN for primary path | `factory.py:697-851` runs 2 steps; `PipelineRunner` unreferenced by factory |
| "MLFactory uses PipelineRunner" (`factory.py:5,170`) | ❌ DISPROVEN | only docstring mentions; `grep -n PipelineRunner src/factory.py` → 2 doc lines |
| "EnsembleOrchestrator — THE single entry point for ensemble training" (`ensemble/__init__.py:138`) | ❌ DISPROVEN | `ensemble_service.py:400` uses its own map; imports orchestrator only for a type at `:554` |
| "EnsembleService delegates to EnsembleOrchestrator" (`ensemble_service.py:5`) | ❌ DISPROVEN | same |
| "All 12 models are production-ready" (CLAUDE.md) | ⚠️ NEEDS REVIEW | 23 registered incl. ensembles; cannot execute (no deps installed) |
| "All training modes deployable" (DIRECTION.md, per DECISIONS.md) | ❌ DISPROVEN | `WalkForwardBundle`/`RegimeBundle`/`MetaLabelingBundle` have 0 producers |
| DECISIONS.md #1 (serving chain dead) | ✅ VERIFIED | `ModelServer` referenced only by `scripts/serve_model.py` |
| DECISIONS.md #2 (special bundles dead) | ✅ VERIFIED | only `src/inference/__init__.py` re-exports |
| DECISIONS.md #4 (contract seq_len not honored) | ✅ VERIFIED **and worse** | `preparation.py:553,563,578,849` vs `oof_alignment.py:432-433` vs `walk_forward.py:411` vs `oof_sequence.py:27` vs `bundle.py:312` |
| DECISIONS.md #5 (5-dimension Optuna island unused) | ✅ VERIFIED | `run_5d_optimization` referenced only by `optimization/__init__.py` re-export and its own docstrings |
| "Import cycles resolved" (implied by phases) | ⚠️ PARTIAL | top-level clean (1 SCC of 3); logical SCC of 29 remains, masked by 278 deferred imports |
| "212/223/473/~600 tests passing" (CLAUDE.md) | ⚠️ UNKNOWN | 511 static `def test_*`; cannot run (no pandas/torch). 5 files use source-grep assertions |
| "~380k lines of markdown" (task brief) | ❌ CORRECTED | 42,881 lines of `.md`; COMPLETION.md is 378,901 **bytes** |

---

## 13. Things I could not determine (UNKNOWN)

- Whether the pipeline runs at all end-to-end: **no dependencies installed**, nothing executable.
- Actual test pass rate, and how many of the 511 tests exercise behaviour vs grep source text.
- Whether `_generate_4d_oof` produces correctly aligned predictions (needs runtime).
- Which feature-governance modules (`bootstrap_stability`, `lifecycle`, `registry`,
  `economic_value`, `robustness_scoring`, `timeframe_budget`, `regime_selection`) are wired into
  `_run_feature_selection_pipeline()` — needs a read of `feature_selection.py` in full.
- (RESOLVED during review: `wf_n_windows`, `wf_window_type`, `wf_min_train_pct`, `wf_test_pct` DO
  exist on `PipelineConfig` — the `getattr` defensive-defaults at `training_ops.py:425-431` are
  unnecessary but harmless. The defensive `getattr` style is still a smell: it hides typos.)
