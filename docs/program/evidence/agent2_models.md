# Agent 2 — Time-Series Model Research

**Scope:** Part A = repo inventory of `src/models/` (evidence, file:line). Part B = research on model families and their fit for THIS platform.
**Constraint honored:** no code changes made.
**Environment note:** `/home/user/Research/.venv` is **empty** (no numpy/torch/sklearn installed) — `.venv/bin/python -c "import src.models"` fails at `src/models/boosting/catboost_model.py:20`. All Part A findings are therefore **static-analysis based**, not runtime-verified. Tagged accordingly.

Tags: **OBSERVED** (read in repo) · **INFERRED** (deduced from observed code) · **PROPOSED** (my recommendation) · **UNKNOWN** (could not determine).

---

## PART A — What exists today

### A.0 Headline: the "12 production-ready models" claim is WRONG (undercount, not overcount)

**OBSERVED** — `src/core/constants.py:81-100` is the canonical model registry-of-record:

```python
MODEL_FAMILIES: dict[str, list[str]] = {
    "boosting": ["xgboost", "lightgbm", "catboost"],
    "classical": ["random_forest", "logistic", "svm"],
    "neural": ["lstm", "gru", "tcn", "tft", "nbeats", "inceptiontime", "resnet1d"],
    "transformer": ["transformer", "patchtst", "itransformer"],
    "ensemble": ["voting", "stacking", "blending"],
    "meta_learner": ["ridge_meta", "mlp_meta", "xgboost_meta", "calibrated_meta"],
}
ALL_MODELS: list[str] = [...]
assert len(ALL_MODELS) == 23, f"Expected 23 models, got {len(ALL_MODELS)}"
```

**16 base learners** exist, not 12. `CLAUDE.md`'s table omits four that are fully implemented and registered:

| Omitted from docs | File | Registered at |
|---|---|---|
| `random_forest` | `src/models/classical/random_forest.py` | `:29` |
| `logistic` | `src/models/classical/logistic.py` | `:35` |
| `svm` | `src/models/classical/svm.py` | `:29` |
| `transformer` (vanilla encoder) | `src/models/neural/transformer_model.py` | `:266` |

Plus 3 ensembles and 4 meta-learners registered as first-class `BaseModel` subclasses.

**This is the single most important Part A finding for the marketplace mission.** The two cheap baselines the mission most needs — logistic regression and random forest — **already exist, already implement the full contract, and are already smoke-tested** (`tests/test_model_smoke.py:146-149`, `:379`). They are simply undocumented and, per Part A.6, invisible to several downstream config registries. The work is *wiring and promotion*, not implementation.

### A.1 Is N-BEATS actually implemented? — YES, and it is real

**OBSERVED** — `src/models/neural/nbeats_model.py`, 766 lines. Not a stub.
- `NBEATSBlock` `:39`, `GenericBasis` `:106`, `TrendBasis` `:125` (polynomial basis), `SeasonalityBasis` `:174` (Fourier basis), `NBEATSStack` `:229`, `NBEATSNetwork` `:295` — the genuine doubly-residual stack architecture with interpretable trend/seasonality basis expansion.
- `NBEATSModel(BaseRNNModel)` registered `:490-496`, aliases `n_beats`, `neural_basis_expansion`.
- Adapted to **classification**, not forecasting: `predict()` `:657` applies `torch.softmax(logits, dim=1)` and `_convert_labels_from_class(...)` — the backcast/forecast basis feeds a classification head.
- `fit()` `:617-629` overrides only to capture `self._seq_len = X_train.shape[1]` then delegates to `super().fit(...)`.
- Honestly self-documents a real limitation: `is_production_safe` returns **False** (`:537-547`) because N-BEATS' fully-connected layers are **non-causal** — every position sees every other position in the window. A `_log_bidirectional_warning()` fires at training time.

**INFERRED** — N-BEATS was designed for univariate point forecasting with a strong trend+seasonality inductive bias. Repurposing its basis expansion as a classification backbone on triple-barrier labels is a *plausible but unproven* use. Its contract caps it at `max_features=20` (`src/core/contracts/model_contract.py:432`) — by far the tightest cap of any model — which effectively makes it a low-dimensional specialist, not a general learner.

### A.2 Is there a `src/models/classical/`? — YES

**OBSERVED** — `src/models/classical/` contains exactly three model files plus `__init__.py`:

| File | Class | Lines |
|---|---|---|
| `logistic.py` | `LogisticModel(BaseModel)` `:41` | 310 |
| `random_forest.py` | `RandomForestModel(BaseModel)` `:35` | 277 |
| `svm.py` | `SVMModel(BaseModel)` `:35` | 333 |

`src/models/classical/__init__.py:26-28` imports all three, and `src/models/__init__.py:48-53` imports the `classical` package to trigger `@register` side effects.

Defaults are sane and financially aware:
- `RandomForestModel.get_default_config()` `:62-75`: `n_estimators=200, max_depth=10, min_samples_split=20, min_samples_leaf=10, max_features="sqrt", class_weight="balanced", oob_score=True, n_jobs=-1`.
- `LogisticModel.get_default_config()` `:73-87`: `saga` solver with `l1_ratio` (0.0=L2, 1.0=L1, in-between=elastic net), `C=1.0`, `class_weight="balanced"`. **This is already a Ridge/Lasso/ElasticNet-capable linear baseline** — no new model needed for that family.
- `SVMModel.get_default_config()` `:66-78`: RBF kernel, `probability=True` (Platt scaling), `max_iter=10000`, `class_weight="balanced"`.

**INFERRED** — SVM with RBF + `probability=True` is O(n²)–O(n³) in samples and will be unusable on the 1.6M-row datasets referenced throughout `CLAUDE.md` (Phases 72–80). Its own docstring `:41-44` admits this. Treat SVM as a small-data diagnostic, not a marketplace citizen.

### A.3 Complete model inventory

Legend: rank from `MODEL_CONTRACTS` (`src/core/contracts/model_contract.py`) cross-checked against the class's own `requires_sequences` / `requires_4d` properties.

#### Boosting — 2D tabular

| Model | Class | File:line | Base | Rank | Scaling | Prob | Feat. imp. |
|---|---|---|---|---|---|---|---|
| xgboost | `XGBoostModel` | `boosting/xgboost_model.py:74` | `BaseModel` | 2D | No (`:105`) | via `predict()` | `:315` |
| lightgbm | `LightGBMModel` | `boosting/lightgbm_model.py:106` | `BaseModel` | 2D | No (`:139`) | via `predict()` | `:370` |
| catboost | `CatBoostModel` | `boosting/catboost_model.py:61` | `BaseModel` | 2D | No (`:100`) | via `predict()` | `:298` |

**OBSERVED inconsistency** — CatBoost is the *only* model not decorated at class definition. It is registered **conditionally at import time** (`boosting/catboost_model.py:377-389`):
```python
if CATBOOST_AVAILABLE:
    from ..registry import register
    _registered_class = register(name="catboost", family="boosting", ...)(CatBoostModel)
    CatBoostModel = _registered_class  # type: ignore[assignment,misc]
```
**INFERRED** — this is a legitimate optional-dependency pattern, and `ModelRegistry.is_available()` (`registry.py:407-435`) plus `get_model_info()`'s try/except (`registry.py:322-360`) exist specifically to absorb it. But it means **registry contents are environment-dependent**: `ModelRegistry.count()` returns 22 or 23 depending on whether CatBoost is installed. A marketplace UI must query `is_available()`, never `is_registered()`.

#### Classical — 2D tabular

| Model | Class | File:line | Base | Rank | Scaling | `predict_proba` | Feat. imp. |
|---|---|---|---|---|---|---|---|
| random_forest | `RandomForestModel` | `classical/random_forest.py:35` | `BaseModel` | 2D | No (`:53`) | **Yes** `:195` | `:242` |
| logistic | `LogisticModel` | `classical/logistic.py:41` | `BaseModel` | 2D | Yes (`:65`) | **Yes** `:212` | `:259` (coefs) |
| svm | `SVMModel` | `classical/svm.py:35` | `BaseModel` | 2D | Yes (`:56`) | **Yes** `:218` | `:272` |

#### Neural RNN/CNN — 3D sequence, all inherit `BaseRNNModel`

`BaseRNNModel(BaseModel)` at `src/models/neural/base_rnn.py:174` provides `model_family="neural"` `:227`, `requires_scaling=True` `:231`, `requires_sequences=True` `:235`, the whole `fit()` loop `:320`, `predict()` `:647`, `save()` `:686`, `load()` `:707`.

| Model | Class | File:line | Rank | Overrides |
|---|---|---|---|---|
| lstm | `LSTMModel` | `neural/lstm_model.py:88` | 3D | network only |
| gru | `GRUModel` | `neural/gru_model.py:93` | 3D | network only |
| tcn | `TCNModel` | `neural/tcn_model.py:162` | 3D | network only |
| inceptiontime | `InceptionTimeModel` | `neural/inceptiontime_model.py:317` | 3D | `predict()` `:462` |
| resnet1d | `ResNet1DModel` | `neural/resnet1d_model.py:340` | 3D | `predict()` `:489` |
| nbeats | `NBEATSModel` | `neural/nbeats_model.py:496` | 3D | `fit()` `:617`, `predict()` `:657` |
| tft | `TFTModel` | `neural/tft_model.py:518` | **3D** | `predict()` `:668`, `get_feature_importance()` `:721` |
| transformer | `TransformerModel` | `neural/transformer_model.py:272` | 3D | `predict()` `:403`, `get_feature_importance()` `:457` |

#### Transformers — the 4D story is only partly true

| Model | Class | File:line | Declared rank | `requires_4d` |
|---|---|---|---|---|
| patchtst | `PatchTSTModel` | `neural/patchtst_model.py:261` | `MULTI_TF_4D` (`contract:376`) | **True** `:300-302` |
| itransformer | `iTransformerModel` | `neural/itransformer_model.py:279` | `MULTI_TF_4D` (`contract:395`) | **True** `:318-320` |
| tft | `TFTModel` | `neural/tft_model.py:518` | `SEQUENCE_3D` (`contract:410`) | **absent → False** |

**OBSERVED DOC BUG** — `CLAUDE.md` places TFT under "Transformer Models (4D input: `[batch, sequence, features, channels]`)". Both the contract (`model_contract.py:407-420`, `input_rank=DataRank.SEQUENCE_3D`, `model_family="neural"`) and the class (no `requires_4d` override) say **TFT is a 3D sequence model in the neural family**. TFT is also registered with `family="neural"` (`tft_model.py:514`), not `"transformer"`.

Also note `CLAUDE.md`'s 4D shape annotation `[batch, sequence, features, channels]` contradicts the code, which is `(batch, n_timeframes, seq_len, features)` — see `base.py:386` and `model_contract.py:376`.

**OBSERVED second family mismatch** — vanilla `transformer` is registered `family="neural"` (`transformer_model.py:268`) and inherits `BaseRNNModel.model_family → "neural"`, but `MODEL_CONTRACTS["transformer"].model_family == "transformer"` (`model_contract.py:361`) and `MODEL_FAMILIES` puts it in `"transformer"` (`constants.py:93`). Same for `patchtst`/`itransformer`: registered `family="transformer"` (`:257`, `:275`) but their runtime `model_family` property resolves to `"neural"` via `BaseRNNModel:228`. **Three different answers to "what family is this model" depending on which source you ask.**

#### Ensembles & meta-learners — also `BaseModel` subclasses

| Model | Class | File:line | Family (registered) |
|---|---|---|---|
| voting | `VotingEnsemble` | `ensemble/voting.py:47` | ensemble |
| stacking | `StackingEnsemble` | `ensemble/stacking.py:44` | ensemble |
| blending | `BlendingEnsemble` | `ensemble/blending.py:34` | ensemble |
| ridge_meta | `RidgeMetaLearner` | `ensemble/ridge_meta.py:34` | meta_learner |
| mlp_meta | `MLPMetaLearner` | `ensemble/mlp_meta.py:33` | meta_learner |
| xgboost_meta | `XGBoostMeta` | `ensemble/xgboost_meta.py:28` | meta_learner |
| calibrated_meta | `CalibratedMetaLearner` | `ensemble/calibrated_meta.py:35` | meta_learner |

**INFERRED, and architecturally important** — ensembles are `BaseModel` instances, so *an ensemble can be a base model of another ensemble*. That is the right foundation for a mix-and-match marketplace. `VotingEnsemble` even computes its own rank from its constituents (`voting.py:100`, `:118`).

### A.4 Interface consistency — mostly good, with sharp edges

**OBSERVED — `fit()` is uniform across all 16 base learners.** Every one has exactly:
```python
def fit(self, X_train, y_train, X_val, y_val,
        sample_weights=None, config=None) -> TrainingMetrics
```
Verified at `base.py:259-267` (abstract), `base_rnn.py:320-328`, `nbeats_model.py:617-625`, `itransformer_model.py:402-410`, `xgboost_model.py:133-141`, `lightgbm_model.py:166-174`, `catboost_model.py:123-131`, `logistic.py:89-97`, `random_forest.py:76-84`, `svm.py:79-87`.

**This is a genuine strength.** There is no fit-signature drift. The `X_val`/`y_val` args are mandatory even for models that cannot early-stop (classical models note this in docstrings, e.g. `logistic.py:101`, `random_forest.py:88`) — slightly wasteful, but consistent.

**OBSERVED — `predict()` is uniform**, always returning `PredictionResult` (canonical at `src/core/interfaces.py:125`, re-exported `models/base.py:27`), with `class_predictions`, `class_probabilities`, `confidence`, `metadata`. Shape validation in `__post_init__` `:172-186`.

**OBSERVED INCONSISTENCY #1 — `predict_proba` is not part of the contract.**
- Present on: all 3 classical (`logistic.py:212`, `random_forest.py:195`, `svm.py:218`), all ensembles and meta-learners (`voting.py:567`, `stacking.py:1086`, `blending.py:408`, `ridge_meta.py:211`, `mlp_meta.py:247`, `xgboost_meta.py:233`, `calibrated_meta.py:234`, `second_level.py:294`, `orchestrator.py:499`).
- **Absent on: all 3 boosting models and all 10 neural models.**
- Not declared in `BaseModel` at all.

**INFERRED** — any marketplace code that duck-types `hasattr(model, "predict_proba")` will silently split the roster in two. Probabilities *are* available everywhere via `predict().class_probabilities`, so the fix is to lift `predict_proba` into `BaseModel` as a concrete method delegating to `predict()`. Cheap, high value.

**OBSERVED INCONSISTENCY #2 — `save`/`load` type signatures drift.**
- `BaseModel.save(self, path: Path)` `:309`, `load(self, path: Path)` `:323`.
- `iTransformerModel.save(self, path: str | Any)` `:523` and `.load(self, path: str | Any)` `:544` — the **only** base learner deviating.
- Ensembles widen further: `second_level.py:533` uses `Path | str`, `orchestrator.py:625` uses `str | Path`.

**OBSERVED INCONSISTENCY #3 — three competing "family" answers**, detailed in A.3.

**OBSERVED INCONSISTENCY #4 — model-specific special-casing exists but is narrow.** Hardcoded name branches survive at:
- `src/models/device.py:487` (`family == "patchtst"`), `:518` (`itransformer`), `:560` (`tft`) — per-model GPU/batch tuning.
- `src/core/utils/config_validator.py:257`: `if model_name in ["lstm", "gru", "tcn", "transformer"]`.
- `src/models/trained_registry/registry.py:474`: `if model_name in ("transformer", "patchtst", "itransformer")`.

**INFERRED** — this is *much* less special-casing than expected. Routing itself is contract-driven: `ModelContract.adapter_id` (`model_contract.py:94-108`) maps rank → `"tabular"|"sequence"|"multi_stream"`, and `src/inference/bundle.py:308-317` reads `getattr(model, "requires_sequences"/"requires_4d")` rather than switching on names. **The plugin architecture is real.** The four sites above are the remaining leaks.

### A.5 Capability matrix

**OBSERVED / INFERRED** per capability:

| Capability | Status |
|---|---|
| **Probabilistic output** | **OBSERVED: universal.** Every `predict()` returns `class_probabilities`. Calibration layered on top: `src/models/calibration/calibrator.py` (`ProbabilityCalibrator`) and `conformal.py` (`ConformalPredictor`). |
| **Multivariate input** | **OBSERVED: universal.** 2D models take `(n, features)`; 3D take `(n, seq, features)`; 4D take `(n, n_timeframes, seq, features)`. Note `base_rnn.py:353-358` flattens 4D→3D as `n_features = n_timeframes * n_features_per_tf`. |
| **Multi-horizon** | **OBSERVED: NOT a model capability.** No model accepts a horizon list. Handled by the orchestrator training one model per horizon and keying results `"xgboost_h5"`, `"xgboost_h20"` — `src/models/training/training_ops.py:49`, `:701`, `:798`; `unified_orchestrator.py:514-516` explicitly refuses to mix horizons in one ensemble. TFT's docstring calls it a "multi-horizon forecaster" (`tft_model.py:515`) but the implementation is single-horizon classification. |
| **Incremental / `partial_fit`** | **OBSERVED: does not exist anywhere.** Zero `partial_fit` definitions in `src/models/`. `LogisticModel` has `warm_start: False` in defaults (`logistic.py:86`) but nothing consumes it. Retraining is full-refit only (walk-forward re-fits per window, `training/modes/walk_forward.py:487`). |
| **Scaling requirements** | **OBSERVED: declared twice, redundantly.** `BaseModel.requires_scaling` property AND `ModelContract.requires_scaling` + `scaler_type`. Values agree on spot-checks but nothing enforces agreement. |
| **Sample weights** | **OBSERVED: universal** — `sample_weights` in every `fit()`. |
| **Causality honesty** | **OBSERVED and commendable** — `is_production_safe` property flags non-causal models: N-BEATS `False` (`nbeats_model.py:537-547`), vanilla Transformer documented non-causal (`transformer_model.py:292-297`), TFT documented non-causal (`tft_model.py:539-541`), `BaseRNNModel:66-76` returns `not bidirectional`. |

### A.6 Four parallel per-model registries — the real fragility

**OBSERVED** — the same model is described independently in at least four places:

1. `MODEL_CONTRACTS` — `src/core/contracts/model_contract.py:229-560` (rank, feature_mode, mtf_mode, seq_len, scaler, min/max features)
2. `MODEL_DATA_REQUIREMENTS` — `src/models/config/data_requirements.py:117+` (feature_set, scaler, seq_len, max_features, input_rank)
3. `MODEL_FEATURE_STRATEGIES` — `src/data/features/strategies.py:147-360`
4. `HYPERPARAMETER_SPACES` — `src/optimization/hyperparameters.py:130-355`, mirrored again in `src/validation/cv/param_spaces.py:135-160`
5. (plus `FEATURE_SET_ALIASES` — `src/data/pipeline/config/feature_sets/core.py:50-93`)

Observed drift already present:

| Drift | Evidence |
|---|---|
| **Phantom model `mlp`** | `data_requirements.py:231-248` defines a full `ModelDataRequirements` for `"mlp"`. It is **not** in `MODEL_FAMILIES`, **not** in `MODEL_CONTRACTS`, **not registered**, has no class. `FEATURE_SET_ALIASES` also lists `"mlp"` (`core.py:64`). Dead config. |
| **Phantom model `informer`** | `FEATURE_SET_ALIASES["informer"]` (`core.py:75`) and `registry.py:212` docstring. No implementation. |
| **feature_set disagreement** | `data_requirements.py:288` gives `patchtst` → `"neural_optimal"` (with an explicit "was transformer_raw - MOD-002 fix" comment); `FEATURE_SET_ALIASES` `core.py:73` gives `patchtst` → `"patchtst_optimal"`. Same for `nbeats`: `"neural_optimal"` (`:350`) vs `"nbeats_optimal"` (`core.py:79`). |
| **Classical models missing from aliases** | `FEATURE_SET_ALIASES` (`core.py:50-93`) has no entry for `random_forest`, `logistic`, or `svm` — while `xgboost`/`lightgbm`/`catboost` all have one. Consistent with A.0: classical models are second-class citizens in the config layer. |
| **`config/models/` is empty** | Only `README.md` exists (`config/models/README.md`). The README states empty is valid, but `data_requirements.py` comments repeatedly claim values "Matches nbeats.yaml", "Matches patchtst.yaml", "Matches tft.yaml" — referencing files that **do not exist**. Those comments are stale provenance claims. |

**PROPOSED** — for a marketplace, collapse (1)+(2)+(3) into one `ModelContract` extended with `feature_set` and `default_hyperparameter_space`, and derive everything else. This is the highest-leverage refactor in the model layer.

### A.7 How models are constructed, and by whom

**OBSERVED** — construction is centralized on `ModelRegistry.create(name, config=...)` (`registry.py:141-175`). Call sites:

| Consumer | Site |
|---|---|
| Single-model trainer | `models/training/trainer.py:115` |
| Walk-forward mode | `models/training/modes/walk_forward.py:487` |
| OOF generation | `models/training/services/oof_generation.py:315` |
| Training ops | `models/training/training_ops.py:659`, `:1039` |
| Voting ensemble | `ensemble/voting.py:367`, `:619` |
| Stacking ensemble | `ensemble/stacking.py:825`, `:1176`, meta at `:532`, `:1184` |
| Blending ensemble | `ensemble/blending.py:252`, `:301`, `:461`; meta `:276`, `:467` |
| Inference bundles | `inference/bundle.py:706` |
| CV tuner (Optuna) | `validation/cv/cv_tuner.py:228` |
| CV feature selection | `validation/cv/cv_feature_selection.py:156` |

**OBSERVED** — `trainer.py:95-118` injects a fixed allowlist of training keys (`max_epochs`, `batch_size`, `early_stopping_patience`, `sequence_length`, `mixed_precision`, `num_workers`, `pin_memory`, `checkpoint_interval`, `keep_n_checkpoints`, `oom_*`) into `config.model_config` before `create()`. **INFERRED**: a new model family with a config key outside this list will not receive it from `TrainerConfig` — a hidden extension point that a marketplace plugin author would trip over.

**INFERRED — verdict on plugin readiness:** the registry + contract + adapter-routing triad is genuinely modular. Adding a new model requires: a `BaseModel` subclass, an `@register` decorator, a `MODEL_CONTRACTS` entry, a `MODEL_DATA_REQUIREMENTS` entry, a `MODEL_FEATURE_STRATEGIES` entry, an entry in `MODEL_FAMILIES` (or the `assert len == 23` at `constants.py:100` fires), and a `HYPERPARAMETER_SPACES` entry (or the check at `hyperparameters.py:362-363` fires). **Six edits in six files.** That is the friction to remove.

### A.8 What is conspicuously ABSENT

**OBSERVED (by exhaustive grep):**
- **No naive / majority-class / dummy baseline.** Zero hits for `DummyClassifier`, `BaselineModel`, `naive_baseline` in `src/models/`. The only "majority" hits are vote-aggregation logic in `voting.py:518` and `inference/pipeline.py:398-400`.
- **No ExtraTrees.**
- **No classical statistical models** (ARIMA, ETS, Theta) — appropriate, see B.1.
- **No DLinear / NLinear.**
- **No N-HiTS, Autoformer, FEDformer, TimesNet.**
- **No foundation-model adapter** (Chronos / TimesFM / Moirai / Lag-Llama / Kronos / TabPFN).
- **No ROCKET / MiniRocket / MultiRocket.**
- **No online/incremental learner** (river, Hoeffding trees, ADWIN).

---

## PART B — Research: which families actually earn their place

Framing that governs every verdict below: the target is **3-class (or binary) classification of triple-barrier labels on financial bars** — not point forecasting. Signal-to-noise is brutal; the meta-goal stated in the mission is to *prove ensembles beat their constituents and simple baselines*. That reframes "model value" away from leaderboard accuracy and toward (a) being a credible floor, (b) adding decorrelated errors to an ensemble, (c) costing little.

### B.0 THE HEADLINE: cheap baselines are the highest-value additions, by a wide margin

**PROPOSED, and I want to be blunt about it.** A marketplace of 16 sophisticated learners with **no naive baseline** cannot make its central claim. If the pipeline reports macro-F1 0.41 for a stacked ensemble of TFT + PatchTST + LightGBM, there is currently *nothing in the repo* that answers "is 0.41 better than always predicting the majority class?" On triple-barrier labels — which `CLAUDE.md` Phase 80 notes were producing a **2.6% long rate at H5** before the symmetric-cost fix — a majority-class predictor can score deceptively well on accuracy and the ensemble may not beat it.

Ranked by (value ÷ effort), the additions I would make first:

| # | Addition | Effort | Why it matters here |
|---|---|---|---|
| **1** | **`DummyModel`** (majority / stratified / prior strategies) | ~150 lines, `BaseModel`, sklearn `DummyClassifier` | **The floor.** Makes every other number interpretable. Also the correct null for the DSR/Bonferroni machinery already in `src/validation/`. Zero risk. |
| **2** | **Promote `logistic` + `random_forest` to documented first-class baselines** | ~0 lines of model code; docs + `FEATURE_SET_ALIASES` entries + CI gate | **Already implemented and smoke-tested.** They are the standard "did the complexity buy anything" controls. `logistic` with `l1_ratio` already covers Ridge/Lasso/ElasticNet. |
| **3** | **`ExtraTreesModel`** | ~120 lines, near-copy of `random_forest.py` | Random split thresholds → trees far less correlated than RF's, individually weaker, **more robust to label noise**. Triple-barrier labels ARE noisy. Cheapest possible source of genuine ensemble diversity. |
| **4** | **A single-feature/rule "primary model" stub** for meta-labeling | small | López de Prado's meta-labeling separates *side* from *size*; the repo has meta-labeling infra (`training/meta_labeling/`) but the primary signal is another ML model. A trivial primary (e.g. momentum sign) is the canonical baseline. |

Everything in B.1–B.9 should be judged *after* these exist. Adding a 17th deep architecture before there is a majority-class floor is, bluntly, bloat.

### B.1 Classical / statistical: ARIMA, ETS, Theta — **SKIP**

**Verdict: do not add. Category error.**

These are univariate point-forecasting models for series with trend/seasonality. The task here is multivariate classification of a path-dependent barrier outcome. Converting an ARIMA forecast into a triple-barrier class requires a hand-built thresholding layer that would itself be the model. Intraday futures bars have negligible exploitable linear autocorrelation. The engineering cost (statsmodels dependency, per-series fitting that does not vectorize, no `sample_weights` path) buys nothing.

**Nuance worth keeping:** ARIMA-family *residuals* and stationarity tooling are useful as **features**, not models — and the repo already has this via fractional differentiation (`src/optimization/feature_selection/frac_diff.py`, Phase 99) and `find_min_d()` ADF scanning. That is the right place for this family.

### B.2 Linear models: Ridge / ElasticNet / logistic — **ALREADY PRESENT; PROMOTE**

**Verdict: highest value, near-zero cost, because it already exists.**

`LogisticModel` (`classical/logistic.py`) with `saga` + `l1_ratio` spans L2, L1, and elastic net. This is *the* canonical baseline for "is there linear signal in these 200 engineered features?" It is fast, its coefficients are directly interpretable (`get_feature_importance()` `:259`), and it is the standard stacking meta-learner (its docstring says so, `:51-52`; `data_requirements.py:611` sets `meta_learner: str = "logistic"` as default).

**PROPOSED:** make `logistic` a mandatory member of every ensemble run and a mandatory row in every comparison report. If a 12-model stacked ensemble does not beat regularized logistic regression on OOF macro-F1 *and* on net-of-cost Sharpe, that is the finding.

### B.3 Trees & boosting: XGB / LGBM / CatBoost / RF / ExtraTrees — **CORE; ADD ExtraTrees**

**Verdict: this is where the real signal lives. Add ExtraTrees, nothing else.**

The 2024–2025 literature consensus holds: **GBDTs remain state of the art on tabular data**, and specifically remain the leading algorithm under **label noise** — the exact regime here. TabNet-style deep tabular models still do not reach GBDT performance; the gap narrows but does not close.

The three boosting models are already the strongest constituents this platform has, and their `requires_scaling=False` + native missing-value handling makes them the lowest-friction models in the marketplace.

**ExtraTrees is the one genuine gap.** Extremely randomized trees sample split thresholds rather than optimizing them, which (a) makes individual trees weaker and (b) makes them **substantially less correlated** than RF trees. Both properties are exactly what a stacking meta-learner wants, and the random thresholds reduce overfitting to mislabeled rows. The repo already has a `DiversityAnalyzer` (`ensemble/diversity.py`) that would immediately show the benefit. **~120 lines. Do it.**

### B.4 Deep sequence: LSTM/GRU/TCN — **KEEP**; N-BEATS/N-HiTS — **KEEP N-BEATS, SKIP N-HiTS**; DLinear/NLinear — **ADD, cheaply**

**LSTM / GRU / TCN — keep all three.** They are implemented, share `BaseRNNModel`, and TCN's causal dilated convolutions make it the only sequence model here that is *architecturally* causal (`is_production_safe` → `True` when not bidirectional). For a trading system, causality is not a nicety.

**N-BEATS — keep, but demote expectations.** It exists and works (A.1). But it is non-causal by its own admission, capped at `max_features=20`, and its trend/seasonality inductive bias is a forecasting prior with weak justification for barrier classification. Keep it as a diversity source; do not invest more in it.

**N-HiTS — skip.** It is N-BEATS + multi-rate sampling, a *forecasting* efficiency improvement. Marginal-on-marginal for a classification task. Pure bloat.

**DLinear / NLinear — ADD, and this is my most contrarian recommendation.** The LTSF-Linear result (AAAI-23 oral) showed a one-layer linear model beating the entire 2021–2022 transformer forecasting literature on most long-horizon benchmarks. The result is contested — Hugging Face's replication argues size-matched transformers win — but that controversy is *precisely* the point: **DLinear is the standard sanity check that a sequence architecture is earning its parameters.** NLinear's normalization step specifically targets train/test distribution shift, which is the defining pathology of financial data.

Cost: ~80 lines each as `BaseRNNModel` subclasses (they are literally `Linear(seq_len → n_classes)` with a decomposition or normalization step). They slot into the existing 3D pipeline with zero new infrastructure. **Best value-per-line of any deep addition.**

### B.5 Transformers: PatchTST / iTransformer / TFT — **KEEP**; Autoformer / FEDformer / TimesNet — **SKIP**

**Keep the three that exist.** PatchTST's patching reduces attention cost and is the strongest of the family on classification benchmarks (~92.6% on UCI-HAR); iTransformer's inverted attention over *features* rather than time is a genuinely different inductive bias (good for ensemble diversity, and appropriate for 200 engineered features); TFT's variable-selection networks give real interpretability.

**Autoformer / FEDformer — skip.** Both are built around **seasonal decomposition and frequency-domain attention** for long-horizon forecasting of series with strong periodicity. Financial bars have weak, unstable periodicity (session effects at best, and the repo already handles those via `session_cumsum()` per Phase 94). These add two large architectures for an inductive bias the data does not have.

**TimesNet — skip, despite the temptation.** It was 2023 SOTA across five TS tasks including classification, but its core mechanism is FFT-driven period detection reshaping 1D series into 2D — again a periodicity prior. And 2025–2026 benchmarks show it being beaten on classification by both PatchTST and much simpler models (TSLANet: 83.2% vs TimesNet 65.3% on UCR; 72.7% vs 66.6% on UEA). Its own maintainers now caution that the Time-Series-Library benchmarks are stale. Adding TimesNet buys a weaker classifier and a large maintenance surface.

**PROPOSED nuance:** if any transformer is added, make it a **ROCKET-family** model instead — see B.7.

### B.6 Foundation models: Chronos / TimesFM / Moirai / Lag-Llama / Kronos / TabPFN — **DEFER, with one exception**

**Verdict: not practical here today, with one caveat worth tracking.**

The blockers are specific, not vague:
1. **Task mismatch.** Chronos, TimesFM, Moirai, and Lag-Llama are *forecasting* models. They emit future values, not barrier-outcome classes. Every one would need a bolted-on classification head, at which point the pretraining advantage is largely lost.
2. **Domain mismatch.** Zero-shot on financial data is genuinely out-of-domain for Chronos/TimesFM — they were pretrained on general corpora that never modeled market microstructure.
3. **Univariate.** TimesFM in particular cannot model cross-feature dependencies — fatal when the whole feature engine produces ~192 engineered multivariate features.
4. **Licensing/weights/size.** Adds a multi-GB dependency and an offline-weights problem to a repo whose `CLAUDE.md` shows sustained battles with 230GB-RAM OOM.

**Two things worth watching, not building:**
- **Kronos** (AAAI-2026) is the first domain-specialist financial TSFM — pretrained on 12B+ K-line records from 45 exchanges with a tokenizer that quantizes OHLCV bars into hierarchical discrete tokens. This is the only foundation model whose pretraining distribution actually matches the input here. Still forecasting-oriented, but it is the one to re-evaluate in 6–12 months.
- **TabPFN v2 / v2.5** is a *tabular* foundation model and therefore the right shape for the 2D branch. v2 handles ≤10K samples / 500 features; v2.5 scales to ~50K samples / 2K features and reportedly beats tuned tree models there. **But** it has a hard architectural class-count limit of 10 (fine — 3 classes) and cannot touch the 1.6M-row datasets this repo targets. **PROPOSED:** if anything from this category is added, it is TabPFN as a *small-window* model in walk-forward mode (each window is small!) — that is a defensible fit, unlike the forecasting TSFMs.

### B.7 Representation learning & ROCKET — **ADD MiniRocket. This is the sleeper.**

**Verdict: MiniRocket/MultiRocket is the best cost/benefit deep-adjacent addition available, and it is missing.**

MiniRocket applies ~10K hard-coded (almost deterministic) dilated convolutional kernels and pools by proportion-of-positive-values, then fits a **linear classifier** on the resulting features. MultiRocket adds three more pooling operators (MPV, MIPV, LSPV) for ~50K features. Reported results: MultiRocket is **not significantly less accurate than HIVE-COTE 2.0** — the most accurate TSC method on the UCR archive — while being **orders of magnitude faster**.

Why it fits this platform unusually well:
- It is a **3D-sequence model** — drops straight into the existing `sequence` adapter and `BaseRNNModel`-style contract.
- Its classifier head is a **ridge/logistic** — so it inherits the interpretability and calibration story already built.
- It is **CPU-fast and deterministic**, sidestepping the entire GPU/OOM/torch.compile apparatus that Phases 72–91 fought.
- Its errors are structurally decorrelated from both GBDTs (which see engineered tabular features) and RNNs (which see the same sequences through learned filters). **That is exactly what stacking needs.**

Contrastive representation learning (TS2Vec, TSLANet-style) — **skip for now.** Two-stage pretrain-then-classify pipelines add real complexity, and the current pipeline has no unlabeled-data advantage to exploit (labels are cheap; they are just noisy).

### B.8 Mixture-of-experts & regime-aware — **ALREADY PARTLY PRESENT; consolidate, don't expand**

**Verdict: the repo already has the useful 80% of this. Do not add an MoE architecture.**

**OBSERVED** — regime infrastructure exists: `src/models/training/regime_detector.py`, `regime_trainer.py`, `src/models/regime_evaluation.py`, per-symbol ADX regime thresholds (Phase 93), regime hysteresis with `min_regime_bars` (Phase 110), regime-conditional feature selection (Phase 98 E5), regime-aware GPU cleanup (Phase 105).

The literature does support regime/MoE ensembles beating single models on financial series — adaptive MoE frameworks report ~6% MAE / ~7% MSE improvements over the strongest single baseline (GRU), and regime-switching maintains distinct predictors per market state. But note *what* those gains are measured against: a **single model**. That is the ensemble argument, which this repo already makes structurally.

**PROPOSED:** a learned gating network (true MoE) is a large, hard-to-validate addition prone to overfitting the regime labels themselves. The existing hard-routed regime trainer is the honest version. Invest in *evaluating* it (does regime-routed beat pooled?) rather than in a softer, fancier router.

### B.9 Online / incremental (river, Hoeffding trees, ADWIN) — **SKIP as models; STEAL the drift detector**

**Verdict: skip the learners, seriously consider the drift detectors.**

The models: Hoeffding trees and Adaptive Random Forest are designed for unbounded streams where full retraining is infeasible. That is not this system — walk-forward *already* refits per window (`training/modes/walk_forward.py`), which is the batch equivalent and is strictly better when you can afford it. Adding a `partial_fit` path would fork the entire training/OOF/calibration/bundle stack for models that would underperform LightGBM. **Real bloat.**

The detectors are different. **ADWIN**, KSWIN, and HDDM detect distribution change with statistical guarantees, and the streaming literature specifically pairs them with **XGBoost** (not just Hoeffding trees) to good effect. Financial markets are the canonical concept-drift domain — manias, panics, crashes.

**PROPOSED:** add ADWIN-style drift detection as a **monitoring/gating** component (a `src/validation/` concern), answering "has the feature or label distribution shifted enough that the deployed bundle should be retrained or stood down?" That is a genuine production capability the repo lacks, and it is ~200 lines with no `river` dependency required.

### B.10 Hybrids — **SKIP**

CNN-LSTM, LSTM-attention, wavelet-LSTM and friends: the marketplace *is* the hybrid mechanism. Stacking a TCN and an LSTM through a meta-learner is a hybrid with an auditable interface and an honest OOF story. Hardcoding hybrid architectures multiplies the model count without multiplying the hypothesis space, and each one needs its own contract, hyperparameter space, and feature strategy (six-file edit, per A.7).

### B.11 Consolidated recommendation

**Tier 1 — do these first (they enable the mission's central claim):**
1. `DummyModel` (majority/stratified/prior) — **PROPOSED**
2. Promote+document+CI-gate `logistic` and `random_forest` as mandatory baselines — **PROPOSED** (code already exists, A.0)
3. `ExtraTreesModel` — **PROPOSED**

**Tier 2 — high value-per-line:**
4. `MiniRocketModel` / `MultiRocketModel` (3D, linear head) — **PROPOSED**
5. `DLinearModel` / `NLinearModel` (3D, ~80 lines each) — **PROPOSED**

**Tier 3 — infrastructure, not models (fixes A.4/A.6):**
6. Lift `predict_proba` into `BaseModel` as a concrete delegate to `predict()`
7. Collapse the four per-model registries into one extended `ModelContract`; delete phantom `mlp`/`informer` entries
8. Fix TFT's documented rank/family; reconcile the three family sources
9. ADWIN-style drift detection as a validation/monitoring component

**Tier 4 — track, do not build:** Kronos, TabPFN v2.5 (small-window walk-forward only).

**Explicitly reject as bloat:** ARIMA/ETS/Theta, N-HiTS, Autoformer, FEDformer, TimesNet, Chronos/TimesFM/Moirai/Lag-Llama adapters, river/Hoeffding online learners, hardcoded hybrid architectures, learned-gating MoE.

---

## Sources (Part B)

- [Are Transformers Effective for Time Series Forecasting? (DLinear/NLinear, AAAI-23)](https://arxiv.org/pdf/2205.13504)
- [LTSF-Linear official implementation](https://github.com/cure-lab/LTSF-Linear)
- [Yes, Transformers are Effective for Time Series Forecasting (counterpoint)](https://huggingface.co/blog/autoformer)
- [Training Gradient Boosted Decision Trees on Tabular Data Containing Label Noise](https://arxiv.org/abs/2409.08647)
- [A Closer Look at Deep Learning Methods on Tabular Datasets](https://arxiv.org/pdf/2407.00956)
- [MultiRocket: Multiple pooling operators and transformations for fast and effective time series classification](https://arxiv.org/pdf/2102.00457)
- [MiniRocket official implementation](https://github.com/angus924/minirocket)
- [Convolution based time series classification in aeon](https://www.aeon-toolkit.org/en/latest/examples/classification/convolution_based.html)
- [TSLANet: Rethinking Transformers for Time Series Representation Learning](https://arxiv.org/pdf/2404.08472)
- [Time-Series-Library (TimesNet, maintainer benchmark caveat)](https://github.com/thuml/Time-Series-Library)
- [PatchTST overview](https://www.emergentmind.com/topics/patchtst)
- [Time Series Foundation Models for Multivariate Financial Time Series Forecasting](https://arxiv.org/html/2507.07296v1)
- [Kronos: A Foundation Model for the Language of Financial Markets (AAAI 2026)](https://arxiv.org/pdf/2508.02739)
- [Kronos overview — TSFM.ai](https://tsfm.ai/blog/kronos-financial-foundation-model)
- [A Closer Look at TabPFN v2: Strength, Limitation, and Extension](https://arxiv.org/html/2502.17361v1)
- [TabPFN-2.5 Model Report](https://priorlabs.ai/technical-reports/tabpfn-2-5-model-report)
- [The state of Tabular Foundation Models (2026)](https://mindfulmodeler.substack.com/p/the-state-of-tabular-foundation-models)
- [Advances in Financial Machine Learning — López de Prado](https://toc.library.ethz.ch/objects/pdf03/e01_978-1-119-48208-6_01.pdf)
- [Does Meta-Labeling Add to Signal Efficacy? — Hudson & Thames](https://hudsonthames.org/wp-content/uploads/2022/04/Does-Meta-Labeling-Add-to-Signal-Efficacy.pdf)
- [Adaptive Market Intelligence: A Mixture of Experts Framework](https://arxiv.org/pdf/2508.02686)
- [Ensemble Multi-Expert Forecasting: Robust Decision-Making in Chaotic Financial Markets](https://www.mdpi.com/1911-8074/18/6/296)
- [Comparative Evaluation of Supervised ML and Concept Drift Detection in Financial Business Problems](https://link.springer.com/chapter/10.1007/978-3-030-75418-1_13)
- [Incremental Market Behavior Classification in Presence of Recurring Concepts](https://www.mdpi.com/1099-4300/21/1/25)
- [Extra Trees vs Random Forest — when to choose each](https://mljourney.com/random-forest-vs-extremely-randomized-trees-extra-trees-when-to-choose-each/)
- [On the Robustness of Decision Tree Learning under Label Noise](https://arxiv.org/pdf/1605.06296)
