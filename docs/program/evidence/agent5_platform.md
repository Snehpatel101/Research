# Agent 5 — ML Platform / Plugin-System Architecture

**Repo:** `/home/user/Research` (ML Factory) — 435 Python files, 156,923 LOC in `src/`
**Date:** 2026-08-24
**Scope:** Investigation + research + design proposal. **No code changes made.**
**Tags:** `OBSERVED` = verified in repo/docs · `INFERRED` = reasoned from evidence · `PROPOSED` = my design · `UNKNOWN` = not determined

---

## 0. Executive Summary

`OBSERVED` The repo already contains **four of the five primitives** a mix-and-match
marketplace needs: a decorator-based `ModelRegistry`, a decorator-based
`AdapterRegistry`, a declarative `ModelContract` table, and a `DataContract` with
strict validation. This is much further along than a typical 157k-LOC monolith.

`OBSERVED` But the composition story fails on five specific counts:

1. **Capability truth is triplicated and already drifted.** `MODEL_CONTRACTS`
   (`src/core/contracts/model_contract.py:229`), `MODEL_FAMILIES`
   (`src/core/constants.py:81`), and per-instance `BaseModel` properties
   (`src/models/base.py:207-238`) each independently claim rank/scaling/family.
   Live drift measured (§A.4).
2. **The registry is closed, not open.** `src/core/constants.py:100` executes
   `assert len(ALL_MODELS) == 23` at import time. Adding a 24th model crashes
   the process. This is a hard blocker for a marketplace.
3. **The one real compatibility validator is dead in the main path.**
   `validate_base_model_compatibility` (`src/models/ensemble/validator.py:227`)
   is only called from the `Voting`/`Blending`/`Stacking` *model classes*, which
   `MLFactory` never constructs. The live path builds a meta-learner directly
   (`src/models/training/services/ensemble_service.py:400-418`).
4. **Ensemble strategy is not expressible.** `EnsembleMethod`
   (`src/config/ensemble.py:28`) defines stacking/voting/blending/weighted — and
   nothing in the training path reads it. The live config surface is
   `models: list[str]` + `build_ensemble: bool` + `meta_learner: str`
   (`src/config/experiment.py:103,128-129`). You cannot express
   `A+B+C -> X` and `A+D+F -> Y` in one experiment, and you cannot pick X at all.
5. **Prediction combinability is checked by string suffix.** The only guard that
   two models' predictions are legally combinable is
   `k.endswith(f"_h{horizon}")` (`src/models/training/unified_orchestrator.py:512-517`).

`PROPOSED` The fix is a **capability-first spine** retrofitted, not a rewrite:
one `ModelSpec` = `ModelCapabilities` + factory callable, registered by decorator
and discoverable via entry points. Every existing lookup table becomes a
*derived view* of the spec table. `BaseModel` stays; it is adapted, not replaced.
A `PredictionSpace` value object makes "can these be combined" a typed equality
check. An `EnsembleStrategy` registry with its own capabilities makes strategy a
first-class swappable component. A `CompositionValidator` runs **before data is
loaded** and returns actionable diagnostics with remedies. Estimated retrofit:
~3–4 weeks, ~2,500 new lines, ~1,800 deleted, zero changes to the 12 model
`fit`/`predict` bodies.

---

# PART A — What Exists Today

## A.1 The contract layer — `src/core/contracts/`

`OBSERVED` Four files, 1,472 lines:

| File | Lines | Contents |
|---|---|---|
| `model_contract.py` | 628 | `ModelContract` frozen dataclass + `MODEL_CONTRACTS` dict of 23 |
| `data_contract.py` | 466 | `DataContract`, `DataContractSchema`, `FeatureMode`, `MTFMode` |
| `feature_spec.py` | 279 | `FeatureSpec` for 5-D Optuna search |
| `__init__.py` | 99 | Re-exports |

### What `ModelContract` declares (`model_contract.py:37-81`)

```python
@dataclass(frozen=True)
class ModelContract:
    model_name: str
    model_family: str          # free-form str, NOT the ModelFamily enum
    input_rank: DataRank = DataRank.TABULAR_2D
    feature_mode: FeatureMode = FeatureMode.ENGINEERED
    mtf_mode: MTFMode = MTFMode.NONE
    primary_timeframe: str = "5min"
    mtf_timeframes: tuple[str, ...] = ()
    sequence_length: int = 60
    patch_length: int | None = None
    requires_scaling: bool = True
    scaler_type: str = "robust"
    min_features: int = 4
    max_features: int = 200
    description: str = ""
```

Derived properties: `requires_sequences` (`:84`), `requires_multi_timeframe`
(`:89`), `adapter_id` (`:94`, an if/elif over `input_rank`).

### Is it capability-driven or a data blob?

`INFERRED` **Both, and that is the problem.** It is genuinely capability-shaped —
it declares *requirements* rather than naming models — and it is genuinely
consumed for routing. But it is a **static module-level dict keyed by name**
(`:229`), not something a model declares about itself. Adding a model means
editing a central file. It is also **incomplete against the product
requirements**: there is no field for task type, `n_classes`, probabilistic
output, incremental training, NaN tolerance, sample-weight support, GPU
requirement, determinism, or prediction representation.

### Is it enforced?

`OBSERVED` **Partially, and only at the data-shape layer.**
`validate_data_contract_strict` is called from exactly three sites, all adapters:

- `src/data/adapters/tabular.py:122`
- `src/data/adapters/sequence.py:153`
- `src/data/adapters/multi_stream.py:231`

So the contract is enforced *after* the adapter has already built the array —
i.e. after data loading, feature engineering and windowing. It is a
post-hoc assertion, not a pre-flight gate.

### Who consumes it?

`OBSERVED` 30+ call sites of `get_model_contract` across `factory.py`,
`trainer.py`, `walk_forward.py`, `training_ops.py`, `feature_selection.py`,
`oof_alignment.py`, `bundle.py`, `timeframe_coordinator.py`. Consumption is
genuinely broad — this is the healthiest part of the architecture and the right
foundation to build on.

`OBSERVED` **Unenforced field:** `ModelContract.sequence_length` is ignored in
standard-mode training but honored in walk-forward mode. Documented as open
**DECISIONS.md item #4**: "TCN at 60 is one bar SHORT of its receptive field —
part of the network literally never sees data… same model trains with different
windows depending on mode." A declared capability that the runtime silently
overrides is worse than no declaration.

---

## A.2 The adapter registry — `src/data/adapters/`

`OBSERVED` 12 files, 5,693 lines.

**`AdapterRegistry` (`registry.py:20-145`) is a proper dict registry with a
decorator** — not if/elif:

```python
class AdapterRegistry:
    _adapters: dict[str, type[BaseAdapter]] = {}

    @classmethod
    def register(cls, adapter_id: str):          # :31
        def decorator(adapter_class): ...
        return decorator

    @classmethod
    def get_for_model(cls, model_name, **kwargs):  # :94
        contract = get_model_contract(model_name)
        return cls.create(contract.adapter_id, **kwargs)
```

`get_adapter()` (`registry.py:148`) dispatches on `adapter_id` **or**
`model_name`. Resolution by model name goes through the contract — good.

`OBSERVED` **But there is a second, parallel adapter dispatcher** —
`AdapterFactory` (`factory.py:56`), which does **not** use the contract. It uses
`MODEL_ADAPTER_MAP` (`factory.py:142`) and then re-branches on adapter type
strings three separate times (`factory.py:157,165` / `:244` / `:311`):

```python
adapter_type = MODEL_ADAPTER_MAP.get(model_key)      # lookup #1
if adapter_type == "sequence":   kwargs[...] = ...   # branch
elif adapter_type == "multi_stream": kwargs[...] = ...
...
adapter_type = MODEL_ADAPTER_MAP.get(model_name.lower())  # lookup #2, same value
if adapter_type == "multi_stream":
    result = adapter.transform(df, additional_dfs=additional_dfs)
else:
    result = adapter.transform(df)
```

`INFERRED` The `additional_dfs` special case is the tell: `BaseAdapter.transform`
has an inconsistent signature across adapters, so the caller must know which
adapter it got. That is a leaked abstraction — the caller re-derives the
adapter's identity to decide how to call it.

`MODEL_ADAPTER_MAP` is at least *derived* from `MODEL_CONTRACTS`
(`src/core/constants.py:125-129`) with a lazy cache and a module `__getattr__`
shim (`:155`), so it is not an independent source of truth — good, but it is an
extra indirection that exists only for backward compatibility.

---

## A.3 `src/core/types.py` — the enum vocabulary

`OBSERVED` 295 lines. Defines `DataRank` (2/3/4, `:32`), `ModelFamily`
(`:69`), `FeatureFamily` (`:102`), `TrainingMode` (`:142`), `CVMethod`
(`:163`), `AdapterType` (`:186`), `LabelingMethod` (`:213`), `ScalingSource`
(`:234`).

`OBSERVED` **`DataRank.from_model` (`:46`) and `ModelFamily.from_model` (`:87`)
are name-keyed lookups into `src/core/constants.py`.** They read
`MODEL_DATA_RANKS` and `MODEL_TO_FAMILY`. So the *type system itself* has a
name→capability lookup baked in. `ModelFamily.from_model` reads `MODEL_TO_FAMILY`
(constants), while `ModelContract.model_family` is a free-form string in the
contract table, while `BaseModel.model_family` is an instance property. Three
answers to one question.

`OBSERVED` **`AdapterType` enum (`:186`) is essentially unused as a type** —
adapter IDs are raw strings everywhere (`"tabular"`, `"sequence"`,
`"multi_stream"` in `ModelContract.adapter_id`, `AdapterRegistry._adapters`,
`AdapterFactory`). The enum exists but the system passes `str`.

---

## A.4 Instantiation trace: `"xgboost"` → model object

`OBSERVED` Full path:

```
ExperimentConfig.training.models = ["xgboost"]     src/config/experiment.py:103
  → MLFactory.run()                                src/factory.py:246
    → MLFactory._run_training()                    src/factory.py:852
      → UnifiedOrchestrator (config.models)        src/models/training/unified_orchestrator.py
        → ModelTrainingService                     src/models/training/services/model_training.py
          → TrainerConfig(model_name="xgboost")    src/models/config/trainer_config.py:228 (contract lookup)
            → Trainer.__init__                     src/models/training/trainer.py:82
              → ModelRegistry.create(name, config) src/models/training/trainer.py:115
                → ModelRegistry._models["xgboost"] src/models/registry.py:174
                  → XGBoostModel(config=...)       src/models/boosting/xgboost_model.py:68 (@register)
```

Registration is **import-side-effect based**: `src/models/__init__.py:48-53`
imports `boosting, classical, ensemble, neural` purely to trigger `@register`
decorators. `OBSERVED` No `entry_points`; `pyproject.toml` declares only
`[project.scripts] ensemble-pipeline`. So **third-party models are impossible
without editing `src/models/__init__.py`**.

### Lookup tables / dispatch points involved in "make me an xgboost"

`OBSERVED` **At least 12 distinct name-keyed lookups**, of which 9 are tables:

| # | Table / dispatch | Location | Keyed by |
|---|---|---|---|
| 1 | `ModelRegistry._models` | `src/models/registry.py:58` | model name + aliases |
| 2 | `ModelRegistry._families` | `src/models/registry.py:59` | family |
| 3 | `MODEL_CONTRACTS` | `src/core/contracts/model_contract.py:229` | model name |
| 4 | `MODEL_FAMILIES` / `MODEL_TO_FAMILY` | `src/core/constants.py:81,107` | family / model name |
| 5 | `MODEL_DATA_RANKS` (derived) | `src/core/constants.py:118` | model name |
| 6 | `MODEL_ADAPTER_MAP` (derived) | `src/core/constants.py:125` | model name |
| 7 | `AdapterRegistry._adapters` | `src/data/adapters/registry.py:28` | adapter id |
| 8 | `MODEL_FEATURE_STRATEGIES` | `src/data/features/strategies.py:146` | model name |
| 9 | `FEATURE_SET_ALIASES` | `src/data/pipeline/config/feature_sets/core.py:~57` | model name |
| 10 | `FeatureSelectionDefaults.get_defaults` mapping | `src/optimization/feature_selection/config.py:~100` | family **and** 3 model names |
| 11 | Instance properties `requires_scaling` / `requires_sequences` / `requires_4d` | `src/models/base.py:207,218,228` | (per-class override) |
| 12 | `meta_learner_map` | `src/models/training/services/ensemble_service.py:400` | meta-learner name |

`INFERRED` #3, #4, #5, #6, #11 all answer *the same questions* (rank, family,
adapter, scaling) from four independent storage locations. #5 and #6 are derived
from #3 (safe). #4 and #11 are **not**.

### Measured drift (run against the repo's venv, 2026-08-24)

`OBSERVED` I compared each registered model's live instance properties against
its `ModelContract`:

```
model            instRank ctrRank  instScale  ctrScale  instFam      ctrFam
mlp_meta         2        2        False      True      meta_learner meta_learner   ← SCALING DRIFT
itransformer     4        4        True       True      neural       transformer    ← FAMILY DRIFT
patchtst         4        4        True       True      neural       transformer    ← FAMILY DRIFT
transformer      3        3        True       True      neural       transformer    ← FAMILY DRIFT
(all others agree)
```

- **`mlp_meta`**: contract says `requires_scaling=True`; the class says `False`.
  Whichever consumer you ask gets a different answer about whether to scale.
- **`patchtst` / `itransformer` / `transformer`**: contract + `MODEL_TO_FAMILY`
  say `transformer`; the class says `neural`. Consumers that branch on family
  (`FeatureSelectionDefaults.get_defaults` at
  `src/optimization/feature_selection/config.py:~100`, `device.py:464,617`)
  will get *different feature-selection defaults* depending on which source
  they consulted.

`INFERRED` No test catches this because nothing cross-checks the two sources.
This is exactly the class of bug a capability spine eliminates by construction.

### The hard blocker

`OBSERVED` `src/core/constants.py:99-100`:

```python
ALL_MODELS: list[str] = [model for models in MODEL_FAMILIES.values() for model in models]
assert len(ALL_MODELS) == 23, f"Expected 23 models, got {len(ALL_MODELS)}"
```

Adding any model raises `AssertionError` **at import time**. There is a
matching `assert TOTAL_BASE_FEATURES == 162` at `:190`. A marketplace whose
plugin count is asserted in a core constants module is not a marketplace.

---

## A.5 Name-based special-casing in generic code

`OBSERVED` Inventory of every place generic code branches on a model name
(excluding docstrings/examples). **Each is a hole in the abstraction.**

| Site | Code | Hole |
|---|---|---|
| `src/models/training/training_ops.py:1077-1130` | `if model_name == "logistic" … elif "random_forest" … elif "xgboost" … elif "lightgbm" … elif "catboost" … else: fallback logistic` | Meta-labeling bet-sizer builds sklearn/booster objects by name, **bypassing `ModelRegistry` entirely**. A 6th model is invisible here. |
| `src/data/pipeline/stages/meta_labeling/primary_model.py:95` | `valid_models = {"logistic","lightgbm","xgboost","random_forest"}` | Hardcoded allow-list. |
| `src/data/pipeline/stages/meta_labeling/primary_model.py:355-389` | 4-way if/elif on `self.base_model` | Second parallel model constructor. |
| `src/models/ensemble/calibrated_meta.py:300-314` | `if estimator_name == "logistic" / "ridge" / "svm"` | Third parallel constructor. |
| `src/models/training/services/ensemble_service.py:400-411` | `meta_learner_map = {"ridge_meta":…, "mlp_meta":…, "xgboost_meta":…, "calibrated_meta":…}` | Meta-learners registered in `ModelRegistry` are **re-listed** here; a registered 5th meta-learner raises `ValueError: Unknown meta_learner`. |
| `src/models/trained_registry/registry.py:468` | `if model_name in ("xgboost","lightgbm","catboost")` | Family inferred from a name tuple. |
| `src/models/trained_registry/registry.py:472` | `if model_name in ("lstm","gru","tcn","tft","nbeats","inceptiontime","resnet1d")` | Ditto. Note `transformer`/`patchtst`/`itransformer` are **absent** — silent fall-through. |
| `src/core/utils/config_validator.py:257` | `if model_name in ["lstm","gru","tcn","transformer"]` | Sequence-model check that omits `nbeats`, `inceptiontime`, `resnet1d`, `tft`. |
| `src/core/utils/config_validator.py:456` | `if model_name == "stacking"` | Ensemble check by name. |
| `src/optimization/feature_selection/config.py:103-105` | `"patchtst": cls.TRANSFORMER, "itransformer": …, "tft": …` | Model names smuggled into a *family* mapping — a workaround for the family drift in §A.4. |
| `src/models/device.py:464,617` | `elif family in ("lstm","gru")` | Memory/GPU heuristics by family string. |
| `src/models/ensemble/validator.py:85` | `if model_name.lower() in ("catboost","cat")` | Hardcoded install hint. |
| `src/data/features/strategies.py:146+` | `MODEL_FEATURE_STRATEGIES` keyed by 12 model names | Per-model baseline feature lists — legitimate content, wrong location (should live with the model spec). |
| `src/data/pipeline/config/feature_sets/core.py:~57` | `FEATURE_SET_ALIASES` with 12 model names + 20 aliases | Ditto. |
| `src/models/ensemble/validator.py:195-205` | Error message hardcodes `"Boosting: xgboost, lightgbm, catboost"`, `"lstm, gru, tcn, transformer"`, `"patchtst, itransformer"` | Even the *diagnostics* are hardcoded, so they go stale. |

`INFERRED` **Three completely separate model-construction paths exist**
(`ModelRegistry.create`, `training_ops._create_meta_model`,
`primary_model._build_model`), plus a fourth partial one in `calibrated_meta`.
Any capability declared on a `ModelSpec` is invisible to three of the four.

---

## A.6 Compatibility validation — what exists, and why it never runs

`OBSERVED` There **is** a real, well-written compatibility validator:
`src/models/ensemble/validator.py` (383 lines). `validate_ensemble_config`
(`:33`) groups models by input rank, allows homogeneous always, allows `{2,3}`
for stacking, rejects anything involving 4D, and
`_build_rank_compatibility_error_message` (`:148`) produces a genuinely
excellent multi-section error with REASON / YOUR CONFIGURATION / SUPPORTED
CONFIGURATIONS / RECOMMENDATIONS. **This is the right shape for what I propose
in Part C.**

`OBSERVED` **It is unreachable from the production path.** Call sites:

| Caller | Line | Reachable from `MLFactory.run()`? |
|---|---|---|
| `VotingEnsemble.__init__` | `voting.py:354` | **No** — `MLFactory` never constructs `VotingEnsemble` (grep for `VotingEnsemble` outside `src/models/ensemble/` returns only a comment at `src/models/__init__.py:51`) |
| `BlendingEnsemble.__init__` | `blending.py:201` | **No** (same) |
| `StackingEnsemble.__init__` | `stacking.py:313` | **No** (same) |
| `Trainer._validate_ensemble` | `trainer.py:391-394` | **Only if `self.model` is itself an ensemble** (`hasattr(self.model, "ensemble_type")`) |
| `loaders.validate_ensemble_base_models` | `loaders.py:264` | Only from YAML ensemble-config loading, not `ExperimentConfig` |

`OBSERVED` The live path is
`MLFactory._run_training` → `UnifiedOrchestrator._build_ensemble`
(`unified_orchestrator.py:494`) → `EnsembleService.build_ensemble`
(`ensemble_service.py:75`) → `_train_meta_learner` (`:346`), which instantiates
`meta_learner_map[name](...)` **directly** at `:416`. `validate_ensemble_config`
is never invoked.

`INFERRED` This explains a contradiction in the project's own docs.
`CLAUDE.md` Phase 57/60 claims *"All 8 ensemble combinations now PASS … 2D+4D,
3D+4D, 4D+4D, 2D+3D+4D all working"*, while `validator.py:141-142` says 4D in a
heterogeneous stack is unsupported and `tests/test_ensemble_input_ranks.py:31`
asserts that `["xgboost","patchtst"]` **raises**. Both are "true": the validator
rejects it, and the production path never asks the validator. **The validator
and the runtime disagree about what the system can do.**

`OBSERVED` The only compatibility check that *does* run in production is inside
`EnsembleService.build_ensemble`, and it is all runtime, post-training:

- `len(oof_predictions) < 2` → returns an error **dict**, not an exception (`:102`)
- all-zero probabilities → error dict (`:117`)
- NaN probabilities → **warning only** (`:124`)
- `OOFAligner.align(..., strategy="intersection")` → `ValueError` caught and
  converted to an error dict (`:135`)

`INFERRED` So an incompatible composition costs a **full training run** (hours
on 1.6M rows per `CLAUDE.md` Phase 89) before failing, and it fails *softly* —
`ensemble_metrics={"error": ...}` — so the pipeline reports success with an
empty ensemble.

### Answering the product's 11 questions today

| Question the architecture must answer | Today | Where |
|---|---|---|
| Are these models compatible? | Partially, in dead code | `validator.py:33` (unreachable) |
| What input shape does each require? | **Yes** | `ModelContract.input_rank` |
| Does each support this task? | **No concept of task** | — |
| Probabilistic output? | Assumed universally true | `PredictionResult.class_probabilities` required (`interfaces.py:157`) |
| Requires scaling? | Yes but **drifted** | contract vs instance (§A.4) |
| Multivariate? | Implicit in rank; not declared | — |
| Multi-horizon? | **No** — horizon is a string suffix | `unified_orchestrator.py:512` |
| Incremental training? | **No** — zero `partial_fit` in `src/models/` | — |
| What prediction representation? | Implicit: always `(n, n_classes)` probs | `interfaces.py:125` |
| Can predictions be combined? | **String-suffix match on `_h{N}`** | `unified_orchestrator.py:515` |
| Which adapter is required? | **Yes** | `ModelContract.adapter_id` |
| Which ensemble strategies are valid? | **No** — only stacking exists in the live path | `ensemble_service.py:400` |
| How are failures/missing predictions handled? | Hardcoded `strategy="intersection"` | `ensemble_service.py:131` |

---

## A.7 The composition-expressiveness gap

`OBSERVED` The live config surface (`src/config/experiment.py:95-130`):

```python
@dataclass
class TrainingSection:
    models: list[str] = field(default_factory=lambda: ["xgboost"])   # :103
    horizons: list[int] = ...                                        # :104
    build_ensemble: bool = True                                      # :128
    meta_learner: str = "ridge_meta"                                 # :129
```

The CLI mirrors it: `--base-models` (comma string) and `--meta-learner`
(`src/cli/commands/train.py:309-313`).

`OBSERVED` A richer `EnsembleConfig` **exists and is unused by training**:

```python
class EnsembleMethod(StrEnum):          # src/config/ensemble.py:28
    STACKING = "stacking"; VOTING = "voting"
    BLENDING = "blending"; WEIGHTED_AVERAGE = "weighted_average"

@dataclass
class EnsembleConfig(BaseConfig):        # :61
    method: str = "stacking"
    base_models: list[str] = ...
    meta_learner: str = "ridge"
    alignment_strategy: str = "intersection"
    n_classes: int = 3
    calibrate_base_models: bool = True
    calibrate_ensemble: bool = True
```

Grep for consumers outside `src/config/`: only re-exports in `__init__.py`
files. `EnsembleMethod` has **zero** functional consumers.

`INFERRED` **The core product requirement — `A+B+C -> X`, then `A+D+F -> Y` —
is not expressible.** You get exactly one flat model list, exactly one implicit
strategy (stacking), and one meta-learner. Running the second composition means
a second full experiment, retraining `A` from scratch. There is no notion of a
named composition, no reuse of base-model OOF across strategies, and no way to
name a strategy other than by choosing a meta-learner.

---

## A.8 What is genuinely good (keep)

`OBSERVED`

- `AdapterRegistry` decorator pattern (`registry.py:31`) — correct shape.
- `ModelRegistry` decorator + alias support + `is_available()` graceful
  optional-dependency handling (`registry.py:408`) — this is the sktime
  `python_dependencies` tag pattern, arrived at independently.
- `ModelContract` as a declarative requirement record — right idea.
- `DataContract` with schema hash + strict validation (`data_contract.py:129-253`).
- `PredictionResult` as a single canonical prediction container
  (`src/core/interfaces.py:125`) with shape validation in `__post_init__`.
- `OOFPrediction` full-length NaN-padded schema **with an explicit contract
  check** (`ensemble_service.py:232-240`) — a real, enforced invariant.
- `_build_rank_compatibility_error_message` — best-in-class error UX, just
  unreachable.
- `TrainerProtocol` / `InferenceBundle` `Protocol`s (`src/core/protocols.py:20,49`)
  — the structural-typing idiom is already in the codebase.

---

# PART B — Research: How Mature Frameworks Compose Heterogeneous Components

## B.1 scikit-learn — the estimator contract + `Tags`

`OBSERVED` (from `sklearn/utils/_tags.py`, `develop.html`)

Since 1.6, capabilities are a **public dataclass tree** returned by
`__sklearn_tags__()`:

```
Tags
├─ estimator_type: "classifier"|"regressor"|"transformer"|"clusterer"|
│                  "outlier_detector"|"density_estimator"|None
├─ target_tags: TargetTags(required, one_d_labels, two_d_labels,
│                          positive_only, multi_output, single_output)
├─ transformer_tags: TransformerTags(preserves_dtype=["float64"]) | None
├─ classifier_tags: ClassifierTags(poor_score, multi_class, multi_label) | None
├─ regressor_tags: RegressorTags(poor_score) | None
├─ input_tags: InputTags(one_d_array, two_d_array, three_d_array, sparse,
│                        categorical, string, dict, positive_only,
│                        allow_nan, pairwise)
├─ array_api_support: bool
├─ no_validation: bool
├─ non_deterministic: bool
└─ requires_fit: bool
```

Key design lessons:

1. **Tags are a typed dataclass tree, not a `dict[str, Any]`.** Typos are type
   errors. Sub-dataclasses are `None` when inapplicable (`classifier_tags is
   None` ⟺ not a classifier) — so *presence* itself carries meaning.
2. **Composition is orthogonal to capability.** `Pipeline` and
   `ColumnTransformer` compose by *position and name*, not by knowing what the
   steps are. `ColumnTransformer` is the direct precedent for routing different
   input subsets to different components.
3. **`get_params`/`set_params` + `clone`** give reproducible reconstruction:
   `set_params` must be equivalent to `__init__`; fitted state gets a trailing
   underscore and is never set in `__init__`. This is what makes
   `GridSearchCV`/serialization generic.
4. **`check_estimator` is a conformance suite**, not documentation. Third-party
   estimators run it in their own CI. Extra mixin-specific checks run
   automatically based on declared type.
5. **Tags are extensible by subclassing** — new fields must have defaults, so
   old consumers never break.

## B.2 sktime / aeon — the maximal capability-tag system

`OBSERVED` (from `sktime/registry/_tags.py`, `aeon/base/_base.py`)

sktime's tag registry is the most complete example of exactly what this repo
needs. Selected tags, grouped:

**Object identity / environment**
`object_type`, `python_version`, `python_dependencies`, `env_marker`,
`requires_cython`, `maintainers`, `authors`, `sktime_version`

**Capability (bool unless noted)**
`capability:missing_values`, `capability:feature_importance`,
`capability:sample_weight`, `capability:multithreading`,
`capability:random_state`, `capability:exogenous`, `capability:insample`,
`capability:pred_int`, `capability:pred_int:insample`,
`capability:multivariate`, `capability:unequal_length`,
`capability:multioutput`, `capability:predict_proba`,
`capability:class_weight`, `capability:categorical_in_X/y`,
`capability:update` (online/stream learning), `capability:contractable`
(bounded fit time), `capability:train_estimate`

**Scitype / data-shape**
`scitype:transform-input` ("Series"|"Panel"),
`scitype:transform-output` ("Series"|"Panel"|"Primitives"),
`scitype:instancewise`, `scitype:transform-labels`, `requires_X`, `requires_y`

**Behavior**
`fit_is_empty`, `property:randomness` ("stochastic"|"deterministic"|
"derandomized"), `requires-fh-in-fit`, `transform-returns-same-time-index`

**Testing/CI**
`tests:core`, `tests:skip_all`, `tests:skip_by_name`, `tests:libs`

Mechanics (aeon `BaseAeonEstimator`):

- Tags declared as a **class-level `_tags: dict`**; resolution walks the MRO via
  `inspect.getmro()` so subclasses inherit and override naturally
  (`get_class_tags()`).
- **Instance-level dynamic overrides** in `_tags_dynamic`, set by `set_tags()`;
  `get_tags()` merges class + dynamic. This is how a *composite* (a pipeline or
  ensemble) computes its own capabilities from its members at construction time.
- **`fit`/`_fit` boilerplate separation**: the public `fit` does dependency
  checks, input validation, scitype conversion, `is_fitted` bookkeeping — then
  delegates to the subclass's `_fit`. Model authors write only `_fit`/`_predict`.
- **`all_estimators(filter_tags=...)`** is a *query* over the registry:
  "give me every forecaster with `capability:pred_int=True`". This is precisely
  the "which models can I mix?" question.
- **`reset()`** restores post-`__init__` state by removing non-hyperparameter
  attributes — clean re-fit without reconstruction.

`INFERRED` The single most transferable idea: **capabilities of a composite are
*computed* from its members, and the framework re-validates the composite's
computed capabilities against what the caller asked for.** This repo already
gropes toward it — `VotingEnsemble.requires_4d` (`voting.py:109-118`) computes
max member rank — but does it ad hoc per ensemble class.

## B.3 darts — capabilities as properties + wrapper-enforced validation

`OBSERVED` (from `darts/models/forecasting/forecasting_model.py`)

`ForecastingModel` declares capabilities as **properties**, not a dict:

`supports_multivariate`, `supports_past_covariates`,
`supports_future_covariates`, `supports_static_covariates`,
`supports_sample_weight`, `supports_transferable_series_prediction`,
`supports_probabilistic_prediction`,
`supports_likelihood_parameter_prediction`,
`supports_optimized_historical_forecasts`, `likelihood`,
`min_train_series_length`, `min_train_samples`, `output_chunk_length`,
`output_chunk_shift`, `extreme_lags` (a 7-tuple).

Enforcement pattern — `_fit_wrapper` / `_predict_wrapper`:

```python
if <covariate provided> and not getattr(self, f"supports_{series_name}"):
    raise ValueError(...)
if len(series) < self.min_train_series_length:
    raise ValueError(...)
```

`INFERRED` Two lessons: (a) **the base class's public wrapper enforces the
declared capabilities generically** — no per-model validation code; (b)
**quantitative capabilities** (`min_train_series_length`, `extreme_lags`,
`output_chunk_length`) matter as much as boolean ones, and the framework uses
them to *plan* (how much history to slice, whether auto-regression is needed).
This repo's `min_features`/`max_features`/`sequence_length` are the same idea,
just not enforced.

## B.4 GluonTS / NeuralForecast / Nixtla

`OBSERVED` GluonTS centers on probabilistic output: models emit a
`Distribution`/`Forecast` object rather than a point array, so "what
representation does this emit" is answered by the *type*. NeuralForecast uses
model adapters for quantile losses vs parametric distributions — the loss/head
is a swappable component, not baked into the architecture. Darts represents
probabilistic output as Monte Carlo samples stored in the `TimeSeries` object.

`INFERRED` All three treat **prediction representation as a first-class type**.
This repo flattens everything to `PredictionResult(class_predictions,
class_probabilities, confidence)` — workable for a pure 3-class classification
product, but it means "what does this emit" is unanswerable because the answer
is always the same by construction. That is fine *if* the product stays
classification-only; it becomes the next bottleneck if regression or quantile
targets are added. Design accordingly (see `PredictionSpace`, §C.4).

## B.5 MLflow — signatures and flavors

`OBSERVED` MLflow separates:
- **Flavor**: how to load/run the artifact (`sklearn`, `pytorch`, `pyfunc`).
  `python_function` is the universal lowest common denominator every flavor
  also implements.
- **Signature**: declared input/output schema, inferred from an
  `input_example`, **enforced at serve time** — inputs are coerced to the
  declared schema and rejected if they cannot be.

`INFERRED` The flavor/pyfunc split maps cleanly onto this repo: the native
model object is the "flavor"; the `TimeSeriesModel` protocol is the "pyfunc"
universal interface. `DataContract`'s schema hash (`data_contract.py:188`) is
already an MLflow-signature analogue — it just is not carried into bundles as a
serving-time gate.

## B.6 Python plugin patterns

`OBSERVED` (Python Packaging User Guide, `importlib.metadata`)

- **Entry points** are the standard mechanism: plugin packages declare
  `[project.entry-points."myapp.plugins"] name = "module:object"`, and the host
  discovers them with `importlib.metadata.entry_points(group="myapp.plugins")`
  — no dependency or prior knowledge of the plugin. Stdlib since 3.8,
  non-provisional since 3.10.
- **Namespace packages** and **decorator registries** are the two alternatives;
  decorators require the module to be imported, which is why entry points +
  lazy loading is preferred for a marketplace.
- **`Protocol` (PEP 544)** gives structural typing so third parties need not
  import and subclass your ABC — decoupling the plugin from the host's class
  hierarchy. `@runtime_checkable` allows an `isinstance` smoke check (methods
  only, not signatures) — so a real conformance suite is still needed.

## B.7 Distilled principles for this repo

`INFERRED`

| # | Principle | Source | Repo status |
|---|---|---|---|
| P1 | Capabilities are a **typed** record, declared **by the component**, not a central table | sklearn `Tags`, sktime `_tags` | Central table, untyped `str` family |
| P2 | Registry supports **capability queries**, not just name lookup | `all_estimators(filter_tags=)` | Name lookup only |
| P3 | Public `fit`/`predict` in the base **enforce declared capabilities**; authors write `_fit`/`_predict` | aeon, darts wrappers | No enforcement layer |
| P4 | Composite capabilities are **computed** from members and re-validated | sktime `set_tags`, darts | Ad hoc per ensemble class |
| P5 | Quantitative capabilities (`min_train_series_length`) drive **planning**, not just rejection | darts | Declared, not enforced |
| P6 | Conformance suite (`check_estimator`) is part of the extension contract | sklearn | Absent |
| P7 | Discovery via **entry points**, not import side effects | packaging guide | Import side effects only |
| P8 | Schema/signature **enforced at the boundary**, coerce-or-reject | MLflow | `DataContract` exists, enforced late |
| P9 | Prediction representation is a **type**, so combinability is a type question | GluonTS/darts | Flattened to one shape |

---

# PART C — Design Proposal

## C.0 Design constraints I am honoring

1. **Retrofittable.** No rewrite of the 12 models' `fit`/`predict` bodies.
   `BaseModel` survives; it gains a `capabilities()` classmethod and is
   *adapted* into the new spine.
2. **Additive first.** Every new module can land while old paths keep working;
   old tables become derived views so drift is impossible during transition.
3. **Fail before you compute.** All composition validation happens before any
   data is loaded.
4. **One source of truth per fact.** If two places can disagree, one of them is
   deleted or derived.

## C.1 Target layer map

```
                    ┌────────────────────────────────────────────────┐
 manifest.yaml ────▶│ ExperimentManifest  (§C.8)                     │
                    │   compositions: [ {members, strategy}, ... ]    │
                    └───────────────┬────────────────────────────────┘
                                    │  (no data loaded yet)
                    ┌───────────────▼────────────────────────────────┐
                    │ CompositionValidator  (§C.6)                   │
                    │   ModelRegistry ⟂ StrategyRegistry ⟂ DataProfile│
                    │   → ValidationReport[Diagnostic]               │
                    └───────────────┬────────────────────────────────┘
                                    │ (only if clean)
   Data ──▶ Transform ──▶ Features ─┼─▶ AdapterPlanner (§C.5)
                                    │      resolves rank/scaler/window per member
                                    ▼
                              ┌─────────────┐
                              │ TimeSeries  │  §C.3 Protocol
                              │   Model     │  §C.2 ModelCapabilities
                              └──────┬──────┘
                                     ▼
                              PredictionBatch (carries PredictionSpace §C.4)
                                     ▼
                              Calibrator (declares in/out PredictionSpace)
                                     ▼
                              ┌─────────────┐
                              │ Ensemble    │  §C.7 EnsembleStrategy
                              │  Strategy   │  + EnsembleCapabilities
                              └──────┬──────┘
                                     ▼
                                Evaluation
```

New package: `src/platform/` (deliberately *not* under `src/core/` — DECISIONS
item #10 flags a 4–5 s import SCC in `src/core`; the platform spine must import
in milliseconds and must not pull torch).

```
src/platform/
├── capabilities.py     ModelCapabilities, TaskType, PredictionKind, ...
├── protocol.py         TimeSeriesModel, Fittable, Incremental, ...
├── spec.py             ModelSpec
├── registry.py         ModelRegistry v2 (+ entry-point discovery + query)
├── space.py            PredictionSpace, PredictionBatch
├── strategy.py         EnsembleStrategy protocol, EnsembleCapabilities, registry
├── planner.py          AdapterPlan / AdapterPlanner
├── validate.py         CompositionValidator, Diagnostic, DiagnosticCode
├── conformance.py      check_model() — the check_estimator analogue
└── manifest.py         ExperimentManifest schema + loader
```

## C.2 `ModelCapabilities` — the declaration

`PROPOSED`

```python
# src/platform/capabilities.py
from __future__ import annotations
from dataclasses import dataclass, field, replace
from enum import StrEnum
from typing import Any

from src.core.types import DataRank


class TaskType(StrEnum):
    """What problem the model solves. Drives label compatibility."""
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    QUANTILE = "quantile"
    META = "meta"              # consumes OOF predictions, not features


class PredictionKind(StrEnum):
    """What representation predict() emits. Drives combinability."""
    CLASS_PROBA = "class_proba"      # (n, n_classes), rows sum to 1
    CLASS_LABEL = "class_label"      # (n,) ints, no proba available
    POINT = "point"                  # (n,) floats
    QUANTILES = "quantiles"          # (n, n_quantiles)
    DISTRIBUTION = "distribution"    # parametric params


class ScalerKind(StrEnum):
    NONE = "none"; ROBUST = "robust"; STANDARD = "standard"; MINMAX = "minmax"


class Determinism(StrEnum):
    DETERMINISTIC = "deterministic"      # bitwise-reproducible
    SEEDED = "seeded"                    # reproducible given random_state
    STOCHASTIC = "stochastic"            # e.g. non-deterministic CUDA kernels


@dataclass(frozen=True, slots=True)
class ModelCapabilities:
    """
    Everything the platform needs to know about a model WITHOUT importing it.

    Frozen + slots: cheap to construct, hashable-adjacent, typo-proof.
    Every new field MUST have a default so third-party specs never break.
    """

    # ---- identity -------------------------------------------------------
    name: str
    family: str                                # free-form; NOT load-bearing
    version: str = "1.0.0"                     # spec version, for bundles

    # ---- task / target --------------------------------------------------
    task: TaskType = TaskType.CLASSIFICATION
    n_classes: int | None = 3                  # None for regression/quantile
    supports_binary: bool = True               # can run with n_classes=2
    supports_multiclass: bool = True
    supports_multi_horizon: bool = False       # one fit → many horizons
    horizons_per_fit: int = 1

    # ---- input shape ----------------------------------------------------
    input_rank: DataRank = DataRank.TABULAR_2D
    supports_multivariate: bool = True         # >1 feature column
    min_features: int = 4
    max_features: int = 200
    sequence_length: int | None = None         # required iff rank >= 3
    min_sequence_length: int | None = None     # e.g. TCN receptive field = 61
    patch_length: int | None = None
    n_timeframes: int | None = None            # required iff rank == 4
    min_train_samples: int = 100               # darts-style planning input

    # ---- preprocessing requirements ------------------------------------
    requires_scaling: bool = True
    scaler_kind: ScalerKind = ScalerKind.ROBUST
    handles_nan: bool = False                  # sklearn InputTags.allow_nan
    handles_categorical: bool = False
    requires_dense: bool = True

    # ---- output ---------------------------------------------------------
    prediction_kind: PredictionKind = PredictionKind.CLASS_PROBA
    supports_proba: bool = True
    proba_is_calibrated: bool = False          # True ⇒ skip ProbabilityCalibrator
    supports_feature_importance: bool = False
    supports_attention: bool = False

    # ---- training capabilities ------------------------------------------
    supports_sample_weight: bool = True
    supports_class_weight: bool = True
    supports_incremental: bool = False         # has partial_fit / warm_start
    supports_warm_start: bool = False
    supports_early_stopping: bool = False
    supports_validation_set: bool = True       # fit() takes X_val/y_val
    determinism: Determinism = Determinism.SEEDED

    # ---- resources -------------------------------------------------------
    supports_gpu: bool = False
    requires_gpu: bool = False
    supports_n_jobs: bool = False
    python_dependencies: tuple[str, ...] = ()  # sktime-style, PEP 440 strings

    # ---- data-pipeline preferences (moved here from strategies.py) ------
    feature_mode: str = "engineered"           # engineered | raw | hybrid | oof_probs
    mtf_mode: str = "none"                     # none | indicators | multi_stream
    primary_timeframe: str = "5min"
    mtf_timeframes: tuple[str, ...] = ()
    preferred_feature_families: tuple[str, ...] = ()

    # ---- free-form escape hatch ------------------------------------------
    extra: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    def __post_init__(self) -> None:
        if self.input_rank.value >= 3 and self.sequence_length is None:
            raise ValueError(
                f"{self.name}: sequence_length is required for rank "
                f"{self.input_rank.value}D models"
            )
        if self.input_rank == DataRank.MULTI_TF_4D and not self.n_timeframes:
            raise ValueError(f"{self.name}: n_timeframes required for 4D models")
        if self.min_features > self.max_features:
            raise ValueError(f"{self.name}: min_features > max_features")
        if self.task is TaskType.CLASSIFICATION and not self.n_classes:
            raise ValueError(f"{self.name}: classification needs n_classes")
        if self.prediction_kind is PredictionKind.CLASS_PROBA and not self.supports_proba:
            raise ValueError(f"{self.name}: CLASS_PROBA implies supports_proba")
        if self.requires_gpu and not self.supports_gpu:
            raise ValueError(f"{self.name}: requires_gpu implies supports_gpu")

    # ---- derived --------------------------------------------------------
    @property
    def requires_sequences(self) -> bool:
        return self.input_rank.value >= 3

    @property
    def requires_multi_timeframe(self) -> bool:
        return self.input_rank == DataRank.MULTI_TF_4D

    @property
    def effective_sequence_length(self) -> int | None:
        """Honor min_sequence_length — closes DECISIONS.md item #4."""
        if self.sequence_length is None:
            return None
        if self.min_sequence_length is None:
            return self.sequence_length
        return max(self.sequence_length, self.min_sequence_length)

    def with_overrides(self, **kw: Any) -> ModelCapabilities:
        """sktime set_tags() analogue — used by composites (§C.7)."""
        return replace(self, **kw)
```

`PROPOSED` **Why this field set.** It answers, by construction, every question
in the product requirement:

| Product question | Field |
|---|---|
| Input shape? | `input_rank`, `sequence_length`, `n_timeframes`, `patch_length` |
| Supports this task? | `task`, `n_classes`, `supports_binary/multiclass` |
| Probabilistic output? | `supports_proba`, `prediction_kind`, `proba_is_calibrated` |
| Requires scaling? | `requires_scaling`, `scaler_kind` |
| Multivariate? | `supports_multivariate`, `min_features`, `max_features` |
| Multi-horizon? | `supports_multi_horizon`, `horizons_per_fit` |
| Incremental? | `supports_incremental`, `supports_warm_start` |
| Prediction representation? | `prediction_kind` |
| Which adapter? | derived from `input_rank` by the planner (§C.5) |
| NaN handling? | `handles_nan` |

## C.3 `TimeSeriesModel` — the canonical Protocol

`PROPOSED` Structural, so third parties never import `BaseModel`.

```python
# src/platform/protocol.py
from __future__ import annotations
from pathlib import Path
from typing import Any, Protocol, runtime_checkable
import numpy as np

from src.core.interfaces import PredictionResult
from src.models.base import TrainingMetrics          # reuse, do not duplicate
from .capabilities import ModelCapabilities


@runtime_checkable
class TimeSeriesModel(Protocol):
    """
    The one interface every model in the marketplace satisfies.

    Deliberately IDENTICAL to today's BaseModel surface (src/models/base.py:244-334)
    plus `capabilities()`. Every existing model already satisfies all of it
    except `capabilities()`, which the adapter shim supplies (§D.1).
    """

    # -- declaration ----------------------------------------------------
    @classmethod
    def capabilities(cls) -> ModelCapabilities:
        """Static, import-cheap capability declaration. MUST NOT touch config."""
        ...

    def get_capabilities(self) -> ModelCapabilities:
        """Instance capabilities: class caps possibly narrowed by this config
        (e.g. n_classes=2 when configured for binary mode)."""
        ...

    # -- configuration (sklearn get_params/set_params analogue) ---------
    @property
    def config(self) -> dict[str, Any]: ...

    def get_default_config(self) -> dict[str, Any]: ...

    # -- state ----------------------------------------------------------
    @property
    def is_fitted(self) -> bool: ...

    # -- training -------------------------------------------------------
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        sample_weights: np.ndarray | None = None,
        config: dict[str, Any] | None = None,
    ) -> TrainingMetrics: ...

    # -- inference ------------------------------------------------------
    def predict(self, X: np.ndarray) -> PredictionResult: ...

    # -- persistence ----------------------------------------------------
    def save(self, path: Path) -> None: ...
    def load(self, path: Path) -> None: ...


@runtime_checkable
class SupportsProba(Protocol):
    def predict_proba(self, X: np.ndarray) -> np.ndarray: ...


@runtime_checkable
class SupportsIncremental(Protocol):
    """Declared via ModelCapabilities.supports_incremental=True."""
    def partial_fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weights: np.ndarray | None = None,
    ) -> TrainingMetrics: ...


@runtime_checkable
class SupportsFeatureImportance(Protocol):
    def get_feature_importance(self) -> dict[str, float] | None: ...
```

`PROPOSED` **Design note (aeon P3).** I deliberately do **not** introduce a
`_fit`/`fit` split. In a green field it is right; here it would mean editing 12
model bodies. Instead the *enforcement wrapper* lives one level up in a
`ModelRunner` (§C.5) that owns the model instance and applies the declared
capabilities — same benefit, zero edits to model code.

## C.4 `PredictionSpace` — "can these legally be combined?"

`PROPOSED` This is the missing concept. Today the answer is
`k.endswith("_h5")`.

```python
# src/platform/space.py
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .capabilities import PredictionKind, TaskType


@dataclass(frozen=True, slots=True)
class PredictionSpace:
    """
    The semantic space a set of predictions lives in.

    Two prediction sets are combinable IFF their PredictionSpaces are equal.
    This replaces the `_h{N}` string-suffix check at
    src/models/training/unified_orchestrator.py:512-517.
    """
    task: TaskType
    kind: PredictionKind
    n_outputs: int                     # n_classes, or n_quantiles, or 1
    class_labels: tuple[int, ...]      # e.g. (-1, 0, 1) or (0, 1)
    horizon: int                       # bars
    label_recipe_id: str               # hash of (method, barriers, costs, symbol)
    index_space_id: str                # hash of (symbol, timeframe, date range)

    def incompatibility(self, other: PredictionSpace) -> str | None:
        """Human-readable reason these cannot be combined, or None."""
        if self.task != other.task:
            return f"task {self.task} vs {other.task}"
        if self.kind != other.kind:
            return f"prediction kind {self.kind} vs {other.kind}"
        if self.n_outputs != other.n_outputs:
            return f"n_outputs {self.n_outputs} vs {other.n_outputs}"
        if self.class_labels != other.class_labels:
            return f"class labels {self.class_labels} vs {other.class_labels}"
        if self.horizon != other.horizon:
            return f"horizon h{self.horizon} vs h{other.horizon}"
        if self.label_recipe_id != other.label_recipe_id:
            return ("labels were generated with different barrier/cost "
                    f"parameters ({self.label_recipe_id[:8]} vs "
                    f"{other.label_recipe_id[:8]})")
        if self.index_space_id != other.index_space_id:
            return "predictions are indexed over different samples"
        return None


@dataclass(frozen=True)
class PredictionBatch:
    """PredictionResult + the space it lives in + coverage."""
    values: np.ndarray                 # (n_total, n_outputs), NaN-padded
    space: PredictionSpace
    valid_index: np.ndarray            # positions with real predictions
    n_total: int
    producer: str                      # model key, e.g. "xgboost@h20"

    @property
    def coverage(self) -> float:
        return len(self.valid_index) / max(self.n_total, 1)
```

`INFERRED` `label_recipe_id` is the sleeper. `CLAUDE.md` Phase 94 records a bug
where labels and backtest used different ATR formulas and different transaction
costs. A `label_recipe_id` in the prediction space makes "these two models were
trained against different labels" a **validation error**, not a silent
correctness bug.

## C.5 Registry v2 + `ModelSpec` + plugin discovery

`PROPOSED`

```python
# src/platform/spec.py
from __future__ import annotations
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any
from .capabilities import ModelCapabilities
from .protocol import TimeSeriesModel


@dataclass(frozen=True)
class ModelSpec:
    """
    THE single source of truth for one model in the marketplace.

    Everything the old system kept in 6 different tables lives here:
      MODEL_CONTRACTS, MODEL_FAMILIES, MODEL_DATA_RANKS, MODEL_ADAPTER_MAP,
      MODEL_FEATURE_STRATEGIES, FEATURE_SET_ALIASES
    all become derived views over the spec table.
    """
    capabilities: ModelCapabilities
    factory: Callable[..., TimeSeriesModel]      # usually the class itself
    aliases: tuple[str, ...] = ()
    description: str = ""
    default_config: dict[str, Any] = field(default_factory=dict)
    source: str = "builtin"                      # "builtin" | entry-point dist name

    @property
    def name(self) -> str:
        return self.capabilities.name

    def create(self, config: dict[str, Any] | None = None, **kw: Any) -> TimeSeriesModel:
        merged = {**self.default_config, **(config or {})}
        return self.factory(config=merged, **kw)
```

```python
# src/platform/registry.py
from __future__ import annotations
import logging
from collections.abc import Callable, Iterator
from importlib.metadata import entry_points
from typing import Any

from .capabilities import ModelCapabilities
from .spec import ModelSpec

logger = logging.getLogger(__name__)
ENTRY_POINT_GROUP = "mlfactory.models"


class ModelNotFound(LookupError):
    def __init__(self, name: str, available: list[str]) -> None:
        near = [a for a in available if a.startswith(name[:3].lower())]
        hint = f" Did you mean: {near}?" if near else ""
        super().__init__(
            f"No model registered under '{name}'.{hint}\n"
            f"Registered ({len(available)}): {available}\n"
            f"Third-party models are discovered from the "
            f"'{ENTRY_POINT_GROUP}' entry-point group."
        )


class ModelRegistry:
    """Open registry. Decorator for builtins, entry points for plugins."""

    _specs: dict[str, ModelSpec] = {}
    _alias: dict[str, str] = {}
    _plugins_loaded: bool = False

    # ---------------- registration ----------------
    @classmethod
    def register(
        cls,
        capabilities: ModelCapabilities,
        *,
        aliases: tuple[str, ...] = (),
        description: str = "",
        default_config: dict[str, Any] | None = None,
    ) -> Callable[[type], type]:
        """
        @ModelRegistry.register(ModelCapabilities(name="xgboost", ...))
        class XGBoostModel(BaseModel): ...
        """
        def decorator(klass: type) -> type:
            spec = ModelSpec(
                capabilities=capabilities,
                factory=klass,
                aliases=aliases,
                description=description or (klass.__doc__ or "").strip().split("\n")[0],
                default_config=default_config or {},
            )
            cls.add(spec)
            # Give the class a conformant capabilities() so it satisfies the Protocol
            klass.capabilities = classmethod(lambda _c, _caps=capabilities: _caps)  # type: ignore[attr-defined]
            return klass
        return decorator

    @classmethod
    def add(cls, spec: ModelSpec, *, replace: bool = False) -> None:
        key = spec.name.lower()
        if key in cls._specs and not replace:
            raise ValueError(
                f"Model '{key}' already registered by {cls._specs[key].source}; "
                f"pass replace=True to override."
            )
        cls._specs[key] = spec
        for a in spec.aliases:
            cls._alias.setdefault(a.lower(), key)

    # ---------------- plugin discovery ----------------
    @classmethod
    def load_plugins(cls, *, strict: bool = False) -> list[str]:
        """
        Discover third-party models from the 'mlfactory.models' entry-point group.
        Each entry point resolves to either a ModelSpec or an iterable of them.
        A broken plugin NEVER takes down the host (unless strict=True).
        """
        if cls._plugins_loaded:
            return []
        loaded: list[str] = []
        for ep in entry_points(group=ENTRY_POINT_GROUP):
            try:
                obj = ep.load()
                specs = obj() if callable(obj) and not isinstance(obj, ModelSpec) else obj
                for spec in ([specs] if isinstance(spec := specs, ModelSpec) else specs):
                    cls.add(ModelSpec(**{**spec.__dict__, "source": ep.value}))
                    loaded.append(spec.name)
            except Exception:                       # noqa: BLE001 - isolation is the point
                logger.exception("Failed to load model plugin %r", ep.name)
                if strict:
                    raise
        cls._plugins_loaded = True
        return loaded

    # ---------------- lookup ----------------
    @classmethod
    def get(cls, name: str) -> ModelSpec:
        key = name.lower().strip()
        key = cls._alias.get(key, key)
        if key not in cls._specs:
            raise ModelNotFound(name, sorted(cls._specs))
        return cls._specs[key]

    @classmethod
    def create(cls, name: str, config: dict[str, Any] | None = None, **kw: Any):
        return cls.get(name).create(config, **kw)

    @classmethod
    def caps(cls, name: str) -> ModelCapabilities:
        """Capability lookup with NO import of the model implementation."""
        return cls.get(name).capabilities

    # ---------------- capability QUERY (sktime all_estimators analogue) ----------------
    @classmethod
    def find(cls, **predicates: Any) -> list[ModelSpec]:
        """
        ModelRegistry.find(input_rank=DataRank.TABULAR_2D, supports_proba=True)
        ModelRegistry.find(task=TaskType.CLASSIFICATION, supports_incremental=True)
        Callable predicates are supported: find(max_features=lambda v: v >= 150)
        """
        out = []
        for spec in cls._specs.values():
            c = spec.capabilities
            ok = True
            for field_name, want in predicates.items():
                got = getattr(c, field_name)
                ok = want(got) if callable(want) else got == want
                if not ok:
                    break
            if ok:
                out.append(spec)
        return sorted(out, key=lambda s: s.name)

    @classmethod
    def compatible_with(cls, name: str) -> list[str]:
        """Models that can share a composition with `name` (same rank + space)."""
        c = cls.caps(name)
        return [s.name for s in cls.find(
            input_rank=c.input_rank, task=c.task, n_classes=c.n_classes
        )]

    @classmethod
    def __iter__(cls) -> Iterator[ModelSpec]:
        return iter(cls._specs.values())
```

`PROPOSED` **Registration then looks like this** (only change to
`xgboost_model.py`: the decorator arguments):

```python
@ModelRegistry.register(
    ModelCapabilities(
        name="xgboost", family="boosting",
        task=TaskType.CLASSIFICATION, n_classes=3,
        input_rank=DataRank.TABULAR_2D,
        min_features=40, max_features=200,
        requires_scaling=False, scaler_kind=ScalerKind.NONE,
        handles_nan=True,                          # XGBoost natively handles NaN
        prediction_kind=PredictionKind.CLASS_PROBA,
        supports_proba=True, proba_is_calibrated=False,
        supports_feature_importance=True,
        supports_sample_weight=True, supports_early_stopping=True,
        supports_incremental=True, supports_warm_start=True,   # xgb_model=
        supports_gpu=True, supports_n_jobs=True,
        determinism=Determinism.SEEDED,
        python_dependencies=("xgboost>=2.0.0",),
        feature_mode="engineered", mtf_mode="indicators",
        primary_timeframe="15min",
    ),
    aliases=("xgb",),
)
class XGBoostModel(BaseModel):
    ...   # body unchanged
```

## C.6 Adapter layer — rank resolution as a *plan*, not a lookup

`PROPOSED` Replace name→adapter lookup with a planner that reads capabilities
and a data profile, and emits an executable plan. Existing adapters are the
executors — unchanged.

```python
# src/platform/planner.py
from __future__ import annotations
from dataclasses import dataclass
import pandas as pd

from src.core.types import DataRank
from src.data.adapters.registry import AdapterRegistry     # reuse as-is
from .capabilities import ModelCapabilities, ScalerKind


@dataclass(frozen=True)
class DataProfile:
    """What the pipeline actually produced. The DataContract's planning view."""
    n_samples: int
    n_features: int
    feature_columns: tuple[str, ...]
    available_timeframes: tuple[str, ...]
    has_nan: bool
    n_classes: int
    class_labels: tuple[int, ...]
    horizon: int
    label_recipe_id: str
    index_space_id: str
    bars_per_day: int


@dataclass(frozen=True)
class AdapterPlan:
    """Fully resolved recipe for turning DataProfile into this model's input."""
    adapter_id: str                      # "tabular" | "sequence" | "multi_stream"
    sequence_length: int | None
    timeframes: tuple[str, ...]
    scaler_kind: ScalerKind
    feature_columns: tuple[str, ...]     # already truncated to max_features
    needs_additional_dfs: bool
    imputation: str                      # "none" | "ffill_then_zero"

    def build(self, **kw):
        return AdapterRegistry.create(self.adapter_id, **kw)


_RANK_TO_ADAPTER = {
    DataRank.TABULAR_2D: "tabular",
    DataRank.SEQUENCE_3D: "sequence",
    DataRank.MULTI_TF_4D: "multi_stream",
}


def plan_adapter(
    caps: ModelCapabilities,
    profile: DataProfile,
    selected_features: tuple[str, ...],
) -> AdapterPlan:
    """Pure function. No I/O, no model import. Fully unit-testable."""
    adapter_id = _RANK_TO_ADAPTER[caps.input_rank]
    tfs: tuple[str, ...] = ()
    if caps.input_rank is DataRank.MULTI_TF_4D:
        tfs = (caps.primary_timeframe, *caps.mtf_timeframes)
    return AdapterPlan(
        adapter_id=adapter_id,
        # honors min_sequence_length — closes DECISIONS.md #4 mechanically
        sequence_length=caps.effective_sequence_length,
        timeframes=tfs,
        scaler_kind=ScalerKind.NONE if not caps.requires_scaling else caps.scaler_kind,
        feature_columns=selected_features[: caps.max_features],
        needs_additional_dfs=caps.input_rank is DataRank.MULTI_TF_4D,
        imputation="none" if caps.handles_nan else "ffill_then_zero",
    )
```

`PROPOSED` **`ModelRunner`** — the darts `_fit_wrapper` analogue, the one place
that enforces declared capabilities generically:

```python
# src/platform/runner.py  (sketch)
class ModelRunner:
    """Owns a TimeSeriesModel + its plan. Enforces caps so models need no checks."""

    def __init__(self, spec: ModelSpec, plan: AdapterPlan, config: dict) -> None:
        self.caps, self.plan = spec.capabilities, plan
        self.model = spec.create(config)

    def fit(self, X, y, X_val, y_val, sample_weights=None):
        c = self.caps
        if X.ndim != c.input_rank.value:
            raise ShapeContractError(c.name, expected=c.input_rank.value, got=X.ndim)
        if X.shape[0] < c.min_train_samples:
            raise InsufficientDataError(c.name, need=c.min_train_samples, got=X.shape[0])
        if sample_weights is not None and not c.supports_sample_weight:
            logger.warning("%s ignores sample_weight (declared unsupported)", c.name)
            sample_weights = None
        if not c.handles_nan and np.isnan(X).any():
            raise NanContractError(c.name)          # today: silent garbage
        return self.model.fit(X, y, X_val, y_val, sample_weights)

    def predict(self, X) -> PredictionBatch:
        r = self.model.predict(X)
        return PredictionBatch(values=r.class_probabilities, space=self.space, ...)
```

## C.7 The compatibility validator — clear, actionable, *pre-flight*

`PROPOSED` This is the piece the product requirement most needs. It runs on the
manifest before a single row is read.

```python
# src/platform/validate.py
from __future__ import annotations
from dataclasses import dataclass, field
from enum import StrEnum
from .capabilities import ModelCapabilities, TaskType
from .planner import DataProfile
from .registry import ModelRegistry, ModelNotFound
from .strategy import StrategyRegistry


class Severity(StrEnum):
    ERROR = "error"; WARNING = "warning"; INFO = "info"


class Code(StrEnum):
    """Stable, greppable diagnostic codes. Documented once, referenced forever."""
    MODEL_UNKNOWN         = "MDL001"
    MODEL_DEPS_MISSING    = "MDL002"
    FEATURE_FLOOR         = "MDL003"   # data has fewer features than min_features
    FEATURE_CEILING       = "MDL004"
    TASK_MISMATCH         = "MDL005"
    NCLASSES_MISMATCH     = "MDL006"
    NAN_UNSUPPORTED       = "MDL007"
    INSUFFICIENT_SAMPLES  = "MDL008"
    SEQ_TOO_SHORT         = "MDL009"   # min_sequence_length not satisfiable
    TIMEFRAME_MISSING     = "MDL010"   # 4D model needs a TF the pipeline lacks
    GPU_REQUIRED          = "MDL011"
    STRATEGY_UNKNOWN      = "ENS001"
    RANK_MIX_UNSUPPORTED  = "ENS002"
    PROBA_REQUIRED        = "ENS003"
    MEMBER_COUNT          = "ENS004"
    SPACE_MISMATCH        = "ENS005"   # PredictionSpace incompatibility
    DUPLICATE_MEMBER      = "ENS006"
    HORIZON_MIX           = "ENS007"
    LOW_DIVERSITY         = "ENS008"   # warning
    OOF_UNAVAILABLE       = "ENS009"


@dataclass(frozen=True)
class Diagnostic:
    code: Code
    severity: Severity
    message: str                     # what is wrong
    offenders: tuple[str, ...] = ()  # which members
    remedy: str = ""                 # what to DO about it
    composition: str = ""

    def render(self) -> str:
        who = f" [{', '.join(self.offenders)}]" if self.offenders else ""
        where = f"{self.composition}: " if self.composition else ""
        fix = f"\n      FIX: {self.remedy}" if self.remedy else ""
        return f"  {self.severity.upper():7} {self.code} {where}{self.message}{who}{fix}"


@dataclass
class ValidationReport:
    diagnostics: list[Diagnostic] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not any(d.severity is Severity.ERROR for d in self.diagnostics)

    def raise_if_failed(self) -> None:
        if self.ok:
            return
        raise CompositionError("Composition is not runnable:\n" + self.render())

    def render(self) -> str:
        order = {Severity.ERROR: 0, Severity.WARNING: 1, Severity.INFO: 2}
        return "\n".join(d.render() for d in sorted(self.diagnostics, key=lambda d: order[d.severity]))


class CompositionValidator:
    """Answers every product question BEFORE training starts."""

    def validate(self, manifest, profile: DataProfile) -> ValidationReport:
        rep = ValidationReport()
        for comp in manifest.compositions:
            self._validate_composition(comp, profile, rep)
        return rep

    def _validate_composition(self, comp, profile, rep) -> None:
        add = lambda **kw: rep.diagnostics.append(Diagnostic(composition=comp.name, **kw))

        # ---- 1. resolve members -------------------------------------
        caps: dict[str, ModelCapabilities] = {}
        for m in comp.members:
            try:
                caps[m.ref] = ModelRegistry.caps(m.ref)
            except ModelNotFound as e:
                add(code=Code.MODEL_UNKNOWN, severity=Severity.ERROR,
                    message=str(e).splitlines()[0], offenders=(m.ref,),
                    remedy="Run `ml models list` to see registered models, or "
                           "install the plugin providing it.")
        if len(caps) != len(comp.members):
            return                                   # cannot continue meaningfully

        if len(set(caps)) != len(comp.members):
            add(code=Code.DUPLICATE_MEMBER, severity=Severity.ERROR,
                message="A model appears twice in one composition.",
                remedy="Give the second instance a distinct `as:` alias, or remove it.")

        # ---- 2. per-member data fit ---------------------------------
        for name, c in caps.items():
            if profile.n_features < c.min_features:
                add(code=Code.FEATURE_FLOOR, severity=Severity.ERROR,
                    message=(f"{name} needs >= {c.min_features} features; the feature "
                             f"pipeline produced {profile.n_features}."),
                    offenders=(name,),
                    remedy=(f"Raise features.n_features to >= {c.min_features}, "
                            f"or swap {name} for one of "
                            f"{[s.name for s in ModelRegistry.find(input_rank=c.input_rank, max_features=lambda v: True) if s.capabilities.min_features <= profile.n_features][:4]}."))
            if profile.has_nan and not c.handles_nan:
                add(code=Code.NAN_UNSUPPORTED, severity=Severity.ERROR,
                    message=f"{name} cannot consume NaN, and the feature frame has NaN.",
                    offenders=(name,),
                    remedy="Enable features.impute, or use a NaN-tolerant model "
                           f"({[s.name for s in ModelRegistry.find(handles_nan=True)][:4]}).")
            if c.task is not TaskType.META and c.n_classes != profile.n_classes:
                add(code=Code.NCLASSES_MISMATCH, severity=Severity.ERROR,
                    message=(f"{name} declares n_classes={c.n_classes}; labels have "
                             f"{profile.n_classes}."),
                    offenders=(name,),
                    remedy=("Set labeling.binary_mode to match, or pick a model with "
                            f"supports_binary=True."))
            if c.requires_multi_timeframe:
                need = {c.primary_timeframe, *c.mtf_timeframes}
                missing = sorted(need - set(profile.available_timeframes))
                if missing:
                    add(code=Code.TIMEFRAME_MISSING, severity=Severity.ERROR,
                        message=f"{name} needs timeframes {missing} which the pipeline does not build.",
                        offenders=(name,),
                        remedy=f"Add {missing} to data.mtf.timeframes and set mtf.enabled=true.")
            seq = c.effective_sequence_length
            if seq and profile.n_samples < seq + c.min_train_samples:
                add(code=Code.INSUFFICIENT_SAMPLES, severity=Severity.ERROR,
                    message=(f"{name} windows at {seq} bars and needs >= "
                             f"{c.min_train_samples} training windows; only "
                             f"{profile.n_samples} rows available."),
                    offenders=(name,),
                    remedy="Widen the date range, or use a 2D model.")

        # ---- 3. strategy admissibility ------------------------------
        try:
            scaps = StrategyRegistry.caps(comp.strategy.kind)
        except LookupError:
            add(code=Code.STRATEGY_UNKNOWN, severity=Severity.ERROR,
                message=f"Unknown ensemble strategy '{comp.strategy.kind}'.",
                remedy=f"Available: {StrategyRegistry.names()}")
            return

        n = len(caps)
        if not (scaps.min_members <= n <= scaps.max_members):
            add(code=Code.MEMBER_COUNT, severity=Severity.ERROR,
                message=(f"Strategy '{comp.strategy.kind}' takes "
                         f"{scaps.min_members}..{scaps.max_members} members; got {n}."),
                remedy="Add or remove members.")

        ranks = {c.input_rank for c in caps.values()}
        if len(ranks) > 1 and not scaps.supports_mixed_rank:
            by_rank: dict[int, list[str]] = {}
            for nm, c in caps.items():
                by_rank.setdefault(c.input_rank.value, []).append(nm)
            alts = StrategyRegistry.find(supports_mixed_rank=True)
            add(code=Code.RANK_MIX_UNSUPPORTED, severity=Severity.ERROR,
                message=(f"Strategy '{comp.strategy.kind}' feeds one X to every member, "
                         f"so all members must share an input rank. Found: "
                         + "; ".join(f"{r}D={v}" for r, v in sorted(by_rank.items()))),
                offenders=tuple(caps),
                remedy=(f"Either switch to a rank-agnostic strategy "
                        f"({[s.name for s in alts]} — these combine OOF predictions, "
                        f"which are always 2D), or keep only the "
                        f"{max(by_rank, key=lambda k: len(by_rank[k]))}D members: "
                        f"{by_rank[max(by_rank, key=lambda k: len(by_rank[k]))]}."))

        if scaps.requires_proba:
            bad = [n for n, c in caps.items() if not c.supports_proba]
            if bad:
                add(code=Code.PROBA_REQUIRED, severity=Severity.ERROR,
                    message=f"Strategy '{comp.strategy.kind}' needs probabilities; these do not emit them.",
                    offenders=tuple(bad),
                    remedy="Use `hard_vote` (labels only), or drop those members.")

        # ---- 4. prediction-space combinability ----------------------
        spaces = {n: profile.space_for(c) for n, c in caps.items()}
        ref_name, ref = next(iter(spaces.items()))
        for nm, sp in spaces.items():
            why = ref.incompatibility(sp)
            if why:
                add(code=Code.SPACE_MISMATCH, severity=Severity.ERROR,
                    message=f"'{nm}' predictions are not combinable with '{ref_name}': {why}.",
                    offenders=(ref_name, nm),
                    remedy="Members of one composition must share task, class set, "
                           "horizon, label recipe and sample index.")

        # ---- 5. soft advice -----------------------------------------
        fams = {c.family for c in caps.values()}
        if len(fams) == 1 and n >= 3:
            add(code=Code.LOW_DIVERSITY, severity=Severity.WARNING,
                message=f"All {n} members are '{fams.pop()}' models; ensemble gain is usually small.",
                remedy=f"Consider adding a member from a different family, e.g. "
                       f"{[s.name for s in ModelRegistry.find(input_rank=next(iter(ranks)))][:5]}.")
```

`PROPOSED` **Sample output** for `A+D+F` where D is 3D and F is 4D:

```
Composition is not runnable:
  ERROR   ENS002 momentum_stack: Strategy 'soft_vote' feeds one X to every member,
          so all members must share an input rank. Found: 2D=['xgboost']; 3D=['lstm']; 4D=['patchtst']
          [xgboost, lstm, patchtst]
      FIX: Either switch to a rank-agnostic strategy (['stack_ridge', 'stack_xgb',
           'rank_average'] — these combine OOF predictions, which are always 2D),
           or keep only the 2D members: ['xgboost'].
  ERROR   MDL010 momentum_stack: patchtst needs timeframes ['15min'] which the pipeline does not build. [patchtst]
      FIX: Add ['15min'] to data.mtf.timeframes and set mtf.enabled=true.
  WARNING ENS008 momentum_stack: All 3 members are 'neural' models; ensemble gain is usually small.
```

## C.8 Ensemble strategy as a first-class plugin

`PROPOSED` Currently "strategy" is implicit (always stacking) and the
meta-learner is chosen from a hardcoded 4-entry dict. Make strategies
registrable with their own capabilities.

```python
# src/platform/strategy.py
from __future__ import annotations
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol, runtime_checkable
import numpy as np
from .space import PredictionBatch, PredictionSpace


class MissingPolicy(StrEnum):
    """How to handle members with gaps (sequence models lose the first seq_len-1)."""
    INTERSECT     = "intersect"       # today's only behavior
    NAN_PROPAGATE = "nan_propagate"
    IMPUTE_PRIOR  = "impute_prior"    # fill with class prior
    DROP_MEMBER   = "drop_member"     # drop members below min_coverage


@dataclass(frozen=True, slots=True)
class EnsembleCapabilities:
    name: str
    supports_mixed_rank: bool          # True ⟺ consumes OOF, not raw X
    requires_oof: bool
    requires_proba: bool
    requires_fit: bool                 # meta-learner needs training
    min_members: int = 2
    max_members: int = 64
    supports_weights: bool = False
    supports_missing: tuple[MissingPolicy, ...] = (MissingPolicy.INTERSECT,)
    emits: PredictionSpace | None = None    # None ⇒ same space as members
    description: str = ""


@runtime_checkable
class EnsembleStrategy(Protocol):
    @classmethod
    def capabilities(cls) -> EnsembleCapabilities: ...

    def fit(self, members: dict[str, PredictionBatch], y: np.ndarray) -> None: ...
    def combine(self, members: dict[str, PredictionBatch]) -> PredictionBatch: ...
    def save(self, path) -> None: ...
    def load(self, path) -> None: ...


class StrategyRegistry:
    """Same decorator + entry-point pattern as ModelRegistry."""
    ENTRY_POINT_GROUP = "mlfactory.strategies"
    ...
```

`PROPOSED` Builtin strategies to ship, all wrapping code that already exists:

| id | `supports_mixed_rank` | `requires_oof` | `requires_proba` | wraps |
|---|---|---|---|---|
| `soft_vote` | ✗ | ✗ | ✓ | `voting.py` soft path |
| `hard_vote` | ✗ | ✗ | ✗ | `voting.py` hard path (+ seeded tie-break, Phase 95) |
| `weighted_average` | ✗ | ✗ | ✓ | new, ~40 lines |
| `stack_ridge` | ✓ | ✓ | ✓ | `RidgeMetaLearner` |
| `stack_mlp` | ✓ | ✓ | ✓ | `MLPMetaLearner` |
| `stack_xgb` | ✓ | ✓ | ✓ | `XGBoostMeta` |
| `stack_calibrated` | ✓ | ✓ | ✓ | `CalibratedMetaLearner` |
| `blend_holdout` | ✓ | ✗ | ✓ | `blending.py` |
| `rank_average` | ✓ | ✓ | ✓ | new, ~30 lines |
| `best_single` | ✓ | ✓ | ✗ | new, ~20 lines (baseline for honest comparison) |

`PROPOSED` Composite capability computation (sktime P4):

```python
def composite_capabilities(members: list[ModelCapabilities],
                           scaps: EnsembleCapabilities) -> ModelCapabilities:
    """An ensemble is itself a model — compute its ModelCapabilities."""
    return ModelCapabilities(
        name=f"ensemble[{scaps.name}]",
        family="ensemble",
        task=members[0].task,
        n_classes=members[0].n_classes,
        input_rank=(DataRank.TABULAR_2D if scaps.supports_mixed_rank
                    else max(m.input_rank for m in members)),
        min_features=max(m.min_features for m in members),
        max_features=min(m.max_features for m in members),
        sequence_length=(None if scaps.supports_mixed_rank
                         else max((m.effective_sequence_length or 0) for m in members) or None),
        requires_scaling=any(m.requires_scaling for m in members),
        handles_nan=all(m.handles_nan for m in members),
        supports_proba=scaps.requires_proba or all(m.supports_proba for m in members),
        supports_incremental=all(m.supports_incremental for m in members) and not scaps.requires_fit,
        supports_gpu=any(m.supports_gpu for m in members),
        requires_gpu=any(m.requires_gpu for m in members),
        python_dependencies=tuple(sorted({d for m in members for d in m.python_dependencies})),
    )
```

This makes **ensembles-of-ensembles** fall out for free, and it is the same
`min`/`max`/`any`/`all` folding that `VotingEnsemble.requires_4d`
(`voting.py:109-118`) does by hand today.

## C.9 The experiment manifest

`PROPOSED` This is the user-facing artifact that makes the product requirement
literal.

```yaml
# experiments/momentum_v3.yaml
version: 1
name: momentum_v3
seed: 42

data:
  symbol: MES
  path: data/mes_1min.parquet
  date_range: [2019-01-01, 2024-12-31]
  timeframes: [1min, 5min, 15min, 60min]

labeling:
  method: triple_barrier
  horizon: 20
  binary_mode: false
  # → label_recipe_id is hashed from THIS block + symbol costs

features:
  set: core_full
  selection:
    method: mda
    n_features: 80

# ── The marketplace: name any number of compositions over a shared model pool ──
models:
  A: {ref: xgboost,      config: {max_depth: 6}}
  B: {ref: lightgbm}
  C: {ref: catboost}
  D: {ref: lstm,         config: {hidden_size: 256}}
  E: {ref: tcn}
  F: {ref: patchtst}
  G: {ref: acme_tsmixer}          # third-party plugin, entry point mlfactory.models

compositions:
  - name: boosting_soft
    members: [A, B, C]
    strategy: {kind: soft_vote, weights: [0.5, 0.3, 0.2]}

  - name: cross_family_stack
    members: [A, D, F]
    strategy:
      kind: stack_ridge
      missing_policy: nan_propagate
      min_coverage: 0.6
    calibration: {members: isotonic, output: none}

  - name: neural_rank
    members: [D, E, F]
    strategy: {kind: rank_average}

  - name: baseline
    members: [A, B, C, D, E, F]
    strategy: {kind: best_single}      # honest control

evaluation:
  compare: [boosting_soft, cross_family_stack, neural_rank, baseline]
  backtest: true
  metrics: [macro_f1, logloss_weighted, sharpe_ratio, deflated_sharpe]
```

`PROPOSED` Key properties:

- **Base models are trained once and their OOF reused across every
  composition.** `A` appears in three compositions; it is fitted once. This is
  the single biggest practical win over today's one-list-per-experiment model,
  and it falls out of separating `models:` (the pool) from `compositions:`.
- **Swapping `A+B+C -> X` for `A+D+F -> Y` is a text edit**, and both can
  coexist and be compared in one run.
- Third-party `G` needs no repo edit.
- `evaluation.compare` gives an apples-to-apples leaderboard, and pairs with the
  existing DSR Bonferroni `num_tests` correction (`CLAUDE.md` Phase 96 C1) —
  `num_tests` becomes `len(compositions)` automatically.

Loader: dataclasses + a hand-written validator (not pydantic — the repo has zero
pydantic dependency today and `src/config/base.py:BaseConfig` already
establishes a `validate() -> list[str]` idiom worth staying consistent with).
`UNKNOWN` whether the team would accept pydantic; if yes it buys better error
messages for free.

## C.10 Conformance suite — the extension contract

`PROPOSED`

```python
# src/platform/conformance.py
def check_model(name_or_spec, *, fast: bool = True) -> list[str]:
    """
    sklearn check_estimator analogue. Third-party plugins run this in their CI;
    `ml models check <name>` runs it locally.

    Returns a list of failure strings; empty ⇒ conformant.
    """
    checks = [
        _check_caps_wellformed,        # ModelCapabilities.__post_init__ passes
        _check_protocol_satisfied,     # isinstance(inst, TimeSeriesModel)
        _check_deps_declared,          # every import in python_dependencies resolves
        _check_default_config_roundtrip,
        _check_fit_predict_shapes,     # synthetic data at declared rank
        _check_predict_before_fit_raises,
        _check_proba_rows_sum_to_one,  # iff supports_proba
        _check_proba_shape_matches_n_classes,
        _check_nan_claim,              # feed NaN iff handles_nan, expect no crash / clean error
        _check_sample_weight_claim,    # weights change the fit iff supports_sample_weight
        _check_min_features_honored,   # min_features-1 columns → clean error, not crash
        _check_save_load_roundtrip,    # predictions identical after save/load
        _check_determinism_claim,      # two seeded fits agree iff determinism != STOCHASTIC
        _check_incremental_claim,      # partial_fit exists iff supports_incremental
        _check_binary_mode,            # n_classes=2 runs iff supports_binary
    ]
    ...
```

`INFERRED` Running this against the current 12 would immediately surface the
`mlp_meta` scaling drift and the `patchtst`/`itransformer`/`transformer` family
drift found in §A.4 — those become CI failures instead of latent bugs.

## C.11 Extension points — "adding a model" in full

`PROPOSED` The complete checklist for a new model, in-tree or third-party:

```python
# 1. Implement the protocol (subclassing BaseModel is the easy path, not required)
class TSMixerModel(BaseModel):
    def fit(self, X_train, y_train, X_val, y_val, sample_weights=None, config=None): ...
    def predict(self, X) -> PredictionResult: ...
    def save(self, path): ...
    def load(self, path): ...
    def get_default_config(self): ...

# 2. Declare capabilities + register
SPEC = ModelSpec(
    capabilities=ModelCapabilities(
        name="tsmixer", family="mlp",
        input_rank=DataRank.SEQUENCE_3D, sequence_length=96, min_sequence_length=96,
        min_features=20, max_features=120,
        requires_scaling=True, scaler_kind=ScalerKind.STANDARD,
        supports_proba=True, prediction_kind=PredictionKind.CLASS_PROBA,
        supports_gpu=True, python_dependencies=("torch>=2.2",),
    ),
    factory=TSMixerModel,
)

# 3a. in-tree:      @ModelRegistry.register(SPEC.capabilities) above the class
# 3b. third-party:  pyproject.toml
#     [project.entry-points."mlfactory.models"]
#     tsmixer = "acme_tsmixer:SPEC"

# 4. Prove it
#    $ ml models check tsmixer
```

**Nothing else.** No edit to `constants.py`, `strategies.py`,
`feature_sets/core.py`, `feature_selection/config.py`, `device.py`,
`trained_registry`, or `validator.py` — because all of those become derived
views over the spec table.

Same three-step story for a new **ensemble strategy** (`mlfactory.strategies`),
a new **adapter** (`mlfactory.adapters`), and a new **calibrator**
(`mlfactory.calibrators`).

---

# PART D — Retrofit Plan: REPAIR / REPLACE / DELETE

`PROPOSED` Ordered so each phase is independently shippable and reversible, and
the old paths keep working throughout.

### Phase P1 — Spine, additive only (~1 week, zero behavior change)

**ADD** `src/platform/{capabilities,protocol,spec,registry,space,planner,strategy,validate,conformance,manifest}.py`.

**REPAIR** `src/core/contracts/model_contract.py` → keep the file, add a
one-way bridge:

```python
def _caps_from_contract(c: ModelContract) -> ModelCapabilities: ...   # migration aid
def _contract_from_caps(c: ModelCapabilities) -> ModelContract: ...   # compat shim
MODEL_CONTRACTS = _DerivedContractView()   # reads ModelRegistry, not a literal dict
```

Every one of the 30+ `get_model_contract()` call sites keeps working, reading
through to the spec. **Drift becomes structurally impossible from day one.**

**DELETE (immediately, low risk):**
- `src/core/constants.py:100` — `assert len(ALL_MODELS) == 23`. **Hard blocker.**
- `src/core/constants.py:190` — `assert TOTAL_BASE_FEATURES == 162`.
- `src/core/constants.py:81-109` — `MODEL_FAMILIES`, `MODEL_TO_FAMILY`, `ALL_MODELS`;
  replace with `ModelRegistry.find(family=...)`.

**REPAIR** `DataRank.from_model` / `ModelFamily.from_model`
(`src/core/types.py:46,87`) to read `ModelRegistry.caps(name)`.

**Exit test:** `check_model` green for all 12; a new `test_no_capability_drift`
that asserts contract-view == registry for every registered model (this test
**fails today** on `mlp_meta` + 3 family mismatches — fixing those is part of P1).

### Phase P2 — Capability declarations (~3 days)

**REPAIR** 23 `@register(...)` decorators → `@ModelRegistry.register(ModelCapabilities(...))`.
Mechanical: values come from `MODEL_CONTRACTS` + the instance properties; the
4 drift cases get adjudicated explicitly.

Newly-declared facts (currently unrepresented, must be researched per model):
`handles_nan` (XGBoost/LightGBM/CatBoost = True; the rest False),
`supports_incremental` (XGB `xgb_model=`, LGBM `init_model=`, CatBoost
`init_model=`, sklearn SGD-family; torch models trivially True with warm start),
`min_sequence_length` (TCN=61, PatchTST≥patch_len, InceptionTime, ResNet1D),
`min_train_samples`, `python_dependencies`, `determinism`.

**Exit test:** `check_model` still green; `ModelRegistry.find(...)` queries
return the expected sets.

### Phase P3 — Validator wired into the live path (~3 days)

**REPAIR** `src/factory.py:246` `MLFactory.run()`: insert
`CompositionValidator().validate(manifest, profile).raise_if_failed()`
immediately after the data profile is known and **before** `_run_training`.

**REPLACE** `src/models/ensemble/validator.py` → thin deprecation shim
delegating to `CompositionValidator`. Preserve the excellent error prose from
`_build_rank_compatibility_error_message` (`:148`) — port it into
`Diagnostic.remedy` templates, but **generate the model lists from the
registry** instead of hardcoding `"xgboost, lightgbm, catboost"` (`:196`).

**Exit test:** `tests/test_ensemble_input_ranks.py` still passes; new test that
`MLFactory.run()` on an incompatible manifest raises within 2 s and never
touches the data file.

### Phase P4 — Strategy registry (~1 week)

**REPLACE** `EnsembleService._train_meta_learner`'s `meta_learner_map`
(`ensemble_service.py:400-418`) with `StrategyRegistry.get(comp.strategy.kind)`.

**REPAIR** `VotingEnsemble` / `BlendingEnsemble` / `StackingEnsemble` — keep
the classes, wrap each as a registered `EnsembleStrategy`. Their bodies do not
change; they finally become **reachable from production**, which resolves the
documented voting/blending dead-code situation.

**REPAIR** `unified_orchestrator.py:512-517` — replace the `_h{N}` suffix filter
with `PredictionSpace` equality.

**DELETE** `src/config/ensemble.py:EnsembleMethod` (zero consumers, superseded
by `StrategyRegistry.names()`); keep `EnsembleConfig` only if the manifest
loader reuses it.

### Phase P5 — Manifest + multi-composition (~1 week)

**ADD** `src/platform/manifest.py` + `ml run <manifest.yaml>`.

**REPAIR** `UnifiedOrchestrator` to loop compositions over a shared
`{model_key: PredictionBatch}` OOF pool. **This is the change that delivers the
product requirement.** The pool already exists as
`UnifiedOrchestrator._oof_predictions` — today it is filtered to one horizon,
consumed once, then `.clear()`ed (`unified_orchestrator.py:463`). Consume it N
times before clearing.

**REPAIR** `src/config/experiment.py:TrainingSection` — keep it as the
single-composition sugar that lowers to a manifest with one composition, so all
existing configs/notebooks keep working.

### Phase P6 — Collapse the name-based special cases (~1 week)

**DELETE** and replace with registry/capability lookups:

| Delete | Replace with |
|---|---|
| `training_ops.py:1077-1130` `_create_meta_model` if/elif | `ModelRegistry.create(name, config)` |
| `meta_labeling/primary_model.py:95,355-389` | `ModelRegistry.create` + `find(input_rank=2D, supports_proba=True)` for the allow-list |
| `calibrated_meta.py:300-314` | registry lookup |
| `trained_registry/registry.py:468,472` name tuples | `ModelRegistry.caps(name).family` / `.input_rank` |
| `config_validator.py:257,456` | `caps.requires_sequences` / `caps.family == "ensemble"` |
| `feature_selection/config.py:103-105` model-names-in-family-map | `caps.family` (correct once P2 fixes the drift) |
| `device.py:464,617` `family in ("lstm","gru")` | new caps fields (`extra["gpu_profile"]`) |
| `ensemble/validator.py:85` catboost hint | `caps.python_dependencies` |

**MOVE** (not delete) `MODEL_FEATURE_STRATEGIES` (`strategies.py:146`) and
`FEATURE_SET_ALIASES` (`feature_sets/core.py`) content into
`ModelCapabilities.preferred_feature_families` / `feature_mode` / `extra`, so
each model's feature preferences live with the model.

### Phase P7 — Adapter planner + close DECISIONS #4 (~4 days)

**REPLACE** `AdapterFactory` (`src/data/adapters/factory.py`, 492 lines) with
`plan_adapter` + `AdapterPlan`. **KEEP** `AdapterRegistry` and all five concrete
adapters unchanged.

**REPAIR** the `additional_dfs` signature leak: give `BaseAdapter.transform` a
uniform `(df, context: AdapterContext)` signature so callers stop re-deriving
adapter identity.

**REPAIR** DECISIONS #4: with `effective_sequence_length` in the plan, standard
and walk-forward modes agree by construction. Per DECISIONS, this **changes
model results** — run it as its own phase with before/after metrics, exactly as
the doc recommends.

### Summary: REPAIR / REPLACE / DELETE

| Verdict | Items |
|---|---|
| **REPAIR** (keep, fix in place) | `ModelContract` → derived view; `BaseModel` (+ `capabilities()`); `AdapterRegistry`; all 5 adapters; all 12 model classes (decorator args only); `Voting`/`Blending`/`Stacking` (wrap as strategies); `PredictionResult`; `OOFPrediction`; `DataContract`; `ProbabilityCalibrator`; `TrainerConfig`; `unified_orchestrator` OOF loop; `types.py` `from_model` classmethods |
| **REPLACE** (new impl, old API preserved) | `ModelRegistry` → spec-based; `ensemble/validator.py` → `CompositionValidator`; `AdapterFactory` → `plan_adapter`; `meta_learner_map` → `StrategyRegistry`; `_h{N}` filter → `PredictionSpace` |
| **DELETE** | `assert len(ALL_MODELS)==23` and `assert TOTAL_BASE_FEATURES==162`; `MODEL_FAMILIES`/`MODEL_TO_FAMILY`/`ALL_MODELS`; `EnsembleMethod`; `training_ops._create_meta_model`; `primary_model` model if/elif + allow-list; `calibrated_meta` estimator if/elif; name-tuple branches in `trained_registry`, `config_validator`, `device`; hardcoded model lists inside `validator.py` error text |
| **OUT OF SCOPE** (pre-existing DECISIONS items) | #1 serving chain, #5 5-D Optuna island, #6 dual `AdapterResult`, #10 import SCC. #6 and #10 *help* this work and could be folded into P1. |

`INFERRED` Rough size: **+2,500 / −1,800 lines**, ~4 weeks, and — critically —
**zero edits to any model's `fit`/`predict` body**.

---

# PART E — Risks, Open Questions, UNKNOWNs

`INFERRED`

1. **`MODEL_CONTRACTS` as a derived view is the linchpin.** 30+ call sites read
   it. If the view is not a faithful drop-in (frozen dataclass, same field
   names, same `adapter_id` property), P1 breaks everything at once. Mitigation:
   the view returns a real `ModelContract` built from caps; a property-based
   test asserts view == old literal dict for all 23 at the P1 boundary.

2. **Import cost.** `src/platform` must not import torch. `ModelRegistry.caps()`
   must answer without importing the model implementation — that is why
   capabilities live on the `ModelSpec`, not only as a classmethod. Entry-point
   loading must be lazy (`ep.load()` only on `create()`, not on `caps()`).
   Interacts with DECISIONS #10.

3. **`min_sequence_length` changes results** (DECISIONS #4). Gate behind a
   config flag for one release.

4. **Capability lying.** A plugin can declare `handles_nan=True` and crash.
   `check_model` mitigates but does not eliminate; `ModelRunner` should catch
   and re-raise as `CapabilityViolation` naming the false claim.

5. `UNKNOWN` **Is 4D heterogeneous stacking actually correct?** `CLAUDE.md`
   Phase 57/60 says yes (via `_generate_4d_oof`); `validator.py:141` and
   `tests/test_ensemble_input_ranks.py:31` say no. Someone must adjudicate
   before `EnsembleCapabilities.supports_mixed_rank` is set for the stacking
   strategies. My reading of `ensemble_service.py` is that once every member
   emits 2D OOF, rank is irrelevant to the meta-learner — so 4D members *should*
   be admissible and the validator is simply stale. **Needs a decision, not a
   guess.**

6. `UNKNOWN` **pydantic or hand-rolled** for the manifest. Repo has no pydantic
   dep; `BaseConfig.validate() -> list[str]` is the existing idiom.

7. `UNKNOWN` **Do we need per-composition feature selection?** `CLAUDE.md`
   Phase 58/94 established per-model, train-only feature selection. Sharing one
   base-model pool across compositions means feature selection is per *model*,
   not per composition — which is already the case, so this should be fine, but
   it deserves an explicit check.

8. **Migration cost of `PredictionSpace`.** `label_recipe_id` requires hashing
   the labeling config. `DataContract._compute_schema_hash`
   (`data_contract.py:188`) is the existing pattern to copy.

---

## Appendix — Evidence index

| Claim | File:line |
|---|---|
| `ModelContract` definition | `src/core/contracts/model_contract.py:37-108` |
| `MODEL_CONTRACTS` table (23 entries) | `src/core/contracts/model_contract.py:229-560` |
| Contract enforced only in adapters | `tabular.py:122`, `sequence.py:153`, `multi_stream.py:231` |
| `AdapterRegistry` decorator | `src/data/adapters/registry.py:20-52` |
| `get_adapter` dispatch | `src/data/adapters/registry.py:148-181` |
| Second adapter dispatcher | `src/data/adapters/factory.py:142,157,165,244,311` |
| `ModelRegistry` decorator + create | `src/models/registry.py:62-175` |
| Instantiation entry | `src/models/training/trainer.py:115` |
| Registration via import side effect | `src/models/__init__.py:48-53` |
| `assert len(ALL_MODELS) == 23` | `src/core/constants.py:100` |
| `assert TOTAL_BASE_FEATURES == 162` | `src/core/constants.py:190` |
| Derived rank/adapter maps | `src/core/constants.py:118-161` |
| `BaseModel` capability properties | `src/models/base.py:194-238` |
| `BaseModel.fit/predict/save/load` | `src/models/base.py:244-334` |
| Ensemble validator (dead in prod) | `src/models/ensemble/validator.py:33-252` |
| Best-in-class error message | `src/models/ensemble/validator.py:148-224` |
| Live ensemble path bypasses validator | `src/models/training/services/ensemble_service.py:75-215` |
| Hardcoded `meta_learner_map` | `src/models/training/services/ensemble_service.py:400-418` |
| `_h{N}` suffix combinability check | `src/models/training/unified_orchestrator.py:512-517` |
| OOF schema contract (enforced) | `src/models/training/services/ensemble_service.py:232-240` |
| Config surface for composition | `src/config/experiment.py:103,128-129` |
| Unused `EnsembleMethod` | `src/config/ensemble.py:28-33` |
| CLI composition flags | `src/cli/commands/train.py:309-313` |
| `MLFactory.run()` — no pre-flight validation | `src/factory.py:246-350` |
| `PredictionResult` canonical | `src/core/interfaces.py:125-197` |
| `TrainerProtocol` / `InferenceBundle` | `src/core/protocols.py:20,49` |
| Name-based special cases | see §A.5 table |
| DECISIONS #4 (contract seq_len) | `DECISIONS.md:84-105`, matrix row 4 |
